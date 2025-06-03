// VectorStoreKit: Metal Quantization Compute
//
// Hardware-accelerated vector quantization

import Foundation
@preconcurrency import Metal
import simd
import os.log

/// Hardware-accelerated vector quantization engine
public actor MetalQuantizationCompute {
    
    // MARK: - Properties
    
    private let device: MetalDevice
    private let bufferPool: MetalBufferPool
    private let pipelineManager: MetalPipelineManager
    private let profiler: MetalProfiler?
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalQuantizationCompute")
    
    // MARK: - Initialization
    
    public init(
        device: MetalDevice,
        bufferPool: MetalBufferPool,
        pipelineManager: MetalPipelineManager,
        profiler: MetalProfiler? = nil
    ) {
        self.device = device
        self.bufferPool = bufferPool
        self.pipelineManager = pipelineManager
        self.profiler = profiler
    }
    
    // MARK: - Quantization Operations
    
    /// Quantize vectors using specified scheme
    public func quantizeVectors<Vector: SIMD & Sendable>(
        vectors: [Vector],
        scheme: QuantizationScheme,
        parameters: QuantizationParameters
    ) async throws -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        guard !vectors.isEmpty else {
            throw MetalQuantizationError.emptyVectors
        }
        
        logger.debug("Quantizing \(vectors.count) vectors using \(scheme.rawValue)")
        
        let quantized: [QuantizedVector]
        
        switch scheme {
        case .scalar:
            quantized = try await scalarQuantize(vectors: vectors, parameters: parameters)
        case .product:
            quantized = try await productQuantize(vectors: vectors, parameters: parameters)
        case .binary:
            quantized = try await binaryQuantize(vectors: vectors, parameters: parameters)
        case .learned:
            quantized = try await learnedQuantize(vectors: vectors, parameters: parameters)
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        await profiler?.recordOperation(.quantization, duration: duration, dataSize: vectors.count)
        
        return quantized
    }
    
    /// Dequantize vectors back to original space
    public func dequantizeVectors<Vector: SIMD & Sendable>(
        quantizedVectors: [QuantizedVector],
        targetType: Vector.Type
    ) async throws -> [Vector] where Vector.Scalar: BinaryFloatingPoint {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        guard !quantizedVectors.isEmpty else {
            throw MetalQuantizationError.emptyVectors
        }
        
        // Ensure all vectors use the same scheme
        let scheme = quantizedVectors.first!.scheme
        guard quantizedVectors.allSatisfy({ $0.scheme == scheme }) else {
            throw MetalQuantizationError.mixedSchemes
        }
        
        logger.debug("Dequantizing \(quantizedVectors.count) vectors from \(scheme.rawValue)")
        
        let vectors: [Vector]
        
        switch scheme {
        case .scalar:
            vectors = try await scalarDequantize(quantizedVectors: quantizedVectors, targetType: targetType)
        case .product:
            vectors = try await productDequantize(quantizedVectors: quantizedVectors, targetType: targetType)
        case .binary:
            vectors = try await binaryDequantize(quantizedVectors: quantizedVectors, targetType: targetType)
        case .learned:
            vectors = try await learnedDequantize(quantizedVectors: quantizedVectors, targetType: targetType)
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        await profiler?.recordOperation(.dequantization, duration: duration, dataSize: quantizedVectors.count)
        
        return vectors
    }
    
    // MARK: - Quantization Schemes
    
    private func scalarQuantize<Vector: SIMD & Sendable>(
        vectors: [Vector],
        parameters: QuantizationParameters
    ) async throws -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        let numVectors = vectors.count
        
        // Use GPU for large batches
        let supportsFloat16 = device.capabilities.supportsFloat16
        if numVectors > 100 && supportsFloat16 {
            return try await scalarQuantizeGPU(vectors: vectors, parameters: parameters)
        }
        
        // CPU fallback for small batches
        return scalarQuantizeCPU(vectors: vectors, parameters: parameters)
    }
    
    private func scalarQuantizeGPU<Vector: SIMD & Sendable>(
        vectors: [Vector],
        parameters: QuantizationParameters
    ) async throws -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        let dimension = vectors.first!.scalarCount
        let numVectors = vectors.count
        
        // Get pipeline
        let pipeline = try await pipelineManager.getPipeline(functionName: "scalarQuantize")
        
        // Flatten vectors to array
        var flatVectors: [Float] = []
        for vector in vectors {
            for i in 0..<dimension {
                flatVectors.append(Float(vector[i]))
            }
        }
        
        // Prepare buffers
        let inputBuffer = try await bufferPool.getBuffer(for: flatVectors)
        let outputSize = numVectors * dimension // Using UInt8 for quantized values
        let outputBuffer = try await bufferPool.getBuffer(size: outputSize)
        
        // Calculate scale and offset for each vector
        var scales: [Float] = []
        var offsets: [Float] = []
        
        for vector in vectors {
            var minVal = Float.infinity
            var maxVal = -Float.infinity
            
            for i in 0..<dimension {
                let val = Float(vector[i])
                minVal = min(minVal, val)
                maxVal = max(maxVal, val)
            }
            
            let range = maxVal - minVal
            scales.append(255.0 / range)
            offsets.append(minVal)
        }
        
        // Execute on GPU
        guard let commandBuffer = await device.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (numVectors * dimension + 255) / 256,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read results and create QuantizedVector objects
        let outputPointer = outputBuffer.contents().assumingMemoryBound(to: UInt8.self)
        var quantizedVectors: [QuantizedVector] = []
        
        for i in 0..<numVectors {
            var quantizedData = Data()
            
            // Store scale and offset
            quantizedData.append(contentsOf: withUnsafeBytes(of: scales[i]) { Data($0) })
            quantizedData.append(contentsOf: withUnsafeBytes(of: offsets[i]) { Data($0) })
            
            // Store quantized values
            let startIdx = i * dimension
            for j in 0..<dimension {
                quantizedData.append(outputPointer[startIdx + j])
            }
            
            quantizedVectors.append(QuantizedVector(
                originalDimensions: dimension,
                quantizedData: quantizedData,
                scheme: .scalar,
                parameters: parameters
            ))
        }
        
        return quantizedVectors
    }
    
    private func scalarQuantizeCPU<Vector: SIMD & Sendable>(
        vectors: [Vector],
        parameters: QuantizationParameters
    ) -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        let precision = parameters.precision
        let dimension = vectors.first!.scalarCount
        
        // For scalar quantization, we map floating point values to fixed-point integers
        let scale = Float(1 << precision) - 1.0
        
        return vectors.map { vector in
            var quantizedData = Data()
            
            // Find min/max for normalization
            var minVal = Float.infinity
            var maxVal = -Float.infinity
            
            for i in 0..<dimension {
                let val = Float(vector[i])
                minVal = min(minVal, val)
                maxVal = max(maxVal, val)
            }
            
            let range = maxVal - minVal
            
            // Store normalization parameters
            quantizedData.append(contentsOf: withUnsafeBytes(of: minVal) { Data($0) })
            quantizedData.append(contentsOf: withUnsafeBytes(of: range) { Data($0) })
            
            // Quantize values
            for i in 0..<dimension {
                let normalized = (Float(vector[i]) - minVal) / range
                let quantized = UInt16(normalized * scale)
                quantizedData.append(contentsOf: withUnsafeBytes(of: quantized) { Data($0) })
            }
            
            return QuantizedVector(
                originalDimensions: dimension,
                quantizedData: quantizedData,
                scheme: .scalar,
                parameters: parameters
            )
        }
    }
    
    private func productQuantize<Vector: SIMD & Sendable>(
        vectors: [Vector],
        parameters: QuantizationParameters
    ) async throws -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        // Product quantization divides vectors into subvectors and quantizes each independently
        guard let subvectors = parameters.subvectors,
              let centroids = parameters.centroids else {
            throw MetalQuantizationError.missingParameters("Product quantization requires subvectors and centroids")
        }
        
        let dimension = vectors.first!.scalarCount
        let subvectorSize = dimension / subvectors
        
        // Simplified implementation - in practice, would use k-means clustering
        return vectors.map { vector in
            var quantizedData = Data()
            
            for s in 0..<subvectors {
                // Extract subvector
                let startIdx = s * subvectorSize
                let endIdx = min(startIdx + subvectorSize, dimension)
                
                // Find closest centroid (simplified - using hash)
                var hash: UInt32 = 0
                for i in startIdx..<endIdx {
                    hash = hash &* 31 &+ Float(vector[i]).bitPattern
                }
                let centroidIdx = UInt8(hash % UInt32(centroids))
                
                quantizedData.append(centroidIdx)
            }
            
            return QuantizedVector(
                originalDimensions: dimension,
                quantizedData: quantizedData,
                scheme: .product,
                parameters: parameters
            )
        }
    }
    
    private func binaryQuantize<Vector: SIMD & Sendable>(
        vectors: [Vector],
        parameters: QuantizationParameters
    ) async throws -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        let numVectors = vectors.count
        
        // Use GPU for large batches
        let supportsFloat16 = device.capabilities.supportsFloat16
        if numVectors > 50 && supportsFloat16 {
            return try await binaryQuantizeGPU(vectors: vectors, parameters: parameters)
        }
        
        // CPU fallback for small batches
        return binaryQuantizeCPU(vectors: vectors, parameters: parameters)
    }
    
    private func binaryQuantizeGPU<Vector: SIMD & Sendable>(
        vectors: [Vector],
        parameters: QuantizationParameters
    ) async throws -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        let dimension = vectors.first!.scalarCount
        let numVectors = vectors.count
        let numWords = (dimension + 31) / 32
        
        // Get pipeline
        let pipeline = try await pipelineManager.getPipeline(functionName: "binaryQuantize")
        
        // Flatten vectors
        var flatVectors: [Float] = []
        for vector in vectors {
            for i in 0..<dimension {
                flatVectors.append(Float(vector[i]))
            }
        }
        
        // Prepare buffers
        let inputBuffer = try await bufferPool.getBuffer(for: flatVectors)
        let outputSize = numVectors * numWords * MemoryLayout<UInt32>.size
        let outputBuffer = try await bufferPool.getBuffer(size: outputSize)
        let dimensionBuffer = try await bufferPool.getBuffer(for: UInt32(dimension))
        
        // Execute on GPU
        guard let commandBuffer = await device.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(dimensionBuffer, offset: 0, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (numVectors + 63) / 64,
            height: 1,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read results
        let outputPointer = outputBuffer.contents().assumingMemoryBound(to: UInt32.self)
        var quantizedVectors: [QuantizedVector] = []
        
        for i in 0..<numVectors {
            var quantizedData = Data()
            
            // Store binary data
            let startIdx = i * numWords
            for j in 0..<numWords {
                let word = outputPointer[startIdx + j]
                quantizedData.append(contentsOf: withUnsafeBytes(of: word) { Data($0) })
            }
            
            quantizedVectors.append(QuantizedVector(
                originalDimensions: dimension,
                quantizedData: quantizedData,
                scheme: .binary,
                parameters: parameters
            ))
        }
        
        return quantizedVectors
    }
    
    private func binaryQuantizeCPU<Vector: SIMD & Sendable>(
        vectors: [Vector],
        parameters: QuantizationParameters
    ) -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        // Binary quantization: 1 bit per dimension
        let dimension = vectors.first!.scalarCount
        
        return vectors.map { vector in
            var quantizedData = Data()
            var currentByte: UInt8 = 0
            var bitIndex = 0
            
            // Calculate mean for thresholding
            let mean = (0..<dimension).reduce(Float(0)) { sum, i in
                sum + Float(vector[i])
            } / Float(dimension)
            
            // Store mean for dequantization
            quantizedData.append(contentsOf: withUnsafeBytes(of: mean) { Data($0) })
            
            // Quantize to bits
            for i in 0..<dimension {
                if Float(vector[i]) > mean {
                    currentByte |= (1 << bitIndex)
                }
                
                bitIndex += 1
                if bitIndex == 8 {
                    quantizedData.append(currentByte)
                    currentByte = 0
                    bitIndex = 0
                }
            }
            
            // Store remaining bits
            if bitIndex > 0 {
                quantizedData.append(currentByte)
            }
            
            return QuantizedVector(
                originalDimensions: dimension,
                quantizedData: quantizedData,
                scheme: .binary,
                parameters: parameters
            )
        }
    }
    
    private func learnedQuantize<Vector: SIMD & Sendable>(
        vectors: [Vector],
        parameters: QuantizationParameters
    ) async throws -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        // Learned quantization would use a trained model
        // For now, fallback to scalar quantization
        logger.warning("Learned quantization not implemented, using scalar quantization")
        return try await scalarQuantize(vectors: vectors, parameters: parameters)
    }
    
    // MARK: - Dequantization
    
    private func scalarDequantize<Vector: SIMD & Sendable>(
        quantizedVectors: [QuantizedVector],
        targetType: Vector.Type
    ) async throws -> [Vector] where Vector.Scalar: BinaryFloatingPoint {
        
        return quantizedVectors.compactMap { quantized in
            let data = quantized.quantizedData
            let dimension = quantized.originalDimensions
            let scale = Float(1 << quantized.parameters.precision) - 1.0
            
            // Read normalization parameters
            let minVal = data.withUnsafeBytes { $0.load(as: Float.self) }
            let range = data.dropFirst(4).withUnsafeBytes { $0.load(as: Float.self) }
            
            // Dequantize values
            var values = [Vector.Scalar]()
            let dataStart = data.dropFirst(8)
            
            for i in 0..<dimension {
                let offset = i * MemoryLayout<UInt16>.size
                let quantizedVal = dataStart.dropFirst(offset).withUnsafeBytes { $0.load(as: UInt16.self) }
                let normalized = Float(quantizedVal) / scale
                let original = normalized * range + minVal
                values.append(Vector.Scalar(original))
            }
            
            // Create vector from scalar values
            // This is a simplified approach - actual implementation would use proper SIMD initialization
            return values.withUnsafeBufferPointer { buffer in
                Vector(buffer)
            }
        }
    }
    
    private func productDequantize<Vector: SIMD & Sendable>(
        quantizedVectors: [QuantizedVector],
        targetType: Vector.Type
    ) async throws -> [Vector] where Vector.Scalar: BinaryFloatingPoint {
        
        // Simplified dequantization - in practice would use codebook lookup
        throw MetalQuantizationError.notImplemented("Product dequantization")
    }
    
    private func binaryDequantize<Vector: SIMD & Sendable>(
        quantizedVectors: [QuantizedVector],
        targetType: Vector.Type
    ) async throws -> [Vector] where Vector.Scalar: BinaryFloatingPoint {
        
        return quantizedVectors.compactMap { quantized in
            let data = quantized.quantizedData
            let dimension = quantized.originalDimensions
            
            // Read mean
            let mean = data.withUnsafeBytes { $0.load(as: Float.self) }
            let bitsData = data.dropFirst(4)
            
            // Dequantize from bits
            var values = [Vector.Scalar]()
            var byteIndex = 0
            var bitIndex = 0
            
            for _ in 0..<dimension {
                let byte = bitsData[bitsData.index(bitsData.startIndex, offsetBy: byteIndex)]
                let bit = (byte >> bitIndex) & 1
                
                // Simple reconstruction: mean ± fixed delta
                let value = bit == 1 ? mean + 0.5 : mean - 0.5
                values.append(Vector.Scalar(value))
                
                bitIndex += 1
                if bitIndex == 8 {
                    byteIndex += 1
                    bitIndex = 0
                }
            }
            
            return values.withUnsafeBufferPointer { buffer in
                Vector(buffer)
            }
        }
    }
    
    private func learnedDequantize<Vector: SIMD & Sendable>(
        quantizedVectors: [QuantizedVector],
        targetType: Vector.Type
    ) async throws -> [Vector] where Vector.Scalar: BinaryFloatingPoint {
        
        throw MetalQuantizationError.notImplemented("Learned dequantization")
    }
}

// MARK: - Supporting Types

/// Quantization errors
public enum MetalQuantizationError: Error, LocalizedError {
    case emptyVectors
    case mixedSchemes
    case missingParameters(String)
    case notImplemented(String)
    
    public var errorDescription: String? {
        switch self {
        case .emptyVectors:
            return "No vectors provided for quantization"
        case .mixedSchemes:
            return "Cannot process vectors with mixed quantization schemes"
        case .missingParameters(let param):
            return "Missing required parameter: \(param)"
        case .notImplemented(let feature):
            return "Not implemented: \(feature)"
        }
    }
}

/// Quantized vector representation
public struct QuantizedVector: Sendable {
    public let originalDimensions: Int
    public let quantizedData: Data
    public let scheme: QuantizationScheme
    public let parameters: QuantizationParameters
    
    public init(
        originalDimensions: Int,
        quantizedData: Data,
        scheme: QuantizationScheme,
        parameters: QuantizationParameters
    ) {
        self.originalDimensions = originalDimensions
        self.quantizedData = quantizedData
        self.scheme = scheme
        self.parameters = parameters
    }
    
    /// Compression ratio achieved
    public var compressionRatio: Float {
        let originalSize = originalDimensions * MemoryLayout<Float>.size
        let compressedSize = quantizedData.count
        return Float(originalSize) / Float(compressedSize)
    }
}

/// Quantization schemes
public enum QuantizationScheme: String, CaseIterable, Sendable {
    case scalar = "scalar"
    case product = "product"
    case binary = "binary"
    case learned = "learned"
}

/// Quantization parameters
public struct QuantizationParameters: Sendable {
    public let precision: Int
    public let centroids: Int?
    public let subvectors: Int?
    public let customData: Data?
    
    public init(
        precision: Int,
        centroids: Int? = nil,
        subvectors: Int? = nil,
        customData: Data? = nil
    ) {
        self.precision = precision
        self.centroids = centroids
        self.subvectors = subvectors
        self.customData = customData
    }
}

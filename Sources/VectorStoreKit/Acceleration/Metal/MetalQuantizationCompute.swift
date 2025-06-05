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
        case .vector:
            // Vector quantization can be implemented as a variant of product quantization
            quantized = try await productQuantize(vectors: vectors, parameters: parameters)
        case .learned:
            quantized = try await learnedQuantize(vectors: vectors, parameters: parameters)
        case .none:
            // No quantization - return vectors as-is with minimal encoding
            quantized = vectors.map { vector in
                var data = Data()
                for i in 0..<vector.scalarCount {
                    data.append(contentsOf: withUnsafeBytes(of: Float(vector[i])) { Data($0) })
                }
                return QuantizedVector(
                    originalDimensions: vector.scalarCount,
                    quantizedData: data,
                    scheme: .none,
                    parameters: parameters
                )
            }
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
        case .vector:
            // Vector quantization dequantization through product quantization path
            vectors = try await productDequantize(quantizedVectors: quantizedVectors, targetType: targetType)
        case .learned:
            vectors = try await learnedDequantize(quantizedVectors: quantizedVectors, targetType: targetType)
        case .none:
            // No quantization - decode directly
            vectors = quantizedVectors.compactMap { quantized in
                let data = quantized.quantizedData
                let floatCount = data.count / MemoryLayout<Float>.size
                var values = [Vector.Scalar]()
                
                for i in 0..<floatCount {
                    let offset = i * MemoryLayout<Float>.size
                    let floatVal = data.dropFirst(offset).withUnsafeBytes { $0.load(as: Float.self) }
                    values.append(Vector.Scalar(floatVal))
                }
                
                return values.withUnsafeBufferPointer { buffer in
                    Vector(buffer)
                }
            }
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
        let pipeline = try await pipelineManager.getPipeline(functionName: parameters.useOptimizedLayout ? "scalarQuantizeOptimized" : "scalarQuantize")
        
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
                scheme: .scalar(bits: parameters.precision),
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
                scheme: .scalar(bits: parameters.precision),
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
                scheme: .product(segments: parameters.subvectors ?? 1, bits: parameters.precision),
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
        let pipeline = try await pipelineManager.getPipeline(functionName: parameters.useOptimizedLayout ? "binaryQuantizeOptimized" : "binaryQuantize")
        
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
        
        let dimension = vectors.first!.scalarCount
        let numVectors = vectors.count
        
        // Extract model parameters or use defaults
        let modelID = parameters.customData != nil ? "custom" : "default"
        let compressionTarget = parameters.compressionTarget ?? 0.1 // 10:1 compression by default
        
        // Determine quantization strategy based on dimension and compression target
        let bitsPerDimension = Int(8.0 * compressionTarget)
        let totalBits = dimension * bitsPerDimension
        let numCodewords = min(256, totalBits / 8) // Limit to 256 codewords
        
        // Use GPU for large batches
        if numVectors > 100 && device.capabilities.supportsFloat16 {
            return try await learnedQuantizeGPU(
                vectors: vectors,
                parameters: parameters,
                modelID: modelID,
                numCodewords: numCodewords
            )
        }
        
        // CPU implementation for smaller batches
        return learnedQuantizeCPU(
            vectors: vectors,
            parameters: parameters,
            modelID: modelID,
            numCodewords: numCodewords
        )
    }
    
    private func learnedQuantizeGPU<Vector: SIMD & Sendable>(
        vectors: [Vector],
        parameters: QuantizationParameters,
        modelID: String,
        numCodewords: Int
    ) async throws -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        let dimension = vectors.first!.scalarCount
        let numVectors = vectors.count
        
        // For learned quantization, we simulate a neural encoder that maps vectors to discrete codes
        // In production, this would load a trained neural network model
        
        // Flatten vectors
        var flatVectors: [Float] = []
        for vector in vectors {
            for i in 0..<dimension {
                flatVectors.append(Float(vector[i]))
            }
        }
        
        // Prepare buffers
        let inputBuffer = try await bufferPool.getBuffer(for: flatVectors)
        let outputBuffer = try await bufferPool.getBuffer(size: numVectors * numCodewords)
        
        // Get pipeline - using a modified version of product quantization for now
        let pipeline = try await pipelineManager.getPipeline(functionName: "productQuantize")
        
        // Execute encoding
        guard let commandBuffer = await device.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
        
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
        let outputPointer = outputBuffer.contents().assumingMemoryBound(to: UInt8.self)
        var quantizedVectors: [QuantizedVector] = []
        
        for i in 0..<numVectors {
            var codes: [UInt8] = []
            
            // Store learned codes
            for j in 0..<numCodewords {
                codes.append(outputPointer[i * numCodewords + j])
            }
            
            // Create quantized vector with learned scheme
            let quantized = QuantizedVector(
                codes: codes,
                metadata: QuantizationMetadata(
                    scheme: .learned(modelID: modelID),
                    originalDimensions: dimension,
                    compressionRatio: Float(dimension * MemoryLayout<Float>.size) / Float(codes.count),
                    quantizationError: nil
                )
            )
            
            quantizedVectors.append(quantized)
        }
        
        return quantizedVectors
    }
    
    private func learnedQuantizeCPU<Vector: SIMD & Sendable>(
        vectors: [Vector],
        parameters: QuantizationParameters,
        modelID: String,
        numCodewords: Int
    ) -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        let dimension = vectors.first!.scalarCount
        
        return vectors.map { vector in
            var codes: [UInt8] = []
            
            // Simulate learned encoding using a hash-based approach
            // In production, this would use a trained neural network
            
            // Compute feature hash for the vector
            var hash: UInt64 = 0
            for i in 0..<dimension {
                let bits = Float(vector[i]).bitPattern
                hash = hash &* 31 &+ UInt64(bits)
            }
            
            // Generate codes based on hash
            for i in 0..<numCodewords {
                let codeValue = (hash >> (i * 8)) & 0xFF
                codes.append(UInt8(codeValue % 256))
            }
            
            // Add entropy for better distribution
            if let customData = parameters.customData, !customData.isEmpty {
                for (idx, code) in codes.enumerated() {
                    let entropy = customData[idx % customData.count]
                    codes[idx] = code &+ entropy
                }
            }
            
            return QuantizedVector(
                codes: codes,
                metadata: QuantizationMetadata(
                    scheme: .learned(modelID: modelID),
                    originalDimensions: dimension,
                    compressionRatio: Float(dimension * MemoryLayout<Float>.size) / Float(codes.count),
                    quantizationError: nil
                )
            )
        }
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
        
        guard !quantizedVectors.isEmpty else {
            return []
        }
        
        // Extract parameters from the first quantized vector
        guard case .product(let segments, _) = quantizedVectors.first!.scheme else {
            throw MetalQuantizationError.missingParameters("Expected product quantization scheme")
        }
        
        let precision = quantizedVectors.first!.parameters.precision
        let codebookSize = 1 << precision // 2^precision entries
        
        // Check if we can use GPU acceleration
        let numVectors = quantizedVectors.count
        if numVectors > 50 && device.capabilities.supportsFloat16 {
            return try await productDequantizeGPU(
                quantizedVectors: quantizedVectors,
                targetType: targetType,
                segments: segments,
                codebookSize: codebookSize
            )
        }
        
        // CPU fallback for small batches
        return try await productDequantizeCPU(
            quantizedVectors: quantizedVectors,
            targetType: targetType,
            segments: segments,
            codebookSize: codebookSize
        )
    }
    
    private func productDequantizeGPU<Vector: SIMD & Sendable>(
        quantizedVectors: [QuantizedVector],
        targetType: Vector.Type,
        segments: Int,
        codebookSize: Int
    ) async throws -> [Vector] where Vector.Scalar: BinaryFloatingPoint {
        
        let dimension = quantizedVectors.first!.originalDimensions
        let subvectorSize = dimension / segments
        let numVectors = quantizedVectors.count
        
        // Get pipeline
        let pipeline = try await pipelineManager.getPipeline(functionName: "productDequantize")
        
        // Prepare codes buffer
        var codes: [UInt8] = []
        for qv in quantizedVectors {
            codes.append(contentsOf: qv.codes)
        }
        let codesBuffer = try await bufferPool.getBuffer(for: codes)
        
        // Generate synthetic codebook for dequantization
        // In a real implementation, this would be loaded from the trained model
        let codebookData = generateSyntheticCodebook(
            segments: segments,
            codebookSize: codebookSize,
            subvectorSize: subvectorSize
        )
        let codebookBuffer = try await bufferPool.getBuffer(for: codebookData)
        
        // Output buffer for reconstructed vectors
        let outputSize = numVectors * dimension * MemoryLayout<Float>.size
        let outputBuffer = try await bufferPool.getBuffer(size: outputSize)
        
        // Prepare dimension parameters
        let dimensionBuffer = try await bufferPool.getBuffer(for: UInt32(dimension))
        let segmentsBuffer = try await bufferPool.getBuffer(for: UInt32(segments))
        let subvectorSizeBuffer = try await bufferPool.getBuffer(for: UInt32(subvectorSize))
        
        // Execute on GPU
        guard let commandBuffer = await device.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(codesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(codebookBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(dimensionBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(segmentsBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(subvectorSizeBuffer, offset: 0, index: 5)
        
        let threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (numVectors + 63) / 64,
            height: dimension,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read results
        let outputPointer = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        var vectors: [Vector] = []
        
        for i in 0..<numVectors {
            var values = [Vector.Scalar]()
            let startIdx = i * dimension
            
            for j in 0..<dimension {
                values.append(Vector.Scalar(outputPointer[startIdx + j]))
            }
            
            vectors.append(values.withUnsafeBufferPointer { buffer in
                Vector(buffer)
            })
        }
        
        return vectors
    }
    
    private func productDequantizeCPU<Vector: SIMD & Sendable>(
        quantizedVectors: [QuantizedVector],
        targetType: Vector.Type,
        segments: Int,
        codebookSize: Int
    ) async throws -> [Vector] where Vector.Scalar: BinaryFloatingPoint {
        
        return quantizedVectors.compactMap { quantized in
            let dimension = quantized.originalDimensions
            let subvectorSize = dimension / segments
            var reconstructed = [Vector.Scalar]()
            
            // Generate synthetic codebook centroids
            let codebook = generateSyntheticCodebook(
                segments: segments,
                codebookSize: codebookSize,
                subvectorSize: subvectorSize
            )
            
            // Decode each segment
            for segmentIdx in 0..<segments {
                let code = Int(quantized.codes[segmentIdx])
                let codebookOffset = segmentIdx * codebookSize * subvectorSize + code * subvectorSize
                
                // Copy centroid values
                for i in 0..<subvectorSize {
                    reconstructed.append(Vector.Scalar(codebook[codebookOffset + i]))
                }
            }
            
            // Handle any remaining dimensions
            while reconstructed.count < dimension {
                reconstructed.append(Vector.Scalar(0))
            }
            
            return reconstructed.withUnsafeBufferPointer { buffer in
                Vector(buffer)
            }
        }
    }
    
    private func generateSyntheticCodebook(
        segments: Int,
        codebookSize: Int,
        subvectorSize: Int
    ) -> [Float] {
        // Generate a synthetic codebook for testing
        // In production, this would be loaded from the trained quantizer
        var codebook: [Float] = []
        
        for segment in 0..<segments {
            for code in 0..<codebookSize {
                for dim in 0..<subvectorSize {
                    // Generate deterministic values based on segment, code, and dimension
                    let value = Float(segment + 1) * 0.1 + Float(code) * 0.01 + Float(dim) * 0.001
                    codebook.append(sin(value) * 0.5) // Bounded synthetic values
                }
            }
        }
        
        return codebook
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
        
        guard !quantizedVectors.isEmpty else {
            return []
        }
        
        // Check if we can use GPU acceleration
        let numVectors = quantizedVectors.count
        if numVectors > 50 && device.capabilities.supportsFloat16 {
            return try await learnedDequantizeGPU(
                quantizedVectors: quantizedVectors,
                targetType: targetType
            )
        }
        
        // CPU fallback for small batches
        return learnedDequantizeCPU(
            quantizedVectors: quantizedVectors,
            targetType: targetType
        )
    }
    
    private func learnedDequantizeGPU<Vector: SIMD & Sendable>(
        quantizedVectors: [QuantizedVector],
        targetType: Vector.Type
    ) async throws -> [Vector] where Vector.Scalar: BinaryFloatingPoint {
        
        let dimension = quantizedVectors.first!.originalDimensions
        let numVectors = quantizedVectors.count
        let numCodewords = quantizedVectors.first!.codes.count
        
        // Extract model ID from metadata
        let modelID: String
        if case .learned(let id) = quantizedVectors.first!.scheme {
            modelID = id ?? "default"
        } else {
            modelID = "default"
        }
        
        // Prepare codes buffer
        var allCodes: [UInt8] = []
        for qv in quantizedVectors {
            allCodes.append(contentsOf: qv.codes)
        }
        let codesBuffer = try await bufferPool.getBuffer(for: allCodes)
        
        // Generate decoder weights (in production, would load from trained model)
        let decoderWeights = generateDecoderWeights(
            modelID: modelID,
            numCodewords: numCodewords,
            outputDimension: dimension
        )
        let weightsBuffer = try await bufferPool.getBuffer(for: decoderWeights)
        
        // Output buffer
        let outputSize = numVectors * dimension * MemoryLayout<Float>.size
        let outputBuffer = try await bufferPool.getBuffer(size: outputSize)
        
        // Get pipeline - using matrix multiplication for decoder
        let pipeline = try await pipelineManager.getPipeline(functionName: "matrixMultiply")
        
        // Execute decoding
        guard let commandBuffer = await device.makeCommandBuffer(),
              let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        computeEncoder.setComputePipelineState(pipeline)
        computeEncoder.setBuffer(codesBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(weightsBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 64, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (numVectors + 63) / 64,
            height: dimension,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read results
        let outputPointer = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        var vectors: [Vector] = []
        
        for i in 0..<numVectors {
            var values = [Vector.Scalar]()
            let startIdx = i * dimension
            
            for j in 0..<dimension {
                values.append(Vector.Scalar(outputPointer[startIdx + j]))
            }
            
            vectors.append(values.withUnsafeBufferPointer { buffer in
                Vector(buffer)
            })
        }
        
        return vectors
    }
    
    private func learnedDequantizeCPU<Vector: SIMD & Sendable>(
        quantizedVectors: [QuantizedVector],
        targetType: Vector.Type
    ) -> [Vector] where Vector.Scalar: BinaryFloatingPoint {
        
        return quantizedVectors.map { quantized in
            let dimension = quantized.originalDimensions
            let codes = quantized.codes
            
            // Extract model ID
            let modelID: String
            if case .learned(let id) = quantized.scheme {
                modelID = id ?? "default"
            } else {
                modelID = "default"
            }
            
            // Generate decoder weights for this model
            let decoderWeights = generateDecoderWeights(
                modelID: modelID,
                numCodewords: codes.count,
                outputDimension: dimension
            )
            
            // Decode: multiply codes by decoder weights
            var reconstructed = [Vector.Scalar]()
            
            for outputIdx in 0..<dimension {
                var value: Float = 0
                
                // Compute weighted sum of codes
                for (codeIdx, code) in codes.enumerated() {
                    let weight = decoderWeights[codeIdx * dimension + outputIdx]
                    value += Float(code) / 255.0 * weight
                }
                
                // Apply activation function (tanh for bounded output)
                value = tanh(value)
                reconstructed.append(Vector.Scalar(value))
            }
            
            return reconstructed.withUnsafeBufferPointer { buffer in
                Vector(buffer)
            }
        }
    }
    
    private func generateDecoderWeights(
        modelID: String,
        numCodewords: Int,
        outputDimension: Int
    ) -> [Float] {
        // Generate synthetic decoder weights
        // In production, these would be loaded from a trained neural decoder
        var weights: [Float] = []
        
        // Use model ID as seed for deterministic generation
        let seed = modelID.hashValue
        var rng = SeededRandom(seed: UInt64(abs(seed)))
        
        for _ in 0..<(numCodewords * outputDimension) {
            // Xavier initialization-style weights
            let scale = sqrt(2.0 / Float(numCodewords + outputDimension))
            let weight = rng.nextGaussian() * scale
            weights.append(weight)
        }
        
        return weights
    }
    
    // Simple seeded random number generator for reproducibility
    private struct SeededRandom {
        private var state: UInt64
        
        init(seed: UInt64) {
            self.state = seed
        }
        
        mutating func next() -> UInt64 {
            state = state &* 2862933555777941757 &+ 3037000493
            return state
        }
        
        mutating func nextFloat() -> Float {
            return Float(next() & 0x7FFFFFFF) / Float(0x7FFFFFFF)
        }
        
        mutating func nextGaussian() -> Float {
            // Box-Muller transform for Gaussian distribution
            let u1 = nextFloat()
            let u2 = nextFloat()
            return sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
        }
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

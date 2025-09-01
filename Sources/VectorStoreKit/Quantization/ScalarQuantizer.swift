// ScalarQuantizer.swift
// VectorStoreKit
//
// Scalar quantization for memory-efficient vector storage

import Foundation
@preconcurrency import Metal
import Accelerate
import os.log

/// Quantization type for different precision/compression trade-offs
public enum ScalarQuantizationType: String, Sendable, Codable {
    case int8 = "int8"       // 4x compression, good accuracy
    case uint8 = "uint8"     // 4x compression, for non-negative values
    case float16 = "float16" // 2x compression, excellent accuracy
    case dynamic = "dynamic" // Adaptive based on value distribution
    
    public var compressionRatio: Float {
        switch self {
        case .int8, .uint8:
            return 4.0
        case .float16:
            return 2.0
        case .dynamic:
            return 0.0 // Determined at runtime
        }
    }
    
    public var bytesPerValue: Int {
        switch self {
        case .int8, .uint8:
            return 1
        case .float16:
            return 2
        case .dynamic:
            return 0 // Determined at runtime
        }
    }
}

/// Statistics for quantization parameters
public struct ScalarQuantizationStatistics: Sendable, Codable {
    public let min: Float
    public let max: Float
    public let mean: Float
    public let stdDev: Float
    public let sparsity: Float // Percentage of near-zero values
    
    /// Recommend optimal quantization type based on statistics
    public var recommendedType: ScalarQuantizationType {
        // If all values are non-negative, use uint8
        if min >= 0 {
            return .uint8
        }
        
        // If range is small and centered, int8 is good
        let range = max - min
        if range < 256 && abs(mean) < 100 {
            return .int8
        }
        
        // For high dynamic range or precision needs
        return .float16
    }
}

/// Quantized vector storage with metadata
public struct QuantizedVectorStore: Sendable, Codable {
    public let quantizedData: Data
    public let quantizationType: ScalarQuantizationType
    public let scale: Float
    public let offset: Float
    public let originalDimension: Int
    public let statistics: ScalarQuantizationStatistics
    
    public var compressionRatio: Float {
        let originalSize = originalDimension * MemoryLayout<Float>.size
        let quantizedSize = quantizedData.count
        return Float(originalSize) / Float(quantizedSize)
    }
}

/// Scalar quantizer for memory-efficient vector storage
public actor ScalarQuantizer {
    
    // MARK: - Properties
    
    private let metalDevice: MetalDevice?
    private let bufferPool: MetalBufferPool?
    private let pipelineManager: MetalPipelineManager?
    
    private var quantizationCache: [String: QuantizationParameters] = [:]
    private let logger = Logger(subsystem: "VectorStoreKit", category: "ScalarQuantizer")
    
    // MARK: - Types
    
    private struct QuantizationParameters {
        let scale: Float
        let offset: Float
        let statistics: ScalarQuantizationStatistics
    }
    
    // MARK: - Initialization
    
    public init(
        metalDevice: MetalDevice? = nil,
        bufferPool: MetalBufferPool? = nil,
        pipelineManager: MetalPipelineManager? = nil
    ) {
        self.metalDevice = metalDevice
        self.bufferPool = bufferPool
        self.pipelineManager = pipelineManager
    }
    
    // MARK: - Quantization
    
    /// Quantize a single vector
    public func quantize(
        vector: [Float],
        type: ScalarQuantizationType = .dynamic
    ) async throws -> QuantizedVectorStore {
        return try await quantizeBatch(
            vectors: [vector],
            type: type
        ).first!
    }
    
    /// Quantize multiple vectors with the same parameters
    public func quantizeBatch(
        vectors: [[Float]],
        type: ScalarQuantizationType = .dynamic
    ) async throws -> [QuantizedVectorStore] {
        guard !vectors.isEmpty else {
            return []
        }
        
        let dimension = vectors[0].count
        guard vectors.allSatisfy({ $0.count == dimension }) else {
            throw ScalarQuantizationError.inconsistentDimensions
        }
        
        // Calculate statistics across all vectors
        let statistics = calculateStatistics(for: vectors)
        let actualType = type == .dynamic ? statistics.recommendedType : type
        
        logger.debug("Quantizing \(vectors.count) vectors to \(String(describing: actualType))")
        
        // Use Metal acceleration if available
        if let metalDevice = metalDevice,
           let bufferPool = bufferPool,
           vectors.count > 100 { // Metal is beneficial for larger batches
            return try await quantizeBatchMetal(
                vectors: vectors,
                type: actualType,
                statistics: statistics
            )
        } else {
            return try await quantizeBatchCPU(
                vectors: vectors,
                type: actualType,
                statistics: statistics
            )
        }
    }
    
    /// Quantize Vector512 optimized type
    public func quantize512(
        vectors: [Vector512],
        type: ScalarQuantizationType = .float16
    ) async throws -> [QuantizedVectorStore] {
        // Convert to arrays for processing
        let arrayVectors = vectors.map { $0.toArray() }
        return try await quantizeBatch(vectors: arrayVectors, type: type)
    }
    
    // MARK: - Dequantization
    
    /// Dequantize a single vector
    public func dequantize(
        _ quantized: QuantizedVectorStore
    ) async throws -> [Float] {
        return try await dequantizeBatch([quantized]).first!
    }
    
    /// Dequantize multiple vectors
    public func dequantizeBatch(
        _ quantized: [QuantizedVectorStore]
    ) async throws -> [[Float]] {
        guard !quantized.isEmpty else {
            return []
        }
        
        // Group by quantization type for efficient processing
        var grouped: [ScalarQuantizationType: [QuantizedVectorStore]] = [:]
        for q in quantized {
            grouped[q.quantizationType, default: []].append(q)
        }
        
        var results: [[Float]] = []
        
        for (type, group) in grouped {
            let dequantized = try await dequantizeGroup(group, type: type)
            results.append(contentsOf: dequantized)
        }
        
        return results
    }
    
    // MARK: - CPU Implementation
    
    private func quantizeBatchCPU(
        vectors: [[Float]],
        type: ScalarQuantizationType,
        statistics: ScalarQuantizationStatistics
    ) async throws -> [QuantizedVectorStore] {
        
        let params = calculateQuantizationParameters(
            type: type,
            statistics: statistics
        )
        
        var results: [QuantizedVectorStore] = []
        
        for vector in vectors {
            let quantized: Data
            
            switch type {
            case .int8:
                quantized = quantizeToInt8(vector, scale: params.scale, offset: params.offset)
            case .uint8:
                quantized = quantizeToUInt8(vector, scale: params.scale, offset: params.offset)
            case .float16:
                quantized = quantizeToFloat16(vector)
            case .dynamic:
                fatalError("Dynamic type should have been resolved")
            }
            
            results.append(QuantizedVectorStore(
                quantizedData: quantized,
                quantizationType: type,
                scale: params.scale,
                offset: params.offset,
                originalDimension: vector.count,
                statistics: statistics
            ))
        }
        
        return results
    }
    
    private func quantizeToInt8(_ vector: [Float], scale: Float, offset: Float) -> Data {
        var mutableScale = scale
        var mutableOffset = offset
        var quantized = [Int8](repeating: 0, count: vector.count)
        
        // vDSP_vsmsa computes: output = input * scale + offset
        var scaledAndOffset = [Float](repeating: 0, count: vector.count)
        vDSP_vsmsa(
            vector, 1,           // Input vector
            &mutableScale,       // Scale (pointer to single value)
            &mutableOffset,      // Offset (pointer to single value)
            &scaledAndOffset, 1, // Output
            vDSP_Length(vector.count)
        )
        
        // Convert float to int8
        vDSP_vfix8(scaledAndOffset, 1, &quantized, 1, vDSP_Length(vector.count))
        
        return Data(bytes: quantized, count: quantized.count)
    }
    
    private func quantizeToUInt8(_ vector: [Float], scale: Float, offset: Float) -> Data {
        var mutableScale = scale
        var mutableOffset = offset
        var scaled = [Float](repeating: 0, count: vector.count)
        var quantized = [UInt8](repeating: 0, count: vector.count)
        
        // Scale and offset
        vDSP_vsmsa(
            vector, 1,
            &mutableScale,
            &mutableOffset,
            &scaled, 1,
            vDSP_Length(vector.count)
        )
        
        // Convert to UInt8
        vDSP_vfixu8(scaled, 1, &quantized, 1, vDSP_Length(vector.count))
        
        return Data(quantized)
    }
    
    private func quantizeToFloat16(_ vector: [Float]) -> Data {
        var float16Values = [UInt16](repeating: 0, count: vector.count)
        
        // Convert Float32 to Float16
        var src = vector
        src.withUnsafeMutableBufferPointer { srcPtr in
            float16Values.withUnsafeMutableBufferPointer { dstPtr in
                var srcBuffer = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!),
                    height: 1,
                    width: vImagePixelCount(vector.count),
                    rowBytes: vector.count * MemoryLayout<Float>.stride
                )
                
                var dstBuffer = vImage_Buffer(
                    data: UnsafeMutableRawPointer(dstPtr.baseAddress!),
                    height: 1,
                    width: vImagePixelCount(vector.count),
                    rowBytes: vector.count * MemoryLayout<UInt16>.stride
                )
                
                vImageConvert_PlanarFtoPlanar16F(&srcBuffer, &dstBuffer, vImage_Flags(kvImageNoFlags))
            }
        }
        
        return Data(bytes: float16Values, count: float16Values.count * 2)
    }
    
    // MARK: - Metal Implementation
    
    private func quantizeBatchMetal(
        vectors: [[Float]],
        type: ScalarQuantizationType,
        statistics: ScalarQuantizationStatistics
    ) async throws -> [QuantizedVectorStore] {
        guard let device = metalDevice,
              let pool = bufferPool,
              let pipelineManager = pipelineManager else {
            // Fallback to CPU
            return try await quantizeBatchCPU(
                vectors: vectors,
                type: type,
                statistics: statistics
            )
        }
        
        // Flatten vectors for GPU processing
        let dimension = vectors[0].count
        let totalElements = vectors.count * dimension
        var flatVectors = [Float](repeating: 0, count: totalElements)
        
        for (i, vector) in vectors.enumerated() {
            let offset = i * dimension
            flatVectors[offset..<offset + dimension] = ArraySlice(vector)
        }
        
        // Create Metal buffers
        let inputBuffer = try await pool.getBuffer(for: flatVectors)
        let outputSize = vectors.count * dimension * type.bytesPerValue
        let outputBuffer = try await pool.getBuffer(size: outputSize)
        
        defer {
            Task {
                await pool.returnBuffer(inputBuffer)
                await pool.returnBuffer(outputBuffer)
            }
        }
        
        // Get appropriate kernel
        let kernelName = getKernelName(for: type)
        let pipeline = try await pipelineManager.getPipeline(functionName: kernelName)
        
        // Execute quantization
        let params = calculateQuantizationParameters(type: type, statistics: statistics)
        try await executeQuantizationKernel(
            pipeline: pipeline,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            params: params,
            vectorCount: vectors.count,
            dimension: dimension
        )
        
        // Extract results
        return extractQuantizedResults(
            from: outputBuffer,
            type: type,
            params: params,
            vectorCount: vectors.count,
            dimension: dimension,
            statistics: statistics
        )
    }
    
    // MARK: - Utilities
    
    private func calculateStatistics(for vectors: [[Float]]) -> ScalarQuantizationStatistics {
        var allValues: [Float] = []
        allValues.reserveCapacity(vectors.count * (vectors.first?.count ?? 0))
        
        for vector in vectors {
            allValues.append(contentsOf: vector)
        }
        
        var min: Float = 0
        var max: Float = 0
        vDSP_minv(allValues, 1, &min, vDSP_Length(allValues.count))
        vDSP_maxv(allValues, 1, &max, vDSP_Length(allValues.count))
        
        var mean: Float = 0
        vDSP_meanv(allValues, 1, &mean, vDSP_Length(allValues.count))
        
        var stdDev: Float = 0
        vDSP_normalize(allValues, 1, nil, 1, &mean, &stdDev, vDSP_Length(allValues.count))
        
        let nearZeroCount = allValues.filter { abs($0) < 0.01 }.count
        let sparsity = Float(nearZeroCount) / Float(allValues.count)
        
        return ScalarQuantizationStatistics(
            min: min,
            max: max,
            mean: mean,
            stdDev: stdDev,
            sparsity: sparsity
        )
    }
    
    private func calculateQuantizationParameters(
        type: ScalarQuantizationType,
        statistics: ScalarQuantizationStatistics
    ) -> QuantizationParameters {
        let scale: Float
        let offset: Float
        
        switch type {
        case .int8:
            let range = statistics.max - statistics.min
            scale = 255.0 / range
            offset = -statistics.min * scale - 128.0
            
        case .uint8:
            let range = statistics.max - statistics.min
            scale = 255.0 / range
            offset = -statistics.min * scale
            
        case .float16:
            scale = 1.0
            offset = 0.0
            
        case .dynamic:
            fatalError("Dynamic type should have been resolved")
        }
        
        return QuantizationParameters(
            scale: scale,
            offset: offset,
            statistics: statistics
        )
    }
    
    private func getKernelName(for type: ScalarQuantizationType) -> String {
        switch type {
        case .int8:
            return "quantizeToInt8"
        case .uint8:
            return "quantizeToUInt8"
        case .float16:
            return "quantizeToFloat16"
        case .dynamic:
            fatalError("Dynamic type should have been resolved")
        }
    }
    
    private func executeQuantizationKernel(
        pipeline: MTLComputePipelineState,
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        params: QuantizationParameters,
        vectorCount: Int,
        dimension: Int
    ) async throws {
        guard let device = metalDevice else { return }
        
        guard let commandBuffer = await device.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ScalarQuantizationError.metalExecutionFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        
        var uniforms = QuantizationUniforms(
            scale: params.scale,
            offset: params.offset,
            dimension: Int32(dimension),
            vectorCount: Int32(vectorCount)
        )
        encoder.setBytes(&uniforms, length: MemoryLayout<QuantizationUniforms>.size, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (vectorCount * dimension + 255) / 256,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
    }
    
    private func extractQuantizedResults(
        from buffer: MTLBuffer,
        type: ScalarQuantizationType,
        params: QuantizationParameters,
        vectorCount: Int,
        dimension: Int,
        statistics: ScalarQuantizationStatistics
    ) -> [QuantizedVectorStore] {
        let bytesPerVector = dimension * type.bytesPerValue
        var results: [QuantizedVectorStore] = []
        
        let contents = buffer.contents()
        
        for i in 0..<vectorCount {
            let offset = i * bytesPerVector
            let data = Data(bytes: contents.advanced(by: offset), count: bytesPerVector)
            
            results.append(QuantizedVectorStore(
                quantizedData: data,
                quantizationType: type,
                scale: params.scale,
                offset: params.offset,
                originalDimension: dimension,
                statistics: statistics
            ))
        }
        
        return results
    }
    
    private func dequantizeGroup(
        _ group: [QuantizedVectorStore],
        type: ScalarQuantizationType
    ) async throws -> [[Float]] {
        // Implementation for dequantization
        var results: [[Float]] = []
        
        for quantized in group {
            let vector: [Float]
            
            switch type {
            case .int8:
                vector = dequantizeFromInt8(quantized)
            case .uint8:
                vector = dequantizeFromUInt8(quantized)
            case .float16:
                vector = dequantizeFromFloat16(quantized)
            case .dynamic:
                fatalError("Dynamic type should have been resolved")
            }
            
            results.append(vector)
        }
        
        return results
    }
    
    private func dequantizeFromInt8(_ quantized: QuantizedVectorStore) -> [Float] {
        let bytes = quantized.quantizedData.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Int8.self))
        }
        var result = [Float](repeating: 0, count: quantized.originalDimension)
        
        // Convert and scale back
        for i in 0..<bytes.count {
            result[i] = (Float(bytes[i]) - quantized.offset) / quantized.scale
        }
        
        return result
    }
    
    private func dequantizeFromUInt8(_ quantized: QuantizedVectorStore) -> [Float] {
        let bytes = [UInt8](quantized.quantizedData)
        var result = [Float](repeating: 0, count: quantized.originalDimension)
        
        for i in 0..<bytes.count {
            result[i] = (Float(bytes[i]) - quantized.offset) / quantized.scale
        }
        
        return result
    }
    
    private func dequantizeFromFloat16(_ quantized: QuantizedVectorStore) -> [Float] {
        let uint16Array = quantized.quantizedData.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: UInt16.self))
        }
        
        var result = [Float](repeating: 0, count: quantized.originalDimension)
        
        uint16Array.withUnsafeBufferPointer { srcPtr in
            result.withUnsafeMutableBufferPointer { dstPtr in
                var srcBuffer = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: srcPtr.baseAddress!),
                    height: 1,
                    width: vImagePixelCount(quantized.originalDimension),
                    rowBytes: quantized.originalDimension * MemoryLayout<UInt16>.stride
                )
                
                var dstBuffer = vImage_Buffer(
                    data: UnsafeMutableRawPointer(dstPtr.baseAddress!),
                    height: 1,
                    width: vImagePixelCount(quantized.originalDimension),
                    rowBytes: quantized.originalDimension * MemoryLayout<Float>.stride
                )
                
                vImageConvert_Planar16FtoPlanarF(&srcBuffer, &dstBuffer, vImage_Flags(kvImageNoFlags))
            }
        }
        
        return result
    }
}

// MARK: - Supporting Types

private struct QuantizationUniforms {
    let scale: Float
    let offset: Float
    let dimension: Int32
    let vectorCount: Int32
}

public enum ScalarQuantizationError: LocalizedError {
    case inconsistentDimensions
    case unsupportedType
    case metalExecutionFailed
    
    public var errorDescription: String? {
        switch self {
        case .inconsistentDimensions:
            return "All vectors must have the same dimension"
        case .unsupportedType:
            return "Unsupported quantization type"
        case .metalExecutionFailed:
            return "Metal execution failed"
        }
    }
}

// MARK: - Extensions

extension ScalarQuantizer {
    /// Calculate memory savings for a dataset
    public func calculateMemorySavings(
        vectorCount: Int,
        dimension: Int,
        quantizationType: ScalarQuantizationType
    ) -> (originalSize: Int, quantizedSize: Int, savedBytes: Int, compressionRatio: Float) {
        let originalSize = vectorCount * dimension * MemoryLayout<Float>.size
        let quantizedSize = vectorCount * dimension * quantizationType.bytesPerValue
        let savedBytes = originalSize - quantizedSize
        let compressionRatio = Float(originalSize) / Float(quantizedSize)
        
        return (originalSize, quantizedSize, savedBytes, compressionRatio)
    }
}
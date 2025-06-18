// VectorStoreKit: Quantization Compression
//
// Efficient quantization-based compression for ML model parameters
//

import Foundation
import Accelerate
@preconcurrency import Metal

/// Quantization-based compression for model parameters
public actor QuantizationCompressor {
    private let device: MTLDevice
    private let quantizer: ScalarQuantizer
    
    public init(device: MTLDevice) async throws {
        self.device = device
        // Initialize without Metal acceleration for now - CPU quantization is sufficient
        self.quantizer = ScalarQuantizer()
    }
    
    /// Compress parameter data using quantization
    public func compress(
        data: Data,
        shape: [Int],
        quantizationType: ScalarQuantizationType = .dynamic
    ) async throws -> CompressedParameter {
        // Convert data to float array
        let floatCount = data.count / MemoryLayout<Float>.stride
        let floatArray = data.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self).prefix(floatCount))
        }
        
        // Quantize the data
        let quantized = try await quantizer.quantize(
            vector: floatArray,
            type: quantizationType
        )
        
        // Package the result
        return CompressedParameter(
            quantizedStore: quantized,
            originalShape: shape,
            compressionRatio: quantized.compressionRatio
        )
    }
    
    /// Decompress quantized parameter data
    public func decompress(
        _ compressed: CompressedParameter
    ) async throws -> Data {
        // Dequantize the data
        let floatArray = try await quantizer.dequantize(compressed.quantizedStore)
        
        // Convert back to Data
        return floatArray.withUnsafeBytes { bytes in
            Data(bytes: bytes.baseAddress!, count: bytes.count)
        }
    }
    
    /// Batch compress multiple parameters
    public func compressBatch(
        parameters: [(name: String, data: Data, shape: [Int])],
        quantizationType: ScalarQuantizationType = .dynamic
    ) async throws -> [String: CompressedParameter] {
        var compressed: [String: CompressedParameter] = [:]
        
        // Group by similar sizes for efficient batch processing
        var sizeGroups: [Int: [(name: String, data: Data, shape: [Int])]] = [:]
        
        for param in parameters {
            let size = param.data.count
            sizeGroups[size, default: []].append(param)
        }
        
        // Process each size group
        for (_, group) in sizeGroups {
            // Convert group to float arrays
            let floatArrays = group.map { param in
                let floatCount = param.data.count / MemoryLayout<Float>.stride
                return param.data.withUnsafeBytes { bytes in
                    Array(bytes.bindMemory(to: Float.self).prefix(floatCount))
                }
            }
            
            // Batch quantize
            let quantizedBatch = try await quantizer.quantizeBatch(
                vectors: floatArrays,
                type: quantizationType
            )
            
            // Store results
            for (param, quantized) in zip(group, quantizedBatch) {
                compressed[param.name] = CompressedParameter(
                    quantizedStore: quantized,
                    originalShape: param.shape,
                    compressionRatio: quantized.compressionRatio
                )
            }
        }
        
        return compressed
    }
    
    /// Estimate compression savings
    public func estimateCompressionSavings(
        parameterSizes: [Int],
        quantizationType: ScalarQuantizationType
    ) -> CompressionEstimate {
        let totalOriginalSize = parameterSizes.reduce(0, +)
        let avgCompressionRatio = quantizationType.compressionRatio
        let estimatedCompressedSize = Int(Float(totalOriginalSize) / avgCompressionRatio)
        let savedBytes = totalOriginalSize - estimatedCompressedSize
        let savingsPercentage = Float(savedBytes) / Float(totalOriginalSize) * 100
        
        return CompressionEstimate(
            originalSize: totalOriginalSize,
            compressedSize: estimatedCompressedSize,
            savedBytes: savedBytes,
            savingsPercentage: savingsPercentage,
            compressionRatio: avgCompressionRatio
        )
    }
}

/// Compressed parameter container
public struct CompressedParameter: Codable, Sendable {
    public let quantizedStore: QuantizedVectorStore
    public let originalShape: [Int]
    public let compressionRatio: Float
}

/// Compression estimate results
public struct CompressionEstimate: Sendable {
    public let originalSize: Int
    public let compressedSize: Int
    public let savedBytes: Int
    public let savingsPercentage: Float
    public let compressionRatio: Float
}

// MARK: - Integration with ModelSerialization

// Removed extension - quantization is handled directly in CheckpointSerializer now

// MARK: - Helpers removed - using CPU-based quantization

// MARK: - Vector512 Support

extension ScalarQuantizer {
    /// Optimized quantization for Vector512 types
    public func quantize512(
        _ vector: Vector512,
        type: ScalarQuantizationType = .float16
    ) async throws -> QuantizedVectorStore {
        let floatArray = vector.toArray()
        return try await quantize(vector: floatArray, type: type)
    }
    
    /// Batch quantization for Vector512 types
    public func quantizeBatch512(
        _ vectors: [Vector512],
        type: ScalarQuantizationType = .float16
    ) async throws -> [QuantizedVectorStore] {
        let floatArrays = vectors.map { $0.toArray() }
        return try await quantizeBatch(vectors: floatArrays, type: type)
    }
}
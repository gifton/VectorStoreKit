// VectorStoreKit: Optimized Product Quantizer
//
// Optimized product quantization with rotation for better compression

import Foundation

// MARK: - Optimized Product Quantization

/// Optimized PQ with rotation for better quantization
public actor OptimizedProductQuantizer {
    private let baseQuantizer: ProductQuantizer
    private var rotationMatrix: [[Float]]?
    
    public var config: ProductQuantizationConfig {
        get async { baseQuantizer.config }
    }
    
    public init(configuration: ProductQuantizationConfig) async throws {
        self.baseQuantizer = try ProductQuantizer(config: configuration)
    }
    
    /// Learn optimal rotation for better quantization
    public func learnRotation(vectors: [[Float]]) async throws {
        // Simplified OPQ implementation
        // In practice, would learn rotation matrix that minimizes quantization error
        let dimensions = vectors.first?.count ?? 0
        rotationMatrix = Array(
            repeating: Array(repeating: Float(0), count: dimensions),
            count: dimensions
        )
        
        // Identity matrix for now
        for i in 0..<dimensions {
            rotationMatrix![i][i] = 1.0
        }
    }
    
    /// Train the quantizer with rotation
    public func train(on vectors: [[Float]]) async throws {
        // Apply rotation if learned
        let rotatedVectors = vectors.map { applyRotation($0) }
        try await baseQuantizer.train(on: rotatedVectors)
    }
    
    /// Encode vector with rotation
    public func encode(_ vector: [Float]) async throws -> QuantizedVector {
        let rotated = applyRotation(vector)
        return try await baseQuantizer.encode(rotated)
    }
    
    /// Decode vector with inverse rotation
    public func decode(_ quantized: QuantizedVector) async throws -> [Float] {
        let decoded = try await baseQuantizer.decode(quantized)
        return applyInverseRotation(decoded)
    }
    
    /// Compute distance with rotation consideration
    public func computeDistance(
        query: [Float],
        quantized: QuantizedVector
    ) async throws -> Float {
        let rotatedQuery = applyRotation(query)
        return try await baseQuantizer.computeDistance(
            query: rotatedQuery,
            quantized: quantized
        )
    }
    
    /// Precompute distance tables for rotated query
    public func precomputeDistanceTables(query: [Float]) async throws -> DistanceTables {
        let rotatedQuery = applyRotation(query)
        return try await baseQuantizer.precomputeDistanceTables(query: rotatedQuery)
    }
    
    /// Analyze quantization error with rotation
    public func analyzeQuantizationError(testVectors: [[Float]]) async throws -> QuantizationAnalysis {
        let rotatedVectors = testVectors.map { applyRotation($0) }
        return try await baseQuantizer.analyzeQuantizationError(testVectors: rotatedVectors)
    }
    
    // MARK: - Private Methods
    
    private func applyRotation(_ vector: [Float]) -> [Float] {
        guard let matrix = rotationMatrix else { return vector }
        
        var rotated = Array(repeating: Float(0), count: vector.count)
        for i in 0..<vector.count {
            for j in 0..<vector.count {
                rotated[i] += matrix[i][j] * vector[j]
            }
        }
        return rotated
    }
    
    private func applyInverseRotation(_ vector: [Float]) -> [Float] {
        guard let matrix = rotationMatrix else { return vector }
        
        // For orthogonal matrices, inverse = transpose
        // This is a simplified implementation
        var result = Array(repeating: Float(0), count: vector.count)
        for i in 0..<vector.count {
            for j in 0..<vector.count {
                result[i] += matrix[j][i] * vector[j]
            }
        }
        return result
    }
}
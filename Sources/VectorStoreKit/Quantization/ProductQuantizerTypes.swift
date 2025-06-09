// VectorStoreKit: Product Quantizer Types
//
// Supporting types for product quantization

import Foundation

// MARK: - Supporting Types

/// Precomputed distance tables for fast search
public struct DistanceTables: Sendable {
    public let tables: [[Float]]  // [segments x codebookSize]
    
    /// Compute distance using precomputed tables
    public func computeDistance(codes: [UInt8]) -> Float {
        var distance: Float = 0
        for (segmentIdx, code) in codes.enumerated() {
            distance += tables[segmentIdx][Int(code)]
        }
        return sqrt(distance)
    }
}

/// Quantization analysis results
public struct QuantizationAnalysis: Sendable {
    public let averageError: Float
    public let maxError: Float
    public let errorVariance: Float
    public let compressionRatio: Float
    public let memoryUsage: Int
}

/// Quantization errors
public enum QuantizationError: LocalizedError {
    case notTrained
    case insufficientTrainingData(required: Int, provided: Int)
    case dimensionMismatch
    case invalidSegmentation(dimensions: Int, segments: Int)
    case invalidCodeSize(Int)
    case invalidQuantizedVector
    
    public var errorDescription: String? {
        switch self {
        case .notTrained:
            return "Product quantizer must be trained before use"
        case .insufficientTrainingData(let required, let provided):
            return "Insufficient training data: \(provided) vectors provided, \(required) required"
        case .dimensionMismatch:
            return "Vector dimensions do not match configuration"
        case .invalidSegmentation(let dimensions, let segments):
            return "Dimensions \(dimensions) not divisible by \(segments) segments"
        case .invalidCodeSize(let size):
            return "Invalid code size: \(size). Must be between 4 and 16"
        case .invalidQuantizedVector:
            return "Invalid quantized vector format"
        }
    }
}
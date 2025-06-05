// VectorStoreKit: Multi-Index Hash Configuration
//
// Configuration for hash-based retrieval

import Foundation

/// Configuration for multi-index hashing
public struct MultiIndexHashConfiguration: Sendable {
    public let dimensions: Int
    public let numHashTables: Int
    public let hashLength: Int          // Number of bits per hash
    public let projectionType: ProjectionType
    public let amplificationFactor: Int // For multi-probe
    public let rerank: Bool
    
    public enum ProjectionType: Sendable {
        case random
        case learned(modelPath: String)
        case spherical
        case crossPolytope
        case pStable(p: Float)
    }
    
    public init(
        dimensions: Int,
        numHashTables: Int = 32,
        hashLength: Int = 16,
        projectionType: ProjectionType = .spherical,
        amplificationFactor: Int = 10,
        rerank: Bool = true
    ) {
        self.dimensions = dimensions
        self.numHashTables = numHashTables
        self.hashLength = hashLength
        self.projectionType = projectionType
        self.amplificationFactor = amplificationFactor
        self.rerank = rerank
    }
    
    public func validate() throws {
        guard dimensions > 0 else {
            throw HashingError.invalidDimensions(dimensions)
        }
        guard numHashTables > 0 else {
            throw HashingError.invalidParameter("numHashTables", numHashTables)
        }
        guard hashLength > 0 && hashLength <= 64 else {
            throw HashingError.invalidParameter("hashLength", hashLength)
        }
    }
}

/// Hashing errors  
public enum HashingError: Error, LocalizedError {
    case invalidDimensions(Int)
    case invalidParameter(String, Int)
    case initializationFailed(String)
    case dimensionMismatch
    case modelLoadFailed
    case modelDimensionMismatch(expected: Int, actual: Int)
    case modelHashLengthMismatch(expected: Int, actual: Int)
    
    public var errorDescription: String? {
        switch self {
        case .invalidDimensions(let dims):
            return "Invalid dimensions: \(dims)"
        case .invalidParameter(let name, let value):
            return "Invalid parameter \(name): \(value)"
        case .initializationFailed(let reason):
            return "Initialization failed: \(reason)"
        case .dimensionMismatch:
            return "Vector dimensions do not match configuration"
        case .modelLoadFailed:
            return "Failed to load hash model"
        case .modelDimensionMismatch(let expected, let actual):
            return "Model dimension mismatch: expected \(expected), got \(actual)"
        case .modelHashLengthMismatch(let expected, let actual):
            return "Model hash length mismatch: expected \(expected), got \(actual)"
        }
    }
}
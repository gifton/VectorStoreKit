import Foundation
import Accelerate

/// Simple vector type for LLM embeddings
public struct EmbeddingVector: Codable, Sendable, Equatable {
    /// Vector dimensions
    public let dimensions: Int
    
    /// Vector data
    public var data: [Float]
    
    /// Initialize with dimensions
    public init(dimensions: Int) {
        self.dimensions = dimensions
        self.data = Array(repeating: 0.0, count: dimensions)
    }
    
    /// Initialize with data
    public init(data: [Float]) {
        self.dimensions = data.count
        self.data = data
    }
    
    /// Normalize the vector to unit length
    public mutating func normalize() {
        let magnitude = sqrt(data.reduce(0) { $0 + $1 * $1 })
        guard magnitude > 0 else { return }
        
        for i in 0..<data.count {
            data[i] /= magnitude
        }
    }
    
    /// Compute dot product with another vector
    public func dot(_ other: EmbeddingVector) -> Float {
        guard dimensions == other.dimensions else { return 0 }
        
        var result: Float = 0
        vDSP_dotpr(data, 1, other.data, 1, &result, vDSP_Length(dimensions))
        return result
    }
    
    /// Compute cosine similarity with another vector
    public func cosineSimilarity(_ other: EmbeddingVector) -> Float {
        let dotProduct = self.dot(other)
        let magnitudeSelf = sqrt(data.reduce(0) { $0 + $1 * $1 })
        let magnitudeOther = sqrt(other.data.reduce(0) { $0 + $1 * $1 })
        
        guard magnitudeSelf > 0 && magnitudeOther > 0 else { return 0 }
        return dotProduct / (magnitudeSelf * magnitudeOther)
    }
    
    /// Compute Euclidean distance to another vector
    public func euclideanDistance(_ other: EmbeddingVector) -> Float {
        guard dimensions == other.dimensions else { return Float.infinity }
        
        var sum: Float = 0
        for i in 0..<dimensions {
            let diff = data[i] - other.data[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
}

// MARK: - Vector Operations

public extension EmbeddingVector {
    /// Add two vectors
    static func + (lhs: EmbeddingVector, rhs: EmbeddingVector) -> EmbeddingVector {
        guard lhs.dimensions == rhs.dimensions else {
            fatalError("Vector dimensions must match")
        }
        
        var result = EmbeddingVector(dimensions: lhs.dimensions)
        vDSP_vadd(lhs.data, 1, rhs.data, 1, &result.data, 1, vDSP_Length(lhs.dimensions))
        return result
    }
    
    /// Subtract two vectors
    static func - (lhs: EmbeddingVector, rhs: EmbeddingVector) -> EmbeddingVector {
        guard lhs.dimensions == rhs.dimensions else {
            fatalError("Vector dimensions must match")
        }
        
        var result = EmbeddingVector(dimensions: lhs.dimensions)
        vDSP_vsub(rhs.data, 1, lhs.data, 1, &result.data, 1, vDSP_Length(lhs.dimensions))
        return result
    }
    
    /// Scale vector by scalar
    static func * (vector: EmbeddingVector, scalar: Float) -> EmbeddingVector {
        var result = EmbeddingVector(dimensions: vector.dimensions)
        vDSP_vsmul(vector.data, 1, [scalar], &result.data, 1, vDSP_Length(vector.dimensions))
        return result
    }
}

// MARK: - Distance Metrics

/// Common distance metrics for vector comparison
public enum EmbeddingDistanceMetric: String, Codable, CaseIterable {
    case euclidean
    case cosine
    case dotProduct
    case manhattan
    
    /// Compute distance between two vectors
    public func distance(_ a: EmbeddingVector, _ b: EmbeddingVector) -> Float {
        switch self {
        case .euclidean:
            return a.euclideanDistance(b)
        case .cosine:
            return 1.0 - a.cosineSimilarity(b)
        case .dotProduct:
            return -a.dot(b) // Negative so smaller is better
        case .manhattan:
            guard a.dimensions == b.dimensions else { return Float.infinity }
            var sum: Float = 0
            for i in 0..<a.dimensions {
                sum += abs(a.data[i] - b.data[i])
            }
            return sum
        }
    }
    
    /// Convert distance to similarity score (0-1, where 1 is most similar)
    public func toSimilarity(_ distance: Float) -> Float {
        switch self {
        case .cosine:
            return 1.0 - distance
        case .euclidean, .manhattan:
            return 1.0 / (1.0 + distance)
        case .dotProduct:
            return max(0, -distance) // Negative distance becomes positive similarity
        }
    }
}
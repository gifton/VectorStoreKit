// VectorStoreKit: Distance Computation Foundation
//
// High-performance distance computation implementations

import Foundation
import simd
import Accelerate
import Metal

// MARK: - Distance Computation Protocol

/// Protocol for distance computation strategies
public protocol DistanceComputation: Sendable {
    /// Compute distance between two vectors
    func distance<Vector: SIMD>(_ a: Vector, _ b: Vector) -> Float 
    where Vector.Scalar: BinaryFloatingPoint
    
    /// Compute distances between a query and multiple vectors (batch operation)
    func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint
    
    /// Metric characteristics
    var isMetric: Bool { get }
    var supportsTriangleInequality: Bool { get }
    var name: String { get }
}

// MARK: - Euclidean Distance

/// Euclidean (L2) distance implementation
public struct EuclideanDistance: DistanceComputation {
    public let isMetric = true
    public let supportsTriangleInequality = true
    public let name = "euclidean"
    
    public init() {}
    
    public func distance<Vector: SIMD>(_ a: Vector, _ b: Vector) -> Float 
    where Vector.Scalar: BinaryFloatingPoint {
        let diff = a - b
        let squared = diff * diff
        return Float(sqrt(squared.sum()))
    }
    
    public func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        // Use Accelerate for batch computation when possible
        if Vector.self == SIMD32<Float>.self {
            return accelerateBatchDistance(
                query: query as! SIMD32<Float>,
                vectors: vectors as! [SIMD32<Float>]
            )
        }
        
        // Fallback to scalar computation
        return vectors.map { distance(query, $0) }
    }
    
    private func accelerateBatchDistance(
        query: SIMD32<Float>,
        vectors: [SIMD32<Float>]
    ) -> [Float] {
        let count = vectors.count
        let dimensions = 32
        var results = [Float](repeating: 0, count: count)
        
        // Flatten vectors for Accelerate
        let flatVectors = vectors.flatMap { vector in
            (0..<dimensions).map { Float(vector[$0]) }
        }
        
        let queryArray = (0..<dimensions).map { Float(query[$0]) }
        
        // Compute distances using Accelerate
        for i in 0..<count {
            let vectorStart = i * dimensions
            var diff = [Float](repeating: 0, count: dimensions)
            
            flatVectors.withUnsafeBufferPointer { flatBuffer in
                let vectorPointer = flatBuffer.baseAddress! + vectorStart
                vDSP_vsub(
                    vectorPointer,
                    1,
                    queryArray,
                    1,
                    &diff,
                    1,
                    vDSP_Length(dimensions)
                )
            }
            
            var squaredSum: Float = 0
            vDSP_dotpr(diff, 1, diff, 1, &squaredSum, vDSP_Length(dimensions))
            
            results[i] = sqrt(squaredSum)
        }
        
        return results
    }
}

// MARK: - Cosine Distance

/// Cosine distance (1 - cosine similarity) implementation
public struct CosineDistance: DistanceComputation {
    public let isMetric = false // Not a true metric
    public let supportsTriangleInequality = false
    public let name = "cosine"
    
    public init() {}
    
    public func distance<Vector: SIMD>(_ a: Vector, _ b: Vector) -> Float 
    where Vector.Scalar: BinaryFloatingPoint {
        let dotProduct = (a * b).sum()
        let normA = sqrt((a * a).sum())
        let normB = sqrt((b * b).sum())
        
        guard normA > 0 && normB > 0 else { return 1.0 }
        
        let similarity = dotProduct / (normA * normB)
        return Float(1.0 - max(-1.0, min(1.0, similarity)))
    }
    
    public func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        let queryNorm = sqrt((query * query).sum())
        guard queryNorm > 0 else { return [Float](repeating: 1.0, count: vectors.count) }
        
        return vectors.map { vector in
            let dotProduct = (query * vector).sum()
            let vectorNorm = sqrt((vector * vector).sum())
            
            guard vectorNorm > 0 else { return Float(1.0) }
            
            let similarity = dotProduct / (queryNorm * vectorNorm)
            return Float(1.0 - max(-1.0, min(1.0, similarity)))
        }
    }
}

// MARK: - Dot Product Distance

/// Negative dot product as distance (for maximum inner product search)
public struct DotProductDistance: DistanceComputation {
    public let isMetric = false
    public let supportsTriangleInequality = false
    public let name = "dot_product"
    
    public init() {}
    
    public func distance<Vector: SIMD>(_ a: Vector, _ b: Vector) -> Float 
    where Vector.Scalar: BinaryFloatingPoint {
        return Float(-(a * b).sum())
    }
    
    public func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        return vectors.map { Float(-(query * $0).sum()) }
    }
}

// MARK: - Manhattan Distance

/// Manhattan (L1) distance implementation
public struct ManhattanDistance: DistanceComputation {
    public let isMetric = true
    public let supportsTriangleInequality = true
    public let name = "manhattan"
    
    public init() {}
    
    public func distance<Vector: SIMD>(_ a: Vector, _ b: Vector) -> Float 
    where Vector.Scalar: BinaryFloatingPoint {
        let diff = a - b
        var sum: Vector.Scalar = 0
        for i in 0..<a.scalarCount {
            sum += abs(diff[i])
        }
        return Float(sum)
    }
    
    public func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        return vectors.map { distance(query, $0) }
    }
}

// MARK: - Hamming Distance

/// Hamming distance for binary vectors
public struct HammingDistance: DistanceComputation {
    public let isMetric = true
    public let supportsTriangleInequality = true
    public let name = "hamming"
    
    public init() {}
    
    public func distance<Vector: SIMD>(_ a: Vector, _ b: Vector) -> Float 
    where Vector.Scalar: BinaryFloatingPoint {
        var count = 0
        for i in 0..<a.scalarCount {
            if a[i] != b[i] {
                count += 1
            }
        }
        return Float(count)
    }
    
    public func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        return vectors.map { distance(query, $0) }
    }
}

// MARK: - Distance Computation Engine

/// High-performance distance computation engine with hardware acceleration
public actor DistanceComputationEngine {
    private let metalDevice: MTLDevice?
    private let metalQueue: MTLCommandQueue?
    private var computePipelines: [String: MTLComputePipelineState] = [:]
    
    /// Preferred computation backend
    public enum Backend: String, Sendable {
        case cpu = "cpu"
        case metal = "metal"
        case accelerate = "accelerate"
        case auto = "auto"
    }
    
    private let preferredBackend: Backend
    
    public init(backend: Backend = .auto) async throws {
        self.preferredBackend = backend
        
        // Initialize Metal if available and requested
        if backend == .metal || backend == .auto {
            self.metalDevice = MTLCreateSystemDefaultDevice()
            self.metalQueue = metalDevice?.makeCommandQueue()
            
            if metalDevice != nil {
                try await setupMetalPipelines()
            }
        } else {
            self.metalDevice = nil
            self.metalQueue = nil
        }
    }
    
    /// Compute distances with automatic backend selection
    public func computeDistances<Vector: SIMD>(
        query: Vector,
        vectors: [Vector],
        metric: DistanceComputation
    ) async throws -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        // Select optimal backend based on vector count and dimensions
        let backend = selectBackend(vectorCount: vectors.count, dimensions: query.scalarCount)
        
        switch backend {
        case .metal:
            if let metalDevice = metalDevice,
               let metalQueue = metalQueue,
               let pipeline = computePipelines[metric.name] {
                return try await computeWithMetal(
                    query: query,
                    vectors: vectors,
                    metric: metric,
                    device: metalDevice,
                    queue: metalQueue,
                    pipeline: pipeline
                )
            }
            fallthrough
            
        case .accelerate:
            if Vector.self == SIMD32<Float>.self && metric.name == "euclidean" {
                return EuclideanDistance().batchDistance(query: query, vectors: vectors)
            }
            fallthrough
            
        case .cpu, .auto:
            return metric.batchDistance(query: query, vectors: vectors)
        }
    }
    
    /// Find k nearest neighbors
    public func findKNearest<Vector: SIMD>(
        query: Vector,
        vectors: [(id: VectorID, vector: Vector)],
        k: Int,
        metric: DistanceComputation
    ) async throws -> [(id: VectorID, distance: Float)] where Vector.Scalar: BinaryFloatingPoint {
        let distances = try await computeDistances(
            query: query,
            vectors: vectors.map { $0.vector },
            metric: metric
        )
        
        // Create tuples of (id, distance)
        let results = zip(vectors.map { $0.id }, distances)
            .map { (id: $0.0, distance: $0.1) }
        
        // Sort by distance and take top k
        return Array(results.sorted { $0.distance < $1.distance }.prefix(k))
    }
    
    // MARK: - Private Methods
    
    private func selectBackend(vectorCount: Int, dimensions: Int) -> Backend {
        guard preferredBackend == .auto else { return preferredBackend }
        
        // Heuristics for backend selection
        let totalOperations = vectorCount * dimensions
        
        if metalDevice != nil && totalOperations > 100_000 {
            return .metal
        } else if dimensions == 32 || dimensions == 64 {
            return .accelerate
        } else {
            return .cpu
        }
    }
    
    private func setupMetalPipelines() async throws {
        // In a real implementation, this would compile Metal shaders
        // for different distance metrics
    }
    
    private func computeWithMetal<Vector: SIMD>(
        query: Vector,
        vectors: [Vector],
        metric: DistanceComputation,
        device: MTLDevice,
        queue: MTLCommandQueue,
        pipeline: MTLComputePipelineState
    ) async throws -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        // Metal GPU computation implementation
        // For now, fallback to CPU
        return metric.batchDistance(query: query, vectors: vectors)
    }
}

// MARK: - SIMD Extensions

extension SIMD where Scalar: BinaryFloatingPoint {
    /// Sum all components of the vector
    func sum() -> Scalar {
        var result: Scalar = 0
        for i in 0..<scalarCount {
            result += self[i]
        }
        return result
    }
}

// MARK: - Additional Distance Metrics

/// Chebyshev distance (L∞ norm)
public struct ChebyshevDistance: DistanceComputation {
    public let isMetric = true
    public let supportsTriangleInequality = true
    public let name = "chebyshev"
    
    public init() {}
    
    public func distance<Vector>(_ a: Vector, _ b: Vector) -> Float where Vector: SIMD, Vector.Scalar: BinaryFloatingPoint {
        let diff = a - b
        return diff.indices.reduce(Float(0)) { max($0, abs(Float(diff[$1]))) }
    }
    
    public func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        return vectors.map { distance(query, $0) }
    }
}

/// Minkowski distance (Lp norm)
public struct MinkowskiDistance: DistanceComputation {
    public let isMetric = true
    public let supportsTriangleInequality = true
    public let name = "minkowski"
    
    let p: Float
    
    public init(p: Float = 2.0) {
        self.p = p
    }
    
    public func distance<Vector>(_ a: Vector, _ b: Vector) -> Float where Vector: SIMD, Vector.Scalar: BinaryFloatingPoint {
        let diff = a - b
        let sum = diff.indices.reduce(Float(0)) { $0 + pow(abs(Float(diff[$1])), p) }
        return pow(sum, 1.0/p)
    }
    
    public func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        return vectors.map { distance(query, $0) }
    }
}

/// Jaccard distance for binary vectors
public struct JaccardDistance: DistanceComputation {
    public let isMetric = true
    public let supportsTriangleInequality = true
    public let name = "jaccard"
    
    public init() {}
    
    public func distance<Vector>(_ a: Vector, _ b: Vector) -> Float where Vector: SIMD, Vector.Scalar: BinaryFloatingPoint {
        var intersection: Float = 0
        var union: Float = 0
        
        for i in a.indices {
            let aVal = Float(a[i]) > 0.5 ? 1.0 : 0.0
            let bVal = Float(b[i]) > 0.5 ? 1.0 : 0.0
            
            if aVal == 1.0 || bVal == 1.0 {
                union += 1.0
                if aVal == 1.0 && bVal == 1.0 {
                    intersection += 1.0
                }
            }
        }
        
        return union > 0 ? 1.0 - (intersection / union) : 0.0
    }
    
    public func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        return vectors.map { distance(query, $0) }
    }
}

/// Mahalanobis distance (requires covariance matrix)
public struct MahalanobisDistance: DistanceComputation {
    public let isMetric = true
    public let supportsTriangleInequality = true
    public let name = "mahalanobis"
    
    public init() {}
    
    // Simplified implementation - would need covariance matrix in practice
    public func distance<Vector>(_ a: Vector, _ b: Vector) -> Float where Vector: SIMD, Vector.Scalar: BinaryFloatingPoint {
        // For now, fall back to Euclidean
        return EuclideanDistance().distance(a, b)
    }
    
    public func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        return vectors.map { distance(query, $0) }
    }
}

/// Earth Mover's Distance (Wasserstein distance)
public struct EarthMoverDistance: DistanceComputation {
    public let isMetric = true
    public let supportsTriangleInequality = true
    public let name = "earth_mover"
    
    public init() {}
    
    // Simplified implementation
    public func distance<Vector>(_ a: Vector, _ b: Vector) -> Float where Vector: SIMD, Vector.Scalar: BinaryFloatingPoint {
        // For now, use L1 distance as approximation
        return ManhattanDistance().distance(a, b)
    }
    
    public func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        return vectors.map { distance(query, $0) }
    }
}

/// Learned distance metric
public struct LearnedDistance: DistanceComputation {
    public let isMetric = false
    public let supportsTriangleInequality = false
    public let name = "learned"
    
    let modelName: String
    
    public init(modelName: String) {
        self.modelName = modelName
    }
    
    public func distance<Vector>(_ a: Vector, _ b: Vector) -> Float where Vector: SIMD, Vector.Scalar: BinaryFloatingPoint {
        // Would use ML model in practice
        return EuclideanDistance().distance(a, b)
    }
    
    public func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        return vectors.map { distance(query, $0) }
    }
}

/// Adaptive distance metric
public struct AdaptiveDistance: DistanceComputation {
    public let isMetric = false
    public let supportsTriangleInequality = false
    public let name = "adaptive"
    
    public init() {}
    
    public func distance<Vector>(_ a: Vector, _ b: Vector) -> Float where Vector: SIMD, Vector.Scalar: BinaryFloatingPoint {
        // Adaptively choose between metrics based on data characteristics
        let cosineScore = CosineDistance().distance(a, b)
        let euclideanScore = EuclideanDistance().distance(a, b)
        
        // Simple adaptive logic - would be more sophisticated in practice
        return min(cosineScore, euclideanScore)
    }
    
    public func batchDistance<Vector: SIMD>(
        query: Vector,
        vectors: [Vector]
    ) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        return vectors.map { distance(query, $0) }
    }
}

// MARK: - Distance Metric Factory

/// Factory for creating distance metric instances
public struct DistanceMetricFactory {
    public static func create(from type: DistanceMetric) -> any DistanceComputation {
        switch type {
        case .euclidean:
            return EuclideanDistance()
        case .cosine:
            return CosineDistance()
        case .dotProduct:
            return DotProductDistance()
        case .manhattan:
            return ManhattanDistance()
        case .hamming:
            return HammingDistance()
        case .chebyshev:
            return ChebyshevDistance()
        case .minkowski:
            return MinkowskiDistance(p: 2.0)  // Default to L2 norm
        case .jaccard:
            return JaccardDistance()
        case .mahalanobis:
            return MahalanobisDistance()
        case .earth_mover:
            return EarthMoverDistance()
        case .learned:
            return LearnedDistance(modelName: "default")
        case .adaptive:
            return AdaptiveDistance()
        }
    }
}
// VectorStoreKit: K-means Clustering
//
// Metal-accelerated K-means clustering for IVF index

import Foundation
import simd

/// Result of K-means clustering
public struct ClusteringResult: Sendable {
    public let centroids: [[Float]]
    public let assignments: [Int]
    public let iterations: Int
    public let converged: Bool
    public let inertia: Float
    
    public init(
        centroids: [[Float]],
        assignments: [Int],
        iterations: Int,
        converged: Bool,
        inertia: Float
    ) {
        self.centroids = centroids
        self.assignments = assignments
        self.iterations = iterations
        self.converged = converged
        self.inertia = inertia
    }
}

/// K-means clustering implementation
public actor KMeansClustering {
    private let configuration: KMeansClusteringConfiguration
    private let metalCompute: VectorStoreKit.MetalCompute?
    
    public init(configuration: KMeansClusteringConfiguration = .default) async throws {
        self.configuration = configuration
        
        // Initialize Metal compute if requested and available
        if configuration.useMetalAcceleration {
            self.metalCompute = try? await VectorStoreKit.MetalCompute(configuration: .efficient)
        } else {
            self.metalCompute = nil
        }
    }
    
    /// Perform K-means clustering on the given vectors
    public func cluster(
        vectors: [[Float]],
        k: Int,
        weights: [Float]? = nil
    ) async throws -> ClusteringResult {
        guard !vectors.isEmpty else {
            throw ClusteringError.emptyDataset
        }
        
        guard k > 0 && k <= vectors.count else {
            throw ClusteringError.invalidK(k, maxK: vectors.count)
        }
        
        let dimensions = vectors[0].count
        guard vectors.allSatisfy({ $0.count == dimensions }) else {
            throw ClusteringError.inconsistentDimensions
        }
        
        // Initialize centroids
        var centroids = try await initializeCentroids(
            from: vectors,
            k: k,
            dimensions: dimensions
        )
        
        // Initialize assignments
        var assignments = Array(repeating: 0, count: vectors.count)
        var previousInertia: Float = .infinity
        var iterations = 0
        var converged = false
        
        // Main K-means loop
        while iterations < configuration.maxIterations && !converged {
            // Assignment step
            let (newAssignments, distances) = try await assignToNearestCentroids(
                vectors: vectors,
                centroids: centroids
            )
            
            // Update step
            let newCentroids = try await updateCentroids(
                vectors: vectors,
                assignments: newAssignments,
                k: k,
                dimensions: dimensions,
                weights: weights
            )
            
            // Check convergence
            let inertia = distances.reduce(0, +)
            converged = abs(inertia - previousInertia) < configuration.tolerance
            
            assignments = newAssignments
            centroids = newCentroids
            previousInertia = inertia
            iterations += 1
        }
        
        return ClusteringResult(
            centroids: centroids,
            assignments: assignments,
            iterations: iterations,
            converged: converged,
            inertia: previousInertia
        )
    }
    
    /// Perform mini-batch K-means for large datasets
    public func miniBatchCluster(
        vectors: [[Float]],
        k: Int,
        batchSize: Int = 1000,
        weights: [Float]? = nil
    ) async throws -> ClusteringResult {
        guard batchSize > 0 && batchSize <= vectors.count else {
            throw ClusteringError.invalidBatchSize(batchSize)
        }
        
        let dimensions = vectors[0].count
        
        // Initialize centroids
        var centroids = try await initializeCentroids(
            from: vectors,
            k: k,
            dimensions: dimensions
        )
        
        var iterations = 0
        let maxIterations = configuration.maxIterations
        
        // Mini-batch K-means loop
        while iterations < maxIterations {
            // Sample batch
            let batch = sampleBatch(from: vectors, size: batchSize)
            
            // Assign batch to centroids
            let (assignments, _) = try await assignToNearestCentroids(
                vectors: batch,
                centroids: centroids
            )
            
            // Update centroids incrementally
            centroids = await updateCentroidsIncremental(
                centroids: centroids,
                batch: batch,
                assignments: assignments,
                learningRate: 1.0 / Float(iterations + 1)
            )
            
            iterations += 1
        }
        
        // Final assignment for all vectors
        let (finalAssignments, distances) = try await assignToNearestCentroids(
            vectors: vectors,
            centroids: centroids
        )
        
        return ClusteringResult(
            centroids: centroids,
            assignments: finalAssignments,
            iterations: iterations,
            converged: true,
            inertia: distances.reduce(0, +)
        )
    }
    
    // MARK: - Private Methods
    
    private func initializeCentroids(
        from vectors: [[Float]],
        k: Int,
        dimensions: Int
    ) async throws -> [[Float]] {
        switch configuration.initMethod {
        case .random:
            return initializeRandom(from: vectors, k: k)
            
        case .kMeansPlusPlus:
            return try await initializeKMeansPlusPlus(from: vectors, k: k)
            
        case .custom(let centroids):
            guard centroids.count == k else {
                throw ClusteringError.invalidCustomCentroids(provided: centroids.count, expected: k)
            }
            return centroids
        }
    }
    
    private func initializeRandom(from vectors: [[Float]], k: Int) -> [[Float]] {
        if let seed = configuration.seed {
            var rng = SeedableRNG(seed: seed)
            let indices = vectors.indices.shuffled(using: &rng).prefix(k)
            return indices.map { vectors[$0] }
        } else {
            var rng = SystemRNG()
            let indices = vectors.indices.shuffled(using: &rng).prefix(k)
            return indices.map { vectors[$0] }
        }
    }
    
    private func initializeKMeansPlusPlus(
        from vectors: [[Float]],
        k: Int
    ) async throws -> [[Float]] {
        var centroids: [[Float]] = []
        
        // Choose first centroid randomly
        let firstIndex: Int
        if let seed = configuration.seed {
            var rng = SeedableRNG(seed: seed)
            firstIndex = Int.random(in: 0..<vectors.count, using: &rng)
        } else {
            firstIndex = Int.random(in: 0..<vectors.count)
        }
        centroids.append(vectors[firstIndex])
        
        // Choose remaining centroids
        for _ in 1..<k {
            // Compute distances to nearest centroid for each point
            let distances = try await computeMinDistances(
                vectors: vectors,
                centroids: centroids
            )
            
            // Convert distances to probabilities
            let sumDistances = distances.reduce(0, +)
            guard sumDistances > 0 else {
                throw ClusteringError.degenerateData
            }
            
            // Sample next centroid proportional to squared distance
            let probabilities = distances.map { $0 * $0 / sumDistances }
            let nextIndex: Int
            if let seed = configuration.seed {
                var rng = SeedableRNG(seed: seed + UInt64(centroids.count))
                nextIndex = weightedSample(
                    probabilities: probabilities,
                    using: &rng
                )
            } else {
                var rng = SystemRNG()
                nextIndex = weightedSample(
                    probabilities: probabilities,
                    using: &rng
                )
            }
            
            centroids.append(vectors[nextIndex])
        }
        
        return centroids
    }
    
    private func assignToNearestCentroids(
        vectors: [[Float]],
        centroids: [[Float]]
    ) async throws -> (assignments: [Int], distances: [Float]) {
        // For now, use CPU computation for array-based vectors
        // TODO: Add support for converting arrays to SIMD vectors when dimensions match
        return assignWithCPU(vectors: vectors, centroids: centroids)
    }
    
    private func assignWithCPU(
        vectors: [[Float]],
        centroids: [[Float]]
    ) -> (assignments: [Int], distances: [Float]) {
        var assignments: [Int] = []
        var distances: [Float] = []
        
        for vector in vectors {
            var minDistance: Float = .infinity
            var minIndex = 0
            
            for (index, centroid) in centroids.enumerated() {
                let distance = euclideanDistance(vector, centroid)
                if distance < minDistance {
                    minDistance = distance
                    minIndex = index
                }
            }
            
            assignments.append(minIndex)
            distances.append(minDistance)
        }
        
        return (assignments, distances)
    }
    
    private func assignWithMetal(
        vectors: [[Float]],
        centroids: [[Float]],
        metalCompute: VectorStoreKit.MetalCompute
    ) async throws -> (assignments: [Int], distances: [Float]) {
        // Compute distances for each vector to all centroids
        var assignments: [Int] = []
        var distances: [Float] = []
        
        // Process in batches for better performance
        let batchSize = 100
        
        for i in stride(from: 0, to: vectors.count, by: batchSize) {
            let endIdx = min(i + batchSize, vectors.count)
            let batch = Array(vectors[i..<endIdx])
            
            // Process each vector in the batch
            for vector in batch {
                // Convert to SIMD vector
                let simdVector = vector.withUnsafeBufferPointer { buffer in
                    SIMD32<Float>(buffer)
                }
                
                let simdCentroids = centroids.map { centroid in
                    centroid.withUnsafeBufferPointer { buffer in
                        SIMD32<Float>(buffer)
                    }
                }
                
                let result = try await metalCompute.computeDistances(
                    query: simdVector,
                    candidates: simdCentroids,
                    metric: .euclidean
                )
                
                // Find minimum distance and assignment
                let (minIndex, minDistance) = result.distances.enumerated().min { $0.1 < $1.1 }!
                assignments.append(minIndex)
                distances.append(minDistance)
            }
        }
        
        return (assignments, distances)
    }
    
    private func updateCentroids(
        vectors: [[Float]],
        assignments: [Int],
        k: Int,
        dimensions: Int,
        weights: [Float]?
    ) async throws -> [[Float]] {
        var newCentroids: [[Float]] = Array(
            repeating: Array(repeating: 0.0, count: dimensions),
            count: k
        )
        var counts = Array(repeating: 0.0 as Float, count: k)
        
        // Accumulate weighted sums
        for (index, assignment) in assignments.enumerated() {
            let weight = weights?[index] ?? 1.0
            let vector = vectors[index]
            
            for d in 0..<dimensions {
                newCentroids[assignment][d] += vector[d] * weight
            }
            counts[assignment] += weight
        }
        
        // Compute means
        for c in 0..<k {
            if counts[c] > 0 {
                for d in 0..<dimensions {
                    newCentroids[c][d] /= counts[c]
                }
            } else {
                // Handle empty cluster by reinitializing randomly
                let randomIndex = Int.random(in: 0..<vectors.count)
                newCentroids[c] = vectors[randomIndex]
            }
        }
        
        return newCentroids
    }
    
    private func updateCentroidsIncremental(
        centroids: [[Float]],
        batch: [[Float]],
        assignments: [Int],
        learningRate: Float
    ) async -> [[Float]] {
        var updatedCentroids = centroids
        
        for (vector, assignment) in zip(batch, assignments) {
            // Incremental update: c = c + lr * (x - c)
            for d in 0..<vector.count {
                let diff = vector[d] - updatedCentroids[assignment][d]
                updatedCentroids[assignment][d] += learningRate * diff
            }
        }
        
        return updatedCentroids
    }
    
    private func computeMinDistances(
        vectors: [[Float]],
        centroids: [[Float]]
    ) async throws -> [Float] {
        var minDistances: [Float] = []
        
        for vector in vectors {
            var minDistance: Float = .infinity
            
            for centroid in centroids {
                let distance = euclideanDistance(vector, centroid)
                minDistance = min(minDistance, distance)
            }
            
            minDistances.append(minDistance)
        }
        
        return minDistances
    }
    
    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    private func sampleBatch(from vectors: [[Float]], size: Int) -> [[Float]] {
        let indices = (0..<vectors.count).shuffled().prefix(size)
        return indices.map { vectors[$0] }
    }
    
    private func weightedSample<T>(probabilities: [Float], using rng: inout T) -> Int where T: Swift.RandomNumberGenerator {
        let random = Float.random(in: 0..<1, using: &rng)
        var cumulative: Float = 0
        
        for (index, probability) in probabilities.enumerated() {
            cumulative += probability
            if random < cumulative {
                return index
            }
        }
        
        return probabilities.count - 1
    }
}

// MARK: - Supporting Types

/// Errors that can occur during clustering
public enum ClusteringError: LocalizedError {
    case emptyDataset
    case invalidK(Int, maxK: Int)
    case inconsistentDimensions
    case invalidBatchSize(Int)
    case invalidCustomCentroids(provided: Int, expected: Int)
    case degenerateData
    
    public var errorDescription: String? {
        switch self {
        case .emptyDataset:
            return "Cannot cluster an empty dataset"
        case .invalidK(let k, let maxK):
            return "Invalid number of clusters: \(k). Must be between 1 and \(maxK)"
        case .inconsistentDimensions:
            return "All vectors must have the same dimensions"
        case .invalidBatchSize(let size):
            return "Invalid batch size: \(size)"
        case .invalidCustomCentroids(let provided, let expected):
            return "Custom centroids count (\(provided)) doesn't match k (\(expected))"
        case .degenerateData:
            return "Data is degenerate (all points are identical)"
        }
    }
}

// Simple seedable RNG for reproducibility
private struct SeedableRNG: Swift.RandomNumberGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed
    }
    
    mutating func next() -> UInt64 {
        state = state &* 2862933555777941757 &+ 3037000493
        return state
    }
}

private struct SystemRNG: Swift.RandomNumberGenerator {
    mutating func next() -> UInt64 {
        return UInt64.random(in: UInt64.min...UInt64.max)
    }
}
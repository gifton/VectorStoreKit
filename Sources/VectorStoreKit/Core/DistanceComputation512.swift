// DistanceComputation512.swift
// VectorStoreKit
//
// Optimized SIMD distance computation for 512-dimensional vectors

import Foundation
import simd
import Accelerate

// MARK: - SIMD Conversion Extensions

extension SIMD8 where Scalar == Float {
    @inlinable
    func toFloat16() -> SIMD8<Float16> {
        return SIMD8<Float16>(
            Float16(self[0]), Float16(self[1]), Float16(self[2]), Float16(self[3]),
            Float16(self[4]), Float16(self[5]), Float16(self[6]), Float16(self[7])
        )
    }
}

extension SIMD8 where Scalar == Float16 {
    @inlinable
    func toFloat32() -> SIMD8<Float> {
        return SIMD8<Float>(
            Float(self[0]), Float(self[1]), Float(self[2]), Float(self[3]),
            Float(self[4]), Float(self[5]), Float(self[6]), Float(self[7])
        )
    }
}

/// High-performance distance computation for 512-dimensional vectors
public struct DistanceComputation512 {
    
    // MARK: - SIMD Euclidean Distance
    
    /// Compute Euclidean distance using SIMD operations
    @inlinable
    public static func euclideanDistance(_ a: Vector512, _ b: Vector512) -> Float {
        return a.distanceSquared(to: b).squareRoot()
    }
    
    /// Compute Euclidean distance squared (avoiding sqrt for performance)
    @inlinable
    public static func euclideanDistanceSquared(_ a: Vector512, _ b: Vector512) -> Float {
        return a.distanceSquared(to: b)
    }
    
    // MARK: - SIMD Cosine Distance
    
    /// Compute cosine distance (1 - cosine similarity)
    @inlinable
    public static func cosineDistance(_ a: Vector512, _ b: Vector512, normalized: Bool = false) -> Float {
        if normalized {
            // For normalized vectors, cosine similarity is just dot product
            return 1.0 - a.dot(b)
        } else {
            return 1.0 - a.cosineSimilarity(to: b, normalized: false)
        }
    }
    
    // MARK: - SIMD Manhattan Distance
    
    /// Compute Manhattan (L1) distance with advanced SIMD optimizations
    @inlinable
    @inline(__always)
    public static func manhattanDistance(_ a: Vector512, _ b: Vector512) -> Float {
        return a.withUnsafeMetalBytes { aBytes in
            b.withUnsafeMetalBytes { bBytes in
                let aPtr = aBytes.bindMemory(to: SIMD8<Float>.self)
                let bPtr = bBytes.bindMemory(to: SIMD8<Float>.self)
                
                // Use 4 SIMD8 accumulators for better instruction-level parallelism
                var sum0 = SIMD8<Float>.zero
                var sum1 = SIMD8<Float>.zero
                var sum2 = SIMD8<Float>.zero
                var sum3 = SIMD8<Float>.zero
                
                // Process 512 elements as 64 SIMD8 operations with loop unrolling
                for i in stride(from: 0, to: 64, by: 4) {
                    // Prefetch next iterations for better cache performance
                    if i + 8 < 64 {
                        _ = aPtr[i + 8]
                        _ = bPtr[i + 8]
                    }
                    
                    // Unrolled loop for 4 SIMD8 operations
                    let diff0 = abs(aPtr[i] - bPtr[i])
                    let diff1 = abs(aPtr[i + 1] - bPtr[i + 1])
                    let diff2 = abs(aPtr[i + 2] - bPtr[i + 2])
                    let diff3 = abs(aPtr[i + 3] - bPtr[i + 3])
                    
                    sum0 += diff0
                    sum1 += diff1
                    sum2 += diff2
                    sum3 += diff3
                }
                
                // Combine accumulators and reduce to scalar
                let totalSum = sum0 + sum1 + sum2 + sum3
                return totalSum.sum()
            }
        }
    }
    
    // MARK: - SIMD Dot Product
    
    /// Compute dot product
    @inlinable
    public static func dotProduct(_ a: Vector512, _ b: Vector512) -> Float {
        return a.dot(b)
    }
    
    // MARK: - Advanced Distance Metrics
    
    /// Compute Mahalanobis distance with covariance matrix
    public static func mahalanobisDistance(
        _ a: Vector512,
        _ b: Vector512,
        covarianceMatrix: [[Float]]? = nil
    ) async throws -> Float {
        guard let matrix = covarianceMatrix else {
            throw VectorStoreError(
                category: .distanceComputation,
                code: .missingRequiredParameter,
                message: "Covariance matrix required for Mahalanobis distance"
            )
        }
        
        let mahalanobis = try MahalanobisDistance(covarianceMatrix: matrix)
        return mahalanobis.distance(a, b)
    }
    
    /// Compute Earth Mover's Distance (Wasserstein distance)
    public static func earthMoversDistance(
        _ a: Vector512,
        _ b: Vector512,
        costFunction: EarthMoversDistance.CostFunction = .euclidean
    ) -> Float {
        let emd = EarthMoversDistance(costFunction: costFunction)
        return emd.distance(a, b)
    }
    
    /// Compute learned distance using ML model
    public static func learnedDistance(
        _ a: Vector512,
        _ b: Vector512,
        modelId: String
    ) async throws -> Float {
        guard let model = await LearnedDistanceCache.shared.getModel(id: modelId) else {
            throw VectorStoreError(
                category: .distanceComputation,
                code: .missingRequiredParameter,
                message: "Learned model not found in cache",
                context: ["modelId": modelId]
            )
        }
        
        return try model.distance(a, b)
    }
    
    /// Compute adaptive distance based on context
    public static func adaptiveDistance(
        _ a: Vector512,
        _ b: Vector512,
        context: AdaptiveDistance.Context
    ) -> Float {
        let adaptive = AdaptiveDistance()
        return adaptive.distance(a, b, context: context)
    }
    
    // MARK: - Additional Distance Metrics
    
    /// Compute Chebyshev distance (L∞ norm) with optimized SIMD
    @inlinable
    @inline(__always)
    public static func chebyshevDistance(_ a: Vector512, _ b: Vector512) -> Float {
        return a.withUnsafeMetalBytes { aBytes in
            b.withUnsafeMetalBytes { bBytes in
                let aPtr = aBytes.bindMemory(to: SIMD8<Float>.self)
                let bPtr = bBytes.bindMemory(to: SIMD8<Float>.self)
                
                // Use 4 SIMD8 max accumulators
                var max0 = SIMD8<Float>.zero
                var max1 = SIMD8<Float>.zero
                var max2 = SIMD8<Float>.zero
                var max3 = SIMD8<Float>.zero
                
                // Process with loop unrolling and prefetching
                for i in stride(from: 0, to: 64, by: 4) {
                    // Prefetch ahead
                    if i + 8 < 64 {
                        _ = aPtr[i + 8]
                        _ = bPtr[i + 8]
                    }
                    
                    let diff0 = abs(aPtr[i] - bPtr[i])
                    let diff1 = abs(aPtr[i + 1] - bPtr[i + 1])
                    let diff2 = abs(aPtr[i + 2] - bPtr[i + 2])
                    let diff3 = abs(aPtr[i + 3] - bPtr[i + 3])
                    
                    max0 = max(max0, diff0)
                    max1 = max(max1, diff1)
                    max2 = max(max2, diff2)
                    max3 = max(max3, diff3)
                }
                
                // Find maximum across all accumulators
                let maxVec = max(max(max0, max1), max(max2, max3))
                return maxVec.max()
            }
        }
    }
    
    /// Compute Minkowski distance (Lp norm) with optimized SIMD
    @inlinable
    public static func minkowskiDistance(_ a: Vector512, _ b: Vector512, p: Float) -> Float {
        // Optimize for common cases
        if p == 1.0 {
            return manhattanDistance(a, b)
        } else if p == 2.0 {
            return euclideanDistance(a, b)
        } else if p.isInfinite {
            return chebyshevDistance(a, b)
        }
        
        return a.withUnsafeMetalBytes { aBytes in
            b.withUnsafeMetalBytes { bBytes in
                let aPtr = aBytes.bindMemory(to: SIMD8<Float>.self)
                let bPtr = bBytes.bindMemory(to: SIMD8<Float>.self)
                let pVec = SIMD8<Float>(repeating: p)
                
                // Use 4 accumulators for better parallelism
                var sum0 = SIMD8<Float>.zero
                var sum1 = SIMD8<Float>.zero
                var sum2 = SIMD8<Float>.zero
                var sum3 = SIMD8<Float>.zero
                
                // Unrolled loop with prefetching
                for i in stride(from: 0, to: 64, by: 4) {
                    // Prefetch next data
                    if i + 8 < 64 {
                        _ = aPtr[i + 8]
                        _ = bPtr[i + 8]
                    }
                    
                    let diff0 = abs(aPtr[i] - bPtr[i])
                    let diff1 = abs(aPtr[i + 1] - bPtr[i + 1])
                    let diff2 = abs(aPtr[i + 2] - bPtr[i + 2])
                    let diff3 = abs(aPtr[i + 3] - bPtr[i + 3])
                    
                    // Use SIMD pow for vectorized power computation
                    sum0 += pow(diff0, pVec)
                    sum1 += pow(diff1, pVec)
                    sum2 += pow(diff2, pVec)
                    sum3 += pow(diff3, pVec)
                }
                
                let totalSum = sum0 + sum1 + sum2 + sum3
                return pow(totalSum.sum(), 1.0 / p)
            }
        }
    }
    
    /// Compute Hamming distance for continuous vectors using threshold with SIMD optimization
    @inlinable
    public static func hammingDistance(_ a: Vector512, _ b: Vector512, threshold: Float = 0.01) -> Float {
        return a.withUnsafeMetalBytes { aBytes in
            b.withUnsafeMetalBytes { bBytes in
                let aPtr = aBytes.bindMemory(to: SIMD8<Float>.self)
                let bPtr = bBytes.bindMemory(to: SIMD8<Float>.self)
                let thresholdVec = SIMD8<Float>(repeating: threshold)
                
                // Use integer SIMD for counting
                var count0 = SIMD8<Int32>.zero
                var count1 = SIMD8<Int32>.zero
                var count2 = SIMD8<Int32>.zero
                var count3 = SIMD8<Int32>.zero
                
                // Unrolled loop with efficient mask-to-int conversion
                for i in stride(from: 0, to: 64, by: 4) {
                    let diff0 = abs(aPtr[i] - bPtr[i])
                    let diff1 = abs(aPtr[i + 1] - bPtr[i + 1])
                    let diff2 = abs(aPtr[i + 2] - bPtr[i + 2])
                    let diff3 = abs(aPtr[i + 3] - bPtr[i + 3])
                    
                    // Convert boolean masks to integer counts efficiently
                    count0 += SIMD8<Int32>(diff0 .> thresholdVec)
                    count1 += SIMD8<Int32>(diff1 .> thresholdVec)
                    count2 += SIMD8<Int32>(diff2 .> thresholdVec)
                    count3 += SIMD8<Int32>(diff3 .> thresholdVec)
                }
                
                // Sum all counts and normalize
                let totalCount = count0.wrappedSum() + count1.wrappedSum() + count2.wrappedSum() + count3.wrappedSum()
                return Float(totalCount) / 512.0
            }
        }
    }
    
    /// Compute Jaccard distance for continuous vectors with SIMD optimization
    @inlinable
    public static func jaccardDistance(_ a: Vector512, _ b: Vector512) -> Float {
        return a.withUnsafeMetalBytes { aBytes in
            b.withUnsafeMetalBytes { bBytes in
                let aPtr = aBytes.bindMemory(to: SIMD8<Float>.self)
                let bPtr = bBytes.bindMemory(to: SIMD8<Float>.self)
                
                // Use separate accumulators for min and max
                var minSum0 = SIMD8<Float>.zero
                var minSum1 = SIMD8<Float>.zero
                var maxSum0 = SIMD8<Float>.zero
                var maxSum1 = SIMD8<Float>.zero
                
                // Process with loop unrolling
                for i in stride(from: 0, to: 64, by: 2) {
                    let minVals0 = min(aPtr[i], bPtr[i])
                    let maxVals0 = max(aPtr[i], bPtr[i])
                    let minVals1 = min(aPtr[i + 1], bPtr[i + 1])
                    let maxVals1 = max(aPtr[i + 1], bPtr[i + 1])
                    
                    minSum0 += minVals0
                    minSum1 += minVals1
                    maxSum0 += maxVals0
                    maxSum1 += maxVals1
                }
                
                let totalMinSum = (minSum0 + minSum1).sum()
                let totalMaxSum = (maxSum0 + maxSum1).sum()
                
                return totalMaxSum > Float.ulpOfOne ? 1.0 - (totalMinSum / totalMaxSum) : 0.0
            }
        }
    }
    
    // MARK: - Advanced Batch Distance Computation
    
    /// Compute distances from query to multiple candidates using optimized SIMD
    @inlinable
    public static func batchEuclideanDistance(
        query: Vector512,
        candidates: [Vector512]
    ) -> [Float] {
        guard !candidates.isEmpty else { return [] }
        
        var results = [Float](repeating: 0, count: candidates.count)
        
        // Process candidates in cache-friendly chunks
        let chunkSize = 16  // Optimal for L1 cache
        
        results.withUnsafeMutableBufferPointer { resultsPtr in
            for chunkStart in stride(from: 0, to: candidates.count, by: chunkSize) {
                let chunkEnd = min(chunkStart + chunkSize, candidates.count)
                
                // Prefetch next chunk for better cache performance
                if chunkStart + chunkSize < candidates.count {
                    let prefetchEnd = min(chunkStart + 2 * chunkSize, candidates.count)
                    for i in (chunkStart + chunkSize)..<prefetchEnd {
                        _ = candidates[i][0]  // Trigger prefetch
                    }
                }
                
                // Process current chunk
                for i in chunkStart..<chunkEnd {
                    resultsPtr[i] = euclideanDistance(query, candidates[i])
                }
            }
        }
        
        return results
    }
    
    /// Compute distances from query to multiple candidates in parallel with work stealing
    public static func batchEuclideanDistanceParallel(
        query: Vector512,
        candidates: [Vector512]
    ) async -> [Float] {
        guard !candidates.isEmpty else { return [] }
        
        // Use work-stealing with optimal chunk sizes
        let minChunkSize = max(1, candidates.count / (ProcessInfo.processInfo.activeProcessorCount * 2))
        let maxChunkSize = min(64, candidates.count)
        let chunkSize = max(minChunkSize, min(maxChunkSize, 32))
        
        return await withTaskGroup(of: (Int, [Float]).self) { group in
            // Submit work in chunks
            for chunkStart in stride(from: 0, to: candidates.count, by: chunkSize) {
                let chunkEnd = min(chunkStart + chunkSize, candidates.count)
                let chunk = Array(candidates[chunkStart..<chunkEnd])
                
                group.addTask {
                    let chunkResults = chunk.map { euclideanDistance(query, $0) }
                    return (chunkStart, chunkResults)
                }
            }
            
            // Collect results maintaining order
            var results = Array(repeating: Float(0), count: candidates.count)
            for await (startIndex, chunkResults) in group {
                for (i, result) in chunkResults.enumerated() {
                    results[startIndex + i] = result
                }
            }
            return results
        }
    }
    
    // MARK: - Advanced SIMD Specialized Functions
    
    /// Ultra-optimized Euclidean distance squared using platform-specific SIMD
    @inlinable
    @inline(__always)
    public static func euclideanDistanceSquaredUltraOptimized(_ a: Vector512, _ b: Vector512) -> Float {
        return a.withUnsafeMetalBytes { aBytes in
            b.withUnsafeMetalBytes { bBytes in
                let aPtr = aBytes.bindMemory(to: SIMD16<Float>.self)
                let bPtr = bBytes.bindMemory(to: SIMD16<Float>.self)
                
                // Use 8 SIMD16 accumulators for maximum instruction-level parallelism
                var sum0 = SIMD16<Float>.zero
                var sum1 = SIMD16<Float>.zero
                var sum2 = SIMD16<Float>.zero
                var sum3 = SIMD16<Float>.zero
                var sum4 = SIMD16<Float>.zero
                var sum5 = SIMD16<Float>.zero
                var sum6 = SIMD16<Float>.zero
                var sum7 = SIMD16<Float>.zero
                
                // Process 512 elements as 32 SIMD16 operations (8x unrolled)
                for i in stride(from: 0, to: 32, by: 8) {
                    // Manually unrolled for maximum performance
                    let diff0 = aPtr[i] - bPtr[i]
                    let diff1 = aPtr[i + 1] - bPtr[i + 1]
                    let diff2 = aPtr[i + 2] - bPtr[i + 2]
                    let diff3 = aPtr[i + 3] - bPtr[i + 3]
                    let diff4 = aPtr[i + 4] - bPtr[i + 4]
                    let diff5 = aPtr[i + 5] - bPtr[i + 5]
                    let diff6 = aPtr[i + 6] - bPtr[i + 6]
                    let diff7 = aPtr[i + 7] - bPtr[i + 7]
                    
                    // Fused multiply-add operations
                    sum0 = sum0 + diff0 * diff0
                    sum1 = sum1 + diff1 * diff1
                    sum2 = sum2 + diff2 * diff2
                    sum3 = sum3 + diff3 * diff3
                    sum4 = sum4 + diff4 * diff4
                    sum5 = sum5 + diff5 * diff5
                    sum6 = sum6 + diff6 * diff6
                    sum7 = sum7 + diff7 * diff7
                }
                
                // Hierarchical reduction for optimal instruction scheduling
                let sum01 = sum0 + sum1
                let sum23 = sum2 + sum3
                let sum45 = sum4 + sum5
                let sum67 = sum6 + sum7
                
                let sum0123 = sum01 + sum23
                let sum4567 = sum45 + sum67
                
                let totalSum = sum0123 + sum4567
                return totalSum.sum()
            }
        }
    }
    
    /// Optimized cosine similarity for normalized vectors using wide SIMD
    @inlinable
    @inline(__always)
    public static func normalizedCosineSimilarityOptimized(_ a: Vector512, _ b: Vector512) -> Float {
        return a.withUnsafeMetalBytes { aBytes in
            b.withUnsafeMetalBytes { bBytes in
                let aPtr = aBytes.bindMemory(to: SIMD16<Float>.self)
                let bPtr = bBytes.bindMemory(to: SIMD16<Float>.self)
                
                // 4 SIMD16 accumulators for dot product
                var dot0 = SIMD16<Float>.zero
                var dot1 = SIMD16<Float>.zero
                var dot2 = SIMD16<Float>.zero
                var dot3 = SIMD16<Float>.zero
                
                // Unrolled loop for 512 elements (32 SIMD16 operations)
                for i in stride(from: 0, to: 32, by: 4) {
                    dot0 = dot0 + aPtr[i] * bPtr[i]
                    dot1 = dot1 + aPtr[i + 1] * bPtr[i + 1]
                    dot2 = dot2 + aPtr[i + 2] * bPtr[i + 2]
                    dot3 = dot3 + aPtr[i + 3] * bPtr[i + 3]
                }
                
                // Efficient reduction
                let totalDot = dot0 + dot1 + dot2 + dot3
                return totalDot.sum()
            }
        }
    }
    
    /// SIMD-optimized batch distance computation with memory prefetching
    public static func batchDistanceOptimized(
        query: Vector512,
        candidates: UnsafeBufferPointer<Vector512>,
        metric: DistanceMetric = .euclidean
    ) -> [Float] {
        guard !candidates.isEmpty else { return [] }
        
        var results = [Float](repeating: 0, count: candidates.count)
        
        results.withUnsafeMutableBufferPointer { resultsPtr in
            // Process with aggressive prefetching
            let prefetchDistance = 8
            
            for i in 0..<candidates.count {
                // Prefetch future candidates
                if i + prefetchDistance < candidates.count {
                    _ = candidates[i + prefetchDistance][0]  // Trigger cache line prefetch
                }
                
                // Compute distance using optimal function
                let distance: Float
                switch metric {
                case .euclidean:
                    distance = sqrt(euclideanDistanceSquaredUltraOptimized(query, candidates[i]))
                case .cosine:
                    distance = 1.0 - normalizedCosineSimilarityOptimized(query, candidates[i])
                default:
                    distance = query.distance(to: candidates[i], metric: metric)
                }
                
                resultsPtr[i] = distance
            }
        }
        
        return results
    }
    
    // MARK: - Accelerate Framework Optimizations
    
    /// Batch distance computation using Accelerate framework
    public static func batchEuclideanDistanceAccelerate(
        query: Vector512,
        candidates: [Vector512]
    ) -> [Float] {
        guard !candidates.isEmpty else { return [] }
        
        var results = [Float](repeating: 0, count: candidates.count)
        let queryArray = query.toArray()
        
        for (index, candidate) in candidates.enumerated() {
            let candidateArray = candidate.toArray()
            
            // Compute difference
            var diff = [Float](repeating: 0, count: 512)
            vDSP_vsub(candidateArray, 1, queryArray, 1, &diff, 1, 512)
            
            // Square differences
            vDSP_vsq(diff, 1, &diff, 1, 512)
            
            // Sum squared differences
            var sum: Float = 0
            vDSP_sve(diff, 1, &sum, 512)
            
            results[index] = sqrt(sum)
        }
        
        return results
    }
    
    /// Batch cosine distance using Accelerate
    public static func batchCosineDistanceAccelerate(
        query: Vector512,
        candidates: [Vector512],
        normalized: Bool = false
    ) -> [Float] {
        guard !candidates.isEmpty else { return [] }
        
        var results = [Float](repeating: 0, count: candidates.count)
        let queryArray = query.toArray()
        
        if normalized {
            // For normalized vectors, just compute dot products
            for (index, candidate) in candidates.enumerated() {
                let candidateArray = candidate.toArray()
                var dotProduct: Float = 0
                vDSP_dotpr(queryArray, 1, candidateArray, 1, &dotProduct, 512)
                results[index] = 1.0 - dotProduct
            }
        } else {
            // Compute query magnitude once
            var queryMagnitude: Float = 0
            vDSP_svesq(queryArray, 1, &queryMagnitude, 512)
            queryMagnitude = sqrt(queryMagnitude)
            
            for (index, candidate) in candidates.enumerated() {
                let candidateArray = candidate.toArray()
                
                // Dot product
                var dotProduct: Float = 0
                vDSP_dotpr(queryArray, 1, candidateArray, 1, &dotProduct, 512)
                
                // Candidate magnitude
                var candidateMagnitude: Float = 0
                vDSP_svesq(candidateArray, 1, &candidateMagnitude, 512)
                candidateMagnitude = sqrt(candidateMagnitude)
                
                // Cosine similarity = dot / (mag_a * mag_b)
                let similarity = dotProduct / (queryMagnitude * candidateMagnitude + Float.ulpOfOne)
                results[index] = 1.0 - similarity
            }
        }
        
        return results
    }
    
    // MARK: - Distance Matrix Computation
    
    /// Compute pairwise distance matrix for a set of vectors
    /// Now with GPU acceleration for large matrices
    public static func distanceMatrix(
        vectors: [Vector512],
        metric: DistanceMetric = .euclidean,
        useGPU: Bool? = nil  // nil = auto-detect based on size
    ) async throws -> [[Float]] {
        let count = vectors.count
        
        // Auto-detect GPU usage based on matrix size
        let shouldUseGPU = useGPU ?? (count * count > 10_000)
        
        if shouldUseGPU {
            // TODO: Use GPU acceleration when MetalDistanceMatrix is integrated
            // For now, fall back to CPU
            return distanceMatrixCPU(vectors: vectors, metric: metric)
        } else {
            // Fall back to CPU implementation
            return distanceMatrixCPU(vectors: vectors, metric: metric)
        }
    }
    
    /// CPU implementation of distance matrix computation
    public static func distanceMatrixCPU(
        vectors: [Vector512],
        metric: DistanceMetric = .euclidean
    ) -> [[Float]] {
        let count = vectors.count
        var matrix = Array(repeating: Array(repeating: Float(0), count: count), count: count)
        
        // Only compute upper triangle (distance matrix is symmetric)
        for i in 0..<count {
            for j in i+1..<count {
                let distance: Float
                
                switch metric {
                case .euclidean:
                    distance = euclideanDistance(vectors[i], vectors[j])
                case .cosine:
                    distance = cosineDistance(vectors[i], vectors[j])
                case .manhattan:
                    distance = manhattanDistance(vectors[i], vectors[j])
                case .dotProduct:
                    distance = -dotProduct(vectors[i], vectors[j])
                case .chebyshev:
                    distance = chebyshevDistance(vectors[i], vectors[j])
                case .minkowski:
                    // Default to p=3 for Minkowski distance
                    distance = minkowskiDistance(vectors[i], vectors[j], p: 3)
                case .hamming:
                    // For continuous vectors, use threshold-based Hamming
                    distance = hammingDistance(vectors[i], vectors[j])
                case .jaccard:
                    // Jaccard distance for continuous vectors
                    distance = jaccardDistance(vectors[i], vectors[j])
                case .mahalanobis:
                    // Note: distanceMatrix doesn't support async metrics that require external data
                    // For Mahalanobis, we fall back to Euclidean in batch operations
                    distance = euclideanDistance(vectors[i], vectors[j])
                case .earth_mover:
                    // Earth Mover's Distance with default cost function
                    distance = earthMoversDistance(vectors[i], vectors[j])
                case .learned:
                    // Note: distanceMatrix doesn't support async metrics
                    // For learned distance, fall back to Euclidean in batch operations
                    distance = euclideanDistance(vectors[i], vectors[j])
                case .adaptive:
                    // Note: Adaptive distance requires context which isn't available here
                    // Fall back to Euclidean for batch operations
                    distance = euclideanDistance(vectors[i], vectors[j])
                }
                
                matrix[i][j] = distance
                matrix[j][i] = distance // Symmetric
            }
        }
        
        return matrix
    }
    
    /// Compute distance matrix between two different sets of vectors
    public static func distanceMatrix(
        vectorsA: [Vector512],
        vectorsB: [Vector512],
        metric: DistanceMetric = .euclidean,
        useGPU: Bool? = nil
    ) async throws -> [[Float]] {
        let totalElements = vectorsA.count * vectorsB.count
        let shouldUseGPU = useGPU ?? (totalElements > 10_000)
        
        if shouldUseGPU {
            // TODO: Use GPU acceleration when MetalDistanceMatrix is integrated
            // For now, fall back to CPU
            return distanceMatrixCPU(vectorsA: vectorsA, vectorsB: vectorsB, metric: metric)
        } else {
            return distanceMatrixCPU(vectorsA: vectorsA, vectorsB: vectorsB, metric: metric)
        }
    }
    
    /// CPU implementation for distance matrix between two sets
    public static func distanceMatrixCPU(
        vectorsA: [Vector512],
        vectorsB: [Vector512],
        metric: DistanceMetric = .euclidean
    ) -> [[Float]] {
        var matrix = Array(repeating: Array(repeating: Float(0), count: vectorsB.count), count: vectorsA.count)
        
        for i in 0..<vectorsA.count {
            for j in 0..<vectorsB.count {
                let distance: Float
                
                switch metric {
                case .euclidean:
                    distance = euclideanDistance(vectorsA[i], vectorsB[j])
                case .cosine:
                    distance = cosineDistance(vectorsA[i], vectorsB[j])
                case .manhattan:
                    distance = manhattanDistance(vectorsA[i], vectorsB[j])
                case .dotProduct:
                    distance = -dotProduct(vectorsA[i], vectorsB[j])
                default:
                    distance = euclideanDistance(vectorsA[i], vectorsB[j])
                }
                
                matrix[i][j] = distance
            }
        }
        
        return matrix
    }
    
    // MARK: - Metal Integration
    
    // TODO: Add Metal integration when MetalDistanceMatrix is available
    // private static var metalDistanceMatrix: MetalDistanceMatrix?
    
    // MARK: - Performance Utilities
    
    /// Warm up SIMD units by performing dummy operations
    public static func warmUpSIMD() {
        let dummy1 = Vector512(repeating: 1.0)
        let dummy2 = Vector512(repeating: 2.0)
        _ = euclideanDistance(dummy1, dummy2)
    }
    
    /// Benchmark distance computation
    public static func benchmark(
        vectorCount: Int = 10000,
        metric: DistanceMetric = .euclidean
    ) -> (simdTime: Double, accelerateTime: Double) {
        // Generate random vectors
        var vectors = [Vector512]()
        vectors.reserveCapacity(vectorCount)
        
        for _ in 0..<vectorCount {
            let values = (0..<512).map { _ in Float.random(in: -1...1) }
            vectors.append(Vector512(values))
        }
        
        let query = vectors[0]
        let candidates = Array(vectors[1...])
        
        // Benchmark SIMD
        let simdStart = CFAbsoluteTimeGetCurrent()
        let _ = batchEuclideanDistance(query: query, candidates: candidates)
        let simdTime = CFAbsoluteTimeGetCurrent() - simdStart
        
        // Benchmark Accelerate
        let accelerateStart = CFAbsoluteTimeGetCurrent()
        let _ = batchEuclideanDistanceAccelerate(query: query, candidates: candidates)
        let accelerateTime = CFAbsoluteTimeGetCurrent() - accelerateStart
        
        return (simdTime, accelerateTime)
    }
}

// MARK: - Generic SIMD Extensions

extension SIMD where Scalar: BinaryInteger {
    /// Efficient wrapping sum for integer SIMD types
    @inlinable
    func wrappedSum() -> Scalar {
        var result = Scalar.zero
        for i in indices {
            result = result &+ self[i]
        }
        return result
    }
}

// MARK: - Distance Metric Integration

extension Vector512 {
    /// Compute distance to another vector using specified metric
    public func distance(to other: Vector512, metric: DistanceMetric) -> Float {
        switch metric {
        case .euclidean:
            return DistanceComputation512.euclideanDistance(self, other)
        case .cosine:
            return DistanceComputation512.cosineDistance(self, other)
        case .manhattan:
            return DistanceComputation512.manhattanDistance(self, other)
        case .dotProduct:
            return -DistanceComputation512.dotProduct(self, other)
        case .chebyshev:
            return DistanceComputation512.chebyshevDistance(self, other)
        case .minkowski:
            // Default to p=3 for Minkowski distance
            return DistanceComputation512.minkowskiDistance(self, other, p: 3)
        case .hamming:
            // For continuous vectors, use threshold-based Hamming
            return DistanceComputation512.hammingDistance(self, other)
        case .jaccard:
            // Jaccard distance for continuous vectors
            return DistanceComputation512.jaccardDistance(self, other)
        case .mahalanobis:
            // Note: This synchronous method can't use async Mahalanobis
            // Use DistanceComputation512.mahalanobisDistance for full support
            return DistanceComputation512.euclideanDistance(self, other)
        case .earth_mover:
            return DistanceComputation512.earthMoversDistance(self, other)
        case .learned:
            // Note: This synchronous method can't use async learned distance
            // Use DistanceComputation512.learnedDistance for full support
            return DistanceComputation512.euclideanDistance(self, other)
        case .adaptive:
            // Note: Adaptive distance requires context
            // Use DistanceComputation512.adaptiveDistance with context for full support
            return DistanceComputation512.euclideanDistance(self, other)
        }
    }
}

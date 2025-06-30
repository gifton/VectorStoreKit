// MahalanobisDistance.swift
// VectorStoreKit
//
// Mahalanobis distance implementation with covariance matrix support

import Foundation
import simd
import Accelerate

/// Mahalanobis distance computation with numerical stability
public struct MahalanobisDistance {
    
    /// Inverted covariance matrix stored in optimized format
    private let invertedCovariance: CovarianceMatrix
    
    /// Thread-safe workspace for computation
    private let workspace: MahalanobisWorkspace
    
    /// Initialize with covariance matrix
    public init(covarianceMatrix: [[Float]]) throws {
        guard covarianceMatrix.count == 512 else {
            throw VectorStoreError(
                category: .distanceComputation,
                code: .invalidInput,
                message: "Covariance matrix must be 512x512",
                context: ["matrixSize": covarianceMatrix.count]
            )
        }
        
        // Validate matrix is square
        for row in covarianceMatrix {
            guard row.count == 512 else {
                throw VectorStoreError(
                    category: .distanceComputation,
                    code: .invalidInput,
                    message: "Covariance matrix must be square",
                    context: ["rowSize": row.count]
                )
            }
        }
        
        // Compute inverted covariance matrix
        self.invertedCovariance = try CovarianceMatrix(matrix: covarianceMatrix)
        self.workspace = MahalanobisWorkspace()
    }
    
    /// Compute Mahalanobis distance between two vectors with SIMD optimization
    public func distance(_ a: Vector512, _ b: Vector512) -> Float {
        // Compute difference vector using SIMD
        let diff = a - b
        
        // Apply inverted covariance matrix with SIMD-optimized transform
        let transformedDiff = invertedCovariance.transformOptimized(diff)
        
        // Compute final dot product using optimized SIMD
        return sqrt(diff.dotOptimized(transformedDiff))
    }
    
    /// Batch computation with SIMD optimization and memory prefetching
    public func batchDistance(query: Vector512, candidates: [Vector512]) -> [Float] {
        guard !candidates.isEmpty else { return [] }
        
        var results = [Float](repeating: 0, count: candidates.count)
        
        // Pre-compute query transformation for efficiency
        let queryArray = query.toArray()
        var preTransformedQuery = [Float](repeating: 0, count: 512)
        
        // Use batch BLAS operation for query pre-transformation
        invertedCovariance.withUnsafeBufferPointer { storagePtr in
            cblas_sgemv(
                CblasColMajor,
                CblasNoTrans,
                512, 512,
                1.0,
                storagePtr.baseAddress!, 512,
                queryArray, 1,
                0.0,
                &preTransformedQuery, 1
            )
        }
        
        // Process candidates with prefetching
        results.withUnsafeMutableBufferPointer { resultsPtr in
            for i in 0..<candidates.count {
                // Prefetch next candidates
                if i + 4 < candidates.count {
                    _ = candidates[i + 4][0]
                }
                
                let diff = query - candidates[i]
                let transformedDiff = invertedCovariance.transformOptimized(diff)
                resultsPtr[i] = sqrt(diff.dotOptimized(transformedDiff))
            }
        }
        
        return results
    }
}

/// Optimized covariance matrix representation
struct CovarianceMatrix {
    // Store in column-major format for Accelerate
    private let storage: [Float]
    private let dimension = 512
    
    init(matrix: [[Float]]) throws {
        // Flatten to column-major format
        var flattened = [Float](repeating: 0, count: 512 * 512)
        for i in 0..<512 {
            for j in 0..<512 {
                flattened[j * 512 + i] = matrix[i][j]
            }
        }
        
        // Compute inverse using LAPACK
        var inversed = flattened
        var n = Int32(dimension)
        var lda = n
        var ipiv = [Int32](repeating: 0, count: Int(n))
        var work = [Float](repeating: 0, count: Int(n))
        var lwork = n
        var info: Int32 = 0
        
        // LU decomposition
        sgetrf_(&n, &n, &inversed, &lda, &ipiv, &info)
        
        guard info == 0 else {
            throw VectorStoreError(
                category: .distanceComputation,
                code: .numericalOverflow,
                message: "Covariance matrix is singular or ill-conditioned",
                context: ["lapackInfo": info]
            )
        }
        
        // Compute inverse
        sgetri_(&n, &inversed, &lda, &ipiv, &work, &lwork, &info)
        
        guard info == 0 else {
            throw VectorStoreError(
                category: .distanceComputation,
                code: .numericalOverflow,
                message: "Failed to invert covariance matrix",
                context: ["lapackInfo": info]
            )
        }
        
        self.storage = inversed
    }
    
    /// Access storage for BLAS operations
    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer(body)
    }
    
    /// Transform vector by the inverted covariance matrix
    func transform(_ vector: Vector512) -> Vector512 {
        let input = vector.toArray()
        var output = [Float](repeating: 0, count: 512)
        
        // Matrix-vector multiplication using BLAS
        cblas_sgemv(
            CblasColMajor,
            CblasNoTrans,
            512, 512,
            1.0,
            storage, 512,
            input, 1,
            0.0,
            &output, 1
        )
        
        return Vector512(output)
    }
    
    /// SIMD-optimized matrix-vector transformation
    @inlinable
    func transformOptimized(_ vector: Vector512) -> Vector512 {
        return vector.withUnsafeMetalBytes { vectorBytes in
            let vectorPtr = vectorBytes.bindMemory(to: Float.self)
            var output = [Float](repeating: 0, count: 512)
            
            // Use optimized BLAS call with better memory layout
            output.withUnsafeMutableBufferPointer { outputPtr in
                guard let outputBase = outputPtr.baseAddress else { return }
                
                // Optimized matrix-vector multiplication
                cblas_sgemv(
                    CblasColMajor,
                    CblasNoTrans,
                    512, 512,
                    1.0,
                    storage, 512,
                    vectorPtr.baseAddress!, 1,
                    0.0,
                    outputBase, 1
                )
            }
            
            return Vector512(output)
        }
    }
    
    /// Batch matrix-vector multiplication for multiple vectors
    func batchTransform(_ vectors: [Vector512]) -> [Vector512] {
        guard !vectors.isEmpty else { return [] }
        
        let batchSize = vectors.count
        let inputSize = 512 * batchSize
        
        // Flatten input vectors for batch processing
        var flatInput = [Float]()
        flatInput.reserveCapacity(inputSize)
        
        for vector in vectors {
            flatInput.append(contentsOf: vector.toArray())
        }
        
        var flatOutput = [Float](repeating: 0, count: inputSize)
        
        // Batch matrix multiplication: Output = Matrix * Input
        cblas_sgemm(
            CblasColMajor,
            CblasNoTrans, CblasNoTrans,
            512, batchSize, 512,
            1.0,
            storage, 512,
            flatInput, 512,
            0.0,
            &flatOutput, 512
        )
        
        // Convert back to Vector512 array
        var result = [Vector512]()
        result.reserveCapacity(batchSize)
        
        for i in 0..<batchSize {
            let startIndex = i * 512
            let endIndex = startIndex + 512
            let vectorData = Array(flatOutput[startIndex..<endIndex])
            result.append(Vector512(vectorData))
        }
        
        return result
    }
}

/// Thread-safe workspace for Mahalanobis computation
final class MahalanobisWorkspace {
    // Pre-allocated buffers for intermediate results
    private let tempBuffer: UnsafeMutablePointer<Float>
    
    init() {
        tempBuffer = UnsafeMutablePointer<Float>.allocate(capacity: 512)
    }
    
    deinit {
        tempBuffer.deallocate()
    }
}

/// Global cache for covariance matrices to avoid recomputation
public actor MahalanobisCache {
    private var cache: [String: MahalanobisDistance] = [:]
    private let maxCacheSize = 10
    
    public static let shared = MahalanobisCache()
    
    public func getDistance(for id: String, covarianceMatrix: [[Float]]?) async throws -> MahalanobisDistance {
        if let cached = cache[id] {
            return cached
        }
        
        guard let matrix = covarianceMatrix else {
            throw VectorStoreError(
                category: .distanceComputation,
                code: .missingRequiredParameter,
                message: "Covariance matrix required for Mahalanobis distance",
                context: ["id": id]
            )
        }
        
        let distance = try MahalanobisDistance(covarianceMatrix: matrix)
        
        // Evict oldest if cache is full
        if cache.count >= maxCacheSize {
            cache.removeFirst()
        }
        
        cache[id] = distance
        return distance
    }
    
    public func clear() {
        cache.removeAll()
    }
}

// MARK: - Vector512 SIMD Extensions for Mahalanobis

extension Vector512 {
    /// SIMD-optimized dot product for Mahalanobis distance
    @inlinable
    func dotOptimized(_ other: Vector512) -> Float {
        return self.withUnsafeMetalBytes { selfBytes in
            other.withUnsafeMetalBytes { otherBytes in
                let selfPtr = selfBytes.bindMemory(to: SIMD8<Float>.self)
                let otherPtr = otherBytes.bindMemory(to: SIMD8<Float>.self)
                
                // Use 4 SIMD8 accumulators for better instruction-level parallelism
                var sum0 = SIMD8<Float>.zero
                var sum1 = SIMD8<Float>.zero
                var sum2 = SIMD8<Float>.zero
                var sum3 = SIMD8<Float>.zero
                
                // Process 512 elements as 64 SIMD8 operations with unrolling
                for i in stride(from: 0, to: 64, by: 4) {
                    sum0 += selfPtr[i] * otherPtr[i]
                    sum1 += selfPtr[i + 1] * otherPtr[i + 1]
                    sum2 += selfPtr[i + 2] * otherPtr[i + 2]
                    sum3 += selfPtr[i + 3] * otherPtr[i + 3]
                }
                
                // Reduce to scalar
                let totalSum = sum0 + sum1 + sum2 + sum3
                return totalSum.sum()
            }
        }
    }
}
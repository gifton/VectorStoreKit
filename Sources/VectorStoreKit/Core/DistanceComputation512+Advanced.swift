// DistanceComputation512+Advanced.swift
// VectorStoreKit
//
// Advanced distance metric extensions for DistanceComputation512

import Foundation
import simd
import Accelerate

// MARK: - Advanced Distance Metrics (Async Support)

extension DistanceComputation512 {
    
    /// Compute distance with full metric support including async operations
    public static func computeDistance(
        _ a: Vector512,
        _ b: Vector512,
        metric: DistanceMetric,
        parameters: DistanceParameters = .default
    ) async throws -> Float {
        switch metric {
        case .euclidean:
            return euclideanDistance(a, b)
            
        case .cosine:
            return cosineDistance(a, b, normalized: parameters.normalized)
            
        case .manhattan:
            return manhattanDistance(a, b)
            
        case .dotProduct:
            return -dotProduct(a, b)
            
        case .chebyshev:
            return chebyshevDistance(a, b)
            
        case .minkowski:
            return minkowskiDistance(a, b, p: parameters.minkowskiP)
            
        case .hamming:
            return hammingDistance(a, b, threshold: parameters.hammingThreshold)
            
        case .jaccard:
            return jaccardDistance(a, b)
            
        case .mahalanobis:
            guard let covarianceMatrix = parameters.covarianceMatrix else {
                throw VectorStoreError(
                    category: .distanceComputation,
                    code: .missingRequiredParameter,
                    message: "Covariance matrix required for Mahalanobis distance"
                )
            }
            return try await mahalanobisDistance(a, b, covarianceMatrix: covarianceMatrix)
            
        case .earth_mover:
            let costFunction = parameters.emdCostFunction ?? .euclidean
            return earthMoversDistance(a, b, costFunction: costFunction)
            
        case .learned:
            guard let modelId = parameters.learnedModelId else {
                throw VectorStoreError(
                    category: .distanceComputation,
                    code: .missingRequiredParameter,
                    message: "Model ID required for learned distance"
                )
            }
            return try await learnedDistance(a, b, modelId: modelId)
            
        case .adaptive:
            guard let context = parameters.adaptiveContext else {
                throw VectorStoreError(
                    category: .distanceComputation,
                    code: .missingRequiredParameter,
                    message: "Context required for adaptive distance"
                )
            }
            return adaptiveDistance(a, b, context: context)
        }
    }
    
    /// Batch distance computation with full metric support
    public static func batchComputeDistance(
        query: Vector512,
        candidates: [Vector512],
        metric: DistanceMetric,
        parameters: DistanceParameters = .default
    ) async throws -> [Float] {
        switch metric {
        case .euclidean, .cosine, .manhattan, .dotProduct, .chebyshev,
             .minkowski, .hamming, .jaccard:
            // Use synchronous batch methods for simple metrics
            return candidates.map { candidate in
                switch metric {
                case .euclidean:
                    return euclideanDistance(query, candidate)
                case .cosine:
                    return cosineDistance(query, candidate, normalized: parameters.normalized)
                case .manhattan:
                    return manhattanDistance(query, candidate)
                case .dotProduct:
                    return -dotProduct(query, candidate)
                case .chebyshev:
                    return chebyshevDistance(query, candidate)
                case .minkowski:
                    return minkowskiDistance(query, candidate, p: parameters.minkowskiP)
                case .hamming:
                    return hammingDistance(query, candidate, threshold: parameters.hammingThreshold)
                case .jaccard:
                    return jaccardDistance(query, candidate)
                default:
                    return 0 // Won't reach here
                }
            }
            
        case .mahalanobis:
            guard let covarianceMatrix = parameters.covarianceMatrix else {
                throw VectorStoreError(
                    category: .distanceComputation,
                    code: .missingRequiredParameter,
                    message: "Covariance matrix required for Mahalanobis distance"
                )
            }
            
            let mahalanobis = try await MahalanobisCache.shared.getDistance(
                for: parameters.mahalanobisCacheId ?? "default",
                covarianceMatrix: covarianceMatrix
            )
            
            return mahalanobis.batchDistance(query: query, candidates: candidates)
            
        case .earth_mover:
            let emd = EarthMoversDistance(
                costFunction: parameters.emdCostFunction ?? .euclidean
            )
            
            // Check cache for optimization
            let cache = EarthMoversDistanceCache.shared
            
            return await withTaskGroup(of: (Int, Float).self) { group in
                for (index, candidate) in candidates.enumerated() {
                    group.addTask {
                        let cacheKey = "\(query.hashValue)-\(candidate.hashValue)"
                        
                        if let cached = await cache.getCachedDistance(key: cacheKey) {
                            return (index, cached)
                        }
                        
                        let distance = emd.distance(query, candidate)
                        await cache.cacheDistance(key: cacheKey, distance: distance)
                        return (index, distance)
                    }
                }
                
                var results = Array(repeating: Float(0), count: candidates.count)
                for await (index, distance) in group {
                    results[index] = distance
                }
                return results
            }
            
        case .learned:
            guard let modelId = parameters.learnedModelId else {
                throw VectorStoreError(
                    category: .distanceComputation,
                    code: .missingRequiredParameter,
                    message: "Model ID required for learned distance"
                )
            }
            
            guard let model = await LearnedDistanceCache.shared.getModel(id: modelId) else {
                throw VectorStoreError(
                    category: .distanceComputation,
                    code: .missingRequiredParameter,
                    message: "Learned model not found",
                    context: ["modelId": modelId]
                )
            }
            
            return try await model.batchDistance(query: query, candidates: candidates)
            
        case .adaptive:
            guard let context = parameters.adaptiveContext else {
                throw VectorStoreError(
                    category: .distanceComputation,
                    code: .missingRequiredParameter,
                    message: "Context required for adaptive distance"
                )
            }
            
            let adaptive = AdaptiveDistance(strategy: parameters.adaptiveStrategy ?? .hybrid)
            
            return candidates.map { candidate in
                adaptive.distance(query, candidate, context: context)
            }
        }
    }
    
    /// Create optimal distance parameters for a given metric
    public static func optimalParameters(
        for metric: DistanceMetric,
        dataset: [Vector512]? = nil
    ) async -> DistanceParameters {
        var params = DistanceParameters()
        
        switch metric {
        case .minkowski:
            // Default to p=2 (Euclidean) unless dataset suggests otherwise
            params.minkowskiP = 2.0
            
        case .hamming:
            // Adaptive threshold based on data if available
            if let dataset = dataset, !dataset.isEmpty {
                // Sample to determine good threshold
                let sample = Array(dataset.prefix(100))
                var diffs: [Float] = []
                
                for i in 0..<min(10, sample.count-1) {
                    for j in i+1..<min(11, sample.count) {
                        let a = sample[i].toArray()
                        let b = sample[j].toArray()
                        for k in 0..<512 {
                            diffs.append(abs(a[k] - b[k]))
                        }
                    }
                }
                
                diffs.sort()
                params.hammingThreshold = diffs[diffs.count / 100] // 1st percentile
            } else {
                params.hammingThreshold = 0.01
            }
            
        case .mahalanobis:
            // Would compute covariance matrix from dataset
            // For now, caller must provide
            break
            
        case .earth_mover:
            params.emdCostFunction = .euclidean
            
        case .learned:
            // Caller must provide model ID
            break
            
        case .adaptive:
            params.adaptiveStrategy = .hybrid
            
            // Build context from dataset if available
            if let dataset = dataset, !dataset.isEmpty {
                let analysis = await analyzeDatasetForAdaptive(dataset)
                params.adaptiveContext = AdaptiveDistance.Context(
                    dataStatistics: analysis.statistics,
                    queryHistory: QueryHistory(),
                    performanceMetrics: AdaptivePerformanceMetrics()
                )
            }
            
        default:
            break
        }
        
        return params
    }
    
    private static func analyzeDatasetForAdaptive(_ dataset: [Vector512]) async -> DatasetAnalysis {
        // Compute statistics for adaptive distance
        var sparsity: Float = 0
        var variance: Float = 0
        
        // Sample for efficiency
        let sample = Array(dataset.prefix(1000))
        
        // Compute sparsity
        var zeroCount = 0
        var totalCount = 0
        for vector in sample {
            for value in vector.toArray() {
                if abs(value) < Float.ulpOfOne {
                    zeroCount += 1
                }
                totalCount += 1
            }
        }
        sparsity = Float(zeroCount) / Float(totalCount)
        
        // Compute dimensional variance
        var means = [Float](repeating: 0, count: 512)
        for vector in sample {
            let values = vector.toArray()
            for i in 0..<512 {
                means[i] += values[i]
            }
        }
        
        for i in 0..<512 {
            means[i] /= Float(sample.count)
        }
        
        var variances = [Float](repeating: 0, count: 512)
        for vector in sample {
            let values = vector.toArray()
            for i in 0..<512 {
                let diff = values[i] - means[i]
                variances[i] += diff * diff
            }
        }
        
        for i in 0..<512 {
            variances[i] /= Float(sample.count)
        }
        
        let meanVariance = variances.reduce(0, +) / Float(variances.count)
        let varianceOfVariances = variances.map { pow($0 - meanVariance, 2) }.reduce(0, +) / Float(variances.count)
        variance = sqrt(varianceOfVariances) / (meanVariance + Float.ulpOfOne)
        
        return DatasetAnalysis(
            statistics: DataStatistics(
                sparsity: sparsity,
                dimensionalVariance: variance,
                outlierRatio: 0.05,
                averageNearestNeighborDistance: 1.0
            )
        )
    }
}

// MARK: - Distance Parameters

/// Parameters for distance computation
public struct DistanceParameters {
    // General parameters
    public var normalized: Bool = false
    
    // Minkowski distance
    public var minkowskiP: Float = 2.0
    
    // Hamming distance
    public var hammingThreshold: Float = 0.01
    
    // Mahalanobis distance
    public var covarianceMatrix: [[Float]]?
    public var mahalanobisCacheId: String?
    
    // Earth Mover's Distance
    public var emdCostFunction: EarthMoversDistance.CostFunction?
    
    // Learned distance
    public var learnedModelId: String?
    
    // Adaptive distance
    public var adaptiveContext: AdaptiveDistance.Context?
    public var adaptiveStrategy: AdaptiveDistance.AdaptationStrategy?
    
    public static let `default` = DistanceParameters()
}

// MARK: - Helper Types

private struct DatasetAnalysis {
    let statistics: DataStatistics
}

// MARK: - Vector512 Extension for Advanced Metrics

extension Vector512 {
    /// Compute distance with async support for advanced metrics
    public func distanceAsync(
        to other: Vector512,
        metric: DistanceMetric,
        parameters: DistanceParameters = .default
    ) async throws -> Float {
        return try await DistanceComputation512.computeDistance(
            self,
            other,
            metric: metric,
            parameters: parameters
        )
    }
}
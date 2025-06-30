// AdaptiveDistance.swift
// VectorStoreKit
//
// Context-adaptive distance metric with auto-tuning capabilities

import Foundation
import simd
import Accelerate

/// Adaptive distance metric that adjusts based on data characteristics
public struct AdaptiveDistance {
    
    /// Adaptation strategy
    public enum AdaptationStrategy {
        case dataDistribution      // Adapt based on data distribution
        case queryPattern         // Adapt based on query patterns
        case hybrid              // Combine multiple signals
        case reinforcement      // Learn from feedback
    }
    
    /// Context for distance computation
    public struct Context {
        public let dataStatistics: DataStatistics
        public let queryHistory: QueryHistory
        public let performanceMetrics: AdaptivePerformanceMetrics
        public let userFeedback: UserFeedback?
        
        public init(
            dataStatistics: DataStatistics,
            queryHistory: QueryHistory,
            performanceMetrics: AdaptivePerformanceMetrics,
            userFeedback: UserFeedback? = nil
        ) {
            self.dataStatistics = dataStatistics
            self.queryHistory = queryHistory
            self.performanceMetrics = performanceMetrics
            self.userFeedback = userFeedback
        }
    }
    
    private var strategy: AdaptationStrategy
    private var parameters: AdaptiveParameters
    private let optimizer: ParameterOptimizer
    
    /// Initialize with adaptation strategy
    public init(strategy: AdaptationStrategy = .hybrid) {
        self.strategy = strategy
        self.parameters = AdaptiveParameters()
        self.optimizer = ParameterOptimizer()
    }
    
    /// Compute adaptive distance between vectors with SIMD optimization
    public func distance(_ a: Vector512, _ b: Vector512, context: Context) -> Float {
        // Select optimal metric based on context
        let selectedMetric = selectMetric(context: context)
        let weight = computeMetricWeight(metric: selectedMetric, context: context)
        
        // Compute base distance
        let baseDistance = computeBaseDistance(a, b, metric: selectedMetric)
        
        // Apply adaptive adjustments
        let adjustedDistance = applyAdjustments(
            distance: baseDistance,
            vectors: (a, b),
            context: context,
            weight: weight
        )
        
        return adjustedDistance
    }
    
    /// Update adaptation parameters based on feedback
    public mutating func updateParameters(feedback: PerformanceFeedback) {
        parameters = optimizer.optimize(
            currentParams: parameters,
            feedback: feedback,
            strategy: strategy
        )
    }
    
    /// Auto-tune parameters based on dataset
    public mutating func autoTune(dataset: [Vector512], sampleQueries: [Vector512]) async throws {
        // Analyze dataset characteristics
        let analysis = try await analyzeDataset(dataset)
        
        // Run sample queries with different parameters
        let candidates = generateParameterCandidates()
        var bestParams = parameters
        var bestScore = Float.infinity
        
        for candidate in candidates {
            parameters = candidate
            let score = try await evaluateParameters(
                dataset: dataset,
                queries: sampleQueries,
                analysis: analysis
            )
            
            if score < bestScore {
                bestScore = score
                bestParams = candidate
            }
        }
        
        parameters = bestParams
    }
    
    // MARK: - Private Methods
    
    private func selectMetric(context: Context) -> DistanceMetric {
        switch strategy {
        case .dataDistribution:
            // Select based on data distribution
            if context.dataStatistics.sparsity > 0.7 {
                return .cosine  // Better for sparse data
            } else if context.dataStatistics.dimensionalVariance > 0.5 {
                return .mahalanobis  // Account for variance
            } else {
                return .euclidean
            }
            
        case .queryPattern:
            // Select based on query patterns
            if context.queryHistory.averageQuerySparsity > 0.6 {
                return .cosine
            } else if context.queryHistory.frequentDimensions.count < 100 {
                return .manhattan  // When only some dimensions matter
            } else {
                return .euclidean
            }
            
        case .hybrid:
            // Combine multiple signals
            let sparsityScore = context.dataStatistics.sparsity * 0.3
            let varianceScore = context.dataStatistics.dimensionalVariance * 0.3
            let queryScore = context.queryHistory.averageQuerySparsity * 0.4
            
            let totalScore = sparsityScore + varianceScore + queryScore
            
            if totalScore > 0.6 {
                return .cosine
            } else if varianceScore > 0.2 {
                return .mahalanobis
            } else {
                return .euclidean
            }
            
        case .reinforcement:
            // Use learned preference
            return parameters.preferredMetric
        }
    }
    
    private func computeMetricWeight(metric: DistanceMetric, context: Context) -> Float {
        // Compute confidence weight based on context
        var weight: Float = 1.0
        
        // Adjust based on data characteristics
        switch metric {
        case .euclidean:
            weight *= (1.0 - context.dataStatistics.sparsity)
        case .cosine:
            weight *= (1.0 + context.dataStatistics.sparsity) / 2.0
        case .manhattan:
            weight *= (1.0 + context.dataStatistics.outlierRatio) / 2.0
        default:
            break
        }
        
        // Adjust based on performance history
        if let perfMetric = context.performanceMetrics.metricPerformance[metric.rawValue] {
            weight *= perfMetric.accuracy
        }
        
        return min(max(weight, 0.1), 2.0)  // Bound weight
    }
    
    private func computeBaseDistance(_ a: Vector512, _ b: Vector512, metric: DistanceMetric) -> Float {
        switch metric {
        case .euclidean:
            return sqrt(a.distanceSquared(to: b))
        case .cosine:
            return 1.0 - a.cosineSimilarity(to: b)
        case .manhattan:
            // Manhattan distance: sum of absolute differences
            let diff = a - b
            var sum: Float = 0
            for i in 0..<512 {
                sum += abs(diff[i])
            }
            return sum
        case .dotProduct:
            // Negative dot product for distance
            var sum: Float = 0
            for i in 0..<512 {
                sum += a[i] * b[i]
            }
            return -sum
        case .chebyshev:
            // Chebyshev distance: max absolute difference
            let diff = a - b
            var maxDiff: Float = 0
            for i in 0..<512 {
                maxDiff = max(maxDiff, abs(diff[i]))
            }
            return maxDiff
        case .minkowski:
            // Minkowski distance: (sum |x_i - y_i|^p)^(1/p)
            let diff = a - b
            var sum: Float = 0
            let p = parameters.minkowskiP
            for i in 0..<512 {
                sum += pow(abs(diff[i]), p)
            }
            return pow(sum, 1.0/p)
        case .hamming:
            // Hamming distance: count of different values
            var count: Float = 0
            let threshold = parameters.hammingThreshold
            for i in 0..<512 {
                if abs(a[i] - b[i]) > threshold {
                    count += 1
                }
            }
            return count
        case .jaccard:
            // Jaccard distance: 1 - (intersection / union)
            var intersection: Float = 0
            var union: Float = 0
            for i in 0..<512 {
                let minVal = min(a[i], b[i])
                let maxVal = max(a[i], b[i])
                intersection += minVal
                union += maxVal
            }
            return union > 0 ? 1.0 - (intersection / union) : 0
        default:
            // For complex metrics, fall back to euclidean
            return sqrt(a.distanceSquared(to: b))
        }
    }
    
    private func applyAdjustments(
        distance: Float,
        vectors: (Vector512, Vector512),
        context: Context,
        weight: Float
    ) -> Float {
        var adjusted = distance * weight
        
        // Apply dimensional weighting if available
        if !parameters.dimensionalWeights.isEmpty {
            let weightedDistance = computeWeightedDistance(
                vectors.0,
                vectors.1,
                weights: parameters.dimensionalWeights
            )
            adjusted = adjusted * 0.7 + weightedDistance * 0.3
        }
        
        // Apply locality-sensitive adjustments
        if parameters.useLocalityAdjustment {
            let localityFactor = computeLocalityFactor(distance, context: context)
            adjusted *= localityFactor
        }
        
        // Apply query-specific adjustments
        if let queryAdjustment = context.queryHistory.querySpecificAdjustment {
            adjusted *= queryAdjustment
        }
        
        return adjusted
    }
    
    /// SIMD-optimized weighted distance computation
    internal func computeWeightedDistance(_ a: Vector512, _ b: Vector512, weights: [Float]) -> Float {
        return a.withUnsafeMetalBytes { aBytes in
            b.withUnsafeMetalBytes { bBytes in
                let aPtr = aBytes.bindMemory(to: SIMD8<Float>.self)
                let bPtr = bBytes.bindMemory(to: SIMD8<Float>.self)
                
                var sum0 = SIMD8<Float>.zero
                var sum1 = SIMD8<Float>.zero
                var sum2 = SIMD8<Float>.zero
                var sum3 = SIMD8<Float>.zero
                
                weights.withUnsafeBufferPointer { weightsPtr in
                    let weightsBase = weightsPtr.baseAddress
                    let weightsCount = weightsPtr.count
                    
                    // Process in groups of 32 elements (4 SIMD8)
                    for i in stride(from: 0, to: 64, by: 4) {
                        let diff0 = aPtr[i] - bPtr[i]
                        let diff1 = aPtr[i + 1] - bPtr[i + 1]
                        let diff2 = aPtr[i + 2] - bPtr[i + 2]
                        let diff3 = aPtr[i + 3] - bPtr[i + 3]
                        
                        // Load weights with bounds checking
                        let w0: SIMD8<Float>
                        let w1: SIMD8<Float>
                        let w2: SIMD8<Float>
                        let w3: SIMD8<Float>
                        
                        let baseIdx = i * 8
                        if baseIdx + 32 <= weightsCount, let wBase = weightsBase {
                            w0 = wBase.advanced(by: baseIdx).withMemoryRebound(to: SIMD8<Float>.self, capacity: 4) { ptr in ptr[0] }
                            w1 = wBase.advanced(by: baseIdx).withMemoryRebound(to: SIMD8<Float>.self, capacity: 4) { ptr in ptr[1] }
                            w2 = wBase.advanced(by: baseIdx).withMemoryRebound(to: SIMD8<Float>.self, capacity: 4) { ptr in ptr[2] }
                            w3 = wBase.advanced(by: baseIdx).withMemoryRebound(to: SIMD8<Float>.self, capacity: 4) { ptr in ptr[3] }
                        } else {
                            // Handle out-of-bounds with scalar fallback
                            var w0_scalar = [Float](repeating: 1.0, count: 8)
                            var w1_scalar = [Float](repeating: 1.0, count: 8)
                            var w2_scalar = [Float](repeating: 1.0, count: 8)
                            var w3_scalar = [Float](repeating: 1.0, count: 8)
                            
                            for j in 0..<8 {
                                let idx0 = baseIdx + j
                                let idx1 = baseIdx + 8 + j
                                let idx2 = baseIdx + 16 + j
                                let idx3 = baseIdx + 24 + j
                                
                                if idx0 < weightsCount { w0_scalar[j] = weights[idx0] }
                                if idx1 < weightsCount { w1_scalar[j] = weights[idx1] }
                                if idx2 < weightsCount { w2_scalar[j] = weights[idx2] }
                                if idx3 < weightsCount { w3_scalar[j] = weights[idx3] }
                            }
                            
                            w0 = SIMD8<Float>(w0_scalar)
                            w1 = SIMD8<Float>(w1_scalar)
                            w2 = SIMD8<Float>(w2_scalar)
                            w3 = SIMD8<Float>(w3_scalar)
                        }
                        
                        // Compute weighted squared differences
                        sum0 += w0 * diff0 * diff0
                        sum1 += w1 * diff1 * diff1
                        sum2 += w2 * diff2 * diff2
                        sum3 += w3 * diff3 * diff3
                    }
                }
                
                let totalSum = sum0 + sum1 + sum2 + sum3
                return sqrt(totalSum.sum())
            }
        }
    }
    
    private func computeLocalityFactor(_ distance: Float, context: Context) -> Float {
        // Adjust based on local density
        let avgDistance = context.dataStatistics.averageNearestNeighborDistance
        
        if distance < avgDistance * 0.5 {
            // Very close vectors - emphasize small differences
            return 1.2
        } else if distance > avgDistance * 2.0 {
            // Far vectors - de-emphasize differences
            return 0.8
        } else {
            // Normal range
            return 1.0
        }
    }
    
    private func analyzeDataset(_ dataset: [Vector512]) async throws -> DatasetAnalysis {
        // Compute various statistics about the dataset
        let sparsity = computeSparsity(dataset)
        let variance = computeDimensionalVariance(dataset)
        let clustering = await computeClusteringCoefficient(dataset)
        
        return DatasetAnalysis(
            sparsity: sparsity,
            dimensionalVariance: variance,
            clusteringCoefficient: clustering,
            size: dataset.count
        )
    }
    
    /// SIMD-optimized sparsity computation
    internal func computeSparsity(_ dataset: [Vector512]) -> Float {
        var zeroCount = 0
        var totalCount = 0
        
        let threshold = SIMD8<Float>(repeating: Float.ulpOfOne)
        
        for vector in dataset.prefix(100) {  // Sample
            vector.withUnsafeMetalBytes { bytes in
                let ptr = bytes.bindMemory(to: SIMD8<Float>.self)
                
                for i in 0..<64 {
                    let values = ptr[i]
                    // Use SIMD8 magnitude for absolute value
                    var absValues = SIMD8<Float>()
                    for k in 0..<8 {
                        absValues[k] = abs(values[k])
                    }
                    let isZero = absValues .< threshold
                    
                    // Count zeros using SIMD mask
                    for j in 0..<8 {
                        if isZero[j] {
                            zeroCount += 1
                        }
                    }
                }
            }
            totalCount += 512
        }
        
        return Float(zeroCount) / Float(totalCount)
    }
    
    /// SIMD-optimized dimensional variance computation
    internal func computeDimensionalVariance(_ dataset: [Vector512]) -> Float {
        guard !dataset.isEmpty else { return 0 }
        
        // Sample for efficiency
        let sample = Array(dataset.prefix(100))
        let sampleCount = Float(sample.count)
        
        // Compute mean for each dimension using SIMD
        var means = [Float](repeating: 0, count: 512)
        var variances = [Float](repeating: 0, count: 512)
        
        // First pass: compute means
        means.withUnsafeMutableBufferPointer { meansPtr in
            meansPtr.baseAddress!.withMemoryRebound(to: SIMD8<Float>.self, capacity: 64) { meansSIMD in
                for vector in sample {
                    vector.withUnsafeMetalBytes { bytes in
                        let vectorSIMD = bytes.bindMemory(to: SIMD8<Float>.self)
                        
                        for i in 0..<64 {
                            meansSIMD[i] += vectorSIMD[i]
                        }
                    }
                }
                
                // Divide by sample count
                let invSampleCount = SIMD8<Float>(repeating: 1.0 / sampleCount)
                for i in 0..<64 {
                    meansSIMD[i] *= invSampleCount
                }
            }
        }
        
        // Second pass: compute variances using SIMD
        variances.withUnsafeMutableBufferPointer { variancesPtr in
            variancesPtr.baseAddress!.withMemoryRebound(to: SIMD8<Float>.self, capacity: 64) { variancesSIMD in
                means.withUnsafeBufferPointer { meansPtr in
                    meansPtr.baseAddress!.withMemoryRebound(to: SIMD8<Float>.self, capacity: 64) { meansSIMD in
                        for vector in sample {
                            vector.withUnsafeMetalBytes { bytes in
                                let vectorSIMD = bytes.bindMemory(to: SIMD8<Float>.self)
                                
                                for i in 0..<64 {
                                    let diff = vectorSIMD[i] - meansSIMD[i]
                                    variancesSIMD[i] += diff * diff
                                }
                            }
                        }
                        
                        // Divide by sample count
                        let invSampleCount = SIMD8<Float>(repeating: 1.0 / sampleCount)
                        for i in 0..<64 {
                            variancesSIMD[i] *= invSampleCount
                        }
                    }
                }
            }
        }
        
        // Compute coefficient of variation using SIMD
        var meanVariance: Float = 0
        var varianceOfVariances: Float = 0
        
        variances.withUnsafeBufferPointer { variancesPtr in
            variancesPtr.baseAddress!.withMemoryRebound(to: SIMD8<Float>.self, capacity: 64) { variancesSIMD in
                var sumVec = SIMD8<Float>.zero
                for i in 0..<64 {
                    sumVec += variancesSIMD[i]
                }
                meanVariance = sumVec.sum() / 512.0
                
                let meanVec = SIMD8<Float>(repeating: meanVariance)
                var sumOfSquareVec = SIMD8<Float>.zero
                
                for i in 0..<64 {
                    let diff = variancesSIMD[i] - meanVec
                    sumOfSquareVec += diff * diff
                }
                varianceOfVariances = sumOfSquareVec.sum() / 512.0
            }
        }
        
        return sqrt(varianceOfVariances) / (meanVariance + Float.ulpOfOne)
    }
    
    private func computeClusteringCoefficient(_ dataset: [Vector512]) async -> Float {
        // Simplified clustering coefficient
        // In practice, would use more sophisticated analysis
        return 0.5
    }
    
    private func generateParameterCandidates() -> [AdaptiveParameters] {
        var candidates: [AdaptiveParameters] = []
        
        // Generate different parameter combinations
        for metric in [DistanceMetric.euclidean, .cosine, .manhattan] {
            for locality in [true, false] {
                for minkowskiP in [2.0, 3.0, 4.0] {
                    var params = AdaptiveParameters()
                    params.preferredMetric = metric
                    params.useLocalityAdjustment = locality
                    params.minkowskiP = minkowskiP
                    candidates.append(params)
                }
            }
        }
        
        return candidates
    }
    
    private func evaluateParameters(
        dataset: [Vector512],
        queries: [Vector512],
        analysis: DatasetAnalysis
    ) async throws -> Float {
        // Evaluate parameter quality using locality preservation
        var totalError: Float = 0
        
        for query in queries.prefix(10) {
            // Find true nearest neighbors with euclidean
            let trueNeighbors = findNearestNeighbors(
                query: query,
                dataset: dataset,
                k: 10,
                metric: .euclidean
            )
            
            // Find neighbors with current parameters
            let context = Context(
                dataStatistics: DataStatistics(from: analysis),
                queryHistory: QueryHistory(),
                performanceMetrics: AdaptivePerformanceMetrics()
            )
            
            let adaptiveNeighbors = findNearestNeighborsAdaptive(
                query: query,
                dataset: dataset,
                k: 10,
                context: context
            )
            
            // Compute ranking error
            let error = computeRankingError(
                true: trueNeighbors,
                predicted: adaptiveNeighbors
            )
            
            totalError += error
        }
        
        return totalError / Float(queries.count)
    }
    
    private func findNearestNeighbors(
        query: Vector512,
        dataset: [Vector512],
        k: Int,
        metric: DistanceMetric
    ) -> [(index: Int, distance: Float)] {
        var distances: [(index: Int, distance: Float)] = []
        
        for (index, vector) in dataset.enumerated() {
            let distance = computeBaseDistance(query, vector, metric: metric)
            distances.append((index, distance))
        }
        
        return Array(distances.sorted { $0.distance < $1.distance }.prefix(k))
    }
    
    private func findNearestNeighborsAdaptive(
        query: Vector512,
        dataset: [Vector512],
        k: Int,
        context: Context
    ) -> [(index: Int, distance: Float)] {
        var distances: [(index: Int, distance: Float)] = []
        
        for (index, vector) in dataset.enumerated() {
            let distance = self.distance(query, vector, context: context)
            distances.append((index, distance))
        }
        
        return Array(distances.sorted { $0.distance < $1.distance }.prefix(k))
    }
    
    private func computeRankingError(
        true trueNeighbors: [(index: Int, distance: Float)],
        predicted: [(index: Int, distance: Float)]
    ) -> Float {
        // Compute normalized discounted cumulative gain (NDCG)
        var dcg: Float = 0
        var idcg: Float = 0
        
        for i in 0..<trueNeighbors.count {
            let trueIndex = trueNeighbors[i].index
            
            // Find position in predicted
            if let predictedPos = predicted.firstIndex(where: { $0.index == trueIndex }) {
                let relevance: Float = 1.0 / Float(i + 1)  // Higher rank = higher relevance
                dcg += relevance / log2(Float(predictedPos + 2))
            }
            
            // Ideal DCG
            idcg += 1.0 / (Float(i + 1) * log2(Float(i + 2)))
        }
        
        let ndcg = dcg / (idcg + Float.ulpOfOne)
        return 1.0 - ndcg  // Convert to error
    }
}

// MARK: - Supporting Types

/// Adaptive parameters that can be tuned
public struct AdaptiveParameters {
    var preferredMetric: DistanceMetric = .euclidean
    var dimensionalWeights: [Float] = []
    var useLocalityAdjustment: Bool = true
    var minkowskiP: Float = 2.0
    var hammingThreshold: Float = 0.01
    var adaptationRate: Float = 0.1
}

/// Statistics about the data
public struct DataStatistics {
    public let sparsity: Float
    public let dimensionalVariance: Float
    public let outlierRatio: Float
    public let averageNearestNeighborDistance: Float
    
    init(from analysis: DatasetAnalysis) {
        self.sparsity = analysis.sparsity
        self.dimensionalVariance = analysis.dimensionalVariance
        self.outlierRatio = 0.05  // Default
        self.averageNearestNeighborDistance = 1.0  // Default
    }
}

/// Query history and patterns
public struct QueryHistory {
    public let averageQuerySparsity: Float
    public let frequentDimensions: Set<Int>
    public let querySpecificAdjustment: Float?
    
    public init(
        averageQuerySparsity: Float = 0.5,
        frequentDimensions: Set<Int> = [],
        querySpecificAdjustment: Float? = nil
    ) {
        self.averageQuerySparsity = averageQuerySparsity
        self.frequentDimensions = frequentDimensions
        self.querySpecificAdjustment = querySpecificAdjustment
    }
}

/// Performance metrics for different distance functions
public struct AdaptivePerformanceMetrics {
    public let metricPerformance: [String: MetricPerformance]
    
    public init(metricPerformance: [String: MetricPerformance] = [:]) {
        self.metricPerformance = metricPerformance
    }
}

public struct MetricPerformance {
    public let accuracy: Float
    public let speed: Float
    public let consistency: Float
}

/// User feedback for reinforcement learning
public struct UserFeedback {
    public let relevantResults: Set<VectorID>
    public let irrelevantResults: Set<VectorID>
    public let preferredOrdering: [VectorID]?
}

/// Performance feedback for parameter updates
public struct PerformanceFeedback {
    public let queryId: String
    public let selectedMetric: DistanceMetric
    public let resultQuality: Float  // 0-1
    public let computationTime: TimeInterval
    public let userSatisfaction: Float?  // 0-1
}

/// Parameter optimizer using gradient-free optimization
struct ParameterOptimizer {
    func optimize(
        currentParams: AdaptiveParameters,
        feedback: PerformanceFeedback,
        strategy: AdaptiveDistance.AdaptationStrategy
    ) -> AdaptiveParameters {
        var newParams = currentParams
        
        // Simple gradient-free optimization
        let qualityGradient = feedback.resultQuality - 0.8  // Target 80% quality
        
        // Update adaptation rate
        if qualityGradient < 0 {
            newParams.adaptationRate = newParams.adaptationRate * 0.9  // Slow down if performing poorly
        } else {
            newParams.adaptationRate = newParams.adaptationRate * 1.1  // Speed up if performing well
        }
        
        // Update metric preference based on feedback
        if feedback.resultQuality > 0.9 {
            newParams.preferredMetric = feedback.selectedMetric
        }
        
        return newParams
    }
}

/// Dataset analysis results
struct DatasetAnalysis {
    let sparsity: Float
    let dimensionalVariance: Float
    let clusteringCoefficient: Float
    let size: Int
}

/// Global cache for adaptive distance computations
public actor AdaptiveDistanceCache {
    private var contextCache: [String: AdaptiveDistance.Context] = [:]
    private var parameterCache: [String: AdaptiveParameters] = [:]
    
    public static let shared = AdaptiveDistanceCache()
    
    public func getContext(id: String) -> AdaptiveDistance.Context? {
        return contextCache[id]
    }
    
    public func cacheContext(id: String, context: AdaptiveDistance.Context) {
        contextCache[id] = context
    }
    
    public func getParameters(id: String) -> AdaptiveParameters? {
        return parameterCache[id]
    }
    
    public func cacheParameters(id: String, parameters: AdaptiveParameters) {
        parameterCache[id] = parameters
    }
    
    public func clear() {
        contextCache.removeAll()
        parameterCache.removeAll()
    }
}
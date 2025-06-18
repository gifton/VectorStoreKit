// AdvancedDistanceMetricsExample.swift
// VectorStoreKit Examples
//
// Demonstrates usage of advanced distance metrics

import Foundation
import VectorStoreKit

@main
struct AdvancedDistanceMetricsExample {
    static func main() async throws {
        print("=== Advanced Distance Metrics Example ===\n")
        
        // Create sample vectors
        let vector1 = Vector512((0..<512).map { Float($0) / 512.0 })
        let vector2 = Vector512((0..<512).map { Float(511 - $0) / 512.0 })
        let vector3 = Vector512((0..<512).map { sin(Float($0) * .pi / 256) })
        
        // Basic metrics (synchronous)
        print("Basic Distance Metrics:")
        print("Euclidean: \(DistanceComputation512.euclideanDistance(vector1, vector2))")
        print("Cosine: \(DistanceComputation512.cosineDistance(vector1, vector2))")
        print("Manhattan: \(DistanceComputation512.manhattanDistance(vector1, vector2))")
        print("")
        
        // Earth Mover's Distance
        print("Earth Mover's Distance:")
        let emdDistance = DistanceComputation512.earthMoversDistance(vector1, vector2)
        print("EMD (Euclidean cost): \(emdDistance)")
        
        let emdApprox = EarthMoversDistance(costFunction: .euclidean)
            .approximateDistance(vector1, vector2, iterations: 100, regularization: 0.05)
        print("EMD (Sinkhorn approximation): \(emdApprox)")
        print("")
        
        // Mahalanobis Distance (requires covariance matrix)
        print("Mahalanobis Distance:")
        do {
            // Create a simple diagonal covariance matrix
            var covarianceMatrix = [[Float]](repeating: [Float](repeating: 0, count: 512), count: 512)
            for i in 0..<512 {
                covarianceMatrix[i][i] = 1.0 + Float(i) / 512.0  // Variable variance
            }
            
            let mahalanobisDistance = try await DistanceComputation512.mahalanobisDistance(
                vector1,
                vector2,
                covarianceMatrix: covarianceMatrix
            )
            print("Mahalanobis: \(mahalanobisDistance)")
        } catch {
            print("Mahalanobis error: \(error)")
        }
        print("")
        
        // Learned Distance
        print("Learned Distance:")
        do {
            // Create and train a simple learned distance model
            let learnedModel = try LearnedDistance(
                modelType: .neural(layers: [256, 128, 64]),
                useGPU: false
            )
            
            // Cache the model
            await LearnedDistanceCache.shared.cacheModel(id: "example_model", model: learnedModel)
            
            // Train with some examples (positive = similar, negative = dissimilar)
            let positives = [(vector1, vector1), (vector2, vector2)]
            let negatives = [(vector1, vector2), (vector1, vector3)]
            
            try await learnedModel.update(positives: positives, negatives: negatives)
            
            // Compute learned distance
            let learnedDist = try await DistanceComputation512.learnedDistance(
                vector1,
                vector2,
                modelId: "example_model"
            )
            print("Learned distance: \(learnedDist)")
        } catch {
            print("Learned distance error: \(error)")
        }
        print("")
        
        // Adaptive Distance
        print("Adaptive Distance:")
        
        // Create context for adaptive distance
        let dataStatistics = DataStatistics(
            sparsity: 0.3,
            dimensionalVariance: 0.5,
            outlierRatio: 0.05,
            averageNearestNeighborDistance: 1.5
        )
        
        let queryHistory = QueryHistory(
            averageQuerySparsity: 0.4,
            frequentDimensions: Set(0..<100),
            querySpecificAdjustment: 1.1
        )
        
        let performanceMetrics = PerformanceMetrics(
            metricPerformance: [
                "euclidean": MetricPerformance(accuracy: 0.9, speed: 0.95, consistency: 0.85),
                "cosine": MetricPerformance(accuracy: 0.85, speed: 0.9, consistency: 0.9)
            ]
        )
        
        let context = AdaptiveDistance.Context(
            dataStatistics: dataStatistics,
            queryHistory: queryHistory,
            performanceMetrics: performanceMetrics
        )
        
        let adaptiveDist = DistanceComputation512.adaptiveDistance(vector1, vector2, context: context)
        print("Adaptive distance: \(adaptiveDist)")
        print("")
        
        // Batch computation with advanced metrics
        print("Batch Distance Computation:")
        let candidates = [vector1, vector2, vector3]
        
        // Setup parameters for batch computation
        var parameters = DistanceParameters()
        parameters.adaptiveContext = context
        
        let batchDistances = try await DistanceComputation512.batchComputeDistance(
            query: vector1,
            candidates: candidates,
            metric: .adaptive,
            parameters: parameters
        )
        
        print("Batch adaptive distances: \(batchDistances)")
        
        // Auto-tune adaptive distance
        print("\nAuto-tuning Adaptive Distance...")
        var adaptiveModel = AdaptiveDistance(strategy: .hybrid)
        
        let dataset = (0..<100).map { _ in
            Vector512((0..<512).map { _ in Float.random(in: -1...1) })
        }
        
        let sampleQueries = (0..<10).map { _ in
            Vector512((0..<512).map { _ in Float.random(in: -1...1) })
        }
        
        try await adaptiveModel.autoTune(dataset: dataset, sampleQueries: sampleQueries)
        print("Auto-tuning complete!")
        
        // Performance comparison
        print("\nPerformance Comparison:")
        let testVector1 = Vector512((0..<512).map { _ in Float.random(in: -1...1) })
        let testVector2 = Vector512((0..<512).map { _ in Float.random(in: -1...1) })
        
        let metrics: [DistanceMetric] = [.euclidean, .cosine, .manhattan, .earth_mover]
        
        for metric in metrics {
            let start = CFAbsoluteTimeGetCurrent()
            
            let distance = try await DistanceComputation512.computeDistance(
                testVector1,
                testVector2,
                metric: metric
            )
            
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            print("\(metric): \(distance) (took \(String(format: "%.3f", elapsed))ms)")
        }
        
        print("\n=== Example Complete ===")
    }
}
// AdvancedDistanceMetricsTests.swift
// VectorStoreKit Tests
//
// Tests for advanced distance metric implementations

import XCTest
@testable import VectorStoreKit

final class AdvancedDistanceMetricsTests: XCTestCase {
    
    var vector1: Vector512!
    var vector2: Vector512!
    var vector3: Vector512!
    
    override func setUp() {
        super.setUp()
        
        // Create test vectors
        vector1 = Vector512((0..<512).map { Float($0) / 512.0 })
        vector2 = Vector512((0..<512).map { Float(511 - $0) / 512.0 })
        vector3 = Vector512((0..<512).map { sin(Float($0) * .pi / 256) })
    }
    
    // MARK: - Earth Mover's Distance Tests
    
    func testEarthMoversDistance() {
        let emd = EarthMoversDistance(costFunction: .euclidean)
        
        // Same vector should have distance 0
        let sameDist = emd.distance(vector1, vector1)
        XCTAssertEqual(sameDist, 0, accuracy: 0.0001)
        
        // Different vectors should have positive distance
        let diffDist = emd.distance(vector1, vector2)
        XCTAssertGreaterThan(diffDist, 0)
        
        // Test symmetry
        let dist12 = emd.distance(vector1, vector2)
        let dist21 = emd.distance(vector2, vector1)
        XCTAssertEqual(dist12, dist21, accuracy: 0.0001)
    }
    
    func testEarthMoversDistanceApproximation() {
        let emd = EarthMoversDistance(costFunction: .euclidean)
        
        // Exact EMD
        let exactDist = emd.distance(vector1, vector2)
        
        // Sinkhorn approximation
        let approxDist = emd.approximateDistance(
            vector1,
            vector2,
            iterations: 100,
            regularization: 0.01
        )
        
        // Approximation should be close to exact
        XCTAssertEqual(exactDist, approxDist, accuracy: exactDist * 0.2) // Within 20%
    }
    
    // MARK: - Mahalanobis Distance Tests
    
    func testMahalanobisDistanceIdentity() async throws {
        // Identity covariance matrix should give Euclidean distance
        var identityMatrix = [[Float]](repeating: [Float](repeating: 0, count: 512), count: 512)
        for i in 0..<512 {
            identityMatrix[i][i] = 1.0
        }
        
        let mahalanobis = try MahalanobisDistance(covarianceMatrix: identityMatrix)
        let mahaDist = mahalanobis.distance(vector1, vector2)
        let euclideanDist = DistanceComputation512.euclideanDistance(vector1, vector2)
        
        XCTAssertEqual(mahaDist, euclideanDist, accuracy: 0.001)
    }
    
    func testMahalanobisDistanceInvalidMatrix() async {
        // Non-square matrix should throw error
        let invalidMatrix = [[Float]](repeating: [Float](repeating: 1, count: 256), count: 512)
        
        do {
            _ = try MahalanobisDistance(covarianceMatrix: invalidMatrix)
            XCTFail("Should have thrown error for non-square matrix")
        } catch let error as VectorStoreError {
            XCTAssertEqual(error.category, .distanceComputation)
            XCTAssertEqual(error.code, .invalidInput)
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }
    
    // MARK: - Learned Distance Tests
    
    func testLearnedDistanceNeural() async throws {
        let model = try LearnedDistance(
            modelType: .neural(layers: [128, 64]),
            useGPU: false,
            batchSize: 16
        )
        
        // Before training, distance should be reasonable
        let initialDist = try model.distance(vector1, vector2)
        XCTAssertGreaterThanOrEqual(initialDist, 0)
        XCTAssertLessThanOrEqual(initialDist, 1)
        
        // Train the model
        let positives = [(vector1, vector1), (vector2, vector2), (vector3, vector3)]
        let negatives = [(vector1, vector2), (vector1, vector3), (vector2, vector3)]
        
        try await model.update(positives: positives, negatives: negatives)
        
        // After training, similar vectors should have lower distance
        let trainedSimilar = try model.distance(vector1, vector1)
        let trainedDifferent = try model.distance(vector1, vector2)
        
        XCTAssertLessThan(trainedSimilar, trainedDifferent)
    }
    
    func testLearnedDistanceBatch() async throws {
        let model = try LearnedDistance(
            modelType: .siamese,
            useGPU: false
        )
        
        let candidates = [vector1, vector2, vector3]
        let batchDistances = try await model.batchDistance(query: vector1, candidates: candidates)
        
        XCTAssertEqual(batchDistances.count, 3)
        XCTAssertEqual(batchDistances[0], 0, accuracy: 0.0001) // Distance to itself
        XCTAssertGreaterThan(batchDistances[1], 0) // Distance to vector2
        XCTAssertGreaterThan(batchDistances[2], 0) // Distance to vector3
    }
    
    // MARK: - Adaptive Distance Tests
    
    func testAdaptiveDistanceBasic() {
        let adaptive = AdaptiveDistance(strategy: .dataDistribution)
        
        let context = AdaptiveDistance.Context(
            dataStatistics: DataStatistics(
                sparsity: 0.1,
                dimensionalVariance: 0.3,
                outlierRatio: 0.05,
                averageNearestNeighborDistance: 1.0
            ),
            queryHistory: QueryHistory(),
            performanceMetrics: PerformanceMetrics()
        )
        
        let distance = adaptive.distance(vector1, vector2, context: context)
        XCTAssertGreaterThan(distance, 0)
    }
    
    func testAdaptiveDistanceAutoTune() async throws {
        var adaptive = AdaptiveDistance(strategy: .hybrid)
        
        // Create dataset
        let dataset = (0..<50).map { _ in
            Vector512((0..<512).map { _ in Float.random(in: -1...1) })
        }
        
        let queries = (0..<5).map { _ in
            Vector512((0..<512).map { _ in Float.random(in: -1...1) })
        }
        
        // Auto-tune should complete without error
        try await adaptive.autoTune(dataset: dataset, sampleQueries: queries)
    }
    
    func testAdaptiveDistanceStrategySelection() {
        let sparseContext = AdaptiveDistance.Context(
            dataStatistics: DataStatistics(
                sparsity: 0.8, // High sparsity
                dimensionalVariance: 0.2,
                outlierRatio: 0.05,
                averageNearestNeighborDistance: 1.0
            ),
            queryHistory: QueryHistory(),
            performanceMetrics: PerformanceMetrics()
        )
        
        let denseContext = AdaptiveDistance.Context(
            dataStatistics: DataStatistics(
                sparsity: 0.1, // Low sparsity
                dimensionalVariance: 0.8, // High variance
                outlierRatio: 0.05,
                averageNearestNeighborDistance: 1.0
            ),
            queryHistory: QueryHistory(),
            performanceMetrics: PerformanceMetrics()
        )
        
        let adaptiveSparse = AdaptiveDistance(strategy: .dataDistribution)
        let adaptiveDense = AdaptiveDistance(strategy: .dataDistribution)
        
        // The adaptive distance should choose different strategies
        let sparseDist = adaptiveSparse.distance(vector1, vector2, context: sparseContext)
        let denseDist = adaptiveDense.distance(vector1, vector2, context: denseContext)
        
        // Both should be valid distances
        XCTAssertGreaterThan(sparseDist, 0)
        XCTAssertGreaterThan(denseDist, 0)
    }
    
    // MARK: - Integration Tests
    
    func testDistanceComputation512Integration() async throws {
        // Test the integrated distance computation
        var params = DistanceParameters()
        
        // Earth Mover's Distance
        let emdDist = try await DistanceComputation512.computeDistance(
            vector1,
            vector2,
            metric: .earth_mover,
            parameters: params
        )
        XCTAssertGreaterThan(emdDist, 0)
        
        // Mahalanobis (should fail without covariance matrix)
        do {
            _ = try await DistanceComputation512.computeDistance(
                vector1,
                vector2,
                metric: .mahalanobis,
                parameters: params
            )
            XCTFail("Should have thrown error for missing covariance matrix")
        } catch let error as VectorStoreError {
            XCTAssertEqual(error.code, .missingRequiredParameter)
        }
        
        // Learned distance (should fail without model ID)
        do {
            _ = try await DistanceComputation512.computeDistance(
                vector1,
                vector2,
                metric: .learned,
                parameters: params
            )
            XCTFail("Should have thrown error for missing model ID")
        } catch let error as VectorStoreError {
            XCTAssertEqual(error.code, .missingRequiredParameter)
        }
        
        // Adaptive distance with context
        params.adaptiveContext = AdaptiveDistance.Context(
            dataStatistics: DataStatistics(
                sparsity: 0.3,
                dimensionalVariance: 0.5,
                outlierRatio: 0.05,
                averageNearestNeighborDistance: 1.0
            ),
            queryHistory: QueryHistory(),
            performanceMetrics: PerformanceMetrics()
        )
        
        let adaptiveDist = try await DistanceComputation512.computeDistance(
            vector1,
            vector2,
            metric: .adaptive,
            parameters: params
        )
        XCTAssertGreaterThan(adaptiveDist, 0)
    }
    
    func testBatchDistanceComputation() async throws {
        let candidates = [vector1, vector2, vector3]
        var params = DistanceParameters()
        
        // Test EMD batch
        let emdBatch = try await DistanceComputation512.batchComputeDistance(
            query: vector1,
            candidates: candidates,
            metric: .earth_mover,
            parameters: params
        )
        
        XCTAssertEqual(emdBatch.count, 3)
        XCTAssertEqual(emdBatch[0], 0, accuracy: 0.0001) // Distance to itself
        XCTAssertGreaterThan(emdBatch[1], 0)
        XCTAssertGreaterThan(emdBatch[2], 0)
        
        // Test adaptive batch
        params.adaptiveContext = AdaptiveDistance.Context(
            dataStatistics: DataStatistics(
                sparsity: 0.3,
                dimensionalVariance: 0.5,
                outlierRatio: 0.05,
                averageNearestNeighborDistance: 1.0
            ),
            queryHistory: QueryHistory(),
            performanceMetrics: PerformanceMetrics()
        )
        
        let adaptiveBatch = try await DistanceComputation512.batchComputeDistance(
            query: vector1,
            candidates: candidates,
            metric: .adaptive,
            parameters: params
        )
        
        XCTAssertEqual(adaptiveBatch.count, 3)
    }
    
    // MARK: - Performance Tests
    
    func testDistanceMetricsPerformance() async throws {
        self.measure {
            let v1 = Vector512((0..<512).map { _ in Float.random(in: -1...1) })
            let v2 = Vector512((0..<512).map { _ in Float.random(in: -1...1) })
            
            // Measure EMD
            let emd = EarthMoversDistance()
            _ = emd.approximateDistance(v1, v2, iterations: 50)
        }
    }
}

// MARK: - Helper Extensions for Testing

extension DataStatistics {
    init(
        sparsity: Float,
        dimensionalVariance: Float,
        outlierRatio: Float,
        averageNearestNeighborDistance: Float
    ) {
        self = DataStatistics(
            from: DatasetAnalysis(
                sparsity: sparsity,
                dimensionalVariance: dimensionalVariance,
                clusteringCoefficient: 0.5,
                size: 1000
            )
        )
    }
}

private struct DatasetAnalysis {
    let sparsity: Float
    let dimensionalVariance: Float
    let clusteringCoefficient: Float
    let size: Int
}
// VectorStoreKit: Neural Clustering Tests
//
// Tests for neural clustering implementation in IVF index

import XCTest
@testable import VectorStoreKit

final class NeuralClusteringTests: XCTestCase {
    
    // MARK: - Test Data Generation
    
    func generateTestVectors(count: Int, dimensions: Int, clusters: Int = 5) -> [[Float]] {
        var vectors: [[Float]] = []
        
        // Generate clustered data
        for _ in 0..<count {
            let clusterIdx = Int.random(in: 0..<clusters)
            let centerOffset = Float(clusterIdx) * 10.0
            
            var vector: [Float] = []
            for d in 0..<dimensions {
                // Add some noise around cluster center
                let value = centerOffset + Float.random(in: -2...2)
                vector.append(value)
            }
            vectors.append(vector)
        }
        
        return vectors
    }
    
    // MARK: - Neural Clustering Tests
    
    func testNeuralClusteringInitialization() async throws {
        let config = NeuralClusteringConfiguration(
            dimensions: 128,
            numberOfClusters: 10,
            trainingEpochs: 10,
            batchSize: 32
        )
        
        let clustering = try await NeuralClustering(configuration: config)
        
        // Should initialize without errors
        let centroids = await clustering.getCentroids()
        XCTAssertEqual(centroids.count, 0) // Not trained yet
    }
    
    func testNeuralClusteringTraining() async throws {
        let dimensions = 64
        let numClusters = 5
        let config = NeuralClusteringConfiguration(
            dimensions: dimensions,
            numberOfClusters: numClusters,
            trainingEpochs: 5, // Quick test
            batchSize: 16
        )
        
        let clustering = try await NeuralClustering(configuration: config)
        let testVectors = generateTestVectors(count: 200, dimensions: dimensions, clusters: numClusters)
        
        // Train clustering
        let result = try await clustering.train(vectors: testVectors)
        
        // Verify results
        XCTAssertEqual(result.centroids.count, numClusters)
        XCTAssertNotNil(result.clusterNetwork)
        
        // Test cluster assignment
        let testVector = testVectors[0]
        let nearestClusters = try await clustering.findNearestCentroids(for: testVector, count: 3)
        XCTAssertEqual(nearestClusters.count, 3)
        XCTAssertTrue(nearestClusters.allSatisfy { $0 >= 0 && $0 < numClusters })
    }
    
    func testAdaptiveProbing() async throws {
        let dimensions = 32
        let numClusters = 10
        let config = NeuralClusteringConfiguration(
            dimensions: dimensions,
            numberOfClusters: numClusters,
            trainingEpochs: 5,
            adaptiveProbing: true,
            defaultProbes: 3
        )
        
        let clustering = try await NeuralClustering(configuration: config)
        let testVectors = generateTestVectors(count: 100, dimensions: dimensions, clusters: numClusters)
        
        // Train clustering
        _ = try await clustering.train(vectors: testVectors)
        
        // Test probe prediction
        let testQuery = testVectors[0]
        let probeCount = try await clustering.predictProbeCount(for: testQuery, targetRecall: 0.95)
        
        XCTAssertGreaterThan(probeCount, 0)
        XCTAssertLessThanOrEqual(probeCount, numClusters)
    }
    
    func testClusterProbabilities() async throws {
        let dimensions = 16
        let numClusters = 4
        let config = NeuralClusteringConfiguration(
            dimensions: dimensions,
            numberOfClusters: numClusters,
            trainingEpochs: 5
        )
        
        let clustering = try await NeuralClustering(configuration: config)
        let testVectors = generateTestVectors(count: 50, dimensions: dimensions, clusters: numClusters)
        
        // Train clustering
        _ = try await clustering.train(vectors: testVectors)
        
        // Test probability distribution
        let testVector = testVectors[0]
        let probabilities = try await clustering.getClusterProbabilities(for: testVector)
        
        XCTAssertEqual(probabilities.count, numClusters)
        XCTAssertEqual(probabilities.reduce(0, +), 1.0, accuracy: 0.01) // Should sum to 1
        XCTAssertTrue(probabilities.allSatisfy { $0 >= 0 && $0 <= 1 }) // Valid probabilities
    }
    
    // MARK: - IVF Integration Tests
    
    func testIVFWithNeuralClustering() async throws {
        let dimensions = 32
        let neuralConfig = NeuralClusteringConfig(
            clusterHiddenSizes: [64, 32],
            epochs: 5,
            batchSize: 16,
            adaptiveProbing: true
        )
        
        let ivfConfig = IVFConfiguration(
            dimensions: dimensions,
            numberOfCentroids: 8,
            numberOfProbes: 3,
            trainingSampleSize: 100,
            neuralClusteringConfig: neuralConfig,
            clusteringMethod: .neural
        )
        
        let index = try await IVFIndex<SIMD32<Float>, TestMetadata>(configuration: ivfConfig)
        
        // Generate training data
        let trainingVectors = generateTestVectors(count: 100, dimensions: dimensions, clusters: 8)
        
        // Train index
        try await index.train(on: trainingVectors)
        
        // Insert test vectors
        for (i, vector) in trainingVectors.prefix(50).enumerated() {
            let simdVector = SIMD32<Float>(vector)
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: simdVector,
                metadata: TestMetadata(label: "test_\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        // Search test
        let queryVector = SIMD32<Float>(trainingVectors[0])
        let results = try await index.search(query: queryVector, k: 5)
        
        XCTAssertEqual(results.count, 5)
        XCTAssertEqual(results[0].id, "vec_0") // Should find itself
    }
    
    func testHybridClustering() async throws {
        let dimensions = 64
        let neuralConfig = NeuralClusteringConfig.fast
        
        let ivfConfig = IVFConfiguration(
            dimensions: dimensions,
            numberOfCentroids: 10,
            neuralClusteringConfig: neuralConfig,
            clusteringMethod: .hybrid // K-means + neural refinement
        )
        
        let index = try await IVFIndex<SIMD64<Float>, TestMetadata>(configuration: ivfConfig)
        
        // Generate training data with clear clusters
        let trainingVectors = generateTestVectors(count: 200, dimensions: dimensions, clusters: 10)
        
        // Train index
        try await index.train(on: trainingVectors)
        
        // Verify training succeeded
        let stats = await index.statistics()
        XCTAssertTrue(stats.trained)
        XCTAssertEqual(stats.numberOfCentroids, 10)
    }
    
    func testOnlineAdaptation() async throws {
        let dimensions = 32
        let neuralConfig = NeuralClusteringConfig(
            clusterHiddenSizes: [64],
            epochs: 5,
            adaptiveProbing: true,
            onlineAdaptation: true,
            queryHistorySize: 100,
            adaptationInterval: 10
        )
        
        let ivfConfig = IVFConfiguration(
            dimensions: dimensions,
            numberOfCentroids: 5,
            neuralClusteringConfig: neuralConfig,
            clusteringMethod: .neural
        )
        
        let index = try await IVFIndex<SIMD32<Float>, TestMetadata>(configuration: ivfConfig)
        
        // Train and populate index
        let vectors = generateTestVectors(count: 100, dimensions: dimensions, clusters: 5)
        try await index.train(on: vectors)
        
        for (i, vector) in vectors.prefix(50).enumerated() {
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: SIMD32<Float>(vector),
                metadata: TestMetadata(label: "test_\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        // Perform multiple searches to trigger adaptation
        for i in 0..<15 {
            let queryIdx = i % vectors.count
            let query = SIMD32<Float>(vectors[queryIdx])
            _ = try await index.search(query: query, k: 3)
        }
        
        // The neural clustering should have adapted to query patterns
        // (In a real scenario, we would verify improved performance)
    }
    
    // MARK: - Performance Tests
    
    func testNeuralClusteringPerformance() async throws {
        // Skip in CI environments
        try XCTSkipIf(ProcessInfo.processInfo.environment["CI"] != nil)
        
        let dimensions = 128
        let numClusters = 20
        let config = NeuralClusteringConfiguration(
            dimensions: dimensions,
            numberOfClusters: numClusters,
            trainingEpochs: 10
        )
        
        let clustering = try await NeuralClustering(configuration: config)
        let testVectors = generateTestVectors(count: 1000, dimensions: dimensions, clusters: numClusters)
        
        let startTime = Date()
        _ = try await clustering.train(vectors: testVectors)
        let trainingTime = Date().timeIntervalSince(startTime)
        
        print("Neural clustering training time: \(trainingTime)s for \(testVectors.count) vectors")
        
        // Measure inference time
        let inferenceStart = Date()
        for _ in 0..<100 {
            _ = try await clustering.findNearestCentroids(for: testVectors[0], count: 5)
        }
        let inferenceTime = Date().timeIntervalSince(inferenceStart) / 100
        
        print("Average inference time: \(inferenceTime * 1000)ms")
        
        // Performance assertions
        XCTAssertLessThan(trainingTime, 30.0) // Should train in under 30 seconds
        XCTAssertLessThan(inferenceTime, 0.01) // Should infer in under 10ms
    }
}

// MARK: - Test Helpers

private struct TestMetadata: Codable, Sendable {
    let label: String
}
// VectorStoreKit: Tests for New Indexes
//
// Comprehensive tests for IVF, Learned, and Hybrid indexes

import XCTest
@testable import VectorStoreKit

final class NewIndexTests: XCTestCase {
    
    // MARK: - K-means Clustering Tests
    
    func testKMeansClustering() async throws {
        let clustering = try await KMeansClustering()
        
        // Generate test data
        let vectors = generateTestVectors(count: 100, dimensions: 8)
        
        // Test clustering
        let result = try await clustering.cluster(vectors: vectors, k: 5)
        
        XCTAssertEqual(result.centroids.count, 5)
        XCTAssertEqual(result.assignments.count, 100)
        XCTAssertTrue(result.converged || result.iterations > 0)
        XCTAssertGreaterThan(result.inertia, 0)
        
        // Verify all points are assigned
        for assignment in result.assignments {
            XCTAssertGreaterThanOrEqual(assignment, 0)
            XCTAssertLessThan(assignment, 5)
        }
    }
    
    func testKMeansWithDifferentInitMethods() async throws {
        let vectors = generateTestVectors(count: 50, dimensions: 4)
        
        // Test random initialization
        let randomConfig = KMeansConfiguration(initMethod: .random, seed: 42)
        let randomClustering = try await KMeansClustering(configuration: randomConfig)
        let randomResult = try await randomClustering.cluster(vectors: vectors, k: 3)
        
        XCTAssertEqual(randomResult.centroids.count, 3)
        
        // Test k-means++ initialization
        let plusPlusConfig = KMeansConfiguration(initMethod: .kMeansPlusPlus, seed: 42)
        let plusPlusClustering = try await KMeansClustering(configuration: plusPlusConfig)
        let plusPlusResult = try await plusPlusClustering.cluster(vectors: vectors, k: 3)
        
        XCTAssertEqual(plusPlusResult.centroids.count, 3)
        
        // Test custom initialization
        let customCentroids = [vectors[0], vectors[10], vectors[20]]
        let customConfig = KMeansConfiguration(initMethod: .custom(customCentroids))
        let customClustering = try await KMeansClustering(configuration: customConfig)
        let customResult = try await customClustering.cluster(vectors: vectors, k: 3)
        
        XCTAssertEqual(customResult.centroids.count, 3)
    }
    
    func testMiniBatchKMeans() async throws {
        let clustering = try await KMeansClustering()
        let vectors = generateTestVectors(count: 1000, dimensions: 16)
        
        let result = try await clustering.miniBatchCluster(
            vectors: vectors,
            k: 10,
            batchSize: 100
        )
        
        XCTAssertEqual(result.centroids.count, 10)
        XCTAssertEqual(result.assignments.count, 1000)
        XCTAssertTrue(result.converged)
        XCTAssertGreaterThan(result.iterations, 0)
    }
    
    // MARK: - IVF Index Tests
    
    func testIVFIndexBasicOperations() async throws {
        let config = IVFConfiguration(
            dimensions: 8,
            numberOfCentroids: 4,
            numberOfProbes: 2
        )
        
        let index = try await IVFIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Generate training data
        let trainingData = generateTestVectors(count: 100, dimensions: 8)
        
        // Train the index
        try await index.train(on: trainingData)
        
        // Test insert
        let vector = SIMD8<Float>(1, 2, 3, 4, 5, 6, 7, 8)
        let metadata = TestMetadata(name: "test", value: 42)
        let entry = VectorEntry(id: "test1", vector: vector, metadata: metadata)
        
        let insertResult = try await index.insert(entry)
        XCTAssertTrue(insertResult.success)
        XCTAssertEqual(await index.count, 1)
        
        // Test search
        let results = try await index.search(query: vector, k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, "test1")
        XCTAssertEqual(results[0].distance, 0, accuracy: 0.01)
        
        // Test contains
        XCTAssertTrue(await index.contains(id: "test1"))
        XCTAssertFalse(await index.contains(id: "nonexistent"))
        
        // Test delete
        let deleted = try await index.delete(id: "test1")
        XCTAssertTrue(deleted)
        XCTAssertEqual(await index.count, 0)
    }
    
    func testIVFIndexMultiProbeSearch() async throws {
        let config = IVFConfiguration(
            dimensions: 8,
            numberOfCentroids: 10,
            numberOfProbes: 5
        )
        
        let index = try await IVFIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Train and populate index
        let vectors = generateTestVectors(count: 200, dimensions: 8)
        try await index.train(on: vectors)
        
        for (i, vec) in vectors.enumerated() {
            let simdVec = vectorToSIMD8(vec)
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: simdVec,
                metadata: TestMetadata(name: "Vector \(i)", value: i)
            )
            _ = try await index.insert(entry)
        }
        
        // Search with different strategies
        let query = vectorToSIMD8(vectors[50])
        
        let exactResults = try await index.search(
            query: query,
            k: 10,
            strategy: .exact
        )
        
        let approxResults = try await index.search(
            query: query,
            k: 10,
            strategy: .approximate(quality: 0.8)
        )
        
        XCTAssertGreaterThan(exactResults.count, 0)
        XCTAssertGreaterThan(approxResults.count, 0)
        XCTAssertEqual(exactResults[0].id, "vec_50") // Should find itself
    }
    
    func testIVFIndexStatistics() async throws {
        let config = IVFConfiguration(
            dimensions: 8,
            numberOfCentroids: 5
        )
        
        let index = try await IVFIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Train index
        let vectors = generateTestVectors(count: 100, dimensions: 8)
        try await index.train(on: vectors)
        
        // Add vectors
        for i in 0..<50 {
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: vectorToSIMD8(vectors[i]),
                metadata: TestMetadata(name: "test", value: i)
            )
            _ = try await index.insert(entry)
        }
        
        // Get statistics
        let stats = await index.statistics()
        
        XCTAssertEqual(stats.vectorCount, 50)
        XCTAssertEqual(stats.numberOfCentroids, 5)
        XCTAssertTrue(stats.trained)
        XCTAssertGreaterThan(stats.memoryUsage, 0)
    }
    
    // MARK: - Learned Index Tests
    
    func testLearnedIndexBasicOperations() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 8,
            modelArchitecture: .linear,
            bucketSize: 10
        )
        
        let index = try await LearnedIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Test insert without training (should use hash-based positioning)
        let vector = SIMD8<Float>(1, 2, 3, 4, 5, 6, 7, 8)
        let metadata = TestMetadata(name: "test", value: 42)
        let entry = VectorEntry(id: "test1", vector: vector, metadata: metadata)
        
        let insertResult = try await index.insert(entry)
        XCTAssertTrue(insertResult.success)
        XCTAssertEqual(await index.count, 1)
        
        // Test search
        let results = try await index.search(query: vector, k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, "test1")
        
        // Test delete
        let deleted = try await index.delete(id: "test1")
        XCTAssertTrue(deleted)
        XCTAssertEqual(await index.count, 0)
    }
    
    func testLearnedIndexTraining() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 8,
            modelArchitecture: .mlp(hiddenSizes: [16, 8]),
            trainingConfig: LearnedIndexConfiguration.TrainingConfiguration(
                epochs: 10,
                batchSize: 16
            ),
            bucketSize: 50
        )
        
        let index = try await LearnedIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Generate training data
        let trainingVectors = generateTestVectors(count: 1000, dimensions: 8)
        
        // Train the model
        try await index.train(on: trainingVectors)
        
        // Insert vectors
        for (i, vec) in trainingVectors.prefix(100).enumerated() {
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: vectorToSIMD8(vec),
                metadata: TestMetadata(name: "Vector \(i)", value: i)
            )
            _ = try await index.insert(entry)
        }
        
        // Verify trained index performs searches
        let query = vectorToSIMD8(trainingVectors[10])
        let results = try await index.search(query: query, k: 5)
        
        XCTAssertGreaterThan(results.count, 0)
        XCTAssertTrue(results.contains { $0.id == "vec_10" })
    }
    
    func testLearnedIndexStatistics() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 8,
            bucketSize: 20
        )
        
        let index = try await LearnedIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Add vectors
        for i in 0..<100 {
            let vec = generateRandomSIMD8()
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: vec,
                metadata: TestMetadata(name: "test", value: i)
            )
            _ = try await index.insert(entry)
        }
        
        let stats = await index.statistics()
        
        XCTAssertEqual(stats.vectorCount, 100)
        XCTAssertGreaterThan(stats.bucketCount, 0)
        XCTAssertGreaterThan(stats.memoryUsage, 0)
    }
    
    // MARK: - Hybrid Index Tests
    
    func testHybridIndexBasicOperations() async throws {
        let config = HybridIndexConfiguration(
            dimensions: 8,
            routingStrategy: .fixed(ivfWeight: 0.5)
        )
        
        let index = try await HybridIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Train the index
        let trainingData = generateTestVectors(count: 100, dimensions: 8)
        try await index.train(on: trainingData)
        
        // Test insert
        let vector = SIMD8<Float>(1, 2, 3, 4, 5, 6, 7, 8)
        let metadata = TestMetadata(name: "test", value: 42)
        let entry = VectorEntry(id: "test1", vector: vector, metadata: metadata)
        
        let insertResult = try await index.insert(entry)
        XCTAssertTrue(insertResult.success)
        XCTAssertEqual(await index.count, 1)
        
        // Test search
        let results = try await index.search(query: vector, k: 1)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, "test1")
        
        // Test delete
        let deleted = try await index.delete(id: "test1")
        XCTAssertTrue(deleted)
        XCTAssertEqual(await index.count, 0)
    }
    
    func testHybridIndexRoutingStrategies() async throws {
        // Test different routing strategies
        let strategies: [HybridIndexConfiguration.RoutingStrategy] = [
            .fixed(ivfWeight: 1.0),    // All IVF
            .fixed(ivfWeight: 0.0),    // All learned
            .fixed(ivfWeight: 0.5),    // 50/50
            .adaptive,                  // Adaptive
            .ensemble,                  // Both
            .hierarchical              // Hierarchical
        ]
        
        for strategy in strategies {
            let config = HybridIndexConfiguration(
                dimensions: 8,
                routingStrategy: strategy
            )
            
            let index = try await HybridIndex<SIMD8<Float>, TestMetadata>(configuration: config)
            
            // Train and add data
            let vectors = generateTestVectors(count: 100, dimensions: 8)
            try await index.train(on: vectors)
            
            for i in 0..<10 {
                let entry = VectorEntry(
                    id: "vec_\(i)",
                    vector: vectorToSIMD8(vectors[i]),
                    metadata: TestMetadata(name: "test", value: i)
                )
                _ = try await index.insert(entry)
            }
            
            // Search
            let query = vectorToSIMD8(vectors[5])
            let results = try await index.search(query: query, k: 3)
            
            XCTAssertGreaterThan(results.count, 0)
            XCTAssertTrue(results.contains { $0.id == "vec_5" })
        }
    }
    
    func testHybridIndexStatistics() async throws {
        let config = HybridIndexConfiguration(
            dimensions: 8,
            routingStrategy: .adaptive
        )
        
        let index = try await HybridIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Train and populate
        let vectors = generateTestVectors(count: 100, dimensions: 8)
        try await index.train(on: vectors)
        
        for i in 0..<50 {
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: vectorToSIMD8(vectors[i]),
                metadata: TestMetadata(name: "test", value: i)
            )
            _ = try await index.insert(entry)
        }
        
        // Perform some searches to generate routing statistics
        for i in 0..<10 {
            let query = vectorToSIMD8(vectors[i * 5])
            _ = try await index.search(query: query, k: 5)
        }
        
        let stats = await index.statistics()
        XCTAssertEqual(stats.vectorCount, 50)
        XCTAssertGreaterThan(stats.memoryUsage, 0)
        
        let routingStats = await index.getRoutingStatistics()
        XCTAssertEqual(routingStats.totalQueries, 10)
        XCTAssertGreaterThanOrEqual(routingStats.ivfRatio + routingStats.learnedRatio + routingStats.hybridRatio, 0.99)
    }
    
    // MARK: - Integration Tests
    
    func testIndexComparison() async throws {
        let dimensions = 8
        let vectorCount = 100
        let vectors = generateTestVectors(count: vectorCount, dimensions: dimensions)
        
        // Create all index types
        let hnswConfig = HNSWConfiguration(dimensions: dimensions, m: 8, efConstruction: 50)
        let hnswIndex = try await HNSWIndex<SIMD8<Float>, TestMetadata>(configuration: hnswConfig)
        
        let ivfConfig = IVFConfiguration(dimensions: dimensions, numberOfCentroids: 10)
        let ivfIndex = try await IVFIndex<SIMD8<Float>, TestMetadata>(configuration: ivfConfig)
        try await ivfIndex.train(on: vectors)
        
        let learnedConfig = LearnedIndexConfiguration(dimensions: dimensions)
        let learnedIndex = try await LearnedIndex<SIMD8<Float>, TestMetadata>(configuration: learnedConfig)
        
        let hybridConfig = HybridIndexConfiguration(dimensions: dimensions)
        let hybridIndex = try await HybridIndex<SIMD8<Float>, TestMetadata>(configuration: hybridConfig)
        try await hybridIndex.train(on: vectors)
        
        // Insert same data into all indexes
        for (i, vec) in vectors.prefix(50).enumerated() {
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: vectorToSIMD8(vec),
                metadata: TestMetadata(name: "test", value: i)
            )
            
            _ = try await hnswIndex.insert(entry)
            _ = try await ivfIndex.insert(entry)
            _ = try await learnedIndex.insert(entry)
            _ = try await hybridIndex.insert(entry)
        }
        
        // Compare search results
        let query = vectorToSIMD8(vectors[25])
        let k = 5
        
        let hnswResults = try await hnswIndex.search(query: query, k: k)
        let ivfResults = try await ivfIndex.search(query: query, k: k)
        let learnedResults = try await learnedIndex.search(query: query, k: k)
        let hybridResults = try await hybridIndex.search(query: query, k: k)
        
        // All should find the query vector itself as the first result
        XCTAssertEqual(hnswResults[0].id, "vec_25")
        XCTAssertEqual(ivfResults[0].id, "vec_25")
        XCTAssertEqual(learnedResults[0].id, "vec_25")
        XCTAssertEqual(hybridResults[0].id, "vec_25")
        
        // All should return k results
        XCTAssertEqual(hnswResults.count, k)
        XCTAssertEqual(ivfResults.count, k)
        XCTAssertEqual(learnedResults.count, k)
        XCTAssertEqual(hybridResults.count, k)
    }
    
    // MARK: - Helper Functions
    
    private func generateTestVectors(count: Int, dimensions: Int) -> [[Float]] {
        var vectors: [[Float]] = []
        for _ in 0..<count {
            var vector: [Float] = []
            for _ in 0..<dimensions {
                vector.append(Float.random(in: -1...1))
            }
            vectors.append(vector)
        }
        return vectors
    }
    
    private func vectorToSIMD8(_ vector: [Float]) -> SIMD8<Float> {
        var simd = SIMD8<Float>()
        for i in 0..<min(8, vector.count) {
            simd[i] = vector[i]
        }
        return simd
    }
    
    private func generateRandomSIMD8() -> SIMD8<Float> {
        var simd = SIMD8<Float>()
        for i in 0..<8 {
            simd[i] = Float.random(in: -1...1)
        }
        return simd
    }
}

// Test metadata
private struct TestMetadata: Codable, Sendable {
    let name: String
    let value: Int
}
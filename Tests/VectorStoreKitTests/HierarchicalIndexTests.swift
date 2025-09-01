// HierarchicalIndexTests.swift
// VectorStoreKitTests
//
// Tests for two-level hierarchical index

import XCTest
@testable import VectorStoreKit
import simd

final class HierarchicalIndexTests: XCTestCase {
    
    // MARK: - Test Helpers
    
    private func generateTestVectors(count: Int, dimension: Int = 512) -> [(Vector512, String, Data?)] {
        (0..<count).map { i in
            let vector = Vector512(repeating: Float(i) / Float(count))
            let id = "vec_\(i)"
            let metadata = "metadata_\(i)".data(using: .utf8)
            return (vector, id, metadata)
        }
    }
    
    private func generateClusteredVectors(
        clusters: Int,
        vectorsPerCluster: Int,
        dimension: Int = 512
    ) -> [(Vector512, String, Data?)] {
        var vectors: [(Vector512, String, Data?)] = []
        
        for cluster in 0..<clusters {
            let clusterCenter = Float(cluster) * 10.0
            
            for i in 0..<vectorsPerCluster {
                var values = [Float](repeating: 0, count: dimension)
                
                // Create vector near cluster center
                for j in 0..<dimension {
                    values[j] = clusterCenter + Float.random(in: -1...1)
                }
                
                let vector = Vector512(values)
                let id = "cluster_\(cluster)_vec_\(i)"
                let metadata = "cluster: \(cluster)".data(using: .utf8)
                
                vectors.append((vector, id, metadata))
            }
        }
        
        return vectors
    }
    
    // MARK: - Configuration Tests
    
    func testConfigurationForDatasetSize() {
        let small = HierarchicalConfiguration.forDatasetSize(10_000)
        XCTAssertEqual(small.topLevelClusters, 100)
        
        let medium = HierarchicalConfiguration.forDatasetSize(250_000)
        XCTAssertEqual(medium.topLevelClusters, 512)
        
        let large = HierarchicalConfiguration.forDatasetSize(750_000)
        XCTAssertEqual(large.topLevelClusters, 1024)
        
        let huge = HierarchicalConfiguration.forDatasetSize(2_000_000)
        XCTAssertEqual(huge.topLevelClusters, 2048)
    }
    
    // MARK: - Initialization Tests
    
    func testHierarchicalIndexInitialization() async throws {
        let config = HierarchicalConfiguration(
            topLevelClusters: 10,
            leafIndexSize: 100,
            probesPerQuery: 3
        )
        
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: config
        )
        
        let stats = await index.getStatistics()
        XCTAssertEqual(stats.totalVectors, 0)
        XCTAssertEqual(stats.leafIndices.count, 0)
    }
    
    // MARK: - Insertion Tests
    
    func testSingleVectorInsertion() async throws {
        let config = HierarchicalConfiguration(
            topLevelClusters: 10,
            leafIndexSize: 100
        )
        
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: config
        )
        
        let vector = Vector512(repeating: 1.0)
        try await index.insert(vector, id: "test1", metadata: nil)
        
        let stats = await index.getStatistics()
        XCTAssertEqual(stats.totalVectors, 1)
    }
    
    func testBatchInsertion() async throws {
        let config = HierarchicalConfiguration(
            topLevelClusters: 10,
            leafIndexSize: 100
        )
        
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: config
        )
        
        let vectors = generateTestVectors(count: 100)
        try await index.insertBatch(vectors)
        
        let stats = await index.getStatistics()
        XCTAssertEqual(stats.totalVectors, 100)
        XCTAssertGreaterThan(stats.leafIndices.count, 0)
    }
    
    func testLargeScaleInsertion() async throws {
        let config = HierarchicalConfiguration(
            topLevelClusters: 50,
            leafIndexSize: 100,
            probesPerQuery: 5
        )
        
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: config
        )
        
        // Insert in batches to test incremental building
        for batch in 0..<10 {
            let vectors = generateTestVectors(
                count: 500,
                dimension: 512
            ).map { (vector, id, metadata) in
                (vector, "batch_\(batch)_\(id)", metadata)
            }
            
            try await index.insertBatch(vectors)
        }
        
        let stats = await index.getStatistics()
        XCTAssertEqual(stats.totalVectors, 5000)
        XCTAssertGreaterThan(stats.leafIndices.count, 10)
    }
    
    // MARK: - Search Tests
    
    func testBasicSearch() async throws {
        let config = HierarchicalConfiguration(
            topLevelClusters: 10,
            leafIndexSize: 100
        )
        
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: config
        )
        
        // Insert clustered vectors
        let vectors = generateClusteredVectors(
            clusters: 5,
            vectorsPerCluster: 50
        )
        try await index.insertBatch(vectors)
        
        // Search for vector from cluster 0
        let query = vectors[0].0
        let results = try await index.search(query: query, k: 10)
        
        XCTAssertEqual(results.count, 10)
        XCTAssertEqual(results[0].id, vectors[0].1) // Should find itself
        
        // Most results should be from the same cluster
        let cluster0Results = results.filter { $0.id.hasPrefix("cluster_0") }
        XCTAssertGreaterThan(cluster0Results.count, 5)
    }
    
    func testSearchWithFilter() async throws {
        let config = HierarchicalConfiguration(
            topLevelClusters: 10,
            leafIndexSize: 100
        )
        
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: config
        )
        
        let vectors = generateClusteredVectors(
            clusters: 5,
            vectorsPerCluster: 20
        )
        try await index.insertBatch(vectors)
        
        // Search with filter for cluster 2
        let query = vectors[50].0 // Vector from cluster 2
        let results = try await index.search(
            query: query,
            k: 10,
            filter: { metadata in
                guard let data = metadata,
                      let str = String(data: data, encoding: .utf8) else {
                    return false
                }
                return str.contains("cluster: 2")
            }
        )
        
        // All results should be from cluster 2
        for result in results {
            XCTAssertTrue(result.id.hasPrefix("cluster_2"))
        }
    }
    
    func testBatchSearch() async throws {
        let config = HierarchicalConfiguration(
            topLevelClusters: 10,
            leafIndexSize: 100
        )
        
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: config
        )
        
        let vectors = generateTestVectors(count: 500)
        try await index.insertBatch(vectors)
        
        // Search for multiple queries
        let queries = vectors.prefix(5).map { $0.0 }
        let batchResults = try await index.batchSearch(
            queries: Array(queries),
            k: 10
        )
        
        XCTAssertEqual(batchResults.count, 5)
        
        for (i, results) in batchResults.enumerated() {
            XCTAssertEqual(results.count, 10)
            XCTAssertEqual(results[0].id, vectors[i].1) // Should find itself
        }
    }
    
    // MARK: - Dynamic Probing Tests
    
    func testDynamicProbing() async throws {
        let config = HierarchicalConfiguration(
            topLevelClusters: 20,
            leafIndexSize: 100,
            probesPerQuery: 3,
            enableDynamicProbing: true
        )
        
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: config
        )
        
        // Create imbalanced dataset
        var vectors: [(Vector512, String, Data?)] = []
        
        // Large cluster
        vectors.append(contentsOf: generateClusteredVectors(
            clusters: 1,
            vectorsPerCluster: 800
        ))
        
        // Small clusters
        for i in 1..<10 {
            let smallCluster = generateClusteredVectors(
                clusters: 1,
                vectorsPerCluster: 20
            ).map { (vector, id, metadata) in
                (vector, "small_\(i)_\(id)", metadata)
            }
            vectors.append(contentsOf: smallCluster)
        }
        
        try await index.insertBatch(vectors)
        
        // Search should probe more clusters due to imbalance
        let query = vectors[0].0
        let results = try await index.search(query: query, k: 50)
        
        XCTAssertEqual(results.count, 50)
    }
    
    // MARK: - Rebalancing Tests
    
    func testRebalancing() async throws {
        let config = HierarchicalConfiguration(
            topLevelClusters: 10,
            leafIndexSize: 50,
            rebalanceThreshold: 0.5
        )
        
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: config
        )
        
        // Insert imbalanced data
        var vectors: [(Vector512, String, Data?)] = []
        
        // One huge cluster
        for i in 0..<400 {
            let vector = Vector512(repeating: 1.0 + Float(i) * 0.001)
            vectors.append((vector, "huge_\(i)", nil))
        }
        
        // Several tiny clusters
        for cluster in 1..<10 {
            for i in 0..<5 {
                let vector = Vector512(repeating: Float(cluster) * 10.0)
                vectors.append((vector, "tiny_\(cluster)_\(i)", nil))
            }
        }
        
        try await index.insertBatch(vectors)
        
        let statsBefore = await index.getStatistics()
        let imbalanceBefore = statsBefore.clusterImbalance
        
        // Trigger rebalance
        try await index.rebalance()
        
        let statsAfter = await index.getStatistics()
        let imbalanceAfter = statsAfter.clusterImbalance
        
        // Imbalance should improve
        XCTAssertLessThan(imbalanceAfter, imbalanceBefore)
    }
    
    // MARK: - Performance Tests
    
    func testSearchPerformance() async throws {
        let config = HierarchicalConfiguration(
            topLevelClusters: 100,
            leafIndexSize: 1000,
            probesPerQuery: 10
        )
        
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: config
        )
        
        // Insert 10K vectors
        let vectors = generateTestVectors(count: 10_000)
        try await index.insertBatch(vectors)
        
        // Measure search performance
        let query = Vector512(repeating: 0.5)
        
        let start = Date()
        let _ = try await index.search(query: query, k: 100)
        let searchTime = Date().timeIntervalSince(start)
        
        // Should complete in reasonable time
        XCTAssertLessThan(searchTime, 0.1) // 100ms
    }
    
    func testRecallAccuracy() async throws {
        let config = HierarchicalConfiguration(
            topLevelClusters: 20,
            leafIndexSize: 100,
            probesPerQuery: 5
        )
        
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: config
        )
        
        // Insert well-separated clusters
        let vectors = generateClusteredVectors(
            clusters: 10,
            vectorsPerCluster: 100
        )
        try await index.insertBatch(vectors)
        
        // Test recall for vectors from same cluster
        var correctResults = 0
        let testCount = 20
        
        for i in 0..<testCount {
            let queryIdx = i * 10 // Sample from different clusters
            let query = vectors[queryIdx].0
            let expectedCluster = queryIdx / 100
            
            let results = try await index.search(query: query, k: 20)
            
            // Count how many results are from the correct cluster
            let correctClusterResults = results.filter { result in
                result.id.hasPrefix("cluster_\(expectedCluster)")
            }
            
            if correctClusterResults.count >= 15 { // 75% recall
                correctResults += 1
            }
        }
        
        let recall = Float(correctResults) / Float(testCount)
        XCTAssertGreaterThan(recall, 0.8) // 80% recall
    }
}
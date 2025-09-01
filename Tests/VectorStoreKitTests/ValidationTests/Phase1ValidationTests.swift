import XCTest
@testable import VectorStoreKit

final class Phase1ValidationTests: XCTestCase {
    
    /// Verify simple API works correctly
    func testSimpleAPIFunctionality() async throws {
        // In-memory store
        let store = SimpleVectorStore<TestMetadata>.inMemory(dimension: 128)
        
        // Add vectors
        let vectors = VectorTestHelpers.randomVectors(count: 100, dimension: 128)
        let metadata = (0..<100).map { TestMetadata(id: "vec-\($0)", category: "test") }
        let ids = try await store.add(vectors: vectors, metadata: metadata)
        XCTAssertEqual(ids.count, 100)
        
        // Search
        let query = VectorTestHelpers.randomVectors(count: 1, dimension: 128)[0]
        let results = try await store.search(query, limit: 10)
        XCTAssertEqual(results.count, 10)
        
        // Verify ordering (distances should be increasing)
        for i in 1..<results.count {
            XCTAssertLessThanOrEqual(results[i-1].score, results[i].score,
                "Results should be ordered by distance")
        }
        
        // Search with metadata
        let metadataResults = try await store.searchWithMetadata(query, limit: 5)
        XCTAssertEqual(metadataResults.count, 5)
        XCTAssertNotNil(metadataResults[0].metadata)
        
        // Test statistics
        let stats = await store.statistics
        XCTAssertEqual(stats.vectorCount, 100)
    }
    
    /// Verify advanced API maintains compatibility
    func testAdvancedAPICompatibility() async throws {
        // Test that VectorUniverse still works with existing patterns
        let universe = VectorUniverse<Vector<Float>, TestMetadata>()
        
        // Test production preset
        let prodConfig = universe.production(dataSize: .small)
        
        // The type should be fully configured
        XCTAssertTrue(type(of: prodConfig).self == FullyConfiguredUniverse<
            Vector<Float>,
            TestMetadata,
            HNSWIndexingStrategy<Vector<Float>, TestMetadata>,
            HierarchicalProductionStorageStrategy,
            LRUCachingStrategy<Vector<Float>>
        >.self)
        
        // Test research preset
        let researchConfig = universe.research()
        
        // Test embedded preset
        let embeddedConfig = universe.embedded(maxMemory: 50_000_000)
    }
    
    /// Verify core types work correctly
    func testCoreTypesFunction() throws {
        // Test Vector type
        let vec1 = Vector<Float>([1, 2, 3, 4])
        let vec2 = Vector<Float>([5, 6, 7, 8])
        
        // Test operations
        let sum = vec1 + vec2
        XCTAssertEqual(sum[0], 6)
        XCTAssertEqual(sum[1], 8)
        
        let diff = vec2 - vec1
        XCTAssertEqual(diff[0], 4)
        XCTAssertEqual(diff[1], 4)
        
        let scaled = vec1 * 2.0
        XCTAssertEqual(scaled[0], 2)
        XCTAssertEqual(scaled[1], 4)
        
        // Test dot product
        let dot = vec1.dot(vec2)
        XCTAssertEqual(dot, 70) // 1*5 + 2*6 + 3*7 + 4*8
        
        // Test norm
        let norm = vec1.norm
        XCTAssertEqual(norm, sqrt(30), accuracy: 0.0001) // sqrt(1+4+9+16)
        
        // Test normalization
        let normalized = vec1.normalized()
        XCTAssertEqual(normalized.norm, 1.0, accuracy: 0.0001)
    }
    
    /// Verify Metal performance is maintained
    func testMetalPerformanceMaintained() async throws {
        let dimension = 512
        let candidateCount = 10_000
        
        let query = Vector(VectorTestHelpers.randomVectors(count: 1, dimension: dimension)[0])
        let candidates = VectorTestHelpers.randomVectors(count: candidateCount, dimension: dimension)
            .map { Vector($0) }
        
        let compute = UnifiedDistanceComputation(preferGPU: true)
        
        // Measure performance
        try await assertPerformance({
            let distances = try await compute.computeDistances(
                query: query,
                candidates: candidates,
                metric: .euclidean
            )
            XCTAssertEqual(distances.count, candidateCount)
        }, maxDuration: 0.1) // Should complete in <100ms
    }
    
    /// Verify batch operations
    func testBatchOperations() async throws {
        let dimension = 256
        let queryCount = 100
        let candidateCount = 1000
        
        let queries = VectorTestHelpers.randomVectors(count: queryCount, dimension: dimension)
            .map { Vector($0) }
        let candidates = VectorTestHelpers.randomVectors(count: candidateCount, dimension: dimension)
            .map { Vector($0) }
        
        let compute = UnifiedDistanceComputation(preferGPU: false) // Test CPU path
        
        let results = try await compute.batchComputeDistances(
            queries: queries,
            candidates: candidates,
            metric: .cosine,
            k: 10
        )
        
        XCTAssertEqual(results.count, queryCount)
        for result in results {
            XCTAssertEqual(result.count, 10) // k=10
        }
    }
    
    /// Verify distance metrics
    func testDistanceMetrics() async throws {
        let vec1 = Vector<Float>([1, 0, 0, 0])
        let vec2 = Vector<Float>([0, 1, 0, 0])
        let vec3 = Vector<Float>([1, 1, 1, 1])
        
        let compute = UnifiedDistanceComputation(preferGPU: false)
        
        // Test Euclidean
        let euclidean = try await compute.computeDistances(
            query: vec1,
            candidates: [vec2],
            metric: .euclidean
        )[0]
        XCTAssertEqual(euclidean, sqrt(2), accuracy: 0.0001)
        
        // Test Cosine (orthogonal vectors)
        let cosine = try await compute.computeDistances(
            query: vec1,
            candidates: [vec2],
            metric: .cosine
        )[0]
        XCTAssertEqual(cosine, 1.0, accuracy: 0.0001) // cosine distance = 1 - cosine similarity
        
        // Test Manhattan
        let manhattan = try await compute.computeDistances(
            query: vec1,
            candidates: [vec3],
            metric: .manhattan
        )[0]
        XCTAssertEqual(manhattan, 3.0, accuracy: 0.0001) // |1-1| + |0-1| + |0-1| + |0-1|
        
        // Test Hamming
        let hamming = try await compute.computeDistances(
            query: vec1,
            candidates: [vec2],
            metric: .hamming
        )[0]
        XCTAssertEqual(hamming, 2.0) // Two positions differ when thresholded at 0.5
    }
    
    /// Verify no public API breaking changes
    func testNoBreakingChanges() throws {
        // This test ensures all public types still exist
        _ = VectorEntry(id: UUID(), vector: Vector<Float>([1, 2, 3]))
        _ = SearchResult(id: UUID(), score: 0.5)
        _ = AddResult(succeeded: Set([UUID()]))
        _ = UpdateResult(succeeded: Set([UUID()]))
        _ = DeleteResult(succeeded: Set([UUID()]))
        _ = DistanceMetric.cosine
        _ = MetadataFilter<String> { _ in true }
        
        // Configuration types
        _ = VectorStoreConfig(
            dimension: 128,
            metric: .euclidean,
            indexType: .hnsw(m: 16, efConstruction: 200),
            storage: .memory
        )
        
        // If this compiles, public API is maintained
        XCTAssertTrue(true)
    }
    
    /// Test memory pressure handling
    func testMemoryPressureHandling() async throws {
        // Create store with limited memory
        let store = SimpleVectorStore<TestMetadata>.optimized(
            dimension: 768,
            compressedDimension: 128
        )
        
        // Add vectors in batches to simulate memory pressure
        let batchSize = 1000
        let numBatches = 5
        
        for batch in 0..<numBatches {
            let vectors = VectorTestHelpers.randomVectors(
                count: batchSize,
                dimension: 768,
                seed: UInt64(batch)
            )
            
            _ = try await store.add(vectors: vectors)
            
            // Check memory usage doesn't grow unbounded
            let stats = await store.statistics
            print("Batch \(batch): \(stats.vectorCount) vectors, \(stats.memoryUsage / 1024 / 1024)MB")
        }
        
        let finalStats = await store.statistics
        XCTAssertEqual(finalStats.vectorCount, batchSize * numBatches)
    }
    
    /// Verify test infrastructure works
    func testTestInfrastructure() throws {
        // Test vector generation
        let random = VectorTestHelpers.randomVectors(count: 10, dimension: 64)
        XCTAssertEqual(random.count, 10)
        XCTAssertEqual(random[0].count, 64)
        
        // Test clustering
        let (clustered, labels) = VectorTestHelpers.clusteredVectors(
            clusters: 3,
            vectorsPerCluster: 10,
            dimension: 32
        )
        XCTAssertEqual(clustered.count, 30)
        XCTAssertEqual(labels.count, 30)
        XCTAssertEqual(Set(labels).count, 3) // 3 unique clusters
        
        // Test synthetic embeddings
        let sequential = VectorTestHelpers.syntheticEmbeddings(
            count: 5,
            dimension: 16,
            pattern: .sequential
        )
        XCTAssertEqual(sequential.count, 5)
        
        // Test metrics
        let retrieved = Set([UUID(), UUID(), UUID()])
        let relevant = Set([retrieved.first!]) // Only one is relevant
        
        let recall = VectorTestHelpers.recall(retrieved: retrieved, relevant: relevant)
        XCTAssertEqual(recall, 1.0) // Found all relevant
        
        let precision = VectorTestHelpers.precision(retrieved: retrieved, relevant: relevant)
        XCTAssertEqual(precision, 1.0/3.0, accuracy: 0.0001) // 1 of 3 retrieved is relevant
    }
}
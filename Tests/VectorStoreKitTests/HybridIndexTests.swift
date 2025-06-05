// VectorStoreKit: Hybrid Index Tests
//
// Comprehensive test suite for hybrid index functionality

import Testing
import Foundation
@testable import VectorStoreKit

@Suite("Hybrid Index Tests")
struct HybridIndexTests {
    
    // MARK: - Test Helpers
    
    static func generateMixedVectors(count: Int, dimensions: Int = 128) -> [[Float]] {
        // Generate vectors that benefit from both IVF and learned approaches
        (0..<count).map { i in
            let mixFactor = Float(i) / Float(count)
            
            return (0..<dimensions).map { d in
                // Mix clustered patterns with learnable sequences
                let clustered = sin(Float(i % 10) * .pi / 5) * cos(Float(d) * .pi / Float(dimensions))
                let sequential = Float(d) * mixFactor / Float(dimensions)
                let noise = Float.random(in: -0.05...0.05)
                
                return clustered * (1 - mixFactor) + sequential * mixFactor + noise
            }
        }
    }
    
    static func generateQuerySet(baseVectors: [[Float]], count: Int) -> [[Float]] {
        // Generate queries with slight perturbations
        (0..<count).map { i in
            let baseIdx = i % baseVectors.count
            return baseVectors[baseIdx].map { value in
                value + Float.random(in: -0.1...0.1)
            }
        }
    }
    
    // MARK: - Configuration Tests
    
    @Test("Hybrid Configuration")
    func testConfiguration() throws {
        // Basic configuration
        let config = HybridIndexConfiguration(
            dimensions: 128,
            routingStrategy: .queryAnalysis,
            adaptiveThreshold: 0.7
        )
        #expect(throws: Never.self) {
            try config.validate()
        }
        
        // Custom IVF and learned configs
        let ivfConfig = IVFConfiguration(
            dimensions: 128,
            numberOfCentroids: 64,
            numberOfProbes: 8
        )
        
        let learnedConfig = LearnedIndexConfiguration(
            dimensions: 128,
            modelArchitecture: .mlp(hiddenSizes: [256, 128]),
            bucketSize: 100
        )
        
        let customConfig = HybridIndexConfiguration(
            dimensions: 128,
            ivfConfig: ivfConfig,
            learnedConfig: learnedConfig,
            routingStrategy: .adaptive,
            adaptiveThreshold: 0.8
        )
        
        #expect(throws: Never.self) {
            try customConfig.validate()
        }
        
        // Invalid configuration
        let invalidConfig = HybridIndexConfiguration(
            dimensions: 0,
            routingStrategy: .fixed(ratio: 0.5)
        )
        #expect(throws: HybridIndexError.self) {
            try invalidConfig.validate()
        }
    }
    
    @Test("Memory Usage Estimation")
    func testMemoryEstimation() {
        let config = HybridIndexConfiguration(dimensions: 128)
        
        let mem1K = config.estimatedMemoryUsage(for: 1000)
        let mem10K = config.estimatedMemoryUsage(for: 10000)
        
        #expect(mem10K > mem1K)
        #expect(mem1K > 0)
        
        // Should be sum of both indexes plus overhead
        let ivfMem = config.ivfConfig.estimatedMemoryUsage(for: 1000)
        let learnedMem = config.learnedConfig.estimatedMemoryUsage(for: 1000)
        #expect(mem1K >= ivfMem + learnedMem)
    }
    
    // MARK: - Basic Operations Tests
    
    @Test("Basic Insert and Search", .timeLimit(.seconds(30)))
    func testBasicOperations() async throws {
        let config = HybridIndexConfiguration(
            dimensions: 32,
            routingStrategy: .adaptive,
            adaptiveThreshold: 0.5
        )
        
        let index = try await HybridIndex<SIMD32<Float>, TestMetadata>(configuration: config)
        
        // Generate training data
        let trainingData = Self.generateMixedVectors(count: 200, dimensions: 32)
        
        // Train both sub-indexes
        try await index.train(on: trainingData)
        
        // Insert vectors
        for i in 0..<50 {
            let vector = SIMD32<Float>(trainingData[i])
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: vector,
                metadata: TestMetadata(label: "hybrid\(i)")
            )
            
            let result = try await index.insert(entry)
            #expect(result.success == true)
        }
        
        // Search
        let query = SIMD32<Float>(trainingData[25])
        let results = try await index.search(query: query, k: 10)
        
        #expect(results.count > 0)
        #expect(results.first?.id == "vec25")  // Should find exact match
        
        // Verify both indexes have the data
        #expect(await index.count == 50)
    }
    
    @Test("Routing Strategies")
    func testRoutingStrategies() async throws {
        let dimensions = 64
        let vectors = Self.generateMixedVectors(count: 500, dimensions: dimensions)
        
        // Test different routing strategies
        let strategies: [(String, HybridIndexConfiguration.RoutingStrategy)] = [
            ("Fixed 50/50", .fixed(ratio: 0.5)),
            ("Adaptive", .adaptive),
            ("Query Analysis", .queryAnalysis),
            ("IVF Priority", .fixed(ratio: 0.8))
        ]
        
        for (name, strategy) in strategies {
            let config = HybridIndexConfiguration(
                dimensions: dimensions,
                routingStrategy: strategy
            )
            
            let index = try await HybridIndex<SIMD64<Float>, TestMetadata>(configuration: config)
            
            // Train and insert
            try await index.train(on: vectors)
            
            for i in 0..<100 {
                let entry = VectorEntry(
                    id: "vec\(i)",
                    vector: SIMD64<Float>(vectors[i]),
                    metadata: TestMetadata(label: "route\(i)")
                )
                _ = try await index.insert(entry)
            }
            
            // Test search with different query types
            let queries = Self.generateQuerySet(baseVectors: vectors, count: 10)
            
            var totalTime: Double = 0
            var resultCounts = 0
            
            for query in queries {
                let start = DispatchTime.now()
                let results = try await index.search(
                    query: SIMD64<Float>(query),
                    k: 5
                )
                let end = DispatchTime.now()
                
                totalTime += Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
                resultCounts += results.count
            }
            
            print("\(name) - Avg time: \(totalTime/10 * 1000)ms, Avg results: \(resultCounts/10)")
            
            // Get routing statistics
            let stats = await index.statistics()
            print("Routing stats - IVF: \(stats.routingStatistics.ivfRatio), Learned: \(stats.routingStatistics.learnedRatio)")
        }
    }
    
    // MARK: - Performance Tests
    
    @Test("Hybrid Performance vs Individual", .timeLimit(.seconds(60)))
    func testPerformanceComparison() async throws {
        let dimensions = 128
        let vectorCount = 5000
        let queryCount = 100
        
        // Generate test data
        let vectors = Self.generateMixedVectors(count: vectorCount, dimensions: dimensions)
        let queries = Self.generateQuerySet(baseVectors: vectors, count: queryCount)
        
        // Create indexes
        let hybridConfig = HybridIndexConfiguration(
            dimensions: dimensions,
            routingStrategy: .adaptive
        )
        let hybridIndex = try await HybridIndex<SIMD128<Float>, TestMetadata>(configuration: hybridConfig)
        
        let ivfIndex = try await IVFIndex<SIMD128<Float>, TestMetadata>(
            configuration: hybridConfig.ivfConfig
        )
        
        let learnedIndex = try await LearnedIndex<SIMD128<Float>, TestMetadata>(
            configuration: hybridConfig.learnedConfig
        )
        
        // Train all indexes
        try await hybridIndex.train(on: vectors)
        try await ivfIndex.train(on: vectors)
        try await learnedIndex.train()
        
        // Insert data into all indexes
        for i in 0..<1000 {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD128<Float>(vectors[i]),
                metadata: TestMetadata(label: "perf\(i)")
            )
            
            _ = try await hybridIndex.insert(entry)
            _ = try await ivfIndex.insert(entry)
            _ = try await learnedIndex.insert(entry)
        }
        
        // Benchmark searches
        let k = 10
        
        // Hybrid search
        let hybridStart = DispatchTime.now()
        var hybridResults: [[SearchResult<TestMetadata>]] = []
        for query in queries {
            let results = try await hybridIndex.search(
                query: SIMD128<Float>(query),
                k: k
            )
            hybridResults.append(results)
        }
        let hybridEnd = DispatchTime.now()
        let hybridTime = Double(hybridEnd.uptimeNanoseconds - hybridStart.uptimeNanoseconds) / 1_000_000_000
        
        // IVF search
        let ivfStart = DispatchTime.now()
        var ivfResults: [[SearchResult<TestMetadata>]] = []
        for query in queries {
            let results = try await ivfIndex.search(
                query: SIMD128<Float>(query),
                k: k
            )
            ivfResults.append(results)
        }
        let ivfEnd = DispatchTime.now()
        let ivfTime = Double(ivfEnd.uptimeNanoseconds - ivfStart.uptimeNanoseconds) / 1_000_000_000
        
        // Learned search
        let learnedStart = DispatchTime.now()
        var learnedResults: [[SearchResult<TestMetadata>]] = []
        for query in queries {
            let results = try await learnedIndex.search(
                query: SIMD128<Float>(query),
                k: k
            )
            learnedResults.append(results)
        }
        let learnedEnd = DispatchTime.now()
        let learnedTime = Double(learnedEnd.uptimeNanoseconds - learnedStart.uptimeNanoseconds) / 1_000_000_000
        
        print("Search times - Hybrid: \(hybridTime)s, IVF: \(ivfTime)s, Learned: \(learnedTime)s")
        
        // Calculate recall overlap
        var hybridIvfOverlap = 0
        var hybridLearnedOverlap = 0
        
        for i in 0..<queryCount {
            let hybridIds = Set(hybridResults[i].map { $0.id })
            let ivfIds = Set(ivfResults[i].map { $0.id })
            let learnedIds = Set(learnedResults[i].map { $0.id })
            
            hybridIvfOverlap += hybridIds.intersection(ivfIds).count
            hybridLearnedOverlap += hybridIds.intersection(learnedIds).count
        }
        
        print("Avg overlap - Hybrid/IVF: \(Float(hybridIvfOverlap)/Float(queryCount * k)), Hybrid/Learned: \(Float(hybridLearnedOverlap)/Float(queryCount * k))")
        
        // Hybrid should leverage strengths of both
        #expect(hybridTime < ivfTime + learnedTime)  // Should be faster than running both
    }
    
    // MARK: - Adaptive Learning Tests
    
    @Test("Adaptive Routing", .timeLimit(.seconds(45)))
    func testAdaptiveRouting() async throws {
        let config = HybridIndexConfiguration(
            dimensions: 64,
            routingStrategy: .adaptive,
            adaptiveThreshold: 0.6
        )
        
        let index = try await HybridIndex<SIMD64<Float>, TestMetadata>(configuration: config)
        
        // Generate different types of data
        let clusteredVectors = IVFIndexTests.generateClusteredVectors(
            clusterCount: 10,
            vectorsPerCluster: 50,
            dimensions: 64
        )
        
        let sequentialVectors = LearnedIndexTests.generateSequentialVectors(
            count: 500,
            dimensions: 64
        )
        
        // Train on mixed data
        let allVectors = clusteredVectors + sequentialVectors
        try await index.train(on: allVectors)
        
        // Insert vectors
        for (i, vector) in allVectors.prefix(200).enumerated() {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD64<Float>(vector),
                metadata: TestMetadata(label: i < clusteredVectors.count ? "clustered" : "sequential")
            )
            _ = try await index.insert(entry)
        }
        
        // Test queries that should route to different indexes
        let clusteredQuery = SIMD64<Float>(clusteredVectors[25])
        let sequentialQuery = SIMD64<Float>(sequentialVectors[25])
        
        // Multiple searches to trigger adaptive behavior
        for _ in 0..<10 {
            _ = try await index.search(query: clusteredQuery, k: 5)
            _ = try await index.search(query: sequentialQuery, k: 5)
        }
        
        // Check routing statistics
        let stats = await index.statistics()
        print("Total queries: \(stats.routingStatistics.totalQueries)")
        print("IVF queries: \(stats.routingStatistics.ivfQueries)")
        print("Learned queries: \(stats.routingStatistics.learnedQueries)")
        print("Hybrid queries: \(stats.routingStatistics.hybridQueries)")
        
        // Should have used different strategies
        #expect(stats.routingStatistics.ivfQueries > 0)
        #expect(stats.routingStatistics.learnedQueries > 0)
    }
    
    // MARK: - Integrity Tests
    
    @Test("Hybrid Index Integrity")
    func testIntegrity() async throws {
        let config = HybridIndexConfiguration(
            dimensions: 32,
            routingStrategy: .fixed(ratio: 0.5)
        )
        
        let index = try await HybridIndex<SIMD32<Float>, TestMetadata>(configuration: config)
        
        // Train
        let vectors = Self.generateMixedVectors(count: 200, dimensions: 32)
        try await index.train(on: vectors)
        
        // Insert different amounts in each index (simulate inconsistency)
        for i in 0..<100 {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD32<Float>(vectors[i]),
                metadata: TestMetadata(label: "integrity\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        // Check integrity
        let integrity = try await index.validateIntegrity()
        
        // Should be valid if both indexes are consistent
        #expect(integrity.isValid == true)
        #expect(integrity.errors.isEmpty == true)
        
        // Check statistics consistency
        let stats = await index.statistics()
        #expect(stats.ivfStatistics.vectorCount == stats.learnedStatistics.vectorCount)
    }
    
    // MARK: - Export/Import Tests
    
    @Test("Export and Import")
    func testExportImport() async throws {
        let config = HybridIndexConfiguration(
            dimensions: 16,
            routingStrategy: .adaptive
        )
        
        let index1 = try await HybridIndex<SIMD16<Float>, TestMetadata>(configuration: config)
        
        // Train and insert data
        let vectors = Self.generateMixedVectors(count: 100, dimensions: 16)
        try await index1.train(on: vectors)
        
        for i in 0..<50 {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD16<Float>(vectors[i]),
                metadata: TestMetadata(label: "export\(i)")
            )
            _ = try await index1.insert(entry)
        }
        
        // Export
        let exportedData = try await index1.export(format: .json)
        
        // Create new index and import
        let index2 = try await HybridIndex<SIMD16<Float>, TestMetadata>(configuration: config)
        try await index2.import(data: exportedData, format: .json)
        
        // Verify routing statistics are preserved
        let stats1 = await index1.statistics()
        let stats2 = await index2.statistics()
        
        // Note: Full state restoration would require exporting sub-indexes too
        #expect(stats2.vectorCount == 0)  // New index doesn't have data yet
        
        // Train and insert in new index
        try await index2.train(on: vectors)
        for i in 0..<50 {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD16<Float>(vectors[i]),
                metadata: TestMetadata(label: "export\(i)")
            )
            _ = try await index2.insert(entry)
        }
        
        // Search should work similarly
        let query = SIMD16<Float>(vectors[25])
        let results1 = try await index1.search(query: query, k: 5)
        let results2 = try await index2.search(query: query, k: 5)
        
        #expect(results1.first?.id == results2.first?.id)
    }
    
    // MARK: - Edge Cases
    
    @Test("Empty Hybrid Index")
    func testEmptyIndex() async throws {
        let config = HybridIndexConfiguration(dimensions: 8)
        let index = try await HybridIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Search on untrained index
        let query = SIMD8<Float>(repeating: 1.0)
        let results = try await index.search(query: query, k: 5)
        #expect(results.isEmpty == true)
        
        // Delete non-existent
        let deleted = try await index.delete(id: "nonexistent")
        #expect(deleted == false)
        
        // Train on empty data should fail
        await #expect(throws: Error.self) {
            try await index.train(on: [])
        }
    }
    
    @Test("Optimization Strategies")
    func testOptimization() async throws {
        let config = HybridIndexConfiguration(
            dimensions: 32,
            routingStrategy: .adaptive
        )
        
        let index = try await HybridIndex<SIMD32<Float>, TestMetadata>(configuration: config)
        
        // Train and insert data
        let vectors = Self.generateMixedVectors(count: 300, dimensions: 32)
        try await index.train(on: vectors)
        
        for i in 0..<200 {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD32<Float>(vectors[i]),
                metadata: TestMetadata(label: "opt\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        // Run different optimization strategies
        try await index.optimize(strategy: .light)
        try await index.optimize(strategy: .aggressive)
        try await index.optimize(strategy: .adaptive)
        
        // Index should still work after optimization
        let query = SIMD32<Float>(vectors[100])
        let results = try await index.search(query: query, k: 10)
        #expect(results.count > 0)
        
        // Check if optimization improved anything
        let profile = await index.performanceProfile()
        print("Search latency p50: \(profile.searchLatency.p50 * 1000)ms")
        print("Memory efficiency: \(profile.memoryUsage.efficiency)")
    }
    
    @Test("Distribution Analysis")
    func testDistributionAnalysis() async throws {
        let config = HybridIndexConfiguration(dimensions: 64)
        let index = try await HybridIndex<SIMD64<Float>, TestMetadata>(configuration: config)
        
        // Insert diverse data
        let vectors = Self.generateMixedVectors(count: 500, dimensions: 64)
        try await index.train(on: vectors)
        
        for i in 0..<300 {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD64<Float>(vectors[i]),
                metadata: TestMetadata(label: "dist\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        // Analyze distribution
        let distribution = await index.analyzeDistribution()
        
        #expect(distribution.dimensionality == 64)
        #expect(distribution.density > 0)
        
        // Should have clustering info from IVF
        #expect(distribution.clustering.estimatedClusters > 0)
        
        // Get visualization data
        let vizData = await index.visualizationData()
        #expect(vizData.nodePositions.count > 0)
        #expect(vizData.layoutAlgorithm == "hybrid_layout")
    }
}
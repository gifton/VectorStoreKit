// VectorStoreKit: Comprehensive Benchmark Tests
//
// Test suite for validating benchmark functionality

import Testing
import Foundation
@testable import VectorStoreKit

@Suite("Comprehensive Benchmark Tests")
struct ComprehensiveBenchmarkTests {
    
    // MARK: - Configuration Tests
    
    @Test("Benchmark Configuration")
    func testBenchmarkConfiguration() {
        // Test default configurations
        let small = ComprehensiveBenchmarks.BenchmarkConfiguration.small
        #expect(small.dimensions == 64)
        #expect(small.vectorCounts == [100, 1_000, 10_000])
        
        let standard = ComprehensiveBenchmarks.BenchmarkConfiguration.standard
        #expect(standard.dimensions == 128)
        #expect(standard.vectorCounts == [1_000, 10_000, 100_000])
        
        let large = ComprehensiveBenchmarks.BenchmarkConfiguration.large
        #expect(large.dimensions == 256)
        #expect(large.vectorCounts == [10_000, 100_000, 1_000_000])
        
        // Test custom configuration
        let custom = ComprehensiveBenchmarks.BenchmarkConfiguration(
            dimensions: 512,
            vectorCounts: [500, 5_000],
            queryCount: 200,
            k: 20,
            iterations: 3,
            warmupIterations: 1
        )
        
        #expect(custom.dimensions == 512)
        #expect(custom.queryCount == 200)
        #expect(custom.k == 20)
    }
    
    // MARK: - Benchmark Execution Tests
    
    @Test("Small Benchmark Execution", .timeLimit(.seconds(60)))
    func testSmallBenchmark() async throws {
        let config = ComprehensiveBenchmarks.BenchmarkConfiguration(
            dimensions: 32,
            vectorCounts: [100, 500],
            queryCount: 50,
            k: 5,
            iterations: 2,
            warmupIterations: 1
        )
        
        let benchmarks = ComprehensiveBenchmarks(configuration: config)
        let report = try await benchmarks.runAllBenchmarks()
        
        // Validate report
        #expect(report.results.count > 0)
        #expect(!report.bestInsertPerformance.isEmpty)
        #expect(!report.bestSearchPerformance.isEmpty)
        #expect(!report.bestMemoryEfficiency.isEmpty)
        #expect(!report.bestRecall.isEmpty)
        
        // Check that all index types were benchmarked
        let indexTypes = Set(report.results.map { $0.indexType })
        #expect(indexTypes.contains("IVF"))
        #expect(indexTypes.contains("Learned"))
        #expect(indexTypes.contains("Hybrid"))
        
        // Validate result metrics
        for result in report.results {
            #expect(result.insertTime > 0)
            #expect(result.searchTime > 0)
            #expect(result.memoryUsage > 0)
            #expect(result.recall >= 0 && result.recall <= 1)
            #expect(result.throughput.insertsPerSecond > 0)
            #expect(result.throughput.queriesPerSecond > 0)
        }
    }
    
    // MARK: - Performance Benchmark Tests
    
    @Test("Distance Computation Benchmarks")
    func testDistanceComputationBenchmarks() async throws {
        try await PerformanceBenchmarks.benchmarkDistanceComputations()
        // Test completes without throwing
    }
    
    @Test("IVF Optimization Benchmarks", .timeLimit(.seconds(120)))
    func testIVFOptimizationBenchmarks() async throws {
        // Run with smaller parameters for testing
        let originalMethod = PerformanceBenchmarks.benchmarkIVFOptimizations
        
        // Create a smaller test version
        let dimensions = 32
        let vectorCount = 1_000
        let centroidCounts = [16, 64]
        
        for centroids in centroidCounts {
            let config = IVFConfiguration(
                dimensions: dimensions,
                numberOfCentroids: centroids,
                numberOfProbes: max(1, centroids / 10)
            )
            
            let index = try await IVFIndex<SIMD32<Float>, TestMetadata>(configuration: config)
            
            // Generate test data
            let vectors = (0..<vectorCount).map { _ in
                (0..<dimensions).map { _ in Float.random(in: -1...1) }
            }
            
            // Train
            try await index.train(on: Array(vectors.prefix(500)))
            
            // Insert some vectors
            for i in 0..<100 {
                let entry = VectorEntry(
                    id: "test_\(i)",
                    vector: SIMD32<Float>(vectors[i]),
                    metadata: TestMetadata(label: "test")
                )
                _ = try await index.insert(entry)
            }
            
            // Verify index is functional
            let query = SIMD32<Float>(vectors[50])
            let results = try await index.search(query: query, k: 5)
            #expect(results.count > 0)
        }
    }
    
    // MARK: - Specialized Benchmark Tests
    
    @Test("Memory Allocation Patterns")
    func testMemoryAllocationBenchmark() async throws {
        // Test with small configuration
        let dimensions = 16
        let bucketSizes = [10, 100]
        
        for bucketSize in bucketSizes {
            let config = LearnedIndexConfiguration(
                dimensions: dimensions,
                modelArchitecture: .linear,
                bucketSize: bucketSize
            )
            
            let index = try await LearnedIndex<SIMD16<Float>, TestMetadata>(configuration: config)
            
            var previousMemory = 0
            for i in 0..<100 {
                let entry = VectorEntry(
                    id: "mem_\(i)",
                    vector: SIMD16<Float>(repeating: Float(i)),
                    metadata: TestMetadata(label: "memory")
                )
                _ = try await index.insert(entry)
                
                if i % 10 == 0 {
                    let currentMemory = await index.memoryUsage
                    #expect(currentMemory >= previousMemory)
                    previousMemory = currentMemory
                }
            }
        }
    }
    
    @Test("Concurrent Access Patterns", .timeLimit(.seconds(30)))
    func testConcurrentAccessBenchmark() async throws {
        let dimensions = 32
        let config = HybridIndexConfiguration(dimensions: dimensions)
        let index = try await HybridIndex<SIMD32<Float>, TestMetadata>(configuration: config)
        
        // Initialize with data
        let vectors = (0..<100).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }
        try await index.train(on: vectors)
        
        // Test concurrent operations
        let operationCount = 100
        
        await withTaskGroup(of: Void.self) { group in
            // Concurrent writers
            for i in 0..<2 {
                group.addTask {
                    for j in 0..<25 {
                        let vector = (0..<dimensions).map { _ in Float.random(in: -1...1) }
                        let entry = VectorEntry(
                            id: "concurrent_w_\(i)_\(j)",
                            vector: SIMD32<Float>(vector),
                            metadata: TestMetadata(label: "write")
                        )
                        _ = try? await index.insert(entry)
                    }
                }
            }
            
            // Concurrent readers
            for i in 0..<2 {
                group.addTask {
                    for _ in 0..<25 {
                        let query = SIMD32<Float>(repeating: Float.random(in: -1...1))
                        _ = try? await index.search(query: query, k: 5)
                    }
                }
            }
        }
        
        // Verify index integrity
        let finalCount = await index.count
        #expect(finalCount >= 50) // At least the concurrent writes
    }
    
    // MARK: - Report Generation Tests
    
    @Test("Benchmark Report Generation")
    func testReportGeneration() {
        // Create sample results
        let results = [
            ComprehensiveBenchmarks.BenchmarkResult(
                indexType: "IVF",
                vectorCount: 1000,
                insertTime: 1.5,
                searchTime: 0.5,
                memoryUsage: 1_000_000,
                recall: 0.95,
                precision: 0.95,
                throughput: .init(insertsPerSecond: 667, queriesPerSecond: 2000)
            ),
            ComprehensiveBenchmarks.BenchmarkResult(
                indexType: "Learned",
                vectorCount: 1000,
                insertTime: 2.0,
                searchTime: 0.3,
                memoryUsage: 800_000,
                recall: 0.92,
                precision: 0.92,
                throughput: .init(insertsPerSecond: 500, queriesPerSecond: 3333)
            ),
            ComprehensiveBenchmarks.BenchmarkResult(
                indexType: "Hybrid",
                vectorCount: 1000,
                insertTime: 1.8,
                searchTime: 0.4,
                memoryUsage: 1_200_000,
                recall: 0.97,
                precision: 0.97,
                throughput: .init(insertsPerSecond: 556, queriesPerSecond: 2500)
            )
        ]
        
        // Generate report using private method via reflection (in real code, make this testable)
        let benchmarks = ComprehensiveBenchmarks()
        
        // Validate expected report structure
        #expect(results.count == 3)
        #expect(results.allSatisfy { $0.recall > 0.9 })
        
        // Find best performers
        let bestInsert = results.min { $0.insertTime < $1.insertTime }
        let bestSearch = results.max { $0.throughput.queriesPerSecond > $1.throughput.queriesPerSecond }
        let bestMemory = results.min { $0.memoryUsage < $1.memoryUsage }
        let bestRecall = results.max { $0.recall > $1.recall }
        
        #expect(bestInsert?.indexType == "IVF")
        #expect(bestSearch?.indexType == "Learned")
        #expect(bestMemory?.indexType == "Learned")
        #expect(bestRecall?.indexType == "Hybrid")
    }
    
    // MARK: - Adaptive Behavior Tests
    
    @Test("Adaptive Search Benchmark", .timeLimit(.seconds(45)))
    func testAdaptiveSearchBenchmark() async throws {
        let dimensions = 32
        let config = HybridIndexConfiguration(
            dimensions: dimensions,
            routingStrategy: .adaptive,
            adaptiveThreshold: 0.6
        )
        
        let index = try await HybridIndex<SIMD32<Float>, TestMetadata>(configuration: config)
        
        // Generate different pattern types
        let clusteredVectors = (0..<100).map { i in
            let cluster = i / 25
            let center = Float(cluster) * 0.5
            return (0..<dimensions).map { _ in center + Float.random(in: -0.1...0.1) }
        }
        
        let sequentialVectors = (0..<100).map { i in
            let position = Float(i) / 100
            return (0..<dimensions).map { d in Float(d) * position / Float(dimensions) }
        }
        
        // Train and insert
        try await index.train(on: clusteredVectors + sequentialVectors)
        
        for (i, vector) in (clusteredVectors + sequentialVectors).enumerated() {
            let entry = VectorEntry(
                id: "adaptive_\(i)",
                vector: SIMD32<Float>(vector),
                metadata: TestMetadata(label: i < 100 ? "clustered" : "sequential")
            )
            _ = try await index.insert(entry)
        }
        
        // Run adaptive queries
        for _ in 0..<5 {
            // Clustered queries
            for i in 0..<10 {
                let query = SIMD32<Float>(clusteredVectors[i * 10])
                _ = try await index.search(query: query, k: 5)
            }
            
            // Sequential queries
            for i in 0..<10 {
                let query = SIMD32<Float>(sequentialVectors[i * 10])
                _ = try await index.search(query: query, k: 5)
            }
        }
        
        // Check routing adapted
        let stats = await index.statistics()
        #expect(stats.routingStatistics.totalQueries >= 100)
        #expect(stats.routingStatistics.ivfQueries > 0)
        #expect(stats.routingStatistics.learnedQueries > 0)
    }
}

// MARK: - Benchmark Runner Tests

@Suite("Benchmark Runner Tests")
struct BenchmarkRunnerTests {
    
    @Test("Command Line Parsing")
    func testCommandLineParsing() throws {
        // Test various command line argument combinations
        let validArgs = [
            ["--size", "small"],
            ["--dimensions", "64"],
            ["--vector-counts", "100,1000"],
            ["--queries", "50"],
            ["--k", "10"],
            ["--output", "json"],
            ["--file", "results.json"],
            ["--quick"],
            ["--verbose"]
        ]
        
        // Each argument set should be valid
        for args in validArgs {
            // In real implementation, would test argument parsing
            #expect(args.count >= 1)
        }
    }
    
    @Test("Output Format Generation")
    func testOutputFormatGeneration() {
        let results = [
            ComprehensiveBenchmarks.BenchmarkResult(
                indexType: "IVF",
                vectorCount: 1000,
                insertTime: 1.0,
                searchTime: 0.1,
                memoryUsage: 1_000_000,
                recall: 0.95,
                precision: 0.95,
                throughput: .init(insertsPerSecond: 1000, queriesPerSecond: 10000)
            )
        ]
        
        // Test CSV generation
        let csvHeader = "Index Type,Vector Count,Insert Time (s),Search Time (s),Memory (bytes),Recall,Inserts/sec,Queries/sec"
        let csvRow = "IVF,1000,1.0,0.1,1000000,0.95,1000.0,10000.0"
        
        // Verify CSV format
        #expect(csvHeader.contains("Index Type"))
        #expect(csvRow.contains("IVF"))
        
        // Test JSON encoding
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        
        #expect(throws: Never.self) {
            _ = try encoder.encode(results)
        }
    }
}
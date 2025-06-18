// VectorStoreKit: Concurrency Benchmarks
//
// Multi-threaded performance benchmarks

import Foundation
import simd

/// Benchmarks for concurrent operations
public struct ConcurrencyBenchmarks {
    
    private let framework: BenchmarkFramework
    private let metrics: PerformanceMetrics
    
    public init(
        framework: BenchmarkFramework = BenchmarkFramework(),
        metrics: PerformanceMetrics = PerformanceMetrics()
    ) {
        self.framework = framework
        self.metrics = metrics
    }
    
    // MARK: - Types
    
    public struct ConcurrencyReport {
        public let operation: String
        public let threadCounts: [Int]
        public let throughputs: [Double]
        public let latencies: [TimeInterval]
        public let contentions: [Double]
        
        public var optimalThreadCount: Int {
            guard let maxIndex = throughputs.firstIndex(of: throughputs.max() ?? 0) else {
                return 1
            }
            return threadCounts[maxIndex]
        }
        
        public var scalabilityEfficiency: Double {
            guard threadCounts.count >= 2 else { return 1.0 }
            let singleThreadThroughput = throughputs.first ?? 1.0
            let maxThreadCount = threadCounts.last ?? 1
            let maxThreadThroughput = throughputs.last ?? 1.0
            let idealThroughput = singleThreadThroughput * Double(maxThreadCount)
            return maxThreadThroughput / idealThroughput
        }
    }
    
    // MARK: - Main Benchmark Suites
    
    /// Run all concurrency benchmarks
    public func runAll() async throws -> [String: BenchmarkFramework.Statistics] {
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        // Read-heavy workload benchmarks
        results.merge(try await runReadHeavyBenchmarks()) { _, new in new }
        
        // Write-heavy workload benchmarks
        results.merge(try await runWriteHeavyBenchmarks()) { _, new in new }
        
        // Mixed workload benchmarks
        results.merge(try await runMixedWorkloadBenchmarks()) { _, new in new }
        
        // Actor contention benchmarks
        results.merge(try await runActorContentionBenchmarks()) { _, new in new }
        
        return results
    }
    
    // MARK: - Read-Heavy Benchmarks
    
    private func runReadHeavyBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let threadCounts = [1, 2, 4, 8, 16, 32]
        let dimensions = 128
        let indexSize = 100_000
        let queriesPerThread = 1_000
        
        // Prepare index
        let index = try await prepareIndex(size: indexSize, dimensions: dimensions)
        let queries = generateQueries(count: queriesPerThread * threadCounts.max()!, dimensions: dimensions)
        
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        for threads in threadCounts {
            let suite = benchmarkSuite(
                name: "Read-Heavy Concurrency (\(threads) threads)",
                description: "Concurrent read performance"
            ) {
                // Pure search workload
                benchmark(name: "concurrent_search_\(threads)threads") {
                    await withTaskGroup(of: Void.self) { group in
                        for threadId in 0..<threads {
                            group.addTask {
                                let startIdx = threadId * queriesPerThread
                                let endIdx = startIdx + queriesPerThread
                                
                                for i in startIdx..<endIdx {
                                    let query = queries[i % queries.count]
                                    _ = try? await index.search(
                                        query: query,
                                        k: 10
                                    )
                                }
                            }
                        }
                    }
                }
                
                // Search with different k values
                benchmark(name: "concurrent_search_varied_k_\(threads)threads") {
                    await withTaskGroup(of: Void.self) { group in
                        for threadId in 0..<threads {
                            group.addTask {
                                let kValues = [1, 10, 50, 100]
                                
                                for i in 0..<queriesPerThread {
                                    let query = queries[i % queries.count]
                                    let k = kValues[i % kValues.count]
                                    _ = try? await index.search(
                                        query: query,
                                        k: k
                                    )
                                }
                            }
                        }
                    }
                }
                
                // Filtered search workload
                benchmark(name: "concurrent_filtered_search_\(threads)threads") {
                    await withTaskGroup(of: Void.self) { group in
                        for threadId in 0..<threads {
                            group.addTask {
                                for i in 0..<queriesPerThread {
                                    let query = queries[i % queries.count]
                                    let filter = createCategoryFilter(
                                        category: "cat_\(threadId % 10)"
                                    )
                                    _ = try? await index.search(
                                        query: query,
                                        k: 10,
                                        filter: filter
                                    )
                                }
                            }
                        }
                    }
                }
            }
            
            let suiteResults = try await framework.run(suite: suite)
            results.merge(suiteResults) { _, new in new }
        }
        
        return results
    }
    
    // MARK: - Write-Heavy Benchmarks
    
    private func runWriteHeavyBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let threadCounts = [1, 2, 4, 8, 16]
        let dimensions = 128
        let insertsPerThread = 1_000
        
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        for threads in threadCounts {
            let suite = benchmarkSuite(
                name: "Write-Heavy Concurrency (\(threads) threads)",
                description: "Concurrent write performance"
            ) {
                // Pure insert workload
                benchmark(name: "concurrent_insert_\(threads)threads") {
                    let index = try await createEmptyIndex(dimensions: dimensions)
                    
                    await withTaskGroup(of: Void.self) { group in
                        for threadId in 0..<threads {
                            group.addTask {
                                for i in 0..<insertsPerThread {
                                    let vector = self.generateVector(dimensions: dimensions)
                                    let entry = VectorEntry(
                                        id: "thread\(threadId)_vec\(i)",
                                        vector: vector,
                                        metadata: TestMetadata(
                                            id: threadId * insertsPerThread + i,
                                            category: "thread_\(threadId)",
                                            timestamp: Date(),
                                            tags: ["concurrent_write"]
                                        )
                                    )
                                    _ = try? await index.insert(entry)
                                }
                            }
                        }
                    }
                    
                    blackHole(index)
                }
                
                // Bulk insert workload
                benchmark(name: "concurrent_bulk_insert_\(threads)threads") {
                    let index = try await createEmptyIndex(dimensions: dimensions)
                    let batchSize = 100
                    
                    await withTaskGroup(of: Void.self) { group in
                        for threadId in 0..<threads {
                            group.addTask {
                                for batch in 0..<(insertsPerThread / batchSize) {
                                    var entries: [VectorEntry<SIMD32<Float>, TestMetadata>] = []
                                    
                                    for i in 0..<batchSize {
                                        let vector = self.generateVector(dimensions: dimensions)
                                        entries.append(VectorEntry(
                                            id: "thread\(threadId)_batch\(batch)_vec\(i)",
                                            vector: vector,
                                            metadata: TestMetadata(
                                                id: threadId * insertsPerThread + batch * batchSize + i,
                                                category: "thread_\(threadId)",
                                                timestamp: Date(),
                                                tags: ["bulk_write"]
                                            )
                                        ))
                                    }
                                    
                                    try? await index.bulkInsert(entries)
                                }
                            }
                        }
                    }
                    
                    blackHole(index)
                }
                
                // Delete workload
                benchmark(name: "concurrent_delete_\(threads)threads") {
                    let index = try await prepareIndex(
                        size: threads * insertsPerThread,
                        dimensions: dimensions
                    )
                    
                    await withTaskGroup(of: Void.self) { group in
                        for threadId in 0..<threads {
                            group.addTask {
                                for i in 0..<insertsPerThread {
                                    let id = "vec_\(threadId * insertsPerThread + i)"
                                    _ = try? await index.delete(id: id)
                                }
                            }
                        }
                    }
                    
                    blackHole(index)
                }
            }
            
            let suiteResults = try await framework.run(suite: suite)
            results.merge(suiteResults) { _, new in new }
        }
        
        return results
    }
    
    // MARK: - Mixed Workload Benchmarks
    
    private func runMixedWorkloadBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let threadCounts = [1, 2, 4, 8, 16]
        let dimensions = 128
        let operationsPerThread = 1_000
        let readWriteRatios = [(read: 0.9, write: 0.1), (read: 0.7, write: 0.3), (read: 0.5, write: 0.5)]
        
        // Prepare base index
        let baseIndex = try await prepareIndex(size: 50_000, dimensions: dimensions)
        let queries = generateQueries(count: 10_000, dimensions: dimensions)
        
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        for threads in threadCounts {
            for ratio in readWriteRatios {
                let ratioStr = "\(Int(ratio.read * 100))r\(Int(ratio.write * 100))w"
                
                let suite = benchmarkSuite(
                    name: "Mixed Workload (\(threads) threads, \(ratioStr))",
                    description: "Mixed read/write performance"
                ) {
                    benchmark(name: "mixed_\(ratioStr)_\(threads)threads") {
                        await withTaskGroup(of: Void.self) { group in
                            for threadId in 0..<threads {
                                group.addTask {
                                    for i in 0..<operationsPerThread {
                                        let isRead = Double.random(in: 0...1) < ratio.read
                                        
                                        if isRead {
                                            // Perform search
                                            let query = queries[i % queries.count]
                                            _ = try? await baseIndex.search(
                                                query: query,
                                                k: 10
                                            )
                                        } else {
                                            // Perform insert
                                            let vector = self.generateVector(dimensions: dimensions)
                                            let entry = VectorEntry(
                                                id: "mixed_thread\(threadId)_op\(i)",
                                                vector: vector,
                                                metadata: TestMetadata(
                                                    id: threadId * operationsPerThread + i,
                                                    category: "mixed_\(threadId)",
                                                    timestamp: Date(),
                                                    tags: ["mixed"]
                                                )
                                            )
                                            _ = try? await baseIndex.insert(entry)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                let suiteResults = try await framework.run(suite: suite)
                results.merge(suiteResults) { _, new in new }
            }
        }
        
        return results
    }
    
    // MARK: - Actor Contention Benchmarks
    
    private func runActorContentionBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let dimensions = 128
        let threadCounts = [1, 2, 4, 8, 16, 32, 64]
        let operationsPerThread = 100
        
        var allBenchmarks: [BenchmarkFramework.Benchmark] = []
        
        for threads in threadCounts {
            allBenchmarks.append(contentsOf: createActorBenchmarks(threads: threads, operationsPerThread: operationsPerThread))
        }
        
        let suite = BenchmarkFramework.Suite(
            name: "Actor Contention",
            description: "Measure actor-based concurrency overhead",
            benchmarks: allBenchmarks
        )
        
        return try await framework.run(suite)
    }
    
    private func createActorBenchmarks(threads: Int, operationsPerThread: Int) -> [BenchmarkFramework.Benchmark] {
        return [
            BenchmarkFramework.Benchmark(
                name: "shared_actor_\(threads)threads",
                run: {
                    let sharedActor = SharedTestActor()
                    
                    await withTaskGroup(of: Void.self) { group in
                        for _ in 0..<threads {
                            group.addTask {
                                for _ in 0..<operationsPerThread {
                                    await sharedActor.performOperation()
                                }
                            }
                        }
                    }
                }
            ),
            
            // Multiple actors (one per thread)
            BenchmarkFramework.Benchmark(
                name: "per_thread_actor_\(threads)threads",
                run: {
                    let actors = (0..<threads).map { _ in SharedTestActor() }
                    
                    await withTaskGroup(of: Void.self) { group in
                        for (index, actor) in actors.enumerated() {
                            group.addTask {
                                for _ in 0..<operationsPerThread {
                                    await actor.performOperation()
                                }
                            }
                        }
                    }
                }
            ),
            
            // Actor pool
            BenchmarkFramework.Benchmark(
                name: "actor_pool_\(threads)threads",
                run: {
                    let poolSize = min(8, threads)
                    let actorPool = (0..<poolSize).map { _ in SharedTestActor() }
                    
                    await withTaskGroup(of: Void.self) { group in
                        for threadId in 0..<threads {
                            group.addTask {
                                let actor = actorPool[threadId % poolSize]
                                for _ in 0..<operationsPerThread {
                                    await actor.performOperation()
                                }
                            }
                        }
                    }
                }
            )
        ]
    }
    
    // MARK: - Helper Functions
    
    private func prepareIndex(
        size: Int,
        dimensions: Int
    ) async throws -> HNSWIndex<SIMD32<Float>, TestMetadata> {
        let config = HNSWIndex<SIMD32<Float>, TestMetadata>.Configuration(
            maxConnections: 16,
            efConstruction: 200
        )
        
        let index = try HNSWIndex<SIMD32<Float>, TestMetadata>(
            configuration: config
        )
        
        for i in 0..<size {
            let vector = generateVector(dimensions: dimensions)
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: vector,
                metadata: TestMetadata(
                    id: i,
                    category: "cat_\(i % 10)",
                    timestamp: Date(),
                    tags: ["prepared"]
                )
            )
            _ = try await index.insert(entry)
        }
        
        return index
    }
    
    private func createEmptyIndex(
        dimensions: Int
    ) async throws -> HNSWIndex<SIMD32<Float>, TestMetadata> {
        let config = HNSWIndex<SIMD32<Float>, TestMetadata>.Configuration(
            maxConnections: 16,
            efConstruction: 200
        )
        
        return try HNSWIndex<SIMD32<Float>, TestMetadata>(
            configuration: config
        )
    }
    
    private func generateVector(dimensions: Int) -> SIMD32<Float> {
        // Use optimized random initialization
        return randomSIMD32(in: -1.0...1.0)
    }
    
    private func generateQueries(count: Int, dimensions: Int) -> [SIMD32<Float>] {
        return (0..<count).map { _ in generateVector(dimensions: dimensions) }
    }
}

// MARK: - Test Types

private actor SharedTestActor {
    private var counter: Int = 0
    private var data: [String: Int] = [:]
    
    func performOperation() {
        counter += 1
        data["key_\(counter % 100)"] = counter
        
        // Simulate some work
        var sum = 0
        for i in 0..<100 {
            sum += i
        }
        blackHole(sum)
    }
}

// Filter implementation for benchmark tests
private func createCategoryFilter(category: String) -> SearchFilter {
    return .metadata(MetadataFilter(
        key: "category",
        operation: .equals,
        value: category
    ))
}
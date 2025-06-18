// VectorStoreKit: Scalability Benchmarks
//
// Performance at different scales (1K to 1M vectors)

import Foundation
import simd

/// Benchmarks for testing scalability
public struct ScalabilityBenchmarks {
    
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
    
    public struct ScalabilityReport {
        public let operation: String
        public let dataPoints: [DataPoint]
        
        public struct DataPoint {
            public let scale: Int
            public let time: TimeInterval
            public let throughput: Double
            public let memory: Int
            public let efficiency: Double
        }
        
        public var scalabilityFactor: Double {
            guard dataPoints.count >= 2 else { return 1.0 }
            let first = dataPoints.first!
            let last = dataPoints.last!
            let expectedSpeedup = Double(last.scale) / Double(first.scale)
            let actualSpeedup = first.throughput / last.throughput
            return actualSpeedup / expectedSpeedup
        }
    }
    
    // MARK: - Main Benchmark Suites
    
    /// Run all scalability benchmarks
    public func runAll() async throws -> [String: BenchmarkFramework.Statistics] {
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        // Index scalability
        results.merge(try await runIndexScalabilityBenchmarks()) { _, new in new }
        
        // Search scalability
        results.merge(try await runSearchScalabilityBenchmarks()) { _, new in new }
        
        // Insert scalability
        results.merge(try await runInsertScalabilityBenchmarks()) { _, new in new }
        
        // Memory scalability
        results.merge(try await runMemoryScalabilityBenchmarks()) { _, new in new }
        
        return results
    }
    
    // MARK: - Index Scalability Benchmarks
    
    private func runIndexScalabilityBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let scales = [1_000, 10_000, 100_000, 1_000_000]
        let dimensions = 128
        
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        // HNSW scalability
        let hnswSuite = benchmarkSuite(
            name: "HNSW Index Scalability",
            description: "HNSW performance at different scales"
        ) {
            for scale in scales {
                benchmark(
                    name: "hnsw_build_\(scale)",
                    setUp: {
                        await metrics.startCollection(
                            name: "hnsw_build_\(scale)",
                            metadata: ["scale": "\(scale)", "dimensions": "\(dimensions)"]
                        )
                    },
                    tearDown: {
                        _ = await metrics.stopCollection()
                    }
                ) {
                    let config = HNSWIndex<SIMD32<Float>, TestMetadata>.Configuration(
                        maxConnections: 16,
                        efConstruction: 200
                    )
                    
                    let index = try HNSWIndex<SIMD32<Float>, TestMetadata>(
                        configuration: config
                    )
                    
                    let vectors = generateTestVectors(count: scale, dimensions: dimensions)
                    
                    for (i, vector) in vectors.enumerated() {
                        _ = try await index.insert(VectorEntry(
                            id: "vec_\(i)",
                            vector: vector.simd32,
                            metadata: vector.metadata
                        ))
                    }
                    
                    blackHole(index)
                }
            }
        }
        
        let hnswResults = try await framework.run(suite: hnswSuite)
        results.merge(hnswResults) { _, new in new }
        
        // IVF scalability
        let ivfSuite = benchmarkSuite(
            name: "IVF Index Scalability",
            description: "IVF performance at different scales"
        ) {
            for scale in scales {
                benchmark(name: "ivf_build_\(scale)") {
                    let centroids = min(1024, scale / 100)
                    let config = IVFConfiguration(
                        dimensions: dimensions,
                        numberOfCentroids: centroids
                    )
                    
                    let index = try await IVFIndex<SIMD32<Float>, TestMetadata>(
                        configuration: config
                    )
                    
                    let vectors = generateTestVectors(count: scale, dimensions: dimensions)
                    
                    // Training phase
                    let trainingData = vectors.prefix(min(10_000, scale)).map { $0.vector }
                    try await index.train(on: trainingData)
                    
                    // Insert phase
                    for (i, vector) in vectors.enumerated() {
                        _ = try await index.insert(VectorEntry(
                            id: "vec_\(i)",
                            vector: vector.simd32,
                            metadata: vector.metadata
                        ))
                    }
                    
                    blackHole(index)
                }
            }
        }
        
        let ivfResults = try await framework.run(suite: ivfSuite)
        results.merge(ivfResults) { _, new in new }
        
        return results
    }
    
    // MARK: - Search Scalability Benchmarks
    
    private func runSearchScalabilityBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let scales = [1_000, 10_000, 100_000, 1_000_000]
        let dimensions = 128
        let k = 100
        let queryCount = 1000
        
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        for scale in scales {
            // Prepare index
            let index = try await prepareHNSWIndex(scale: scale, dimensions: dimensions)
            let queries = generateTestVectors(count: queryCount, dimensions: dimensions)
            
            let suite = benchmarkSuite(
                name: "Search Scalability (\(scale) vectors)",
                description: "Search performance at scale"
            ) {
                // Single query latency
                benchmark(name: "search_latency_\(scale)") {
                    let query = queries.randomElement()!
                    let results = try await index.search(
                        query: query.simd32,
                        k: k
                    )
                    blackHole(results)
                }
                
                // Batch query throughput
                benchmark(name: "search_throughput_\(scale)") {
                    for query in queries.prefix(100) {
                        let results = try await index.search(
                            query: query.simd32,
                            k: k
                        )
                        blackHole(results)
                    }
                }
                
                // Concurrent search
                benchmark(name: "search_concurrent_\(scale)") {
                    await withTaskGroup(of: Void.self) { group in
                        for query in queries.prefix(100) {
                            group.addTask {
                                let results = try? await index.search(
                                    query: query.simd32,
                                    k: k
                                )
                                blackHole(results)
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
    
    // MARK: - Insert Scalability Benchmarks
    
    private func runInsertScalabilityBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let scales = [1_000, 10_000, 100_000]
        let dimensions = 128
        let batchSizes = [1, 10, 100, 1000]
        
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        for scale in scales {
            for batchSize in batchSizes {
                if batchSize > scale { continue }
                
                let suite = benchmarkSuite(
                    name: "Insert Scalability (\(scale) vectors, batch \(batchSize))",
                    description: "Insert performance at scale"
                ) {
                    // Sequential inserts
                    benchmark(name: "insert_sequential_\(scale)_batch\(batchSize)") {
                        let index = try await prepareEmptyHNSWIndex(dimensions: dimensions)
                        let vectors = generateTestVectors(count: scale, dimensions: dimensions)
                        
                        for batch in vectors.chunked(into: batchSize) {
                            for (i, vector) in batch.enumerated() {
                                _ = try await index.insert(VectorEntry(
                                    id: "vec_\(i)",
                                    vector: vector.simd32,
                                    metadata: vector.metadata
                                ))
                            }
                        }
                        
                        blackHole(index)
                    }
                    
                    // Bulk insert
                    if batchSize > 1 {
                        benchmark(name: "insert_bulk_\(scale)_batch\(batchSize)") {
                            let index = try await prepareEmptyHNSWIndex(dimensions: dimensions)
                            let vectors = generateTestVectors(count: scale, dimensions: dimensions)
                            
                            for batch in vectors.chunked(into: batchSize) {
                                let entries = batch.enumerated().map { i, vector in
                                    VectorEntry(
                                        id: "vec_\(i)",
                                        vector: vector.simd32,
                                        metadata: vector.metadata
                                    )
                                }
                                try await index.bulkInsert(entries)
                            }
                            
                            blackHole(index)
                        }
                    }
                }
                
                let suiteResults = try await framework.run(suite: suite)
                results.merge(suiteResults) { _, new in new }
            }
        }
        
        return results
    }
    
    // MARK: - Memory Scalability Benchmarks
    
    private func runMemoryScalabilityBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let scales = [1_000, 10_000, 100_000, 1_000_000]
        let dimensions = [64, 128, 256, 512]
        
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        let suite = benchmarkSuite(
            name: "Memory Scalability",
            description: "Memory usage at different scales"
        ) {
            for scale in scales {
                for dim in dimensions {
                    if !(scale > 100_000 && dim > 256) { // Skip very large combinations
                        benchmark(
                            name: "memory_usage_\(scale)_\(dim)d",
                        setUp: {
                            await metrics.startCollection(
                                name: "memory_\(scale)_\(dim)d",
                                metadata: ["scale": "\(scale)", "dimensions": "\(dim)"]
                            )
                        },
                        tearDown: {
                            _ = await metrics.stopCollection()
                        }
                    ) {
                        // Measure memory for raw vectors
                        let vectors = generateTestVectors(count: scale, dimensions: dim)
                        let baselineMemory = scale * dim * MemoryLayout<Float>.stride
                        
                        await metrics.recordMemory(
                            name: "baseline",
                            bytes: baselineMemory
                        )
                        
                        // Measure memory for HNSW index
                        let hnswIndex = try await prepareHNSWIndex(
                            scale: scale,
                            dimensions: dim
                        )
                        let hnswMemory = await hnswIndex.memoryUsage
                        
                        await metrics.recordMemory(
                            name: "hnsw",
                            bytes: hnswMemory
                        )
                        
                        // Measure memory for IVF index
                        let ivfIndex = try await prepareIVFIndex(
                            scale: scale,
                            dimensions: dim
                        )
                        let ivfMemory = await ivfIndex.memoryUsage
                        
                        await metrics.recordMemory(
                            name: "ivf",
                            bytes: ivfMemory
                        )
                        
                        // Calculate overhead
                        let hnswOverhead = Double(hnswMemory) / Double(baselineMemory)
                        let ivfOverhead = Double(ivfMemory) / Double(baselineMemory)
                        
                        await metrics.recordCustom(
                            name: "hnsw_overhead",
                            value: hnswOverhead,
                            unit: "ratio"
                        )
                        
                        await metrics.recordCustom(
                            name: "ivf_overhead",
                            value: ivfOverhead,
                            unit: "ratio"
                        )
                        
                        blackHole((vectors, hnswIndex, ivfIndex))
                    }
                    }
                }
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Helper Functions
    
    private struct TestVector {
        let vector: [Float]
        let simd32: SIMD32<Float>
        let metadata: TestMetadata
    }
    
    private func generateTestVectors(count: Int, dimensions: Int) -> [TestVector] {
        return (0..<count).map { i in
            let vector = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            var simd32 = SIMD32<Float>()
            for j in 0..<min(32, dimensions) {
                simd32[j] = vector[j]
            }
            
            return TestVector(
                vector: vector,
                simd32: simd32,
                metadata: TestMetadata(
                    id: i,
                    category: "cat_\(i % 10)",
                    timestamp: Date(),
                    tags: ["scale_test"]
                )
            )
        }
    }
    
    private func prepareHNSWIndex(
        scale: Int,
        dimensions: Int
    ) async throws -> HNSWIndex<SIMD32<Float>, TestMetadata> {
        let config = HNSWIndex<SIMD32<Float>, TestMetadata>.Configuration(
            maxConnections: 16,
            efConstruction: 200
        )
        
        let index = try HNSWIndex<SIMD32<Float>, TestMetadata>(
            configuration: config
        )
        
        let vectors = generateTestVectors(count: scale, dimensions: dimensions)
        for (i, vector) in vectors.enumerated() {
            _ = try await index.insert(VectorEntry(
                id: "vec_\(i)",
                vector: vector.simd32,
                metadata: vector.metadata
            ))
        }
        
        return index
    }
    
    private func prepareIVFIndex(
        scale: Int,
        dimensions: Int
    ) async throws -> IVFIndex<SIMD32<Float>, TestMetadata> {
        let centroids = min(1024, scale / 100)
        let config = IVFConfiguration(
            dimensions: dimensions,
            numberOfCentroids: centroids
        )
        
        let index = try await IVFIndex<SIMD32<Float>, TestMetadata>(
            configuration: config
        )
        
        let vectors = generateTestVectors(count: scale, dimensions: dimensions)
        let trainingData = vectors.prefix(min(10_000, scale)).map { $0.vector }
        try await index.train(on: trainingData)
        
        for (i, vector) in vectors.enumerated() {
            _ = try await index.insert(VectorEntry(
                id: "vec_\(i)",
                vector: vector.simd32,
                metadata: vector.metadata
            ))
        }
        
        return index
    }
    
    private func prepareEmptyHNSWIndex(
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
}

// MARK: - Array Extension for Chunking

// chunked(into:) extension is already defined in MetalDistanceCompute.swift
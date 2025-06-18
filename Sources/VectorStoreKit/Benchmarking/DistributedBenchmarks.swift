// VectorStoreKit: Distributed Benchmarks
//
// Partitioning and rebalancing overhead benchmarks

import Foundation

/// Benchmarks for distributed operations
public struct DistributedBenchmarks {
    
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
    
    struct PartitionTestData {
        let vectors: [[Float]]
        let metadata: [TestMetadata]
        let dimensions: Int
        let count: Int
    }
    
    // MARK: - Main Benchmark Suites
    
    /// Run all distributed benchmarks
    public func runAll() async throws -> [String: BenchmarkFramework.Statistics] {
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        // Partitioning strategy benchmarks
        results.merge(try await runPartitioningBenchmarks()) { _, new in new }
        
        // Rebalancing benchmarks
        results.merge(try await runRebalancingBenchmarks()) { _, new in new }
        
        // Cross-partition query benchmarks
        results.merge(try await runCrossPartitionQueryBenchmarks()) { _, new in new }
        
        // Partition scaling benchmarks
        results.merge(try await runPartitionScalingBenchmarks()) { _, new in new }
        
        return results
    }
    
    // MARK: - Partitioning Strategy Benchmarks
    
    private func runPartitioningBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let vectorCounts = [10_000, 100_000, 1_000_000]
        let partitionCounts = [2, 4, 8, 16, 32]
        let dimensions = 128
        
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        for count in vectorCounts {
            for partitions in partitionCounts {
                let testData = generatePartitionTestData(
                    count: count,
                    dimensions: dimensions
                )
                
                let suite = benchmarkSuite(
                    name: "Partitioning (\(count) vectors, \(partitions) partitions)",
                    description: "Compare partitioning strategies"
                ) {
                    // Hash-based partitioning
                    benchmark(name: "hash_partitioning_\(count)_\(partitions)") {
                        let partitioner = BenchmarkHashPartitioner(partitionCount: partitions)
                        let partitioned = try await partitioner.partition(testData.vectors)
                        blackHole(partitioned)
                    }
                    
                    // Range-based partitioning
                    benchmark(name: "range_partitioning_\(count)_\(partitions)") {
                        let partitioner = BenchmarkRangePartitioner(partitionCount: partitions)
                        let partitioned = try await partitioner.partition(testData.vectors)
                        blackHole(partitioned)
                    }
                    
                    // Clustering-based partitioning
                    benchmark(name: "cluster_partitioning_\(count)_\(partitions)") {
                        let partitioner = ClusterPartitioner(
                            partitionCount: partitions,
                            dimensions: dimensions
                        )
                        let partitioned = try await partitioner.partition(testData.vectors)
                        blackHole(partitioned)
                    }
                    
                    // Consistent hashing
                    benchmark(name: "consistent_hash_partitioning_\(count)_\(partitions)") {
                        let partitioner = BenchmarkConsistentHashPartitioner(
                            virtualNodes: partitions * 100
                        )
                        let partitioned = try await partitioner.partition(testData.vectors)
                        blackHole(partitioned)
                    }
                }
                
                let suiteResults = try await framework.run(suite: suite)
                results.merge(suiteResults) { _, new in new }
            }
        }
        
        return results
    }
    
    // MARK: - Rebalancing Benchmarks
    
    private func runRebalancingBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Partition Rebalancing",
            description: "Benchmark rebalancing operations"
        ) {
            let dimensions = 128
            let initialPartitions = 8
            
            // Node addition scenario
            benchmark(name: "rebalance_add_node") {
                let vectors = generateVectors(count: 100_000, dimensions: dimensions)
                let partitioner = BenchmarkConsistentHashPartitioner(virtualNodes: 1000)
                
                // Initial partitioning
                var partitions = try await partitioner.partition(
                    vectors,
                    partitionCount: initialPartitions
                )
                
                // Add new node and rebalance
                let newPartitions = try await partitioner.rebalance(
                    partitions: partitions,
                    newPartitionCount: initialPartitions + 1
                )
                
                blackHole(newPartitions)
            }
            
            // Node removal scenario
            benchmark(name: "rebalance_remove_node") {
                let vectors = generateVectors(count: 100_000, dimensions: dimensions)
                let partitioner = BenchmarkConsistentHashPartitioner(virtualNodes: 1000)
                
                // Initial partitioning
                var partitions = try await partitioner.partition(
                    vectors,
                    partitionCount: initialPartitions
                )
                
                // Remove node and rebalance
                let newPartitions = try await partitioner.rebalance(
                    partitions: partitions,
                    newPartitionCount: initialPartitions - 1
                )
                
                blackHole(newPartitions)
            }
            
            // Load-based rebalancing
            benchmark(name: "rebalance_load_based") {
                let vectors = generateSkewedVectors(
                    count: 100_000,
                    dimensions: dimensions,
                    skewFactor: 0.8
                )
                let partitioner = LoadAwarePartitioner()
                
                // Initial partitioning
                var partitions = try await partitioner.partition(
                    vectors,
                    partitionCount: initialPartitions
                )
                
                // Measure load and rebalance
                let loads = partitions.map { $0.count }
                let newPartitions = try await partitioner.rebalanceByLoad(
                    partitions: partitions,
                    targetLoadVariance: 0.1
                )
                
                blackHole(newPartitions)
            }
            
            // Incremental rebalancing
            benchmark(name: "rebalance_incremental") {
                let vectors = generateVectors(count: 100_000, dimensions: dimensions)
                let partitioner = IncrementalPartitioner(
                    basePartitions: initialPartitions
                )
                
                var partitions = try await partitioner.partition(vectors)
                
                // Add vectors incrementally and rebalance
                for _ in 0..<10 {
                    let newVectors = generateVectors(count: 10_000, dimensions: dimensions)
                    partitions = try await partitioner.addAndRebalance(
                        newVectors: newVectors,
                        existingPartitions: partitions
                    )
                }
                
                blackHole(partitions)
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Cross-Partition Query Benchmarks
    
    private func runCrossPartitionQueryBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Cross-Partition Queries",
            description: "Benchmark queries across multiple partitions"
        ) {
            let dimensions = 128
            let vectorsPerPartition = 10_000
            let partitionCounts = [2, 4, 8, 16]
            let k = 100
            
            for partitions in partitionCounts {
                // Single partition query (baseline)
                benchmark(name: "query_single_partition_\(partitions)") {
                    let index = createPartitionedIndex(
                        partitions: 1,
                        vectorsPerPartition: vectorsPerPartition * partitions,
                        dimensions: dimensions
                    )
                    
                    let query = generateVector(dimensions: dimensions)
                    let results = try await index.search(
                        query: query,
                        k: k,
                        partitions: [0]
                    )
                    blackHole(results)
                }
                
                // All partitions query
                benchmark(name: "query_all_partitions_\(partitions)") {
                    let index = createPartitionedIndex(
                        partitions: partitions,
                        vectorsPerPartition: vectorsPerPartition,
                        dimensions: dimensions
                    )
                    
                    let query = generateVector(dimensions: dimensions)
                    let results = try await index.search(
                        query: query,
                        k: k,
                        partitions: Array(0..<partitions)
                    )
                    blackHole(results)
                }
                
                // Parallel partition query
                benchmark(name: "query_parallel_partitions_\(partitions)") {
                    let index = createPartitionedIndex(
                        partitions: partitions,
                        vectorsPerPartition: vectorsPerPartition,
                        dimensions: dimensions
                    )
                    
                    let query = generateVector(dimensions: dimensions)
                    let results = try await index.parallelSearch(
                        query: query,
                        k: k,
                        partitions: Array(0..<partitions)
                    )
                    blackHole(results)
                }
                
                // Smart routing query
                benchmark(name: "query_smart_routing_\(partitions)") {
                    let index = createPartitionedIndex(
                        partitions: partitions,
                        vectorsPerPartition: vectorsPerPartition,
                        dimensions: dimensions,
                        enableSmartRouting: true
                    )
                    
                    let query = generateVector(dimensions: dimensions)
                    let results = try await index.smartSearch(
                        query: query,
                        k: k,
                        maxPartitions: partitions / 2
                    )
                    blackHole(results)
                }
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Partition Scaling Benchmarks
    
    private func runPartitionScalingBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Partition Scaling",
            description: "Benchmark scaling behavior with partition count"
        ) {
            let dimensions = 128
            let totalVectors = 1_000_000
            
            // Weak scaling (constant load per partition)
            benchmark(name: "weak_scaling") {
                let partitionCounts = [1, 2, 4, 8, 16]
                var times: [TimeInterval] = []
                
                for partitions in partitionCounts {
                    let vectorsPerPartition = totalVectors / partitions
                    let index = createPartitionedIndex(
                        partitions: partitions,
                        vectorsPerPartition: vectorsPerPartition,
                        dimensions: dimensions
                    )
                    
                    let start = Date()
                    // Perform operations
                    for _ in 0..<100 {
                        let query = generateVector(dimensions: dimensions)
                        _ = try await index.search(query: query, k: 10)
                    }
                    let elapsed = Date().timeIntervalSince(start)
                    times.append(elapsed)
                }
                
                blackHole(times)
            }
            
            // Strong scaling (constant total load)
            benchmark(name: "strong_scaling") {
                let partitionCounts = [1, 2, 4, 8, 16]
                var times: [TimeInterval] = []
                
                for partitions in partitionCounts {
                    let index = createPartitionedIndex(
                        partitions: partitions,
                        vectorsPerPartition: totalVectors / partitions,
                        dimensions: dimensions
                    )
                    
                    let start = Date()
                    // Perform operations
                    for _ in 0..<100 {
                        let query = generateVector(dimensions: dimensions)
                        _ = try await index.parallelSearch(
                            query: query,
                            k: 10,
                            partitions: Array(0..<partitions)
                        )
                    }
                    let elapsed = Date().timeIntervalSince(start)
                    times.append(elapsed)
                }
                
                blackHole(times)
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Helper Functions
    
    private func generatePartitionTestData(
        count: Int,
        dimensions: Int
    ) -> PartitionTestData {
        let vectors = generateVectors(count: count, dimensions: dimensions)
        let metadata = (0..<count).map { i in
            TestMetadata(
                id: i,
                category: "cat_\(i % 10)",
                timestamp: Date(),
                tags: ["test"]
            )
        }
        
        return PartitionTestData(
            vectors: vectors,
            metadata: metadata,
            dimensions: dimensions,
            count: count
        )
    }
    
    private func generateVectors(count: Int, dimensions: Int) -> [[Float]] {
        return (0..<count).map { _ in
            generateVector(dimensions: dimensions)
        }
    }
    
    private func generateVector(dimensions: Int) -> [Float] {
        return (0..<dimensions).map { _ in Float.random(in: -1...1) }
    }
    
    private func generateSkewedVectors(
        count: Int,
        dimensions: Int,
        skewFactor: Float
    ) -> [[Float]] {
        // Generate vectors with skewed distribution
        let hotspotCount = Int(Float(count) * (1.0 - skewFactor))
        let normalCount = count - hotspotCount
        
        var vectors: [[Float]] = []
        
        // Hot spot vectors (clustered)
        let center = generateVector(dimensions: dimensions)
        for _ in 0..<hotspotCount {
            let vector = center.map { $0 + Float.random(in: -0.1...0.1) }
            vectors.append(vector)
        }
        
        // Normal vectors (distributed)
        for _ in 0..<normalCount {
            vectors.append(generateVector(dimensions: dimensions))
        }
        
        return vectors.shuffled()
    }
    
    private func createPartitionedIndex(
        partitions: Int,
        vectorsPerPartition: Int,
        dimensions: Int,
        enableSmartRouting: Bool = false
    ) -> MockPartitionedIndex {
        return MockPartitionedIndex(
            partitions: partitions,
            vectorsPerPartition: vectorsPerPartition,
            dimensions: dimensions,
            enableSmartRouting: enableSmartRouting
        )
    }
}

// MARK: - Mock Types for Benchmarking

private actor MockPartitionedIndex {
    let partitions: Int
    let vectorsPerPartition: Int
    let dimensions: Int
    let enableSmartRouting: Bool
    
    init(
        partitions: Int,
        vectorsPerPartition: Int,
        dimensions: Int,
        enableSmartRouting: Bool
    ) {
        self.partitions = partitions
        self.vectorsPerPartition = vectorsPerPartition
        self.dimensions = dimensions
        self.enableSmartRouting = enableSmartRouting
    }
    
    func search(
        query: [Float],
        k: Int,
        partitions: [Int]? = nil
    ) async throws -> [SearchResult<TestMetadata>] {
        // Simulate search delay
        try await Task.sleep(nanoseconds: UInt64(vectorsPerPartition) * 100)
        return []
    }
    
    func parallelSearch(
        query: [Float],
        k: Int,
        partitions: [Int]
    ) async throws -> [SearchResult<TestMetadata>] {
        // Simulate parallel search
        try await Task.sleep(nanoseconds: UInt64(vectorsPerPartition) * 50)
        return []
    }
    
    func smartSearch(
        query: [Float],
        k: Int,
        maxPartitions: Int
    ) async throws -> [SearchResult<TestMetadata>] {
        // Simulate smart routing search
        try await Task.sleep(nanoseconds: UInt64(vectorsPerPartition) * 30)
        return []
    }
}

private struct BenchmarkDistributedSearchResult {
    let id: String
    let distance: Float
}

// MARK: - Mock Partitioners

private actor BenchmarkHashPartitioner {
    let partitionCount: Int
    
    init(partitionCount: Int) {
        self.partitionCount = partitionCount
    }
    
    func partition(_ vectors: [[Float]]) async throws -> [[Int]] {
        var partitions = Array(repeating: [Int](), count: partitionCount)
        for (index, vector) in vectors.enumerated() {
            let hash = vector.hashValue
            let partition = abs(hash) % partitionCount
            partitions[partition].append(index)
        }
        return partitions
    }
}

private actor BenchmarkRangePartitioner {
    let partitionCount: Int
    
    init(partitionCount: Int) {
        self.partitionCount = partitionCount
    }
    
    func partition(_ vectors: [[Float]]) async throws -> [[Int]] {
        let rangeSize = vectors.count / partitionCount
        var partitions = Array(repeating: [Int](), count: partitionCount)
        
        for (index, _) in vectors.enumerated() {
            let partition = min(index / rangeSize, partitionCount - 1)
            partitions[partition].append(index)
        }
        return partitions
    }
}

private actor ClusterPartitioner {
    let partitionCount: Int
    let dimensions: Int
    
    init(partitionCount: Int, dimensions: Int) {
        self.partitionCount = partitionCount
        self.dimensions = dimensions
    }
    
    func partition(_ vectors: [[Float]]) async throws -> [[Int]] {
        // Simplified k-means clustering simulation
        var partitions = Array(repeating: [Int](), count: partitionCount)
        for (index, _) in vectors.enumerated() {
            let partition = index % partitionCount
            partitions[partition].append(index)
        }
        return partitions
    }
}

private actor BenchmarkConsistentHashPartitioner {
    let virtualNodes: Int
    
    init(virtualNodes: Int) {
        self.virtualNodes = virtualNodes
    }
    
    func partition(_ vectors: [[Float]], partitionCount: Int? = nil) async throws -> [[Int]] {
        let count = partitionCount ?? 8
        var partitions = Array(repeating: [Int](), count: count)
        
        for (index, vector) in vectors.enumerated() {
            let hash = vector.hashValue
            let vnode = abs(hash) % virtualNodes
            let partition = vnode % count
            partitions[partition].append(index)
        }
        return partitions
    }
    
    func rebalance(
        partitions: [[Int]],
        newPartitionCount: Int
    ) async throws -> [[Int]] {
        var allIndices: [Int] = []
        for partition in partitions {
            allIndices.append(contentsOf: partition)
        }
        
        var newPartitions = Array(repeating: [Int](), count: newPartitionCount)
        for index in allIndices {
            let partition = index % newPartitionCount
            newPartitions[partition].append(index)
        }
        
        return newPartitions
    }
}

private actor LoadAwarePartitioner {
    func partition(_ vectors: [[Float]], partitionCount: Int) async throws -> [[Int]] {
        var partitions = Array(repeating: [Int](), count: partitionCount)
        var loads = Array(repeating: 0, count: partitionCount)
        
        for (index, _) in vectors.enumerated() {
            let minLoadPartition = loads.enumerated().min(by: { $0.element < $1.element })!.offset
            partitions[minLoadPartition].append(index)
            loads[minLoadPartition] += 1
        }
        
        return partitions
    }
    
    func rebalanceByLoad(
        partitions: [[Int]],
        targetLoadVariance: Double
    ) async throws -> [[Int]] {
        // Simulate load-based rebalancing
        return partitions
    }
}

private actor IncrementalPartitioner {
    let basePartitions: Int
    
    init(basePartitions: Int) {
        self.basePartitions = basePartitions
    }
    
    func partition(_ vectors: [[Float]]) async throws -> [[Int]] {
        var partitions = Array(repeating: [Int](), count: basePartitions)
        for (index, _) in vectors.enumerated() {
            let partition = index % basePartitions
            partitions[partition].append(index)
        }
        return partitions
    }
    
    func addAndRebalance(
        newVectors: [[Float]],
        existingPartitions: [[Int]]
    ) async throws -> [[Int]] {
        // Simulate incremental rebalancing
        var partitions = existingPartitions
        let startIndex = existingPartitions.flatMap { $0 }.count
        
        for (offset, _) in newVectors.enumerated() {
            let partition = offset % basePartitions
            partitions[partition].append(startIndex + offset)
        }
        
        return partitions
    }
}
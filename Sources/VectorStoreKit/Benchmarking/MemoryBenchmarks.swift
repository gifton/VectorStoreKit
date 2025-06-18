// VectorStoreKit: Memory Benchmarks
//
// Memory usage and allocation pattern benchmarks

import Foundation
import simd
#if canImport(Darwin)
import Darwin
#endif

/// Benchmarks for memory usage patterns
public struct MemoryBenchmarks {
    
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
    
    public struct MemoryProfile {
        public let operation: String
        public let allocations: Int
        public let deallocations: Int
        public let peakMemory: Int
        public let averageMemory: Int
        public let fragmentationRatio: Double
        public let cacheEfficiency: Double
        
        public var leakDetected: Bool {
            allocations > deallocations + 100 // Allow small discrepancy
        }
    }
    
    // MARK: - Main Benchmark Suites
    
    /// Run all memory benchmarks
    public func runAll() async throws -> [String: BenchmarkFramework.Statistics] {
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        // Allocation pattern benchmarks
        results.merge(try await runAllocationPatternBenchmarks()) { _, new in new }
        
        // Memory pressure benchmarks
        results.merge(try await runMemoryPressureBenchmarks()) { _, new in new }
        
        // Cache efficiency benchmarks
        results.merge(try await runCacheEfficiencyBenchmarks()) { _, new in new }
        
        // Memory leak detection benchmarks
        results.merge(try await runMemoryLeakBenchmarks()) { _, new in new }
        
        return results
    }
    
    // MARK: - Allocation Pattern Benchmarks
    
    private func runAllocationPatternBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Memory Allocation Patterns",
            description: "Benchmark different allocation strategies"
        ) {
            let dimensions = 512
            let vectorCount = 10_000
            
            // Bulk allocation
            benchmark(
                name: "bulk_allocation",
                setUp: {
                    await metrics.startCollection(name: "bulk_allocation")
                },
                tearDown: {
                    _ = await metrics.stopCollection()
                }
            ) {
                // Allocate all at once
                let vectors = [[Float]](
                    unsafeUninitializedCapacity: vectorCount
                ) { buffer, initializedCount in
                    for i in 0..<vectorCount {
                        buffer[i] = [Float](repeating: 0, count: dimensions)
                    }
                    initializedCount = vectorCount
                }
                
                // Fill with data
                for i in 0..<vectorCount {
                    for j in 0..<dimensions {
                        vectors[i][j] = Float.random(in: -1...1)
                    }
                }
                
                blackHole(vectors)
            }
            
            // Incremental allocation
            benchmark(
                name: "incremental_allocation",
                setUp: {
                    await metrics.startCollection(name: "incremental_allocation")
                },
                tearDown: {
                    _ = await metrics.stopCollection()
                }
            ) {
                var vectors: [[Float]] = []
                vectors.reserveCapacity(vectorCount)
                
                for i in 0..<vectorCount {
                    let vector = (0..<dimensions).map { _ in Float.random(in: -1...1) }
                    vectors.append(vector)
                    
                    // Record memory periodically
                    if i % 1000 == 0 {
                        await metrics.recordMemory(
                            name: "incremental",
                            bytes: getMemoryUsage()
                        )
                    }
                }
                
                blackHole(vectors)
            }
            
            // Pool-based allocation
            benchmark(
                name: "pool_allocation",
                setUp: {
                    await metrics.startCollection(name: "pool_allocation")
                },
                tearDown: {
                    _ = await metrics.stopCollection()
                }
            ) {
                let pool = VectorPool(capacity: 1000, dimensions: dimensions)
                var allocated: [VectorPool.Handle] = []
                
                // Allocate and deallocate in patterns
                for i in 0..<vectorCount {
                    if i > 0 && i % 100 == 0 {
                        // Return some vectors to pool
                        for _ in 0..<50 {
                            if let handle = allocated.popLast() {
                                pool.release(handle)
                            }
                        }
                    }
                    
                    if let handle = pool.acquire() {
                        allocated.append(handle)
                        
                        // Fill with data
                        for j in 0..<dimensions {
                            handle.vector[j] = Float.random(in: -1...1)
                        }
                    }
                }
                
                blackHole(allocated)
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Memory Pressure Benchmarks
    
    private func runMemoryPressureBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Memory Pressure",
            description: "Behavior under memory constraints"
        ) {
            let dimensions = 256
            
            // Growing memory usage
            benchmark(name: "memory_growth") {
                var indices: [Any] = []
                let growthSteps = 10
                let vectorsPerStep = 10_000
                
                for step in 0..<growthSteps {
                    // Create index with current size
                    let index = try await createIndex(
                        size: (step + 1) * vectorsPerStep,
                        dimensions: dimensions
                    )
                    indices.append(index)
                    
                    let memory = getMemoryUsage()
                    await metrics.recordMemory(
                        name: "growth_step_\(step)",
                        bytes: memory
                    )
                    
                    // Perform operations to stress memory
                    for _ in 0..<100 {
                        let query = generateVector(dimensions: dimensions)
                        _ = try? await index.search(query: query, k: 10)
                    }
                }
                
                blackHole(indices)
            }
            
            // Memory spike handling
            benchmark(name: "memory_spike") {
                let baseIndex = try await createIndex(size: 10_000, dimensions: dimensions)
                
                // Create temporary spike
                let spikeData = (0..<100_000).map { _ in
                    [Float](repeating: Float.random(in: -1...1), count: dimensions)
                }
                
                let memoryBeforeSpike = getMemoryUsage()
                await metrics.recordMemory(name: "before_spike", bytes: memoryBeforeSpike)
                
                // Process spike data
                for batch in spikeData.chunked(into: 1000) {
                    for vector in batch {
                        _ = try? await baseIndex.search(
                            query: SIMD32<Float>(vector),
                            k: 10
                        )
                    }
                }
                
                let memoryDuringSpike = getMemoryUsage()
                await metrics.recordMemory(name: "during_spike", bytes: memoryDuringSpike)
                
                // Clear spike data
                blackHole(spikeData)
                
                // Allow cleanup
                for _ in 0..<10 {
                    try await Task.sleep(nanoseconds: 100_000_000) // 0.1s
                }
                
                let memoryAfterSpike = getMemoryUsage()
                await metrics.recordMemory(name: "after_spike", bytes: memoryAfterSpike)
                
                blackHole(baseIndex)
            }
            
            // Out-of-memory simulation
            benchmark(name: "oom_handling") {
                var indices: [Any] = []
                let maxIndices = 100
                
                for i in 0..<maxIndices {
                    do {
                        let index = try await createIndex(
                            size: 100_000,
                            dimensions: dimensions
                        )
                        indices.append(index)
                        
                        if i % 10 == 0 {
                            let memory = getMemoryUsage()
                            await metrics.recordMemory(
                                name: "oom_step_\(i)",
                                bytes: memory
                            )
                        }
                    } catch {
                        // Expected to fail at some point
                        await metrics.recordCustom(
                            name: "oom_failed_at",
                            value: Double(i),
                            unit: "indices"
                        )
                        break
                    }
                }
                
                blackHole(indices)
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Cache Efficiency Benchmarks
    
    private func runCacheEfficiencyBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Cache Efficiency",
            description: "Memory access patterns and cache usage"
        ) {
            let dimensions = 512
            let vectorCount = 100_000
            
            // Sequential access pattern
            benchmark(name: "sequential_access") {
                let vectors = generateVectorArray(count: vectorCount, dimensions: dimensions)
                var sum: Float = 0
                
                // Access vectors sequentially
                for vector in vectors {
                    for value in vector {
                        sum += value
                    }
                }
                
                blackHole(sum)
            }
            
            // Random access pattern
            benchmark(name: "random_access") {
                let vectors = generateVectorArray(count: vectorCount, dimensions: dimensions)
                let indices = (0..<10_000).map { _ in Int.random(in: 0..<vectorCount) }
                var sum: Float = 0
                
                // Access vectors randomly
                for index in indices {
                    for value in vectors[index] {
                        sum += value
                    }
                }
                
                blackHole(sum)
            }
            
            // Strided access pattern
            benchmark(name: "strided_access") {
                let vectors = generateVectorArray(count: vectorCount, dimensions: dimensions)
                let stride = 64 // Cache line size
                var sum: Float = 0
                
                // Access with stride
                for i in Swift.stride(from: 0, to: vectorCount, by: stride) {
                    for value in vectors[i] {
                        sum += value
                    }
                }
                
                blackHole(sum)
            }
            
            // Tiled access pattern
            benchmark(name: "tiled_access") {
                let vectors = generateVectorArray(count: vectorCount, dimensions: dimensions)
                let tileSize = 8
                var sum: Float = 0
                
                // Access in tiles for better cache usage
                for tileStart in Swift.stride(from: 0, to: vectorCount, by: tileSize) {
                    let tileEnd = min(tileStart + tileSize, vectorCount)
                    
                    // Process tile
                    for i in tileStart..<tileEnd {
                        for j in 0..<dimensions {
                            sum += vectors[i][j]
                        }
                    }
                }
                
                blackHole(sum)
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Memory Leak Detection Benchmarks
    
    private func runMemoryLeakBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Memory Leak Detection",
            description: "Check for memory leaks in operations"
        ) {
            let dimensions = 128
            let iterations = 100
            
            // Index creation/destruction cycles
            benchmark(name: "index_lifecycle_leak_check") {
                let initialMemory = getMemoryUsage()
                
                for i in 0..<iterations {
                    autoreleasepool {
                        let index = try? HNSWIndex<SIMD32<Float>, TestMetadata>(
                            configuration: .init(maxConnections: 16)
                        )
                        
                        // Add some data
                        for j in 0..<1000 {
                            let vector = generateVector(dimensions: dimensions)
                            let entry = VectorEntry(
                                id: "leak_test_\(i)_\(j)",
                                vector: vector,
                                metadata: TestMetadata(
                                    id: i * 1000 + j,
                                    category: "leak_test",
                                    timestamp: Date(),
                                    tags: []
                                )
                            )
                            _ = try? await index?.insert(entry)
                        }
                        
                        // Force deallocation
                        _ = index
                    }
                    
                    if i % 10 == 0 {
                        let currentMemory = getMemoryUsage()
                        let growth = currentMemory - initialMemory
                        await metrics.recordCustom(
                            name: "memory_growth",
                            value: Double(growth),
                            unit: "bytes"
                        )
                    }
                }
                
                let finalMemory = getMemoryUsage()
                let totalGrowth = finalMemory - initialMemory
                await metrics.recordCustom(
                    name: "total_memory_growth",
                    value: Double(totalGrowth),
                    unit: "bytes"
                )
            }
            
            // Search operation leak check
            benchmark(name: "search_leak_check") {
                let index = try await createIndex(size: 10_000, dimensions: dimensions)
                let queries = (0..<1000).map { _ in generateVector(dimensions: dimensions) }
                
                let initialMemory = getMemoryUsage()
                
                for i in 0..<iterations {
                    for query in queries {
                        autoreleasepool {
                            _ = try? await index.search(query: query, k: 100)
                        }
                    }
                    
                    if i % 10 == 0 {
                        let currentMemory = getMemoryUsage()
                        let growth = currentMemory - initialMemory
                        await metrics.recordCustom(
                            name: "search_memory_growth",
                            value: Double(growth),
                            unit: "bytes"
                        )
                    }
                }
            }
            
            // Metadata leak check
            benchmark(name: "metadata_leak_check") {
                let index = try await createIndex(size: 1000, dimensions: dimensions)
                let initialMemory = getMemoryUsage()
                
                for i in 0..<iterations {
                    // Update metadata repeatedly
                    for j in 0..<100 {
                        let id = "vec_\(j)"
                        let newMetadata = TestMetadata(
                            id: j,
                            category: "updated_\(i)",
                            timestamp: Date(),
                            tags: Array(repeating: "tag_\(i)", count: 100) // Large metadata
                        )
                        _ = try? await index.updateMetadata(id: id, metadata: newMetadata)
                    }
                    
                    if i % 10 == 0 {
                        let currentMemory = getMemoryUsage()
                        let growth = currentMemory - initialMemory
                        await metrics.recordCustom(
                            name: "metadata_memory_growth",
                            value: Double(growth),
                            unit: "bytes"
                        )
                    }
                }
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Helper Functions
    
    private func createIndex(
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
                    tags: ["memory_test"]
                )
            )
            _ = try await index.insert(entry)
        }
        
        return index
    }
    
    private func generateVector(dimensions: Int) -> SIMD32<Float> {
        var vector = SIMD32<Float>()
        for i in 0..<min(32, dimensions) {
            vector[i] = Float.random(in: -1...1)
        }
        return vector
    }
    
    private func generateVectorArray(count: Int, dimensions: Int) -> [[Float]] {
        return (0..<count).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }
    }
    
    private func getMemoryUsage() -> Int {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }
        
        return result == KERN_SUCCESS ? Int(info.resident_size) : 0
        #else
        return 0
        #endif
    }
}

// MARK: - Helper Types

private class VectorPool {
    struct Handle {
        let vector: UnsafeMutableBufferPointer<Float>
        fileprivate let index: Int
    }
    
    private let capacity: Int
    private let dimensions: Int
    private var pool: [UnsafeMutableBufferPointer<Float>]
    private var available: Set<Int>
    
    init(capacity: Int, dimensions: Int) {
        self.capacity = capacity
        self.dimensions = dimensions
        self.pool = []
        self.available = Set(0..<capacity)
        
        // Pre-allocate vectors
        for _ in 0..<capacity {
            let buffer = UnsafeMutableBufferPointer<Float>.allocate(capacity: dimensions)
            pool.append(buffer)
        }
    }
    
    deinit {
        for buffer in pool {
            buffer.deallocate()
        }
    }
    
    func acquire() -> Handle? {
        guard let index = available.first else { return nil }
        available.remove(index)
        return Handle(vector: pool[index], index: index)
    }
    
    func release(_ handle: Handle) {
        available.insert(handle.index)
    }
}

// MARK: - Array Extension
// Using the chunked extension from MetalDistanceCompute.swift
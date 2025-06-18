// VectorStoreKit: Cache Benchmarks
//
// Performance benchmarks for caching layers

import Foundation

/// Benchmarks for cache implementations
public struct CacheBenchmarks {
    
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
    
    struct CacheTestData {
        let key: String
        let value: Data
        let size: Int
        
        init(id: Int, size: Int = 1024) {
            self.key = "key_\(id)"
            self.value = Data(repeating: UInt8(id % 256), count: size)
            self.size = size
        }
    }
    
    // MARK: - Main Benchmark Suites
    
    /// Run all cache benchmarks
    public func runAll() async throws -> [String: BenchmarkFramework.Statistics] {
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        // Cache implementation benchmarks
        results.merge(try await runCacheImplementationBenchmarks()) { _, new in new }
        
        // Hit rate benchmarks
        results.merge(try await runHitRateBenchmarks()) { _, new in new }
        
        // Eviction policy benchmarks
        results.merge(try await runEvictionBenchmarks()) { _, new in new }
        
        // Concurrent access benchmarks
        results.merge(try await runConcurrentAccessBenchmarks()) { _, new in new }
        
        return results
    }
    
    // MARK: - Cache Implementation Benchmarks
    
    private func runCacheImplementationBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let cacheSizes = [100, 1_000, 10_000]
        let itemSizes = [1_024, 10_240, 102_400] // 1KB, 10KB, 100KB
        
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        for cacheSize in cacheSizes {
            for itemSize in itemSizes {
                let suite = benchmarkSuite(
                    name: "Cache Implementations (size: \(cacheSize), item: \(formatBytes(itemSize)))",
                    description: "Compare different cache implementations"
                ) {
                    // LRU Cache
                    benchmark(name: "lru_cache_\(cacheSize)_\(itemSize)") {
                        try await benchmarkLRUCache(
                            capacity: cacheSize,
                            itemSize: itemSize
                        )
                    }
                    
                    // LFU Cache
                    benchmark(name: "lfu_cache_\(cacheSize)_\(itemSize)") {
                        try await benchmarkLFUCache(
                            capacity: cacheSize,
                            itemSize: itemSize
                        )
                    }
                    
                    // FIFO Cache
                    benchmark(name: "fifo_cache_\(cacheSize)_\(itemSize)") {
                        try await benchmarkFIFOCache(
                            capacity: cacheSize,
                            itemSize: itemSize
                        )
                    }
                    
                    // No-op Cache (baseline)
                    benchmark(name: "noop_cache_\(cacheSize)_\(itemSize)") {
                        try await benchmarkNoOpCache(
                            capacity: cacheSize,
                            itemSize: itemSize
                        )
                    }
                }
                
                let suiteResults = try await framework.run(suite: suite)
                results.merge(suiteResults) { _, new in new }
            }
        }
        
        return results
    }
    
    // MARK: - Hit Rate Benchmarks
    
    private func runHitRateBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Cache Hit Rates",
            description: "Measure cache hit rates under different access patterns"
        ) {
            let cacheSize = 1_000
            let totalAccesses = 10_000
            
            // Sequential access pattern
            benchmark(name: "sequential_access_pattern") {
                let cache = LRUCache<String, Data>(
                    configuration: LRUCacheConfiguration(maxSize: cacheSize)
                )
                
                var hits = 0
                for i in 0..<totalAccesses {
                    let key = "key_\(i % (cacheSize * 2))"
                    
                    if await cache.get(key) != nil {
                        hits += 1
                    } else {
                        let data = CacheTestData(id: i)
                        await cache.set(key, data.value)
                    }
                }
                
                let hitRate = Double(hits) / Double(totalAccesses)
                await metrics.recordCacheHitRate(
                    name: "sequential",
                    hits: hits,
                    total: totalAccesses
                )
                blackHole(hitRate)
            }
            
            // Random access pattern
            benchmark(name: "random_access_pattern") {
                let cache = LRUCache<String, Data>(
                    configuration: LRUCacheConfiguration(maxSize: cacheSize)
                )
                
                var hits = 0
                for _ in 0..<totalAccesses {
                    let id = Int.random(in: 0..<(cacheSize * 3))
                    let key = "key_\(id)"
                    
                    if await cache.get(key) != nil {
                        hits += 1
                    } else {
                        let data = CacheTestData(id: id)
                        await cache.set(key, data.value)
                    }
                }
                
                let hitRate = Double(hits) / Double(totalAccesses)
                await metrics.recordCacheHitRate(
                    name: "random",
                    hits: hits,
                    total: totalAccesses
                )
                blackHole(hitRate)
            }
            
            // Zipfian distribution (realistic)
            benchmark(name: "zipfian_access_pattern") {
                let cache = LRUCache<String, Data>(
                    configuration: LRUCacheConfiguration(maxSize: cacheSize)
                )
                
                var hits = 0
                for _ in 0..<totalAccesses {
                    let id = generateZipfian(n: cacheSize * 2)
                    let key = "key_\(id)"
                    
                    if await cache.get(key) != nil {
                        hits += 1
                    } else {
                        let data = CacheTestData(id: id)
                        await cache.set(key, data.value)
                    }
                }
                
                let hitRate = Double(hits) / Double(totalAccesses)
                await metrics.recordCacheHitRate(
                    name: "zipfian",
                    hits: hits,
                    total: totalAccesses
                )
                blackHole(hitRate)
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Eviction Benchmarks
    
    private func runEvictionBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Cache Eviction",
            description: "Benchmark eviction performance"
        ) {
            let cacheSize = 1_000
            let overflowFactor = 2 // Insert 2x cache capacity
            
            // LRU eviction
            benchmark(name: "lru_eviction") {
                let cache = LRUCache<String, Data>(
                    configuration: LRUCacheConfiguration(maxSize: cacheSize)
                )
                
                for i in 0..<(cacheSize * overflowFactor) {
                    let data = CacheTestData(id: i)
                    await cache.set(data.key, data.value)
                }
                
                let size = await cache.currentSize()
                blackHole(size)
            }
            
            // LFU eviction
            benchmark(name: "lfu_eviction") {
                let cache = LFUCache<String, Data>(
                    configuration: LFUCacheConfiguration(maxSize: cacheSize)
                )
                
                for i in 0..<(cacheSize * overflowFactor) {
                    let data = CacheTestData(id: i)
                    await cache.set(data.key, data.value)
                }
                
                let size = await cache.currentSize()
                blackHole(size)
            }
            
            // FIFO eviction
            benchmark(name: "fifo_eviction") {
                let cache = FIFOCache<String, Data>(
                    configuration: FIFOCacheConfiguration(maxSize: cacheSize)
                )
                
                for i in 0..<(cacheSize * overflowFactor) {
                    let data = CacheTestData(id: i)
                    await cache.set(data.key, data.value)
                }
                
                let size = await cache.currentSize()
                blackHole(size)
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Concurrent Access Benchmarks
    
    private func runConcurrentAccessBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Concurrent Cache Access",
            description: "Benchmark cache performance under concurrent load"
        ) {
            let cacheSize = 10_000
            let operationsPerTask = 1_000
            
            // Create benchmarks for each concurrency level
            // LRU with 1 thread
            benchmark(name: "lru_concurrent_1") {
                let cache = LRUCache<String, Data>(
                    configuration: LRUCacheConfiguration(maxSize: cacheSize)
                )
                
                await withTaskGroup(of: Void.self) { group in
                    group.addTask {
                        for i in 0..<operationsPerTask {
                            let id = i % (cacheSize * 2)
                            let key = "key_\(id)"
                            
                            // Mix of reads and writes
                            if i % 3 == 0 {
                                let data = CacheTestData(id: id)
                                await cache.set(key, data.value)
                            } else {
                                _ = await cache.get(key)
                            }
                        }
                    }
                }
                
                let finalSize = await cache.currentSize()
                blackHole(finalSize)
            }
            
            // LRU with 4 threads
            benchmark(name: "lru_concurrent_4") {
                let cache = LRUCache<String, Data>(
                    configuration: LRUCacheConfiguration(maxSize: cacheSize)
                )
                
                await withTaskGroup(of: Void.self) { group in
                    for taskId in 0..<4 {
                        group.addTask {
                            for i in 0..<operationsPerTask {
                                let id = (taskId * operationsPerTask + i) % (cacheSize * 2)
                                let key = "key_\(id)"
                                
                                // Mix of reads and writes
                                if i % 3 == 0 {
                                    let data = CacheTestData(id: id)
                                    await cache.set(key, data.value)
                                } else {
                                    _ = await cache.get(key)
                                }
                            }
                        }
                    }
                }
                
                let finalSize = await cache.currentSize()
                blackHole(finalSize)
            }
            
            // LRU with 8 threads
            benchmark(name: "lru_concurrent_8") {
                let cache = LRUCache<String, Data>(
                    configuration: LRUCacheConfiguration(maxSize: cacheSize)
                )
                
                await withTaskGroup(of: Void.self) { group in
                    for taskId in 0..<8 {
                        group.addTask {
                            for i in 0..<operationsPerTask {
                                let id = (taskId * operationsPerTask + i) % (cacheSize * 2)
                                let key = "key_\(id)"
                                
                                // Mix of reads and writes
                                if i % 3 == 0 {
                                    let data = CacheTestData(id: id)
                                    await cache.set(key, data.value)
                                } else {
                                    _ = await cache.get(key)
                                }
                            }
                        }
                    }
                }
                
                let finalSize = await cache.currentSize()
                blackHole(finalSize)
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Helper Functions
    
    private func benchmarkLRUCache(capacity: Int, itemSize: Int) async throws {
        let cache = LRUCache<String, Data>(
            configuration: LRUCacheConfiguration(maxSize: capacity)
        )
        
        // Warm up cache
        for i in 0..<capacity {
            let data = CacheTestData(id: i, size: itemSize)
            await cache.set(data.key, data.value)
        }
        
        // Benchmark mixed operations
        for i in 0..<capacity {
            if i % 2 == 0 {
                _ = await cache.get("key_\(i)")
            } else {
                let data = CacheTestData(id: capacity + i, size: itemSize)
                await cache.set(data.key, data.value)
            }
        }
    }
    
    private func benchmarkLFUCache(capacity: Int, itemSize: Int) async throws {
        let cache = LFUCache<String, Data>(
            configuration: LFUCacheConfiguration(maxSize: capacity)
        )
        
        // Similar implementation to LRU
        for i in 0..<capacity {
            let data = CacheTestData(id: i, size: itemSize)
            await cache.set(data.key, data.value)
        }
        
        for i in 0..<capacity {
            if i % 2 == 0 {
                _ = await cache.get("key_\(i)")
            } else {
                let data = CacheTestData(id: capacity + i, size: itemSize)
                await cache.set(data.key, data.value)
            }
        }
    }
    
    private func benchmarkFIFOCache(capacity: Int, itemSize: Int) async throws {
        let cache = FIFOCache<String, Data>(
            configuration: FIFOCacheConfiguration(maxSize: capacity)
        )
        
        for i in 0..<capacity {
            let data = CacheTestData(id: i, size: itemSize)
            await cache.set(data.key, data.value)
        }
        
        for i in 0..<capacity {
            if i % 2 == 0 {
                _ = await cache.get("key_\(i)")
            } else {
                let data = CacheTestData(id: capacity + i, size: itemSize)
                await cache.set(data.key, data.value)
            }
        }
    }
    
    private func benchmarkNoOpCache(capacity: Int, itemSize: Int) async throws {
        let cache = NoOpCache<String, Data>(
            configuration: NoOpCacheConfiguration()
        )
        
        for i in 0..<capacity {
            let data = CacheTestData(id: i, size: itemSize)
            await cache.set(data.key, data.value)
            _ = await cache.get(data.key)
        }
    }
    
    private func generateZipfian(n: Int, alpha: Double = 1.0) -> Int {
        // Simple Zipfian distribution generator
        let u = Double.random(in: 0...1)
        return Int(pow(Double(n + 1), 1.0 - u) - 1)
    }
    
    private func formatBytes(_ bytes: Int) -> String {
        let units = ["B", "KB", "MB", "GB"]
        var size = Double(bytes)
        var unitIndex = 0
        
        while size >= 1024 && unitIndex < units.count - 1 {
            size /= 1024
            unitIndex += 1
        }
        
        return String(format: "%.0f%@", size, units[unitIndex])
    }
}

// MARK: - Cache Performance Report

public extension CacheBenchmarks {
    
    struct CachePerformanceReport {
        public let cacheType: String
        public let capacity: Int
        public let hitRate: Double
        public let avgGetLatency: TimeInterval
        public let avgSetLatency: TimeInterval
        public let evictionRate: Double
        public let memoryUsage: Int
        
        public var summary: String {
            return String(format: """
                Cache Type: %@
                  Capacity: %d
                  Hit Rate: %.1f%%
                  Avg Get: %.2f μs
                  Avg Set: %.2f μs
                  Eviction Rate: %.1f%%
                  Memory: %@
                """,
                cacheType,
                capacity,
                hitRate * 100,
                avgGetLatency * 1_000_000,
                avgSetLatency * 1_000_000,
                evictionRate * 100,
                formatBytes(memoryUsage)
            )
        }
        
        private func formatBytes(_ bytes: Int) -> String {
            let units = ["B", "KB", "MB", "GB"]
            var size = Double(bytes)
            var unitIndex = 0
            
            while size >= 1024 && unitIndex < units.count - 1 {
                size /= 1024
                unitIndex += 1
            }
            
            return String(format: "%.1f %@", size, units[unitIndex])
        }
    }
}
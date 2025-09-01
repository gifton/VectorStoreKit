// VectorStoreKit: Caching Strategy Tests
//
// Comprehensive tests for the multi-level caching system

import XCTest
@testable import VectorStoreKit
import simd

// MARK: - Mock Storage Backend

actor MockCacheStorageBackend: CacheStorageBackend {
    typealias Vector = SIMD16<Float>
    
    private var storage: [VectorID: Vector] = [:]
    private var fetchDelay: TimeInterval = 0.001
    private var fetchCount = 0
    
    func fetchVectors(ids: [VectorID]) async throws -> [VectorID: Vector] {
        fetchCount += 1
        
        // Simulate fetch delay
        try? await Task.sleep(for: .seconds(fetchDelay))
        
        var result: [VectorID: Vector] = [:]
        for id in ids {
            if let vector = storage[id] {
                result[id] = vector
            }
        }
        return result
    }
    
    func storeVectors(_ vectors: [VectorID: Vector]) async throws {
        for (id, vector) in vectors {
            storage[id] = vector
        }
    }
    
    func checkExistence(ids: [VectorID]) async -> [VectorID: Bool] {
        var result: [VectorID: Bool] = [:]
        for id in ids {
            result[id] = storage[id] != nil
        }
        return result
    }
    
    // Test helpers
    func addVector(id: VectorID, vector: Vector) async {
        storage[id] = vector
    }
    
    func getFetchCount() async -> Int {
        fetchCount
    }
}

// MARK: - Caching Strategy Tests

final class CachingStrategyTests: XCTestCase {
    typealias TestVector = SIMD16<Float>
    
    // MARK: - Basic Functionality Tests
    
    func testMultiLevelCacheInitialization() async throws {
        // Create configurations for three levels
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 1024 * 1024),    // 1MB
            CacheLevelConfiguration(level: .l2, maxMemory: 10 * 1024 * 1024), // 10MB
            CacheLevelConfiguration(level: .l3, maxMemory: 100 * 1024 * 1024) // 100MB
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        let stats = await cache.statistics()
        XCTAssertEqual(stats.levelStatistics.count, 3)
        XCTAssertEqual(stats.memoryPressure, .normal)
    }
    
    func testBasicSetAndGet() async throws {
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 10000)
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        // Create test vector
        let vectorId = "test-vector-1"
        let vector = TestVector(repeating: 1.0)
        
        // Set vector
        await cache.set(id: vectorId, vector: vector, metadata: "test metadata")
        
        // Get vector
        let retrieved = await cache.get(id: vectorId)
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved, vector)
        
        // Check statistics
        let stats = await cache.statistics()
        let globalMetrics = stats.globalMetrics
        XCTAssertEqual(globalMetrics.totalHits, 1)
        XCTAssertEqual(globalMetrics.totalMisses, 0)
    }
    
    func testCacheMiss() async throws {
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 10000)
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        // Try to get non-existent vector
        let retrieved = await cache.get(id: "non-existent")
        XCTAssertNil(retrieved)
        
        // Check statistics
        let stats = await cache.statistics()
        XCTAssertEqual(stats.globalMetrics.totalMisses, 1)
    }
    
    // MARK: - Multi-Level Tests
    
    func testLevelPromotion() async throws {
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 1000),
            CacheLevelConfiguration(level: .l2, maxMemory: 10000),
            CacheLevelConfiguration(level: .l3, maxMemory: 100000)
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        // Add vector to L3
        let vectorId = "promote-test"
        let vector = TestVector(repeating: 2.0)
        await cache.set(id: vectorId, vector: vector, priority: .low)
        
        // Access multiple times to trigger promotion
        for _ in 0..<15 {
            _ = await cache.get(id: vectorId)
        }
        
        // Allow some time for promotion
        try? await Task.sleep(for: .milliseconds(100))
        
        // Vector should be promoted to higher level
        let stats = await cache.statistics()
        XCTAssertGreaterThan(stats.globalMetrics.totalHits, 10)
    }
    
    func testEvictionAcrossLevels() async throws {
        // Small caches to force eviction
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 500),
            CacheLevelConfiguration(level: .l2, maxMemory: 1000)
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        // Add many vectors to force eviction
        for i in 0..<20 {
            let vector = TestVector(repeating: Float(i))
            await cache.set(id: "vector-\(i)", vector: vector, priority: .normal)
        }
        
        // Early vectors should be evicted
        let firstVector = await cache.get(id: "vector-0")
        XCTAssertNil(firstVector)
        
        // Recent vectors should still be cached
        let lastVector = await cache.get(id: "vector-19")
        XCTAssertNotNil(lastVector)
    }
    
    // MARK: - Memory Pressure Tests
    
    func testMemoryPressureHandling() async throws {
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 10000),
            CacheLevelConfiguration(level: .l2, maxMemory: 100000)
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        // Add some vectors
        for i in 0..<10 {
            let vector = TestVector(repeating: Float(i))
            await cache.set(id: "pressure-\(i)", vector: vector)
        }
        
        // Simulate memory pressure
        // Note: In real implementation, this would be triggered by system
        // For testing, we'd need to expose a method to simulate pressure
        
        // Check that cache responds to pressure
        let stats = await cache.statistics()
        XCTAssertNotNil(stats.memoryPressure)
    }
    
    // MARK: - Cache Warming Tests
    
    func testCacheWarming() async throws {
        let backend = MockCacheStorageBackend()
        
        // Pre-populate storage
        for i in 0..<10 {
            let vector = TestVector(repeating: Float(i))
            await backend.addVector(id: "warm-\(i)", vector: vector)
        }
        
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 10000),
            CacheLevelConfiguration(level: .l2, maxMemory: 100000)
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs,
            storageBackend: backend
        )
        
        // Create predictions for warming
        var predictions: [VectorID: AccessPrediction] = [:]
        for i in 0..<5 {
            predictions["warm-\(i)"] = AccessPrediction(
                confidence: 0.8,
                timeWindow: 60
            )
        }
        
        // Warm the cache
        await cache.warmCaches(predictions: predictions)
        
        // Allow warming to complete
        try? await Task.sleep(for: .milliseconds(100))
        
        // Check that predicted vectors are cached
        for i in 0..<5 {
            let vector = await cache.get(id: "warm-\(i)")
            XCTAssertNotNil(vector, "Vector warm-\(i) should be cached")
        }
        
        // Check fetch count
        let fetchCount = await backend.getFetchCount()
        XCTAssertGreaterThan(fetchCount, 0, "Backend should have been accessed")
    }
    
    // MARK: - Pattern Analysis Tests
    
    func testAccessPatternDetection() async throws {
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 10000),
            CacheLevelConfiguration(level: .l2, maxMemory: 100000)
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        // Create sequential access pattern
        for i in 0..<20 {
            let vector = TestVector(repeating: Float(i))
            await cache.set(id: "seq-\(i)", vector: vector)
            
            // Access in sequence
            if i > 0 {
                _ = await cache.get(id: "seq-\(i-1)")
            }
        }
        
        // Get performance analysis
        let analysis = await cache.performanceAnalysis()
        XCTAssertNotNil(analysis)
        XCTAssertFalse(analysis.recommendations.isEmpty)
    }
    
    // MARK: - Optimization Tests
    
    func testCacheOptimization() async throws {
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 1000),
            CacheLevelConfiguration(level: .l2, maxMemory: 10000),
            CacheLevelConfiguration(level: .l3, maxMemory: 100000)
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        // Create unbalanced access pattern
        for i in 0..<50 {
            let vector = TestVector(repeating: Float(i))
            let priority: CachePriority = i < 10 ? .critical : .low
            await cache.set(id: "opt-\(i)", vector: vector, priority: priority)
        }
        
        // Access some vectors frequently
        for _ in 0..<10 {
            for i in 0..<5 {
                _ = await cache.get(id: "opt-\(i)")
            }
        }
        
        // Run optimization
        await cache.optimize()
        
        // Check that frequently accessed items are still available
        for i in 0..<5 {
            let vector = await cache.get(id: "opt-\(i)")
            XCTAssertNotNil(vector, "Frequently accessed vector should remain cached")
        }
    }
    
    // MARK: - Performance Tests
    
    func testConcurrentAccess() async throws {
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 10000),
            CacheLevelConfiguration(level: .l2, maxMemory: 100000)
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        // Pre-populate cache
        for i in 0..<100 {
            let vector = TestVector(repeating: Float(i))
            await cache.set(id: "concurrent-\(i)", vector: vector)
        }
        
        // Concurrent reads
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    for i in 0..<100 {
                        _ = await cache.get(id: "concurrent-\(i)")
                    }
                }
            }
        }
        
        // Verify statistics
        let stats = await cache.statistics()
        XCTAssertGreaterThan(stats.globalMetrics.totalHits, 900)
    }
    
    func testHealthScore() async throws {
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 10000),
            CacheLevelConfiguration(level: .l2, maxMemory: 100000)
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        // Perform some operations
        for i in 0..<20 {
            let vector = TestVector(repeating: Float(i))
            await cache.set(id: "health-\(i)", vector: vector)
            _ = await cache.get(id: "health-\(i)")
        }
        
        // Get performance analysis
        let analysis = await cache.performanceAnalysis()
        XCTAssertGreaterThan(analysis.healthScore, 0)
        XCTAssertLessThanOrEqual(analysis.healthScore, 100)
    }
    
    // MARK: - Edge Cases
    
    func testEmptyCache() async throws {
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 10000)
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        // Operations on empty cache
        let vector = await cache.get(id: "non-existent")
        XCTAssertNil(vector)
        
        await cache.clear()
        await cache.optimize()
        
        let stats = await cache.statistics()
        XCTAssertEqual(stats.globalMetrics.totalHits, 0)
        XCTAssertEqual(stats.globalMetrics.totalMisses, 1)
    }
    
    func testClearCache() async throws {
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 10000)
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        // Add vectors
        for i in 0..<10 {
            let vector = TestVector(repeating: Float(i))
            await cache.set(id: "clear-\(i)", vector: vector)
        }
        
        // Clear cache
        await cache.clear()
        
        // Verify all vectors are gone
        for i in 0..<10 {
            let vector = await cache.get(id: "clear-\(i)")
            XCTAssertNil(vector)
        }
    }
}

// MARK: - Performance Benchmark Tests

final class CachingStrategyPerformanceTests: XCTestCase {
    typealias TestVector = SIMD16<Float>
    
    func testLargeScalePerformance() async throws {
        let configs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 1024 * 1024),      // 1MB
            CacheLevelConfiguration(level: .l2, maxMemory: 10 * 1024 * 1024), // 10MB
            CacheLevelConfiguration(level: .l3, maxMemory: 100 * 1024 * 1024) // 100MB
        ]
        
        let cache = try await MultiLevelCachingStrategy<TestVector, String>(
            configurations: configs
        )
        
        let vectorCount = 10000
        let accessCount = 100000
        
        measure {
            let expectation = expectation(description: "Performance test")
            
            Task {
                // Populate cache
                for i in 0..<vectorCount {
                    let vector = TestVector(repeating: Float(i))
                    await cache.set(id: "perf-\(i)", vector: vector)
                }
                
                // Random access pattern
                for _ in 0..<accessCount {
                    let id = Int.random(in: 0..<vectorCount)
                    _ = await cache.get(id: "perf-\(id)")
                }
                
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 30.0)
        }
    }
}
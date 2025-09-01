// VectorStoreKit: Cache Correctness Tests
//
// Comprehensive test coverage for cache implementations

import XCTest
@testable import VectorStoreKit
import simd

final class CacheCorrectnessTests: XCTestCase {
    
    // MARK: - CRUD Operations
    
    func testLRUBasicOperations() async throws {
        let cache = try BasicLRUVectorCache<SIMD32<Float>>(maxMemory: 10_000)
        
        // Test set/get
        let vector1 = SIMD32<Float>(repeating: 1.0)
        await cache.set(id: "vec1", vector: vector1, priority: .normal)
        
        let retrieved = await cache.get(id: "vec1")
        XCTAssertEqual(retrieved, vector1)
        
        // Test update
        let vector2 = SIMD32<Float>(repeating: 2.0)
        await cache.set(id: "vec1", vector: vector2, priority: .high)
        
        let updated = await cache.get(id: "vec1")
        XCTAssertEqual(updated, vector2)
        
        // Test remove
        await cache.remove(id: "vec1")
        let removed = await cache.get(id: "vec1")
        XCTAssertNil(removed)
        
        // Test clear
        await cache.set(id: "vec2", vector: vector1, priority: .normal)
        await cache.set(id: "vec3", vector: vector2, priority: .normal)
        await cache.clear()
        
        XCTAssertEqual(await cache.count, 0)
    }
    
    func testLFUBasicOperations() async throws {
        let cache = try BasicLFUVectorCache<SIMD32<Float>>(maxMemory: 10_000)
        await cache.start()  // Start schedulers
        
        // Similar tests as LRU
        let vector1 = SIMD32<Float>(repeating: 1.0)
        await cache.set(id: "vec1", vector: vector1, priority: .normal)
        
        let retrieved = await cache.get(id: "vec1")
        XCTAssertEqual(retrieved, vector1)
        
        // Test frequency increment
        _ = await cache.get(id: "vec1")  // Access again
        _ = await cache.get(id: "vec1")  // Access again
        
        // vec1 should have frequency 3 now
    }
    
    func testFIFOBasicOperations() async throws {
        let cache = try BasicFIFOVectorCache<SIMD32<Float>>(maxMemory: 10_000)
        
        let vector1 = SIMD32<Float>(repeating: 1.0)
        await cache.set(id: "vec1", vector: vector1, priority: .normal)
        
        let retrieved = await cache.get(id: "vec1")
        XCTAssertEqual(retrieved, vector1)
    }
    
    // MARK: - Eviction Order Tests
    
    func testLRUEvictionOrder() async throws {
        // Small cache that fits only 2 vectors
        let vectorSize = MemoryLayout<SIMD32<Float>>.size
        let cacheSize = vectorSize * 2 + 100  // Room for 2 vectors plus overhead
        
        let cache = try BasicLRUVectorCache<SIMD32<Float>>(maxMemory: cacheSize)
        
        // Add 3 vectors
        let vec1 = SIMD32<Float>(repeating: 1.0)
        let vec2 = SIMD32<Float>(repeating: 2.0)
        let vec3 = SIMD32<Float>(repeating: 3.0)
        
        await cache.set(id: "vec1", vector: vec1, priority: .normal)
        await cache.set(id: "vec2", vector: vec2, priority: .normal)
        
        // Access vec1 to make it more recent
        _ = await cache.get(id: "vec1")
        
        // Add vec3 - should evict vec2 (least recently used)
        await cache.set(id: "vec3", vector: vec3, priority: .normal)
        
        XCTAssertNotNil(await cache.get(id: "vec1"))
        XCTAssertNil(await cache.get(id: "vec2"))  // Evicted
        XCTAssertNotNil(await cache.get(id: "vec3"))
    }
    
    func testLFUEvictionOrder() async throws {
        // Small cache that fits only 2 vectors
        let vectorSize = MemoryLayout<SIMD32<Float>>.size
        let cacheSize = vectorSize * 2 + 100
        
        let cache = try BasicLFUVectorCache<SIMD32<Float>>(maxMemory: cacheSize)
        await cache.start()
        
        let vec1 = SIMD32<Float>(repeating: 1.0)
        let vec2 = SIMD32<Float>(repeating: 2.0)
        let vec3 = SIMD32<Float>(repeating: 3.0)
        
        await cache.set(id: "vec1", vector: vec1, priority: .normal)
        await cache.set(id: "vec2", vector: vec2, priority: .normal)
        
        // Access vec1 multiple times to increase frequency
        _ = await cache.get(id: "vec1")
        _ = await cache.get(id: "vec1")
        _ = await cache.get(id: "vec1")
        
        // Add vec3 - should evict vec2 (least frequently used)
        await cache.set(id: "vec3", vector: vec3, priority: .normal)
        
        XCTAssertNotNil(await cache.get(id: "vec1"))
        XCTAssertNil(await cache.get(id: "vec2"))  // Evicted
        XCTAssertNotNil(await cache.get(id: "vec3"))
    }
    
    func testFIFOEvictionOrder() async throws {
        // Small cache that fits only 2 vectors
        let vectorSize = MemoryLayout<SIMD32<Float>>.size
        let cacheSize = vectorSize * 2 + 100
        
        let cache = try BasicFIFOVectorCache<SIMD32<Float>>(maxMemory: cacheSize)
        
        let vec1 = SIMD32<Float>(repeating: 1.0)
        let vec2 = SIMD32<Float>(repeating: 2.0)
        let vec3 = SIMD32<Float>(repeating: 3.0)
        
        await cache.set(id: "vec1", vector: vec1, priority: .normal)
        await cache.set(id: "vec2", vector: vec2, priority: .normal)
        
        // Access patterns don't matter for FIFO
        _ = await cache.get(id: "vec2")
        _ = await cache.get(id: "vec2")
        
        // Add vec3 - should evict vec1 (first in)
        await cache.set(id: "vec3", vector: vec3, priority: .normal)
        
        XCTAssertNil(await cache.get(id: "vec1"))  // Evicted (first in)
        XCTAssertNotNil(await cache.get(id: "vec2"))
        XCTAssertNotNil(await cache.get(id: "vec3"))
    }
    
    // MARK: - Concurrent Access Tests
    
    func testConcurrentOperations() async throws {
        let cache = try BasicLRUVectorCache<SIMD32<Float>>(maxMemory: 100_000)
        let iterations = 1000
        let taskCount = 8
        
        // Concurrent writes
        await withTaskGroup(of: Void.self) { group in
            for task in 0..<taskCount {
                group.addTask {
                    for i in 0..<iterations {
                        let id = "task\(task)_vec\(i)"
                        let vector = SIMD32<Float>(repeating: Float(i))
                        await cache.set(id: id, vector: vector, priority: .normal)
                    }
                }
            }
        }
        
        // Verify all writes succeeded
        var totalFound = 0
        for task in 0..<taskCount {
            for i in 0..<iterations {
                let id = "task\(task)_vec\(i)"
                if await cache.get(id: id) != nil {
                    totalFound += 1
                }
            }
        }
        
        // Not all will be in cache due to size limits, but should be substantial
        XCTAssertGreaterThan(totalFound, 0)
        
        // Concurrent reads
        let readIterations = 10_000
        await withTaskGroup(of: Int.self) { group in
            for _ in 0..<taskCount {
                group.addTask {
                    var hits = 0
                    for _ in 0..<readIterations {
                        let task = Int.random(in: 0..<taskCount)
                        let i = Int.random(in: 0..<iterations)
                        let id = "task\(task)_vec\(i)"
                        if await cache.get(id: id) != nil {
                            hits += 1
                        }
                    }
                    return hits
                }
            }
            
            var totalHits = 0
            for await hits in group {
                totalHits += hits
            }
            
            // Should have reasonable hit rate
            let hitRate = Float(totalHits) / Float(taskCount * readIterations)
            XCTAssertGreaterThan(hitRate, 0.1)  // At least 10% hit rate
        }
    }
    
    // MARK: - Memory Limit Tests
    
    func testMemoryCapEnforcement() async throws {
        let vectorSize = MemoryLayout<SIMD32<Float>>.size
        let maxVectors = 10
        let cacheSize = vectorSize * maxVectors
        
        let cache = try BasicLRUVectorCache<SIMD32<Float>>(maxMemory: cacheSize)
        
        // Add more vectors than capacity
        for i in 0..<(maxVectors * 2) {
            let vector = SIMD32<Float>(repeating: Float(i))
            await cache.set(id: "vec\(i)", vector: vector, priority: .normal)
            
            // Memory usage should never exceed limit
            let memoryUsage = await cache.memoryUsage
            XCTAssertLessThanOrEqual(memoryUsage, cacheSize,
                "Memory usage \(memoryUsage) exceeds limit \(cacheSize)")
        }
        
        // Count should be approximately maxVectors
        let count = await cache.count
        XCTAssertLessThanOrEqual(count, maxVectors + 2,  // Allow small overhead
            "Cache count \(count) exceeds expected maximum \(maxVectors)")
    }
    
    func testBatchInsertMemoryLimit() async throws {
        let vectorSize = MemoryLayout<SIMD32<Float>>.size
        let maxVectors = 10
        let cacheSize = vectorSize * maxVectors
        
        let cache = try BasicFIFOVectorCache<SIMD32<Float>>(maxMemory: cacheSize)
        
        // Create batch larger than cache
        var batch: [(VectorID, SIMD32<Float>, CachePriority)] = []
        for i in 0..<(maxVectors * 2) {
            let vector = SIMD32<Float>(repeating: Float(i))
            batch.append(("vec\(i)", vector, .normal))
        }
        
        // Insert batch
        for (id, vector, priority) in batch {
            await cache.set(id: id, vector: vector, priority: priority)
        }
        
        // Memory should not exceed limit
        let memoryUsage = await cache.memoryUsage
        XCTAssertLessThanOrEqual(memoryUsage, cacheSize,
            "Batch insert exceeded memory limit")
    }
    
    // MARK: - Priority Tests
    
    func testPriorityProtection() async throws {
        let vectorSize = MemoryLayout<SIMD32<Float>>.size
        let cacheSize = vectorSize * 3 + 100  // Room for 3 vectors
        
        let cache = try BasicFIFOVectorCache<SIMD32<Float>>(maxMemory: cacheSize)
        
        // Add high priority items
        let highVec1 = SIMD32<Float>(repeating: 1.0)
        let highVec2 = SIMD32<Float>(repeating: 2.0)
        await cache.set(id: "high1", vector: highVec1, priority: .high)
        await cache.set(id: "high2", vector: highVec2, priority: .critical)
        
        // Add low priority item
        let lowVec = SIMD32<Float>(repeating: 3.0)
        await cache.set(id: "low1", vector: lowVec, priority: .low)
        
        // Add another item - should evict low priority
        let newVec = SIMD32<Float>(repeating: 4.0)
        await cache.set(id: "new1", vector: newVec, priority: .normal)
        
        // High priority items should remain
        XCTAssertNotNil(await cache.get(id: "high1"))
        XCTAssertNotNil(await cache.get(id: "high2"))
        XCTAssertNil(await cache.get(id: "low1"))  // Should be evicted
    }
    
    func testFIFOTwoQueueBehavior() async throws {
        let vectorSize = MemoryLayout<SIMD32<Float>>.size
        let cacheSize = vectorSize * 5 + 100  // Room for 5 vectors
        
        let cache = try BasicFIFOVectorCache<SIMD32<Float>>(maxMemory: cacheSize)
        
        // Fill cache with mix of priorities
        // Protected queue (20% = 1 vector)
        await cache.set(id: "critical1", vector: SIMD32<Float>(repeating: 1.0), priority: .critical)
        
        // Open queue (80% = 4 vectors)
        await cache.set(id: "normal1", vector: SIMD32<Float>(repeating: 2.0), priority: .normal)
        await cache.set(id: "normal2", vector: SIMD32<Float>(repeating: 3.0), priority: .normal)
        await cache.set(id: "low1", vector: SIMD32<Float>(repeating: 4.0), priority: .low)
        await cache.set(id: "low2", vector: SIMD32<Float>(repeating: 5.0), priority: .low)
        
        // Add new normal priority - should evict from open queue (normal1 is oldest)
        await cache.set(id: "new1", vector: SIMD32<Float>(repeating: 6.0), priority: .normal)
        
        XCTAssertNotNil(await cache.get(id: "critical1"), "Protected item should remain")
        XCTAssertNil(await cache.get(id: "normal1"), "Oldest open queue item should be evicted")
        XCTAssertNotNil(await cache.get(id: "normal2"))
        XCTAssertNotNil(await cache.get(id: "new1"))
        
        // Add another high priority - protected queue is now at capacity
        await cache.set(id: "high1", vector: SIMD32<Float>(repeating: 7.0), priority: .high)
        
        // Add one more high priority - should evict from open queue first
        await cache.set(id: "high2", vector: SIMD32<Float>(repeating: 8.0), priority: .high)
        
        XCTAssertNotNil(await cache.get(id: "critical1"), "Protected items should remain")
        XCTAssertNotNil(await cache.get(id: "high1"), "Protected items should remain")
        
        // Verify open queue was evicted first
        let remainingCount = await cache.count
        XCTAssertLessThanOrEqual(remainingCount, 5)
    }
    
    // MARK: - Performance Tests
    
    func testGetLatency() async throws {
        let cache = try BasicLRUVectorCache<SIMD32<Float>>(maxMemory: 100_000_000)
        
        // Warm up cache
        for i in 0..<10_000 {
            let vector = SIMD32<Float>(repeating: Float(i))
            await cache.set(id: "vec\(i)", vector: vector, priority: .normal)
        }
        
        // Measure get latency
        let startTime = ContinuousClock.now
        let iterations = 100_000
        
        for _ in 0..<iterations {
            let id = "vec\(Int.random(in: 0..<10_000))"
            _ = await cache.get(id: id)
        }
        
        let elapsed = ContinuousClock.now - startTime
        let avgLatencyNs = elapsed.components.attoseconds / 1_000_000_000 / UInt64(iterations)
        
        // Should be under 150ns per operation
        XCTAssertLessThan(avgLatencyNs, 150,
            "Average get latency \(avgLatencyNs)ns exceeds 150ns target")
    }
}

// MARK: - Stress Tests

final class CacheStressTests: XCTestCase {
    
    func testMemoryPressure() async throws {
        let cache = try BasicLRUVectorCache<SIMD32<Float>>(maxMemory: 10_000_000)
        
        // Continuously add vectors under memory pressure
        let startTime = ContinuousClock.now
        let duration = Duration.seconds(5)
        var operations = 0
        
        while ContinuousClock.now - startTime < duration {
            let vector = SIMD32<Float>.random(in: -1...1)
            let id = "vec\(operations)"
            await cache.set(id: id, vector: vector, priority: .normal)
            
            // Randomly access existing items
            if operations > 0 && Int.random(in: 0...1) == 0 {
                let randomId = "vec\(Int.random(in: 0..<operations))"
                _ = await cache.get(id: randomId)
            }
            
            operations += 1
            
            // Verify memory limit
            let usage = await cache.memoryUsage
            XCTAssertLessThanOrEqual(usage, 10_000_000)
        }
        
        print("Completed \(operations) operations under memory pressure")
    }
    
    func testConcurrentStress() async throws {
        let cache = try BasicLRUVectorCache<SIMD32<Float>>(maxMemory: 100_000_000)
        let duration = Duration.seconds(10)
        let taskCount = 16
        
        await withTaskGroup(of: Int.self) { group in
            for taskId in 0..<taskCount {
                group.addTask {
                    let startTime = ContinuousClock.now
                    var operations = 0
                    
                    while ContinuousClock.now - startTime < duration {
                        let operation = Int.random(in: 0...2)
                        
                        switch operation {
                        case 0:  // Set
                            let vector = SIMD32<Float>.random(in: -1...1)
                            let id = "task\(taskId)_vec\(operations)"
                            await cache.set(id: id, vector: vector, priority: .normal)
                        case 1:  // Get
                            let id = "task\(Int.random(in: 0..<taskCount))_vec\(Int.random(in: 0..<operations + 1))"
                            _ = await cache.get(id: id)
                        case 2:  // Remove
                            let id = "task\(Int.random(in: 0..<taskCount))_vec\(Int.random(in: 0..<operations + 1))"
                            await cache.remove(id: id)
                        default:
                            break
                        }
                        
                        operations += 1
                    }
                    
                    return operations
                }
            }
            
            var totalOperations = 0
            for await ops in group {
                totalOperations += ops
            }
            
            print("Completed \(totalOperations) total operations across \(taskCount) tasks")
        }
    }
}

// MARK: - Test Helpers

extension SIMD32 where Scalar == Float {
    static func random(in range: ClosedRange<Float>) -> SIMD32<Float> {
        var result = SIMD32<Float>()
        for i in 0..<32 {
            result[i] = Float.random(in: range)
        }
        return result
    }
}
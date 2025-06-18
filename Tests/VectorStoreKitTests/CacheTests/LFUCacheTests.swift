import XCTest
import simd
@testable import VectorStoreKit

final class LFUCacheTests: XCTestCase {
    typealias TestVector = SIMD4<Float>
    
    func testBasicCRUDOperations() async throws {
        let cache = try BasicLFUVectorCache<TestVector>(maxMemory: 10000)
        
        // Test set and get
        let vector1 = TestVector(1, 2, 3, 4)
        await cache.set(id: "vec1", vector: vector1, priority: .normal)
        
        let retrieved = await cache.get(id: "vec1")
        XCTAssertEqual(retrieved, vector1)
        
        // Test contains
        let contains = await cache.contains(id: "vec1")
        XCTAssertTrue(contains)
        
        // Test remove
        await cache.remove(id: "vec1")
        let afterRemove = await cache.get(id: "vec1")
        XCTAssertNil(afterRemove)
        
        // Test clear
        await cache.set(id: "vec2", vector: TestVector(5, 6, 7, 8), priority: .normal)
        await cache.clear()
        let afterClear = await cache.get(id: "vec2")
        XCTAssertNil(afterClear)
    }
    
    func testFrequencyBasedEviction() async throws {
        // Small cache that fits ~3 vectors
        let cache = try BasicLFUVectorCache<TestVector>(maxMemory: 300)
        
        // Add vectors
        await cache.set(id: "vec1", vector: TestVector(1, 1, 1, 1), priority: .normal)
        await cache.set(id: "vec2", vector: TestVector(2, 2, 2, 2), priority: .normal)
        await cache.set(id: "vec3", vector: TestVector(3, 3, 3, 3), priority: .normal)
        
        // Access vec1 and vec3 multiple times to increase frequency
        for _ in 0..<3 {
            _ = await cache.get(id: "vec1")
            _ = await cache.get(id: "vec3")
        }
        
        // vec2 accessed only once (during insertion)
        _ = await cache.get(id: "vec2")
        
        // Add vec4, should evict vec2 (least frequently used)
        await cache.set(id: "vec4", vector: TestVector(4, 4, 4, 4), priority: .normal)
        
        XCTAssertNotNil(await cache.get(id: "vec1"))
        XCTAssertNil(await cache.get(id: "vec2")) // Should be evicted
        XCTAssertNotNil(await cache.get(id: "vec3"))
        XCTAssertNotNil(await cache.get(id: "vec4"))
    }
    
    func testMemoryCapacityEnforcement() async throws {
        let maxMemory = 500
        let cache = try BasicLFUVectorCache<TestVector>(maxMemory: maxMemory)
        
        // Fill cache
        for i in 0..<10 {
            await cache.set(id: "vec\(i)", vector: TestVector(Float(i), 0, 0, 0), priority: .normal)
        }
        
        // Check memory usage doesn't exceed limit
        let memoryUsage = await cache.memoryUsage
        XCTAssertLessThanOrEqual(memoryUsage, maxMemory, "Memory usage \(memoryUsage) exceeds max \(maxMemory)")
        
        // Verify cache isn't empty
        let count = await cache.count
        XCTAssertGreaterThan(count, 0, "Cache should contain some entries")
    }
    
    func testMinFrequencyBugFix() async throws {
        // Test the specific bug where minFrequency wasn't updated correctly
        let cache = try BasicLFUVectorCache<TestVector>(maxMemory: 300)
        
        // Add vectors
        await cache.set(id: "vec1", vector: TestVector(1, 1, 1, 1), priority: .normal)
        await cache.set(id: "vec2", vector: TestVector(2, 2, 2, 2), priority: .normal)
        
        // Access vec1 to increase its frequency
        _ = await cache.get(id: "vec1")
        
        // Remove vec2 (which has frequency 1, the minimum)
        await cache.remove(id: "vec2")
        
        // Add new vectors - this should work correctly even with minFrequency update
        await cache.set(id: "vec3", vector: TestVector(3, 3, 3, 3), priority: .normal)
        await cache.set(id: "vec4", vector: TestVector(4, 4, 4, 4), priority: .normal)
        
        // Should be able to evict vec3 or vec4 (both have frequency 1)
        await cache.set(id: "vec5", vector: TestVector(5, 5, 5, 5), priority: .normal)
        
        // vec1 should still be there (higher frequency)
        XCTAssertNotNil(await cache.get(id: "vec1"))
    }
    
    func testConcurrentAccess() async throws {
        let cache = try BasicLFUVectorCache<TestVector>(maxMemory: 50000)
        let operationCount = 100
        let taskCount = 8
        
        // Concurrent writes
        await withTaskGroup(of: Void.self) { group in
            for taskId in 0..<taskCount {
                group.addTask {
                    for i in 0..<operationCount {
                        let id = "task\(taskId)_vec\(i)"
                        let vector = TestVector(Float(taskId), Float(i), 0, 0)
                        await cache.set(id: id, vector: vector, priority: .normal)
                    }
                }
            }
        }
        
        // Concurrent reads
        await withTaskGroup(of: Void.self) { group in
            for taskId in 0..<taskCount {
                group.addTask {
                    for i in 0..<operationCount {
                        let id = "task\(taskId)_vec\(i)"
                        _ = await cache.get(id: id)
                    }
                }
            }
        }
        
        // Verify cache state
        let count = await cache.count
        XCTAssertGreaterThan(count, 0, "Cache should contain entries after concurrent operations")
        
        let memoryUsage = await cache.memoryUsage
        XCTAssertLessThanOrEqual(memoryUsage, 50000, "Memory usage should not exceed limit")
    }
    
    func testFrequencyDecay() async throws {
        let cache = try BasicLFUVectorCache<TestVector>(maxMemory: 10000)
        
        // Add vector and access it multiple times
        await cache.set(id: "vec1", vector: TestVector(1, 1, 1, 1), priority: .normal)
        for _ in 0..<5 {
            _ = await cache.get(id: "vec1")
        }
        
        // Force decay by calling optimize
        await cache.optimize()
        
        // Verify cache still works after decay
        let retrieved = await cache.get(id: "vec1")
        XCTAssertNotNil(retrieved)
    }
    
    func testAdaptiveThreshold() async throws {
        let cache = try BasicLFUVectorCache<TestVector>(maxMemory: 10000)
        
        // Generate poor hit rate
        for i in 0..<100 {
            await cache.set(id: "vec\(i)", vector: TestVector(Float(i), 0, 0, 0), priority: .normal)
            _ = await cache.get(id: "nonexistent\(i)") // All misses
        }
        
        // Let threshold adapt by setting new items
        for i in 100..<110 {
            await cache.set(id: "vec\(i)", vector: TestVector(Float(i), 0, 0, 0), priority: .normal)
        }
        
        // Cache should still function properly
        XCTAssertNotNil(await cache.get(id: "vec109"))
    }
    
    func testStatistics() async throws {
        let cache = try BasicLFUVectorCache<TestVector>(maxMemory: 10000)
        
        // Generate some activity
        await cache.set(id: "vec1", vector: TestVector(1, 1, 1, 1), priority: .normal)
        _ = await cache.get(id: "vec1")
        _ = await cache.get(id: "missing")
        
        let stats = await cache.statistics()
        XCTAssertEqual(stats.hitCount, 1)
        XCTAssertEqual(stats.missCount, 1)
        XCTAssertGreaterThan(stats.memoryUsage, 0)
    }
}
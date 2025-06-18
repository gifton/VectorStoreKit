import XCTest
import simd
@testable import VectorStoreKit

final class LRUCacheTests: XCTestCase {
    typealias TestVector = SIMD4<Float>
    
    func testBasicCRUDOperations() async throws {
        let cache = try BasicLRUVectorCache<TestVector>(maxMemory: 10000)
        
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
    
    func testEvictionOrder() async throws {
        // Small cache that fits ~3 vectors
        let cache = try BasicLRUVectorCache<TestVector>(maxMemory: 300)
        
        // Add vectors in order
        await cache.set(id: "vec1", vector: TestVector(1, 1, 1, 1), priority: .normal)
        await cache.set(id: "vec2", vector: TestVector(2, 2, 2, 2), priority: .normal)
        await cache.set(id: "vec3", vector: TestVector(3, 3, 3, 3), priority: .normal)
        
        // Access vec1 to make it most recently used
        _ = await cache.get(id: "vec1")
        
        // Add vec4, should evict vec2 (least recently used)
        await cache.set(id: "vec4", vector: TestVector(4, 4, 4, 4), priority: .normal)
        
        XCTAssertNotNil(await cache.get(id: "vec1"))
        XCTAssertNil(await cache.get(id: "vec2")) // Should be evicted
        XCTAssertNotNil(await cache.get(id: "vec3"))
        XCTAssertNotNil(await cache.get(id: "vec4"))
    }
    
    func testMemoryCapacityEnforcement() async throws {
        let maxMemory = 500
        let cache = try BasicLRUVectorCache<TestVector>(maxMemory: maxMemory)
        
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
    
    func testConcurrentAccess() async throws {
        let cache = try BasicLRUVectorCache<TestVector>(maxMemory: 50000)
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
    
    func testHitRateCalculation() async throws {
        let cache = try BasicLRUVectorCache<TestVector>(maxMemory: 10000)
        
        // Add some vectors
        await cache.set(id: "vec1", vector: TestVector(1, 1, 1, 1), priority: .normal)
        await cache.set(id: "vec2", vector: TestVector(2, 2, 2, 2), priority: .normal)
        
        // Generate hits and misses
        _ = await cache.get(id: "vec1") // hit
        _ = await cache.get(id: "vec2") // hit
        _ = await cache.get(id: "vec3") // miss
        _ = await cache.get(id: "vec1") // hit
        _ = await cache.get(id: "vec4") // miss
        
        let hitRate = await cache.hitRate
        XCTAssertEqual(hitRate, 0.6, accuracy: 0.01, "Hit rate should be 3/5 = 0.6")
    }
    
    func testPriorityEviction() async throws {
        let cache = try BasicLRUVectorCache<TestVector>(maxMemory: 300)
        
        // Add high priority vector
        await cache.set(id: "high", vector: TestVector(1, 1, 1, 1), priority: .high)
        
        // Add normal priority vectors
        await cache.set(id: "normal1", vector: TestVector(2, 2, 2, 2), priority: .normal)
        await cache.set(id: "normal2", vector: TestVector(3, 3, 3, 3), priority: .normal)
        
        // Adding another should preferentially evict normal priority
        await cache.set(id: "new", vector: TestVector(4, 4, 4, 4), priority: .normal)
        
        // High priority should still be there
        XCTAssertNotNil(await cache.get(id: "high"))
    }
    
    func testStatistics() async throws {
        let cache = try BasicLRUVectorCache<TestVector>(maxMemory: 10000)
        
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
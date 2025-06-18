import XCTest
import simd
@testable import VectorStoreKit

final class FIFOCacheTests: XCTestCase {
    typealias TestVector = SIMD4<Float>
    
    func testBasicCRUDOperations() async throws {
        let cache = try BasicFIFOVectorCache<TestVector>(maxMemory: 10000)
        
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
    
    func testFIFOEvictionOrder() async throws {
        // Small cache that fits ~3 vectors
        let cache = try BasicFIFOVectorCache<TestVector>(maxMemory: 300)
        
        // Add vectors in order
        await cache.set(id: "vec1", vector: TestVector(1, 1, 1, 1), priority: .normal)
        await cache.set(id: "vec2", vector: TestVector(2, 2, 2, 2), priority: .normal)
        await cache.set(id: "vec3", vector: TestVector(3, 3, 3, 3), priority: .normal)
        
        // Add vec4, should evict vec1 (first in)
        await cache.set(id: "vec4", vector: TestVector(4, 4, 4, 4), priority: .normal)
        
        XCTAssertNil(await cache.get(id: "vec1")) // Should be evicted
        XCTAssertNotNil(await cache.get(id: "vec2"))
        XCTAssertNotNil(await cache.get(id: "vec3"))
        XCTAssertNotNil(await cache.get(id: "vec4"))
    }
    
    func testMemoryCapacityEnforcement() async throws {
        let maxMemory = 500
        let cache = try BasicFIFOVectorCache<TestVector>(maxMemory: maxMemory)
        
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
    
    func testBatchProcessing() async throws {
        let cache = try BasicFIFOVectorCache<TestVector>(maxMemory: 5000)
        
        // Add many items quickly to trigger batch processing
        for i in 0..<150 {
            await cache.set(id: "vec\(i)", vector: TestVector(Float(i), 0, 0, 0), priority: .normal)
        }
        
        // Force batch processing by optimizing
        await cache.optimize()
        
        // Check memory limit is respected
        let memoryUsage = await cache.memoryUsage
        XCTAssertLessThanOrEqual(memoryUsage, 5000, "Memory usage should not exceed limit after batch")
        
        // Verify some entries exist
        let count = await cache.count
        XCTAssertGreaterThan(count, 0, "Cache should contain entries after batch processing")
    }
    
    func testPriorityProtection() async throws {
        let cache = try BasicFIFOVectorCache<TestVector>(maxMemory: 300)
        
        // Add high priority vector first
        await cache.set(id: "high1", vector: TestVector(1, 1, 1, 1), priority: .high)
        
        // Add normal priority vectors
        await cache.set(id: "normal1", vector: TestVector(2, 2, 2, 2), priority: .normal)
        await cache.set(id: "normal2", vector: TestVector(3, 3, 3, 3), priority: .normal)
        
        // Add another normal priority - should evict normal1, not high1
        await cache.set(id: "normal3", vector: TestVector(4, 4, 4, 4), priority: .normal)
        
        // High priority should still be there
        XCTAssertNotNil(await cache.get(id: "high1"))
        XCTAssertNil(await cache.get(id: "normal1")) // Should be evicted
    }
    
    func testConcurrentAccess() async throws {
        let cache = try BasicFIFOVectorCache<TestVector>(maxMemory: 50000)
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
    
    func testBulkEviction() async throws {
        let cache = try BasicFIFOVectorCache<TestVector>(maxMemory: 1000)
        
        // Fill cache with mixed priorities
        await cache.set(id: "low1", vector: TestVector(1, 1, 1, 1), priority: .low)
        await cache.set(id: "normal1", vector: TestVector(2, 2, 2, 2), priority: .normal)
        await cache.set(id: "high1", vector: TestVector(3, 3, 3, 3), priority: .high)
        await cache.set(id: "low2", vector: TestVector(4, 4, 4, 4), priority: .low)
        await cache.set(id: "normal2", vector: TestVector(5, 5, 5, 5), priority: .normal)
        
        // Add many items to trigger bulk eviction
        for i in 0..<20 {
            await cache.set(id: "bulk\(i)", vector: TestVector(Float(i), 0, 0, 0), priority: .normal)
        }
        
        // Force batch processing
        await cache.optimize()
        
        // High priority should survive bulk eviction
        XCTAssertNotNil(await cache.get(id: "high1"))
        
        // Low priority items should be evicted first
        XCTAssertNil(await cache.get(id: "low1"))
    }
    
    func testStatistics() async throws {
        let cache = try BasicFIFOVectorCache<TestVector>(maxMemory: 10000)
        
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
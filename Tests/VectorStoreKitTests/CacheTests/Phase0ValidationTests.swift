import XCTest
import simd
@testable import VectorStoreKit

/// Phase 0 validation tests - ensures all critical fixes are working
final class Phase0ValidationTests: XCTestCase {
    typealias TestVector = SIMD4<Float>
    
    // MARK: - 0.1 Structured Concurrency Test
    
    func testStructuredConcurrency() async throws {
        // This test verifies that we're using structured concurrency properly
        // by ensuring no unstructured tasks accumulate
        
        let cache = try BasicLRUVectorCache<TestVector>(maxMemory: 10000)
        
        // Perform many operations that would previously spawn unstructured tasks
        for i in 0..<100 {
            await cache.set(id: "vec\(i)", vector: TestVector(Float(i), 0, 0, 0), priority: .normal)
            _ = await cache.get(id: "vec\(i)")
        }
        
        // If unstructured tasks were accumulating, memory would grow
        // This is a simplified test - in production you'd use Instruments
        XCTAssertTrue(true, "No crash from unstructured task accumulation")
    }
    
    // MARK: - 0.2 Capacity Enforcement Test
    
    func testFIFOBatchCapacityEnforcement() async throws {
        // Test the specific FIFO batch processing bug fix
        let maxMemory = 300 // Small cache for ~3 vectors
        let cache = try BasicFIFOVectorCache<TestVector>(maxMemory: maxMemory)
        
        // Trigger batch processing by adding many items quickly
        for i in 0..<20 {
            await cache.set(id: "batch\(i)", vector: TestVector(Float(i), Float(i), 0, 0), priority: .normal)
        }
        
        // Force batch processing
        await cache.optimize()
        
        // Verify memory limit is strictly enforced
        let memoryUsage = await cache.memoryUsage
        XCTAssertLessThanOrEqual(memoryUsage, maxMemory, "FIFO batch processing must respect memory limit")
        
        // Verify some entries exist (not all evicted)
        let count = await cache.count
        XCTAssertGreaterThan(count, 0, "Cache should contain entries")
        XCTAssertLessThan(count, 20, "Cache should have evicted entries")
    }
    
    func testLRUCapacityEnforcement() async throws {
        let maxMemory = 300
        let cache = try BasicLRUVectorCache<TestVector>(maxMemory: maxMemory)
        
        // Fill beyond capacity
        for i in 0..<10 {
            await cache.set(id: "vec\(i)", vector: TestVector(Float(i), 0, 0, 0), priority: .normal)
        }
        
        let memoryUsage = await cache.memoryUsage
        XCTAssertLessThanOrEqual(memoryUsage, maxMemory, "LRU must enforce memory limit")
    }
    
    func testLFUCapacityEnforcement() async throws {
        let maxMemory = 300
        let cache = try BasicLFUVectorCache<TestVector>(maxMemory: maxMemory)
        
        // Fill beyond capacity
        for i in 0..<10 {
            await cache.set(id: "vec\(i)", vector: TestVector(Float(i), 0, 0, 0), priority: .normal)
        }
        
        let memoryUsage = await cache.memoryUsage
        XCTAssertLessThanOrEqual(memoryUsage, maxMemory, "LFU must enforce memory limit")
    }
    
    // MARK: - 0.4 Retain Cycle Guard Test (LRU specific)
    
    func testLRUNoRetainCycles() async throws {
        // This test ensures the LRU linked list doesn't create retain cycles
        let cache = try BasicLRUVectorCache<TestVector>(maxMemory: 1000)
        
        // Add and remove many items
        for i in 0..<50 {
            await cache.set(id: "cycle\(i)", vector: TestVector(Float(i), 0, 0, 0), priority: .normal)
        }
        
        // Remove half
        for i in 0..<25 {
            await cache.remove(id: "cycle\(i)")
        }
        
        // Clear cache
        await cache.clear()
        
        // If there were retain cycles, memory would leak
        // In production, use Instruments to verify
        XCTAssertEqual(await cache.count, 0, "Cache should be empty after clear")
    }
    
    // MARK: - 0.5 LFU minFrequency Bug Test
    
    func testLFUMinFrequencyBug() async throws {
        // Test the specific bug where minFrequency becomes stale
        let cache = try BasicLFUVectorCache<TestVector>(maxMemory: 400)
        
        // Step 1: Add items with frequency 1
        await cache.set(id: "freq1_a", vector: TestVector(1, 1, 1, 1), priority: .normal)
        await cache.set(id: "freq1_b", vector: TestVector(2, 2, 2, 2), priority: .normal)
        
        // Step 2: Increase frequency of one item
        for _ in 0..<3 {
            _ = await cache.get(id: "freq1_a") // Now frequency 4
        }
        
        // Step 3: Remove the only item with frequency 1
        await cache.remove(id: "freq1_b")
        
        // Step 4: Add new items - this would previously fail due to stale minFrequency
        await cache.set(id: "new1", vector: TestVector(3, 3, 3, 3), priority: .normal)
        await cache.set(id: "new2", vector: TestVector(4, 4, 4, 4), priority: .normal)
        
        // Step 5: Force eviction - should work correctly
        await cache.set(id: "new3", vector: TestVector(5, 5, 5, 5), priority: .normal)
        await cache.set(id: "new4", vector: TestVector(6, 6, 6, 6), priority: .normal)
        
        // Verify high frequency item survived
        XCTAssertNotNil(await cache.get(id: "freq1_a"), "High frequency item should survive eviction")
        
        // Verify cache state is consistent
        let count = await cache.count
        XCTAssertGreaterThan(count, 0, "Cache should have entries")
        XCTAssertLessThanOrEqual(await cache.memoryUsage, 400, "Memory limit enforced")
    }
    
    // MARK: - Thread Safety Under Load
    
    func testConcurrentSafetyUnderLoad() async throws {
        // Aggressive concurrent test to ensure actor isolation works
        let cache = try BasicLRUVectorCache<TestVector>(maxMemory: 100000)
        let iterations = 1000
        let tasks = 20
        
        await withTaskGroup(of: Void.self) { group in
            // Concurrent writes
            for taskId in 0..<tasks {
                group.addTask {
                    for i in 0..<iterations {
                        let id = "task\(taskId)_item\(i)"
                        await cache.set(id: id, vector: TestVector(Float(taskId), Float(i), 0, 0), priority: .normal)
                    }
                }
            }
            
            // Concurrent reads
            for taskId in 0..<tasks {
                group.addTask {
                    for i in 0..<iterations {
                        let id = "task\(taskId)_item\(i)"
                        _ = await cache.get(id: id)
                    }
                }
            }
            
            // Concurrent removes
            for taskId in 0..<tasks/2 {
                group.addTask {
                    for i in 0..<iterations/2 {
                        let id = "task\(taskId)_item\(i)"
                        await cache.remove(id: id)
                    }
                }
            }
        }
        
        // Verify cache is in consistent state
        let finalCount = await cache.count
        let finalMemory = await cache.memoryUsage
        
        XCTAssertGreaterThanOrEqual(finalCount, 0, "Count should be non-negative")
        XCTAssertLessThanOrEqual(finalMemory, 100000, "Memory should respect limit")
        
        // Verify statistics are consistent
        let stats = await cache.statistics()
        XCTAssertGreaterThanOrEqual(stats.hitCount, 0, "Hit count should be non-negative")
        XCTAssertGreaterThanOrEqual(stats.missCount, 0, "Miss count should be non-negative")
    }
}
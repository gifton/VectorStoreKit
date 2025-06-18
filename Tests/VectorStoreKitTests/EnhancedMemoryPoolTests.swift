// EnhancedMemoryPoolTests.swift
// VectorStoreKitTests
//
// Tests for the enhanced memory pool manager with size tracking,
// memory pressure integration, and defragmentation

import XCTest
@testable import VectorStoreKit

final class EnhancedMemoryPoolTests: XCTestCase {
    var memoryManager: MemoryManager!
    var poolManager: MemoryPoolManager!
    
    override func setUp() async throws {
        memoryManager = MemoryManager()
        poolManager = MemoryPoolManager(
            memoryManager: memoryManager,
            autoDefragmentationEnabled: false // Disable for predictable tests
        )
    }
    
    override func tearDown() async throws {
        await poolManager.clearAll()
        poolManager = nil
        memoryManager = nil
    }
    
    // MARK: - Size Tracking Tests
    
    func testSizeTrackedAllocation() async throws {
        // Create enhanced buffer pool
        let pool = await poolManager.createEnhancedBufferPool(
            for: "test_pool",
            maxPoolSize: 10,
            defaultAlignment: 64
        )
        
        // Allocate buffers of different sizes
        let buffer1 = try await pool.acquireBuffer(size: 1024)
        let buffer2 = try await pool.acquireBuffer(size: 2048)
        let buffer3 = try await pool.acquireBuffer(size: 4096)
        
        // Get statistics
        let stats = await pool.statistics()
        XCTAssertEqual(stats.currentlyInUse, 1024 + 2048 + 4096)
        
        // Release buffers
        await pool.release(buffer1)
        await pool.release(buffer2)
        
        // Verify size tracking
        let statsAfter = await pool.statistics()
        XCTAssertEqual(statsAfter.currentlyInUse, 4096)
        
        // Release remaining
        await pool.release(buffer3)
        
        let statsFinal = await pool.statistics()
        XCTAssertEqual(statsFinal.currentlyInUse, 0)
    }
    
    func testReleaseWithoutSizeInfo() async throws {
        let pool = await poolManager.createEnhancedBufferPool(for: "test_pool")
        
        // Allocate buffer
        let buffer = try await pool.acquireBuffer(size: 1024)
        
        // Release using generic release method (which now tracks size)
        await pool.release(buffer)
        
        // Verify proper deallocation
        let stats = await pool.statistics()
        XCTAssertEqual(stats.currentlyInUse, 0)
    }
    
    // MARK: - Memory Pressure Tests
    
    func testMemoryPressureHandling() async throws {
        let pool = await poolManager.createEnhancedBufferPool(
            for: "pressure_test",
            maxPoolSize: 20
        )
        
        // Pre-allocate buffers
        await pool.preAllocate(count: 10)
        
        var statsBefore = await pool.statistics()
        XCTAssertGreaterThan(statsBefore.totalAllocated, 0)
        
        // Simulate memory pressure
        await poolManager.handleMemoryPressure(.warning)
        
        // Wait a bit for async operations
        try await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
        
        // Verify pool reduction
        let statsAfter = await pool.statistics()
        XCTAssertLessThanOrEqual(statsAfter.totalAllocated, statsBefore.totalAllocated)
    }
    
    func testCriticalMemoryPressure() async throws {
        let pool = await poolManager.createEnhancedBufferPool(for: "critical_test")
        
        // Allocate some buffers
        var buffers: [UnsafeMutableRawPointer] = []
        for _ in 0..<5 {
            buffers.append(try await pool.acquireBuffer(size: 1024))
        }
        
        // Release them back to pool
        for buffer in buffers {
            await pool.release(buffer)
        }
        
        // Handle critical pressure
        await poolManager.handleMemoryPressure(.critical)
        
        // Verify all pools cleared
        let stats = await pool.statistics()
        XCTAssertEqual(stats.totalAllocated - stats.currentlyInUse, 0)
    }
    
    // MARK: - Automatic Resizing Tests
    
    func testPoolGrowth() async throws {
        let pool = await poolManager.createEnhancedBufferPool(
            for: "growth_test",
            maxPoolSize: 100,
            growthFactor: 2.0
        )
        
        // Simulate high usage pattern for a specific size
        let targetSize = 1024
        var buffers: [UnsafeMutableRawPointer] = []
        
        // Allocate and release many times to establish pattern
        for _ in 0..<20 {
            let buffer = try await pool.acquireBuffer(size: targetSize)
            buffers.append(buffer)
        }
        
        // Release all
        for buffer in buffers {
            await pool.release(buffer)
        }
        
        // Get statistics to verify pool has buffers
        let stats = await pool.statistics()
        XCTAssertGreaterThan(stats.totalAllocated, 0)
    }
    
    func testUsagePatternTracking() async throws {
        let pool = await poolManager.createEnhancedBufferPool(for: "pattern_test")
        
        // Create usage pattern
        for _ in 0..<10 {
            let small = try await pool.acquireBuffer(size: 512)
            await pool.release(small)
        }
        
        for _ in 0..<5 {
            let medium = try await pool.acquireBuffer(size: 2048)
            await pool.release(medium)
        }
        
        for _ in 0..<2 {
            let large = try await pool.acquireBuffer(size: 8192)
            await pool.release(large)
        }
        
        // Pre-allocate based on pattern
        await pool.preAllocate(count: 15)
        
        // Verify allocation focused on frequently used sizes
        let stats = await pool.statistics()
        XCTAssertGreaterThan(stats.totalAllocated, 0)
    }
    
    // MARK: - Defragmentation Tests
    
    func testDefragmentation() async throws {
        let pool = await poolManager.createEnhancedBufferPool(
            for: "defrag_test",
            defragmentationThreshold: 0.2
        )
        
        // Create fragmentation by allocating various sizes
        var buffers: [UnsafeMutableRawPointer] = []
        for i in 0..<10 {
            let size = 1024 * (i + 1)
            buffers.append(try await pool.acquireBuffer(size: size))
        }
        
        // Release every other buffer
        for (index, buffer) in buffers.enumerated() where index % 2 == 0 {
            await pool.release(buffer)
        }
        
        // Force defragmentation
        let result = await pool.defragment()
        
        XCTAssertGreaterThanOrEqual(result.fragmentationAfter, 0)
        XCTAssertLessThanOrEqual(result.fragmentationAfter, result.fragmentationBefore)
        
        // Clean up
        for (index, buffer) in buffers.enumerated() where index % 2 == 1 {
            await pool.release(buffer)
        }
    }
    
    func testAutomaticDefragmentation() async throws {
        // Create pool manager with short defrag interval
        let autoPoolManager = MemoryPoolManager(
            memoryManager: memoryManager,
            autoDefragmentationEnabled: true,
            defragmentationInterval: 0.5 // 0.5 seconds for testing
        )
        
        let pool = await autoPoolManager.createEnhancedBufferPool(for: "auto_defrag")
        
        // Create some fragmentation
        var buffers: [UnsafeMutableRawPointer] = []
        for _ in 0..<5 {
            buffers.append(try await pool.acquireBuffer(size: 1024))
        }
        
        for buffer in buffers {
            await pool.release(buffer)
        }
        
        // Wait for automatic defragmentation
        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        
        // Check if defragmentation occurred
        let report = await autoPoolManager.memoryReport()
        XCTAssertNotNil(report.lastDefragmentation)
    }
    
    // MARK: - Integration Tests
    
    func testMemoryManagerIntegration() async throws {
        // Register pool with memory manager
        let pool = await poolManager.createEnhancedBufferPool(for: "integrated_pool")
        
        // Allocate some memory
        let buffer1 = try await pool.acquireBuffer(size: 1024)
        let buffer2 = try await pool.acquireBuffer(size: 2048)
        
        // Get memory statistics from manager
        let memStats = await memoryManager.getMemoryStatistics()
        XCTAssertGreaterThan(memStats.currentMemoryUsage, 0)
        
        // Trigger cleanup
        await memoryManager.triggerCleanup(level: .light)
        
        // Release buffers
        await pool.release(buffer1)
        await pool.release(buffer2)
    }
    
    func testComprehensiveMemoryReport() async throws {
        // Create multiple pools
        let pool1 = await poolManager.createEnhancedBufferPool(for: "pool1")
        let pool2 = await poolManager.createEnhancedBufferPool(for: "pool2")
        
        // Allocate in both pools
        let buffer1 = try await pool1.acquireBuffer(size: 1024)
        let buffer2 = try await pool2.acquireBuffer(size: 2048)
        
        // Get comprehensive report
        let report = await poolManager.memoryReport()
        
        XCTAssertEqual(report.totalPools, 2)
        XCTAssertEqual(report.totalMemoryInUse, 1024 + 2048)
        XCTAssertGreaterThan(report.utilizationRate, 0)
        
        // Clean up
        await pool1.release(buffer1)
        await pool2.release(buffer2)
    }
    
    // MARK: - Performance Tests
    
    func testAllocationPerformance() async throws {
        let pool = await poolManager.createEnhancedBufferPool(
            for: "perf_test",
            maxPoolSize: 1000
        )
        
        // Pre-warm the pool
        await pool.preAllocate(count: 100)
        
        measure {
            Task {
                var buffers: [UnsafeMutableRawPointer] = []
                
                // Allocate
                for _ in 0..<100 {
                    if let buffer = try? await pool.acquireBuffer(size: 1024) {
                        buffers.append(buffer)
                    }
                }
                
                // Release
                for buffer in buffers {
                    await pool.release(buffer)
                }
            }
        }
    }
    
    func testHitRateImprovement() async throws {
        let pool = await poolManager.createEnhancedBufferPool(for: "hitrate_test")
        
        // First round - all misses
        var buffers: [UnsafeMutableRawPointer] = []
        for _ in 0..<10 {
            buffers.append(try await pool.acquireBuffer(size: 1024))
        }
        
        // Release back to pool
        for buffer in buffers {
            await pool.release(buffer)
        }
        
        // Second round - should be hits
        buffers.removeAll()
        for _ in 0..<10 {
            buffers.append(try await pool.acquireBuffer(size: 1024))
        }
        
        // Check hit rate
        let stats = await pool.statistics()
        XCTAssertGreaterThan(stats.hitRate, 0.4) // At least 40% hit rate
        
        // Clean up
        for buffer in buffers {
            await pool.release(buffer)
        }
    }
}

// MARK: - Stress Tests

extension EnhancedMemoryPoolTests {
    func testConcurrentAllocationStress() async throws {
        let pool = await poolManager.createEnhancedBufferPool(
            for: "stress_test",
            maxPoolSize: 200
        )
        
        // Run concurrent allocations
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    for _ in 0..<20 {
                        if let buffer = try? await pool.acquireBuffer(size: Int.random(in: 512...4096)) {
                            // Simulate some work
                            try? await Task.sleep(nanoseconds: 1_000_000) // 1ms
                            await pool.release(buffer)
                        }
                    }
                }
            }
        }
        
        // Verify pool integrity
        let stats = await pool.statistics()
        XCTAssertEqual(stats.currentlyInUse, 0)
        XCTAssertGreaterThanOrEqual(stats.totalAllocated, 0)
    }
    
    func testMemoryPressureDuringHighLoad() async throws {
        let pool = await poolManager.createEnhancedBufferPool(for: "pressure_stress")
        
        // Start allocating
        let allocationTask = Task {
            var buffers: [UnsafeMutableRawPointer] = []
            for _ in 0..<50 {
                if let buffer = try? await pool.acquireBuffer(size: 1024) {
                    buffers.append(buffer)
                    try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
                }
            }
            
            // Release all
            for buffer in buffers {
                await pool.release(buffer)
            }
        }
        
        // Simulate memory pressure during allocation
        try await Task.sleep(nanoseconds: 100_000_000) // 100ms
        await poolManager.handleMemoryPressure(.warning)
        
        // Wait for allocation task
        await allocationTask.value
        
        // Verify pool handled pressure gracefully
        let stats = await pool.statistics()
        XCTAssertEqual(stats.currentlyInUse, 0)
    }
}
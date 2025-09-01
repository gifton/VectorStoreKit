// MemoryPoolManagerTests.swift
// VectorStoreKit

import XCTest
@testable import VectorStoreKit

final class MemoryPoolManagerTests: XCTestCase {
    
    func testMemoryPoolManagerInitialization() async throws {
        let manager = MemoryPoolManager()
        
        // Test that we can register pools
        let objectPool = ObjectPool<String>(
            factory: { "test" },
            reset: nil,
            initialSize: 5,
            maxSize: 10
        )
        
        await manager.register(objectPool, for: "string_pool")
        
        // Verify the pool was registered
        let retrievedPool = await manager.getPool(for: "string_pool", type: String.self)
        XCTAssertNotNil(retrievedPool)
        
        // Get statistics
        let stats = await manager.allStatistics()
        XCTAssertEqual(stats.count, 1)
        XCTAssertNotNil(stats["string_pool"])
    }
    
    func testMemoryPressureMonitoring() async throws {
        let manager = MemoryPoolManager()
        
        // Setup some pools
        await manager.setupVectorPools()
        
        // Verify pools are created
        let stats = await manager.allStatistics()
        XCTAssertGreaterThan(stats.count, 0)
        
        // Test clearing all pools
        await manager.clearAll()
        
        // Verify memory usage is 0 after clearing
        let totalUsage = await manager.totalMemoryUsage()
        XCTAssertEqual(totalUsage, 0)
        
        // Stop monitoring
        await manager.stopMemoryPressureMonitoring()
    }
    
    func testArrayMemoryPool() async throws {
        let pool = ArrayMemoryPool<Float>(
            bucketSizes: [16, 32, 64],
            maxArraysPerBucket: 10
        )
        
        // Acquire arrays of different sizes
        let array1 = await pool.acquireWithCapacity(10)
        XCTAssertGreaterThanOrEqual(array1.capacity, 10)
        
        let array2 = await pool.acquireWithCapacity(25)
        XCTAssertGreaterThanOrEqual(array2.capacity, 25)
        
        let array3 = await pool.acquireWithCapacity(50)
        XCTAssertGreaterThanOrEqual(array3.capacity, 50)
        
        // Release arrays back to pool
        await pool.release(array1)
        await pool.release(array2)
        await pool.release(array3)
        
        // Check statistics
        let stats = await pool.statistics()
        XCTAssertEqual(stats.currentlyInUse, 0)
        XCTAssertGreaterThan(stats.hitRate, 0) // Should have hits on reuse
    }
    
    func testAlignedBufferPool() async throws {
        let pool = AlignedBufferPool(
            maxBuffersPerSize: 5,
            defaultAlignment: 64
        )
        
        // Acquire buffers
        let buffer1 = await pool.acquireBuffer(size: 1024, alignment: 64)
        let buffer2 = await pool.acquireBuffer(size: 4096, alignment: 128)
        
        // Verify alignment
        let address1 = Int(bitPattern: buffer1)
        let address2 = Int(bitPattern: buffer2)
        XCTAssertEqual(address1 % 64, 0)
        XCTAssertEqual(address2 % 128, 0)
        
        // Release buffers
        await pool.releaseBuffer(buffer1, size: 1024, alignment: 64)
        await pool.releaseBuffer(buffer2, size: 4096, alignment: 128)
        
        // Check statistics
        let stats = await pool.statistics()
        XCTAssertEqual(stats.currentlyInUse, 0)
    }
}
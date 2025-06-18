// BufferPoolTests.swift
// VectorStoreKit Tests
//
// Tests for the BufferPool actor to ensure correct buffer management

import XCTest
import Metal
@testable import VectorStoreKit

@MainActor
final class BufferPoolTests: XCTestCase {
    var device: MTLDevice!
    var pool: BufferPool!
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Get the default Metal device
        guard let metalDevice = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this device")
        }
        device = metalDevice
        pool = BufferPool(device: device)
    }
    
    override func tearDown() async throws {
        await pool.clear()
        pool = nil
        device = nil
        try await super.tearDown()
    }
    
    // MARK: - Basic Functionality Tests
    
    func testBufferAcquisition() async throws {
        // Test acquiring a buffer
        let buffer = try await pool.acquire(size: 1000)
        
        // Verify buffer properties
        XCTAssertGreaterThanOrEqual(buffer.count, 1000)
        XCTAssertEqual(buffer.count, 1024) // Should round up to 1024 (2^10)
        XCTAssertNotNil(buffer.buffer)
    }
    
    func testBufferReuse() async throws {
        // Acquire and release a buffer
        let buffer1 = try await pool.acquire(size: 1000)
        let bufferPtr1 = buffer1.buffer
        await pool.release(buffer1)
        
        // Acquire another buffer of the same size
        let buffer2 = try await pool.acquire(size: 1000)
        let bufferPtr2 = buffer2.buffer
        
        // Should get the same buffer back
        XCTAssertTrue(bufferPtr1 === bufferPtr2, "Buffer should be reused from pool")
        
        // Check statistics
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.reuseCount, 1)
        XCTAssertEqual(stats.allocationCount, 1)
    }
    
    func testPowerOfTwoRounding() async throws {
        // Test various sizes to ensure proper rounding
        let testCases: [(input: Int, expected: Int)] = [
            (1, 1),
            (2, 2),
            (3, 4),
            (5, 8),
            (15, 16),
            (17, 32),
            (100, 128),
            (500, 512),
            (1000, 1024),
            (2048, 2048)  // Already power of two
        ]
        
        for (input, expected) in testCases {
            let buffer = try await pool.acquire(size: input)
            XCTAssertEqual(buffer.count, expected, "Size \(input) should round to \(expected)")
            await pool.release(buffer)
        }
    }
    
    func testPoolSizeLimit() async throws {
        // Acquire and release more buffers than maxPoolSize (100)
        let bufferCount = 120
        var buffers: [MetalBuffer] = []
        
        // Acquire all buffers
        for _ in 0..<bufferCount {
            let buffer = try await pool.acquire(size: 512)
            buffers.append(buffer)
        }
        
        // Release all buffers
        for buffer in buffers {
            await pool.release(buffer)
        }
        
        // Check pool state
        let poolState = await pool.getPoolState()
        if let count = poolState[512] {
            XCTAssertLessThanOrEqual(count, 100, "Pool should not exceed maxPoolSize")
        }
        
        // Check statistics
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.pooledCount, 100)
        XCTAssertEqual(stats.discardedCount, 20)
    }
    
    func testMultipleSizeBuckets() async throws {
        // Create buffers of different sizes
        let sizes = [64, 128, 256, 512, 1024]
        var buffers: [(size: Int, buffer: MetalBuffer)] = []
        
        // Acquire buffers of different sizes
        for size in sizes {
            let buffer = try await pool.acquire(size: size)
            buffers.append((size, buffer))
        }
        
        // Release all buffers
        for (_, buffer) in buffers {
            await pool.release(buffer)
        }
        
        // Check pool state has multiple buckets
        let poolState = await pool.getPoolState()
        XCTAssertEqual(poolState.count, sizes.count)
        
        for size in sizes {
            XCTAssertEqual(poolState[size], 1, "Should have one buffer in bucket for size \(size)")
        }
    }
    
    func testClearPool() async throws {
        // Add some buffers to the pool
        for size in [128, 256, 512] {
            for _ in 0..<5 {
                let buffer = try await pool.acquire(size: size)
                await pool.release(buffer)
            }
        }
        
        // Verify buffers are pooled
        let statsBefore = await pool.getStatistics()
        XCTAssertGreaterThan(statsBefore.currentPooledCount, 0)
        
        // Clear the pool
        await pool.clear()
        
        // Verify pool is empty
        let statsAfter = await pool.getStatistics()
        XCTAssertEqual(statsAfter.currentPooledCount, 0)
        XCTAssertEqual(statsAfter.bucketCount, 0)
        
        let poolState = await pool.getPoolState()
        XCTAssertTrue(poolState.isEmpty)
    }
    
    // MARK: - Performance Tests
    
    func testBufferReusePerformance() async throws {
        // Measure performance with buffer reuse
        let iterations = 1000
        
        measure {
            let expectation = XCTestExpectation(description: "Buffer reuse performance")
            
            Task {
                for _ in 0..<iterations {
                    let buffer = try await pool.acquire(size: 1024)
                    await pool.release(buffer)
                }
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10.0)
        }
        
        // Check that most buffers were reused
        let stats = await pool.getStatistics()
        let reuseRate = stats.reuseRate
        XCTAssertGreaterThan(reuseRate, 95.0, "Reuse rate should be high for same-sized buffers")
    }
    
    func testStatistics() async throws {
        // Perform various operations
        let buffer1 = try await pool.acquire(size: 256)
        let buffer2 = try await pool.acquire(size: 256)
        await pool.release(buffer1)
        
        let buffer3 = try await pool.acquire(size: 256) // Should reuse buffer1
        await pool.release(buffer2)
        await pool.release(buffer3)
        
        // Check statistics
        let stats = await pool.getStatistics()
        
        XCTAssertEqual(stats.acquisitionCount, 3)
        XCTAssertEqual(stats.allocationCount, 2)
        XCTAssertEqual(stats.reuseCount, 1)
        XCTAssertEqual(stats.releaseCount, 3)
        XCTAssertEqual(stats.pooledCount, 3)
        XCTAssertEqual(stats.currentPooledCount, 2)
        
        // Check calculated rates
        XCTAssertEqual(stats.reuseRate, 100.0 / 3.0, accuracy: 0.01)
        XCTAssertEqual(stats.poolingEfficiency, 100.0, accuracy: 0.01)
    }
    
    // MARK: - Edge Cases
    
    func testZeroSizeRequest() async throws {
        // Should handle zero size gracefully by rounding up to 1
        let buffer = try await pool.acquire(size: 0)
        XCTAssertEqual(buffer.count, 1)
    }
    
    func testLargeSizeRequest() async throws {
        // Test with a large size
        let largeSize = 1_000_000
        let buffer = try await pool.acquire(size: largeSize)
        
        // Should round up to next power of two
        let expectedSize = 1_048_576 // 2^20
        XCTAssertEqual(buffer.count, expectedSize)
    }
    
    func testConcurrentAccess() async throws {
        // Test concurrent acquisitions and releases
        let concurrentTasks = 100
        
        await withTaskGroup(of: Void.self) { group in
            for i in 0..<concurrentTasks {
                group.addTask {
                    let size = 128 * (i % 4 + 1) // Vary sizes
                    if let buffer = try? await self.pool.acquire(size: size) {
                        // Simulate some work
                        try? await Task.sleep(nanoseconds: 1_000_000) // 1ms
                        await self.pool.release(buffer)
                    }
                }
            }
        }
        
        // Verify pool is still in a consistent state
        let stats = await pool.getStatistics()
        XCTAssertEqual(stats.acquisitionCount, concurrentTasks)
        XCTAssertEqual(stats.releaseCount, concurrentTasks)
    }
}
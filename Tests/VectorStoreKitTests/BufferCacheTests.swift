// BufferCacheTests.swift
// VectorStoreKit
//
// Tests for the BufferCache implementation to ensure proper memory management
// and LRU eviction policy functionality.

import XCTest
import Metal
@testable import VectorStoreKit

final class BufferCacheTests: XCTestCase {
    
    private var device: MTLDevice!
    private var cache: BufferCache!
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Create Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }
        self.device = device
        
        // Create cache with small memory limit for testing (10MB)
        self.cache = BufferCache(device: device, maxMemory: 10_485_760)
    }
    
    override func tearDown() async throws {
        await cache.clear()
        cache = nil
        device = nil
        try await super.tearDown()
    }
    
    // MARK: - Basic Functionality Tests
    
    func testStoreAndRetrieve() async throws {
        // Create a test buffer
        let buffer = try createTestBuffer(size: 1000)
        
        // Store buffer
        try await cache.store(buffer: buffer, for: "test_key")
        
        // Retrieve buffer
        let retrieved = await cache.retrieve(for: "test_key")
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved?.count, buffer.count)
    }
    
    func testRetrieveNonExistent() async throws {
        let retrieved = await cache.retrieve(for: "non_existent")
        XCTAssertNil(retrieved)
    }
    
    func testGetOrCreate() async throws {
        var createCalled = false
        
        // First call should create
        let buffer1 = try await cache.getOrCreate(key: "test_key") {
            createCalled = true
            return try createTestBuffer(size: 1000)
        }
        
        XCTAssertTrue(createCalled)
        XCTAssertEqual(buffer1.count, 1000)
        
        // Second call should retrieve from cache
        createCalled = false
        let buffer2 = try await cache.getOrCreate(key: "test_key") {
            createCalled = true
            return try createTestBuffer(size: 1000)
        }
        
        XCTAssertFalse(createCalled)
        XCTAssertEqual(buffer2.count, buffer1.count)
    }
    
    // MARK: - Memory Management Tests
    
    func testMemoryLimit() async throws {
        // Create buffers that total more than the cache limit
        // Each buffer is 1MB, cache limit is 10MB
        let bufferSize = 1_048_576 / MemoryLayout<Float>.stride // 1MB worth of floats
        
        // Store 12 buffers (should trigger eviction)
        for i in 0..<12 {
            let buffer = try createTestBuffer(size: bufferSize)
            try await cache.store(buffer: buffer, for: "buffer_\(i)")
        }
        
        // Check statistics
        let stats = await cache.getStatistics()
        XCTAssertLessThanOrEqual(stats.currentMemoryUsage, stats.memoryLimit)
        XCTAssertGreaterThan(stats.evictionCount, 0)
        
        // Early buffers should be evicted
        let firstBuffer = await cache.retrieve(for: "buffer_0")
        XCTAssertNil(firstBuffer)
        
        // Recent buffers should still be in cache
        let lastBuffer = await cache.retrieve(for: "buffer_11")
        XCTAssertNotNil(lastBuffer)
    }
    
    func testLRUEviction() async throws {
        // Create smaller buffers for controlled testing
        let bufferSize = 524_288 / MemoryLayout<Float>.stride // 512KB worth of floats
        
        // Fill cache close to limit
        for i in 0..<18 {
            let buffer = try createTestBuffer(size: bufferSize)
            try await cache.store(buffer: buffer, for: "buffer_\(i)")
        }
        
        // Access some buffers to update their access time
        _ = await cache.retrieve(for: "buffer_0")
        _ = await cache.retrieve(for: "buffer_1")
        _ = await cache.retrieve(for: "buffer_2")
        
        // Add more buffers to trigger eviction
        for i in 18..<22 {
            let buffer = try createTestBuffer(size: bufferSize)
            try await cache.store(buffer: buffer, for: "buffer_\(i)")
        }
        
        // Buffers 0, 1, 2 should still be in cache (recently accessed)
        XCTAssertNotNil(await cache.retrieve(for: "buffer_0"))
        XCTAssertNotNil(await cache.retrieve(for: "buffer_1"))
        XCTAssertNotNil(await cache.retrieve(for: "buffer_2"))
        
        // Buffers in the middle should be evicted (not recently accessed)
        XCTAssertNil(await cache.retrieve(for: "buffer_5"))
        XCTAssertNil(await cache.retrieve(for: "buffer_6"))
    }
    
    func testBufferTooLarge() async throws {
        // Try to store a buffer larger than the entire cache
        let hugeSize = 11_534_336 / MemoryLayout<Float>.stride // 11MB worth of floats
        let buffer = try createTestBuffer(size: hugeSize)
        
        do {
            try await cache.store(buffer: buffer, for: "huge_buffer")
            XCTFail("Should have thrown an error for oversized buffer")
        } catch BufferCacheError.bufferTooLarge {
            // Expected error
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
    
    // MARK: - Performance Tests
    
    func testCacheHitPerformance() async throws {
        // Store a buffer
        let buffer = try createTestBuffer(size: 10000)
        try await cache.store(buffer: buffer, for: "perf_test")
        
        // Measure cache hit performance
        let startTime = CFAbsoluteTimeGetCurrent()
        let iterations = 1000
        
        for _ in 0..<iterations {
            _ = await cache.retrieve(for: "perf_test")
        }
        
        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        let avgTime = elapsed / Double(iterations) * 1000 // Convert to milliseconds
        
        print("Average cache hit time: \(avgTime)ms")
        XCTAssertLessThan(avgTime, 0.1) // Should be less than 0.1ms
    }
    
    // MARK: - Statistics Tests
    
    func testStatistics() async throws {
        // Initial state
        var stats = await cache.getStatistics()
        XCTAssertEqual(stats.currentBufferCount, 0)
        XCTAssertEqual(stats.currentMemoryUsage, 0)
        
        // Store some buffers
        for i in 0..<5 {
            let buffer = try createTestBuffer(size: 1000)
            try await cache.store(buffer: buffer, for: "buffer_\(i)")
        }
        
        // Check hit/miss counts
        _ = await cache.retrieve(for: "buffer_0") // Hit
        _ = await cache.retrieve(for: "buffer_1") // Hit
        _ = await cache.retrieve(for: "non_existent") // Miss
        
        stats = await cache.getStatistics()
        XCTAssertEqual(stats.hitCount, 2)
        XCTAssertEqual(stats.missCount, 1)
        XCTAssertEqual(stats.currentBufferCount, 5)
        XCTAssertGreaterThan(stats.hitRate, 60) // Should be ~66%
    }
    
    func testMemoryPressureHandling() async throws {
        // Fill cache
        let bufferSize = 1_048_576 / MemoryLayout<Float>.stride // 1MB
        for i in 0..<10 {
            let buffer = try createTestBuffer(size: bufferSize)
            try await cache.store(buffer: buffer, for: "buffer_\(i)")
        }
        
        let statsBeforePressure = await cache.getStatistics()
        
        // Handle memory pressure
        await cache.handleMemoryPressure()
        
        let statsAfterPressure = await cache.getStatistics()
        
        // Should have reduced memory usage by approximately half
        XCTAssertLessThanOrEqual(
            statsAfterPressure.currentMemoryUsage,
            statsBeforePressure.currentMemoryUsage / 2 + 1_048_576 // Allow 1MB tolerance
        )
    }
    
    // MARK: - Helper Methods
    
    private func createTestBuffer(size: Int) throws -> MetalBuffer {
        let byteSize = size * MemoryLayout<Float>.stride
        guard let mtlBuffer = device.makeBuffer(length: byteSize, options: .storageModeShared) else {
            throw BufferCacheError.allocationFailed(reason: "Failed to create test buffer")
        }
        
        // Fill with test data
        let pointer = mtlBuffer.contents().bindMemory(to: Float.self, capacity: size)
        for i in 0..<size {
            pointer[i] = Float(i)
        }
        
        // Create MetalBuffer using the existing initializer
        return MetalBuffer(buffer: mtlBuffer, count: size)
    }
}

// MARK: - Edge Case Tests

extension BufferCacheTests {
    
    func testConcurrentAccess() async throws {
        // Test concurrent stores and retrieves
        await withTaskGroup(of: Void.self) { group in
            // Concurrent stores
            for i in 0..<100 {
                group.addTask {
                    let buffer = try! self.createTestBuffer(size: 100)
                    try! await self.cache.store(buffer: buffer, for: "concurrent_\(i)")
                }
            }
            
            // Concurrent retrieves
            for i in 0..<100 {
                group.addTask {
                    _ = await self.cache.retrieve(for: "concurrent_\(i)")
                }
            }
        }
        
        let stats = await cache.getStatistics()
        XCTAssertGreaterThan(stats.currentBufferCount, 0)
    }
    
    func testClearCache() async throws {
        // Store some buffers
        for i in 0..<10 {
            let buffer = try createTestBuffer(size: 1000)
            try await cache.store(buffer: buffer, for: "buffer_\(i)")
        }
        
        let statsBeforeClear = await cache.getStatistics()
        XCTAssertGreaterThan(statsBeforeClear.currentBufferCount, 0)
        
        // Clear cache
        await cache.clear()
        
        let statsAfterClear = await cache.getStatistics()
        XCTAssertEqual(statsAfterClear.currentBufferCount, 0)
        XCTAssertEqual(statsAfterClear.currentMemoryUsage, 0)
        
        // Verify buffers are gone
        for i in 0..<10 {
            XCTAssertNil(await cache.retrieve(for: "buffer_\(i)"))
        }
    }
}
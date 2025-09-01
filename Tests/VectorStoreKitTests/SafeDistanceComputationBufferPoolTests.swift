// SafeDistanceComputationBufferPoolTests.swift
// VectorStoreKit
//
// Tests for SafeDistanceComputationBufferPool MemoryPressureAware conformance

import XCTest
@testable import VectorStoreKit

final class SafeDistanceComputationBufferPoolTests: XCTestCase {
    
    func testMemoryPressureAwareConformance() async throws {
        // Create a buffer pool
        let pool = SafeDistanceComputationBufferPool(bufferSize: 1024, maxBuffers: 10)
        
        // Test normal pressure
        await pool.handleMemoryPressure(.normal)
        
        // Get memory usage
        let memoryUsage = await pool.getCurrentMemoryUsage()
        XCTAssertGreaterThan(memoryUsage, 0, "Memory usage should be greater than 0")
        
        // Get statistics
        let stats = await pool.getMemoryStatistics()
        XCTAssertEqual(stats.componentName, "SafeDistanceComputationBufferPool")
        XCTAssertEqual(stats.pressureEventCount, 1)
        XCTAssertNotNil(stats.lastPressureHandled)
        
        // Test warning pressure
        await pool.handleMemoryPressure(.warning)
        let statsAfterWarning = await pool.getMemoryStatistics()
        XCTAssertEqual(statsAfterWarning.pressureEventCount, 2)
        
        // Test critical pressure (should clear pool)
        await pool.handleMemoryPressure(.critical)
        let statsAfterCritical = await pool.getMemoryStatistics()
        XCTAssertEqual(statsAfterCritical.pressureEventCount, 3)
        
        // Memory usage should be minimal after critical pressure
        let memoryAfterCritical = await pool.getCurrentMemoryUsage()
        XCTAssertLessThanOrEqual(memoryAfterCritical, memoryUsage, "Memory usage should decrease after critical pressure")
    }
    
    func testBufferAcquisitionUnderPressure() async throws {
        let pool = SafeDistanceComputationBufferPool(bufferSize: 1024, maxBuffers: 5)
        
        // Acquire some buffers
        let buffer1 = await pool.acquireBuffer()
        let buffer2 = await pool.acquireBuffer()
        
        // Apply critical pressure
        await pool.handleMemoryPressure(.critical)
        
        // Buffers should still be valid
        XCTAssertEqual(buffer1.count, 1024)
        XCTAssertEqual(buffer2.count, 1024)
        
        // New buffers can still be acquired (but won't be pooled)
        let buffer3 = await pool.acquireBuffer()
        XCTAssertEqual(buffer3.count, 1024)
        
        // Release buffers (they won't be pooled under pressure)
        await pool.releaseBuffer(buffer1)
        await pool.releaseBuffer(buffer2)
        await pool.releaseBuffer(buffer3)
        
        // Return to normal pressure
        await pool.handleMemoryPressure(.normal)
        
        // Now buffers should be pooled again
        let buffer4 = await pool.acquireBuffer()
        await pool.releaseBuffer(buffer4)
        
        let poolStats = pool.statistics
        XCTAssertGreaterThan(poolStats.available, 0, "Pool should have available buffers after returning to normal pressure")
    }
}
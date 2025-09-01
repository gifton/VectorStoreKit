// RunCacheTests.swift
// Simple test runner to verify cache implementation

import XCTest
@testable import VectorStoreKit

class RunCacheTests: XCTestCase {
    
    func testFIFOTwoQueueBehaviorSimple() async throws {
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
        
        // Verify all are in cache
        let count1 = await cache.count
        XCTAssertEqual(count1, 5, "Should have 5 vectors in cache")
        
        // Add new normal priority - should evict from open queue (normal1 is oldest)
        await cache.set(id: "new1", vector: SIMD32<Float>(repeating: 6.0), priority: .normal)
        
        let critical1 = await cache.get(id: "critical1")
        let normal1 = await cache.get(id: "normal1")
        let normal2 = await cache.get(id: "normal2")
        let new1 = await cache.get(id: "new1")
        
        XCTAssertNotNil(critical1, "Protected item should remain")
        XCTAssertNil(normal1, "Oldest open queue item should be evicted")
        XCTAssertNotNil(normal2)
        XCTAssertNotNil(new1)
        
        print("✅ FIFO Two-Queue behavior test passed!")
    }
}
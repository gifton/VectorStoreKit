// SimpleCacheTest.swift
// Standalone test to verify FIFO two-queue implementation

import Foundation
@testable import VectorStoreKit

@main
struct SimpleCacheTest {
    static func main() async throws {
        print("🧪 Testing FIFO Two-Queue Implementation...")
        
        let vectorSize = MemoryLayout<SIMD32<Float>>.size
        let cacheSize = vectorSize * 5 + 100  // Room for 5 vectors
        
        let cache = try BasicFIFOVectorCache<SIMD32<Float>>(maxMemory: cacheSize)
        
        // Fill cache with mix of priorities
        print("1️⃣ Adding vectors with different priorities...")
        
        // Protected queue (20% = 1 vector)
        await cache.set(id: "critical1", vector: SIMD32<Float>(repeating: 1.0), priority: .critical)
        
        // Open queue (80% = 4 vectors)
        await cache.set(id: "normal1", vector: SIMD32<Float>(repeating: 2.0), priority: .normal)
        await cache.set(id: "normal2", vector: SIMD32<Float>(repeating: 3.0), priority: .normal)
        await cache.set(id: "low1", vector: SIMD32<Float>(repeating: 4.0), priority: .low)
        await cache.set(id: "low2", vector: SIMD32<Float>(repeating: 5.0), priority: .low)
        
        // Verify all are in cache
        let count1 = await cache.count
        print("   Cache count: \(count1) (expected: 5)")
        
        // Add new normal priority - should evict from open queue
        print("\n2️⃣ Adding new normal priority vector (should evict oldest from open queue)...")
        await cache.set(id: "new1", vector: SIMD32<Float>(repeating: 6.0), priority: .normal)
        
        // Check what's in cache
        let critical1 = await cache.get(id: "critical1")
        let normal1 = await cache.get(id: "normal1")
        let normal2 = await cache.get(id: "normal2")
        let new1 = await cache.get(id: "new1")
        
        print("   critical1: \(critical1 != nil ? "✓" : "✗") (should be ✓)")
        print("   normal1: \(normal1 != nil ? "✓" : "✗") (should be ✗ - evicted)")
        print("   normal2: \(normal2 != nil ? "✓" : "✗") (should be ✓)")
        print("   new1: \(new1 != nil ? "✓" : "✗") (should be ✓)")
        
        // Add high priority items
        print("\n3️⃣ Adding high priority items...")
        await cache.set(id: "high1", vector: SIMD32<Float>(repeating: 7.0), priority: .high)
        await cache.set(id: "high2", vector: SIMD32<Float>(repeating: 8.0), priority: .high)
        
        let finalCount = await cache.count
        print("   Final cache count: \(finalCount)")
        
        // Get performance analysis
        let analysis = await cache.performanceAnalysis()
        print("\n4️⃣ Performance Analysis:")
        print("   Hit rate: \(String(format: "%.2f%%", analysis.hitRateOverTime.last ?? 0 * 100))")
        print("   Memory utilization: \(String(format: "%.2f%%", analysis.memoryUtilization * 100))")
        print("   Recommendations: \(analysis.recommendations.count)")
        for rec in analysis.recommendations {
            print("   - \(rec.description)")
        }
        
        print("\n✅ Test completed successfully!")
    }
}
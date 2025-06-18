import Foundation
import VectorStoreKit

// Example demonstrating the VectorQuality caching functionality
@main
struct VectorQualityCacheExample {
    static func main() async {
        print("=== VectorQuality Cache Example ===\n")
        
        // Create test vectors
        let vector1 = SIMD8<Float>(1, 2, 3, 4, 5, 6, 7, 8)
        let vector2 = SIMD8<Float>(1, 2, 3, 4, 5, 6, 7, 8) // Same values as vector1
        let vector3 = SIMD8<Float>(8, 7, 6, 5, 4, 3, 2, 1) // Different values
        
        // Clear cache to start fresh
        await VectorQuality.clearCache()
        print("Cache cleared")
        
        // First assessment - will compute
        print("\n1. First assessment of vector1:")
        let start1 = Date()
        let quality1 = await VectorQuality.assessWithCache(vector1)
        let time1 = Date().timeIntervalSince(start1) * 1000 // Convert to milliseconds
        print("   Time: \(String(format: "%.3f", time1)) ms")
        print("   Magnitude: \(quality1.magnitude)")
        print("   Sparsity: \(quality1.sparsity)")
        
        // Second assessment of same vector - should hit cache
        print("\n2. Second assessment of same vector (cache hit):")
        let start2 = Date()
        let quality2 = await VectorQuality.assessWithCache(vector1)
        let time2 = Date().timeIntervalSince(start2) * 1000
        print("   Time: \(String(format: "%.3f", time2)) ms")
        print("   Speedup: \(String(format: "%.1fx", time1/time2))")
        
        // Assessment of vector with same values - should also hit cache
        print("\n3. Assessment of vector2 (same values as vector1):")
        let start3 = Date()
        let quality3 = await VectorQuality.assessWithCache(vector2)
        let time3 = Date().timeIntervalSince(start3) * 1000
        print("   Time: \(String(format: "%.3f", time3)) ms")
        print("   Cache hit: \(quality3.magnitude == quality1.magnitude)")
        
        // Assessment of different vector - will compute
        print("\n4. Assessment of vector3 (different values):")
        let start4 = Date()
        let quality4 = await VectorQuality.assessWithCache(vector3)
        let time4 = Date().timeIntervalSince(start4) * 1000
        print("   Time: \(String(format: "%.3f", time4)) ms")
        print("   Magnitude: \(quality4.magnitude)")
        
        // Demonstrate VectorEntry factory method
        print("\n5. Creating VectorEntry with cached assessment:")
        let entry = await VectorEntry.createWithCache(
            id: "example-1",
            vector: vector1,
            metadata: ["type": "example"]
        )
        print("   Entry created with ID: \(entry.id)")
        print("   Quality magnitude: \(entry.quality.magnitude)")
        
        // Show cache performance with multiple vectors
        print("\n6. Batch performance comparison:")
        let vectors = (0..<100).map { i in
            SIMD8<Float>(repeating: Float(i % 10)) // Only 10 unique vectors
        }
        
        // Without cache (using regular assess)
        let startNonCached = Date()
        for vector in vectors {
            _ = VectorQuality.assess(vector)
        }
        let timeNonCached = Date().timeIntervalSince(startNonCached) * 1000
        
        // With cache
        await VectorQuality.clearCache()
        let startCached = Date()
        for vector in vectors {
            _ = await VectorQuality.assessWithCache(vector)
        }
        let timeCached = Date().timeIntervalSince(startCached) * 1000
        
        print("   Non-cached (100 assessments): \(String(format: "%.1f", timeNonCached)) ms")
        print("   Cached (10 unique, 90 hits): \(String(format: "%.1f", timeCached)) ms")
        print("   Speedup: \(String(format: "%.1fx", timeNonCached/timeCached))")
        
        print("\n=== Example Complete ===")
    }
}
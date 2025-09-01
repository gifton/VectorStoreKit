import XCTest
@testable import VectorStoreKit

final class VectorQualityCacheTests: XCTestCase {
    
    func testQualityAssessmentCaching() async throws {
        // Create test vectors
        let vector1 = SIMD16<Float>(repeating: 1.0)
        let vector2 = SIMD16<Float>(repeating: 2.0)
        let vector3 = SIMD16<Float>(repeating: 1.0) // Same as vector1
        
        // Clear cache before test
        await VectorQuality.clearCache()
        
        // First assessment should compute
        let start1 = Date()
        let quality1 = await VectorQuality.assessWithCache(vector1)
        let time1 = Date().timeIntervalSince(start1)
        
        // Second assessment of same vector should be cached
        let start2 = Date()
        let quality2 = await VectorQuality.assessWithCache(vector1)
        let time2 = Date().timeIntervalSince(start2)
        
        // Cache hit should be much faster
        XCTAssert(time2 < time1 * 0.1, "Cache hit should be at least 10x faster")
        
        // Quality should be identical
        XCTAssertEqual(quality1.magnitude, quality2.magnitude)
        XCTAssertEqual(quality1.sparsity, quality2.sparsity)
        XCTAssertEqual(quality1.entropy, quality2.entropy)
        
        // Different vector should compute
        let quality3 = await VectorQuality.assessWithCache(vector2)
        XCTAssertNotEqual(quality1.magnitude, quality3.magnitude)
        
        // Same values but different instance should still hit cache
        let quality4 = await VectorQuality.assessWithCache(vector3)
        XCTAssertEqual(quality1.magnitude, quality4.magnitude)
    }
    
    func testVectorEntryFactoryMethod() async throws {
        let vector = SIMD8<Float>(1, 2, 3, 4, 5, 6, 7, 8)
        let metadata = ["key": "value"]
        
        // Create entry using cached assessment
        let entry = await VectorEntry.createWithCache(
            id: "test-1",
            vector: vector,
            metadata: metadata
        )
        
        XCTAssertEqual(entry.id, "test-1")
        XCTAssertEqual(entry.vector, vector)
        XCTAssertEqual(entry.metadata, metadata)
        XCTAssertNotNil(entry.quality)
        
        // Creating another entry with same vector should use cache
        let start = Date()
        let entry2 = await VectorEntry.createWithCache(
            id: "test-2",
            vector: vector,
            metadata: metadata
        )
        let elapsed = Date().timeIntervalSince(start)
        
        XCTAssert(elapsed < 0.001, "Cached creation should be very fast")
        XCTAssertEqual(entry.quality.magnitude, entry2.quality.magnitude)
    }
    
    func testCacheClearance() async throws {
        let vector = SIMD4<Float>(1, 2, 3, 4)
        
        // Assess and cache
        _ = await VectorQuality.assessWithCache(vector)
        
        // Clear cache
        await VectorQuality.clearCache()
        
        // Next assessment should recompute (we can't easily test timing here,
        // but we can verify it still works correctly)
        let quality = await VectorQuality.assessWithCache(vector)
        XCTAssertGreaterThan(quality.magnitude, 0)
    }
}
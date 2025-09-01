import XCTest
@testable import VectorStoreKit

final class OptimizedHNSWIndexTests: XCTestCase {
    
    struct TestMetadata: Codable, Sendable {
        let id: Int
        let description: String
    }
    
    func testOptimizedHNSWIndexCompiles() async throws {
        // Create a simple configuration
        let config = OptimizedHNSWIndex<SIMD8<Float>, TestMetadata>.Configuration(
            maxConnections: 16,
            efConstruction: 200
        )
        
        // Create the index
        let index = try OptimizedHNSWIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Create test data
        let vector = SIMD8<Float>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        let metadata = TestMetadata(id: 1, description: "Test vector")
        let entry = VectorEntry(id: "test1", vector: vector, metadata: metadata)
        
        // Test insert
        let result = try await index.insert(entry)
        XCTAssertTrue(result.success)
        
        // Test search
        let searchResults = try await index.search(
            query: vector,
            k: 1,
            strategy: .approximate
        )
        XCTAssertFalse(searchResults.isEmpty)
    }
}
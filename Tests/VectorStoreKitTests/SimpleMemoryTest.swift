import XCTest
@testable import VectorStoreKit

final class SimpleMemoryTest: XCTestCase {
    
    func testHNSWMemoryUsage() async throws {
        // Create a simple HNSW index
        let config = HNSWIndex<SIMD8<Float>, String>.Configuration()
        let index = try HNSWIndex<SIMD8<Float>, String>(configuration: config)
        
        // Check initial memory
        let initialMemory = await index.memoryUsage
        print("Initial memory usage: \(initialMemory) bytes")
        XCTAssertGreaterThan(initialMemory, 0)
        
        // Add a vector
        let vector = SIMD8<Float>(1, 2, 3, 4, 5, 6, 7, 8)
        let entry = VectorEntry(id: "test1", vector: vector, metadata: "Hello World")
        _ = try await index.insert(entry)
        
        // Check memory after insertion
        let afterInsertMemory = await index.memoryUsage
        print("Memory after 1 insert: \(afterInsertMemory) bytes")
        XCTAssertGreaterThan(afterInsertMemory, initialMemory)
        
        // Add more vectors
        for i in 2...10 {
            let v = SIMD8<Float>(repeating: Float(i))
            let e = VectorEntry(id: "test\(i)", vector: v, metadata: "Metadata \(i)")
            _ = try await index.insert(e)
        }
        
        // Check final memory
        let finalMemory = await index.memoryUsage
        print("Memory after 10 inserts: \(finalMemory) bytes")
        XCTAssertGreaterThan(finalMemory, afterInsertMemory)
        
        // Calculate average memory per node
        let nodeCount = await index.count
        let avgMemory = finalMemory / nodeCount
        print("Average memory per node: \(avgMemory) bytes")
        
        // Should be reasonable - at least vector size + some overhead
        XCTAssertGreaterThan(avgMemory, MemoryLayout<SIMD8<Float>>.size)
    }
}
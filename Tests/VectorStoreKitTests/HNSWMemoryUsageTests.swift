// VectorStoreKit: HNSW Memory Usage Tests
//
// Tests for proper memory usage calculation in HNSW index

import XCTest
@testable import VectorStoreKit

final class HNSWMemoryUsageTests: XCTestCase {
    
    func testMemoryUsageCalculation() async throws {
        // Create a simple HNSW index
        let config = HNSWIndex<SIMD8<Float>, String>.Configuration(
            maxConnections: 16,
            efConstruction: 200,
            enableAnalytics: true
        )
        
        let index = try HNSWIndex<SIMD8<Float>, String>(configuration: config)
        
        // Get initial memory usage
        let initialMemory = await index.memoryUsage
        XCTAssertGreaterThan(initialMemory, 0, "Initial memory usage should be greater than 0")
        
        // Add some vectors
        let vector1 = SIMD8<Float>(1, 2, 3, 4, 5, 6, 7, 8)
        let entry1 = VectorEntry(id: "vec1", vector: vector1, metadata: "First vector")
        _ = try await index.insert(entry1)
        
        let memoryAfterOne = await index.memoryUsage
        XCTAssertGreaterThan(memoryAfterOne, initialMemory, "Memory should increase after insertion")
        
        // Add more vectors to test connection memory
        for i in 2...10 {
            let vector = SIMD8<Float>(
                Float(i), Float(i+1), Float(i+2), Float(i+3),
                Float(i+4), Float(i+5), Float(i+6), Float(i+7)
            )
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: vector,
                metadata: "Vector \(i)"
            )
            _ = try await index.insert(entry)
        }
        
        let finalMemory = await index.memoryUsage
        XCTAssertGreaterThan(finalMemory, memoryAfterOne, "Memory should increase with more vectors")
        
        // Verify memory calculation includes various components
        let stats = await index.statistics()
        XCTAssertEqual(stats.memoryUsage, finalMemory, "Statistics should report same memory usage")
        
        // Test memory calculation components
        let nodeCount = await index.count
        let avgMemoryPerNode = finalMemory / max(nodeCount, 1)
        
        // Each node should use at least the size of the vector + some overhead
        let minExpectedPerNode = MemoryLayout<SIMD8<Float>>.size + 100 // Vector + minimal overhead
        XCTAssertGreaterThan(avgMemoryPerNode, minExpectedPerNode, 
                            "Average memory per node should exceed minimum expected")
    }
    
    func testMemoryUsageWithDifferentMetadataSizes() async throws {
        let config = HNSWIndex<SIMD4<Float>, HNSWTestMetadata>.Configuration(
            maxConnections: 8,
            efConstruction: 100,
            enableAnalytics: false // Reduce analytics memory
        )
        
        let index = try HNSWIndex<SIMD4<Float>, HNSWTestMetadata>(configuration: config)
        
        // Add vector with small metadata
        let smallMetadata = HNSWTestMetadata(data: "small")
        let entry1 = VectorEntry(
            id: "small",
            vector: SIMD4<Float>(1, 2, 3, 4),
            metadata: smallMetadata
        )
        _ = try await index.insert(entry1)
        let memorySmall = await index.memoryUsage
        
        // Add vector with large metadata
        let largeData = String(repeating: "x", count: 1000)
        let largeMetadata = HNSWTestMetadata(data: largeData)
        let entry2 = VectorEntry(
            id: "large",
            vector: SIMD4<Float>(5, 6, 7, 8),
            metadata: largeMetadata
        )
        _ = try await index.insert(entry2)
        let memoryLarge = await index.memoryUsage
        
        // Memory should increase by at least the difference in metadata size
        let increase = memoryLarge - memorySmall
        XCTAssertGreaterThan(increase, 900, "Memory increase should reflect larger metadata")
    }
}

// Test metadata structure
private struct HNSWTestMetadata: Codable, Sendable {
    let data: String
}
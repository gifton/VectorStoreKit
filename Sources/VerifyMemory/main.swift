import VectorStoreKit
import Foundation

//@main
struct VerifyMemory {
    static func main() async throws {
        print("Testing HNSW Memory Usage Calculation")
        print("=====================================")
        
        // Create a simple HNSW index
        let config = HNSWIndex<SIMD8<Float>, String>.Configuration(
            maxConnections: 16,
            efConstruction: 200,
            enableAnalytics: true
        )
        
        let index = try HNSWIndex<SIMD8<Float>, String>(configuration: config)
        
        // Check initial memory
        let initialMemory = await index.memoryUsage
        print("\nInitial memory usage: \(formatBytes(initialMemory))")
        
        // Add first vector
        let vector1 = SIMD8<Float>(1, 2, 3, 4, 5, 6, 7, 8)
        let entry1 = VectorEntry(id: "vec1", vector: vector1, metadata: "First vector")
        _ = try await index.insert(entry1)
        
        let memoryAfter1 = await index.memoryUsage
        print("\nAfter 1 vector:")
        print("  Total memory: \(formatBytes(memoryAfter1))")
        print("  Increase: \(formatBytes(memoryAfter1 - initialMemory))")
        
        // Add 9 more vectors
        for i in 2...10 {
            let vector = SIMD8<Float>(
                Float(i), Float(i+1), Float(i+2), Float(i+3),
                Float(i+4), Float(i+5), Float(i+6), Float(i+7)
            )
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: vector,
                metadata: "Vector \(i) with some metadata"
            )
            _ = try await index.insert(entry)
        }
        
        let finalMemory = await index.memoryUsage
        let nodeCount = await index.count
        let avgMemoryPerNode = finalMemory / nodeCount
        
        print("\nAfter \(nodeCount) vectors:")
        print("  Total memory: \(formatBytes(finalMemory))")
        print("  Total increase: \(formatBytes(finalMemory - initialMemory))")
        print("  Average per node: \(formatBytes(avgMemoryPerNode))")
        
        // Get detailed statistics
        let stats = await index.statistics()
        print("\nDetailed Statistics:")
        print("  Vector count: \(stats.vectorCount)")
        print("  Memory usage: \(formatBytes(stats.memoryUsage))")
        print("  Layers: \(stats.layers)")
        print("  Avg connections: \(stats.averageConnections)")
        
        // Test with larger metadata
        print("\n\nTesting with larger metadata...")
        let largeMetadata = String(repeating: "X", count: 1000)
        let largeEntry = VectorEntry(
            id: "large",
            vector: SIMD8<Float>(9, 9, 9, 9, 9, 9, 9, 9),
            metadata: largeMetadata
        )
        
        let memoryBeforeLarge = await index.memoryUsage
        _ = try await index.insert(largeEntry)
        let memoryAfterLarge = await index.memoryUsage
        
        print("Large metadata increase: \(formatBytes(memoryAfterLarge - memoryBeforeLarge))")
        print("(Metadata was \(largeMetadata.count) characters)")
    }
    
    static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .memory
        return formatter.string(fromByteCount: Int64(bytes))
    }
}

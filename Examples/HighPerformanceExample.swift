// High Performance VectorStoreKit Example
//
// Demonstrates integration of new high-performance components with VectorStore API

import Foundation
import VectorStoreKit
import Metal

@main
struct HighPerformanceExample {
    static func main() async throws {
        print("=== VectorStoreKit High Performance Example ===\n")
        
        // 1. Create optimized VectorStore for 512-dimensional vectors
        print("1. Creating optimized VectorStore for 512-dimensional vectors...")
        let store = try await createOptimizedStore()
        
        // 2. Demonstrate batch processing
        print("\n2. Batch processing example...")
        try await demonstrateBatchProcessing(store: store)
        
        // 3. Demonstrate hierarchical indexing
        print("\n3. Hierarchical indexing example...")
        try await demonstrateHierarchicalIndexing()
        
        // 4. Demonstrate quantization
        print("\n4. Quantization example...")
        try await demonstrateQuantization()
        
        // 5. Performance comparison
        print("\n5. Performance comparison...")
        try await performanceComparison()
        
        print("\n✅ High performance example complete!")
    }
    
    // MARK: - 1. Create Optimized Store
    
    static func createOptimizedStore() async throws -> VectorStore<Vector512, DocumentMetadata, HNSWIndex<Vector512, DocumentMetadata>, InMemoryStorage, BasicLRUVectorCache<Vector512>> {
        // Use the fluent API with optimizations for Vector512
        let universe = VectorUniverse<Vector512, DocumentMetadata>
            .optimized512(accelerator: .metal)
            .indexing(HNSWIndexingStrategy<Vector512, DocumentMetadata>(
                maxConnections: 32,
                efConstruction: 400,
                useAdaptiveTuning: true
            ))
            .storage(InMemoryPerformanceStorageStrategy())
            .caching(LRUCachingStrategy(maxMemory: 500_000_000))
        
        let store = try await universe.materialize()
        
        print("✓ Created optimized VectorStore")
        print("  - Vector dimension: 512")
        print("  - Index: HNSW (M=32, ef=400)")
        print("  - Storage: In-memory")
        print("  - Cache: LRU with 500MB limit")
        print("  - Acceleration: Metal GPU")
        
        return store
    }
    
    // MARK: - 2. Batch Processing
    
    static func demonstrateBatchProcessing(store: VectorStore<Vector512, DocumentMetadata, HNSWIndex<Vector512, DocumentMetadata>, InMemoryStorage, BasicLRUVectorCache<Vector512>>) async throws {
        print("Generating 10,000 vectors...")
        
        // Create a large dataset
        let vectors = (0..<10_000).map { i in
            VectorEntry(
                id: "doc_\(i)",
                vector: Vector512(repeating: Float(i) / 10_000.0),
                metadata: DocumentMetadata(
                    title: "Document \(i)",
                    content: "Content for document \(i)",
                    timestamp: Date()
                )
            )
        }
        
        // Add vectors using batch processing
        let startTime = Date()
        
        let result = try await store.addBatch(
            vectors,
            batchSize: 1000  // Optimal batch size for Vector512
        )
        
        let elapsed = Date().timeIntervalSince(startTime)
        let throughput = Double(vectors.count) / elapsed
        
        print("✓ Batch processing complete")
        print("  - Vectors added: \(result.insertedCount)")
        print("  - Time elapsed: \(String(format: "%.2f", elapsed))s")
        print("  - Throughput: \(String(format: "%.0f", throughput)) vectors/sec")
        print("  - Errors: \(result.errorCount)")
    }
    
    // MARK: - 3. Hierarchical Indexing
    
    static func demonstrateHierarchicalIndexing() async throws {
        print("Creating hierarchical index for 1M+ vectors...")
        
        // Create a universe with hierarchical indexing
        let universe = VectorUniverse<Vector512, DocumentMetadata>
            .optimized512()
            .indexing(HierarchicalIndexingStrategy<Vector512, DocumentMetadata>())
            .storage(InMemoryPerformanceStorageStrategy())
            .withoutCache()
        
        let store = try await universe.materialize()
        
        // Add some test vectors
        let testVectors = (0..<1000).map { i in
            VectorEntry(
                id: "hier_\(i)",
                vector: generateRandomVector512(seed: i),
                metadata: DocumentMetadata(
                    title: "Hierarchical \(i)",
                    content: "Test content",
                    timestamp: Date()
                )
            )
        }
        
        _ = try await store.add(testVectors)
        
        // Perform search
        let query = generateRandomVector512(seed: 42)
        let results = try await store.search(
            query: query,
            k: 10,
            strategy: .hybrid
        )
        
        print("✓ Hierarchical indexing demonstrated")
        print("  - Index type: Two-level (IVF + HNSW)")
        print("  - Vectors indexed: 1000")
        print("  - Search results: \(results.results.count)")
        print("  - Query time: \(String(format: "%.3f", results.queryTime))s")
    }
    
    // MARK: - 4. Quantization
    
    static func demonstrateQuantization() async throws {
        print("Demonstrating scalar quantization...")
        
        let quantizer = ScalarQuantizer()
        
        // Generate test vectors
        let vectors = (0..<100).map { i in
            (0..<512).map { j in
                sin(Float(i * 512 + j) / 100.0)
            }
        }
        
        // Test different quantization types
        let types: [(ScalarQuantizationType, String)] = [
            (.int8, "INT8"),
            (.uint8, "UINT8"),
            (.float16, "FLOAT16")
        ]
        
        for (type, name) in types {
            let startTime = Date()
            
            // Quantize
            let quantized = try await quantizer.quantizeBatch(
                vectors: vectors,
                type: type
            )
            
            // Calculate metrics
            let compressionRatio = quantized.first?.compressionRatio ?? 0
            let quantizeTime = Date().timeIntervalSince(startTime)
            
            // Dequantize to measure accuracy
            let dequantStart = Date()
            let restored = try await quantizer.dequantizeBatch(quantized)
            let dequantTime = Date().timeIntervalSince(dequantStart)
            
            // Calculate error
            var maxError: Float = 0
            for (original, restored) in zip(vectors, restored) {
                for (o, r) in zip(original, restored) {
                    maxError = max(maxError, abs(o - r))
                }
            }
            
            print("\n  \(name) Quantization:")
            print("    - Compression ratio: \(String(format: "%.1fx", compressionRatio))")
            print("    - Quantize time: \(String(format: "%.3f", quantizeTime))s")
            print("    - Dequantize time: \(String(format: "%.3f", dequantTime))s")
            print("    - Max error: \(String(format: "%.6f", maxError))")
        }
    }
    
    // MARK: - 5. Performance Comparison
    
    static func performanceComparison() async throws {
        print("Comparing standard vs optimized implementations...")
        
        let vectorCount = 10_000
        let queryCount = 100
        
        // Generate test data
        let vectors = (0..<vectorCount).map { i in
            generateRandomVector512(seed: i)
        }
        
        let queries = (0..<queryCount).map { i in
            generateRandomVector512(seed: vectorCount + i)
        }
        
        // Standard implementation (simulated)
        print("\nStandard Implementation:")
        let standardStart = Date()
        var standardDistances = 0
        
        for query in queries {
            for vector in vectors {
                // Simulate standard distance computation
                var sum: Float = 0
                for i in 0..<512 {
                    let diff = query[i] - vector[i]
                    sum += diff * diff
                }
                standardDistances += 1
            }
        }
        
        let standardTime = Date().timeIntervalSince(standardStart)
        let standardOps = Double(standardDistances) / standardTime
        
        // Optimized implementation
        print("\nOptimized Implementation (Vector512):")
        let optimizedStart = Date()
        var optimizedDistances = 0
        
        for query in queries {
            for vector in vectors {
                _ = DistanceComputation512.euclideanDistance(query, vector)
                optimizedDistances += 1
            }
        }
        
        let optimizedTime = Date().timeIntervalSince(optimizedStart)
        let optimizedOps = Double(optimizedDistances) / optimizedTime
        
        // Results
        print("\n📊 Performance Results:")
        print("  Standard implementation:")
        print("    - Time: \(String(format: "%.3f", standardTime))s")
        print("    - Operations/sec: \(String(format: "%.0f", standardOps))")
        
        print("\n  Optimized implementation:")
        print("    - Time: \(String(format: "%.3f", optimizedTime))s")
        print("    - Operations/sec: \(String(format: "%.0f", optimizedOps))")
        
        print("\n  Speedup: \(String(format: "%.1fx", optimizedOps / standardOps))")
    }
    
    // MARK: - Helper Functions
    
    static func generateRandomVector512(seed: Int) -> Vector512 {
        var vector = Vector512()
        
        // Use seed for reproducibility
        var state = UInt64(seed)
        
        for i in 0..<512 {
            // Simple linear congruential generator
            state = state &* 1664525 &+ 1013904223
            let normalized = Float(state % 1000000) / 1000000.0
            vector[i] = normalized * 2.0 - 1.0  // Range [-1, 1]
        }
        
        // Normalize to unit vector
        let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        if magnitude > 0 {
            for i in 0..<512 {
                vector[i] /= magnitude
            }
        }
        
        return vector
    }
}

// MARK: - Document Metadata

struct DocumentMetadata: Codable, Sendable {
    let title: String
    let content: String
    let timestamp: Date
}

// MARK: - Extensions for Demo

extension VectorEntry {
    var tier: StorageTier {
        .hot  // Default to hot tier for demo
    }
}
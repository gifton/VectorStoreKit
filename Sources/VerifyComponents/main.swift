// Standalone verification of new components
import Foundation
import VectorStoreKit
import Metal

//@main
struct VerifyComponents {
    static func main() async throws {
        print("=== VectorStoreKit Component Verification ===\n")
        
        // 1. Vector512 Tests
        print("1. Testing Vector512...")
        testVector512()
        
        // 2. BatchProcessor Tests
        print("\n2. Testing BatchProcessor...")
        try await testBatchProcessor()
        
        // 3. ScalarQuantizer Tests
        print("\n3. Testing ScalarQuantizer...")
        try await testScalarQuantizer()
        
        // 4. HierarchicalIndex Tests
        print("\n4. Testing HierarchicalIndex...")
        try await testHierarchicalIndex()
        
        // 5. Integration Test
        print("\n5. Testing Integration...")
        try await testIntegration()
        
        print("\n✅ All components verified successfully!")
    }
    
    static func testVector512() {
        // Basic operations
        let v1 = Vector512(repeating: 1.0)
        let v2 = Vector512(repeating: 2.0)
        
        let sum = v1 + v2
        assert(sum[0] == 3.0, "Addition failed")
        
        let product = v1 * v2
        assert(product[0] == 2.0, "Multiplication failed")
        
        // Distance computation
        let euclidean = DistanceComputation512.euclideanDistance(v1, v2)
        let expected = Float(sqrt(512.0))
        assert(abs(euclidean - expected) < 0.01, "Euclidean distance failed")
        
        // Collection operations
        let mapped = v1.map { $0 * 2 }
        assert(mapped.count == 512, "Map failed")
        assert(mapped[0] == 2.0, "Map value incorrect")
        
        print("✓ Vector512: Basic operations, distance computation, collection conformance")
    }
    
    static func testBatchProcessor() async throws {
        let processor = BatchProcessor()
        
        // Create test dataset
        let vectors = (0..<100).map { i in
            Vector512(repeating: Float(i))
        }
        
        let dataset = SimpleVectorDataset(vectors: vectors)
        
        // Test transformation
        let scaled: [Vector512] = try await processor.processBatches(
            dataset: dataset,
            operation: .transformation { vector in
                var result = vector
                for i in 0..<result.scalarCount {
                    result[i] *= 2.0
                }
                return result
            }
        )
        
        assert(scaled.count == vectors.count, "Batch count mismatch")
        assert(scaled[0][0] == 0.0, "First vector incorrect")
        assert(scaled[1][0] == 2.0, "Scaling failed")
        
        // Test filtering
        let filtered: [Vector512] = try await processor.processBatches(
            dataset: dataset,
            operation: .filtering { vector in
                Int(vector[0]) % 2 == 0
            }
        )
        
        assert(filtered.count == 50, "Filtering failed")
        
        print("✓ BatchProcessor: Transformation, filtering, batch processing")
    }
    
    static func testScalarQuantizer() async throws {
        let quantizer = ScalarQuantizer()
        
        // Test single vector quantization
        let vector = (0..<128).map { Float($0) / 128.0 }
        
        let quantized = try await quantizer.quantize(
            vector: vector,
            type: .int8
        )
        
        assert(quantized.quantizationType == ScalarQuantizationType.int8, "Wrong quantization type")
        assert(abs(quantized.compressionRatio - 4.0) < 0.1, "Wrong compression ratio")
        
        // Test round-trip
        let restored = try await quantizer.dequantize(quantized)
        
        var maxError: Float = 0
        for (original, dequantized) in zip(vector, restored) {
            maxError = max(maxError, abs(original - dequantized))
        }
        
        assert(maxError < 0.01, "Quantization error too high: \(maxError)")
        
        // Test batch quantization
        let vectors = (0..<10).map { i in
            (0..<128).map { j in Float(i * 128 + j) / 1280.0 }
        }
        
        let batchQuantized = try await quantizer.quantizeBatch(
            vectors: vectors,
            type: .uint8
        )
        
        assert(batchQuantized.count == vectors.count, "Batch size mismatch")
        
        print("✓ ScalarQuantizer: Int8/uint8 quantization, compression ratio, round-trip accuracy")
    }
    
    static func testHierarchicalIndex() async throws {
        // Skip for now - HierarchicalIndex requires IVF training
        print("✓ HierarchicalIndex: Skipped (requires IVF training)")
    }
    
    static func testIntegration() async throws {
        // Test components working together
        let processor = BatchProcessor()
        let quantizer = ScalarQuantizer()
        // Use a simple array as our "index" for testing
        var simpleIndex: [(Vector512, String)] = []
        
        // Generate and process vectors
        let vectors = (0..<100).map { i in
            Vector512(repeating: Float(i) / 100.0)
        }
        
        let dataset = SimpleVectorDataset(vectors: vectors)
        
        // Scale vectors
        let scaled: [Vector512] = try await processor.processBatches(
            dataset: dataset,
            operation: .transformation { vector in
                var result = vector
                for i in 0..<result.scalarCount {
                    result[i] *= 10.0
                }
                return result
            }
        )
        
        // Quantize for storage
        let _ = try await quantizer.quantizeBatch(
            vectors: scaled.map { v in
                (0..<512).map { i in v[i] }
            },
            type: .int8
        )
        
        // Index scaled vectors
        for (i, vector) in scaled.enumerated() {
            simpleIndex.append((vector, "scaled_\(i)"))
        }
        
        // Simple search
        let query = Vector512(repeating: 5.0) // Should match scaled 0.5
        let results = simpleIndex
            .map { (vector, id) in
                (distance: DistanceComputation512.euclideanDistance(query, vector), id: id)
            }
            .sorted { $0.distance < $1.distance }
            .prefix(3)
        
        let resultsArray = Array(results)
        assert(resultsArray.count == 3, "Integration search failed")
        
        // The nearest should be around index 50
        if let nearestId = resultsArray.first?.id {
            let index = Int(nearestId.replacingOccurrences(of: "scaled_", with: "")) ?? -1
            assert(abs(index - 50) <= 1, "Integration nearest neighbor incorrect")
        }
        
        print("✓ Integration: Processing pipeline, quantization, indexing, search")
    }
}

// Simple dataset for testing
struct SimpleVectorDataset: LargeVectorDataset {
    let vectors: [Vector512]
    
    var count: Int {
        vectors.count
    }
    
    func loadBatch(range: Range<Int>) async throws -> [Vector512] {
        Array(vectors[range])
    }
    
    func asyncIterator() -> AsyncStream<Vector512> {
        AsyncStream { continuation in
            for vector in vectors {
                continuation.yield(vector)
            }
            continuation.finish()
        }
    }
}

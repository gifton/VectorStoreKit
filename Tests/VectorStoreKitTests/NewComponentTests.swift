// NewComponentTests.swift
// VectorStoreKitTests
//
// Simple test runner for new components

import XCTest
@testable import VectorStoreKit
import Metal

final class NewComponentTests: XCTestCase {
    
    // MARK: - Vector512 Basic Tests
    
    func testVector512BasicOperations() {
        // Test initialization
        let v1 = Vector512(repeating: 1.0)
        XCTAssertEqual(v1[0], 1.0)
        XCTAssertEqual(v1[511], 1.0)
        
        // Test arithmetic
        let v2 = Vector512(repeating: 2.0)
        let sum = v1 + v2
        XCTAssertEqual(sum[0], 3.0)
        
        // Test collection conformance
        let mapped = v1.map { $0 * 2 }
        XCTAssertEqual(mapped.count, 512)
        XCTAssertEqual(mapped[0], 2.0)
    }
    
    func testVector512DistanceComputation() {
        let v1 = Vector512(repeating: 1.0)
        let v2 = Vector512(repeating: 2.0)
        
        // Euclidean distance
        let euclidean = DistanceComputation512.euclideanDistance(v1, v2)
        let expected = sqrt(512.0) // sqrt(512 * (2-1)^2)
        XCTAssertEqual(euclidean, expected, accuracy: 0.01)
        
        // Dot product
        let dot = DistanceComputation512.dotProduct(v1, v2)
        XCTAssertEqual(dot, 1024.0, accuracy: 0.01) // 512 * 1 * 2
    }
    
    // MARK: - BatchProcessor Basic Tests
    
    func testBatchProcessorConfiguration() {
        let config = BatchProcessingConfiguration()
        
        // Test batch size calculation
        let smallVectorBatch = config.optimalBatchSize(forVectorSize: 128)
        let largeVectorBatch = config.optimalBatchSize(forVectorSize: 2048)
        
        XCTAssertGreaterThan(smallVectorBatch, largeVectorBatch)
    }
    
    func testBatchProcessorSimpleOperation() async throws {
        let processor = BatchProcessor()
        
        // Create simple dataset
        let vectors = (0..<10).map { i in
            Vector512(repeating: Float(i))
        }
        
        let dataset = SimpleDataset(items: vectors)
        
        // Test identity transformation
        let identityProcessor: @Sendable ([Vector512]) async throws -> [Vector512] = { $0 }
        
        let results = try await processor.processBatches(
            dataset: dataset,
            processor: identityProcessor
        )
        
        XCTAssertEqual(results.count, vectors.count)
        
        // Verify values preserved
        for (i, result) in results.enumerated() {
            XCTAssertEqual(result[0], Float(i))
        }
    }
    
    // MARK: - ScalarQuantizer Basic Tests
    
    func testScalarQuantizerInt8() async throws {
        let quantizer = ScalarQuantizer()
        
        let vector = (0..<100).map { Float($0) / 100.0 }
        
        // Quantize
        let quantized = try await quantizer.quantize(
            vector,
            type: .int8,
            statistics: nil
        )
        
        XCTAssertEqual(quantized.quantizationType, .int8)
        XCTAssertEqual(quantized.originalDimension, vector.count)
        XCTAssertEqual(quantized.compressionRatio, 4.0, accuracy: 0.1)
        
        // Dequantize
        let restored = try await quantizer.dequantize(quantized)
        
        // Check accuracy
        for (original, dequantized) in zip(vector, restored) {
            XCTAssertEqual(original, dequantized, accuracy: 0.01)
        }
    }
    
    func testScalarQuantizerStatistics() async throws {
        let quantizer = ScalarQuantizer()
        
        let vectors = [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 5.0]
        ].map { $0.map { Float($0) } }
        
        let stats = try await quantizer.computeStatistics(for: vectors)
        
        XCTAssertEqual(stats.min, 0.0)
        XCTAssertEqual(stats.max, 5.0)
        XCTAssertEqual(stats.mean, 2.5, accuracy: 0.01)
    }
    
    // MARK: - StreamingBufferManager Basic Tests
    
    func testMemoryMappedFile() throws {
        // Create temporary file
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString + ".bin")
        
        let testData = Data(repeating: 42, count: 1024)
        try testData.write(to: tempURL)
        
        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }
        
        // Test memory mapping
        let mmFile = try MemoryMappedFile(url: tempURL)
        let region = try mmFile.mapRegion(offset: 0, length: 100)
        
        XCTAssertEqual(region.count, 100)
        XCTAssertEqual(region[0], 42)
    }
    
    // MARK: - HierarchicalIndex Basic Tests
    
    func testHierarchicalConfiguration() {
        let small = HierarchicalConfiguration.forDatasetSize(10_000)
        XCTAssertEqual(small.topLevelClusters, 100)
        
        let large = HierarchicalConfiguration.forDatasetSize(1_000_000)
        XCTAssertEqual(large.topLevelClusters, 1024)
    }
    
    func testHierarchicalIndexCreation() async throws {
        let config = HierarchicalConfiguration(
            topLevelClusters: 10,
            leafIndexSize: 100
        )
        
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: config
        )
        
        let stats = await index.getStatistics()
        XCTAssertEqual(stats.totalVectors, 0)
    }
    
    // MARK: - Integration Test
    
    func testBasicIntegration() async throws {
        // This tests that all components can work together
        let processor = BatchProcessor()
        let quantizer = ScalarQuantizer()
        
        // Generate test vectors
        let vectors = (0..<100).map { i in
            Vector512(repeating: Float(i) / 100.0)
        }
        
        // Process and quantize
        let dataset = SimpleDataset(items: vectors)
        
        let transformed: [Vector512] = try await processor.processBatches(
            dataset: dataset,
            operation: .transformation { vector in
                // Scale vectors
                var scaled = vector
                for i in 0..<scaled.scalarCount {
                    scaled[i] *= 2.0
                }
                return scaled
            }
        )
        
        // Quantize transformed vectors
        let quantizedVectors = try await quantizer.quantizeBatch(
            vectors: transformed.map { vector in
                (0..<512).map { i in vector[i] }
            },
            type: .int8
        )
        
        XCTAssertEqual(quantizedVectors.count, vectors.count)
        
        // Verify compression
        for q in quantizedVectors {
            XCTAssertEqual(q.compressionRatio, 4.0, accuracy: 0.1)
        }
    }
}

// MARK: - Test Helpers

struct SimpleDataset<T: Sendable>: LargeVectorDataset {
    let items: [T]
    
    var count: Int {
        items.count
    }
    
    func loadBatch(range: Range<Int>) async throws -> [T] {
        Array(items[range])
    }
    
    func asyncIterator() -> AsyncStream<T> {
        AsyncStream { continuation in
            for item in items {
                continuation.yield(item)
            }
            continuation.finish()
        }
    }
}
// ScalarQuantizerTests.swift
// VectorStoreKitTests
//
// Tests for scalar quantization

import XCTest
@testable import VectorStoreKit
import Metal
import Accelerate

final class ScalarQuantizerTests: XCTestCase {
    
    var quantizer: ScalarQuantizer!
    
    override func setUp() async throws {
        try await super.setUp()
        quantizer = ScalarQuantizer()
    }
    
    // MARK: - Statistics Tests
    
    func testComputeStatistics() async throws {
        let vectors = [
            [Float](arrayLiteral: 0.0, 1.0, 2.0, 3.0, 4.0),
            [Float](arrayLiteral: -1.0, 0.0, 1.0, 2.0, 3.0),
            [Float](arrayLiteral: 1.0, 2.0, 3.0, 4.0, 5.0)
        ]
        
        let stats = try await quantizer.computeStatistics(for: vectors)
        
        XCTAssertEqual(stats.min, -1.0)
        XCTAssertEqual(stats.max, 5.0)
        XCTAssertEqual(stats.mean, 2.0, accuracy: 0.01)
        XCTAssertGreaterThan(stats.stdDev, 0)
        
        // Test recommended type
        XCTAssertEqual(stats.recommendedType, .int8)
    }
    
    func testStatisticsForNonNegativeData() async throws {
        let vectors = [
            [Float](arrayLiteral: 0.0, 1.0, 2.0, 3.0),
            [Float](arrayLiteral: 1.0, 2.0, 3.0, 4.0),
            [Float](arrayLiteral: 2.0, 3.0, 4.0, 5.0)
        ]
        
        let stats = try await quantizer.computeStatistics(for: vectors)
        
        XCTAssertEqual(stats.min, 0.0)
        XCTAssertEqual(stats.recommendedType, .uint8)
    }
    
    // MARK: - Quantization Tests
    
    func testInt8Quantization() async throws {
        let vector = [Float](arrayLiteral: -1.0, -0.5, 0.0, 0.5, 1.0)
        
        let quantized = try await quantizer.quantize(
            vector,
            type: .int8,
            statistics: nil
        )
        
        XCTAssertEqual(quantized.quantizationType, .int8)
        XCTAssertEqual(quantized.originalDimension, vector.count)
        XCTAssertEqual(quantized.quantizedData.count, vector.count) // 1 byte per value
        
        // Test round-trip
        let dequantized = try await quantizer.dequantize(quantized)
        
        for (original, restored) in zip(vector, dequantized) {
            XCTAssertEqual(original, restored, accuracy: 0.1) // Some quantization error expected
        }
    }
    
    func testUInt8Quantization() async throws {
        let vector = [Float](arrayLiteral: 0.0, 0.25, 0.5, 0.75, 1.0)
        
        let quantized = try await quantizer.quantize(
            vector,
            type: .uint8,
            statistics: nil
        )
        
        XCTAssertEqual(quantized.quantizationType, .uint8)
        XCTAssertEqual(quantized.quantizedData.count, vector.count)
        
        // Test round-trip
        let dequantized = try await quantizer.dequantize(quantized)
        
        for (original, restored) in zip(vector, dequantized) {
            XCTAssertEqual(original, restored, accuracy: 0.01)
        }
    }
    
    func testFloat16Quantization() async throws {
        let vector = [Float](arrayLiteral: -100.5, -10.25, 0.0, 10.25, 100.5)
        
        let quantized = try await quantizer.quantize(
            vector,
            type: .float16,
            statistics: nil
        )
        
        XCTAssertEqual(quantized.quantizationType, .float16)
        XCTAssertEqual(quantized.quantizedData.count, vector.count * 2) // 2 bytes per value
        
        // Test round-trip
        let dequantized = try await quantizer.dequantize(quantized)
        
        for (original, restored) in zip(vector, dequantized) {
            XCTAssertEqual(original, restored, accuracy: 0.01) // Float16 has good precision
        }
    }
    
    func testDynamicQuantization() async throws {
        // Test with different value ranges
        let testCases: [([Float], ScalarQuantizationType)] = [
            // Non-negative small range -> uint8
            ([0.0, 1.0, 2.0, 3.0], .uint8),
            // Mixed sign small range -> int8
            ([-1.0, -0.5, 0.0, 0.5, 1.0], .int8),
            // Large range -> float16
            ([-1000.0, -100.0, 0.0, 100.0, 1000.0], .float16)
        ]
        
        for (vector, expectedType) in testCases {
            let quantized = try await quantizer.quantize(
                vector,
                type: .dynamic,
                statistics: nil
            )
            
            XCTAssertEqual(quantized.quantizationType, expectedType)
            
            // Verify round-trip
            let dequantized = try await quantizer.dequantize(quantized)
            for (original, restored) in zip(vector, dequantized) {
                XCTAssertEqual(original, restored, accuracy: 0.1)
            }
        }
    }
    
    // MARK: - Batch Processing Tests
    
    func testBatchQuantization() async throws {
        let vectors = (0..<10).map { i in
            (0..<128).map { j in
                Float(i * 128 + j) / 1000.0
            }
        }
        
        let quantizedBatch = try await quantizer.quantizeBatch(
            vectors: vectors,
            type: .int8
        )
        
        XCTAssertEqual(quantizedBatch.count, vectors.count)
        
        // Verify all have same parameters
        let firstScale = quantizedBatch[0].scale
        let firstOffset = quantizedBatch[0].offset
        
        for quantized in quantizedBatch {
            XCTAssertEqual(quantized.scale, firstScale)
            XCTAssertEqual(quantized.offset, firstOffset)
        }
    }
    
    func testBatchDequantization() async throws {
        let vectors = (0..<5).map { i in
            (0..<64).map { j in
                Float(i * 64 + j) / 100.0
            }
        }
        
        let quantizedBatch = try await quantizer.quantizeBatch(
            vectors: vectors,
            type: .uint8
        )
        
        let dequantizedBatch = try await quantizer.dequantizeBatch(quantizedBatch)
        
        XCTAssertEqual(dequantizedBatch.count, vectors.count)
        
        // Verify accuracy
        for (original, restored) in zip(vectors, dequantizedBatch) {
            for (o, r) in zip(original, restored) {
                XCTAssertEqual(o, r, accuracy: 0.01)
            }
        }
    }
    
    // MARK: - Error Handling Tests
    
    func testInconsistentDimensions() async throws {
        let vectors = [
            [Float](arrayLiteral: 1.0, 2.0, 3.0),
            [Float](arrayLiteral: 4.0, 5.0) // Different dimension
        ]
        
        do {
            _ = try await quantizer.quantizeBatch(vectors: vectors, type: .int8)
            XCTFail("Should throw inconsistentDimensions error")
        } catch ScalarQuantizationError.inconsistentDimensions {
            // Expected
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }
    
    // MARK: - Compression Tests
    
    func testCompressionRatio() async throws {
        let vector = (0..<1024).map { Float($0) }
        
        let originalSize = vector.count * MemoryLayout<Float>.size
        
        // Test different quantization types
        let types: [ScalarQuantizationType] = [.int8, .uint8, .float16]
        
        for type in types {
            let quantized = try await quantizer.quantize(vector, type: type)
            let compressedSize = quantized.quantizedData.count
            let ratio = Float(originalSize) / Float(compressedSize)
            
            switch type {
            case .int8, .uint8:
                XCTAssertEqual(ratio, 4.0, accuracy: 0.1)
            case .float16:
                XCTAssertEqual(ratio, 2.0, accuracy: 0.1)
            case .dynamic:
                break // Variable ratio
            }
            
            XCTAssertEqual(quantized.compressionRatio, ratio, accuracy: 0.1)
        }
    }
    
    // MARK: - Analysis Tests
    
    func testQuantizationAnalysis() async throws {
        let vectors = (0..<100).map { i in
            (0..<128).map { j in
                Float(i + j) + Float.random(in: -0.1...0.1)
            }
        }
        
        let analysis = try await quantizer.analyzeQuantization(
            vectors: vectors,
            type: .int8
        )
        
        XCTAssertGreaterThan(analysis.averageError, 0)
        XCTAssertLessThan(analysis.averageError, 1.0) // Should be small
        XCTAssertGreaterThan(analysis.maxError, analysis.averageError)
        XCTAssertGreaterThan(analysis.compressionRatio, 3.0)
        XCTAssertLessThan(analysis.signalToNoiseRatio, 100.0)
        
        // Distribution should sum to approximately 1.0
        let distributionSum = analysis.errorDistribution.reduce(0, +)
        XCTAssertEqual(distributionSum, 1.0, accuracy: 0.01)
    }
    
    // MARK: - Performance Tests
    
    func testQuantizationPerformance() throws {
        let vectors = (0..<1000).map { _ in
            (0..<512).map { _ in Float.random(in: -1...1) }
        }
        
        measure {
            let expectation = self.expectation(description: "Quantization complete")
            
            Task {
                _ = try await quantizer.quantizeBatch(
                    vectors: vectors,
                    type: .int8
                )
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 5.0)
        }
    }
    
    func testDequantizationPerformance() async throws {
        let vectors = (0..<1000).map { _ in
            (0..<512).map { _ in Float.random(in: -1...1) }
        }
        
        let quantizedBatch = try await quantizer.quantizeBatch(
            vectors: vectors,
            type: .uint8
        )
        
        measure {
            let expectation = self.expectation(description: "Dequantization complete")
            
            Task {
                _ = try await quantizer.dequantizeBatch(quantizedBatch)
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 5.0)
        }
    }
}
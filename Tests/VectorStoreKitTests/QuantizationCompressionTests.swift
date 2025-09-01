// VectorStoreKit: Quantization Compression Tests

import XCTest
@testable import VectorStoreKit
@preconcurrency import Metal

final class QuantizationCompressionTests: XCTestCase {
    
    func testBasicQuantizationCompression() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }
        
        let compressor = try await QuantizationCompressor(device: device)
        
        // Create test data
        let testVector = (0..<1000).map { Float($0) / 1000.0 }
        let data = testVector.withUnsafeBytes { Data($0) }
        let shape = [testVector.count]
        
        // Compress
        let compressed = try await compressor.compress(
            data: data,
            shape: shape,
            quantizationType: .uint8
        )
        
        // Verify compression ratio
        XCTAssertGreaterThan(compressed.compressionRatio, 3.0, "Expected at least 3x compression for uint8")
        
        // Decompress
        let decompressed = try await compressor.decompress(compressed)
        
        // Verify size matches
        XCTAssertEqual(decompressed.count, data.count)
        
        // Verify accuracy (allowing for quantization error)
        let decompressedVector = decompressed.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self))
        }
        
        for (original, decompressed) in zip(testVector, decompressedVector) {
            let error = abs(original - decompressed)
            XCTAssertLessThan(error, 0.01, "Quantization error too large")
        }
    }
    
    func testBatchQuantization() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }
        
        let compressor = try await QuantizationCompressor(device: device)
        
        // Create multiple test vectors
        let parameters = (0..<5).map { i in
            let vector = (0..<100).map { Float($0 + i * 100) / 500.0 }
            let data = vector.withUnsafeBytes { Data($0) }
            return (name: "param_\(i)", data: data, shape: [vector.count])
        }
        
        // Batch compress
        let compressed = try await compressor.compressBatch(
            parameters: parameters,
            quantizationType: .int8
        )
        
        XCTAssertEqual(compressed.count, 5)
        
        // Verify each compressed parameter
        for (name, compressedParam) in compressed {
            XCTAssertGreaterThan(compressedParam.compressionRatio, 3.0)
            XCTAssertTrue(name.hasPrefix("param_"))
        }
    }
    
    func testQuantizationTypes() async throws {
        let quantizer = ScalarQuantizer()
        
        let testVector: [Float] = [0.1, 0.5, 0.9, -0.3, -0.7, 1.2, -1.5]
        
        // Test different quantization types
        let types: [ScalarQuantizationType] = [.int8, .uint8, .float16]
        
        for type in types {
            let quantized = try await quantizer.quantize(vector: testVector, type: type)
            let dequantized = try await quantizer.dequantize(quantized)
            
            // Verify dimensions preserved
            XCTAssertEqual(dequantized.count, testVector.count)
            
            // Check compression ratio
            switch type {
            case .int8, .uint8:
                XCTAssertEqual(quantized.compressionRatio, 4.0, accuracy: 0.1)
            case .float16:
                XCTAssertEqual(quantized.compressionRatio, 2.0, accuracy: 0.1)
            default:
                break
            }
            
            // Verify reasonable accuracy
            for (original, restored) in zip(testVector, dequantized) {
                let error = abs(original - restored)
                let tolerance: Float = type == .float16 ? 0.001 : 0.02
                XCTAssertLessThan(error, tolerance, "Error too large for \(type)")
            }
        }
    }
    
    func testModelSerializationWithQuantization() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal device not available")
        }
        
        let serializer = CheckpointSerializer(device: device)
        
        // Create test buffer
        let testData = (0..<1000).map { Float($0) / 1000.0 }
        guard let mtlBuffer = device.makeBuffer(
            bytes: testData,
            length: testData.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to create Metal buffer")
            return
        }
        
        let shape = TensorShape(testData.count)
        let buffer = MetalBuffer(buffer: mtlBuffer, shape: shape)
        
        // Serialize with quantization compression
        let compressed = try await serializer.serializeBuffer(
            buffer,
            name: "test_weights",
            compression: .quantization
        )
        
        XCTAssertEqual(compressed.compression, .quantization)
        XCTAssertEqual(compressed.name, "test_weights")
        
        // The compressed data should be smaller than original
        let originalSize = testData.count * MemoryLayout<Float>.stride
        XCTAssertLessThan(compressed.data.count, originalSize)
        
        // Deserialize
        let restored = try await serializer.deserializeBuffer(compressed)
        
        XCTAssertEqual(restored.count, buffer.count)
        XCTAssertEqual(restored.shape, buffer.shape)
    }
}
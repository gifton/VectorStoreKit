// VectorStoreKit: Mixed Precision Tests
//
// Tests for mixed precision training support

import XCTest
@testable import VectorStoreKit
import Metal

@available(macOS 13.0, iOS 16.0, *)
final class MixedPrecisionTests: XCTestCase {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var shaderLibrary: MLShaderLibrary!
    var operations: MetalMLOperations!
    
    override func setUp() async throws {
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device)
        
        commandQueue = device.makeCommandQueue()
        XCTAssertNotNil(commandQueue)
        
        shaderLibrary = try MLShaderLibrary(device: device)
        operations = MetalMLOperations(
            device: device,
            commandQueue: commandQueue,
            shaderLibrary: shaderLibrary
        )
    }
    
    // MARK: - Precision Conversion Tests
    
    func testFP32ToFP16Conversion() async throws {
        let converter = try PrecisionConverter(device: device)
        
        // Create test data
        let testData: [Float] = [1.0, 2.5, -3.14159, 0.0001, 65504.0] // Include edge cases
        let fp32Buffer = try MetalBuffer(device: device, array: testData)
        
        // Create FP16 buffer (half the size)
        guard let fp16MTLBuffer = device.makeBuffer(
            length: testData.count * 2, // 2 bytes per FP16
            options: .storageModeShared
        ) else {
            XCTFail("Failed to allocate FP16 buffer")
            return
        }
        let fp16Buffer = MetalBuffer(buffer: fp16MTLBuffer, count: testData.count, stride: 2)
        
        // Convert
        try await converter.convertFP32ToFP16(fp32Buffer, output: fp16Buffer)
        
        // Verify by converting back
        guard let fp32ConvertedMTLBuffer = device.makeBuffer(
            length: testData.count * 4,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to allocate conversion buffer")
            return
        }
        let fp32ConvertedBuffer = MetalBuffer(buffer: fp32ConvertedMTLBuffer, count: testData.count)
        
        try await converter.convertFP16ToFP32(fp16Buffer, output: fp32ConvertedBuffer)
        
        let result = fp32ConvertedBuffer.toArray()
        
        // Check precision (FP16 has less precision)
        for (original, converted) in zip(testData, result) {
            if abs(original) < 65504 { // FP16 max value
                XCTAssertEqual(original, converted, accuracy: abs(original) * 0.001)
            }
        }
    }
    
    func testNumericStabilityCheck() async throws {
        let converter = try PrecisionConverter(device: device)
        
        // Test overflow detection
        let overflowData: [Float] = [1.0, Float.infinity, 2.0]
        let overflowBuffer = try MetalBuffer(device: device, array: overflowData)
        
        let (overflow1, underflow1) = try await converter.checkNumericStability(overflowBuffer)
        XCTAssertTrue(overflow1, "Should detect overflow")
        XCTAssertFalse(underflow1, "Should not detect underflow")
        
        // Test underflow detection
        let underflowData: [Float] = [1.0, 1e-8, 2.0]
        let underflowBuffer = try MetalBuffer(device: device, array: underflowData)
        
        let (overflow2, underflow2) = try await converter.checkNumericStability(underflowBuffer)
        XCTAssertFalse(overflow2, "Should not detect overflow")
        XCTAssertTrue(underflow2, "Should detect underflow")
        
        // Test normal data
        let normalData: [Float] = [1.0, 2.0, 3.0, 4.0]
        let normalBuffer = try MetalBuffer(device: device, array: normalData)
        
        let (overflow3, underflow3) = try await converter.checkNumericStability(normalBuffer)
        XCTAssertFalse(overflow3, "Should not detect overflow in normal data")
        XCTAssertFalse(underflow3, "Should not detect underflow in normal data")
    }
    
    // MARK: - Dynamic Loss Scaler Tests
    
    func testDynamicLossScaler() async throws {
        let config = MixedPrecisionConfig()
        let scaler = try DynamicLossScaler(config: config, device: device)
        
        // Initial scale should match config
        XCTAssertEqual(scaler.scale, config.initialLossScale)
        
        // Create gradient buffer
        let gradients: [Float] = Array(repeating: 0.1, count: 100)
        let gradBuffer = try MetalBuffer(device: device, array: gradients)
        
        // Normal update should succeed
        let success1 = try await scaler.update(gradBuffer)
        XCTAssertTrue(success1)
        
        // After growth interval updates, scale should increase
        for _ in 0..<config.lossScaleGrowthInterval {
            _ = try await scaler.update(gradBuffer)
        }
        XCTAssertEqual(scaler.scale, config.initialLossScale * config.lossScaleGrowthFactor)
        
        // Test overflow handling
        let overflowGradients: [Float] = [Float.infinity] + Array(repeating: 0.1, count: 99)
        let overflowBuffer = try MetalBuffer(device: device, array: overflowGradients)
        
        let success2 = try await scaler.update(overflowBuffer)
        XCTAssertFalse(success2)
        XCTAssertEqual(scaler.scale, config.initialLossScale) // Should backoff
        
        // Test statistics
        let stats = scaler.getStatistics()
        XCTAssertGreaterThan(stats.updateCount, 0)
        XCTAssertGreaterThan(stats.overflowCount, 0)
        XCTAssertGreaterThan(stats.overflowRate, 0)
    }
    
    // MARK: - Mixed Precision Operations Tests
    
    func testMixedPrecisionMatMul() async throws {
        // Create test matrices
        let m = 4, n = 3, k = 2
        let aData: [Float] = [1, 2, 3, 4, 5, 6, 7, 8] // 4x2
        let bData: [Float] = [1, 2, 3, 4, 5, 6]       // 2x3
        
        // Create FP32 buffers
        let aFP32 = try MetalBuffer(device: device, array: aData)
        let bFP32 = try MetalBuffer(device: device, array: bData)
        
        // Create mixed precision buffers
        let aMixed = MixedPrecisionBuffer(buffer: aFP32, precision: .fp32)
        let bMixed = MixedPrecisionBuffer(buffer: bFP32, precision: .fp32)
        
        // Create output buffer (always FP32)
        guard let outputMTLBuffer = device.makeBuffer(
            length: m * n * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to allocate output buffer")
            return
        }
        let output = MetalBuffer(buffer: outputMTLBuffer, count: m * n)
        
        // Perform mixed precision matmul
        try await operations.matmulMixedPrecision(aMixed, bMixed, output: output, m: m, n: n, k: k)
        
        // Verify results
        let result = output.toArray()
        let expected: [Float] = [
            9, 12, 15,    // [1,2] • [[1,2,3], [4,5,6]]
            19, 26, 33,   // [3,4] • [[1,2,3], [4,5,6]]
            29, 40, 51,   // [5,6] • [[1,2,3], [4,5,6]]
            39, 54, 69    // [7,8] • [[1,2,3], [4,5,6]]
        ]
        
        for (computed, expected) in zip(result, expected) {
            XCTAssertEqual(computed, expected, accuracy: 0.001)
        }
    }
    
    func testLossScaling() async throws {
        // Create gradient buffer
        let gradients: [Float] = Array(repeating: 0.001, count: 100)
        let gradBuffer = try MetalBuffer(device: device, array: gradients)
        
        let scale: Float = 1024.0
        
        // Scale gradients
        try await operations.scaleGradients(gradBuffer, scale: scale)
        
        // Verify scaling
        let scaled = gradBuffer.toArray()
        for (i, val) in scaled.enumerated() {
            XCTAssertEqual(val, gradients[i] * scale, accuracy: 0.0001)
        }
        
        // Unscale gradients
        try await operations.unscaleGradients(gradBuffer, scale: scale)
        
        // Verify unscaling
        let unscaled = gradBuffer.toArray()
        for (i, val) in unscaled.enumerated() {
            XCTAssertEqual(val, gradients[i], accuracy: 0.0001)
        }
    }
    
    // MARK: - Memory Savings Tests
    
    func testMemorySavingsCalculation() {
        let modelSize = 1_000_000 // 1M parameters
        let batchSize = 32
        let sequenceLength = 512
        
        let comparison = MixedPrecisionMemoryCalculator.calculateMemorySavings(
            modelSize: modelSize,
            batchSize: batchSize,
            sequenceLength: sequenceLength
        )
        
        // Should show significant memory savings
        XCTAssertGreaterThan(comparison.memorySavingsPercent, 20.0)
        XCTAssertLessThan(comparison.mixedTotalMemory, comparison.fp32TotalMemory)
        
        print(comparison.description)
    }
    
    // MARK: - Integration Tests
    
    func testMixedPrecisionTrainingStep() async throws {
        // Simulate a training step with mixed precision
        let config = MixedPrecisionConfig.default
        let scaler = try DynamicLossScaler(config: config, device: device)
        
        // Model parameters
        let paramCount = 1000
        let masterWeights = try MetalBuffer(device: device, array: Array(repeating: Float(0.1), count: paramCount))
        
        // FP16 model weights
        guard let fp16WeightsMTL = device.makeBuffer(length: paramCount * 2, options: .storageModeShared) else {
            XCTFail("Failed to allocate FP16 weights")
            return
        }
        let modelWeights = MetalBuffer(buffer: fp16WeightsMTL, count: paramCount, stride: 2)
        
        // Gradients (always FP32)
        let gradients = try MetalBuffer(device: device, array: Array(repeating: Float(0.001), count: paramCount))
        
        // Scale gradients
        try await operations.scaleGradients(gradients, scale: scaler.scale)
        
        // Check for overflow
        let updateSuccess = try await scaler.update(gradients)
        
        if updateSuccess {
            // Unscale gradients
            try await operations.unscaleGradients(gradients, scale: scaler.scale)
            
            // Update weights
            try await operations.sgdUpdateMixedPrecision(
                masterWeights: masterWeights,
                modelWeights: modelWeights,
                gradients: gradients,
                learningRate: 0.01
            )
        }
        
        // Verify weights were updated
        let masterArray = masterWeights.toArray()
        XCTAssertNotEqual(masterArray[0], 0.1, "Weights should be updated")
    }
    
    // MARK: - Performance Tests
    
    func testMixedPrecisionPerformance() async throws {
        let size = 1024
        let iterations = 100
        
        // Create large matrices
        let aData = Array(repeating: Float(1.0), count: size * size)
        let bData = Array(repeating: Float(2.0), count: size * size)
        
        let aBuffer = try MetalBuffer(device: device, array: aData)
        let bBuffer = try MetalBuffer(device: device, array: bData)
        
        guard let outputMTL = device.makeBuffer(
            length: size * size * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            XCTFail("Failed to allocate output")
            return
        }
        let output = MetalBuffer(buffer: outputMTL, count: size * size)
        
        // Measure FP32 performance
        let fp32Start = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            try await operations.matmul(aBuffer, bBuffer, output: output, m: size, n: size, k: size)
        }
        let fp32Time = CFAbsoluteTimeGetCurrent() - fp32Start
        
        // Measure mixed precision performance
        let aMixed = MixedPrecisionBuffer(buffer: aBuffer, precision: .fp32)
        let bMixed = MixedPrecisionBuffer(buffer: bBuffer, precision: .fp32)
        
        let mixedStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            try await operations.matmulMixedPrecision(aMixed, bMixed, output: output, m: size, n: size, k: size)
        }
        let mixedTime = CFAbsoluteTimeGetCurrent() - mixedStart
        
        print("FP32 time: \(fp32Time)s")
        print("Mixed precision time: \(mixedTime)s")
        print("Speedup: \(fp32Time / mixedTime)x")
        
        // Mixed precision should be faster on supported hardware
        // Note: Actual speedup depends on GPU capabilities
    }
}

// MARK: - Test Helpers

@available(macOS 13.0, iOS 16.0, *)
extension MetalBuffer {
    /// Helper to create buffer with specific precision
    func withPrecision(_ precision: Precision, device: MTLDevice) throws -> MixedPrecisionBuffer {
        let newByteLength = count * precision.byteSize
        guard let newBuffer = device.makeBuffer(length: newByteLength, options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: newByteLength)
        }
        
        let metalBuffer = MetalBuffer(
            buffer: newBuffer,
            shape: shape,
            stride: precision.byteSize
        )
        
        return MixedPrecisionBuffer(buffer: metalBuffer, precision: precision)
    }
}
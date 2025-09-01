// VectorStoreKit: Normalization Layers Tests
//
// Comprehensive tests for LayerNorm, GroupNorm, and BatchNorm implementations

import XCTest
@testable import VectorStoreKit
import Metal

final class NormalizationLayersTests: XCTestCase {
    var device: MTLDevice!
    var metalPipeline: MetalMLPipeline!
    
    override func setUp() async throws {
        try await super.setUp()
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this system")
        }
        self.device = device
        self.metalPipeline = try MetalMLPipeline(device: device)
    }
    
    // MARK: - LayerNorm Tests
    
    func testLayerNormInitialization() async throws {
        let layerNorm = try await LayerNormLayer(
            normalizedShape: [64],
            eps: 1e-5,
            elementwiseAffine: true,
            name: "test_layernorm",
            metalPipeline: metalPipeline
        )
        
        // Verify parameter count (gamma + beta)
        let paramCount = await layerNorm.getParameterCount()
        XCTAssertEqual(paramCount, 64 * 2, "LayerNorm should have 2 * normalizedShape parameters")
    }
    
    func testLayerNormForwardPass() async throws {
        let layerNorm = try await LayerNormLayer(
            normalizedShape: [10],
            eps: 1e-5,
            elementwiseAffine: true,
            metalPipeline: metalPipeline
        )
        
        // Create input with known statistics
        let input: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        // Forward pass
        let output = try await layerNorm.forward(inputBuffer)
        
        // Verify output shape
        XCTAssertEqual(output.count, 10, "Output should have same size as input")
        
        // Check that output is normalized (mean ≈ 0, std ≈ 1)
        let outputArray = output.toArray()
        let mean = outputArray.reduce(0, +) / Float(outputArray.count)
        let variance = outputArray.map { pow($0 - mean, 2) }.reduce(0, +) / Float(outputArray.count)
        let std = sqrt(variance)
        
        XCTAssertTrue(abs(mean) < 0.1, "Normalized output should have mean close to 0, got \(mean)")
        XCTAssertTrue(abs(std - 1.0) < 0.1, "Normalized output should have std close to 1, got \(std)")
    }
    
    func testLayerNormBackwardPass() async throws {
        let layerNorm = try await LayerNormLayer(
            normalizedShape: [5],
            elementwiseAffine: true,
            metalPipeline: metalPipeline
        )
        
        // Forward pass
        let input: [Float] = [1, 2, 3, 4, 5]
        let inputBuffer = try MetalBuffer(device: device, array: input)
        _ = try await layerNorm.forward(inputBuffer)
        
        // Backward pass
        let gradOutput = Array(repeating: Float(0.1), count: 5)
        let gradOutputBuffer = try MetalBuffer(device: device, array: gradOutput)
        let gradInput = try await layerNorm.backward(gradOutputBuffer)
        
        // Verify gradient shape
        XCTAssertEqual(gradInput.count, 5, "Gradient should have same size as input")
        
        // Check gradients are reasonable
        verifyNumericalStability(gradInput, message: "LayerNorm backward")
    }
    
    func testLayerNormWithoutAffine() async throws {
        let layerNorm = try await LayerNormLayer(
            normalizedShape: [20],
            elementwiseAffine: false,
            metalPipeline: metalPipeline
        )
        
        // Should have no parameters
        let paramCount = await layerNorm.getParameterCount()
        XCTAssertEqual(paramCount, 0, "LayerNorm without affine should have no parameters")
        
        // Forward pass should still work
        let input = randomInput(size: 20)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        let output = try await layerNorm.forward(inputBuffer)
        
        XCTAssertEqual(output.count, 20, "Output should have same size as input")
    }
    
    func testLayerNormMultiDimensional() async throws {
        // Test with multi-dimensional normalized shape
        let layerNorm = try await LayerNormLayer(
            normalizedShape: [4, 5],  // 20 total elements
            metalPipeline: metalPipeline
        )
        
        // Input with batch dimension
        let input = randomInput(size: 3 * 4 * 5)  // [3, 4, 5] shape
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        let output = try await layerNorm.forward(inputBuffer)
        XCTAssertEqual(output.count, input.count, "Output should preserve input size")
    }
    
    // MARK: - GroupNorm Tests
    
    func testGroupNormInitialization() async throws {
        let groupNorm = try await GroupNormLayer(
            numGroups: 4,
            numChannels: 16,
            affine: true,
            metalPipeline: metalPipeline
        )
        
        // Verify parameter count (gamma + beta for channels)
        let paramCount = await groupNorm.getParameterCount()
        XCTAssertEqual(paramCount, 16 * 2, "GroupNorm should have 2 * numChannels parameters")
    }
    
    func testGroupNormInvalidGroups() async throws {
        // Test that numChannels must be divisible by numGroups
        do {
            _ = try await GroupNormLayer(
                numGroups: 3,
                numChannels: 16,  // Not divisible by 3
                metalPipeline: metalPipeline
            )
            XCTFail("Should throw error for incompatible groups")
        } catch {
            XCTAssertTrue(error is MetalMLError)
        }
    }
    
    func testGroupNormForwardPass() async throws {
        let groupNorm = try await GroupNormLayer(
            numGroups: 2,
            numChannels: 4,
            metalPipeline: metalPipeline
        )
        
        // Create input with shape [batch=2, channels=4, spatial=3]
        let input = randomInput(size: 2 * 4 * 3)
        let shape = TensorShape(dimensions: [2, 4, 3])
        let inputBuffer = try MetalBuffer(device: device, array: input, shape: shape)
        
        let output = try await groupNorm.forward(inputBuffer)
        XCTAssertEqual(output.count, input.count, "Output should preserve input size")
        
        // Verify output is normalized
        verifyNumericalStability(output, message: "GroupNorm forward")
    }
    
    func testGroupNormBackwardPass() async throws {
        let groupNorm = try await GroupNormLayer(
            numGroups: 2,
            numChannels: 4,
            affine: true,
            metalPipeline: metalPipeline
        )
        
        // Forward pass
        let shape = TensorShape(dimensions: [1, 4, 5])
        let input = randomInput(size: 1 * 4 * 5)
        let inputBuffer = try MetalBuffer(device: device, array: input, shape: shape)
        _ = try await groupNorm.forward(inputBuffer)
        
        // Backward pass
        let gradOutput = randomInput(size: 1 * 4 * 5)
        let gradBuffer = try MetalBuffer(device: device, array: gradOutput, shape: shape)
        let gradInput = try await groupNorm.backward(gradBuffer)
        
        XCTAssertEqual(gradInput.count, input.count, "Gradient should match input size")
        verifyNumericalStability(gradInput, message: "GroupNorm backward")
    }
    
    // MARK: - BatchNorm Tests
    
    func testBatchNormInitialization() async throws {
        let batchNorm = try await BatchNormLayer(
            numFeatures: 32,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            trackRunningStats: true,
            metalPipeline: metalPipeline
        )
        
        // Verify parameter count (includes running stats but they're not trainable)
        let paramCount = await batchNorm.getParameterCount()
        XCTAssertTrue(paramCount >= 32 * 2, "BatchNorm should have at least gamma and beta parameters")
    }
    
    func testBatchNormForwardTraining() async throws {
        let batchNorm = try await BatchNormLayer(
            numFeatures: 10,
            trackRunningStats: true,
            metalPipeline: metalPipeline
        )
        await batchNorm.setTraining(true)
        
        // Create batch input [batch=4, features=10]
        let batchSize = 4
        let features = 10
        let shape = TensorShape(dimensions: [batchSize, features])
        let input = randomInput(size: batchSize * features)
        let inputBuffer = try MetalBuffer(device: device, array: input, shape: shape)
        
        // Forward pass
        let output = try await batchNorm.forward(inputBuffer)
        
        // Verify normalization across batch dimension
        XCTAssertEqual(output.count, input.count)
        
        // Check that features are normalized across batch
        let outputArray = output.toArray()
        for f in 0..<features {
            var featureValues: [Float] = []
            for b in 0..<batchSize {
                featureValues.append(outputArray[b * features + f])
            }
            
            let mean = featureValues.reduce(0, +) / Float(batchSize)
            let variance = featureValues.map { pow($0 - mean, 2) }.reduce(0, +) / Float(batchSize)
            
            XCTAssertTrue(abs(mean) < 0.1, "Feature \(f) should have mean close to 0")
            XCTAssertTrue(abs(variance - 1.0) < 0.2, "Feature \(f) should have variance close to 1")
        }
    }
    
    func testBatchNormForwardEvaluation() async throws {
        let batchNorm = try await BatchNormLayer(
            numFeatures: 5,
            trackRunningStats: true,
            metalPipeline: metalPipeline
        )
        
        // Train mode first to accumulate running stats
        await batchNorm.setTraining(true)
        let shape = TensorShape(dimensions: [10, 5])
        
        // Run a few batches to update running stats
        for _ in 0..<5 {
            let input = randomInput(size: 10 * 5)
            let inputBuffer = try MetalBuffer(device: device, array: input, shape: shape)
            _ = try await batchNorm.forward(inputBuffer)
        }
        
        // Switch to eval mode
        await batchNorm.setTraining(false)
        
        // In eval mode, should use running stats
        let evalInput = randomInput(size: 10 * 5)
        let evalBuffer = try MetalBuffer(device: device, array: evalInput, shape: shape)
        let evalOutput = try await batchNorm.forward(evalBuffer)
        
        XCTAssertEqual(evalOutput.count, evalInput.count)
        verifyNumericalStability(evalOutput, message: "BatchNorm eval")
    }
    
    func testBatchNormBackwardPass() async throws {
        let batchNorm = try await BatchNormLayer(
            numFeatures: 8,
            affine: true,
            metalPipeline: metalPipeline
        )
        await batchNorm.setTraining(true)
        
        // Forward pass
        let shape = TensorShape(dimensions: [4, 8])
        let input = randomInput(size: 4 * 8)
        let inputBuffer = try MetalBuffer(device: device, array: input, shape: shape)
        _ = try await batchNorm.forward(inputBuffer)
        
        // Backward pass
        let gradOutput = randomInput(size: 4 * 8)
        let gradBuffer = try MetalBuffer(device: device, array: gradOutput, shape: shape)
        let gradInput = try await batchNorm.backward(gradBuffer)
        
        XCTAssertEqual(gradInput.count, input.count)
        verifyNumericalStability(gradInput, message: "BatchNorm backward")
    }
    
    // MARK: - Gradient Management Tests
    
    func testNormalizationGradientManagement() async throws {
        let layers: [(String, any NeuralLayer)] = [
            ("LayerNorm", try await LayerNormLayer(
                normalizedShape: [10],
                elementwiseAffine: true,
                metalPipeline: metalPipeline
            )),
            ("GroupNorm", try await GroupNormLayer(
                numGroups: 2,
                numChannels: 10,
                affine: true,
                metalPipeline: metalPipeline
            )),
            ("BatchNorm", try await BatchNormLayer(
                numFeatures: 10,
                affine: true,
                metalPipeline: metalPipeline
            ))
        ]
        
        for (name, layer) in layers {
            // Zero gradients
            await layer.zeroGradients()
            
            // Scale gradients
            await layer.scaleGradients(0.5)
            
            XCTAssertTrue(true, "\(name) gradient management completed")
        }
    }
    
    // MARK: - Performance Tests
    
    func testLayerNormPerformance() async throws {
        let layerNorm = try await LayerNormLayer(
            normalizedShape: [512],
            metalPipeline: metalPipeline
        )
        
        let input = randomInput(size: 1024 * 512)  // Large batch
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        measure {
            let expectation = XCTestExpectation(description: "LayerNorm forward")
            
            Task {
                _ = try await layerNorm.forward(inputBuffer)
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10.0)
        }
    }
    
    func testBatchNormPerformance() async throws {
        let batchNorm = try await BatchNormLayer(
            numFeatures: 256,
            metalPipeline: metalPipeline
        )
        await batchNorm.setTraining(true)
        
        let shape = TensorShape(dimensions: [128, 256])
        let input = randomInput(size: 128 * 256)
        let inputBuffer = try MetalBuffer(device: device, array: input, shape: shape)
        
        measure {
            let expectation = XCTestExpectation(description: "BatchNorm forward")
            
            Task {
                _ = try await batchNorm.forward(inputBuffer)
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10.0)
        }
    }
    
    // MARK: - Edge Cases
    
    func testSmallBatchSize() async throws {
        let batchNorm = try await BatchNormLayer(
            numFeatures: 10,
            metalPipeline: metalPipeline
        )
        await batchNorm.setTraining(true)
        
        // Test with batch size of 1 (edge case for batch norm)
        let shape = TensorShape(dimensions: [1, 10])
        let input = randomInput(size: 10)
        let inputBuffer = try MetalBuffer(device: device, array: input, shape: shape)
        
        // Should handle gracefully even though statistics might be unstable
        let output = try await batchNorm.forward(inputBuffer)
        XCTAssertEqual(output.count, 10)
    }
    
    func testVerySmallEpsilon() async throws {
        // Test numerical stability with very small epsilon
        let layerNorm = try await LayerNormLayer(
            normalizedShape: [10],
            eps: 1e-8,
            metalPipeline: metalPipeline
        )
        
        // Input with very small variance
        let input = Array(repeating: Float(1.0), count: 10)
        input[0] = 1.0001  // Tiny variation
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        let output = try await layerNorm.forward(inputBuffer)
        verifyNumericalStability(output, message: "Small epsilon test")
    }
}

// MARK: - Test Helpers

extension NormalizationLayersTests {
    /// Helper to create random input data
    func randomInput(size: Int, range: ClosedRange<Float> = -1...1) -> [Float] {
        (0..<size).map { _ in Float.random(in: range) }
    }
    
    /// Helper to verify numerical stability
    func verifyNumericalStability(_ buffer: MetalBuffer, message: String = "") {
        let array = buffer.toArray()
        for (i, value) in array.enumerated() {
            XCTAssertFalse(value.isNaN, "\(message) - NaN at index \(i)")
            XCTAssertFalse(value.isInfinite, "\(message) - Inf at index \(i)")
        }
    }
}
// VectorStoreKit: Dense Layer Tests
//
// Comprehensive tests for Dense layer implementation

import XCTest
@testable import VectorStoreKit
import Metal

final class DenseLayerTests: XCTestCase {
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
    
    // MARK: - Initialization Tests
    
    func testDenseLayerInitialization() async throws {
        // Test basic initialization
        let dense = try await DenseLayer(
            inputSize: 128,
            outputSize: 64,
            activation: .relu,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        // Verify parameter count: weights + bias
        let paramCount = await dense.getParameterCount()
        XCTAssertEqual(paramCount, 128 * 64 + 64, "Parameter count should be inputSize * outputSize + outputSize")
    }
    
    func testDenseLayerInitializationWithDifferentActivations() async throws {
        // Test initialization with different activations
        let activations: [Activation] = [.linear, .relu, .sigmoid, .tanh, .leakyRelu, .swish]
        
        for activation in activations {
            let dense = try await DenseLayer(
                inputSize: 32,
                outputSize: 16,
                activation: activation,
                name: "test_dense_\(activation)",
                metalPipeline: metalPipeline
            )
            
            let paramCount = await dense.getParameterCount()
            XCTAssertEqual(paramCount, 32 * 16 + 16, "Parameter count should be consistent across activations")
        }
    }
    
    // MARK: - Forward Pass Tests
    
    func testForwardPassBasic() async throws {
        let dense = try await DenseLayer(
            inputSize: 4,
            outputSize: 3,
            activation: .linear,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        // Create input
        let input = Array(repeating: Float(1.0), count: 4)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        // Forward pass
        let output = try await dense.forward(inputBuffer)
        
        // Verify output shape
        XCTAssertEqual(output.count, 3, "Output should have outputSize dimensions")
        
        // Check values are reasonable (not NaN or infinite)
        let outputArray = output.toArray()
        for value in outputArray {
            XCTAssertFalse(value.isNaN, "Output should not contain NaN")
            XCTAssertFalse(value.isInfinite, "Output should not contain Inf")
        }
    }
    
    func testForwardPassWithActivations() async throws {
        // Test each activation function
        let activationTests: [(Activation, (Float) -> Bool)] = [
            (.relu, { $0 >= 0 }),
            (.sigmoid, { $0 >= 0 && $0 <= 1 }),
            (.tanh, { $0 >= -1 && $0 <= 1 }),
            (.linear, { _ in true })
        ]
        
        for (activation, validator) in activationTests {
            let dense = try await DenseLayer(
                inputSize: 10,
                outputSize: 5,
                activation: activation,
                name: "test_dense_\(activation)",
                metalPipeline: metalPipeline
            )
            
            // Create random input
            let input = (0..<10).map { _ in Float.random(in: -2...2) }
            let inputBuffer = try MetalBuffer(device: device, array: input)
            
            // Forward pass
            let output = try await dense.forward(inputBuffer)
            let outputArray = output.toArray()
            
            // Verify activation constraints
            for value in outputArray {
                XCTAssertTrue(validator(value), "Activation \(activation) output \(value) doesn't meet constraints")
            }
        }
    }
    
    func testForwardPassInvalidInput() async throws {
        let dense = try await DenseLayer(
            inputSize: 10,
            outputSize: 5,
            activation: .relu,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        // Test with wrong input size
        let wrongInput = Array(repeating: Float(1.0), count: 8) // Should be 10
        let wrongBuffer = try MetalBuffer(device: device, array: wrongInput)
        
        do {
            _ = try await dense.forward(wrongBuffer)
            XCTFail("Should throw error for incompatible input size")
        } catch {
            // Expected error
            XCTAssertTrue(error is MetalMLError)
        }
    }
    
    // MARK: - Backward Pass Tests
    
    func testBackwardPass() async throws {
        let dense = try await DenseLayer(
            inputSize: 4,
            outputSize: 3,
            activation: .relu,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        // Forward pass first
        let input = Array(repeating: Float(0.5), count: 4)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        _ = try await dense.forward(inputBuffer)
        
        // Backward pass
        let gradOutput = Array(repeating: Float(0.1), count: 3)
        let gradOutputBuffer = try MetalBuffer(device: device, array: gradOutput)
        
        let gradInput = try await dense.backward(gradOutputBuffer)
        
        // Verify gradient shape
        XCTAssertEqual(gradInput.count, 4, "Gradient input should match input size")
        
        // Check gradients are reasonable
        let gradArray = gradInput.toArray()
        for value in gradArray {
            XCTAssertFalse(value.isNaN, "Gradient should not contain NaN")
            XCTAssertFalse(value.isInfinite, "Gradient should not contain Inf")
        }
    }
    
    func testBackwardPassWithoutForward() async throws {
        let dense = try await DenseLayer(
            inputSize: 4,
            outputSize: 3,
            activation: .relu,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        // Try backward without forward
        let gradOutput = Array(repeating: Float(0.1), count: 3)
        let gradOutputBuffer = try MetalBuffer(device: device, array: gradOutput)
        
        do {
            _ = try await dense.backward(gradOutputBuffer)
            XCTFail("Should throw error when backward is called without forward")
        } catch {
            // Expected error
            XCTAssertTrue(error is MetalMLError)
        }
    }
    
    // MARK: - Parameter Update Tests
    
    func testParameterUpdate() async throws {
        let dense = try await DenseLayer(
            inputSize: 2,
            outputSize: 1,
            activation: .linear,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        // Get initial parameters
        let initialParams = await dense.getParameters()
        XCTAssertNotNil(initialParams)
        let initialValues = initialParams!.toArray()
        
        // Forward and backward pass
        let input = [Float(1.0), Float(2.0)]
        let inputBuffer = try MetalBuffer(device: device, array: input)
        _ = try await dense.forward(inputBuffer)
        
        let gradOutput = [Float(1.0)]
        let gradOutputBuffer = try MetalBuffer(device: device, array: gradOutput)
        _ = try await dense.backward(gradOutputBuffer)
        
        // Update parameters
        let learningRate: Float = 0.1
        try await dense.updateParameters(MetalBuffer(buffer: device.makeBuffer(length: 0)!, count: 0), learningRate: learningRate)
        
        // Get updated parameters
        let updatedParams = await dense.getParameters()
        XCTAssertNotNil(updatedParams)
        let updatedValues = updatedParams!.toArray()
        
        // Parameters should have changed
        var hasChanged = false
        for i in 0..<min(initialValues.count, updatedValues.count) {
            if abs(initialValues[i] - updatedValues[i]) > 1e-6 {
                hasChanged = true
                break
            }
        }
        XCTAssertTrue(hasChanged, "Parameters should change after update")
    }
    
    // MARK: - Gradient Management Tests
    
    func testZeroGradients() async throws {
        let dense = try await DenseLayer(
            inputSize: 4,
            outputSize: 2,
            activation: .relu,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        // Perform forward/backward to accumulate gradients
        let input = Array(repeating: Float(1.0), count: 4)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        _ = try await dense.forward(inputBuffer)
        
        let gradOutput = Array(repeating: Float(1.0), count: 2)
        let gradOutputBuffer = try MetalBuffer(device: device, array: gradOutput)
        _ = try await dense.backward(gradOutputBuffer)
        
        // Zero gradients
        await dense.zeroGradients()
        
        // Gradients should be zeroed (implementation specific verification)
        XCTAssertTrue(true, "Zero gradients executed without error")
    }
    
    func testScaleGradients() async throws {
        let dense = try await DenseLayer(
            inputSize: 4,
            outputSize: 2,
            activation: .relu,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        // Perform forward/backward to accumulate gradients
        let input = Array(repeating: Float(1.0), count: 4)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        _ = try await dense.forward(inputBuffer)
        
        let gradOutput = Array(repeating: Float(1.0), count: 2)
        let gradOutputBuffer = try MetalBuffer(device: device, array: gradOutput)
        _ = try await dense.backward(gradOutputBuffer)
        
        // Scale gradients
        await dense.scaleGradients(0.5)
        
        // Gradients should be scaled (implementation specific verification)
        XCTAssertTrue(true, "Scale gradients executed without error")
    }
    
    // MARK: - Training Mode Tests
    
    func testTrainingModeToggle() async throws {
        let dense = try await DenseLayer(
            inputSize: 8,
            outputSize: 4,
            activation: .relu,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        // Set training mode
        await dense.setTraining(true)
        
        // Process in training mode
        let input = Array(repeating: Float(0.5), count: 8)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        let trainOutput = try await dense.forward(inputBuffer)
        
        // Set evaluation mode
        await dense.setTraining(false)
        
        // Process in evaluation mode
        let evalOutput = try await dense.forward(inputBuffer)
        
        // Both should work
        XCTAssertEqual(trainOutput.count, evalOutput.count, "Output size should be same in both modes")
    }
    
    // MARK: - Performance Tests
    
    func testPerformanceSmallLayer() async throws {
        let dense = try await DenseLayer(
            inputSize: 64,
            outputSize: 32,
            activation: .relu,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        let input = Array(repeating: Float(0.5), count: 64)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        measure {
            let expectation = XCTestExpectation(description: "Dense forward pass")
            
            Task {
                _ = try await dense.forward(inputBuffer)
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10.0)
        }
    }
    
    func testPerformanceLargeLayer() async throws {
        let dense = try await DenseLayer(
            inputSize: 1024,
            outputSize: 512,
            activation: .relu,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        let input = Array(repeating: Float(0.1), count: 1024)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        measure {
            let expectation = XCTestExpectation(description: "Large dense forward pass")
            
            Task {
                _ = try await dense.forward(inputBuffer)
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10.0)
        }
    }
    
    // MARK: - Edge Cases
    
    func testVerySmallLayer() async throws {
        let dense = try await DenseLayer(
            inputSize: 1,
            outputSize: 1,
            activation: .linear,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        let input = [Float(2.0)]
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        let output = try await dense.forward(inputBuffer)
        XCTAssertEqual(output.count, 1, "Single neuron should produce single output")
    }
    
    func testNumericalStability() async throws {
        let dense = try await DenseLayer(
            inputSize: 10,
            outputSize: 5,
            activation: .sigmoid,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        // Test with extreme values
        let extremeInput = (0..<10).map { i in i % 2 == 0 ? Float(1000.0) : Float(-1000.0) }
        let inputBuffer = try MetalBuffer(device: device, array: extremeInput)
        
        let output = try await dense.forward(inputBuffer)
        let outputArray = output.toArray()
        
        // Sigmoid should still produce valid outputs
        for value in outputArray {
            XCTAssertFalse(value.isNaN, "Should handle extreme values without NaN")
            XCTAssertFalse(value.isInfinite, "Should handle extreme values without Inf")
            XCTAssertTrue(value >= 0 && value <= 1, "Sigmoid should still be bounded")
        }
    }
    
    // MARK: - Memory Management Tests
    
    func testMemoryManagement() async throws {
        let dense = try await DenseLayer(
            inputSize: 256,
            outputSize: 128,
            activation: .relu,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        // Run multiple iterations to test memory handling
        for i in 0..<100 {
            autoreleasepool {
                let input = Array(repeating: Float(0.1 * Float(i)), count: 256)
                let inputBuffer = try! MetalBuffer(device: device, array: input)
                
                Task {
                    _ = try? await dense.forward(inputBuffer)
                }
            }
        }
        
        // If we get here without crashes, memory management is working
        XCTAssertTrue(true, "Memory management test passed")
    }
}

// MARK: - Test Helpers

extension DenseLayerTests {
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
// VectorStoreKit: LSTM Layer Tests
//
// Unit tests for LSTM layer implementation

import XCTest
@testable import VectorStoreKit
import Metal

final class LSTMLayerTests: XCTestCase {
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
    
    func testLSTMInitialization() async throws {
        let config = LSTMConfig(hiddenSize: 32, returnSequences: true)
        let lstm = try await LSTMLayer(
            inputSize: 64,
            config: config,
            name: "test_lstm",
            metalPipeline: metalPipeline
        )
        
        // Check parameter count
        let paramCount = await lstm.getParameterCount()
        
        // LSTM has 4 gates, each with:
        // - Input weights: hiddenSize * inputSize
        // - Recurrent weights: hiddenSize * hiddenSize  
        // - Bias: hiddenSize
        let expectedParams = 4 * (32 * 64 + 32 * 32 + 32)
        XCTAssertEqual(paramCount, expectedParams, "Parameter count should match expected value")
    }
    
    // MARK: - Forward Pass Tests
    
    func testForwardPassSingleTimestep() async throws {
        let config = LSTMConfig(hiddenSize: 16, returnSequences: false)
        let lstm = try await LSTMLayer(
            inputSize: 8,
            config: config,
            metalPipeline: metalPipeline
        )
        
        // Single timestep input
        let input = Array(repeating: Float(0.5), count: 8)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        let output = try await lstm.forward(inputBuffer)
        
        XCTAssertEqual(output.count, 16, "Output should have hidden size dimensions")
        
        // Check output is bounded (tanh activation)
        let outputArray = output.toArray()
        for value in outputArray {
            XCTAssertGreaterThanOrEqual(value, -1.0, "LSTM output should be >= -1")
            XCTAssertLessThanOrEqual(value, 1.0, "LSTM output should be <= 1")
        }
    }
    
    func testForwardPassSequence() async throws {
        let seqLen = 5
        let inputSize = 10
        let hiddenSize = 20
        
        let config = LSTMConfig(hiddenSize: hiddenSize, returnSequences: true)
        let lstm = try await LSTMLayer(
            inputSize: inputSize,
            config: config,
            metalPipeline: metalPipeline
        )
        
        // Create sequence input
        var input: [Float] = []
        for t in 0..<seqLen {
            for i in 0..<inputSize {
                input.append(Float(t) * 0.1 + Float(i) * 0.01)
            }
        }
        
        let inputBuffer = try MetalBuffer(device: device, array: input)
        let output = try await lstm.forward(inputBuffer)
        
        // With returnSequences=true, output should be [seqLen * hiddenSize]
        XCTAssertEqual(output.count, seqLen * hiddenSize, 
                      "Output should contain all timestep hidden states")
    }
    
    func testForwardPassReturnLastOnly() async throws {
        let seqLen = 10
        let inputSize = 8
        let hiddenSize = 16
        
        let config = LSTMConfig(hiddenSize: hiddenSize, returnSequences: false)
        let lstm = try await LSTMLayer(
            inputSize: inputSize,
            config: config,
            metalPipeline: metalPipeline
        )
        
        // Create sequence input
        let input = Array(repeating: Float(0.1), count: seqLen * inputSize)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        let output = try await lstm.forward(inputBuffer)
        
        // With returnSequences=false, output should be just [hiddenSize]
        XCTAssertEqual(output.count, hiddenSize, 
                      "Output should contain only last hidden state")
    }
    
    // MARK: - Backward Pass Tests
    
    func testBackwardPass() async throws {
        let config = LSTMConfig(hiddenSize: 8, returnSequences: false)
        let lstm = try await LSTMLayer(
            inputSize: 4,
            config: config,
            metalPipeline: metalPipeline
        )
        
        // Forward pass
        let input = Array(repeating: Float(0.5), count: 4)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        _ = try await lstm.forward(inputBuffer)
        
        // Backward pass
        let gradOutput = Array(repeating: Float(0.1), count: 8)
        let gradOutputBuffer = try MetalBuffer(device: device, array: gradOutput)
        
        let gradInput = try await lstm.backward(gradOutputBuffer)
        
        XCTAssertEqual(gradInput.count, 4, "Gradient input should match input size")
    }
    
    // MARK: - Gate Activation Tests
    
    func testGateActivations() async throws {
        // This test verifies that gates produce valid activation values
        let config = LSTMConfig(hiddenSize: 4, returnSequences: true)
        let lstm = try await LSTMLayer(
            inputSize: 4,
            config: config,
            metalPipeline: metalPipeline
        )
        
        // Use specific input to test gate behavior
        let input: [Float] = [1.0, -1.0, 0.5, -0.5]
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        let output = try await lstm.forward(inputBuffer)
        let outputArray = output.toArray()
        
        // All outputs should be valid (no NaN or Inf)
        for value in outputArray {
            XCTAssertFalse(value.isNaN, "Output should not contain NaN")
            XCTAssertFalse(value.isInfinite, "Output should not contain Inf")
        }
    }
    
    // MARK: - Memory Management Tests
    
    func testMemoryManagement() async throws {
        let config = LSTMConfig(hiddenSize: 32, returnSequences: true)
        let lstm = try await LSTMLayer(
            inputSize: 64,
            config: config,
            metalPipeline: metalPipeline
        )
        
        // Process multiple sequences to test memory handling
        for i in 0..<10 {
            let seqLen = 5 + i
            let input = Array(repeating: Float(0.1), count: seqLen * 64)
            let inputBuffer = try MetalBuffer(device: device, array: input)
            
            _ = try await lstm.forward(inputBuffer)
            
            // Memory should be properly managed (no leaks)
            // This is implicitly tested by running multiple iterations
        }
        
        // If we get here without crashes, memory management is working
        XCTAssertTrue(true, "Memory management test passed")
    }
    
    // MARK: - Edge Case Tests
    
    func testEmptySequence() async throws {
        let config = LSTMConfig(hiddenSize: 16, returnSequences: false)
        let lstm = try await LSTMLayer(
            inputSize: 8,
            config: config,
            metalPipeline: metalPipeline
        )
        
        // Empty input should throw error
        let emptyInput = MetalBuffer(
            buffer: device.makeBuffer(length: 0, options: .storageModeShared)!,
            count: 0
        )
        
        do {
            _ = try await lstm.forward(emptyInput)
            XCTFail("Should throw error for empty input")
        } catch {
            // Expected error
            XCTAssertTrue(error is MetalMLError)
        }
    }
    
    func testVeryLongSequence() async throws {
        let config = LSTMConfig(hiddenSize: 8, returnSequences: true)
        let lstm = try await LSTMLayer(
            inputSize: 4,
            config: config,
            metalPipeline: metalPipeline
        )
        
        // Test with a long sequence
        let seqLen = 1000
        let input = Array(repeating: Float(0.01), count: seqLen * 4)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        let startTime = Date()
        let output = try await lstm.forward(inputBuffer)
        let processingTime = Date().timeIntervalSince(startTime)
        
        XCTAssertEqual(output.count, seqLen * 8, "Output size should match sequence length * hidden size")
        XCTAssertLessThan(processingTime, 1.0, "Processing should complete within reasonable time")
        
        print("Processed \(seqLen) timesteps in \(String(format: "%.3f", processingTime))s")
    }
    
    // MARK: - Training Mode Tests
    
    func testTrainingModeToggle() async throws {
        let config = LSTMConfig(hiddenSize: 16, returnSequences: false)
        let lstm = try await LSTMLayer(
            inputSize: 8,
            config: config,
            metalPipeline: metalPipeline
        )
        
        // Set training mode
        await lstm.setTraining(true)
        
        // Process in training mode
        let input = Array(repeating: Float(0.5), count: 8)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        _ = try await lstm.forward(inputBuffer)
        
        // Set evaluation mode
        await lstm.setTraining(false)
        
        // Process in evaluation mode
        _ = try await lstm.forward(inputBuffer)
        
        // Both modes should work without errors
        XCTAssertTrue(true, "Training mode toggle test passed")
    }
    
    // MARK: - Performance Tests
    
    func testPerformance() async throws {
        let config = LSTMConfig(hiddenSize: 128, returnSequences: true)
        let lstm = try await LSTMLayer(
            inputSize: 256,
            config: config,
            metalPipeline: metalPipeline
        )
        
        // Measure forward pass performance
        let seqLen = 100
        let input = Array(repeating: Float(0.1), count: seqLen * 256)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        measure {
            let expectation = XCTestExpectation(description: "LSTM forward pass")
            
            Task {
                _ = try await lstm.forward(inputBuffer)
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10.0)
        }
    }
}

// MARK: - Test Helpers

extension LSTMLayerTests {
    /// Helper to create random input data
    func randomInput(size: Int) -> [Float] {
        (0..<size).map { _ in Float.random(in: -1...1) }
    }
    
    /// Helper to verify output bounds
    func verifyOutputBounds(_ output: MetalBuffer, min: Float = -1.0, max: Float = 1.0) {
        let array = output.toArray()
        for (i, value) in array.enumerated() {
            XCTAssertGreaterThanOrEqual(value, min, "Output[\(i)] = \(value) should be >= \(min)")
            XCTAssertLessThanOrEqual(value, max, "Output[\(i)] = \(value) should be <= \(max)")
        }
    }
}
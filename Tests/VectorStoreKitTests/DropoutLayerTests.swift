// VectorStoreKit: Dropout Layer Tests
//
// Comprehensive tests for Dropout layer implementation

import XCTest
@testable import VectorStoreKit
import Metal

final class DropoutLayerTests: XCTestCase {
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
    
    func testDropoutInitialization() async throws {
        let dropout = try await DropoutLayer(rate: 0.5, metalPipeline: metalPipeline)
        
        // Verify no parameters
        let paramCount = await dropout.getParameterCount()
        XCTAssertEqual(paramCount, 0, "Dropout should have no trainable parameters")
        
        let params = await dropout.getParameters()
        XCTAssertNil(params, "Dropout should return nil for parameters")
    }
    
    func testDropoutRateValidation() async throws {
        // Test valid dropout rates
        let validRates: [Float] = [0.0, 0.1, 0.5, 0.9, 1.0]
        
        for rate in validRates {
            let dropout = try await DropoutLayer(rate: rate, metalPipeline: metalPipeline)
            XCTAssertEqual(dropout.rate, rate, "Dropout rate should be set correctly")
        }
    }
    
    // MARK: - Forward Pass Tests
    
    func testForwardPassTrainingMode() async throws {
        let dropout = try await DropoutLayer(rate: 0.5, metalPipeline: metalPipeline)
        await dropout.setTraining(true)
        
        // Create input
        let inputSize = 1000
        let input = Array(repeating: Float(1.0), count: inputSize)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        // Forward pass
        let output = try await dropout.forward(inputBuffer)
        
        // Verify output shape
        XCTAssertEqual(output.count, inputSize, "Output should have same size as input")
        
        // Check that approximately 50% of values are zeroed
        let outputArray = output.toArray()
        let zeroCount = outputArray.filter { $0 == 0 }.count
        let zeroPercentage = Float(zeroCount) / Float(inputSize)
        
        // Allow for statistical variation
        XCTAssertTrue(abs(zeroPercentage - 0.5) < 0.1, 
                     "Approximately 50% of values should be dropped, got \(zeroPercentage * 100)%")
        
        // Check that non-zero values are scaled correctly
        let nonZeroValues = outputArray.filter { $0 != 0 }
        if !nonZeroValues.isEmpty {
            let expectedScale = 1.0 / (1.0 - dropout.rate)
            for value in nonZeroValues {
                XCTAssertTrue(abs(value - expectedScale) < 0.001,
                             "Non-zero values should be scaled by 1/(1-rate)")
            }
        }
    }
    
    func testForwardPassEvaluationMode() async throws {
        let dropout = try await DropoutLayer(rate: 0.5, metalPipeline: metalPipeline)
        await dropout.setTraining(false)
        
        // Create input
        let input = Array(repeating: Float(2.0), count: 100)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        // Forward pass
        let output = try await dropout.forward(inputBuffer)
        
        // In evaluation mode, output should equal input
        let outputArray = output.toArray()
        for (i, value) in outputArray.enumerated() {
            XCTAssertEqual(value, input[i], 
                          "In evaluation mode, dropout should not modify input")
        }
    }
    
    func testForwardPassZeroDropoutRate() async throws {
        let dropout = try await DropoutLayer(rate: 0.0, metalPipeline: metalPipeline)
        await dropout.setTraining(true)
        
        // Create input
        let input = randomInput(size: 50)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        // Forward pass
        let output = try await dropout.forward(inputBuffer)
        
        // With 0% dropout, output should equal input
        let outputArray = output.toArray()
        for (i, value) in outputArray.enumerated() {
            XCTAssertEqual(value, input[i], 
                          "With 0% dropout rate, no values should be dropped")
        }
    }
    
    func testForwardPassFullDropoutRate() async throws {
        let dropout = try await DropoutLayer(rate: 1.0, metalPipeline: metalPipeline)
        await dropout.setTraining(true)
        
        // Create input
        let input = randomInput(size: 50, range: 1...5)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        // Forward pass
        let output = try await dropout.forward(inputBuffer)
        
        // With 100% dropout, all values should be zero
        let outputArray = output.toArray()
        for value in outputArray {
            XCTAssertEqual(value, 0.0, 
                          "With 100% dropout rate, all values should be zero")
        }
    }
    
    // MARK: - Backward Pass Tests
    
    func testBackwardPass() async throws {
        let dropout = try await DropoutLayer(rate: 0.5, metalPipeline: metalPipeline)
        await dropout.setTraining(true)
        
        // Forward pass first
        let input = Array(repeating: Float(1.0), count: 100)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        let forwardOutput = try await dropout.forward(inputBuffer)
        
        // Backward pass
        let gradOutput = Array(repeating: Float(0.1), count: 100)
        let gradOutputBuffer = try MetalBuffer(device: device, array: gradOutput)
        let gradInput = try await dropout.backward(gradOutputBuffer)
        
        // Verify gradient shape
        XCTAssertEqual(gradInput.count, input.count, 
                      "Gradient input should have same size as input")
        
        // Check that gradients match the dropout mask
        let forwardArray = forwardOutput.toArray()
        let gradArray = gradInput.toArray()
        
        for i in 0..<forwardArray.count {
            if forwardArray[i] == 0 {
                XCTAssertEqual(gradArray[i], 0.0, 
                              "Gradient should be zero where forward output was dropped")
            } else {
                XCTAssertTrue(abs(gradArray[i]) > 0, 
                             "Gradient should be non-zero where forward output was not dropped")
            }
        }
    }
    
    func testBackwardPassEvaluationMode() async throws {
        let dropout = try await DropoutLayer(rate: 0.5, metalPipeline: metalPipeline)
        await dropout.setTraining(false)
        
        // Forward pass in eval mode
        let input = randomInput(size: 50)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        _ = try await dropout.forward(inputBuffer)
        
        // Backward pass
        let gradOutput = Array(repeating: Float(0.2), count: 50)
        let gradOutputBuffer = try MetalBuffer(device: device, array: gradOutput)
        let gradInput = try await dropout.backward(gradOutputBuffer)
        
        // In eval mode, gradients should pass through unchanged
        let gradArray = gradInput.toArray()
        for (i, value) in gradArray.enumerated() {
            XCTAssertEqual(value, gradOutput[i], 
                          "In evaluation mode, gradients should pass through unchanged")
        }
    }
    
    // MARK: - Consistency Tests
    
    func testDropoutConsistency() async throws {
        let dropout = try await DropoutLayer(rate: 0.3, metalPipeline: metalPipeline)
        await dropout.setTraining(true)
        
        // Same input should produce different outputs (due to randomness)
        let input = randomInput(size: 100)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        let output1 = try await dropout.forward(inputBuffer)
        let output2 = try await dropout.forward(inputBuffer)
        
        let array1 = output1.toArray()
        let array2 = output2.toArray()
        
        // Outputs should be different due to random dropout
        var isDifferent = false
        for i in 0..<array1.count {
            if array1[i] != array2[i] {
                isDifferent = true
                break
            }
        }
        
        XCTAssertTrue(isDifferent, 
                     "Different forward passes should produce different dropout patterns")
    }
    
    // MARK: - Parameter Update Tests
    
    func testParameterUpdate() async throws {
        let dropout = try await DropoutLayer(rate: 0.5, metalPipeline: metalPipeline)
        
        // Dropout has no parameters, so update should do nothing
        let dummyGradients = try MetalBuffer(device: device, array: [Float](repeating: 0, count: 1))
        
        // Should not throw
        try await dropout.updateParameters(dummyGradients, learningRate: 0.1)
        
        // Parameters should still be nil
        let params = await dropout.getParameters()
        XCTAssertNil(params, "Dropout should have no parameters after update")
    }
    
    // MARK: - Gradient Management Tests
    
    func testZeroGradients() async throws {
        let dropout = try await DropoutLayer(rate: 0.5, metalPipeline: metalPipeline)
        
        // Should complete without error (no-op for dropout)
        await dropout.zeroGradients()
        XCTAssertTrue(true, "Zero gradients should complete without error")
    }
    
    func testScaleGradients() async throws {
        let dropout = try await DropoutLayer(rate: 0.5, metalPipeline: metalPipeline)
        
        // Should complete without error (no-op for dropout)
        await dropout.scaleGradients(0.5)
        XCTAssertTrue(true, "Scale gradients should complete without error")
    }
    
    // MARK: - Edge Cases
    
    func testEmptyInput() async throws {
        let dropout = try await DropoutLayer(rate: 0.5, metalPipeline: metalPipeline)
        
        // Create empty buffer
        let emptyBuffer = MetalBuffer(
            buffer: device.makeBuffer(length: 0, options: .storageModeShared)!,
            count: 0
        )
        
        // Should handle empty input gracefully
        let output = try await dropout.forward(emptyBuffer)
        XCTAssertEqual(output.count, 0, "Empty input should produce empty output")
    }
    
    func testLargeInput() async throws {
        let dropout = try await DropoutLayer(rate: 0.2, metalPipeline: metalPipeline)
        await dropout.setTraining(true)
        
        // Test with large input
        let largeSize = 10000
        let input = Array(repeating: Float(1.0), count: largeSize)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        let output = try await dropout.forward(inputBuffer)
        
        // Verify statistical properties hold for large input
        let outputArray = output.toArray()
        let droppedCount = outputArray.filter { $0 == 0 }.count
        let dropRate = Float(droppedCount) / Float(largeSize)
        
        // Should be close to specified rate with large sample
        XCTAssertTrue(abs(dropRate - 0.2) < 0.02, 
                     "Drop rate should converge to specified rate with large input")
    }
    
    // MARK: - Performance Tests
    
    func testPerformanceSmallInput() async throws {
        let dropout = try await DropoutLayer(rate: 0.5, metalPipeline: metalPipeline)
        await dropout.setTraining(true)
        
        let input = randomInput(size: 100)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        measure {
            let expectation = XCTestExpectation(description: "Dropout forward pass")
            
            Task {
                _ = try await dropout.forward(inputBuffer)
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10.0)
        }
    }
    
    func testPerformanceLargeInput() async throws {
        let dropout = try await DropoutLayer(rate: 0.5, metalPipeline: metalPipeline)
        await dropout.setTraining(true)
        
        let input = randomInput(size: 100000)
        let inputBuffer = try MetalBuffer(device: device, array: input)
        
        measure {
            let expectation = XCTestExpectation(description: "Large dropout forward pass")
            
            Task {
                _ = try await dropout.forward(inputBuffer)
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10.0)
        }
    }
    
    // MARK: - Statistical Tests
    
    func testDropoutDistribution() async throws {
        let rates: [Float] = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for rate in rates {
            let dropout = try await DropoutLayer(rate: rate, metalPipeline: metalPipeline)
            await dropout.setTraining(true)
            
            // Run multiple trials
            let trials = 10
            let inputSize = 1000
            var totalDropped = 0
            
            for _ in 0..<trials {
                let input = Array(repeating: Float(1.0), count: inputSize)
                let inputBuffer = try MetalBuffer(device: device, array: input)
                
                let output = try await dropout.forward(inputBuffer)
                let outputArray = output.toArray()
                totalDropped += outputArray.filter { $0 == 0 }.count
            }
            
            let averageDropRate = Float(totalDropped) / Float(trials * inputSize)
            let tolerance: Float = 0.05 // 5% tolerance
            
            XCTAssertTrue(abs(averageDropRate - rate) < tolerance,
                         "Average drop rate \(averageDropRate) should be close to \(rate)")
        }
    }
}

// MARK: - Test Helpers

extension DropoutLayerTests {
    /// Helper to create random input data
    func randomInput(size: Int, range: ClosedRange<Float> = -1...1) -> [Float] {
        (0..<size).map { _ in Float.random(in: range) }
    }
}
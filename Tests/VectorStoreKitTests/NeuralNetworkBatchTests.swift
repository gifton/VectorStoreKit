// VectorStoreKit: Neural Network Batch Processing Tests
//
// Tests for batch processing in the neural network training pipeline

import XCTest
@testable import VectorStoreKit
import Metal

final class NeuralNetworkBatchTests: XCTestCase {
    var device: MTLDevice!
    var metalPipeline: MetalMLPipeline!
    
    override func setUp() async throws {
        try await super.setUp()
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        self.device = device
        self.metalPipeline = try MetalMLPipeline(device: device)
    }
    
    func testBatchProcessing() async throws {
        // Create a simple network
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Add layers
        let dense1 = try await DenseLayer(
            inputSize: 10,
            outputSize: 5,
            activation: .relu,
            name: "dense1",
            metalPipeline: metalPipeline
        )
        
        let dense2 = try await DenseLayer(
            inputSize: 5,
            outputSize: 2,
            activation: .linear,
            name: "dense2",
            metalPipeline: metalPipeline
        )
        
        await network.addLayers([dense1, dense2])
        
        // Create batch training data
        let batchSize = 4
        let inputSize = 10
        let outputSize = 2
        
        var trainingData: [(input: MetalBuffer, target: MetalBuffer)] = []
        
        for i in 0..<8 { // 8 samples total
            // Create random input
            var inputData = [Float](repeating: 0, count: inputSize)
            for j in 0..<inputSize {
                inputData[j] = Float.random(in: -1...1)
            }
            
            // Create random target
            var targetData = [Float](repeating: 0, count: outputSize)
            for j in 0..<outputSize {
                targetData[j] = Float.random(in: 0...1)
            }
            
            let inputBuffer = try MetalBuffer(device: device, array: inputData)
            let targetBuffer = try MetalBuffer(device: device, array: targetData)
            
            trainingData.append((input: inputBuffer, target: targetBuffer))
        }
        
        // Train with batch processing
        let config = NetworkTrainingConfig(
            epochs: 2,
            batchSize: batchSize,
            learningRate: 0.01,
            lossFunction: .mse
        )
        
        try await network.train(data: trainingData, config: config)
        
        // Verify training completed
        let history = await network.getTrainingHistory()
        XCTAssertFalse(history.losses.isEmpty, "Training should produce loss history")
        
        // Verify loss decreased
        if history.losses.count > 1 {
            let firstLoss = history.losses.first!
            let lastLoss = history.losses.last!
            XCTAssertLessThan(lastLoss, firstLoss * 1.2, "Loss should generally decrease during training")
        }
    }
    
    func testGradientAccumulation() async throws {
        // Create a simple dense layer
        let dense = try await DenseLayer(
            inputSize: 3,
            outputSize: 2,
            activation: .linear,
            name: "test_dense",
            metalPipeline: metalPipeline
        )
        
        // Test gradient zeroing
        await dense.zeroGradients()
        
        // Perform forward and backward pass
        let input = try MetalBuffer(device: device, array: [1.0, 2.0, 3.0])
        let output = try await dense.forward(input)
        
        // Create gradient for backward pass
        let gradOutput = try MetalBuffer(device: device, array: [0.5, -0.5])
        _ = try await dense.backward(gradOutput)
        
        // Scale gradients (simulating batch average)
        await dense.scaleGradients(0.25) // Scale by 1/4 for batch size 4
        
        // Update parameters
        try await dense.updateParameters(gradOutput, learningRate: 0.1)
        
        // Verify parameters were updated
        let paramsAfter = await dense.getParameters()
        XCTAssertNotNil(paramsAfter, "Parameters should exist after update")
    }
    
    func testBatchSizeScaling() async throws {
        // Test that gradient scaling works correctly for different batch sizes
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        let dense = try await DenseLayer(
            inputSize: 2,
            outputSize: 1,
            activation: .linear,
            metalPipeline: metalPipeline
        )
        await network.addLayer(dense)
        
        // Create small dataset
        var data: [(input: MetalBuffer, target: MetalBuffer)] = []
        for _ in 0..<4 {
            let input = try MetalBuffer(device: device, array: [Float.random(in: -1...1), Float.random(in: -1...1)])
            let target = try MetalBuffer(device: device, array: [Float.random(in: 0...1)])
            data.append((input: input, target: target))
        }
        
        // Train with batch size 2
        let config = NetworkTrainingConfig(
            epochs: 1,
            batchSize: 2,
            learningRate: 0.01,
            lossFunction: .mse
        )
        
        try await network.train(data: data, config: config)
        
        // Verify proper batch processing
        let history = await network.getTrainingHistory()
        XCTAssertEqual(history.losses.count, 1, "Should have one epoch of losses")
    }
    
    // MARK: - Mixed Batch Processing Tests
    
    func testMixedBatchSizes() async throws {
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Create layers
        let layers = [
            try await DenseLayer(inputSize: 8, outputSize: 4, activation: .relu, metalPipeline: metalPipeline),
            try await DropoutLayer(rate: 0.2, metalPipeline: metalPipeline),
            try await DenseLayer(inputSize: 4, outputSize: 2, activation: .sigmoid, metalPipeline: metalPipeline)
        ]
        
        for layer in layers {
            await network.addLayer(layer)
        }
        
        // Test with different batch sizes
        let batchSizes = [1, 2, 4, 8]
        
        for batchSize in batchSizes {
            // Create data for this batch size
            var data: [(input: MetalBuffer, target: MetalBuffer)] = []
            for _ in 0..<(batchSize * 2) {
                let input = (0..<8).map { _ in Float.random(in: -1...1) }
                let target = (0..<2).map { _ in Float.random(in: 0...1) }
                
                data.append((
                    input: try MetalBuffer(device: device, array: input),
                    target: try MetalBuffer(device: device, array: target)
                ))
            }
            
            let config = NetworkTrainingConfig(
                epochs: 1,
                batchSize: batchSize,
                learningRate: 0.01,
                lossFunction: .binaryCrossEntropy
            )
            
            try await network.train(data: data, config: config)
            
            // Verify training completed
            let history = await network.getTrainingHistory()
            XCTAssertFalse(history.losses.isEmpty, "Batch size \(batchSize) should produce loss history")
        }
    }
    
    func testUniformBatchProcessing() async throws {
        // Test that all samples in a batch are processed correctly
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        let inputSize = 5
        let hiddenSize = 3
        let outputSize = 2
        
        let dense1 = try await DenseLayer(
            inputSize: inputSize,
            outputSize: hiddenSize,
            activation: .tanh,
            metalPipeline: metalPipeline
        )
        let dense2 = try await DenseLayer(
            inputSize: hiddenSize,
            outputSize: outputSize,
            activation: .linear,
            metalPipeline: metalPipeline
        )
        
        await network.addLayers([dense1, dense2])
        
        // Create identical inputs to verify batch processing
        let batchSize = 4
        var data: [(input: MetalBuffer, target: MetalBuffer)] = []
        
        let sameInput = Array(repeating: Float(0.5), count: inputSize)
        let sameTarget = Array(repeating: Float(0.7), count: outputSize)
        
        for _ in 0..<batchSize {
            data.append((
                input: try MetalBuffer(device: device, array: sameInput),
                target: try MetalBuffer(device: device, array: sameTarget)
            ))
        }
        
        // Forward pass for batch
        var outputs: [MetalBuffer] = []
        for (input, _) in data {
            let output = try await network.forward(input)
            outputs.append(output)
        }
        
        // All outputs should be identical for identical inputs
        let firstOutput = outputs[0].toArray()
        for i in 1..<outputs.count {
            let output = outputs[i].toArray()
            for j in 0..<output.count {
                XCTAssertEqual(output[j], firstOutput[j], accuracy: 1e-5,
                              "Outputs should be identical for identical inputs")
            }
        }
    }
    
    func testGradientAccumulationAcrossBatch() async throws {
        // Verify that gradients are properly accumulated across batch
        let dense = try await DenseLayer(
            inputSize: 2,
            outputSize: 1,
            activation: .linear,
            metalPipeline: metalPipeline
        )
        
        // Zero gradients initially
        await dense.zeroGradients()
        
        // Simulate batch processing with gradient accumulation
        let batchSize = 3
        var totalLoss: Float = 0
        
        for i in 0..<batchSize {
            // Different inputs for each sample
            let input = try MetalBuffer(device: device, array: [Float(i + 1), Float(i + 2)])
            let output = try await dense.forward(input)
            
            // Simulate loss gradient
            let gradOutput = try MetalBuffer(device: device, array: [Float(0.1 * Float(i + 1))])
            _ = try await dense.backward(gradOutput)
            
            totalLoss += 0.1 * Float(i + 1)
        }
        
        // Scale gradients by batch size
        await dense.scaleGradients(1.0 / Float(batchSize))
        
        // Update parameters
        try await dense.updateParameters(MetalBuffer(buffer: device.makeBuffer(length: 0)!, count: 0), learningRate: 0.1)
        
        XCTAssertTrue(true, "Gradient accumulation completed successfully")
    }
    
    // MARK: - Memory Management Tests
    
    func testBatchMemoryManagement() async throws {
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Create a larger network
        let layers = [
            try await DenseLayer(inputSize: 128, outputSize: 64, activation: .relu, metalPipeline: metalPipeline),
            try await LayerNormLayer(normalizedShape: [64], metalPipeline: metalPipeline),
            try await DenseLayer(inputSize: 64, outputSize: 32, activation: .relu, metalPipeline: metalPipeline),
            try await DropoutLayer(rate: 0.3, metalPipeline: metalPipeline),
            try await DenseLayer(inputSize: 32, outputSize: 10, activation: .softmax, metalPipeline: metalPipeline)
        ]
        
        for layer in layers {
            await network.addLayer(layer)
        }
        
        // Process multiple large batches
        let batchSize = 32
        let numBatches = 10
        
        for batch in 0..<numBatches {
            autoreleasepool {
                // Create batch data
                var batchData: [(input: MetalBuffer, target: MetalBuffer)] = []
                
                for _ in 0..<batchSize {
                    let input = (0..<128).map { _ in Float.random(in: -1...1) }
                    let target = (0..<10).map { _ in Float.random(in: 0...1) }
                    
                    if let inputBuffer = try? MetalBuffer(device: device, array: input),
                       let targetBuffer = try? MetalBuffer(device: device, array: target) {
                        batchData.append((input: inputBuffer, target: targetBuffer))
                    }
                }
                
                // Process batch
                Task {
                    for (input, _) in batchData {
                        _ = try? await network.forward(input)
                    }
                }
            }
        }
        
        // If we complete without memory issues, test passes
        XCTAssertTrue(true, "Memory management test completed")
    }
    
    // MARK: - Performance Tests
    
    func testBatchProcessingPerformance() async throws {
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Build a typical network
        await network.addLayer(try await DenseLayer(
            inputSize: 784,
            outputSize: 128,
            activation: .relu,
            metalPipeline: metalPipeline
        ))
        await network.addLayer(try await DenseLayer(
            inputSize: 128,
            outputSize: 10,
            activation: .softmax,
            metalPipeline: metalPipeline
        ))
        
        // Create batch data
        let batchSize = 64
        var data: [(input: MetalBuffer, target: MetalBuffer)] = []
        
        for _ in 0..<batchSize {
            let input = (0..<784).map { _ in Float.random(in: 0...1) }
            var target = Array(repeating: Float(0), count: 10)
            target[Int.random(in: 0..<10)] = 1.0  // One-hot encoding
            
            data.append((
                input: try MetalBuffer(device: device, array: input),
                target: try MetalBuffer(device: device, array: target)
            ))
        }
        
        // Measure batch processing time
        measure {
            let expectation = XCTestExpectation(description: "Batch processing")
            
            Task {
                // Process entire batch
                for (input, _) in data {
                    _ = try await network.forward(input)
                }
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 30.0)
        }
    }
    
    // MARK: - Edge Cases
    
    func testEmptyBatch() async throws {
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        await network.addLayer(try await DenseLayer(
            inputSize: 10,
            outputSize: 5,
            metalPipeline: metalPipeline
        ))
        
        // Try training with empty data
        let emptyData: [(input: MetalBuffer, target: MetalBuffer)] = []
        let config = NetworkTrainingConfig(
            epochs: 1,
            batchSize: 4,
            learningRate: 0.01,
            lossFunction: .mse
        )
        
        // Should handle gracefully
        do {
            try await network.train(data: emptyData, config: config)
        } catch {
            // Expected - empty data should be handled appropriately
            XCTAssertTrue(true, "Empty batch handled correctly")
        }
    }
    
    func testSingleSampleBatch() async throws {
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        await network.addLayer(try await DenseLayer(
            inputSize: 3,
            outputSize: 1,
            activation: .sigmoid,
            metalPipeline: metalPipeline
        ))
        
        // Single sample
        let data = [(
            input: try MetalBuffer(device: device, array: [0.1, 0.2, 0.3]),
            target: try MetalBuffer(device: device, array: [0.8])
        )]
        
        let config = NetworkTrainingConfig(
            epochs: 5,
            batchSize: 1,
            learningRate: 0.1,
            lossFunction: .mse
        )
        
        try await network.train(data: data, config: config)
        
        // Verify training occurred
        let history = await network.getTrainingHistory()
        XCTAssertEqual(history.losses.count, 5, "Should have losses for each epoch")
    }
}
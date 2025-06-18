// VectorStoreKit Model Serialization Example
//
// Demonstrates checkpoint save/load functionality for neural networks

import Foundation
import VectorStoreKit
import Metal

@main
struct ModelSerializationExample {
    static func main() async throws {
        print("=== VectorStoreKit Model Serialization Example ===\n")
        
        // Initialize Metal
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal not available")
            return
        }
        
        print("Using device: \(device.name ?? "Unknown")")
        
        // Create Metal ML pipeline
        let metalPipeline = try MetalMLPipeline(device: device)
        
        // Create a neural network
        print("\n1. Creating neural network...")
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Add layers
        await network.addLayers([
            try await DenseLayer(
                inputSize: 784,
                outputSize: 128,
                activation: .relu,
                name: "hidden1",
                metalPipeline: metalPipeline
            ),
            try await BatchNormLayer(
                numFeatures: 128,
                name: "bn1",
                metalPipeline: metalPipeline
            ),
            try await DropoutLayer(
                rate: 0.2,
                metalPipeline: metalPipeline
            ),
            try await DenseLayer(
                inputSize: 128,
                outputSize: 64,
                activation: .relu,
                name: "hidden2",
                metalPipeline: metalPipeline
            ),
            try await DenseLayer(
                inputSize: 64,
                outputSize: 10,
                activation: .softmax,
                name: "output",
                metalPipeline: metalPipeline
            )
        ])
        
        let paramCount = await network.parameterCount()
        print("Network created with \(paramCount) parameters")
        
        // Train the network briefly to initialize parameters
        print("\n2. Training network briefly...")
        
        // Create simple training data
        let batchSize = 32
        let inputSize = 784
        let outputSize = 10
        
        var trainingData: [(input: MetalBuffer, target: MetalBuffer)] = []
        
        for _ in 0..<5 { // 5 batches
            // Random input
            let inputArray = (0..<inputSize).map { _ in Float.random(in: -1...1) }
            let inputBuffer = try await metalPipeline.allocateBuffer(size: inputSize)
            let inputPtr = inputBuffer.buffer.contents().bindMemory(to: Float.self, capacity: inputSize)
            for i in 0..<inputSize {
                inputPtr[i] = inputArray[i]
            }
            
            // Random one-hot target
            let targetArray = (0..<outputSize).map { i in
                i == Int.random(in: 0..<outputSize) ? Float(1.0) : Float(0.0)
            }
            let targetBuffer = try await metalPipeline.allocateBuffer(size: outputSize)
            let targetPtr = targetBuffer.buffer.contents().bindMemory(to: Float.self, capacity: outputSize)
            for i in 0..<outputSize {
                targetPtr[i] = targetArray[i]
            }
            
            trainingData.append((inputBuffer, targetBuffer))
        }
        
        let trainingConfig = NetworkTrainingConfig(
            epochs: 2,
            batchSize: 1,
            learningRate: 0.01,
            lossFunction: .crossEntropy,
            logInterval: 1
        )
        
        try await network.train(data: trainingData, config: trainingConfig)
        
        // Save checkpoint
        print("\n3. Saving checkpoint...")
        let checkpointURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("model_checkpoint.json")
        
        let saveOptions = CheckpointOptions(
            compression: .zstd,
            includeTrainingState: true,
            description: "Example neural network checkpoint",
            customMetadata: [
                "example": "ModelSerializationExample",
                "purpose": "demonstration"
            ]
        )
        
        try await network.saveCheckpoint(to: checkpointURL, options: saveOptions)
        
        let fileSize = try FileManager.default.attributesOfItem(atPath: checkpointURL.path)[.size] as? Int ?? 0
        print("Checkpoint saved to: \(checkpointURL.path)")
        print("File size: \(ByteCountFormatter.string(fromByteCount: Int64(fileSize), countStyle: .file))")
        
        // Load checkpoint metadata
        print("\n4. Loading checkpoint metadata...")
        let metadata = try await NeuralNetwork.checkpointMetadata(from: checkpointURL)
        print("Checkpoint created: \(metadata.timestamp)")
        print("Description: \(metadata.description ?? "None")")
        print("Metrics: \(metadata.metrics)")
        print("Hardware: \(metadata.hardware.deviceName)")
        
        // Create a new network and load checkpoint
        print("\n5. Creating new network and loading checkpoint...")
        let newNetwork = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        try await newNetwork.loadCheckpoint(from: checkpointURL)
        
        let newParamCount = await newNetwork.parameterCount()
        print("New network loaded with \(newParamCount) parameters")
        
        // Verify the networks are equivalent
        print("\n6. Verifying loaded network...")
        
        // Test with same input
        let testInput = try await metalPipeline.allocateBuffer(size: inputSize)
        let testPtr = testInput.buffer.contents().bindMemory(to: Float.self, capacity: inputSize)
        for i in 0..<inputSize {
            testPtr[i] = Float.random(in: -1...1)
        }
        
        let originalOutput = try await network.predict(testInput)
        let loadedOutput = try await newNetwork.predict(testInput)
        
        // Compare outputs
        let originalPtr = originalOutput.buffer.contents().bindMemory(to: Float.self, capacity: outputSize)
        let loadedPtr = loadedOutput.buffer.contents().bindMemory(to: Float.self, capacity: outputSize)
        
        var maxDiff: Float = 0
        for i in 0..<outputSize {
            let diff = abs(originalPtr[i] - loadedPtr[i])
            maxDiff = max(maxDiff, diff)
        }
        
        print("Maximum output difference: \(maxDiff)")
        print("Networks are \(maxDiff < 1e-5 ? "equivalent ✓" : "different ✗")")
        
        // Test compression ratios
        print("\n7. Testing compression algorithms...")
        
        for compression in [CompressionAlgorithm.none, .lz4, .zstd] {
            let testURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("test_\(compression.rawValue).json")
            
            let options = CheckpointOptions(
                compression: compression,
                includeTrainingState: false
            )
            
            try await network.saveCheckpoint(to: testURL, options: options)
            
            let size = try FileManager.default.attributesOfItem(atPath: testURL.path)[.size] as? Int ?? 0
            let ratio = Float(fileSize) / Float(size)
            
            print("\(compression.rawValue): \(ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)) (ratio: \(String(format: "%.2f", ratio))x)")
            
            try? FileManager.default.removeItem(at: testURL)
        }
        
        // Cleanup
        try? FileManager.default.removeItem(at: checkpointURL)
        
        print("\n✅ Model serialization example completed successfully!")
    }
}
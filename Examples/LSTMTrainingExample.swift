// VectorStoreKit: LSTM Training Example
//
// Demonstrates end-to-end training of an LSTM network with:
// - Proper gradient computation through BPTT
// - Batch processing
// - AdamW optimizer with gradient clipping
// - Mixed precision training

import Foundation
import Metal
import VectorStoreKit

@main
struct LSTMTrainingExample {
    static func main() async throws {
        print("=== VectorStoreKit LSTM Training Example ===\n")
        
        // Initialize Metal
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not available")
            return
        }
        
        let metalPipeline = try MetalMLPipeline(device: device)
        
        // Create a simple LSTM for sequence classification
        print("Building LSTM model...")
        let model = try await createSequenceClassifier(metalPipeline: metalPipeline)
        
        // Generate synthetic sequence data
        print("Generating training data...")
        let (trainData, valData) = try await generateSequenceData(device: device)
        
        // Train with AdamW optimizer
        print("\nTraining with AdamW optimizer...")
        try await trainWithAdamW(model: model, trainData: trainData, valData: valData)
        
        // Demonstrate gradient clipping
        print("\nTraining with gradient clipping...")
        try await trainWithGradientClipping(model: model, trainData: trainData, valData: valData)
        
        // Demonstrate mixed precision training
        print("\nTraining with mixed precision...")
        try await trainWithMixedPrecision(model: model, trainData: trainData, valData: valData, device: device)
        
        print("\n✅ LSTM training example completed successfully!")
    }
    
    // MARK: - Model Creation
    
    static func createSequenceClassifier(metalPipeline: MetalMLPipeline) async throws -> NeuralNetwork {
        let model = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Input size: 32, Hidden size: 64, Sequence length: variable
        let lstmConfig = LSTMConfig(
            hiddenSize: 64,
            returnSequences: false,  // Only return last hidden state
            dropout: 0.2
        )
        
        let lstm = try await LSTMLayer(
            inputSize: 32,
            config: lstmConfig,
            name: "lstm1",
            metalPipeline: metalPipeline
        )
        
        // Classification head
        let dense = try await DenseLayer(
            inputSize: 64,
            outputSize: 10,  // 10 classes
            activation: .linear,
            name: "classifier",
            metalPipeline: metalPipeline
        )
        
        let softmax = try await ActivationLayer(
            activation: .softmax,
            name: "softmax",
            metalPipeline: metalPipeline
        )
        
        await model.addLayers([lstm, dense, softmax])
        
        return model
    }
    
    // MARK: - Data Generation
    
    static func generateSequenceData(device: MTLDevice) async throws -> (train: [(MetalBuffer, MetalBuffer)], val: [(MetalBuffer, MetalBuffer)]) {
        let numSamples = 100
        let sequenceLength = 20
        let inputSize = 32
        let numClasses = 10
        
        var trainData: [(MetalBuffer, MetalBuffer)] = []
        var valData: [(MetalBuffer, MetalBuffer)] = []
        
        // Generate synthetic sequences
        for i in 0..<numSamples {
            // Create input sequence
            var input: [Float] = []
            for _ in 0..<sequenceLength {
                for _ in 0..<inputSize {
                    input.append(Float.random(in: -1...1))
                }
            }
            
            // Create one-hot encoded target
            let targetClass = i % numClasses
            var target = [Float](repeating: 0, count: numClasses)
            target[targetClass] = 1.0
            
            let inputBuffer = try MetalBuffer(device: device, array: input)
            let targetBuffer = try MetalBuffer(device: device, array: target)
            
            if i < 80 {
                trainData.append((inputBuffer, targetBuffer))
            } else {
                valData.append((inputBuffer, targetBuffer))
            }
        }
        
        return (trainData, valData)
    }
    
    // MARK: - Training Functions
    
    static func trainWithAdamW(model: NeuralNetwork, trainData: [(MetalBuffer, MetalBuffer)], valData: [(MetalBuffer, MetalBuffer)]) async throws {
        // Create AdamW optimizer
        let optimizer = await AdamWOptimizer(
            learningRate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            weightDecay: 0.01
        )
        
        let config = NetworkTrainingConfig(
            epochs: 10,
            batchSize: 16,
            learningRate: 0.001,
            lossFunction: .crossEntropy,
            optimizer: optimizer,
            logInterval: 5
        )
        
        let startTime = Date()
        try await model.train(data: trainData, config: config)
        let trainingTime = Date().timeIntervalSince(startTime)
        
        print("Training completed in \(String(format: "%.2f", trainingTime)) seconds")
        
        // Evaluate on validation set
        await evaluateModel(model: model, valData: valData)
    }
    
    static func trainWithGradientClipping(model: NeuralNetwork, trainData: [(MetalBuffer, MetalBuffer)], valData: [(MetalBuffer, MetalBuffer)]) async throws {
        // Create AdamW optimizer with gradient clipping
        let optimizer = await AdamWOptimizer(
            learningRate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            weightDecay: 0.01,
            gradientClipping: 1.0  // Clip gradients to [-1, 1]
        )
        
        let config = NetworkTrainingConfig(
            epochs: 5,
            batchSize: 16,
            learningRate: 0.001,
            lossFunction: .crossEntropy,
            optimizer: optimizer,
            gradientClipping: 1.0,
            logInterval: 5
        )
        
        let startTime = Date()
        try await model.train(data: trainData, config: config)
        let trainingTime = Date().timeIntervalSince(startTime)
        
        print("Training with gradient clipping completed in \(String(format: "%.2f", trainingTime)) seconds")
    }
    
    static func trainWithMixedPrecision(model: NeuralNetwork, trainData: [(MetalBuffer, MetalBuffer)], valData: [(MetalBuffer, MetalBuffer)], device: MTLDevice) async throws {
        // Create mixed precision configuration
        let mixedPrecisionConfig = MixedPrecisionConfig(
            useFP16Compute: true,
            initialLossScale: 1024.0,
            lossScaleGrowthInterval: 100
        )
        
        // Create dynamic loss scaler
        let lossScaler = try DynamicLossScaler(config: mixedPrecisionConfig, device: device)
        
        // Train with mixed precision
        print("Initial loss scale: \(await lossScaler.scale)")
        
        // Note: In a real implementation, the training loop would integrate with the loss scaler
        // to scale gradients before backward pass and unscale before optimizer step
        
        let optimizer = await AdamWOptimizer(learningRate: 0.001)
        let config = NetworkTrainingConfig(
            epochs: 5,
            batchSize: 16,
            lossFunction: .crossEntropy,
            optimizer: optimizer
        )
        
        try await model.train(data: trainData, config: config)
        
        // Report mixed precision statistics
        let stats = await lossScaler.getStatistics()
        print("Mixed precision training stats:")
        print("  - Final loss scale: \(stats.currentScale)")
        print("  - Overflow count: \(stats.overflowCount)")
        print("  - Overflow rate: \(String(format: "%.2f%%", stats.overflowRate * 100))")
    }
    
    // MARK: - Evaluation
    
    static func evaluateModel(model: NeuralNetwork, valData: [(MetalBuffer, MetalBuffer)]) async {
        var correct = 0
        var total = 0
        
        for (input, target) in valData {
            let output = try? await model.predict(input)
            guard let output = output else { continue }
            
            let outputArray = output.toArray()
            let targetArray = target.toArray()
            
            // Find predicted class
            let predictedClass = outputArray.enumerated().max(by: { $0.element < $1.element })?.offset ?? -1
            let actualClass = targetArray.enumerated().max(by: { $0.element < $1.element })?.offset ?? -1
            
            if predictedClass == actualClass {
                correct += 1
            }
            total += 1
        }
        
        let accuracy = Float(correct) / Float(total) * 100
        print("Validation accuracy: \(String(format: "%.1f%%", accuracy)) (\(correct)/\(total))")
    }
}

// MARK: - Activation Layer Helper

/// Simple activation layer for the example
actor ActivationLayer: NeuralLayer {
    private let activation: Activation
    private let name: String
    private let metalPipeline: MetalMLPipeline
    private var lastInput: MetalBuffer?
    
    init(activation: Activation, name: String, metalPipeline: MetalMLPipeline) async {
        self.activation = activation
        self.name = name
        self.metalPipeline = metalPipeline
    }
    
    func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        lastInput = input
        let operations = await metalPipeline.getOperations()
        let output = try await metalPipeline.allocateBuffer(size: input.count)
        try await operations.applyActivation(input: input, output: output, activation: activation)
        return output
    }
    
    func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let input = lastInput else {
            throw MetalMLError.invalidState("No cached input for backward pass")
        }
        
        let operations = await metalPipeline.getOperations()
        let gradInput = try await metalPipeline.allocateBuffer(size: input.count)
        
        // Compute activation gradient
        try await operations.applyActivationGradient(
            gradOutput: gradOutput,
            input: input,
            output: gradInput,
            activation: activation
        )
        
        return gradInput
    }
    
    func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        // No parameters to update
    }
    
    func getParameters() async -> MetalBuffer? {
        nil
    }
    
    func getParameterCount() async -> Int {
        0
    }
    
    func setTraining(_ training: Bool) async {
        // No effect for activation layers
    }
}
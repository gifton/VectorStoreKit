// VectorStoreKit: Neural Network
//
// Metal-accelerated neural network with automatic differentiation

import Foundation
@preconcurrency import Metal

/// Neural network model with Metal acceleration
public actor NeuralNetwork {
    // MARK: - Properties
    private var layers: [any NeuralLayer]
    private let metalPipeline: MetalMLPipeline
    private let computationGraph: ComputationGraph
    private var trainingHistory: NetworkTrainingHistory
    
    // MARK: - Initialization
    public init(metalPipeline: MetalMLPipeline) async throws {
        self.metalPipeline = metalPipeline
        self.computationGraph = ComputationGraph(metalPipeline: metalPipeline)
        self.layers = []
        self.trainingHistory = NetworkTrainingHistory()
    }
    
    // MARK: - Layer Management
    public func addLayer(_ layer: any NeuralLayer) async {
        layers.append(layer)
    }
    
    public func addLayers(_ newLayers: [any NeuralLayer]) async {
        layers.append(contentsOf: newLayers)
    }
    
    // MARK: - Forward Pass
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        var output = input
        
        for layer in layers {
            output = try await layer.forward(output)
        }
        
        return output
    }
    
    // MARK: - Training
    public func train(
        data: [(input: MetalBuffer, target: MetalBuffer)],
        config: NetworkTrainingConfig
    ) async throws {
        var epochLosses: [Float] = []
        
        for epoch in 0..<config.epochs {
            var totalLoss: Float = 0
            var batchCount = 0
            
            // Shuffle data if requested
            let trainingData = config.shuffle ? data.shuffled() : data
            
            // Process mini-batches
            for batchStart in stride(from: 0, to: trainingData.count, by: config.batchSize) {
                let batchEnd = min(batchStart + config.batchSize, trainingData.count)
                var batchLoss: Float = 0
                
                // Clear computation graph for this batch
                await computationGraph.clear()
                
                // Process each sample in batch
                for idx in batchStart..<batchEnd {
                    let (input, target) = trainingData[idx]
                    
                    // Forward pass
                    let output = try await forward(input)
                    
                    // Compute loss
                    let loss = try await computeLoss(
                        prediction: output,
                        target: target,
                        lossFunction: config.lossFunction
                    )
                    
                    batchLoss += loss
                    
                    // Backward pass
                    let gradOutput = try await computeLossGradient(
                        prediction: output,
                        target: target,
                        lossFunction: config.lossFunction
                    )
                    
                    // Propagate gradients through layers
                    var currentGrad = gradOutput
                    for layer in layers.reversed() {
                        currentGrad = try await layer.backward(currentGrad)
                    }
                    
                    // Update parameters
                    for layer in layers {
                        try await layer.updateParameters(currentGrad, learningRate: config.learningRate)
                    }
                }
                
                batchLoss /= Float(batchEnd - batchStart)
                totalLoss += batchLoss
                batchCount += 1
            }
            
            let avgLoss = totalLoss / Float(batchCount)
            epochLosses.append(avgLoss)
            
            // Logging
            if epoch % config.logInterval == 0 {
                print("Epoch \(epoch): Loss = \(avgLoss)")
            }
            
            // Early stopping check
            if config.earlyStoppingPatience > 0 {
                if shouldStopEarly(losses: epochLosses, patience: config.earlyStoppingPatience) {
                    print("Early stopping at epoch \(epoch)")
                    break
                }
            }
        }
        
        // Update training history
        trainingHistory.losses = epochLosses
        trainingHistory.finalLoss = epochLosses.last ?? Float.infinity
    }
    
    // MARK: - Prediction
    public func predict(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Set all layers to evaluation mode
        for layer in layers {
            await layer.setTraining(false)
        }
        
        let output = try await forward(input)
        
        // Reset to training mode
        for layer in layers {
            await layer.setTraining(true)
        }
        
        return output
    }
    
    // MARK: - Loss Computation
    private func computeLoss(
        prediction: MetalBuffer,
        target: MetalBuffer,
        lossFunction: LossFunction
    ) async throws -> Float {
        guard prediction.count == target.count else {
            throw MetalMLError.incompatibleBufferSize(
                expected: target.count,
                actual: prediction.count
            )
        }
        
        // For now, compute on CPU - will be replaced with Metal kernel
        let predPtr = prediction.buffer.contents().bindMemory(to: Float.self, capacity: prediction.count)
        let targetPtr = target.buffer.contents().bindMemory(to: Float.self, capacity: target.count)
        
        var loss: Float = 0
        
        switch lossFunction {
        case .mse:
            for i in 0..<prediction.count {
                let diff = predPtr[i] - targetPtr[i]
                loss += diff * diff
            }
            loss /= Float(prediction.count)
            
        case .mae:
            for i in 0..<prediction.count {
                loss += abs(predPtr[i] - targetPtr[i])
            }
            loss /= Float(prediction.count)
            
        case .crossEntropy:
            for i in 0..<prediction.count {
                loss += -targetPtr[i] * log(max(predPtr[i], 1e-7))
            }
            
        case .binaryCrossEntropy:
            for i in 0..<prediction.count {
                loss += -targetPtr[i] * log(max(predPtr[i], 1e-7)) - (1 - targetPtr[i]) * log(max(1 - predPtr[i], 1e-7))
            }
            loss /= Float(prediction.count)
            
        case .huber:
            let delta: Float = 1.0
            for i in 0..<prediction.count {
                let diff = abs(predPtr[i] - targetPtr[i])
                if diff <= delta {
                    loss += 0.5 * diff * diff
                } else {
                    loss += delta * (diff - 0.5 * delta)
                }
            }
            loss /= Float(prediction.count)
        }
        
        return loss
    }
    
    private func computeLossGradient(
        prediction: MetalBuffer,
        target: MetalBuffer,
        lossFunction: LossFunction
    ) async throws -> MetalBuffer {
        let gradient = try await metalPipeline.allocateBuffer(size: prediction.count)
        
        let predPtr = prediction.buffer.contents().bindMemory(to: Float.self, capacity: prediction.count)
        let targetPtr = target.buffer.contents().bindMemory(to: Float.self, capacity: target.count)
        let gradPtr = gradient.buffer.contents().bindMemory(to: Float.self, capacity: gradient.count)
        
        switch lossFunction {
        case .mse:
            for i in 0..<prediction.count {
                gradPtr[i] = 2 * (predPtr[i] - targetPtr[i]) / Float(prediction.count)
            }
            
        case .mae:
            for i in 0..<prediction.count {
                gradPtr[i] = predPtr[i] > targetPtr[i] ? 1.0 / Float(prediction.count) : -1.0 / Float(prediction.count)
            }
            
        case .crossEntropy:
            // For softmax + cross entropy, gradient is simply prediction - target
            for i in 0..<prediction.count {
                gradPtr[i] = predPtr[i] - targetPtr[i]
            }
            
        case .binaryCrossEntropy:
            for i in 0..<prediction.count {
                gradPtr[i] = (predPtr[i] - targetPtr[i]) / (max(predPtr[i] * (1 - predPtr[i]), 1e-7) * Float(prediction.count))
            }
            
        case .huber:
            let delta: Float = 1.0
            for i in 0..<prediction.count {
                let diff = predPtr[i] - targetPtr[i]
                let absDiff = abs(diff)
                if absDiff <= delta {
                    gradPtr[i] = diff / Float(prediction.count)
                } else {
                    gradPtr[i] = delta * (diff > 0 ? 1 : -1) / Float(prediction.count)
                }
            }
        }
        
        return gradient
    }
    
    // MARK: - Model Management
    public func parameterCount() async -> Int {
        var count = 0
        for layer in layers {
            count += await layer.getParameterCount()
        }
        return count
    }
    
    public func getTrainingHistory() async -> NetworkTrainingHistory {
        trainingHistory
    }
    
    // MARK: - Compatibility API for Autoencoders
    
    /// Set training mode for all layers
    public func setTraining(_ training: Bool) async {
        for layer in layers {
            await layer.setTraining(training)
        }
    }
    
    /// Forward pass with Float array (compatibility)
    public func forward(_ input: [Float]) async -> [Float] {
        do {
            // Convert input to MetalBuffer with shape
            let inputBuffer = try await metalPipeline.allocateBuffer(shape: TensorShape(input.count))
            let ptr = inputBuffer.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
            for i in 0..<input.count {
                ptr[i] = input[i]
            }
            
            // Run forward pass
            let outputBuffer = try await forward(inputBuffer)
            
            // Convert output to Float array
            let outputPtr = outputBuffer.buffer.contents().bindMemory(to: Float.self, capacity: outputBuffer.count)
            var output = [Float](repeating: 0, count: outputBuffer.count)
            for i in 0..<outputBuffer.count {
                output[i] = outputPtr[i]
            }
            
            // Release buffers
            await metalPipeline.releaseBuffer(inputBuffer)
            await metalPipeline.releaseBuffer(outputBuffer)
            
            return output
        } catch {
            print("Error in forward pass: \(error)")
            return [Float](repeating: 0, count: input.count)
        }
    }
    
    /// Backward pass with Float arrays (compatibility)
    public func backward(_ gradOutput: [Float], learningRate: Float) async -> [LayerGradients] {
        do {
            // Convert gradient to MetalBuffer
            let gradBuffer = try await metalPipeline.allocateBuffer(size: gradOutput.count)
            let ptr = gradBuffer.buffer.contents().bindMemory(to: Float.self, capacity: gradOutput.count)
            for i in 0..<gradOutput.count {
                ptr[i] = gradOutput[i]
            }
            
            // Propagate gradients through layers
            var currentGrad = gradBuffer
            var allGradients: [LayerGradients] = []
            
            for layer in layers.reversed() {
                currentGrad = try await layer.backward(currentGrad)
                
                // For now, create empty LayerGradients
                // This would need to be expanded based on layer type
                allGradients.append(LayerGradients())
            }
            
            // Update parameters
            for layer in layers {
                try await layer.updateParameters(currentGrad, learningRate: learningRate)
            }
            
            // Release buffer
            await metalPipeline.releaseBuffer(gradBuffer)
            
            return allGradients.reversed()
        } catch {
            print("Error in backward pass: \(error)")
            return []
        }
    }
    
    /// Get input gradient for encoder-decoder architectures
    public func getInputGradient() async -> [Float] {
        // This is a placeholder - would need proper implementation
        return []
    }
    
    /// Update weights with gradients (compatibility)
    public func updateWeights(gradients: [LayerGradients], optimizer: any Optimizer) async {
        // This is a simplified implementation
        // In practice, each layer would handle its own gradient updates
        for (layer, gradient) in zip(layers, gradients) {
            if let weightsGrad = gradient.weights {
                let buffer = try? await metalPipeline.allocateBuffer(size: weightsGrad.count)
                if let buffer = buffer {
                    let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: weightsGrad.count)
                    for i in 0..<weightsGrad.count {
                        ptr[i] = weightsGrad[i]
                    }
                    let lr = await optimizer.getCurrentLearningRate()
                    try? await layer.updateParameters(buffer, learningRate: lr)
                    await metalPipeline.releaseBuffer(buffer)
                }
            }
        }
    }
    
    /// Get layer count for autoencoder compatibility
    public var layerCount: Int {
        get async { layers.count }
    }
    
    // MARK: - Private Methods
    private func shouldStopEarly(losses: [Float], patience: Int) -> Bool {
        guard losses.count > patience else { return false }
        
        let recentLosses = Array(losses.suffix(patience))
        let minRecentLoss = recentLosses.min() ?? Float.infinity
        let previousLosses = Array(losses.dropLast(patience))
        let minPreviousLoss = previousLosses.min() ?? Float.infinity
        
        return minRecentLoss >= minPreviousLoss
    }
}

// MARK: - Supporting Types

/// Training configuration
public struct NetworkTrainingConfig: Sendable {
    public let epochs: Int
    public let batchSize: Int
    public let learningRate: Float
    public let lossFunction: LossFunction
    public let shuffle: Bool
    public let logInterval: Int
    public let earlyStoppingPatience: Int
    
    public init(
        epochs: Int = 100,
        batchSize: Int = 32,
        learningRate: Float = 0.01,
        lossFunction: LossFunction = .mse,
        shuffle: Bool = true,
        logInterval: Int = 10,
        earlyStoppingPatience: Int = 0
    ) {
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.lossFunction = lossFunction
        self.shuffle = shuffle
        self.logInterval = logInterval
        self.earlyStoppingPatience = earlyStoppingPatience
    }
}

/// Training history
public struct NetworkTrainingHistory: Codable, Sendable {
    public var losses: [Float] = []
    public var validationLosses: [Float] = []
    public var metrics: [String: [Float]] = [:]
    public var finalLoss: Float = Float.infinity
    public var bestEpoch: Int = 0
}

/// Loss functions
public enum LossFunction: String, Codable, Sendable {
    case mse = "mean_squared_error"
    case mae = "mean_absolute_error"
    case crossEntropy = "cross_entropy"
    case binaryCrossEntropy = "binary_cross_entropy"
    case huber = "huber"
}

/// Neural network errors
public enum NeuralNetworkError: LocalizedError {
    case mismatchedData(inputCount: Int, targetCount: Int)
    case invalidArchitecture(String)
    case trainingFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .mismatchedData(let inputCount, let targetCount):
            return "Mismatched data: \(inputCount) inputs vs \(targetCount) targets"
        case .invalidArchitecture(let message):
            return "Invalid architecture: \(message)"
        case .trainingFailed(let message):
            return "Training failed: \(message)"
        }
    }
}
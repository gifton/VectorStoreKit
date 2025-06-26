// VectorStoreKit: Neural Network
//
// Metal-accelerated neural network with automatic differentiation

import Foundation
@preconcurrency import Metal

/// Neural network model with Metal acceleration
public actor NeuralNetwork {
    // MARK: - Properties
    internal var layers: [any NeuralLayer]
    internal let metalPipeline: MetalMLPipeline
    internal var trainingHistory: NetworkTrainingHistory
    
    // MARK: - Initialization
    public init(metalPipeline: MetalMLPipeline) async throws {
        self.metalPipeline = metalPipeline
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
                let actualBatchSize = batchEnd - batchStart
                
                // Prepare batch data
                var batchInputs: [MetalBuffer] = []
                var batchTargets: [MetalBuffer] = []
                
                for idx in batchStart..<batchEnd {
                    let (input, target) = trainingData[idx]
                    batchInputs.append(input)
                    batchTargets.append(target)
                }
                
                // Concatenate batch inputs and targets
                let batchInput = try await concatenateBatch(batchInputs)
                let batchTarget = try await concatenateBatch(batchTargets)
                
                // Forward pass on entire batch
                let batchOutput = try await forward(batchInput)
                
                // Compute loss on batch
                let batchLoss = try await computeLoss(
                    prediction: batchOutput,
                    target: batchTarget,
                    lossFunction: config.lossFunction
                )
                
                // Backward pass on batch
                let gradOutput = try await computeLossGradient(
                    prediction: batchOutput,
                    target: batchTarget,
                    lossFunction: config.lossFunction
                )
                
                // Zero gradients before accumulation
                for layer in layers {
                    await layer.zeroGradients()
                }
                
                // Propagate gradients through layers
                var currentGrad = gradOutput
                for layer in layers.reversed() {
                    currentGrad = try await layer.backward(currentGrad)
                }
                
                // Scale gradients by batch size for proper averaging
                let scaleFactor = 1.0 / Float(actualBatchSize)
                for layer in layers {
                    await layer.scaleGradients(scaleFactor)
                }
                
                // Update parameters using optimizer
                if let optimizer = config.optimizer {
                    for layer in layers {
                        try await layer.updateParametersWithOptimizer(optimizer)
                    }
                } else {
                    // Simple SGD update
                    for layer in layers {
                        // updateParameters expects the layer's accumulated gradients, not currentGrad
                        if let params = await layer.getParameters() {
                            try await layer.updateParameters(params, learningRate: config.learningRate)
                        }
                    }
                }
                
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
        
        // Allocate buffers for loss computation
        let lossBuffer = try await metalPipeline.allocateBuffer(size: prediction.count)
        let totalLossBuffer = try await metalPipeline.allocateBuffer(size: 1)
        
        // Initialize total loss to 0
        let totalLossPtr = totalLossBuffer.buffer.contents().bindMemory(to: Float.self, capacity: 1)
        totalLossPtr[0] = 0.0
        
        // Get shader library and create command buffer
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        // Select appropriate loss kernel
        let functionName: String
        switch lossFunction {
        case .mse:
            functionName = MLShaderLibrary.LossFunction.mseLoss.rawValue
        case .crossEntropy:
            functionName = MLShaderLibrary.LossFunction.crossEntropyLoss.rawValue
        }
        
        let pipeline = try await shaderLibrary.pipeline(for: functionName)
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(prediction.buffer, offset: 0, index: 0)
            encoder.setBuffer(target.buffer, offset: 0, index: 1)
            encoder.setBuffer(lossBuffer.buffer, offset: 0, index: 2)
            encoder.setBuffer(totalLossBuffer.buffer, offset: 0, index: 3)
            
            switch lossFunction {
            case .crossEntropy:
                // For cross entropy, we need batch size and num classes
                // Assuming single batch for now
                var batchSize = UInt32(1)
                var numClasses = UInt32(prediction.count)
                encoder.setBytes(&batchSize, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.setBytes(&numClasses, length: MemoryLayout<UInt32>.size, index: 5)
            default:
                var size = UInt32(prediction.count)
                encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 4)
            }
            
            let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadgroupCount = MTLSize(
                width: (prediction.count + threadgroupSize.width - 1) / threadgroupSize.width,
                height: 1,
                depth: 1
            )
            
            encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // Read back the total loss
        let totalLoss = totalLossPtr[0]
        
        // Release buffers
        await metalPipeline.releaseBuffer(lossBuffer)
        await metalPipeline.releaseBuffer(totalLossBuffer)
        
        return totalLoss
    }
    
    private func computeLossGradient(
        prediction: MetalBuffer,
        target: MetalBuffer,
        lossFunction: LossFunction
    ) async throws -> MetalBuffer {
        let gradient = try await metalPipeline.allocateBuffer(size: prediction.count)
        
        // Get shader library and create command buffer
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        // Select appropriate gradient kernel
        let functionName: String
        switch lossFunction {
        case .mse:
            functionName = MLShaderLibrary.LossFunction.mseGradient.rawValue
        case .crossEntropy:
            functionName = MLShaderLibrary.LossFunction.crossEntropyGradient.rawValue
        }
        
        let pipeline = try await shaderLibrary.pipeline(for: functionName)
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(prediction.buffer, offset: 0, index: 0)
            encoder.setBuffer(target.buffer, offset: 0, index: 1)
            encoder.setBuffer(gradient.buffer, offset: 0, index: 2)
            
            switch lossFunction {
            case .crossEntropy:
                // For cross entropy gradient, we need batch size and num classes
                var batchSize = UInt32(1)
                var numClasses = UInt32(prediction.count)
                encoder.setBytes(&batchSize, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&numClasses, length: MemoryLayout<UInt32>.size, index: 4)
                
                // Use 2D thread configuration for cross entropy
                let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
                let threadgroupCount = MTLSize(
                    width: (1 + threadgroupSize.width - 1) / threadgroupSize.width,
                    height: (prediction.count + threadgroupSize.height - 1) / threadgroupSize.height,
                    depth: 1
                )
                encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
            default:
                var size = UInt32(prediction.count)
                encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
                
                let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
                let threadgroupCount = MTLSize(
                    width: (prediction.count + threadgroupSize.width - 1) / threadgroupSize.width,
                    height: 1,
                    depth: 1
                )
                encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
            }
            
            encoder.endEncoding()
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
    
    // MARK: - Batch Processing Helpers
    
    private func concatenateBatch(_ buffers: [MetalBuffer]) async throws -> MetalBuffer {
        guard !buffers.isEmpty else {
            throw MetalMLError.invalidBufferSize("Empty batch")
        }
        
        let elementSize = buffers[0].count
        let totalSize = buffers.count * elementSize
        let concatenated = try await metalPipeline.allocateBuffer(size: totalSize)
        
        // Copy each buffer to the appropriate offset
        let ptr = concatenated.buffer.contents().bindMemory(to: Float.self, capacity: totalSize)
        for (i, buffer) in buffers.enumerated() {
            let bufferPtr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
            let offset = i * elementSize
            for j in 0..<buffer.count {
                ptr[offset + j] = bufferPtr[j]
            }
        }
        
        return concatenated
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
    public let optimizer: (any Optimizer)?
    public let gradientClipping: Float?
    
    public init(
        epochs: Int = 100,
        batchSize: Int = 32,
        learningRate: Float = 0.01,
        lossFunction: LossFunction = .mse,
        shuffle: Bool = true,
        logInterval: Int = 10,
        earlyStoppingPatience: Int = 0,
        optimizer: (any Optimizer)? = nil,
        gradientClipping: Float? = nil
    ) {
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.lossFunction = lossFunction
        self.shuffle = shuffle
        self.logInterval = logInterval
        self.earlyStoppingPatience = earlyStoppingPatience
        self.optimizer = optimizer
        self.gradientClipping = gradientClipping
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
    case crossEntropy = "cross_entropy"
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
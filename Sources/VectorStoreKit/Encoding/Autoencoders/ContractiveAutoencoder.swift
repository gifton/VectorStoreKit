// VectorStoreKit: Contractive Autoencoder Implementation
//
// Contractive autoencoder with Jacobian penalty for robust feature learning

import Foundation
import Accelerate
@preconcurrency import Metal

/// Configuration for contractive autoencoder
public struct ContractiveAutoencoderConfiguration: AutoencoderConfiguration {
    public let inputDimensions: Int
    public let encodedDimensions: Int
    public let encoderLayers: [Int]
    public let decoderLayers: [Int]
    public let contractiveWeight: Float
    public let training: TrainingConfiguration
    public let regularization: RegularizationConfig
    
    public init(
        inputDimensions: Int,
        encodedDimensions: Int,
        encoderLayers: [Int] = [512, 256],
        decoderLayers: [Int] = [256, 512],
        contractiveWeight: Float = 0.1,
        training: TrainingConfiguration = TrainingConfiguration(),
        regularization: RegularizationConfig = RegularizationConfig()
    ) {
        self.inputDimensions = inputDimensions
        self.encodedDimensions = encodedDimensions
        self.encoderLayers = encoderLayers
        self.decoderLayers = decoderLayers
        self.contractiveWeight = contractiveWeight
        self.training = training
        self.regularization = regularization
    }
}

/// Contractive autoencoder implementation
public actor ContractiveAutoencoder: Autoencoder {
    public typealias Config = ContractiveAutoencoderConfiguration
    
    // MARK: - Properties
    
    private let configuration: ContractiveAutoencoderConfiguration
    private let encoder: NeuralNetwork
    private let decoder: NeuralNetwork
    private var trained: Bool = false
    private var trainingHistory: TrainingHistory
    private let metalPipeline: MetalMLPipeline?
    
    public var isTrained: Bool {
        trained
    }
    
    // MARK: - Initialization
    
    public init(
        configuration: ContractiveAutoencoderConfiguration,
        metalPipeline: MetalMLPipeline? = nil
    ) async throws {
        self.configuration = configuration
        self.metalPipeline = metalPipeline
        self.trainingHistory = TrainingHistory()
        
        // Build encoder network
        var encoderLayers: [any NeuralLayer] = []
        var currentInput = configuration.inputDimensions
        
        // Create metalPipeline if not provided
        let pipeline: MetalMLPipeline
        if let metalPipeline = metalPipeline {
            pipeline = metalPipeline
        } else {
            guard let device = MTLCreateSystemDefaultDevice() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            pipeline = try await MetalMLPipeline(device: device)
        }
        
        // Hidden layers
        for hiddenSize in configuration.encoderLayers {
            encoderLayers.append(try await DenseLayer(
                inputSize: currentInput,
                outputSize: hiddenSize,
                activation: .sigmoid,  // Sigmoid for smooth gradients
                metalPipeline: pipeline
            ))
            currentInput = hiddenSize
        }
        
        // Output layer with sigmoid activation
        encoderLayers.append(try await DenseLayer(
            inputSize: currentInput,
            outputSize: configuration.encodedDimensions,
            activation: .sigmoid,
            metalPipeline: pipeline
        ))
        
        // Build decoder network
        var decoderLayers: [any NeuralLayer] = []
        currentInput = configuration.encodedDimensions
        
        // Hidden layers
        for hiddenSize in configuration.decoderLayers {
            decoderLayers.append(try await DenseLayer(
                inputSize: currentInput,
                outputSize: hiddenSize,
                activation: .sigmoid,
                metalPipeline: pipeline
            ))
            currentInput = hiddenSize
        }
        
        // Output layer
        decoderLayers.append(try await DenseLayer(
            inputSize: currentInput,
            outputSize: configuration.inputDimensions,
            activation: .sigmoid,
            metalPipeline: pipeline
        ))
        
        // Initialize networks
        self.encoder = try await NeuralNetwork(metalPipeline: pipeline)
        await self.encoder.addLayers(encoderLayers)
        
        self.decoder = try await NeuralNetwork(metalPipeline: pipeline)
        await self.decoder.addLayers(decoderLayers)
    }
    
    // MARK: - Encoding/Decoding
    
    public func encode(_ vectors: [[Float]]) async throws -> [[Float]] {
        guard trained else {
            throw AutoencoderError.notTrained
        }
        
        // Set to evaluation mode
        await encoder.setTraining(false)
        
        var results: [[Float]] = []
        for vector in vectors {
            let encoded = await encoder.forward(vector)
            results.append(encoded)
        }
        
        return results
    }
    
    public func decode(_ embeddings: [[Float]]) async throws -> [[Float]] {
        guard trained else {
            throw AutoencoderError.notTrained
        }
        
        // Set to evaluation mode
        await decoder.setTraining(false)
        
        var results: [[Float]] = []
        for embedding in embeddings {
            let decoded = await decoder.forward(embedding)
            results.append(decoded)
        }
        
        return results
    }
    
    public func reconstruct(_ vectors: [[Float]]) async throws -> [[Float]] {
        let encoded = try await encode(vectors)
        return try await decode(encoded)
    }
    
    public func reconstructionError(_ vectors: [[Float]]) async throws -> Float {
        let reconstructed = try await reconstruct(vectors)
        
        var totalError: Float = 0
        for (original, recon) in zip(vectors, reconstructed) {
            let error = zip(original, recon).map { pow($0 - $1, 2) }.reduce(0, +)
            totalError += error / Float(original.count)  // MSE
        }
        
        return totalError / Float(vectors.count)
    }
    
    // MARK: - Training
    
    public func train(on data: [[Float]], validationData: [[Float]]? = nil) async throws {
        guard data.count > configuration.training.batchSize else {
            throw AutoencoderError.insufficientTrainingData
        }
        
        // Set to training mode
        await encoder.setTraining(true)
        await decoder.setTraining(true)
        
        // Create optimizer
        let optimizer = createOptimizer()
        
        // Create loss function
        let lossFunction = LossFunction.mse
        
        // Training loop
        for epoch in 0..<configuration.training.epochs {
            var epochLoss: Float = 0
            var epochReconLoss: Float = 0
            var epochContractLoss: Float = 0
            var batchCount = 0
            
            // Shuffle data
            let shuffled = data.shuffled()
            
            // Process batches
            for batchStart in stride(from: 0, to: shuffled.count, by: configuration.training.batchSize) {
                let batchEnd = min(batchStart + configuration.training.batchSize, shuffled.count)
                let batch = Array(shuffled[batchStart..<batchEnd])
                
                // Compute batch loss and gradients
                let (loss, reconLoss, contractLoss, gradients) = await computeContractiveBatchLossAndGradients(
                    batch: batch,
                    lossFunction: lossFunction,
                    optimizer: optimizer
                )
                
                epochLoss += loss
                epochReconLoss += reconLoss
                epochContractLoss += contractLoss
                batchCount += 1
                
                // Apply gradients
                await applyGradients(gradients, optimizer: optimizer)
                
                // Update learning rate if needed
                if let schedule = configuration.training.lrSchedule {
                    await updateLearningRate(
                        optimizer: optimizer,
                        schedule: schedule,
                        epoch: epoch,
                        step: batchCount
                    )
                }
            }
            
            // Average epoch losses
            epochLoss /= Float(batchCount)
            epochReconLoss /= Float(batchCount)
            epochContractLoss /= Float(batchCount)
            
            // Validation
            let validationLoss = if let validationData = validationData {
                try await computeValidationLoss(validationData, lossFunction: lossFunction)
            } else {
                epochLoss
            }
            
            // Update training history
            trainingHistory.addEpoch(
                epoch: epoch,
                trainLoss: epochLoss,
                validationLoss: validationLoss
            )
            
            // Log progress
            if epoch % 10 == 0 {
                print("Epoch \(epoch): Loss = \(epochLoss) (Recon: \(epochReconLoss), Contract: \(epochContractLoss)), Val Loss = \(validationLoss)")
            }
            
            // Early stopping
            if trainingHistory.shouldStop() {
                print("Early stopping at epoch \(epoch)")
                break
            }
        }
        
        trained = true
        
        // Set back to evaluation mode
        await encoder.setTraining(false)
        await decoder.setTraining(false)
    }
    
    // MARK: - Private Methods
    
    private func createOptimizer() -> any Optimizer {
        let learningRate = configuration.training.learningRate
        
        switch configuration.training.optimizerType {
        case .sgd:
            return SGD(learningRate: learningRate, momentum: 0.9)
        case .adam:
            return Adam(learningRate: learningRate)
        case .rmsprop:
            return RMSprop(learningRate: learningRate)
        case .adagrad:
            return AdaGrad(learningRate: learningRate)
        }
    }
    
    private func computeContractiveBatchLossAndGradients(
        batch: [[Float]],
        lossFunction: LossFunction,
        optimizer: any Optimizer
    ) async -> (loss: Float, reconLoss: Float, contractLoss: Float, gradients: [LayerGradients]) {
        var totalLoss: Float = 0
        var totalReconLoss: Float = 0
        var totalContractLoss: Float = 0
        var accumulatedEncoderGrads: [LayerGradients] = []
        var accumulatedDecoderGrads: [LayerGradients] = []
        
        for input in batch {
            // Forward pass through encoder
            let encoded = await encoder.forward(input)
            
            // Forward pass through decoder
            let reconstructed = await decoder.forward(encoded)
            
            // Compute reconstruction loss
            let reconLoss = lossFunction.compute(prediction: reconstructed, target: input)
            totalReconLoss += reconLoss
            
            // Compute contractive penalty (Frobenius norm of Jacobian)
            let jacobianPenalty = await computeJacobianPenalty(
                input: input,
                encoded: encoded
            )
            totalContractLoss += jacobianPenalty
            
            // Total loss
            let loss = reconLoss + configuration.contractiveWeight * jacobianPenalty
            totalLoss += loss
            
            // Compute reconstruction gradient
            let reconGrad = lossFunction.gradient(prediction: reconstructed, target: input)
            
            // Backward pass through decoder
            let currentLR = await optimizer.getCurrentLearningRate()
            let decoderGrads = await decoder.backward(reconGrad, learningRate: currentLR)
            
            // Get gradient for encoder from decoder's input gradient
            var encoderOutputGrad = await decoder.getInputGradient()
            
            // Add contractive gradient
            let contractiveGrad = await computeContractiveGradient(
                input: input,
                encoded: encoded
            )
            
            // Combine gradients
            for i in 0..<encoderOutputGrad.count {
                encoderOutputGrad[i] += configuration.contractiveWeight * contractiveGrad[i]
            }
            
            // Backward pass through encoder
            let encoderGrads = await encoder.backward(encoderOutputGrad, learningRate: currentLR)
            
            // Accumulate gradients
            if accumulatedEncoderGrads.isEmpty {
                accumulatedEncoderGrads = encoderGrads
                accumulatedDecoderGrads = decoderGrads
            } else {
                for i in 0..<encoderGrads.count {
                    accumulatedEncoderGrads[i] = addGradients(accumulatedEncoderGrads[i], encoderGrads[i])
                }
                for i in 0..<decoderGrads.count {
                    accumulatedDecoderGrads[i] = addGradients(accumulatedDecoderGrads[i], decoderGrads[i])
                }
            }
        }
        
        // Average gradients
        let batchSize = Float(batch.count)
        let avgEncoderGrads = accumulatedEncoderGrads.map { scaleGradients($0, by: 1.0 / batchSize) }
        let avgDecoderGrads = accumulatedDecoderGrads.map { scaleGradients($0, by: 1.0 / batchSize) }
        
        // Combine gradients
        let allGradients = avgEncoderGrads + avgDecoderGrads
        
        return (
            totalLoss / batchSize,
            totalReconLoss / batchSize,
            totalContractLoss / batchSize,
            allGradients
        )
    }
    
    private func computeJacobianPenalty(
        input: [Float],
        encoded: [Float]
    ) async -> Float {
        // Compute approximate Jacobian penalty using finite differences
        let epsilon: Float = 1e-4
        var jacobianFrobeniusSquared: Float = 0
        
        // For each input dimension
        for i in 0..<input.count {
            // Create perturbed input
            var perturbedInput = input
            perturbedInput[i] += epsilon
            
            // Get encoded representation of perturbed input
            let perturbedEncoded = await encoder.forward(perturbedInput)
            
            // Compute finite difference approximation of Jacobian column
            for j in 0..<encoded.count {
                let jacobianElement = (perturbedEncoded[j] - encoded[j]) / epsilon
                jacobianFrobeniusSquared += jacobianElement * jacobianElement
            }
        }
        
        return jacobianFrobeniusSquared / Float(input.count)
    }
    
    private func computeContractiveGradient(
        input: [Float],
        encoded: [Float]
    ) async -> [Float] {
        // Gradient of contractive penalty with respect to encoder output
        // This is a simplified approximation
        var gradient = Array(repeating: Float(0), count: encoded.count)
        
        // For sigmoid activation, the gradient involves h(1-h)
        for i in 0..<encoded.count {
            let h = encoded[i]
            gradient[i] = 2.0 * h * (1 - h) * (1 - 2 * h)
        }
        
        return gradient
    }
    
    private func addGradients(_ g1: LayerGradients, _ g2: LayerGradients) -> LayerGradients {
        let addedWeights: [Float]? = if let w1 = g1.weights, let w2 = g2.weights {
            zip(w1, w2).map { $0 + $1 }
        } else {
            nil
        }
        
        let addedBias: [Float]? = if let b1 = g1.bias, let b2 = g2.bias {
            zip(b1, b2).map { $0 + $1 }
        } else {
            nil
        }
        
        return LayerGradients(weights: addedWeights, bias: addedBias)
    }
    
    private func scaleGradients(_ gradients: LayerGradients, by scale: Float) -> LayerGradients {
        let scaledWeights = gradients.weights?.map { $0 * scale }
        let scaledBias = gradients.bias?.map { $0 * scale }
        
        return LayerGradients(weights: scaledWeights, bias: scaledBias)
    }
    
    private func applyGradients(_ gradients: [LayerGradients], optimizer: any Optimizer) async {
        // Apply gradients to encoder and decoder
        let encoderGradCount = await encoder.layerCount
        let encoderGrads = Array(gradients.prefix(encoderGradCount))
        let decoderGrads = Array(gradients.suffix(from: encoderGradCount))
        
        await encoder.updateWeights(gradients: encoderGrads, optimizer: optimizer)
        await decoder.updateWeights(gradients: decoderGrads, optimizer: optimizer)
    }
    
    private func updateLearningRate(
        optimizer: any Optimizer,
        schedule: TrainingConfiguration.LearningRateSchedule,
        epoch: Int,
        step: Int
    ) async {
        switch schedule {
        case .exponentialDecay(let rate):
            if step == 1 {
                let newLR = configuration.training.learningRate * pow(rate, Float(epoch))
                await optimizer.setLearningRate(newLR)
            }
            
        case .stepDecay(let stepSize, let gamma):
            if epoch % stepSize == 0 && step == 1 {
                let currentLR = await optimizer.getCurrentLearningRate()
                await optimizer.setLearningRate(currentLR * gamma)
            }
            
        case .cosineAnnealing(let tMax):
            if step == 1 {
                let progress = Float(epoch % tMax) / Float(tMax)
                let newLR = configuration.training.learningRate * 0.5 * (1 + cos(Float.pi * progress))
                await optimizer.setLearningRate(newLR)
            }
            
        case .warmupCosine(let warmupSteps):
            let totalSteps = epoch * step
            if totalSteps < warmupSteps {
                let warmupProgress = Float(totalSteps) / Float(warmupSteps)
                let newLR = configuration.training.learningRate * warmupProgress
                await optimizer.setLearningRate(newLR)
            } else {
                let progress = Float(epoch) / Float(configuration.training.epochs)
                let newLR = configuration.training.learningRate * 0.5 * (1 + cos(Float.pi * progress))
                await optimizer.setLearningRate(newLR)
            }
        }
    }
    
    private func computeValidationLoss(
        _ data: [[Float]],
        lossFunction: LossFunction
    ) async throws -> Float {
        // Set to evaluation mode
        await encoder.setTraining(false)
        await decoder.setTraining(false)
        
        var totalLoss: Float = 0
        
        for input in data {
            let encoded = await encoder.forward(input)
            let reconstructed = await decoder.forward(encoded)
            
            // Only reconstruction loss for validation (no contractive penalty)
            let loss = lossFunction.compute(prediction: reconstructed, target: input)
            totalLoss += loss
        }
        
        // Set back to training mode
        await encoder.setTraining(true)
        await decoder.setTraining(true)
        
        return totalLoss / Float(data.count)
    }
}
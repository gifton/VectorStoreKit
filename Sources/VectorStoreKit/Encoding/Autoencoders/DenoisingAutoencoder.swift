// VectorStoreKit: Denoising Autoencoder Implementation
//
// Denoising autoencoder that learns to reconstruct clean data from corrupted inputs

import Foundation
@preconcurrency import Metal

/// Configuration for denoising autoencoder
public struct DenoisingAutoencoderConfiguration: AutoencoderConfiguration {
    public let inputDimensions: Int
    public let encodedDimensions: Int
    public let encoderLayers: [Int]
    public let decoderLayers: [Int]
    public let noiseLevel: Float
    public let noiseType: NoiseType
    public let training: AutoencoderTrainingConfiguration
    public let regularization: RegularizationConfig
    
    public enum NoiseType: String, Sendable, Codable {
        case gaussian = "gaussian"
        case masking = "masking"
        case saltAndPepper = "saltAndPepper"
        case combined = "combined"
    }
    
    public init(
        inputDimensions: Int,
        encodedDimensions: Int,
        encoderLayers: [Int] = [512, 256],
        decoderLayers: [Int] = [256, 512],
        noiseLevel: Float = 0.3,
        noiseType: NoiseType = .gaussian,
        training: AutoencoderTrainingConfiguration = AutoencoderTrainingConfiguration(),
        regularization: RegularizationConfig = RegularizationConfig()
    ) {
        self.inputDimensions = inputDimensions
        self.encodedDimensions = encodedDimensions
        self.encoderLayers = encoderLayers
        self.decoderLayers = decoderLayers
        self.noiseLevel = noiseLevel
        self.noiseType = noiseType
        self.training = training
        self.regularization = regularization
    }
}

/// Denoising autoencoder implementation
public actor DenoisingAutoencoder: Autoencoder {
    public typealias Config = DenoisingAutoencoderConfiguration
    
    // MARK: - Properties
    
    private let configuration: DenoisingAutoencoderConfiguration
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
        configuration: DenoisingAutoencoderConfiguration,
        metalPipeline: MetalMLPipeline? = nil
    ) async throws {
        self.configuration = configuration
        self.metalPipeline = metalPipeline
        self.trainingHistory = TrainingHistory()
        
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
        
        // Build encoder network
        var encoderLayers: [any NeuralLayer] = []
        var currentInput = configuration.inputDimensions
        
        // Hidden layers
        for hiddenSize in configuration.encoderLayers {
            encoderLayers.append(try await DenseLayer(
                inputSize: currentInput,
                outputSize: hiddenSize,
                activation: .relu,
                metalPipeline: pipeline
            ))
            
            // Add dropout if specified
            if configuration.regularization.dropout > 0 {
                encoderLayers.append(try await DropoutLayer(
                    rate: configuration.regularization.dropout,
                    metalPipeline: pipeline
                ))
            }
            
            currentInput = hiddenSize
        }
        
        // Output layer
        encoderLayers.append(try await DenseLayer(
            inputSize: currentInput,
            outputSize: configuration.encodedDimensions,
            activation: .linear,
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
                activation: .relu,
                metalPipeline: pipeline
            ))
            
            // Add dropout if specified
            if configuration.regularization.dropout > 0 {
                decoderLayers.append(try await DropoutLayer(
                    rate: configuration.regularization.dropout,
                    metalPipeline: pipeline
                ))
            }
            
            currentInput = hiddenSize
        }
        
        // Output layer with sigmoid for bounded outputs
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
    
    /// Denoise input vectors
    public func denoise(_ noisyVectors: [[Float]]) async throws -> [[Float]] {
        // Denoising is just reconstruction for a denoising autoencoder
        return try await reconstruct(noisyVectors)
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
            var batchCount = 0
            
            // Shuffle data
            let shuffled = data.shuffled()
            
            // Process batches
            for batchStart in stride(from: 0, to: shuffled.count, by: configuration.training.batchSize) {
                let batchEnd = min(batchStart + configuration.training.batchSize, shuffled.count)
                let batch = Array(shuffled[batchStart..<batchEnd])
                
                // Compute batch loss and gradients
                let (loss, gradients) = await computeDenoisingBatchLossAndGradients(
                    batch: batch,
                    lossFunction: lossFunction,
                    optimizer: optimizer
                )
                
                epochLoss += loss
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
            
            // Average epoch loss
            epochLoss /= Float(batchCount)
            
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
                print("Epoch \(epoch): Train Loss = \(epochLoss), Val Loss = \(validationLoss)")
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
    
    private func addNoise(to vector: [Float]) -> [Float] {
        switch configuration.noiseType {
        case .gaussian:
            return addGaussianNoise(to: vector, level: configuration.noiseLevel)
            
        case .masking:
            return addMaskingNoise(to: vector, probability: configuration.noiseLevel)
            
        case .saltAndPepper:
            return addSaltAndPepperNoise(to: vector, probability: configuration.noiseLevel)
            
        case .combined:
            // Apply multiple noise types
            var noisy = vector
            noisy = addGaussianNoise(to: noisy, level: configuration.noiseLevel * 0.5)
            noisy = addMaskingNoise(to: noisy, probability: configuration.noiseLevel * 0.3)
            return noisy
        }
    }
    
    private func addGaussianNoise(to vector: [Float], level: Float) -> [Float] {
        vector.map { value in
            // Box-Muller transform for normal distribution
            let u1 = Float.random(in: 0.0001...0.9999)
            let u2 = Float.random(in: 0.0001...0.9999)
            let gaussian = sqrt(-2.0 * log(u1)) * cos(2.0 * Float.pi * u2)
            return value + gaussian * level
        }
    }
    
    private func addMaskingNoise(to vector: [Float], probability: Float) -> [Float] {
        vector.map { value in
            Float.random(in: 0...1) < probability ? 0 : value
        }
    }
    
    private func addSaltAndPepperNoise(to vector: [Float], probability: Float) -> [Float] {
        vector.map { value in
            let rand = Float.random(in: 0...1)
            if rand < probability / 2 {
                return 0  // Pepper (minimum value)
            } else if rand < probability {
                return 1  // Salt (maximum value)
            } else {
                return value
            }
        }
    }
    
    private func computeDenoisingBatchLossAndGradients(
        batch: [[Float]],
        lossFunction: LossFunction,
        optimizer: any Optimizer
    ) async -> (loss: Float, gradients: [LayerGradients]) {
        var totalLoss: Float = 0
        var accumulatedEncoderGrads: [LayerGradients] = []
        var accumulatedDecoderGrads: [LayerGradients] = []
        
        for cleanInput in batch {
            // Add noise to input
            let noisyInput = addNoise(to: cleanInput)
            
            // Forward pass through encoder with noisy input
            let encoded = await encoder.forward(noisyInput)
            
            // Forward pass through decoder
            let reconstructed = await decoder.forward(encoded)
            
            // Compute loss against clean input
            let loss = lossFunction.compute(prediction: reconstructed, target: cleanInput)
            totalLoss += loss
            
            // Compute output gradient (compare with clean input)
            let outputGrad = lossFunction.gradient(prediction: reconstructed, target: cleanInput)
            
            // Backward pass through decoder
            let currentLR = await optimizer.getCurrentLearningRate()
            let decoderGrads = await decoder.backward(outputGrad, learningRate: currentLR)
            
            // Get gradient for encoder from decoder's input gradient
            let encoderOutputGrad = await decoder.getInputGradient()
            
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
        
        // Apply gradient clipping if specified
        let clippedEncoderGrads = if let clipValue = configuration.regularization.gradientClipping {
            avgEncoderGrads.map { clipGradients($0, clipValue: clipValue) }
        } else {
            avgEncoderGrads
        }
        
        let clippedDecoderGrads = if let clipValue = configuration.regularization.gradientClipping {
            avgDecoderGrads.map { clipGradients($0, clipValue: clipValue) }
        } else {
            avgDecoderGrads
        }
        
        // Combine gradients
        let allGradients = clippedEncoderGrads + clippedDecoderGrads
        
        return (totalLoss / batchSize, allGradients)
    }
    
    private func clipGradients(_ gradients: LayerGradients, clipValue: Float) -> LayerGradients {
        let clippedWeights = gradients.weights?.map { grad in
            max(-clipValue, min(clipValue, grad))
        }
        
        let clippedBias = gradients.bias?.map { grad in
            max(-clipValue, min(clipValue, grad))
        }
        
        return LayerGradients(weights: clippedWeights, bias: clippedBias)
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
        schedule: AutoencoderTrainingConfiguration.LearningRateSchedule,
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
        
        for cleanInput in data {
            // Add noise for validation too
            let noisyInput = addNoise(to: cleanInput)
            let encoded = await encoder.forward(noisyInput)
            let reconstructed = await decoder.forward(encoded)
            
            // Compare with clean input
            let loss = lossFunction.compute(prediction: reconstructed, target: cleanInput)
            totalLoss += loss
        }
        
        // Set back to training mode
        await encoder.setTraining(true)
        await decoder.setTraining(true)
        
        return totalLoss / Float(data.count)
    }
}
// VectorStoreKit: Sparse Autoencoder Implementation
//
// Sparse autoencoder with sparsity constraints on hidden activations

import Foundation

/// Configuration for sparse autoencoder
public struct SparseAutoencoderConfiguration: AutoencoderConfiguration {
    public let inputDimensions: Int
    public let encodedDimensions: Int
    public let encoderLayers: [Int]
    public let decoderLayers: [Int]
    public let sparsityTarget: Float
    public let sparsityWeight: Float
    public let training: TrainingConfiguration
    public let regularization: RegularizationConfig
    
    public init(
        inputDimensions: Int,
        encodedDimensions: Int,
        encoderLayers: [Int] = [512, 256],
        decoderLayers: [Int] = [256, 512],
        sparsityTarget: Float = 0.05,
        sparsityWeight: Float = 1.0,
        training: TrainingConfiguration = TrainingConfiguration(),
        regularization: RegularizationConfig = RegularizationConfig()
    ) {
        self.inputDimensions = inputDimensions
        self.encodedDimensions = encodedDimensions
        self.encoderLayers = encoderLayers
        self.decoderLayers = decoderLayers
        self.sparsityTarget = sparsityTarget
        self.sparsityWeight = sparsityWeight
        self.training = training
        self.regularization = regularization
    }
}

/// Sparse autoencoder implementation
public actor SparseAutoencoder: Autoencoder {
    public typealias Config = SparseAutoencoderConfiguration
    
    // MARK: - Properties
    
    private let configuration: SparseAutoencoderConfiguration
    private let encoder: NeuralNetwork
    private let decoder: NeuralNetwork
    private var trained: Bool = false
    private var trainingHistory: TrainingHistory
    private let metalCompute: MetalCompute?
    
    // Track average activations for sparsity
    private var averageActivations: [Float] = []
    private let activationDecay: Float = 0.99
    
    public var isTrained: Bool {
        trained
    }
    
    // MARK: - Initialization
    
    public init(
        configuration: SparseAutoencoderConfiguration,
        metalCompute: MetalCompute? = nil
    ) async {
        self.configuration = configuration
        self.metalCompute = metalCompute
        self.trainingHistory = TrainingHistory()
        
        // Initialize average activations
        self.averageActivations = Array(
            repeating: configuration.sparsityTarget,
            count: configuration.encodedDimensions
        )
        
        // Build encoder network
        var encoderLayers: [any NeuralLayer] = []
        var currentInput = configuration.inputDimensions
        
        // Hidden layers
        for hiddenSize in configuration.encoderLayers {
            encoderLayers.append(DenseLayer(
                inputSize: currentInput,
                outputSize: hiddenSize,
                activation: .relu,
                metalCompute: metalCompute
            ))
            currentInput = hiddenSize
        }
        
        // Output layer with sigmoid activation to encourage sparsity
        encoderLayers.append(DenseLayer(
            inputSize: currentInput,
            outputSize: configuration.encodedDimensions,
            activation: .sigmoid,
            metalCompute: metalCompute
        ))
        
        // Build decoder network
        var decoderLayers: [any NeuralLayer] = []
        currentInput = configuration.encodedDimensions
        
        // Hidden layers
        for hiddenSize in configuration.decoderLayers {
            decoderLayers.append(DenseLayer(
                inputSize: currentInput,
                outputSize: hiddenSize,
                activation: .relu,
                metalCompute: metalCompute
            ))
            currentInput = hiddenSize
        }
        
        // Output layer
        decoderLayers.append(DenseLayer(
            inputSize: currentInput,
            outputSize: configuration.inputDimensions,
            activation: .sigmoid,
            metalCompute: metalCompute
        ))
        
        // Initialize networks
        self.encoder = await NeuralNetwork(layers: encoderLayers)
        self.decoder = await NeuralNetwork(layers: decoderLayers)
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
            totalError += sqrt(error)
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
            var epochSparsityLoss: Float = 0
            var batchCount = 0
            
            // Shuffle data
            let shuffled = data.shuffled()
            
            // Process batches
            for batchStart in stride(from: 0, to: shuffled.count, by: configuration.training.batchSize) {
                let batchEnd = min(batchStart + configuration.training.batchSize, shuffled.count)
                let batch = Array(shuffled[batchStart..<batchEnd])
                
                // Compute batch loss and gradients
                let (loss, reconLoss, sparsityLoss, gradients) = await computeSparseBatchLossAndGradients(
                    batch: batch,
                    lossFunction: lossFunction,
                    optimizer: optimizer
                )
                
                epochLoss += loss
                epochReconLoss += reconLoss
                epochSparsityLoss += sparsityLoss
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
            epochSparsityLoss /= Float(batchCount)
            
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
                print("Epoch \(epoch): Loss = \(epochLoss) (Recon: \(epochReconLoss), Sparsity: \(epochSparsityLoss)), Val Loss = \(validationLoss)")
                print("Average activation: \(averageActivations.reduce(0, +) / Float(averageActivations.count))")
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
    
    private func computeSparseBatchLossAndGradients(
        batch: [[Float]],
        lossFunction: LossFunction,
        optimizer: any Optimizer
    ) async -> (loss: Float, reconLoss: Float, sparsityLoss: Float, gradients: [LayerGradients]) {
        var totalLoss: Float = 0
        var totalReconLoss: Float = 0
        var totalSparsityLoss: Float = 0
        var accumulatedEncoderGrads: [LayerGradients] = []
        var accumulatedDecoderGrads: [LayerGradients] = []
        var batchActivations = Array(repeating: Float(0), count: configuration.encodedDimensions)
        
        for input in batch {
            // Forward pass through encoder
            let encoded = await encoder.forward(input)
            
            // Update batch activations
            for (i, activation) in encoded.enumerated() {
                batchActivations[i] += activation
            }
            
            // Forward pass through decoder
            let reconstructed = await decoder.forward(encoded)
            
            // Compute reconstruction loss
            let reconLoss = lossFunction.compute(prediction: reconstructed, target: input)
            totalReconLoss += reconLoss
            
            // Compute sparsity loss (KL divergence between target and actual sparsity)
            let sparsityLoss = computeSparsityLoss(activations: encoded)
            totalSparsityLoss += sparsityLoss
            
            // Total loss
            let loss = reconLoss + configuration.sparsityWeight * sparsityLoss
            totalLoss += loss
            
            // Compute reconstruction gradient
            let reconGrad = lossFunction.gradient(prediction: reconstructed, target: input)
            
            // Backward pass through decoder
            let currentLR = await optimizer.getCurrentLearningRate()
            let decoderGrads = await decoder.backward(reconGrad, learningRate: currentLR)
            
            // Get gradient for encoder from decoder's input gradient
            var encoderOutputGrad = await decoder.getInputGradient()
            
            // Add sparsity gradient
            let sparsityGrad = computeSparsityGradient(
                activations: encoded,
                averageActivations: averageActivations
            )
            
            // Combine gradients
            for i in 0..<encoderOutputGrad.count {
                encoderOutputGrad[i] += configuration.sparsityWeight * sparsityGrad[i]
            }
            
            // Add L1 regularization gradient if specified
            if configuration.regularization.l1Weight > 0 {
                for i in 0..<encoded.count {
                    encoderOutputGrad[i] += configuration.regularization.l1Weight * (encoded[i] > 0 ? 1 : -1)
                }
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
        
        // Update average activations
        let batchSize = Float(batch.count)
        for i in 0..<averageActivations.count {
            let batchAvg = batchActivations[i] / batchSize
            averageActivations[i] = activationDecay * averageActivations[i] + (1 - activationDecay) * batchAvg
        }
        
        // Average gradients
        let avgEncoderGrads = accumulatedEncoderGrads.map { scaleGradients($0, by: 1.0 / batchSize) }
        let avgDecoderGrads = accumulatedDecoderGrads.map { scaleGradients($0, by: 1.0 / batchSize) }
        
        // Combine gradients
        let allGradients = avgEncoderGrads + avgDecoderGrads
        
        return (
            totalLoss / batchSize,
            totalReconLoss / batchSize,
            totalSparsityLoss / batchSize,
            allGradients
        )
    }
    
    private func computeSparsityLoss(activations: [Float]) -> Float {
        // KL divergence between target sparsity and actual sparsity
        var klDivergence: Float = 0
        
        for i in activations.indices {
            let rho = averageActivations[i]
            let rhoHat = configuration.sparsityTarget
            
            klDivergence += rhoHat * log(max(rhoHat / max(rho, 1e-7), 1e-7)) +
                           (1 - rhoHat) * log(max((1 - rhoHat) / max(1 - rho, 1e-7), 1e-7))
        }
        
        return klDivergence
    }
    
    private func computeSparsityGradient(
        activations: [Float],
        averageActivations: [Float]
    ) -> [Float] {
        // Gradient of KL divergence with respect to activations
        var gradients: [Float] = []
        
        for (i, activation) in activations.enumerated() {
            let rho = averageActivations[i]
            let rhoHat = configuration.sparsityTarget
            
            let grad = -rhoHat / max(rho, 1e-7) + (1 - rhoHat) / max(1 - rho, 1e-7)
            
            // Apply activation derivative (sigmoid in this case)
            let sigmoidDeriv = activation * (1 - activation)
            gradients.append(grad * sigmoidDeriv * (1 - activationDecay))
        }
        
        return gradients
    }
    
    private func addGradients(_ g1: LayerGradients, _ g2: LayerGradients) -> LayerGradients {
        var result = LayerGradients()
        
        if let w1 = g1.weights, let w2 = g2.weights {
            result.weights = zip(w1, w2).map { $0 + $1 }
        }
        
        if let b1 = g1.bias, let b2 = g2.bias {
            result.bias = zip(b1, b2).map { $0 + $1 }
        }
        
        return result
    }
    
    private func scaleGradients(_ gradients: LayerGradients, by scale: Float) -> LayerGradients {
        var result = LayerGradients()
        
        if let weights = gradients.weights {
            result.weights = weights.map { $0 * scale }
        }
        
        if let bias = gradients.bias {
            result.bias = bias.map { $0 * scale }
        }
        
        return result
    }
    
    private func applyGradients(_ gradients: [LayerGradients], optimizer: any Optimizer) async {
        // Apply gradients to encoder and decoder
        let encoderGradCount = await encoder.layers.count
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
            
            // Only reconstruction loss for validation
            let loss = lossFunction.compute(prediction: reconstructed, target: input)
            totalLoss += loss
        }
        
        // Set back to training mode
        await encoder.setTraining(true)
        await decoder.setTraining(true)
        
        return totalLoss / Float(data.count)
    }
}
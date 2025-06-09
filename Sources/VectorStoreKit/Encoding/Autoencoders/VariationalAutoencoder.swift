// VectorStoreKit: Variational Autoencoder Implementation
//
// VAE implementation using Core/ML components and VAE-specific layers

import Foundation

/// Configuration for variational autoencoder
public struct VAEConfiguration: AutoencoderConfiguration {
    public let inputDimensions: Int
    public let encodedDimensions: Int  // This is the latent dimension
    public let encoderLayers: [Int]
    public let decoderLayers: [Int]
    public let training: TrainingConfiguration
    public let regularization: RegularizationConfig
    public let klWeight: Float
    public let reconstructionWeight: Float
    
    public init(
        inputDimensions: Int,
        encodedDimensions: Int,
        encoderLayers: [Int] = [512, 256],
        decoderLayers: [Int] = [256, 512],
        training: TrainingConfiguration = TrainingConfiguration(),
        regularization: RegularizationConfig = RegularizationConfig(),
        klWeight: Float = 1.0,
        reconstructionWeight: Float = 1.0
    ) {
        self.inputDimensions = inputDimensions
        self.encodedDimensions = encodedDimensions
        self.encoderLayers = encoderLayers
        self.decoderLayers = decoderLayers
        self.training = training
        self.regularization = regularization
        self.klWeight = klWeight
        self.reconstructionWeight = reconstructionWeight
    }
}

/// Variational autoencoder implementation
public actor VariationalAutoencoder: Autoencoder {
    public typealias Config = VAEConfiguration
    
    // MARK: - Properties
    
    private let configuration: VAEConfiguration
    private let encoder: NeuralNetwork
    private let samplingLayer: VAESamplingLayer
    private let decoder: NeuralNetwork
    private var trained: Bool = false
    private var trainingHistory: TrainingHistory
    private let metalCompute: MetalCompute?
    
    public var isTrained: Bool {
        trained
    }
    
    // MARK: - Initialization
    
    public init(
        configuration: VAEConfiguration,
        metalCompute: MetalCompute? = nil
    ) async {
        self.configuration = configuration
        self.metalCompute = metalCompute
        self.trainingHistory = TrainingHistory()
        
        // Build encoder network (outputs to hidden representation before sampling)
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
            
            // Add dropout if specified
            if configuration.regularization.dropout > 0 {
                encoderLayers.append(DropoutLayer(
                    rate: configuration.regularization.dropout
                ))
            }
            
            currentInput = hiddenSize
        }
        
        // Initialize encoder network
        self.encoder = await NeuralNetwork(layers: encoderLayers)
        
        // Initialize VAE sampling layer (handles mean, log_var, and reparameterization)
        self.samplingLayer = VAESamplingLayer(
            inputSize: currentInput,
            latentDim: configuration.encodedDimensions,
            metalCompute: metalCompute
        )
        
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
            
            // Add dropout if specified
            if configuration.regularization.dropout > 0 {
                decoderLayers.append(DropoutLayer(
                    rate: configuration.regularization.dropout
                ))
            }
            
            currentInput = hiddenSize
        }
        
        // Output layer with sigmoid activation for VAE
        decoderLayers.append(DenseLayer(
            inputSize: currentInput,
            outputSize: configuration.inputDimensions,
            activation: .sigmoid,
            metalCompute: metalCompute
        ))
        
        // Initialize decoder network
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
            // Pass through encoder network
            let hidden = await encoder.forward(vector)
            
            // Get latent representation (only z, not mean and log_var)
            let (z, _, _) = await samplingLayer.forward(hidden)
            results.append(z)
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
        
        // Use binary cross entropy for VAE
        var totalError: Float = 0
        for (original, recon) in zip(vectors, reconstructed) {
            var error: Float = 0
            for (t, p) in zip(original, recon) {
                let clampedP = max(p, 1e-7)
                let clampedOneMinusP = max(1 - p, 1e-7)
                error += -(t * log(clampedP) + (1 - t) * log(clampedOneMinusP))
            }
            totalError += error / Float(original.count)
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
        
        // Create VAE loss
        let vaeLoss = VAELoss(
            reconstructionWeight: configuration.reconstructionWeight,
            klWeight: configuration.klWeight
        )
        
        // Training loop
        for epoch in 0..<configuration.training.epochs {
            var epochLoss: Float = 0
            var epochReconLoss: Float = 0
            var epochKLLoss: Float = 0
            var batchCount = 0
            
            // Shuffle data
            let shuffled = data.shuffled()
            
            // Process batches
            for batchStart in stride(from: 0, to: shuffled.count, by: configuration.training.batchSize) {
                let batchEnd = min(batchStart + configuration.training.batchSize, shuffled.count)
                let batch = Array(shuffled[batchStart..<batchEnd])
                
                // Compute batch loss and gradients
                let (loss, reconLoss, klLoss, gradients) = await computeVAEBatchLossAndGradients(
                    batch: batch,
                    vaeLoss: vaeLoss,
                    optimizer: optimizer
                )
                
                epochLoss += loss
                epochReconLoss += reconLoss
                epochKLLoss += klLoss
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
            epochKLLoss /= Float(batchCount)
            
            // Validation
            let validationLoss = if let validationData = validationData {
                try await computeValidationLoss(validationData, vaeLoss: vaeLoss)
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
                print("Epoch \(epoch): Loss = \(epochLoss) (Recon: \(epochReconLoss), KL: \(epochKLLoss)), Val Loss = \(validationLoss)")
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
    
    private func computeVAEBatchLossAndGradients(
        batch: [[Float]],
        vaeLoss: VAELoss,
        optimizer: any Optimizer
    ) async -> (loss: Float, reconLoss: Float, klLoss: Float, gradients: [LayerGradients]) {
        var totalLoss: Float = 0
        var totalReconLoss: Float = 0
        var totalKLLoss: Float = 0
        var accumulatedGradients: [LayerGradients] = []
        
        for input in batch {
            // Forward pass through encoder
            let hidden = await encoder.forward(input)
            
            // Get mean, log_var, and sampled z
            let (z, mean, logVar) = await samplingLayer.forward(hidden)
            
            // Forward pass through decoder
            let reconstructed = await decoder.forward(z)
            
            // Compute VAE loss
            let loss = vaeLoss.compute(
                original: input,
                reconstructed: reconstructed,
                mean: mean,
                logVar: logVar
            )
            
            // Compute individual losses for monitoring
            let reconLoss = LossFunction.binaryCrossEntropy.compute(
                prediction: reconstructed,
                target: input
            )
            let klLoss = LossFunction.klDivergence(mean: mean, logVar: logVar)
            
            totalLoss += loss
            totalReconLoss += reconLoss
            totalKLLoss += klLoss
            
            // Compute gradients
            let (reconGrad, meanGrad, logVarGrad) = vaeLoss.gradient(
                original: input,
                reconstructed: reconstructed,
                mean: mean,
                logVar: logVar
            )
            
            // Backward pass through decoder
            let currentLR = await optimizer.getCurrentLearningRate()
            let decoderGrads = await decoder.backward(reconGrad, learningRate: currentLR)
            
            // Get gradient for sampling layer from decoder
            let zGrad = await decoder.getInputGradient()
            
            // Backward pass through sampling layer
            // Note: We need to handle gradients for both mean and log_var
            let samplingGrads = await computeSamplingLayerGradients(
                zGrad: zGrad,
                meanGrad: meanGrad,
                logVarGrad: logVarGrad,
                mean: mean,
                logVar: logVar,
                z: z
            )
            
            // Backward pass through encoder
            let encoderGrads = await encoder.backward(samplingGrads, learningRate: currentLR)
            
            // Accumulate all gradients
            let allGrads = encoderGrads + [await samplingLayer.getMeanLayer().getGradients()] + 
                          [await samplingLayer.getLogVarLayer().getGradients()] + decoderGrads
            
            if accumulatedGradients.isEmpty {
                accumulatedGradients = allGrads
            } else {
                for i in 0..<allGrads.count {
                    accumulatedGradients[i] = addGradients(accumulatedGradients[i], allGrads[i])
                }
            }
        }
        
        // Average gradients
        let batchSize = Float(batch.count)
        let avgGradients = accumulatedGradients.map { scaleGradients($0, by: 1.0 / batchSize) }
        
        return (
            totalLoss / batchSize,
            totalReconLoss / batchSize,
            totalKLLoss / batchSize,
            avgGradients
        )
    }
    
    private func computeSamplingLayerGradients(
        zGrad: [Float],
        meanGrad: [Float],
        logVarGrad: [Float],
        mean: [Float],
        logVar: [Float],
        z: [Float]
    ) async -> [Float] {
        // Gradient through reparameterization trick
        // z = mean + epsilon * exp(0.5 * log_var)
        // dL/d_hidden = dL/d_mean * d_mean/d_hidden + dL/d_logVar * d_logVar/d_hidden
        
        // Compute gradient with respect to z
        // dL/dz propagates to both mean and log_var through the reparameterization
        var combinedGrad = Array(repeating: Float(0), count: meanGrad.count)
        
        // Gradient from reconstruction loss through z
        for i in 0..<zGrad.count {
            combinedGrad[i] += zGrad[i]
        }
        
        // Add direct gradients from KL divergence
        for i in 0..<combinedGrad.count {
            combinedGrad[i] += meanGrad[i] + logVarGrad[i]
        }
        
        // Account for the reparameterization gradient
        // dz/d_logVar = 0.5 * epsilon * exp(0.5 * log_var)
        // Since epsilon = (z - mean) / exp(0.5 * log_var), we can compute:
        for i in 0..<z.count {
            let std = exp(0.5 * logVar[i])
            let epsilon = (z[i] - mean[i]) / std
            // Additional gradient through log_var from reparameterization
            combinedGrad[i] += zGrad[i] * 0.5 * epsilon * std
        }
        
        return combinedGrad
    }
    
    private func addGradients(_ g1: LayerGradients, _ g2: LayerGradients) -> LayerGradients {
        var result = LayerGradients()
        
        // Add weight gradients
        if let w1 = g1.weightsGrad, let w2 = g2.weightsGrad {
            result.weightsGrad = zip(w1, w2).map { row1, row2 in
                zip(row1, row2).map { $0 + $1 }
            }
        }
        
        // Add bias gradients
        if let b1 = g1.biasGrad, let b2 = g2.biasGrad {
            result.biasGrad = zip(b1, b2).map { $0 + $1 }
        }
        
        // Add batch norm gradients if present
        if let gamma1 = g1.batchNormGammaGrad, let gamma2 = g2.batchNormGammaGrad {
            result.batchNormGammaGrad = zip(gamma1, gamma2).map { $0 + $1 }
        }
        
        if let beta1 = g1.batchNormBetaGrad, let beta2 = g2.batchNormBetaGrad {
            result.batchNormBetaGrad = zip(beta1, beta2).map { $0 + $1 }
        }
        
        return result
    }
    
    private func scaleGradients(_ gradients: LayerGradients, by scale: Float) -> LayerGradients {
        var result = LayerGradients()
        
        // Scale weight gradients
        if let weights = gradients.weightsGrad {
            result.weightsGrad = weights.map { row in
                row.map { $0 * scale }
            }
        }
        
        // Scale bias gradients
        if let bias = gradients.biasGrad {
            result.biasGrad = bias.map { $0 * scale }
        }
        
        // Scale batch norm gradients if present
        if let gamma = gradients.batchNormGammaGrad {
            result.batchNormGammaGrad = gamma.map { $0 * scale }
        }
        
        if let beta = gradients.batchNormBetaGrad {
            result.batchNormBetaGrad = beta.map { $0 * scale }
        }
        
        return result
    }
    
    private func applyGradients(_ gradients: [LayerGradients], optimizer: any Optimizer) async {
        // Apply gradients to all networks
        let encoderLayerCount = await encoder.layers.count
        let encoderGrads = Array(gradients.prefix(encoderLayerCount))
        
        // Skip mean and logVar layer gradients (handled separately)
        let remainingGrads = Array(gradients.suffix(from: encoderLayerCount + 2))
        
        await encoder.updateWeights(gradients: encoderGrads, optimizer: optimizer)
        
        // Update sampling layer (mean and logVar layers)
        // Note: Sampling layer update would be handled internally by the VAESamplingLayer
        
        await decoder.updateWeights(gradients: remainingGrads, optimizer: optimizer)
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
        vaeLoss: VAELoss
    ) async throws -> Float {
        // Set to evaluation mode
        await encoder.setTraining(false)
        await decoder.setTraining(false)
        
        var totalLoss: Float = 0
        
        for input in data {
            let hidden = await encoder.forward(input)
            let (z, mean, logVar) = await samplingLayer.forward(hidden)
            let reconstructed = await decoder.forward(z)
            
            let loss = vaeLoss.compute(
                original: input,
                reconstructed: reconstructed,
                mean: mean,
                logVar: logVar
            )
            totalLoss += loss
        }
        
        // Set back to training mode
        await encoder.setTraining(true)
        await decoder.setTraining(true)
        
        return totalLoss / Float(data.count)
    }
}

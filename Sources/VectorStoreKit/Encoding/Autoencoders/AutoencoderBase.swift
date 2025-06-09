// VectorStoreKit: Base Autoencoder Implementation
//
// Standard autoencoder using Core/ML neural network components

import Foundation

/// Configuration for standard autoencoder
public struct StandardAutoencoderConfiguration: AutoencoderConfiguration {
    public let inputDimensions: Int
    public let encodedDimensions: Int
    public let encoderLayers: [Int]
    public let decoderLayers: [Int]
    public let encoderActivation: Activation
    public let decoderActivation: Activation
    public let finalActivation: Activation
    public let training: TrainingConfiguration
    public let regularization: RegularizationConfig
    
    public init(
        inputDimensions: Int,
        encodedDimensions: Int,
        encoderLayers: [Int] = [512, 256],
        decoderLayers: [Int] = [256, 512],
        encoderActivation: Activation = .relu,
        decoderActivation: Activation = .relu,
        finalActivation: Activation = .tanh,
        training: TrainingConfiguration = TrainingConfiguration(),
        regularization: RegularizationConfig = RegularizationConfig()
    ) {
        self.inputDimensions = inputDimensions
        self.encodedDimensions = encodedDimensions
        self.encoderLayers = encoderLayers
        self.decoderLayers = decoderLayers
        self.encoderActivation = encoderActivation
        self.decoderActivation = decoderActivation
        self.finalActivation = finalActivation
        self.training = training
        self.regularization = regularization
    }
}

/// Base autoencoder implementation
public actor AutoencoderBase: Autoencoder {
    public typealias Config = StandardAutoencoderConfiguration
    
    // MARK: - Properties
    
    private let configuration: StandardAutoencoderConfiguration
    private let encoder: NeuralNetwork
    private let decoder: NeuralNetwork
    private var trained: Bool = false
    private var trainingHistory: TrainingHistory
    private let metalCompute: MetalCompute?
    
    public var isTrained: Bool {
        trained
    }
    
    // MARK: - Initialization
    
    public init(
        configuration: StandardAutoencoderConfiguration,
        metalCompute: MetalCompute? = nil
    ) async {
        self.configuration = configuration
        self.metalCompute = metalCompute
        self.trainingHistory = TrainingHistory()
        
        // Build encoder network
        var encoderLayers: [any NeuralLayer] = []
        var currentInput = configuration.inputDimensions
        
        // Hidden layers
        for hiddenSize in configuration.encoderLayers {
            encoderLayers.append(DenseLayer(
                inputSize: currentInput,
                outputSize: hiddenSize,
                activation: configuration.encoderActivation,
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
        
        // Output layer
        encoderLayers.append(DenseLayer(
            inputSize: currentInput,
            outputSize: configuration.encodedDimensions,
            activation: .linear,
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
                activation: configuration.decoderActivation,
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
        
        // Output layer
        decoderLayers.append(DenseLayer(
            inputSize: currentInput,
            outputSize: configuration.inputDimensions,
            activation: configuration.finalActivation,
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
            totalError += sqrt(error)  // Euclidean distance
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
            var batchCount = 0
            
            // Shuffle data
            let shuffled = data.shuffled()
            
            // Process batches
            for batchStart in stride(from: 0, to: shuffled.count, by: configuration.training.batchSize) {
                let batchEnd = min(batchStart + configuration.training.batchSize, shuffled.count)
                let batch = Array(shuffled[batchStart..<batchEnd])
                
                // Compute batch loss and gradients
                let (loss, gradients) = await computeBatchLossAndGradients(
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
    
    private func computeBatchLossAndGradients(
        batch: [[Float]],
        lossFunction: LossFunction,
        optimizer: any Optimizer
    ) async -> (loss: Float, gradients: [LayerGradients]) {
        var totalLoss: Float = 0
        var accumulatedEncoderGrads: [LayerGradients] = []
        var accumulatedDecoderGrads: [LayerGradients] = []
        
        for input in batch {
            // Forward pass through encoder
            let encoded = await encoder.forward(input)
            
            // Forward pass through decoder
            let reconstructed = await decoder.forward(encoded)
            
            // Compute loss
            let loss = lossFunction.compute(prediction: reconstructed, target: input)
            totalLoss += loss
            
            // Compute output gradient
            let outputGrad = lossFunction.gradient(prediction: reconstructed, target: input)
            
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
                // Add to accumulated gradients
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
        
        return (totalLoss / batchSize, allGradients)
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
            if step == 1 {  // Update at the start of each epoch
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
                // Cosine annealing after warmup
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
            let loss = lossFunction.compute(prediction: reconstructed, target: input)
            totalLoss += loss
        }
        
        // Set back to training mode
        await encoder.setTraining(true)
        await decoder.setTraining(true)
        
        return totalLoss / Float(data.count)
    }
}
// VectorStoreKit: Variational Autoencoder Implementation
//
// VAE implementation using Core/ML components and VAE-specific layers

import Foundation
@preconcurrency import Metal

/// Configuration for variational autoencoder
public struct VAEConfiguration: AutoencoderConfiguration {
    public let inputDimensions: Int
    public let encodedDimensions: Int  // This is the latent dimension
    public let encoderLayers: [Int]
    public let decoderLayers: [Int]
    public let training: AutoencoderTrainingConfiguration
    public let regularization: RegularizationConfig
    public let klWeight: Float
    public let reconstructionWeight: Float
    
    public init(
        inputDimensions: Int,
        encodedDimensions: Int,
        encoderLayers: [Int] = [512, 256],
        decoderLayers: [Int] = [256, 512],
        training: AutoencoderTrainingConfiguration = AutoencoderTrainingConfiguration(),
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
    private let metalPipeline: MetalMLPipeline?
    
    public var isTrained: Bool {
        trained
    }
    
    // MARK: - Initialization
    
    public init(
        configuration: VAEConfiguration,
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
        
        // Build encoder network (outputs to hidden representation before sampling)
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
        
        // Initialize encoder network
        self.encoder = try await NeuralNetwork(metalPipeline: pipeline)
        await self.encoder.addLayers(encoderLayers)
        
        // TODO: Initialize VAE sampling layer (handles mean, log_var, and reparameterization)
        // For now, create a placeholder that will need to be implemented
        // self.samplingLayer = VAESamplingLayer(
        //     inputSize: currentInput,
        //     latentDim: configuration.encodedDimensions,
        //     metalPipeline: pipeline
        // )
        
        // Create VAE sampling layer
        self.samplingLayer = try await VAESamplingLayer(
            inputDimension: currentInput,
            latentDimension: configuration.encodedDimensions,
            metalPipeline: pipeline
        )
        
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
        
        // Output layer with sigmoid activation for VAE
        decoderLayers.append(try await DenseLayer(
            inputSize: currentInput,
            outputSize: configuration.inputDimensions,
            activation: .sigmoid,
            metalPipeline: pipeline
        ))
        
        // Initialize decoder network
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
            // Pass through encoder network
            let hidden = await encoder.forward(vector)
            
            // Get latent representation (only z, not mean and log_var)
            let (z, _, _) = try await samplingLayer.forward(hidden)
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
                let (loss, reconLoss, klLoss, gradients) = try await computeVAEBatchLossAndGradients(
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
    ) async throws -> (loss: Float, reconLoss: Float, klLoss: Float, gradients: [LayerGradients]) {
        var totalLoss: Float = 0
        var totalReconLoss: Float = 0
        var totalKLLoss: Float = 0
        var accumulatedGradients: [LayerGradients] = []
        
        for input in batch {
            // Forward pass through encoder
            let hidden = await encoder.forward(input)
            
            // Get mean, log_var, and sampled z
            let (z, mean, logVar) = try await samplingLayer.forward(hidden)
            
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
            let reconLoss = LossFunctions.compute(
                lossFunction: LossFunction.mse,
                prediction: reconstructed,
                target: input
            )
            // Compute KL divergence manually
            var klLoss: Float = 0
            for i in 0..<mean.count {
                klLoss += -0.5 * (1 + logVar[i] - mean[i] * mean[i] - exp(logVar[i]))
            }
            
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
            let allGrads = encoderGrads + decoderGrads
            
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
        // Add weight gradients
        let addedWeights: [Float]? = if let w1 = g1.weights, let w2 = g2.weights {
            zip(w1, w2).map { $0 + $1 }
        } else {
            nil
        }
        
        // Add bias gradients
        let addedBias: [Float]? = if let b1 = g1.bias, let b2 = g2.bias {
            zip(b1, b2).map { $0 + $1 }
        } else {
            nil
        }
        
        // Add batch norm gradients if present
        let addedBatchNormGamma: [Float]? = if let gamma1 = g1.batchNormGamma, let gamma2 = g2.batchNormGamma {
            zip(gamma1, gamma2).map { $0 + $1 }
        } else {
            nil
        }
        
        let addedBatchNormBeta: [Float]? = if let beta1 = g1.batchNormBeta, let beta2 = g2.batchNormBeta {
            zip(beta1, beta2).map { $0 + $1 }
        } else {
            nil
        }
        
        return LayerGradients(
            weights: addedWeights,
            bias: addedBias,
            batchNormGamma: addedBatchNormGamma,
            batchNormBeta: addedBatchNormBeta
        )
    }
    
    private func scaleGradients(_ gradients: LayerGradients, by scale: Float) -> LayerGradients {
        // Scale weight gradients
        let scaledWeights = gradients.weights?.map { $0 * scale }
        
        // Scale bias gradients
        let scaledBias = gradients.bias?.map { $0 * scale }
        
        // Scale batch norm gradients if present
        let scaledBatchNormGamma = gradients.batchNormGamma?.map { $0 * scale }
        let scaledBatchNormBeta = gradients.batchNormBeta?.map { $0 * scale }
        
        return LayerGradients(
            weights: scaledWeights,
            bias: scaledBias,
            batchNormGamma: scaledBatchNormGamma,
            batchNormBeta: scaledBatchNormBeta
        )
    }
    
    private func applyGradients(_ gradients: [LayerGradients], optimizer: any Optimizer) async {
        // Apply gradients to all networks
        let encoderLayerCount = await encoder.layerCount
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
        vaeLoss: VAELoss
    ) async throws -> Float {
        // Set to evaluation mode
        await encoder.setTraining(false)
        await decoder.setTraining(false)
        
        var totalLoss: Float = 0
        
        for input in data {
            let hidden = await encoder.forward(input)
            let (z, mean, logVar) = try await samplingLayer.forward(hidden)
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

// VAESamplingLayer with Metal acceleration
public actor VAESamplingLayer: NeuralLayer {
    let meanLayer: DenseLayer
    let logVarLayer: DenseLayer
    let metalPipeline: MetalMLPipeline
    private let reparameterizationShader: MTLComputePipelineState
    
    public var isTraining: Bool = true
    
    public init(
        inputDimension: Int,
        latentDimension: Int,
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.metalPipeline = metalPipeline
        
        // Load ML shaders if not already loaded
        // try await metalPipeline.pipelineManager.loadMLOptimizationShaders()
        
        // Get reparameterization shader
        self.reparameterizationShader = try await metalPipeline.pipelineManager.getMLPipeline(MetalPipelineManager.MLPipeline.vaeReparameterization)
        
        // Initialize projection layers
        self.meanLayer = try await DenseLayer(
            inputSize: inputDimension,
            outputSize: latentDimension,
            activation: .linear,
            metalPipeline: metalPipeline
        )
        
        self.logVarLayer = try await DenseLayer(
            inputSize: inputDimension,
            outputSize: latentDimension,
            activation: .linear,
            metalPipeline: metalPipeline
        )
    }
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Get mean and log variance
        let mean = try await meanLayer.forward(input)
        let logVar = try await logVarLayer.forward(input)
        
        if !isTraining {
            // During inference, just return the mean
            return mean
        }
        
        // Generate random epsilon
        let epsilon = try await metalPipeline.allocateBuffer(size: mean.count)
        generateRandomNormal(buffer: epsilon)
        
        // Allocate output buffer
        let output = try await metalPipeline.allocateBuffer(size: mean.count)
        
        // Apply reparameterization trick on GPU
        let commandQueue = await metalPipeline.getCommandQueue()
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        
        encoder.setComputePipelineState(reparameterizationShader)
        encoder.setBuffer(mean.buffer, offset: 0, index: 0)
        encoder.setBuffer(logVar.buffer, offset: 0, index: 1)
        encoder.setBuffer(epsilon.buffer, offset: 0, index: 2)
        encoder.setBuffer(output.buffer, offset: 0, index: 3)
        
        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (mean.count + 255) / 256,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Clean up epsilon buffer
        await metalPipeline.releaseBuffer(epsilon)
        
        return output
    }
    
    public func setTraining(_ training: Bool) {
        self.isTraining = training
        Task {
            await meanLayer.setTraining(training)
            await logVarLayer.setTraining(training)
        }
    }
    
    private func generateRandomNormal(buffer: MetalBuffer) {
        let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
        for i in 0..<buffer.count {
            // Box-Muller transform for normal distribution
            let u1 = Float.random(in: 0..<1)
            let u2 = Float.random(in: 0..<1)
            ptr[i] = sqrt(-2.0 * log(u1)) * cos(2.0 * .pi * u2)
        }
    }
    
    // Compatibility method for existing code
    func forward(_ input: [Float]) async throws -> (z: [Float], mean: [Float], logVar: [Float]) {
        // Convert input to MetalBuffer
        let inputBuffer = try await metalPipeline.allocateBuffer(size: input.count)
        let inputPtr = inputBuffer.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
        for i in 0..<input.count {
            inputPtr[i] = input[i]
        }
        
        // Forward through layers
        let meanBuffer = try await meanLayer.forward(inputBuffer)
        let logVarBuffer = try await logVarLayer.forward(inputBuffer)
        
        // Convert back to arrays
        let meanPtr = meanBuffer.buffer.contents().bindMemory(to: Float.self, capacity: meanBuffer.count)
        let logVarPtr = logVarBuffer.buffer.contents().bindMemory(to: Float.self, capacity: logVarBuffer.count)
        
        var mean: [Float] = []
        var logVar: [Float] = []
        
        for i in 0..<meanBuffer.count {
            mean.append(meanPtr[i])
        }
        for i in 0..<logVarBuffer.count {
            logVar.append(logVarPtr[i])
        }
        
        // Reparameterization trick: z = mean + epsilon * exp(0.5 * log_var)
        let epsilon = (0..<mean.count).map { _ in Float.random(in: -1...1) }
        let std = logVar.map { exp(0.5 * $0) }
        let z = zip(zip(mean, epsilon), std).map { (meanEps, std) in
            meanEps.0 + meanEps.1 * std
        }
        
        // Clean up buffers
        await metalPipeline.releaseBuffer(inputBuffer)
        await metalPipeline.releaseBuffer(meanBuffer)
        await metalPipeline.releaseBuffer(logVarBuffer)
        
        return (z, mean, logVar)
    }
    
    func getMeanLayer() -> DenseLayer {
        meanLayer
    }
    
    func getLogVarLayer() -> DenseLayer {
        logVarLayer
    }
    
    // MARK: - Required NeuralLayer Protocol Methods
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        // For VAE, the gradient flows through both mean and logVar layers
        // This is a simplified implementation - in practice you'd need to handle
        // the reparameterization trick gradient properly
        return try await meanLayer.backward(gradOutput)
    }
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        // Update both mean and logVar layer parameters
        try await meanLayer.updateParameters(gradients, learningRate: learningRate)
        try await logVarLayer.updateParameters(gradients, learningRate: learningRate)
    }
    
    public func getParameters() async -> MetalBuffer? {
        // Return concatenated parameters from both layers
        return await meanLayer.getParameters()
    }
    
    public func getParameterCount() async -> Int {
        let meanCount = await meanLayer.getParameterCount()
        let logVarCount = await logVarLayer.getParameterCount()
        return meanCount + logVarCount
    }
    
    public func zeroGradients() async {
        await meanLayer.zeroGradients()
        await logVarLayer.zeroGradients()
    }
    
    public func scaleGradients(_ scale: Float) async {
        await meanLayer.scaleGradients(scale)
        await logVarLayer.scaleGradients(scale)
    }
    
    public func updateParametersWithOptimizer(_ optimizer: any Optimizer) async throws {
        try await meanLayer.updateParametersWithOptimizer(optimizer)
        try await logVarLayer.updateParametersWithOptimizer(optimizer)
    }
}

// VectorStoreKit: Neural Encoder/Decoder
//
// Advanced neural encoding for learned embeddings with autoencoder architectures

import Foundation
import Accelerate
import simd

/// Configuration for neural encoder/decoder
public struct NeuralEncoderConfiguration: Sendable, Codable {
    public let inputDimensions: Int
    public let encodedDimensions: Int
    public let architecture: EncoderArchitecture
    public let training: TrainingConfiguration
    public let regularization: RegularizationConfig
    
    public enum EncoderArchitecture: Sendable, Codable {
        case autoencoder(encoder: [Int], decoder: [Int])
        case variational(encoder: [Int], latentDim: Int, decoder: [Int])
        case sparse(sparsityTarget: Float, hiddenLayers: [Int])
        case denoising(noiseLevel: Float, hiddenLayers: [Int])
        case contractive(contractiveWeight: Float, hiddenLayers: [Int])
    }
    
    public struct TrainingConfiguration: Sendable, Codable {
        public let batchSize: Int
        public let epochs: Int
        public let learningRate: Float
        public let optimizerType: OptimizerType
        public let lrSchedule: LearningRateSchedule?
        
        public enum OptimizerType: String, Sendable, Codable {
            case sgd = "sgd"
            case adam = "adam"
            case rmsprop = "rmsprop"
            case adagrad = "adagrad"
        }
        
        public enum LearningRateSchedule: Sendable, Codable {
            case exponentialDecay(rate: Float)
            case stepDecay(stepSize: Int, gamma: Float)
            case cosineAnnealing(tMax: Int)
            case warmupCosine(warmupSteps: Int)
        }
        
        public init(
            batchSize: Int,
            epochs: Int,
            learningRate: Float,
            optimizerType: OptimizerType,
            lrSchedule: LearningRateSchedule? = nil
        ) {
            self.batchSize = batchSize
            self.epochs = epochs
            self.learningRate = learningRate
            self.optimizerType = optimizerType
            self.lrSchedule = lrSchedule
        }
    }
    
    public struct RegularizationConfig: Sendable, Codable {
        public let l1Weight: Float
        public let l2Weight: Float
        public let dropout: Float
        public let gradientClipping: Float?
        
        public init(
            l1Weight: Float,
            l2Weight: Float,
            dropout: Float,
            gradientClipping: Float? = nil
        ) {
            self.l1Weight = l1Weight
            self.l2Weight = l2Weight
            self.dropout = dropout
            self.gradientClipping = gradientClipping
        }
    }
    
    public init(
        inputDimensions: Int,
        encodedDimensions: Int,
        architecture: EncoderArchitecture = .autoencoder(
            encoder: [512, 256],
            decoder: [256, 512]
        ),
        training: TrainingConfiguration = TrainingConfiguration(
            batchSize: 128,
            epochs: 100,
            learningRate: 0.001,
            optimizerType: .adam,
            lrSchedule: .exponentialDecay(rate: 0.95)
        ),
        regularization: RegularizationConfig = RegularizationConfig(
            l1Weight: 0,
            l2Weight: 0.0001,
            dropout: 0.1,
            gradientClipping: 5.0
        )
    ) {
        self.inputDimensions = inputDimensions
        self.encodedDimensions = encodedDimensions
        self.architecture = architecture
        self.training = training
        self.regularization = regularization
    }
}

/// Neural encoder/decoder for vector embeddings
public actor NeuralEncoder {
    
    // MARK: - Properties
    
    private let configuration: NeuralEncoderConfiguration
    private var encoder: NeuralNetwork
    private var decoder: NeuralNetwork
    private var trained: Bool = false
    private var trainingHistory: TrainingHistory
    private let metalCompute: MetalCompute?
    
    // MARK: - Initialization
    
    public init(
        configuration: NeuralEncoderConfiguration,
        metalCompute: MetalCompute? = nil
    ) {
        self.configuration = configuration
        self.metalCompute = metalCompute
        self.trainingHistory = TrainingHistory()
        
        // Initialize encoder and decoder networks
        switch configuration.architecture {
        case .autoencoder(let encoderLayers, let decoderLayers):
            self.encoder = NeuralNetwork(
                layers: Self.buildLayers(
                    input: configuration.inputDimensions,
                    hidden: encoderLayers,
                    output: configuration.encodedDimensions,
                    activation: .relu,
                    finalActivation: .linear
                )
            )
            self.decoder = NeuralNetwork(
                layers: Self.buildLayers(
                    input: configuration.encodedDimensions,
                    hidden: decoderLayers,
                    output: configuration.inputDimensions,
                    activation: .relu,
                    finalActivation: .tanh
                )
            )
            
        case .variational(let encoderLayers, let latentDim, let decoderLayers):
            // VAE has two outputs from encoder: mean and log_var
            self.encoder = NeuralNetwork(
                layers: Self.buildLayers(
                    input: configuration.inputDimensions,
                    hidden: encoderLayers,
                    output: latentDim * 2,  // mean + log_var
                    activation: .relu,
                    finalActivation: .linear
                )
            )
            self.decoder = NeuralNetwork(
                layers: Self.buildLayers(
                    input: latentDim,
                    hidden: decoderLayers,
                    output: configuration.inputDimensions,
                    activation: .relu,
                    finalActivation: .sigmoid
                )
            )
            
        case .sparse(_, let hiddenLayers),
             .denoising(_, let hiddenLayers),
             .contractive(_, let hiddenLayers):
            self.encoder = NeuralNetwork(
                layers: Self.buildLayers(
                    input: configuration.inputDimensions,
                    hidden: hiddenLayers,
                    output: configuration.encodedDimensions,
                    activation: .relu,
                    finalActivation: .linear
                )
            )
            self.decoder = NeuralNetwork(
                layers: Self.buildLayers(
                    input: configuration.encodedDimensions,
                    hidden: Array(hiddenLayers.reversed()),
                    output: configuration.inputDimensions,
                    activation: .relu,
                    finalActivation: .sigmoid
                )
            )
        }
    }
    
    // MARK: - Encoding/Decoding
    
    /// Encode vectors to lower-dimensional representations
    public func encode(_ vectors: [[Float]]) async throws -> [[Float]] {
        guard trained else {
            throw NeuralEncoderError.notTrained
        }
        
        if let metalCompute = metalCompute {
            return try await metalEncode(vectors)
        } else {
            return vectors.map { encoder.forward($0) }
        }
    }
    
    /// Decode embeddings back to original space
    public func decode(_ embeddings: [[Float]]) async throws -> [[Float]] {
        guard trained else {
            throw NeuralEncoderError.notTrained
        }
        
        if let metalCompute = metalCompute {
            return try await metalDecode(embeddings)
        } else {
            return embeddings.map { decoder.forward($0) }
        }
    }
    
    /// Encode and decode for reconstruction
    public func reconstruct(_ vectors: [[Float]]) async throws -> [[Float]] {
        let encoded = try await encode(vectors)
        return try await decode(encoded)
    }
    
    /// Get reconstruction error
    public func reconstructionError(_ vectors: [[Float]]) async throws -> Float {
        let reconstructed = try await reconstruct(vectors)
        
        var totalError: Float = 0
        for (original, recon) in zip(vectors, reconstructed) {
            totalError += euclideanDistance(original, recon)
        }
        
        return totalError / Float(vectors.count)
    }
    
    // MARK: - Training
    
    /// Train the encoder/decoder on data
    public func train(on data: [[Float]], validationData: [[Float]]? = nil) async throws {
        guard data.count > configuration.training.batchSize else {
            throw NeuralEncoderError.insufficientTrainingData
        }
        
        // Initialize optimizer
        var optimizer = createOptimizer()
        
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
                
                // Forward pass
                let (loss, encoderGrads, decoderGrads) = computeLossAndGradients(batch: batch)
                
                // Backward pass and update
                optimizer.update(
                    encoder: &encoder,
                    decoder: &decoder,
                    encoderGrads: encoderGrads,
                    decoderGrads: decoderGrads,
                    epoch: epoch
                )
                
                epochLoss += loss
                batchCount += 1
            }
            
            // Average epoch loss
            epochLoss /= Float(batchCount)
            
            // Validation
            let validationLoss = if let validationData = validationData {
                try await computeValidationLoss(validationData)
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
    }
    
    /// Fine-tune on new data
    public func finetune(on data: [[Float]], epochs: Int = 10) async throws {
        guard trained else {
            throw NeuralEncoderError.notTrained
        }
        
        // Use lower learning rate for fine-tuning
        var config = configuration
        config.training.learningRate *= 0.1
        
        // Train for fewer epochs
        let originalEpochs = configuration.training.epochs
        config.training.epochs = epochs
        
        try await train(on: data)
        
        config.training.epochs = originalEpochs
    }
    
    // MARK: - Loss Computation
    
    private func computeLossAndGradients(batch: [[Float]]) -> (loss: Float, encoderGrads: [LayerGradients], decoderGrads: [LayerGradients]) {
        var totalLoss: Float = 0
        var encoderGrads: [[LayerGradients]] = []
        var decoderGrads: [[LayerGradients]] = []
        
        for input in batch {
            // Forward pass
            let (encoded, encoderActivations) = encoder.forwardWithActivations(input)
            let (reconstructed, decoderActivations) = decoder.forwardWithActivations(encoded)
            
            // Compute loss based on architecture
            let loss: Float
            let encoderGradient: [Float]
            
            switch configuration.architecture {
            case .autoencoder:
                // Mean squared error
                loss = meanSquaredError(input, reconstructed)
                let outputGrad = reconstructed.enumerated().map { i, pred in
                    2.0 * (pred - input[i]) / Float(input.count)
                }
                
                // Backward through decoder
                let (decoderLayerGrads, decoderInputGrad) = decoder.backward(
                    outputGradient: outputGrad,
                    activations: decoderActivations
                )
                decoderGrads.append(decoderLayerGrads)
                encoderGradient = decoderInputGrad
                
            case .variational(_, let latentDim, _):
                // VAE loss: reconstruction + KL divergence
                let mean = Array(encoded[0..<latentDim])
                let logVar = Array(encoded[latentDim..<encoded.count])
                
                // Sample from latent distribution
                let sampled = sampleLatent(mean: mean, logVar: logVar)
                let (vaeReconstructed, vaeDecoderActivations) = decoder.forwardWithActivations(sampled)
                
                // Reconstruction loss
                let reconLoss = binaryCrossEntropy(input, vaeReconstructed)
                
                // KL divergence
                let klLoss = -0.5 * zip(mean, logVar).map { mu, lv in
                    1 + lv - mu * mu - exp(lv)
                }.reduce(0, +) / Float(latentDim)
                
                loss = reconLoss + klLoss
                
                // Gradients
                let outputGrad = vaeReconstructed.enumerated().map { i, pred in
                    (pred - input[i]) / Float(input.count)
                }
                
                let (vaeDecoderGrads, latentGrad) = decoder.backward(
                    outputGradient: outputGrad,
                    activations: vaeDecoderActivations
                )
                decoderGrads.append(vaeDecoderGrads)
                
                // Gradient through sampling
                var fullEncoderGrad = Array(repeating: Float(0), count: encoded.count)
                for i in 0..<latentDim {
                    fullEncoderGrad[i] = latentGrad[i] + mean[i] / Float(batch.count)
                    fullEncoderGrad[i + latentDim] = latentGrad[i] * 0.5 * exp(0.5 * logVar[i]) +
                        0.5 * (exp(logVar[i]) - 1) / Float(batch.count)
                }
                encoderGradient = fullEncoderGrad
                
            case .sparse(let sparsityTarget, _):
                // Sparse autoencoder with L1 penalty on activations
                loss = meanSquaredError(input, reconstructed)
                let sparsityLoss = configuration.regularization.l1Weight *
                    encoded.map { abs($0) }.reduce(0, +) / Float(encoded.count)
                
                let outputGrad = reconstructed.enumerated().map { i, pred in
                    2.0 * (pred - input[i]) / Float(input.count)
                }
                
                let (sparseDecoderGrads, decoderInputGrad) = decoder.backward(
                    outputGradient: outputGrad,
                    activations: decoderActivations
                )
                decoderGrads.append(sparseDecoderGrads)
                
                // Add sparsity gradient
                encoderGradient = decoderInputGrad.enumerated().map { i, grad in
                    grad + configuration.regularization.l1Weight * (encoded[i] > 0 ? 1 : -1) / Float(encoded.count)
                }
                
            case .denoising(let noiseLevel, _):
                // Denoising autoencoder
                let noisyInput = addNoise(to: input, level: noiseLevel)
                let (denoisedEncoded, _) = encoder.forwardWithActivations(noisyInput)
                let (denoisedReconstructed, denoisedActivations) = decoder.forwardWithActivations(denoisedEncoded)
                
                loss = meanSquaredError(input, denoisedReconstructed)  // Compare with clean input
                
                let outputGrad = denoisedReconstructed.enumerated().map { i, pred in
                    2.0 * (pred - input[i]) / Float(input.count)
                }
                
                let (denoisingDecoderGrads, denoisingEncoderGrad) = decoder.backward(
                    outputGradient: outputGrad,
                    activations: denoisedActivations
                )
                decoderGrads.append(denoisingDecoderGrads)
                encoderGradient = denoisingEncoderGrad
                
            case .contractive(let contractiveWeight, _):
                // Contractive autoencoder with Jacobian penalty
                loss = meanSquaredError(input, reconstructed)
                
                // Approximate Jacobian penalty
                let jacobianPenalty = computeJacobianPenalty(
                    input: input,
                    encoded: encoded,
                    weight: contractiveWeight
                )
                
                let outputGrad = reconstructed.enumerated().map { i, pred in
                    2.0 * (pred - input[i]) / Float(input.count)
                }
                
                let (contractiveDecoderGrads, contractiveEncoderGrad) = decoder.backward(
                    outputGradient: outputGrad,
                    activations: decoderActivations
                )
                decoderGrads.append(contractiveDecoderGrads)
                
                // Add Jacobian gradient (simplified)
                encoderGradient = contractiveEncoderGrad.map { $0 * (1 + contractiveWeight) }
            }
            
            totalLoss += loss
            
            // Backward through encoder
            let (encoderLayerGrads, _) = encoder.backward(
                outputGradient: encoderGradient,
                activations: encoderActivations
            )
            encoderGrads.append(encoderLayerGrads)
        }
        
        // Average gradients
        let avgEncoderGrads = averageGradients(encoderGrads)
        let avgDecoderGrads = averageGradients(decoderGrads)
        
        return (totalLoss / Float(batch.count), avgEncoderGrads, avgDecoderGrads)
    }
    
    // MARK: - Helper Methods
    
    private static func buildLayers(
        input: Int,
        hidden: [Int],
        output: Int,
        activation: ActivationType,
        finalActivation: ActivationType
    ) -> [NeuralLayer] {
        var layers: [NeuralLayer] = []
        var currentInput = input
        
        // Hidden layers
        for hiddenSize in hidden {
            layers.append(NeuralLayer(
                inputSize: currentInput,
                outputSize: hiddenSize,
                activation: activation
            ))
            currentInput = hiddenSize
        }
        
        // Output layer
        layers.append(NeuralLayer(
            inputSize: currentInput,
            outputSize: output,
            activation: finalActivation
        ))
        
        return layers
    }
    
    private func createOptimizer() -> Optimizer {
        switch configuration.training.optimizerType {
        case .sgd:
            return SGDOptimizer(
                learningRate: configuration.training.learningRate,
                momentum: 0.9
            )
        case .adam:
            return AdamOptimizer(
                learningRate: configuration.training.learningRate,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8
            )
        case .rmsprop:
            return RMSPropOptimizer(
                learningRate: configuration.training.learningRate,
                alpha: 0.99,
                epsilon: 1e-8
            )
        case .adagrad:
            return AdaGradOptimizer(
                learningRate: configuration.training.learningRate,
                epsilon: 1e-8
            )
        }
    }
    
    private func meanSquaredError(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { pow($0 - $1, 2) }.reduce(0, +) / Float(a.count)
    }
    
    private func binaryCrossEntropy(_ target: [Float], _ predicted: [Float]) -> Float {
        -zip(target, predicted).map { t, p in
            t * log(max(p, 1e-7)) + (1 - t) * log(max(1 - p, 1e-7))
        }.reduce(0, +) / Float(target.count)
    }
    
    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        sqrt(zip(a, b).map { pow($0 - $1, 2) }.reduce(0, +))
    }
    
    private func sampleLatent(mean: [Float], logVar: [Float]) -> [Float] {
        zip(mean, logVar).map { mu, lv in
            let std = exp(0.5 * lv)
            let epsilon = Float.random(in: -1...1)  // Should use normal distribution
            return mu + epsilon * std
        }
    }
    
    private func addNoise(to vector: [Float], level: Float) -> [Float] {
        vector.map { value in
            value + Float.random(in: -level...level)
        }
    }
    
    private func computeJacobianPenalty(input: [Float], encoded: [Float], weight: Float) -> Float {
        // Simplified Jacobian penalty approximation
        // In practice, would compute actual Jacobian matrix
        return weight * encoded.map { $0 * $0 }.reduce(0, +) / Float(encoded.count)
    }
    
    private func averageGradients(_ gradientsList: [[LayerGradients]]) -> [LayerGradients] {
        guard !gradientsList.isEmpty else { return [] }
        
        let numLayers = gradientsList[0].count
        var averaged: [LayerGradients] = []
        
        for layerIdx in 0..<numLayers {
            let layerGrads = gradientsList.map { $0[layerIdx] }
            
            // Average weight gradients
            let avgWeightGrad = averageMatrices(layerGrads.map { $0.weights })
            let avgBiasGrad = averageVectors(layerGrads.map { $0.bias })
            
            averaged.append(LayerGradients(weights: avgWeightGrad, bias: avgBiasGrad))
        }
        
        return averaged
    }
    
    private func averageMatrices(_ matrices: [[[Float]]]) -> [[Float]] {
        guard !matrices.isEmpty else { return [] }
        
        let rows = matrices[0].count
        let cols = matrices[0][0].count
        var result = Array(repeating: Array(repeating: Float(0), count: cols), count: rows)
        
        for matrix in matrices {
            for i in 0..<rows {
                for j in 0..<cols {
                    result[i][j] += matrix[i][j]
                }
            }
        }
        
        let count = Float(matrices.count)
        return result.map { row in row.map { $0 / count } }
    }
    
    private func averageVectors(_ vectors: [[Float]]) -> [Float] {
        guard !vectors.isEmpty else { return [] }
        
        let size = vectors[0].count
        var result = Array(repeating: Float(0), count: size)
        
        for vector in vectors {
            for i in 0..<size {
                result[i] += vector[i]
            }
        }
        
        return result.map { $0 / Float(vectors.count) }
    }
    
    private func computeValidationLoss(_ data: [[Float]]) async throws -> Float {
        let reconstructed = try await reconstruct(data)
        
        var totalLoss: Float = 0
        for (original, recon) in zip(data, reconstructed) {
            totalLoss += meanSquaredError(original, recon)
        }
        
        return totalLoss / Float(data.count)
    }
    
    // MARK: - Metal Acceleration
    
    private func metalEncode(_ vectors: [[Float]]) async throws -> [[Float]] {
        guard let metalCompute = metalCompute else {
            // Fallback to CPU if Metal is not available
            return vectors.map { encoder.forward($0) }
        }
        
        // Process in batches for better GPU utilization
        let batchSize = min(256, vectors.count)
        var results: [[Float]] = []
        
        for batchStart in stride(from: 0, to: vectors.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, vectors.count)
            let batch = Array(vectors[batchStart..<batchEnd])
            
            // Process batch through encoder layers
            let batchResults = try await processNeuralNetworkBatch(
                batch: batch,
                network: encoder,
                metalCompute: metalCompute
            )
            
            results.append(contentsOf: batchResults)
        }
        
        return results
    }
    
    private func metalDecode(_ embeddings: [[Float]]) async throws -> [[Float]] {
        guard let metalCompute = metalCompute else {
            // Fallback to CPU if Metal is not available
            return embeddings.map { decoder.forward($0) }
        }
        
        // Process in batches for better GPU utilization
        let batchSize = min(256, embeddings.count)
        var results: [[Float]] = []
        
        for batchStart in stride(from: 0, to: embeddings.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, embeddings.count)
            let batch = Array(embeddings[batchStart..<batchEnd])
            
            // Process batch through decoder layers
            let batchResults = try await processNeuralNetworkBatch(
                batch: batch,
                network: decoder,
                metalCompute: metalCompute
            )
            
            results.append(contentsOf: batchResults)
        }
        
        return results
    }
    
    /// Process a batch of inputs through a neural network using Metal acceleration
    private func processNeuralNetworkBatch(
        batch: [[Float]],
        network: NeuralNetwork,
        metalCompute: MetalCompute
    ) async throws -> [[Float]] {
        
        guard !batch.isEmpty else { return [] }
        
        let batchCount = batch.count
        let weights = network.getWeights()
        
        // Convert batch to matrix format (each row is an input vector)
        var currentMatrix = batch
        
        // Process through each layer
        for (layerIdx, layer) in network.layers.enumerated() {
            let layerWeights = weights[layerIdx]
            
            // Reshape weights to matrix format (transposed for multiplication)
            let weightsMatrix = layerWeights.weights
            
            // Perform matrix multiplication: output = input × weights^T
            var outputMatrix = try await metalCompute.matrixCompute.matrixMultiply(
                matrixA: currentMatrix,
                matrixB: transposeMatrix(weightsMatrix)
            )
            
            // Add bias to each row
            for i in 0..<outputMatrix.count {
                for j in 0..<outputMatrix[i].count {
                    outputMatrix[i][j] += layerWeights.bias[j]
                }
            }
            
            // Apply activation function
            outputMatrix = applyActivationBatch(
                outputMatrix,
                activation: layer.activation
            )
            
            currentMatrix = outputMatrix
        }
        
        return currentMatrix
    }
    
    /// Transpose a matrix
    private func transposeMatrix(_ matrix: [[Float]]) -> [[Float]] {
        guard !matrix.isEmpty else { return [] }
        
        let rows = matrix.count
        let cols = matrix[0].count
        var transposed = Array(repeating: Array(repeating: Float(0), count: rows), count: cols)
        
        for i in 0..<rows {
            for j in 0..<cols {
                transposed[j][i] = matrix[i][j]
            }
        }
        
        return transposed
    }
    
    /// Apply activation function to a batch of vectors
    private func applyActivationBatch(_ batch: [[Float]], activation: ActivationType) -> [[Float]] {
        batch.map { vector in
            switch activation {
            case .relu:
                return vector.map { max(0, $0) }
            case .sigmoid:
                return vector.map { 1 / (1 + exp(-$0)) }
            case .tanh:
                return vector.map { tanh($0) }
            case .linear:
                return vector
            }
        }
    }
    
    // MARK: - Model Persistence
    
    public func save(to url: URL) async throws {
        let modelData = NeuralEncoderModel(
            configuration: configuration,
            encoderWeights: encoder.getWeights(),
            decoderWeights: decoder.getWeights(),
            trained: trained,
            trainingHistory: trainingHistory
        )
        
        let data = try JSONEncoder().encode(modelData)
        try data.write(to: url)
    }
    
    public func load(from url: URL) async throws {
        let data = try Data(contentsOf: url)
        let modelData = try JSONDecoder().decode(NeuralEncoderModel.self, from: data)
        
        encoder.setWeights(modelData.encoderWeights)
        decoder.setWeights(modelData.decoderWeights)
        trained = modelData.trained
        trainingHistory = modelData.trainingHistory
    }
}

// MARK: - Supporting Types

/// Neural network implementation
private struct NeuralNetwork {
    var layers: [NeuralLayer]
    
    func forward(_ input: [Float]) -> [Float] {
        var current = input
        for layer in layers {
            current = layer.forward(current)
        }
        return current
    }
    
    func forwardWithActivations(_ input: [Float]) -> (output: [Float], activations: [[Float]]) {
        var activations: [[Float]] = [input]
        var current = input
        
        for layer in layers {
            current = layer.forward(current)
            activations.append(current)
        }
        
        return (current, activations)
    }
    
    func backward(
        outputGradient: [Float],
        activations: [[Float]]
    ) -> (layerGradients: [LayerGradients], inputGradient: [Float]) {
        var layerGradients: [LayerGradients] = []
        var currentGrad = outputGradient
        
        for (layerIdx, layer) in layers.enumerated().reversed() {
            let input = activations[layerIdx]
            let output = activations[layerIdx + 1]
            
            let (weightGrad, biasGrad, inputGrad) = layer.backward(
                outputGradient: currentGrad,
                input: input,
                output: output
            )
            
            layerGradients.insert(
                LayerGradients(weights: weightGrad, bias: biasGrad),
                at: 0
            )
            currentGrad = inputGrad
        }
        
        return (layerGradients, currentGrad)
    }
    
    func getWeights() -> [LayerWeights] {
        layers.map { LayerWeights(weights: $0.weights, bias: $0.bias) }
    }
    
    mutating func setWeights(_ weights: [LayerWeights]) {
        for (i, layerWeights) in weights.enumerated() {
            layers[i].weights = layerWeights.weights
            layers[i].bias = layerWeights.bias
        }
    }
}

/// Single neural network layer
private struct NeuralLayer {
    var weights: [[Float]]
    var bias: [Float]
    let activation: ActivationType
    
    init(inputSize: Int, outputSize: Int, activation: ActivationType) {
        // Xavier initialization
        let scale = sqrt(2.0 / Float(inputSize + outputSize))
        self.weights = (0..<outputSize).map { _ in
            (0..<inputSize).map { _ in Float.random(in: -scale...scale) }
        }
        self.bias = Array(repeating: 0, count: outputSize)
        self.activation = activation
    }
    
    func forward(_ input: [Float]) -> [Float] {
        // Compute weighted sum
        var output = Array(repeating: Float(0), count: bias.count)
        
        for (i, weightRow) in weights.enumerated() {
            output[i] = zip(input, weightRow).map { $0 * $1 }.reduce(0, +) + bias[i]
        }
        
        // Apply activation
        return applyActivation(output)
    }
    
    func backward(
        outputGradient: [Float],
        input: [Float],
        output: [Float]
    ) -> (weightGrad: [[Float]], biasGrad: [Float], inputGrad: [Float]) {
        // Gradient through activation
        let activationGrad = activationDerivative(output, gradient: outputGradient)
        
        // Weight gradients
        let weightGrad = activationGrad.map { grad in
            input.map { $0 * grad }
        }
        
        // Bias gradients
        let biasGrad = activationGrad
        
        // Input gradients
        var inputGrad = Array(repeating: Float(0), count: input.count)
        for (i, grad) in activationGrad.enumerated() {
            for (j, weight) in weights[i].enumerated() {
                inputGrad[j] += weight * grad
            }
        }
        
        return (weightGrad, biasGrad, inputGrad)
    }
    
    private func applyActivation(_ input: [Float]) -> [Float] {
        switch activation {
        case .relu:
            return input.map { max(0, $0) }
        case .sigmoid:
            return input.map { 1 / (1 + exp(-$0)) }
        case .tanh:
            return input.map { tanh($0) }
        case .linear:
            return input
        }
    }
    
    private func activationDerivative(_ output: [Float], gradient: [Float]) -> [Float] {
        switch activation {
        case .relu:
            return zip(output, gradient).map { $0 > 0 ? $1 : 0 }
        case .sigmoid:
            return zip(output, gradient).map { o, g in o * (1 - o) * g }
        case .tanh:
            return zip(output, gradient).map { o, g in (1 - o * o) * g }
        case .linear:
            return gradient
        }
    }
}

/// Activation function types
private enum ActivationType: String, Codable {
    case relu, sigmoid, tanh, linear
}

/// Layer gradients
private struct LayerGradients {
    let weights: [[Float]]
    let bias: [Float]
}

/// Layer weights for persistence
private struct LayerWeights: Codable {
    let weights: [[Float]]
    let bias: [Float]
}

/// Optimizer protocol
private protocol Optimizer {
    mutating func update(
        encoder: inout NeuralNetwork,
        decoder: inout NeuralNetwork,
        encoderGrads: [LayerGradients],
        decoderGrads: [LayerGradients],
        epoch: Int
    )
}

/// SGD optimizer
private struct SGDOptimizer: Optimizer {
    let learningRate: Float
    let momentum: Float
    var velocityEncoder: [[LayerVelocity]] = []
    var velocityDecoder: [[LayerVelocity]] = []
    
    struct LayerVelocity {
        var weights: [[Float]]
        var bias: [Float]
    }
    
    mutating func update(
        encoder: inout NeuralNetwork,
        decoder: inout NeuralNetwork,
        encoderGrads: [LayerGradients],
        decoderGrads: [LayerGradients],
        epoch: Int
    ) {
        // Initialize velocities if needed
        if velocityEncoder.isEmpty {
            velocityEncoder = encoderGrads.map { grad in
                LayerVelocity(
                    weights: grad.weights.map { $0.map { _ in Float(0) } },
                    bias: grad.bias.map { _ in Float(0) }
                )
            }
        }
        if velocityDecoder.isEmpty {
            velocityDecoder = decoderGrads.map { grad in
                LayerVelocity(
                    weights: grad.weights.map { $0.map { _ in Float(0) } },
                    bias: grad.bias.map { _ in Float(0) }
                )
            }
        }
        
        // Update encoder
        for (i, grad) in encoderGrads.enumerated() {
            // Update velocities
            for j in 0..<grad.weights.count {
                for k in 0..<grad.weights[j].count {
                    velocityEncoder[i].weights[j][k] = momentum * velocityEncoder[i].weights[j][k] -
                        learningRate * grad.weights[j][k]
                    encoder.layers[i].weights[j][k] += velocityEncoder[i].weights[j][k]
                }
            }
            
            for j in 0..<grad.bias.count {
                velocityEncoder[i].bias[j] = momentum * velocityEncoder[i].bias[j] -
                    learningRate * grad.bias[j]
                encoder.layers[i].bias[j] += velocityEncoder[i].bias[j]
            }
        }
        
        // Update decoder
        for (i, grad) in decoderGrads.enumerated() {
            for j in 0..<grad.weights.count {
                for k in 0..<grad.weights[j].count {
                    velocityDecoder[i].weights[j][k] = momentum * velocityDecoder[i].weights[j][k] -
                        learningRate * grad.weights[j][k]
                    decoder.layers[i].weights[j][k] += velocityDecoder[i].weights[j][k]
                }
            }
            
            for j in 0..<grad.bias.count {
                velocityDecoder[i].bias[j] = momentum * velocityDecoder[i].bias[j] -
                    learningRate * grad.bias[j]
                decoder.layers[i].bias[j] += velocityDecoder[i].bias[j]
            }
        }
    }
}

/// Adam optimizer
private struct AdamOptimizer: Optimizer {
    let learningRate: Float
    let beta1: Float
    let beta2: Float
    let epsilon: Float
    var t: Int = 0
    var mEncoder: [[LayerMomentum]] = []
    var vEncoder: [[LayerMomentum]] = []
    var mDecoder: [[LayerMomentum]] = []
    var vDecoder: [[LayerMomentum]] = []
    
    struct LayerMomentum {
        var weights: [[Float]]
        var bias: [Float]
    }
    
    mutating func update(
        encoder: inout NeuralNetwork,
        decoder: inout NeuralNetwork,
        encoderGrads: [LayerGradients],
        decoderGrads: [LayerGradients],
        epoch: Int
    ) {
        t += 1
        
        // Initialize moments if needed
        if mEncoder.isEmpty {
            mEncoder = initializeMoments(from: encoderGrads)
            vEncoder = initializeMoments(from: encoderGrads)
        }
        if mDecoder.isEmpty {
            mDecoder = initializeMoments(from: decoderGrads)
            vDecoder = initializeMoments(from: decoderGrads)
        }
        
        // Bias correction
        let beta1Power = pow(beta1, Float(t))
        let beta2Power = pow(beta2, Float(t))
        let correctedLR = learningRate * sqrt(1 - beta2Power) / (1 - beta1Power)
        
        // Update encoder
        updateNetwork(&encoder, gradients: encoderGrads, m: &mEncoder, v: &vEncoder, lr: correctedLR)
        
        // Update decoder
        updateNetwork(&decoder, gradients: decoderGrads, m: &mDecoder, v: &vDecoder, lr: correctedLR)
    }
    
    private func initializeMoments(from gradients: [LayerGradients]) -> [LayerMomentum] {
        gradients.map { grad in
            LayerMomentum(
                weights: grad.weights.map { $0.map { _ in Float(0) } },
                bias: grad.bias.map { _ in Float(0) }
            )
        }
    }
    
    private mutating func updateNetwork(
        _ network: inout NeuralNetwork,
        gradients: [LayerGradients],
        m: inout [LayerMomentum],
        v: inout [LayerMomentum],
        lr: Float
    ) {
        for (i, grad) in gradients.enumerated() {
            // Update weights
            for j in 0..<grad.weights.count {
                for k in 0..<grad.weights[j].count {
                    m[i].weights[j][k] = beta1 * m[i].weights[j][k] + (1 - beta1) * grad.weights[j][k]
                    v[i].weights[j][k] = beta2 * v[i].weights[j][k] + (1 - beta2) * grad.weights[j][k] * grad.weights[j][k]
                    
                    network.layers[i].weights[j][k] -= lr * m[i].weights[j][k] / (sqrt(v[i].weights[j][k]) + epsilon)
                }
            }
            
            // Update bias
            for j in 0..<grad.bias.count {
                m[i].bias[j] = beta1 * m[i].bias[j] + (1 - beta1) * grad.bias[j]
                v[i].bias[j] = beta2 * v[i].bias[j] + (1 - beta2) * grad.bias[j] * grad.bias[j]
                
                network.layers[i].bias[j] -= lr * m[i].bias[j] / (sqrt(v[i].bias[j]) + epsilon)
            }
        }
    }
}

/// RMSProp optimizer
private struct RMSPropOptimizer: Optimizer {
    let learningRate: Float
    let alpha: Float
    let epsilon: Float
    
    func update(
        encoder: inout NeuralNetwork,
        decoder: inout NeuralNetwork,
        encoderGrads: [LayerGradients],
        decoderGrads: [LayerGradients],
        epoch: Int
    ) {
        // Simplified implementation
        let sgd = SGDOptimizer(learningRate: learningRate, momentum: 0)
        var mutableSelf = sgd
        mutableSelf.update(
            encoder: &encoder,
            decoder: &decoder,
            encoderGrads: encoderGrads,
            decoderGrads: decoderGrads,
            epoch: epoch
        )
    }
}

/// AdaGrad optimizer
private struct AdaGradOptimizer: Optimizer {
    let learningRate: Float
    let epsilon: Float
    
    func update(
        encoder: inout NeuralNetwork,
        decoder: inout NeuralNetwork,
        encoderGrads: [LayerGradients],
        decoderGrads: [LayerGradients],
        epoch: Int
    ) {
        // Simplified implementation
        let sgd = SGDOptimizer(learningRate: learningRate, momentum: 0)
        var mutableSelf = sgd
        mutableSelf.update(
            encoder: &encoder,
            decoder: &decoder,
            encoderGrads: encoderGrads,
            decoderGrads: decoderGrads,
            epoch: epoch
        )
    }
}

/// Training history
private struct TrainingHistory: Codable {
    var epochs: [EpochHistory] = []
    let patience: Int = 10
    let minDelta: Float = 0.0001
    
    struct EpochHistory: Codable {
        let epoch: Int
        let trainLoss: Float
        let validationLoss: Float
        let timestamp: Date
    }
    
    mutating func addEpoch(epoch: Int, trainLoss: Float, validationLoss: Float) {
        epochs.append(EpochHistory(
            epoch: epoch,
            trainLoss: trainLoss,
            validationLoss: validationLoss,
            timestamp: Date()
        ))
    }
    
    func shouldStop() -> Bool {
        guard epochs.count > patience else { return false }
        
        let recentEpochs = Array(epochs.suffix(patience))
        let bestLoss = recentEpochs.map { $0.validationLoss }.min() ?? Float.infinity
        let currentLoss = epochs.last?.validationLoss ?? Float.infinity
        
        return currentLoss > bestLoss - minDelta
    }
}

/// Model data for persistence
private struct NeuralEncoderModel: Codable {
    let configuration: NeuralEncoderConfiguration
    let encoderWeights: [LayerWeights]
    let decoderWeights: [LayerWeights]
    let trained: Bool
    let trainingHistory: TrainingHistory
}

/// Neural encoder errors
public enum NeuralEncoderError: LocalizedError {
    case notTrained
    case insufficientTrainingData
    case dimensionMismatch
    case invalidConfiguration
    
    public var errorDescription: String? {
        switch self {
        case .notTrained:
            return "Neural encoder must be trained before use"
        case .insufficientTrainingData:
            return "Insufficient training data provided"
        case .dimensionMismatch:
            return "Input dimensions do not match configuration"
        case .invalidConfiguration:
            return "Invalid encoder configuration"
        }
    }
}
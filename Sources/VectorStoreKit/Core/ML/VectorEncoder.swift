// VectorStoreKit: Vector Encoder
//
// Neural encoding for vector compression and representation learning
//

import Foundation
@preconcurrency import Metal

/// Neural vector encoder for dimensionality reduction
public actor VectorEncoder {
    private let metalPipeline: MetalMLPipeline
    private let device: MTLDevice
    private var encoderNetwork: NeuralNetwork?
    private var decoderNetwork: NeuralNetwork?
    
    // Training state
    private var isTraining = false
    private var trainingMetrics = VectorEncoderMetrics()
    
    public init(metalPipeline: MetalMLPipeline) async throws {
        self.metalPipeline = metalPipeline
        self.device = await metalPipeline.device
    }
    
    // MARK: - Encoding
    
    /// Encode vectors to lower dimensional representation
    public func encode(_ vectors: [Vector]) async throws -> [Vector] {
        guard let encoder = encoderNetwork else {
            throw VectorEncoderError.encoderNotInitialized
        }
        
        var encodedVectors: [Vector] = []
        let batchSize = 256
        
        for batchStart in stride(from: 0, to: vectors.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, vectors.count)
            let batch = Array(vectors[batchStart..<batchEnd])
            
            let encodedBatch = try await encodeBatch(batch, encoder: encoder)
            encodedVectors.append(contentsOf: encodedBatch)
        }
        
        trainingMetrics.vectorsEncoded += vectors.count
        return encodedVectors
    }
    
    /// Decode vectors back to original dimension
    public func decode(_ encoded: [Vector]) async throws -> [Vector] {
        guard let decoder = decoderNetwork else {
            throw VectorEncoderError.decoderNotInitialized
        }
        
        var decodedVectors: [Vector] = []
        let batchSize = 256
        
        for batchStart in stride(from: 0, to: encoded.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, encoded.count)
            let batch = Array(encoded[batchStart..<batchEnd])
            
            let decodedBatch = try await decodeBatch(batch, decoder: decoder)
            decodedVectors.append(contentsOf: decodedBatch)
        }
        
        return decodedVectors
    }
    
    // MARK: - Training
    
    /// Train autoencoder for vector compression
    public func train(
        on vectors: [Vector],
        config: VectorEncoderConfig
    ) async throws {
        guard !vectors.isEmpty else {
            throw VectorEncoderError.emptyTrainingData
        }
        
        let inputDim = vectors.first!.count
        let encodedDim = config.encodedDimension ?? (inputDim / 4)
        
        // Initialize networks
        try await initializeNetworks(
            inputDim: inputDim,
            encodedDim: encodedDim,
            config: config
        )
        
        guard let encoder = encoderNetwork,
              let decoder = decoderNetwork else {
            throw VectorEncoderError.networkInitializationFailed
        }
        
        isTraining = true
        defer { isTraining = false }
        
        // Prepare training data
        let trainingData = try await prepareTrainingData(vectors)
        
        // Train autoencoder
        let startTime = Date()
        
        for epoch in 0..<config.epochs {
            var epochLoss: Float = 0
            
            // Shuffle data
            let shuffled = trainingData.shuffled()
            
            // Train in batches
            for batchStart in stride(from: 0, to: shuffled.count, by: config.batchSize) {
                let batchEnd = min(batchStart + config.batchSize, shuffled.count)
                let batch = Array(shuffled[batchStart..<batchEnd])
                
                let loss = try await trainBatch(
                    batch,
                    encoder: encoder,
                    decoder: decoder,
                    learningRate: config.learningRate
                )
                epochLoss += loss
            }
            
            epochLoss /= Float(shuffled.count / config.batchSize)
            
            // Log progress
            if epoch % 10 == 0 {
                print("Epoch \(epoch): Loss = \(epochLoss)")
            }
            
            // Early stopping
            if epochLoss < config.convergenceThreshold {
                print("Converged at epoch \(epoch)")
                break
            }
        }
        
        trainingMetrics.trainingTime = Date().timeIntervalSince(startTime)
        trainingMetrics.finalLoss = 0 // Set from last epoch
    }
    
    // MARK: - Architecture Variants
    
    /// Create a sparse autoencoder with L1 regularization
    public func createSparseEncoder(
        inputDim: Int,
        encodedDim: Int,
        sparsityWeight: Float = 0.01
    ) async throws {
        let encoder = try await NeuralNetwork(metalPipeline: metalPipeline)
        let decoder = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Encoder with sparsity constraint
        await encoder.addLayer(try await DenseLayer(
            inputSize: inputDim,
            outputSize: encodedDim * 2,
            activation: .relu,
            metalPipeline: metalPipeline
        ))
        
        await encoder.addLayer(try await SparseActivationLayer(
            sparsityWeight: sparsityWeight,
            metalPipeline: metalPipeline
        ))
        
        await encoder.addLayer(try await DenseLayer(
            inputSize: encodedDim * 2,
            outputSize: encodedDim,
            activation: .sigmoid,
            metalPipeline: metalPipeline
        ))
        
        // Decoder
        await decoder.addLayer(try await DenseLayer(
            inputSize: encodedDim,
            outputSize: encodedDim * 2,
            activation: .relu,
            metalPipeline: metalPipeline
        ))
        
        await decoder.addLayer(try await DenseLayer(
            inputSize: encodedDim * 2,
            outputSize: inputDim,
            activation: .linear,
            metalPipeline: metalPipeline
        ))
        
        self.encoderNetwork = encoder
        self.decoderNetwork = decoder
    }
    
    /// Create a denoising autoencoder
    public func createDenoisingEncoder(
        inputDim: Int,
        encodedDim: Int,
        noiseLevel: Float = 0.1
    ) async throws {
        let encoder = try await NeuralNetwork(metalPipeline: metalPipeline)
        let decoder = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Add noise layer at input
        await encoder.addLayer(try await GaussianNoiseLayer(
            stddev: noiseLevel,
            metalPipeline: metalPipeline
        ))
        
        // Standard encoder architecture
        await encoder.addLayer(try await DenseLayer(
            inputSize: inputDim,
            outputSize: encodedDim * 2,
            activation: .relu,
            metalPipeline: metalPipeline
        ))
        
        await encoder.addLayer(try await DenseLayer(
            inputSize: encodedDim * 2,
            outputSize: encodedDim,
            activation: .tanh,
            metalPipeline: metalPipeline
        ))
        
        // Decoder
        await decoder.addLayer(try await DenseLayer(
            inputSize: encodedDim,
            outputSize: encodedDim * 2,
            activation: .relu,
            metalPipeline: metalPipeline
        ))
        
        await decoder.addLayer(try await DenseLayer(
            inputSize: encodedDim * 2,
            outputSize: inputDim,
            activation: .linear,
            metalPipeline: metalPipeline
        ))
        
        self.encoderNetwork = encoder
        self.decoderNetwork = decoder
    }
    
    // MARK: - Metrics
    
    public func getMetrics() -> VectorEncoderMetrics {
        trainingMetrics
    }
    
    public func resetMetrics() {
        trainingMetrics = VectorEncoderMetrics()
    }
    
    // MARK: - Private Methods
    
    private func initializeNetworks(
        inputDim: Int,
        encodedDim: Int,
        config: VectorEncoderConfig
    ) async throws {
        // Initialize encoder
        self.encoderNetwork = try await NeuralNetwork(metalPipeline: metalPipeline)
        guard let encoder = encoderNetwork else { return }
        
        var currentDim = inputDim
        
        // Encoder layers
        for hiddenSize in config.encoderHiddenSizes {
            await encoder.addLayer(try await DenseLayer(
                inputSize: currentDim,
                outputSize: hiddenSize,
                activation: .relu,
                metalPipeline: metalPipeline
            ))
            
            if config.useDropout {
                await encoder.addLayer(try await DropoutLayer(
                    rate: config.dropoutRate,
                    metalPipeline: metalPipeline
                ))
            }
            
            if config.useBatchNorm {
                await encoder.addLayer(try await BatchNormLayer(
                    numFeatures: hiddenSize,
                    metalPipeline: metalPipeline
                ))
            }
            
            currentDim = hiddenSize
        }
        
        // Bottleneck layer
        await encoder.addLayer(try await DenseLayer(
            inputSize: currentDim,
            outputSize: encodedDim,
            activation: config.bottleneckActivation,
            metalPipeline: metalPipeline
        ))
        
        // Initialize decoder (mirror of encoder)
        self.decoderNetwork = try await NeuralNetwork(metalPipeline: metalPipeline)
        guard let decoder = decoderNetwork else { return }
        
        currentDim = encodedDim
        let reversedSizes = config.encoderHiddenSizes.reversed()
        
        for hiddenSize in reversedSizes {
            await decoder.addLayer(try await DenseLayer(
                inputSize: currentDim,
                outputSize: hiddenSize,
                activation: .relu,
                metalPipeline: metalPipeline
            ))
            
            if config.useBatchNorm {
                await decoder.addLayer(try await BatchNormLayer(
                    numFeatures: hiddenSize,
                    metalPipeline: metalPipeline
                ))
            }
            
            currentDim = hiddenSize
        }
        
        // Output layer
        await decoder.addLayer(try await DenseLayer(
            inputSize: currentDim,
            outputSize: inputDim,
            activation: .linear,
            metalPipeline: metalPipeline
        ))
    }
    
    private func encodeBatch(
        _ batch: [Vector],
        encoder: NeuralNetwork
    ) async throws -> [Vector] {
        let batchBuffer = try await vectorsToMetalBuffer(batch)
        let encoded = try await encoder.forward(batchBuffer)
        let vectors = try await metalBufferToVectors(encoded, count: batch.count)
        
        await metalPipeline.releaseBuffer(batchBuffer)
        await metalPipeline.releaseBuffer(encoded)
        
        return vectors
    }
    
    private func decodeBatch(
        _ batch: [Vector],
        decoder: NeuralNetwork
    ) async throws -> [Vector] {
        let batchBuffer = try await vectorsToMetalBuffer(batch)
        let decoded = try await decoder.forward(batchBuffer)
        let vectors = try await metalBufferToVectors(decoded, count: batch.count)
        
        await metalPipeline.releaseBuffer(batchBuffer)
        await metalPipeline.releaseBuffer(decoded)
        
        return vectors
    }
    
    private func trainBatch(
        _ batch: [(input: MetalBuffer, target: MetalBuffer)],
        encoder: NeuralNetwork,
        decoder: NeuralNetwork,
        learningRate: Float
    ) async throws -> Float {
        var totalLoss: Float = 0
        
        for (input, target) in batch {
            // Forward pass through encoder
            let encoded = try await encoder.forward(input)
            
            // Forward pass through decoder
            let reconstructed = try await decoder.forward(encoded)
            
            // Compute reconstruction loss (MSE)
            let loss = try await computeMSELoss(
                predicted: reconstructed,
                target: target
            )
            totalLoss += loss
            
            // Backward pass
            let gradOutput = try await computeMSEGradient(
                predicted: reconstructed,
                target: target
            )
            
            // Convert gradient to array for backward compatibility
            let gradArray = gradOutput.toArray()
            
            // Decoder backward
            let decoderGradients = await decoder.backward(gradArray, learningRate: learningRate)
            
            // Extract gradient for encoder (we need the input gradient from decoder)
            // For now, use a simple approach - this should be improved
            let encodedGradArray = Array(repeating: Float(0), count: encoded.count)
            
            // Encoder backward
            _ = await encoder.backward(encodedGradArray, learningRate: learningRate)
            
            // Update parameters using Adam optimizer
            await updateParametersWithAdam(
                encoder: encoder,
                decoder: decoder,
                learningRate: learningRate,
                iteration: batchIdx + epoch * batches.count
            )
            
            // Release intermediate buffers
            await metalPipeline.releaseBuffer(encoded)
            await metalPipeline.releaseBuffer(reconstructed)
            await metalPipeline.releaseBuffer(gradOutput)
        }
        
        return totalLoss / Float(batch.count)
    }
    
    private func prepareTrainingData(_ vectors: [Vector]) async throws -> [(input: MetalBuffer, target: MetalBuffer)] {
        var trainingData: [(input: MetalBuffer, target: MetalBuffer)] = []
        
        for vector in vectors {
            let buffer = try await vectorToMetalBuffer(vector)
            // For autoencoder, input and target are the same
            trainingData.append((input: buffer, target: buffer))
        }
        
        return trainingData
    }
    
    private func vectorToMetalBuffer(_ vector: Vector) async throws -> MetalBuffer {
        let buffer = try await metalPipeline.allocateBuffer(size: vector.count)
        buffer.buffer.contents().copyMemory(
            from: vector,
            byteCount: vector.count * MemoryLayout<Float>.stride
        )
        return buffer
    }
    
    private func vectorsToMetalBuffer(_ vectors: [Vector]) async throws -> MetalBuffer {
        guard !vectors.isEmpty else {
            throw VectorEncoderError.emptyInput
        }
        
        let vectorSize = vectors.first!.count
        let totalSize = vectors.count * vectorSize
        
        let buffer = try await metalPipeline.allocateBuffer(size: totalSize)
        let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: totalSize)
        
        for (idx, vector) in vectors.enumerated() {
            let offset = idx * vectorSize
            for i in 0..<vectorSize {
                ptr[offset + i] = vector[i]
            }
        }
        
        return buffer
    }
    
    private func metalBufferToVectors(_ buffer: MetalBuffer, count: Int) async throws -> [Vector] {
        let vectorSize = buffer.count / count
        let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
        
        var vectors: [Vector] = []
        for i in 0..<count {
            let offset = i * vectorSize
            let values = (0..<vectorSize).map { ptr[offset + $0] }
            vectors.append(Vector(values))
        }
        
        return vectors
    }
    
    private func computeMSELoss(predicted: MetalBuffer, target: MetalBuffer) async throws -> Float {
        // Use Metal shader for efficient loss computation
        return try await computeLossWithMetal(
            predictions: predicted,
            targets: target,
            lossType: .mse
        )
    }
    
    private func computeMSEGradient(predicted: MetalBuffer, target: MetalBuffer) async throws -> MetalBuffer {
        let gradient = try await metalPipeline.allocateBuffer(size: predicted.count)
        
        let predPtr = predicted.buffer.contents().bindMemory(to: Float.self, capacity: predicted.count)
        let targetPtr = target.buffer.contents().bindMemory(to: Float.self, capacity: target.count)
        let gradPtr = gradient.buffer.contents().bindMemory(to: Float.self, capacity: gradient.count)
        
        let scale = 2.0 / Float(predicted.count)
        for i in 0..<predicted.count {
            gradPtr[i] = scale * (predPtr[i] - targetPtr[i])
        }
        
        return gradient
    }
    
    // MARK: - Adam Optimizer
    
    private var adamFirstMoments: [String: MetalBuffer] = [:]
    private var adamSecondMoments: [String: MetalBuffer] = [:]
    private let beta1: Float = 0.9
    private let beta2: Float = 0.999
    private let epsilon: Float = 1e-8
    
    private func updateParametersWithAdam(
        encoder: NeuralNetwork,
        decoder: NeuralNetwork,
        learningRate: Float,
        iteration: Int
    ) async {
        // Bias correction
        let biasCorrection1 = 1.0 - pow(beta1, Float(iteration + 1))
        let biasCorrection2 = 1.0 - pow(beta2, Float(iteration + 1))
        
        // Update encoder parameters
        await updateNetworkParametersWithAdam(
            network: encoder,
            prefix: "encoder",
            learningRate: learningRate,
            biasCorrection1: biasCorrection1,
            biasCorrection2: biasCorrection2
        )
        
        // Update decoder parameters
        await updateNetworkParametersWithAdam(
            network: decoder,
            prefix: "decoder",
            learningRate: learningRate,
            biasCorrection1: biasCorrection1,
            biasCorrection2: biasCorrection2
        )
    }
    
    private func updateNetworkParametersWithAdam(
        network: NeuralNetwork,
        prefix: String,
        learningRate: Float,
        biasCorrection1: Float,
        biasCorrection2: Float
    ) async {
        // This would integrate with the neural network's parameter update mechanism
        // For now, using the built-in gradient descent
        // In a full implementation, this would access the network's parameters
        // and apply Adam updates using Metal shaders
    }
    
    // MARK: - Metal Loss Computation
    
    private func computeLossWithMetal(
        predictions: MetalBuffer,
        targets: MetalBuffer,
        lossType: LossFunction
    ) async throws -> Float {
        // Load ML optimization shaders if needed
        try await metalPipeline.pipelineManager.loadMLOptimizationShaders()
        
        // Allocate atomic float for thread-safe accumulation
        let lossBuffer = try await metalPipeline.allocateBuffer(size: MemoryLayout<Float>.size)
        
        // Clear the buffer
        lossBuffer.buffer.contents().bindMemory(to: Float.self, capacity: 1).pointee = 0.0
        
        let commandBuffer = metalPipeline.commandQueue.makeCommandBuffer()!
        
        // First pass: accumulate loss
        let encoder1 = commandBuffer.makeComputeCommandEncoder()!
        
        let operation: MLOperation = switch lossType {
        case .mse: .mseLoss
        case .crossEntropy: .crossEntropyLoss
        case .contrastive: .contrastiveLoss
        }
        
        let lossShader = try await metalPipeline.pipelineManager.getMLPipeline(operation)
        
        encoder1.setComputePipelineState(lossShader)
        encoder1.setBuffer(predictions.buffer, offset: 0, index: 0)
        encoder1.setBuffer(targets.buffer, offset: 0, index: 1)
        encoder1.setBuffer(lossBuffer.buffer, offset: 0, index: 2)
        
        var count = UInt32(predictions.count)
        encoder1.setBytes(&count, length: 4, index: 3)
        
        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (predictions.count + 255) / 256,
            height: 1,
            depth: 1
        )
        
        encoder1.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder1.endEncoding()
        
        // Second pass: reduce (for MSE)
        if lossType == .mse {
            let encoder2 = commandBuffer.makeComputeCommandEncoder()!
            let reduceShader = try await metalPipeline.pipelineManager.getMLPipeline(.mseLossReduce)
            
            encoder2.setComputePipelineState(reduceShader)
            encoder2.setBuffer(lossBuffer.buffer, offset: 0, index: 0)
            encoder2.setBytes(&count, length: 4, index: 1)
            
            encoder2.dispatchThreadgroups(
                MTLSize(width: 1, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
            )
            encoder2.endEncoding()
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Read result
        let lossPointer = lossBuffer.buffer.contents().bindMemory(to: Float.self, capacity: 1)
        let loss = lossPointer.pointee
        
        await metalPipeline.releaseBuffer(lossBuffer)
        return loss
    }
}

private enum VectorEncoderLossFunction {
    case mse
    case crossEntropy
    case contrastive
}

// Extension for ML operations
extension MetalPipelineManager {
    public enum MLOperation {
        case mseLoss
        case mseLossReduce
        case crossEntropyLoss
        case contrastiveLoss
        case adamUpdate
        case vaeReparameterization
        case tripletGradients
        case randomUniform
        case randomNormal
        
        var shaderName: String {
            switch self {
            case .mseLoss: return "mse_loss"
            case .mseLossReduce: return "mse_loss_reduce"
            case .crossEntropyLoss: return "cross_entropy_loss"
            case .contrastiveLoss: return "contrastive_loss"
            case .adamUpdate: return "adam_update"
            case .vaeReparameterization: return "vae_reparameterization"
            case .tripletGradients: return "triplet_loss_gradients"
            case .randomUniform: return "uniform_random_init"
            case .randomNormal: return "normal_random_transform"
            }
        }
    }
    
    public func getMLPipeline(_ operation: MLOperation) async throws -> MTLComputePipelineState {
        return try await getOrCreatePipeline(functionName: operation.shaderName)
    }
    
    public func loadMLOptimizationShaders() async throws {
        let shaderNames = [
            "vae_reparameterization",
            "mse_loss",
            "mse_loss_reduce",
            "cross_entropy_loss",
            "contrastive_loss",
            "adam_update",
            "triplet_loss_gradients",
            "uniform_random_init",
            "normal_random_transform"
        ]
        
        for shaderName in shaderNames {
            _ = try await getOrCreatePipeline(functionName: shaderName)
        }
    }
}

// MARK: - Supporting Types

/// Configuration for vector encoder
public struct VectorEncoderConfig: Sendable {
    public let encoderHiddenSizes: [Int]
    public let encodedDimension: Int?
    public let epochs: Int
    public let batchSize: Int
    public let learningRate: Float
    public let useDropout: Bool
    public let dropoutRate: Float
    public let useBatchNorm: Bool
    public let bottleneckActivation: ActivationType
    public let convergenceThreshold: Float
    
    public init(
        encoderHiddenSizes: [Int] = [512, 256, 128],
        encodedDimension: Int? = nil,
        epochs: Int = 100,
        batchSize: Int = 32,
        learningRate: Float = 0.001,
        useDropout: Bool = true,
        dropoutRate: Float = 0.2,
        useBatchNorm: Bool = false,
        bottleneckActivation: ActivationType = .tanh,
        convergenceThreshold: Float = 0.001
    ) {
        self.encoderHiddenSizes = encoderHiddenSizes
        self.encodedDimension = encodedDimension
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.useDropout = useDropout
        self.dropoutRate = dropoutRate
        self.useBatchNorm = useBatchNorm
        self.bottleneckActivation = bottleneckActivation
        self.convergenceThreshold = convergenceThreshold
    }
}

/// Metrics for encoder performance
public struct VectorEncoderMetrics: Sendable {
    public var vectorsEncoded: Int = 0
    public var trainingTime: TimeInterval = 0
    public var finalLoss: Float = 0
    public var compressionRatio: Float = 0
    
    public var encodingThroughput: Double {
        guard trainingTime > 0 else { return 0 }
        return Double(vectorsEncoded) / trainingTime
    }
}

/// Sparse activation layer for sparse autoencoders
actor SparseActivationLayer: NeuralLayer {
    private let sparsityWeight: Float
    private let metalPipeline: MetalMLPipeline
    private var lastActivations: MetalBuffer?
    
    init(sparsityWeight: Float, metalPipeline: MetalMLPipeline) {
        self.sparsityWeight = sparsityWeight
        self.metalPipeline = metalPipeline
    }
    
    func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Apply sparsity constraint
        let output = try await metalPipeline.allocateBuffer(size: input.count)
        
        let inPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        
        // Apply soft thresholding for sparsity
        let threshold = sparsityWeight
        for i in 0..<input.count {
            let val = inPtr[i]
            if abs(val) > threshold {
                outPtr[i] = val - threshold * (val > 0 ? 1 : -1)
            } else {
                outPtr[i] = 0
            }
        }
        
        lastActivations = output
        return output
    }
    
    func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let activations = lastActivations else {
            throw VectorEncoderError.backwardBeforeForward
        }
        
        let gradInput = try await metalPipeline.allocateBuffer(size: gradOutput.count)
        
        let gradOutPtr = gradOutput.buffer.contents().bindMemory(to: Float.self, capacity: gradOutput.count)
        let gradInPtr = gradInput.buffer.contents().bindMemory(to: Float.self, capacity: gradInput.count)
        let actPtr = activations.buffer.contents().bindMemory(to: Float.self, capacity: activations.count)
        
        // Gradient passes through where activation is non-zero
        for i in 0..<gradOutput.count {
            gradInPtr[i] = actPtr[i] != 0 ? gradOutPtr[i] : 0
        }
        
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
    
    func setTraining(_ training: Bool) {
        // No effect
    }
    
    func zeroGradients() async {
        // No gradients to zero
    }
    
    func scaleGradients(_ scale: Float) async {
        // No gradients to scale
    }
    
    func updateParametersWithOptimizer(_ optimizer: any Optimizer) async throws {
        // No parameters to update
    }
}

/// Gaussian noise layer for denoising autoencoders
actor GaussianNoiseLayer: NeuralLayer {
    private let stddev: Float
    private let metalPipeline: MetalMLPipeline
    private var isTraining = true
    
    init(stddev: Float, metalPipeline: MetalMLPipeline) {
        self.stddev = stddev
        self.metalPipeline = metalPipeline
    }
    
    func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        guard isTraining else {
            // No noise during inference
            return input
        }
        
        let output = try await metalPipeline.allocateBuffer(size: input.count)
        
        let inPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        
        // Add Gaussian noise
        for i in 0..<input.count {
            let noise = Float.random(in: -1...1) * stddev
            outPtr[i] = inPtr[i] + noise
        }
        
        return output
    }
    
    func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        // Gradient passes through unchanged
        return gradOutput
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
    
    func setTraining(_ training: Bool) {
        isTraining = training
    }
    
    func zeroGradients() async {
        // No gradients to zero
    }
    
    func scaleGradients(_ scale: Float) async {
        // No gradients to scale
    }
    
    func updateParametersWithOptimizer(_ optimizer: any Optimizer) async throws {
        // No parameters to update
    }
}

// MARK: - Errors

public enum VectorEncoderError: LocalizedError {
    case encoderNotInitialized
    case decoderNotInitialized
    case emptyTrainingData
    case emptyInput
    case networkInitializationFailed
    case backwardBeforeForward
    
    public var errorDescription: String? {
        switch self {
        case .encoderNotInitialized:
            return "Encoder network not initialized"
        case .decoderNotInitialized:
            return "Decoder network not initialized"
        case .emptyTrainingData:
            return "Training data is empty"
        case .emptyInput:
            return "Input is empty"
        case .networkInitializationFailed:
            return "Failed to initialize neural networks"
        case .backwardBeforeForward:
            return "Backward called before forward pass"
        }
    }
}
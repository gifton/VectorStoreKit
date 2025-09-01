// VectorStoreKit: Optimized Vector Encoder
//
// Phase 2 optimized neural encoder/decoder with Metal acceleration
// Leverages unified buffer management and optimized shader compilation

import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate
import simd
import os.log

/// Optimized neural vector encoder using Phase 2 acceleration
public actor OptimizedVectorEncoder {
    
    // MARK: - Properties
    
    private let accelerationEngine: MetalAccelerationEngine
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let logger = Logger(subsystem: "VectorStoreKit", category: "OptimizedVectorEncoder")
    
    // Neural network components
    private var encoder: OptimizedEncoderNetwork?
    private var decoder: OptimizedDecoderNetwork?
    
    // Optimization state
    private var isTraining = false
    private var batchNormRunningStats: BatchNormStats?
    private var adaptiveLearningRate: Float = 0.001
    
    // Performance monitoring
    private var metrics = EncoderMetrics()
    
    // MARK: - Initialization
    
    public init() async throws {
        self.accelerationEngine = MetalAccelerationEngine.shared
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw VectorEncoderError.metalNotAvailable
        }
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw VectorEncoderError.commandQueueCreationFailed
        }
        self.commandQueue = queue
        
        logger.info("Initialized optimized vector encoder with Metal acceleration")
    }
    
    // MARK: - Configuration
    
    /// Configure encoder with optimized architecture
    public func configure(
        inputDimension: Int,
        encodedDimension: Int,
        config: EncoderConfiguration = .default
    ) async throws {
        
        // Create optimized encoder architecture
        encoder = try await OptimizedEncoderNetwork(
            inputDim: inputDimension,
            outputDim: encodedDimension,
            hiddenLayers: config.encoderHiddenLayers,
            activation: config.activation,
            device: device,
            accelerationEngine: accelerationEngine
        )
        
        // Create optimized decoder architecture (mirror of encoder)
        decoder = try await OptimizedDecoderNetwork(
            inputDim: encodedDimension,
            outputDim: inputDimension,
            hiddenLayers: config.decoderHiddenLayers,
            activation: config.activation,
            device: device,
            accelerationEngine: accelerationEngine
        )
        
        // Initialize batch normalization statistics
        batchNormRunningStats = BatchNormStats(
            dimensions: config.encoderHiddenLayers + [encodedDimension]
        )
        
        logger.info("Configured encoder: \(inputDimension) -> \(encodedDimension) dimensions")
    }
    
    // MARK: - Encoding/Decoding
    
    /// Encode vectors using optimized Metal acceleration
    public func encode(_ vectors: [[Float]]) async throws -> [[Float]] {
        guard let encoder = encoder else {
            throw VectorEncoderError.encoderNotInitialized
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Convert to Metal-friendly format
        let inputTensor = try await createTensor(from: vectors)
        
        // Forward pass through encoder
        let encoded = try await encoder.forward(
            inputTensor,
            training: false,
            batchNormStats: batchNormRunningStats
        )
        
        // Convert back to array format
        let result = try await tensorToArray(encoded)
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        metrics.encodingTime += duration
        metrics.vectorsEncoded += vectors.count
        
        logger.debug("Encoded \(vectors.count) vectors in \(duration * 1000)ms")
        
        return result
    }
    
    /// Decode vectors using optimized Metal acceleration
    public func decode(_ encoded: [[Float]]) async throws -> [[Float]] {
        guard let decoder = decoder else {
            throw VectorEncoderError.decoderNotInitialized
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Convert to Metal-friendly format
        let inputTensor = try await createTensor(from: encoded)
        
        // Forward pass through decoder
        let decoded = try await decoder.forward(
            inputTensor,
            training: false,
            batchNormStats: batchNormRunningStats
        )
        
        // Convert back to array format
        let result = try await tensorToArray(decoded)
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        metrics.decodingTime += duration
        metrics.vectorsDecoded += encoded.count
        
        return result
    }
    
    // MARK: - Training
    
    /// Train autoencoder with optimized Metal acceleration
    public func train(
        on vectors: [[Float]],
        config: TrainingConfiguration = .default,
        progressHandler: ((TrainingProgress) -> Void)? = nil
    ) async throws {
        
        guard let encoder = encoder, let decoder = decoder else {
            throw VectorEncoderError.networkNotInitialized
        }
        
        isTraining = true
        defer { isTraining = false }
        
        logger.info("Starting training on \(vectors.count) vectors")
        
        let batchSize = config.batchSize
        let epochs = config.epochs
        
        // Initialize optimizer states
        let encoderOptimizer = AdamOptimizer(
            parameters: encoder.parameters,
            learningRate: config.learningRate,
            beta1: 0.9,
            beta2: 0.999
        )
        
        let decoderOptimizer = AdamOptimizer(
            parameters: decoder.parameters,
            learningRate: config.learningRate,
            beta1: 0.9,
            beta2: 0.999
        )
        
        var totalLoss: Float = 0
        
        for epoch in 0..<epochs {
            let epochStartTime = CFAbsoluteTimeGetCurrent()
            var epochLoss: Float = 0
            var batchCount = 0
            
            // Shuffle data for each epoch
            let shuffled = vectors.shuffled()
            
            // Process batches
            for batchStart in stride(from: 0, to: vectors.count, by: batchSize) {
                let batchEnd = min(batchStart + batchSize, vectors.count)
                let batch = Array(shuffled[batchStart..<batchEnd])
                
                // Forward pass
                let (loss, encodedGradient) = try await trainStep(
                    batch: batch,
                    encoder: encoder,
                    decoder: decoder
                )
                
                // Backward pass and update
                try await updateWeights(
                    encoder: encoder,
                    decoder: decoder,
                    encoderOptimizer: encoderOptimizer,
                    decoderOptimizer: decoderOptimizer,
                    encodedGradient: encodedGradient,
                    learningRate: adaptiveLearningRate
                )
                
                epochLoss += loss
                batchCount += 1
                
                // Update progress
                if let handler = progressHandler {
                    let progress = TrainingProgress(
                        epoch: epoch,
                        batch: batchCount,
                        totalBatches: (vectors.count + batchSize - 1) / batchSize,
                        loss: loss,
                        learningRate: adaptiveLearningRate
                    )
                    handler(progress)
                }
            }
            
            // Calculate epoch metrics
            epochLoss /= Float(batchCount)
            totalLoss = epochLoss
            
            // Adaptive learning rate decay
            if epoch > 0 && epoch % config.learningRateDecayEpochs == 0 {
                adaptiveLearningRate *= config.learningRateDecay
            }
            
            let epochDuration = CFAbsoluteTimeGetCurrent() - epochStartTime
            
            logger.info("""
                Epoch \(epoch + 1)/\(epochs) completed:
                - Loss: \(epochLoss)
                - Duration: \(epochDuration)s
                - Learning rate: \(adaptiveLearningRate)
                """)
        }
        
        metrics.trainingLoss = totalLoss
        logger.info("Training completed with final loss: \(totalLoss)")
    }
    
    // MARK: - Private Methods
    
    private func trainStep(
        batch: [[Float]],
        encoder: OptimizedEncoderNetwork,
        decoder: OptimizedDecoderNetwork
    ) async throws -> (loss: Float, encodedGradient: MetalTensor) {
        
        // Convert batch to tensor
        let inputTensor = try await createTensor(from: batch)
        
        // Forward pass through autoencoder
        let encoded = try await encoder.forward(
            inputTensor,
            training: true,
            batchNormStats: batchNormRunningStats
        )
        
        let reconstructed = try await decoder.forward(
            encoded,
            training: true,
            batchNormStats: batchNormRunningStats
        )
        
        // Compute reconstruction loss (MSE)
        let loss = try await computeMSELoss(
            predicted: reconstructed,
            target: inputTensor
        )
        
        // Compute gradients
        let outputGradient = try await computeMSEGradient(
            predicted: reconstructed,
            target: inputTensor
        )
        
        // Backward pass through decoder
        let encodedGradient = try await decoder.backward(
            outputGradient: outputGradient,
            cached: decoder.lastForwardCache!
        )
        
        return (loss, encodedGradient)
    }
    
    private func updateWeights(
        encoder: OptimizedEncoderNetwork,
        decoder: OptimizedDecoderNetwork,
        encoderOptimizer: AdamOptimizer,
        decoderOptimizer: AdamOptimizer,
        encodedGradient: MetalTensor,
        learningRate: Float
    ) async throws {
        
        // Backward pass through encoder
        let inputGradient = try await encoder.backward(
            outputGradient: encodedGradient,
            cached: encoder.lastForwardCache!
        )
        
        // Update encoder weights
        try await encoderOptimizer.update(
            gradients: encoder.gradients,
            learningRate: learningRate
        )
        
        // Update decoder weights
        try await decoderOptimizer.update(
            gradients: decoder.gradients,
            learningRate: learningRate
        )
    }
    
    private func createTensor(from vectors: [[Float]]) async throws -> MetalTensor {
        let batchSize = vectors.count
        let dimension = vectors[0].count
        
        // Flatten data
        let flatData = vectors.flatMap { $0 }
        
        // Create Metal buffer
        guard let buffer = device.makeBuffer(bytes: flatData, length: flatData.count * MemoryLayout<Float>.size, options: .storageModeShared) else {
            throw VectorEncoderError.metalNotAvailable
        }
        
        return MetalTensor(
            buffer: buffer,
            shape: [batchSize, dimension],
            device: device
        )
    }
    
    private func tensorToArray(_ tensor: MetalTensor) async throws -> [[Float]] {
        let batchSize = tensor.shape[0]
        let dimension = tensor.shape[1]
        
        // Copy data from GPU
        var result = [[Float]](repeating: [Float](repeating: 0, count: dimension), count: batchSize)
        
        let pointer = tensor.buffer.contents().bindMemory(to: Float.self, capacity: batchSize * dimension)
        
        for i in 0..<batchSize {
            for j in 0..<dimension {
                result[i][j] = pointer[i * dimension + j]
            }
        }
        
        return result
    }
    
    private func computeMSELoss(predicted: MetalTensor, target: MetalTensor) async throws -> Float {
        // Use Metal acceleration for loss computation
        let diff = try await accelerationEngine.subtractTensors(predicted, target)
        let squared = try await accelerationEngine.squareTensor(diff)
        let sum = try await accelerationEngine.reduceSumTensor(squared)
        
        let count = predicted.shape.reduce(1, *)
        return sum / Float(count)
    }
    
    private func computeMSEGradient(predicted: MetalTensor, target: MetalTensor) async throws -> MetalTensor {
        // Gradient of MSE: 2 * (predicted - target) / n
        let diff = try await accelerationEngine.subtractTensors(predicted, target)
        let count = Float(predicted.shape.reduce(1, *))
        let scale = 2.0 / count
        
        return try await accelerationEngine.scaleTensor(diff, by: scale)
    }
    
    // MARK: - Metrics
    
    public func getMetrics() -> EncoderMetrics {
        metrics
    }
    
    public func resetMetrics() {
        metrics = EncoderMetrics()
    }
}

// MARK: - Optimized Network Components

/// Optimized encoder network with Metal acceleration
private actor OptimizedEncoderNetwork {
    let layers: [OptimizedLayer]
    let device: MTLDevice
    let accelerationEngine: MetalAccelerationEngine
    
    var parameters: [MetalTensor] = []
    var gradients: [MetalTensor] = []
    var lastForwardCache: ForwardCache?
    
    init(
        inputDim: Int,
        outputDim: Int,
        hiddenLayers: [Int],
        activation: ActivationType,
        device: MTLDevice,
        accelerationEngine: MetalAccelerationEngine
    ) async throws {
        self.device = device
        self.accelerationEngine = accelerationEngine
        
        // Build layer architecture
        var layers: [OptimizedLayer] = []
        var currentDim = inputDim
        
        // Hidden layers
        for (i, hiddenDim) in hiddenLayers.enumerated() {
            layers.append(
                try await OptimizedLinearLayer(
                    inputDim: currentDim,
                    outputDim: hiddenDim,
                    device: device,
                    accelerationEngine: accelerationEngine
                )
            )
            
            layers.append(
                OptimizedBatchNormLayer(
                    dimension: hiddenDim,
                    device: device,
                    accelerationEngine: accelerationEngine
                )
            )
            
            layers.append(
                OptimizedActivationLayer(
                    activation: activation,
                    device: device,
                    accelerationEngine: accelerationEngine
                )
            )
            
            currentDim = hiddenDim
        }
        
        // Output layer
        layers.append(
            try await OptimizedLinearLayer(
                inputDim: currentDim,
                outputDim: outputDim,
                device: device,
                accelerationEngine: accelerationEngine
            )
        )
        
        self.layers = layers
        
        // Collect parameters
        for layer in layers {
            parameters.append(contentsOf: await layer.getParameters())
        }
    }
    
    func forward(
        _ input: MetalTensor,
        training: Bool,
        batchNormStats: BatchNormStats?
    ) async throws -> MetalTensor {
        var output = input
        var layerOutputs: [MetalTensor] = []
        
        for layer in layers {
            output = try await layer.forward(output, training: training)
            layerOutputs.append(output)
        }
        
        lastForwardCache = ForwardCache(
            input: input,
            layerOutputs: layerOutputs
        )
        
        return output
    }
    
    func backward(
        outputGradient: MetalTensor,
        cached: ForwardCache
    ) async throws -> MetalTensor {
        var gradient = outputGradient
        gradients = []
        
        // Backward through layers in reverse order
        for (i, layer) in layers.enumerated().reversed() {
            let input = i > 0 ? cached.layerOutputs[i - 1] : cached.input
            let (inputGrad, layerGrads) = try await layer.backward(
                outputGradient: gradient,
                input: input,
                output: cached.layerOutputs[i]
            )
            
            gradient = inputGrad
            gradients.insert(contentsOf: layerGrads, at: 0)
        }
        
        return gradient
    }
}

/// Optimized decoder network (similar structure to encoder)
private actor OptimizedDecoderNetwork {
    // Similar implementation to OptimizedEncoderNetwork
    // but with reversed layer dimensions
    let layers: [OptimizedLayer]
    let device: MTLDevice
    let accelerationEngine: MetalAccelerationEngine
    
    var parameters: [MetalTensor] = []
    var gradients: [MetalTensor] = []
    var lastForwardCache: ForwardCache?
    
    init(
        inputDim: Int,
        outputDim: Int,
        hiddenLayers: [Int],
        activation: ActivationType,
        device: MTLDevice,
        accelerationEngine: MetalAccelerationEngine
    ) async throws {
        self.device = device
        self.accelerationEngine = accelerationEngine
        
        // Build layer architecture (reversed from encoder)
        var layers: [OptimizedLayer] = []
        var currentDim = inputDim
        
        // Hidden layers (in reverse order)
        for hiddenDim in hiddenLayers.reversed() {
            layers.append(
                try await OptimizedLinearLayer(
                    inputDim: currentDim,
                    outputDim: hiddenDim,
                    device: device,
                    accelerationEngine: accelerationEngine
                )
            )
            
            layers.append(
                OptimizedBatchNormLayer(
                    dimension: hiddenDim,
                    device: device,
                    accelerationEngine: accelerationEngine
                )
            )
            
            layers.append(
                OptimizedActivationLayer(
                    activation: activation,
                    device: device,
                    accelerationEngine: accelerationEngine
                )
            )
            
            currentDim = hiddenDim
        }
        
        // Output layer
        layers.append(
            try await OptimizedLinearLayer(
                inputDim: currentDim,
                outputDim: outputDim,
                device: device,
                accelerationEngine: accelerationEngine
            )
        )
        
        self.layers = layers
        
        // Collect parameters
        for layer in layers {
            parameters.append(contentsOf: await layer.getParameters())
        }
    }
    
    func forward(
        _ input: MetalTensor,
        training: Bool,
        batchNormStats: BatchNormStats?
    ) async throws -> MetalTensor {
        var output = input
        var layerOutputs: [MetalTensor] = []
        
        for layer in layers {
            output = try await layer.forward(output, training: training)
            layerOutputs.append(output)
        }
        
        lastForwardCache = ForwardCache(
            input: input,
            layerOutputs: layerOutputs
        )
        
        return output
    }
    
    func backward(
        outputGradient: MetalTensor,
        cached: ForwardCache
    ) async throws -> MetalTensor {
        var gradient = outputGradient
        gradients = []
        
        // Backward through layers in reverse order
        for (i, layer) in layers.enumerated().reversed() {
            let input = i > 0 ? cached.layerOutputs[i - 1] : cached.input
            let (inputGrad, layerGrads) = try await layer.backward(
                outputGradient: gradient,
                input: input,
                output: cached.layerOutputs[i]
            )
            
            gradient = inputGrad
            gradients.insert(contentsOf: layerGrads, at: 0)
        }
        
        return gradient
    }
}

// MARK: - Supporting Types

/// Configuration for encoder architecture
public struct EncoderConfiguration {
    public let encoderHiddenLayers: [Int]
    public let decoderHiddenLayers: [Int]
    public let activation: ActivationType
    public let dropout: Float
    public let useBatchNorm: Bool
    
    public static let `default` = EncoderConfiguration(
        encoderHiddenLayers: [512, 256],
        decoderHiddenLayers: [256, 512],
        activation: .relu,
        dropout: 0.1,
        useBatchNorm: true
    )
    
    public static let small = EncoderConfiguration(
        encoderHiddenLayers: [256, 128],
        decoderHiddenLayers: [128, 256],
        activation: .relu,
        dropout: 0.1,
        useBatchNorm: true
    )
    
    public static let large = EncoderConfiguration(
        encoderHiddenLayers: [1024, 512, 256],
        decoderHiddenLayers: [256, 512, 1024],
        activation: .gelu,
        dropout: 0.2,
        useBatchNorm: true
    )
}

/// Training configuration
public struct TrainingConfiguration {
    public let batchSize: Int
    public let epochs: Int
    public let learningRate: Float
    public let learningRateDecay: Float
    public let learningRateDecayEpochs: Int
    public let validationSplit: Float
    
    public static let `default` = TrainingConfiguration(
        batchSize: 256,
        epochs: 100,
        learningRate: 0.001,
        learningRateDecay: 0.95,
        learningRateDecayEpochs: 10,
        validationSplit: 0.1
    )
}

/// Training progress information
public struct TrainingProgress {
    public let epoch: Int
    public let batch: Int
    public let totalBatches: Int
    public let loss: Float
    public let learningRate: Float
}

/// Encoder performance metrics
public struct EncoderMetrics {
    public var vectorsEncoded: Int = 0
    public var vectorsDecoded: Int = 0
    public var encodingTime: TimeInterval = 0
    public var decodingTime: TimeInterval = 0
    public var trainingLoss: Float = 0
    
    public var averageEncodingTime: TimeInterval {
        vectorsEncoded > 0 ? encodingTime / Double(vectorsEncoded) : 0
    }
    
    public var averageDecodingTime: TimeInterval {
        vectorsDecoded > 0 ? decodingTime / Double(vectorsDecoded) : 0
    }
}

/// Batch normalization running statistics
private struct BatchNormStats {
    var runningMean: [[Float]]
    var runningVar: [[Float]]
    let momentum: Float = 0.9
    
    init(dimensions: [Int]) {
        runningMean = dimensions.map { dim in
            [Float](repeating: 0, count: dim)
        }
        runningVar = dimensions.map { dim in
            [Float](repeating: 1, count: dim)
        }
    }
}

/// Forward pass cache for backpropagation
private struct ForwardCache {
    let input: MetalTensor
    let layerOutputs: [MetalTensor]
}

/// Metal tensor wrapper
private struct MetalTensor {
    let buffer: MTLBuffer
    let shape: [Int]
    let device: MTLDevice
}

/// Layer protocol for optimized layers
private protocol OptimizedLayer {
    func forward(_ input: MetalTensor, training: Bool) async throws -> MetalTensor
    func backward(outputGradient: MetalTensor, input: MetalTensor, output: MetalTensor) async throws -> (inputGradient: MetalTensor, parameterGradients: [MetalTensor])
    func getParameters() async -> [MetalTensor]
}

/// Optimized linear layer with Metal acceleration
private actor OptimizedLinearLayer: OptimizedLayer {
    let weight: MetalTensor
    let bias: MetalTensor
    let accelerationEngine: MetalAccelerationEngine
    let device: MTLDevice
    
    init(inputDim: Int, outputDim: Int, device: MTLDevice, accelerationEngine: MetalAccelerationEngine) async throws {
        self.accelerationEngine = accelerationEngine
        self.device = device
        
        // Initialize weights with Xavier/He initialization
        let scale = sqrt(2.0 / Float(inputDim))
        let weightData = (0..<inputDim * outputDim).map { _ in Float.random(in: -scale...scale) }
        let biasData = [Float](repeating: 0, count: outputDim)
        
        let weightBuffer = device.makeBuffer(bytes: weightData, length: weightData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let biasBuffer = device.makeBuffer(bytes: biasData, length: biasData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        
        self.weight = MetalTensor(buffer: weightBuffer, shape: [inputDim, outputDim], device: device)
        self.bias = MetalTensor(buffer: biasBuffer, shape: [outputDim], device: device)
    }
    
    func forward(_ input: MetalTensor, training: Bool) async throws -> MetalTensor {
        // Linear transformation: output = input * weight + bias
        // For simplicity, we'll use CPU computation with Accelerate
        let batchSize = input.shape[0]
        let inputDim = weight.shape[0]
        let outputDim = weight.shape[1]
        
        // Create output buffer
        let outputBuffer = device.makeBuffer(
            length: batchSize * outputDim * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        
        // Get pointers
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: batchSize * inputDim)
        let weightPtr = weight.buffer.contents().bindMemory(to: Float.self, capacity: inputDim * outputDim)
        let biasPtr = bias.buffer.contents().bindMemory(to: Float.self, capacity: outputDim)
        let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * outputDim)
        
        // Perform matrix multiplication using Accelerate
        // output = input * weight
        cblas_sgemm(
            CblasRowMajor,           // row-major order
            CblasNoTrans,            // don't transpose A (input)
            CblasNoTrans,            // don't transpose B (weight)
            Int32(batchSize),        // M: rows of A and C
            Int32(outputDim),        // N: columns of B and C
            Int32(inputDim),         // K: columns of A, rows of B
            1.0,                     // alpha
            inputPtr,                // A
            Int32(inputDim),         // lda
            weightPtr,               // B
            Int32(outputDim),        // ldb
            0.0,                     // beta
            outputPtr,               // C
            Int32(outputDim)         // ldc
        )
        
        // Add bias to each output
        for i in 0..<batchSize {
            let outputRowPtr = outputPtr.advanced(by: i * outputDim)
            vDSP_vadd(outputRowPtr, 1, biasPtr, 1, outputRowPtr, 1, vDSP_Length(outputDim))
        }
        
        return MetalTensor(buffer: outputBuffer, shape: [batchSize, outputDim], device: device)
    }
    
    func backward(outputGradient: MetalTensor, input: MetalTensor, output: MetalTensor) async throws -> (inputGradient: MetalTensor, parameterGradients: [MetalTensor]) {
        let batchSize = input.shape[0]
        let inputDim = weight.shape[0]
        let outputDim = weight.shape[1]
        
        // Compute weight gradient: weightGrad = input^T * outputGradient
        let weightGradBuffer = device.makeBuffer(
            length: inputDim * outputDim * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: batchSize * inputDim)
        let outputGradPtr = outputGradient.buffer.contents().bindMemory(to: Float.self, capacity: batchSize * outputDim)
        let weightGradPtr = weightGradBuffer.contents().bindMemory(to: Float.self, capacity: inputDim * outputDim)
        
        cblas_sgemm(
            CblasRowMajor,
            CblasTrans,              // transpose input
            CblasNoTrans,            // don't transpose outputGradient
            Int32(inputDim),         // M
            Int32(outputDim),        // N
            Int32(batchSize),        // K
            1.0 / Float(batchSize),  // alpha (average over batch)
            inputPtr,                // A
            Int32(inputDim),         // lda
            outputGradPtr,           // B
            Int32(outputDim),        // ldb
            0.0,                     // beta
            weightGradPtr,           // C
            Int32(outputDim)         // ldc
        )
        
        // Compute bias gradient: biasGrad = sum(outputGradient, axis=0)
        let biasGradBuffer = device.makeBuffer(
            length: outputDim * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        let biasGradPtr = biasGradBuffer.contents().bindMemory(to: Float.self, capacity: outputDim)
        
        // Initialize to zero
        memset(biasGradPtr, 0, outputDim * MemoryLayout<Float>.size)
        
        // Sum over batch dimension
        for i in 0..<batchSize {
            let gradRowPtr = outputGradPtr.advanced(by: i * outputDim)
            vDSP_vadd(biasGradPtr, 1, gradRowPtr, 1, biasGradPtr, 1, vDSP_Length(outputDim))
        }
        
        // Average over batch
        var scale = 1.0 / Float(batchSize)
        vDSP_vsmul(biasGradPtr, 1, &scale, biasGradPtr, 1, vDSP_Length(outputDim))
        
        // Compute input gradient: inputGrad = outputGradient * weight^T
        let inputGradBuffer = device.makeBuffer(
            length: batchSize * inputDim * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        
        let weightPtr = weight.buffer.contents().bindMemory(to: Float.self, capacity: inputDim * outputDim)
        let inputGradPtr = inputGradBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * inputDim)
        
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,            // don't transpose outputGradient
            CblasTrans,              // transpose weight
            Int32(batchSize),        // M
            Int32(inputDim),         // N
            Int32(outputDim),        // K
            1.0,                     // alpha
            outputGradPtr,           // A
            Int32(outputDim),        // lda
            weightPtr,               // B
            Int32(outputDim),        // ldb (before transpose)
            0.0,                     // beta
            inputGradPtr,           // C
            Int32(inputDim)          // ldc
        )
        
        let weightGrad = MetalTensor(buffer: weightGradBuffer, shape: [inputDim, outputDim], device: device)
        let biasGrad = MetalTensor(buffer: biasGradBuffer, shape: [outputDim], device: device)
        let inputGrad = MetalTensor(buffer: inputGradBuffer, shape: [batchSize, inputDim], device: device)
        
        return (inputGrad, [weightGrad, biasGrad])
    }
    
    func getParameters() async -> [MetalTensor] {
        [weight, bias]
    }
}

/// Batch normalization layer with simplified implementation
private struct OptimizedBatchNormLayer: OptimizedLayer {
    let dimension: Int
    let device: MTLDevice
    let accelerationEngine: MetalAccelerationEngine
    let gamma: MetalTensor  // Scale parameter
    let beta: MetalTensor   // Shift parameter
    let eps: Float = 1e-5
    
    init(dimension: Int, device: MTLDevice, accelerationEngine: MetalAccelerationEngine) {
        self.dimension = dimension
        self.device = device
        self.accelerationEngine = accelerationEngine
        
        // Initialize gamma to 1 and beta to 0
        let gammaData = [Float](repeating: 1.0, count: dimension)
        let betaData = [Float](repeating: 0.0, count: dimension)
        
        let gammaBuffer = device.makeBuffer(bytes: gammaData, length: gammaData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let betaBuffer = device.makeBuffer(bytes: betaData, length: betaData.count * MemoryLayout<Float>.size, options: .storageModeShared)!
        
        self.gamma = MetalTensor(buffer: gammaBuffer, shape: [dimension], device: device)
        self.beta = MetalTensor(buffer: betaBuffer, shape: [dimension], device: device)
    }
    
    func forward(_ input: MetalTensor, training: Bool) async throws -> MetalTensor {
        // Simplified batch norm: just normalize by feature dimension
        let batchSize = input.shape[0]
        let features = input.shape[1]
        
        let outputBuffer = device.makeBuffer(
            length: batchSize * features * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: batchSize * features)
        let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: batchSize * features)
        let gammaPtr = gamma.buffer.contents().bindMemory(to: Float.self, capacity: features)
        let betaPtr = beta.buffer.contents().bindMemory(to: Float.self, capacity: features)
        
        // Compute mean and variance for each feature
        var mean = [Float](repeating: 0, count: features)
        var variance = [Float](repeating: 0, count: features)
        
        // Compute mean
        for i in 0..<batchSize {
            let rowPtr = inputPtr.advanced(by: i * features)
            vDSP_vadd(mean, 1, rowPtr, 1, &mean, 1, vDSP_Length(features))
        }
        var scale = 1.0 / Float(batchSize)
        vDSP_vsmul(mean, 1, &scale, &mean, 1, vDSP_Length(features))
        
        // Compute variance
        for i in 0..<batchSize {
            let rowPtr = inputPtr.advanced(by: i * features)
            for j in 0..<features {
                let diff = rowPtr[j] - mean[j]
                variance[j] += diff * diff
            }
        }
        vDSP_vsmul(variance, 1, &scale, &variance, 1, vDSP_Length(features))
        
        // Normalize and scale
        for i in 0..<batchSize {
            let inRowPtr = inputPtr.advanced(by: i * features)
            let outRowPtr = outputPtr.advanced(by: i * features)
            
            for j in 0..<features {
                let normalized = (inRowPtr[j] - mean[j]) / sqrt(variance[j] + eps)
                outRowPtr[j] = normalized * gammaPtr[j] + betaPtr[j]
            }
        }
        
        return MetalTensor(buffer: outputBuffer, shape: input.shape, device: device)
    }
    
    func backward(outputGradient: MetalTensor, input: MetalTensor, output: MetalTensor) async throws -> (inputGradient: MetalTensor, parameterGradients: [MetalTensor]) {
        // Simplified: just pass gradients through
        let gammaGrad = MetalTensor(
            buffer: device.makeBuffer(length: dimension * MemoryLayout<Float>.size, options: .storageModeShared)!,
            shape: [dimension],
            device: device
        )
        let betaGrad = MetalTensor(
            buffer: device.makeBuffer(length: dimension * MemoryLayout<Float>.size, options: .storageModeShared)!,
            shape: [dimension],
            device: device
        )
        
        return (outputGradient, [gammaGrad, betaGrad])
    }
    
    func getParameters() async -> [MetalTensor] {
        [gamma, beta]
    }
}

/// Activation layer with common activation functions
private struct OptimizedActivationLayer: OptimizedLayer {
    let activation: ActivationType
    let device: MTLDevice
    let accelerationEngine: MetalAccelerationEngine
    
    func forward(_ input: MetalTensor, training: Bool) async throws -> MetalTensor {
        let totalElements = input.shape.reduce(1, *)
        let outputBuffer = device.makeBuffer(
            length: totalElements * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let outputPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        
        switch activation {
        case .relu:
            // ReLU: max(0, x)
            var zero: Float = 0
            vDSP_vthres(inputPtr, 1, &zero, outputPtr, 1, vDSP_Length(totalElements))
            
        case .gelu:
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            for i in 0..<totalElements {
                let x = inputPtr[i]
                let a = 0.044715 * x * x * x
                let inner = sqrt(2.0 / .pi) * (x + a)
                outputPtr[i] = 0.5 * x * (1.0 + tanh(inner))
            }
            
        case .tanh:
            // Tanh activation
            vvtanhf(outputPtr, inputPtr, &totalElements)
            
        case .sigmoid:
            // Sigmoid: 1 / (1 + exp(-x))
            for i in 0..<totalElements {
                outputPtr[i] = 1.0 / (1.0 + exp(-inputPtr[i]))
            }
        }
        
        return MetalTensor(buffer: outputBuffer, shape: input.shape, device: device)
    }
    
    func backward(outputGradient: MetalTensor, input: MetalTensor, output: MetalTensor) async throws -> (inputGradient: MetalTensor, parameterGradients: [MetalTensor]) {
        let totalElements = input.shape.reduce(1, *)
        let gradBuffer = device.makeBuffer(
            length: totalElements * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let gradOutPtr = outputGradient.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let gradInPtr = gradBuffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        
        switch activation {
        case .relu:
            // ReLU gradient: 1 if x > 0, else 0
            for i in 0..<totalElements {
                gradInPtr[i] = inputPtr[i] > 0 ? gradOutPtr[i] : 0
            }
            
        case .gelu:
            // GELU gradient (approximation)
            for i in 0..<totalElements {
                let x = inputPtr[i]
                let a = 0.044715 * x * x * x
                let inner = sqrt(2.0 / .pi) * (x + a)
                let tanhInner = tanh(inner)
                let sechSquared = 1.0 - tanhInner * tanhInner
                let dInner = sqrt(2.0 / .pi) * (1.0 + 3.0 * 0.044715 * x * x)
                let derivative = 0.5 * (1.0 + tanhInner) + 0.5 * x * sechSquared * dInner
                gradInPtr[i] = gradOutPtr[i] * derivative
            }
            
        case .tanh:
            // Tanh gradient: 1 - tanh(x)^2
            for i in 0..<totalElements {
                let tanhVal = outputPtr[i]
                gradInPtr[i] = gradOutPtr[i] * (1.0 - tanhVal * tanhVal)
            }
            
        case .sigmoid:
            // Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x))
            for i in 0..<totalElements {
                let sigVal = outputPtr[i]
                gradInPtr[i] = gradOutPtr[i] * sigVal * (1.0 - sigVal)
            }
        }
        
        return (MetalTensor(buffer: gradBuffer, shape: input.shape, device: device), [])
    }
    
    func getParameters() async -> [MetalTensor] {
        []
    }
}

/// Adam optimizer implementation
private actor AdamOptimizer {
    let parameters: [MetalTensor]
    let learningRate: Float
    let beta1: Float
    let beta2: Float
    let eps: Float = 1e-8
    
    // Adam state
    private var m: [MetalTensor] = []  // First moment estimates
    private var v: [MetalTensor] = []  // Second moment estimates
    private var t: Int = 0             // Time step
    
    init(parameters: [MetalTensor], learningRate: Float, beta1: Float, beta2: Float) {
        self.parameters = parameters
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        
        // Initialize moment estimates
        for param in parameters {
            let totalElements = param.shape.reduce(1, *)
            let mBuffer = param.device.makeBuffer(
                length: totalElements * MemoryLayout<Float>.size,
                options: .storageModeShared
            )!
            let vBuffer = param.device.makeBuffer(
                length: totalElements * MemoryLayout<Float>.size,
                options: .storageModeShared
            )!
            
            // Initialize to zero
            memset(mBuffer.contents(), 0, totalElements * MemoryLayout<Float>.size)
            memset(vBuffer.contents(), 0, totalElements * MemoryLayout<Float>.size)
            
            m.append(MetalTensor(buffer: mBuffer, shape: param.shape, device: param.device))
            v.append(MetalTensor(buffer: vBuffer, shape: param.shape, device: param.device))
        }
    }
    
    func update(gradients: [MetalTensor], learningRate: Float) async throws {
        guard parameters.count == gradients.count else {
            throw VectorEncoderError.networkInitializationFailed
        }
        
        t += 1
        let biasCorrection1 = 1.0 - pow(beta1, Float(t))
        let biasCorrection2 = 1.0 - pow(beta2, Float(t))
        let stepSize = learningRate * sqrt(biasCorrection2) / biasCorrection1
        
        for i in 0..<parameters.count {
            let param = parameters[i]
            let grad = gradients[i]
            let totalElements = param.shape.reduce(1, *)
            
            let paramPtr = param.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
            let gradPtr = grad.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
            let mPtr = m[i].buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
            let vPtr = v[i].buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
            
            // Update biased first moment estimate
            // m = beta1 * m + (1 - beta1) * grad
            var oneMinusBeta1 = 1.0 - beta1
            vDSP_vsmul(gradPtr, 1, &oneMinusBeta1, mPtr, 1, vDSP_Length(totalElements))
            var beta1Copy = beta1
            vDSP_vsma(mPtr, 1, &beta1Copy, mPtr, 1, mPtr, 1, vDSP_Length(totalElements))
            
            // Update biased second moment estimate
            // v = beta2 * v + (1 - beta2) * grad^2
            var gradSquared = [Float](repeating: 0, count: totalElements)
            vDSP_vsq(gradPtr, 1, &gradSquared, 1, vDSP_Length(totalElements))
            
            var oneMinusBeta2 = 1.0 - beta2
            vDSP_vsmul(&gradSquared, 1, &oneMinusBeta2, vPtr, 1, vDSP_Length(totalElements))
            var beta2Copy = beta2
            vDSP_vsma(vPtr, 1, &beta2Copy, vPtr, 1, vPtr, 1, vDSP_Length(totalElements))
            
            // Update parameters
            // param = param - stepSize * m / (sqrt(v) + eps)
            for j in 0..<totalElements {
                let mHat = mPtr[j]
                let vHat = vPtr[j]
                paramPtr[j] -= stepSize * mHat / (sqrt(vHat) + eps)
            }
        }
    }
}

// ActivationType is defined in Core/ML/Activations.swift

/// Encoder errors
public enum VectorEncoderError: Error {
    case metalNotAvailable
    case commandQueueCreationFailed
    case encoderNotInitialized
    case decoderNotInitialized
    case networkNotInitialized
    case emptyTrainingData
    case networkInitializationFailed
}
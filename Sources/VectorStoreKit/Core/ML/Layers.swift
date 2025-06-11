// VectorStoreKit: Neural Network Layers
//
// Essential actor-based layers with Metal acceleration

import Foundation
import Accelerate
@preconcurrency import Metal

/// Dense (fully connected) layer
public actor DenseLayer: NeuralLayer {
    // MARK: - Properties
    private let inputSize: Int
    private let outputSize: Int
    private let activation: Activation
    private let name: String
    private let metalPipeline: MetalMLPipeline
    private var isTraining: Bool = true
    
    // Parameter names
    private let weightsName: String
    private let biasName: String
    private let weightsGradName: String
    private let biasGradName: String
    
    // Parameter store
    private let parameterStore: ParameterStore
    
    // Cached buffers for backward pass
    private var lastInput: MetalBuffer?
    private var lastOutput: MetalBuffer?
    private var lastPreActivation: MetalBuffer?
    
    // MARK: - Initialization
    public init(
        inputSize: Int,
        outputSize: Int,
        activation: Activation = .linear,
        name: String = "dense",
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = activation
        self.name = name
        
        // Initialize parameter names
        self.weightsName = "\(name)_weights"
        self.biasName = "\(name)_bias"
        self.weightsGradName = "\(name)_weights_grad"
        self.biasGradName = "\(name)_bias_grad"
        
        // Initialize pipeline and parameter store
        self.metalPipeline = metalPipeline
        self.parameterStore = await ParameterStore(device: metalPipeline.device)
        
        // Initialize parameters
        try await initializeParameters()
    }
    
    private func initializeParameters() async throws {
        let paramStore = parameterStore
        
        // Allocate weight buffer (inputSize x outputSize)
        let weightsBuffer = try await paramStore.allocateParameter(
            name: weightsName,
            size: inputSize * outputSize
        )
        
        // Allocate bias buffer
        let biasBuffer = try await paramStore.allocateParameter(
            name: biasName,
            size: outputSize
        )
        
        // Initialize weights using He/Xavier initialization
        let scale: Float = switch activation {
        case .relu, .leakyRelu:
            sqrt(2.0 / Float(inputSize)) // He initialization
        default:
            sqrt(1.0 / Float(inputSize)) // Xavier initialization
        }
        
        // Initialize weights with random values
        let weightsPtr = weightsBuffer.buffer.contents().bindMemory(to: Float.self, capacity: weightsBuffer.count)
        for i in 0..<weightsBuffer.count {
            weightsPtr[i] = Float.random(in: -scale..<scale)
        }
        
        // Initialize bias to zero
        let biasPtr = biasBuffer.buffer.contents().bindMemory(to: Float.self, capacity: biasBuffer.count)
        for i in 0..<biasBuffer.count {
            biasPtr[i] = 0.0
        }
        
        // Allocate gradient buffers
        _ = try await paramStore.allocateGradient(name: weightsGradName, size: inputSize * outputSize)
        _ = try await paramStore.allocateGradient(name: biasGradName, size: outputSize)
    }
    
    // MARK: - NeuralLayer Protocol
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        guard input.count == inputSize else {
            throw MetalMLError.incompatibleBufferSize(expected: inputSize, actual: input.count)
        }
        
        // Store input for backward pass
        self.lastInput = input
        
        let paramStore = parameterStore
        guard let weights = await paramStore.getParameter(name: weightsName),
              let bias = await paramStore.getParameter(name: biasName) else {
            throw MetalMLError.parameterNotFound(name: weightsName)
        }
        
        // Allocate output buffer
        let preActivation = try await allocateBuffer(size: outputSize)
        
        // Perform matrix multiplication using Metal
        let operations = await metalPipeline.getOperations()
        
        // Weights are stored as [outputSize x inputSize], input is [inputSize x 1]
        // We need to compute: output = weights * input + bias
        // This is equivalent to a matrix-vector multiplication
        try await operations.matmul(
            weights,
            input,
            output: preActivation,
            m: outputSize,
            n: 1,
            k: inputSize,
            useTiling: false
        )
        
        // Add bias
        try await operations.addBias(
            matrix: preActivation,
            bias: bias,
            rows: 1,
            cols: outputSize
        )
        
        // Store pre-activation for backward pass
        self.lastPreActivation = preActivation
        
        // Apply activation function
        let activated = try await applyActivation(preActivation)
        self.lastOutput = activated
        
        return activated
    }
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let input = lastInput,
              let preActivation = lastPreActivation else {
            throw MetalMLError.incompatibleBufferSize(expected: 0, actual: 0)
        }
        
        let paramStore = parameterStore
        guard let weights = await paramStore.getParameter(name: weightsName),
              let weightsGrad = await paramStore.getGradient(name: weightsGradName),
              let biasGrad = await paramStore.getGradient(name: biasGradName) else {
            throw MetalMLError.parameterNotFound(name: weightsName)
        }
        
        // Apply activation derivative
        let gradPreActivation = try await applyActivationDerivative(gradOutput, preActivation: preActivation)
        
        // Compute weight gradients: gradW = gradOutput * input^T
        let gradOutputPtr = gradPreActivation.buffer.contents().bindMemory(to: Float.self, capacity: gradPreActivation.count)
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let weightsGradPtr = weightsGrad.buffer.contents().bindMemory(to: Float.self, capacity: weightsGrad.count)
        
        for i in 0..<outputSize {
            for j in 0..<inputSize {
                weightsGradPtr[i * inputSize + j] = gradOutputPtr[i] * inputPtr[j]
            }
        }
        
        // Compute bias gradients: gradB = gradOutput
        let biasGradPtr = biasGrad.buffer.contents().bindMemory(to: Float.self, capacity: biasGrad.count)
        for i in 0..<outputSize {
            biasGradPtr[i] = gradOutputPtr[i]
        }
        
        // Compute input gradients: gradInput = weights^T * gradOutput
        let gradInput = try await allocateBuffer(size: inputSize)
        let gradInputPtr = gradInput.buffer.contents().bindMemory(to: Float.self, capacity: gradInput.count)
        let weightsPtr = weights.buffer.contents().bindMemory(to: Float.self, capacity: weights.count)
        
        for j in 0..<inputSize {
            var sum: Float = 0.0
            for i in 0..<outputSize {
                sum += weightsPtr[i * inputSize + j] * gradOutputPtr[i]
            }
            gradInputPtr[j] = sum
        }
        
        return gradInput
    }
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        let paramStore = parameterStore
        
        // Update weights
        if let weightsGrad = await paramStore.getGradient(name: weightsGradName) {
            try await paramStore.updateParameter(name: weightsName, with: weightsGrad, learningRate: learningRate)
        }
        
        // Update bias
        if let biasGrad = await paramStore.getGradient(name: biasGradName) {
            try await paramStore.updateParameter(name: biasName, with: biasGrad, learningRate: learningRate)
        }
    }
    
    public func getParameters() async -> MetalBuffer? {
        let paramStore = parameterStore
        return await paramStore.getParameter(name: weightsName)
    }
    
    public func getParameterCount() async -> Int {
        inputSize * outputSize + outputSize
    }
    
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
    
    private func allocateBuffer(size: Int) async throws -> MetalBuffer {
        try await metalPipeline.allocateBuffer(size: size)
    }
    
    // MARK: - Activation Functions
    private func applyActivation(_ input: MetalBuffer) async throws -> MetalBuffer {
        let output = try await allocateBuffer(size: input.count)
        let operations = await metalPipeline.getOperations()
        
        try await operations.applyActivation(
            input,
            output: output,
            activation: activation
        )
        
        return output
    }
    
    private func applyActivationDerivative(_ gradOutput: MetalBuffer, preActivation: MetalBuffer) async throws -> MetalBuffer {
        let gradInput = try await allocateBuffer(size: gradOutput.count)
        let operations = await metalPipeline.getOperations()
        
        // Note: lastOutput contains the post-activation values needed for some derivatives
        let output = self.lastOutput ?? preActivation
        
        try await operations.applyActivationDerivative(
            gradOutput: gradOutput,
            input: preActivation,
            output: output,
            activation: activation,
            gradInput: gradInput
        )
        
        return gradInput
    }
}

/// Dropout layer for regularization
public actor DropoutLayer: NeuralLayer {
    private let rate: Float
    private var mask: MetalBuffer?
    private let metalPipeline: MetalMLPipeline
    private var isTraining: Bool = true
    
    public init(rate: Float, metalPipeline: MetalMLPipeline) async throws {
        self.rate = rate
        self.metalPipeline = metalPipeline
    }
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        guard isTraining else {
            return input // No dropout during evaluation
        }
        
        // Generate dropout mask and apply using Metal shader
        let output = try await allocateBuffer(size: input.count)
        let mask = try await allocateBuffer(size: input.count)
        
        // Store mask for backward pass
        self.mask = mask
        
        // Use Metal shader for dropout
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "dropout_forward")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        encoder.setBuffer(mask.buffer, offset: 0, index: 2)
        
        var dropoutRate = rate
        var seed = UInt32.random(in: 0..<UInt32.max)
        encoder.setBytes(&dropoutRate, length: 4, index: 3)
        encoder.setBytes(&seed, length: 4, index: 4)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (input.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return output
    }
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let mask = self.mask else {
            return gradOutput // No mask means no dropout was applied
        }
        
        // Apply mask to gradients using Metal shader
        let gradInput = try await allocateBuffer(size: gradOutput.count)
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "dropout_backward")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
        encoder.setBuffer(mask.buffer, offset: 0, index: 1)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (gradOutput.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return gradInput
    }
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        // No parameters to update
    }
    
    public func getParameters() async -> MetalBuffer? {
        nil
    }
    
    public func getParameterCount() async -> Int {
        0
    }
    
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
    
    private func allocateBuffer(size: Int) async throws -> MetalBuffer {
        try await metalPipeline.allocateBuffer(size: size)
    }
}

/// Batch normalization layer
public actor BatchNormLayer: NeuralLayer {
    private let numFeatures: Int
    private let momentum: Float
    private let epsilon: Float
    private let metalPipeline: MetalMLPipeline
    private var isTraining: Bool = true
    
    // Parameter names
    private let gammaName: String
    private let betaName: String
    private let runningMeanName: String
    private let runningVarName: String
    
    // Parameter store
    private let parameterStore: ParameterStore
    
    // Cached values for backward pass
    private var lastInput: MetalBuffer?
    private var lastNormalized: MetalBuffer?
    private var lastMean: MetalBuffer?
    private var lastVar: MetalBuffer?
    
    public init(
        numFeatures: Int,
        momentum: Float = 0.9,
        epsilon: Float = 1e-5,
        name: String = "batchnorm",
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.numFeatures = numFeatures
        self.momentum = momentum
        self.epsilon = epsilon
        self.metalPipeline = metalPipeline
        
        self.gammaName = "\(name)_gamma"
        self.betaName = "\(name)_beta"
        self.runningMeanName = "\(name)_running_mean"
        self.runningVarName = "\(name)_running_var"
        
        self.parameterStore = await ParameterStore(device: metalPipeline.device)
        
        try await initializeParameters()
    }
    
    private func initializeParameters() async throws {
        let paramStore = parameterStore
        
        // Allocate learnable parameters
        let gamma = try await paramStore.allocateParameter(name: gammaName, size: numFeatures)
        let beta = try await paramStore.allocateParameter(name: betaName, size: numFeatures)
        
        // Initialize gamma to 1 and beta to 0
        let gammaPtr = gamma.buffer.contents().bindMemory(to: Float.self, capacity: gamma.count)
        let betaPtr = beta.buffer.contents().bindMemory(to: Float.self, capacity: beta.count)
        
        for i in 0..<numFeatures {
            gammaPtr[i] = 1.0
            betaPtr[i] = 0.0
        }
        
        // Allocate running statistics
        let runningMean = try await paramStore.allocateParameter(name: runningMeanName, size: numFeatures)
        let runningVar = try await paramStore.allocateParameter(name: runningVarName, size: numFeatures)
        
        // Initialize to 0 and 1
        let meanPtr = runningMean.buffer.contents().bindMemory(to: Float.self, capacity: runningMean.count)
        let varPtr = runningVar.buffer.contents().bindMemory(to: Float.self, capacity: runningVar.count)
        
        for i in 0..<numFeatures {
            meanPtr[i] = 0.0
            varPtr[i] = 1.0
        }
        
        // Allocate gradient buffers
        _ = try await paramStore.allocateGradient(name: "\(gammaName)_grad", size: numFeatures)
        _ = try await paramStore.allocateGradient(name: "\(betaName)_grad", size: numFeatures)
    }
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        guard input.count == numFeatures else {
            throw MetalMLError.incompatibleBufferSize(expected: numFeatures, actual: input.count)
        }
        
        self.lastInput = input
        let paramStore = parameterStore
        
        guard let gamma = await paramStore.getParameter(name: gammaName),
              let beta = await paramStore.getParameter(name: betaName),
              let runningMean = await paramStore.getParameter(name: runningMeanName),
              let runningVar = await paramStore.getParameter(name: runningVarName) else {
            throw MetalMLError.parameterNotFound(name: gammaName)
        }
        
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let gammaPtr = gamma.buffer.contents().bindMemory(to: Float.self, capacity: gamma.count)
        let betaPtr = beta.buffer.contents().bindMemory(to: Float.self, capacity: beta.count)
        
        let mean: MetalBuffer
        let variance: MetalBuffer
        
        if isTraining {
            // Note: This implementation assumes feature-wise normalization
            // For proper batch normalization, we would need the full batch tensor
            // Currently treating single sample as the batch
            mean = try await allocateBuffer(size: numFeatures)
            variance = try await allocateBuffer(size: numFeatures)
            
            let meanPtr = mean.buffer.contents().bindMemory(to: Float.self, capacity: mean.count)
            let varPtr = variance.buffer.contents().bindMemory(to: Float.self, capacity: variance.count)
            
            // Compute mean and variance for each feature
            // In a full implementation, this would compute across the batch dimension
            for i in 0..<numFeatures {
                meanPtr[i] = inputPtr[i]
                // Variance of single sample is 0, using small epsilon for numerical stability
                varPtr[i] = epsilon
            }
            
            // Update running statistics with exponential moving average
            let runningMeanPtr = runningMean.buffer.contents().bindMemory(to: Float.self, capacity: runningMean.count)
            let runningVarPtr = runningVar.buffer.contents().bindMemory(to: Float.self, capacity: runningVar.count)
            
            for i in 0..<numFeatures {
                runningMeanPtr[i] = momentum * runningMeanPtr[i] + (1 - momentum) * meanPtr[i]
                runningVarPtr[i] = momentum * runningVarPtr[i] + (1 - momentum) * varPtr[i]
            }
        } else {
            // Use running statistics during evaluation
            mean = runningMean
            variance = runningVar
        }
        
        self.lastMean = mean
        self.lastVar = variance
        
        // Normalize
        let normalized = try await allocateBuffer(size: numFeatures)
        let output = try await allocateBuffer(size: numFeatures)
        
        let normalizedPtr = normalized.buffer.contents().bindMemory(to: Float.self, capacity: normalized.count)
        let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        let meanPtr = mean.buffer.contents().bindMemory(to: Float.self, capacity: mean.count)
        let varPtr = variance.buffer.contents().bindMemory(to: Float.self, capacity: variance.count)
        
        for i in 0..<numFeatures {
            normalizedPtr[i] = (inputPtr[i] - meanPtr[i]) / sqrt(varPtr[i] + epsilon)
            outputPtr[i] = gammaPtr[i] * normalizedPtr[i] + betaPtr[i]
        }
        
        self.lastNormalized = normalized
        
        return output
    }
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let input = lastInput,
              let normalized = lastNormalized,
              let _ = lastMean,
              let variance = lastVar else {
            throw MetalMLError.incompatibleBufferSize(expected: 0, actual: 0)
        }
        
        let paramStore = parameterStore
        guard let gamma = await paramStore.getParameter(name: gammaName),
              let gammaGrad = await paramStore.getGradient(name: "\(gammaName)_grad"),
              let betaGrad = await paramStore.getGradient(name: "\(betaName)_grad") else {
            throw MetalMLError.parameterNotFound(name: gammaName)
        }
        
        // Batch normalization backward pass
        let gradInput = try await allocateBuffer(size: input.count)
        
        let gradOutputPtr = gradOutput.buffer.contents().bindMemory(to: Float.self, capacity: gradOutput.count)
        let normalizedPtr = normalized.buffer.contents().bindMemory(to: Float.self, capacity: normalized.count)
        let gammaPtr = gamma.buffer.contents().bindMemory(to: Float.self, capacity: gamma.count)
        let gammaGradPtr = gammaGrad.buffer.contents().bindMemory(to: Float.self, capacity: gammaGrad.count)
        let betaGradPtr = betaGrad.buffer.contents().bindMemory(to: Float.self, capacity: betaGrad.count)
        let gradInputPtr = gradInput.buffer.contents().bindMemory(to: Float.self, capacity: gradInput.count)
        let varPtr = variance.buffer.contents().bindMemory(to: Float.self, capacity: variance.count)
        
        // Compute parameter gradients
        for i in 0..<numFeatures {
            // Gradient w.r.t gamma: sum over batch of gradOutput * normalized
            gammaGradPtr[i] = gradOutputPtr[i] * normalizedPtr[i]
            
            // Gradient w.r.t beta: sum over batch of gradOutput
            betaGradPtr[i] = gradOutputPtr[i]
            
            // Gradient w.r.t input (simplified for single sample)
            // Full implementation would include batch statistics gradients
            let stdInv = 1.0 / sqrt(varPtr[i] + epsilon)
            gradInputPtr[i] = gradOutputPtr[i] * gammaPtr[i] * stdInv
        }
        
        return gradInput
    }
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        let paramStore = parameterStore
        
        if let gammaGrad = await paramStore.getGradient(name: "\(gammaName)_grad") {
            try await paramStore.updateParameter(name: gammaName, with: gammaGrad, learningRate: learningRate)
        }
        
        if let betaGrad = await paramStore.getGradient(name: "\(betaName)_grad") {
            try await paramStore.updateParameter(name: betaName, with: betaGrad, learningRate: learningRate)
        }
    }
    
    public func getParameters() async -> MetalBuffer? {
        let paramStore = parameterStore
        return await paramStore.getParameter(name: gammaName)
    }
    
    public func getParameterCount() async -> Int {
        2 * numFeatures // gamma and beta
    }
    
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
    
    private func allocateBuffer(size: Int) async throws -> MetalBuffer {
        try await metalPipeline.allocateBuffer(size: size)
    }
}
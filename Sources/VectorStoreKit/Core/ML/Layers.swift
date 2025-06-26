// VectorStoreKit: Neural Network Layers
//
// Essential actor-based layers with Metal acceleration

import Foundation
import Accelerate
@preconcurrency import Metal

/// Dense (fully connected) layer
public actor DenseLayer: NeuralLayer {
    // MARK: - Properties
    public let inputSize: Int
    public let outputSize: Int
    public let activation: Activation
    public let name: String
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
    
    // Gradient accumulation buffers
    private var accumulatedWeightsGrad: MetalBuffer?
    private var accumulatedBiasGrad: MetalBuffer?
    
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
        let weightsGrad = try await paramStore.allocateGradient(name: weightsGradName, size: inputSize * outputSize)
        let biasGrad = try await paramStore.allocateGradient(name: biasGradName, size: outputSize)
        
        // Initialize gradient accumulators
        self.accumulatedWeightsGrad = weightsGrad
        self.accumulatedBiasGrad = biasGrad
        
        // Zero initialize gradients
        let wGradPtr = weightsGrad.buffer.contents().bindMemory(to: Float.self, capacity: weightsGrad.count)
        let bGradPtr = biasGrad.buffer.contents().bindMemory(to: Float.self, capacity: biasGrad.count)
        for i in 0..<weightsGrad.count {
            wGradPtr[i] = 0.0
        }
        for i in 0..<biasGrad.count {
            bGradPtr[i] = 0.0
        }
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
        
        // Compute weight gradients using Metal: gradW = gradOutput * input^T
        let operations = await metalPipeline.getOperations()
        
        // Use outer product for weight gradients
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let outerProductPipeline = try await shaderLibrary.pipeline(for: MLShaderLibrary.MatrixOperation.outerProduct.rawValue)
        let copyPipeline = try await shaderLibrary.pipeline(for: MLShaderLibrary.MatrixOperation.copyMatrix.rawValue)
        
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            // Use accumulation version to add to existing gradients
            let accumulatePipeline = try await shaderLibrary.pipeline(for: MLShaderLibrary.MatrixOperation.outerProductAccumulate.rawValue)
            encoder.setComputePipelineState(accumulatePipeline)
            encoder.setBuffer(gradPreActivation.buffer, offset: 0, index: 0)  // [outputSize x 1]
            encoder.setBuffer(input.buffer, offset: 0, index: 1)               // [inputSize x 1]
            encoder.setBuffer(weightsGrad.buffer, offset: 0, index: 2)         // [outputSize x inputSize]
            
            var M = UInt32(outputSize)
            var N = UInt32(inputSize)
            encoder.setBytes(&M, length: 4, index: 3)
            encoder.setBytes(&N, length: 4, index: 4)
            
            let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
            let threadgroupCount = MTLSize(
                width: (inputSize + 15) / 16,
                height: (outputSize + 15) / 16,
                depth: 1
            )
            
            encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
            
            // Compute bias gradients using copy operation
            guard let copyEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            // Use accumulate operation for bias gradients
            let biasAccumulatePipeline = try await shaderLibrary.pipeline(for: MLShaderLibrary.OptimizationOperation.accumulateGradients.rawValue)
            copyEncoder.setComputePipelineState(biasAccumulatePipeline)
            copyEncoder.setBuffer(biasGrad.buffer, offset: 0, index: 0)  // Existing gradients
            copyEncoder.setBuffer(gradPreActivation.buffer, offset: 0, index: 1)  // New gradients to add
            
            var count = UInt32(outputSize)
            copyEncoder.setBytes(&count, length: 4, index: 2)
            
            let copyThreads = MTLSize(width: (outputSize + 255) / 256, height: 1, depth: 1)
            let copyThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            
            copyEncoder.dispatchThreadgroups(copyThreads, threadsPerThreadgroup: copyThreadgroup)
            copyEncoder.endEncoding()
        }
        
        // Compute input gradients: gradInput = weights^T * gradOutput
        let gradInput = try await allocateBuffer(size: inputSize)
        
        // First transpose the weights, then multiply
        // For dense layer: weights are [outputSize x inputSize]
        // We need weights^T which is [inputSize x outputSize]
        let transposedWeights = try await allocateBuffer(size: weights.count)
        
        // Use transpose kernel
        let transposePipeline = try await shaderLibrary.pipeline(for: MLShaderLibrary.MatrixOperation.transpose.rawValue)
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let transposeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            transposeEncoder.setComputePipelineState(transposePipeline)
            transposeEncoder.setBuffer(weights.buffer, offset: 0, index: 0)
            transposeEncoder.setBuffer(transposedWeights.buffer, offset: 0, index: 1)
            
            var rows = UInt32(outputSize)
            var cols = UInt32(inputSize)
            transposeEncoder.setBytes(&rows, length: 4, index: 2)
            transposeEncoder.setBytes(&cols, length: 4, index: 3)
            
            let transposeThreads = MTLSize(width: 16, height: 16, depth: 1)
            let transposeGroups = MTLSize(
                width: (inputSize + 15) / 16,
                height: (outputSize + 15) / 16,
                depth: 1
            )
            
            transposeEncoder.dispatchThreadgroups(transposeGroups, threadsPerThreadgroup: transposeThreads)
            transposeEncoder.endEncoding()
        }
        
        // Now multiply transposed weights with gradients
        try await operations.matmul(
            transposedWeights,
            gradPreActivation,
            output: gradInput,
            m: inputSize,
            n: 1,
            k: outputSize,
            useTiling: false
        )
        
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
    
    public func setTraining(_ training: Bool) {
        self.isTraining = training
    }
    
    public func getParameterStore() async -> ParameterStore {
        parameterStore
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
    
    // MARK: - Gradient Management
    
    public func zeroGradients() async {
        let paramStore = parameterStore
        
        // Zero weight gradients
        if let weightsGrad = await paramStore.getGradient(name: weightsGradName) {
            let shaderLibrary = await metalPipeline.getShaderLibrary()
            let commandQueue = await metalPipeline.getMetalCommandQueue()
            
            do {
                let zeroPipeline = try await shaderLibrary.pipeline(for: MLShaderLibrary.OptimizationOperation.zeroGradients.rawValue)
                
                try await commandQueue.submitAsync { commandBuffer in
                    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                        throw MetalMLError.commandQueueCreationFailed
                    }
                    
                    encoder.setComputePipelineState(zeroPipeline)
                    encoder.setBuffer(weightsGrad.buffer, offset: 0, index: 0)
                    var count = UInt32(weightsGrad.count)
                    encoder.setBytes(&count, length: 4, index: 1)
                    
                    let threads = MTLSize(width: (weightsGrad.count + 255) / 256, height: 1, depth: 1)
                    let threadgroup = MTLSize(width: 256, height: 1, depth: 1)
                    encoder.dispatchThreadgroups(threads, threadsPerThreadgroup: threadgroup)
                    encoder.endEncoding()
                }
            } catch {
                // Fallback to CPU zeroing
                let ptr = weightsGrad.buffer.contents().bindMemory(to: Float.self, capacity: weightsGrad.count)
                for i in 0..<weightsGrad.count {
                    ptr[i] = 0.0
                }
            }
        }
        
        // Zero bias gradients
        if let biasGrad = await paramStore.getGradient(name: biasGradName) {
            let shaderLibrary = await metalPipeline.getShaderLibrary()
            let commandQueue = await metalPipeline.getMetalCommandQueue()
            
            do {
                let zeroPipeline = try await shaderLibrary.pipeline(for: MLShaderLibrary.OptimizationOperation.zeroGradients.rawValue)
                
                try await commandQueue.submitAsync { commandBuffer in
                    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                        throw MetalMLError.commandQueueCreationFailed
                    }
                    
                    encoder.setComputePipelineState(zeroPipeline)
                    encoder.setBuffer(biasGrad.buffer, offset: 0, index: 0)
                    var count = UInt32(biasGrad.count)
                    encoder.setBytes(&count, length: 4, index: 1)
                    
                    let threads = MTLSize(width: (biasGrad.count + 255) / 256, height: 1, depth: 1)
                    let threadgroup = MTLSize(width: 256, height: 1, depth: 1)
                    encoder.dispatchThreadgroups(threads, threadsPerThreadgroup: threadgroup)
                    encoder.endEncoding()
                }
            } catch {
                // Fallback to CPU zeroing
                let ptr = biasGrad.buffer.contents().bindMemory(to: Float.self, capacity: biasGrad.count)
                for i in 0..<biasGrad.count {
                    ptr[i] = 0.0
                }
            }
        }
    }
    
    public func scaleGradients(_ scale: Float) async {
        let paramStore = parameterStore
        
        // Scale weight gradients
        if let weightsGrad = await paramStore.getGradient(name: weightsGradName) {
            let shaderLibrary = await metalPipeline.getShaderLibrary()
            let commandQueue = await metalPipeline.getMetalCommandQueue()
            
            do {
                let scalePipeline = try await shaderLibrary.pipeline(for: MLShaderLibrary.OptimizationOperation.scaleGradients.rawValue)
                
                try await commandQueue.submitAsync { commandBuffer in
                    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                        throw MetalMLError.commandQueueCreationFailed
                    }
                    
                    encoder.setComputePipelineState(scalePipeline)
                    encoder.setBuffer(weightsGrad.buffer, offset: 0, index: 0)
                    var scaleValue = scale
                    encoder.setBytes(&scaleValue, length: 4, index: 1)
                    var count = UInt32(weightsGrad.count)
                    encoder.setBytes(&count, length: 4, index: 2)
                    
                    let threads = MTLSize(width: (weightsGrad.count + 255) / 256, height: 1, depth: 1)
                    let threadgroup = MTLSize(width: 256, height: 1, depth: 1)
                    encoder.dispatchThreadgroups(threads, threadsPerThreadgroup: threadgroup)
                    encoder.endEncoding()
                }
            } catch {
                // Fallback to CPU scaling
                let ptr = weightsGrad.buffer.contents().bindMemory(to: Float.self, capacity: weightsGrad.count)
                for i in 0..<weightsGrad.count {
                    ptr[i] *= scale
                }
            }
        }
        
        // Scale bias gradients
        if let biasGrad = await paramStore.getGradient(name: biasGradName) {
            let shaderLibrary = await metalPipeline.getShaderLibrary()
            let commandQueue = await metalPipeline.getMetalCommandQueue()
            
            do {
                let scalePipeline = try await shaderLibrary.pipeline(for: MLShaderLibrary.OptimizationOperation.scaleGradients.rawValue)
                
                try await commandQueue.submitAsync { commandBuffer in
                    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                        throw MetalMLError.commandQueueCreationFailed
                    }
                    
                    encoder.setComputePipelineState(scalePipeline)
                    encoder.setBuffer(biasGrad.buffer, offset: 0, index: 0)
                    var scaleValue = scale
                    encoder.setBytes(&scaleValue, length: 4, index: 1)
                    var count = UInt32(biasGrad.count)
                    encoder.setBytes(&count, length: 4, index: 2)
                    
                    let threads = MTLSize(width: (biasGrad.count + 255) / 256, height: 1, depth: 1)
                    let threadgroup = MTLSize(width: 256, height: 1, depth: 1)
                    encoder.dispatchThreadgroups(threads, threadsPerThreadgroup: threadgroup)
                    encoder.endEncoding()
                }
            } catch {
                // Fallback to CPU scaling
                let ptr = biasGrad.buffer.contents().bindMemory(to: Float.self, capacity: biasGrad.count)
                for i in 0..<biasGrad.count {
                    ptr[i] *= scale
                }
            }
        }
    }
    
    public func updateParametersWithOptimizer(_ optimizer: any Optimizer) async throws {
        let paramStore = parameterStore
        let lr = await optimizer.getCurrentLearningRate()
        
        // Update weights using gradients
        if let weightsGrad = await paramStore.getGradient(name: weightsGradName) {
            try await paramStore.updateParameter(name: weightsName, with: weightsGrad, learningRate: lr)
        }
        
        // Update bias using gradients
        if let biasGrad = await paramStore.getGradient(name: biasGradName) {
            try await paramStore.updateParameter(name: biasName, with: biasGrad, learningRate: lr)
        }
    }
    
    // MARK: - Weight Loading
    
    /// Load pre-trained weights into the layer
    public func loadWeights(weights: [Float], bias: [Float]) async throws {
        // Validate dimensions
        guard weights.count == inputSize * outputSize else {
            throw VectorStoreError.dimensionMismatch(
                expected: inputSize * outputSize,
                actual: weights.count
            )
        }
        
        guard bias.count == outputSize else {
            throw VectorStoreError.dimensionMismatch(
                expected: outputSize,
                actual: bias.count
            )
        }
        
        let paramStore = parameterStore
        
        // Get weight buffer
        guard let weightsBuffer = await paramStore.getParameter(name: weightsName) else {
            throw VectorStoreError(
                category: .initialization,
                code: .resourcesUnavailable,
                message: "Weight buffer not initialized"
            )
        }
        
        // Get bias buffer  
        guard let biasBuffer = await paramStore.getParameter(name: biasName) else {
            throw VectorStoreError(
                category: .initialization,
                code: .resourcesUnavailable,
                message: "Bias buffer not initialized"
            )
        }
        
        // Copy weights
        let weightsPtr = weightsBuffer.buffer.contents().bindMemory(
            to: Float.self,
            capacity: weightsBuffer.count
        )
        for i in 0..<weights.count {
            weightsPtr[i] = weights[i]
        }
        
        // Copy bias
        let biasPtr = biasBuffer.buffer.contents().bindMemory(
            to: Float.self,
            capacity: biasBuffer.count
        )
        for i in 0..<bias.count {
            biasPtr[i] = bias[i]
        }
    }
}

/// Dropout layer for regularization
public actor DropoutLayer: NeuralLayer {
    public let rate: Float
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
        let pipeline = try await shaderLibrary.pipeline(for: "dropout_forward")
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
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
        }
        
        return output
    }
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let mask = self.mask else {
            return gradOutput // No mask means no dropout was applied
        }
        
        // Apply mask to gradients using Metal shader
        let gradInput = try await allocateBuffer(size: gradOutput.count)
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try await shaderLibrary.pipeline(for: "dropout_backward")
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
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
        }
        
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
    
    public func setTraining(_ training: Bool) {
        self.isTraining = training
    }
    
    public func zeroGradients() async {
        // Dropout doesn't have gradients to zero
    }
    
    public func scaleGradients(_ scale: Float) async {
        // Dropout doesn't have gradients to scale
    }
    
    public func updateParametersWithOptimizer(_ optimizer: any Optimizer) async throws {
        // Dropout has no parameters to update
    }
    
    private func allocateBuffer(size: Int) async throws -> MetalBuffer {
        try await metalPipeline.allocateBuffer(size: size)
    }
}


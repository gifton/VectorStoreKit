// VectorStoreKit: Normalization Layers
//
// Layer normalization and other normalization techniques

import Foundation
@preconcurrency import Metal

/// Layer Normalization
public actor LayerNormLayer: NeuralLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let parameterStore: ParameterStore
    private let operations: MetalMLOperations
    
    // Configuration
    private let normalizedShape: [Int]
    private let eps: Float
    private let elementwiseAffine: Bool
    
    // Parameters
    private let gammaName: String
    private let betaName: String
    private let gammaGradName: String
    private let betaGradName: String
    
    // Cached values for backward pass
    private var lastInput: MetalBuffer?
    private var lastMean: MetalBuffer?
    private var lastVar: MetalBuffer?
    private var lastNormalized: MetalBuffer?
    private var isTraining: Bool = true
    
    // MARK: - Initialization
    
    public init(
        normalizedShape: [Int],
        eps: Float = 1e-5,
        elementwiseAffine: Bool = true,
        name: String = "layernorm",
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.normalizedShape = normalizedShape
        self.eps = eps
        self.elementwiseAffine = elementwiseAffine
        
        self.gammaName = "\(name)_gamma"
        self.betaName = "\(name)_beta"
        self.gammaGradName = "\(name)_gamma_grad"
        self.betaGradName = "\(name)_beta_grad"
        
        self.metalPipeline = metalPipeline
        self.parameterStore = await ParameterStore(device: metalPipeline.device)
        self.operations = await metalPipeline.getOperations()
        
        // Initialize parameters
        if elementwiseAffine {
            try await initializeParameters()
        }
    }
    
    private func initializeParameters() async throws {
        let paramSize = normalizedShape.reduce(1, *)
        
        // Allocate gamma (scale) and beta (shift)
        let gamma = try await parameterStore.allocateParameter(
            name: gammaName,
            size: paramSize
        )
        let beta = try await parameterStore.allocateParameter(
            name: betaName,
            size: paramSize
        )
        
        // Initialize gamma to 1 and beta to 0
        let gammaPtr = gamma.buffer.contents().bindMemory(to: Float.self, capacity: paramSize)
        let betaPtr = beta.buffer.contents().bindMemory(to: Float.self, capacity: paramSize)
        
        for i in 0..<paramSize {
            gammaPtr[i] = 1.0
            betaPtr[i] = 0.0
        }
        
        // Allocate gradient buffers
        _ = try await parameterStore.allocateGradient(name: gammaGradName, size: paramSize)
        _ = try await parameterStore.allocateGradient(name: betaGradName, size: paramSize)
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Save input for backward pass
        if isTraining {
            lastInput = input
        }
        
        let normalizedSize = normalizedShape.reduce(1, *)
        let numGroups = input.count / normalizedSize
        
        // Allocate buffers for statistics
        let mean = try await metalPipeline.allocateBuffer(size: numGroups)
        let variance = try await metalPipeline.allocateBuffer(size: numGroups)
        let output = try await metalPipeline.allocateBuffer(size: input.count)
        
        // Use Metal kernel for layer normalization (computes stats and normalizes in one pass)
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.NormalizationFunction.layerNormForward.rawValue)
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        // Get gamma and beta before the async block if using affine
        let gamma = elementwiseAffine ? await parameterStore.getParameter(name: gammaName) : nil
        let beta = elementwiseAffine ? await parameterStore.getParameter(name: betaName) : nil
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBuffer(mean.buffer, offset: 0, index: 2)
            encoder.setBuffer(variance.buffer, offset: 0, index: 3)
            
            // Set gamma and beta buffers if using affine
            if elementwiseAffine, let gamma = gamma, let beta = beta {
                encoder.setBuffer(gamma.buffer, offset: 0, index: 4)
                encoder.setBuffer(beta.buffer, offset: 0, index: 5)
            }
            
            var normalizedSizeVal = UInt32(normalizedSize)
            var numGroupsVal = UInt32(numGroups)
            var epsVal = eps
            var hasAffineVal = UInt32(elementwiseAffine ? 1 : 0)
            
            encoder.setBytes(&normalizedSizeVal, length: 4, index: 6)
            encoder.setBytes(&numGroupsVal, length: 4, index: 7)
            encoder.setBytes(&epsVal, length: 4, index: 8)
            encoder.setBytes(&hasAffineVal, length: 4, index: 9)
            
            // Each thread group processes one normalization group
            let threadsPerThreadgroup = MTLSize(width: min(normalizedSize, 256), height: 1, depth: 1)
            let threadgroups = MTLSize(width: numGroups, height: 1, depth: 1)
            
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
        
        // Save statistics for backward pass
        if isTraining {
            lastMean = mean
            lastVar = variance
            lastNormalized = output  // We don't have a separate normalized buffer now
        }
        
        return output
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let input = lastInput,
              let mean = lastMean,
              let variance = lastVar,
              let normalized = lastNormalized else {
            throw MetalMLError.parameterNotFound(name: "cached normalization values")
        }
        
        // Backward through affine transformation is handled in the backward kernel
        let gradNormalized = gradOutput
        
        // Backward through normalization
        let gradInput = try await backwardNormalize(
            gradNormalized: gradNormalized,
            input: input,
            mean: mean,
            variance: variance
        )
        
        return gradInput
    }
    
    // MARK: - Parameter Updates
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        guard elementwiseAffine else { return }
        
        if let gammaGrad = await parameterStore.getGradient(name: gammaGradName) {
            try await parameterStore.updateParameter(
                name: gammaName,
                with: gammaGrad,
                learningRate: learningRate
            )
        }
        
        if let betaGrad = await parameterStore.getGradient(name: betaGradName) {
            try await parameterStore.updateParameter(
                name: betaName,
                with: betaGrad,
                learningRate: learningRate
            )
        }
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func getParameters() async -> MetalBuffer? {
        guard elementwiseAffine else { return nil }
        return await parameterStore.getParameter(name: gammaName)
    }
    
    public func getParameterCount() async -> Int {
        guard elementwiseAffine else { return 0 }
        let paramSize = normalizedShape.reduce(1, *)
        return paramSize * 2  // gamma and beta
    }
    
    nonisolated public func setTraining(_ training: Bool) {
        Task { @MainActor in
            await self.updateTrainingMode(training)
        }
    }
    
    private func updateTrainingMode(_ training: Bool) async {
        self.isTraining = training
    }
    
    public func zeroGradients() async {
        // Zero out gradient buffers if elementwise affine is enabled
        guard elementwiseAffine else { return }
        
        let paramSize = normalizedShape.reduce(1, *)
        
        if let gammaGrad = await parameterStore.getGradient(name: gammaGradName) {
            let gradPtr = gammaGrad.buffer.contents().bindMemory(to: Float.self, capacity: paramSize)
            for i in 0..<paramSize {
                gradPtr[i] = 0.0
            }
        }
        
        if let betaGrad = await parameterStore.getGradient(name: betaGradName) {
            let gradPtr = betaGrad.buffer.contents().bindMemory(to: Float.self, capacity: paramSize)
            for i in 0..<paramSize {
                gradPtr[i] = 0.0
            }
        }
    }
    
    public func scaleGradients(_ scale: Float) async {
        // Scale gradient buffers if elementwise affine is enabled
        guard elementwiseAffine else { return }
        
        let paramSize = normalizedShape.reduce(1, *)
        
        if let gammaGrad = await parameterStore.getGradient(name: gammaGradName) {
            let gradPtr = gammaGrad.buffer.contents().bindMemory(to: Float.self, capacity: paramSize)
            for i in 0..<paramSize {
                gradPtr[i] *= scale
            }
        }
        
        if let betaGrad = await parameterStore.getGradient(name: betaGradName) {
            let gradPtr = betaGrad.buffer.contents().bindMemory(to: Float.self, capacity: paramSize)
            for i in 0..<paramSize {
                gradPtr[i] *= scale
            }
        }
    }
    
    public func updateParametersWithOptimizer(_ optimizer: any Optimizer) async throws {
        // Use the default implementation for now
        if let params = await getParameters() {
            try await updateParameters(params, learningRate: await optimizer.getCurrentLearningRate())
        }
    }
    
    // MARK: - Private Methods
    
    private func backwardNormalize(
        gradNormalized: MetalBuffer,
        input: MetalBuffer,
        mean: MetalBuffer,
        variance: MetalBuffer
    ) async throws -> MetalBuffer {
        // Compute gradients through layer normalization
        let gradInput = try await metalPipeline.allocateBuffer(size: input.count)
        
        // Use Metal shader for efficient computation
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.NormalizationFunction.layerNormBackward.rawValue)
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        // Get parameter gradient buffers if using affine
        let gamma = elementwiseAffine ? await parameterStore.getParameter(name: gammaName) : nil
        let gammaGrad = elementwiseAffine ? await parameterStore.getGradient(name: gammaGradName) : nil
        let betaGrad = elementwiseAffine ? await parameterStore.getGradient(name: betaGradName) : nil
        
        let normalizedSize = normalizedShape.reduce(1, *)
        let numGroups = input.count / normalizedSize
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(gradNormalized.buffer, offset: 0, index: 0)
            encoder.setBuffer(input.buffer, offset: 0, index: 1)
            encoder.setBuffer(mean.buffer, offset: 0, index: 2)
            encoder.setBuffer(variance.buffer, offset: 0, index: 3)
            encoder.setBuffer(gradInput.buffer, offset: 0, index: 4)
            
            // Set parameter gradient buffers if using affine
            if elementwiseAffine,
               let gamma = gamma,
               let gammaGrad = gammaGrad,
               let betaGrad = betaGrad {
                encoder.setBuffer(gamma.buffer, offset: 0, index: 5)
                encoder.setBuffer(gammaGrad.buffer, offset: 0, index: 6)
                encoder.setBuffer(betaGrad.buffer, offset: 0, index: 7)
            }
            
            var normalizedSizeVal = UInt32(normalizedSize)
            var numGroupsVal = UInt32(numGroups)
            var epsVal = eps
            
            encoder.setBytes(&normalizedSizeVal, length: 4, index: 8)
            encoder.setBytes(&numGroupsVal, length: 4, index: 9)
            encoder.setBytes(&epsVal, length: 4, index: 10)
            
            var hasAffineVal = UInt32(elementwiseAffine ? 1 : 0)
            encoder.setBytes(&hasAffineVal, length: 4, index: 11)
            
            // Each thread group processes one normalization group
            let threadsPerThreadgroup = MTLSize(width: min(normalizedSize, 256), height: 1, depth: 1)
            let threadgroups = MTLSize(width: numGroups, height: 1, depth: 1)
            
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
        
        return gradInput
    }
}

// MARK: - Group Normalization

/// Group Normalization layer
public actor GroupNormLayer: NeuralLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let parameterStore: ParameterStore
    private let operations: MetalMLOperations
    
    private let numGroups: Int
    private let numChannels: Int
    private let eps: Float
    private let affine: Bool
    
    // Parameters
    private let gammaName: String
    private let betaName: String
    
    // Cached values
    private var isTraining: Bool = true
    
    // MARK: - Initialization
    
    public init(
        numGroups: Int,
        numChannels: Int,
        eps: Float = 1e-5,
        affine: Bool = true,
        name: String = "groupnorm",
        metalPipeline: MetalMLPipeline
    ) async throws {
        guard numChannels % numGroups == 0 else {
            throw MetalMLError.incompatibleBufferSize(
                expected: numChannels,
                actual: numGroups
            )
        }
        
        self.numGroups = numGroups
        self.numChannels = numChannels
        self.eps = eps
        self.affine = affine
        
        self.gammaName = "\(name)_gamma"
        self.betaName = "\(name)_beta"
        
        self.metalPipeline = metalPipeline
        self.parameterStore = await ParameterStore(device: metalPipeline.device)
        self.operations = await metalPipeline.getOperations()
        
        if affine {
            try await initializeParameters()
        }
    }
    
    private func initializeParameters() async throws {
        // Allocate gamma and beta
        let gamma = try await parameterStore.allocateParameter(
            name: gammaName,
            size: numChannels
        )
        let beta = try await parameterStore.allocateParameter(
            name: betaName,
            size: numChannels
        )
        
        // Initialize
        let gammaPtr = gamma.buffer.contents().bindMemory(to: Float.self, capacity: numChannels)
        let betaPtr = beta.buffer.contents().bindMemory(to: Float.self, capacity: numChannels)
        
        for i in 0..<numChannels {
            gammaPtr[i] = 1.0
            betaPtr[i] = 0.0
        }
        
        // Allocate gradient buffers
        _ = try await parameterStore.allocateGradient(name: "\(gammaName)_grad", size: numChannels)
        _ = try await parameterStore.allocateGradient(name: "\(betaName)_grad", size: numChannels)
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Input shape: [N, C, H, W] or [N, C, *]
        guard input.shape.rank >= 2 else {
            throw MetalMLError.invalidArchitecture("GroupNorm expects at least 2D input")
        }
        
        let batchSize = input.shape.dimensions[0]
        let channels = input.shape.dimensions[1]
        guard channels == numChannels else {
            throw MetalMLError.incompatibleBufferSize(expected: numChannels, actual: channels)
        }
        
        // Calculate spatial size
        let spatialSize = input.count / (batchSize * channels)
        let channelsPerGroup = channels / numGroups
        
        // Allocate output
        let output = try await metalPipeline.allocateBuffer(shape: input.shape)
        
        // Use Metal shader for group normalization
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "groupnorm_forward")
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        // Get affine parameters before async block
        let gamma = affine ? await parameterStore.getParameter(name: gammaName) : nil
        let beta = affine ? await parameterStore.getParameter(name: betaName) : nil
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            
            if affine, let gamma = gamma, let beta = beta {
                encoder.setBuffer(gamma.buffer, offset: 0, index: 2)
                encoder.setBuffer(beta.buffer, offset: 0, index: 3)
            }
            
            var batchSizeVal = UInt32(batchSize)
            var channelsVal = UInt32(channels)
            var spatialSizeVal = UInt32(spatialSize)
            var numGroupsVal = UInt32(numGroups)
            var channelsPerGroupVal = UInt32(channelsPerGroup)
            var epsVal = eps
            var affineVal = UInt32(affine ? 1 : 0)
            
            encoder.setBytes(&batchSizeVal, length: 4, index: 4)
            encoder.setBytes(&channelsVal, length: 4, index: 5)
            encoder.setBytes(&spatialSizeVal, length: 4, index: 6)
            encoder.setBytes(&numGroupsVal, length: 4, index: 7)
            encoder.setBytes(&channelsPerGroupVal, length: 4, index: 8)
            encoder.setBytes(&epsVal, length: 4, index: 9)
            encoder.setBytes(&affineVal, length: 4, index: 10)
            
            // Each thread group processes one group
            let threadsPerThreadgroup = MTLSize(width: min(channelsPerGroup * spatialSize, 256), height: 1, depth: 1)
            let threadgroups = MTLSize(width: batchSize * numGroups, height: 1, depth: 1)
            
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
        
        return output
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        let batchSize = gradOutput.shape.dimensions[0]
        let channels = gradOutput.shape.dimensions[1]
        let spatialSize = gradOutput.count / (batchSize * channels)
        let channelsPerGroup = channels / numGroups
        
        // Allocate gradient input
        let gradInput = try await metalPipeline.allocateBuffer(shape: gradOutput.shape)
        
        // Allocate buffers for parameter gradients if affine
        var gradGamma: MetalBuffer?
        var gradBeta: MetalBuffer?
        if affine {
            gradGamma = await parameterStore.getGradient(name: "\(gammaName)_grad")
            gradBeta = await parameterStore.getGradient(name: "\(betaName)_grad")
        }
        
        // Use Metal shader for backward pass
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "groupnorm_backward")
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        // Get affine parameters before async block
        let gamma = affine ? await parameterStore.getParameter(name: gammaName) : nil
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
            encoder.setBuffer(gradInput.buffer, offset: 0, index: 1)
            
            if affine,
               let gamma = gamma,
               let gGamma = gradGamma,
               let gBeta = gradBeta {
                encoder.setBuffer(gamma.buffer, offset: 0, index: 2)
                encoder.setBuffer(gGamma.buffer, offset: 0, index: 3)
                encoder.setBuffer(gBeta.buffer, offset: 0, index: 4)
            }
            
            var batchSizeVal = UInt32(batchSize)
            var channelsVal = UInt32(channels)
            var spatialSizeVal = UInt32(spatialSize)
            var numGroupsVal = UInt32(numGroups)
            var channelsPerGroupVal = UInt32(channelsPerGroup)
            var epsVal = eps
            var affineVal = UInt32(affine ? 1 : 0)
            
            encoder.setBytes(&batchSizeVal, length: 4, index: 5)
            encoder.setBytes(&channelsVal, length: 4, index: 6)
            encoder.setBytes(&spatialSizeVal, length: 4, index: 7)
            encoder.setBytes(&numGroupsVal, length: 4, index: 8)
            encoder.setBytes(&channelsPerGroupVal, length: 4, index: 9)
            encoder.setBytes(&epsVal, length: 4, index: 10)
            encoder.setBytes(&affineVal, length: 4, index: 11)
            
            // Each thread group processes one group
            let threadsPerThreadgroup = MTLSize(width: min(channelsPerGroup * spatialSize, 256), height: 1, depth: 1)
            let threadgroups = MTLSize(width: batchSize * numGroups, height: 1, depth: 1)
            
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
        
        return gradInput
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        guard affine else { return }
        
        if let gammaGrad = await parameterStore.getGradient(name: "\(gammaName)_grad") {
            try await parameterStore.updateParameter(
                name: gammaName,
                with: gammaGrad,
                learningRate: learningRate
            )
        }
        
        if let betaGrad = await parameterStore.getGradient(name: "\(betaName)_grad") {
            try await parameterStore.updateParameter(
                name: betaName,
                with: betaGrad,
                learningRate: learningRate
            )
        }
    }
    
    public func getParameters() async -> MetalBuffer? {
        guard affine else { return nil }
        return await parameterStore.getParameter(name: gammaName)
    }
    
    public func getParameterCount() async -> Int {
        guard affine else { return 0 }
        return numChannels * 2
    }
    
    nonisolated public func setTraining(_ training: Bool) {
        Task { @MainActor in
            await self.updateTrainingMode(training)
        }
    }
    
    private func updateTrainingMode(_ training: Bool) async {
        self.isTraining = training
    }
    
    public func zeroGradients() async {
        // Zero out gradient buffers if affine transformation is enabled
        guard affine else { return }
        
        if let gammaGrad = await parameterStore.getGradient(name: "\(gammaName)_grad") {
            let gradPtr = gammaGrad.buffer.contents().bindMemory(to: Float.self, capacity: numChannels)
            for i in 0..<numChannels {
                gradPtr[i] = 0.0
            }
        }
        
        if let betaGrad = await parameterStore.getGradient(name: "\(betaName)_grad") {
            let gradPtr = betaGrad.buffer.contents().bindMemory(to: Float.self, capacity: numChannels)
            for i in 0..<numChannels {
                gradPtr[i] = 0.0
            }
        }
    }
    
    public func scaleGradients(_ scale: Float) async {
        // Scale gradient buffers if affine transformation is enabled
        guard affine else { return }
        
        if let gammaGrad = await parameterStore.getGradient(name: "\(gammaName)_grad") {
            let gradPtr = gammaGrad.buffer.contents().bindMemory(to: Float.self, capacity: numChannels)
            for i in 0..<numChannels {
                gradPtr[i] *= scale
            }
        }
        
        if let betaGrad = await parameterStore.getGradient(name: "\(betaName)_grad") {
            let gradPtr = betaGrad.buffer.contents().bindMemory(to: Float.self, capacity: numChannels)
            for i in 0..<numChannels {
                gradPtr[i] *= scale
            }
        }
    }
    
    public func updateParametersWithOptimizer(_ optimizer: any Optimizer) async throws {
        // Use the default implementation for now
        if let params = await getParameters() {
            try await updateParameters(params, learningRate: await optimizer.getCurrentLearningRate())
        }
    }
}

// MARK: - Batch Normalization

/// Batch Normalization layer
public actor BatchNormLayer: NeuralLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let parameterStore: ParameterStore
    private let operations: MetalMLOperations
    
    // Configuration
    private let numFeatures: Int
    private let eps: Float
    private let momentum: Float
    private let affine: Bool
    private let trackRunningStats: Bool
    
    // Parameters
    private let gammaName: String
    private let betaName: String
    private let gammaGradName: String
    private let betaGradName: String
    
    // Running statistics
    private let runningMeanName: String
    private let runningVarName: String
    private var numBatchesTracked: Int = 0
    
    // Cached values for backward pass
    private var lastInput: MetalBuffer?
    private var lastBatchMean: MetalBuffer?
    private var lastBatchVar: MetalBuffer?
    private var lastNormalized: MetalBuffer?
    private var isTraining: Bool = true
    
    // MARK: - Initialization
    
    public init(
        numFeatures: Int,
        eps: Float = 1e-5,
        momentum: Float = 0.1,
        affine: Bool = true,
        trackRunningStats: Bool = true,
        name: String = "batchnorm",
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.numFeatures = numFeatures
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.trackRunningStats = trackRunningStats
        
        self.gammaName = "\(name)_gamma"
        self.betaName = "\(name)_beta"
        self.gammaGradName = "\(name)_gamma_grad"
        self.betaGradName = "\(name)_beta_grad"
        self.runningMeanName = "\(name)_running_mean"
        self.runningVarName = "\(name)_running_var"
        
        self.metalPipeline = metalPipeline
        self.parameterStore = await ParameterStore(device: metalPipeline.device)
        self.operations = await metalPipeline.getOperations()
        
        // Initialize parameters
        try await initializeParameters()
    }
    
    private func initializeParameters() async throws {
        // Initialize affine parameters
        if affine {
            // Allocate gamma (scale) and beta (shift)
            let gamma = try await parameterStore.allocateParameter(
                name: gammaName,
                size: numFeatures
            )
            let beta = try await parameterStore.allocateParameter(
                name: betaName,
                size: numFeatures
            )
            
            // Initialize gamma to 1 and beta to 0
            let gammaPtr = gamma.buffer.contents().bindMemory(to: Float.self, capacity: numFeatures)
            let betaPtr = beta.buffer.contents().bindMemory(to: Float.self, capacity: numFeatures)
            
            for i in 0..<numFeatures {
                gammaPtr[i] = 1.0
                betaPtr[i] = 0.0
            }
            
            // Allocate gradient buffers
            _ = try await parameterStore.allocateGradient(name: gammaGradName, size: numFeatures)
            _ = try await parameterStore.allocateGradient(name: betaGradName, size: numFeatures)
        }
        
        // Initialize running statistics
        if trackRunningStats {
            let runningMean = try await parameterStore.allocateParameter(
                name: runningMeanName,
                size: numFeatures
            )
            let runningVar = try await parameterStore.allocateParameter(
                name: runningVarName,
                size: numFeatures
            )
            
            // Initialize running mean to 0 and variance to 1
            let meanPtr = runningMean.buffer.contents().bindMemory(to: Float.self, capacity: numFeatures)
            let varPtr = runningVar.buffer.contents().bindMemory(to: Float.self, capacity: numFeatures)
            
            for i in 0..<numFeatures {
                meanPtr[i] = 0.0
                varPtr[i] = 1.0
            }
        }
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Input shape should be [batch_size, num_features] or [batch_size, num_features, *]
        guard input.shape.dimensions.count >= 2 else {
            throw MetalMLError.invalidArchitecture("BatchNorm expects at least 2D input")
        }
        
        let batchSize = input.shape.dimensions[0]
        let features = input.shape.dimensions[1]
        guard features == numFeatures else {
            throw MetalMLError.incompatibleBufferSize(expected: numFeatures, actual: features)
        }
        
        // Save input for backward pass
        if isTraining {
            lastInput = input
        }
        
        // Allocate output buffer
        let output = try await metalPipeline.allocateBuffer(shape: input.shape)
        
        // Allocate buffers for batch statistics
        let batchMean = try await metalPipeline.allocateBuffer(size: numFeatures)
        let batchVar = try await metalPipeline.allocateBuffer(size: numFeatures)
        
        // Get parameters
        let gamma = affine ? await parameterStore.getParameter(name: gammaName) : nil
        let beta = affine ? await parameterStore.getParameter(name: betaName) : nil
        let runningMean = trackRunningStats ? await parameterStore.getParameter(name: runningMeanName) : nil
        let runningVar = trackRunningStats ? await parameterStore.getParameter(name: runningVarName) : nil
        
        // Use Metal kernel for batch normalization
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.NormalizationFunction.batchNormForward.rawValue)
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            encoder.setBuffer(batchMean.buffer, offset: 0, index: 2)
            encoder.setBuffer(batchVar.buffer, offset: 0, index: 3)
            
            // Set gamma and beta if using affine
            if affine, let gamma = gamma, let beta = beta {
                encoder.setBuffer(gamma.buffer, offset: 0, index: 4)
                encoder.setBuffer(beta.buffer, offset: 0, index: 5)
            }
            
            // Set running statistics if available
            if trackRunningStats, let runningMean = runningMean, let runningVar = runningVar {
                encoder.setBuffer(runningMean.buffer, offset: 0, index: 6)
                encoder.setBuffer(runningVar.buffer, offset: 0, index: 7)
            }
            
            var batchSizeVal = UInt32(batchSize)
            var numFeaturesVal = UInt32(numFeatures)
            var epsVal = eps
            var isTrainingVal = UInt32(isTraining ? 1 : 0)
            var momentumVal = momentum
            
            encoder.setBytes(&batchSizeVal, length: 4, index: 8)
            encoder.setBytes(&numFeaturesVal, length: 4, index: 9)
            encoder.setBytes(&epsVal, length: 4, index: 10)
            encoder.setBytes(&isTrainingVal, length: 4, index: 11)
            encoder.setBytes(&momentumVal, length: 4, index: 12)
            
            // Thread configuration
            let threadsPerThreadgroup = MTLSize(width: min(256, numFeatures), height: 1, depth: 1)
            let threadgroups = MTLSize(width: numFeatures, height: batchSize, depth: 1)
            
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
        
        // Update running statistics in training mode
        if isTraining && trackRunningStats {
            try await updateRunningStats(batchMean: batchMean, batchVar: batchVar)
        }
        
        // Save statistics for backward pass
        if isTraining {
            lastBatchMean = batchMean
            lastBatchVar = batchVar
            lastNormalized = output
        }
        
        return output
    }
    
    // MARK: - Update Running Statistics
    
    private func updateRunningStats(batchMean: MetalBuffer, batchVar: MetalBuffer) async throws {
        guard let runningMean = await parameterStore.getParameter(name: runningMeanName),
              let runningVar = await parameterStore.getParameter(name: runningVarName) else {
            return
        }
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.NormalizationFunction.batchNormUpdateRunningStats.rawValue)
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(batchMean.buffer, offset: 0, index: 0)
            encoder.setBuffer(batchVar.buffer, offset: 0, index: 1)
            encoder.setBuffer(runningMean.buffer, offset: 0, index: 2)
            encoder.setBuffer(runningVar.buffer, offset: 0, index: 3)
            
            var momentumVal = momentum
            var numFeaturesVal = UInt32(numFeatures)
            
            encoder.setBytes(&momentumVal, length: 4, index: 4)
            encoder.setBytes(&numFeaturesVal, length: 4, index: 5)
            
            let threadsPerThreadgroup = MTLSize(width: min(256, numFeatures), height: 1, depth: 1)
            let threadgroups = MTLSize(width: (numFeatures + 255) / 256, height: 1, depth: 1)
            
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
        
        numBatchesTracked += 1
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let input = lastInput,
              let batchMean = lastBatchMean,
              let batchVar = lastBatchVar else {
            throw MetalMLError.parameterNotFound(name: "cached batch normalization values")
        }
        
        let batchSize = input.shape.dimensions[0]
        
        // Allocate gradient input buffer
        let gradInput = try await metalPipeline.allocateBuffer(shape: input.shape)
        
        // Get parameters and gradients
        let gamma = affine ? await parameterStore.getParameter(name: gammaName) : nil
        let gradGamma = affine ? await parameterStore.getGradient(name: gammaGradName) : nil
        let gradBeta = affine ? await parameterStore.getGradient(name: betaGradName) : nil
        
        // Use Metal kernel for backward pass
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.NormalizationFunction.batchNormBackward.rawValue)
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
            encoder.setBuffer(input.buffer, offset: 0, index: 1)
            encoder.setBuffer(batchMean.buffer, offset: 0, index: 2)
            encoder.setBuffer(batchVar.buffer, offset: 0, index: 3)
            encoder.setBuffer(gradInput.buffer, offset: 0, index: 4)
            
            if affine, let gamma = gamma, let gradGamma = gradGamma, let gradBeta = gradBeta {
                encoder.setBuffer(gamma.buffer, offset: 0, index: 5)
                encoder.setBuffer(gradGamma.buffer, offset: 0, index: 6)
                encoder.setBuffer(gradBeta.buffer, offset: 0, index: 7)
            }
            
            var batchSizeVal = UInt32(batchSize)
            var numFeaturesVal = UInt32(numFeatures)
            var epsVal = eps
            
            encoder.setBytes(&batchSizeVal, length: 4, index: 8)
            encoder.setBytes(&numFeaturesVal, length: 4, index: 9)
            encoder.setBytes(&epsVal, length: 4, index: 10)
            
            // Thread configuration
            let threadsPerThreadgroup = MTLSize(width: min(256, numFeatures), height: 1, depth: 1)
            let threadgroups = MTLSize(width: numFeatures, height: batchSize, depth: 1)
            
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
        
        return gradInput
    }
    
    // MARK: - Parameter Updates
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        guard affine else { return }
        
        if let gammaGrad = await parameterStore.getGradient(name: gammaGradName) {
            try await parameterStore.updateParameter(
                name: gammaName,
                with: gammaGrad,
                learningRate: learningRate
            )
        }
        
        if let betaGrad = await parameterStore.getGradient(name: betaGradName) {
            try await parameterStore.updateParameter(
                name: betaName,
                with: betaGrad,
                learningRate: learningRate
            )
        }
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func getParameters() async -> MetalBuffer? {
        guard affine else { return nil }
        return await parameterStore.getParameter(name: gammaName)
    }
    
    public func getParameterCount() async -> Int {
        var count = 0
        if affine {
            count += numFeatures * 2  // gamma and beta
        }
        if trackRunningStats {
            count += numFeatures * 2  // running mean and variance (not trainable)
        }
        return count
    }
    
    nonisolated public func setTraining(_ training: Bool) {
        Task { @MainActor in
            await self.updateTrainingMode(training)
        }
    }
    
    private func updateTrainingMode(_ training: Bool) async {
        self.isTraining = training
    }
    
    public func zeroGradients() async {
        // Zero out gradient buffers if affine transformation is enabled
        guard affine else { return }
        
        if let gammaGrad = await parameterStore.getGradient(name: gammaGradName) {
            let gradPtr = gammaGrad.buffer.contents().bindMemory(to: Float.self, capacity: numFeatures)
            for i in 0..<numFeatures {
                gradPtr[i] = 0.0
            }
        }
        
        if let betaGrad = await parameterStore.getGradient(name: betaGradName) {
            let gradPtr = betaGrad.buffer.contents().bindMemory(to: Float.self, capacity: numFeatures)
            for i in 0..<numFeatures {
                gradPtr[i] = 0.0
            }
        }
    }
    
    public func scaleGradients(_ scale: Float) async {
        // Scale gradient buffers if affine transformation is enabled
        guard affine else { return }
        
        if let gammaGrad = await parameterStore.getGradient(name: gammaGradName) {
            let gradPtr = gammaGrad.buffer.contents().bindMemory(to: Float.self, capacity: numFeatures)
            for i in 0..<numFeatures {
                gradPtr[i] *= scale
            }
        }
        
        if let betaGrad = await parameterStore.getGradient(name: betaGradName) {
            let gradPtr = betaGrad.buffer.contents().bindMemory(to: Float.self, capacity: numFeatures)
            for i in 0..<numFeatures {
                gradPtr[i] *= scale
            }
        }
    }
    
    public func updateParametersWithOptimizer(_ optimizer: any Optimizer) async throws {
        // Use the default implementation for now
        if let params = await getParameters() {
            try await updateParameters(params, learningRate: await optimizer.getCurrentLearningRate())
        }
    }
}
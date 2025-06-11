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
        
        // Calculate normalization statistics
        let (mean, variance) = try await calculateStatistics(input)
        
        // Save statistics for backward pass
        if isTraining {
            lastMean = mean
            lastVar = variance
        }
        
        // Normalize input
        let normalized = try await normalize(
            input: input,
            mean: mean,
            variance: variance
        )
        
        // Save normalized values for backward pass
        if isTraining {
            lastNormalized = normalized
        }
        
        // Apply affine transformation if enabled
        let output: MetalBuffer
        if elementwiseAffine {
            output = try await applyAffine(normalized)
        } else {
            output = normalized
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
        
        var gradNormalized = gradOutput
        
        // Backward through affine transformation
        if elementwiseAffine {
            gradNormalized = try await backwardAffine(
                gradOutput: gradOutput,
                normalized: normalized
            )
        }
        
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
    
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
    
    // MARK: - Private Methods
    
    private func calculateStatistics(_ input: MetalBuffer) async throws -> (mean: MetalBuffer, variance: MetalBuffer) {
        // Calculate mean and variance across normalized dimensions
        // For now, simplified implementation
        let statSize = input.count / normalizedShape.reduce(1, *)
        
        let mean = try await metalPipeline.allocateBuffer(size: statSize)
        let variance = try await metalPipeline.allocateBuffer(size: statSize)
        
        // Placeholder - would use Metal kernel for parallel reduction
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let meanPtr = mean.buffer.contents().bindMemory(to: Float.self, capacity: statSize)
        let varPtr = variance.buffer.contents().bindMemory(to: Float.self, capacity: statSize)
        
        let normalizedSize = normalizedShape.reduce(1, *)
        
        for i in 0..<statSize {
            var sum: Float = 0
            let offset = i * normalizedSize
            
            // Calculate mean
            for j in 0..<normalizedSize {
                sum += inputPtr[offset + j]
            }
            meanPtr[i] = sum / Float(normalizedSize)
            
            // Calculate variance
            sum = 0
            for j in 0..<normalizedSize {
                let diff = inputPtr[offset + j] - meanPtr[i]
                sum += diff * diff
            }
            varPtr[i] = sum / Float(normalizedSize)
        }
        
        return (mean, variance)
    }
    
    private func normalize(
        input: MetalBuffer,
        mean: MetalBuffer,
        variance: MetalBuffer
    ) async throws -> MetalBuffer {
        let output = try await metalPipeline.allocateBuffer(size: input.count)
        
        // Placeholder - would use Metal kernel
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        let meanPtr = mean.buffer.contents().bindMemory(to: Float.self, capacity: mean.count)
        let varPtr = variance.buffer.contents().bindMemory(to: Float.self, capacity: variance.count)
        
        let normalizedSize = normalizedShape.reduce(1, *)
        let statSize = mean.count
        
        for i in 0..<statSize {
            let offset = i * normalizedSize
            let m = meanPtr[i]
            let v = varPtr[i]
            let scale = 1.0 / sqrt(v + eps)
            
            for j in 0..<normalizedSize {
                let idx = offset + j
                outputPtr[idx] = (inputPtr[idx] - m) * scale
            }
        }
        
        return output
    }
    
    private func applyAffine(_ normalized: MetalBuffer) async throws -> MetalBuffer {
        guard let gamma = await parameterStore.getParameter(name: gammaName),
              let beta = await parameterStore.getParameter(name: betaName) else {
            return normalized
        }
        
        let output = try await metalPipeline.allocateBuffer(size: normalized.count)
        
        // Placeholder - would use Metal kernel
        let normalizedPtr = normalized.buffer.contents().bindMemory(to: Float.self, capacity: normalized.count)
        let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        let gammaPtr = gamma.buffer.contents().bindMemory(to: Float.self, capacity: gamma.count)
        let betaPtr = beta.buffer.contents().bindMemory(to: Float.self, capacity: beta.count)
        
        let normalizedSize = normalizedShape.reduce(1, *)
        let numGroups = normalized.count / normalizedSize
        
        for i in 0..<numGroups {
            let offset = i * normalizedSize
            for j in 0..<normalizedSize {
                let idx = offset + j
                outputPtr[idx] = normalizedPtr[idx] * gammaPtr[j] + betaPtr[j]
            }
        }
        
        return output
    }
    
    private func backwardAffine(
        gradOutput: MetalBuffer,
        normalized: MetalBuffer
    ) async throws -> MetalBuffer {
        guard let gammaGrad = await parameterStore.getGradient(name: gammaGradName),
              let betaGrad = await parameterStore.getGradient(name: betaGradName),
              let gamma = await parameterStore.getParameter(name: gammaName) else {
            return gradOutput
        }
        
        // Compute parameter gradients
        // gradGamma = sum(gradOutput * normalized, axis=0)
        // gradBeta = sum(gradOutput, axis=0)
        
        let gradOutputPtr = gradOutput.buffer.contents().bindMemory(to: Float.self, capacity: gradOutput.count)
        let normalizedPtr = normalized.buffer.contents().bindMemory(to: Float.self, capacity: normalized.count)
        let gammaGradPtr = gammaGrad.buffer.contents().bindMemory(to: Float.self, capacity: gammaGrad.count)
        let betaGradPtr = betaGrad.buffer.contents().bindMemory(to: Float.self, capacity: betaGrad.count)
        
        let normalizedSize = normalizedShape.reduce(1, *)
        let numGroups = gradOutput.count / normalizedSize
        
        // Zero out gradients
        for j in 0..<normalizedSize {
            gammaGradPtr[j] = 0
            betaGradPtr[j] = 0
        }
        
        // Accumulate gradients
        for i in 0..<numGroups {
            let offset = i * normalizedSize
            for j in 0..<normalizedSize {
                let idx = offset + j
                gammaGradPtr[j] += gradOutputPtr[idx] * normalizedPtr[idx]
                betaGradPtr[j] += gradOutputPtr[idx]
            }
        }
        
        // Compute gradient with respect to normalized input
        let gradNormalized = try await metalPipeline.allocateBuffer(size: gradOutput.count)
        let gradNormalizedPtr = gradNormalized.buffer.contents().bindMemory(to: Float.self, capacity: gradNormalized.count)
        let gammaPtr = gamma.buffer.contents().bindMemory(to: Float.self, capacity: gamma.count)
        
        for i in 0..<numGroups {
            let offset = i * normalizedSize
            for j in 0..<normalizedSize {
                let idx = offset + j
                gradNormalizedPtr[idx] = gradOutputPtr[idx] * gammaPtr[j]
            }
        }
        
        return gradNormalized
    }
    
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
        let pipeline = try shaderLibrary.pipeline(for: "layernorm_backward")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradNormalized.buffer, offset: 0, index: 0)
        encoder.setBuffer(input.buffer, offset: 0, index: 1)
        encoder.setBuffer(mean.buffer, offset: 0, index: 2)
        encoder.setBuffer(variance.buffer, offset: 0, index: 3)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 4)
        
        let normalizedSize = normalizedShape.reduce(1, *)
        let numGroups = input.count / normalizedSize
        
        var normalizedSizeVal = UInt32(normalizedSize)
        var numGroupsVal = UInt32(numGroups)
        var epsVal = eps
        
        encoder.setBytes(&normalizedSizeVal, length: 4, index: 5)
        encoder.setBytes(&numGroupsVal, length: 4, index: 6)
        encoder.setBytes(&epsVal, length: 4, index: 7)
        
        // Each thread group processes one normalization group
        let threadsPerThreadgroup = MTLSize(width: min(normalizedSize, 256), height: 1, depth: 1)
        let threadgroups = MTLSize(width: numGroups, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
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
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        
        if affine {
            if let gamma = await parameterStore.getParameter(name: gammaName),
               let beta = await parameterStore.getParameter(name: betaName) {
                encoder.setBuffer(gamma.buffer, offset: 0, index: 2)
                encoder.setBuffer(beta.buffer, offset: 0, index: 3)
            }
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
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
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
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 1)
        
        if affine {
            if let gamma = await parameterStore.getParameter(name: gammaName),
               let gGamma = gradGamma,
               let gBeta = gradBeta {
                encoder.setBuffer(gamma.buffer, offset: 0, index: 2)
                encoder.setBuffer(gGamma.buffer, offset: 0, index: 3)
                encoder.setBuffer(gBeta.buffer, offset: 0, index: 4)
            }
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
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
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
    
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
}
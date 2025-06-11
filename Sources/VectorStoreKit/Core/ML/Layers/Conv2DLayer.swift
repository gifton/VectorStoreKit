// VectorStoreKit: Conv2D Layer
//
// Convolutional layer with Metal acceleration

import Foundation
@preconcurrency import Metal

/// 2D Convolutional layer with Metal acceleration
public actor Conv2DLayer: NeuralLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let parameterStore: ParameterStore
    private let operations: MetalMLOperations
    
    // Layer configuration
    private let inputChannels: Int
    private let outputChannels: Int
    private let kernelSize: (height: Int, width: Int)
    private let stride: (height: Int, width: Int)
    private let padding: Padding
    private let dilation: (height: Int, width: Int)
    private let groups: Int
    private let activation: Activation
    private let useBias: Bool
    
    // Parameter names
    private let weightsName: String
    private let biasName: String
    private let weightsGradName: String
    private let biasGradName: String
    
    // Cached values for backward pass
    private var lastInput: MetalBuffer?
    private var lastOutput: MetalBuffer?
    private var lastPreActivation: MetalBuffer?
    private var isTraining: Bool = true
    
    /// Padding mode for convolution
    public enum Padding: Sendable {
        case valid      // No padding
        case same       // Pad to keep output size same as input
        case custom(height: Int, width: Int)  // Custom padding
        
        func calculate(inputSize: (height: Int, width: Int), 
                      kernelSize: (height: Int, width: Int),
                      stride: (height: Int, width: Int),
                      dilation: (height: Int, width: Int)) -> (height: Int, width: Int) {
            switch self {
            case .valid:
                return (0, 0)
            case .same:
                let outputHeight = (inputSize.height + stride.height - 1) / stride.height
                let outputWidth = (inputSize.width + stride.width - 1) / stride.width
                let padHeight = max(0, (outputHeight - 1) * stride.height + (kernelSize.height - 1) * dilation.height + 1 - inputSize.height)
                let padWidth = max(0, (outputWidth - 1) * stride.width + (kernelSize.width - 1) * dilation.width + 1 - inputSize.width)
                return (padHeight / 2, padWidth / 2)
            case .custom(let h, let w):
                return (h, w)
            }
        }
    }
    
    // MARK: - Initialization
    
    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: (height: Int, width: Int),
        stride: (height: Int, width: Int) = (1, 1),
        padding: Padding = .valid,
        dilation: (height: Int, width: Int) = (1, 1),
        groups: Int = 1,
        activation: Activation = .linear,
        useBias: Bool = true,
        name: String = "conv2d",
        metalPipeline: MetalMLPipeline
    ) async throws {
        // Validate parameters
        guard inputChannels % groups == 0 && outputChannels % groups == 0 else {
            throw MetalMLError.incompatibleBufferSize(
                expected: inputChannels,
                actual: groups
            )
        }
        
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.activation = activation
        self.useBias = useBias
        
        // Initialize parameter names
        self.weightsName = "\(name)_weights"
        self.biasName = "\(name)_bias"
        self.weightsGradName = "\(name)_weights_grad"
        self.biasGradName = "\(name)_bias_grad"
        
        // Initialize pipeline and parameter store
        self.metalPipeline = metalPipeline
        self.parameterStore = await ParameterStore(device: metalPipeline.device)
        self.operations = await metalPipeline.getOperations()
        
        // Initialize parameters
        try await initializeParameters()
    }
    
    private func initializeParameters() async throws {
        // Weight shape: [outputChannels, inputChannels/groups, kernelHeight, kernelWidth]
        let channelsPerGroup = inputChannels / groups
        let weightsSize = outputChannels * channelsPerGroup * kernelSize.height * kernelSize.width
        
        // Allocate weight buffer
        let weightsBuffer = try await parameterStore.allocateParameter(
            name: weightsName,
            size: weightsSize
        )
        
        // Initialize weights using He initialization
        let scale = sqrt(2.0 / Float(channelsPerGroup * kernelSize.height * kernelSize.width))
        let weightsPtr = weightsBuffer.buffer.contents().bindMemory(to: Float.self, capacity: weightsSize)
        for i in 0..<weightsSize {
            weightsPtr[i] = Float.random(in: -scale...scale)
        }
        
        // Allocate and initialize bias if needed
        if useBias {
            let biasBuffer = try await parameterStore.allocateParameter(
                name: biasName,
                size: outputChannels
            )
            
            // Initialize bias to zero
            let biasPtr = biasBuffer.buffer.contents().bindMemory(to: Float.self, capacity: outputChannels)
            for i in 0..<outputChannels {
                biasPtr[i] = 0
            }
        }
        
        // Allocate gradient buffers
        _ = try await parameterStore.allocateGradient(name: weightsGradName, size: weightsSize)
        if useBias {
            _ = try await parameterStore.allocateGradient(name: biasGradName, size: outputChannels)
        }
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Validate input shape: [batch, channels, height, width]
        guard input.shape.rank == 4 else {
            throw MetalMLError.invalidArchitecture("Conv2D expects 4D input (batch, channels, height, width)")
        }
        
        let inputShape = input.shape.dimensions
        guard inputShape[1] == inputChannels else {
            throw MetalMLError.incompatibleBufferSize(
                expected: inputChannels,
                actual: inputShape[1]
            )
        }
        
        // Save input for backward pass
        if isTraining {
            lastInput = input
        }
        
        // Get weights
        guard let weights = await parameterStore.getParameter(name: weightsName) else {
            throw MetalMLError.parameterNotFound(name: weightsName)
        }
        
        // Calculate output shape
        let outputShape = calculateOutputShape(inputShape: inputShape)
        
        // Allocate output buffer with shape
        let output = try await metalPipeline.allocateBuffer(shape: outputShape)
        
        // Perform convolution (placeholder - would use Metal kernel)
        try await performConvolution(
            input: input,
            weights: weights,
            output: output
        )
        
        // Add bias if needed
        if useBias, let bias = await parameterStore.getParameter(name: biasName) {
            try await addBias(output: output, bias: bias)
        }
        
        // Save pre-activation for backward pass
        if isTraining {
            lastPreActivation = output
        }
        
        // Apply activation
        let activated = if activation != .linear {
            try await applyActivation(output)
        } else {
            output
        }
        
        // Save output for backward pass
        if isTraining {
            lastOutput = activated
        }
        
        return activated
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let input = lastInput,
              let weights = await parameterStore.getParameter(name: weightsName) else {
            throw MetalMLError.parameterNotFound(name: "cached values")
        }
        
        var currentGrad = gradOutput
        
        // Backpropagate through activation
        if activation != .linear, let preActivation = lastPreActivation {
            currentGrad = try await backwardActivation(
                gradOutput: currentGrad,
                preActivation: preActivation
            )
        }
        
        // Compute weight gradients
        if let weightsGrad = await parameterStore.getGradient(name: weightsGradName) {
            try await computeWeightGradients(
                input: input,
                gradOutput: currentGrad,
                weightsGrad: weightsGrad
            )
        }
        
        // Compute bias gradients
        if useBias, let biasGrad = await parameterStore.getGradient(name: biasGradName) {
            try await computeBiasGradients(
                gradOutput: currentGrad,
                biasGrad: biasGrad
            )
        }
        
        // Compute input gradients
        let gradInput = try await computeInputGradients(
            weights: weights,
            gradOutput: currentGrad,
            inputShape: getShape(of: input)
        )
        
        return gradInput
    }
    
    // MARK: - Parameter Updates
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        // Update weights
        if let weightsGrad = await parameterStore.getGradient(name: weightsGradName) {
            try await parameterStore.updateParameter(
                name: weightsName,
                with: weightsGrad,
                learningRate: learningRate
            )
        }
        
        // Update bias
        if useBias, let biasGrad = await parameterStore.getGradient(name: biasGradName) {
            try await parameterStore.updateParameter(
                name: biasName,
                with: biasGrad,
                learningRate: learningRate
            )
        }
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func getParameters() async -> MetalBuffer? {
        await parameterStore.getParameter(name: weightsName)
    }
    
    public func getParameterCount() async -> Int {
        let channelsPerGroup = inputChannels / groups
        let weightsCount = outputChannels * channelsPerGroup * kernelSize.height * kernelSize.width
        let biasCount = useBias ? outputChannels : 0
        return weightsCount + biasCount
    }
    
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
    
    // MARK: - Private Methods
    
    private func calculateOutputShape(inputShape: [Int]) -> TensorShape {
        // Input shape: [batch, channels, height, width]
        let batchSize = inputShape[0]
        let inputHeight = inputShape[2]
        let inputWidth = inputShape[3]
        
        let dilatedKernelHeight = kernelSize.height + (kernelSize.height - 1) * (dilation.height - 1)
        let dilatedKernelWidth = kernelSize.width + (kernelSize.width - 1) * (dilation.width - 1)
        
        // Calculate output dimensions based on convolution parameters
        let outputHeight: Int
        let outputWidth: Int
        
        switch padding {
        case .valid:
            outputHeight = (inputHeight - dilatedKernelHeight) / stride.height + 1
            outputWidth = (inputWidth - dilatedKernelWidth) / stride.width + 1
            
        case .same:
            outputHeight = (inputHeight + stride.height - 1) / stride.height
            outputWidth = (inputWidth + stride.width - 1) / stride.width
            
        case .custom(let h, let w):
            outputHeight = (inputHeight + 2 * h - dilatedKernelHeight) / stride.height + 1
            outputWidth = (inputWidth + 2 * w - dilatedKernelWidth) / stride.width + 1
        }
        
        // Return shape: [batch, outputChannels, outputHeight, outputWidth]
        return TensorShape(batchSize, outputChannels, outputHeight, outputWidth)
    }
    
    private func getShape(of buffer: MetalBuffer) -> (batch: Int, channels: Int, height: Int, width: Int) {
        // For now, assume square images and derive dimensions from buffer size
        // In practice, this would be stored as metadata with the buffer
        // Determine channels based on buffer size
        // If buffer size matches input size, it's likely an input buffer
        let channels = inputChannels
        let elementsPerChannel = buffer.count / channels
        let spatialSize = Int(sqrt(Double(elementsPerChannel)))
        
        // Check if this is a single sample or a batch
        if buffer.count == channels * spatialSize * spatialSize {
            return (1, channels, spatialSize, spatialSize)
        } else {
            // Try to infer batch size
            let batchSize = buffer.count / (channels * spatialSize * spatialSize)
            return (batchSize, channels, spatialSize, spatialSize)
        }
    }
    
    private func performConvolution(
        input: MetalBuffer,
        weights: MetalBuffer,
        output: MetalBuffer
    ) async throws {
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "conv2d_forward")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        let shape = getShape(of: input)
        let outputShape = getShape(of: output)
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        
        // Pass convolution parameters
        let (padH, padW) = getPadding(inputHeight: shape.height, inputWidth: shape.width)
        var batch = UInt32(shape.batch)
        var inC = UInt32(inputChannels)
        var outC = UInt32(outputChannels)
        var inH = UInt32(shape.height)
        var inW = UInt32(shape.width)
        var kH = UInt32(kernelSize.height)
        var kW = UInt32(kernelSize.width)
        var sH = UInt32(stride.height)
        var sW = UInt32(stride.width)
        var pH = UInt32(padH)
        var pW = UInt32(padW)
        var dH = UInt32(dilation.height)
        var dW = UInt32(dilation.width)
        var g = UInt32(groups)
        
        encoder.setBytes(&batch, length: 4, index: 3)
        encoder.setBytes(&inC, length: 4, index: 4)
        encoder.setBytes(&outC, length: 4, index: 5)
        encoder.setBytes(&inH, length: 4, index: 6)
        encoder.setBytes(&inW, length: 4, index: 7)
        encoder.setBytes(&kH, length: 4, index: 8)
        encoder.setBytes(&kW, length: 4, index: 9)
        encoder.setBytes(&sH, length: 4, index: 10)
        encoder.setBytes(&sW, length: 4, index: 11)
        encoder.setBytes(&pH, length: 4, index: 12)
        encoder.setBytes(&pW, length: 4, index: 13)
        encoder.setBytes(&dH, length: 4, index: 14)
        encoder.setBytes(&dW, length: 4, index: 15)
        encoder.setBytes(&g, length: 4, index: 16)
        
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (outputShape.width + 7) / 8,
            height: (outputShape.height + 7) / 8,
            depth: outputChannels * shape.batch
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func getPadding(inputHeight: Int, inputWidth: Int) -> (Int, Int) {
        let dilatedKernelHeight = kernelSize.height + (kernelSize.height - 1) * (dilation.height - 1)
        let dilatedKernelWidth = kernelSize.width + (kernelSize.width - 1) * (dilation.width - 1)
        
        switch padding {
        case .valid:
            return (0, 0)
        case .same:
            let padH = ((inputHeight - 1) * stride.height + dilatedKernelHeight - inputHeight) / 2
            let padW = ((inputWidth - 1) * stride.width + dilatedKernelWidth - inputWidth) / 2
            return (padH, padW)
        case .custom(let h, let w):
            return (h, w)
        }
    }
    
    private func addBias(output: MetalBuffer, bias: MetalBuffer) async throws {
        let shape = getShape(of: output)
        
        // Reshape for bias addition - bias is added per output channel
        try await operations.addBias(
            matrix: output,
            bias: bias,
            rows: shape.batch * shape.height * shape.width,
            cols: outputChannels
        )
    }
    
    private func applyActivation(_ input: MetalBuffer) async throws -> MetalBuffer {
        let output = try await metalPipeline.allocateBuffer(size: input.count)
        try await operations.applyActivation(input, output: output, activation: activation)
        return output
    }
    
    private func backwardActivation(
        gradOutput: MetalBuffer,
        preActivation: MetalBuffer
    ) async throws -> MetalBuffer {
        let gradInput = try await metalPipeline.allocateBuffer(size: gradOutput.count)
        
        if let output = lastOutput {
            try await operations.applyActivationDerivative(
                gradOutput: gradOutput,
                input: preActivation,
                output: output,
                activation: activation,
                gradInput: gradInput
            )
        }
        
        return gradInput
    }
    
    private func computeWeightGradients(
        input: MetalBuffer,
        gradOutput: MetalBuffer,
        weightsGrad: MetalBuffer
    ) async throws {
        // Compute gradients w.r.t weights: gradW = input^T @ gradOutput
        // For convolution: gradW[oc][ic][kh][kw] = sum over batch and spatial positions
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "conv2d_backward_weight")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 1)
        encoder.setBuffer(weightsGrad.buffer, offset: 0, index: 2)
        
        // Get shapes
        let inputShape = getShape(of: input)
        let outputShape = getShape(of: gradOutput)
        let (padH, padW) = getPadding(inputHeight: inputShape.height, inputWidth: inputShape.width)
        
        // Set shapes as uint4
        var inShape = SIMD4<UInt32>(UInt32(inputShape.batch), UInt32(inputChannels), UInt32(inputShape.height), UInt32(inputShape.width))
        var outShape = SIMD4<UInt32>(UInt32(outputShape.batch), UInt32(outputChannels), UInt32(outputShape.height), UInt32(outputShape.width))
        var weightShape = SIMD4<UInt32>(UInt32(outputChannels), UInt32(inputChannels/groups), UInt32(kernelSize.height), UInt32(kernelSize.width))
        var strideVal = SIMD2<UInt32>(UInt32(stride.height), UInt32(stride.width))
        var paddingVal = SIMD2<UInt32>(UInt32(padH), UInt32(padW))
        var dilationVal = SIMD2<UInt32>(UInt32(dilation.height), UInt32(dilation.width))
        var groupsVal = UInt32(groups)
        
        encoder.setBytes(&inShape, length: 16, index: 3)
        encoder.setBytes(&outShape, length: 16, index: 4)
        encoder.setBytes(&weightShape, length: 16, index: 5)
        encoder.setBytes(&strideVal, length: 8, index: 6)
        encoder.setBytes(&paddingVal, length: 8, index: 7)
        encoder.setBytes(&dilationVal, length: 8, index: 8)
        encoder.setBytes(&groupsVal, length: 4, index: 9)
        
        // Each thread computes one weight gradient
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 8)
        let threadgroups = MTLSize(
            width: (Int(kW) + 7) / 8,
            height: (Int(kH) + 7) / 8,
            depth: (Int(outChannels * inChannels) + 7) / 8
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func computeBiasGradients(
        gradOutput: MetalBuffer,
        biasGrad: MetalBuffer
    ) async throws {
        // Compute gradients w.r.t bias: sum gradOutput over batch and spatial dimensions
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "conv2d_backward_bias")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
        encoder.setBuffer(biasGrad.buffer, offset: 0, index: 1)
        
        let outputShape = getShape(of: gradOutput)
        
        // Set shape as uint4
        var outShape = SIMD4<UInt32>(UInt32(outputShape.batch), UInt32(outputChannels), UInt32(outputShape.height), UInt32(outputShape.width))
        encoder.setBytes(&outShape, length: 16, index: 2)
        
        // One thread per output channel
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (outputChannels + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func computeInputGradients(
        weights: MetalBuffer,
        gradOutput: MetalBuffer,
        inputShape: (batch: Int, channels: Int, height: Int, width: Int)
    ) async throws -> MetalBuffer {
        let gradInput = try await metalPipeline.allocateBuffer(
            size: inputShape.batch * inputShape.channels * inputShape.height * inputShape.width
        )
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "conv2d_backward_input")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        let outputShape = getShape(of: gradOutput)
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 2)
        
        // Pass parameters for transposed convolution
        let (padH, padW) = getPadding(inputHeight: inputShape.height, inputWidth: inputShape.width)
        
        // Set shapes as required by shader
        var gradOutShape = SIMD4<UInt32>(UInt32(outputShape.batch), UInt32(outputChannels), UInt32(outputShape.height), UInt32(outputShape.width))
        var weightShape = SIMD4<UInt32>(UInt32(outputChannels), UInt32(inputChannels/groups), UInt32(kernelSize.height), UInt32(kernelSize.width))
        var gradInShape = SIMD4<UInt32>(UInt32(inputShape.batch), UInt32(inputChannels), UInt32(inputShape.height), UInt32(inputShape.width))
        var strideVal = SIMD2<UInt32>(UInt32(stride.height), UInt32(stride.width))
        var paddingVal = SIMD2<UInt32>(UInt32(padH), UInt32(padW))
        var dilationVal = SIMD2<UInt32>(UInt32(dilation.height), UInt32(dilation.width))
        var groupsVal = UInt32(groups)
        
        encoder.setBytes(&gradOutShape, length: 16, index: 3)
        encoder.setBytes(&weightShape, length: 16, index: 4)
        encoder.setBytes(&gradInShape, length: 16, index: 5)
        encoder.setBytes(&strideVal, length: 8, index: 6)
        encoder.setBytes(&paddingVal, length: 8, index: 7)
        encoder.setBytes(&dilationVal, length: 8, index: 8)
        encoder.setBytes(&groupsVal, length: 4, index: 9)
        
        // Dispatch threads for input gradient computation
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (inputShape.width + 7) / 8,
            height: (inputShape.height + 7) / 8,
            depth: inputChannels * inputShape.batch
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return gradInput
    }
}

// MARK: - Conv2D Builder

public extension Conv2DLayer {
    /// Convenience initializer for common Conv2D configurations
    static func conv2d(
        _ outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Padding = .same,
        activation: Activation = .relu,
        metalPipeline: MetalMLPipeline
    ) -> (Int) async throws -> Conv2DLayer {
        return { inputChannels in
            try await Conv2DLayer(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                kernelSize: (kernelSize, kernelSize),
                stride: (stride, stride),
                padding: padding,
                activation: activation,
                metalPipeline: metalPipeline
            )
        }
    }
}
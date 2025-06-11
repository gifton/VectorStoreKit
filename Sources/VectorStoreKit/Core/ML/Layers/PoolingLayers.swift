// VectorStoreKit: Pooling Layers
//
// Max and Average pooling layers with Metal acceleration

import Foundation
@preconcurrency import Metal

/// Base protocol for pooling layers
public protocol PoolingLayer: NeuralLayer {
    var poolSize: (height: Int, width: Int) { get }
    var stride: (height: Int, width: Int) { get }
    var padding: Conv2DLayer.Padding { get }
}

/// 2D Max Pooling layer
public actor MaxPool2DLayer: PoolingLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let operations: MetalMLOperations
    
    public let poolSize: (height: Int, width: Int)
    public let stride: (height: Int, width: Int)
    public let padding: Conv2DLayer.Padding
    
    // Cached values for backward pass
    private var lastInput: MetalBuffer?
    private var lastOutput: MetalBuffer?
    private var maxIndices: MetalBuffer?
    private var isTraining: Bool = true
    
    // MARK: - Initialization
    
    public init(
        poolSize: (height: Int, width: Int) = (2, 2),
        stride: (height: Int, width: Int)? = nil,
        padding: Conv2DLayer.Padding = .valid,
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.poolSize = poolSize
        self.stride = stride ?? poolSize  // Default stride equals pool size
        self.padding = padding
        self.metalPipeline = metalPipeline
        self.operations = await metalPipeline.getOperations()
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Validate input shape
        guard input.shape.rank == 4 else {
            throw MetalMLError.invalidArchitecture("MaxPool2D expects 4D input (batch, channels, height, width)")
        }
        
        // Save input for backward pass
        if isTraining {
            lastInput = input
        }
        
        // Calculate output shape
        let outputShape = calculateOutputShape(inputShape: input.shape)
        
        // Allocate output buffer
        let output = try await metalPipeline.allocateBuffer(shape: outputShape)
        
        // Allocate indices buffer for backward pass if training
        if isTraining {
            maxIndices = try await metalPipeline.allocateBuffer(shape: outputShape)
        }
        
        // Perform max pooling
        try await performMaxPooling(
            input: input,
            output: output,
            indices: maxIndices
        )
        
        // Save output for potential use
        if isTraining {
            lastOutput = output
        }
        
        return output
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let input = lastInput,
              let indices = maxIndices else {
            throw MetalMLError.parameterNotFound(name: "cached values")
        }
        
        // Calculate gradient input size (same as original input)
        let gradInput = try await metalPipeline.allocateBuffer(size: input.count)
        
        // Perform backward max pooling
        try await performMaxPoolingBackward(
            gradOutput: gradOutput,
            indices: indices,
            gradInput: gradInput
        )
        
        return gradInput
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        // No parameters to update in pooling layers
    }
    
    public func getParameters() async -> MetalBuffer? {
        nil  // No parameters in pooling layers
    }
    
    public func getParameterCount() async -> Int {
        0  // No parameters in pooling layers
    }
    
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
    
    // MARK: - Private Methods
    
    private func calculateOutputShape(inputShape: TensorShape) -> TensorShape {
        let dims = inputShape.dimensions
        let batch = dims[0]
        let channels = dims[1]
        let height = dims[2]
        let width = dims[3]
        
        let outputHeight = (height - poolSize.height) / stride.height + 1
        let outputWidth = (width - poolSize.width) / stride.width + 1
        
        return TensorShape(batch, channels, outputHeight, outputWidth)
    }
    
    private func getInputShape(_ buffer: MetalBuffer) -> (batch: Int, channels: Int, height: Int, width: Int) {
        // Use shape information from MetalBuffer
        if buffer.shape.rank == 4 {
            let dims = buffer.shape.dimensions
            return (dims[0], dims[1], dims[2], dims[3])
        }
        
        // Fallback for 1D buffers (legacy support)
        let totalElements = buffer.count
        
        // Try to infer dimensions - this is a simplification
        // Check common image sizes
        for size in [224, 112, 56, 28, 14, 7] {
            for channels in [3, 32, 64, 128, 256, 512] {
                if totalElements == channels * size * size {
                    return (1, channels, size, size)
                } else if totalElements % (channels * size * size) == 0 {
                    let batch = totalElements / (channels * size * size)
                    return (batch, channels, size, size)
                }
            }
        }
        
        // Fallback to square assumption
        let spatialElements = totalElements / 64  // Assume 64 channels
        let spatialSize = Int(sqrt(Double(spatialElements)))
        return (1, 64, spatialSize, spatialSize)
    }
    
    private func performMaxPooling(
        input: MetalBuffer,
        output: MetalBuffer,
        indices: MetalBuffer?
    ) async throws {
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "maxpool2d_forward")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        let inputShape = getInputShape(input)
        let outputHeight = (inputShape.height - poolSize.height) / stride.height + 1
        let outputWidth = (inputShape.width - poolSize.width) / stride.width + 1
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        if let indices = indices {
            encoder.setBuffer(indices.buffer, offset: 0, index: 2)
        }
        
        // Pass pooling parameters
        var batch = UInt32(inputShape.batch)
        var channels = UInt32(inputShape.channels)
        var inH = UInt32(inputShape.height)
        var inW = UInt32(inputShape.width)
        var outH = UInt32(outputHeight)
        var outW = UInt32(outputWidth)
        var poolH = UInt32(poolSize.height)
        var poolW = UInt32(poolSize.width)
        var strideH = UInt32(stride.height)
        var strideW = UInt32(stride.width)
        
        encoder.setBytes(&batch, length: 4, index: 3)
        encoder.setBytes(&channels, length: 4, index: 4)
        encoder.setBytes(&inH, length: 4, index: 5)
        encoder.setBytes(&inW, length: 4, index: 6)
        encoder.setBytes(&outH, length: 4, index: 7)
        encoder.setBytes(&outW, length: 4, index: 8)
        encoder.setBytes(&poolH, length: 4, index: 9)
        encoder.setBytes(&poolW, length: 4, index: 10)
        encoder.setBytes(&strideH, length: 4, index: 11)
        encoder.setBytes(&strideW, length: 4, index: 12)
        
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (outputWidth + 7) / 8,
            height: (outputHeight + 7) / 8,
            depth: Int(channels * batch)
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func performMaxPoolingBackward(
        gradOutput: MetalBuffer,
        indices: MetalBuffer,
        gradInput: MetalBuffer
    ) async throws {
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "maxpool2d_backward")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        // Get dimensions from grad buffers
        let outputShape = getOutputShapeFromGrad(gradOutput)
        let inputShape = getInputShapeFromGrad(gradInput)
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
        encoder.setBuffer(indices.buffer, offset: 0, index: 1)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 2)
        
        // Pass dimensions
        var batch = UInt32(inputShape.batch)
        var channels = UInt32(inputShape.channels)
        var inH = UInt32(inputShape.height)
        var inW = UInt32(inputShape.width)
        var outH = UInt32(outputShape.height)
        var outW = UInt32(outputShape.width)
        
        encoder.setBytes(&batch, length: 4, index: 3)
        encoder.setBytes(&channels, length: 4, index: 4)
        encoder.setBytes(&inH, length: 4, index: 5)
        encoder.setBytes(&inW, length: 4, index: 6)
        encoder.setBytes(&outH, length: 4, index: 7)
        encoder.setBytes(&outW, length: 4, index: 8)
        
        // First clear gradInput
        let clearPipeline = try shaderLibrary.pipeline(for: "clear_buffer")
        encoder.setComputePipelineState(clearPipeline)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 0)
        var size = UInt32(gradInput.count)
        encoder.setBytes(&size, length: 4, index: 1)
        
        let clearThreads = MTLSize(width: (gradInput.count + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(clearThreads, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        
        // Then scatter gradients using indices
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
        encoder.setBuffer(indices.buffer, offset: 0, index: 1)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 2)
        
        encoder.setBytes(&batch, length: 4, index: 3)
        encoder.setBytes(&channels, length: 4, index: 4)
        encoder.setBytes(&inH, length: 4, index: 5)
        encoder.setBytes(&inW, length: 4, index: 6)
        encoder.setBytes(&outH, length: 4, index: 7)
        encoder.setBytes(&outW, length: 4, index: 8)
        
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (outputShape.width + 7) / 8,
            height: (outputShape.height + 7) / 8,
            depth: Int(channels * batch)
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func getOutputShapeFromGrad(_ buffer: MetalBuffer) -> (batch: Int, channels: Int, height: Int, width: Int) {
        // Infer output dimensions from gradient buffer
        let shape = getInputShape(buffer)
        // For max pooling output
        return shape
    }
    
    private func getInputShapeFromGrad(_ buffer: MetalBuffer) -> (batch: Int, channels: Int, height: Int, width: Int) {
        // Infer input dimensions from gradient buffer
        return getInputShape(buffer)
    }
}

/// 2D Average Pooling layer
public actor AvgPool2DLayer: PoolingLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let operations: MetalMLOperations
    
    public let poolSize: (height: Int, width: Int)
    public let stride: (height: Int, width: Int)
    public let padding: Conv2DLayer.Padding
    private let includepadding: Bool
    
    // Cached values for backward pass
    private var lastInputShape: (batch: Int, channels: Int, height: Int, width: Int)?
    private var isTraining: Bool = true
    
    // MARK: - Initialization
    
    public init(
        poolSize: (height: Int, width: Int) = (2, 2),
        stride: (height: Int, width: Int)? = nil,
        padding: Conv2DLayer.Padding = .valid,
        includePadding: Bool = true,
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.poolSize = poolSize
        self.stride = stride ?? poolSize  // Default stride equals pool size
        self.padding = padding
        self.includepadding = includePadding
        self.metalPipeline = metalPipeline
        self.operations = await metalPipeline.getOperations()
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Save input shape for backward pass
        if isTraining {
            lastInputShape = getShape(of: input)
        }
        
        // Calculate output size
        let outputSize = calculateOutputSize(inputBuffer: input)
        
        // Allocate output buffer
        let output = try await metalPipeline.allocateBuffer(size: outputSize)
        
        // Perform average pooling
        try await performAvgPooling(input: input, output: output)
        
        return output
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let inputShape = lastInputShape else {
            throw MetalMLError.parameterNotFound(name: "input shape")
        }
        
        // Calculate gradient input shape
        let gradInputShape = TensorShape(inputShape.batch, inputShape.channels, inputShape.height, inputShape.width)
        let gradInput = try await metalPipeline.allocateBuffer(shape: gradInputShape)
        
        // Perform backward average pooling
        try await performAvgPoolingBackward(
            gradOutput: gradOutput,
            gradInput: gradInput,
            inputShape: inputShape
        )
        
        return gradInput
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        // No parameters to update in pooling layers
    }
    
    public func getParameters() async -> MetalBuffer? {
        nil  // No parameters in pooling layers
    }
    
    public func getParameterCount() async -> Int {
        0  // No parameters in pooling layers
    }
    
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
    
    // MARK: - Private Methods
    
    private func calculateOutputSize(inputBuffer: MetalBuffer) -> Int {
        let shape = getShape(of: inputBuffer)
        let outputHeight = (shape.height - poolSize.height) / stride.height + 1
        let outputWidth = (shape.width - poolSize.width) / stride.width + 1
        
        return shape.batch * shape.channels * outputHeight * outputWidth
    }
    
    private func getShape(of buffer: MetalBuffer) -> (batch: Int, channels: Int, height: Int, width: Int) {
        // Use shape information from MetalBuffer
        if buffer.shape.rank == 4 {
            let dims = buffer.shape.dimensions
            return (dims[0], dims[1], dims[2], dims[3])
        }
        
        // Fallback for 1D buffers - try to infer dimensions
        return getInputShape(buffer)
    }
    
    private func getInputShape(_ buffer: MetalBuffer) -> (batch: Int, channels: Int, height: Int, width: Int) {
        // Use shape information from MetalBuffer
        if buffer.shape.rank == 4 {
            let dims = buffer.shape.dimensions
            return (dims[0], dims[1], dims[2], dims[3])
        }
        
        // Fallback for legacy 1D buffers
        let totalElements = buffer.count
        
        // Try to infer dimensions - this is a simplification
        // Check common image sizes
        for size in [224, 112, 56, 28, 14, 7] {
            for channels in [3, 32, 64, 128, 256, 512] {
                if totalElements == channels * size * size {
                    return (1, channels, size, size)
                } else if totalElements % (channels * size * size) == 0 {
                    let batch = totalElements / (channels * size * size)
                    return (batch, channels, size, size)
                }
            }
        }
        
        // Fallback to square assumption
        let spatialElements = totalElements / 64  // Assume 64 channels
        let spatialSize = Int(sqrt(Double(spatialElements)))
        return (1, 64, spatialSize, spatialSize)
    }
    
    private func performAvgPooling(
        input: MetalBuffer,
        output: MetalBuffer
    ) async throws {
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "avgpool2d_forward")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        let inputShape = getShape(of: input)
        let outputHeight = (inputShape.height - poolSize.height) / stride.height + 1
        let outputWidth = (inputShape.width - poolSize.width) / stride.width + 1
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        
        // Pass pooling parameters
        var batch = UInt32(inputShape.batch)
        var channels = UInt32(inputShape.channels)
        var inH = UInt32(inputShape.height)
        var inW = UInt32(inputShape.width)
        var outH = UInt32(outputHeight)
        var outW = UInt32(outputWidth)
        var poolH = UInt32(poolSize.height)
        var poolW = UInt32(poolSize.width)
        var strideH = UInt32(stride.height)
        var strideW = UInt32(stride.width)
        var includePad = includepadding ? UInt32(1) : UInt32(0)
        
        encoder.setBytes(&batch, length: 4, index: 2)
        encoder.setBytes(&channels, length: 4, index: 3)
        encoder.setBytes(&inH, length: 4, index: 4)
        encoder.setBytes(&inW, length: 4, index: 5)
        encoder.setBytes(&outH, length: 4, index: 6)
        encoder.setBytes(&outW, length: 4, index: 7)
        encoder.setBytes(&poolH, length: 4, index: 8)
        encoder.setBytes(&poolW, length: 4, index: 9)
        encoder.setBytes(&strideH, length: 4, index: 10)
        encoder.setBytes(&strideW, length: 4, index: 11)
        encoder.setBytes(&includePad, length: 4, index: 12)
        
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (outputWidth + 7) / 8,
            height: (outputHeight + 7) / 8,
            depth: Int(channels * batch)
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func performAvgPoolingBackward(
        gradOutput: MetalBuffer,
        gradInput: MetalBuffer,
        inputShape: (batch: Int, channels: Int, height: Int, width: Int)
    ) async throws {
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "avgpool2d_backward")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        let outputHeight = (inputShape.height - poolSize.height) / stride.height + 1
        let outputWidth = (inputShape.width - poolSize.width) / stride.width + 1
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 1)
        
        // Pass dimensions
        var batch = UInt32(inputShape.batch)
        var channels = UInt32(inputShape.channels)
        var inH = UInt32(inputShape.height)
        var inW = UInt32(inputShape.width)
        var outH = UInt32(outputHeight)
        var outW = UInt32(outputWidth)
        var poolH = UInt32(poolSize.height)
        var poolW = UInt32(poolSize.width)
        var strideH = UInt32(stride.height)
        var strideW = UInt32(stride.width)
        
        encoder.setBytes(&batch, length: 4, index: 2)
        encoder.setBytes(&channels, length: 4, index: 3)
        encoder.setBytes(&inH, length: 4, index: 4)
        encoder.setBytes(&inW, length: 4, index: 5)
        encoder.setBytes(&outH, length: 4, index: 6)
        encoder.setBytes(&outW, length: 4, index: 7)
        encoder.setBytes(&poolH, length: 4, index: 8)
        encoder.setBytes(&poolW, length: 4, index: 9)
        encoder.setBytes(&strideH, length: 4, index: 10)
        encoder.setBytes(&strideW, length: 4, index: 11)
        
        // First clear gradInput
        let clearPipeline = try shaderLibrary.pipeline(for: "clear_buffer")
        encoder.setComputePipelineState(clearPipeline)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 0)
        var size = UInt32(gradInput.count)
        encoder.setBytes(&size, length: 4, index: 1)
        
        let clearThreads = MTLSize(width: (gradInput.count + 255) / 256, height: 1, depth: 1)
        encoder.dispatchThreadgroups(clearThreads, threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        
        // Then distribute gradients
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 1)
        
        encoder.setBytes(&batch, length: 4, index: 2)
        encoder.setBytes(&channels, length: 4, index: 3)
        encoder.setBytes(&inH, length: 4, index: 4)
        encoder.setBytes(&inW, length: 4, index: 5)
        encoder.setBytes(&outH, length: 4, index: 6)
        encoder.setBytes(&outW, length: 4, index: 7)
        encoder.setBytes(&poolH, length: 4, index: 8)
        encoder.setBytes(&poolW, length: 4, index: 9)
        encoder.setBytes(&strideH, length: 4, index: 10)
        encoder.setBytes(&strideW, length: 4, index: 11)
        
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (outputWidth + 7) / 8,
            height: (outputHeight + 7) / 8,
            depth: Int(channels * batch)
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}

/// Global Average Pooling layer
public actor GlobalAvgPool2DLayer: NeuralLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let operations: MetalMLOperations
    
    // Cached values for backward pass
    private var lastInputShape: (batch: Int, channels: Int, height: Int, width: Int)?
    private var isTraining: Bool = true
    
    // MARK: - Initialization
    
    public init(metalPipeline: MetalMLPipeline) async throws {
        self.metalPipeline = metalPipeline
        self.operations = await metalPipeline.getOperations()
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Save input shape for backward pass
        let shape = getShape(of: input)
        if isTraining {
            lastInputShape = shape
        }
        
        // Output size is just batch * channels
        let outputSize = shape.batch * shape.channels
        let output = try await metalPipeline.allocateBuffer(size: outputSize)
        
        // Perform global average pooling
        try await performGlobalAvgPooling(
            input: input,
            output: output,
            shape: shape
        )
        
        return output
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let inputShape = lastInputShape else {
            throw MetalMLError.parameterNotFound(name: "input shape")
        }
        
        // Calculate gradient input shape
        let gradInputShape = TensorShape(inputShape.batch, inputShape.channels, inputShape.height, inputShape.width)
        let gradInput = try await metalPipeline.allocateBuffer(shape: gradInputShape)
        
        // Distribute gradient uniformly across spatial dimensions
        try await performGlobalAvgPoolingBackward(
            gradOutput: gradOutput,
            gradInput: gradInput,
            inputShape: inputShape
        )
        
        return gradInput
    }
    
    // MARK: - NeuralLayer Protocol
    
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
    
    // MARK: - Private Methods
    
    private func getShape(of buffer: MetalBuffer) -> (batch: Int, channels: Int, height: Int, width: Int) {
        // Use shape information from MetalBuffer
        if buffer.shape.rank == 4 {
            let dims = buffer.shape.dimensions
            return (dims[0], dims[1], dims[2], dims[3])
        }
        
        // Fallback for legacy 1D buffers
        let totalElements = buffer.count
        
        // Try to infer dimensions - this is a simplification
        // Check common image sizes
        for size in [224, 112, 56, 28, 14, 7] {
            for channels in [3, 32, 64, 128, 256, 512] {
                if totalElements == channels * size * size {
                    return (1, channels, size, size)
                } else if totalElements % (channels * size * size) == 0 {
                    let batch = totalElements / (channels * size * size)
                    return (batch, channels, size, size)
                }
            }
        }
        
        // Fallback to square assumption
        let spatialElements = totalElements / 64  // Assume 64 channels
        let spatialSize = Int(sqrt(Double(spatialElements)))
        return (1, 64, spatialSize, spatialSize)
    }
    
    private func performGlobalAvgPooling(
        input: MetalBuffer,
        output: MetalBuffer,
        shape: (batch: Int, channels: Int, height: Int, width: Int)
    ) async throws {
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "global_avgpool2d_forward")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        
        // Pass dimensions
        var batch = UInt32(shape.batch)
        var channels = UInt32(shape.channels)
        var height = UInt32(shape.height)
        var width = UInt32(shape.width)
        
        encoder.setBytes(&batch, length: 4, index: 2)
        encoder.setBytes(&channels, length: 4, index: 3)
        encoder.setBytes(&height, length: 4, index: 4)
        encoder.setBytes(&width, length: 4, index: 5)
        
        // Each thread processes one channel
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (shape.channels * shape.batch + 255) / 256,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func performGlobalAvgPoolingBackward(
        gradOutput: MetalBuffer,
        gradInput: MetalBuffer,
        inputShape: (batch: Int, channels: Int, height: Int, width: Int)
    ) async throws {
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "global_avgpool2d_backward")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 1)
        
        // Pass dimensions
        var batch = UInt32(inputShape.batch)
        var channels = UInt32(inputShape.channels)
        var height = UInt32(inputShape.height)
        var width = UInt32(inputShape.width)
        
        encoder.setBytes(&batch, length: 4, index: 2)
        encoder.setBytes(&channels, length: 4, index: 3)
        encoder.setBytes(&height, length: 4, index: 4)
        encoder.setBytes(&width, length: 4, index: 5)
        
        // Each thread processes one spatial position
        let spatialSize = inputShape.height * inputShape.width
        let totalThreads = inputShape.batch * inputShape.channels * spatialSize
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (totalThreads + 255) / 256,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
// VectorStoreKit: Metal ML Operations
//
// High-level interface for ML compute operations

import Foundation
@preconcurrency import Metal

/// Manages Metal compute operations for ML
public actor MetalMLOperations {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let shaderLibrary: MLShaderLibrary
    
    public init(device: MTLDevice, commandQueue: MTLCommandQueue, shaderLibrary: MLShaderLibrary) {
        self.device = device
        self.commandQueue = commandQueue
        self.shaderLibrary = shaderLibrary
    }
    
    // MARK: - Matrix Operations
    
    /// Perform matrix multiplication: C = A * B
    public func matmul(
        _ a: MetalBuffer,      // [M x K]
        _ b: MetalBuffer,      // [K x N]
        output: MetalBuffer,   // [M x N]
        m: Int, n: Int, k: Int,
        useTiling: Bool = true
    ) async throws {
        let functionName = useTiling ? MLShaderLibrary.MatrixOperation.matmulTiled : .matmulForward
        let pipeline = try shaderLibrary.pipeline(for: functionName.rawValue)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a.buffer, offset: 0, index: 0)
        encoder.setBuffer(b.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        
        var mVal = UInt32(m)
        var nVal = UInt32(n)
        var kVal = UInt32(k)
        encoder.setBytes(&mVal, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&kVal, length: MemoryLayout<UInt32>.size, index: 5)
        
        let workSize = MTLSize(width: n, height: m, depth: 1)
        let (threadgroupSize, threadgroupCount) = shaderLibrary.threadConfiguration(
            for: pipeline,
            workSize: workSize
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    /// Add bias to matrix
    public func addBias(
        matrix: MetalBuffer,    // [rows x cols]
        bias: MetalBuffer,      // [cols]
        rows: Int,
        cols: Int
    ) async throws {
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.MatrixOperation.addBias.rawValue)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(matrix.buffer, offset: 0, index: 0)
        encoder.setBuffer(bias.buffer, offset: 0, index: 1)
        
        var rowsVal = UInt32(rows)
        var colsVal = UInt32(cols)
        encoder.setBytes(&rowsVal, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&colsVal, length: MemoryLayout<UInt32>.size, index: 3)
        
        let workSize = MTLSize(width: cols, height: rows, depth: 1)
        let (threadgroupSize, threadgroupCount) = shaderLibrary.threadConfiguration(
            for: pipeline,
            workSize: workSize
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    // MARK: - Activation Functions
    
    /// Apply activation function
    public func applyActivation(
        _ input: MetalBuffer,
        output: MetalBuffer,
        activation: Activation
    ) async throws {
        let functionName: String
        var alpha: Float = 0.01 // For leaky ReLU
        
        switch activation {
        case .relu:
            functionName = MLShaderLibrary.ActivationFunction.reluForward.rawValue
        case .leakyRelu:
            functionName = MLShaderLibrary.ActivationFunction.leakyReluForward.rawValue
            alpha = 0.01 // Default leaky ReLU alpha
        case .sigmoid:
            functionName = MLShaderLibrary.ActivationFunction.sigmoidForward.rawValue
        case .tanh:
            functionName = MLShaderLibrary.ActivationFunction.tanhForward.rawValue
        case .elu:
            functionName = MLShaderLibrary.ActivationFunction.eluForward.rawValue
        case .selu:
            functionName = MLShaderLibrary.ActivationFunction.seluForward.rawValue
        case .gelu:
            functionName = MLShaderLibrary.ActivationFunction.geluForward.rawValue
        case .swish:
            functionName = MLShaderLibrary.ActivationFunction.swishForward.rawValue
        default:
            // Linear activation - just copy
            try await copyBuffer(from: input, to: output)
            return
        }
        
        let pipeline = try shaderLibrary.pipeline(for: functionName)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        
        if activation.isParametric {
            encoder.setBytes(&alpha, length: MemoryLayout<Float>.size, index: 2)
            var size = UInt32(input.count)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
        } else {
            var size = UInt32(input.count)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        }
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (input.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    /// Apply activation derivative
    public func applyActivationDerivative(
        gradOutput: MetalBuffer,
        input: MetalBuffer,
        output: MetalBuffer,
        activation: Activation,
        gradInput: MetalBuffer
    ) async throws {
        let functionName: String
        var alpha: Float = 0.01
        
        switch activation {
        case .relu:
            functionName = MLShaderLibrary.ActivationFunction.reluBackward.rawValue
        case .leakyRelu:
            functionName = MLShaderLibrary.ActivationFunction.leakyReluBackward.rawValue
            alpha = 0.01 // Default leaky ReLU alpha
        case .sigmoid:
            functionName = MLShaderLibrary.ActivationFunction.sigmoidBackward.rawValue
        case .tanh:
            functionName = MLShaderLibrary.ActivationFunction.tanhBackward.rawValue
        case .elu:
            functionName = MLShaderLibrary.ActivationFunction.eluBackward.rawValue
        case .gelu:
            functionName = MLShaderLibrary.ActivationFunction.geluBackward.rawValue
        case .swish:
            functionName = MLShaderLibrary.ActivationFunction.swishBackward.rawValue
        default:
            // Linear activation - gradient passes through
            try await copyBuffer(from: gradOutput, to: gradInput)
            return
        }
        
        let pipeline = try shaderLibrary.pipeline(for: functionName)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
        encoder.setBuffer(input.buffer, offset: 0, index: 1)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 2)
        
        var bufferIndex = 3
        
        // Some activations need the output
        let needsOutput: [Activation] = [.sigmoid, .tanh, .elu, .swish]
        if needsOutput.contains(activation) {
            encoder.setBuffer(output.buffer, offset: 0, index: 2)
            encoder.setBuffer(gradInput.buffer, offset: 0, index: 3)
            bufferIndex = 4
        }
        
        if activation.isParametric {
            encoder.setBytes(&alpha, length: MemoryLayout<Float>.size, index: bufferIndex)
            bufferIndex += 1
        }
        
        var size = UInt32(input.count)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: bufferIndex)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (input.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    // MARK: - Backward Pass Operations
    
    /// Compute gradients for dense layer weights
    public func computeWeightGradients(
        gradOutput: MetalBuffer,  // [outputSize]
        input: MetalBuffer,       // [inputSize]
        gradWeights: MetalBuffer, // [outputSize x inputSize]
        outputSize: Int,
        inputSize: Int
    ) async throws {
        let pipeline = try shaderLibrary.pipeline(for: "outer_product")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
        encoder.setBuffer(input.buffer, offset: 0, index: 1)
        encoder.setBuffer(gradWeights.buffer, offset: 0, index: 2)
        
        var M = UInt32(outputSize)
        var N = UInt32(inputSize)
        encoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 4)
        
        let workSize = MTLSize(width: inputSize, height: outputSize, depth: 1)
        let (threadgroupSize, threadgroupCount) = shaderLibrary.threadConfiguration(
            for: pipeline,
            workSize: workSize
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    /// Compute gradients for dense layer input
    public func computeInputGradients(
        weights: MetalBuffer,     // [outputSize x inputSize]
        gradOutput: MetalBuffer,  // [outputSize]
        gradInput: MetalBuffer,   // [inputSize]
        outputSize: Int,
        inputSize: Int
    ) async throws {
        // Matrix-vector multiplication: gradInput = weights^T * gradOutput
        try await matmul(
            weights,
            gradOutput,
            output: gradInput,
            m: inputSize,
            n: 1,
            k: outputSize,
            useTiling: false
        )
    }
    
    // MARK: - Utility Operations
    
    /// Copy buffer
    public func copyBuffer(from: MetalBuffer, to: MetalBuffer) async throws {
        guard from.count == to.count else {
            throw MetalMLError.incompatibleBufferSize(expected: from.count, actual: to.count)
        }
        
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.MatrixOperation.copyMatrix.rawValue)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(from.buffer, offset: 0, index: 0)
        encoder.setBuffer(to.buffer, offset: 0, index: 1)
        
        var size = UInt32(from.count)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (from.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    /// Add two buffers element-wise: output = a + b
    public func addBuffers(
        _ a: MetalBuffer,
        _ b: MetalBuffer,
        output: MetalBuffer
    ) async throws {
        guard a.count == b.count && a.count == output.count else {
            throw MetalMLError.incompatibleBufferSize(expected: a.count, actual: b.count)
        }
        
        let pipeline = try shaderLibrary.pipeline(for: "element_add")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a.buffer, offset: 0, index: 0)
        encoder.setBuffer(b.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        
        var size = UInt32(a.count)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (a.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    /// Scale buffer by a scalar: output = input * scale
    public func scaleBuffer(
        _ input: MetalBuffer,
        scale: Float,
        output: MetalBuffer
    ) async throws {
        guard input.count == output.count else {
            throw MetalMLError.incompatibleBufferSize(expected: input.count, actual: output.count)
        }
        
        let pipeline = try shaderLibrary.pipeline(for: "scale_buffer")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        
        var scaleVal = scale
        encoder.setBytes(&scaleVal, length: MemoryLayout<Float>.size, index: 2)
        
        var size = UInt32(input.count)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (input.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}

// MARK: - Activation Extensions

extension Activation {
    var isParametric: Bool {
        switch self {
        case .leakyRelu, .elu:
            return true
        default:
            return false
        }
    }
}
// VectorStoreKit: ML Activation Functions
//
// Metal-accelerated activation functions for neural networks
//

import Foundation
@preconcurrency import Metal

/// Activation functions for neural networks
public enum ActivationType: String, Codable, Sendable, CaseIterable {
    case linear = "linear"
    case relu = "relu"
    case leakyRelu = "leakyRelu"
    case sigmoid = "sigmoid"
    case tanh = "tanh"
    case softmax = "softmax"
    case gelu = "gelu"
    case swish = "swish"
    case elu = "elu"
    case mish = "mish"
    
    /// Default parameter values for parametric activations
    public var defaultParameter: Float {
        switch self {
        case .leakyRelu:
            return 0.01  // alpha
        case .elu:
            return 1.0   // alpha
        default:
            return 0.0
        }
    }
    
    /// Whether this activation has learnable parameters
    public var hasParameters: Bool {
        switch self {
        case .leakyRelu, .elu:
            return true
        default:
            return false
        }
    }
    
    /// Metal shader function name for forward pass
    public var forwardFunctionName: String {
        switch self {
        case .linear:
            return "linear_forward"
        case .relu:
            return "relu_forward"
        case .leakyRelu:
            return "leaky_relu_forward"
        case .sigmoid:
            return "sigmoid_forward"
        case .tanh:
            return "tanh_forward"
        case .softmax:
            return "softmax_forward"
        case .gelu:
            return "gelu_forward"
        case .swish:
            return "swish_forward"
        case .elu:
            return "elu_forward"
        case .mish:
            return "mish_forward"
        }
    }
    
    /// Metal shader function name for backward pass
    public var backwardFunctionName: String {
        switch self {
        case .linear:
            return "linear_backward"
        case .relu:
            return "relu_backward"
        case .leakyRelu:
            return "leaky_relu_backward"
        case .sigmoid:
            return "sigmoid_backward"
        case .tanh:
            return "tanh_backward"
        case .softmax:
            return "softmax_backward"
        case .gelu:
            return "gelu_backward"
        case .swish:
            return "swish_backward"
        case .elu:
            return "elu_backward"
        case .mish:
            return "mish_backward"
        }
    }
}

/// Metal-accelerated activation layer
public actor ActivationLayer: NeuralLayer {
    public let activation: ActivationType
    private let metalPipeline: MetalMLPipeline
    public var parameter: Float
    private var lastInput: MetalBuffer?
    private var lastOutput: MetalBuffer?
    private var isTraining: Bool = true
    
    // Cached pipelines
    private var forwardPipeline: MTLComputePipelineState?
    private var backwardPipeline: MTLComputePipelineState?
    
    public init(
        activation: ActivationType,
        parameter: Float? = nil,
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.activation = activation
        self.metalPipeline = metalPipeline
        self.parameter = parameter ?? activation.defaultParameter
        
        // Load pipelines
        try await loadPipelines()
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        guard let pipeline = forwardPipeline else {
            throw ActivationError.pipelineNotLoaded
        }
        
        // Store for backward pass
        if isTraining {
            lastInput = input
        }
        
        // Allocate output buffer
        let output = try await metalPipeline.allocateBuffer(shape: input.shape)
        
        // Execute forward pass
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        try await commandQueue.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            
            var size = UInt32(input.count)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
            
            // Set parameters for parametric activations
            if activation.hasParameters {
                var param = parameter
                encoder.setBytes(&param, length: MemoryLayout<Float>.size, index: 3)
            }
            
            // Special handling for softmax (needs threadgroup memory)
            if activation == .softmax {
                let threadsPerThreadgroup = min(256, pipeline.maxTotalThreadsPerThreadgroup)
                let threadgroupMemoryLength = threadsPerThreadgroup * MemoryLayout<Float>.size * 2
                encoder.setThreadgroupMemoryLength(threadgroupMemoryLength, index: 0)
                
                let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
                let threads = MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1)
                encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threads)
            } else {
                // Standard dispatch for element-wise activations
                let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
                let threadgroups = MTLSize(
                    width: (input.count + 255) / 256,
                    height: 1,
                    depth: 1
                )
                encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            }
            
            encoder.endEncoding()
        }
        
        // Store for backward pass
        if isTraining {
            lastOutput = output
        }
        
        return output
    }
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let pipeline = backwardPipeline,
              let input = lastInput else {
            throw ActivationError.backwardBeforeForward
        }
        
        // Allocate gradient input buffer
        let gradInput = try await metalPipeline.allocateBuffer(shape: input.shape)
        
        // Execute backward pass
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        
        try await commandQueue.execute { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
            
            // Different buffer layouts for different activations
            switch activation {
            case .sigmoid, .tanh, .elu:
                // These need the output from forward pass
                guard let output = lastOutput else {
                    throw ActivationError.backwardBeforeForward
                }
                encoder.setBuffer(output.buffer, offset: 0, index: 1)
                encoder.setBuffer(gradInput.buffer, offset: 0, index: 2)
                
            case .swish:
                // Swish needs both input and output
                guard let output = lastOutput else {
                    throw ActivationError.backwardBeforeForward
                }
                encoder.setBuffer(input.buffer, offset: 0, index: 1)
                encoder.setBuffer(output.buffer, offset: 0, index: 2)
                encoder.setBuffer(gradInput.buffer, offset: 0, index: 3)
                
            default:
                // Most activations just need input
                encoder.setBuffer(input.buffer, offset: 0, index: 1)
                encoder.setBuffer(gradInput.buffer, offset: 0, index: 2)
            }
            
            var size = UInt32(input.count)
            let sizeIndex = activation == .swish ? 4 : 3
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: sizeIndex)
            
            // Set parameters for parametric activations
            if activation.hasParameters {
                var param = parameter
                let paramIndex = sizeIndex + 1
                encoder.setBytes(&param, length: MemoryLayout<Float>.size, index: paramIndex)
            }
            
            // Dispatch
            let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            let threadgroups = MTLSize(
                width: (input.count + 255) / 256,
                height: 1,
                depth: 1
            )
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
        
        return gradInput
    }
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        // Most activations don't have parameters to update
        // For parametric activations, this would update the parameter
        if activation.hasParameters {
            // In a full implementation, we'd accumulate gradients for the parameter
            // and update it here
        }
    }
    
    public func getParameters() async -> MetalBuffer? {
        if activation.hasParameters {
            // Return parameter as a 1-element buffer
            let buffer = try? await metalPipeline.allocateBuffer(size: 1)
            if let buffer = buffer {
                let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: 1)
                ptr[0] = parameter
                return buffer
            }
        }
        return nil
    }
    
    public func getParameterCount() async -> Int {
        activation.hasParameters ? 1 : 0
    }
    
    public func setTraining(_ training: Bool) {
        isTraining = training
    }
    
    public func zeroGradients() async {
        // Activation layers typically don't accumulate gradients
        // This is a no-op for most activations
    }
    
    public func scaleGradients(_ scale: Float) async {
        // Activation layers typically don't store gradients
        // This is a no-op for most activations
    }
    
    public func updateParametersWithOptimizer(_ optimizer: any Optimizer) async throws {
        // Most activations don't have parameters
        // For parametric activations, we would update the parameter here
        if activation.hasParameters {
            // In a full implementation with gradient tracking, we would:
            // 1. Get accumulated parameter gradients
            // 2. Apply optimizer update rule
            // 3. Update the parameter value
            // For now, this is a placeholder
        }
    }
    
    // MARK: - Private Methods
    
    private func loadPipelines() async throws {
        let library = await metalPipeline.getShaderLibrary()
        
        // Load forward pipeline
        forwardPipeline = try await library.makeComputePipeline(
            functionName: activation.forwardFunctionName
        )
        
        // Load backward pipeline
        backwardPipeline = try await library.makeComputePipeline(
            functionName: activation.backwardFunctionName
        )
    }
}


// MARK: - Errors

public enum ActivationError: LocalizedError {
    case pipelineNotLoaded
    case backwardBeforeForward
    case unsupportedActivation(String)
    
    public var errorDescription: String? {
        switch self {
        case .pipelineNotLoaded:
            return "Metal pipeline not loaded for activation"
        case .backwardBeforeForward:
            return "Backward pass called before forward pass"
        case .unsupportedActivation(let name):
            return "Unsupported activation function: \(name)"
        }
    }
}

// MARK: - Compatibility

/// Compatibility typealias
public typealias Activation = ActivationType
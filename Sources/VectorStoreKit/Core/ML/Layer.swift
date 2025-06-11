// VectorStoreKit: Neural Network Layers
//
// Modern actor-based layer system with Metal acceleration

import Foundation
import Accelerate
@preconcurrency import Metal

/// Base protocol for neural network layers
public protocol NeuralLayer: Actor {
    /// Forward pass through the layer
    func forward(_ input: MetalBuffer) async throws -> MetalBuffer
    
    /// Backward pass through the layer
    func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer
    
    /// Update layer parameters using gradients
    func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws
    
    /// Get current parameters as Metal buffer
    func getParameters() async -> MetalBuffer?
    
    /// Number of trainable parameters
    func getParameterCount() async -> Int
    
    /// Set training mode
    func setTraining(_ training: Bool) async
}

/// Container for layer gradients
public struct LayerGradients: Sendable {
    public let weights: [Float]?
    public let bias: [Float]?
    public let input: [Float]?
    public let batchNormGamma: [Float]?
    public let batchNormBeta: [Float]?
    
    public init(
        weights: [Float]? = nil,
        bias: [Float]? = nil,
        input: [Float]? = nil,
        batchNormGamma: [Float]? = nil,
        batchNormBeta: [Float]? = nil
    ) {
        self.weights = weights
        self.bias = bias
        self.input = input
        self.batchNormGamma = batchNormGamma
        self.batchNormBeta = batchNormBeta
    }
}

/// Base neural layer with common functionality
public actor BaseNeuralLayer {
    // MARK: - Properties
    public let metalPipeline: MetalMLPipeline
    public private(set) var isTraining: Bool = true
    private let parameterStore: ParameterStore
    
    // MARK: - Initialization
    public init(metalPipeline: MetalMLPipeline) async throws {
        self.metalPipeline = metalPipeline
        self.parameterStore = await ParameterStore(device: metalPipeline.device)
    }
    
    // MARK: - Common Methods
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
    
    public func allocateBuffer(size: Int) async throws -> MetalBuffer {
        try await metalPipeline.allocateBuffer(size: size)
    }
    
    public func getParameterStore() async -> ParameterStore {
        parameterStore
    }
}

/// Shape information for tensors
public struct TensorShape: Sendable, Equatable {
    public let dimensions: [Int]
    
    public init(_ dimensions: Int...) {
        self.dimensions = dimensions
    }
    
    public init(dimensions: [Int]) {
        self.dimensions = dimensions
    }
    
    public var rank: Int { dimensions.count }
    
    public var count: Int { dimensions.reduce(1, *) }
    
    public func isCompatible(with other: TensorShape) -> Bool {
        return dimensions == other.dimensions
    }
    
    public func broadcasted(with other: TensorShape) -> TensorShape? {
        // Simple broadcasting rules
        guard rank == other.rank else { return nil }
        var result: [Int] = []
        for (d1, d2) in zip(dimensions, other.dimensions) {
            if d1 == d2 {
                result.append(d1)
            } else if d1 == 1 {
                result.append(d2)
            } else if d2 == 1 {
                result.append(d1)
            } else {
                return nil
            }
        }
        return TensorShape(dimensions: result)
    }
}

/// Metal buffer wrapper for GPU memory with shape information
public struct MetalBuffer: Sendable {
    public let buffer: MTLBuffer
    public let shape: TensorShape
    public let stride: Int
    
    public init(buffer: MTLBuffer, shape: TensorShape, stride: Int = MemoryLayout<Float>.stride) {
        self.buffer = buffer
        self.shape = shape
        self.stride = stride
    }
    
    public init(buffer: MTLBuffer, count: Int, stride: Int = MemoryLayout<Float>.stride) {
        self.buffer = buffer
        self.shape = TensorShape(count)
        self.stride = stride
    }
    
    public var count: Int { shape.count }
    
    public var byteLength: Int {
        count * stride
    }
    
    public func reshaped(to newShape: TensorShape) -> MetalBuffer? {
        guard newShape.count == shape.count else { return nil }
        return MetalBuffer(buffer: buffer, shape: newShape, stride: stride)
    }
}

/// Centralized parameter storage with Metal buffer management
public actor ParameterStore {
    private let device: MTLDevice
    private var parameters: [String: MetalBuffer] = [:]
    private var gradients: [String: MetalBuffer] = [:]
    
    public init(device: MTLDevice) {
        self.device = device
    }
    
    public func allocateParameter(name: String, size: Int) throws -> MetalBuffer {
        guard let buffer = device.makeBuffer(length: size * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: size)
        }
        let metalBuffer = MetalBuffer(buffer: buffer, count: size)
        parameters[name] = metalBuffer
        return metalBuffer
    }
    
    public func getParameter(name: String) -> MetalBuffer? {
        parameters[name]
    }
    
    public func allocateGradient(name: String, size: Int) throws -> MetalBuffer {
        guard let buffer = device.makeBuffer(length: size * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: size)
        }
        let metalBuffer = MetalBuffer(buffer: buffer, count: size)
        gradients[name] = metalBuffer
        return metalBuffer
    }
    
    public func getGradient(name: String) -> MetalBuffer? {
        gradients[name]
    }
    
    public func updateParameter(name: String, with gradient: MetalBuffer, learningRate: Float) async throws {
        guard let parameter = parameters[name] else {
            throw MetalMLError.parameterNotFound(name: name)
        }
        
        // Use Metal kernel for parameter update
        guard let commandBuffer = device.makeCommandQueue()?.makeCommandBuffer() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        // Try to get shader library from device
        let library = device.makeDefaultLibrary()
        let pipeline: MTLComputePipelineState
        
        do {
            guard let library = library,
                  let function = library.makeFunction(name: "sgd_update") else {
                // Fallback to CPU implementation if Metal shader not available
                let paramPtr = parameter.buffer.contents().bindMemory(to: Float.self, capacity: parameter.count)
                let gradPtr = gradient.buffer.contents().bindMemory(to: Float.self, capacity: gradient.count)
                
                for i in 0..<parameter.count {
                    paramPtr[i] -= learningRate * gradPtr[i]
                }
                return
            }
            pipeline = try await device.makeComputePipelineState(function: function)
        } catch {
            // Fallback to CPU implementation if pipeline creation fails
            let paramPtr = parameter.buffer.contents().bindMemory(to: Float.self, capacity: parameter.count)
            let gradPtr = gradient.buffer.contents().bindMemory(to: Float.self, capacity: gradient.count)
            
            for i in 0..<parameter.count {
                paramPtr[i] -= learningRate * gradPtr[i]
            }
            return
        }
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(parameter.buffer, offset: 0, index: 0)
        encoder.setBuffer(gradient.buffer, offset: 0, index: 1)
        
        var lr = learningRate
        var count = UInt32(parameter.count)
        encoder.setBytes(&lr, length: MemoryLayout<Float>.size, index: 2)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (parameter.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    public func getAllParameters() -> [String: MetalBuffer] {
        parameters
    }
}

/// Metal ML Pipeline for compute operations
public actor MetalMLPipeline {
    public let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let shaderLibrary: MLShaderLibrary
    private let operations: MetalMLOperations
    private var bufferCache: [Int: [MetalBuffer]] = [:]
    
    public init(device: MTLDevice) throws {
        self.device = device
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
        self.shaderLibrary = try MLShaderLibrary(device: device)
        self.operations = MetalMLOperations(
            device: device,
            commandQueue: commandQueue,
            shaderLibrary: shaderLibrary
        )
    }
    
    public func allocateBuffer(size: Int) throws -> MetalBuffer {
        // Check cache first
        let alignedSize = nextPowerOfTwo(size)
        if var cached = bufferCache[alignedSize], !cached.isEmpty {
            let buffer = cached.removeLast()
            bufferCache[alignedSize] = cached
            return buffer
        }
        
        // Allocate new buffer
        guard let buffer = device.makeBuffer(length: alignedSize * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: alignedSize)
        }
        return MetalBuffer(buffer: buffer, count: size)
    }
    
    public func allocateBuffer(shape: TensorShape) throws -> MetalBuffer {
        let size = shape.count
        let alignedSize = nextPowerOfTwo(size)
        
        // Try to reuse cached buffer with matching size
        if var cached = bufferCache[alignedSize], !cached.isEmpty {
            let buffer = cached.removeLast()
            bufferCache[alignedSize] = cached
            // Create new MetalBuffer with the requested shape
            return MetalBuffer(buffer: buffer.buffer, shape: shape)
        }
        
        // Allocate new buffer
        guard let buffer = device.makeBuffer(length: alignedSize * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: alignedSize)
        }
        return MetalBuffer(buffer: buffer, shape: shape)
    }
    
    public func releaseBuffer(_ buffer: MetalBuffer) {
        let alignedSize = nextPowerOfTwo(buffer.count)
        if bufferCache[alignedSize] == nil {
            bufferCache[alignedSize] = []
        }
        bufferCache[alignedSize]?.append(buffer)
    }
    
    public func getShaderLibrary() -> MLShaderLibrary {
        shaderLibrary
    }
    
    public func getCommandQueue() -> MTLCommandQueue {
        commandQueue
    }
    
    public func getOperations() -> MetalMLOperations {
        operations
    }
    
    private func nextPowerOfTwo(_ n: Int) -> Int {
        var power = 1
        while power < n {
            power *= 2
        }
        return power
    }
}



/// Metal ML Errors
public enum MetalMLError: LocalizedError {
    case bufferAllocationFailed(size: Int)
    case commandQueueCreationFailed
    case parameterNotFound(name: String)
    case incompatibleBufferSize(expected: Int, actual: Int)
    case shaderCompilationFailed(String)
    case invalidArchitecture(String)
    
    public var errorDescription: String? {
        switch self {
        case .bufferAllocationFailed(let size):
            return "Failed to allocate Metal buffer of size \(size)"
        case .commandQueueCreationFailed:
            return "Failed to create Metal command queue"
        case .parameterNotFound(let name):
            return "Parameter '\(name)' not found in store"
        case .incompatibleBufferSize(let expected, let actual):
            return "Buffer size mismatch: expected \(expected), got \(actual)"
        case .shaderCompilationFailed(let message):
            return "Shader compilation failed: \(message)"
        case .invalidArchitecture(let message):
            return "Invalid architecture: \(message)"
        }
    }
}
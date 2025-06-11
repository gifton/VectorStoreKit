// VectorStoreKit: Layer Composition Utilities
//
// Utilities for composing layers including residual connections and sequential models

import Foundation
@preconcurrency import Metal

/// Residual connection wrapper for any layer
public actor ResidualBlock: NeuralLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let operations: MetalMLOperations
    private let layer: any NeuralLayer
    private let downsample: (any NeuralLayer)?
    
    // MARK: - Initialization
    
    public init(
        layer: any NeuralLayer,
        downsample: (any NeuralLayer)? = nil,
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.layer = layer
        self.downsample = downsample
        self.metalPipeline = metalPipeline
        self.operations = await metalPipeline.getOperations()
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Forward through main layer
        let output = try await layer.forward(input)
        
        // Get residual (possibly downsampled)
        let residual: MetalBuffer
        if let downsample = downsample {
            residual = try await downsample.forward(input)
        } else {
            residual = input
        }
        
        // Add residual connection
        let result = try await addBuffers(output, residual)
        
        return result
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        // Gradient flows through both paths
        let gradMain = try await layer.backward(gradOutput)
        
        let gradResidual: MetalBuffer
        if let downsample = downsample {
            gradResidual = try await downsample.backward(gradOutput)
        } else {
            gradResidual = gradOutput
        }
        
        // Sum gradients
        let gradInput = try await addBuffers(gradMain, gradResidual)
        
        return gradInput
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        try await layer.updateParameters(gradients, learningRate: learningRate)
        if let downsample = downsample {
            try await downsample.updateParameters(gradients, learningRate: learningRate)
        }
    }
    
    public func getParameters() async -> MetalBuffer? {
        await layer.getParameters()
    }
    
    public func getParameterCount() async -> Int {
        let mainCount = await layer.getParameterCount()
        let downsampleCount = await downsample?.getParameterCount() ?? 0
        return mainCount + downsampleCount
    }
    
    public func setTraining(_ training: Bool) async {
        await layer.setTraining(training)
        await downsample?.setTraining(training)
    }
    
    // MARK: - Helper Methods
    
    private func addBuffers(_ a: MetalBuffer, _ b: MetalBuffer) async throws -> MetalBuffer {
        guard a.count == b.count else {
            throw MetalMLError.incompatibleBufferSize(expected: a.count, actual: b.count)
        }
        
        let output = try await metalPipeline.allocateBuffer(size: a.count)
        
        // Placeholder - would use Metal kernel for element-wise addition
        let aPtr = a.buffer.contents().bindMemory(to: Float.self, capacity: a.count)
        let bPtr = b.buffer.contents().bindMemory(to: Float.self, capacity: b.count)
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        
        for i in 0..<a.count {
            outPtr[i] = aPtr[i] + bPtr[i]
        }
        
        return output
    }
}

/// Sequential container for layers
public actor Sequential: NeuralLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private var layers: [any NeuralLayer]
    
    // MARK: - Initialization
    
    public init(
        layers: [any NeuralLayer] = [],
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.layers = layers
        self.metalPipeline = metalPipeline
    }
    
    public init(
        _ layerBuilders: [(any NeuralLayer)],
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.layers = layerBuilders
        self.metalPipeline = metalPipeline
    }
    
    // MARK: - Layer Management
    
    public func add(_ layer: any NeuralLayer) async {
        layers.append(layer)
    }
    
    public func addLayers(_ newLayers: [any NeuralLayer]) async {
        layers.append(contentsOf: newLayers)
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        var output = input
        
        for layer in layers {
            output = try await layer.forward(output)
        }
        
        return output
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        var grad = gradOutput
        
        // Backward through layers in reverse order
        for layer in layers.reversed() {
            grad = try await layer.backward(grad)
        }
        
        return grad
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        for layer in layers {
            try await layer.updateParameters(gradients, learningRate: learningRate)
        }
    }
    
    public func getParameters() async -> MetalBuffer? {
        // Return first layer's parameters as representative
        guard !layers.isEmpty else { return nil }
        return await layers[0].getParameters()
    }
    
    public func getParameterCount() async -> Int {
        var count = 0
        for layer in layers {
            count += await layer.getParameterCount()
        }
        return count
    }
    
    public func setTraining(_ training: Bool) async {
        for layer in layers {
            await layer.setTraining(training)
        }
    }
}

/// Parallel container for layers (ensemble)
public actor Parallel: NeuralLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let operations: MetalMLOperations
    private let layers: [any NeuralLayer]
    private let aggregation: AggregationType
    
    public enum AggregationType: Sendable {
        case sum
        case mean
        case concat
        case max
    }
    
    // MARK: - Initialization
    
    public init(
        layers: [any NeuralLayer],
        aggregation: AggregationType = .sum,
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.layers = layers
        self.aggregation = aggregation
        self.metalPipeline = metalPipeline
        self.operations = await metalPipeline.getOperations()
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Forward through all layers in parallel
        let outputs = try await withThrowingTaskGroup(of: MetalBuffer.self) { group in
            for layer in layers {
                group.addTask {
                    try await layer.forward(input)
                }
            }
            
            var results: [MetalBuffer] = []
            for try await output in group {
                results.append(output)
            }
            return results
        }
        
        // Aggregate outputs
        return try await aggregate(outputs)
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        // Backward through aggregation
        let gradOutputs = try await disaggregate(gradOutput)
        
        // Backward through all layers in parallel
        let gradInputs = try await withThrowingTaskGroup(of: MetalBuffer.self) { group in
            for (layer, grad) in zip(layers, gradOutputs) {
                group.addTask {
                    try await layer.backward(grad)
                }
            }
            
            var results: [MetalBuffer] = []
            for try await gradInput in group {
                results.append(gradInput)
            }
            return results
        }
        
        // Sum input gradients
        return try await sumBuffers(gradInputs)
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        for layer in layers {
            try await layer.updateParameters(gradients, learningRate: learningRate)
        }
    }
    
    public func getParameters() async -> MetalBuffer? {
        guard !layers.isEmpty else { return nil }
        return await layers[0].getParameters()
    }
    
    public func getParameterCount() async -> Int {
        var count = 0
        for layer in layers {
            count += await layer.getParameterCount()
        }
        return count
    }
    
    public func setTraining(_ training: Bool) async {
        for layer in layers {
            await layer.setTraining(training)
        }
    }
    
    // MARK: - Helper Methods
    
    private func aggregate(_ outputs: [MetalBuffer]) async throws -> MetalBuffer {
        guard !outputs.isEmpty else {
            throw MetalMLError.invalidArchitecture("No outputs to aggregate")
        }
        
        switch aggregation {
        case .sum:
            return try await sumBuffers(outputs)
            
        case .mean:
            let sum = try await sumBuffers(outputs)
            return try await scaleBuffer(sum, by: 1.0 / Float(outputs.count))
            
        case .concat:
            return try await concatenateBuffers(outputs)
            
        case .max:
            return try await maxBuffers(outputs)
        }
    }
    
    private func disaggregate(_ gradOutput: MetalBuffer) async throws -> [MetalBuffer] {
        switch aggregation {
        case .sum, .mean:
            // Gradient flows equally to all branches
            let scale: Float = aggregation == .mean ? 1.0 / Float(layers.count) : 1.0
            var results: [MetalBuffer] = []
            for _ in 0..<layers.count {
                let scaled = try await self.scaleBuffer(gradOutput, by: scale)
                results.append(scaled)
            }
            return results
            
        case .concat:
            // Split gradient
            return try await splitBuffer(gradOutput, parts: layers.count)
            
        case .max:
            // Gradient flows only to max branches (placeholder)
            return Array(repeating: gradOutput, count: layers.count)
        }
    }
    
    private func sumBuffers(_ buffers: [MetalBuffer]) async throws -> MetalBuffer {
        guard let first = buffers.first else {
            throw MetalMLError.invalidArchitecture("No buffers to sum")
        }
        
        let output = try await metalPipeline.allocateBuffer(size: first.count)
        
        // Placeholder - would use Metal kernel
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        
        // Initialize to zero
        for i in 0..<output.count {
            outPtr[i] = 0
        }
        
        // Sum all buffers
        for buffer in buffers {
            let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
            for i in 0..<buffer.count {
                outPtr[i] += ptr[i]
            }
        }
        
        return output
    }
    
    private func scaleBuffer(_ buffer: MetalBuffer, by scale: Float) async throws -> MetalBuffer {
        let output = try await metalPipeline.allocateBuffer(size: buffer.count)
        
        let inPtr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        
        for i in 0..<buffer.count {
            outPtr[i] = inPtr[i] * scale
        }
        
        return output
    }
    
    private func concatenateBuffers(_ buffers: [MetalBuffer]) async throws -> MetalBuffer {
        let totalSize = buffers.reduce(0) { $0 + $1.count }
        let output = try await metalPipeline.allocateBuffer(size: totalSize)
        
        var offset = 0
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        
        for buffer in buffers {
            let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
            for i in 0..<buffer.count {
                outPtr[offset + i] = ptr[i]
            }
            offset += buffer.count
        }
        
        return output
    }
    
    private func splitBuffer(_ buffer: MetalBuffer, parts: Int) async throws -> [MetalBuffer] {
        let partSize = buffer.count / parts
        var results: [MetalBuffer] = []
        
        let inPtr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
        
        for p in 0..<parts {
            let part = try await metalPipeline.allocateBuffer(size: partSize)
            let partPtr = part.buffer.contents().bindMemory(to: Float.self, capacity: partSize)
            
            let offset = p * partSize
            for i in 0..<partSize {
                partPtr[i] = inPtr[offset + i]
            }
            
            results.append(part)
        }
        
        return results
    }
    
    private func maxBuffers(_ buffers: [MetalBuffer]) async throws -> MetalBuffer {
        guard let first = buffers.first else {
            throw MetalMLError.invalidArchitecture("No buffers for max")
        }
        
        let output = try await metalPipeline.allocateBuffer(size: first.count)
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        
        // Initialize with first buffer
        let firstPtr = first.buffer.contents().bindMemory(to: Float.self, capacity: first.count)
        for i in 0..<first.count {
            outPtr[i] = firstPtr[i]
        }
        
        // Take element-wise maximum
        for buffer in buffers.dropFirst() {
            let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
            for i in 0..<buffer.count {
                outPtr[i] = max(outPtr[i], ptr[i])
            }
        }
        
        return output
    }
}

// MARK: - Extensions

extension Array where Element == any NeuralLayer {
    /// Create a sequential model from an array of layers
    public func sequential(metalPipeline: MetalMLPipeline) async throws -> Sequential {
        try await Sequential(layers: self, metalPipeline: metalPipeline)
    }
    
    /// Create a parallel model from an array of layers
    public func parallel(
        aggregation: Parallel.AggregationType = .sum,
        metalPipeline: MetalMLPipeline
    ) async throws -> Parallel {
        try await Parallel(layers: self, aggregation: aggregation, metalPipeline: metalPipeline)
    }
}

extension Array {
    func asyncMap<T>(_ transform: @Sendable (Element) async throws -> T) async rethrows -> [T] {
        var results: [T] = []
        results.reserveCapacity(count)
        
        for element in self {
            let result = try await transform(element)
            results.append(result)
        }
        
        return results
    }
}
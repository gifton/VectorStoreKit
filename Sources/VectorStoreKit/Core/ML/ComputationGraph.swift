// VectorStoreKit: Computation Graph
//
// Automatic differentiation and gradient tracking

import Foundation
@preconcurrency import Metal

/// Node in the computation graph
public actor GraphNode {
    public let id: UUID
    public let operation: GraphOperation
    public private(set) var inputs: [GraphNode]
    public private(set) var output: MetalBuffer?
    public private(set) var gradient: MetalBuffer?
    private let metalPipeline: MetalMLPipeline
    
    public init(
        operation: GraphOperation,
        inputs: [GraphNode] = [],
        metalPipeline: MetalMLPipeline
    ) {
        self.id = UUID()
        self.operation = operation
        self.inputs = inputs
        self.metalPipeline = metalPipeline
    }
    
    public func setOutput(_ buffer: MetalBuffer) {
        self.output = buffer
    }
    
    public func setGradient(_ buffer: MetalBuffer) {
        self.gradient = buffer
    }
    
    public func backward() async throws -> [MetalBuffer] {
        guard let gradient = self.gradient else {
            throw MetalMLError.incompatibleBufferSize(expected: 0, actual: 0)
        }
        
        // Compute gradients with respect to inputs
        var inputBuffers: [MetalBuffer] = []
        for node in inputs {
            if let output = await node.output {
                inputBuffers.append(output)
            }
        }
        return try await operation.backward(gradient, inputs: inputBuffers, metalPipeline: metalPipeline)
    }
}

/// Operation types in the computation graph
public enum GraphOperation: Sendable {
    case input
    case matmul
    case add
    case activation(Activation)
    case loss(LossType)
    case custom(String)
    
    func backward(_ gradOutput: MetalBuffer, inputs: [MetalBuffer], metalPipeline: MetalMLPipeline) async throws -> [MetalBuffer] {
        switch self {
        case .input:
            return [gradOutput]
            
        case .matmul:
            // For C = A * B, compute:
            // gradA = gradC * B^T
            // gradB = A^T * gradC
            guard inputs.count >= 2 else {
                throw MetalMLError.incompatibleBufferSize(expected: 2, actual: inputs.count)
            }
            
            let _ = await metalPipeline.getOperations()
            let shaderLibrary = await metalPipeline.getShaderLibrary()
            
            // Assuming square matrices for now - would need shape info
            let size = Int(sqrt(Double(inputs[0].count)))
            
            // Compute gradA = gradC * B^T
            let gradA = try await metalPipeline.allocateBuffer(size: inputs[0].count)
            let gradAPipeline = try shaderLibrary.pipeline(for: "matmul_backward_A")
            
            // Compute gradB = A^T * gradC  
            let gradB = try await metalPipeline.allocateBuffer(size: inputs[1].count)
            let gradBPipeline = try shaderLibrary.pipeline(for: "matmul_backward_B")
            
            // Execute gradient computations
            guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            // gradA computation
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(gradAPipeline)
                encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
                encoder.setBuffer(inputs[1].buffer, offset: 0, index: 1)
                encoder.setBuffer(gradA.buffer, offset: 0, index: 2)
                
                var M = UInt32(size)
                var N = UInt32(size)
                var K = UInt32(size)
                encoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.setBytes(&K, length: MemoryLayout<UInt32>.size, index: 5)
                
                let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
                let threadgroups = MTLSize(
                    width: (size + 15) / 16,
                    height: (size + 15) / 16,
                    depth: 1
                )
                encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
            }
            
            // gradB computation
            if let encoder = commandBuffer.makeComputeCommandEncoder() {
                encoder.setComputePipelineState(gradBPipeline)
                encoder.setBuffer(inputs[0].buffer, offset: 0, index: 0)
                encoder.setBuffer(gradOutput.buffer, offset: 0, index: 1)
                encoder.setBuffer(gradB.buffer, offset: 0, index: 2)
                
                var M = UInt32(size)
                var N = UInt32(size)
                var K = UInt32(size)
                encoder.setBytes(&M, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&N, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.setBytes(&K, length: MemoryLayout<UInt32>.size, index: 5)
                
                let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
                let threadgroups = MTLSize(
                    width: (size + 15) / 16,
                    height: (size + 15) / 16,
                    depth: 1
                )
                encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
            }
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            return [gradA, gradB]
            
        case .add:
            // Gradient for addition is just pass-through
            return [gradOutput, gradOutput]
            
        case .activation(let type):
            // Activation-specific gradients
            guard let input = inputs.first else {
                throw MetalMLError.incompatibleBufferSize(expected: 1, actual: 0)
            }
            
            let operations = await metalPipeline.getOperations()
            let gradInput = try await metalPipeline.allocateBuffer(size: input.count)
            
            // Use Metal shader for activation derivative
            try await operations.applyActivationDerivative(
                gradOutput: gradOutput,
                input: input,
                output: input, // For some activations we need the output
                activation: type,
                gradInput: gradInput
            )
            
            return [gradInput]
            
        case .loss:
            // Loss gradients
            return [gradOutput]
            
        case .custom:
            return [gradOutput]
        }
    }
}

/// Computation graph for automatic differentiation
public actor ComputationGraph {
    private var nodes: [UUID: GraphNode] = [:]
    private var tapeOrder: [UUID] = []
    private let metalPipeline: MetalMLPipeline
    
    public init(metalPipeline: MetalMLPipeline) {
        self.metalPipeline = metalPipeline
    }
    
    /// Create an input node
    public func input(buffer: MetalBuffer) async -> GraphNode {
        let node = GraphNode(operation: .input, metalPipeline: metalPipeline)
        await node.setOutput(buffer)
        nodes[node.id] = node
        tapeOrder.append(node.id)
        return node
    }
    
    /// Create a matrix multiplication node
    public func matmul(_ a: GraphNode, _ b: GraphNode, m: Int, n: Int, k: Int) async throws -> GraphNode {
        let node = GraphNode(operation: .matmul, inputs: [a, b], metalPipeline: metalPipeline)
        nodes[node.id] = node
        tapeOrder.append(node.id)
        
        // Compute forward pass
        guard let aOutput = await a.output,
              let bOutput = await b.output else {
            throw MetalMLError.incompatibleBufferSize(expected: 0, actual: 0)
        }
        
        // Allocate output buffer
        let output = try await metalPipeline.allocateBuffer(size: m * n)
        
        // Perform matrix multiplication using Metal
        let operations = await metalPipeline.getOperations()
        try await operations.matmul(
            aOutput,
            bOutput,
            output: output,
            m: m,
            n: n,
            k: k,
            useTiling: true
        )
        
        await node.setOutput(output)
        
        return node
    }
    
    /// Create an addition node
    public func add(_ a: GraphNode, _ b: GraphNode) async throws -> GraphNode {
        let node = GraphNode(operation: .add, inputs: [a, b], metalPipeline: metalPipeline)
        nodes[node.id] = node
        tapeOrder.append(node.id)
        
        // Compute forward pass
        guard let aOutput = await a.output,
              let bOutput = await b.output else {
            throw MetalMLError.incompatibleBufferSize(expected: 0, actual: 0)
        }
        
        let output = try await metalPipeline.allocateBuffer(size: aOutput.count)
        
        // Perform addition
        let aPtr = aOutput.buffer.contents().bindMemory(to: Float.self, capacity: aOutput.count)
        let bPtr = bOutput.buffer.contents().bindMemory(to: Float.self, capacity: bOutput.count)
        let outPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        
        for i in 0..<aOutput.count {
            outPtr[i] = aPtr[i] + bPtr[i]
        }
        
        await node.setOutput(output)
        return node
    }
    
    /// Create an activation node
    public func activation(_ input: GraphNode, type: Activation) async throws -> GraphNode {
        let node = GraphNode(operation: .activation(type), inputs: [input], metalPipeline: metalPipeline)
        nodes[node.id] = node
        tapeOrder.append(node.id)
        
        // Compute forward pass
        guard let inputBuffer = await input.output else {
            throw MetalMLError.incompatibleBufferSize(expected: 0, actual: 0)
        }
        
        let output = try await applyActivation(inputBuffer, activation: type)
        await node.setOutput(output)
        
        return node
    }
    
    /// Compute backward pass through the graph
    public func backward(from node: GraphNode) async throws {
        // Initialize gradient of output node
        guard let output = await node.output else {
            throw MetalMLError.incompatibleBufferSize(expected: 0, actual: 0)
        }
        
        let initialGradient = try await metalPipeline.allocateBuffer(size: output.count)
        let gradPtr = initialGradient.buffer.contents().bindMemory(to: Float.self, capacity: initialGradient.count)
        
        // Set initial gradient to 1
        for i in 0..<initialGradient.count {
            gradPtr[i] = 1.0
        }
        
        await node.setGradient(initialGradient)
        
        // Traverse tape in reverse order
        for id in tapeOrder.reversed() {
            guard let currentNode = nodes[id],
                  await currentNode.gradient != nil else {
                continue
            }
            
            // Compute gradients for input nodes
            let inputGradients = try await currentNode.backward()
            
            // Accumulate gradients to input nodes
            for (input, grad) in zip(await currentNode.inputs, inputGradients) {
                await accumulateGradient(node: input, gradient: grad)
            }
        }
    }
    
    /// Clear the computation graph
    public func clear() {
        nodes.removeAll()
        tapeOrder.removeAll()
    }
    
    // MARK: - Private Methods
    
    private func accumulateGradient(node: GraphNode, gradient: MetalBuffer) async {
        if let existingGrad = await node.gradient {
            // Accumulate gradients
            let existingPtr = existingGrad.buffer.contents().bindMemory(to: Float.self, capacity: existingGrad.count)
            let newPtr = gradient.buffer.contents().bindMemory(to: Float.self, capacity: gradient.count)
            
            for i in 0..<existingGrad.count {
                existingPtr[i] += newPtr[i]
            }
        } else {
            // First gradient for this node
            await node.setGradient(gradient)
        }
    }
    
    private func applyActivation(_ input: MetalBuffer, activation: Activation) async throws -> MetalBuffer {
        let output = try await metalPipeline.allocateBuffer(size: input.count)
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        
        switch activation {
        case .relu:
            for i in 0..<input.count {
                outputPtr[i] = max(0, inputPtr[i])
            }
        case .sigmoid:
            for i in 0..<input.count {
                outputPtr[i] = 1.0 / (1.0 + exp(-inputPtr[i]))
            }
        case .tanh:
            for i in 0..<input.count {
                outputPtr[i] = tanh(inputPtr[i])
            }
        default:
            memcpy(outputPtr, inputPtr, input.byteLength)
        }
        
        return output
    }
}

/// Loss types for computation graph
public enum LossType: Sendable {
    case mse
    case crossEntropy
    case binaryCrossEntropy
}

/// Gradient tape for automatic differentiation
public actor GradientTape {
    private let graph: ComputationGraph
    private var isRecording: Bool = false
    
    public init(metalPipeline: MetalMLPipeline) {
        self.graph = ComputationGraph(metalPipeline: metalPipeline)
    }
    
    public func startRecording() async {
        isRecording = true
        await graph.clear()
    }
    
    public func stopRecording() {
        isRecording = false
    }
    
    public func reset() async {
        await graph.clear()
        isRecording = false
    }
    
    public func getGraph() -> ComputationGraph {
        graph
    }
}
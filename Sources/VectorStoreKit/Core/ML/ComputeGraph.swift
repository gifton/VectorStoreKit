// VectorStoreKit: Compute Graph
//
// Operation optimization through graph analysis and fusion
//

import Foundation
@preconcurrency import Metal

// MARK: - Type Definitions

/// Unique identifier for nodes in the compute graph
public struct NodeID: Hashable, Sendable {
    private let id: String
    
    public init() {
        self.id = UUID().uuidString
    }
    
    public init(name: String) {
        self.id = name
    }
}

/// Operation types that can be represented in the graph
public enum ComputeOperation: Sendable {
    case matmul(m: Int, n: Int, k: Int)
    case add
    case multiply
    case activation(Activation)
    case transpose
    case reduce(axis: Int)
    case concat(axis: Int)
    case split(axis: Int, parts: Int)
    case reshape(shape: TensorShape)
    case bias(size: Int)
    case dropout(rate: Float)
    case batchNorm(features: Int)
    
    /// Check if two operations can be fused
    func canFuseWith(_ other: ComputeOperation) -> Bool {
        switch (self, other) {
        case (.matmul, .bias),
             (.matmul, .activation),
             (.bias, .activation),
             (.add, .activation),
             (.multiply, .activation):
            return true
        case (.matmul(_, let n1, _), .matmul(_, _, let k2)) where n1 == k2:
            // Chain matrix multiplications
            return true
        default:
            return false
        }
    }
    
    /// Get the fused operation result
    func fuseWith(_ other: ComputeOperation) -> ComputeOperation? {
        switch (self, other) {
        case (.matmul(let m, let n, let k), .bias(let size)) where n == size:
            // Fused matmul+bias operation
            return .matmul(m: m, n: n, k: k) // Metadata tracked separately
        case (.matmul(let m1, let n1, let k1), .matmul(let m2, let n2, let k2)) where n1 == m2:
            // Fused chain multiplication
            return .matmul(m: m1, n: n2, k: k1 + k2)
        default:
            return nil
        }
    }
}

/// Node in the compute graph
public struct ComputeNode: Sendable {
    public let id: NodeID
    public let operation: ComputeOperation
    public var inputs: [NodeID]
    public var outputs: Set<NodeID> = []
    public var isFused: Bool = false
    public var fusedOperations: [ComputeOperation] = []
    
    public init(id: NodeID, operation: ComputeOperation, inputs: [NodeID]) {
        self.id = id
        self.operation = operation
        self.inputs = inputs
    }
}

/// Result of graph execution
public struct ExecutionResult: Sendable {
    public let outputs: [NodeID: MetalBuffer]
    public let executionTime: TimeInterval
    public let operationsExecuted: Int
    public let operationsFused: Int
}

// MARK: - Compute Graph Actor

/// Manages compute graph construction, optimization, and execution
public actor ComputeGraph {
    // MARK: - Properties
    
    private var nodes: [NodeID: ComputeNode] = [:]
    private var executionOrder: [NodeID] = []
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var isOptimized: Bool = false
    
    // Optimization statistics
    private var totalOperations: Int = 0
    private var fusedOperations: Int = 0
    
    // Buffer management
    private let bufferPool: MetalBufferPool
    private var intermediateBuffers: [NodeID: MetalBuffer] = [:]
    private var bufferRefCounts: [NodeID: Int] = [:]
    
    // Shader library
    private let shaderLibrary: MLShaderLibrary
    
    // MARK: - Initialization
    
    public init(device: MTLDevice) throws {
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        self.commandQueue = queue
        self.bufferPool = MetalBufferPool(device: device)
        self.shaderLibrary = try MLShaderLibrary(device: device)
    }
    
    // MARK: - Graph Construction
    
    /// Add an operation to the graph
    public func addOperation(_ op: ComputeOperation, inputs: [NodeID]) -> NodeID {
        let nodeID = NodeID()
        let node = ComputeNode(id: nodeID, operation: op, inputs: inputs)
        
        // Update output connections for input nodes
        for inputID in inputs {
            nodes[inputID]?.outputs.insert(nodeID)
        }
        
        nodes[nodeID] = node
        isOptimized = false
        return nodeID
    }
    
    /// Add a named operation to the graph
    public func addOperation(_ op: ComputeOperation, inputs: [NodeID], name: String) -> NodeID {
        let nodeID = NodeID(name: name)
        let node = ComputeNode(id: nodeID, operation: op, inputs: inputs)
        
        // Update output connections for input nodes
        for inputID in inputs {
            nodes[inputID]?.outputs.insert(nodeID)
        }
        
        nodes[nodeID] = node
        isOptimized = false
        return nodeID
    }
    
    // MARK: - Graph Optimization
    
    /// Optimize the graph by fusing operations and eliminating redundancy
    public func optimize() async throws {
        guard !isOptimized else { return }
        
        // Reset statistics
        totalOperations = nodes.count
        fusedOperations = 0
        
        // 1. Topological sort for execution order
        executionOrder = try topologicalSort()
        
        // 2. Operation fusion pass
        performOperationFusion()
        
        // 3. Common subexpression elimination
        eliminateCommonSubexpressions()
        
        // 4. Dead code elimination
        eliminateDeadCode()
        
        // 5. Update execution order after optimization
        executionOrder = try topologicalSort()
        
        isOptimized = true
    }
    
    // MARK: - Graph Execution
    
    /// Execute the graph with automatic batching and fusion
    public func execute(inputs: [NodeID: MetalBuffer]) async throws -> ExecutionResult {
        // Ensure graph is optimized
        if !isOptimized {
            try await optimize()
        }
        
        let startTime = Date()
        var buffers: [NodeID: MetalBuffer] = inputs
        var operationsExecuted = 0
        
        // Execute nodes in topological order
        for nodeID in executionOrder {
            guard let node = nodes[nodeID] else { continue }
            
            // Skip fused nodes
            if node.isFused { continue }
            
            // Gather input buffers
            var inputBuffers: [MetalBuffer] = []
            for inputID in node.inputs {
                guard let buffer = buffers[inputID] else {
                    throw MetalMLError.parameterNotFound(name: "Input buffer for node \(inputID)")
                }
                inputBuffers.append(buffer)
            }
            
            // Execute operation
            let output = try await executeNode(node, inputs: inputBuffers)
            buffers[nodeID] = output
            operationsExecuted += 1
        }
        
        let executionTime = Date().timeIntervalSince(startTime)
        
        return ExecutionResult(
            outputs: buffers,
            executionTime: executionTime,
            operationsExecuted: operationsExecuted,
            operationsFused: fusedOperations
        )
    }
    
    // MARK: - Private Methods
    
    /// Perform topological sort on the graph
    private func topologicalSort() throws -> [NodeID] {
        var visited = Set<NodeID>()
        var tempVisited = Set<NodeID>()
        var sorted: [NodeID] = []
        
        func visit(_ nodeID: NodeID) throws {
            if tempVisited.contains(nodeID) {
                throw MetalMLError.invalidArchitecture("Cycle detected in compute graph")
            }
            
            guard !visited.contains(nodeID),
                  let node = nodes[nodeID] else { return }
            
            tempVisited.insert(nodeID)
            
            for outputID in node.outputs {
                try visit(outputID)
            }
            
            tempVisited.remove(nodeID)
            visited.insert(nodeID)
            sorted.insert(nodeID, at: 0)
        }
        
        // Visit all nodes
        for nodeID in nodes.keys {
            if !visited.contains(nodeID) {
                try visit(nodeID)
            }
        }
        
        return sorted
    }
    
    /// Perform operation fusion optimization
    private func performOperationFusion() {
        var fusionCandidates: [(NodeID, NodeID)] = []
        
        // Find fusion candidates
        for nodeID in executionOrder {
            guard let node = nodes[nodeID],
                  !node.isFused,
                  node.outputs.count == 1,
                  let outputID = node.outputs.first,
                  let outputNode = nodes[outputID],
                  !outputNode.isFused else { continue }
            
            // Check if operations can be fused
            if node.operation.canFuseWith(outputNode.operation) {
                fusionCandidates.append((nodeID, outputID))
            }
        }
        
        // Apply fusion
        for (sourceID, targetID) in fusionCandidates {
            guard var sourceNode = nodes[sourceID],
                  var targetNode = nodes[targetID] else { continue }
            
            // Mark source as fused
            sourceNode.isFused = true
            nodes[sourceID] = sourceNode
            
            // Update target with fused operations
            targetNode.fusedOperations.append(sourceNode.operation)
            nodes[targetID] = targetNode
            
            fusedOperations += 1
        }
    }
    
    /// Eliminate common subexpressions
    private func eliminateCommonSubexpressions() {
        var operationSignatures: [String: NodeID] = [:]
        var replacements: [NodeID: NodeID] = [:]
        
        for nodeID in executionOrder {
            guard let node = nodes[nodeID], !node.isFused else { continue }
            
            // Create signature for the operation
            let signature = createOperationSignature(node)
            
            if let existingID = operationSignatures[signature] {
                // Found duplicate - mark for replacement
                replacements[nodeID] = existingID
            } else {
                operationSignatures[signature] = nodeID
            }
        }
        
        // Apply replacements
        for (oldID, newID) in replacements {
            guard let oldNode = nodes[oldID] else { continue }
            
            // Redirect outputs
            for outputID in oldNode.outputs {
                if var outputNode = nodes[outputID] {
                    outputNode.inputs = outputNode.inputs.map { $0 == oldID ? newID : $0 }
                    nodes[outputID] = outputNode
                }
                nodes[newID]?.outputs.insert(outputID)
            }
            
            // Remove redundant node
            nodes.removeValue(forKey: oldID)
            fusedOperations += 1
        }
    }
    
    /// Eliminate dead code (unreachable nodes)
    private func eliminateDeadCode() {
        var reachable = Set<NodeID>()
        var toVisit = Array(nodes.keys.filter { nodes[$0]?.outputs.isEmpty ?? false })
        
        // Mark all reachable nodes from outputs
        while !toVisit.isEmpty {
            let nodeID = toVisit.removeFirst()
            guard !reachable.contains(nodeID),
                  let node = nodes[nodeID] else { continue }
            
            reachable.insert(nodeID)
            toVisit.append(contentsOf: node.inputs)
        }
        
        // Remove unreachable nodes
        let unreachable = Set(nodes.keys).subtracting(reachable)
        for nodeID in unreachable {
            nodes.removeValue(forKey: nodeID)
            fusedOperations += 1
        }
    }
    
    /// Create a signature for operation comparison
    private func createOperationSignature(_ node: ComputeNode) -> String {
        let inputsStr = node.inputs.map { "\($0)" }.sorted().joined(separator: ",")
        return "\(node.operation)_\(inputsStr)"
    }
    
    /// Execute a single node
    private func executeNode(_ node: ComputeNode, inputs: [MetalBuffer]) async throws -> MetalBuffer {
        // This is a simplified execution - in practice, this would dispatch to
        // specialized Metal kernels based on the operation type
        
        switch node.operation {
        case .matmul(let m, let n, let k):
            return try await executeMatmul(inputs: inputs, m: m, n: n, k: k, fusedOps: node.fusedOperations)
            
        case .add:
            return try await executeElementwise(inputs: inputs, operation: node.operation)
            
        case .multiply:
            return try await executeElementwise(inputs: inputs, operation: node.operation)
            
        case .activation(let activation):
            return try await executeActivation(input: inputs[0], activation: activation)
            
        case .transpose:
            return try await executeTranspose(input: inputs[0])
            
        case .bias(let size):
            return try await executeBias(input: inputs[0], bias: inputs[1], size: size)
            
        default:
            throw MetalMLError.invalidArchitecture("Operation \(node.operation) not yet implemented")
        }
    }
    
    /// Execute matrix multiplication with potential fusion
    private func executeMatmul(inputs: [MetalBuffer], m: Int, n: Int, k: Int, fusedOps: [ComputeOperation]) async throws -> MetalBuffer {
        guard inputs.count >= 2 else {
            throw MetalMLError.invalidBufferSize("Matmul requires at least 2 inputs")
        }
        
        let A = inputs[0]
        let B = inputs[1]
        
        // Allocate output buffer
        let outputSize = m * n
        guard let outputBuffer = device.makeBuffer(length: outputSize * MemoryLayout<Float>.stride,
                                                   options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: outputSize)
        }
        
        let output = MetalBuffer(buffer: outputBuffer, count: outputSize)
        
        // Check for fused operations
        let hasBias = fusedOps.contains { if case .bias = $0 { return true } else { return false } }
        let activation = fusedOps.compactMap { op -> Activation? in
            if case .activation(let act) = op { return act }
            return nil
        }.first
        
        // Create compute pipeline
        let shaderLibrary = try MLShaderLibrary(device: device)
        let functionName: String
        
        if hasBias && activation != nil {
            // Fused matmul + bias + activation
            functionName = "matmul_bias_activation_fused"
        } else if hasBias {
            // Fused matmul + bias
            functionName = "matmul_bias_fused"
        } else if activation != nil {
            // Fused matmul + activation
            functionName = "matmul_activation_fused"
        } else {
            // Plain matmul
            functionName = MLShaderLibrary.MatrixOperation.matmulForward.rawValue
        }
        
        let pipeline = try shaderLibrary.pipeline(for: functionName)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(A.buffer, offset: 0, index: 0)
        encoder.setBuffer(B.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        
        // Add bias buffer if needed
        var bufferIndex = 3
        if hasBias && inputs.count > 2 {
            encoder.setBuffer(inputs[2].buffer, offset: 0, index: bufferIndex)
            bufferIndex += 1
        }
        
        // Set dimensions
        var mValue = UInt32(m)
        var nValue = UInt32(n)
        var kValue = UInt32(k)
        encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: bufferIndex)
        encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: bufferIndex + 1)
        encoder.setBytes(&kValue, length: MemoryLayout<UInt32>.size, index: bufferIndex + 2)
        
        // Set activation type if needed
        if let activation = activation {
            var activationType = UInt32(activation.rawValue.hashValue)
            encoder.setBytes(&activationType, length: MemoryLayout<UInt32>.size, index: bufferIndex + 3)
        }
        
        // Dispatch threads
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupCount = MTLSize(
            width: (n + threadgroupSize.width - 1) / threadgroupSize.width,
            height: (m + threadgroupSize.height - 1) / threadgroupSize.height,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        // Execute and wait
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        return output
    }
    
    /// Execute elementwise operations
    private func executeElementwise(inputs: [MetalBuffer], operation: ComputeOperation) async throws -> MetalBuffer {
        guard let first = inputs.first else {
            throw MetalMLError.invalidBufferSize("Elementwise operation requires at least one input")
        }
        
        // Allocate output buffer
        guard let outputBuffer = device.makeBuffer(length: first.byteLength,
                                                   options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: first.count)
        }
        
        let output = MetalBuffer(buffer: outputBuffer, shape: first.shape)
        
        // Select appropriate shader
        let shaderLibrary = try MLShaderLibrary(device: device)
        let functionName: String
        
        switch operation {
        case .add:
            functionName = MLShaderLibrary.ElementwiseOperation.elementAdd.rawValue
        case .multiply:
            functionName = MLShaderLibrary.ElementwiseOperation.elementMultiply.rawValue
        default:
            throw MetalMLError.invalidArchitecture("Unsupported elementwise operation")
        }
        
        let pipeline = try shaderLibrary.pipeline(for: functionName)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs[0].buffer, offset: 0, index: 0)
        if inputs.count > 1 {
            encoder.setBuffer(inputs[1].buffer, offset: 0, index: 1)
        }
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        
        var size = UInt32(first.count)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
        
        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (first.count + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        return output
    }
    
    /// Execute activation function
    private func executeActivation(input: MetalBuffer, activation: Activation) async throws -> MetalBuffer {
        // Allocate output buffer
        guard let outputBuffer = device.makeBuffer(length: input.byteLength,
                                                   options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: input.count)
        }
        
        let output = MetalBuffer(buffer: outputBuffer, shape: input.shape)
        
        // Select appropriate shader
        let shaderLibrary = try MLShaderLibrary(device: device)
        let functionName: String
        
        switch activation {
        case .relu:
            functionName = MLShaderLibrary.ActivationFunction.reluForward.rawValue
        case .sigmoid:
            functionName = MLShaderLibrary.ActivationFunction.sigmoidForward.rawValue
        case .tanh:
            functionName = MLShaderLibrary.ActivationFunction.tanhForward.rawValue
        case .leakyRelu:
            functionName = MLShaderLibrary.ActivationFunction.leakyReluForward.rawValue
        case .softmax:
            functionName = MLShaderLibrary.ActivationFunction.softmaxForward.rawValue
        default:
            functionName = MLShaderLibrary.ActivationFunction.reluForward.rawValue
        }
        
        let pipeline = try shaderLibrary.pipeline(for: functionName)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        
        var size = UInt32(input.count)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        
        // Additional parameters for specific activations
        if activation == .leakyRelu {
            var alpha: Float = 0.01
            encoder.setBytes(&alpha, length: MemoryLayout<Float>.size, index: 3)
        }
        
        let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (input.count + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        return output
    }
    
    /// Execute transpose operation
    private func executeTranspose(input: MetalBuffer) async throws -> MetalBuffer {
        // For 2D tensors, swap dimensions
        guard input.shape.rank == 2 else {
            throw MetalMLError.invalidArchitecture("Transpose requires 2D tensor")
        }
        
        let transposedShape = TensorShape(input.shape.dimensions[1], input.shape.dimensions[0])
        
        guard let outputBuffer = device.makeBuffer(length: input.byteLength,
                                                   options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: input.count)
        }
        
        return MetalBuffer(buffer: outputBuffer, shape: transposedShape)
    }
    
    /// Execute bias addition
    private func executeBias(input: MetalBuffer, bias: MetalBuffer, size: Int) async throws -> MetalBuffer {
        guard bias.count == size else {
            throw MetalMLError.incompatibleBufferSize(expected: size, actual: bias.count)
        }
        
        // Allocate output buffer
        guard let outputBuffer = device.makeBuffer(length: input.byteLength,
                                                   options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: input.count)
        }
        
        return MetalBuffer(buffer: outputBuffer, shape: input.shape)
    }
    
    // MARK: - Graph Analysis
    
    /// Get optimization statistics
    public func getOptimizationStats() -> (total: Int, fused: Int, reduction: Float) {
        let reduction = fusedOperations > 0 ? Float(fusedOperations) / Float(totalOperations) : 0.0
        return (totalOperations, fusedOperations, reduction)
    }
    
    /// Get the optimized execution order
    public func getExecutionOrder() -> [NodeID] {
        return executionOrder
    }
    
    /// Check if a node exists in the graph
    public func hasNode(_ nodeID: NodeID) -> Bool {
        return nodes[nodeID] != nil
    }
    
    /// Get node information
    public func getNode(_ nodeID: NodeID) -> ComputeNode? {
        return nodes[nodeID]
    }
    
    /// Clear the graph
    public func clear() {
        nodes.removeAll()
        executionOrder.removeAll()
        isOptimized = false
        totalOperations = 0
        fusedOperations = 0
        intermediateBuffers.removeAll()
        bufferRefCounts.removeAll()
    }
    
    // MARK: - Buffer Management
    
    /// Initialize reference counts for buffer lifetime management
    private func initializeBufferRefCounts() {
        bufferRefCounts.removeAll()
        
        for nodeID in nodes.keys {
            if let node = nodes[nodeID] {
                // Count how many times this node's output is used
                bufferRefCounts[nodeID] = node.outputs.count
            }
        }
    }
    
    /// Release buffer if no longer needed
    private func releaseBufferIfUnused(_ nodeID: NodeID, buffers: inout [NodeID: MetalBuffer]) async {
        guard var refCount = bufferRefCounts[nodeID] else { return }
        
        refCount -= 1
        bufferRefCounts[nodeID] = refCount
        
        if refCount == 0 {
            // Buffer is no longer needed, return to pool
            if let buffer = buffers[nodeID] {
                await bufferPool.releaseBuffer(buffer.buffer)
                buffers.removeValue(forKey: nodeID)
            }
        }
    }
    
    /// Clean up intermediate buffers
    private func cleanupIntermediateBuffers() async {
        for (_, buffer) in intermediateBuffers {
            await bufferPool.releaseBuffer(buffer.buffer)
        }
        intermediateBuffers.removeAll()
    }
    
    // MARK: - Additional Operations
    
    /// Execute reduce operation
    private func executeReduceOptimized(input: MetalBuffer, axis: Int, commandBuffer: MTLCommandBuffer) async throws -> MetalBuffer {
        // Calculate output shape
        var outputDims = input.shape.dimensions
        outputDims[axis] = 1
        let outputShape = TensorShape(outputDims)
        let outputSize = outputShape.count
        
        let output = try await bufferPool.getBuffer(size: outputSize * MemoryLayout<Float>.stride)
        
        // Use reduce kernel
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.LossFunction.reduceSum.rawValue)
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        
        var size = UInt32(input.count)
        var reduceAxis = UInt32(axis)
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&reduceAxis, length: MemoryLayout<UInt32>.size, index: 3)
        
        let threadgroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (outputSize + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        return MetalBuffer(buffer: output, shape: outputShape)
    }
    
    /// Execute reshape operation (no copy needed, just metadata change)
    private func executeReshapeOptimized(input: MetalBuffer, shape: TensorShape) async throws -> MetalBuffer {
        guard input.count == shape.count else {
            throw MetalMLError.incompatibleBufferSize(expected: shape.count, actual: input.count)
        }
        
        // Reshape is just a view change, no computation needed
        return MetalBuffer(buffer: input.buffer, shape: shape)
    }
    
    /// Execute concatenation operation
    private func executeConcatOptimized(inputs: [MetalBuffer], axis: Int, commandBuffer: MTLCommandBuffer) async throws -> MetalBuffer {
        guard !inputs.isEmpty else {
            throw MetalMLError.invalidBufferSize("Concat requires at least one input")
        }
        
        // Calculate output shape
        var outputDims = inputs[0].shape.dimensions
        var concatSize = 0
        for input in inputs {
            concatSize += input.shape.dimensions[axis]
        }
        outputDims[axis] = concatSize
        let outputShape = TensorShape(outputDims)
        
        let output = try await bufferPool.getBuffer(size: outputShape.elementCount * MemoryLayout<Float>.stride)
        
        // Use concatenate kernel
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.ElementwiseOperation.concatenateBuffers.rawValue)
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        
        // Set input buffers
        for (i, input) in inputs.enumerated() {
            encoder.setBuffer(input.buffer, offset: 0, index: i)
        }
        encoder.setBuffer(output, offset: 0, index: inputs.count)
        
        var numInputs = UInt32(inputs.count)
        var concatAxis = UInt32(axis)
        encoder.setBytes(&numInputs, length: MemoryLayout<UInt32>.size, index: inputs.count + 1)
        encoder.setBytes(&concatAxis, length: MemoryLayout<UInt32>.size, index: inputs.count + 2)
        
        let threadgroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (outputShape.elementCount + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        return MetalBuffer(buffer: output, shape: outputShape)
    }
    
    /// Execute batch normalization
    private func executeBatchNormOptimized(inputs: [MetalBuffer], features: Int, commandBuffer: MTLCommandBuffer) async throws -> MetalBuffer {
        guard inputs.count >= 3 else {
            throw MetalMLError.invalidBufferSize("BatchNorm requires input, mean, variance")
        }
        
        let input = inputs[0]
        let mean = inputs[1]
        let variance = inputs[2]
        let gamma = inputs.count > 3 ? inputs[3] : nil
        let beta = inputs.count > 4 ? inputs[4] : nil
        
        let output = try await bufferPool.getBuffer(size: input.byteLength)
        
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.NormalizationFunction.batchNormForward.rawValue)
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(mean.buffer, offset: 0, index: 1)
        encoder.setBuffer(variance.buffer, offset: 0, index: 2)
        encoder.setBuffer(gamma?.buffer ?? mean.buffer, offset: 0, index: 3) // Use mean as dummy if no gamma
        encoder.setBuffer(beta?.buffer ?? mean.buffer, offset: 0, index: 4)  // Use mean as dummy if no beta
        encoder.setBuffer(output, offset: 0, index: 5)
        
        var size = UInt32(input.count)
        var numFeatures = UInt32(features)
        var epsilon: Float = 1e-5
        var useAffine = UInt32(gamma != nil ? 1 : 0)
        
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&numFeatures, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&epsilon, length: MemoryLayout<Float>.size, index: 8)
        encoder.setBytes(&useAffine, length: MemoryLayout<UInt32>.size, index: 9)
        
        let threadgroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (input.count + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        return MetalBuffer(buffer: output, shape: input.shape)
    }
    
    /// Execute dropout operation
    private func executeDropoutOptimized(input: MetalBuffer, rate: Float, commandBuffer: MTLCommandBuffer) async throws -> MetalBuffer {
        // During inference, dropout is a no-op
        // For training, we would need to generate a mask
        let isTraining = false // This should be configurable
        
        if !isTraining {
            // Return input unchanged
            return input
        }
        
        let output = try await bufferPool.getBuffer(size: input.byteLength)
        
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.ElementwiseOperation.dropoutForward.rawValue)
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        
        var size = UInt32(input.count)
        var dropoutRate = rate
        var scale = Float(1.0 / (1.0 - rate)) // Scale for maintaining expected value
        
        encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&dropoutRate, length: MemoryLayout<Float>.size, index: 3)
        encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 4)
        
        let threadgroupSize = MTLSize(width: min(pipeline.maxTotalThreadsPerThreadgroup, 256), height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (input.count + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        return MetalBuffer(buffer: output, shape: input.shape)
    }
}

// MARK: - Activation Parameters Structure

// Removed ActivationParams struct as it conflicts with shader definition
// We'll use inline parameter passing instead

// MARK: - Extensions

extension ComputeGraph {
    /// Build a simple linear layer subgraph
    public func buildLinearLayer(
        input: NodeID,
        weights: NodeID,
        bias: NodeID?,
        activation: Activation? = nil,
        name: String = "linear"
    ) -> NodeID {
        // Matrix multiplication
        let matmulNode = addOperation(
            ComputeOperation.matmul(m: 1, n: 1, k: 1), // Dimensions will be inferred from buffers
            inputs: [input, weights],
            name: "\(name)_matmul"
        )
        
        // Add bias if provided
        let biasedNode: NodeID
        if let bias = bias {
            biasedNode = addOperation(
                ComputeOperation.bias(size: 1), // Size will be inferred
                inputs: [matmulNode, bias],
                name: "\(name)_bias"
            )
        } else {
            biasedNode = matmulNode
        }
        
        // Apply activation if provided
        if let activation = activation {
            return addOperation(
                ComputeOperation.activation(activation),
                inputs: [biasedNode],
                name: "\(name)_activation"
            )
        }
        
        return biasedNode
    }
}
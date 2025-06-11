// VectorStoreKit: Attention Layers
//
// Multi-head attention and transformer components

import Foundation
@preconcurrency import Metal

/// Multi-Head Attention layer
public actor MultiHeadAttentionLayer: NeuralLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let parameterStore: ParameterStore
    private let operations: MetalMLOperations
    
    // Configuration
    private let embedDim: Int
    private let numHeads: Int
    private let dropout: Float
    private let kdim: Int
    private let vdim: Int
    private let batchFirst: Bool
    private let needWeights: Bool
    
    // Derived dimensions
    private let headDim: Int
    
    // Parameter names
    private let qProjWeight: String
    private let kProjWeight: String
    private let vProjWeight: String
    private let outProjWeight: String
    private let qProjBias: String
    private let kProjBias: String
    private let vProjBias: String
    private let outProjBias: String
    
    // Cached values for backward pass
    private var cachedAttentionWeights: MetalBuffer?
    private var cachedQ: MetalBuffer?
    private var cachedK: MetalBuffer?
    private var cachedV: MetalBuffer?
    private var isTraining: Bool = true
    
    // MARK: - Initialization
    
    public init(
        embedDim: Int,
        numHeads: Int,
        dropout: Float = 0.0,
        bias: Bool = true,
        kdim: Int? = nil,
        vdim: Int? = nil,
        batchFirst: Bool = false,
        needWeights: Bool = true,
        name: String = "multihead_attention",
        metalPipeline: MetalMLPipeline
    ) async throws {
        guard embedDim % numHeads == 0 else {
            throw MetalMLError.incompatibleBufferSize(
                expected: embedDim,
                actual: numHeads
            )
        }
        
        self.embedDim = embedDim
        self.numHeads = numHeads
        self.dropout = dropout
        self.kdim = kdim ?? embedDim
        self.vdim = vdim ?? embedDim
        self.batchFirst = batchFirst
        self.needWeights = needWeights
        self.headDim = embedDim / numHeads
        
        // Initialize parameter names
        self.qProjWeight = "\(name)_q_proj_weight"
        self.kProjWeight = "\(name)_k_proj_weight"
        self.vProjWeight = "\(name)_v_proj_weight"
        self.outProjWeight = "\(name)_out_proj_weight"
        self.qProjBias = "\(name)_q_proj_bias"
        self.kProjBias = "\(name)_k_proj_bias"
        self.vProjBias = "\(name)_v_proj_bias"
        self.outProjBias = "\(name)_out_proj_bias"
        
        self.metalPipeline = metalPipeline
        self.parameterStore = await ParameterStore(device: metalPipeline.device)
        self.operations = await metalPipeline.getOperations()
        
        // Initialize parameters
        try await initializeParameters(useBias: bias)
    }
    
    private func initializeParameters(useBias: Bool) async throws {
        // Query projection: embedDim x embedDim
        let qWeight = try await parameterStore.allocateParameter(
            name: qProjWeight,
            size: embedDim * embedDim
        )
        initializeWeight(qWeight, inputDim: embedDim, outputDim: embedDim)
        
        // Key projection: kdim x embedDim
        let kWeight = try await parameterStore.allocateParameter(
            name: kProjWeight,
            size: kdim * embedDim
        )
        initializeWeight(kWeight, inputDim: kdim, outputDim: embedDim)
        
        // Value projection: vdim x embedDim
        let vWeight = try await parameterStore.allocateParameter(
            name: vProjWeight,
            size: vdim * embedDim
        )
        initializeWeight(vWeight, inputDim: vdim, outputDim: embedDim)
        
        // Output projection: embedDim x embedDim
        let outWeight = try await parameterStore.allocateParameter(
            name: outProjWeight,
            size: embedDim * embedDim
        )
        initializeWeight(outWeight, inputDim: embedDim, outputDim: embedDim)
        
        // Initialize biases if needed
        if useBias {
            for biasName in [qProjBias, kProjBias, vProjBias, outProjBias] {
                let bias = try await parameterStore.allocateParameter(
                    name: biasName,
                    size: embedDim
                )
                let ptr = bias.buffer.contents().bindMemory(to: Float.self, capacity: embedDim)
                for i in 0..<embedDim {
                    ptr[i] = 0
                }
            }
        }
        
        // Allocate gradient buffers
        _ = try await parameterStore.allocateGradient(name: "\(qProjWeight)_grad", size: embedDim * embedDim)
        _ = try await parameterStore.allocateGradient(name: "\(kProjWeight)_grad", size: kdim * embedDim)
        _ = try await parameterStore.allocateGradient(name: "\(vProjWeight)_grad", size: vdim * embedDim)
        _ = try await parameterStore.allocateGradient(name: "\(outProjWeight)_grad", size: embedDim * embedDim)
        
        if useBias {
            for biasName in [qProjBias, kProjBias, vProjBias, outProjBias] {
                _ = try await parameterStore.allocateGradient(name: "\(biasName)_grad", size: embedDim)
            }
        }
    }
    
    private func initializeWeight(_ buffer: MetalBuffer, inputDim: Int, outputDim: Int) {
        let scale = sqrt(1.0 / Float(inputDim))
        let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
        for i in 0..<buffer.count {
            ptr[i] = Float.random(in: -scale...scale)
        }
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Input shape: [seq_len, batch, embed_dim] or [batch, seq_len, embed_dim]
        // Validate input shape
        guard input.shape.rank == 3 else {
            throw MetalMLError.invalidArchitecture("MultiHeadAttention expects 3D input")
        }
        
        let shape = getInputShape(input)
        
        // For self-attention, query = key = value = input
        let query = input
        let key = input
        let value = input
        
        // Project Q, K, V
        let q = try await project(query, weight: qProjWeight, bias: qProjBias)
        let k = try await project(key, weight: kProjWeight, bias: kProjBias)
        let v = try await project(value, weight: vProjWeight, bias: vProjBias)
        
        // Cache for backward pass
        if isTraining {
            cachedQ = q
            cachedK = k
            cachedV = v
        }
        
        // Reshape for multi-head attention
        let (qHeads, kHeads, vHeads) = try await reshapeForHeads(q: q, k: k, v: v, shape: shape)
        
        // Compute scaled dot-product attention
        let (attnOutput, attnWeights) = try await scaledDotProductAttention(
            q: qHeads,
            k: kHeads,
            v: vHeads,
            shape: shape
        )
        
        // Cache attention weights if needed
        if isTraining && needWeights {
            cachedAttentionWeights = attnWeights
        }
        
        // Reshape back and project output
        let concatenated = try await reshapeFromHeads(attnOutput, shape: shape)
        let output = try await project(concatenated, weight: outProjWeight, bias: outProjBias)
        
        return output
    }
    
    // MARK: - Attention Computation
    
    private func scaledDotProductAttention(
        q: MetalBuffer,
        k: MetalBuffer,
        v: MetalBuffer,
        shape: InputShape
    ) async throws -> (output: MetalBuffer, weights: MetalBuffer) {
        // let seqLen = shape.seqLen  // Currently unused
        // let batchSize = shape.batchSize  // Currently unused
        
        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let scores = try await computeAttentionScores(q: q, k: k, scale: 1.0 / sqrt(Float(headDim)))
        
        // Apply softmax to get attention weights
        let weights = try await applySoftmax(scores, dim: -1)
        
        // Apply dropout if training
        let droppedWeights = if isTraining && dropout > 0 {
            try await applyDropout(weights, rate: dropout)
        } else {
            weights
        }
        
        // Compute weighted sum: weights @ V
        let output = try await matmulWeightsValues(weights: droppedWeights, values: v)
        
        return (output, weights)
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let q = cachedQ,
              let k = cachedK,
              let v = cachedV else {
            throw MetalMLError.parameterNotFound(name: "cached attention values")
        }
        
        // Backward through output projection
        let (gradConcat, gradOutWeight, gradOutBias) = try await projectBackward(
            gradOutput: gradOutput,
            input: q,  // Placeholder - should be concatenated attention output
            weightName: outProjWeight
        )
        
        // Store output projection gradients
        if let gradW = await parameterStore.getGradient(name: "\(outProjWeight)_grad") {
            try await accumulate(gradOutWeight, into: gradW)
        }
        if let gradB = gradOutBias,
           let gradBias = await parameterStore.getGradient(name: "\(outProjBias)_grad") {
            try await accumulate(gradB, into: gradBias)
        }
        
        // Backward through attention
        let (gradQ, gradK, gradV) = try await attentionBackward(
            gradOutput: gradConcat,
            q: q, k: k, v: v
        )
        
        // Backward through Q, K, V projections
        let gradInput = try await projectionsBackward(
            gradQ: gradQ, gradK: gradK, gradV: gradV
        )
        
        return gradInput
    }
    
    private func attentionBackward(
        gradOutput: MetalBuffer,
        q: MetalBuffer, k: MetalBuffer, v: MetalBuffer
    ) async throws -> (gradQ: MetalBuffer, gradK: MetalBuffer, gradV: MetalBuffer) {
        // Backward through attention mechanism
        // gradOutput: gradient w.r.t attention output [batch*heads, seq_len, head_dim]
        
        guard let weights = cachedAttentionWeights else {
            throw MetalMLError.parameterNotFound(name: "cached attention weights")
        }
        
        let shape = getInputShape(q)
        let scale = 1.0 / sqrt(Float(headDim))
        
        // Allocate gradient buffers
        let gradQ = try await metalPipeline.allocateBuffer(shape: q.shape)
        let gradK = try await metalPipeline.allocateBuffer(shape: k.shape)
        let gradV = try await metalPipeline.allocateBuffer(shape: v.shape)
        
        // Backward through weighted sum: gradWeights = gradOutput @ V^T
        let gradWeights = try await computeGradWeights(gradOutput: gradOutput, values: v)
        
        // Backward through values: gradV = weights^T @ gradOutput
        try await computeGradValues(weights: weights, gradOutput: gradOutput, gradV: gradV)
        
        // Backward through softmax
        let gradScores = try await softmaxBackward(gradWeights: gradWeights, weights: weights)
        
        // Backward through attention scores
        try await computeGradQK(gradScores: gradScores, q: q, k: k, gradQ: gradQ, gradK: gradK, scale: scale)
        
        return (gradQ, gradK, gradV)
    }
    
    private func computeGradWeights(gradOutput: MetalBuffer, values: MetalBuffer) async throws -> MetalBuffer {
        // Compute gradient w.r.t attention weights: gradOutput @ V^T
        let seqLen = 10  // Would be extracted from metadata
        let gradWeights = try await metalPipeline.allocateBuffer(shape: TensorShape(weights.shape.dimensions[0], seqLen, seqLen))
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "attention_grad_weights")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
        encoder.setBuffer(values.buffer, offset: 0, index: 1)
        encoder.setBuffer(gradWeights.buffer, offset: 0, index: 2)
        
        // Set dimensions
        var batchHeads = UInt32(gradOutput.shape.dimensions[0])
        var seqLenVal = UInt32(seqLen)
        var headDimVal = UInt32(headDim)
        
        encoder.setBytes(&batchHeads, length: 4, index: 3)
        encoder.setBytes(&seqLenVal, length: 4, index: 4)
        encoder.setBytes(&headDimVal, length: 4, index: 5)
        
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (seqLen + 7) / 8,
            height: (seqLen + 7) / 8,
            depth: Int(batchHeads)
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return gradWeights
    }
    
    private func computeGradValues(weights: MetalBuffer, gradOutput: MetalBuffer, gradV: MetalBuffer) async throws {
        // Compute gradient w.r.t values: weights^T @ gradOutput
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "attention_grad_values")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights.buffer, offset: 0, index: 0)
        encoder.setBuffer(gradOutput.buffer, offset: 0, index: 1)
        encoder.setBuffer(gradV.buffer, offset: 0, index: 2)
        
        // Set dimensions from shapes
        let dims = weights.shape.dimensions
        var batchHeads = UInt32(dims[0])
        var seqLen = UInt32(dims[1])
        var headDimVal = UInt32(headDim)
        
        encoder.setBytes(&batchHeads, length: 4, index: 3)
        encoder.setBytes(&seqLen, length: 4, index: 4)
        encoder.setBytes(&headDimVal, length: 4, index: 5)
        
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (Int(headDimVal) + 7) / 8,
            height: (Int(seqLen) + 7) / 8,
            depth: Int(batchHeads)
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func softmaxBackward(gradWeights: MetalBuffer, weights: MetalBuffer) async throws -> MetalBuffer {
        // Backward through softmax: grad * output * (1 - output) for diagonal, -grad * output_i * output_j for off-diagonal
        let gradScores = try await metalPipeline.allocateBuffer(shape: weights.shape)
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "softmax_backward")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradWeights.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer, offset: 0, index: 1)
        encoder.setBuffer(gradScores.buffer, offset: 0, index: 2)
        
        let dims = weights.shape.dimensions
        var numMatrices = UInt32(dims[0])
        var size = UInt32(dims[1])
        
        encoder.setBytes(&numMatrices, length: 4, index: 3)
        encoder.setBytes(&size, length: 4, index: 4)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (Int(numMatrices * size) + 255) / 256,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return gradScores
    }
    
    private func computeGradQK(
        gradScores: MetalBuffer,
        q: MetalBuffer, k: MetalBuffer,
        gradQ: MetalBuffer, gradK: MetalBuffer,
        scale: Float
    ) async throws {
        // Compute gradients w.r.t Q and K
        // gradQ = (gradScores @ K) * scale
        // gradK = (gradScores^T @ Q) * scale
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "attention_grad_qk")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradScores.buffer, offset: 0, index: 0)
        encoder.setBuffer(q.buffer, offset: 0, index: 1)
        encoder.setBuffer(k.buffer, offset: 0, index: 2)
        encoder.setBuffer(gradQ.buffer, offset: 0, index: 3)
        encoder.setBuffer(gradK.buffer, offset: 0, index: 4)
        
        var scaleVal = scale
        encoder.setBytes(&scaleVal, length: 4, index: 5)
        
        // Extract dimensions
        let batchHeads = q.shape.dimensions[0]
        let seqLen = q.shape.dimensions[1]
        var batchHeadsVal = UInt32(batchHeads)
        var seqLenVal = UInt32(seqLen)
        var headDimVal = UInt32(headDim)
        
        encoder.setBytes(&batchHeadsVal, length: 4, index: 6)
        encoder.setBytes(&seqLenVal, length: 4, index: 7)
        encoder.setBytes(&headDimVal, length: 4, index: 8)
        
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (headDim + 7) / 8,
            height: (Int(seqLenVal) + 7) / 8,
            depth: batchHeads
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func projectionsBackward(
        gradQ: MetalBuffer, gradK: MetalBuffer, gradV: MetalBuffer
    ) async throws -> MetalBuffer {
        // Compute gradients through Q, K, V projections
        // For self-attention, input gradient is sum of all three projection gradients
        
        // Backward through Q projection
        let (gradInputQ, gradQWeight, gradQBias) = try await projectBackward(
            gradOutput: gradQ,
            input: lastInput!,
            weightName: qProjWeight
        )
        
        // Store Q projection gradients
        if let gradW = await parameterStore.getGradient(name: "\(qProjWeight)_grad") {
            try await accumulate(gradQWeight, into: gradW)
        }
        if let gradB = gradQBias,
           let gradBias = await parameterStore.getGradient(name: "\(qProjBias)_grad") {
            try await accumulate(gradB, into: gradBias)
        }
        
        // Backward through K projection
        let (gradInputK, gradKWeight, gradKBias) = try await projectBackward(
            gradOutput: gradK,
            input: lastInput!,
            weightName: kProjWeight
        )
        
        // Store K projection gradients
        if let gradW = await parameterStore.getGradient(name: "\(kProjWeight)_grad") {
            try await accumulate(gradKWeight, into: gradW)
        }
        if let gradB = gradKBias,
           let gradBias = await parameterStore.getGradient(name: "\(kProjBias)_grad") {
            try await accumulate(gradB, into: gradBias)
        }
        
        // Backward through V projection
        let (gradInputV, gradVWeight, gradVBias) = try await projectBackward(
            gradOutput: gradV,
            input: lastInput!,
            weightName: vProjWeight
        )
        
        // Store V projection gradients
        if let gradW = await parameterStore.getGradient(name: "\(vProjWeight)_grad") {
            try await accumulate(gradVWeight, into: gradW)
        }
        if let gradB = gradVBias,
           let gradBias = await parameterStore.getGradient(name: "\(vProjBias)_grad") {
            try await accumulate(gradB, into: gradBias)
        }
        
        // Sum input gradients (for self-attention, all projections receive same input)
        let gradInput = try await metalPipeline.allocateBuffer(shape: gradInputQ.shape)
        try await operations.addBuffers(gradInputQ, gradInputK, output: gradInput)
        try await operations.addBuffers(gradInput, gradInputV, output: gradInput)
        
        return gradInput
    }
    
    // MARK: - Parameter Updates
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        // Update all projection weights and biases
        for weightName in [qProjWeight, kProjWeight, vProjWeight, outProjWeight] {
            if let grad = await parameterStore.getGradient(name: "\(weightName)_grad") {
                try await parameterStore.updateParameter(
                    name: weightName,
                    with: grad,
                    learningRate: learningRate
                )
            }
        }
        
        for biasName in [qProjBias, kProjBias, vProjBias, outProjBias] {
            if let grad = await parameterStore.getGradient(name: "\(biasName)_grad") {
                try await parameterStore.updateParameter(
                    name: biasName,
                    with: grad,
                    learningRate: learningRate
                )
            }
        }
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func getParameters() async -> MetalBuffer? {
        await parameterStore.getParameter(name: qProjWeight)
    }
    
    public func getParameterCount() async -> Int {
        let weightCount = embedDim * embedDim + kdim * embedDim + vdim * embedDim + embedDim * embedDim
        let biasCount = embedDim * 4  // 4 biases if used
        return weightCount + biasCount
    }
    
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
    
    // MARK: - Helper Types and Methods
    
    private struct InputShape {
        let seqLen: Int
        let batchSize: Int
        let embedDim: Int
    }
    
    private func getInputShape(_ input: MetalBuffer) -> InputShape {
        // Extract from buffer shape
        if input.shape.rank == 3 {
            let dims = input.shape.dimensions
            if batchFirst {
                // [batch, seq_len, embed_dim]
                return InputShape(seqLen: dims[1], batchSize: dims[0], embedDim: dims[2])
            } else {
                // [seq_len, batch, embed_dim]
                return InputShape(seqLen: dims[0], batchSize: dims[1], embedDim: dims[2])
            }
        }
        // Fallback
        return InputShape(seqLen: 10, batchSize: 1, embedDim: embedDim)
    }
    
    private func project(_ input: MetalBuffer, weight: String, bias: String?) async throws -> MetalBuffer {
        guard let _ = await parameterStore.getParameter(name: weight) else {
            throw MetalMLError.parameterNotFound(name: weight)
        }
        
        // Infer output shape based on input shape
        let inputShape = input.shape
        let outputShape = if inputShape.rank == 3 {
            TensorShape(inputShape.dimensions[0], inputShape.dimensions[1], embedDim)
        } else {
            TensorShape(embedDim)
        }
        let output = try await metalPipeline.allocateBuffer(shape: outputShape)
        
        // Placeholder - would use Metal kernel for matrix multiplication
        if let biasName = bias,
           let b = await parameterStore.getParameter(name: biasName) {
            try await operations.addBias(matrix: output, bias: b, rows: 1, cols: embedDim)
        }
        
        return output
    }
    
    private func projectBackward(
        gradOutput: MetalBuffer,
        input: MetalBuffer,
        weightName: String
    ) async throws -> (gradInput: MetalBuffer, gradWeight: MetalBuffer, gradBias: MetalBuffer?) {
        // Placeholder
        let gradInput = try await metalPipeline.allocateBuffer(shape: input.shape)
        let gradWeight = try await metalPipeline.allocateBuffer(shape: TensorShape(embedDim, embedDim))
        let gradBias = try await metalPipeline.allocateBuffer(shape: TensorShape(embedDim))
        
        return (gradInput, gradWeight, gradBias)
    }
    
    private func reshapeForHeads(
        q: MetalBuffer, k: MetalBuffer, v: MetalBuffer,
        shape: InputShape
    ) async throws -> (q: MetalBuffer, k: MetalBuffer, v: MetalBuffer) {
        // Reshape [seq_len, batch, embed_dim] -> [batch * num_heads, seq_len, head_dim]
        // or [batch, seq_len, embed_dim] -> [batch * num_heads, seq_len, head_dim]
        
        let qHeads = try await reshapeToHeads(q, shape: shape)
        let kHeads = try await reshapeToHeads(k, shape: shape)
        let vHeads = try await reshapeToHeads(v, shape: shape)
        
        return (qHeads, kHeads, vHeads)
    }
    
    private func reshapeToHeads(_ tensor: MetalBuffer, shape: InputShape) async throws -> MetalBuffer {
        // Reshape tensor for multi-head attention
        let output = try await metalPipeline.allocateBuffer(size: tensor.count)
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "reshape_for_heads")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(tensor.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        
        var batchSize = UInt32(shape.batchSize)
        var seqLen = UInt32(shape.seqLen)
        var numHeadsVal = UInt32(numHeads)
        var headDimVal = UInt32(headDim)
        var batchFirstVal = UInt32(batchFirst ? 1 : 0)
        
        encoder.setBytes(&batchSize, length: 4, index: 2)
        encoder.setBytes(&seqLen, length: 4, index: 3)
        encoder.setBytes(&numHeadsVal, length: 4, index: 4)
        encoder.setBytes(&headDimVal, length: 4, index: 5)
        encoder.setBytes(&batchFirstVal, length: 4, index: 6)
        
        let totalThreads = tensor.count
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (totalThreads + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return output
    }
    
    private func reshapeFromHeads(_ input: MetalBuffer, shape: InputShape) async throws -> MetalBuffer {
        // Reshape [batch * num_heads, seq_len, head_dim] back to original shape
        let output = try await metalPipeline.allocateBuffer(size: input.count)
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "reshape_from_heads")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        
        var batchSize = UInt32(shape.batchSize)
        var seqLen = UInt32(shape.seqLen)
        var numHeadsVal = UInt32(numHeads)
        var headDimVal = UInt32(headDim)
        var batchFirstVal = UInt32(batchFirst ? 1 : 0)
        
        encoder.setBytes(&batchSize, length: 4, index: 2)
        encoder.setBytes(&seqLen, length: 4, index: 3)
        encoder.setBytes(&numHeadsVal, length: 4, index: 4)
        encoder.setBytes(&headDimVal, length: 4, index: 5)
        encoder.setBytes(&batchFirstVal, length: 4, index: 6)
        
        let totalThreads = input.count
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (totalThreads + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return output
    }
    
    private func computeAttentionScores(q: MetalBuffer, k: MetalBuffer, scale: Float) async throws -> MetalBuffer {
        // Compute Q @ K^T * scale
        // q: [batch * num_heads, seq_len, head_dim]
        // k: [batch * num_heads, seq_len, head_dim]
        // scores: [batch * num_heads, seq_len, seq_len]
        
        // Get dimensions from input shapes
        let seqLen = 10  // Would be extracted from metadata
        let batchHeads = q.count / (seqLen * headDim)
        
        // Allocate score matrix
        let scores = try await metalPipeline.allocateBuffer(size: batchHeads * seqLen * seqLen)
        
        // Use Metal shader for batched matrix multiplication
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "attention_scores")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q.buffer, offset: 0, index: 0)
        encoder.setBuffer(k.buffer, offset: 0, index: 1)
        encoder.setBuffer(scores.buffer, offset: 0, index: 2)
        
        var seqLenVal = UInt32(seqLen)
        var headDimVal = UInt32(headDim)
        var scaleVal = scale
        var batchHeadsVal = UInt32(batchHeads)
        
        encoder.setBytes(&batchHeadsVal, length: 4, index: 3)
        encoder.setBytes(&seqLenVal, length: 4, index: 4)
        encoder.setBytes(&headDimVal, length: 4, index: 5)
        encoder.setBytes(&scaleVal, length: 4, index: 6)
        
        // Launch with appropriate thread configuration
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (seqLen + 7) / 8,
            height: (seqLen + 7) / 8,
            depth: batchHeads
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return scores
    }
    
    private func applySoftmax(_ input: MetalBuffer, dim: Int) async throws -> MetalBuffer {
        // Apply softmax along last dimension (seq_len dimension for attention)
        // input: [batch * num_heads, seq_len, seq_len]
        let output = try await metalPipeline.allocateBuffer(size: input.count)
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "softmax_2d")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        // Infer dimensions
        let seqLen = 10  // Would be extracted from metadata
        let numMatrices = input.count / (seqLen * seqLen)
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        
        var seqLenVal = UInt32(seqLen)
        var numMatricesVal = UInt32(numMatrices)
        
        encoder.setBytes(&numMatricesVal, length: 4, index: 2)
        encoder.setBytes(&seqLenVal, length: 4, index: 3)
        
        // Each thread group processes one row of the attention matrix
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (numMatrices * seqLen + 255) / 256,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return output
    }
    
    private func applyDropout(_ input: MetalBuffer, rate: Float) async throws -> MetalBuffer {
        // Placeholder - would apply dropout
        return input
    }
    
    private func matmulWeightsValues(weights: MetalBuffer, values: MetalBuffer) async throws -> MetalBuffer {
        // Compute attention weights @ values
        // weights: [batch * num_heads, seq_len, seq_len]
        // values: [batch * num_heads, seq_len, head_dim]
        // output: [batch * num_heads, seq_len, head_dim]
        
        // Infer dimensions
        let seqLen = 10  // Would be extracted from metadata
        let batchHeads = values.count / (seqLen * headDim)
        
        let output = try await metalPipeline.allocateBuffer(size: values.count)
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "attention_weighted_sum")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights.buffer, offset: 0, index: 0)
        encoder.setBuffer(values.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        
        var batchHeadsVal = UInt32(batchHeads)
        var seqLenVal = UInt32(seqLen)
        var headDimVal = UInt32(headDim)
        
        encoder.setBytes(&batchHeadsVal, length: 4, index: 3)
        encoder.setBytes(&seqLenVal, length: 4, index: 4)
        encoder.setBytes(&headDimVal, length: 4, index: 5)
        
        // Each thread computes one element of the output
        let threadsPerThreadgroup = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroups = MTLSize(
            width: (headDim + 7) / 8,
            height: (seqLen + 7) / 8,
            depth: batchHeads
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return output
    }
    
    private func accumulate(_ from: MetalBuffer, into: MetalBuffer) async throws {
        // Accumulate gradients using Metal kernel
        try await operations.addBuffers(from, into, output: into)
    }
}

// MARK: - Transformer Encoder Layer

/// Transformer Encoder layer combining multi-head attention and feed-forward network
public actor TransformerEncoderLayer: NeuralLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let selfAttention: MultiHeadAttentionLayer
    private let feedForward: FeedForwardNetwork
    private let norm1: LayerNormLayer
    private let norm2: LayerNormLayer
    private let dropout: DropoutLayer
    
    private let dModel: Int
    private let dropoutRate: Float
    
    // MARK: - Initialization
    
    public init(
        dModel: Int,
        nHead: Int,
        dimFeedforward: Int = 2048,
        dropout: Float = 0.1,
        activation: Activation = .relu,
        layerNormEps: Float = 1e-5,
        batchFirst: Bool = false,
        name: String = "transformer_encoder",
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.dModel = dModel
        self.dropoutRate = dropout
        self.metalPipeline = metalPipeline
        
        // Initialize sub-layers
        self.selfAttention = try await MultiHeadAttentionLayer(
            embedDim: dModel,
            numHeads: nHead,
            dropout: dropout,
            batchFirst: batchFirst,
            name: "\(name)_self_attn",
            metalPipeline: metalPipeline
        )
        
        self.feedForward = try await FeedForwardNetwork(
            dModel: dModel,
            dFeedforward: dimFeedforward,
            dropout: dropout,
            activation: activation,
            name: "\(name)_ffn",
            metalPipeline: metalPipeline
        )
        
        self.norm1 = try await LayerNormLayer(
            normalizedShape: [dModel],
            eps: layerNormEps,
            name: "\(name)_norm1",
            metalPipeline: metalPipeline
        )
        
        self.norm2 = try await LayerNormLayer(
            normalizedShape: [dModel],
            eps: layerNormEps,
            name: "\(name)_norm2",
            metalPipeline: metalPipeline
        )
        
        self.dropout = try await DropoutLayer(
            rate: dropout,
            metalPipeline: metalPipeline
        )
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Self-attention block with residual connection
        let attnOutput = try await selfAttention.forward(input)
        let droppedAttn = try await dropout.forward(attnOutput)
        let residual1 = try await addResidual(input, droppedAttn)
        let normed1 = try await norm1.forward(residual1)
        
        // Feed-forward block with residual connection
        let ffOutput = try await feedForward.forward(normed1)
        let droppedFF = try await dropout.forward(ffOutput)
        let residual2 = try await addResidual(normed1, droppedFF)
        let output = try await norm2.forward(residual2)
        
        return output
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        // Backward through norm2
        let gradNorm2 = try await norm2.backward(gradOutput)
        
        // Backward through residual connection
        let (gradNormed1, gradDroppedFF) = try await splitResidualGrad(gradNorm2)
        
        // Backward through dropout and feed-forward
        let gradFF = try await dropout.backward(gradDroppedFF)
        let gradFFInput = try await feedForward.backward(gradFF)
        
        // Add gradient from skip connection
        let gradNorm1Input = try await addBuffers(gradNormed1, gradFFInput)
        
        // Backward through norm1
        let gradNorm1 = try await norm1.backward(gradNorm1Input)
        
        // Backward through residual connection
        let (gradInput1, gradDroppedAttn) = try await splitResidualGrad(gradNorm1)
        
        // Backward through dropout and self-attention
        let gradAttn = try await dropout.backward(gradDroppedAttn)
        let gradAttnInput = try await selfAttention.backward(gradAttn)
        
        // Add gradient from skip connection
        let gradInput = try await addBuffers(gradInput1, gradAttnInput)
        
        return gradInput
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        try await selfAttention.updateParameters(gradients, learningRate: learningRate)
        try await feedForward.updateParameters(gradients, learningRate: learningRate)
        try await norm1.updateParameters(gradients, learningRate: learningRate)
        try await norm2.updateParameters(gradients, learningRate: learningRate)
    }
    
    public func getParameters() async -> MetalBuffer? {
        await selfAttention.getParameters()
    }
    
    public func getParameterCount() async -> Int {
        let attnCount = await selfAttention.getParameterCount()
        let ffCount = await feedForward.getParameterCount()
        let norm1Count = await norm1.getParameterCount()
        let norm2Count = await norm2.getParameterCount()
        return attnCount + ffCount + norm1Count + norm2Count
    }
    
    public func setTraining(_ training: Bool) async {
        await selfAttention.setTraining(training)
        await feedForward.setTraining(training)
        await norm1.setTraining(training)
        await norm2.setTraining(training)
        await dropout.setTraining(training)
    }
    
    // MARK: - Helper Methods
    
    private func addResidual(_ input: MetalBuffer, _ residual: MetalBuffer) async throws -> MetalBuffer {
        // Add tensors element-wise for residual connection
        let output = try await metalPipeline.allocateBuffer(shape: input.shape)
        try await operations.addBuffers(input, residual, output: output)
        return output
    }
    
    private func splitResidualGrad(_ grad: MetalBuffer) async throws -> (grad1: MetalBuffer, grad2: MetalBuffer) {
        // For residual connection, gradient flows to both paths
        return (grad, grad)
    }
    
    private func addBuffers(_ a: MetalBuffer, _ b: MetalBuffer) async throws -> MetalBuffer {
        // Add buffers element-wise
        let output = try await metalPipeline.allocateBuffer(shape: a.shape)
        try await operations.addBuffers(a, b, output: output)
        return output
    }
}

// MARK: - Feed-Forward Network

/// Position-wise feed-forward network used in transformers
actor FeedForwardNetwork: NeuralLayer {
    private let linear1: DenseLayer
    private let linear2: DenseLayer
    private let dropout: DropoutLayer
    private let activation: Activation
    
    init(
        dModel: Int,
        dFeedforward: Int,
        dropout: Float,
        activation: Activation,
        name: String,
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.activation = activation
        
        self.linear1 = try await DenseLayer(
            inputSize: dModel,
            outputSize: dFeedforward,
            activation: activation,
            name: "\(name)_linear1",
            metalPipeline: metalPipeline
        )
        
        self.dropout = try await DropoutLayer(
            rate: dropout,
            metalPipeline: metalPipeline
        )
        
        self.linear2 = try await DenseLayer(
            inputSize: dFeedforward,
            outputSize: dModel,
            activation: .linear,
            name: "\(name)_linear2",
            metalPipeline: metalPipeline
        )
    }
    
    func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        let hidden = try await linear1.forward(input)
        let dropped = try await dropout.forward(hidden)
        let output = try await linear2.forward(dropped)
        return output
    }
    
    func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        let gradLinear2 = try await linear2.backward(gradOutput)
        let gradDropout = try await dropout.backward(gradLinear2)
        let gradInput = try await linear1.backward(gradDropout)
        return gradInput
    }
    
    func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        try await linear1.updateParameters(gradients, learningRate: learningRate)
        try await linear2.updateParameters(gradients, learningRate: learningRate)
    }
    
    func getParameters() async -> MetalBuffer? {
        await linear1.getParameters()
    }
    
    func getParameterCount() async -> Int {
        let count1 = await linear1.getParameterCount()
        let count2 = await linear2.getParameterCount()
        return count1 + count2
    }
    
    func setTraining(_ training: Bool) async {
        await linear1.setTraining(training)
        await linear2.setTraining(training)
        await dropout.setTraining(training)
    }
}
// VectorStoreKit: LSTM Layer
//
// Long Short-Term Memory layer with Metal acceleration

import Foundation
@preconcurrency import Metal

/// LSTM (Long Short-Term Memory) layer
public actor LSTMLayer: NeuralLayer {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let parameterStore: ParameterStore
    private let operations: MetalMLOperations
    
    // Layer configuration
    private let inputSize: Int
    private let hiddenSize: Int
    private let numLayers: Int
    private let dropout: Float
    private let bidirectional: Bool
    private let returnSequences: Bool
    
    // Parameter names for each gate
    private let weightsIH: [String]  // Input-to-hidden weights
    private let weightsHH: [String]  // Hidden-to-hidden weights
    private let biasIH: [String]     // Input-to-hidden bias
    private let biasHH: [String]     // Hidden-to-hidden bias
    
    // Cached values for backward pass
    private var cachedStates: LSTMStates?
    private var isTraining: Bool = true
    
    /// LSTM states for forward/backward pass
    struct LSTMStates {
        var inputs: [MetalBuffer]        // Input at each timestep
        var hiddenStates: [MetalBuffer]  // Hidden states at each timestep
        var cellStates: [MetalBuffer]    // Cell states at each timestep
        var gates: [GateOutputs]         // Gate outputs at each timestep
        
        struct GateOutputs {
            let input: MetalBuffer       // Input gate
            let forget: MetalBuffer      // Forget gate
            let cell: MetalBuffer        // Cell gate (candidate)
            let output: MetalBuffer      // Output gate
        }
    }
    
    // MARK: - Initialization
    
    public init(
        inputSize: Int,
        hiddenSize: Int,
        numLayers: Int = 1,
        dropout: Float = 0.0,
        bidirectional: Bool = false,
        returnSequences: Bool = true,
        name: String = "lstm",
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.returnSequences = returnSequences
        
        self.metalPipeline = metalPipeline
        self.parameterStore = await ParameterStore(device: metalPipeline.device)
        self.operations = await metalPipeline.getOperations()
        
        // Initialize parameter names
        let numDirections = bidirectional ? 2 : 1
        var wih: [String] = []
        var whh: [String] = []
        var bih: [String] = []
        var bhh: [String] = []
        
        for layer in 0..<numLayers {
            for direction in 0..<numDirections {
                let prefix = "\(name)_l\(layer)_d\(direction)"
                wih.append("\(prefix)_weight_ih")
                whh.append("\(prefix)_weight_hh")
                bih.append("\(prefix)_bias_ih")
                bhh.append("\(prefix)_bias_hh")
            }
        }
        
        self.weightsIH = wih
        self.weightsHH = whh
        self.biasIH = bih
        self.biasHH = bhh
        
        // Initialize parameters
        try await initializeParameters()
    }
    
    private func initializeParameters() async throws {
        let numDirections = bidirectional ? 2 : 1
        
        for layer in 0..<numLayers {
            for direction in 0..<numDirections {
                let idx = layer * numDirections + direction
                let inputDim = layer == 0 ? inputSize : hiddenSize * numDirections
                
                // Each weight matrix contains 4 gates (i, f, g, o) concatenated
                let wihSize = 4 * hiddenSize * inputDim
                let whhSize = 4 * hiddenSize * hiddenSize
                let biasSize = 4 * hiddenSize
                
                // Allocate weight buffers
                let wih = try await parameterStore.allocateParameter(
                    name: weightsIH[idx],
                    size: wihSize
                )
                let whh = try await parameterStore.allocateParameter(
                    name: weightsHH[idx],
                    size: whhSize
                )
                
                // Initialize weights using Xavier/Glorot initialization
                let scaleIH = sqrt(2.0 / Float(inputDim + hiddenSize))
                let scaleHH = sqrt(2.0 / Float(hiddenSize + hiddenSize))
                
                initializeBuffer(wih, scale: scaleIH)
                initializeBuffer(whh, scale: scaleHH)
                
                // Allocate and initialize biases
                let bih = try await parameterStore.allocateParameter(
                    name: biasIH[idx],
                    size: biasSize
                )
                let bhh = try await parameterStore.allocateParameter(
                    name: biasHH[idx],
                    size: biasSize
                )
                
                // Initialize biases to zero except forget gate bias (set to 1)
                initializeLSTMBias(bih)
                initializeLSTMBias(bhh)
                
                // Allocate gradient buffers
                _ = try await parameterStore.allocateGradient(name: "\(weightsIH[idx])_grad", size: wihSize)
                _ = try await parameterStore.allocateGradient(name: "\(weightsHH[idx])_grad", size: whhSize)
                _ = try await parameterStore.allocateGradient(name: "\(biasIH[idx])_grad", size: biasSize)
                _ = try await parameterStore.allocateGradient(name: "\(biasHH[idx])_grad", size: biasSize)
            }
        }
    }
    
    private func initializeBuffer(_ buffer: MetalBuffer, scale: Float) {
        let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
        for i in 0..<buffer.count {
            ptr[i] = Float.random(in: -scale...scale)
        }
    }
    
    private func initializeLSTMBias(_ buffer: MetalBuffer) {
        let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
        let gateSize = buffer.count / 4
        
        // Initialize all to zero
        for i in 0..<buffer.count {
            ptr[i] = 0
        }
        
        // Set forget gate bias to 1
        for i in gateSize..<(2 * gateSize) {
            ptr[i] = 1.0
        }
    }
    
    // MARK: - Forward Pass
    
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Input shape: [batch, seq_len, input_size] or [seq_len, batch, input_size]
        // Validate input shape
        guard input.shape.rank == 3 else {
            throw MetalMLError.invalidArchitecture("LSTM expects 3D input (seq_len, batch, input_size)")
        }
        
        let shape = getInputShape(input)
        let seqLen = shape.seqLen
        let batchSize = shape.batchSize
        
        // Initialize hidden and cell states
        var h = try await initializeHiddenState(batchSize: batchSize)
        var c = try await initializeCellState(batchSize: batchSize)
        
        // Process sequence
        var outputs: [MetalBuffer] = []
        var states = LSTMStates(
            inputs: [],
            hiddenStates: [],
            cellStates: [],
            gates: []
        )
        
        for t in 0..<seqLen {
            // Extract input at timestep t
            let xt = try await extractTimestep(input, timestep: t, shape: shape)
            
            // Forward through LSTM cell
            let (ht, ct, gates) = try await lstmCell(
                input: xt,
                hidden: h[0],  // Use first layer hidden state
                cell: c[0],    // Use first layer cell state
                layer: 0
            )
            
            // Update states
            h[0] = ht
            c[0] = ct
            
            // Store for backward pass if training
            if isTraining {
                states.inputs.append(xt)
                states.hiddenStates.append(ht)
                states.cellStates.append(ct)
                states.gates.append(gates)
            }
            
            outputs.append(ht)
        }
        
        // Save states for backward pass
        if isTraining {
            cachedStates = states
        }
        
        // Return based on configuration
        if returnSequences {
            // Concatenate all outputs
            return try await concatenateSequence(outputs)
        } else {
            // Return only last output
            return outputs.last!
        }
    }
    
    // MARK: - LSTM Cell
    
    private func lstmCell(
        input: MetalBuffer,
        hidden: MetalBuffer,
        cell: MetalBuffer,
        layer: Int
    ) async throws -> (hidden: MetalBuffer, cell: MetalBuffer, gates: LSTMStates.GateOutputs) {
        // Get parameters
        guard let wih = await parameterStore.getParameter(name: weightsIH[layer]),
              let whh = await parameterStore.getParameter(name: weightsHH[layer]),
              let bih = await parameterStore.getParameter(name: biasIH[layer]),
              let bhh = await parameterStore.getParameter(name: biasHH[layer]) else {
            throw MetalMLError.parameterNotFound(name: "LSTM weights")
        }
        
        // Compute input transformation: W_ih @ x + b_ih
        let ihTransform = try await metalPipeline.allocateBuffer(size: 4 * hiddenSize)
        try await computeLinearTransform(
            input: input,
            weight: wih,
            bias: bih,
            output: ihTransform
        )
        
        // Compute hidden transformation: W_hh @ h + b_hh
        let hhTransform = try await metalPipeline.allocateBuffer(size: 4 * hiddenSize)
        try await computeLinearTransform(
            input: hidden,
            weight: whh,
            bias: bhh,
            output: hhTransform
        )
        
        // Add transformations
        let combined = try await metalPipeline.allocateBuffer(size: 4 * hiddenSize)
        _ = try await addBuffers(ihTransform, hhTransform, output: combined)
        
        // Split into gates
        let gates = try await splitGates(combined)
        
        // Apply gate activations and compute new states
        let (newHidden, newCell) = try await computeLSTMStates(
            gates: gates,
            prevCell: cell
        )
        
        return (newHidden, newCell, gates)
    }
    
    private func splitGates(_ combined: MetalBuffer) async throws -> LSTMStates.GateOutputs {
        let gateSize = hiddenSize
        
        let i = try await extractGate(combined, offset: 0, size: gateSize)
        let f = try await extractGate(combined, offset: gateSize, size: gateSize)
        let g = try await extractGate(combined, offset: 2 * gateSize, size: gateSize)
        let o = try await extractGate(combined, offset: 3 * gateSize, size: gateSize)
        
        return LSTMStates.GateOutputs(input: i, forget: f, cell: g, output: o)
    }
    
    private func computeLSTMStates(
        gates: LSTMStates.GateOutputs,
        prevCell: MetalBuffer
    ) async throws -> (hidden: MetalBuffer, cell: MetalBuffer) {
        // Apply sigmoid to input, forget, and output gates
        let i_sig = try await applySigmoid(gates.input)
        let f_sig = try await applySigmoid(gates.forget)
        let o_sig = try await applySigmoid(gates.output)
        
        // Apply tanh to cell gate
        let g_tanh = try await applyTanh(gates.cell)
        
        // Compute new cell state: c_t = f_t * c_{t-1} + i_t * g_t
        let forgotten = try await elementwiseMultiply(f_sig, prevCell)
        let input_contrib = try await elementwiseMultiply(i_sig, g_tanh)
        let newCell = try await addBuffers(forgotten, input_contrib)
        
        // Compute new hidden state: h_t = o_t * tanh(c_t)
        let cell_tanh = try await applyTanh(newCell)
        let newHidden = try await elementwiseMultiply(o_sig, cell_tanh)
        
        return (newHidden, newCell)
    }
    
    // MARK: - Backward Pass
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        guard let states = cachedStates else {
            throw MetalMLError.parameterNotFound(name: "cached LSTM states")
        }
        
        // Initialize gradient accumulators
        let gradInput = try await metalPipeline.allocateBuffer(
            size: states.inputs.reduce(0) { $0 + $1.count }
        )
        
        // Backward through time
        var gradH = try await metalPipeline.allocateBuffer(size: hiddenSize)
        var gradC = try await metalPipeline.allocateBuffer(size: hiddenSize)
        
        // Process in reverse order
        for t in (0..<states.inputs.count).reversed() {
            // Extract gradient for this timestep
            let gradY = try await extractTimestepGrad(gradOutput, timestep: t)
            
            // Add to hidden gradient
            gradH = try await addBuffers(gradH, gradY)
            
            // Backward through LSTM cell
            let (gradX, newGradH, newGradC) = try await lstmCellBackward(
                gradH: gradH,
                gradC: gradC,
                states: states,
                timestep: t,
                layer: 0
            )
            
            // Update gradients
            gradH = newGradH
            gradC = newGradC
            
            // Accumulate input gradient
            try await accumulateInputGrad(gradInput, gradX, timestep: t)
        }
        
        return gradInput
    }
    
    private func lstmCellBackward(
        gradH: MetalBuffer,
        gradC: MetalBuffer,
        states: LSTMStates,
        timestep: Int,
        layer: Int
    ) async throws -> (gradInput: MetalBuffer, gradH: MetalBuffer, gradC: MetalBuffer) {
        // Get saved states for this timestep
        guard timestep < states.gates.count else {
            throw MetalMLError.incompatibleBufferSize(expected: timestep + 1, actual: states.gates.count)
        }
        
        let gates = states.gates[timestep]
        let prevC = timestep > 0 ? states.cellStates[timestep - 1] : 
                    try await metalPipeline.allocateBuffer(size: hiddenSize) // Zero initial cell state
        let currentC = states.cellStates[timestep]
        
        // Apply activation functions to gates
        let i_sig = try await applySigmoid(gates.input)
        let f_sig = try await applySigmoid(gates.forget)
        let o_sig = try await applySigmoid(gates.output)
        let g_tanh = try await applyTanh(gates.cell)
        let c_tanh = try await applyTanh(currentC)
        
        // Compute gradients through output gate
        let gradO = try await elementwiseMultiply(gradH, c_tanh)
        let gradCFromH = try await elementwiseMultiply(gradH, o_sig)
        
        // Add gradient from cell state
        var totalGradC = try await addBuffers(gradC, gradCFromH)
        
        // Compute gradients through cell state update
        let gradCTanh = try await computeTanhDerivative(gradCFromH, output: c_tanh)
        totalGradC = try await addBuffers(totalGradC, gradCTanh)
        
        // Gradients through forget gate
        let gradF = try await elementwiseMultiply(totalGradC, prevC)
        let gradPrevC = try await elementwiseMultiply(totalGradC, f_sig)
        
        // Gradients through input gate
        let gradI = try await elementwiseMultiply(totalGradC, g_tanh)
        let gradG = try await elementwiseMultiply(totalGradC, i_sig)
        
        // Apply gate activation derivatives
        let gradIGate = try await computeSigmoidDerivative(gradI, output: i_sig)
        let gradFGate = try await computeSigmoidDerivative(gradF, output: f_sig)
        let gradGGate = try await computeTanhDerivative(gradG, output: g_tanh)
        let gradOGate = try await computeSigmoidDerivative(gradO, output: o_sig)
        
        // Combine gate gradients
        let combinedGradGates = try await concatenateGates(
            gradIGate, gradFGate, gradGGate, gradOGate
        )
        
        // Compute gradients w.r.t. input and hidden state
        guard let wih = await parameterStore.getParameter(name: weightsIH[layer]),
              let whh = await parameterStore.getParameter(name: weightsHH[layer]) else {
            throw MetalMLError.parameterNotFound(name: "LSTM weights")
        }
        
        // gradInput = wih^T @ combinedGradGates
        let gradInput = try await metalPipeline.allocateBuffer(size: inputSize)
        try await operations.matmul(
            wih,
            combinedGradGates,
            output: gradInput,
            m: inputSize,
            n: 1,
            k: 4 * hiddenSize,
            useTiling: false
        )
        
        // gradHidden = whh^T @ combinedGradGates
        let gradHidden = try await metalPipeline.allocateBuffer(size: hiddenSize)
        try await operations.matmul(
            whh,
            combinedGradGates,
            output: gradHidden,
            m: hiddenSize,
            n: 1,
            k: 4 * hiddenSize,
            useTiling: false
        )
        
        // Update weight gradients
        try await accumulateWeightGradients(
            input: states.inputs[timestep],
            hidden: timestep > 0 ? states.hiddenStates[timestep - 1] : 
                   try await metalPipeline.allocateBuffer(size: hiddenSize),
            gradGates: combinedGradGates,
            layer: layer
        )
        
        return (gradInput, gradHidden, gradPrevC)
    }
    
    private func computeSigmoidDerivative(_ grad: MetalBuffer, output: MetalBuffer) async throws -> MetalBuffer {
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        let _ = try await metalPipeline.allocateBuffer(size: grad.count)
        let oneMinusOutput = try await metalPipeline.allocateBuffer(size: output.count)
        
        // Compute 1 - output
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "one_minus")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(output.buffer, offset: 0, index: 0)
        encoder.setBuffer(oneMinusOutput.buffer, offset: 0, index: 1)
        
        var size = UInt32(output.count)
        encoder.setBytes(&size, length: 4, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (output.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Multiply grad * output * (1 - output)
        let temp = try await elementwiseMultiply(grad, output)
        return try await elementwiseMultiply(temp, oneMinusOutput)
    }
    
    private func computeTanhDerivative(_ grad: MetalBuffer, output: MetalBuffer) async throws -> MetalBuffer {
        // tanh'(x) = 1 - tanh(x)^2
        let _ = try await metalPipeline.allocateBuffer(size: grad.count)
        let outputSquared = try await elementwiseMultiply(output, output)
        let oneMinusOutputSquared = try await computeOneMinus(outputSquared)
        return try await elementwiseMultiply(grad, oneMinusOutputSquared)
    }
    
    private func computeOneMinus(_ x: MetalBuffer) async throws -> MetalBuffer {
        let result = try await metalPipeline.allocateBuffer(size: x.count)
        
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "one_minus")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(x.buffer, offset: 0, index: 0)
        encoder.setBuffer(result.buffer, offset: 0, index: 1)
        
        var size = UInt32(x.count)
        encoder.setBytes(&size, length: 4, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (x.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return result
    }
    
    private func concatenateGates(_ i: MetalBuffer, _ f: MetalBuffer, _ g: MetalBuffer, _ o: MetalBuffer) async throws -> MetalBuffer {
        let combined = try await metalPipeline.allocateBuffer(size: 4 * hiddenSize)
        
        // Copy each gate to its position in the combined buffer
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "copy_to_offset")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        let gates = [(i, 0), (f, hiddenSize), (g, 2 * hiddenSize), (o, 3 * hiddenSize)]
        
        for (gate, offset) in gates {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(gate.buffer, offset: 0, index: 0)
            encoder.setBuffer(combined.buffer, offset: 0, index: 1)
            
            var offsetVal = UInt32(offset)
            var size = UInt32(hiddenSize)
            encoder.setBytes(&offsetVal, length: 4, index: 2)
            encoder.setBytes(&size, length: 4, index: 3)
            
            let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            let threadgroups = MTLSize(width: (hiddenSize + 255) / 256, height: 1, depth: 1)
            
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
        }
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return combined
    }
    
    private func accumulateWeightGradients(
        input: MetalBuffer,
        hidden: MetalBuffer,
        gradGates: MetalBuffer,
        layer: Int
    ) async throws {
        // Compute weight gradients and accumulate
        guard let wihGrad = await parameterStore.getGradient(name: "\(weightsIH[layer])_grad"),
              let whhGrad = await parameterStore.getGradient(name: "\(weightsHH[layer])_grad"),
              let bihGrad = await parameterStore.getGradient(name: "\(biasIH[layer])_grad"),
              let bhhGrad = await parameterStore.getGradient(name: "\(biasHH[layer])_grad") else {
            return
        }
        
        // Compute outer products for weight gradients
        try await operations.computeWeightGradients(
            gradOutput: gradGates,
            input: input,
            gradWeights: wihGrad,
            outputSize: 4 * hiddenSize,
            inputSize: inputSize
        )
        
        try await operations.computeWeightGradients(
            gradOutput: gradGates,
            input: hidden,
            gradWeights: whhGrad,
            outputSize: 4 * hiddenSize,
            inputSize: hiddenSize
        )
        
        // Accumulate bias gradients (just copy gradGates)
        try await operations.copyBuffer(from: gradGates, to: bihGrad)
        try await operations.copyBuffer(from: gradGates, to: bhhGrad)
    }
    
    // MARK: - Parameter Updates
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        // Update all LSTM parameters
        for i in 0..<weightsIH.count {
            if let wihGrad = await parameterStore.getGradient(name: "\(weightsIH[i])_grad") {
                try await parameterStore.updateParameter(
                    name: weightsIH[i],
                    with: wihGrad,
                    learningRate: learningRate
                )
            }
            
            if let whhGrad = await parameterStore.getGradient(name: "\(weightsHH[i])_grad") {
                try await parameterStore.updateParameter(
                    name: weightsHH[i],
                    with: whhGrad,
                    learningRate: learningRate
                )
            }
            
            if let bihGrad = await parameterStore.getGradient(name: "\(biasIH[i])_grad") {
                try await parameterStore.updateParameter(
                    name: biasIH[i],
                    with: bihGrad,
                    learningRate: learningRate
                )
            }
            
            if let bhhGrad = await parameterStore.getGradient(name: "\(biasHH[i])_grad") {
                try await parameterStore.updateParameter(
                    name: biasHH[i],
                    with: bhhGrad,
                    learningRate: learningRate
                )
            }
        }
    }
    
    // MARK: - NeuralLayer Protocol
    
    public func getParameters() async -> MetalBuffer? {
        // Return first weight matrix as representative
        await parameterStore.getParameter(name: weightsIH[0])
    }
    
    public func getParameterCount() async -> Int {
        let numDirections = bidirectional ? 2 : 1
        var count = 0
        
        for layer in 0..<numLayers {
            let inputDim = layer == 0 ? inputSize : hiddenSize * numDirections
            count += 4 * hiddenSize * (inputDim + hiddenSize + 2) * numDirections
        }
        
        return count
    }
    
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
    
    // MARK: - Helper Methods
    
    private func getInputShape(_ input: MetalBuffer) -> (seqLen: Int, batchSize: Int, inputSize: Int) {
        // Extract from buffer shape
        if input.shape.rank == 3 {
            let dims = input.shape.dimensions
            // Assume [seq_len, batch, input_size] format
            return (dims[0], dims[1], dims[2])
        }
        // Fallback for legacy
        return (10, 1, inputSize)  // seq_len=10, batch=1
    }
    
    private func initializeHiddenState(batchSize: Int) async throws -> [MetalBuffer] {
        var states: [MetalBuffer] = []
        let numDirections = bidirectional ? 2 : 1
        
        for _ in 0..<(numLayers * numDirections) {
            let h = try await metalPipeline.allocateBuffer(shape: TensorShape(batchSize, hiddenSize))
            // Initialize to zero
            let ptr = h.buffer.contents().bindMemory(to: Float.self, capacity: h.count)
            for i in 0..<h.count {
                ptr[i] = 0
            }
            states.append(h)
        }
        
        return states
    }
    
    private func initializeCellState(batchSize: Int) async throws -> [MetalBuffer] {
        // Same as hidden state initialization
        return try await initializeHiddenState(batchSize: batchSize)
    }
    
    private func extractTimestep(
        _ input: MetalBuffer,
        timestep: Int,
        shape: (seqLen: Int, batchSize: Int, inputSize: Int)
    ) async throws -> MetalBuffer {
        // Extract single timestep from sequence
        let timestepSize = shape.batchSize * shape.inputSize
        let output = try await metalPipeline.allocateBuffer(shape: TensorShape(shape.batchSize, shape.inputSize))
        
        // Copy data (placeholder)
        let offset = timestep * timestepSize
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
        let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        
        for i in 0..<timestepSize {
            outputPtr[i] = inputPtr[offset + i]
        }
        
        return output
    }
    
    private func concatenateSequence(_ outputs: [MetalBuffer]) async throws -> MetalBuffer {
        let outputShape = TensorShape(outputs[0].shape.dimensions[0], outputs.count, outputs[0].shape.dimensions[1])
        let result = try await metalPipeline.allocateBuffer(shape: outputShape)
        
        // Concatenate (placeholder)
        var offset = 0
        let resultPtr = result.buffer.contents().bindMemory(to: Float.self, capacity: result.count)
        
        for output in outputs {
            let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
            for i in 0..<output.count {
                resultPtr[offset + i] = outputPtr[i]
            }
            offset += output.count
        }
        
        return result
    }
    
    private func computeLinearTransform(
        input: MetalBuffer,
        weight: MetalBuffer,
        bias: MetalBuffer,
        output: MetalBuffer
    ) async throws {
        // Perform matrix-vector multiplication: output = weight @ input + bias
        // Weight is [4*hiddenSize, inputSize], input is [inputSize]
        let inputSize = input.count
        let outputSize = output.count
        
        // First compute weight @ input
        try await operations.matmul(
            weight,
            input,
            output: output,
            m: outputSize,
            n: 1,
            k: inputSize,
            useTiling: false
        )
        
        // Then add bias
        try await operations.addBias(
            matrix: output,
            bias: bias,
            rows: 1,
            cols: outputSize
        )
    }
    
    private func addBuffers(_ a: MetalBuffer, _ b: MetalBuffer, output: MetalBuffer? = nil) async throws -> MetalBuffer {
        let out: MetalBuffer
        if let output = output {
            out = output
        } else {
            out = try await metalPipeline.allocateBuffer(size: a.count)
        }
        
        // Use Metal shader for element-wise addition
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "element_add")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a.buffer, offset: 0, index: 0)
        encoder.setBuffer(b.buffer, offset: 0, index: 1)
        encoder.setBuffer(out.buffer, offset: 0, index: 2)
        
        var size = UInt32(a.count)
        encoder.setBytes(&size, length: 4, index: 3)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (a.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return out
    }
    
    private func extractGate(_ buffer: MetalBuffer, offset: Int, size: Int) async throws -> MetalBuffer {
        let gate = try await metalPipeline.allocateBuffer(size: size)
        
        // Use Metal shader to extract a slice from the buffer
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "extract_slice")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(gate.buffer, offset: 0, index: 1)
        
        var offsetVal = UInt32(offset)
        var sizeVal = UInt32(size)
        encoder.setBytes(&offsetVal, length: 4, index: 2)
        encoder.setBytes(&sizeVal, length: 4, index: 3)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (size + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return gate
    }
    
    private func applySigmoid(_ input: MetalBuffer) async throws -> MetalBuffer {
        let output = try await metalPipeline.allocateBuffer(size: input.count)
        try await operations.applyActivation(input, output: output, activation: .sigmoid)
        return output
    }
    
    private func applyTanh(_ input: MetalBuffer) async throws -> MetalBuffer {
        let output = try await metalPipeline.allocateBuffer(size: input.count)
        try await operations.applyActivation(input, output: output, activation: .tanh)
        return output
    }
    
    private func elementwiseMultiply(_ a: MetalBuffer, _ b: MetalBuffer) async throws -> MetalBuffer {
        let output = try await metalPipeline.allocateBuffer(size: a.count)
        
        // Use Metal shader for element-wise multiplication
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "element_multiply")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a.buffer, offset: 0, index: 0)
        encoder.setBuffer(b.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        
        var size = UInt32(a.count)
        encoder.setBytes(&size, length: 4, index: 3)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (a.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return output
    }
    
    private func extractTimestepGrad(_ gradOutput: MetalBuffer, timestep: Int) async throws -> MetalBuffer {
        // Extract gradient for a specific timestep from the output gradient
        let shape = gradOutput.shape
        let timestepSize = hiddenSize
        
        // For returnSequences=true, gradOutput contains gradients for all timesteps
        // Shape: [seqLen * hiddenSize] or [batch, seqLen, hiddenSize]
        let gradTimestep = try await metalPipeline.allocateBuffer(shape: TensorShape(timestepSize))
        
        // Copy the relevant portion
        let offset = timestep * timestepSize
        let gradOutputPtr = gradOutput.buffer.contents().bindMemory(to: Float.self, capacity: gradOutput.count)
        let gradTimestepPtr = gradTimestep.buffer.contents().bindMemory(to: Float.self, capacity: timestepSize)
        
        for i in 0..<timestepSize {
            gradTimestepPtr[i] = offset + i < gradOutput.count ? gradOutputPtr[offset + i] : 0.0
        }
        
        return gradTimestep
    }
    
    private func accumulateInputGrad(_ gradInput: MetalBuffer, _ gradX: MetalBuffer, timestep: Int) async throws {
        // Accumulate input gradient at specific timestep
        let shape = getInputShape(gradInput)
        let timestepSize = shape.batchSize * inputSize
        let offset = timestep * timestepSize
        
        // Use Metal shader for accumulation
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: "accumulate_at_offset")
        
        guard let commandBuffer = await metalPipeline.getCommandQueue().makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradX.buffer, offset: 0, index: 0)
        encoder.setBuffer(gradInput.buffer, offset: 0, index: 1)
        
        var offsetVal = UInt32(offset)
        var size = UInt32(timestepSize)
        encoder.setBytes(&offsetVal, length: 4, index: 2)
        encoder.setBytes(&size, length: 4, index: 3)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (timestepSize + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
// VectorStoreKit: LSTM Layer Implementation
//
// Long Short-Term Memory layer with Metal acceleration

import Foundation
import Accelerate
@preconcurrency import Metal

/// LSTM layer configuration
public struct LSTMConfig: Sendable {
    public let hiddenSize: Int
    public let returnSequences: Bool
    public let dropout: Float
    public let recurrentDropout: Float
    
    public init(
        hiddenSize: Int,
        returnSequences: Bool = true,
        dropout: Float = 0.0,
        recurrentDropout: Float = 0.0
    ) {
        self.hiddenSize = hiddenSize
        self.returnSequences = returnSequences
        self.dropout = dropout
        self.recurrentDropout = recurrentDropout
    }
}

/// LSTM (Long Short-Term Memory) layer
public actor LSTMLayer: NeuralLayer {
    // MARK: - Properties
    public let inputSize: Int
    public let hiddenSize: Int
    public let config: LSTMConfig
    public let name: String
    internal let metalPipeline: MetalMLPipeline
    internal var isTraining: Bool = true
    
    // Convenience property for serialization
    public var returnSequences: Bool {
        config.returnSequences
    }
    
    // Parameter names for gates
    private let inputGateWeightsName: String
    private let inputGateBiasName: String
    private let forgetGateWeightsName: String
    private let forgetGateBiasName: String
    private let cellGateWeightsName: String
    private let cellGateBiasName: String
    private let outputGateWeightsName: String
    private let outputGateBiasName: String
    
    // Recurrent weight names
    private let inputGateRecurrentWeightsName: String
    private let forgetGateRecurrentWeightsName: String
    private let cellGateRecurrentWeightsName: String
    private let outputGateRecurrentWeightsName: String
    
    // Gradient names
    private let gradientSuffix = "_grad"
    
    // Parameter store
    private let parameterStore: ParameterStore
    
    // Cached states for backward pass
    private var sequenceLength: Int = 0
    private var batchSize: Int = 0
    private var lastInputSequence: MetalBuffer?
    private var hiddenStates: [MetalBuffer] = []
    private var cellStates: [MetalBuffer] = []
    private var gateOutputs: [String: [MetalBuffer]] = [:]
    
    // MARK: - Initialization
    public init(
        inputSize: Int,
        config: LSTMConfig,
        name: String = "lstm",
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.inputSize = inputSize
        self.hiddenSize = config.hiddenSize
        self.config = config
        self.name = name
        self.metalPipeline = metalPipeline
        
        // Initialize parameter names
        self.inputGateWeightsName = "\(name)_input_gate_weights"
        self.inputGateBiasName = "\(name)_input_gate_bias"
        self.forgetGateWeightsName = "\(name)_forget_gate_weights"
        self.forgetGateBiasName = "\(name)_forget_gate_bias"
        self.cellGateWeightsName = "\(name)_cell_gate_weights"
        self.cellGateBiasName = "\(name)_cell_gate_bias"
        self.outputGateWeightsName = "\(name)_output_gate_weights"
        self.outputGateBiasName = "\(name)_output_gate_bias"
        
        // Recurrent weight names
        self.inputGateRecurrentWeightsName = "\(name)_input_gate_recurrent_weights"
        self.forgetGateRecurrentWeightsName = "\(name)_forget_gate_recurrent_weights"
        self.cellGateRecurrentWeightsName = "\(name)_cell_gate_recurrent_weights"
        self.outputGateRecurrentWeightsName = "\(name)_output_gate_recurrent_weights"
        
        // Initialize parameter store
        self.parameterStore = await ParameterStore(device: metalPipeline.device)
        
        // Initialize parameters
        try await initializeParameters()
    }
    
    private func initializeParameters() async throws {
        let scale = sqrt(1.0 / Float(inputSize + hiddenSize))
        
        // Initialize weights and biases for each gate
        let gateConfigs = [
            (inputGateWeightsName, inputGateBiasName, inputGateRecurrentWeightsName),
            (forgetGateWeightsName, forgetGateBiasName, forgetGateRecurrentWeightsName),
            (cellGateWeightsName, cellGateBiasName, cellGateRecurrentWeightsName),
            (outputGateWeightsName, outputGateBiasName, outputGateRecurrentWeightsName)
        ]
        
        for (weightsName, biasName, recurrentWeightsName) in gateConfigs {
            // Input weights: [hiddenSize x inputSize]
            let weights = try await parameterStore.allocateParameter(
                name: weightsName,
                size: hiddenSize * inputSize
            )
            initializeBuffer(weights, scale: scale)
            
            // Recurrent weights: [hiddenSize x hiddenSize]
            let recurrentWeights = try await parameterStore.allocateParameter(
                name: recurrentWeightsName,
                size: hiddenSize * hiddenSize
            )
            initializeBuffer(recurrentWeights, scale: scale)
            
            // Bias: [hiddenSize]
            let bias = try await parameterStore.allocateParameter(
                name: biasName,
                size: hiddenSize
            )
            
            // Initialize forget gate bias to 1.0, others to 0.0
            let biasValue: Float = biasName.contains("forget") ? 1.0 : 0.0
            let biasPtr = bias.buffer.contents().bindMemory(to: Float.self, capacity: bias.count)
            for i in 0..<bias.count {
                biasPtr[i] = biasValue
            }
            
            // Allocate gradient buffers
            _ = try await parameterStore.allocateGradient(
                name: weightsName + gradientSuffix,
                size: hiddenSize * inputSize
            )
            _ = try await parameterStore.allocateGradient(
                name: recurrentWeightsName + gradientSuffix,
                size: hiddenSize * hiddenSize
            )
            _ = try await parameterStore.allocateGradient(
                name: biasName + gradientSuffix,
                size: hiddenSize
            )
        }
    }
    
    private func initializeBuffer(_ buffer: MetalBuffer, scale: Float) {
        let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
        for i in 0..<buffer.count {
            ptr[i] = Float.random(in: -scale..<scale)
        }
    }
    
    // MARK: - NeuralLayer Protocol
    public func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Input shape: [seq_len * batch_size, input_size]
        // We need to determine seq_len and batch_size from context
        
        // For now, assume batch_size = 1
        let batchSize = 1
        let totalElements = input.count / inputSize
        let seqLen = totalElements / batchSize
        
        guard input.count == seqLen * batchSize * inputSize else {
            throw MetalMLError.incompatibleBufferSize(
                expected: seqLen * batchSize * inputSize,
                actual: input.count
            )
        }
        
        self.sequenceLength = seqLen
        self.batchSize = batchSize
        self.lastInputSequence = input
        
        // Clear previous states
        hiddenStates.removeAll()
        cellStates.removeAll()
        gateOutputs = [
            "input": [],
            "forget": [],
            "cell": [],
            "output": []
        ]
        
        let operations = await metalPipeline.getOperations()
        _ = await metalPipeline.getShaderLibrary()
        
        // Initialize hidden and cell states
        var h_t = try await allocateBuffer(size: batchSize * hiddenSize)
        var c_t = try await allocateBuffer(size: batchSize * hiddenSize)
        
        // Zero initialize states
        let zeroPtr = h_t.buffer.contents().bindMemory(to: Float.self, capacity: h_t.count)
        for i in 0..<h_t.count {
            zeroPtr[i] = 0.0
        }
        let cellPtr = c_t.buffer.contents().bindMemory(to: Float.self, capacity: c_t.count)
        for i in 0..<c_t.count {
            cellPtr[i] = 0.0
        }
        
        // Process sequence
        var outputs: [MetalBuffer] = []
        
        for t in 0..<seqLen {
            // Extract timestep from input
            let x_t = try await extractTimestep(
                from: input,
                timestep: t,
                batchSize: batchSize,
                inputSize: inputSize,
                seqLen: seqLen
            )
            
            // Compute gates
            let (i_t, f_t, g_t, o_t) = try await computeGates(
                x_t: x_t,
                h_t: h_t,
                operations: operations
            )
            
            // Store gate outputs for backward pass
            gateOutputs["input"]!.append(i_t)
            gateOutputs["forget"]!.append(f_t)
            gateOutputs["cell"]!.append(g_t)
            gateOutputs["output"]!.append(o_t)
            
            // Update cell state: c_t = f_t * c_{t-1} + i_t * g_t
            let new_c_t = try await updateCellState(
                c_prev: c_t,
                f_t: f_t,
                i_t: i_t,
                g_t: g_t,
                operations: operations
            )
            
            // Update hidden state: h_t = o_t * tanh(c_t)
            let new_h_t = try await updateHiddenState(
                c_t: new_c_t,
                o_t: o_t,
                operations: operations
            )
            
            // Store states
            cellStates.append(new_c_t)
            hiddenStates.append(new_h_t)
            
            if config.returnSequences {
                outputs.append(new_h_t)
            }
            
            // Update states for next timestep
            h_t = new_h_t
            c_t = new_c_t
        }
        
        // Return appropriate output
        if config.returnSequences {
            // Concatenate all outputs
            return try await concatenateSequenceOutputs(outputs)
        } else {
            // Return only the last hidden state
            return h_t
        }
    }
    
    public func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        // Implement BPTT (Backpropagation Through Time)
        let operations = await metalPipeline.getOperations()
        
        let gradInput = try await allocateBuffer(size: sequenceLength * batchSize * inputSize)
        
        // Initialize gradient states
        var grad_h_next = try await allocateBuffer(size: batchSize * hiddenSize)
        var grad_c_next = try await allocateBuffer(size: batchSize * hiddenSize)
        
        // Zero initialize
        zeroBuffer(grad_h_next)
        zeroBuffer(grad_c_next)
        
        // Process gradients in reverse order
        for t in (0..<sequenceLength).reversed() {
            // Extract gradient for this timestep
            let grad_h_t: MetalBuffer
            if config.returnSequences {
                grad_h_t = try await extractTimestepGradient(
                    from: gradOutput,
                    timestep: t,
                    hiddenSize: hiddenSize
                )
            } else if t == sequenceLength - 1 {
                grad_h_t = gradOutput
            } else {
                grad_h_t = try await allocateBuffer(size: batchSize * hiddenSize)
                zeroBuffer(grad_h_t)
            }
            
            // Add gradient from next timestep
            try await operations.addBuffers(grad_h_t, grad_h_next, output: grad_h_t)
            
            // Compute gradients through gates
            let gradients = try await computeGateGradients(
                grad_h_t: grad_h_t,
                grad_c_next: grad_c_next,
                timestep: t,
                operations: operations
            )
            
            // Update gradient states
            grad_h_next = gradients.grad_h_prev
            grad_c_next = gradients.grad_c_prev
            
            // Accumulate parameter gradients
            try await accumulateParameterGradients(
                timestep: t,
                gradients: gradients,
                operations: operations
            )
            
            // Store input gradient for this timestep
            try await storeTimestepInputGradient(
                gradInput: gradInput,
                timestep: t,
                gradient: gradients.grad_x_t
            )
        }
        
        return gradInput
    }
    
    public func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        // Parameters are updated by the optimizer through the parameter store
        // This method is called by the training loop but actual updates
        // happen through the optimizer
    }
    
    public func getParameters() async -> MetalBuffer? {
        // Return all parameters concatenated
        let params = [
            inputGateWeightsName, inputGateBiasName, inputGateRecurrentWeightsName,
            forgetGateWeightsName, forgetGateBiasName, forgetGateRecurrentWeightsName,
            cellGateWeightsName, cellGateBiasName, cellGateRecurrentWeightsName,
            outputGateWeightsName, outputGateBiasName, outputGateRecurrentWeightsName
        ]
        
        var totalSize = 0
        var buffers: [MetalBuffer] = []
        
        for name in params {
            if let buffer = await parameterStore.getParameter(name: name) {
                totalSize += buffer.count
                buffers.append(buffer)
            }
        }
        
        guard !buffers.isEmpty else { return nil }
        
        // Concatenate all parameters
        let concatenated = try? await allocateBuffer(size: totalSize)
        guard let result = concatenated else { return nil }
        
        var offset = 0
        let resultPtr = result.buffer.contents().bindMemory(to: Float.self, capacity: result.count)
        
        for buffer in buffers {
            let bufferPtr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
            for i in 0..<buffer.count {
                resultPtr[offset + i] = bufferPtr[i]
            }
            offset += buffer.count
        }
        
        return result
    }
    
    public func getParameterCount() async -> Int {
        // 4 gates * (input weights + recurrent weights + bias)
        let inputWeights = hiddenSize * inputSize * 4
        let recurrentWeights = hiddenSize * hiddenSize * 4
        let biases = hiddenSize * 4
        return inputWeights + recurrentWeights + biases
    }
    
    public func setTraining(_ training: Bool) {
        self.isTraining = training
    }
    
    public func zeroGradients() async {
        // Zero all gradient buffers
        let gradientNames = [
            inputGateWeightsName + gradientSuffix,
            inputGateBiasName + gradientSuffix,
            inputGateRecurrentWeightsName + gradientSuffix,
            forgetGateWeightsName + gradientSuffix,
            forgetGateBiasName + gradientSuffix,
            forgetGateRecurrentWeightsName + gradientSuffix,
            cellGateWeightsName + gradientSuffix,
            cellGateBiasName + gradientSuffix,
            cellGateRecurrentWeightsName + gradientSuffix,
            outputGateWeightsName + gradientSuffix,
            outputGateBiasName + gradientSuffix,
            outputGateRecurrentWeightsName + gradientSuffix
        ]
        
        for name in gradientNames {
            if let gradBuffer = await parameterStore.getGradient(name: name) {
                zeroBuffer(gradBuffer)
            }
        }
    }
    
    public func scaleGradients(_ scale: Float) async {
        // Scale all gradient buffers
        let gradientNames = [
            inputGateWeightsName + gradientSuffix,
            inputGateBiasName + gradientSuffix,
            inputGateRecurrentWeightsName + gradientSuffix,
            forgetGateWeightsName + gradientSuffix,
            forgetGateBiasName + gradientSuffix,
            forgetGateRecurrentWeightsName + gradientSuffix,
            cellGateWeightsName + gradientSuffix,
            cellGateBiasName + gradientSuffix,
            cellGateRecurrentWeightsName + gradientSuffix,
            outputGateWeightsName + gradientSuffix,
            outputGateBiasName + gradientSuffix,
            outputGateRecurrentWeightsName + gradientSuffix
        ]
        
        for name in gradientNames {
            if let gradBuffer = await parameterStore.getGradient(name: name) {
                let ptr = gradBuffer.buffer.contents().bindMemory(to: Float.self, capacity: gradBuffer.count)
                for i in 0..<gradBuffer.count {
                    ptr[i] *= scale
                }
            }
        }
    }
    
    public func updateParametersWithOptimizer(_ optimizer: any Optimizer) async throws {
        // Update all parameters using the optimizer
        let paramPairs = [
            (inputGateWeightsName, inputGateWeightsName + gradientSuffix),
            (inputGateBiasName, inputGateBiasName + gradientSuffix),
            (inputGateRecurrentWeightsName, inputGateRecurrentWeightsName + gradientSuffix),
            (forgetGateWeightsName, forgetGateWeightsName + gradientSuffix),
            (forgetGateBiasName, forgetGateBiasName + gradientSuffix),
            (forgetGateRecurrentWeightsName, forgetGateRecurrentWeightsName + gradientSuffix),
            (cellGateWeightsName, cellGateWeightsName + gradientSuffix),
            (cellGateBiasName, cellGateBiasName + gradientSuffix),
            (cellGateRecurrentWeightsName, cellGateRecurrentWeightsName + gradientSuffix),
            (outputGateWeightsName, outputGateWeightsName + gradientSuffix),
            (outputGateBiasName, outputGateBiasName + gradientSuffix),
            (outputGateRecurrentWeightsName, outputGateRecurrentWeightsName + gradientSuffix)
        ]
        
        for (paramName, gradName) in paramPairs {
            guard let param = await parameterStore.getParameter(name: paramName),
                  let grad = await parameterStore.getGradient(name: gradName) else {
                continue
            }
            
            let paramPtr = param.buffer.contents().bindMemory(to: Float.self, capacity: param.count)
            let gradPtr = grad.buffer.contents().bindMemory(to: Float.self, capacity: grad.count)
            
            for i in 0..<param.count {
                let newValue = await optimizer.update(
                    parameter: paramPtr[i],
                    gradient: gradPtr[i],
                    name: "\(paramName)_\(i)"
                )
                paramPtr[i] = newValue
            }
        }
    }
    
    // Add this method if it doesn't exist
    public func getParameterStore() async -> ParameterStore {
        return parameterStore
    }
    
    // MARK: - Helper Methods
    
    private func allocateBuffer(size: Int) async throws -> MetalBuffer {
        try await metalPipeline.allocateBuffer(size: size)
    }
    
    private func extractTimestep(
        from input: MetalBuffer,
        timestep: Int,
        batchSize: Int,
        inputSize: Int,
        seqLen: Int
    ) async throws -> MetalBuffer {
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.MatrixOperation.extractTimestep.rawValue)
        
        let output = try await allocateBuffer(size: batchSize * inputSize)
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            
            var timestepValue = UInt32(timestep)
            var batchSizeValue = UInt32(batchSize)
            var inputSizeValue = UInt32(inputSize)
            var seqLenValue = UInt32(seqLen)
            
            encoder.setBytes(&timestepValue, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&batchSizeValue, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&inputSizeValue, length: MemoryLayout<UInt32>.size, index: 4)
            encoder.setBytes(&seqLenValue, length: MemoryLayout<UInt32>.size, index: 5)
            
            let gridSize = MTLSize(width: batchSize * inputSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, gridSize.width), height: 1, depth: 1)
            
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        return output
    }
    
    private func computeGates(
        x_t: MetalBuffer,
        h_t: MetalBuffer,
        operations: MetalMLOperations
    ) async throws -> (i_t: MetalBuffer, f_t: MetalBuffer, g_t: MetalBuffer, o_t: MetalBuffer) {
        // Get parameters
        guard let Wi = await parameterStore.getParameter(name: inputGateWeightsName),
              let Wf = await parameterStore.getParameter(name: forgetGateWeightsName),
              let Wg = await parameterStore.getParameter(name: cellGateWeightsName),
              let Wo = await parameterStore.getParameter(name: outputGateWeightsName),
              let Ui = await parameterStore.getParameter(name: inputGateRecurrentWeightsName),
              let Uf = await parameterStore.getParameter(name: forgetGateRecurrentWeightsName),
              let Ug = await parameterStore.getParameter(name: cellGateRecurrentWeightsName),
              let Uo = await parameterStore.getParameter(name: outputGateRecurrentWeightsName),
              let bi = await parameterStore.getParameter(name: inputGateBiasName),
              let bf = await parameterStore.getParameter(name: forgetGateBiasName),
              let bg = await parameterStore.getParameter(name: cellGateBiasName),
              let bo = await parameterStore.getParameter(name: outputGateBiasName) else {
            throw MetalMLError.parameterNotFound(name: "LSTM gate parameters")
        }
        
        // Compute gate activations
        // i_t = sigmoid(Wi @ x_t + Ui @ h_t + bi)
        let i_t = try await computeGateActivation(
            W: Wi, U: Ui, b: bi,
            x_t: x_t, h_t: h_t,
            activation: .sigmoid,
            operations: operations
        )
        
        // f_t = sigmoid(Wf @ x_t + Uf @ h_t + bf)
        let f_t = try await computeGateActivation(
            W: Wf, U: Uf, b: bf,
            x_t: x_t, h_t: h_t,
            activation: .sigmoid,
            operations: operations
        )
        
        // g_t = tanh(Wg @ x_t + Ug @ h_t + bg)
        let g_t = try await computeGateActivation(
            W: Wg, U: Ug, b: bg,
            x_t: x_t, h_t: h_t,
            activation: .tanh,
            operations: operations
        )
        
        // o_t = sigmoid(Wo @ x_t + Uo @ h_t + bo)
        let o_t = try await computeGateActivation(
            W: Wo, U: Uo, b: bo,
            x_t: x_t, h_t: h_t,
            activation: .sigmoid,
            operations: operations
        )
        
        return (i_t, f_t, g_t, o_t)
    }
    
    private func computeGateActivation(
        W: MetalBuffer, U: MetalBuffer, b: MetalBuffer,
        x_t: MetalBuffer, h_t: MetalBuffer,
        activation: Activation,
        operations: MetalMLOperations
    ) async throws -> MetalBuffer {
        // Allocate temporary buffers
        let wx = try await allocateBuffer(size: hiddenSize)
        let uh = try await allocateBuffer(size: hiddenSize)
        
        // W @ x_t
        try await operations.matmul(
            W, x_t, output: wx,
            m: hiddenSize, n: 1, k: inputSize,
            useTiling: false
        )
        
        // U @ h_t
        try await operations.matmul(
            U, h_t, output: uh,
            m: hiddenSize, n: 1, k: hiddenSize,
            useTiling: false
        )
        
        // wx + uh + b
        let preActivation = try await allocateBuffer(size: hiddenSize)
        try await operations.addBuffers(wx, uh, output: preActivation)
        try await operations.addBias(matrix: preActivation, bias: b, rows: 1, cols: hiddenSize)
        
        // Apply activation
        let activated = try await allocateBuffer(size: hiddenSize)
        try await operations.applyActivation(
            preActivation,
            output: activated,
            activation: activation
        )
        
        return activated
    }
    
    private func updateCellState(
        c_prev: MetalBuffer,
        f_t: MetalBuffer,
        i_t: MetalBuffer,
        g_t: MetalBuffer,
        operations: MetalMLOperations
    ) async throws -> MetalBuffer {
        // c_t = f_t * c_{t-1} + i_t * g_t
        let fc = try await allocateBuffer(size: hiddenSize)
        let ig = try await allocateBuffer(size: hiddenSize)
        let c_t = try await allocateBuffer(size: hiddenSize)
        
        // f_t * c_{t-1}
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let multiplyPipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.ElementwiseOperation.elementMultiply.rawValue)
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(multiplyPipeline)
            encoder.setBuffer(f_t.buffer, offset: 0, index: 0)
            encoder.setBuffer(c_prev.buffer, offset: 0, index: 1)
            encoder.setBuffer(fc.buffer, offset: 0, index: 2)
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // i_t * g_t
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(multiplyPipeline)
            encoder.setBuffer(i_t.buffer, offset: 0, index: 0)
            encoder.setBuffer(g_t.buffer, offset: 0, index: 1)
            encoder.setBuffer(ig.buffer, offset: 0, index: 2)
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // fc + ig
        try await operations.addBuffers(fc, ig, output: c_t)
        
        return c_t
    }
    
    private func updateHiddenState(
        c_t: MetalBuffer,
        o_t: MetalBuffer,
        operations: MetalMLOperations
    ) async throws -> MetalBuffer {
        // h_t = o_t * tanh(c_t)
        let tanh_c_t = try await allocateBuffer(size: hiddenSize)
        try await operations.applyActivation(
            c_t,
            output: tanh_c_t,
            activation: .tanh
        )
        
        let h_t = try await allocateBuffer(size: hiddenSize)
        
        // Element-wise multiply
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let multiplyPipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.ElementwiseOperation.elementMultiply.rawValue)
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(multiplyPipeline)
            encoder.setBuffer(o_t.buffer, offset: 0, index: 0)
            encoder.setBuffer(tanh_c_t.buffer, offset: 0, index: 1)
            encoder.setBuffer(h_t.buffer, offset: 0, index: 2)
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        return h_t
    }
    
    private func concatenateSequenceOutputs(_ outputs: [MetalBuffer]) async throws -> MetalBuffer {
        let totalSize = outputs.reduce(0) { $0 + $1.count }
        let concatenated = try await allocateBuffer(size: totalSize)
        
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let concatenatePipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.MatrixOperation.concatenateSequence.rawValue)
        
        // First, copy all outputs to a temporary buffer
        let tempBuffer = try await allocateBuffer(size: totalSize)
        var offset = 0
        let tempPtr = tempBuffer.buffer.contents().bindMemory(to: Float.self, capacity: tempBuffer.count)
        
        for output in outputs {
            let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
            for i in 0..<output.count {
                tempPtr[offset + i] = outputPtr[i]
            }
            offset += output.count
        }
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(concatenatePipeline)
            encoder.setBuffer(tempBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(concatenated.buffer, offset: 0, index: 1)
            
            var batchSizeValue = UInt32(batchSize)
            var seqLenValue = UInt32(sequenceLength)
            var hiddenSizeValue = UInt32(hiddenSize)
            
            encoder.setBytes(&batchSizeValue, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&seqLenValue, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&hiddenSizeValue, length: MemoryLayout<UInt32>.size, index: 4)
            
            let gridSize = MTLSize(width: totalSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, gridSize.width), height: 1, depth: 1)
            
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        return concatenated
    }
    
    private func extractTimestepGradient(
        from gradOutput: MetalBuffer,
        timestep: Int,
        hiddenSize: Int
    ) async throws -> MetalBuffer {
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let pipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.MatrixOperation.extractTimestepGrad.rawValue)
        
        let output = try await allocateBuffer(size: hiddenSize)
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(gradOutput.buffer, offset: 0, index: 0)
            encoder.setBuffer(output.buffer, offset: 0, index: 1)
            
            var timestepValue = UInt32(timestep)
            var hiddenSizeValue = UInt32(hiddenSize)
            var seqLenValue = UInt32(sequenceLength)
            
            encoder.setBytes(&timestepValue, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&hiddenSizeValue, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&seqLenValue, length: MemoryLayout<UInt32>.size, index: 4)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, gridSize.width), height: 1, depth: 1)
            
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        return output
    }
    
    private struct GateGradients {
        let grad_i_t: MetalBuffer
        let grad_f_t: MetalBuffer
        let grad_g_t: MetalBuffer
        let grad_o_t: MetalBuffer
        let grad_c_prev: MetalBuffer
        let grad_h_prev: MetalBuffer
        let grad_x_t: MetalBuffer
    }
    
    private func computeGateGradients(
        grad_h_t: MetalBuffer,
        grad_c_next: MetalBuffer,
        timestep: Int,
        operations: MetalMLOperations
    ) async throws -> GateGradients {
        // Get saved states
        let h_t = hiddenStates[timestep]
        let c_t = cellStates[timestep]
        let c_prev = timestep > 0 ? cellStates[timestep - 1] : try await allocateBuffer(size: hiddenSize)
        
        // Zero initialize c_prev if it's the first timestep
        if timestep == 0 {
            zeroBuffer(c_prev)
        }
        
        let i_t = gateOutputs["input"]![timestep]
        let f_t = gateOutputs["forget"]![timestep]
        let g_t = gateOutputs["cell"]![timestep]
        let o_t = gateOutputs["output"]![timestep]
        
        // Get h_prev for computing grad_h_prev
        let h_prev = timestep > 0 ? hiddenStates[timestep - 1] : try await allocateBuffer(size: hiddenSize)
        if timestep == 0 {
            zeroBuffer(h_prev)
        }
        
        // Allocate gradient buffers
        let grad_o_t = try await allocateBuffer(size: hiddenSize)
        let grad_c_t = try await allocateBuffer(size: hiddenSize)
        let grad_i_t = try await allocateBuffer(size: hiddenSize)
        let grad_f_t = try await allocateBuffer(size: hiddenSize)
        let grad_g_t = try await allocateBuffer(size: hiddenSize)
        let grad_c_prev = try await allocateBuffer(size: hiddenSize)
        let grad_h_prev = try await allocateBuffer(size: hiddenSize)
        let grad_x_t = try await allocateBuffer(size: inputSize)
        
        // Get Metal resources
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        
        // Step 1: Compute grad_o_t = grad_h_t * tanh(c_t) * o_t * (1 - o_t)
        let tanh_c_t = try await allocateBuffer(size: hiddenSize)
        try await operations.applyActivation(c_t, output: tanh_c_t, activation: .tanh)
        
        // First multiply grad_h_t * tanh(c_t)
        let temp_grad_o = try await allocateBuffer(size: hiddenSize)
        let elementMultiplyPipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.ElementwiseOperation.elementMultiply.rawValue)
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(elementMultiplyPipeline)
            encoder.setBuffer(grad_h_t.buffer, offset: 0, index: 0)
            encoder.setBuffer(tanh_c_t.buffer, offset: 0, index: 1)
            encoder.setBuffer(temp_grad_o.buffer, offset: 0, index: 2)
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // Then apply sigmoid derivative: result * o_t * (1 - o_t)
        let sigmoidBackwardPipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.ActivationFunction.sigmoidBackward.rawValue)
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(sigmoidBackwardPipeline)
            encoder.setBuffer(temp_grad_o.buffer, offset: 0, index: 0)  // gradOutput
            encoder.setBuffer(o_t.buffer, offset: 0, index: 1)          // output (sigmoid value)
            encoder.setBuffer(grad_o_t.buffer, offset: 0, index: 2)     // gradInput
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // Step 2: Compute grad_c_t = grad_h_t * o_t * (1 - tanh²(c_t)) + grad_c_next
        // First compute grad_h_t * o_t
        let temp_grad_c = try await allocateBuffer(size: hiddenSize)
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(elementMultiplyPipeline)
            encoder.setBuffer(grad_h_t.buffer, offset: 0, index: 0)
            encoder.setBuffer(o_t.buffer, offset: 0, index: 1)
            encoder.setBuffer(temp_grad_c.buffer, offset: 0, index: 2)
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // Apply tanh derivative
        let tanhBackwardPipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.ActivationFunction.tanhBackward.rawValue)
        let tanh_deriv_result = try await allocateBuffer(size: hiddenSize)
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(tanhBackwardPipeline)
            encoder.setBuffer(temp_grad_c.buffer, offset: 0, index: 0)  // gradOutput
            encoder.setBuffer(tanh_c_t.buffer, offset: 0, index: 1)     // output (tanh value)
            encoder.setBuffer(tanh_deriv_result.buffer, offset: 0, index: 2) // gradInput
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // Add grad_c_next to get final grad_c_t
        try await operations.addBuffers(tanh_deriv_result, grad_c_next, output: grad_c_t)
        
        // Step 3: Compute gate gradients
        // grad_f_t = grad_c_t * c_{t-1} * f_t * (1 - f_t)
        let temp1 = try await allocateBuffer(size: hiddenSize)
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(elementMultiplyPipeline)
            encoder.setBuffer(grad_c_t.buffer, offset: 0, index: 0)
            encoder.setBuffer(c_prev.buffer, offset: 0, index: 1)
            encoder.setBuffer(temp1.buffer, offset: 0, index: 2)
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // Apply sigmoid derivative for forget gate
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(sigmoidBackwardPipeline)
            encoder.setBuffer(temp1.buffer, offset: 0, index: 0)     // gradOutput
            encoder.setBuffer(f_t.buffer, offset: 0, index: 1)       // output (sigmoid value)
            encoder.setBuffer(grad_f_t.buffer, offset: 0, index: 2)  // gradInput
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // grad_i_t = grad_c_t * g_t * i_t * (1 - i_t)
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(elementMultiplyPipeline)
            encoder.setBuffer(grad_c_t.buffer, offset: 0, index: 0)
            encoder.setBuffer(g_t.buffer, offset: 0, index: 1)
            encoder.setBuffer(temp1.buffer, offset: 0, index: 2)
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // Apply sigmoid derivative for input gate
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(sigmoidBackwardPipeline)
            encoder.setBuffer(temp1.buffer, offset: 0, index: 0)     // gradOutput
            encoder.setBuffer(i_t.buffer, offset: 0, index: 1)       // output (sigmoid value)
            encoder.setBuffer(grad_i_t.buffer, offset: 0, index: 2)  // gradInput
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // grad_g_t = grad_c_t * i_t * (1 - g_t²)
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(elementMultiplyPipeline)
            encoder.setBuffer(grad_c_t.buffer, offset: 0, index: 0)
            encoder.setBuffer(i_t.buffer, offset: 0, index: 1)
            encoder.setBuffer(temp1.buffer, offset: 0, index: 2)
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // Apply tanh derivative for cell gate
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(tanhBackwardPipeline)
            encoder.setBuffer(temp1.buffer, offset: 0, index: 0)     // gradOutput
            encoder.setBuffer(g_t.buffer, offset: 0, index: 1)       // output (tanh value)
            encoder.setBuffer(grad_g_t.buffer, offset: 0, index: 2)  // gradInput
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // Step 4: Compute grad_c_prev = grad_c_t * f_t
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(elementMultiplyPipeline)
            encoder.setBuffer(grad_c_t.buffer, offset: 0, index: 0)
            encoder.setBuffer(f_t.buffer, offset: 0, index: 1)
            encoder.setBuffer(grad_c_prev.buffer, offset: 0, index: 2)
            
            var size = UInt32(hiddenSize)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 3)
            
            let gridSize = MTLSize(width: hiddenSize, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(256, hiddenSize), height: 1, depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // Step 5: Compute input gradients
        // grad_x_t and grad_h_prev require backpropagation through the gates
        try await computeInputGradients(
            grad_i_t: grad_i_t,
            grad_f_t: grad_f_t,
            grad_g_t: grad_g_t,
            grad_o_t: grad_o_t,
            h_prev: h_prev,
            timestep: timestep,
            grad_x_t: grad_x_t,
            grad_h_prev: grad_h_prev,
            operations: operations
        )
        
        return GateGradients(
            grad_i_t: grad_i_t,
            grad_f_t: grad_f_t,
            grad_g_t: grad_g_t,
            grad_o_t: grad_o_t,
            grad_c_prev: grad_c_prev,
            grad_h_prev: grad_h_prev,
            grad_x_t: grad_x_t
        )
    }
    
    private func accumulateParameterGradients(
        timestep: Int,
        gradients: GateGradients,
        operations: MetalMLOperations
    ) async throws {
        // Get input for this timestep
        let x_t = try await extractTimestep(
            from: lastInputSequence!,
            timestep: timestep,
            batchSize: batchSize,
            inputSize: inputSize,
            seqLen: sequenceLength
        )
        
        // Get h_prev
        let h_prev = timestep > 0 ? hiddenStates[timestep - 1] : try await allocateBuffer(size: hiddenSize)
        if timestep == 0 {
            zeroBuffer(h_prev)
        }
        
        // Accumulate gradients for each gate
        let gates = [
            (gradients.grad_i_t, inputGateWeightsName, inputGateRecurrentWeightsName, inputGateBiasName),
            (gradients.grad_f_t, forgetGateWeightsName, forgetGateRecurrentWeightsName, forgetGateBiasName),
            (gradients.grad_g_t, cellGateWeightsName, cellGateRecurrentWeightsName, cellGateBiasName),
            (gradients.grad_o_t, outputGateWeightsName, outputGateRecurrentWeightsName, outputGateBiasName)
        ]
        
        for (gateGrad, weightsName, recurrentWeightsName, biasName) in gates {
            // Get gradient buffers
            guard let weightsGrad = await parameterStore.getGradient(name: weightsName + gradientSuffix),
                  let recurrentWeightsGrad = await parameterStore.getGradient(name: recurrentWeightsName + gradientSuffix),
                  let biasGrad = await parameterStore.getGradient(name: biasName + gradientSuffix) else {
                throw MetalMLError.parameterNotFound(name: "Gradient buffers")
            }
            
            // Compute weight gradients: dW += grad_gate @ x_t^T
            // Note: We need to accumulate across timesteps
            try await accumulateOuterProduct(
                a: gateGrad,
                b: x_t,
                output: weightsGrad,
                m: hiddenSize,
                n: inputSize,
                operations: operations
            )
            
            // Compute recurrent weight gradients: dU += grad_gate @ h_prev^T
            try await accumulateOuterProduct(
                a: gateGrad,
                b: h_prev,
                output: recurrentWeightsGrad,
                m: hiddenSize,
                n: hiddenSize,
                operations: operations
            )
            
            // Compute bias gradients: db += grad_gate
            try await accumulateBiasGradient(
                gateGrad: gateGrad,
                biasGrad: biasGrad,
                operations: operations
            )
        }
    }
    
    private func computeInputGradients(
        grad_i_t: MetalBuffer,
        grad_f_t: MetalBuffer,
        grad_g_t: MetalBuffer,
        grad_o_t: MetalBuffer,
        h_prev: MetalBuffer,
        timestep: Int,
        grad_x_t: MetalBuffer,
        grad_h_prev: MetalBuffer,
        operations: MetalMLOperations
    ) async throws {
        // Get weight parameters
        let gates = [
            (grad_i_t, inputGateWeightsName, inputGateRecurrentWeightsName),
            (grad_f_t, forgetGateWeightsName, forgetGateRecurrentWeightsName),
            (grad_g_t, cellGateWeightsName, cellGateRecurrentWeightsName),
            (grad_o_t, outputGateWeightsName, outputGateRecurrentWeightsName)
        ]
        
        // Initialize gradients to zero
        zeroBuffer(grad_x_t)
        zeroBuffer(grad_h_prev)
        
        // Temporary buffers for accumulation
        let temp_x_grad = try await allocateBuffer(size: inputSize)
        let temp_h_grad = try await allocateBuffer(size: hiddenSize)
        
        // Get shader resources
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        
        for (gateGrad, weightsName, recurrentWeightsName) in gates {
            guard let W = await parameterStore.getParameter(name: weightsName),
                  let U = await parameterStore.getParameter(name: recurrentWeightsName) else {
                throw MetalMLError.parameterNotFound(name: "Gate weights")
            }
            
            // grad_x_t += W^T @ grad_gate
            // W is [hiddenSize x inputSize], grad_gate is [hiddenSize x 1]
            // W^T @ grad_gate gives [inputSize x 1]
            // Use matmul_backward_B which computes A^T @ gradC
            let matmulBackwardBPipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.MatrixOperation.matmulBackwardB.rawValue)
            
            try await commandQueue.submitAsync { commandBuffer in
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw MetalMLError.encoderCreationFailed
                }
                
                encoder.setComputePipelineState(matmulBackwardBPipeline)
                encoder.setBuffer(W.buffer, offset: 0, index: 0)  // A (transposed in shader)
                encoder.setBuffer(gateGrad.buffer, offset: 0, index: 1)  // gradC
                encoder.setBuffer(temp_x_grad.buffer, offset: 0, index: 2)  // gradB (output)
                
                var m = UInt32(hiddenSize)  // rows of W
                var n = UInt32(1)  // cols of result
                var k = UInt32(inputSize)  // cols of W
                
                encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)
                
                let gridSize = MTLSize(width: 1, height: inputSize, depth: 1)
                let threadgroupSize = MTLSize(width: 1, height: min(16, inputSize), depth: 1)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
                encoder.endEncoding()
            }
            
            try await operations.addBuffers(grad_x_t, temp_x_grad, output: grad_x_t)
            
            // grad_h_prev += U^T @ grad_gate
            // U is [hiddenSize x hiddenSize], grad_gate is [hiddenSize x 1]
            // U^T @ grad_gate gives [hiddenSize x 1]
            try await commandQueue.submitAsync { commandBuffer in
                guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                    throw MetalMLError.encoderCreationFailed
                }
                
                encoder.setComputePipelineState(matmulBackwardBPipeline)
                encoder.setBuffer(U.buffer, offset: 0, index: 0)  // A (transposed in shader)
                encoder.setBuffer(gateGrad.buffer, offset: 0, index: 1)  // gradC
                encoder.setBuffer(temp_h_grad.buffer, offset: 0, index: 2)  // gradB (output)
                
                var m = UInt32(hiddenSize)  // rows of U
                var n = UInt32(1)  // cols of result
                var k = UInt32(hiddenSize)  // cols of U
                
                encoder.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
                encoder.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
                encoder.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)
                
                let gridSize = MTLSize(width: 1, height: hiddenSize, depth: 1)
                let threadgroupSize = MTLSize(width: 1, height: min(16, hiddenSize), depth: 1)
                encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
                encoder.endEncoding()
            }
            
            try await operations.addBuffers(grad_h_prev, temp_h_grad, output: grad_h_prev)
        }
    }
    
    private func accumulateOuterProduct(
        a: MetalBuffer,
        b: MetalBuffer,
        output: MetalBuffer,
        m: Int,
        n: Int,
        operations: MetalMLOperations
    ) async throws {
        // Compute outer product a @ b^T and accumulate to output
        let temp = try await allocateBuffer(size: m * n)
        
        // Outer product: result[i,j] = a[i] * b[j]
        let commandQueue = await metalPipeline.getMetalCommandQueue()
        let shaderLibrary = await metalPipeline.getShaderLibrary()
        let outerProductPipeline = try shaderLibrary.pipeline(for: MLShaderLibrary.MatrixOperation.outerProduct.rawValue)
        
        try await commandQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.encoderCreationFailed
            }
            
            encoder.setComputePipelineState(outerProductPipeline)
            encoder.setBuffer(a.buffer, offset: 0, index: 0)
            encoder.setBuffer(b.buffer, offset: 0, index: 1)
            encoder.setBuffer(temp.buffer, offset: 0, index: 2)
            
            var mValue = UInt32(m)
            var nValue = UInt32(n)
            encoder.setBytes(&mValue, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&nValue, length: MemoryLayout<UInt32>.size, index: 4)
            
            let gridSize = MTLSize(width: n, height: m, depth: 1)
            let threadgroupSize = MTLSize(width: min(16, n), height: min(16, m), depth: 1)
            encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
        
        // Accumulate to output
        try await operations.addBuffers(output, temp, output: output)
    }
    
    private func accumulateBiasGradient(
        gateGrad: MetalBuffer,
        biasGrad: MetalBuffer,
        operations: MetalMLOperations
    ) async throws {
        // Simply accumulate the gate gradient to bias gradient
        try await operations.addBuffers(biasGrad, gateGrad, output: biasGrad)
    }
    
    private func storeTimestepInputGradient(
        gradInput: MetalBuffer,
        timestep: Int,
        gradient: MetalBuffer
    ) async throws {
        // Store gradient at appropriate offset
        let offset = timestep * batchSize * inputSize
        let gradInputPtr = gradInput.buffer.contents().bindMemory(to: Float.self, capacity: gradInput.count)
        let gradientPtr = gradient.buffer.contents().bindMemory(to: Float.self, capacity: gradient.count)
        
        for i in 0..<gradient.count {
            gradInputPtr[offset + i] = gradientPtr[i]
        }
    }
    
    private func zeroBuffer(_ buffer: MetalBuffer) {
        let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
        for i in 0..<buffer.count {
            ptr[i] = 0.0
        }
    }
}

// MARK: - Extensions

// Extension removed as getMetalCommandQueue() is already implemented in MetalMLPipeline
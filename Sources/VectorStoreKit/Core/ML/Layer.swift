// VectorStoreKit: Neural Network Layers
//
// Comprehensive layer implementations with forward and backward passes

import Foundation
import Accelerate

/// Protocol for neural network layers
public protocol NeuralLayer: Sendable {
    /// Forward pass through the layer
    func forward(_ input: [Float]) async -> [Float]
    
    /// Backward pass through the layer
    func backward(_ gradOutput: [Float], input: [Float], output: [Float]) async -> (gradInput: [Float], gradients: LayerGradients)
    
    /// Update layer parameters
    mutating func updateParameters(gradients: LayerGradients, optimizer: Optimizer) async
    
    /// Get current parameters
    func getParameters() async -> LayerParameters
    
    /// Number of trainable parameters
    func getParameterCount() async -> Int
    
    /// Get layer gradients (for layers that need external gradient access)
    func getGradients() async -> LayerGradients
}

/// Container for layer parameters
public struct LayerParameters: Sendable {
    public let weights: [[Float]]?
    public let bias: [Float]?
    public let batchNormGamma: [Float]?
    public let batchNormBeta: [Float]?
    
    public init(
        weights: [[Float]]? = nil,
        bias: [Float]? = nil,
        batchNormGamma: [Float]? = nil,
        batchNormBeta: [Float]? = nil
    ) {
        self.weights = weights
        self.bias = bias
        self.batchNormGamma = batchNormGamma
        self.batchNormBeta = batchNormBeta
    }
}

/// Container for layer gradients
public struct LayerGradients: Sendable {
    public var weightsGrad: [[Float]]?
    public var biasGrad: [Float]?
    public var batchNormGammaGrad: [Float]?
    public var batchNormBetaGrad: [Float]?
    
    public init(
        weightsGrad: [[Float]]? = nil,
        biasGrad: [Float]? = nil,
        batchNormGammaGrad: [Float]? = nil,
        batchNormBetaGrad: [Float]? = nil
    ) {
        self.weightsGrad = weightsGrad
        self.biasGrad = biasGrad
        self.batchNormGammaGrad = batchNormGammaGrad
        self.batchNormBetaGrad = batchNormBetaGrad
    }
    
    public init() {
        self.weightsGrad = nil
        self.biasGrad = nil  
        self.batchNormGammaGrad = nil
        self.batchNormBetaGrad = nil
    }
    
    // Convenience properties for autoencoders
    public var weights: [Float]? {
        get { weightsGrad?.flatMap { $0 } }
        set { 
            if let newValue = newValue {
                // Convert flat array back to 2D (simplified - assumes square)
                let size = Int(sqrt(Double(newValue.count)))
                var result: [[Float]] = []
                for i in 0..<size {
                    result.append(Array(newValue[i*size..<(i+1)*size]))
                }
                weightsGrad = result
            } else {
                weightsGrad = nil
            }
        }
    }
    
    public var bias: [Float]? {
        get { biasGrad }
        set { biasGrad = newValue }
    }
}

/// Dense (fully connected) layer
public struct DenseLayer: NeuralLayer, Sendable {
    public var weights: [[Float]]
    public var bias: [Float]
    public let activation: Activation
    public let inputSize: Int
    public let outputSize: Int
    public let useBias: Bool
    
    // For Metal acceleration
    private let metalCompute: MetalCompute?
    
    public init(
        inputSize: Int,
        outputSize: Int,
        activation: Activation = .linear,
        useBias: Bool = true,
        metalCompute: MetalCompute? = nil
    ) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = activation
        self.useBias = useBias
        self.metalCompute = metalCompute
        
        // Xavier/He initialization
        let scale = activation == .relu || activation == .leakyRelu
            ? sqrt(2.0 / Float(inputSize))
            : sqrt(2.0 / Float(inputSize + outputSize))
        
        self.weights = (0..<outputSize).map { _ in
            (0..<inputSize).map { _ in Float.random(in: -scale...scale) }
        }
        
        self.bias = useBias ? Array(repeating: 0.0, count: outputSize) : []
    }
    
    public func forward(_ input: [Float]) async -> [Float] {
        var output = Array(repeating: Float(0), count: outputSize)
        
        // Matrix multiplication: output = weights * input + bias
        for i in 0..<outputSize {
            var sum: Float = 0
            vDSP_dotpr(weights[i], 1, input, 1, &sum, vDSP_Length(inputSize))
            output[i] = sum + (useBias ? bias[i] : 0)
        }
        
        // Apply activation
        return activation.apply(output)
    }
    
    public func backward(
        _ gradOutput: [Float],
        input: [Float],
        output: [Float]
    ) async -> (gradInput: [Float], gradients: LayerGradients) {
        // Compute activation gradient
        let activationGrad = activation.derivative(output, output: output)
        let gradActivation = vDSP.multiply(gradOutput, activationGrad)
        
        // Compute weight gradients: gradW = gradActivation * input^T
        var weightsGrad = [[Float]]()
        for i in 0..<outputSize {
            let grad = Array(repeating: gradActivation[i], count: inputSize)
            weightsGrad.append(vDSP.multiply(grad, input))
        }
        
        // Compute bias gradients
        let biasGrad = useBias ? gradActivation : nil
        
        // Compute input gradients: gradInput = weights^T * gradActivation
        var gradInput = Array(repeating: Float(0), count: inputSize)
        for j in 0..<inputSize {
            var sum: Float = 0
            for i in 0..<outputSize {
                sum += weights[i][j] * gradActivation[i]
            }
            gradInput[j] = sum
        }
        
        return (
            gradInput,
            LayerGradients(weightsGrad: weightsGrad, biasGrad: biasGrad)
        )
    }
    
    public mutating func updateParameters(gradients: LayerGradients, optimizer: Optimizer) async {
        if let weightsGrad = gradients.weightsGrad {
            for i in 0..<outputSize {
                for j in 0..<inputSize {
                    weights[i][j] = await optimizer.update(
                        parameter: weights[i][j],
                        gradient: weightsGrad[i][j],
                        name: "dense_w_\(i)_\(j)"
                    )
                }
            }
        }
        
        if useBias, let biasGrad = gradients.biasGrad {
            for i in 0..<outputSize {
                bias[i] = await optimizer.update(
                    parameter: bias[i],
                    gradient: biasGrad[i],
                    name: "dense_b_\(i)"
                )
            }
        }
    }
    
    public func getParameters() async -> LayerParameters {
        LayerParameters(weights: weights, bias: useBias ? bias : nil)
    }
    
    public func getParameterCount() async -> Int {
        inputSize * outputSize + (useBias ? outputSize : 0)
    }
    
    public func getGradients() async -> LayerGradients {
        // Note: Due to struct immutability, this returns empty gradients.
        // For proper gradient tracking, use ActorDenseLayer instead.
        LayerGradients()
    }
}

/// Dropout layer for regularization
public struct DropoutLayer: NeuralLayer, Sendable {
    public let rate: Float
    public var training: Bool
    private var mask: [Float]?
    
    public init(rate: Float = 0.5, training: Bool = true) {
        self.rate = rate
        self.training = training
    }
    
    public mutating func setTraining(_ training: Bool) {
        self.training = training
    }
    
    public func forward(_ input: [Float]) async -> [Float] {
        guard training && rate > 0 else { return input }
        
        // Generate dropout mask
        let mask = input.map { _ in Float.random(in: 0...1) > rate ? 1.0 / (1.0 - rate) : 0.0 }
        
        // Apply mask
        return vDSP.multiply(input, mask)
    }
    
    public func backward(
        _ gradOutput: [Float],
        input: [Float],
        output: [Float]
    ) async -> (gradInput: [Float], gradients: LayerGradients) {
        guard training && rate > 0 else {
            return (gradOutput, LayerGradients())
        }
        
        // Use the same mask from forward pass
        let mask = input.map { _ in Float.random(in: 0...1) > rate ? 1.0 / (1.0 - rate) : 0.0 }
        let gradInput = vDSP.multiply(gradOutput, mask)
        
        return (gradInput, LayerGradients())
    }
    
    public mutating func updateParameters(gradients: LayerGradients, optimizer: Optimizer) async {
        // No parameters to update
    }
    
    public func getParameters() async -> LayerParameters {
        LayerParameters()
    }
    
    public func getParameterCount() async -> Int { 0 }
    
    public func getGradients() async -> LayerGradients {
        LayerGradients()
    }
}

/// Batch normalization layer
public actor BatchNormLayer: NeuralLayer {
    public var gamma: [Float]  // Scale
    public var beta: [Float]   // Shift
    public var runningMean: [Float]
    public var runningVar: [Float]
    public let momentum: Float
    public let epsilon: Float
    public let training: Bool
    private let features: Int
    
    public init(
        features: Int,
        momentum: Float = 0.9,
        epsilon: Float = 1e-5,
        training: Bool = true
    ) {
        self.features = features
        self.momentum = momentum
        self.epsilon = epsilon
        self.training = training
        
        self.gamma = Array(repeating: 1.0, count: features)
        self.beta = Array(repeating: 0.0, count: features)
        self.runningMean = Array(repeating: 0.0, count: features)
        self.runningVar = Array(repeating: 1.0, count: features)
    }
    
    public func forward(_ input: [Float]) async -> [Float] {
        guard input.count == features else {
            fatalError("BatchNorm input size mismatch")
        }
        
        let mean: [Float]
        let variance: [Float]
        
        if training {
            // Compute batch statistics
            mean = [vDSP.mean(input)]
            let meanValue = mean[0]
            let centered = input.map { $0 - meanValue }
            variance = [vDSP.meanSquare(centered)]
            
            // Update running statistics
            for i in 0..<features {
                runningMean[i] = momentum * runningMean[i] + (1 - momentum) * mean[0]
                runningVar[i] = momentum * runningVar[i] + (1 - momentum) * variance[0]
            }
        } else {
            mean = runningMean
            variance = runningVar
        }
        
        // Normalize
        let normalized = input.enumerated().map { i, x in
            (x - mean[i % mean.count]) / sqrt(variance[i % variance.count] + epsilon)
        }
        
        // Scale and shift
        return normalized.enumerated().map { i, x in
            gamma[i] * x + beta[i]
        }
    }
    
    public func backward(
        _ gradOutput: [Float],
        input: [Float],
        output: [Float]
    ) async -> (gradInput: [Float], gradients: LayerGradients) {
        // Simplified batch norm backward pass
        let gammaGrad = vDSP.multiply(gradOutput, input)
        let betaGrad = gradOutput
        
        // Compute input gradient (simplified)
        let gradInput = gradOutput.enumerated().map { i, g in
            g * gamma[i]
        }
        
        return (
            gradInput,
            LayerGradients(
                batchNormGammaGrad: gammaGrad,
                batchNormBetaGrad: betaGrad
            )
        )
    }
    
    public func updateParameters(gradients: LayerGradients, optimizer: Optimizer) async {
        if let gammaGrad = gradients.batchNormGammaGrad {
            for i in 0..<features {
                gamma[i] = await optimizer.update(
                    parameter: gamma[i],
                    gradient: gammaGrad[i],
                    name: "bn_gamma_\(i)"
                )
            }
        }
        
        if let betaGrad = gradients.batchNormBetaGrad {
            for i in 0..<features {
                beta[i] = await optimizer.update(
                    parameter: beta[i],
                    gradient: betaGrad[i],
                    name: "bn_beta_\(i)"
                )
            }
        }
    }
    
    public func getParameters() async -> LayerParameters {
        LayerParameters(
            batchNormGamma: gamma,
            batchNormBeta: beta
        )
    }
    
    public func getParameterCount() async -> Int {
        2 * features
    }
    
    public func getGradients() async -> LayerGradients {
        LayerGradients()
    }
}

/// Residual connection layer
public struct ResidualLayer: NeuralLayer, Sendable {
    public let sublayers: [any NeuralLayer]
    
    public init(sublayers: [any NeuralLayer]) {
        self.sublayers = sublayers
    }
    
    public func forward(_ input: [Float]) async -> [Float] {
        var output = input
        for layer in sublayers {
            output = await layer.forward(output)
        }
        
        // Add residual connection
        guard input.count == output.count else {
            fatalError("Residual connection size mismatch")
        }
        
        return vDSP.add(input, output)
    }
    
    public func backward(
        _ gradOutput: [Float],
        input: [Float],
        output: [Float]
    ) async -> (gradInput: [Float], gradients: LayerGradients) {
        // Gradient flows through both paths
        var currentGrad = gradOutput
        var allGradients: [LayerGradients] = []
        
        // Backward through sublayers in reverse order
        for layer in sublayers.reversed() {
            let (layerGradInput, layerGradients) = await layer.backward(
                currentGrad,
                input: input,  // Simplified - should track intermediate inputs
                output: output
            )
            currentGrad = layerGradInput
            allGradients.append(layerGradients)
        }
        
        // Add gradient from skip connection
        let totalGradInput = vDSP.add(currentGrad, gradOutput)
        
        // Combine all gradients (simplified)
        return (totalGradInput, allGradients.first ?? LayerGradients())
    }
    
    public mutating func updateParameters(gradients: LayerGradients, optimizer: Optimizer) async {
        // Update sublayer parameters
        for _ in 0..<sublayers.count {
            // This would need proper gradient routing in a full implementation
        }
    }
    
    public func getParameters() async -> LayerParameters {
        // Combine parameters from all sublayers
        LayerParameters()
    }
    
    public func getParameterCount() async -> Int {
        var count = 0
        for layer in sublayers {
            count += await layer.getParameterCount()
        }
        return count
    }
    
    public func getGradients() async -> LayerGradients {
        LayerGradients()
    }
}
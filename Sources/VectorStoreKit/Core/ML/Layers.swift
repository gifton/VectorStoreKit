// VectorStoreKit: Additional Neural Network Layers
//
// Essential layers for neural clustering

import Foundation
import Accelerate

/// Batch normalization layer (simplified version)
public actor SimpleBatchNormLayer: NeuralLayer {
    // MARK: - Properties
    private let size: Int
    private var gamma: [Float]
    private var beta: [Float]
    private var runningMean: [Float]
    private var runningVar: [Float]
    private let momentum: Float
    private let epsilon: Float
    private var isTraining: Bool = true
    
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
    
    // MARK: - Initialization
    public init(
        size: Int,
        momentum: Float = 0.99,
        epsilon: Float = 1e-5
    ) {
        self.size = size
        self.momentum = momentum
        self.epsilon = epsilon
        
        // Initialize learnable parameters
        self.gamma = Array(repeating: 1.0, count: size)
        self.beta = Array(repeating: 0.0, count: size)
        
        // Initialize running statistics
        self.runningMean = Array(repeating: 0.0, count: size)
        self.runningVar = Array(repeating: 1.0, count: size)
    }
    
    // MARK: - NeuralLayer Protocol
    public func forward(_ input: [Float]) async -> [Float] {
        guard input.count == size else {
            return input
        }
        
        var output = input
        
        if isTraining {
            // Compute batch statistics
            let mean = input.reduce(0, +) / Float(size)
            let variance = input.map { pow($0 - mean, 2) }.reduce(0, +) / Float(size)
            
            // Update running statistics
            runningMean = runningMean.enumerated().map { i, oldMean in
                momentum * oldMean + (1 - momentum) * mean
            }
            runningVar = runningVar.enumerated().map { i, oldVar in
                momentum * oldVar + (1 - momentum) * variance
            }
            
            // Normalize
            output = input.enumerated().map { i, x in
                let normalized = (x - mean) / sqrt(variance + epsilon)
                return gamma[i] * normalized + beta[i]
            }
        } else {
            // Use running statistics
            output = input.enumerated().map { i, x in
                let normalized = (x - runningMean[i]) / sqrt(runningVar[i] + epsilon)
                return gamma[i] * normalized + beta[i]
            }
        }
        
        return output
    }
    
    public func backward(
        _ gradOutput: [Float],
        input: [Float],
        output: [Float]
    ) async -> (gradInput: [Float], gradients: LayerGradients) {
        // Compute batch statistics
        let mean = input.reduce(0, +) / Float(size)
        let variance = input.map { pow($0 - mean, 2) }.reduce(0, +) / Float(size)
        let stdDev = sqrt(variance + epsilon)
        
        // Compute normalized values
        let normalized = input.map { ($0 - mean) / stdDev }
        
        // Compute gradients with respect to gamma and beta
        let gradGamma = zip(gradOutput, normalized).map { $0 * $1 }
        let gradBeta = gradOutput
        
        // Compute gradient with respect to normalized input
        let gradNormalized = zip(gradOutput, gamma).map { $0 * $1 }
        
        // Compute gradient with respect to variance
        let gradVar = zip(gradNormalized, input).map { gradNorm, x in
            gradNorm * (x - mean) * -0.5 * pow(variance + epsilon, -1.5)
        }.reduce(0, +)
        
        // Compute gradient with respect to mean
        let gradMean = gradNormalized.map { -$0 / stdDev }.reduce(0, +) +
                      gradVar * (-2.0 / Float(size)) * input.map { $0 - mean }.reduce(0, +)
        
        // Compute gradient with respect to input
        let gradInput = zip(gradNormalized, input).map { pair in
            let (gradNorm, x) = pair
            return gradNorm / stdDev + gradVar * 2.0 * (x - mean) / Float(size) + gradMean / Float(size)
        }
        
        let gradients = LayerGradients(
            batchNormGammaGrad: gradGamma,
            batchNormBetaGrad: gradBeta
        )
        
        return (gradInput, gradients)
    }
    
    public func updateParameters(gradients: LayerGradients, optimizer: any Optimizer) async {
        // Update gamma and beta
        if let gammaGrads = gradients.batchNormGammaGrad {
            for i in 0..<gamma.count {
                gamma[i] = await optimizer.update(
                    parameter: gamma[i],
                    gradient: gammaGrads[i],
                    name: "bn_gamma_\(i)"
                )
            }
        }
        if let betaGrads = gradients.batchNormBetaGrad {
            for i in 0..<beta.count {
                beta[i] = await optimizer.update(
                    parameter: beta[i],
                    gradient: betaGrads[i],
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
        return size * 2 // gamma and beta
    }
    
    public func getGradients() async -> LayerGradients {
        LayerGradients()
    }
}

/// Dropout layer for regularization (actor-based version)
public actor ActorDropoutLayer: NeuralLayer {
    private let rate: Float
    private var mask: [Float] = []
    private var isTraining: Bool = true
    
    public init(rate: Float) {
        self.rate = rate
    }
    
    public func forward(_ input: [Float]) async -> [Float] {
        guard isTraining else {
            return input
        }
        
        // Generate dropout mask
        mask = input.map { _ in Float.random(in: 0..<1) > rate ? 1.0 / (1.0 - rate) : 0.0 }
        
        // Apply dropout
        return zip(input, mask).map { $0 * $1 }
    }
    
    public func backward(
        _ gradOutput: [Float],
        input: [Float],
        output: [Float]
    ) async -> (gradInput: [Float], gradients: LayerGradients) {
        // Apply mask to gradients
        let gradInput = zip(gradOutput, mask).map { $0 * $1 }
        return (gradInput, LayerGradients())
    }
    
    public func updateParameters(gradients: LayerGradients, optimizer: any Optimizer) async {
        // No parameters to update
    }
    
    public func getParameters() async -> LayerParameters {
        LayerParameters()
    }
    
    public func getParameterCount() async -> Int {
        return 0
    }
    
    public func setTraining(_ training: Bool) async {
        self.isTraining = training
    }
    
    public func getGradients() async -> LayerGradients {
        LayerGradients()
    }
}

/// Dense (fully connected) layer (actor-based version)
public actor ActorDenseLayer: NeuralLayer {
    // MARK: - Properties
    private let inputSize: Int
    private let outputSize: Int
    private var weights: [[Float]]
    private var bias: [Float]
    private let activation: Activation
    private let name: String
    private var lastGradients: LayerGradients = LayerGradients()
    
    // MARK: - Initialization
    public init(
        inputSize: Int,
        outputSize: Int,
        activation: Activation = .linear,
        name: String = "dense"
    ) {
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.activation = activation
        self.name = name
        
        // Xavier/He initialization
        let scale: Float
        switch activation {
        case .relu, .leakyRelu:
            scale = sqrt(2.0 / Float(inputSize)) // He initialization
        default:
            scale = sqrt(1.0 / Float(inputSize)) // Xavier initialization
        }
        
        // Initialize weights
        self.weights = (0..<outputSize).map { _ in
            (0..<inputSize).map { _ in Float.random(in: -scale..<scale) }
        }
        
        // Initialize bias to zero
        self.bias = Array(repeating: 0.0, count: outputSize)
    }
    
    // MARK: - NeuralLayer Protocol
    public func forward(_ input: [Float]) async -> [Float] {
        guard input.count == inputSize else {
            fatalError("Input size mismatch: expected \(inputSize), got \(input.count)")
        }
        
        var output = Array(repeating: Float(0), count: outputSize)
        
        // Matrix multiplication: output = weights * input + bias
        for i in 0..<outputSize {
            var sum: Float = 0
            for j in 0..<inputSize {
                sum += weights[i][j] * input[j]
            }
            output[i] = sum + bias[i]
        }
        
        // Apply activation
        return activation.apply(output)
    }
    
    public func backward(
        _ gradOutput: [Float],
        input: [Float],
        output: [Float]
    ) async -> (gradInput: [Float], gradients: LayerGradients) {
        // Apply activation derivative
        let activationGrad = activation.derivative(output)
        let gradActivated = zip(gradOutput, activationGrad).map { $0 * $1 }
        
        // Compute weight gradients
        var weightGradients: [[Float]] = []
        for i in 0..<outputSize {
            var rowGradients: [Float] = []
            for j in 0..<inputSize {
                rowGradients.append(gradActivated[i] * input[j])
            }
            weightGradients.append(rowGradients)
        }
        
        // Compute bias gradients
        let biasGradients = gradActivated
        
        // Compute input gradients
        var gradInput = Array(repeating: Float(0), count: inputSize)
        for j in 0..<inputSize {
            var sum: Float = 0
            for i in 0..<outputSize {
                sum += weights[i][j] * gradActivated[i]
            }
            gradInput[j] = sum
        }
        
        let gradients = LayerGradients(
            weightsGrad: weightGradients,
            biasGrad: biasGradients
        )
        
        // Store gradients for later retrieval
        lastGradients = gradients
        
        return (gradInput, gradients)
    }
    
    public func updateParameters(gradients: LayerGradients, optimizer: any Optimizer) async {
        if let weightsGrad = gradients.weightsGrad {
            for i in 0..<outputSize {
                for j in 0..<inputSize {
                    weights[i][j] = await optimizer.update(
                        parameter: weights[i][j],
                        gradient: weightsGrad[i][j],
                        name: "\(name)_w_\(i)_\(j)"
                    )
                }
            }
        }
        
        if let biasGrad = gradients.biasGrad {
            for i in 0..<outputSize {
                bias[i] = await optimizer.update(
                    parameter: bias[i],
                    gradient: biasGrad[i],
                    name: "\(name)_b_\(i)"
                )
            }
        }
    }
    
    public func getParameters() async -> LayerParameters {
        LayerParameters(
            weights: weights,
            bias: bias
        )
    }
    
    public func getParameterCount() async -> Int {
        return inputSize * outputSize + outputSize
    }
    
    public func getGradients() async -> LayerGradients {
        // Return the stored gradients from the last backward pass
        lastGradients
    }
}


// Extension to make NeuralLayer conform to training mode
extension NeuralLayer {
    public func setTraining(_ training: Bool) async {
        // Default implementation - layers can override if needed
    }
}
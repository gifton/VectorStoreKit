// VectorStoreKit: VAE-specific Layers
//
// Specialized layers for Variational Autoencoders

import Foundation

/// Reparameterization layer for VAE
public struct ReparameterizationLayer: NeuralLayer, Sendable {
    private let latentDim: Int
    private var lastMean: [Float] = []
    private var lastLogVar: [Float] = []
    private var lastEpsilon: [Float] = []
    
    public init(latentDim: Int) {
        self.latentDim = latentDim
    }
    
    public func forward(_ input: [Float]) async -> [Float] {
        guard input.count == 2 * latentDim else {
            fatalError("Reparameterization layer expects input of size 2 * latentDim")
        }
        
        // Split input into mean and log variance
        let mean = Array(input[0..<latentDim])
        let logVar = Array(input[latentDim..<input.count])
        
        // Generate random noise
        let epsilon = (0..<latentDim).map { _ in
            // Box-Muller transform for normal distribution
            let u1 = Float.random(in: 0.0001...0.9999)
            let u2 = Float.random(in: 0.0001...0.9999)
            return sqrt(-2.0 * log(u1)) * cos(2.0 * Float.pi * u2)
        }
        
        // Store for backward pass
        var mutableSelf = self
        mutableSelf.lastMean = mean
        mutableSelf.lastLogVar = logVar
        mutableSelf.lastEpsilon = epsilon
        
        // Reparameterization: z = mean + epsilon * exp(0.5 * log_var)
        let z = zip(zip(mean, logVar), epsilon).map { (mv, eps) in
            let (m, lv) = mv
            return m + eps * exp(0.5 * lv)
        }
        
        return z
    }
    
    public func backward(
        _ gradOutput: [Float],
        input: [Float],
        output: [Float]
    ) async -> (gradInput: [Float], gradients: LayerGradients) {
        guard gradOutput.count == latentDim else {
            fatalError("Gradient output size mismatch")
        }
        
        var gradInput = Array(repeating: Float(0), count: 2 * latentDim)
        
        // Gradient w.r.t mean (direct pass-through)
        for i in 0..<latentDim {
            gradInput[i] = gradOutput[i]
        }
        
        // Gradient w.r.t log_var
        for i in 0..<latentDim {
            let gradStd = gradOutput[i] * lastEpsilon[i]
            gradInput[latentDim + i] = 0.5 * exp(0.5 * lastLogVar[i]) * gradStd
        }
        
        return (gradInput, LayerGradients())
    }
    
    public mutating func updateParameters(gradients: LayerGradients, optimizer: Optimizer) async {
        // No parameters to update
    }
    
    public func getParameters() async -> LayerParameters {
        LayerParameters()
    }
    
    public func getParameterCount() async -> Int {
        0
    }
    
    public func getGradients() async -> LayerGradients {
        LayerGradients()
    }
}

/// Split layer that divides input into mean and log variance
public struct SplitLayer: NeuralLayer, Sendable {
    private let inputDim: Int
    private let latentDim: Int
    
    public init(inputDim: Int, latentDim: Int) {
        self.inputDim = inputDim
        self.latentDim = latentDim
    }
    
    public func forward(_ input: [Float]) async -> [Float] {
        // This is handled by the dense layer before reparameterization
        // Just pass through
        return input
    }
    
    public func backward(
        _ gradOutput: [Float],
        input: [Float],
        output: [Float]
    ) async -> (gradInput: [Float], gradients: LayerGradients) {
        // Pass gradients through
        return (gradOutput, LayerGradients())
    }
    
    public mutating func updateParameters(gradients: LayerGradients, optimizer: Optimizer) async {
        // No parameters to update
    }
    
    public func getParameters() async -> LayerParameters {
        LayerParameters()
    }
    
    public func getParameterCount() async -> Int {
        0
    }
    
    public func getGradients() async -> LayerGradients {
        LayerGradients()
    }
}

/// Sampling layer that combines mean and log variance layers for VAE
public actor VAESamplingLayer {
    private let meanLayer: DenseLayer
    private let logVarLayer: DenseLayer
    private let reparameterization: ReparameterizationLayer
    private let latentDim: Int
    
    public init(inputSize: Int, latentDim: Int, metalCompute: MetalCompute? = nil) {
        self.latentDim = latentDim
        self.meanLayer = DenseLayer(
            inputSize: inputSize,
            outputSize: latentDim,
            activation: .linear,
            metalCompute: metalCompute
        )
        self.logVarLayer = DenseLayer(
            inputSize: inputSize,
            outputSize: latentDim,
            activation: .linear,
            metalCompute: metalCompute
        )
        self.reparameterization = ReparameterizationLayer(latentDim: latentDim)
    }
    
    public func forward(_ input: [Float]) async -> (z: [Float], mean: [Float], logVar: [Float]) {
        let mean = await meanLayer.forward(input)
        let logVar = await logVarLayer.forward(input)
        
        // Combine mean and logVar for reparameterization
        let combined = mean + logVar
        let z = await reparameterization.forward(combined)
        
        return (z, mean, logVar)
    }
    
    public func getMeanLayer() -> DenseLayer {
        meanLayer
    }
    
    public func getLogVarLayer() -> DenseLayer {
        logVarLayer
    }
}
// VectorStoreKit: Neural Network Optimizers
//
// Advanced optimization algorithms for neural network training

import Foundation

/// Protocol for neural network optimizers
public protocol Optimizer: Sendable {
    /// Update a parameter given its gradient
    func update(parameter: Float, gradient: Float, name: String) async -> Float
    
    /// Reset optimizer state
    mutating func reset() async
    
    /// Get current learning rate
    func getCurrentLearningRate() async -> Float
    
    /// Set learning rate
    func setLearningRate(_ learningRate: Float) async
}

// MARK: - Optimizer Implementations

/// Stochastic Gradient Descent optimizer
public actor SGDOptimizer: Optimizer {
    public var learningRate: Float
    private let momentum: Float
    private let weightDecay: Float
    private let nesterov: Bool
    private var velocities: [String: Float] = [:]
    
    public init(
        learningRate: Float = 0.01,
        momentum: Float = 0.0,
        weightDecay: Float = 0.0,
        nesterov: Bool = false
    ) {
        self.learningRate = learningRate
        self.momentum = momentum
        self.weightDecay = weightDecay
        self.nesterov = nesterov
    }
    
    public func update(parameter: Float, gradient: Float, name: String) async -> Float {
        var grad = gradient
        
        // Apply weight decay (L2 regularization)
        if weightDecay > 0 {
            grad += weightDecay * parameter
        }
        
        if momentum > 0 {
            let velocity = velocities[name] ?? 0.0
            let newVelocity = momentum * velocity - learningRate * grad
            velocities[name] = newVelocity
            
            if nesterov {
                // Nesterov accelerated gradient
                return parameter + momentum * newVelocity - learningRate * grad
            } else {
                return parameter + newVelocity
            }
        } else {
            return parameter - learningRate * grad
        }
    }
    
    public func reset() async {
        velocities.removeAll()
    }
    
    public func getCurrentLearningRate() async -> Float {
        learningRate
    }
    
    public func setLearningRate(_ learningRate: Float) async {
        self.learningRate = learningRate
    }
}

/// Adam optimizer
public actor AdamOptimizer: Optimizer {
    public var learningRate: Float
    private let beta1: Float
    private let beta2: Float
    private let epsilon: Float
    private let weightDecay: Float
    private var firstMoments: [String: Float] = [:]
    private var secondMoments: [String: Float] = [:]
    private var timestep: Int = 0
    
    public init(
        learningRate: Float = 0.001,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        epsilon: Float = 1e-8,
        weightDecay: Float = 0.0
    ) {
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weightDecay = weightDecay
    }
    
    public func update(parameter: Float, gradient: Float, name: String) async -> Float {
        timestep += 1
        
        var grad = gradient
        
        // Apply weight decay
        if weightDecay > 0 {
            grad += weightDecay * parameter
        }
        
        // Update biased first moment estimate
        let m = firstMoments[name] ?? 0.0
        let newM = beta1 * m + (1 - beta1) * grad
        firstMoments[name] = newM
        
        // Update biased second raw moment estimate
        let v = secondMoments[name] ?? 0.0
        let newV = beta2 * v + (1 - beta2) * grad * grad
        secondMoments[name] = newV
        
        // Compute bias-corrected first moment estimate
        let mHat = newM / (1 - pow(beta1, Float(timestep)))
        
        // Compute bias-corrected second raw moment estimate
        let vHat = newV / (1 - pow(beta2, Float(timestep)))
        
        // Update parameters
        return parameter - learningRate * mHat / (sqrt(vHat) + epsilon)
    }
    
    public func reset() async {
        firstMoments.removeAll()
        secondMoments.removeAll()
        timestep = 0
    }
    
    public func getCurrentLearningRate() async -> Float {
        learningRate
    }
    
    public func setLearningRate(_ learningRate: Float) async {
        self.learningRate = learningRate
    }
}

/// RMSprop optimizer
public actor RMSpropOptimizer: Optimizer {
    public var learningRate: Float
    private let decay: Float
    private let epsilon: Float
    private let momentum: Float
    private let centered: Bool
    private var meanSquares: [String: Float] = [:]
    private var meanGrads: [String: Float] = [:]
    private var momentumBuffer: [String: Float] = [:]
    
    public init(
        learningRate: Float = 0.01,
        decay: Float = 0.99,
        epsilon: Float = 1e-8,
        momentum: Float = 0.0,
        centered: Bool = false
    ) {
        self.learningRate = learningRate
        self.decay = decay
        self.epsilon = epsilon
        self.momentum = momentum
        self.centered = centered
    }
    
    public func update(parameter: Float, gradient: Float, name: String) async -> Float {
        let grad = gradient
        
        // Update mean squared gradient
        let ms = meanSquares[name] ?? 0.0
        let newMs = decay * ms + (1 - decay) * grad * grad
        meanSquares[name] = newMs
        
        var denom = sqrt(newMs) + epsilon
        
        if centered {
            // Update mean gradient
            let mg = meanGrads[name] ?? 0.0
            let newMg = decay * mg + (1 - decay) * grad
            meanGrads[name] = newMg
            
            denom = sqrt(newMs - newMg * newMg) + epsilon
        }
        
        let update: Float
        if momentum > 0 {
            let buf = momentumBuffer[name] ?? 0.0
            let newBuf = momentum * buf + grad / denom
            momentumBuffer[name] = newBuf
            update = learningRate * newBuf
        } else {
            update = learningRate * grad / denom
        }
        
        return parameter - update
    }
    
    public func reset() async {
        meanSquares.removeAll()
        meanGrads.removeAll()
        momentumBuffer.removeAll()
    }
    
    public func getCurrentLearningRate() async -> Float {
        learningRate
    }
    
    public func setLearningRate(_ learningRate: Float) async {
        self.learningRate = learningRate
    }
}

/// AdaGrad optimizer
public actor AdaGradOptimizer: Optimizer {
    public var learningRate: Float
    private let epsilon: Float
    private let initialAccumulatorValue: Float
    private var accumulators: [String: Float] = [:]
    
    public init(
        learningRate: Float = 0.01,
        epsilon: Float = 1e-10,
        initialAccumulatorValue: Float = 0.0
    ) {
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.initialAccumulatorValue = initialAccumulatorValue
    }
    
    public func update(parameter: Float, gradient: Float, name: String) async -> Float {
        let grad = gradient
        
        // Update accumulator
        let acc = accumulators[name] ?? initialAccumulatorValue
        let newAcc = acc + grad * grad
        accumulators[name] = newAcc
        
        // Update parameter
        return parameter - learningRate * grad / (sqrt(newAcc) + epsilon)
    }
    
    public func reset() async {
        accumulators.removeAll()
    }
    
    public func getCurrentLearningRate() async -> Float {
        learningRate
    }
    
    public func setLearningRate(_ learningRate: Float) async {
        self.learningRate = learningRate
    }
}

// MARK: - Type Aliases for Convenience

public typealias SGD = SGDOptimizer
public typealias Adam = AdamOptimizer
public typealias RMSprop = RMSpropOptimizer
public typealias AdaGrad = AdaGradOptimizer

/// Learning rate scheduler
public protocol LearningRateScheduler: Sendable {
    /// Get learning rate for current step
    func getLearningRate(step: Int, currentLR: Float) -> Float
}

/// Step decay learning rate scheduler
public struct StepDecayScheduler: LearningRateScheduler {
    public let stepSize: Int
    public let gamma: Float
    
    public init(stepSize: Int = 10, gamma: Float = 0.1) {
        self.stepSize = stepSize
        self.gamma = gamma
    }
    
    public func getLearningRate(step: Int, currentLR: Float) -> Float {
        let epochs = step / stepSize
        return currentLR * pow(gamma, Float(epochs))
    }
}

/// Exponential decay learning rate scheduler
public struct ExponentialDecayScheduler: LearningRateScheduler {
    public let decayRate: Float
    
    public init(decayRate: Float = 0.96) {
        self.decayRate = decayRate
    }
    
    public func getLearningRate(step: Int, currentLR: Float) -> Float {
        return currentLR * pow(decayRate, Float(step))
    }
}

/// Cosine annealing learning rate scheduler
public struct CosineAnnealingScheduler: LearningRateScheduler {
    public let totalSteps: Int
    public let minLR: Float
    
    public init(totalSteps: Int, minLR: Float = 0.0) {
        self.totalSteps = totalSteps
        self.minLR = minLR
    }
    
    public func getLearningRate(step: Int, currentLR: Float) -> Float {
        let progress = Float(step) / Float(totalSteps)
        return minLR + (currentLR - minLR) * 0.5 * (1 + cos(Float.pi * progress))
    }
}

/// Warm-up learning rate scheduler
public struct WarmupScheduler: LearningRateScheduler {
    public let warmupSteps: Int
    public let targetLR: Float
    
    public init(warmupSteps: Int = 1000, targetLR: Float = 0.001) {
        self.warmupSteps = warmupSteps
        self.targetLR = targetLR
    }
    
    public func getLearningRate(step: Int, currentLR: Float) -> Float {
        if step < warmupSteps {
            return targetLR * Float(step) / Float(warmupSteps)
        }
        return targetLR
    }
}
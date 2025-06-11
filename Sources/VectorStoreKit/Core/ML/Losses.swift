// VectorStoreKit: Loss Functions
//
// Extended loss functions for neural networks including autoencoder-specific losses

import Foundation

/// Loss function utilities for neural networks
public struct LossFunctions {
    
    /// Compute loss for a given loss function type
    public static func compute(
        lossFunction: LossFunction,
        prediction: [Float],
        target: [Float]
    ) -> Float {
        guard prediction.count == target.count else {
            return Float.infinity
        }
        
        var loss: Float = 0
        
        switch lossFunction {
        case .mse:
            for i in 0..<prediction.count {
                let diff = prediction[i] - target[i]
                loss += diff * diff
            }
            loss /= Float(prediction.count)
            
        case .mae:
            for i in 0..<prediction.count {
                loss += abs(prediction[i] - target[i])
            }
            loss /= Float(prediction.count)
            
        case .crossEntropy:
            for i in 0..<prediction.count {
                loss += -target[i] * log(max(prediction[i], 1e-7))
            }
            
        case .binaryCrossEntropy:
            for i in 0..<prediction.count {
                loss += -target[i] * log(max(prediction[i], 1e-7)) - (1 - target[i]) * log(max(1 - prediction[i], 1e-7))
            }
            loss /= Float(prediction.count)
            
        case .huber:
            let delta: Float = 1.0
            for i in 0..<prediction.count {
                let diff = abs(prediction[i] - target[i])
                if diff <= delta {
                    loss += 0.5 * diff * diff
                } else {
                    loss += delta * (diff - 0.5 * delta)
                }
            }
            loss /= Float(prediction.count)
        }
        
        return loss
    }
    
    /// Compute gradient for a given loss function type
    public static func gradient(
        lossFunction: LossFunction,
        prediction: [Float],
        target: [Float]
    ) -> [Float] {
        guard prediction.count == target.count else {
            return []
        }
        
        var grad = [Float](repeating: 0, count: prediction.count)
        
        switch lossFunction {
        case .mse:
            for i in 0..<prediction.count {
                grad[i] = 2.0 * (prediction[i] - target[i]) / Float(prediction.count)
            }
            
        case .mae:
            for i in 0..<prediction.count {
                let diff = prediction[i] - target[i]
                grad[i] = diff > 0 ? 1.0 / Float(prediction.count) : -1.0 / Float(prediction.count)
            }
            
        case .crossEntropy:
            for i in 0..<prediction.count {
                grad[i] = -target[i] / max(prediction[i], 1e-7)
            }
            
        case .binaryCrossEntropy:
            for i in 0..<prediction.count {
                grad[i] = (-target[i] / max(prediction[i], 1e-7) + (1 - target[i]) / max(1 - prediction[i], 1e-7)) / Float(prediction.count)
            }
            
        case .huber:
            let delta: Float = 1.0
            for i in 0..<prediction.count {
                let diff = prediction[i] - target[i]
                if abs(diff) <= delta {
                    grad[i] = diff / Float(prediction.count)
                } else {
                    grad[i] = delta * (diff > 0 ? 1.0 : -1.0) / Float(prediction.count)
                }
            }
        }
        
        return grad
    }
    
    /// KL Divergence loss for VAE
    public static func klDivergence(mean: [Float], logVar: [Float]) -> Float {
        var klLoss: Float = 0
        for (mu, lv) in zip(mean, logVar) {
            klLoss += -0.5 * (1 + lv - mu * mu - exp(lv))
        }
        
        return klLoss / Float(mean.count)
    }
    
    /// Reconstruction loss for autoencoders
    public static func reconstructionLoss(original: [Float], reconstructed: [Float], useMAE: Bool = false) -> Float {
        if useMAE {
            return compute(lossFunction: .mae, prediction: reconstructed, target: original)
        } else {
            return compute(lossFunction: .mse, prediction: reconstructed, target: original)
        }
    }
    
    /// Sparsity loss for sparse autoencoders
    public static func sparsityLoss(activations: [Float], targetSparsity: Float = 0.05, weight: Float = 1.0) -> Float {
        let avgActivation = activations.reduce(0, +) / Float(activations.count)
        let klSparsity = targetSparsity * log(targetSparsity / max(avgActivation, 1e-7)) +
                         (1 - targetSparsity) * log((1 - targetSparsity) / max(1 - avgActivation, 1e-7))
        return weight * klSparsity
    }
    
    /// Contractive loss (approximation using Frobenius norm of Jacobian)
    public static func contractiveLoss(encodedGradients: [[Float]], weight: Float = 0.1) -> Float {
        var frobeniusNorm: Float = 0
        for row in encodedGradients {
            for value in row {
                frobeniusNorm += value * value
            }
        }
        return weight * sqrt(frobeniusNorm)
    }
}

/// VAE-specific loss combining reconstruction and KL divergence
public struct VAELoss {
    public let reconstructionWeight: Float
    public let klWeight: Float
    
    public init(reconstructionWeight: Float = 1.0, klWeight: Float = 1.0) {
        self.reconstructionWeight = reconstructionWeight
        self.klWeight = klWeight
    }
    
    public func compute(
        original: [Float],
        reconstructed: [Float],
        mean: [Float],
        logVar: [Float]
    ) -> Float {
        let reconLoss = LossFunctions.compute(
            lossFunction: .binaryCrossEntropy,
            prediction: reconstructed,
            target: original
        )
        let klLoss = LossFunctions.klDivergence(mean: mean, logVar: logVar)
        
        return reconstructionWeight * reconLoss + klWeight * klLoss
    }
    
    public func gradient(
        original: [Float],
        reconstructed: [Float],
        mean: [Float],
        logVar: [Float]
    ) -> (reconGrad: [Float], meanGrad: [Float], logVarGrad: [Float]) {
        // Reconstruction gradient
        let reconGrad = LossFunctions.gradient(
            lossFunction: .binaryCrossEntropy,
            prediction: reconstructed,
            target: original
        ).map { $0 * reconstructionWeight }
        
        // KL gradients
        let meanGrad = mean.map { $0 * klWeight / Float(mean.count) }
        let logVarGrad = logVar.map { lv in
            0.5 * klWeight * (exp(lv) - 1) / Float(logVar.count)
        }
        
        return (reconGrad, meanGrad, logVarGrad)
    }
}

/// Extension to LossFunction enum for convenience
public extension LossFunction {
    func compute(prediction: [Float], target: [Float]) -> Float {
        LossFunctions.compute(lossFunction: self, prediction: prediction, target: target)
    }
    
    func gradient(prediction: [Float], target: [Float]) -> [Float] {
        LossFunctions.gradient(lossFunction: self, prediction: prediction, target: target)
    }
}
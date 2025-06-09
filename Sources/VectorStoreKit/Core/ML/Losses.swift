// VectorStoreKit: Loss Functions
//
// Extended loss functions for neural networks including autoencoder-specific losses

import Foundation

/// Extended loss functions for neural networks
public extension LossFunction {
    
    /// KL Divergence loss for VAE
    static func klDivergence(mean: [Float], logVar: [Float]) -> Float {
        var klLoss: Float = 0
        for (mu, lv) in zip(mean, logVar) {
            klLoss += -0.5 * (1 + lv - mu * mu - exp(lv))
        }
        
        return klLoss / Float(mean.count)
    }
    
    /// Reconstruction loss for autoencoders (using binary cross entropy)
    static func reconstructionLoss(original: [Float], reconstructed: [Float], useMAE: Bool = false) -> Float {
        if useMAE {
            return LossFunction.mae.compute(prediction: reconstructed, target: original)
        } else {
            return LossFunction.mse.compute(prediction: reconstructed, target: original)
        }
    }
    
    /// Sparsity loss for sparse autoencoders
    static func sparsityLoss(activations: [Float], targetSparsity: Float = 0.05, weight: Float = 1.0) -> Float {
        let avgActivation = activations.reduce(0, +) / Float(activations.count)
        let klSparsity = targetSparsity * log(targetSparsity / max(avgActivation, 1e-7)) +
                         (1 - targetSparsity) * log((1 - targetSparsity) / max(1 - avgActivation, 1e-7))
        return weight * klSparsity
    }
    
    /// Contractive loss (approximation using Frobenius norm of Jacobian)
    static func contractiveLoss(encodedGradients: [[Float]], weight: Float = 0.1) -> Float {
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
        let reconLoss = LossFunction.binaryCrossEntropy.compute(
            prediction: reconstructed,
            target: original
        )
        let klLoss = LossFunction.klDivergence(mean: mean, logVar: logVar)
        
        return reconstructionWeight * reconLoss + klWeight * klLoss
    }
    
    public func gradient(
        original: [Float],
        reconstructed: [Float],
        mean: [Float],
        logVar: [Float]
    ) -> (reconGrad: [Float], meanGrad: [Float], logVarGrad: [Float]) {
        // Reconstruction gradient
        let reconGrad = LossFunction.binaryCrossEntropy.gradient(
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
// VectorStoreKit: Optimizer Wrapper
//
// Wrapper to adapt optimizers for batch parameter updates

import Foundation

/// Wrapper to adapt single-parameter optimizers for batch updates
public actor OptimizerWrapper {
    private let baseOptimizer: any Optimizer
    
    public init(baseOptimizer: any Optimizer) {
        self.baseOptimizer = baseOptimizer
    }
    
    /// Update a batch of parameters
    public func update(
        parameters: [Float],
        gradients: [Float],
        name: String
    ) async -> [Float] {
        guard parameters.count == gradients.count else {
            fatalError("Parameter and gradient count mismatch")
        }
        
        var updatedParams: [Float] = []
        
        for (i, (param, grad)) in zip(parameters, gradients).enumerated() {
            let updated = await baseOptimizer.update(
                parameter: param,
                gradient: grad,
                name: "\(name)_\(i)"
            )
            updatedParams.append(updated)
        }
        
        return updatedParams
    }
    
    /// Get current learning rate
    public func getCurrentLearningRate() async -> Float {
        await baseOptimizer.getCurrentLearningRate()
    }
}

/// Extension to create wrapped optimizers
public extension Optimizer {
    func wrapped() -> OptimizerWrapper {
        OptimizerWrapper(baseOptimizer: self)
    }
}
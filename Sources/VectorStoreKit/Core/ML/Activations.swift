// VectorStoreKit: ML Activation Functions
//
// Comprehensive activation functions for neural networks with derivatives

import Foundation
import Accelerate

/// Activation functions for neural networks
public enum Activation: String, Codable, Sendable, CaseIterable {
    case linear
    case relu
    case leakyRelu
    case sigmoid
    case tanh
    case softmax
    case elu
    case selu
    case gelu
    case swish
    
    /// Apply activation function to input
    public func apply(_ input: [Float]) -> [Float] {
        switch self {
        case .linear:
            return input
            
        case .relu:
            return vDSP.threshold(input, to: 0, with: .zeroFill)
            
        case .leakyRelu:
            return input.map { $0 > 0 ? $0 : 0.01 * $0 }
            
        case .sigmoid:
            return input.map { 1.0 / (1.0 + exp(-$0)) }
            
        case .tanh:
            return input.map { Foundation.tanh($0) }
            
        case .softmax:
            let maxVal = input.max() ?? 0
            let expValues = input.map { exp($0 - maxVal) }
            let sum = expValues.reduce(0, +)
            return expValues.map { $0 / sum }
            
        case .elu:
            let alpha: Float = 1.0
            return input.map { $0 > 0 ? $0 : alpha * (exp($0) - 1) }
            
        case .selu:
            let alpha: Float = 1.6732632423543772848170429916717
            let scale: Float = 1.0507009873554804934193349852946
            return input.map { scale * ($0 > 0 ? $0 : alpha * (exp($0) - 1)) }
            
        case .gelu:
            // Gaussian Error Linear Unit approximation
            return input.map { x in
                0.5 * x * (1 + Foundation.tanh(sqrt(2 / Float.pi) * (x + 0.044715 * pow(x, 3))))
            }
            
        case .swish:
            return zip(input, apply(input, activation: .sigmoid)).map { $0 * $1 }
        }
    }
    
    /// Compute derivative of activation function
    public func derivative(_ input: [Float], output: [Float]? = nil) -> [Float] {
        switch self {
        case .linear:
            return Array(repeating: 1.0, count: input.count)
            
        case .relu:
            return input.map { $0 > 0 ? 1.0 : 0.0 }
            
        case .leakyRelu:
            return input.map { $0 > 0 ? 1.0 : 0.01 }
            
        case .sigmoid:
            let sig = output ?? apply(input)
            return zip(sig, sig).map { $0 * (1 - $1) }
            
        case .tanh:
            let tanh_out = output ?? apply(input)
            return tanh_out.map { 1 - $0 * $0 }
            
        case .softmax:
            // For softmax, derivative is computed differently in loss function
            return Array(repeating: 1.0, count: input.count)
            
        case .elu:
            let alpha: Float = 1.0
            return input.map { $0 > 0 ? 1.0 : alpha * exp($0) }
            
        case .selu:
            let alpha: Float = 1.6732632423543772848170429916717
            let scale: Float = 1.0507009873554804934193349852946
            return input.map { $0 > 0 ? scale : scale * alpha * exp($0) }
            
        case .gelu:
            // GELU derivative approximation
            return input.map { x in
                let tanh_arg = sqrt(2 / Float.pi) * (x + 0.044715 * pow(x, 3))
                let tanh_val = Foundation.tanh(tanh_arg)
                let sech2 = 1 - tanh_val * tanh_val
                return 0.5 * (1 + tanh_val) + 0.5 * x * sech2 * sqrt(2 / Float.pi) * (1 + 0.134145 * x * x)
            }
            
        case .swish:
            let sig = apply(input, activation: .sigmoid)
            return zip(input, sig).map { x, s in s * (1 + x * (1 - s)) }
        }
    }
    
    /// Helper method for internal use
    private func apply(_ input: [Float], activation: Activation) -> [Float] {
        activation.apply(input)
    }
}

/// Activation function utilities
public struct ActivationUtils {
    
    /// Apply activation function with Metal acceleration if available
    public static func applyActivation(
        _ input: [Float],
        activation: Activation,
        metalCompute: MetalCompute? = nil
    ) async -> [Float] {
        // For now, use CPU implementation
        // Metal acceleration can be added later
        return activation.apply(input)
    }
    
    /// Batch apply activation to multiple inputs
    public static func batchApply(
        _ inputs: [[Float]],
        activation: Activation
    ) -> [[Float]] {
        inputs.map { activation.apply($0) }
    }
    
    /// Compute Jacobian matrix for activation function
    public static func jacobian(
        _ input: [Float],
        activation: Activation
    ) -> [[Float]] {
        let n = input.count
        var jacobian = Array(repeating: Array(repeating: Float(0), count: n), count: n)
        
        switch activation {
        case .softmax:
            let output = activation.apply(input)
            for i in 0..<n {
                for j in 0..<n {
                    if i == j {
                        jacobian[i][j] = output[i] * (1 - output[i])
                    } else {
                        jacobian[i][j] = -output[i] * output[j]
                    }
                }
            }
            
        default:
            // For element-wise activations, Jacobian is diagonal
            let derivatives = activation.derivative(input)
            for i in 0..<n {
                jacobian[i][i] = derivatives[i]
            }
        }
        
        return jacobian
    }
}
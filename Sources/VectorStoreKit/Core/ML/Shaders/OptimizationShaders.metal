// VectorStoreKit: Optimization Operations Metal Shaders
//
// Parameter update operations for various optimizers

#include <metal_stdlib>
using namespace metal;

// MARK: - SGD Update

/// Stochastic Gradient Descent parameter update
kernel void sgd_update(
    device float* parameters [[buffer(0)]],
    constant float* gradients [[buffer(1)]],
    constant float& learningRate [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    // SGD update rule: param = param - lr * gradient
    parameters[gid] -= learningRate * gradients[gid];
}

// MARK: - SGD with Momentum

/// SGD with momentum parameter update
kernel void sgd_momentum_update(
    device float* parameters [[buffer(0)]],
    constant float* gradients [[buffer(1)]],
    device float* velocity [[buffer(2)]],
    constant float& learningRate [[buffer(3)]],
    constant float& momentum [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    // Update velocity: v = momentum * v - lr * gradient
    float v = momentum * velocity[gid] - learningRate * gradients[gid];
    velocity[gid] = v;
    
    // Update parameters: param = param + v
    parameters[gid] += v;
}

// MARK: - Adam Optimizer

/// Adam optimizer parameter update
kernel void adam_update(
    device float* parameters [[buffer(0)]],
    constant float* gradients [[buffer(1)]],
    device float* m [[buffer(2)]],  // First moment estimate
    device float* v [[buffer(3)]],  // Second moment estimate
    constant float& learningRate [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& epsilon [[buffer(7)]],
    constant uint& timestep [[buffer(8)]],
    constant uint& count [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float grad = gradients[gid];
    
    // Update biased first moment estimate
    m[gid] = beta1 * m[gid] + (1.0f - beta1) * grad;
    
    // Update biased second raw moment estimate
    v[gid] = beta2 * v[gid] + (1.0f - beta2) * grad * grad;
    
    // Compute bias-corrected first moment estimate with numerical stability
    float beta1_pow = pow(beta1, float(timestep));
    if (isnan(beta1_pow) || isinf(beta1_pow)) beta1_pow = 0.0f;
    float bias_correction1 = 1.0f - beta1_pow;
    if (bias_correction1 < 1e-7f) bias_correction1 = 1e-7f;
    float mHat = m[gid] / bias_correction1;
    
    // Compute bias-corrected second raw moment estimate with numerical stability
    float beta2_pow = pow(beta2, float(timestep));
    if (isnan(beta2_pow) || isinf(beta2_pow)) beta2_pow = 0.0f;
    float bias_correction2 = 1.0f - beta2_pow;
    if (bias_correction2 < 1e-7f) bias_correction2 = 1e-7f;
    float vHat = v[gid] / bias_correction2;
    
    // Update parameters with numerical stability
    float denominator = sqrt(vHat) + epsilon;
    if (denominator < 1e-7f) denominator = 1e-7f;
    
    // Check for NaN/Inf in gradient update
    float update = learningRate * mHat / denominator;
    if (isnan(update) || isinf(update)) {
        update = 0.0f;
    }
    
    parameters[gid] -= update;
}

// MARK: - RMSprop Optimizer

/// RMSprop optimizer parameter update
kernel void rmsprop_update(
    device float* parameters [[buffer(0)]],
    constant float* gradients [[buffer(1)]],
    device float* squaredGradAvg [[buffer(2)]],
    constant float& learningRate [[buffer(3)]],
    constant float& decay [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float grad = gradients[gid];
    
    // Update squared gradient average
    squaredGradAvg[gid] = decay * squaredGradAvg[gid] + (1.0f - decay) * grad * grad;
    
    // Update parameters with numerical stability
    float denominator = sqrt(squaredGradAvg[gid]) + epsilon;
    if (denominator < 1e-7f) denominator = 1e-7f;
    
    // Check for NaN/Inf in gradient update
    float update = learningRate * grad / denominator;
    if (isnan(update) || isinf(update)) {
        update = 0.0f;
    }
    
    parameters[gid] -= update;
}

// MARK: - AdaGrad Optimizer

/// AdaGrad optimizer parameter update
kernel void adagrad_update(
    device float* parameters [[buffer(0)]],
    constant float* gradients [[buffer(1)]],
    device float* accumulatedGrad [[buffer(2)]],
    constant float& learningRate [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float grad = gradients[gid];
    
    // Accumulate squared gradients
    accumulatedGrad[gid] += grad * grad;
    
    // Update parameters with numerical stability
    float denominator = sqrt(accumulatedGrad[gid]) + epsilon;
    if (denominator < 1e-7f) denominator = 1e-7f;
    
    // Check for NaN/Inf in gradient update
    float update = learningRate * grad / denominator;
    if (isnan(update) || isinf(update)) {
        update = 0.0f;
    }
    
    parameters[gid] -= update;
}

// MARK: - Gradient Operations

/// Zero gradients
kernel void zero_gradients(
    device float* gradients [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    gradients[gid] = 0.0f;
}

/// Accumulate gradients (gradients += new_gradients)
kernel void accumulate_gradients(
    device float* gradients [[buffer(0)]],
    constant float* new_gradients [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    gradients[gid] += new_gradients[gid];
}

/// Scale gradients by a factor
kernel void scale_gradients(
    device float* gradients [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    gradients[gid] *= scale;
}

// MARK: - Gradient Clipping

/// Clip gradients by global norm
kernel void clip_gradients_by_norm(
    device float* gradients [[buffer(0)]],
    constant float& globalNorm [[buffer(1)]],
    constant float& maxNorm [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    if (globalNorm > maxNorm) {
        // Protect against division by zero
        float divisor = globalNorm;
        if (divisor < 1e-7f) divisor = 1e-7f;
        
        float scale = maxNorm / divisor;
        
        // Check for NaN/Inf
        if (isnan(scale) || isinf(scale)) {
            scale = 1.0f;
        }
        
        gradients[gid] *= scale;
    }
}

/// Clip gradients by value
kernel void clip_gradients_by_value(
    device float* gradients [[buffer(0)]],
    constant float& minValue [[buffer(1)]],
    constant float& maxValue [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    gradients[gid] = clamp(gradients[gid], minValue, maxValue);
}
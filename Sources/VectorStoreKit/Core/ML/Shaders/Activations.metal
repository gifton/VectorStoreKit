// VectorStoreKit: Activation Functions Metal Shaders
//
// GPU-accelerated activation functions for neural networks

#include <metal_stdlib>
using namespace metal;

// MARK: - Forward Pass Activations

/// ReLU activation: f(x) = max(0, x)
kernel void relu_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = max(0.0f, input[gid]);
}

/// Leaky ReLU activation: f(x) = max(alpha * x, x)
kernel void leaky_relu_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float x = input[gid];
    output[gid] = x > 0 ? x : alpha * x;
}

/// Sigmoid activation: f(x) = 1 / (1 + exp(-x))
kernel void sigmoid_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    // Clamp input to prevent exp overflow
    float x = clamp(input[gid], -88.0f, 88.0f);
    float exp_neg_x = exp(-x);
    
    // Handle numerical edge cases
    if (isnan(exp_neg_x) || isinf(exp_neg_x)) {
        output[gid] = x > 0 ? 1.0f : 0.0f;
    } else {
        output[gid] = 1.0f / (1.0f + exp_neg_x);
    }
}

/// Tanh activation: f(x) = tanh(x)
kernel void tanh_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = tanh(input[gid]);
}

/// ELU activation: f(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
kernel void elu_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& alpha [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float x = input[gid];
    
    if (x > 0) {
        output[gid] = x;
    } else {
        // Clamp to prevent exp underflow
        x = max(x, -88.0f);
        float exp_x = exp(x);
        
        // Handle numerical edge cases
        if (isnan(exp_x) || isinf(exp_x)) {
            output[gid] = -alpha;
        } else {
            output[gid] = alpha * (exp_x - 1.0f);
        }
    }
}

/// SELU activation
kernel void selu_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    constexpr float alpha = 1.6732632423543772848170429916717f;
    constexpr float scale = 1.0507009873554804934193349852946f;
    
    float x = input[gid];
    
    if (x > 0) {
        output[gid] = scale * x;
    } else {
        // Clamp to prevent exp underflow
        x = max(x, -88.0f);
        float exp_x = exp(x);
        
        // Handle numerical edge cases
        if (isnan(exp_x) || isinf(exp_x)) {
            output[gid] = -scale * alpha;
        } else {
            output[gid] = scale * alpha * (exp_x - 1.0f);
        }
    }
}

/// GELU activation: f(x) = x * Φ(x)
kernel void gelu_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float x = input[gid];
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    float x_cubed = x * x * x;
    float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    output[gid] = 0.5f * x * (1.0f + tanh(tanh_arg));
}

/// Linear activation: f(x) = x
kernel void linear_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = input[gid];
}

/// Mish activation: f(x) = x * tanh(ln(1 + exp(x)))
kernel void mish_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float x = input[gid];
    
    // Compute softplus = ln(1 + exp(x)) with numerical stability
    float softplus;
    if (x > 20.0f) {
        // For large x, ln(1 + exp(x)) ≈ x
        softplus = x;
    } else if (x < -20.0f) {
        // For very negative x, ln(1 + exp(x)) ≈ exp(x)
        softplus = exp(x);
    } else {
        // Standard computation
        softplus = log(1.0f + exp(x));
    }
    
    output[gid] = x * tanh(softplus);
}

/// Swish activation: f(x) = x * sigmoid(x)
kernel void swish_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float x = input[gid];
    
    // Clamp to prevent exp overflow
    float clamped_x = clamp(x, -88.0f, 88.0f);
    float exp_neg_x = exp(-clamped_x);
    
    // Handle numerical edge cases
    if (isnan(exp_neg_x) || isinf(exp_neg_x)) {
        output[gid] = x > 0 ? x : 0.0f;
    } else {
        output[gid] = x / (1.0f + exp_neg_x);
    }
}

/// Softmax activation (along last dimension)
kernel void softmax_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& num_classes [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch_size) return;
    
    uint offset = gid * num_classes;
    
    // Find max for numerical stability
    float max_val = input[offset];
    for (uint i = 1; i < num_classes; i++) {
        max_val = max(max_val, input[offset + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (uint i = 0; i < num_classes; i++) {
        // Clamp to prevent exp overflow
        float diff = clamp(input[offset + i] - max_val, -88.0f, 88.0f);
        float exp_val = exp(diff);
        
        // Handle NaN/Inf
        if (isnan(exp_val) || isinf(exp_val)) {
            exp_val = 0.0f;
        }
        
        output[offset + i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize with division protection
    if (sum < 1e-7f) sum = 1e-7f;
    float inv_sum = 1.0f / sum;
    
    for (uint i = 0; i < num_classes; i++) {
        output[offset + i] *= inv_sum;
    }
}

// MARK: - Backward Pass Derivatives

/// ReLU backward: f'(x) = 1 if x > 0, 0 otherwise
kernel void relu_backward(
    constant float* gradOutput [[buffer(0)]],
    constant float* input [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    gradInput[gid] = input[gid] > 0 ? gradOutput[gid] : 0.0f;
}

/// Leaky ReLU backward
kernel void leaky_relu_backward(
    constant float* gradOutput [[buffer(0)]],
    constant float* input [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant float& alpha [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    gradInput[gid] = input[gid] > 0 ? gradOutput[gid] : alpha * gradOutput[gid];
}

/// Sigmoid backward: f'(x) = f(x) * (1 - f(x))
kernel void sigmoid_backward(
    constant float* gradOutput [[buffer(0)]],
    constant float* output [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float o = output[gid];
    gradInput[gid] = gradOutput[gid] * o * (1.0f - o);
}

/// Tanh backward: f'(x) = 1 - f(x)^2
kernel void tanh_backward(
    constant float* gradOutput [[buffer(0)]],
    constant float* output [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float o = output[gid];
    gradInput[gid] = gradOutput[gid] * (1.0f - o * o);
}

/// ELU backward
kernel void elu_backward(
    constant float* gradOutput [[buffer(0)]],
    constant float* input [[buffer(1)]],
    constant float* output [[buffer(2)]],
    device float* gradInput [[buffer(3)]],
    constant float& alpha [[buffer(4)]],
    constant uint& size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float x = input[gid];
    gradInput[gid] = x > 0 ? gradOutput[gid] : gradOutput[gid] * (output[gid] + alpha);
}

/// GELU backward
kernel void gelu_backward(
    constant float* gradOutput [[buffer(0)]],
    constant float* input [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float x = input[gid];
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    
    // Compute CDF and PDF of standard normal
    float x_sq = x * x;
    float x_cubed = x * x_sq;
    float tanh_arg = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    float tanh_val = tanh(tanh_arg);
    float cdf = 0.5f * (1.0f + tanh_val);
    
    // Derivative of tanh
    float sech_sq = 1.0f - tanh_val * tanh_val;
    float pdf_approx = 0.5f * sqrt_2_over_pi * (1.0f + 0.134145f * x_sq) * sech_sq;
    
    gradInput[gid] = gradOutput[gid] * (cdf + x * pdf_approx);
}

/// Swish backward: f'(x) = f(x) + sigmoid(x) * (1 - f(x))
kernel void swish_backward(
    constant float* gradOutput [[buffer(0)]],
    constant float* input [[buffer(1)]],
    constant float* output [[buffer(2)]],
    device float* gradInput [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float x = input[gid];
    
    // Clamp to prevent exp overflow
    float clamped_x = clamp(x, -88.0f, 88.0f);
    float exp_neg_x = exp(-clamped_x);
    float sigmoid_x;
    
    // Handle numerical edge cases
    if (isnan(exp_neg_x) || isinf(exp_neg_x)) {
        sigmoid_x = x > 0 ? 1.0f : 0.0f;
    } else {
        sigmoid_x = 1.0f / (1.0f + exp_neg_x);
    }
    
    float swish_x = output[gid];
    gradInput[gid] = gradOutput[gid] * (swish_x + sigmoid_x * (1.0f - swish_x));
}

/// Softmax backward
kernel void softmax_backward(
    constant float* gradOutput [[buffer(0)]],
    constant float* output [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch_size) return;
    
    uint offset = gid * num_classes;
    
    // Compute sum of gradOutput * output
    float sum = 0.0f;
    for (uint i = 0; i < num_classes; i++) {
        sum += gradOutput[offset + i] * output[offset + i];
    }
    
    // Compute gradient
    for (uint i = 0; i < num_classes; i++) {
        gradInput[offset + i] = output[offset + i] * (gradOutput[offset + i] - sum);
    }
}

/// Linear backward: f'(x) = 1
kernel void linear_backward(
    constant float* gradOutput [[buffer(0)]],
    constant float* input [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    gradInput[gid] = gradOutput[gid];
}

/// SELU backward
kernel void selu_backward(
    constant float* gradOutput [[buffer(0)]],
    constant float* input [[buffer(1)]],
    constant float* output [[buffer(2)]],
    device float* gradInput [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    constexpr float alpha = 1.6732632423543772848170429916717f;
    constexpr float scale = 1.0507009873554804934193349852946f;
    
    float x = input[gid];
    if (x > 0) {
        gradInput[gid] = gradOutput[gid] * scale;
    } else {
        // For x <= 0: derivative is scale * alpha * exp(x)
        // Since output = scale * alpha * (exp(x) - 1), we have:
        // exp(x) = (output / (scale * alpha)) + 1
        float exp_x = (output[gid] / (scale * alpha)) + 1.0f;
        gradInput[gid] = gradOutput[gid] * scale * alpha * exp_x;
    }
}

/// Mish backward: f'(x) = tanh(softplus(x)) + x * sigmoid(x) * sech^2(softplus(x))
kernel void mish_backward(
    constant float* gradOutput [[buffer(0)]],
    constant float* input [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float x = input[gid];
    
    // Compute softplus = ln(1 + exp(x)) with numerical stability
    float softplus;
    float exp_x;
    if (x > 20.0f) {
        softplus = x;
        exp_x = exp(x);
    } else if (x < -20.0f) {
        softplus = exp(x);
        exp_x = exp(x);
    } else {
        exp_x = exp(x);
        softplus = log(1.0f + exp_x);
    }
    
    // Compute sigmoid(x) = 1 / (1 + exp(-x))
    float sigmoid_x;
    if (x > 0) {
        float exp_neg_x = exp(-x);
        sigmoid_x = 1.0f / (1.0f + exp_neg_x);
    } else {
        sigmoid_x = exp_x / (1.0f + exp_x);
    }
    
    // Compute tanh(softplus)
    float tanh_softplus = tanh(softplus);
    
    // Compute sech^2(softplus) = 1 - tanh^2(softplus)
    float sech2_softplus = 1.0f - tanh_softplus * tanh_softplus;
    
    // Mish derivative: tanh(softplus) + x * sigmoid(x) * sech^2(softplus)
    gradInput[gid] = gradOutput[gid] * (tanh_softplus + x * sigmoid_x * sech2_softplus);
}
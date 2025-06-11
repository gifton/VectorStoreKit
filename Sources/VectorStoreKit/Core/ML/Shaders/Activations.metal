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
    output[gid] = 1.0f / (1.0f + exp(-input[gid]));
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
    output[gid] = x > 0 ? x : alpha * (exp(x) - 1.0f);
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
    output[gid] = scale * (x > 0 ? x : alpha * (exp(x) - 1.0f));
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

/// Swish activation: f(x) = x * sigmoid(x)
kernel void swish_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float x = input[gid];
    output[gid] = x / (1.0f + exp(-x));
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
        float exp_val = exp(input[offset + i] - max_val);
        output[offset + i] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
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
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    float swish_x = output[gid];
    
    gradInput[gid] = gradOutput[gid] * (swish_x + sigmoid_x * (1.0f - swish_x));
}
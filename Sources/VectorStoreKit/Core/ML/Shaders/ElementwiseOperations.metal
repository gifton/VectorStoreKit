// VectorStoreKit: Element-wise Operations Metal Shaders
//
// Element-wise operations for neural networks

#include <metal_stdlib>
using namespace metal;

// MARK: - Basic Element-wise Operations

/// Element-wise addition
kernel void element_add(
    constant float* a [[buffer(0)]],
    constant float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = a[gid] + b[gid];
}

/// Element-wise multiplication
kernel void element_multiply(
    constant float* a [[buffer(0)]],
    constant float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = a[gid] * b[gid];
}

/// Compute 1 - x
kernel void one_minus(
    constant float* x [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = 1.0f - x[gid];
}

// MARK: - Slice Operations

/// Extract a slice from buffer
kernel void extract_slice(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& offset [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = input[offset + gid];
}

/// Copy to offset in buffer
kernel void copy_to_offset(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& offset [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[offset + gid] = input[gid];
}

/// Accumulate values at a specific offset in the output buffer
kernel void accumulate_at_offset(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& offset [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[offset + gid] += input[gid];
}

// MARK: - Dropout Operations

/// Generate random number using Xorshift algorithm
inline float xorshift_random(uint seed, uint id) {
    uint state = seed + id * 1664525u + 1013904223u;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return float(state) / float(0xFFFFFFFF);
}

/// Forward pass of dropout
kernel void dropout_forward(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* mask [[buffer(2)]],      // Store mask for backward pass
    constant float& dropoutRate [[buffer(3)]],
    constant uint& seed [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    // Generate random value for this element
    float rand = xorshift_random(seed, gid);
    
    // Apply dropout with inverted scaling
    float scale = 1.0f / (1.0f - dropoutRate);
    float maskValue = (rand > dropoutRate) ? scale : 0.0f;
    
    mask[gid] = maskValue;
    output[gid] = input[gid] * maskValue;
}

/// Backward pass of dropout
kernel void dropout_backward(
    constant float* gradOutput [[buffer(0)]],
    constant float* mask [[buffer(1)]],
    device float* gradInput [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    gradInput[gid] = gradOutput[gid] * mask[gid];
}

// MARK: - Basic Buffer Operations

/// Scale buffer by a scalar
kernel void scale_buffer(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = input[gid] * scale;
}
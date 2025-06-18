// VectorStoreKit: Element-wise Operations Metal Shaders
//
// Element-wise operations for neural networks

#include <metal_stdlib>
using namespace metal;

// MARK: - Basic Element-wise Operations

/// Element-wise addition with bounds checking
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

/// Element-wise addition with FP16
kernel void element_add_fp16(
    constant half* a [[buffer(0)]],
    constant half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    // Perform addition in FP32 for better accuracy
    float result = float(a[gid]) + float(b[gid]);
    output[gid] = half(result);
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

/// Element-wise multiplication with FP16
kernel void element_multiply_fp16(
    constant half* a [[buffer(0)]],
    constant half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    // Perform multiplication in FP32 to avoid underflow
    float result = float(a[gid]) * float(b[gid]);
    output[gid] = half(result);
}

/// Fused multiply-add operation (a * b + c)
kernel void fused_multiply_add(
    constant float* a [[buffer(0)]],
    constant float* b [[buffer(1)]],
    constant float* c [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = fma(a[gid], b[gid], c[gid]); // Hardware-accelerated FMA
}

/// Fused multiply-add with FP16
kernel void fused_multiply_add_fp16(
    constant half* a [[buffer(0)]],
    constant half* b [[buffer(1)]],
    constant half* c [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    // Use FP32 for intermediate computation
    float result = fma(float(a[gid]), float(b[gid]), float(c[gid]));
    output[gid] = half(result);
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
    float denominator = 1.0f - dropoutRate;
    if (denominator < 1e-7f) denominator = 1e-7f;
    
    float scale = 1.0f / denominator;
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

// MARK: - CompositionLayer Operations

/// Sum multiple buffers element-wise
kernel void sum_buffers(
    constant float* input1 [[buffer(0)]],
    constant float* input2 [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = input1[gid] + input2[gid];
}

/// Concatenate two buffers
kernel void concatenate_buffers(
    constant float* input1 [[buffer(0)]],
    constant float* input2 [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size1 [[buffer(3)]],
    constant uint& size2 [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint totalSize = size1 + size2;
    if (gid >= totalSize) return;
    
    if (gid < size1) {
        output[gid] = input1[gid];
    } else {
        output[gid] = input2[gid - size1];
    }
}

/// Split buffer into two parts
kernel void split_buffer(
    constant float* input [[buffer(0)]],
    device float* output1 [[buffer(1)]],
    device float* output2 [[buffer(2)]],
    constant uint& splitPoint [[buffer(3)]],
    constant uint& totalSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= totalSize) return;
    
    if (gid < splitPoint) {
        output1[gid] = input[gid];
    } else {
        output2[gid - splitPoint] = input[gid];
    }
}

/// Element-wise maximum of two buffers
kernel void max_buffers(
    constant float* input1 [[buffer(0)]],
    constant float* input2 [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = max(input1[gid], input2[gid]);
}

/// Add multiple buffers with accumulation
kernel void add_buffers(
    constant float* inputs [[buffer(0)]],    // Flattened array of input buffers
    device float* output [[buffer(1)]],
    constant uint& bufferSize [[buffer(2)]],
    constant uint& numBuffers [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= bufferSize) return;
    
    float sum = 0.0f;
    for (uint i = 0; i < numBuffers; i++) {
        sum += inputs[i * bufferSize + gid];
    }
    output[gid] = sum;
}

// MARK: - Mathematical Operations

/// Element-wise square operation
kernel void element_square(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float val = input[gid];
    output[gid] = val * val;
}

/// Element-wise absolute value
kernel void element_abs(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = abs(input[gid]);
}

/// Element-wise square root
kernel void element_sqrt(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    // Protect against negative values
    float val = input[gid];
    if (val < 0.0f) val = 0.0f;
    
    output[gid] = sqrt(val);
}

// MARK: - Reduction Operations

/// Parallel reduction to compute sum
/// Each threadgroup reduces a portion of the input and writes one output value
kernel void reduce_sum(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& inputSize [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]]
) {
    // Shared memory for reduction
    threadgroup float sharedData[256];
    
    // Calculate global thread ID
    uint globalId = tgid * tgSize + tid;
    
    // Load data into shared memory
    float value = 0.0f;
    if (globalId < inputSize) {
        value = input[globalId];
    }
    sharedData[tid] = value;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction in shared memory
    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride && globalId + stride < inputSize) {
            sharedData[tid] += sharedData[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result for this threadgroup
    if (tid == 0) {
        output[tgid] = sharedData[0];
    }
}

/// Parallel reduction to compute maximum
kernel void reduce_max(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& inputSize [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float sharedData[256];
    
    uint globalId = tgid * tgSize + tid;
    
    float value = -INFINITY;
    if (globalId < inputSize) {
        value = input[globalId];
    }
    sharedData[tid] = value;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride && globalId + stride < inputSize) {
            sharedData[tid] = max(sharedData[tid], sharedData[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[tgid] = sharedData[0];
    }
}

/// Parallel reduction to compute minimum
kernel void reduce_min(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& inputSize [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float sharedData[256];
    
    uint globalId = tgid * tgSize + tid;
    
    float value = INFINITY;
    if (globalId < inputSize) {
        value = input[globalId];
    }
    sharedData[tid] = value;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride && globalId + stride < inputSize) {
            sharedData[tid] = min(sharedData[tid], sharedData[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        output[tgid] = sharedData[0];
    }
}
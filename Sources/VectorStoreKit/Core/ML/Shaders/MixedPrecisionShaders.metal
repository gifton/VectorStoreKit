// VectorStoreKit: Mixed Precision Metal Shaders
//
// Shaders for FP16/FP32 conversion and mixed precision operations

#include <metal_stdlib>
using namespace metal;

// MARK: - Precision Conversion

/// Convert FP32 buffer to FP16
kernel void convert_fp32_to_fp16(
    constant float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    // Direct conversion with potential precision loss
    output[gid] = half(input[gid]);
}

/// Convert FP16 buffer to FP32
kernel void convert_fp16_to_fp32(
    constant half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    // Lossless conversion from FP16 to FP32
    output[gid] = float(input[gid]);
}

// MARK: - Numeric Stability Checking

/// Check for overflow and underflow in a buffer
kernel void check_numeric_stability(
    constant float* input [[buffer(0)]],
    device atomic_uint* flags [[buffer(1)]], // [overflow_flag, underflow_flag]
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float val = input[gid];
    
    // Check for infinity (overflow)
    if (isinf(val)) {
        atomic_store_explicit(&flags[0], 1, memory_order_relaxed);
    }
    
    // Check for NaN
    if (isnan(val)) {
        atomic_store_explicit(&flags[0], 1, memory_order_relaxed);
    }
    
    // Check for values that would overflow FP16 (>65504)
    if (abs(val) > 65504.0f) {
        atomic_store_explicit(&flags[0], 1, memory_order_relaxed);
    }
    
    // Check for underflow (very small but non-zero values)
    if (val != 0.0f && abs(val) < 1e-7f) {
        atomic_store_explicit(&flags[1], 1, memory_order_relaxed);
    }
}

/// Check for overflow and underflow in FP16 buffer
kernel void check_numeric_stability_fp16(
    constant half* input [[buffer(0)]],
    device atomic_uint* flags [[buffer(1)]], // [overflow_flag, underflow_flag]
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    half val = input[gid];
    
    // Check for infinity (overflow)
    if (isinf(val)) {
        atomic_store_explicit(&flags[0], 1, memory_order_relaxed);
    }
    
    // Check for NaN
    if (isnan(val)) {
        atomic_store_explicit(&flags[0], 1, memory_order_relaxed);
    }
    
    // FP16 max value is already bounded at 65504, so any finite value is valid
    
    // Check for underflow (FP16 has smaller range)
    if (val != half(0.0f) && abs(float(val)) < 6e-5f) { // ~2^-14
        atomic_store_explicit(&flags[1], 1, memory_order_relaxed);
    }
}

// MARK: - Loss Scaling Operations

/// Check if gradients would overflow when scaled
kernel void check_gradient_overflow(
    constant float* gradients [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    device atomic_uint* overflow_flag [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float scaled = gradients[gid] * scale;
    
    // Check if scaled value would overflow FP16
    if (abs(scaled) > 65504.0f || isinf(scaled) || isnan(scaled)) {
        atomic_store_explicit(overflow_flag, 1, memory_order_relaxed);
    }
}

/// Scale gradients by loss scale factor
kernel void scale_gradients(
    device float* gradients [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    gradients[gid] *= scale;
}

/// Unscale gradients by loss scale factor
kernel void unscale_gradients(
    device float* gradients [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    gradients[gid] /= scale;
}

/// Scale FP16 gradients by loss scale factor
kernel void scale_gradients_fp16(
    device half* gradients [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    gradients[gid] = half(float(gradients[gid]) * scale);
}

/// Unscale FP16 gradients by loss scale factor
kernel void unscale_gradients_fp16(
    device half* gradients [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    gradients[gid] = half(float(gradients[gid]) / scale);
}

// MARK: - Mixed Precision Matrix Operations

/// Matrix multiplication with mixed precision (FP16 compute, FP32 accumulate)
kernel void matmul_mixed_precision(
    constant half* a [[buffer(0)]],      // [M x K]
    constant half* b [[buffer(1)]],      // [K x N]
    device float* c [[buffer(2)]],       // [M x N]
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= N || gid.y >= M) return;
    
    // Accumulate in FP32 for better precision
    float sum = 0.0f;
    
    for (uint k = 0; k < K; k++) {
        // Convert to FP32 before multiplication
        float aVal = float(a[gid.y * K + k]);
        float bVal = float(b[k * N + gid.x]);
        sum += aVal * bVal;
    }
    
    c[gid.y * N + gid.x] = sum;
}

/// Fused linear layer with mixed precision (matmul + bias + activation)
kernel void linear_mixed_precision(
    constant half* input [[buffer(0)]],    // [batch_size x input_size]
    constant half* weights [[buffer(1)]],  // [output_size x input_size]
    constant float* bias [[buffer(2)]],    // [output_size] - kept in FP32
    device half* output [[buffer(3)]],     // [batch_size x output_size]
    constant uint& batch_size [[buffer(4)]],
    constant uint& input_size [[buffer(5)]],
    constant uint& output_size [[buffer(6)]],
    constant uint& activation_type [[buffer(7)]], // 0=linear, 1=relu, 2=gelu
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch = gid.y;
    uint out_idx = gid.x;
    
    if (batch >= batch_size || out_idx >= output_size) return;
    
    // Accumulate in FP32
    float sum = 0.0f;
    
    // Matrix multiplication
    for (uint i = 0; i < input_size; i++) {
        float inputVal = float(input[batch * input_size + i]);
        float weightVal = float(weights[out_idx * input_size + i]);
        sum += inputVal * weightVal;
    }
    
    // Add bias (FP32)
    sum += bias[out_idx];
    
    // Apply activation
    switch (activation_type) {
        case 1: // ReLU
            sum = max(sum, 0.0f);
            break;
        case 2: // GELU approximation
            sum = 0.5f * sum * (1.0f + tanh(0.7978845608f * (sum + 0.044715f * sum * sum * sum)));
            break;
        // case 0 is linear (no activation)
    }
    
    // Convert back to FP16
    output[batch * output_size + out_idx] = half(sum);
}

// MARK: - Batch Normalization with Mixed Precision

/// Batch normalization forward pass with mixed precision
kernel void batchnorm_mixed_precision(
    constant half* input [[buffer(0)]],        // [batch x channels]
    constant float* running_mean [[buffer(1)]], // [channels] - FP32 for stability
    constant float* running_var [[buffer(2)]],  // [channels] - FP32 for stability
    constant half* gamma [[buffer(3)]],        // [channels]
    constant half* beta [[buffer(4)]],         // [channels]
    device half* output [[buffer(5)]],         // [batch x channels]
    constant float& epsilon [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    constant uint& channels [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch = gid.y;
    uint channel = gid.x;
    
    if (batch >= batch_size || channel >= channels) return;
    
    uint idx = batch * channels + channel;
    
    // Normalize using FP32 statistics
    float x = float(input[idx]);
    float mean = running_mean[channel];
    float var = running_var[channel];
    
    float norm = (x - mean) / sqrt(var + epsilon);
    
    // Scale and shift
    float scaled = norm * float(gamma[channel]) + float(beta[channel]);
    
    // Convert back to FP16
    output[idx] = half(scaled);
}

// MARK: - Optimizer Updates with Mixed Precision

/// SGD update with mixed precision (FP32 master weights, FP16 model weights)
kernel void sgd_update_mixed_precision(
    device float* master_weights [[buffer(0)]],  // FP32 master copy
    device half* model_weights [[buffer(1)]],    // FP16 model weights
    constant float* gradients [[buffer(2)]],     // FP32 gradients
    constant float& learning_rate [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    // Update master weights in FP32
    master_weights[gid] -= learning_rate * gradients[gid];
    
    // Copy to model weights in FP16
    model_weights[gid] = half(master_weights[gid]);
}

/// Adam optimizer update with mixed precision
kernel void adam_update_mixed_precision(
    device float* master_weights [[buffer(0)]],  // FP32 master copy
    device half* model_weights [[buffer(1)]],    // FP16 model weights
    constant float* gradients [[buffer(2)]],     // FP32 gradients
    device float* m [[buffer(3)]],               // First moment (FP32)
    device float* v [[buffer(4)]],               // Second moment (FP32)
    constant float& learning_rate [[buffer(5)]],
    constant float& beta1 [[buffer(6)]],
    constant float& beta2 [[buffer(7)]],
    constant float& epsilon [[buffer(8)]],
    constant uint& timestep [[buffer(9)]],
    constant uint& count [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float grad = gradients[gid];
    
    // Update biased first moment estimate
    m[gid] = beta1 * m[gid] + (1.0f - beta1) * grad;
    
    // Update biased second moment estimate
    v[gid] = beta2 * v[gid] + (1.0f - beta2) * grad * grad;
    
    // Bias correction
    float m_corrected = m[gid] / (1.0f - pow(beta1, float(timestep)));
    float v_corrected = v[gid] / (1.0f - pow(beta2, float(timestep)));
    
    // Update master weights in FP32
    master_weights[gid] -= learning_rate * m_corrected / (sqrt(v_corrected) + epsilon);
    
    // Copy to model weights in FP16
    model_weights[gid] = half(master_weights[gid]);
}

// MARK: - Reduction Operations with Mixed Precision

/// Reduce sum with mixed precision (FP16 input, FP32 accumulation)
kernel void reduce_sum_mixed_precision(
    constant half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& inputSize [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]]
) {
    // Shared memory for reduction (FP32 for accuracy)
    threadgroup float sharedData[256];
    
    uint globalId = tgid * tgSize + tid;
    
    // Load data and convert to FP32
    float value = 0.0f;
    if (globalId < inputSize) {
        value = float(input[globalId]);
    }
    sharedData[tid] = value;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction in FP32
    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride && globalId + stride < inputSize) {
            sharedData[tid] += sharedData[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    if (tid == 0) {
        output[tgid] = sharedData[0];
    }
}

// MARK: - Gradient Clipping Operations

/// Clip gradients by global norm
kernel void clip_gradients_by_norm(
    device float* gradients [[buffer(0)]],
    constant float& maxNorm [[buffer(1)]],
    constant float& globalNorm [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    if (globalNorm > maxNorm) {
        float scale = maxNorm / globalNorm;
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

/// Compute L2 norm of gradients
kernel void compute_gradient_norm(
    constant float* gradients [[buffer(0)]],
    device float* partialNorms [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float sharedData[256];
    
    uint globalId = tgid * tgSize + tid;
    
    // Compute squared values
    float value = 0.0f;
    if (globalId < count) {
        float grad = gradients[globalId];
        value = grad * grad;
    }
    sharedData[tid] = value;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride && globalId + stride < count) {
            sharedData[tid] += sharedData[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write partial norm
    if (tid == 0) {
        partialNorms[tgid] = sharedData[0];
    }
}

// MARK: - Validation and Debugging Operations

/// Validate buffer for NaN and Inf values
kernel void validate_buffer(
    constant float* buffer [[buffer(0)]],
    device atomic_uint* errorFlags [[buffer(1)]], // [nan_count, inf_count, zero_count]
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float value = buffer[gid];
    
    if (isnan(value)) {
        atomic_fetch_add_explicit(&errorFlags[0], 1, memory_order_relaxed);
    } else if (isinf(value)) {
        atomic_fetch_add_explicit(&errorFlags[1], 1, memory_order_relaxed);
    } else if (value == 0.0f) {
        atomic_fetch_add_explicit(&errorFlags[2], 1, memory_order_relaxed);
    }
}

/// Validate FP16 buffer for NaN and Inf values
kernel void validate_buffer_fp16(
    constant half* buffer [[buffer(0)]],
    device atomic_uint* errorFlags [[buffer(1)]], // [nan_count, inf_count, zero_count]
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    half value = buffer[gid];
    
    if (isnan(value)) {
        atomic_fetch_add_explicit(&errorFlags[0], 1, memory_order_relaxed);
    } else if (isinf(value)) {
        atomic_fetch_add_explicit(&errorFlags[1], 1, memory_order_relaxed);
    } else if (value == half(0.0f)) {
        atomic_fetch_add_explicit(&errorFlags[2], 1, memory_order_relaxed);
    }
}

/// Replace NaN/Inf values with safe defaults
kernel void sanitize_buffer(
    device float* buffer [[buffer(0)]],
    constant float& nanReplacement [[buffer(1)]],
    constant float& infReplacement [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    device atomic_uint* replacementCount [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    
    float value = buffer[gid];
    
    if (isnan(value)) {
        buffer[gid] = nanReplacement;
        atomic_fetch_add_explicit(replacementCount, 1, memory_order_relaxed);
    } else if (isinf(value)) {
        buffer[gid] = sign(value) * infReplacement;
        atomic_fetch_add_explicit(replacementCount, 1, memory_order_relaxed);
    }
}

/// Compute buffer statistics for debugging
kernel void compute_buffer_stats(
    constant float* buffer [[buffer(0)]],
    device float* stats [[buffer(1)]], // [min, max, mean, variance]
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tgSize [[threads_per_threadgroup]]
) {
    threadgroup float minVals[256];
    threadgroup float maxVals[256];
    threadgroup float sumVals[256];
    threadgroup float sumSqVals[256];
    
    uint globalId = tgid * tgSize + tid;
    
    // Initialize with extreme values
    float localMin = FLT_MAX;
    float localMax = -FLT_MAX;
    float localSum = 0.0f;
    float localSumSq = 0.0f;
    
    // Process elements
    if (globalId < count) {
        float value = buffer[globalId];
        localMin = min(localMin, value);
        localMax = max(localMax, value);
        localSum = value;
        localSumSq = value * value;
    }
    
    minVals[tid] = localMin;
    maxVals[tid] = localMax;
    sumVals[tid] = localSum;
    sumSqVals[tid] = localSumSq;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride && globalId + stride < count) {
            minVals[tid] = min(minVals[tid], minVals[tid + stride]);
            maxVals[tid] = max(maxVals[tid], maxVals[tid + stride]);
            sumVals[tid] += sumVals[tid + stride];
            sumSqVals[tid] += sumSqVals[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write results for final reduction on CPU
    if (tid == 0) {
        uint offset = tgid * 4;
        stats[offset + 0] = minVals[0];
        stats[offset + 1] = maxVals[0];
        stats[offset + 2] = sumVals[0];
        stats[offset + 3] = sumSqVals[0];
    }
}
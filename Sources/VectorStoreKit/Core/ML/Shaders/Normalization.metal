// VectorStoreKit: Normalization Metal Shaders
//
// Batch normalization and layer normalization operations

#include <metal_stdlib>
using namespace metal;

// MARK: - Batch Normalization

/// Compute mean and variance for batch normalization
kernel void batch_norm_compute_stats(
    constant float* input [[buffer(0)]],
    device float* mean [[buffer(1)]],
    device float* variance [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& num_features [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_features) return;
    
    // Compute mean
    float sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        sum += input[b * num_features + gid];
    }
    float feature_mean = sum / float(batch_size);
    mean[gid] = feature_mean;
    
    // Compute variance
    float var_sum = 0.0f;
    for (uint b = 0; b < batch_size; b++) {
        float diff = input[b * num_features + gid] - feature_mean;
        var_sum += diff * diff;
    }
    variance[gid] = var_sum / float(batch_size);
}

/// Batch normalization forward pass
kernel void batch_norm_forward(
    constant float* input [[buffer(0)]],
    constant float* mean [[buffer(1)]],
    constant float* variance [[buffer(2)]],
    constant float* gamma [[buffer(3)]],
    constant float* beta [[buffer(4)]],
    device float* output [[buffer(5)]],
    device float* normalized [[buffer(6)]],
    constant float& epsilon [[buffer(7)]],
    constant uint& batch_size [[buffer(8)]],
    constant uint& num_features [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch = gid.x;
    uint feature = gid.y;
    
    if (batch >= batch_size || feature >= num_features) return;
    
    uint idx = batch * num_features + feature;
    
    // Normalize
    float norm = (input[idx] - mean[feature]) / sqrt(variance[feature] + epsilon);
    normalized[idx] = norm;
    
    // Scale and shift
    output[idx] = gamma[feature] * norm + beta[feature];
}

/// Update running statistics
kernel void batch_norm_update_running_stats(
    constant float* batch_mean [[buffer(0)]],
    constant float* batch_var [[buffer(1)]],
    device float* running_mean [[buffer(2)]],
    device float* running_var [[buffer(3)]],
    constant float& momentum [[buffer(4)]],
    constant uint& num_features [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= num_features) return;
    
    running_mean[gid] = momentum * running_mean[gid] + (1.0f - momentum) * batch_mean[gid];
    running_var[gid] = momentum * running_var[gid] + (1.0f - momentum) * batch_var[gid];
}

/// Batch normalization backward pass - compute gradients
kernel void batch_norm_backward(
    constant float* grad_output [[buffer(0)]],
    constant float* normalized [[buffer(1)]],
    constant float* gamma [[buffer(2)]],
    constant float* variance [[buffer(3)]],
    device float* grad_input [[buffer(4)]],
    device float* grad_gamma [[buffer(5)]],
    device float* grad_beta [[buffer(6)]],
    device atomic_float* grad_mean [[buffer(7)]],
    device atomic_float* grad_var [[buffer(8)]],
    constant float& epsilon [[buffer(9)]],
    constant uint& batch_size [[buffer(10)]],
    constant uint& num_features [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch = gid.x;
    uint feature = gid.y;
    
    if (batch >= batch_size || feature >= num_features) return;
    
    uint idx = batch * num_features + feature;
    float inv_batch_size = 1.0f / float(batch_size);
    
    // Compute per-sample gradients
    float grad_out = grad_output[idx];
    float norm = normalized[idx];
    
    // Accumulate gradients for gamma and beta
    if (batch == 0) {
        grad_gamma[feature] = 0.0f;
        grad_beta[feature] = 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    atomic_fetch_add_explicit(&grad_gamma[feature], grad_out * norm, memory_order_relaxed);
    atomic_fetch_add_explicit(&grad_beta[feature], grad_out, memory_order_relaxed);
    
    // Compute gradients w.r.t normalized input
    float grad_norm = grad_out * gamma[feature];
    
    // Compute gradients w.r.t variance
    float std_inv = 1.0f / sqrt(variance[feature] + epsilon);
    float grad_var_sample = grad_norm * norm * -0.5f * std_inv * std_inv * std_inv;
    
    atomic_fetch_add_explicit(&grad_var[feature], grad_var_sample, memory_order_relaxed);
    
    // Compute gradients w.r.t mean
    float grad_mean_sample = grad_norm * -std_inv;
    atomic_fetch_add_explicit(&grad_mean[feature], grad_mean_sample, memory_order_relaxed);
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Compute final gradient w.r.t input
    if (batch == 0) {
        float final_grad_mean = grad_mean[feature] * inv_batch_size;
        float final_grad_var = grad_var[feature] * 2.0f * inv_batch_size;
        
        for (uint b = 0; b < batch_size; b++) {
            uint b_idx = b * num_features + feature;
            float x_norm = normalized[b_idx];
            grad_input[b_idx] = grad_norm * std_inv + final_grad_var * x_norm + final_grad_mean;
        }
    }
}

// MARK: - Layer Normalization

/// Layer normalization forward pass
kernel void layer_norm_forward(
    constant float* input [[buffer(0)]],
    constant float* gamma [[buffer(1)]],
    constant float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* mean [[buffer(4)]],
    device float* variance [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    constant uint& batch_size [[buffer(7)]],
    constant uint& num_features [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch_size) return;
    
    uint offset = gid * num_features;
    
    // Compute mean
    float sum = 0.0f;
    for (uint i = 0; i < num_features; i++) {
        sum += input[offset + i];
    }
    float sample_mean = sum / float(num_features);
    mean[gid] = sample_mean;
    
    // Compute variance
    float var_sum = 0.0f;
    for (uint i = 0; i < num_features; i++) {
        float diff = input[offset + i] - sample_mean;
        var_sum += diff * diff;
    }
    float sample_var = var_sum / float(num_features);
    variance[gid] = sample_var;
    
    // Normalize and scale
    float std_inv = 1.0f / sqrt(sample_var + epsilon);
    for (uint i = 0; i < num_features; i++) {
        float normalized = (input[offset + i] - sample_mean) * std_inv;
        output[offset + i] = gamma[i] * normalized + beta[i];
    }
}

/// Layer normalization backward pass
kernel void layer_norm_backward(
    constant float* grad_output [[buffer(0)]],
    constant float* input [[buffer(1)]],
    constant float* mean [[buffer(2)]],
    constant float* variance [[buffer(3)]],
    constant float* gamma [[buffer(4)]],
    device float* grad_input [[buffer(5)]],
    device float* grad_gamma [[buffer(6)]],
    device float* grad_beta [[buffer(7)]],
    constant float& epsilon [[buffer(8)]],
    constant uint& batch_size [[buffer(9)]],
    constant uint& num_features [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch_size) return;
    
    uint offset = gid * num_features;
    float sample_mean = mean[gid];
    float sample_var = variance[gid];
    float std_inv = 1.0f / sqrt(sample_var + epsilon);
    
    // Compute gradients for gamma and beta (accumulate across batch)
    if (gid == 0) {
        for (uint i = 0; i < num_features; i++) {
            grad_gamma[i] = 0.0f;
            grad_beta[i] = 0.0f;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    for (uint i = 0; i < num_features; i++) {
        float normalized = (input[offset + i] - sample_mean) * std_inv;
        atomic_fetch_add_explicit(&grad_gamma[i], grad_output[offset + i] * normalized, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_beta[i], grad_output[offset + i], memory_order_relaxed);
    }
    
    // Compute gradient w.r.t input
    float grad_mean = 0.0f;
    float grad_var = 0.0f;
    
    for (uint i = 0; i < num_features; i++) {
        float grad_norm = grad_output[offset + i] * gamma[i];
        float normalized = (input[offset + i] - sample_mean) * std_inv;
        
        grad_mean += grad_norm * -std_inv;
        grad_var += grad_norm * normalized * -0.5f * std_inv * std_inv * std_inv;
    }
    
    grad_mean /= float(num_features);
    grad_var *= 2.0f / float(num_features);
    
    for (uint i = 0; i < num_features; i++) {
        float x_centered = input[offset + i] - sample_mean;
        float grad_norm = grad_output[offset + i] * gamma[i];
        grad_input[offset + i] = grad_norm * std_inv + grad_var * x_centered + grad_mean;
    }
}

// MARK: - Dropout

/// Generate dropout mask
kernel void dropout_generate_mask(
    device float* mask [[buffer(0)]],
    constant float& rate [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& seed [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    // Simple linear congruential generator for reproducibility
    uint state = gid * 73856093u ^ seed;
    state = state * 1103515245u + 12345u;
    float random = float(state & 0x7fffffffu) / float(0x7fffffff);
    
    mask[gid] = random > rate ? scale : 0.0f;
}

/// Apply dropout mask
kernel void dropout_forward(
    constant float* input [[buffer(0)]],
    constant float* mask [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = input[gid] * mask[gid];
}

/// Dropout backward pass
kernel void dropout_backward(
    constant float* grad_output [[buffer(0)]],
    constant float* mask [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    grad_input[gid] = grad_output[gid] * mask[gid];
}
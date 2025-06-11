// VectorStoreKit: Loss Operations Metal Shaders
//
// GPU-accelerated loss computations for neural networks

#include <metal_stdlib>
using namespace metal;

// MARK: - Mean Squared Error

/// Compute MSE loss
kernel void mse_loss(
    constant float* prediction [[buffer(0)]],
    constant float* target [[buffer(1)]],
    device float* loss [[buffer(2)]],
    device atomic_float* total_loss [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float diff = prediction[gid] - target[gid];
    float squared_diff = diff * diff;
    loss[gid] = squared_diff;
    
    atomic_fetch_add_explicit(total_loss, squared_diff / float(size), memory_order_relaxed);
}

/// Compute MSE gradient
kernel void mse_gradient(
    constant float* prediction [[buffer(0)]],
    constant float* target [[buffer(1)]],
    device float* gradient [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    gradient[gid] = 2.0f * (prediction[gid] - target[gid]) / float(size);
}

// MARK: - Mean Absolute Error

/// Compute MAE loss
kernel void mae_loss(
    constant float* prediction [[buffer(0)]],
    constant float* target [[buffer(1)]],
    device float* loss [[buffer(2)]],
    device atomic_float* total_loss [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float abs_diff = abs(prediction[gid] - target[gid]);
    loss[gid] = abs_diff;
    
    atomic_fetch_add_explicit(total_loss, abs_diff / float(size), memory_order_relaxed);
}

/// Compute MAE gradient
kernel void mae_gradient(
    constant float* prediction [[buffer(0)]],
    constant float* target [[buffer(1)]],
    device float* gradient [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float diff = prediction[gid] - target[gid];
    gradient[gid] = diff > 0 ? 1.0f / float(size) : -1.0f / float(size);
}

// MARK: - Cross Entropy

/// Compute cross entropy loss (with softmax)
kernel void cross_entropy_loss(
    constant float* logits [[buffer(0)]],
    constant float* target [[buffer(1)]],
    device float* loss [[buffer(2)]],
    device atomic_float* total_loss [[buffer(3)]],
    constant uint& batch_size [[buffer(4)]],
    constant uint& num_classes [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch_size) return;
    
    uint offset = gid * num_classes;
    
    // Find max for numerical stability
    float max_logit = logits[offset];
    for (uint i = 1; i < num_classes; i++) {
        max_logit = max(max_logit, logits[offset + i]);
    }
    
    // Compute log-sum-exp
    float sum_exp = 0.0f;
    for (uint i = 0; i < num_classes; i++) {
        sum_exp += exp(logits[offset + i] - max_logit);
    }
    float log_sum_exp = log(sum_exp) + max_logit;
    
    // Compute loss for this sample
    float sample_loss = 0.0f;
    for (uint i = 0; i < num_classes; i++) {
        sample_loss -= target[offset + i] * (logits[offset + i] - log_sum_exp);
    }
    
    loss[gid] = sample_loss;
    atomic_fetch_add_explicit(total_loss, sample_loss / float(batch_size), memory_order_relaxed);
}

/// Compute cross entropy gradient (softmax - target)
kernel void cross_entropy_gradient(
    constant float* logits [[buffer(0)]],
    constant float* target [[buffer(1)]],
    device float* gradient [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint batch = gid.x;
    uint cls = gid.y;
    
    if (batch >= batch_size || cls >= num_classes) return;
    
    uint offset = batch * num_classes;
    
    // Compute softmax
    float max_logit = logits[offset];
    for (uint i = 1; i < num_classes; i++) {
        max_logit = max(max_logit, logits[offset + i]);
    }
    
    float sum_exp = 0.0f;
    for (uint i = 0; i < num_classes; i++) {
        sum_exp += exp(logits[offset + i] - max_logit);
    }
    
    float softmax_val = exp(logits[offset + cls] - max_logit) / sum_exp;
    
    // Gradient is softmax - target
    gradient[offset + cls] = (softmax_val - target[offset + cls]) / float(batch_size);
}

// MARK: - Binary Cross Entropy

/// Compute binary cross entropy loss
kernel void binary_cross_entropy_loss(
    constant float* prediction [[buffer(0)]],
    constant float* target [[buffer(1)]],
    device float* loss [[buffer(2)]],
    device atomic_float* total_loss [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float pred = prediction[gid];
    float targ = target[gid];
    
    // Clamp prediction to avoid log(0)
    pred = clamp(pred, 1e-7f, 1.0f - 1e-7f);
    
    float sample_loss = -targ * log(pred) - (1.0f - targ) * log(1.0f - pred);
    loss[gid] = sample_loss;
    
    atomic_fetch_add_explicit(total_loss, sample_loss / float(size), memory_order_relaxed);
}

/// Compute binary cross entropy gradient
kernel void binary_cross_entropy_gradient(
    constant float* prediction [[buffer(0)]],
    constant float* target [[buffer(1)]],
    device float* gradient [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float pred = prediction[gid];
    float targ = target[gid];
    
    // Clamp to avoid division by zero
    pred = clamp(pred, 1e-7f, 1.0f - 1e-7f);
    
    gradient[gid] = (pred - targ) / (pred * (1.0f - pred) * float(size));
}

// MARK: - Huber Loss

/// Compute Huber loss
kernel void huber_loss(
    constant float* prediction [[buffer(0)]],
    constant float* target [[buffer(1)]],
    device float* loss [[buffer(2)]],
    device atomic_float* total_loss [[buffer(3)]],
    constant float& delta [[buffer(4)]],
    constant uint& size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float diff = abs(prediction[gid] - target[gid]);
    float sample_loss;
    
    if (diff <= delta) {
        sample_loss = 0.5f * diff * diff;
    } else {
        sample_loss = delta * (diff - 0.5f * delta);
    }
    
    loss[gid] = sample_loss;
    atomic_fetch_add_explicit(total_loss, sample_loss / float(size), memory_order_relaxed);
}

/// Compute Huber loss gradient
kernel void huber_gradient(
    constant float* prediction [[buffer(0)]],
    constant float* target [[buffer(1)]],
    device float* gradient [[buffer(2)]],
    constant float& delta [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    
    float diff = prediction[gid] - target[gid];
    float abs_diff = abs(diff);
    
    if (abs_diff <= delta) {
        gradient[gid] = diff / float(size);
    } else {
        gradient[gid] = delta * (diff > 0 ? 1.0f : -1.0f) / float(size);
    }
}

// MARK: - Reduction Operations

/// Reduce sum
kernel void reduce_sum(
    constant float* input [[buffer(0)]],
    device atomic_float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    atomic_fetch_add_explicit(output, input[gid], memory_order_relaxed);
}

/// Reduce mean
kernel void reduce_mean(
    constant float* input [[buffer(0)]],
    device atomic_float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    atomic_fetch_add_explicit(output, input[gid] / float(size), memory_order_relaxed);
}
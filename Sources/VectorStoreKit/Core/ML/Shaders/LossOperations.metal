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
    
    // Protect against division by zero
    float divisor = float(size);
    if (divisor < 1e-7f) divisor = 1e-7f;
    atomic_fetch_add_explicit(total_loss, squared_diff / divisor, memory_order_relaxed);
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
    
    // Protect against division by zero
    float divisor = float(size);
    if (divisor < 1e-7f) divisor = 1e-7f;
    
    gradient[gid] = 2.0f * (prediction[gid] - target[gid]) / divisor;
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
    
    // Protect against division by zero
    float divisor = float(size);
    if (divisor < 1e-7f) divisor = 1e-7f;
    atomic_fetch_add_explicit(total_loss, abs_diff / divisor, memory_order_relaxed);
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
    
    // Protect against division by zero
    float divisor = float(size);
    if (divisor < 1e-7f) divisor = 1e-7f;
    
    // Handle near-zero differences
    if (abs(diff) < 1e-7f) {
        gradient[gid] = 0.0f;
    } else {
        gradient[gid] = diff > 0 ? 1.0f / divisor : -1.0f / divisor;
    }
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
        // Clamp to prevent exp overflow
        float diff = clamp(logits[offset + i] - max_logit, -88.0f, 88.0f);
        float exp_val = exp(diff);
        
        // Handle NaN/Inf
        if (isnan(exp_val) || isinf(exp_val)) {
            exp_val = 0.0f;
        }
        
        sum_exp += exp_val;
    }
    
    // Protect against log(0)
    if (sum_exp < 1e-7f) sum_exp = 1e-7f;
    float log_sum_exp = log(sum_exp) + max_logit;
    
    // Compute loss for this sample
    float sample_loss = 0.0f;
    for (uint i = 0; i < num_classes; i++) {
        sample_loss -= target[offset + i] * (logits[offset + i] - log_sum_exp);
    }
    
    loss[gid] = sample_loss;
    
    // Protect against division by zero
    float divisor = float(batch_size);
    if (divisor < 1e-7f) divisor = 1e-7f;
    atomic_fetch_add_explicit(total_loss, sample_loss / divisor, memory_order_relaxed);
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
        // Clamp to prevent exp overflow
        float diff = clamp(logits[offset + i] - max_logit, -88.0f, 88.0f);
        float exp_val = exp(diff);
        
        // Handle NaN/Inf
        if (isnan(exp_val) || isinf(exp_val)) {
            exp_val = 0.0f;
        }
        
        sum_exp += exp_val;
    }
    
    // Protect against division by zero
    if (sum_exp < 1e-7f) sum_exp = 1e-7f;
    
    // Compute softmax with clamping
    float cls_diff = clamp(logits[offset + cls] - max_logit, -88.0f, 88.0f);
    float exp_cls = exp(cls_diff);
    if (isnan(exp_cls) || isinf(exp_cls)) exp_cls = 0.0f;
    
    float softmax_val = exp_cls / sum_exp;
    
    // Protect batch_size division
    float batch_divisor = float(batch_size);
    if (batch_divisor < 1e-7f) batch_divisor = 1e-7f;
    
    // Gradient is softmax - target
    gradient[offset + cls] = (softmax_val - target[offset + cls]) / batch_divisor;
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
    
    // Check for NaN/Inf in loss
    if (isnan(sample_loss) || isinf(sample_loss)) {
        sample_loss = 0.0f;
    }
    
    loss[gid] = sample_loss;
    
    // Protect against division by zero
    float divisor = float(size);
    if (divisor < 1e-7f) divisor = 1e-7f;
    atomic_fetch_add_explicit(total_loss, sample_loss / divisor, memory_order_relaxed);
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
    
    // Compute denominator with protection
    float denominator = pred * (1.0f - pred) * float(size);
    if (denominator < 1e-7f) denominator = 1e-7f;
    
    gradient[gid] = (pred - targ) / denominator;
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
    
    // Protect against division by zero
    float divisor = float(size);
    if (divisor < 1e-7f) divisor = 1e-7f;
    atomic_fetch_add_explicit(total_loss, sample_loss / divisor, memory_order_relaxed);
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
    
    // Protect against division by zero
    float divisor = float(size);
    if (divisor < 1e-7f) divisor = 1e-7f;
    
    if (abs_diff <= delta) {
        gradient[gid] = diff / divisor;
    } else {
        gradient[gid] = delta * (diff > 0 ? 1.0f : -1.0f) / divisor;
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
    
    // Protect against division by zero
    float divisor = float(size);
    if (divisor < 1e-7f) divisor = 1e-7f;
    
    atomic_fetch_add_explicit(output, input[gid] / divisor, memory_order_relaxed);
}
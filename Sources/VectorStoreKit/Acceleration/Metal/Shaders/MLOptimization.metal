#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// MARK: - Constants and Types

struct AdamConstants {
    float learningRate;
    float beta1;
    float beta2;
    float epsilon;
    float biasCorrection1;
    float biasCorrection2;
};

// MARK: - VAE Reparameterization

kernel void vae_reparameterization(
    device const float* mean [[buffer(0)]],
    device const float* logVar [[buffer(1)]],
    device const float* epsilon [[buffer(2)]],
    device float* z [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    // Reparameterization trick: z = mean + epsilon * exp(0.5 * log_var)
    float std = exp(0.5f * logVar[tid]);
    z[tid] = mean[tid] + epsilon[tid] * std;
}

// MARK: - Loss Functions

kernel void mse_loss(
    device const float* predictions [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device atomic_float* loss [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        float diff = predictions[tid] - targets[tid];
        float squared_diff = diff * diff;
        
        // Atomic add for thread-safe accumulation
        atomic_fetch_add_explicit(loss, squared_diff, memory_order_relaxed);
    }
}

kernel void mse_loss_reduce(
    device atomic_float* loss [[buffer(0)]],
    constant uint& count [[buffer(1)]]
) {
    // Final reduction - divide by count
    float current = atomic_load_explicit(loss, memory_order_relaxed);
    atomic_store_explicit(loss, current / float(count), memory_order_relaxed);
}

kernel void cross_entropy_loss(
    device const float* predictions [[buffer(0)]],
    device const float* targets [[buffer(1)]],
    device atomic_float* loss [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        // Numerical stability: clamp predictions
        float pred = clamp(predictions[tid], 1e-7f, 1.0f - 1e-7f);
        float target = targets[tid];
        
        float loss_value = -target * log(pred) - (1.0f - target) * log(1.0f - pred);
        atomic_fetch_add_explicit(loss, loss_value, memory_order_relaxed);
    }
}

kernel void contrastive_loss(
    device const float* embeddings1 [[buffer(0)]],
    device const float* embeddings2 [[buffer(1)]],
    device const float* labels [[buffer(2)]],     // 1 for similar, 0 for dissimilar
    device atomic_float* loss [[buffer(3)]],
    constant uint& embedding_size [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    constant float& margin [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < batch_size) {
        uint offset = tid * embedding_size;
        
        // Compute squared Euclidean distance
        float dist_sq = 0.0f;
        for (uint i = 0; i < embedding_size; ++i) {
            float diff = embeddings1[offset + i] - embeddings2[offset + i];
            dist_sq += diff * diff;
        }
        
        float label = labels[tid];
        float loss_value = label * dist_sq + 
                          (1.0f - label) * max(0.0f, margin - sqrt(dist_sq));
        
        atomic_fetch_add_explicit(loss, loss_value, memory_order_relaxed);
    }
}

// MARK: - Adam Optimizer

kernel void adam_update(
    device float* parameter [[buffer(0)]],
    device const float* gradient [[buffer(1)]],
    device float* firstMoment [[buffer(2)]],
    device float* secondMoment [[buffer(3)]],
    constant AdamConstants& constants [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // Update biased first moment estimate
    firstMoment[tid] = constants.beta1 * firstMoment[tid] + 
                       (1.0f - constants.beta1) * gradient[tid];
    
    // Update biased second moment estimate  
    secondMoment[tid] = constants.beta2 * secondMoment[tid] + 
                        (1.0f - constants.beta2) * gradient[tid] * gradient[tid];
    
    // Compute bias-corrected moments
    float m_hat = firstMoment[tid] / constants.biasCorrection1;
    float v_hat = secondMoment[tid] / constants.biasCorrection2;
    
    // Update parameters
    parameter[tid] -= constants.learningRate * m_hat / (sqrt(v_hat) + constants.epsilon);
}

// MARK: - Triplet Loss Gradients

kernel void triplet_loss_gradients(
    device const float* anchor [[buffer(0)]],
    device const float* positive [[buffer(1)]],
    device const float* negative [[buffer(2)]],
    device float* anchorGrad [[buffer(3)]],
    device float* positiveGrad [[buffer(4)]],
    device float* negativeGrad [[buffer(5)]],
    constant uint& dimension [[buffer(6)]],
    constant float& scale [[buffer(7)]],      // 2.0 / dimension for normalized gradients
    uint tid [[thread_position_in_grid]]
) {
    if (tid < dimension) {
        // Gradient computation for triplet loss
        // dL/da = 2(a - p) - 2(a - n) = 2(n - p)
        // dL/dp = 2(p - a)
        // dL/dn = 2(a - n)
        
        float a = anchor[tid];
        float p = positive[tid];
        float n = negative[tid];
        
        anchorGrad[tid] = scale * (n - p);
        positiveGrad[tid] = scale * (p - a);
        negativeGrad[tid] = scale * (a - n);
    }
}

// MARK: - Random Number Generation Support

kernel void uniform_random_init(
    device float* buffer [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    constant uint& seed [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        // Simple linear congruential generator
        uint state = seed + tid * 1664525u;
        state = state * 1664525u + 1013904223u;
        state = state * 1664525u + 1013904223u;
        
        // Convert to float in [0, 1)
        buffer[tid] = float(state) / float(UINT_MAX);
    }
}

kernel void normal_random_transform(
    device float* uniform1 [[buffer(0)]],
    device float* uniform2 [[buffer(1)]],
    device float* normal [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid < count) {
        // Box-Muller transform
        float u1 = max(uniform1[tid], 1e-7f);  // Avoid log(0)
        float u2 = uniform2[tid];
        
        float radius = sqrt(-2.0f * log(u1));
        float theta = 2.0f * M_PI_F * u2;
        
        normal[tid] = radius * cos(theta);
    }
}
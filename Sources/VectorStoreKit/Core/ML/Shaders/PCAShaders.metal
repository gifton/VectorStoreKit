// VectorStoreKit: PCA and SVD Metal Shaders
//
// High-performance PCA operations using Metal
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Covariance Matrix Computation

/// Compute covariance matrix from centered data
/// Input: data [n x d] (centered), Output: cov [d x d]
kernel void compute_covariance(
    constant float* data [[buffer(0)]],    // [n x d] centered data
    device float* covariance [[buffer(1)]], // [d x d] output
    constant uint& n [[buffer(2)]],        // number of samples
    constant uint& d [[buffer(3)]],        // dimensions
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y; // row
    uint j = gid.x; // col
    
    if (i >= d || j >= d) return;
    
    float sum = 0.0f;
    
    // Compute dot product of columns i and j
    for (uint k = 0; k < n; k++) {
        sum += data[k * d + i] * data[k * d + j];
    }
    
    // Normalize by n-1 for unbiased estimator
    covariance[i * d + j] = sum / float(n - 1);
}

/// Center data by subtracting mean
kernel void center_data(
    device float* data [[buffer(0)]],      // [n x d] data to center in-place
    constant float* mean [[buffer(1)]],    // [d] mean vector
    constant uint& n [[buffer(2)]],        // number of samples
    constant uint& d [[buffer(3)]],        // dimensions
    uint2 gid [[thread_position_in_grid]]
) {
    uint sample = gid.y;
    uint dim = gid.x;
    
    if (sample >= n || dim >= d) return;
    
    uint idx = sample * d + dim;
    data[idx] -= mean[dim];
}

/// Compute mean of data
kernel void compute_mean(
    constant float* data [[buffer(0)]],    // [n x d] input data
    device float* mean [[buffer(1)]],      // [d] output mean
    constant uint& n [[buffer(2)]],        // number of samples
    constant uint& d [[buffer(3)]],        // dimensions
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= d) return;
    
    float sum = 0.0f;
    for (uint i = 0; i < n; i++) {
        sum += data[i * d + gid];
    }
    
    mean[gid] = sum / float(n);
}

// MARK: - SVD Components for PCA

/// Power iteration method for dominant eigenvector
/// One iteration of v = A^T * A * v / ||A^T * A * v||
kernel void power_iteration_step(
    constant float* covariance [[buffer(0)]],  // [d x d] covariance matrix
    constant float* v_in [[buffer(1)]],        // [d] current eigenvector estimate
    device float* v_out [[buffer(2)]],         // [d] updated eigenvector
    device float* eigenvalue [[buffer(3)]],    // scalar eigenvalue estimate
    constant uint& d [[buffer(4)]],            // dimensions
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= d) return;
    
    // Compute A * v
    float sum = 0.0f;
    for (uint j = 0; j < d; j++) {
        sum += covariance[gid * d + j] * v_in[j];
    }
    v_out[gid] = sum;
    
    // Will need a separate kernel to normalize
}

/// Normalize vector and compute magnitude
kernel void normalize_vector(
    device float* vector [[buffer(0)]],        // [d] vector to normalize in-place
    device float* magnitude [[buffer(1)]],     // scalar output magnitude
    constant uint& d [[buffer(2)]],            // dimensions
    uint gid [[thread_position_in_grid]]
) {
    // First, compute magnitude with parallel reduction
    threadgroup float partial_sums[256];
    uint tid = gid;
    
    float local_sum = 0.0f;
    for (uint i = tid; i < d; i += 256) {
        float val = vector[i];
        local_sum += val * val;
    }
    
    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        float mag = sqrt(partial_sums[0]);
        *magnitude = mag;
        
        // Normalize vector
        float inv_mag = 1.0f / mag;
        for (uint i = 0; i < d; i++) {
            vector[i] *= inv_mag;
        }
    }
}

/// Project data onto principal components
kernel void project_data(
    constant float* centered_data [[buffer(0)]],   // [n x d] centered data
    constant float* components [[buffer(1)]],      // [k x d] principal components (row-major)
    device float* projected [[buffer(2)]],         // [n x k] projected data
    constant uint& n [[buffer(3)]],                // number of samples
    constant uint& d [[buffer(4)]],                // original dimensions
    constant uint& k [[buffer(5)]],                // reduced dimensions
    uint2 gid [[thread_position_in_grid]]
) {
    uint sample = gid.y;
    uint comp = gid.x;
    
    if (sample >= n || comp >= k) return;
    
    float sum = 0.0f;
    for (uint i = 0; i < d; i++) {
        sum += centered_data[sample * d + i] * components[comp * d + i];
    }
    
    projected[sample * k + comp] = sum;
}

/// Gram-Schmidt orthogonalization for eigenvectors
kernel void gram_schmidt_step(
    device float* vectors [[buffer(0)]],       // [k x d] vectors to orthogonalize
    constant uint& k [[buffer(1)]],            // number of vectors
    constant uint& d [[buffer(2)]],            // dimensions
    constant uint& current [[buffer(3)]],      // current vector index to orthogonalize
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= d) return;
    
    // For vector[current], subtract projections onto all previous vectors
    float v_current = vectors[current * d + gid];
    
    for (uint i = 0; i < current; i++) {
        // Compute dot product <v_current, v_i>
        float dot = 0.0f;
        for (uint j = 0; j < d; j++) {
            dot += vectors[current * d + j] * vectors[i * d + j];
        }
        
        // Subtract projection
        v_current -= dot * vectors[i * d + gid];
    }
    
    vectors[current * d + gid] = v_current;
}

/// Extract top k eigenvectors using deflation
kernel void deflate_matrix(
    device float* matrix [[buffer(0)]],        // [d x d] matrix to deflate in-place
    constant float* eigenvector [[buffer(1)]], // [d] eigenvector to remove
    constant float& eigenvalue [[buffer(2)]],  // corresponding eigenvalue
    constant uint& d [[buffer(3)]],            // dimensions
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    
    if (i >= d || j >= d) return;
    
    // A' = A - λ * v * v^T
    matrix[i * d + j] -= eigenvalue * eigenvector[i] * eigenvector[j];
}

/// Compute explained variance ratio
kernel void compute_variance_ratio(
    constant float* eigenvalues [[buffer(0)]],     // [k] top k eigenvalues
    device float* ratios [[buffer(1)]],            // [k] explained variance ratios
    constant float& total_variance [[buffer(2)]],  // sum of all eigenvalues
    constant uint& k [[buffer(3)]],                // number of components
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= k) return;
    
    ratios[gid] = eigenvalues[gid] / total_variance;
}
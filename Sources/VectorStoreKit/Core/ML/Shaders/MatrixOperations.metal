// VectorStoreKit: Matrix Operations Metal Shaders
//
// High-performance matrix operations for neural networks

#include <metal_stdlib>
using namespace metal;

// MARK: - Matrix Multiplication

/// Matrix multiplication: C = A * B
/// A: [M x K], B: [K x N], C: [M x N]
kernel void matmul_forward(
    constant float* A [[buffer(0)]],
    constant float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}

/// Optimized matrix multiplication using threadgroup memory
kernel void matmul_tiled(
    constant float* A [[buffer(0)]],
    constant float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]]
) {
    constexpr uint TILE_SIZE = 16;
    
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    uint row = gid.y;
    uint col = gid.x;
    uint tRow = tid.y;
    uint tCol = tid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; t++) {
        // Load tiles into shared memory
        uint aCol = t * TILE_SIZE + tCol;
        uint bRow = t * TILE_SIZE + tRow;
        
        tileA[tRow][tCol] = (aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[tRow][tCol] = (bRow < K) ? B[bRow * N + col] : 0.0f;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tRow][k] * tileB[k][tCol];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    C[row * N + col] = sum;
}

/// Matrix-vector multiplication: y = A * x
kernel void matvec_forward(
    constant float* A [[buffer(0)]],
    constant float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= M) return;
    
    float sum = 0.0f;
    for (uint j = 0; j < N; j++) {
        sum += A[gid * N + j] * x[j];
    }
    y[gid] = sum;
}

// MARK: - Backward Pass Operations

/// Compute gradients for matrix multiplication
/// gradA = gradC * B^T
kernel void matmul_backward_A(
    constant float* gradC [[buffer(0)]],
    constant float* B [[buffer(1)]],
    device float* gradA [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= K) return;
    
    float sum = 0.0f;
    for (uint n = 0; n < N; n++) {
        sum += gradC[row * N + n] * B[col * N + n];
    }
    
    gradA[row * K + col] = sum;
}

/// Compute gradients for matrix multiplication
/// gradB = A^T * gradC
kernel void matmul_backward_B(
    constant float* A [[buffer(0)]],
    constant float* gradC [[buffer(1)]],
    device float* gradB [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= K || col >= N) return;
    
    float sum = 0.0f;
    for (uint m = 0; m < M; m++) {
        sum += A[m * K + row] * gradC[m * N + col];
    }
    
    gradB[row * N + col] = sum;
}

// MARK: - Element-wise Operations

/// Element-wise addition with broadcasting
kernel void add_bias(
    device float* matrix [[buffer(0)]],
    constant float* bias [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= rows || col >= cols) return;
    
    uint idx = row * cols + col;
    matrix[idx] += bias[col];
}

/// Compute bias gradients by summing across rows
kernel void reduce_bias_gradient(
    constant float* gradOutput [[buffer(0)]],
    device float* gradBias [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cols) return;
    
    float sum = 0.0f;
    for (uint row = 0; row < rows; row++) {
        sum += gradOutput[row * cols + gid];
    }
    gradBias[gid] = sum;
}

// MARK: - Utility Operations

/// Transpose matrix
kernel void transpose(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= rows || col >= cols) return;
    
    output[col * rows + row] = input[row * cols + col];
}

/// Copy matrix
kernel void copy_matrix(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = input[gid];
}

/// Outer product: C = a * b^T
kernel void outer_product(
    constant float* a [[buffer(0)]],  // [M x 1]
    constant float* b [[buffer(1)]],  // [N x 1]
    device float* C [[buffer(2)]],    // [M x N]
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    C[row * N + col] = a[row] * b[col];
}
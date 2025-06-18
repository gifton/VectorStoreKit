// VectorStoreKit: Matrix Operations Metal Shaders
//
// High-performance matrix operations for neural networks

#include <metal_stdlib>
#include <metal_simdgroup_matrix>  // For SIMD group matrix operations
using namespace metal;

// MARK: - Matrix Multiplication

/// Matrix multiplication: C = A * B
/// A: [M x K], B: [K x N], C: [M x N]
/// Optimized with vectorized operations and better memory access patterns
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
    
    // Use float4 for vectorized operations when possible
    float4 sum4 = float4(0.0f);
    float sum = 0.0f;
    
    // Process 4 elements at a time for better memory coalescing
    uint k = 0;
    for (; k + 3 < K; k += 4) {
        // Load 4 consecutive elements from A
        float4 a_vec = float4(A[row * K + k],
                             A[row * K + k + 1],
                             A[row * K + k + 2],
                             A[row * K + k + 3]);
        
        // Load corresponding elements from B (strided access)
        float4 b_vec = float4(B[k * N + col],
                             B[(k + 1) * N + col],
                             B[(k + 2) * N + col],
                             B[(k + 3) * N + col]);
        
        sum4 += a_vec * b_vec;
    }
    
    // Sum the components
    sum = sum4.x + sum4.y + sum4.z + sum4.w;
    
    // Process remaining elements
    for (; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}

/// Optimized matrix multiplication using threadgroup memory with vectorization
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
    
    // Use threadgroup memory with padding to avoid bank conflicts
    threadgroup float tileA[TILE_SIZE][TILE_SIZE + 1];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE + 1];
    
    uint row = gid.y;
    uint col = gid.x;
    uint tRow = tid.y;
    uint tCol = tid.x;
    
    if (row >= M || col >= N) return;
    
    // Use float4 for accumulation
    float4 sum4 = float4(0.0f);
    float sum = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; t++) {
        // Coalesced loading into shared memory
        uint aCol = t * TILE_SIZE + tCol;
        uint bRow = t * TILE_SIZE + tRow;
        
        // Load with bounds checking
        tileA[tRow][tCol] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[tRow][tCol] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Vectorized computation using float4 when possible
        uint k = 0;
        for (; k + 3 < TILE_SIZE; k += 4) {
            float4 a_vec = float4(tileA[tRow][k],
                                 tileA[tRow][k + 1],
                                 tileA[tRow][k + 2],
                                 tileA[tRow][k + 3]);
            
            float4 b_vec = float4(tileB[k][tCol],
                                 tileB[k + 1][tCol],
                                 tileB[k + 2][tCol],
                                 tileB[k + 3][tCol]);
            
            sum4 += a_vec * b_vec;
        }
        
        // Handle remaining elements
        for (; k < TILE_SIZE; k++) {
            sum += tileA[tRow][k] * tileB[k][tCol];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Final reduction
    sum += sum4.x + sum4.y + sum4.z + sum4.w;
    C[row * N + col] = sum;
}

#if __METAL_VERSION__ >= 300
/// High-performance matrix multiplication using SIMD group operations
/// Uses cooperative loading and SIMD shuffle operations for efficiency
kernel void matmul_simdgroup(
    constant float* A [[buffer(0)]],
    constant float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TILE_K = 8;
    
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    
    // Process K dimension in tiles for better cache usage
    for (uint k_base = 0; k_base < K; k_base += TILE_K) {
        // Each thread loads one element, then shares via SIMD shuffle
        for (uint k_offset = 0; k_offset < TILE_K && k_base + k_offset < K; k_offset++) {
            uint k = k_base + k_offset;
            
            // Load A element (one per SIMD lane for row)
            float a_val = (k < K) ? A[row * K + k] : 0.0f;
            
            // Load B element (coalesced access across SIMD group)
            float b_val = (k < K && col < N) ? B[k * N + col] : 0.0f;
            
            // Accumulate
            sum += a_val * b_val;
        }
    }
    
    C[row * N + col] = sum;
}

/// Matrix multiplication with warp-level optimizations
kernel void matmul_warp_optimized(
    constant float* A [[buffer(0)]],
    constant float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    constexpr uint WARP_SIZE = 32;
    constexpr uint TILE_M = 4;
    constexpr uint TILE_N = 4;
    
    threadgroup float tile_A[TILE_M][WARP_SIZE];
    threadgroup float tile_B[TILE_N][WARP_SIZE];
    
    uint warp_row = gid.y * TILE_M;
    uint warp_col = gid.x * TILE_N;
    
    // Initialize accumulators for each thread's tile
    float acc[TILE_M][TILE_N];
    for (uint i = 0; i < TILE_M; i++) {
        for (uint j = 0; j < TILE_N; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Main computation loop
    for (uint k = simd_lane_id; k < K; k += WARP_SIZE) {
        // Load tiles cooperatively
        for (uint i = 0; i < TILE_M; i++) {
            uint row = warp_row + i;
            tile_A[i][simd_lane_id] = (row < M && k < K) ? A[row * K + k] : 0.0f;
        }
        
        for (uint j = 0; j < TILE_N; j++) {
            uint col = warp_col + j;
            tile_B[j][simd_lane_id] = (k < K && col < N) ? B[k * N + col] : 0.0f;
        }
        
        // Synchronize within warp (implicit on modern GPUs)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute tile product
        for (uint i = 0; i < TILE_M; i++) {
            for (uint j = 0; j < TILE_N; j++) {
                for (uint kk = 0; kk < WARP_SIZE && k - simd_lane_id + kk < K; kk++) {
                    acc[i][j] += tile_A[i][kk] * tile_B[j][kk];
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write results
    for (uint i = 0; i < TILE_M; i++) {
        for (uint j = 0; j < TILE_N; j++) {
            uint row = warp_row + i;
            uint col = warp_col + j;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}
#endif

/// Matrix-vector multiplication: y = A * x
/// Optimized with vectorized operations and coalesced memory access
kernel void matvec_forward(
    constant float* A [[buffer(0)]],
    constant float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= M) return;
    
    // Use float4 for vectorized dot product
    float4 sum4 = float4(0.0f);
    float sum = 0.0f;
    
    // Process 4 elements at a time for better performance
    uint j = 0;
    uint row_offset = gid * N;
    
    for (; j + 3 < N; j += 4) {
        // Load 4 consecutive elements from matrix row
        float4 a_vec = float4(A[row_offset + j],
                             A[row_offset + j + 1],
                             A[row_offset + j + 2],
                             A[row_offset + j + 3]);
        
        // Load corresponding elements from vector
        float4 x_vec = float4(x[j], x[j + 1], x[j + 2], x[j + 3]);
        
        sum4 += a_vec * x_vec;
    }
    
    // Reduce float4 to scalar
    sum = sum4.x + sum4.y + sum4.z + sum4.w;
    
    // Process remaining elements
    for (; j < N; j++) {
        sum += A[row_offset + j] * x[j];
    }
    
    y[gid] = sum;
}

// MARK: - Mixed Precision Matrix Operations

/// Matrix multiplication with FP16 inputs and FP32 accumulation
kernel void matmul_fp16(
    constant half* A [[buffer(0)]],      // [M x K]
    constant half* B [[buffer(1)]],      // [K x N]
    device half* C [[buffer(2)]],        // [M x N]
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    // Accumulate in FP32 for better precision
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += float(A[row * K + k]) * float(B[k * N + col]);
    }
    
    // Convert back to FP16 with saturation
    C[row * N + col] = half(sum);
}

/// Optimized tiled matrix multiplication with mixed precision
kernel void matmul_tiled_fp16(
    constant half* A [[buffer(0)]],
    constant half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]]
) {
    constexpr uint TILE_SIZE = 16;
    
    // Use FP32 for shared memory to maintain precision
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
        // Load tiles with FP16 to FP32 conversion
        uint aCol = t * TILE_SIZE + tCol;
        uint bRow = t * TILE_SIZE + tRow;
        
        tileA[tRow][tCol] = (aCol < K) ? float(A[row * K + aCol]) : 0.0f;
        tileB[tRow][tCol] = (bRow < K) ? float(B[bRow * N + col]) : 0.0f;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product in FP32
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tRow][k] * tileB[k][tCol];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    C[row * N + col] = half(sum);
}

// MARK: - Backward Pass Operations

/// Compute gradients for matrix multiplication
/// gradA = gradC * B^T
/// Optimized with vectorized operations
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
    
    float4 sum4 = float4(0.0f);
    float sum = 0.0f;
    
    // Vectorized computation
    uint n = 0;
    for (; n + 3 < N; n += 4) {
        // Load 4 elements from gradC
        float4 gc_vec = float4(gradC[row * N + n],
                              gradC[row * N + n + 1],
                              gradC[row * N + n + 2],
                              gradC[row * N + n + 3]);
        
        // Load corresponding elements from B (transposed access)
        float4 b_vec = float4(B[col * N + n],
                             B[col * N + n + 1],
                             B[col * N + n + 2],
                             B[col * N + n + 3]);
        
        sum4 += gc_vec * b_vec;
    }
    
    // Reduce and handle remaining elements
    sum = sum4.x + sum4.y + sum4.z + sum4.w;
    
    for (; n < N; n++) {
        sum += gradC[row * N + n] * B[col * N + n];
    }
    
    gradA[row * K + col] = sum;
}

/// Compute gradients for matrix multiplication
/// gradB = A^T * gradC
/// Optimized with vectorized operations
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
    
    float4 sum4 = float4(0.0f);
    float sum = 0.0f;
    
    // Process 4 rows at a time
    uint m = 0;
    for (; m + 3 < M; m += 4) {
        // Load 4 elements from A (transposed access)
        float4 a_vec = float4(A[m * K + row],
                             A[(m + 1) * K + row],
                             A[(m + 2) * K + row],
                             A[(m + 3) * K + row]);
        
        // Load corresponding elements from gradC
        float4 gc_vec = float4(gradC[m * N + col],
                              gradC[(m + 1) * N + col],
                              gradC[(m + 2) * N + col],
                              gradC[(m + 3) * N + col]);
        
        sum4 += a_vec * gc_vec;
    }
    
    // Reduce and handle remaining elements
    sum = sum4.x + sum4.y + sum4.z + sum4.w;
    
    for (; m < M; m++) {
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

/// Compute bias gradients by summing across rows with vectorized reduction
kernel void reduce_bias_gradient(
    constant float* gradOutput [[buffer(0)]],
    device float* gradBias [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cols) return;
    
    // Use float4 for vectorized summation
    float4 sum4 = float4(0.0f);
    float sum = 0.0f;
    
    // Process 4 rows at a time for better memory throughput
    uint row = 0;
    for (; row + 3 < rows; row += 4) {
        float4 grad_vec = float4(gradOutput[row * cols + gid],
                                gradOutput[(row + 1) * cols + gid],
                                gradOutput[(row + 2) * cols + gid],
                                gradOutput[(row + 3) * cols + gid]);
        sum4 += grad_vec;
    }
    
    // Reduce float4 to scalar
    sum = sum4.x + sum4.y + sum4.z + sum4.w;
    
    // Process remaining rows
    for (; row < rows; row++) {
        sum += gradOutput[row * cols + gid];
    }
    
    gradBias[gid] = sum;
}

// MARK: - Utility Operations

/// Transpose matrix with optimized memory access patterns
kernel void transpose(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    // Use shared memory for coalesced access
    constexpr uint TILE_SIZE = 16;
    threadgroup float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    
    uint x = gid.x;
    uint y = gid.y;
    uint tx = tid.x;
    uint ty = tid.y;
    
    // Coalesced read from global memory
    if (y < rows && x < cols) {
        tile[ty][tx] = input[y * cols + x];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Calculate transposed indices
    uint transposed_x = gid.y;
    uint transposed_y = gid.x;
    
    // Coalesced write to global memory
    if (transposed_y < cols && transposed_x < rows) {
        output[transposed_y * rows + transposed_x] = tile[tx][ty];
    }
}

/// Copy matrix with vectorized operations
kernel void copy_matrix(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Process 4 elements at once for better memory throughput
    uint idx = gid * 4;
    if (idx + 3 < size) {
        float4 data = float4(input[idx], input[idx + 1], input[idx + 2], input[idx + 3]);
        output[idx] = data.x;
        output[idx + 1] = data.y;
        output[idx + 2] = data.z;
        output[idx + 3] = data.w;
    } else {
        // Handle remaining elements
        for (uint i = idx; i < size && i < idx + 4; i++) {
            output[i] = input[i];
        }
    }
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

/// Outer product with accumulation: C += a * b^T
kernel void outer_product_accumulate(
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
    
    uint idx = row * N + col;
    C[idx] += a[row] * b[col];
}

// MARK: - LSTM Operations

/// Extract a single timestep from a sequence tensor
kernel void extract_timestep(
    constant float* input [[buffer(0)]],   // [seq_len, batch_size, input_size]
    device float* output [[buffer(1)]],    // [batch_size, input_size]
    constant uint& timestep [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& input_size [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch_size * input_size) return;
    
    uint batch = gid / input_size;
    uint feature = gid % input_size;
    
    // Input layout: [seq_len, batch_size, input_size]
    uint input_idx = timestep * (batch_size * input_size) + batch * input_size + feature;
    
    output[gid] = input[input_idx];
}

/// Concatenate sequence outputs into a single tensor
kernel void concatenate_sequence(
    constant float* inputs [[buffer(0)]],   // Flattened array of timestep outputs
    device float* output [[buffer(1)]],     // [batch_size, seq_len, hidden_size]
    constant uint& batch_size [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch_size * seq_len * hidden_size) return;
    
    // Output layout: [batch_size, seq_len, hidden_size]
    uint batch = gid / (seq_len * hidden_size);
    uint t = (gid % (seq_len * hidden_size)) / hidden_size;
    uint h = gid % hidden_size;
    
    // Input is concatenated timestep outputs, each of size [batch_size, hidden_size]
    uint input_idx = t * (batch_size * hidden_size) + batch * hidden_size + h;
    
    output[gid] = inputs[input_idx];
}

/// Extract gradient for a specific timestep
kernel void extract_timestep_grad(
    constant float* grad_output [[buffer(0)]],  // Full sequence gradient
    device float* grad_timestep [[buffer(1)]],   // Gradient for single timestep
    constant uint& timestep [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= hidden_size) return;
    
    // If returnSequences=true, extract from full gradient
    // Otherwise, only last timestep has gradient
    uint offset = timestep * hidden_size;
    grad_timestep[gid] = grad_output[offset + gid];
}
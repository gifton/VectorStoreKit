// VectorStoreKit: Matrix Operation Shaders
//
// GPU kernels for matrix operations
// These shaders provide fundamental linear algebra operations optimized for
// Apple Silicon GPUs, supporting vector database operations and ML workloads

#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// MARK: - Matrix Multiplication

/// Basic matrix multiplication: C = A * B
/// Computes the product of two matrices using the standard algorithm
/// 
/// Mathematical operation:
/// C[i,j] = Σ(A[i,k] * B[k,j]) for k = 0 to colsA-1
///
/// Thread organization:
/// - Each thread computes one element of the result matrix
/// - 2D grid where thread (x,y) computes C[y,x]
/// - Natural parallelism: all output elements independent
///
/// Performance characteristics:
/// - O(colsA) operations per thread
/// - Memory access pattern can be inefficient for large matrices
/// - Consider tiled version for better cache utilization
///
/// Matrix dimensions:
/// - A: rowsA x colsA
/// - B: colsA x colsB (colsA must match for valid multiplication)
/// - C: rowsA x colsB
///
/// @param matrixA Left matrix in row-major format
/// @param matrixB Right matrix in row-major format
/// @param matrixC Output matrix in row-major format
/// @param rowsA Number of rows in matrix A
/// @param colsA Number of columns in A (must equal rows in B)
/// @param colsB Number of columns in matrix B
/// @param id 2D thread position (x: output column, y: output row)
kernel void matrixMultiply(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* matrixC [[buffer(2)]],
    constant uint& rowsA [[buffer(3)]],
    constant uint& colsA [[buffer(4)]],
    constant uint& colsB [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint row = id.y;
    uint col = id.x;
    
    // Bounds check for threads beyond matrix dimensions
    if (row >= rowsA || col >= colsB) return;
    
    // Compute dot product of row from A with column from B
    float sum = 0.0;
    for (uint k = 0; k < colsA; ++k) {
        // A[row][k] * B[k][col]
        sum += matrixA[row * colsA + k] * matrixB[k * colsB + col];
    }
    
    // Store result in C[row][col]
    matrixC[row * colsB + col] = sum;
}

// MARK: - Tiled Matrix Multiplication (Optimized)

/// Tile size for shared memory optimization
/// 16x16 is optimal for many GPUs balancing occupancy and shared memory usage
constant uint TILE_SIZE = 16;

/// Optimized matrix multiplication using tiling for better cache utilization
/// This implementation loads tiles of matrices into fast threadgroup memory,
/// dramatically reducing global memory accesses
///
/// Algorithm overview:
/// 1. Divide matrices into TILE_SIZE x TILE_SIZE tiles
/// 2. Load one tile from A and B into shared memory
/// 3. Compute partial products using shared memory
/// 4. Accumulate results across all tiles
///
/// Performance improvements:
/// - Reduces global memory accesses by factor of TILE_SIZE
/// - Exploits data reuse within threadgroup
/// - Coalesced memory access patterns
/// - 2-10x faster than naive implementation for large matrices
///
/// Shared memory usage:
/// - 2 * TILE_SIZE * TILE_SIZE * sizeof(float) bytes per threadgroup
/// - Threads cooperatively load tiles
/// - Synchronization ensures data consistency
///
/// @param matrixA Input matrix A (rowsA x colsA)
/// @param matrixB Input matrix B (colsA x colsB)
/// @param matrixC Output matrix C (rowsA x colsB)
/// @param rowsA Rows in matrix A
/// @param colsA Columns in A / Rows in B
/// @param colsB Columns in matrix B
/// @param gid Global thread position in grid
/// @param tid Local thread position in threadgroup
/// @param tgSize Threadgroup dimensions
kernel void tiledMatrixMultiply(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* matrixC [[buffer(2)]],
    constant uint& rowsA [[buffer(3)]],
    constant uint& colsA [[buffer(4)]],
    constant uint& colsB [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]]
) {
    // Allocate shared memory for tiles
    // Each threadgroup has its own copy of these tiles
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    // Global position for this thread's output element
    uint row = gid.y;
    uint col = gid.x;
    // Local position within threadgroup
    uint tx = tid.x;
    uint ty = tid.y;
    
    // Early exit for threads beyond matrix bounds
    if (row >= rowsA || col >= colsB) return;
    
    float sum = 0.0;
    // Calculate number of tiles needed to cover the shared dimension
    uint numTiles = (colsA + TILE_SIZE - 1) / TILE_SIZE;
    
    // Iterate over tiles along the shared dimension
    for (uint t = 0; t < numTiles; ++t) {
        // Cooperatively load tile from matrix A
        // Each thread loads one element
        uint aCol = t * TILE_SIZE + tx;
        uint aRow = row;
        if (aCol < colsA && aRow < rowsA) {
            tileA[ty][tx] = matrixA[aRow * colsA + aCol];
        } else {
            // Pad with zeros for out-of-bounds accesses
            tileA[ty][tx] = 0.0;
        }
        
        // Cooperatively load tile from matrix B
        uint bRow = t * TILE_SIZE + ty;
        uint bCol = col;
        if (bRow < colsA && bCol < colsB) {
            tileB[ty][tx] = matrixB[bRow * colsB + bCol];
        } else {
            tileB[ty][tx] = 0.0;
        }
        
        // Synchronize to ensure all threads have loaded their data
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product using shared memory
        // This is much faster than accessing global memory
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        // Synchronize before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write final result to global memory
    matrixC[row * colsB + col] = sum;
}

// MARK: - Matrix Transpose

/// Simple matrix transpose operation: B = A^T
/// Swaps rows and columns of the input matrix
///
/// Mathematical operation:
/// B[j,i] = A[i,j] for all valid i,j
///
/// Use cases:
/// - Data layout transformation
/// - Preparation for certain matrix operations
/// - Converting between row-major and column-major formats
///
/// Memory access pattern:
/// - Input: Sequential read along rows
/// - Output: Strided write along columns
/// - Can cause cache inefficiency for large matrices
/// - Consider coalesced version for better performance
///
/// @param input Source matrix in row-major format (rows x cols)
/// @param output Destination matrix in row-major format (cols x rows)
/// @param rows Number of rows in input matrix
/// @param cols Number of columns in input matrix
/// @param id 2D thread index (x: input column, y: input row)
kernel void matrixTranspose(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint inRow = id.y;
    uint inCol = id.x;
    
    // Bounds check
    if (inRow >= rows || inCol >= cols) return;
    
    // Calculate linear indices
    // Input: row-major indexing
    uint inIndex = inRow * cols + inCol;
    // Output: transposed position
    uint outIndex = inCol * rows + inRow;
    
    // Copy element to transposed position
    output[outIndex] = input[inIndex];
}

// MARK: - Coalesced Transpose (Optimized for memory access)

/// Tile size for transpose operation
/// 32x32 provides good balance of occupancy and shared memory usage
constant uint TRANSPOSE_TILE_SIZE = 32;

/// Optimized matrix transpose using shared memory for coalesced access
/// This implementation dramatically improves memory bandwidth utilization
///
/// Key optimizations:
/// 1. Coalesced reads: Threads in a warp read consecutive memory locations
/// 2. Shared memory: Tile loaded once, accessed multiple times
/// 3. Bank conflict avoidance: +1 padding prevents shared memory conflicts
/// 4. Coalesced writes: After transpose in shared memory
///
/// Performance benefits:
/// - Up to 10x faster than naive transpose for large matrices
/// - Near-optimal memory bandwidth utilization
/// - Reduced memory traffic through caching
///
/// Bank conflict prevention:
/// - Without padding: threads access same bank when reading column
/// - With +1 padding: shifts columns to different banks
/// - Results in conflict-free shared memory access
///
/// @param input Source matrix (rows x cols)
/// @param output Destination matrix (cols x rows)
/// @param rows Input matrix rows
/// @param cols Input matrix columns
/// @param gid Global thread position
/// @param tid Local thread position in threadgroup
kernel void matrixTransposeCoalesced(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    // Shared memory tile with padding to avoid bank conflicts
    // The +1 padding ensures threads don't access the same memory bank
    threadgroup float tile[TRANSPOSE_TILE_SIZE][TRANSPOSE_TILE_SIZE + 1];
    
    uint globalRow = gid.y;
    uint globalCol = gid.x;
    
    // Phase 1: Cooperatively load tile from global to shared memory
    // This read is coalesced - consecutive threads read consecutive addresses
    if (globalRow < rows && globalCol < cols) {
        tile[tid.y][tid.x] = input[globalRow * cols + globalCol];
    }
    
    // Ensure all threads have completed loading
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: Calculate position in transposed matrix
    // Complex indexing handles the tile-based transpose
    uint transposedRow = gid.x / TRANSPOSE_TILE_SIZE * TRANSPOSE_TILE_SIZE + tid.y;
    uint transposedCol = gid.y / TRANSPOSE_TILE_SIZE * TRANSPOSE_TILE_SIZE + tid.x;
    
    // Phase 3: Write transposed data from shared to global memory
    // Note the swapped indices [tid.x][tid.y] to perform transpose
    // This write is also coalesced after the transpose
    if (transposedRow < cols && transposedCol < rows) {
        output[transposedRow * rows + transposedCol] = tile[tid.x][tid.y];
    }
}

// MARK: - Element-wise Operations

/// Element-wise matrix addition: C = A + B
/// Each element C[i,j] = A[i,j] + B[i,j]
///
/// Requirements:
/// - Matrices must have identical dimensions
/// - Highly parallel - each element processed independently
/// - Memory bandwidth limited operation
///
/// @param matrixA First input matrix
/// @param matrixB Second input matrix 
/// @param result Output matrix
/// @param id Linear thread index = element index
kernel void matrixAdd(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = matrixA[id] + matrixB[id];
}

/// Element-wise matrix subtraction: C = A - B
/// Each element C[i,j] = A[i,j] - B[i,j]
///
/// Requirements:
/// - Matrices must have identical dimensions
/// - Order matters: A - B ≠ B - A
///
/// @param matrixA First input matrix (minuend)
/// @param matrixB Second input matrix (subtrahend)
/// @param result Output matrix
/// @param id Linear thread index
kernel void matrixSubtract(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = matrixA[id] - matrixB[id];
}

/// Scalar multiplication: C = α * A
/// Multiply every element of matrix by a scalar value
///
/// Use cases:
/// - Scaling matrices
/// - Learning rate application in gradient descent
/// - Normalization operations
///
/// Performance:
/// - Purely memory bandwidth bound
/// - Consider fusing with other operations when possible
///
/// @param matrix Input matrix
/// @param scalar Multiplication factor
/// @param result Output matrix
/// @param id Linear thread index
kernel void matrixScalarMultiply(
    device const float* matrix [[buffer(0)]],
    constant float& scalar [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = matrix[id] * scalar;
}

kernel void matrixMultiplyElementwise(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = matrixA[id] * matrixB[id];
}

kernel void matrixDivideElementwise(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float b = matrixB[id];
    result[id] = (b != 0.0) ? matrixA[id] / b : 0.0;
}

kernel void matrixMaxElementwise(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = max(matrixA[id], matrixB[id]);
}

kernel void matrixMinElementwise(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = min(matrixA[id], matrixB[id]);
}

// MARK: - Matrix Reduction Operations

/// Sum elements along each row of a matrix
/// Reduces a matrix from shape (rows, cols) to vector of shape (rows,)
///
/// Use cases:
/// - Computing row-wise statistics
/// - Marginalization in probability matrices
/// - Feature aggregation in ML
///
/// Performance notes:
/// - Each thread handles one complete row
/// - Sequential memory access pattern
/// - Consider using parallel reduction for very wide matrices
///
/// @param matrix Input matrix in row-major format
/// @param rowSums Output vector containing sum of each row
/// @param cols Number of columns in the matrix
/// @param id Thread index = row index
kernel void matrixRowSum(
    device const float* matrix [[buffer(0)]],
    device float* rowSums [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint row = id;
    float sum = 0.0;
    
    // Sum all elements in this row
    // Sequential access pattern is cache-friendly
    for (uint col = 0; col < cols; ++col) {
        sum += matrix[row * cols + col];
    }
    
    rowSums[row] = sum;
}

/// Sum elements along each column of a matrix
/// Reduces a matrix from shape (rows, cols) to vector of shape (cols,)
///
/// Use cases:
/// - Computing column-wise statistics
/// - Aggregating features across samples
/// - Normalization preprocessing
///
/// Performance considerations:
/// - Strided memory access pattern (less cache-friendly)
/// - Each thread jumps by 'cols' elements
/// - Consider transposing first for better access pattern
///
/// @param matrix Input matrix in row-major format
/// @param colSums Output vector containing sum of each column
/// @param rows Number of rows in the matrix
/// @param cols Number of columns in the matrix
/// @param id Thread index = column index
kernel void matrixColSum(
    device const float* matrix [[buffer(0)]],
    device float* colSums [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint col = id;
    float sum = 0.0;
    
    // Sum all elements in this column
    // Strided access pattern - less efficient than row sum
    for (uint row = 0; row < rows; ++row) {
        sum += matrix[row * cols + col];
    }
    
    colSums[col] = sum;
}

// MARK: - Batch Operations

kernel void batchMatrixAdd(
    device const float* matricesA [[buffer(0)]],
    device const float* matricesB [[buffer(1)]],
    device float* results [[buffer(2)]],
    constant uint& matrixSize [[buffer(3)]],
    constant uint& batchSize [[buffer(4)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint batchIdx = id.y;
    uint elementIdx = id.x;
    
    if (batchIdx >= batchSize || elementIdx >= matrixSize) return;
    
    uint offset = batchIdx * matrixSize + elementIdx;
    results[offset] = matricesA[offset] + matricesB[offset];
}

// MARK: - Advanced Operations

/// Compute Frobenius norm of a matrix (L2 norm of all elements)
/// Frobenius norm = sqrt(ΣΣ(A[i,j]²))
///
/// This kernel uses parallel reduction for efficient computation:
/// 1. Each thread computes partial sum of squared elements
/// 2. Threads cooperatively reduce partial sums in shared memory
/// 3. Final thread computes square root
///
/// Parallel reduction pattern:
/// - Tree-based reduction in log(threadgroup_size) steps
/// - Minimizes synchronization overhead
/// - Achieves near-optimal parallelism
///
/// Use cases:
/// - Matrix normalization
/// - Convergence checking in iterative algorithms
/// - Error measurement
///
/// Performance:
/// - O(n) work divided among threads
/// - O(log p) reduction steps where p = threadgroup size
/// - Bandwidth limited for large matrices
///
/// @param matrix Input matrix (flattened)
/// @param result Single-element output array for the norm
/// @param size Total number of elements in matrix
/// @param sharedMem Threadgroup memory for parallel reduction
/// @param tid Thread index within threadgroup
/// @param tgSize Number of threads in threadgroup
kernel void matrixNorm(
    device const float* matrix [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    threadgroup float* sharedMem [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    float localSum = 0.0;
    
    // Phase 1: Each thread computes partial sum of squared elements
    // Strided access ensures work is distributed evenly
    for (uint i = tid; i < size; i += tgSize) {
        float val = matrix[i];
        localSum += val * val;
    }
    
    // Store partial sum in shared memory
    sharedMem[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Phase 2: Parallel reduction in shared memory
    // Tree-based reduction: each step halves the active threads
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        // Synchronize after each reduction step
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Phase 3: Final thread computes square root and writes result
    if (tid == 0) {
        result[0] = sqrt(sharedMem[0]);
    }
}
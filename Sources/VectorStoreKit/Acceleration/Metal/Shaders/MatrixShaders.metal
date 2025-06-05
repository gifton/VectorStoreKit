// VectorStoreKit: Matrix Operation Shaders
//
// GPU kernels for matrix operations

#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// MARK: - Matrix Multiplication

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
    
    if (row >= rowsA || col >= colsB) return;
    
    float sum = 0.0;
    for (uint k = 0; k < colsA; ++k) {
        sum += matrixA[row * colsA + k] * matrixB[k * colsB + col];
    }
    
    matrixC[row * colsB + col] = sum;
}

// MARK: - Tiled Matrix Multiplication (Optimized)

constant uint TILE_SIZE = 16;

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
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    uint row = gid.y;
    uint col = gid.x;
    uint tx = tid.x;
    uint ty = tid.y;
    
    if (row >= rowsA || col >= colsB) return;
    
    float sum = 0.0;
    uint numTiles = (colsA + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint t = 0; t < numTiles; ++t) {
        // Load tile from matrix A
        uint aCol = t * TILE_SIZE + tx;
        uint aRow = row;
        if (aCol < colsA && aRow < rowsA) {
            tileA[ty][tx] = matrixA[aRow * colsA + aCol];
        } else {
            tileA[ty][tx] = 0.0;
        }
        
        // Load tile from matrix B
        uint bRow = t * TILE_SIZE + ty;
        uint bCol = col;
        if (bRow < colsA && bCol < colsB) {
            tileB[ty][tx] = matrixB[bRow * colsB + bCol];
        } else {
            tileB[ty][tx] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    matrixC[row * colsB + col] = sum;
}

// MARK: - Matrix Transpose

kernel void matrixTranspose(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint inRow = id.y;
    uint inCol = id.x;
    
    if (inRow >= rows || inCol >= cols) return;
    
    uint inIndex = inRow * cols + inCol;
    uint outIndex = inCol * rows + inRow;
    
    output[outIndex] = input[inIndex];
}

// MARK: - Coalesced Transpose (Optimized for memory access)

constant uint TRANSPOSE_TILE_SIZE = 32;

kernel void matrixTransposeCoalesced(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    threadgroup float tile[TRANSPOSE_TILE_SIZE][TRANSPOSE_TILE_SIZE + 1]; // +1 to avoid bank conflicts
    
    uint globalRow = gid.y;
    uint globalCol = gid.x;
    
    // Load tile into shared memory
    if (globalRow < rows && globalCol < cols) {
        tile[tid.y][tid.x] = input[globalRow * cols + globalCol];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Calculate transposed position
    uint transposedRow = gid.x / TRANSPOSE_TILE_SIZE * TRANSPOSE_TILE_SIZE + tid.y;
    uint transposedCol = gid.y / TRANSPOSE_TILE_SIZE * TRANSPOSE_TILE_SIZE + tid.x;
    
    // Write transposed tile to output
    if (transposedRow < cols && transposedCol < rows) {
        output[transposedRow * rows + transposedCol] = tile[tid.x][tid.y];
    }
}

// MARK: - Element-wise Operations

kernel void matrixAdd(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = matrixA[id] + matrixB[id];
}

kernel void matrixSubtract(
    device const float* matrixA [[buffer(0)]],
    device const float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = matrixA[id] - matrixB[id];
}

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

kernel void matrixRowSum(
    device const float* matrix [[buffer(0)]],
    device float* rowSums [[buffer(1)]],
    constant uint& cols [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint row = id;
    float sum = 0.0;
    
    for (uint col = 0; col < cols; ++col) {
        sum += matrix[row * cols + col];
    }
    
    rowSums[row] = sum;
}

kernel void matrixColSum(
    device const float* matrix [[buffer(0)]],
    device float* colSums [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    uint col = id;
    float sum = 0.0;
    
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

kernel void matrixNorm(
    device const float* matrix [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    threadgroup float* sharedMem [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    float localSum = 0.0;
    
    // Each thread computes partial sum
    for (uint i = tid; i < size; i += tgSize) {
        float val = matrix[i];
        localSum += val * val;
    }
    
    sharedMem[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = tgSize / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedMem[tid] += sharedMem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        result[0] = sqrt(sharedMem[0]);
    }
}
// VectorStoreKit: Matrix Operation Shaders
//
// GPU kernels for matrix operations

#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// MARK: - Matrix Multiplication

kernel void matrixMultiply(
    constant float* matrixA [[buffer(0)]],
    constant float* matrixB [[buffer(1)]],
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
    constant float* matrixA [[buffer(0)]],
    constant float* matrixB [[buffer(1)]],
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
    constant float* input [[buffer(0)]],
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

// MARK: - Element-wise Operations

kernel void matrixAdd(
    constant float* matrixA [[buffer(0)]],
    constant float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = matrixA[id] + matrixB[id];
}

kernel void matrixSubtract(
    constant float* matrixA [[buffer(0)]],
    constant float* matrixB [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = matrixA[id] - matrixB[id];
}

kernel void matrixScalarMultiply(
    constant float* matrix [[buffer(0)]],
    constant float& scalar [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = matrix[id] * scalar;
}

// MARK: - Matrix Reduction Operations

kernel void matrixRowSum(
    constant float* matrix [[buffer(0)]],
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
    constant float* matrix [[buffer(0)]],
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
// VectorStoreKit: Distance Matrix Computation Shaders
//
// GPU kernels for computing full distance matrices between vector sets
// Optimized for massive parallelism and memory efficiency
//

#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>
using namespace metal;

// MARK: - Constants

constant uint TILE_SIZE [[function_constant(0)]]; // Default 32
constant uint WARP_SIZE = 32;
constant uint BANK_PADDING = 1;

// MARK: - Helper Functions

/// Warp-level reduction for maximum value
inline float warpReduceMax(float val) {
    val = max(val, simd_shuffle_down(val, 16));
    val = max(val, simd_shuffle_down(val, 8));
    val = max(val, simd_shuffle_down(val, 4));
    val = max(val, simd_shuffle_down(val, 2));
    val = max(val, simd_shuffle_down(val, 1));
    return val;
}

/// Warp-level reduction for sum
inline float warpReduceSum(float val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

// MARK: - Distance Matrix Kernels

/// Compute full distance matrix using tiled algorithm with shared memory
/// This kernel efficiently computes all pairwise distances between two vector sets
///
/// Algorithm:
/// 1. Tile the computation into blocks to maximize data reuse
/// 2. Load tiles into shared memory for fast access
/// 3. Compute partial distances for each tile
/// 4. Accumulate results across tiles
///
/// Memory access pattern:
/// - Coalesced global memory reads
/// - Bank-conflict-free shared memory access
/// - Maximized data reuse within threadgroups
///
/// @param vectorsA First set of vectors [numVectorsA * dimension]
/// @param vectorsB Second set of vectors [numVectorsB * dimension]
/// @param distanceMatrix Output matrix [numVectorsA * numVectorsB]
/// @param dimension Vector dimensionality
/// @param numVectorsA Number of vectors in set A
/// @param numVectorsB Number of vectors in set B
/// @param metric Distance metric (0=euclidean, 1=cosine, 2=manhattan)
[[kernel]]
void distanceMatrixTiled(
    constant float* vectorsA [[buffer(0)]],
    constant float* vectorsB [[buffer(1)]],
    device float* distanceMatrix [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant uint& numVectorsA [[buffer(4)]],
    constant uint& numVectorsB [[buffer(5)]],
    constant uint& metric [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgSize [[threads_per_threadgroup]]
) {
    const uint rowA = gid.y;
    const uint colB = gid.x;
    
    if (rowA >= numVectorsA || colB >= numVectorsB) return;
    
    // Allocate shared memory for tiles
    threadgroup float tileA[32][32 + BANK_PADDING];
    threadgroup float tileB[32][32 + BANK_PADDING];
    
    float accumulator = 0.0f;
    float magnitudeA = 0.0f;
    float magnitudeB = 0.0f;
    
    // Process dimension in tiles
    const uint numTiles = (dimension + TILE_SIZE - 1) / TILE_SIZE;
    
    for (uint tile = 0; tile < numTiles; ++tile) {
        // Cooperative loading of tiles
        const uint dimStart = tile * TILE_SIZE;
        const uint dimEnd = min(dimStart + TILE_SIZE, dimension);
        const uint tileWidth = dimEnd - dimStart;
        
        // Load tile A
        if (tid.x < tileWidth && tid.y < 1) {
            tileA[0][tid.x] = vectorsA[rowA * dimension + dimStart + tid.x];
        }
        
        // Load tile B
        if (tid.x < tileWidth && tid.y < 1) {
            tileB[0][tid.x] = vectorsB[colB * dimension + dimStart + tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial distance based on metric
        if (metric == 0) { // Euclidean
            for (uint i = 0; i < tileWidth; ++i) {
                float diff = tileA[0][i] - tileB[0][i];
                accumulator += diff * diff;
            }
        } else if (metric == 1) { // Cosine
            for (uint i = 0; i < tileWidth; ++i) {
                accumulator += tileA[0][i] * tileB[0][i];
                magnitudeA += tileA[0][i] * tileA[0][i];
                magnitudeB += tileB[0][i] * tileB[0][i];
            }
        } else if (metric == 2) { // Manhattan
            for (uint i = 0; i < tileWidth; ++i) {
                accumulator += abs(tileA[0][i] - tileB[0][i]);
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Finalize distance computation
    float distance;
    if (metric == 0) { // Euclidean
        distance = sqrt(accumulator);
    } else if (metric == 1) { // Cosine
        float similarity = accumulator / (sqrt(magnitudeA) * sqrt(magnitudeB) + 1e-8f);
        distance = 1.0f - similarity;
    } else { // Manhattan
        distance = accumulator;
    }
    
    // Write result
    distanceMatrix[rowA * numVectorsB + colB] = distance;
}

/// Optimized distance matrix for 512-dimensional vectors using float4
/// Specifically tuned for common embedding dimensions
[[kernel]]
void distanceMatrix512_euclidean(
    constant float4* vectorsA [[buffer(0)]],      // [numVectorsA * 128]
    constant float4* vectorsB [[buffer(1)]],      // [numVectorsB * 128]
    device float* distanceMatrix [[buffer(2)]],   // [numVectorsA * numVectorsB]
    constant uint& numVectorsA [[buffer(3)]],
    constant uint& numVectorsB [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint rowA = gid.y;
    const uint colB = gid.x;
    
    if (rowA >= numVectorsA || colB >= numVectorsB) return;
    
    constant float4* vecA = vectorsA + (rowA * 128);
    constant float4* vecB = vectorsB + (colB * 128);
    
    float4 sum = float4(0.0f);
    
    // Unrolled loop for 512 dimensions (128 float4s)
    #pragma unroll(16)
    for (uint i = 0; i < 128; i += 8) {
        float4 diff0 = vecA[i] - vecB[i];
        float4 diff1 = vecA[i+1] - vecB[i+1];
        float4 diff2 = vecA[i+2] - vecB[i+2];
        float4 diff3 = vecA[i+3] - vecB[i+3];
        float4 diff4 = vecA[i+4] - vecB[i+4];
        float4 diff5 = vecA[i+5] - vecB[i+5];
        float4 diff6 = vecA[i+6] - vecB[i+6];
        float4 diff7 = vecA[i+7] - vecB[i+7];
        
        sum += diff0 * diff0;
        sum += diff1 * diff1;
        sum += diff2 * diff2;
        sum += diff3 * diff3;
        sum += diff4 * diff4;
        sum += diff5 * diff5;
        sum += diff6 * diff6;
        sum += diff7 * diff7;
    }
    
    float distance = sqrt(sum.x + sum.y + sum.z + sum.w);
    distanceMatrix[rowA * numVectorsB + colB] = distance;
}

/// Symmetric distance matrix computation (only upper triangle)
/// Exploits symmetry property: d(a,b) = d(b,a)
[[kernel]]
void distanceMatrixSymmetric(
    constant float* vectors [[buffer(0)]],
    device float* distanceMatrix [[buffer(1)]],
    constant uint& dimension [[buffer(2)]],
    constant uint& numVectors [[buffer(3)]],
    constant uint& metric [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint i = gid.y;
    const uint j = gid.x;
    
    // Only compute upper triangle (including diagonal)
    if (i > j || i >= numVectors || j >= numVectors) return;
    
    if (i == j) {
        // Distance to self is always 0
        distanceMatrix[i * numVectors + j] = 0.0f;
        return;
    }
    
    constant float* vecA = vectors + (i * dimension);
    constant float* vecB = vectors + (j * dimension);
    
    float accumulator = 0.0f;
    float magnitudeA = 0.0f;
    float magnitudeB = 0.0f;
    
    // Compute distance based on metric
    if (metric == 0) { // Euclidean
        for (uint d = 0; d < dimension; ++d) {
            float diff = vecA[d] - vecB[d];
            accumulator += diff * diff;
        }
        accumulator = sqrt(accumulator);
    } else if (metric == 1) { // Cosine
        for (uint d = 0; d < dimension; ++d) {
            accumulator += vecA[d] * vecB[d];
            magnitudeA += vecA[d] * vecA[d];
            magnitudeB += vecB[d] * vecB[d];
        }
        float similarity = accumulator / (sqrt(magnitudeA) * sqrt(magnitudeB) + 1e-8f);
        accumulator = 1.0f - similarity;
    } else if (metric == 2) { // Manhattan
        for (uint d = 0; d < dimension; ++d) {
            accumulator += abs(vecA[d] - vecB[d]);
        }
    }
    
    // Write both upper and lower triangle
    distanceMatrix[i * numVectors + j] = accumulator;
    distanceMatrix[j * numVectors + i] = accumulator;
}

/// Block-based distance matrix computation with shared memory optimization
/// Processes matrix in blocks for better cache utilization
[[kernel]]
void distanceMatrixBlocked(
    constant float4* vectorsA [[buffer(0)]],
    constant float4* vectorsB [[buffer(1)]],
    device float* distanceMatrix [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],       // Actual dimension (not float4 count)
    constant uint& numVectorsA [[buffer(4)]],
    constant uint& numVectorsB [[buffer(5)]],
    constant uint& metric [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint2 tgSize [[threads_per_threadgroup]]
) {
    const uint BLOCK_SIZE = 16;
    const uint float4Dimension = (dimension + 3) / 4;
    
    // Shared memory for block processing
    threadgroup float4 blockA[16][16];
    threadgroup float4 blockB[16][16];
    
    // Global indices
    const uint globalRowA = bid.y * BLOCK_SIZE + tid.y;
    const uint globalColB = bid.x * BLOCK_SIZE + tid.x;
    
    if (globalRowA >= numVectorsA || globalColB >= numVectorsB) return;
    
    float accumulator = 0.0f;
    float magnitudeA = 0.0f;
    float magnitudeB = 0.0f;
    
    // Process dimension in chunks
    for (uint dimChunk = 0; dimChunk < float4Dimension; dimChunk += BLOCK_SIZE) {
        // Load block data
        if (dimChunk + tid.x < float4Dimension) {
            blockA[tid.y][tid.x] = vectorsA[globalRowA * float4Dimension + dimChunk + tid.x];
            blockB[tid.y][tid.x] = vectorsB[globalColB * float4Dimension + dimChunk + tid.x];
        } else {
            blockA[tid.y][tid.x] = float4(0.0f);
            blockB[tid.y][tid.x] = float4(0.0f);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial distances
        for (uint k = 0; k < min(BLOCK_SIZE, float4Dimension - dimChunk); ++k) {
            float4 a = blockA[tid.y][k];
            float4 b = blockB[tid.x][k];
            
            if (metric == 0) { // Euclidean
                float4 diff = a - b;
                float4 squared = diff * diff;
                accumulator += squared.x + squared.y + squared.z + squared.w;
            } else if (metric == 1) { // Cosine
                float4 prod = a * b;
                accumulator += prod.x + prod.y + prod.z + prod.w;
                float4 a2 = a * a;
                float4 b2 = b * b;
                magnitudeA += a2.x + a2.y + a2.z + a2.w;
                magnitudeB += b2.x + b2.y + b2.z + b2.w;
            } else if (metric == 2) { // Manhattan
                float4 diff = abs(a - b);
                accumulator += diff.x + diff.y + diff.z + diff.w;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Finalize distance
    float distance;
    if (metric == 0) { // Euclidean
        distance = sqrt(accumulator);
    } else if (metric == 1) { // Cosine
        float similarity = accumulator / (sqrt(magnitudeA) * sqrt(magnitudeB) + 1e-8f);
        distance = 1.0f - similarity;
    } else { // Manhattan
        distance = accumulator;
    }
    
    distanceMatrix[globalRowA * numVectorsB + globalColB] = distance;
}

/// Streaming distance matrix computation for large datasets
/// Processes data in streaming fashion to handle datasets larger than GPU memory
[[kernel]]
void distanceMatrixStreaming(
    constant float* vectorsA [[buffer(0)]],
    constant float* vectorsB [[buffer(1)]],
    device float* distanceMatrix [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant uint& numVectorsA [[buffer(4)]],
    constant uint& numVectorsB [[buffer(5)]],
    constant uint& batchOffsetA [[buffer(6)]],    // Starting index for this batch in A
    constant uint& batchOffsetB [[buffer(7)]],    // Starting index for this batch in B
    constant uint& metric [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint localRowA = gid.y;
    const uint localColB = gid.x;
    
    // Global indices accounting for batch offset
    const uint globalRowA = batchOffsetA + localRowA;
    const uint globalColB = batchOffsetB + localColB;
    
    if (globalRowA >= numVectorsA || globalColB >= numVectorsB) return;
    
    constant float* vecA = vectorsA + (localRowA * dimension);
    constant float* vecB = vectorsB + (localColB * dimension);
    
    float accumulator = 0.0f;
    
    // Simple computation for streaming - optimize based on metric
    if (metric == 0) { // Euclidean
        for (uint d = 0; d < dimension; ++d) {
            float diff = vecA[d] - vecB[d];
            accumulator += diff * diff;
        }
        accumulator = sqrt(accumulator);
    } else if (metric == 1) { // Cosine
        float magnitudeA = 0.0f;
        float magnitudeB = 0.0f;
        for (uint d = 0; d < dimension; ++d) {
            accumulator += vecA[d] * vecB[d];
            magnitudeA += vecA[d] * vecA[d];
            magnitudeB += vecB[d] * vecB[d];
        }
        float similarity = accumulator / (sqrt(magnitudeA) * sqrt(magnitudeB) + 1e-8f);
        accumulator = 1.0f - similarity;
    } else if (metric == 2) { // Manhattan
        for (uint d = 0; d < dimension; ++d) {
            accumulator += abs(vecA[d] - vecB[d]);
        }
    }
    
    // Write to global position in matrix
    distanceMatrix[globalRowA * numVectorsB + globalColB] = accumulator;
}

/// Hybrid CPU/GPU distance matrix computation decision kernel
/// Computes a small sample to estimate computation time
[[kernel]]
void distanceMatrixHeuristicSample(
    constant float* vectorsA [[buffer(0)]],
    constant float* vectorsB [[buffer(1)]],
    device float* sampleDistances [[buffer(2)]],
    device float* timingInfo [[buffer(3)]],       // [avgTimePerDistance, estimatedTotalTime]
    constant uint& dimension [[buffer(4)]],
    constant uint& sampleSize [[buffer(5)]],      // Number of samples to compute
    uint id [[thread_position_in_grid]]
) {
    if (id >= sampleSize) return;
    
    // Sample random pairs
    uint rowA = id % 10;  // Sample first 10 vectors from A
    uint colB = id % 10;  // Sample first 10 vectors from B
    
    constant float* vecA = vectorsA + (rowA * dimension);
    constant float* vecB = vectorsB + (colB * dimension);
    
    // Time the computation
    uint64_t startTime = mach_absolute_time();
    
    float sum = 0.0f;
    for (uint d = 0; d < dimension; ++d) {
        float diff = vecA[d] - vecB[d];
        sum += diff * diff;
    }
    
    uint64_t endTime = mach_absolute_time();
    
    sampleDistances[id] = sqrt(sum);
    
    // Store timing information for first thread
    if (id == 0) {
        timingInfo[0] = float(endTime - startTime) / float(sampleSize);
    }
}

/// Optimized distance matrix for normalized vectors (cosine = dot product)
[[kernel]]
void distanceMatrixNormalized(
    constant float4* vectorsA [[buffer(0)]],
    constant float4* vectorsB [[buffer(1)]],
    device float* distanceMatrix [[buffer(2)]],
    constant uint& dimension [[buffer(3)]],
    constant uint& numVectorsA [[buffer(4)]],
    constant uint& numVectorsB [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint rowA = gid.y;
    const uint colB = gid.x;
    
    if (rowA >= numVectorsA || colB >= numVectorsB) return;
    
    const uint float4Dimension = (dimension + 3) / 4;
    constant float4* vecA = vectorsA + (rowA * float4Dimension);
    constant float4* vecB = vectorsB + (colB * float4Dimension);
    
    float4 dotProduct = float4(0.0f);
    
    // Compute dot product
    for (uint i = 0; i < float4Dimension; ++i) {
        dotProduct += vecA[i] * vecB[i];
    }
    
    float similarity = dotProduct.x + dotProduct.y + dotProduct.z + dotProduct.w;
    
    // Handle partial last float4 if dimension not divisible by 4
    if (dimension % 4 != 0) {
        uint lastIdx = float4Dimension - 1;
        float4 lastA = vecA[lastIdx];
        float4 lastB = vecB[lastIdx];
        float4 prod = lastA * lastB;
        
        uint remainder = dimension % 4;
        if (remainder == 1) similarity -= prod.y + prod.z + prod.w;
        else if (remainder == 2) similarity -= prod.z + prod.w;
        else if (remainder == 3) similarity -= prod.w;
    }
    
    distanceMatrix[rowA * numVectorsB + colB] = 1.0f - similarity;
}
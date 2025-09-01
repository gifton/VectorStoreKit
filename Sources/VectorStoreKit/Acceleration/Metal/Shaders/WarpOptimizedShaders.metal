// VectorStoreKit: Warp-Optimized Distance Shaders
//
// Advanced GPU kernels using warp-level primitives for maximum performance
//

#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>
using namespace metal;

// MARK: - Constants

#define WARP_SIZE 32        // Apple Silicon warp size
#define TILE_SIZE 16        // Optimal tile size for matrix operations
#define UNROLL_FACTOR 4     // Loop unrolling factor

// MARK: - Warp-Optimized Euclidean Distance

/// Euclidean distance with collaborative warp loading
/// Each warp loads and processes data cooperatively
kernel void euclideanDistanceWarpOptimized(
    constant float* queryVector [[buffer(0)]],
    constant float* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant int& vectorDimension [[buffer(3)]],
    threadgroup float* sharedQuery [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint warp_id [[simdgroup_index_in_threadgroup]]
) {
    // Collaborative loading of query vector to shared memory
    // Each thread loads multiple elements to reduce latency
    const int elementsPerThread = (vectorDimension + threadgroup_size - 1) / threadgroup_size;
    
    for (int i = 0; i < elementsPerThread; ++i) {
        int idx = tid + i * threadgroup_size;
        if (idx < vectorDimension) {
            sharedQuery[idx] = queryVector[idx];
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Each thread computes distance for one candidate
    if (gid >= vectorDimension) return;
    
    float sum = 0.0f;
    const int candidateOffset = gid * vectorDimension;
    
    // Process in chunks that fit in registers
    // Unroll by 4 for better instruction scheduling
    int i = 0;
    for (; i <= vectorDimension - 4; i += 4) {
        float4 q = float4(sharedQuery[i], sharedQuery[i+1], sharedQuery[i+2], sharedQuery[i+3]);
        float4 c = float4(candidateVectors[candidateOffset + i],
                         candidateVectors[candidateOffset + i + 1],
                         candidateVectors[candidateOffset + i + 2],
                         candidateVectors[candidateOffset + i + 3]);
        float4 diff = q - c;
        float4 sq = diff * diff;
        sum += sq.x + sq.y + sq.z + sq.w;
    }
    
    // Handle remainder
    for (; i < vectorDimension; ++i) {
        float diff = sharedQuery[i] - candidateVectors[candidateOffset + i];
        sum += diff * diff;
    }
    
    distances[gid] = sqrt(sum);
}

// MARK: - Warp-Reduced Cosine Distance

/// Cosine distance using warp-level reduction for dot products
kernel void cosineDistanceWarpReduced(
    constant float* queryVector [[buffer(0)]],
    constant float* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant int& vectorDimension [[buffer(3)]],
    threadgroup float* sharedData [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint warp_id [[simdgroup_index_in_threadgroup]],
    uint threadgroup_size [[threads_per_threadgroup]]
) {
    const int candidateIdx = gid;
    const int candidateOffset = candidateIdx * vectorDimension;
    
    // Each warp processes part of the vector
    const int warpsPerThreadgroup = threadgroup_size / WARP_SIZE;
    const int elementsPerWarp = (vectorDimension + warpsPerThreadgroup - 1) / warpsPerThreadgroup;
    const int warpStart = warp_id * elementsPerWarp;
    const int warpEnd = min(warpStart + elementsPerWarp, vectorDimension);
    
    // Warp-level computation
    float localDot = 0.0f;
    float localQueryMag = 0.0f;
    float localCandMag = 0.0f;
    
    // Each thread in warp processes strided elements
    for (int i = warpStart + lane_id; i < warpEnd; i += WARP_SIZE) {
        float q = queryVector[i];
        float c = candidateVectors[candidateOffset + i];
        
        localDot += q * c;
        localQueryMag += q * q;
        localCandMag += c * c;
    }
    
    // Warp-level reduction using shuffle operations
    localDot = simd_sum(localDot);
    localQueryMag = simd_sum(localQueryMag);
    localCandMag = simd_sum(localCandMag);
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        sharedData[warp_id * 3] = localDot;
        sharedData[warp_id * 3 + 1] = localQueryMag;
        sharedData[warp_id * 3 + 2] = localCandMag;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction by first warp
    if (warp_id == 0 && lane_id < warpsPerThreadgroup) {
        float dot = sharedData[lane_id * 3];
        float qMag = sharedData[lane_id * 3 + 1];
        float cMag = sharedData[lane_id * 3 + 2];
        
        dot = simd_sum(dot);
        qMag = simd_sum(qMag);
        cMag = simd_sum(cMag);
        
        if (lane_id == 0) {
            float similarity = dot / (sqrt(qMag) * sqrt(cMag) + 1e-8f);
            distances[candidateIdx] = 1.0f - similarity;
        }
    }
}

// MARK: - Tiled Matrix Distance Computation

/// Compute distances for multiple queries using tiled approach
/// Optimizes for cache locality and shared memory usage
kernel void tiledBatchDistance(
    constant float* queries [[buffer(0)]],
    constant float* candidates [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant int& vectorDimension [[buffer(3)]],
    constant int& numQueries [[buffer(4)]],
    constant int& numCandidates [[buffer(5)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint2 threadgroup_pos [[threadgroup_position_in_grid]]
) {
    // Each threadgroup computes a TILE_SIZE x TILE_SIZE block of the output
    const int queryIdx = threadgroup_pos.y * TILE_SIZE + tid.y;
    const int candidateIdx = threadgroup_pos.x * TILE_SIZE + tid.x;
    
    float sum = 0.0f;
    
    // Process vector in tiles to maximize cache reuse
    for (int k = 0; k < vectorDimension; k += TILE_SIZE) {
        // Collaborative loading of tiles
        if (queryIdx < numQueries && k + tid.x < vectorDimension) {
            tileA[tid.y * TILE_SIZE + tid.x] = queries[queryIdx * vectorDimension + k + tid.x];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }
        
        if (candidateIdx < numCandidates && k + tid.y < vectorDimension) {
            tileB[tid.y * TILE_SIZE + tid.x] = candidates[candidateIdx * vectorDimension + k + tid.y];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial distance for this tile
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            float diff = tileA[tid.y * TILE_SIZE + i] - tileB[i * TILE_SIZE + tid.x];
            sum += diff * diff;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write final result
    if (queryIdx < numQueries && candidateIdx < numCandidates) {
        distances[queryIdx * numCandidates + candidateIdx] = sqrt(sum);
    }
}

// MARK: - Adaptive Precision Distance

/// Distance computation with adaptive precision based on magnitude
/// Uses half precision for small values, full precision for large
kernel void adaptivePrecisionDistance(
    constant float* queryVector [[buffer(0)]],
    constant float* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant int& vectorDimension [[buffer(3)]],
    constant float& precisionThreshold [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const int candidateOffset = gid * vectorDimension;
    
    float sum_high = 0.0f;  // High precision accumulator
    half sum_low = 0.0h;    // Low precision accumulator
    
    for (int i = 0; i < vectorDimension; ++i) {
        float q = queryVector[i];
        float c = candidateVectors[candidateOffset + i];
        float diff = q - c;
        
        // Use half precision for small differences
        if (abs(diff) < precisionThreshold) {
            sum_low += half(diff) * half(diff);
        } else {
            sum_high += diff * diff;
        }
    }
    
    distances[gid] = sqrt(sum_high + float(sum_low));
}
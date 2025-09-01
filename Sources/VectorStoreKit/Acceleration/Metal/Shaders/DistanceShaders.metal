// VectorStoreKit: Distance Computation Shaders
//
// GPU kernels for distance metric calculations
// These shaders implement various distance metrics optimized for vector similarity search
// on Apple Silicon GPUs with unified memory architecture

#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>
using namespace metal;

// MARK: - Constants and Configuration

/// Function constants allow runtime specialization of shaders
constant uint THREADGROUP_SIZE [[function_constant(0)]];   // Optimal threadgroup size
constant bool USE_FAST_MATH [[function_constant(1)]];      // Trade accuracy for speed

// Hardware-specific constants tuned for Apple Silicon
#define SIMD_WIDTH 32              // Width of SIMD execution unit
#define CACHE_LINE_SIZE 128        // Size of cache line in bytes
#define BANK_CONFLICT_PADDING 1    // Padding to avoid shared memory bank conflicts

// MARK: - Helper Functions

/// Fast approximate square root using hardware accelerated function
inline float fastSqrt(float x) {
    return metal::fast::sqrt(x);
}

/// Warp-level parallel reduction using SIMD shuffle operations
inline float warpReduce(float val) {
    // Tree reduction: 32 -> 16 -> 8 -> 4 -> 2 -> 1
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

// MARK: - Euclidean Distance

/// Compute Euclidean (L2) distance between a query vector and multiple candidates
/// Euclidean distance is the straight-line distance between two points in space
/// Formula: d(p,q) = sqrt(Σ(p_i - q_i)²)
///
/// Use cases:
/// - General similarity search when vectors represent positions in space
/// - Image similarity, facial recognition
/// - When magnitude differences matter
///
/// Performance characteristics:
/// - Memory access: Sequential read pattern, cache-friendly
/// - Computation: O(d) operations per candidate where d = dimensions
/// - Parallelism: One thread per candidate, perfect for GPU
///
/// @param queryVector Single query vector to compare against all candidates
/// @param candidateVectors Flattened array of candidate vectors [num_candidates * dimensions]
/// @param distances Output array for computed distances
/// @param vectorDimension Number of dimensions per vector
/// @param id Thread index = candidate index
kernel void euclideanDistance(
    constant float* queryVector [[buffer(0)]],
    constant float* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float sum = 0.0;
    // Calculate starting position for this candidate in flattened array
    uint candidateOffset = id * vectorDimension;
    
    // Compute sum of squared differences
    // Unrolling this loop can improve performance for known dimensions
    for (uint i = 0; i < vectorDimension; ++i) {
        float diff = queryVector[i] - candidateVectors[candidateOffset + i];
        sum += diff * diff;
    }
    
    // Take square root for actual Euclidean distance
    // Note: For nearest neighbor search, you can skip sqrt and compare squared distances
    distances[id] = sqrt(sum);
}

// MARK: - Cosine Distance

/// Compute cosine distance (1 - cosine similarity) between vectors
/// Cosine similarity measures the angle between vectors, ignoring magnitude
/// Formula: similarity = (a·b) / (||a|| * ||b||), distance = 1 - similarity
///
/// Use cases:
/// - Text similarity (TF-IDF, word embeddings)
/// - Recommendation systems
/// - When direction matters more than magnitude
/// - Normalized embeddings from neural networks
///
/// Performance notes:
/// - Requires 3 passes: dot product + 2 magnitude calculations
/// - Can be optimized if vectors are pre-normalized
/// - The epsilon (1e-8) prevents division by zero
///
/// @param queryVector Query vector for comparison
/// @param candidateVectors Array of candidate vectors
/// @param distances Output cosine distances (0 = identical, 2 = opposite)
/// @param vectorDimension Dimensionality of vectors
/// @param id Thread/candidate index
kernel void cosineDistance(
    constant float* queryVector [[buffer(0)]],
    constant float* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float dotProduct = 0.0;
    float queryMagnitude = 0.0;
    float candidateMagnitude = 0.0;
    uint candidateOffset = id * vectorDimension;
    
    // Single pass to compute dot product and both magnitudes
    // This maximizes data reuse while values are in registers
    for (uint i = 0; i < vectorDimension; ++i) {
        float q = queryVector[i];
        float c = candidateVectors[candidateOffset + i];
        dotProduct += q * c;  // a·b
        queryMagnitude += q * q;  // ||a||²
        candidateMagnitude += c * c;  // ||b||²
    }
    
    // Compute cosine similarity with numerical stability
    // The epsilon prevents division by zero for zero vectors
    float similarity = dotProduct / (sqrt(queryMagnitude) * sqrt(candidateMagnitude) + 1e-8);
    
    // Convert to distance (0 = identical, 1 = orthogonal, 2 = opposite)
    distances[id] = 1.0 - similarity;
}

// MARK: - Manhattan Distance

/// Compute Manhattan (L1) distance between vectors
/// Manhattan distance is the sum of absolute differences across dimensions
/// Formula: d(p,q) = Σ|p_i - q_i|
///
/// Use cases:
/// - Grid-based pathfinding
/// - When movement is restricted to axes (like city blocks)
/// - Robust to outliers compared to Euclidean distance
/// - Sparse data where L1 norm is meaningful
///
/// Performance advantages:
/// - No multiplication or square root operations
/// - abs() is typically a single instruction
/// - Better numerical stability than Euclidean
///
/// @param queryVector Query vector for comparison
/// @param candidateVectors Array of candidate vectors
/// @param distances Output Manhattan distances
/// @param vectorDimension Number of dimensions
/// @param id Thread/candidate index
kernel void manhattanDistance(
    constant float* queryVector [[buffer(0)]],
    constant float* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float sum = 0.0;
    uint candidateOffset = id * vectorDimension;
    
    // Sum of absolute differences
    // abs() is typically implemented as a bit manipulation, very fast
    for (uint i = 0; i < vectorDimension; ++i) {
        sum += abs(queryVector[i] - candidateVectors[candidateOffset + i]);
    }
    
    distances[id] = sum;
}

// MARK: - Dot Product

/// Compute negative dot product as a distance measure
/// Dot product measures similarity; we negate it to create a distance metric
/// Formula: d(p,q) = -Σ(p_i * q_i)
///
/// Use cases:
/// - Maximum inner product search (MIPS)
/// - When vectors are normalized and dot product = cosine similarity
/// - Neural network embeddings optimized for dot product
/// - Retrieval systems with learned embeddings
///
/// Why negative:
/// - Larger dot product = more similar = smaller "distance"
/// - Negation makes it compatible with distance-based algorithms
///
/// @param queryVector Query vector
/// @param candidateVectors Candidate vectors array
/// @param distances Output negative dot products
/// @param vectorDimension Vector dimensionality
/// @param id Thread/candidate index
kernel void dotProduct(
    constant float* queryVector [[buffer(0)]],
    constant float* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float product = 0.0;
    uint candidateOffset = id * vectorDimension;
    
    // Standard dot product computation
    // Can benefit from FMA (fused multiply-add) instructions
    for (uint i = 0; i < vectorDimension; ++i) {
        product += queryVector[i] * candidateVectors[candidateOffset + i];
    }
    
    // Negative dot product for distance (higher dot product = smaller distance)
    // This allows nearest neighbor algorithms to find most similar vectors
    distances[id] = -product;
}

// MARK: - Batch Distance Computation

/// Compute Euclidean distances between multiple query and candidate vectors
/// This kernel efficiently processes a matrix of pairwise distances
///
/// Output matrix layout:
/// - Row i = distances from query i to all candidates
/// - Column j = distances from all queries to candidate j
/// - distances[i][j] = distance from query i to candidate j
///
/// Thread organization:
/// - 2D grid where each thread computes one distance
/// - x dimension = candidate index
/// - y dimension = query index
/// - Enables massive parallelism for batch processing
///
/// Performance optimization opportunities:
/// - Tile the computation for better cache usage
/// - Use shared memory for frequently accessed data
/// - Consider warp-level primitives for reductions
///
/// @param queries Array of query vectors [numQueries * vectorDimension]
/// @param candidates Array of candidate vectors [numCandidates * vectorDimension]
/// @param distances Output distance matrix [numQueries * numCandidates]
/// @param vectorDimension Dimensionality of vectors
/// @param numQueries Number of query vectors
/// @param numCandidates Number of candidate vectors
/// @param id 2D thread position (x: candidate, y: query)
kernel void batchEuclideanDistance(
    constant float* queries [[buffer(0)]],          // Multiple query vectors
    constant float* candidates [[buffer(1)]],       // Candidate vectors
    device float* distances [[buffer(2)]],          // Output distances matrix
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numQueries [[buffer(4)]],
    constant uint& numCandidates [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]           // x: candidate index, y: query index
) {
    // Bounds check for 2D grid
    if (id.x >= numCandidates || id.y >= numQueries) return;
    
    float sum = 0.0;
    // Calculate offsets for this query-candidate pair
    uint queryOffset = id.y * vectorDimension;
    uint candidateOffset = id.x * vectorDimension;
    
    // Compute squared Euclidean distance
    for (uint i = 0; i < vectorDimension; ++i) {
        float diff = queries[queryOffset + i] - candidates[candidateOffset + i];
        sum += diff * diff;
    }
    
    // Store in row-major order: distances[query][candidate]
    distances[id.y * numCandidates + id.x] = sqrt(sum);
}

// MARK: - Batch Cosine Distance

/// Batch computation of cosine distances between multiple query-candidate pairs
/// Processes an entire distance matrix in parallel
///
/// Optimization note:
/// If either queries or candidates are pre-normalized, you can:
/// 1. Skip magnitude computation for normalized set
/// 2. Use simplified formula: distance = 1 - dot_product
/// 3. Consider caching magnitudes if vectors are reused
///
/// Memory access pattern:
/// - Each thread reads one query and one candidate vector
/// - No data sharing between threads
/// - Coalesced memory access when threads in warp access consecutive candidates
///
/// @param queries Query vector array
/// @param candidates Candidate vector array
/// @param distances Output distance matrix
/// @param vectorDimension Vector dimensionality
/// @param numQueries Number of queries
/// @param numCandidates Number of candidates
/// @param id 2D thread index
kernel void batchCosineDistance(
    constant float* queries [[buffer(0)]],
    constant float* candidates [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numQueries [[buffer(4)]],
    constant uint& numCandidates [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    if (id.x >= numCandidates || id.y >= numQueries) return;
    
    float dotProduct = 0.0;
    float queryMagnitude = 0.0;
    float candidateMagnitude = 0.0;
    
    uint queryOffset = id.y * vectorDimension;
    uint candidateOffset = id.x * vectorDimension;
    
    // Single pass computation for efficiency
    for (uint i = 0; i < vectorDimension; ++i) {
        float q = queries[queryOffset + i];
        float c = candidates[candidateOffset + i];
        dotProduct += q * c;
        queryMagnitude += q * q;
        candidateMagnitude += c * c;
    }
    
    // Cosine similarity with numerical stability
    float similarity = dotProduct / (sqrt(queryMagnitude) * sqrt(candidateMagnitude) + 1e-8);
    
    // Convert to distance and store in matrix
    distances[id.y * numCandidates + id.x] = 1.0 - similarity;
}

// MARK: - Batch Manhattan Distance

kernel void batchManhattanDistance(
    constant float* queries [[buffer(0)]],
    constant float* candidates [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numQueries [[buffer(4)]],
    constant uint& numCandidates [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    if (id.x >= numCandidates || id.y >= numQueries) return;
    
    float sum = 0.0;
    uint queryOffset = id.y * vectorDimension;
    uint candidateOffset = id.x * vectorDimension;
    
    for (uint i = 0; i < vectorDimension; ++i) {
        sum += abs(queries[queryOffset + i] - candidates[candidateOffset + i]);
    }
    
    distances[id.y * numCandidates + id.x] = sum;
}

// MARK: - Batch Dot Product

kernel void batchDotProduct(
    constant float* queries [[buffer(0)]],
    constant float* candidates [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numQueries [[buffer(4)]],
    constant uint& numCandidates [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    if (id.x >= numCandidates || id.y >= numQueries) return;
    
    float product = 0.0;
    uint queryOffset = id.y * vectorDimension;
    uint candidateOffset = id.x * vectorDimension;
    
    for (uint i = 0; i < vectorDimension; ++i) {
        product += queries[queryOffset + i] * candidates[candidateOffset + i];
    }
    
    // Negative dot product for distance (higher dot product = smaller distance)
    distances[id.y * numCandidates + id.x] = -product;
}

// MARK: - Normalized Vector Operations

/// Normalize vectors to unit length (L2 normalization)
/// This preprocessing step is crucial for cosine similarity calculations
/// Normalized vectors allow dot product to equal cosine similarity
///
/// Benefits of normalization:
/// - Simplifies cosine similarity to dot product
/// - Improves numerical stability
/// - Required for some embedding models (e.g., face recognition)
/// - Enables faster similarity search
///
/// Implementation notes:
/// - In-place normalization saves memory
/// - Epsilon (1e-8) prevents division by zero for null vectors
/// - Two-pass approach: first compute magnitude, then normalize
/// - Could be optimized with reciprocal square root approximation
///
/// @param vectors Array of vectors to normalize in-place
/// @param vectorDimension Dimensionality of each vector
/// @param numVectors Total number of vectors to process
/// @param id Thread index = vector index
kernel void normalizeVectors(
    device float* vectors [[buffer(0)]],
    constant uint& vectorDimension [[buffer(1)]],
    constant uint& numVectors [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numVectors) return;
    
    uint offset = id * vectorDimension;
    float magnitude = 0.0;
    
    // First pass: Calculate L2 norm (magnitude)
    for (uint i = 0; i < vectorDimension; ++i) {
        float val = vectors[offset + i];
        magnitude += val * val;
    }
    
    // Add epsilon to handle zero vectors gracefully
    magnitude = sqrt(magnitude) + 1e-8;
    
    // Second pass: Normalize each component
    // Division is more expensive than multiplication, so consider
    // computing reciprocal once: float inv_magnitude = 1.0 / magnitude;
    for (uint i = 0; i < vectorDimension; ++i) {
        vectors[offset + i] /= magnitude;
    }
}

// MARK: - Enhanced Euclidean Distance with Warp Primitives

/// Advanced Euclidean distance computation using warp-level primitives
/// This kernel maximizes performance through multiple optimization techniques
kernel void euclideanDistanceWarpOptimized(
    constant float4* queryVector [[buffer(0)]],
    constant float4* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numCandidates [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (gid >= numCandidates) return;
    
    const uint float4Dimension = (vectorDimension + 3) / 4;
    
    // Allocate threadgroup memory with padding to avoid bank conflicts
    threadgroup float4 sharedQuery[128 + BANK_CONFLICT_PADDING];
    
    // Cooperative loading: threads work together to load query vector
    for (uint i = tid; i < float4Dimension; i += THREADGROUP_SIZE) {
        sharedQuery[i] = queryVector[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    constant float4* candidate = candidateVectors + (gid * float4Dimension);
    
    // Use multiple accumulators to hide FPU latency
    float4 sum0 = float4(0.0);
    float4 sum1 = float4(0.0);
    float4 sum2 = float4(0.0);
    float4 sum3 = float4(0.0);
    
    // Main computation loop - unrolled for maximum performance
    uint i = 0;
    for (; i + 7 < float4Dimension; i += 8) {
        // Prefetch next cache line to hide memory latency
        __builtin_prefetch(&candidate[i + 8], 0, 0);
        
        float4 diff0 = sharedQuery[i] - candidate[i];
        float4 diff1 = sharedQuery[i+1] - candidate[i+1];
        float4 diff2 = sharedQuery[i+2] - candidate[i+2];
        float4 diff3 = sharedQuery[i+3] - candidate[i+3];
        float4 diff4 = sharedQuery[i+4] - candidate[i+4];
        float4 diff5 = sharedQuery[i+5] - candidate[i+5];
        float4 diff6 = sharedQuery[i+6] - candidate[i+6];
        float4 diff7 = sharedQuery[i+7] - candidate[i+7];
        
        // Use FMA for better accuracy and performance
        sum0 = metal::fma(diff0, diff0, sum0);
        sum1 = metal::fma(diff1, diff1, sum1);
        sum2 = metal::fma(diff2, diff2, sum2);
        sum3 = metal::fma(diff3, diff3, sum3);
        sum0 = metal::fma(diff4, diff4, sum0);
        sum1 = metal::fma(diff5, diff5, sum1);
        sum2 = metal::fma(diff6, diff6, sum2);
        sum3 = metal::fma(diff7, diff7, sum3);
    }
    
    // Handle remaining elements
    for (; i < float4Dimension; ++i) {
        float4 diff = sharedQuery[i] - candidate[i];
        sum0 = metal::fma(diff, diff, sum0);
    }
    
    float4 sum = sum0 + sum1 + sum2 + sum3;
    float distance = sum.x + sum.y + sum.z + sum.w;
    
    // Handle partial last float4 if dimension not divisible by 4
    if (vectorDimension % 4 != 0) {
        uint lastIdx = float4Dimension - 1;
        float4 lastDiff = sharedQuery[lastIdx] - candidate[lastIdx];
        float4 squared = lastDiff * lastDiff;
        
        uint remainder = vectorDimension % 4;
        if (remainder == 1) distance -= squared.y + squared.z + squared.w;
        else if (remainder == 2) distance -= squared.z + squared.w;
        else if (remainder == 3) distance -= squared.w;
    }
    
    // Use warp-level reduction for additional optimization
    distance = warpReduce(distance);
    
    // Write result - only one thread per SIMD group needs to write
    if (simd_lane == 0 || SIMD_WIDTH == 1) {
        distances[gid] = USE_FAST_MATH ? fastSqrt(distance) : sqrt(distance);
    }
}

// MARK: - Half Precision Distance Computation

/// Euclidean distance using 16-bit half precision floating point
kernel void euclideanDistanceHalf(
    constant half4* queryVector [[buffer(0)]],
    constant half4* candidateVectors [[buffer(1)]],
    device half* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numCandidates [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numCandidates) return;
    
    const uint half4Dimension = (vectorDimension + 3) / 4;
    constant half4* candidate = candidateVectors + (gid * half4Dimension);
    
    half4 sum = half4(0.0h);
    
    for (uint i = 0; i < half4Dimension; ++i) {
        half4 diff = queryVector[i] - candidate[i];
        sum = metal::fma(diff, diff, sum);
    }
    
    half distance = sum.x + sum.y + sum.z + sum.w;
    
    if (vectorDimension % 4 != 0) {
        uint lastIdx = half4Dimension - 1;
        half4 lastDiff = queryVector[lastIdx] - candidate[lastIdx];
        half4 squared = lastDiff * lastDiff;
        
        uint remainder = vectorDimension % 4;
        if (remainder == 1) distance -= squared.y + squared.z + squared.w;
        else if (remainder == 2) distance -= squared.z + squared.w;
        else if (remainder == 3) distance -= squared.w;
    }
    
    distances[gid] = sqrt(distance);
}

// MARK: - Tiled Matrix Distance Computation

/// Tiled distance computation for batch processing with shared memory optimization
[[max_total_threads_per_threadgroup(256)]]
kernel void tiledBatchDistance(
    constant float4* queries [[buffer(0)]],
    constant float4* candidates [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numQueries [[buffer(4)]],
    constant uint& numCandidates [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgSize [[threads_per_threadgroup]]
) {
    const uint queryIdx = gid.y;
    const uint candidateIdx = gid.x;
    
    if (queryIdx >= numQueries || candidateIdx >= numCandidates) return;
    
    const uint float4Dimension = (vectorDimension + 3) / 4;
    const uint TILE_SIZE = 32;
    
    threadgroup float4 tileQueries[32][32 + BANK_CONFLICT_PADDING];
    threadgroup float4 tileCandidates[32][32 + BANK_CONFLICT_PADDING];
    
    float4 accumulator = float4(0.0);
    
    for (uint tileStart = 0; tileStart < float4Dimension; tileStart += TILE_SIZE) {
        uint tileEnd = min(tileStart + TILE_SIZE, float4Dimension);
        uint tileSize = tileEnd - tileStart;
        
        if (tid.x < tileSize && tid.y == 0) {
            tileQueries[0][tid.x] = queries[queryIdx * float4Dimension + tileStart + tid.x];
        }
        
        if (tid.y < tileSize && tid.x == 0) {
            tileCandidates[tid.y][0] = candidates[candidateIdx * float4Dimension + tileStart + tid.y];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint i = 0; i < tileSize; ++i) {
            float4 diff = tileQueries[0][i] - tileCandidates[i][0];
            accumulator = metal::fma(diff, diff, accumulator);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float distance = accumulator.x + accumulator.y + accumulator.z + accumulator.w;
    distances[queryIdx * numCandidates + candidateIdx] = sqrt(distance);
}

// MARK: - Fast Approximate Distance (No Square Root)

/// Compute squared Euclidean distance without final square root
kernel void euclideanDistanceSquaredFast(
    constant float4* queryVector [[buffer(0)]],
    constant float4* candidateVectors [[buffer(1)]],
    device float* distancesSquared [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numCandidates [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= numCandidates) return;
    
    const uint float4Dimension = (vectorDimension + 3) / 4;
    constant float4* candidate = candidateVectors + (gid * float4Dimension);
    
    float4 sum = float4(0.0);
    
    for (uint i = 0; i < float4Dimension; ++i) {
        float4 diff = queryVector[i] - candidate[i];
        sum = metal::fma(diff, diff, sum);
    }
    
    float distanceSquared = sum.x + sum.y + sum.z + sum.w;
    
    if (vectorDimension % 4 != 0) {
        uint lastIdx = float4Dimension - 1;
        float4 lastDiff = queryVector[lastIdx] - candidate[lastIdx];
        float4 squared = lastDiff * lastDiff;
        
        uint remainder = vectorDimension % 4;
        if (remainder == 1) distanceSquared -= squared.y + squared.z + squared.w;
        else if (remainder == 2) distanceSquared -= squared.z + squared.w;
        else if (remainder == 3) distanceSquared -= squared.w;
    }
    
    distancesSquared[gid] = distanceSquared;
}

// MARK: - Optimized 512-Dimensional Distance Kernels

/// Highly optimized Euclidean distance for 512-dimensional vectors
/// This kernel is specifically tuned for 512-dim embeddings common in ML models
/// (e.g., face recognition, text embeddings, image features)
///
/// Optimization techniques:
/// 1. SIMD vectorization: Uses float4 to process 4 dimensions simultaneously
/// 2. Loop unrolling: Processes 8 float4s (32 floats) per iteration
/// 3. Instruction-level parallelism: Multiple independent operations
/// 4. Register blocking: Maximizes register usage for reduced memory traffic
///
/// Performance characteristics:
/// - 128 float4 operations = 512 scalar operations
/// - Unrolling factor of 8 balances register pressure and ILP
/// - Expected 4x speedup over scalar implementation
/// - Memory bandwidth limited on large datasets
///
/// Why 512 dimensions?
/// - Common output size for many neural networks
/// - Balances expressiveness with computational efficiency
/// - Fits well in GPU cache hierarchies
///
/// @param queryVector Query as 128 float4 vectors
/// @param candidateVectors Candidates as arrays of 128 float4s each
/// @param distances Output distance array
/// @param id Thread/candidate index
kernel void euclideanDistance512_simd(
    constant float4* queryVector [[buffer(0)]],      // 128 float4s = 512 floats
    constant float4* candidateVectors [[buffer(1)]], // Multiple candidates
    device float* distances [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float4 sum = float4(0.0);
    constant float4* candidate = candidateVectors + (id * 128);
    
    // Unroll loop for 512 dimensions (128 float4 operations)
    // Process 8 float4s at a time for better pipelining
    // The pragma unroll hints the compiler to unroll 16 iterations (128/8)
    #pragma unroll(16)
    for (int i = 0; i < 128; i += 8) {
        // Load 8 float4s (32 floats) and compute differences
        // This exploits instruction-level parallelism
        float4 diff0 = queryVector[i] - candidate[i];
        float4 diff1 = queryVector[i+1] - candidate[i+1];
        float4 diff2 = queryVector[i+2] - candidate[i+2];
        float4 diff3 = queryVector[i+3] - candidate[i+3];
        float4 diff4 = queryVector[i+4] - candidate[i+4];
        float4 diff5 = queryVector[i+5] - candidate[i+5];
        float4 diff6 = queryVector[i+6] - candidate[i+6];
        float4 diff7 = queryVector[i+7] - candidate[i+7];
        
        // Accumulate squared differences
        // Compiler can schedule these operations in parallel
        sum += diff0 * diff0;
        sum += diff1 * diff1;
        sum += diff2 * diff2;
        sum += diff3 * diff3;
        sum += diff4 * diff4;
        sum += diff5 * diff5;
        sum += diff6 * diff6;
        sum += diff7 * diff7;
    }
    
    // Horizontal sum across float4 components and final sqrt
    distances[id] = sqrt(sum.x + sum.y + sum.z + sum.w);
}

kernel void cosineDistance512_simd(
    constant float4* queryVector [[buffer(0)]],
    constant float4* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float4 dotProduct = float4(0.0);
    float4 queryMag = float4(0.0);
    float4 candidateMag = float4(0.0);
    constant float4* candidate = candidateVectors + (id * 128);
    
    #pragma unroll(16)
    for (int i = 0; i < 128; i += 8) {
        float4 q0 = queryVector[i];
        float4 q1 = queryVector[i+1];
        float4 q2 = queryVector[i+2];
        float4 q3 = queryVector[i+3];
        float4 q4 = queryVector[i+4];
        float4 q5 = queryVector[i+5];
        float4 q6 = queryVector[i+6];
        float4 q7 = queryVector[i+7];
        
        float4 c0 = candidate[i];
        float4 c1 = candidate[i+1];
        float4 c2 = candidate[i+2];
        float4 c3 = candidate[i+3];
        float4 c4 = candidate[i+4];
        float4 c5 = candidate[i+5];
        float4 c6 = candidate[i+6];
        float4 c7 = candidate[i+7];
        
        dotProduct += q0 * c0;
        dotProduct += q1 * c1;
        dotProduct += q2 * c2;
        dotProduct += q3 * c3;
        dotProduct += q4 * c4;
        dotProduct += q5 * c5;
        dotProduct += q6 * c6;
        dotProduct += q7 * c7;
        
        queryMag += q0 * q0;
        queryMag += q1 * q1;
        queryMag += q2 * q2;
        queryMag += q3 * q3;
        queryMag += q4 * q4;
        queryMag += q5 * q5;
        queryMag += q6 * q6;
        queryMag += q7 * q7;
        
        candidateMag += c0 * c0;
        candidateMag += c1 * c1;
        candidateMag += c2 * c2;
        candidateMag += c3 * c3;
        candidateMag += c4 * c4;
        candidateMag += c5 * c5;
        candidateMag += c6 * c6;
        candidateMag += c7 * c7;
    }
    
    float dp = dotProduct.x + dotProduct.y + dotProduct.z + dotProduct.w;
    float qm = sqrt(queryMag.x + queryMag.y + queryMag.z + queryMag.w);
    float cm = sqrt(candidateMag.x + candidateMag.y + candidateMag.z + candidateMag.w);
    
    float similarity = dp / (qm * cm + 1e-8);
    distances[id] = 1.0 - similarity;
}

/// Optimized cosine distance for pre-normalized 512-dimensional vectors
/// When vectors are already normalized, cosine similarity = dot product
/// This eliminates expensive magnitude calculations
///
/// Use cases:
/// - Face recognition embeddings (usually L2-normalized)
/// - Sentence embeddings from transformers
/// - Any system where vectors are pre-normalized
///
/// Performance benefits vs full cosine:
/// - 3x fewer operations (no magnitude calculation)
/// - No square root operations
/// - Single pass through data
/// - Better cache utilization
///
/// Accuracy notes:
/// - Assumes ||query|| = ||candidate|| = 1
/// - If not normalized, results will be incorrect
/// - Consider periodic re-normalization for numerical stability
///
/// @param queryVector Pre-normalized query vector
/// @param candidateVectors Pre-normalized candidate vectors
/// @param distances Output distances (0 = identical, 2 = opposite)
/// @param id Thread/candidate index
kernel void cosineDistance512_normalized(
    constant float4* queryVector [[buffer(0)]],
    constant float4* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float4 dotProduct = float4(0.0);
    constant float4* candidate = candidateVectors + (id * 128);
    
    // Compute dot product with aggressive unrolling
    // For normalized vectors: cos(θ) = a·b
    #pragma unroll(16)
    for (int i = 0; i < 128; i += 8) {
        // Accumulate 8 float4 dot products
        // Each operation computes 4 scalar products in parallel
        dotProduct += queryVector[i] * candidate[i];
        dotProduct += queryVector[i+1] * candidate[i+1];
        dotProduct += queryVector[i+2] * candidate[i+2];
        dotProduct += queryVector[i+3] * candidate[i+3];
        dotProduct += queryVector[i+4] * candidate[i+4];
        dotProduct += queryVector[i+5] * candidate[i+5];
        dotProduct += queryVector[i+6] * candidate[i+6];
        dotProduct += queryVector[i+7] * candidate[i+7];
    }
    
    // Horizontal sum to get final dot product
    float similarity = dotProduct.x + dotProduct.y + dotProduct.z + dotProduct.w;
    
    // Convert similarity to distance
    distances[id] = 1.0 - similarity;
}

/// Batch Euclidean distance computation for 512-dimensional vectors
/// Computes full distance matrix between query and candidate sets
///
/// Matrix computation benefits:
/// - Amortizes kernel launch overhead
/// - Better GPU utilization with 2D parallelism
/// - Enables batch processing in retrieval systems
///
/// Memory layout considerations:
/// - Row-major storage: distances[query_idx][candidate_idx]
/// - Coalesced access when threads in same warp access consecutive candidates
/// - Consider tiling for very large matrices to improve cache usage
///
/// Scalability:
/// - O(numQueries * numCandidates) threads
/// - Memory requirement: 4 * numQueries * numCandidates bytes
/// - Consider chunking for very large datasets
///
/// @param queries Array of 512-dim query vectors
/// @param candidates Array of 512-dim candidate vectors
/// @param distances Output matrix [numQueries x numCandidates]
/// @param numQueries Number of query vectors
/// @param numCandidates Number of candidate vectors
/// @param id 2D thread index (x: candidate, y: query)
kernel void batchEuclideanDistance512_simd(
    constant float4* queries [[buffer(0)]],          // Multiple 512-dim queries
    constant float4* candidates [[buffer(1)]],       // Multiple 512-dim candidates
    device float* distances [[buffer(2)]],           // Output distances matrix
    constant uint& numQueries [[buffer(3)]],
    constant uint& numCandidates [[buffer(4)]],
    uint2 id [[thread_position_in_grid]]            // x: candidate index, y: query index
) {
    // Bounds check for 2D grid
    if (id.x >= numCandidates || id.y >= numQueries) return;
    
    float4 sum = float4(0.0);
    // Each vector is 128 float4s (512 floats)
    constant float4* query = queries + (id.y * 128);
    constant float4* candidate = candidates + (id.x * 128);
    
    // Optimized distance computation with unrolling
    #pragma unroll(16)
    for (int i = 0; i < 128; i += 8) {
        // Process 32 dimensions per iteration
        float4 diff0 = query[i] - candidate[i];
        float4 diff1 = query[i+1] - candidate[i+1];
        float4 diff2 = query[i+2] - candidate[i+2];
        float4 diff3 = query[i+3] - candidate[i+3];
        float4 diff4 = query[i+4] - candidate[i+4];
        float4 diff5 = query[i+5] - candidate[i+5];
        float4 diff6 = query[i+6] - candidate[i+6];
        float4 diff7 = query[i+7] - candidate[i+7];
        
        // Accumulate squared differences
        sum += diff0 * diff0;
        sum += diff1 * diff1;
        sum += diff2 * diff2;
        sum += diff3 * diff3;
        sum += diff4 * diff4;
        sum += diff5 * diff5;
        sum += diff6 * diff6;
        sum += diff7 * diff7;
    }
    
    // Store result in row-major order
    distances[id.y * numCandidates + id.x] = sqrt(sum.x + sum.y + sum.z + sum.w);
}
// VectorStoreKit: Distance Computation Shaders
//
// GPU kernels for distance metric calculations

#include <metal_stdlib>
#include <metal_math>
using namespace metal;

// MARK: - Euclidean Distance

kernel void euclideanDistance(
    constant float* queryVector [[buffer(0)]],
    constant float* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float sum = 0.0;
    uint candidateOffset = id * vectorDimension;
    
    for (uint i = 0; i < vectorDimension; ++i) {
        float diff = queryVector[i] - candidateVectors[candidateOffset + i];
        sum += diff * diff;
    }
    
    distances[id] = sqrt(sum);
}

// MARK: - Cosine Distance

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
    
    for (uint i = 0; i < vectorDimension; ++i) {
        float q = queryVector[i];
        float c = candidateVectors[candidateOffset + i];
        dotProduct += q * c;
        queryMagnitude += q * q;
        candidateMagnitude += c * c;
    }
    
    float similarity = dotProduct / (sqrt(queryMagnitude) * sqrt(candidateMagnitude) + 1e-8);
    distances[id] = 1.0 - similarity;
}

// MARK: - Manhattan Distance

kernel void manhattanDistance(
    constant float* queryVector [[buffer(0)]],
    constant float* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float sum = 0.0;
    uint candidateOffset = id * vectorDimension;
    
    for (uint i = 0; i < vectorDimension; ++i) {
        sum += abs(queryVector[i] - candidateVectors[candidateOffset + i]);
    }
    
    distances[id] = sum;
}

// MARK: - Dot Product

kernel void dotProduct(
    constant float* queryVector [[buffer(0)]],
    constant float* candidateVectors [[buffer(1)]],
    device float* distances [[buffer(2)]],
    constant uint& vectorDimension [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float product = 0.0;
    uint candidateOffset = id * vectorDimension;
    
    for (uint i = 0; i < vectorDimension; ++i) {
        product += queryVector[i] * candidateVectors[candidateOffset + i];
    }
    
    // Negative dot product for distance (higher dot product = smaller distance)
    distances[id] = -product;
}

// MARK: - Batch Distance Computation

kernel void batchEuclideanDistance(
    constant float* queries [[buffer(0)]],          // Multiple query vectors
    constant float* candidates [[buffer(1)]],       // Candidate vectors
    device float* distances [[buffer(2)]],          // Output distances matrix
    constant uint& vectorDimension [[buffer(3)]],
    constant uint& numQueries [[buffer(4)]],
    constant uint& numCandidates [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]           // x: candidate index, y: query index
) {
    if (id.x >= numCandidates || id.y >= numQueries) return;
    
    float sum = 0.0;
    uint queryOffset = id.y * vectorDimension;
    uint candidateOffset = id.x * vectorDimension;
    
    for (uint i = 0; i < vectorDimension; ++i) {
        float diff = queries[queryOffset + i] - candidates[candidateOffset + i];
        sum += diff * diff;
    }
    
    distances[id.y * numCandidates + id.x] = sqrt(sum);
}

// MARK: - Batch Cosine Distance

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
    
    for (uint i = 0; i < vectorDimension; ++i) {
        float q = queries[queryOffset + i];
        float c = candidates[candidateOffset + i];
        dotProduct += q * c;
        queryMagnitude += q * q;
        candidateMagnitude += c * c;
    }
    
    float similarity = dotProduct / (sqrt(queryMagnitude) * sqrt(candidateMagnitude) + 1e-8);
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

kernel void normalizeVectors(
    device float* vectors [[buffer(0)]],
    constant uint& vectorDimension [[buffer(1)]],
    constant uint& numVectors [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= numVectors) return;
    
    uint offset = id * vectorDimension;
    float magnitude = 0.0;
    
    // Calculate magnitude
    for (uint i = 0; i < vectorDimension; ++i) {
        float val = vectors[offset + i];
        magnitude += val * val;
    }
    
    magnitude = sqrt(magnitude) + 1e-8; // Avoid division by zero
    
    // Normalize
    for (uint i = 0; i < vectorDimension; ++i) {
        vectors[offset + i] /= magnitude;
    }
}
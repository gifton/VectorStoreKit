// VectorStoreKit: Clustering Shaders
//
// Metal shaders for K-means clustering operations

#include <metal_stdlib>
using namespace metal;

// MARK: - Centroid Assignment

/// Assign each vector to its nearest centroid
kernel void assign_to_centroids(
    constant float* vectors [[buffer(0)]],           // Input vectors (flattened)
    constant float* centroids [[buffer(1)]],         // Centroids (flattened)
    device uint* assignments [[buffer(2)]],          // Output assignments
    device float* distances [[buffer(3)]],           // Output distances
    constant uint& num_vectors [[buffer(4)]],        // Number of vectors
    constant uint& num_centroids [[buffer(5)]],      // Number of centroids (k)
    constant uint& dimensions [[buffer(6)]],         // Vector dimensions
    uint id [[thread_position_in_grid]])
{
    if (id >= num_vectors) return;
    
    uint vector_offset = id * dimensions;
    float min_distance = INFINITY;
    uint min_centroid = 0;
    
    // Find nearest centroid
    for (uint c = 0; c < num_centroids; c++) {
        uint centroid_offset = c * dimensions;
        float distance = 0.0f;
        
        // Compute squared Euclidean distance
        for (uint d = 0; d < dimensions; d++) {
            float diff = vectors[vector_offset + d] - centroids[centroid_offset + d];
            distance += diff * diff;
        }
        
        if (distance < min_distance) {
            min_distance = distance;
            min_centroid = c;
        }
    }
    
    assignments[id] = min_centroid;
    distances[id] = sqrt(min_distance);
}

// MARK: - Centroid Update

/// Update centroids based on assignments (step 1: accumulate)
kernel void accumulate_centroids(
    constant float* vectors [[buffer(0)]],           // Input vectors
    constant uint* assignments [[buffer(1)]],        // Vector assignments
    device atomic_float* centroid_sums [[buffer(2)]],// Accumulated sums (atomic)
    device atomic_uint* centroid_counts [[buffer(3)]],// Counts per centroid (atomic)
    constant uint& dimensions [[buffer(4)]],         // Vector dimensions
    uint id [[thread_position_in_grid]])
{
    uint vector_offset = id * dimensions;
    uint assignment = assignments[id];
    uint centroid_offset = assignment * dimensions;
    
    // Accumulate vector into assigned centroid
    for (uint d = 0; d < dimensions; d++) {
        atomic_fetch_add_explicit(
            &centroid_sums[centroid_offset + d],
            vectors[vector_offset + d],
            memory_order_relaxed
        );
    }
    
    // Increment count
    atomic_fetch_add_explicit(
        &centroid_counts[assignment],
        1,
        memory_order_relaxed
    );
}

/// Update centroids based on assignments (step 2: divide)
kernel void finalize_centroids(
    device float* centroid_sums [[buffer(0)]],      // Accumulated sums
    constant uint* centroid_counts [[buffer(1)]],   // Counts per centroid
    device float* centroids [[buffer(2)]],          // Output centroids
    constant uint& num_centroids [[buffer(3)]],     // Number of centroids
    constant uint& dimensions [[buffer(4)]],        // Vector dimensions
    uint2 id [[thread_position_in_grid]])
{
    uint c = id.x;  // Centroid index
    uint d = id.y;  // Dimension index
    
    if (c >= num_centroids || d >= dimensions) return;
    
    uint idx = c * dimensions + d;
    uint count = centroid_counts[c];
    
    if (count > 0) {
        centroids[idx] = centroid_sums[idx] / float(count);
    } else {
        // Keep existing centroid if no points assigned
        // (centroids buffer should be pre-populated)
    }
}

// MARK: - K-means++ Initialization

/// Compute minimum distances to existing centroids
kernel void compute_min_distances(
    constant float* vectors [[buffer(0)]],           // Input vectors
    constant float* centroids [[buffer(1)]],         // Current centroids
    device float* min_distances [[buffer(2)]],       // Output min distances
    constant uint& num_vectors [[buffer(3)]],        // Number of vectors
    constant uint& num_centroids [[buffer(4)]],      // Current number of centroids
    constant uint& dimensions [[buffer(5)]],         // Vector dimensions
    uint id [[thread_position_in_grid]])
{
    if (id >= num_vectors) return;
    
    uint vector_offset = id * dimensions;
    float min_distance = INFINITY;
    
    // Find minimum distance to any centroid
    for (uint c = 0; c < num_centroids; c++) {
        uint centroid_offset = c * dimensions;
        float distance = 0.0f;
        
        for (uint d = 0; d < dimensions; d++) {
            float diff = vectors[vector_offset + d] - centroids[centroid_offset + d];
            distance += diff * diff;
        }
        
        min_distance = min(min_distance, distance);
    }
    
    min_distances[id] = min_distance;
}

// MARK: - Mini-batch K-means

/// Perform incremental centroid update for mini-batch K-means
kernel void update_centroids_incremental(
    device float* centroids [[buffer(0)]],          // Centroids to update
    constant float* batch_vectors [[buffer(1)]],    // Mini-batch vectors
    constant uint* batch_assignments [[buffer(2)]],  // Assignments for batch
    constant float& learning_rate [[buffer(3)]],    // Learning rate
    constant uint& batch_size [[buffer(4)]],        // Batch size
    constant uint& dimensions [[buffer(5)]],        // Vector dimensions
    uint2 id [[thread_position_in_grid]])
{
    uint batch_idx = id.x;
    uint dim = id.y;
    
    if (batch_idx >= batch_size || dim >= dimensions) return;
    
    uint assignment = batch_assignments[batch_idx];
    float vector_value = batch_vectors[batch_idx * dimensions + dim];
    
    // Incremental update: c = c + lr * (x - c)
    uint centroid_idx = assignment * dimensions + dim;
    float old_value = centroids[centroid_idx];
    float new_value = old_value + learning_rate * (vector_value - old_value);
    
    centroids[centroid_idx] = new_value;
}

// MARK: - Utility Functions

/// Compute inertia (sum of squared distances to assigned centroids)
kernel void compute_inertia(
    constant float* distances [[buffer(0)]],         // Distances to assigned centroids
    device atomic_float* inertia [[buffer(1)]],     // Output inertia (atomic)
    constant uint& num_vectors [[buffer(2)]],        // Number of vectors
    uint id [[thread_position_in_grid]])
{
    if (id >= num_vectors) return;
    
    float squared_distance = distances[id] * distances[id];
    atomic_fetch_add_explicit(inertia, squared_distance, memory_order_relaxed);
}

/// Initialize atomic buffers to zero
kernel void clear_atomic_buffers(
    device atomic_float* buffer [[buffer(0)]],      // Buffer to clear
    constant uint& size [[buffer(1)]],              // Buffer size
    uint id [[thread_position_in_grid]])
{
    if (id >= size) return;
    atomic_store_explicit(&buffer[id], 0.0f, memory_order_relaxed);
}

kernel void clear_atomic_uint_buffers(
    device atomic_uint* buffer [[buffer(0)]],       // Buffer to clear
    constant uint& size [[buffer(1)]],              // Buffer size
    uint id [[thread_position_in_grid]])
{
    if (id >= size) return;
    atomic_store_explicit(&buffer[id], 0, memory_order_relaxed);
}
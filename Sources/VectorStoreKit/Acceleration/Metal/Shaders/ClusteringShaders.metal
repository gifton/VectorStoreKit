// VectorStoreKit: Clustering Shaders
//
// Metal shaders for K-means clustering operations
// These kernels implement GPU-accelerated K-means clustering with optimizations
// for Apple Silicon's unified memory architecture

#include <metal_stdlib>
using namespace metal;

// MARK: - Centroid Assignment

/// Assign each vector to its nearest centroid using Euclidean distance
/// This kernel is the core of K-means clustering - it determines which cluster
/// each data point belongs to by finding the centroid with minimum distance
///
/// Performance characteristics:
/// - Memory access: O(k * d) per thread where k = num_centroids, d = dimensions
/// - Computation: O(k * d) distance calculations per thread
/// - Parallelism: One thread per vector, scales linearly with data size
///
/// @param vectors Input data points stored as flattened array [num_vectors * dimensions]
/// @param centroids Current cluster centers stored as flattened array [num_centroids * dimensions]
/// @param assignments Output array storing the assigned centroid index for each vector
/// @param distances Output array storing the distance to assigned centroid (for convergence checking)
/// @param num_vectors Total number of data points to cluster
/// @param num_centroids Number of clusters (k in K-means)
/// @param dimensions Dimensionality of each vector
/// @param id Thread index corresponding to vector index
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
    // Early exit for threads beyond data bounds
    if (id >= num_vectors) return;
    
    // Calculate base offset for this vector in the flattened array
    // Memory layout: vectors[vector_id][dimension] = vectors[vector_id * dimensions + dimension]
    uint vector_offset = id * dimensions;
    float min_distance = INFINITY;
    uint min_centroid = 0;
    
    // Find nearest centroid by checking distance to each cluster center
    // This is the computationally intensive part of K-means
    for (uint c = 0; c < num_centroids; c++) {
        // Calculate base offset for current centroid
        uint centroid_offset = c * dimensions;
        float distance = 0.0f;
        
        // Compute squared Euclidean distance: sum((v[i] - c[i])^2)
        // We use squared distance to avoid expensive sqrt() until final result
        for (uint d = 0; d < dimensions; d++) {
            float diff = vectors[vector_offset + d] - centroids[centroid_offset + d];
            distance += diff * diff;
        }
        
        // Track the centroid with minimum distance
        if (distance < min_distance) {
            min_distance = distance;
            min_centroid = c;
        }
    }
    
    // Store results: which centroid this vector is assigned to
    assignments[id] = min_centroid;
    // Store actual distance (with sqrt) for convergence checking and analysis
    distances[id] = sqrt(min_distance);
}

// MARK: - Centroid Update

/// Update centroids based on assignments (step 1: accumulate)
/// This kernel implements the "update" step of K-means clustering using atomic operations
/// to handle concurrent updates from multiple threads
///
/// The centroid update is split into two phases:
/// 1. Accumulate: Sum all vectors assigned to each centroid (this kernel)
/// 2. Finalize: Divide sums by counts to get new centroid positions
///
/// Performance considerations:
/// - Uses atomic operations which can create contention if many vectors map to same centroid
/// - memory_order_relaxed is safe here as we only need eventual consistency
/// - Atomic operations on Apple Silicon are optimized for unified memory
///
/// @param vectors Input data points to be accumulated
/// @param assignments Current cluster assignments from assign_to_centroids kernel
/// @param centroid_sums Atomic buffer to accumulate vector sums per centroid
/// @param centroid_counts Atomic buffer to count vectors per centroid
/// @param dimensions Vector dimensionality
/// @param id Thread index corresponding to vector index
kernel void accumulate_centroids(
    constant float* vectors [[buffer(0)]],           // Input vectors
    constant uint* assignments [[buffer(1)]],        // Vector assignments
    device atomic_float* centroid_sums [[buffer(2)]],// Accumulated sums (atomic)
    device atomic_uint* centroid_counts [[buffer(3)]],// Counts per centroid (atomic)
    constant uint& dimensions [[buffer(4)]],         // Vector dimensions
    uint id [[thread_position_in_grid]])
{
    // Calculate offsets for this vector and its assigned centroid
    uint vector_offset = id * dimensions;
    uint assignment = assignments[id];
    uint centroid_offset = assignment * dimensions;
    
    // Accumulate this vector's values into its assigned centroid's sum
    // Using atomic operations ensures correctness when multiple threads
    // update the same centroid simultaneously
    for (uint d = 0; d < dimensions; d++) {
        atomic_fetch_add_explicit(
            &centroid_sums[centroid_offset + d],
            vectors[vector_offset + d],
            memory_order_relaxed  // Relaxed ordering is sufficient for accumulation
        );
    }
    
    // Increment the count of vectors assigned to this centroid
    // This count will be used in finalize_centroids to compute the mean
    atomic_fetch_add_explicit(
        &centroid_counts[assignment],
        1,
        memory_order_relaxed
    );
}

/// Update centroids based on assignments (step 2: divide)
/// This kernel completes the centroid update by computing the mean position
/// of all vectors assigned to each centroid
///
/// Thread organization:
/// - 2D grid where x = centroid index, y = dimension index
/// - This allows parallel computation of all centroid dimensions
/// - More efficient than having one thread compute entire centroid
///
/// Edge cases handled:
/// - Empty clusters: Centroid position is preserved if no vectors assigned
/// - This prevents centroid collapse and maintains K clusters
///
/// @param centroid_sums Accumulated sums from accumulate_centroids kernel
/// @param centroid_counts Number of vectors assigned to each centroid
/// @param centroids Output buffer for new centroid positions
/// @param num_centroids Total number of centroids (k)
/// @param dimensions Vector dimensionality
/// @param id 2D thread position (x: centroid, y: dimension)
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
    
    // Bounds checking for 2D thread grid
    if (c >= num_centroids || d >= dimensions) return;
    
    // Calculate linear index into flattened centroid array
    uint idx = c * dimensions + d;
    uint count = centroid_counts[c];
    
    if (count > 0) {
        // Compute mean by dividing sum by count
        // This gives us the new centroid position as the center of mass
        // of all vectors assigned to this cluster
        centroids[idx] = centroid_sums[idx] / float(count);
    } else {
        // Handle empty cluster case:
        // Keep existing centroid position to prevent cluster collapse
        // The centroids buffer should retain previous iteration's values
        // Alternative strategies: reinitialize to random vector or furthest point
    }
}

// MARK: - K-means++ Initialization

/// Compute minimum distances to existing centroids for K-means++ initialization
/// K-means++ is an initialization method that chooses initial centroids to be far apart,
/// leading to better convergence and final cluster quality
///
/// Algorithm overview:
/// 1. Choose first centroid randomly
/// 2. For each remaining centroid:
///    a. Compute min distance from each point to existing centroids (this kernel)
///    b. Choose new centroid with probability proportional to squared distance
///
/// This kernel computes step 2a, which is used for weighted random selection
///
/// Performance notes:
/// - Similar structure to assign_to_centroids but only tracks minimum distance
/// - Output is squared distances (no sqrt) for use in probability calculation
/// - Can be optimized with SIMD operations for high-dimensional data
///
/// @param vectors All data points in the dataset
/// @param centroids Currently selected centroids (partial set during initialization)
/// @param min_distances Output buffer for minimum squared distances
/// @param num_vectors Total number of data points
/// @param num_centroids Current number of selected centroids
/// @param dimensions Vector dimensionality
/// @param id Thread index corresponding to vector index
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
    
    // Find minimum distance to any existing centroid
    // This determines how far this vector is from current cluster centers
    for (uint c = 0; c < num_centroids; c++) {
        uint centroid_offset = c * dimensions;
        float distance = 0.0f;
        
        // Squared Euclidean distance computation
        for (uint d = 0; d < dimensions; d++) {
            float diff = vectors[vector_offset + d] - centroids[centroid_offset + d];
            distance += diff * diff;
        }
        
        // Track minimum across all centroids
        min_distance = min(min_distance, distance);
    }
    
    // Store squared distance for probability-weighted selection
    // Points with larger distances are more likely to be chosen as next centroid
    min_distances[id] = min_distance;
}

// MARK: - Mini-batch K-means

/// Perform incremental centroid update for mini-batch K-means
/// Mini-batch K-means is a variant that updates centroids using small batches
/// rather than the entire dataset, enabling clustering of very large datasets
///
/// Algorithm benefits:
/// - Memory efficient: Only processes batch_size vectors at a time
/// - Fast convergence: Updates centroids after each batch
/// - Online learning: Can adapt to streaming data
///
/// Update formula: c_new = c_old + learning_rate * (x - c_old)
/// This is a gradient descent step towards the batch mean
///
/// Thread organization:
/// - 2D grid: x = batch vector index, y = dimension
/// - Each thread updates one dimension of one vector's assigned centroid
/// - Potential race conditions handled by atomic operations or careful scheduling
///
/// @param centroids Current centroid positions to be updated in-place
/// @param batch_vectors Mini-batch of vectors for this iteration
/// @param batch_assignments Pre-computed assignments for the batch
/// @param learning_rate Step size for gradient descent (typically decreases over time)
/// @param batch_size Number of vectors in this mini-batch
/// @param dimensions Vector dimensionality
/// @param id 2D thread position (x: batch index, y: dimension)
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
    
    // Bounds check for 2D thread grid
    if (batch_idx >= batch_size || dim >= dimensions) return;
    
    // Get the centroid assignment for this batch vector
    uint assignment = batch_assignments[batch_idx];
    float vector_value = batch_vectors[batch_idx * dimensions + dim];
    
    // Incremental update using gradient descent
    // This moves the centroid towards the new data point
    // learning_rate controls how much influence each batch has
    uint centroid_idx = assignment * dimensions + dim;
    float old_value = centroids[centroid_idx];
    float new_value = old_value + learning_rate * (vector_value - old_value);
    
    // Update centroid position
    // NOTE: This may have race conditions if multiple batch vectors
    // are assigned to the same centroid. Consider atomic operations
    // or ensure batch assignments are processed sequentially per centroid
    centroids[centroid_idx] = new_value;
}

// MARK: - Utility Functions

/// Compute inertia (sum of squared distances to assigned centroids)
/// Inertia is the key metric for K-means convergence and quality assessment
/// Lower inertia indicates tighter, more cohesive clusters
///
/// Mathematical definition:
/// Inertia = Σ(||x_i - c_j||²) where x_i belongs to cluster j with centroid c_j
///
/// Uses:
/// - Convergence criterion: Stop when inertia change is below threshold
/// - Elbow method: Plot inertia vs K to find optimal number of clusters
/// - Quality metric: Compare different runs or algorithms
///
/// @param distances Pre-computed distances from assign_to_centroids kernel
/// @param inertia Atomic accumulator for total inertia (single value)
/// @param num_vectors Total number of vectors being clustered
/// @param id Thread index corresponding to vector index
kernel void compute_inertia(
    constant float* distances [[buffer(0)]],         // Distances to assigned centroids
    device atomic_float* inertia [[buffer(1)]],     // Output inertia (atomic)
    constant uint& num_vectors [[buffer(2)]],        // Number of vectors
    uint id [[thread_position_in_grid]])
{
    if (id >= num_vectors) return;
    
    // Square the distance (inertia uses squared Euclidean distance)
    float squared_distance = distances[id] * distances[id];
    
    // Atomically add to global inertia sum
    // All threads contribute to single value, hence atomic operation
    atomic_fetch_add_explicit(inertia, squared_distance, memory_order_relaxed);
}

/// Initialize atomic float buffers to zero
/// Essential for preparing atomic accumulator buffers before parallel operations
///
/// Metal doesn't guarantee buffer initialization, so explicit clearing is needed
/// This kernel ensures all atomic values start at zero before accumulation
///
/// @param buffer Atomic float buffer to clear
/// @param size Number of elements in the buffer
/// @param id Thread index for parallel clearing
kernel void clear_atomic_buffers(
    device atomic_float* buffer [[buffer(0)]],      // Buffer to clear
    constant uint& size [[buffer(1)]],              // Buffer size
    uint id [[thread_position_in_grid]])
{
    if (id >= size) return;
    
    // Atomic store ensures proper initialization even with concurrent access
    // memory_order_relaxed is sufficient for initialization
    atomic_store_explicit(&buffer[id], 0.0f, memory_order_relaxed);
}

/// Initialize atomic uint buffers to zero
/// Specialized version for unsigned integer atomic buffers
/// Used for clearing centroid count buffers before accumulation
///
/// @param buffer Atomic uint buffer to clear
/// @param size Number of elements in the buffer
/// @param id Thread index for parallel clearing
kernel void clear_atomic_uint_buffers(
    device atomic_uint* buffer [[buffer(0)]],       // Buffer to clear
    constant uint& size [[buffer(1)]],              // Buffer size
    uint id [[thread_position_in_grid]])
{
    if (id >= size) return;
    
    // Initialize to zero for count accumulation
    atomic_store_explicit(&buffer[id], 0, memory_order_relaxed);
}
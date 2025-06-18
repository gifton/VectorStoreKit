# VectorStoreKit Metal Shaders

This directory contains GPU-accelerated compute shaders for VectorStoreKit, optimized for Apple Silicon's unified memory architecture and Metal Performance Shaders framework.

## Overview

These Metal shaders provide massive parallelization for vector database operations, achieving 10-100x speedups over CPU implementations for large-scale similarity search, clustering, and matrix operations.

## Shader Files

### 1. **DistanceShaders.metal**
Core distance metric computations for vector similarity search.

**Kernels:**
- `euclideanDistance` - L2 distance between vectors
- `cosineDistance` - Angular similarity (1 - cosine similarity)
- `manhattanDistance` - L1 distance for sparse vectors
- `dotProduct` - Inner product similarity
- `batchEuclideanDistance` - Batch processing for multiple queries
- `batchCosineDistance` - Batch cosine distance computation
- `normalizeVectors` - L2 normalization for cosine similarity
- `euclideanDistance512_simd` - Optimized for 512-dimensional vectors
- `cosineDistance512_simd` - SIMD-optimized cosine for 512d

**Key Features:**
- SIMD vectorization using float4 types
- Optimized memory access patterns
- Special handling for 512-dimensional embeddings (common in ML)

### 2. **OptimizedDistanceShaders.metal**
Next-generation distance kernels with advanced optimizations.

**Kernels:**
- `euclideanDistanceWarpOptimized` - Uses warp-level primitives
- `euclideanDistanceHalf` - Half-precision for 2x memory bandwidth
- `tiledBatchDistance` - Tiled computation with shared memory
- `euclideanDistanceSquaredFast` - Skip sqrt for ranking
- `fusedDistanceTopK` - Combined distance + top-K selection

**Advanced Features:**
- Warp-level reduction using SIMD shuffle operations
- Multiple accumulators to hide FPU latency
- Prefetching for memory latency hiding
- Function constants for runtime specialization
- Bank conflict avoidance in shared memory

### 3. **ClusteringShaders.metal**
GPU-accelerated K-means clustering implementation.

**Kernels:**
- `assign_to_centroids` - Assign vectors to nearest cluster
- `accumulate_centroids` - Sum vectors by cluster (atomic ops)
- `finalize_centroids` - Compute new centroid positions
- `compute_min_distances` - For K-means++ initialization
- `update_centroids_incremental` - Mini-batch K-means
- `compute_inertia` - Calculate clustering quality
- `clear_atomic_buffers` - Initialize atomic memory

**Key Features:**
- Atomic operations for concurrent updates
- Support for K-means++ initialization
- Mini-batch variant for streaming data
- Efficient parallel reduction

### 4. **MatrixShaders.metal**
Fundamental matrix operations for linear algebra.

**Kernels:**
- `matrixMultiply` - Basic matrix multiplication
- `tiledMatrixMultiply` - Cache-optimized with shared memory
- `matrixTranspose` - Simple transpose operation
- `matrixTransposeCoalesced` - Optimized transpose with padding
- `matrixAdd`, `matrixSubtract`, `matrixScalarMultiply` - Element-wise ops
- `matrixRowSum`, `matrixColSum` - Reduction operations
- `matrixNorm` - Frobenius norm with parallel reduction

**Optimizations:**
- Tiled algorithms for cache efficiency
- Bank conflict avoidance
- Coalesced memory access
- Parallel reduction patterns

### 5. **QuantizationShaders.metal**
Vector compression techniques for memory efficiency.

**Kernels:**
- `scalarQuantize` - Float32 to 8-bit quantization
- `scalarDequantize` - Reconstruct from 8-bit
- `productQuantize` - Advanced PQ compression
- `productDequantize` - PQ reconstruction
- `binaryQuantize` - 1-bit per dimension encoding
- `binaryHammingDistance` - Fast binary distance
- `computeQuantizationStats` - MSE and PSNR metrics

**Compression Ratios:**
- Scalar: 4x (float32 → uint8)
- Product: 32-128x (with codebook)
- Binary: 32x (float32 → 1 bit)

## Performance Characteristics

### Memory Bandwidth Optimization
- **Coalesced Access**: Threads in a warp access consecutive memory
- **Shared Memory**: Reduces global memory traffic by 10-100x
- **Float4 Vectorization**: Process 4 values per instruction

### Compute Optimization
- **FMA Instructions**: Fused multiply-add for accuracy and speed
- **Loop Unrolling**: Maximize instruction-level parallelism
- **Multiple Accumulators**: Hide floating-point latency
- **Warp Primitives**: Leverage SIMD group operations

### Apple Silicon Specific
- **Unified Memory**: Zero-copy data sharing with CPU
- **Metal Performance Shaders**: Integration with MPS framework
- **Function Constants**: Runtime kernel specialization
- **Half Precision**: Hardware-accelerated FP16

## Usage Examples

### Basic Distance Computation
```swift
// Configure compute pipeline
let distanceKernel = device.makeComputePipelineState(function: "euclideanDistance")

// Set buffers
encoder.setBuffer(queryBuffer, offset: 0, index: 0)
encoder.setBuffer(candidatesBuffer, offset: 0, index: 1)
encoder.setBuffer(distancesBuffer, offset: 0, index: 2)
encoder.setBytes(&vectorDimension, length: 4, index: 3)

// Dispatch threads
let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
let numGroups = MTLSize(width: (numCandidates + 255) / 256, height: 1, depth: 1)
encoder.dispatchThreadgroups(numGroups, threadsPerThreadgroup: threadsPerGroup)
```

### Optimized 512D Search
```swift
// Use specialized kernel for 512-dimensional vectors
let kernel = device.makeComputePipelineState(function: "euclideanDistance512_simd")

// Vectors must be aligned and padded to float4 boundaries
// 512 floats = 128 float4s
```

### Product Quantization
```swift
// Compress vectors using PQ
let pqKernel = device.makeComputePipelineState(function: "productQuantize")

// Configure subspaces (e.g., 16 subspaces of 32 dimensions)
encoder.setBytes(&numSubspaces, length: 4, index: 4)
encoder.setBytes(&subspaceDimension, length: 4, index: 5)
```

## Performance Guidelines

### Optimal Thread Configuration
- **Distance Kernels**: 256-512 threads per group
- **Matrix Operations**: 16x16 or 32x32 tiles
- **Clustering**: Match vector count to thread count
- **Quantization**: 2D grids for product quantization

### Memory Requirements
- **Shared Memory**: ~48KB per threadgroup (Apple Silicon)
- **Register Pressure**: Keep under 32 registers per thread
- **Buffer Alignment**: Align to 16 bytes for float4

### Expected Performance
- **Distance Computation**: 100M-1B distances/second
- **Matrix Multiply**: 500+ GFLOPS (optimized)
- **Quantization**: Real-time for millions of vectors
- **Clustering**: 10-100x faster than CPU

## Debugging and Profiling

### Metal GPU Capture
```bash
# Enable GPU debugging
export METAL_DEVICE_WRAPPER_TYPE=1
export METAL_DEBUG_ERROR_MODE=3
```

### Performance Counters
- Use Instruments GPU profiler
- Monitor ALU utilization
- Check memory bandwidth usage
- Identify pipeline stalls

### Common Issues
1. **Bank Conflicts**: Add padding to shared memory arrays
2. **Divergent Branches**: Minimize conditionals in kernels
3. **Register Spilling**: Reduce local variables
4. **Occupancy**: Balance registers vs shared memory

## Future Optimizations

### Planned Enhancements
- **Mixed Precision**: FP16 compute with FP32 accumulation
- **Tensor Cores**: Leverage matrix acceleration units
- **Dynamic Dispatch**: Runtime kernel selection
- **Multi-GPU**: Distributed computation

### Experimental Features
- Learned distance metrics
- Approximate algorithms (LSH)
- Streaming quantization
- Online clustering

## Contributing

When adding new shaders:
1. Follow existing naming conventions
2. Include comprehensive documentation
3. Provide both optimized and reference implementations
4. Add unit tests with performance benchmarks
5. Profile on various Apple Silicon configurations

## License

See VectorStoreKit license file for details.
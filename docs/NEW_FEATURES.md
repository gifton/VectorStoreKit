# VectorStoreKit New Features Documentation

This document describes the new high-performance components added to VectorStoreKit for optimized vector operations on Apple Silicon.

## Table of Contents

1. [Vector512 - Optimized 512-Dimensional Vectors](#vector512)
2. [BatchProcessor - High-Performance Batch Processing](#batchprocessor)
3. [HierarchicalIndex - Two-Level Indexing for Large Datasets](#hierarchicalindex)
4. [ScalarQuantizer - Memory-Efficient Vector Storage](#scalarquantizer)
5. [StreamingBufferManager - Large Dataset Streaming](#streamingbuffermanager)

---

## Vector512

A highly optimized 512-dimensional vector type leveraging Apple Silicon's SIMD capabilities.

### Overview

`Vector512` provides hardware-accelerated operations for 512-dimensional vectors, a common dimension in modern embeddings (e.g., OpenAI's text-embedding-ada-002).

### Key Features

- **Aligned Memory**: 64-byte alignment for optimal SIMD operations
- **SIMD Operations**: Hardware-accelerated arithmetic using Metal and Accelerate
- **Collection Conformance**: Full Swift Collection protocol support
- **Zero-Copy Operations**: Efficient memory management

### Usage

```swift
// Create vectors
let v1 = Vector512(repeating: 1.0)
let v2 = Vector512([1.0, 2.0, 3.0, ...]) // 512 values

// SIMD operations
let sum = v1 + v2
let product = v1 * v2
let dotProduct = simd_dot(v1, v2)

// Collection operations
let filtered = v1.filter { $0 > 0.5 }
let mapped = v1.map { $0 * 2.0 }
```

### Performance Characteristics

- **Memory**: 2KB per vector (512 × 4 bytes)
- **Alignment**: 64-byte aligned for cache efficiency
- **Operations**: ~10x faster than scalar operations on Apple Silicon

---

## BatchProcessor

High-performance batch processing system for large-scale vector operations.

### Overview

`BatchProcessor` enables efficient processing of large vector datasets with automatic batching, Metal acceleration, and memory management.

### Key Features

- **Automatic Batching**: Optimal batch sizes based on memory constraints
- **Metal Acceleration**: GPU-accelerated operations when available
- **Concurrent Processing**: Multi-threaded batch processing
- **Progress Tracking**: Real-time progress monitoring
- **Memory Management**: Adaptive batch sizing to prevent memory pressure

### Configuration

```swift
let config = BatchProcessingConfiguration(
    optimalBatchSize: 1000,
    maxConcurrentBatches: 8,
    memoryLimit: 1_073_741_824, // 1GB
    useMetalAcceleration: true
)

let processor = BatchProcessor(configuration: config)
```

### Operations

```swift
// Transform vectors
let scaled = try await processor.processBatches(
    dataset: dataset,
    operation: .transformation { vector in
        vector * 2.0
    }
)

// Filter vectors
let filtered = try await processor.processBatches(
    dataset: dataset,
    operation: .filtering { vector in
        vector.magnitude > threshold
    }
)

// Aggregate vectors
let mean = try await processor.processBatches(
    dataset: dataset,
    operation: .aggregation { vectors in
        vectors.reduce(Vector512(), +) / Float(vectors.count)
    }
)
```

### Progress Monitoring

```swift
let results = try await processor.processBatches(
    dataset: dataset,
    operation: operation,
    progressHandler: { progress in
        print("Processed \(progress.processedItems)/\(progress.totalItems)")
        print("Speed: \(progress.itemsPerSecond) items/sec")
        print("ETA: \(progress.estimatedTimeRemaining ?? 0) seconds")
    }
)
```

---

## HierarchicalIndex

Two-level hierarchical index optimized for billion-scale vector search.

### Overview

`HierarchicalIndex` implements a two-level indexing structure:
1. **Top Level**: IVF (Inverted File) index for coarse quantization
2. **Leaf Level**: HNSW indexes for fine-grained search within clusters

### Architecture

```
┌─────────────────────────┐
│   IVF Top-Level Index   │
│   (1000+ clusters)      │
└─────────┬───────────────┘
          │
    ┌─────┴─────┬─────────┬─────────┐
    │           │         │         │
┌───▼───┐ ┌────▼───┐ ┌───▼───┐ ┌───▼───┐
│ HNSW  │ │  HNSW  │ │ HNSW  │ │ HNSW  │
│Leaf 1 │ │ Leaf 2 │ │Leaf 3 │ │Leaf N │
└───────┘ └────────┘ └───────┘ └───────┘
```

### Configuration

```swift
// Automatic configuration based on dataset size
let config = HierarchicalConfiguration.forDatasetSize(1_000_000)

// Manual configuration
let config = HierarchicalConfiguration(
    topLevelClusters: 1024,      // Number of coarse clusters
    leafIndexSize: 1000,         // Vectors per leaf index
    probesPerQuery: 10,          // Clusters to search
    enableDynamicProbing: true,  // Adaptive search
    rebalanceThreshold: 2.0      // Imbalance trigger
)

let index = try await HierarchicalIndex<Vector512>(
    dimension: 512,
    configuration: config
)
```

### Operations

```swift
// Insert vectors
try await index.insert(vector, id: "vec_001", metadata: metadata)
try await index.insertBatch(vectors)

// Search
let results = try await index.search(
    query: queryVector,
    k: 10,
    filter: { metadata in
        // Optional filtering
        return metadata?.category == "electronics"
    }
)

// Batch search
let batchResults = try await index.batchSearch(
    queries: queryVectors,
    k: 10
)

// Rebalancing
try await index.rebalance() // Redistribute imbalanced clusters
```

### Performance Optimization

- **Dynamic Probing**: Automatically adjusts search breadth based on data distribution
- **Parallel Search**: Concurrent search across leaf indexes
- **Adaptive Rebalancing**: Maintains balanced cluster sizes

---

## ScalarQuantizer

Memory-efficient vector storage through scalar quantization.

### Overview

`ScalarQuantizer` reduces memory usage by quantizing float32 vectors to int8/uint8/float16, achieving 2-4x compression with minimal accuracy loss.

### Quantization Types

| Type | Compression | Use Case |
|------|------------|----------|
| `int8` | 4x | General purpose, symmetric data |
| `uint8` | 4x | Non-negative data only |
| `float16` | 2x | High precision requirements |
| `dynamic` | Variable | Automatic type selection |

### Usage

```swift
let quantizer = ScalarQuantizer()

// Quantize single vector
let quantized = try await quantizer.quantize(
    vector,
    type: .int8,
    statistics: nil  // Auto-compute
)

// Batch quantization with shared statistics
let stats = try await quantizer.computeStatistics(for: vectors)
let quantizedBatch = try await quantizer.quantizeBatch(
    vectors: vectors,
    type: .dynamic,  // Auto-select best type
    statistics: stats
)

// Dequantize
let restored = try await quantizer.dequantize(quantized)
```

### Analysis

```swift
// Analyze quantization quality
let analysis = try await quantizer.analyzeQuantization(
    vectors: testVectors,
    type: .int8
)

print("Compression: \(analysis.compressionRatio)x")
print("Avg Error: \(analysis.averageError)")
print("Max Error: \(analysis.maxError)")
print("SNR: \(analysis.signalToNoiseRatio) dB")
```

### Metal Acceleration

Quantization automatically uses Metal compute shaders for batches > 100 vectors:
- Parallel quantization on GPU
- Efficient memory transfers
- Hardware-accelerated statistics computation

---

## StreamingBufferManager

Efficient streaming and memory management for datasets larger than RAM.

### Overview

`StreamingBufferManager` enables processing of massive vector datasets through:
- Memory-mapped file access
- Multi-tier storage (hot/warm/cold)
- Intelligent prefetching
- Metal buffer integration

### Architecture

```
┌─────────────────────────────────┐
│     StreamingBufferManager      │
├─────────────────────────────────┤
│  Hot Tier  │ Warm Tier │ Cold   │
│  (Memory)  │  (SSD)    │ (HDD)  │
├─────────────────────────────────┤
│      Memory-Mapped Files        │
│      Prefetch Cache             │
│      Metal Buffer Pool          │
└─────────────────────────────────┘
```

### Usage

```swift
// Initialize manager
let manager = try await StreamingBufferManager(
    device: metalDevice,
    targetSize: 10_737_418_240,  // 10GB dataset
    pageSize: 1_048_576          // 1MB pages
)

// Add vector files
try await manager.addFile(
    url: vectorFileURL,
    tier: .warm,
    metadata: FileMetadata(
        vectorCount: 1_000_000,
        dimension: 512,
        vectorSize: 2048
    )
)

// Load vectors on-demand
let data = try await manager.loadVectors(
    tier: .warm,
    offset: 0,
    count: 1000
)

// Stream vectors
let stream = await manager.streamVectors(
    tier: .warm,
    batchSize: 100
)

for await batch in stream {
    // Process batch
}
```

### Prefetching

```swift
// Prefetch for sequential access
try await manager.prefetchVectors(
    tier: .warm,
    ranges: [
        0..<1000,
        5000..<6000,
        10000..<11000
    ]
)
```

### Metal Integration

```swift
// Create Metal buffer from streamed data
let metalBuffer = try await manager.createMetalBuffer(
    from: data,
    options: .storageModeShared
)
```

### Performance Tips

1. **Page Size**: Match system page size (usually 4KB) for optimal performance
2. **Prefetching**: Prefetch 2-3 batches ahead for sequential processing
3. **Tier Management**: Keep frequently accessed vectors in hot tier
4. **Memory Pressure**: Monitor and respond to system memory warnings

---

## Integration Example

Here's how these components work together for a complete vector search system:

```swift
// 1. Initialize components
let bufferManager = try await StreamingBufferManager(
    device: device,
    targetSize: datasetSize
)

let index = try await HierarchicalIndex<Vector512>(
    dimension: 512,
    configuration: .forDatasetSize(1_000_000)
)

let quantizer = ScalarQuantizer()
let processor = BatchProcessor()

// 2. Build index with quantized vectors
let stream = await bufferManager.streamVectors(tier: .warm)

for await batch in stream {
    // Process batch
    let vectors: [Vector512] = parseVectors(from: batch)
    
    // Quantize for storage efficiency
    let quantized = try await quantizer.quantizeBatch(
        vectors: vectors,
        type: .int8
    )
    
    // Index vectors
    try await processor.processVector512Batches(
        vectors: vectors,
        operation: .indexing(index: index)
    )
}

// 3. Search with dequantization
let results = try await index.search(
    query: queryVector,
    k: 10
)

// Retrieve and dequantize full vectors
let fullVectors = try await results.asyncMap { result in
    let quantized = loadQuantizedVector(id: result.id)
    return try await quantizer.dequantize(quantized)
}
```

## Performance Guidelines

1. **Batch Size**: Use `BatchProcessingConfiguration.optimalBatchSize()` for your vector dimensions
2. **Metal Usage**: Enable for operations > 1000 vectors
3. **Quantization**: Use int8 for 4x compression with < 1% accuracy loss
4. **Hierarchical Index**: Probe sqrt(clusters) for 95% recall
5. **Streaming**: Use 1MB pages for SSD, 4MB for HDD

## Migration Guide

To integrate these new components into existing VectorStoreKit code:

1. Replace generic vector arrays with `Vector512` for 512-dim vectors
2. Wrap large-scale operations in `BatchProcessor`
3. Use `HierarchicalIndex` for datasets > 100K vectors
4. Apply `ScalarQuantizer` for memory-constrained environments
5. Implement `StreamingBufferManager` for out-of-core datasets
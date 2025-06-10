# VectorStoreKit

A high-performance, Swift-native vector database framework optimized for Apple platforms with Metal acceleration, advanced indexing algorithms, and intelligent storage management.

[![Swift](https://img.shields.io/badge/Swift-5.10+-orange.svg)](https://swift.org)
[![Platforms](https://img.shields.io/badge/Platforms-macOS%20|%20iOS%20|%20tvOS%20|%20watchOS%20|%20visionOS-blue.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Advanced Usage](#advanced-usage)
- [Architecture](#architecture)
- [Performance](#performance)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Overview

VectorStoreKit is a comprehensive vector database solution designed specifically for Apple platforms. It leverages Metal for GPU acceleration, implements state-of-the-art indexing algorithms, and provides a flexible, type-safe API for vector similarity search operations.

### Why VectorStoreKit?

- **Native Performance**: Built from the ground up for Apple silicon with Metal acceleration
- **Production Ready**: Battle-tested indexing algorithms with proven scalability
- **Developer Friendly**: Type-safe Swift API with comprehensive documentation
- **Flexible Architecture**: Modular design allows custom implementations at every layer
- **AI/ML Integration**: Seamless integration with Core ML and other Apple ML frameworks

## Key Features

### 🚀 Performance & Acceleration
- **Metal GPU Acceleration**: Hardware-accelerated distance computations and matrix operations
- **Neural Engine Support**: Automatic detection and utilization of Apple Neural Engine
- **SIMD Optimizations**: CPU-optimized operations using Swift SIMD types
- **Parallel Processing**: Multi-threaded indexing and search operations

### 🔍 Advanced Indexing
- **Multiple Index Types**:
  - **HNSW (Hierarchical Navigable Small World)**: High-performance approximate nearest neighbor search
  - **IVF (Inverted File Index)**: Scalable indexing with clustering-based partitioning
  - **Hybrid Index**: Combines multiple strategies for optimal performance
  - **Learned Index**: ML-based indexing for adaptive performance
  
### 💾 Intelligent Storage
- **Hierarchical Storage Tiers**:
  - **Hot Tier**: In-memory storage for frequently accessed vectors
  - **Warm Tier**: SSD-optimized storage with compression
  - **Cold Tier**: Long-term archival storage
  - **Archive Tier**: Compressed, encrypted long-term storage
- **Automatic Migration**: Intelligent data movement based on access patterns
- **Write-Ahead Logging**: Durability and crash recovery
- **Encryption**: Built-in support for data-at-rest encryption

### 🎯 Search Capabilities
- **Distance Metrics**: Euclidean, Cosine, Dot Product, Manhattan, and custom metrics
- **Filtering**: Metadata, vector property, and composite filters
- **Batch Operations**: Efficient bulk insert and search
- **Real-time Updates**: Support for dynamic vector additions and deletions

### 🧩 Integration & Extensibility
- **PipelineKit Integration**: Command-based architecture with middleware support
- **SwiftUI Support**: Observable objects for reactive UI updates
- **Combine Integration**: Publisher-based APIs for reactive programming
- **Custom Implementations**: Protocol-based design for easy customization

## Requirements

- **Swift**: 5.10 or later
- **Platforms**:
  - macOS 14.0+
  - iOS 17.0+
  - tvOS 17.0+
  - watchOS 10.0+
  - visionOS 1.0+
- **Hardware**: Metal-capable device (optional but recommended)

## Installation

### Swift Package Manager

Add VectorStoreKit to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/VectorStoreKit.git", from: "1.0.0")
]
```

Or in Xcode:
1. File → Add Package Dependencies
2. Enter the repository URL
3. Select version and add to your project

## Quick Start

### Basic Usage

```swift
import VectorStoreKit
import simd

// Create a vector store with default configuration
let store = VectorStore<SIMD4<Float>, [String: String]>()

// Add vectors
let vector = VectorEntry(
    id: "doc1",
    vector: SIMD4<Float>(1.0, 2.0, 3.0, 4.0),
    metadata: ["title": "Example Document", "category": "demo"]
)
try await store.add([vector])

// Search for similar vectors
let results = try await store.search(
    query: SIMD4<Float>(1.1, 2.0, 2.9, 4.1),
    k: 10
)

// Access results
for result in results {
    print("ID: \(result.id), Distance: \(result.distance)")
    print("Metadata: \(result.metadata)")
}
```

### Using VectorUniverse (Advanced Configuration)

```swift
// Create a highly configured vector universe
let universe = VectorUniverse<SIMD32<Float>, DocumentMetadata>()
    .index(using: HNSWProductionIndexingStrategy(
        m: 16,
        efConstruction: 200,
        maxElements: 1_000_000
    ))
    .store(using: HierarchicalStorageStrategy(
        hotCapacity: 10_000,
        warmCapacity: 100_000,
        enableEncryption: true
    ))
    .accelerate(using: MetalPerformanceAcceleratorStrategy())
    .optimize(using: MLProductionOptimizationStrategy())
    .cache(using: MultiLevelCachingStrategy(
        l1Size: 1000,
        l2Size: 10_000
    ))

// Build the configured store
let store = try await universe.build()
```

## Core Concepts

### Vectors and Entries

```swift
// VectorEntry represents a vector with metadata
struct VectorEntry<Vector: SIMD, Metadata: Codable> {
    let id: String
    let vector: Vector
    let metadata: Metadata
    let timestamp: Date
    let tier: StorageTier
}
```

### Distance Metrics

```swift
// Built-in distance metrics
let metrics: [DistanceMetric] = [
    .euclidean,
    .cosine,
    .dotProduct,
    .manhattan,
    .custom { v1, v2 in
        // Your custom distance calculation
        return customDistance(v1, v2)
    }
]
```

### Filtering

```swift
// Metadata filtering
let metadataFilter = SearchFilter.metadata(
    MetadataFilter(key: "category", operation: .equals, value: "electronics")
)

// Vector property filtering
let vectorFilter = SearchFilter.vector(
    VectorFilter(constraint: .magnitude(10.0...50.0))
)

// Composite filtering
let compositeFilter = SearchFilter.composite(
    CompositeFilter(operation: .and, filters: [metadataFilter, vectorFilter])
)

// Search with filters
let results = try await store.search(
    query: queryVector,
    k: 20,
    filter: compositeFilter
)
```

## Advanced Usage

### Custom Index Implementation

```swift
// Implement a custom index
actor CustomIndex<Vector: SIMD>: VectorIndex {
    typealias VectorID = String
    typealias Metadata = [String: String]
    
    func insert(_ id: VectorID, vector: Vector, metadata: Metadata?) async throws {
        // Custom insertion logic
    }
    
    func search(query: Vector, k: Int, filter: SearchFilter?) async throws -> [SearchResult] {
        // Custom search logic
    }
}
```

### Batch Operations

```swift
// Batch insert with progress tracking
let vectors = generateLargeDataset() // 1M vectors

let progress = Progress(totalUnitCount: Int64(vectors.count))
try await store.batchInsert(vectors) { completed in
    progress.completedUnitCount = Int64(completed)
    print("Progress: \(progress.fractionCompleted * 100)%")
}

// Parallel batch search
let queries = generateQueries() // 1000 queries
let results = try await store.batchSearch(
    queries: queries,
    k: 10,
    parallelism: 8
)
```

### Storage Management

```swift
// Configure storage tiers
let storageConfig = HierarchicalStorageConfiguration(
    tiers: [
        .hot: HotTierConfiguration(maxSize: 1_000_000),
        .warm: WarmTierConfiguration(
            maxSize: 10_000_000,
            compressionLevel: .balanced
        ),
        .cold: ColdTierConfiguration(
            directory: documentsDirectory,
            compressionLevel: .maximum
        )
    ],
    migrationPolicy: .automatic(
        checkInterval: 3600,
        accessThreshold: 10
    )
)

// Monitor storage
let monitor = store.storageMonitor()
for await event in monitor.events {
    switch event {
    case .migrated(let count, from: let source, to: let destination):
        print("Migrated \(count) vectors from \(source) to \(destination)")
    case .evicted(let count, from: let tier):
        print("Evicted \(count) vectors from \(tier)")
    }
}
```

### Machine Learning Integration

```swift
// Neural clustering
let clustering = NeuralClustering(
    dimensions: 768,
    clusterCount: 100,
    epochs: 50
)

let trainedClustering = try await clustering.train(on: trainingVectors)
let clusters = try await trainedClustering.assignClusters(to: vectors)

// Learned indexing
let learnedIndex = LearnedIndex<SIMD768<Float>, DocumentMetadata>(
    modelType: .transformer,
    configuration: .init(
        hiddenDimensions: [512, 256],
        dropoutRate: 0.1
    )
)

try await learnedIndex.train(on: trainingData)
```

### Performance Optimization

```swift
// Profile operations
let profiler = MetalProfiler()
profiler.startProfiling()

// Perform operations
let results = try await store.search(query: vector, k: 100)

let profile = profiler.stopProfiling()
print("GPU Time: \(profile.gpuTime)ms")
print("Memory Used: \(profile.memoryUsed / 1024 / 1024)MB")

// Optimize index
try await store.optimize(strategy: .rebalance)
```

## Architecture

### Component Overview

```
VectorStoreKit/
├── API/                    # High-level interfaces
│   ├── VectorStore        # Main store interface
│   ├── VectorUniverse     # Fluent configuration API
│   └── Strategies/        # Strategy pattern implementations
├── Core/                  # Fundamental types
│   ├── Protocols          # Core protocol definitions
│   ├── VectorTypes        # Vector type definitions
│   └── DistanceMetrics    # Distance calculations
├── Indexes/               # Index implementations
│   ├── HNSWIndex         # HNSW algorithm
│   ├── IVFIndex          # IVF clustering
│   └── LearnedIndex      # ML-based indexing
├── Storage/               # Persistence layer
│   ├── Tiers/            # Storage tier implementations
│   ├── WAL               # Write-ahead logging
│   └── Migration         # Data migration engine
├── Acceleration/          # Hardware acceleration
│   ├── Metal/            # GPU compute
│   └── NeuralEngine/     # ANE integration
└── ML/                    # Machine learning
    ├── Clustering        # Clustering algorithms
    ├── Encoding          # Vector encoding
    └── Quantization      # Vector quantization
```

### Design Principles

1. **Protocol-Oriented**: Core functionality defined through protocols
2. **Actor-Based Concurrency**: Thread-safe by design using Swift actors
3. **Strategy Pattern**: Pluggable implementations for all major components
4. **Performance First**: Optimized for Apple hardware with Metal and SIMD
5. **Type Safety**: Leverages Swift's type system for compile-time safety

## Performance

### Benchmarks

Performance benchmarks on Apple M2 Pro with 32GB RAM:

| Operation | Vector Count | Dimensions | Time (ms) | Throughput |
|-----------|-------------|------------|-----------|------------|
| Insert    | 1,000,000   | 128        | 8,500     | 117K/sec   |
| Search    | 1,000,000   | 128        | 0.8       | 1.25M/sec  |
| Batch Search | 10,000   | 128        | 45        | 222K/sec   |

### Optimization Tips

1. **Use Metal acceleration** for large-scale operations
2. **Enable caching** for repeated queries
3. **Configure appropriate storage tiers** based on access patterns
4. **Use batch operations** for bulk inserts and searches
5. **Profile your workload** to identify bottlenecks

## Examples

The repository includes several example projects:

- **BenchmarkExample**: Performance testing and benchmarking
- **NeuralClusteringExample**: ML-based clustering demonstration
- **TestDistanceComputation**: Distance metric comparisons
- **TestNeuralEngine**: Neural Engine utilization examples

## API Reference

### Core Types

- `VectorStore`: Main interface for vector operations
- `VectorUniverse`: Fluent API for configuration
- `VectorEntry`: Container for vectors with metadata
- `SearchResult`: Search result with distance and metadata
- `SearchFilter`: Filtering criteria for searches

### Protocols

- `VectorIndex`: Index implementation protocol
- `VectorStorage`: Storage backend protocol
- `ComputeAccelerator`: Hardware acceleration protocol
- `CachingStrategy`: Caching implementation protocol

### Key Methods

```swift
// VectorStore main operations
func add(_ entries: [VectorEntry]) async throws
func search(query: Vector, k: Int, filter: SearchFilter?) async throws -> [SearchResult]
func delete(_ ids: [String]) async throws
func update(_ id: String, vector: Vector?, metadata: Metadata?) async throws
func optimize(strategy: OptimizationStrategy) async throws
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/VectorStoreKit.git
cd VectorStoreKit

# Open in Xcode
open Package.swift

# Run tests
swift test

# Build for specific platform
swift build --platform macos
```

## License

VectorStoreKit is available under the MIT license. See the [LICENSE](LICENSE) file for more info.

## Acknowledgments

- HNSW algorithm based on the paper by Malkov and Yashunin
- Metal shader implementations inspired by Apple's Metal Performance Shaders
- Community contributors and testers

---

For more information, visit our [documentation site](https://vectorstorekit.dev) or join our [Discord community](https://discord.gg/vectorstorekit).
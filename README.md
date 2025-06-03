# VectorStoreKit

A high-performance, Swift-native vector database optimized for Apple platforms with Metal acceleration.

## Features

- 🚀 **Metal-Accelerated Operations**: GPU-accelerated distance computations and matrix operations
- 🔍 **Advanced Search**: HNSW indexing with multiple distance metrics (Euclidean, Cosine, Dot Product)
- 💾 **Hierarchical Storage**: Intelligent hot/warm/cold tiering with automatic migration
- 🧩 **PipelineKit Integration**: Command-based architecture with middleware support
- 🛡️ **Type-Safe**: Full Swift 6 concurrency support with actor-based design
- 📊 **Rich Analytics**: Built-in performance monitoring and access pattern analysis

## Requirements

- Swift 5.10+
- macOS 14.0+ / iOS 17.0+ / tvOS 17.0+ / watchOS 10.0+ / visionOS 1.0+
- Metal support (optional but recommended for best performance)

## Installation

### Swift Package Manager

Add VectorStoreKit to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/yourusername/VectorStoreKit.git", from: "1.0.0")
]
```

## Quick Start

```swift
import VectorStoreKit
import simd

// Create a vector universe
let universe = VectorUniverse<simd_float4, String>()

// Configure with strategies
let configured = universe
    .index(using: HNSWProductionIndexingStrategy())
    .store(using: HierarchicalProductionStorageStrategy())
    .accelerate(using: MetalAcceleratorStrategy())
    .optimize(using: AdaptiveOptimizationStrategy())

// Build the store
let store = try await configured.build()

// Store vectors
let entry = VectorEntry(
    id: "vec1",
    vector: simd_float4(1.0, 2.0, 3.0, 4.0),
    metadata: "example metadata",
    tier: .hot
)

let result = try await store.add([entry])

// Search for similar vectors
let searchResult = try await store.search(
    query: simd_float4(1.1, 2.1, 3.1, 4.1),
    k: 5
)
```

## PipelineKit Integration

VectorStoreKit includes full PipelineKit integration for command-based operations:

```swift
// Create commands
let storeCommand = StoreEmbeddingCommand(
    embedding: simd_float4(1.0, 2.0, 3.0, 4.0),
    metadata: "document metadata"
)

// Use middleware
let pipeline = Pipeline()
    .use(VectorStoreCachingMiddleware(cache: cache))
    .use(VectorStoreMetricsMiddleware())
    .use(VectorStoreAccessControlMiddleware(accessChecker: checker))

// Execute commands
let result = try await pipeline.execute(storeCommand)
```

## Architecture

VectorStoreKit uses a modular architecture with clear separation of concerns:

- **API Layer**: High-level interfaces and strategy patterns
- **Core**: Fundamental types and protocols
- **Indexes**: Vector indexing implementations (HNSW, etc.)
- **Storage**: Persistence layer with tiered storage
- **Acceleration**: Metal compute shaders and optimizations
- **Caching**: Multi-level caching for performance
- **Pipeline Integration**: Command pattern support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

VectorStoreKit is available under the MIT license. See the LICENSE file for more info.
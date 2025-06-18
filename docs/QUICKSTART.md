# VectorStoreKit Quick Start Guide

Get up and running with VectorStoreKit in minutes! This guide covers installation, basic usage, and common configurations.

## Table of Contents

1. [Installation](#installation)
2. [Basic Vector Store](#basic-vector-store)
3. [Inserting Vectors](#inserting-vectors)
4. [Searching Vectors](#searching-vectors)
5. [Using Different Index Types](#using-different-index-types)
6. [Distributed Setup](#distributed-setup)
7. [Next Steps](#next-steps)

## Installation

### Swift Package Manager

Add VectorStoreKit to your `Package.swift` file:

```swift
// Package.swift
import PackageDescription

let package = Package(
    name: "YourApp",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .watchOS(.v10),
        .visionOS(.v1)
    ],
    dependencies: [
        .package(url: "https://github.com/your-org/VectorStoreKit.git", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "YourApp",
            dependencies: ["VectorStoreKit"]
        )
    ]
)
```

### Xcode

1. Open your project in Xcode
2. Go to **File → Add Package Dependencies**
3. Enter the repository URL: `https://github.com/your-org/VectorStoreKit.git`
4. Select the version and add to your project

### Import

```swift
import VectorStoreKit
import simd  // For SIMD vector types
```

## Basic Vector Store

### Quick Start with Default Configuration

```swift
import VectorStoreKit
import simd

// Define your metadata type
struct DocumentMetadata: Codable, Sendable {
    let title: String
    let author: String
    let timestamp: Date
}

// Create a vector store using the quick start configuration
let store = try await VectorUniverse<SIMD32<Float>, DocumentMetadata>
    .quickStart()
    .materialize()

print("Vector store created and ready!")
```

### Custom Configuration

```swift
// Create a customized vector store
let store = try await VectorUniverse<SIMD128<Float>, DocumentMetadata>()
    .indexing(HNSWIndexingStrategy(maxConnections: 16))
    .storage(HierarchicalResearchStorageStrategy())
    .caching(LRUCachingStrategy(maxMemory: 100_000_000))
    .materialize()
```

## Inserting Vectors

### Single Vector Insertion

```swift
// Create a vector entry
let document = VectorEntry(
    id: "doc_001",
    vector: SIMD32<Float>(
        1.0, 0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7,
        0.4, 0.6, 0.2, 0.8, 0.3, 0.7, 0.5, 0.9,
        0.1, 0.6, 0.4, 0.8, 0.2, 0.7, 0.3, 0.9,
        0.5, 0.1, 0.8, 0.4, 0.6, 0.2, 0.7, 0.3
    ),
    metadata: DocumentMetadata(
        title: "Introduction to Swift",
        author: "Jane Doe",
        timestamp: Date()
    )
)

// Insert the vector
let result = try await store.add([document])
print("Inserted \(result.insertedCount) vectors")
```

### Batch Insertion

```swift
// Generate multiple vectors
var documents: [VectorEntry<SIMD32<Float>, DocumentMetadata>] = []

for i in 0..<1000 {
    // Generate random vector (in practice, use real embeddings)
    var vector = SIMD32<Float>()
    for j in 0..<32 {
        vector[j] = Float.random(in: -1...1)
    }
    
    let entry = VectorEntry(
        id: "doc_\(i)",
        vector: vector,
        metadata: DocumentMetadata(
            title: "Document \(i)",
            author: "Author \(i % 10)",
            timestamp: Date().addingTimeInterval(Double(i) * 3600)
        )
    )
    documents.append(entry)
}

// Batch insert
let result = try await store.add(documents)
print("Inserted \(result.insertedCount) vectors in \(result.totalTime)s")
```

### Insertion Options

```swift
// Fast insertion (no compression, minimal validation)
let fastResult = try await store.add(documents, options: .fast)

// Safe insertion (with compression and validation)
let safeResult = try await store.add(documents, options: .safe)

// Custom options
let customOptions = InsertOptions(
    useCompression: true,
    durabilityLevel: .standard,
    validateIntegrity: true,
    parallel: true
)
let customResult = try await store.add(documents, options: customOptions)
```

## Searching Vectors

### Basic Search

```swift
// Create a query vector
let queryVector = SIMD32<Float>(
    0.9, 0.4, 0.2, 0.7, 0.3, 0.8, 0.1, 0.6,
    0.5, 0.7, 0.3, 0.9, 0.2, 0.6, 0.4, 0.8,
    0.2, 0.5, 0.3, 0.7, 0.1, 0.8, 0.4, 0.9,
    0.6, 0.2, 0.7, 0.3, 0.5, 0.1, 0.8, 0.4
)

// Search for 10 nearest neighbors
let results = try await store.search(
    query: queryVector,
    k: 10
)

// Process results
for result in results.results {
    print("ID: \(result.id)")
    print("Distance: \(result.distance)")
    print("Title: \(result.metadata.title)")
    print("---")
}
```

### Search with Filters

```swift
// Search with metadata filter
let filteredResults = try await store.search(
    query: queryVector,
    k: 10,
    filter: .metadata(
        MetadataFilter(
            key: "author",
            operation: .equals,
            value: "Jane Doe"
        )
    )
)

// Search with date range filter
let dateFilter = SearchFilter.metadata(
    MetadataFilter(
        key: "timestamp",
        operation: .greaterThan,
        value: ISO8601DateFormatter().string(from: Date().addingTimeInterval(-86400))
    )
)

let recentResults = try await store.search(
    query: queryVector,
    k: 20,
    filter: dateFilter
)

// Composite filters
let compositeFilter = SearchFilter.composite(
    CompositeFilter(
        operation: .and,
        filters: [
            .metadata(MetadataFilter(key: "author", operation: .equals, value: "Jane Doe")),
            .metadata(MetadataFilter(key: "title", operation: .contains, value: "Swift"))
        ]
    )
)

let complexResults = try await store.search(
    query: queryVector,
    k: 5,
    filter: compositeFilter
)
```

### Search Strategies

```swift
// Exact search (slower but accurate)
let exactResults = try await store.search(
    query: queryVector,
    k: 10,
    strategy: .exact
)

// Approximate search (faster)
let approxResults = try await store.search(
    query: queryVector,
    k: 10,
    strategy: .approximate
)

// Adaptive search (automatically chooses strategy)
let adaptiveResults = try await store.search(
    query: queryVector,
    k: 10,
    strategy: .adaptive
)
```

## Using Different Index Types

### HNSW Index (High Performance)

```swift
// Configure HNSW for high-performance approximate search
let hnswStore = try await VectorUniverse<SIMD256<Float>, DocumentMetadata>()
    .indexing(HNSWIndexingStrategy(
        maxConnections: 32,          // Higher M = better recall, more memory
        efConstruction: 400,         // Higher = better index quality
        useAdaptiveTuning: true      // Auto-tune parameters
    ))
    .storage(HierarchicalResearchStorageStrategy())
    .caching(LRUCachingStrategy(maxMemory: 500_000_000))
    .materialize()

// HNSW is great for:
// - High recall requirements (>95%)
// - Moderate dataset sizes (up to 10M vectors)
// - Low latency requirements (<10ms)
```

### IVF Index (Scalable)

```swift
// Configure IVF for large-scale datasets
let ivfStore = try await VectorUniverse<SIMD768<Float>, DocumentMetadata>()
    .indexing(IVFIndexingStrategy(
        numberOfCentroids: 1024,     // More centroids = better accuracy
        numberOfProbes: 32,          // More probes = better recall
        quantizer: .product(         // Use product quantization
            subquantizers: 96,
            bits: 8
        )
    ))
    .storage(HierarchicalResearchStorageStrategy())
    .caching(LFUCachingStrategy(maxMemory: 1_000_000_000))
    .materialize()

// Train the IVF index (required before use)
let trainingVectors = generateTrainingData(count: 10_000, dimensions: 768)
if let ivfIndex = ivfStore.index as? IVFIndex<SIMD768<Float>, DocumentMetadata> {
    try await ivfIndex.train(on: trainingVectors)
}

// IVF is great for:
// - Very large datasets (>10M vectors)
// - Tunable recall/speed trade-off
// - Memory-constrained environments
```

### Hybrid Index (Adaptive)

```swift
// Configure Hybrid index for mixed workloads
let hybridStore = try await VectorUniverse<SIMD512<Float>, DocumentMetadata>()
    .indexing(HybridIndexingStrategy(
        routingStrategy: .adaptive,   // ML-based query routing
        adaptiveThreshold: 0.7,      // Confidence threshold
        hnswConfig: HNSWConfiguration(
            dimensions: 512,
            maxConnections: 16,
            efConstruction: 200
        ),
        ivfConfig: IVFConfiguration(
            dimensions: 512,
            numberOfCentroids: 512,
            numberOfProbes: 16
        ),
        learnedConfig: LearnedIndexConfiguration(
            dimensions: 512,
            modelArchitecture: .mlp(hiddenSizes: [256, 128]),
            bucketSize: 100
        )
    ))
    .storage(HierarchicalResearchStorageStrategy())
    .caching(LRUCachingStrategy(maxMemory: 2_000_000_000))
    .materialize()

// Train the hybrid index
if let hybridIndex = hybridStore.index as? HybridIndex<SIMD512<Float>, DocumentMetadata> {
    let trainingData = generateTrainingData(count: 20_000, dimensions: 512)
    try await hybridIndex.train(on: trainingData)
}

// Hybrid is great for:
// - Mixed query patterns
// - Adaptive performance
// - Best of multiple approaches
```

### Learned Index (ML-Powered)

```swift
// Configure Learned index for adaptive performance
let learnedStore = try await VectorUniverse<SIMD384<Float>, DocumentMetadata>()
    .indexing(LearnedIndexingStrategy(
        modelArchitecture: .transformer(heads: 8, layers: 4),
        bucketSize: 200,
        epochs: 50,
        learningRate: 0.001
    ))
    .storage(HierarchicalResearchStorageStrategy())
    .caching(LRUCachingStrategy(maxMemory: 1_500_000_000))
    .materialize()

// The learned index will train itself based on your data patterns
// It adapts to your specific workload over time

// Learned is great for:
// - Workloads with patterns
// - Self-optimizing systems
// - Research and experimentation
```

## Distributed Setup

### Basic Sharding

```swift
// Configure a distributed vector store with sharding
struct ShardedVectorStore {
    let shards: [VectorStore<SIMD256<Float>, DocumentMetadata>]
    let shardCount: Int
    
    init(shardCount: Int = 4) async throws {
        self.shardCount = shardCount
        
        // Create multiple shards
        var shards: [VectorStore<SIMD256<Float>, DocumentMetadata>] = []
        for i in 0..<shardCount {
            let shard = try await VectorUniverse<SIMD256<Float>, DocumentMetadata>()
                .indexing(HNSWIndexingStrategy(maxConnections: 16))
                .storage(HierarchicalResearchStorageStrategy(
                    customConfig: HierarchicalStorage.Configuration(
                        baseDirectory: FileManager.default
                            .temporaryDirectory
                            .appendingPathComponent("vectorstore/shard_\(i)")
                    )
                ))
                .caching(LRUCachingStrategy(maxMemory: 250_000_000))
                .materialize()
            shards.append(shard)
        }
        self.shards = shards
    }
    
    // Shard assignment based on vector ID hash
    func getShard(for id: String) -> VectorStore<SIMD256<Float>, DocumentMetadata> {
        let hash = id.hashValue
        let shardIndex = abs(hash) % shardCount
        return shards[shardIndex]
    }
    
    // Distributed insert
    func add(_ entries: [VectorEntry<SIMD256<Float>, DocumentMetadata>]) async throws {
        // Group entries by shard
        var shardedEntries: [Int: [VectorEntry<SIMD256<Float>, DocumentMetadata>]] = [:]
        for entry in entries {
            let shardIndex = abs(entry.id.hashValue) % shardCount
            shardedEntries[shardIndex, default: []].append(entry)
        }
        
        // Insert in parallel
        await withTaskGroup(of: Void.self) { group in
            for (shardIndex, entries) in shardedEntries {
                group.addTask {
                    try? await self.shards[shardIndex].add(entries)
                }
            }
        }
    }
    
    // Distributed search
    func search(
        query: SIMD256<Float>,
        k: Int,
        filter: SearchFilter? = nil
    ) async throws -> [SearchResult<DocumentMetadata>] {
        // Search all shards in parallel
        let results = await withTaskGroup(
            of: [SearchResult<DocumentMetadata>].self
        ) { group in
            for shard in shards {
                group.addTask {
                    let result = try? await shard.search(
                        query: query,
                        k: k,
                        filter: filter
                    )
                    return result?.results ?? []
                }
            }
            
            // Collect and merge results
            var allResults: [SearchResult<DocumentMetadata>] = []
            for await shardResults in group {
                allResults.append(contentsOf: shardResults)
            }
            return allResults
        }
        
        // Sort and return top-k
        return Array(results.sorted { $0.distance < $1.distance }.prefix(k))
    }
}

// Usage
let distributedStore = try await ShardedVectorStore(shardCount: 4)

// Add vectors (automatically distributed)
try await distributedStore.add(documents)

// Search across all shards
let results = try await distributedStore.search(
    query: queryVector,
    k: 10
)
```

### Replica Configuration

```swift
// Configure replicated storage for high availability
struct ReplicatedVectorStore {
    let primary: VectorStore<SIMD256<Float>, DocumentMetadata>
    let replicas: [VectorStore<SIMD256<Float>, DocumentMetadata>]
    
    init(replicaCount: Int = 2) async throws {
        // Create primary
        self.primary = try await VectorUniverse<SIMD256<Float>, DocumentMetadata>()
            .indexing(HNSWIndexingStrategy(maxConnections: 32))
            .storage(HierarchicalResearchStorageStrategy())
            .caching(LRUCachingStrategy(maxMemory: 1_000_000_000))
            .materialize()
        
        // Create replicas
        var replicas: [VectorStore<SIMD256<Float>, DocumentMetadata>] = []
        for i in 0..<replicaCount {
            let replica = try await VectorUniverse<SIMD256<Float>, DocumentMetadata>()
                .indexing(HNSWIndexingStrategy(maxConnections: 32))
                .storage(HierarchicalResearchStorageStrategy(
                    customConfig: HierarchicalStorage.Configuration(
                        baseDirectory: FileManager.default
                            .temporaryDirectory
                            .appendingPathComponent("vectorstore/replica_\(i)")
                    )
                ))
                .caching(LRUCachingStrategy(maxMemory: 500_000_000))
                .materialize()
            replicas.append(replica)
        }
        self.replicas = replicas
    }
    
    // Write to primary and replicate
    func add(_ entries: [VectorEntry<SIMD256<Float>, DocumentMetadata>]) async throws {
        // Write to primary
        let result = try await primary.add(entries)
        
        // Replicate to secondaries asynchronously
        Task {
            await withTaskGroup(of: Void.self) { group in
                for replica in replicas {
                    group.addTask {
                        try? await replica.add(entries)
                    }
                }
            }
        }
        
        return result
    }
    
    // Read from primary or replicas with load balancing
    func search(
        query: SIMD256<Float>,
        k: Int,
        preferReplica: Bool = false
    ) async throws -> ComprehensiveSearchResult<DocumentMetadata> {
        if preferReplica && !replicas.isEmpty {
            // Round-robin replica selection
            let replica = replicas.randomElement()!
            return try await replica.search(query: query, k: k)
        } else {
            return try await primary.search(query: query, k: k)
        }
    }
}
```

### Partitioning Strategies

```swift
// Configure semantic partitioning
struct SemanticPartitionedStore {
    let partitions: [String: VectorStore<SIMD256<Float>, DocumentMetadata>]
    let router: SemanticRouter
    
    init(categories: [String]) async throws {
        var partitions: [String: VectorStore<SIMD256<Float>, DocumentMetadata>] = [:]
        
        // Create partition for each category
        for category in categories {
            let partition = try await VectorUniverse<SIMD256<Float>, DocumentMetadata>()
                .indexing(HNSWIndexingStrategy(
                    maxConnections: 16,
                    efConstruction: 200
                ))
                .storage(HierarchicalResearchStorageStrategy())
                .caching(LRUCachingStrategy(maxMemory: 200_000_000))
                .materialize()
            partitions[category] = partition
        }
        
        self.partitions = partitions
        self.router = SemanticRouter(categories: categories)
    }
    
    // Route vectors to appropriate partitions
    func add(_ entries: [VectorEntry<SIMD256<Float>, DocumentMetadata>]) async throws {
        // Group by semantic category
        var partitionedEntries: [String: [VectorEntry<SIMD256<Float>, DocumentMetadata>]] = [:]
        
        for entry in entries {
            let category = router.categorize(vector: entry.vector)
            partitionedEntries[category, default: []].append(entry)
        }
        
        // Insert into respective partitions
        await withTaskGroup(of: Void.self) { group in
            for (category, entries) in partitionedEntries {
                if let partition = partitions[category] {
                    group.addTask {
                        try? await partition.add(entries)
                    }
                }
            }
        }
    }
    
    // Search relevant partitions
    func search(
        query: SIMD256<Float>,
        k: Int,
        searchAllPartitions: Bool = false
    ) async throws -> [SearchResult<DocumentMetadata>] {
        if searchAllPartitions {
            // Search all partitions
            return try await searchAllPartitionsInParallel(query: query, k: k)
        } else {
            // Route to most relevant partition
            let category = router.categorize(vector: query)
            if let partition = partitions[category] {
                let result = try await partition.search(query: query, k: k)
                return result.results
            }
            return []
        }
    }
    
    private func searchAllPartitionsInParallel(
        query: SIMD256<Float>,
        k: Int
    ) async throws -> [SearchResult<DocumentMetadata>] {
        let results = await withTaskGroup(
            of: [SearchResult<DocumentMetadata>].self
        ) { group in
            for (_, partition) in partitions {
                group.addTask {
                    let result = try? await partition.search(query: query, k: k)
                    return result?.results ?? []
                }
            }
            
            var allResults: [SearchResult<DocumentMetadata>] = []
            for await partitionResults in group {
                allResults.append(contentsOf: partitionResults)
            }
            return allResults
        }
        
        // Return top-k across all partitions
        return Array(results.sorted { $0.distance < $1.distance }.prefix(k))
    }
}

// Simple semantic router (in practice, use ML model)
struct SemanticRouter {
    let categories: [String]
    
    func categorize(vector: SIMD256<Float>) -> String {
        // Simple example: use first component to determine category
        // In practice, use a trained classifier
        let index = Int(abs(vector[0]) * Float(categories.count)) % categories.count
        return categories[index]
    }
}
```

## Next Steps

### 1. Explore Advanced Features

- **Metal Acceleration**: Enable GPU acceleration for large-scale operations
- **Neural Clustering**: Use ML-based clustering for better organization
- **Product Quantization**: Reduce memory usage with vector compression
- **Custom Distance Metrics**: Implement domain-specific similarity measures

### 2. Performance Optimization

```swift
// Enable Metal acceleration
let metalAccelerated = try await VectorUniverse<SIMD512<Float>, DocumentMetadata>()
    .indexing(HNSWIndexingStrategy(maxConnections: 32))
    .storage(HierarchicalResearchStorageStrategy())
    .caching(LRUCachingStrategy(maxMemory: 1_000_000_000))
    .accelerate(using: MetalComputeAccelerator())
    .materialize()

// Profile your operations
let profiler = MetalProfiler()
profiler.startProfiling()

// Perform operations...

let profile = profiler.stopProfiling()
print("GPU Time: \(profile.gpuTime)ms")
print("Memory Used: \(profile.memoryUsed / 1024 / 1024)MB")
```

### 3. Production Deployment

```swift
// Production-ready configuration
let productionStore = try await VectorUniverse<SIMD768<Float>, DocumentMetadata>
    .production(
        dimensions: 768,
        estimatedVectorCount: 10_000_000
    )
    .materialize()

// Set up monitoring
let monitor = VectorStoreMonitor(store: productionStore)
await monitor.startMonitoring()

// Health checks
let healthCheck = VectorStoreHealthCheck(store: productionStore)
let status = await healthCheck.performHealthCheck()
print("Health status: \(status)")
```

### 4. Learn More

- Read the full [API Documentation](API.md)
- Explore [Examples](../Examples/) in the repository
- Check out [Performance Benchmarks](../Benchmarks/)
- Join our community for support and discussions

## Common Patterns

### Embedding Generation

```swift
// Example: Using Core ML for embeddings
import CoreML
import Vision

func generateEmbedding(for text: String) async throws -> [Float] {
    // Use your embedding model here
    // This is a placeholder - integrate with your actual model
    let embedding = (0..<768).map { _ in Float.random(in: -1...1) }
    return embedding
}

// Batch embedding generation
func generateEmbeddings(for texts: [String]) async throws -> [[Float]] {
    await withTaskGroup(of: [Float].self) { group in
        for text in texts {
            group.addTask {
                try? await generateEmbedding(for: text) ?? []
            }
        }
        
        var embeddings: [[Float]] = []
        for await embedding in group {
            embeddings.append(embedding)
        }
        return embeddings
    }
}
```

### Error Handling

```swift
// Robust error handling
do {
    let results = try await store.search(query: queryVector, k: 10)
    // Process results
} catch VectorStoreError.dimensionMismatch(let expected, let actual) {
    print("Vector dimension mismatch: expected \(expected), got \(actual)")
} catch VectorStoreError.indexFull(let capacity) {
    print("Index is full (capacity: \(capacity))")
} catch {
    print("Unexpected error: \(error)")
}
```

### Batch Processing

```swift
// Process large datasets efficiently
extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0..<Swift.min($0 + size, count)])
        }
    }
}

// Process in batches
let allVectors = loadLargeDataset() // 1M vectors
let batchSize = 1000

for (index, batch) in allVectors.chunked(into: batchSize).enumerated() {
    print("Processing batch \(index + 1)...")
    
    let entries = batch.map { vector in
        VectorEntry(
            id: UUID().uuidString,
            vector: vector,
            metadata: generateMetadata()
        )
    }
    
    try await store.add(entries, options: .fast)
}
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   ```swift
   // Reduce memory usage
   let lowMemoryStore = try await VectorUniverse<SIMD128<Float>, Metadata>()
       .indexing(IVFIndexingStrategy(
           numberOfCentroids: 256,
           quantizer: .product(subquantizers: 16, bits: 8)
       ))
       .storage(HierarchicalResearchStorageStrategy())
       .caching(NoOpCachingStrategy())  // Disable cache
       .materialize()
   ```

2. **Slow Searches**
   ```swift
   // Optimize for speed
   let fastSearchStore = try await VectorUniverse<SIMD256<Float>, Metadata>()
       .indexing(HNSWIndexingStrategy(
           maxConnections: 8,  // Lower M for speed
           efSearch: 20        // Lower ef for speed
       ))
       .storage(InMemoryStorageStrategy())  // All in memory
       .caching(LRUCachingStrategy(maxMemory: 500_000_000))
       .materialize()
   ```

3. **Poor Recall**
   ```swift
   // Optimize for accuracy
   let accurateStore = try await VectorUniverse<SIMD256<Float>, Metadata>()
       .indexing(HNSWIndexingStrategy(
           maxConnections: 48,   // Higher M for recall
           efConstruction: 500,  // Higher construction quality
           efSearch: 200        // Higher search effort
       ))
       .storage(HierarchicalResearchStorageStrategy())
       .caching(LRUCachingStrategy(maxMemory: 1_000_000_000))
       .materialize()
   ```

Happy vector searching!
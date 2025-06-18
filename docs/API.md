# VectorStoreKit API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [API Reference](#api-reference)
   - [VectorStore](#vectorstore)
   - [VectorUniverse](#vectoruniverse)
   - [Indexes](#indexes)
   - [Storage](#storage)
   - [Caching](#caching)
   - [ML Components](#ml-components)
4. [Common Use Cases](#common-use-cases)
5. [Performance Considerations](#performance-considerations)
6. [Best Practices](#best-practices)

## Overview

VectorStoreKit is a high-performance vector database framework designed specifically for Apple platforms. It provides:

- **Type-safe APIs** leveraging Swift's strong type system
- **Actor-based concurrency** for thread-safe operations
- **Hardware acceleration** through Metal and Neural Engine
- **Flexible architecture** with pluggable components
- **Production-ready algorithms** including HNSW, IVF, and learned indexes

### Key Features

- **Multiple Index Types**: HNSW, IVF, Hybrid, and Learned indexes
- **Hierarchical Storage**: Hot, warm, cold, and archive tiers with automatic migration
- **Advanced Caching**: LRU, LFU, FIFO with adaptive policies
- **ML Integration**: Neural clustering, autoencoders, and quantization
- **Metal Acceleration**: GPU-accelerated distance computations and matrix operations

## Core Concepts

### Vectors and Entries

VectorStoreKit works with SIMD vector types for optimal performance:

```swift
// Vector entry represents a vector with associated metadata
public struct VectorEntry<Vector: SIMD & Sendable, Metadata: Codable & Sendable> {
    public let id: VectorID
    public let vector: Vector
    public let metadata: Metadata
    public let tier: StorageTier = .hot
}

// Example usage
let entry = VectorEntry(
    id: "doc_001",
    vector: SIMD32<Float>(...), // 32-dimensional vector
    metadata: ["title": "Document Title", "category": "tech"]
)
```

### Distance Metrics

The framework supports multiple distance metrics:

```swift
public enum DistanceMetric {
    case euclidean      // L2 distance
    case cosine        // Cosine similarity
    case dotProduct    // Dot product similarity
    case manhattan     // L1 distance
    case hamming       // Hamming distance (for binary vectors)
    case custom((Vector, Vector) -> Float)
}
```

### Search Strategies

Different search strategies optimize for various use cases:

```swift
public enum SearchStrategy {
    case exact          // Brute-force exact search
    case approximate    // Fast approximate search
    case adaptive       // Automatically choose based on query
    case hierarchical   // Multi-level search
}
```

### Storage Tiers

Automatic data tiering based on access patterns:

```swift
public enum StorageTier {
    case hot     // In-memory, frequently accessed
    case warm    // SSD-optimized, moderate access
    case cold    // Compressed storage, infrequent access
    case archive // Long-term archival storage
}
```

## API Reference

### VectorStore

The main interface for vector operations:

```swift
public actor VectorStore<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
> {
    // Initialize with components
    public init(
        index: Index,
        storage: Storage,
        cache: Cache,
        configuration: StoreConfiguration = .default
    ) async throws
    
    // Core operations
    public func add(
        _ entries: [VectorEntry<Vector, Metadata>],
        options: InsertOptions = .default
    ) async throws -> DetailedInsertResult
    
    public func search(
        query: Vector,
        k: Int,
        strategy: SearchStrategy = .adaptive,
        filter: SearchFilter? = nil
    ) async throws -> ComprehensiveSearchResult<Metadata>
    
    public func update(
        id: VectorID,
        vector: Vector? = nil,
        metadata: Metadata? = nil
    ) async throws -> UpdateResult
    
    public func delete(id: VectorID) async throws -> DeleteResult
    
    // Advanced operations
    public func optimize(strategy: OptimizationStrategy = .adaptive) async throws
    public func statistics() async -> StoreStatistics
    public func export(format: ExportFormat = .binary) async throws -> Data
}
```

#### Insert Options

```swift
public struct InsertOptions {
    public let useCompression: Bool
    public let durabilityLevel: DurabilityLevel
    public let validateIntegrity: Bool
    public let parallel: Bool
    
    // Predefined configurations
    public static let `default` = InsertOptions()
    public static let fast = InsertOptions(
        useCompression: false,
        durabilityLevel: .none,
        validateIntegrity: false
    )
    public static let safe = InsertOptions(
        useCompression: true,
        durabilityLevel: .strict,
        validateIntegrity: true
    )
}
```

#### Search Filters

```swift
// Metadata filtering
let filter = SearchFilter.metadata(
    MetadataFilter(
        key: "category",
        operation: .equals,
        value: "electronics"
    )
)

// Vector constraints
let vectorFilter = SearchFilter.vector(
    VectorFilter(
        constraint: .magnitude(10.0...50.0)
    )
)

// Composite filters
let composite = SearchFilter.composite(
    CompositeFilter(
        operation: .and,
        filters: [filter, vectorFilter]
    )
)
```

### VectorUniverse

Fluent API for building configured vector stores:

```swift
public struct VectorUniverse<Vector: SIMD & Sendable, Metadata: Codable & Sendable> {
    // Configure indexing strategy
    public func indexing<I: IndexingStrategy>(_ strategy: I) -> IndexedUniverse<...>
    
    // Quick start configurations
    public static func quickStart(
        maxConnections: Int = 16,
        cacheMemory: Int = 100_000_000
    ) -> FullyConfiguredUniverse<...>
    
    public static func research() -> FullyConfiguredUniverse<...>
}

// Example usage
let universe = VectorUniverse<SIMD128<Float>, DocumentMetadata>()
    .indexing(HNSWIndexingStrategy(maxConnections: 32))
    .storage(HierarchicalResearchStorageStrategy())
    .caching(LRUCachingStrategy(maxMemory: 500_000_000))

let store = try await universe.materialize()
```

### Indexes

#### HNSW Index

Hierarchical Navigable Small World index for high-performance approximate search:

```swift
public actor HNSWIndex<Vector: SIMD, Metadata: Codable> {
    public init(configuration: HNSWConfiguration) async throws
    
    // Configuration
    public struct HNSWConfiguration {
        public let dimensions: Int
        public let maxConnections: Int        // M parameter
        public let efConstruction: Int        // Build-time accuracy
        public let efSearch: Int              // Search-time accuracy
        public let seed: Int?
        public let distanceMetric: DistanceMetric
        public let useHeuristic: Bool
        public let extendCandidates: Bool
        public let keepPrunedConnections: Bool
    }
}

// Usage
let config = HNSWConfiguration(
    dimensions: 768,
    maxConnections: 16,
    efConstruction: 200,
    efSearch: 50
)
let index = try await HNSWIndex<SIMD768<Float>, DocMetadata>(configuration: config)
```

#### IVF Index

Inverted File index with clustering for scalable search:

```swift
public actor IVFIndex<Vector: SIMD, Metadata: Codable> {
    public init(configuration: IVFConfiguration) async throws
    
    // Training required before use
    public func train(on samples: [[Float]]) async throws
    
    // Configuration
    public struct IVFConfiguration {
        public let dimensions: Int
        public let numberOfCentroids: Int     // Number of clusters
        public let numberOfProbes: Int        // Clusters to search
        public let quantizer: QuantizerType?
        public let distanceMetric: DistanceMetric
        public let maxIterations: Int
        public let tolerance: Float
    }
}
```

#### Hybrid Index

Combines multiple indexing strategies with intelligent routing:

```swift
public actor HybridIndex<Vector: SIMD, Metadata: Codable> {
    public init(configuration: HybridIndexConfiguration) async throws
    
    // Configuration with routing strategies
    public struct HybridIndexConfiguration {
        public let dimensions: Int
        public let routingStrategy: RoutingStrategy
        public let adaptiveThreshold: Float
        public let hnswConfig: HNSWConfiguration?
        public let ivfConfig: IVFConfiguration?
        public let learnedConfig: LearnedIndexConfiguration?
    }
    
    public enum RoutingStrategy {
        case fixed(primary: IndexType)
        case adaptive                    // ML-based routing
        case sizeBasedHierarchy
        case queryComplexity
    }
}
```

#### Learned Index

Machine learning-based index for adaptive performance:

```swift
public actor LearnedIndex<Vector: SIMD, Metadata: Codable> {
    public init(configuration: LearnedIndexConfiguration) async throws
    
    // Train the index model
    public func train() async throws
    
    // Configuration
    public struct LearnedIndexConfiguration {
        public let dimensions: Int
        public let modelArchitecture: ModelArchitecture
        public let bucketSize: Int
        public let epochs: Int
        public let learningRate: Float
        public let batchSize: Int
    }
    
    public enum ModelArchitecture {
        case linear
        case mlp(hiddenSizes: [Int])
        case transformer(heads: Int, layers: Int)
    }
}
```

### Storage

#### Hierarchical Storage

Automatic tiered storage with migration:

```swift
public actor HierarchicalStorage {
    public init(configuration: Configuration) async throws
    
    public struct Configuration {
        public let hotTierMemoryLimit: Int
        public let warmTierFileSizeLimit: Int
        public let coldTierCompression: CompressionAlgorithm
        public let encryptionSettings: EncryptionSettings
        public let migrationSettings: MigrationSettings
        public let walConfiguration: WALConfiguration
        public let monitoringSettings: MonitoringSettings
        public let baseDirectory: URL
    }
}

// Migration settings
public struct MigrationSettings {
    public let strategy: MigrationStrategy
    public let hotToWarmThreshold: AccessThreshold
    public let warmToColdThreshold: AccessThreshold
    public let coldToArchiveThreshold: AccessThreshold
    
    // Predefined configurations
    public static let aggressive = MigrationSettings(...)
    public static let balanced = MigrationSettings(...)
    public static let conservative = MigrationSettings(...)
    public static let intelligent = MigrationSettings(...)
}
```

#### Storage Monitoring

```swift
public actor StoragePerformanceMonitor {
    // Monitor storage events
    public func startMonitoring() async -> AsyncStream<StorageEvent>
    
    // Get performance metrics
    public func metrics() async -> StorageMetrics
    
    // Storage events
    public enum StorageEvent {
        case migrated(count: Int, from: StorageTier, to: StorageTier)
        case evicted(count: Int, from: StorageTier)
        case compressed(tier: StorageTier, ratio: Float)
        case encrypted(tier: StorageTier)
    }
}
```

### Caching

#### Cache Types

```swift
// LRU Cache
public actor BasicLRUVectorCache<Vector: SIMD> {
    public init(maxMemory: Int) throws
    public func set(id: VectorID, vector: Vector, priority: CachePriority) async
    public func get(id: VectorID) async -> Vector?
}

// LFU Cache
public actor BasicLFUVectorCache<Vector: SIMD> {
    public init(maxMemory: Int) throws
    // Tracks access frequency for eviction
}

// FIFO Cache
public actor BasicFIFOVectorCache<Vector: SIMD> {
    public init(maxMemory: Int) throws
    // Simple first-in-first-out eviction
}
```

#### Cache Configuration

```swift
public protocol CacheConfiguration {
    func memoryBudget() -> Int
    func evictionPolicy() -> EvictionPolicy
}

public enum EvictionPolicy {
    case lru           // Least Recently Used
    case lfu           // Least Frequently Used
    case fifo          // First In First Out
    case arc           // Adaptive Replacement Cache
    case learned       // ML-based eviction
}
```

### ML Components

#### Neural Clustering

```swift
public struct NeuralClustering {
    public init(configuration: NeuralClusteringConfiguration)
    
    // Train clustering model
    public func train(
        vectors: [[Float]],
        epochs: Int = 100
    ) async throws -> TrainedNeuralClustering
    
    // Configuration
    public struct NeuralClusteringConfiguration {
        public let dimensions: Int
        public let clusterCount: Int
        public let encoderArchitecture: [Int]
        public let decoderArchitecture: [Int]
        public let latentDimension: Int
        public let learningRate: Float
        public let regularization: RegularizationConfig
    }
}

// Trained model
public struct TrainedNeuralClustering {
    public func assignCluster(to vector: [Float]) async throws -> Int
    public func getCentroids() async -> [[Float]]
    public func encode(_ vector: [Float]) async throws -> [Float]
}
```

#### Autoencoders

```swift
// Variational Autoencoder
public class VariationalAutoencoder: AutoencoderBase {
    public init(configuration: VAEConfiguration) throws
    
    public struct VAEConfiguration {
        public let inputDimensions: Int
        public let encoderLayers: [Int]
        public let latentDimensions: Int
        public let decoderLayers: [Int]
        public let klWeight: Float
    }
}

// Denoising Autoencoder
public class DenoisingAutoencoder: AutoencoderBase {
    public init(configuration: DenoisingAutoencoderConfiguration) throws
    
    public struct DenoisingAutoencoderConfiguration {
        public let inputDimensions: Int
        public let hiddenLayers: [Int]
        public let noiseLevel: Float
        public let noiseType: NoiseType
    }
}
```

#### Product Quantization

```swift
public actor OptimizedProductQuantizer {
    public init(configuration: ProductQuantizationConfig) async throws
    
    // Train quantizer
    public func train(on vectors: [[Float]]) async throws
    
    // Quantize vectors
    public func quantize(_ vector: [Float]) async throws -> QuantizedVector
    
    // Configuration
    public struct ProductQuantizationConfig {
        public let dimensions: Int
        public let numberOfSubquantizers: Int
        public let bitsPerSubquantizer: Int
        public let distanceMetric: DistanceMetric
        public let trainingIterations: Int
        public let optimizationMethod: OptimizationMethod
    }
}
```

## Common Use Cases

### 1. Semantic Search Application

```swift
// Configure for semantic search with large language model embeddings
let universe = VectorUniverse<SIMD768<Float>, DocumentMetadata>()
    .indexing(HNSWIndexingStrategy(
        maxConnections: 32,
        efConstruction: 400,
        useAdaptiveTuning: true
    ))
    .storage(HierarchicalResearchStorageStrategy(
        customConfig: HierarchicalStorage.Configuration(
            hotTierMemoryLimit: 2_000_000_000,  // 2GB hot tier
            warmTierFileSizeLimit: 50_000_000_000,  // 50GB warm tier
            coldTierCompression: .zstd,
            encryptionSettings: .none,
            migrationSettings: .intelligent
        )
    ))
    .caching(LRUCachingStrategy(maxMemory: 500_000_000))

let store = try await universe.materialize()

// Add documents
let documents = loadDocuments()
let embeddings = await generateEmbeddings(documents)

let entries = zip(documents, embeddings).map { doc, embedding in
    VectorEntry(
        id: doc.id,
        vector: SIMD768<Float>(embedding),
        metadata: DocumentMetadata(
            title: doc.title,
            content: doc.content,
            timestamp: doc.timestamp,
            category: doc.category
        )
    )
}

let result = try await store.add(entries, options: .default)
print("Indexed \(result.insertedCount) documents")

// Search with filtering
let queryEmbedding = await generateEmbedding("How to optimize Swift performance?")
let results = try await store.search(
    query: SIMD768<Float>(queryEmbedding),
    k: 10,
    filter: .metadata(
        MetadataFilter(key: "category", operation: .equals, value: "programming")
    )
)
```

### 2. Image Similarity Search

```swift
// Configure for image feature vectors
let imageStore = try await VectorUniverse<SIMD512<Float>, ImageMetadata>()
    .indexing(IVFIndexingStrategy(
        numberOfCentroids: 1024,
        numberOfProbes: 32,
        quantizer: .product(subquantizers: 64, bits: 8)
    ))
    .storage(HierarchicalResearchStorageStrategy())
    .caching(LFUCachingStrategy(maxMemory: 1_000_000_000))
    .materialize()

// Train IVF index
let trainingImages = loadTrainingImages()
let trainingFeatures = await extractFeatures(trainingImages)
if let ivfIndex = imageStore.index as? IVFIndex {
    try await ivfIndex.train(on: trainingFeatures)
}

// Add images
for batch in imageBatches {
    let features = await extractFeatures(batch)
    let entries = zip(batch, features).map { image, feature in
        VectorEntry(
            id: image.id,
            vector: SIMD512<Float>(feature),
            metadata: ImageMetadata(
                filename: image.filename,
                width: image.width,
                height: image.height,
                tags: image.tags
            )
        )
    }
    try await imageStore.add(entries)
}

// Visual search
let queryImage = loadQueryImage()
let queryFeatures = await extractFeatures([queryImage])[0]
let similar = try await imageStore.search(
    query: SIMD512<Float>(queryFeatures),
    k: 20,
    strategy: .approximate
)
```

### 3. Real-time Recommendation System

```swift
// Configure for high-throughput recommendations
let recommender = try await VectorUniverse<SIMD256<Float>, UserItemMetadata>()
    .indexing(HybridIndexingStrategy(
        routingStrategy: .adaptive,
        hnswConfig: HNSWConfiguration(
            dimensions: 256,
            maxConnections: 16,
            efConstruction: 200
        ),
        learnedConfig: LearnedIndexConfiguration(
            dimensions: 256,
            modelArchitecture: .mlp(hiddenSizes: [128, 64]),
            bucketSize: 100
        )
    ))
    .storage(InMemoryStorageStrategy())  // All in memory for speed
    .caching(NoOpCachingStrategy())  // No caching needed
    .materialize()

// Continuous updates
let updateStream = getUserInteractionStream()
for await interaction in updateStream {
    // Update user embedding based on interaction
    let newEmbedding = await updateUserEmbedding(
        userId: interaction.userId,
        itemId: interaction.itemId,
        action: interaction.action
    )
    
    try await recommender.update(
        id: interaction.userId,
        vector: SIMD256<Float>(newEmbedding)
    )
    
    // Get real-time recommendations
    let recommendations = try await recommender.search(
        query: SIMD256<Float>(newEmbedding),
        k: 10,
        strategy: .adaptive
    )
    
    // Send recommendations
    await sendRecommendations(
        userId: interaction.userId,
        items: recommendations.map { $0.id }
    )
}
```

### 4. Multi-modal Search

```swift
// Configure for combined text and image search
struct MultiModalMetadata: Codable, Sendable {
    let textEmbedding: [Float]
    let imageEmbedding: [Float]
    let combinedWeight: Float
    let sourceType: SourceType
    let content: String
    
    enum SourceType: String, Codable {
        case text, image, video, audio
    }
}

let multiModalStore = try await VectorUniverse<SIMD1024<Float>, MultiModalMetadata>()
    .indexing(LearnedIndexingStrategy(
        modelArchitecture: .transformer(heads: 8, layers: 4),
        bucketSize: 200,
        epochs: 50
    ))
    .storage(HierarchicalResearchStorageStrategy())
    .caching(LRUCachingStrategy(maxMemory: 2_000_000_000))
    .materialize()

// Index multi-modal content
let content = loadMultiModalContent()
for item in content {
    let textEmb = await generateTextEmbedding(item.text)
    let imageEmb = await generateImageEmbedding(item.image)
    let combined = combineEmbeddings(text: textEmb, image: imageEmb, weight: 0.5)
    
    let entry = VectorEntry(
        id: item.id,
        vector: SIMD1024<Float>(combined),
        metadata: MultiModalMetadata(
            textEmbedding: textEmb,
            imageEmbedding: imageEmb,
            combinedWeight: 0.5,
            sourceType: .video,
            content: item.description
        )
    )
    try await multiModalStore.add([entry])
}

// Cross-modal search
func crossModalSearch(text: String? = nil, image: UIImage? = nil) async throws -> [SearchResult<MultiModalMetadata>] {
    var queryEmbedding: [Float]
    
    if let text = text, let image = image {
        let textEmb = await generateTextEmbedding(text)
        let imageEmb = await generateImageEmbedding(image)
        queryEmbedding = combineEmbeddings(text: textEmb, image: imageEmb, weight: 0.5)
    } else if let text = text {
        let textEmb = await generateTextEmbedding(text)
        queryEmbedding = padEmbedding(textEmb, to: 1024)
    } else if let image = image {
        let imageEmb = await generateImageEmbedding(image)
        queryEmbedding = padEmbedding(imageEmb, to: 1024)
    } else {
        throw SearchError.noQuery
    }
    
    return try await multiModalStore.search(
        query: SIMD1024<Float>(queryEmbedding),
        k: 20,
        strategy: .adaptive
    )
}
```

## Performance Considerations

### 1. Hardware Acceleration

VectorStoreKit automatically leverages available hardware:

```swift
// Metal acceleration for distance computations
let metalConfig = MetalComputeConfiguration(
    preferredDevice: .default,
    commandQueueType: .concurrent,
    maxBuffersInFlight: 3,
    enableProfiling: true
)

// Configure Metal compute
let metalCompute = try await MetalCompute(configuration: metalConfig)

// Profile GPU operations
let profiler = MetalProfiler()
profiler.startProfiling()

// Perform operations...

let profile = profiler.stopProfiling()
print("GPU Utilization: \(profile.gpuUtilization)%")
print("Memory Bandwidth: \(profile.memoryBandwidth) GB/s")
```

### 2. Batch Processing

Optimize throughput with batching:

```swift
// Batch configuration
let batchProcessor = BatchProcessor<SIMD512<Float>>(
    batchSize: 1000,
    maxConcurrentBatches: 4,
    enablePrefetching: true
)

// Process large dataset
let vectors = loadLargeDataset() // 10M vectors
await batchProcessor.process(vectors) { batch in
    let entries = batch.enumerated().map { index, vector in
        VectorEntry(
            id: "vec_\(index)",
            vector: vector,
            metadata: generateMetadata(index)
        )
    }
    try await store.add(entries, options: .fast)
}
```

### 3. Memory Management

Control memory usage across components:

```swift
// Configure memory limits
let memoryConfig = MemoryConfiguration(
    indexMemoryLimit: 4_000_000_000,      // 4GB for index
    cacheMemoryLimit: 1_000_000_000,      // 1GB for cache
    bufferPoolSize: 500_000_000,          // 500MB for buffers
    enableMemoryPressureHandling: true
)

// Monitor memory usage
let monitor = MemoryMonitor()
await monitor.startMonitoring { event in
    switch event {
    case .pressureWarning(let level):
        print("Memory pressure: \(level)")
        // Reduce cache size or trigger garbage collection
    case .limitExceeded(let component):
        print("Memory limit exceeded: \(component)")
        // Take corrective action
    }
}
```

### 4. Query Optimization

Optimize search performance:

```swift
// Search configuration for different scenarios
struct SearchOptimization {
    // High recall, slower
    static let highQuality = SearchParameters(
        efSearch: 200,
        probes: 64,
        rerank: true,
        expansionFactor: 2.0
    )
    
    // Balanced performance
    static let balanced = SearchParameters(
        efSearch: 50,
        probes: 16,
        rerank: false,
        expansionFactor: 1.0
    )
    
    // Maximum speed
    static let fast = SearchParameters(
        efSearch: 10,
        probes: 4,
        rerank: false,
        expansionFactor: 0.5
    )
}

// Adaptive search based on query complexity
func adaptiveSearch(query: Vector, k: Int) async throws -> [SearchResult] {
    let complexity = analyzeQueryComplexity(query)
    
    let params = switch complexity {
    case .simple: SearchOptimization.fast
    case .moderate: SearchOptimization.balanced
    case .complex: SearchOptimization.highQuality
    }
    
    return try await store.search(
        query: query,
        k: k,
        parameters: params
    )
}
```

### 5. Concurrent Operations

Handle concurrent access efficiently:

```swift
// Concurrent read/write patterns
await withTaskGroup(of: Result<Void, Error>.self) { group in
    // Multiple concurrent searches
    for query in queries {
        group.addTask {
            do {
                let results = try await store.search(query: query, k: 10)
                await processResults(results)
                return .success(())
            } catch {
                return .failure(error)
            }
        }
    }
    
    // Concurrent inserts
    for batch in vectorBatches {
        group.addTask {
            do {
                try await store.add(batch, options: .fast)
                return .success(())
            } catch {
                return .failure(error)
            }
        }
    }
    
    // Collect results
    for await result in group {
        if case .failure(let error) = result {
            print("Operation failed: \(error)")
        }
    }
}
```

## Best Practices

### 1. Index Selection

Choose the right index for your use case:

| Index Type | Best For | Trade-offs |
|------------|----------|------------|
| **HNSW** | High recall, moderate dataset size | Higher memory usage |
| **IVF** | Large datasets, tunable recall | Requires training |
| **Hybrid** | Mixed query patterns | More complex |
| **Learned** | Adaptive performance | Training overhead |

```swift
// Decision matrix
func selectIndex(datasetSize: Int, recallRequirement: Float, latencyBudget: TimeInterval) -> IndexingStrategy {
    switch (datasetSize, recallRequirement, latencyBudget) {
    case (0...100_000, 0.95..., _):
        return HNSWIndexingStrategy(maxConnections: 32)
    case (100_000...10_000_000, _, 0...0.01):
        return IVFIndexingStrategy(numberOfCentroids: 1024, numberOfProbes: 16)
    case (_, _, _):
        return HybridIndexingStrategy(routingStrategy: .adaptive)
    }
}
```

### 2. Vector Preprocessing

Normalize and validate vectors before indexing:

```swift
extension SIMD where Scalar: FloatingPoint {
    // L2 normalization
    func normalized() -> Self {
        let magnitude = sqrt((self * self).sum())
        return magnitude > 0 ? self / magnitude : self
    }
    
    // Validation
    func isValid() -> Bool {
        return !self.contains { $0.isNaN || $0.isInfinite }
    }
}

// Preprocessing pipeline
func preprocessVectors(_ vectors: [[Float]]) -> [SIMD512<Float>] {
    vectors.compactMap { vector in
        guard vector.count == 512 else {
            print("Invalid dimension: \(vector.count)")
            return nil
        }
        
        let simd = SIMD512<Float>(vector)
        guard simd.isValid() else {
            print("Invalid vector values")
            return nil
        }
        
        return simd.normalized()
    }
}
```

### 3. Metadata Design

Structure metadata for efficient filtering:

```swift
// Good: Structured, typed metadata
struct ProductMetadata: Codable, Sendable {
    let id: UUID
    let name: String
    let category: Category
    let price: Decimal
    let tags: Set<String>
    let attributes: [String: String]
    
    enum Category: String, Codable, CaseIterable {
        case electronics, clothing, books, home, sports
    }
}

// Efficient filtering
let filter = SearchFilter.composite(
    CompositeFilter(
        operation: .and,
        filters: [
            .metadata(MetadataFilter(key: "category", operation: .equals, value: "electronics")),
            .metadata(MetadataFilter(key: "price", operation: .lessThan, value: "500"))
        ]
    )
)
```

### 4. Error Handling

Implement comprehensive error handling:

```swift
enum VectorStoreOperation {
    case insert([VectorEntry])
    case search(Vector, k: Int)
    case delete(VectorID)
    
    func execute(on store: VectorStore) async throws {
        do {
            switch self {
            case .insert(let entries):
                let result = try await store.add(entries)
                if result.errorCount > 0 {
                    print("Partial failure: \(result.errorCount) errors")
                    for error in result.errors {
                        handleError(error)
                    }
                }
            case .search(let query, let k):
                let results = try await store.search(query: query, k: k)
                validateSearchResults(results)
            case .delete(let id):
                let result = try await store.delete(id: id)
                if !result.success {
                    throw VectorStoreError.deletionFailed(id)
                }
            }
        } catch {
            switch error {
            case VectorStoreError.dimensionMismatch(let expected, let actual):
                print("Dimension mismatch: expected \(expected), got \(actual)")
            case VectorStoreError.indexFull:
                print("Index capacity reached, consider scaling")
            case VectorStoreError.storageError(let message):
                print("Storage error: \(message)")
            default:
                print("Unexpected error: \(error)")
            }
            throw error
        }
    }
}
```

### 5. Monitoring and Observability

Implement comprehensive monitoring:

```swift
// Performance monitoring
class VectorStoreMonitor {
    let store: VectorStore
    let metrics = MetricsCollector()
    
    func startMonitoring() async {
        // Periodic statistics collection
        Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { _ in
            Task {
                let stats = await self.store.statistics()
                self.metrics.record("vector_count", value: stats.vectorCount)
                self.metrics.record("memory_usage", value: stats.memoryUsage)
                self.metrics.record("cache_hit_rate", value: stats.cacheStatistics?.hitRate ?? 0)
                
                // Check health
                if stats.memoryUsage > self.memoryThreshold {
                    await self.handleHighMemoryUsage()
                }
            }
        }
        
        // Real-time operation tracking
        await store.setOperationCallback { operation, duration in
            self.metrics.recordOperation(operation, duration: duration)
        }
    }
    
    func generateReport() -> PerformanceReport {
        return PerformanceReport(
            averageSearchLatency: metrics.average("search_latency"),
            averageInsertLatency: metrics.average("insert_latency"),
            cacheHitRate: metrics.latest("cache_hit_rate"),
            memoryEfficiency: calculateMemoryEfficiency(),
            recommendations: generateRecommendations()
        )
    }
}
```

### 6. Testing Strategies

Comprehensive testing approach:

```swift
// Performance regression testing
class VectorStorePerformanceTests: XCTestCase {
    func testSearchPerformance() async throws {
        let store = try await createTestStore()
        let vectors = generateTestVectors(count: 100_000, dimensions: 768)
        
        // Baseline performance
        try await store.add(vectors)
        
        measure {
            let expectation = self.expectation(description: "Search completes")
            Task {
                let results = try await store.search(
                    query: vectors[0].vector,
                    k: 100
                )
                XCTAssertEqual(results.count, 100)
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 1.0)
        }
    }
    
    // Accuracy testing
    func testSearchAccuracy() async throws {
        let store = try await createTestStore()
        let dataset = loadGroundTruthDataset()
        
        // Index dataset
        try await store.add(dataset.vectors)
        
        // Test recall
        var totalRecall = 0.0
        for query in dataset.queries {
            let results = try await store.search(query: query.vector, k: 10)
            let recall = calculateRecall(
                results: results.map { $0.id },
                groundTruth: query.groundTruth
            )
            totalRecall += recall
        }
        
        let averageRecall = totalRecall / Double(dataset.queries.count)
        XCTAssertGreaterThan(averageRecall, 0.95, "Recall should be > 95%")
    }
}
```

### 7. Production Deployment

Production-ready configuration:

```swift
// Production configuration
extension VectorUniverse {
    static func production<V: SIMD, M: Codable & Sendable>(
        dimensions: Int,
        estimatedVectorCount: Int
    ) -> FullyConfiguredUniverse<V, M, HybridIndexingStrategy<V, M>, HierarchicalResearchStorageStrategy, LRUCachingStrategy<V>> {
        // Calculate optimal parameters
        let indexParams = calculateOptimalIndexParams(
            dimensions: dimensions,
            vectorCount: estimatedVectorCount
        )
        
        return VectorUniverse<V, M>()
            .indexing(HybridIndexingStrategy(
                routingStrategy: .adaptive,
                hnswConfig: HNSWConfiguration(
                    dimensions: dimensions,
                    maxConnections: indexParams.m,
                    efConstruction: indexParams.efConstruction
                ),
                ivfConfig: IVFConfiguration(
                    dimensions: dimensions,
                    numberOfCentroids: indexParams.centroids,
                    numberOfProbes: indexParams.probes
                )
            ))
            .storage(HierarchicalResearchStorageStrategy(
                customConfig: HierarchicalStorage.Configuration(
                    hotTierMemoryLimit: calculateHotTierSize(estimatedVectorCount),
                    warmTierFileSizeLimit: calculateWarmTierSize(estimatedVectorCount),
                    coldTierCompression: .zstd,
                    encryptionSettings: .chacha20,
                    migrationSettings: .intelligent,
                    walConfiguration: .highDurability,
                    monitoringSettings: .comprehensive
                )
            ))
            .caching(LRUCachingStrategy(
                maxMemory: calculateOptimalCacheSize(estimatedVectorCount)
            ))
    }
}

// Health checks
class VectorStoreHealthCheck {
    let store: VectorStore
    
    func performHealthCheck() async -> HealthStatus {
        do {
            // Test basic operations
            let testVector = generateTestVector()
            let testEntry = VectorEntry(id: "health_check", vector: testVector, metadata: [:])
            
            // Test insert
            let insertResult = try await store.add([testEntry])
            guard insertResult.insertedCount == 1 else {
                return .unhealthy("Insert failed")
            }
            
            // Test search
            let searchResults = try await store.search(query: testVector, k: 1)
            guard !searchResults.results.isEmpty else {
                return .unhealthy("Search failed")
            }
            
            // Test delete
            let deleteResult = try await store.delete(id: "health_check")
            guard deleteResult.success else {
                return .unhealthy("Delete failed")
            }
            
            // Check resource usage
            let stats = await store.statistics()
            if stats.memoryUsage > memoryLimit {
                return .degraded("High memory usage")
            }
            
            return .healthy
        } catch {
            return .unhealthy("Health check error: \(error)")
        }
    }
}
```

## Summary

VectorStoreKit provides a comprehensive, production-ready vector database solution for Apple platforms. Key takeaways:

1. **Use the VectorUniverse API** for easy configuration with sensible defaults
2. **Choose the right index** based on your dataset size and performance requirements  
3. **Leverage hardware acceleration** with Metal for optimal performance
4. **Implement proper error handling** and monitoring for production deployments
5. **Follow preprocessing best practices** to ensure data quality
6. **Test thoroughly** including performance, accuracy, and edge cases

For more examples and detailed implementations, see the [Examples](../Examples/) directory in the repository.
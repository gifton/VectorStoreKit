// VectorStoreKit: Vector Universe Builder
//
// Fluent API for building sophisticated vector storage systems

import Foundation
import simd

// MARK: - Supporting Types

/// Data size categories for optimization
public enum DataSize: String, Sendable {
    case small = "small"     // < 100K vectors
    case medium = "medium"   // 100K - 1M vectors
    case large = "large"     // 1M - 10M vectors
    case xlarge = "xlarge"   // > 10M vectors
    
    var cacheSize: Int {
        switch self {
        case .small: return 100_000_000    // 100MB
        case .medium: return 500_000_000   // 500MB
        case .large: return 1_000_000_000  // 1GB
        case .xlarge: return 2_000_000_000 // 2GB
        }
    }
}

// MARK: - Vector Universe Builder

/// Entry point for building vector stores using a fluent API
///
/// VectorUniverse provides a type-safe, composable way to configure
/// and instantiate vector stores with specific strategies. It uses a
/// builder pattern to ensure all required components are configured
/// before creating the final store.
///
/// ## Example
/// ```swift
/// let store = try await VectorUniverse<SIMD768<Float>, DocumentMetadata>()
///     .indexing(HNSWIndexingStrategy(maxConnections: 32))
///     .storage(ResearchStorageStrategy())
///     .caching(LRUCachingStrategy(maxMemory: 500_000_000))
///     .materialize()
/// ```
///
/// ## Quick Start Options
/// ```swift
/// // For rapid prototyping
/// let store = try await VectorUniverse<SIMD128<Float>, MyMetadata>
///     .quickStart()
///     .materialize()
///
/// // For research applications
/// let store = try await VectorUniverse<SIMD768<Float>, MyMetadata>
///     .research()
///     .materialize()
/// ```
public struct VectorUniverse<Vector: SIMD & Sendable, Metadata: Codable & Sendable> where Vector.Scalar: BinaryFloatingPoint {
    
    private let configuration: UniverseConfiguration
    
    /// Create a new vector universe with optional configuration
    ///
    /// - Parameter config: Universe configuration (uses defaults if not provided)
    public init(config: UniverseConfiguration = .init()) {
        self.configuration = config
    }
    
    /// Configure the indexing strategy for vector search
    ///
    /// - Parameter strategy: The indexing strategy to use (HNSW, IVF, Hybrid, or Learned)
    /// - Returns: An `IndexedUniverse` ready for storage configuration
    ///
    /// ## Available Strategies
    /// - `HNSWIndexingStrategy`: High-performance approximate search
    /// - `IVFIndexingStrategy`: Scalable inverted file index
    public func indexing<I: IndexingStrategy>(
        _ strategy: I
    ) -> IndexedUniverse<Vector, Metadata, I> 
    where I.IndexType.Vector == Vector, I.IndexType.Metadata == Metadata {
        IndexedUniverse(
            configuration: configuration,
            indexingStrategy: strategy
        )
    }
}

// MARK: - Quick Presets

public extension VectorUniverse {
    /// Quick start configuration with simple memory storage for rapid prototyping
    /// - Returns: A fully configured universe optimized for quick development
    func quickStart() -> FullyConfiguredUniverse<
        Vector,
        Metadata,
        HNSWIndexingStrategy<Vector, Metadata>,
        SimpleStorageStrategy,
        NoOpCachingStrategy<Vector>
    > {
        self.indexing(HNSWIndexingStrategy<Vector, Metadata>(
            maxConnections: 16,
            efConstruction: 200
        ))
        .storage(SimpleStorageStrategy())
        .caching(NoOpCachingStrategy<Vector>())
    }
    
    /// Research configuration with all features enabled
    /// - Returns: A fully configured universe optimized for research with adaptive strategies
    func research() -> FullyConfiguredUniverse<
        Vector, 
        Metadata,
        IVFIndexingStrategy<Vector, Metadata>,
        ResearchStorageStrategy,
        LRUCachingStrategy<Vector>
    > {
        self.indexing(IVFIndexingStrategy<Vector, Metadata>(
            numClusters: 1024,
            nprobe: 16
        ))
            .storage(ResearchStorageStrategy())
            .caching(LRUCachingStrategy<Vector>())
    }
    
    /// Production configuration optimized for reliability and performance
    /// - Parameter dataSize: Expected data size to optimize configuration
    /// - Returns: A fully configured universe optimized for production use
    func production(
        dataSize: DataSize = .medium
    ) -> FullyConfiguredUniverse<
        Vector,
        Metadata,
        HNSWIndexingStrategy<Vector, Metadata>,
        ThreeTierStorageStrategy,
        LRUCachingStrategy<Vector>
    > {
        let (m, efConstruction) = dataSize.hnswParameters()
        
        return self.indexing(HNSWIndexingStrategy<Vector, Metadata>(
            maxConnections: m,
            efConstruction: efConstruction
        ))
        .storage(ThreeTierStorageStrategy())
        .caching(LRUCachingStrategy<Vector>(maxMemory: dataSize.cacheSize))
    }
    
    /// Enterprise configuration with 3-tier storage and automatic migration
    /// - Parameter config: Optional 3-tier storage configuration
    /// - Returns: A fully configured universe optimized for enterprise use
    func enterprise(
        storageConfig: ThreeTierStorageConfiguration? = nil
    ) -> FullyConfiguredUniverse<
        Vector,
        Metadata,
        HNSWIndexingStrategy<Vector, Metadata>,
        ThreeTierStorageStrategy,
        LRUCachingStrategy<Vector>
    > {
        self.indexing(HNSWIndexingStrategy<Vector, Metadata>(
            maxConnections: 32,
            efConstruction: 800
        ))
        .storage(ThreeTierStorageStrategy(configuration: storageConfig))
        .caching(LRUCachingStrategy<Vector>())
    }
    
    /// Embedded configuration for on-device usage with minimal resources
    /// - Parameter maxMemory: Maximum memory allocation in bytes
    /// - Returns: A fully configured universe optimized for embedded/mobile use
    func embedded(
        maxMemory: Int = 100_000_000 // 100MB
    ) -> FullyConfiguredUniverse<
        Vector,
        Metadata,
        HNSWIndexingStrategy<Vector, Metadata>,
        InMemoryPerformanceStorageStrategy,
        NoOpCachingStrategy<Vector>
    > {
        self.indexing(HNSWIndexingStrategy<Vector, Metadata>(
            maxConnections: 8,
            efConstruction: 100
        ))
        .storage(InMemoryPerformanceStorageStrategy())
        .caching(NoOpCachingStrategy<Vector>())
    }
}

// MARK: - DataSize Extensions

private extension DataSize {
    /// Get recommended HNSW parameters for data size
    func hnswParameters() -> (m: Int, efConstruction: Int) {
        switch self {
        case .small:
            return (m: 12, efConstruction: 100)
        case .medium:
            return (m: 16, efConstruction: 200)
        case .large:
            return (m: 24, efConstruction: 400)
        case .xlarge:
            return (m: 32, efConstruction: 800)
        }
    }
}

// MARK: - Indexed Universe

/// A vector universe with indexing configured
public struct IndexedUniverse<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    IndexStrategy: IndexingStrategy
>
where 
    Vector.Scalar: BinaryFloatingPoint,
    IndexStrategy.IndexType.Vector == Vector,
    IndexStrategy.IndexType.Metadata == Metadata 
{
    
    let configuration: UniverseConfiguration
    let indexingStrategy: IndexStrategy
    
    /// Configure storage strategy
    public func storage<S: StorageStrategy>(
        _ strategy: S
    ) -> StorageConfiguredUniverse<Vector, Metadata, IndexStrategy, S> {
        StorageConfiguredUniverse(
            configuration: configuration,
            indexingStrategy: indexingStrategy,
            storageStrategy: strategy
        )
    }
}

// MARK: - Storage Configured Universe

/// A vector universe with indexing and storage configured
public struct StorageConfiguredUniverse<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    IndexStrategy: IndexingStrategy,
    Storage: StorageStrategy
>
where 
    Vector.Scalar: BinaryFloatingPoint,
    IndexStrategy.IndexType.Vector == Vector,
    IndexStrategy.IndexType.Metadata == Metadata 
{
    
    let configuration: UniverseConfiguration
    let indexingStrategy: IndexStrategy
    let storageStrategy: Storage
    
    /// Configure caching strategy
    public func caching<C: CachingStrategy>(
        _ strategy: C
    ) -> FullyConfiguredUniverse<Vector, Metadata, IndexStrategy, Storage, C>
    where C.CacheType.Vector == Vector {
        FullyConfiguredUniverse(
            configuration: configuration,
            indexingStrategy: indexingStrategy,
            storageStrategy: storageStrategy,
            cachingStrategy: strategy
        )
    }
    
    /// Use default caching (no cache)
    public func withoutCache() -> FullyConfiguredUniverse<Vector, Metadata, IndexStrategy, Storage, NoOpCachingStrategy<Vector>> {
        FullyConfiguredUniverse(
            configuration: configuration,
            indexingStrategy: indexingStrategy,
            storageStrategy: storageStrategy,
            cachingStrategy: NoOpCachingStrategy<Vector>()
        )
    }
}

// MARK: - Fully Configured Universe

/// A fully configured vector universe ready for materialization
public struct FullyConfiguredUniverse<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    IndexStrategy: IndexingStrategy,
    Storage: StorageStrategy,
    Caching: CachingStrategy
>
where 
    Vector.Scalar: BinaryFloatingPoint,
    IndexStrategy.IndexType.Vector == Vector,
    IndexStrategy.IndexType.Metadata == Metadata,
    Caching.CacheType.Vector == Vector 
{
    
    let configuration: UniverseConfiguration
    let indexingStrategy: IndexStrategy
    let storageStrategy: Storage
    let cachingStrategy: Caching
    
    /// Create the configured vector store
    public func materialize() async throws -> VectorStore<
        Vector,
        Metadata,
        IndexStrategy.IndexType,
        Storage.BackendType,
        Caching.CacheType
    > {
        // Create index
        let index = try await indexingStrategy.createIndex()
        
        // Create storage backend
        let storage = try await storageStrategy.createBackend(configuration: storageStrategy.defaultConfiguration())
        
        // Create cache
        let cache = try await cachingStrategy.createCache()
        
        // Create store configuration
        let storeConfig = configuration.buildStoreConfiguration()
        
        // Create and return the vector store
        return try await VectorStore(
            index: index,
            storage: storage,
            cache: cache,
            configuration: storeConfig
        )
    }
}

// MARK: - Strategy Protocols

/// Protocol for caching strategies
public protocol CachingStrategy {
    associatedtype CacheType: VectorCache
    associatedtype Config: CacheConfiguration = CacheType.Configuration
    
    var configuration: Config? { get }
    
    func createCache() async throws -> CacheType
}

// MARK: - Built-in Caching Strategies

/// No-operation caching strategy
public struct NoOpCachingStrategy<Vector: SIMD & Sendable>: CachingStrategy where Vector.Scalar: BinaryFloatingPoint {
    public typealias CacheType = NoOpVectorCache<Vector>
    
    public let configuration: NoOpCacheConfiguration? = nil
    
    public init() {}
    
    public func createCache() async throws -> NoOpVectorCache<Vector> {
        NoOpVectorCache<Vector>()
    }
}

/// LRU caching strategy
public struct LRUCachingStrategy<Vector: SIMD & Sendable>: CachingStrategy where Vector.Scalar: BinaryFloatingPoint {
    public typealias CacheType = BasicLRUVectorCache<Vector>
    public typealias Config = LRUCacheConfiguration
    
    public let configuration: LRUCacheConfiguration?
    
    public init(maxMemory: Int = 100_000_000) {
        self.configuration = LRUCacheConfiguration(maxMemory: maxMemory)
    }
    
    public func createCache() async throws -> BasicLRUVectorCache<Vector> {
        return try BasicLRUVectorCache<Vector>(maxMemory: configuration?.maxMemory ?? 100_000_000)
    }
}

/// LFU caching strategy
public struct LFUCachingStrategy<Vector: SIMD & Sendable>: CachingStrategy where Vector.Scalar: BinaryFloatingPoint {
    public typealias CacheType = BasicLFUVectorCache<Vector>
    public typealias Config = LFUCacheConfiguration
    
    public let configuration: LFUCacheConfiguration?
    
    public init(maxMemory: Int = 100_000_000) {
        self.configuration = LFUCacheConfiguration(maxMemory: maxMemory)
    }
    
    public func createCache() async throws -> BasicLFUVectorCache<Vector> {
        return try BasicLFUVectorCache<Vector>(maxMemory: configuration?.maxMemory ?? 100_000_000)
    }
}

/// FIFO caching strategy
public struct FIFOCachingStrategy<Vector: SIMD & Sendable>: CachingStrategy where Vector.Scalar: BinaryFloatingPoint {
    public typealias CacheType = BasicFIFOVectorCache<Vector>
    public typealias Config = FIFOCacheConfiguration
    
    public let configuration: FIFOCacheConfiguration?
    
    public init(maxMemory: Int = 100_000_000) {
        self.configuration = FIFOCacheConfiguration(maxMemory: maxMemory)
    }
    
    public func createCache() async throws -> BasicFIFOVectorCache<Vector> {
        return try BasicFIFOVectorCache<Vector>(maxMemory: configuration?.maxMemory ?? 100_000_000)
    }
}


// MARK: - Convenience Extensions

// MARK: - Caching Strategy Configurations

// Note: AdaptiveCachingStrategy removed - use LRU, LFU, or FIFO directly

extension VectorUniverse {
    /// Quick setup with HNSW index, hierarchical storage, and LRU cache
    public static func quickStart(
        maxConnections: Int = 16,
        cacheMemory: Int = 100_000_000
    ) -> FullyConfiguredUniverse<Vector, Metadata, HNSWIndexingStrategy<Vector, Metadata>, ResearchStorageStrategy, LRUCachingStrategy<Vector>> {
        VectorUniverse<Vector, Metadata>()
            .indexing(HNSWIndexingStrategy<Vector, Metadata>(maxConnections: maxConnections))
            .storage(ResearchStorageStrategy())
            .caching(LRUCachingStrategy(maxMemory: cacheMemory))
    }
    
    /// Research configuration with advanced features
    public static func research() -> FullyConfiguredUniverse<Vector, Metadata, HNSWIndexingStrategy<Vector, Metadata>, ResearchStorageStrategy, LRUCachingStrategy<Vector>> {
        VectorUniverse<Vector, Metadata>()
            .indexing(HNSWIndexingStrategy<Vector, Metadata>(
                maxConnections: 32,
                efConstruction: 400,
                useAdaptiveTuning: true
            ))
            .storage(ResearchStorageStrategy())
            .caching(LRUCachingStrategy<Vector>(maxMemory: 500_000_000))
    }
}

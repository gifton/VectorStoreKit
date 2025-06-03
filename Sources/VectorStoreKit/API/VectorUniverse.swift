// VectorStoreKit: Vector Universe Builder
//
// Fluent API for building sophisticated vector storage systems

import Foundation
import simd

// MARK: - Vector Universe Builder

/// Entry point for building vector stores using a fluent API
///
/// VectorUniverse provides a type-safe, composable way to configure
/// and instantiate vector stores with specific strategies.
public struct VectorUniverse<Vector: SIMD & Sendable, Metadata: Codable & Sendable> where Vector.Scalar: BinaryFloatingPoint {
    
    private let configuration: UniverseConfiguration
    
    /// Create a new vector universe
    public init(config: UniverseConfiguration = .init()) {
        self.configuration = config
    }
    
    /// Configure indexing strategy
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
    
    public let configuration: LRUCacheConfiguration?
    
    public init(maxMemory: Int = 100_000_000) {
        self.configuration = LRUCacheConfiguration(maxMemory: maxMemory)
    }
    
    public func createCache() async throws -> BasicLRUVectorCache<Vector> {
        try BasicLRUVectorCache<Vector>(maxMemory: configuration?.maxMemory ?? 100_000_000)
    }
}

/// LFU caching strategy
public struct LFUCachingStrategy<Vector: SIMD & Sendable>: CachingStrategy where Vector.Scalar: BinaryFloatingPoint {
    public typealias CacheType = BasicLFUVectorCache<Vector>
    
    public let configuration: LFUCacheConfiguration?
    
    public init(maxMemory: Int = 100_000_000) {
        self.configuration = LFUCacheConfiguration(maxMemory: maxMemory)
    }
    
    public func createCache() async throws -> BasicLFUVectorCache<Vector> {
        try BasicLFUVectorCache<Vector>(maxMemory: configuration?.maxMemory ?? 100_000_000)
    }
}

// MARK: - Universe Configuration

/// Configuration container for vector universe
public struct UniverseConfiguration: Sendable {
    private var settings: [String: any Sendable] = [:]
    
    public init() {
        // Default settings
        settings["enableProfiling"] = true
        settings["enableAnalytics"] = true
    }
    
    /// Build store configuration from universe settings
    func buildStoreConfiguration() -> StoreConfiguration {
        StoreConfiguration(
            name: settings["name"] as? String ?? "VectorStore",
            enableProfiling: settings["enableProfiling"] as? Bool ?? true,
            enableAnalytics: settings["enableAnalytics"] as? Bool ?? true,
            integrityCheckInterval: settings["integrityCheckInterval"] as? TimeInterval ?? 3600,
            optimizationThreshold: settings["optimizationThreshold"] as? Int ?? 100_000
        )
    }
    
    /// Get a configuration value with a default
    public func get<T>(_ key: String, default defaultValue: T) -> T {
        return settings[key] as? T ?? defaultValue
    }
    
    /// Set a configuration value
    public mutating func set<T: Sendable>(_ key: String, value: T) {
        settings[key] = value
    }
}

// MARK: - Convenience Extensions

extension VectorUniverse {
    /// Quick setup with HNSW index, hierarchical storage, and LRU cache
    public static func quickStart(
        maxConnections: Int = 16,
        cacheMemory: Int = 100_000_000
    ) -> FullyConfiguredUniverse<Vector, Metadata, HNSWIndexingStrategy<Vector, Metadata>, HierarchicalResearchStorageStrategy, LRUCachingStrategy<Vector>> {
        VectorUniverse<Vector, Metadata>()
            .indexing(HNSWIndexingStrategy<Vector, Metadata>(maxConnections: maxConnections))
            .storage(HierarchicalResearchStorageStrategy())
            .caching(LRUCachingStrategy(maxMemory: cacheMemory))
    }
    
    /// Research configuration with advanced features
    public static func research() -> FullyConfiguredUniverse<Vector, Metadata, HNSWIndexingStrategy<Vector, Metadata>, HierarchicalResearchStorageStrategy, LFUCachingStrategy<Vector>> {
        VectorUniverse<Vector, Metadata>()
            .indexing(HNSWIndexingStrategy<Vector, Metadata>(
                maxConnections: 32,
                efConstruction: 400,
                useAdaptiveTuning: true
            ))
            .storage(HierarchicalResearchStorageStrategy(
                customConfig: HierarchicalStorage.Configuration(
                    hotTierMemoryLimit: 1_000_000_000,
                    warmTierFileSizeLimit: 10_000_000_000,
                    coldTierCompression: .zstd,
                    encryptionSettings: .chacha20,
                    migrationSettings: .intelligent,
                    walConfiguration: .highPerformance,
                    monitoringSettings: .comprehensive,
                    baseDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("vectorstore/research")
                )
            ))
            .caching(LFUCachingStrategy(maxMemory: 500_000_000))
    }
}

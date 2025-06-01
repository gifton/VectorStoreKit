// VectorStoreKit: Advanced Protocol Definitions
//
// Research-grade protocols for building sophisticated vector storage systems

import Foundation
import simd

// MARK: - Core Index Protocol

/// Advanced vector index protocol with research capabilities
public protocol VectorIndex: Actor {
    
    // MARK: - Associated Types
    
    /// Vector type with SIMD optimization
    associatedtype Vector: SIMD where Vector.Scalar: BinaryFloatingPoint
    
    /// Metadata type with serialization support
    associatedtype Metadata: Codable & Sendable
    
    /// Index-specific configuration
    associatedtype Configuration: IndexConfiguration
    
    /// Index-specific statistics
    associatedtype Statistics: IndexStatistics
    
    // MARK: - Core Properties
    
    /// Current number of vectors in the index
    var count: Int { get async }
    
    /// Maximum capacity of this index
    var capacity: Int { get async }
    
    /// Current memory usage in bytes
    var memoryUsage: Int { get async }
    
    /// Index configuration
    var configuration: Configuration { get async }
    
    /// Whether the index is optimized
    var isOptimized: Bool { get async }
    
    // MARK: - Core Operations
    
    /// Insert a vector entry into the index
    /// - Parameter entry: The vector entry to insert
    /// - Returns: Insert operation result with metrics
    func insert(_ entry: VectorEntry<Vector, Metadata>) async throws -> InsertResult
    
    /// Search for similar vectors
    /// - Parameters:
    ///   - query: Query vector
    ///   - k: Number of nearest neighbors
    ///   - strategy: Search strategy to use
    ///   - filter: Optional filter for results
    /// - Returns: Search results with comprehensive metrics
    func search(
        query: Vector,
        k: Int,
        strategy: SearchStrategy,
        filter: SearchFilter?
    ) async throws -> [SearchResult<Metadata>]
    
    /// Update an existing vector
    /// - Parameters:
    ///   - id: Vector identifier
    ///   - vector: New vector data (optional)
    ///   - metadata: New metadata (optional)
    /// - Returns: Whether the update succeeded
    func update(id: VectorID, vector: Vector?, metadata: Metadata?) async throws -> Bool
    
    /// Delete a vector from the index
    /// - Parameter id: Vector identifier to delete
    /// - Returns: Whether the deletion succeeded
    func delete(id: VectorID) async throws -> Bool
    
    /// Check if a vector exists in the index
    /// - Parameter id: Vector identifier
    /// - Returns: Whether the vector exists
    func contains(id: VectorID) async -> Bool
    
    // MARK: - Advanced Operations
    
    /// Optimize the index for better performance
    /// - Parameter strategy: Optimization strategy
    func optimize(strategy: OptimizationStrategy) async throws
    
    /// Compact the index to reclaim space
    func compact() async throws
    
    /// Get comprehensive index statistics
    func statistics() async -> Statistics
    
    /// Validate index integrity
    func validateIntegrity() async throws -> IntegrityReport
    
    /// Export index for analysis or backup
    func export(format: ExportFormat) async throws -> Data
    
    /// Import index data
    func import(data: Data, format: ExportFormat) async throws
    
    // MARK: - Research & Analysis
    
    /// Analyze vector distribution in the index
    func analyzeDistribution() async -> DistributionAnalysis
    
    /// Get performance characteristics for different query types
    func performanceProfile() async -> PerformanceProfile
    
    /// Generate index visualization data
    func visualizationData() async -> VisualizationData
}

// MARK: - Storage Backend Protocol

/// Advanced storage backend with research capabilities
public protocol StorageBackend: Actor {
    
    // MARK: - Associated Types
    
    /// Backend-specific configuration
    associatedtype Configuration: StorageConfiguration
    
    /// Backend-specific statistics
    associatedtype Statistics: StorageStatistics
    
    // MARK: - Core Properties
    
    /// Storage configuration
    var configuration: Configuration { get async }
    
    /// Whether the backend is ready for operations
    var isReady: Bool { get async }
    
    /// Current storage size in bytes
    var size: Int { get async }
    
    // MARK: - Core Operations
    
    /// Store data with a key
    /// - Parameters:
    ///   - key: Storage key
    ///   - data: Data to store
    ///   - options: Storage options
    func store(key: String, data: Data, options: StorageOptions) async throws
    
    /// Retrieve data by key
    /// - Parameter key: Storage key
    /// - Returns: Retrieved data or nil if not found
    func retrieve(key: String) async throws -> Data?
    
    /// Delete data by key
    /// - Parameter key: Storage key
    func delete(key: String) async throws
    
    /// Check if key exists
    /// - Parameter key: Storage key
    /// - Returns: Whether the key exists
    func exists(key: String) async -> Bool
    
    /// Scan keys with a prefix
    /// - Parameter prefix: Key prefix to scan
    /// - Returns: Async stream of key-data pairs
    func scan(prefix: String) async throws -> AsyncStream<(String, Data)>
    
    // MARK: - Advanced Operations
    
    /// Compact storage to reclaim space
    func compact() async throws
    
    /// Get storage statistics
    func statistics() async -> Statistics
    
    /// Validate storage integrity
    func validateIntegrity() async throws -> StorageIntegrityReport
    
    /// Create a snapshot
    func createSnapshot() async throws -> SnapshotIdentifier
    
    /// Restore from snapshot
    func restoreSnapshot(_ identifier: SnapshotIdentifier) async throws
    
    // MARK: - Batch Operations
    
    /// Store multiple items in a batch
    func batchStore(_ items: [(key: String, data: Data, options: StorageOptions)]) async throws
    
    /// Retrieve multiple items in a batch
    func batchRetrieve(_ keys: [String]) async throws -> [String: Data?]
    
    /// Delete multiple items in a batch
    func batchDelete(_ keys: [String]) async throws
}

// MARK: - Cache Protocol

/// Advanced caching with ML-driven eviction policies
public protocol VectorCache: Actor {
    
    // MARK: - Associated Types
    
    /// Vector type
    associatedtype Vector: SIMD where Vector.Scalar: BinaryFloatingPoint
    
    /// Cache configuration
    associatedtype Configuration: CacheConfiguration
    
    /// Cache statistics
    associatedtype Statistics: CacheStatistics
    
    // MARK: - Core Properties
    
    /// Cache configuration
    var configuration: Configuration { get async }
    
    /// Current number of cached items
    var count: Int { get async }
    
    /// Current memory usage
    var memoryUsage: Int { get async }
    
    /// Cache hit rate
    var hitRate: Float { get async }
    
    // MARK: - Core Operations
    
    /// Get a vector from cache
    /// - Parameter id: Vector identifier
    /// - Returns: Cached vector or nil
    func get(id: VectorID) async -> Vector?
    
    /// Set a vector in cache
    /// - Parameters:
    ///   - id: Vector identifier
    ///   - vector: Vector to cache
    ///   - priority: Cache priority
    func set(id: VectorID, vector: Vector, priority: CachePriority) async
    
    /// Remove a vector from cache
    /// - Parameter id: Vector identifier
    func remove(id: VectorID) async
    
    /// Clear the cache
    func clear() async
    
    /// Get cache statistics
    func statistics() async -> Statistics
    
    // MARK: - Advanced Operations
    
    /// Prefetch vectors based on predicted access
    func prefetch(_ predictions: [VectorID: Float]) async
    
    /// Optimize cache based on access patterns
    func optimize() async
    
    /// Analyze cache performance
    func performanceAnalysis() async -> CachePerformanceAnalysis
}

// MARK: - Configuration Protocols

/// Base protocol for index configurations
public protocol IndexConfiguration: Codable, Sendable {
    /// Validate configuration parameters
    func validate() throws
    
    /// Get memory requirements estimate
    func estimatedMemoryUsage(for vectorCount: Int) -> Int
    
    /// Get computational complexity estimate
    func computationalComplexity() -> ComputationalComplexity
}

/// Base protocol for storage configurations
public protocol StorageConfiguration: Codable, Sendable {
    /// Validate configuration parameters
    func validate() throws
    
    /// Get storage overhead estimate
    func storageOverhead() -> Float
    
    /// Get compression capabilities
    func compressionCapabilities() -> CompressionCapabilities
}

/// Base protocol for cache configurations
public protocol CacheConfiguration: Codable, Sendable {
    /// Validate configuration parameters
    func validate() throws
    
    /// Get memory budget
    func memoryBudget() -> Int
    
    /// Get eviction policy
    func evictionPolicy() -> EvictionPolicy
}

// MARK: - Statistics Protocols

/// Base protocol for index statistics
public protocol IndexStatistics: Codable, Sendable {
    /// Total number of vectors
    var vectorCount: Int { get }
    
    /// Memory usage in bytes
    var memoryUsage: Int { get }
    
    /// Average search latency
    var averageSearchLatency: TimeInterval { get }
    
    /// Index quality metrics
    var qualityMetrics: IndexQualityMetrics { get }
}

/// Base protocol for storage statistics
public protocol StorageStatistics: Codable, Sendable {
    /// Total storage size
    var totalSize: Int { get }
    
    /// Compression ratio achieved
    var compressionRatio: Float { get }
    
    /// Average access latency
    var averageLatency: TimeInterval { get }
    
    /// Storage health metrics
    var healthMetrics: StorageHealthMetrics { get }
}

/// Base protocol for cache statistics
public protocol CacheStatistics: Codable, Sendable {
    /// Total cache hits
    var hits: Int { get }
    
    /// Total cache misses
    var misses: Int { get }
    
    /// Cache hit rate
    var hitRate: Float { get }
    
    /// Memory efficiency
    var memoryEfficiency: Float { get }
}

// MARK: - Supporting Types

/// Insert operation result
public struct InsertResult: Sendable {
    public let success: Bool
    public let insertTime: TimeInterval
    public let memoryImpact: Int
    public let indexReorganization: Bool
    
    public init(success: Bool, insertTime: TimeInterval, memoryImpact: Int, indexReorganization: Bool) {
        self.success = success
        self.insertTime = insertTime
        self.memoryImpact = memoryImpact
        self.indexReorganization = indexReorganization
    }
}

/// Storage options for advanced control
public struct StorageOptions: Sendable {
    public let compression: CompressionLevel
    public let durability: DurabilityLevel
    public let priority: StoragePriority
    public let ttl: TimeInterval?
    
    public init(
        compression: CompressionLevel = .adaptive,
        durability: DurabilityLevel = .standard,
        priority: StoragePriority = .normal,
        ttl: TimeInterval? = nil
    ) {
        self.compression = compression
        self.durability = durability
        self.priority = priority
        self.ttl = ttl
    }
    
    public static let `default` = StorageOptions()
}

/// Cache priority levels
public enum CachePriority: Int, Sendable, CaseIterable {
    case low = 0
    case normal = 1
    case high = 2
    case critical = 3
}

/// Storage priority levels
public enum StoragePriority: Int, Sendable, CaseIterable {
    case background = 0
    case normal = 1
    case high = 2
    case immediate = 3
}

/// Durability guarantees
public enum DurabilityLevel: Sendable {
    case none          // No durability guarantee
    case eventual      // Eventually consistent
    case standard      // Standard ACID properties
    case strict        // Strict consistency
}

/// Optimization strategies
public enum OptimizationStrategy: Sendable {
    case none
    case light
    case aggressive
    case learned(model: String)
    case adaptive
}

/// Export formats for research and backup
public enum ExportFormat: String, Sendable, CaseIterable {
    case binary = "binary"
    case json = "json"
    case hdf5 = "hdf5"
    case arrow = "arrow"
    case custom = "custom"
}

/// Search filters for advanced querying
public enum SearchFilter: Sendable {
    case metadata(MetadataFilter)
    case vector(VectorFilter)
    case composite(CompositeFilter)
    case learned(LearnedFilter)
}

public struct MetadataFilter: Sendable {
    public let key: String
    public let operation: FilterOperation
    public let value: Any
    
    public init(key: String, operation: FilterOperation, value: Any) {
        self.key = key
        self.operation = operation
        self.value = value
    }
}

public enum FilterOperation: Sendable {
    case equals, notEquals, lessThan, lessThanOrEqual, greaterThan, greaterThanOrEqual
    case contains, notContains, startsWith, endsWith
    case `in`, notIn
    case regex
}

public struct VectorFilter: Sendable {
    public let dimension: Int?
    public let range: ClosedRange<Float>?
    public let constraint: VectorConstraint
}

public enum VectorConstraint: Sendable {
    case magnitude(ClosedRange<Float>)
    case sparsity(ClosedRange<Float>)
    case custom((any SIMD) -> Bool)
}

public struct CompositeFilter: Sendable {
    public let operation: LogicalOperation
    public let filters: [SearchFilter]
}

public enum LogicalOperation: Sendable {
    case and, or, not
}

public struct LearnedFilter: Sendable {
    public let modelIdentifier: String
    public let confidence: Float
    public let parameters: [String: Any]
}

/// Compression capabilities
public struct CompressionCapabilities: Sendable {
    public let algorithms: [CompressionAlgorithm]
    public let maxRatio: Float
    public let lossless: Bool
}

public enum CompressionAlgorithm: String, Sendable, CaseIterable {
    case none = "none"
    case lz4 = "lz4"
    case zstd = "zstd"
    case brotli = "brotli"
    case quantization = "quantization"
    case learned = "learned"
}

/// Eviction policies for caching
public enum EvictionPolicy: Sendable {
    case lru           // Least Recently Used
    case lfu           // Least Frequently Used
    case arc           // Adaptive Replacement Cache
    case learned       // ML-based eviction
    case hybrid        // Combination approach
}

/// Quality metrics for indexes
public struct IndexQualityMetrics: Codable, Sendable {
    public let recall: Float
    public let precision: Float
    public let buildTime: TimeInterval
    public let memoryEfficiency: Float
    public let searchLatency: TimeInterval
    
    public init(recall: Float, precision: Float, buildTime: TimeInterval, memoryEfficiency: Float, searchLatency: TimeInterval) {
        self.recall = recall
        self.precision = precision
        self.buildTime = buildTime
        self.memoryEfficiency = memoryEfficiency
        self.searchLatency = searchLatency
    }
}

/// Health metrics for storage
public struct StorageHealthMetrics: Codable, Sendable {
    public let errorRate: Float
    public let latencyP99: TimeInterval
    public let throughput: Float
    public let fragmentation: Float
    
    public init(errorRate: Float, latencyP99: TimeInterval, throughput: Float, fragmentation: Float) {
        self.errorRate = errorRate
        self.latencyP99 = latencyP99
        self.throughput = throughput
        self.fragmentation = fragmentation
    }
}

/// Integrity reports
public struct IntegrityReport: Sendable {
    public let isValid: Bool
    public let errors: [IntegrityError]
    public let warnings: [IntegrityWarning]
    public let statistics: IntegrityStatistics
}

public struct IntegrityError: Sendable {
    public let type: ErrorType
    public let description: String
    public let severity: Severity
}

public struct IntegrityWarning: Sendable {
    public let type: WarningType
    public let description: String
    public let recommendation: String
}

public struct IntegrityStatistics: Sendable {
    public let totalChecks: Int
    public let passedChecks: Int
    public let failedChecks: Int
    public let checkDuration: TimeInterval
}

public enum ErrorType: String, Sendable {
    case corruption = "corruption"
    case inconsistency = "inconsistency"
    case missing = "missing"
    case invalid = "invalid"
}

public enum WarningType: String, Sendable {
    case performance = "performance"
    case memory = "memory"
    case configuration = "configuration"
    case optimization = "optimization"
}

public enum Severity: String, Sendable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case critical = "critical"
}

/// Storage integrity report
public struct StorageIntegrityReport: Sendable {
    public let isHealthy: Bool
    public let issues: [StorageIssue]
    public let recommendations: [String]
    public let lastCheck: Date
}

public struct StorageIssue: Sendable {
    public let type: StorageIssueType
    public let description: String
    public let impact: ImpactLevel
}

public enum StorageIssueType: String, Sendable {
    case corruption = "corruption"
    case performance = "performance"
    case space = "space"
    case consistency = "consistency"
}

public enum ImpactLevel: String, Sendable {
    case negligible = "negligible"
    case minor = "minor"
    case moderate = "moderate"
    case major = "major"
    case critical = "critical"
}

/// Snapshot identifier
public struct SnapshotIdentifier: Codable, Sendable {
    public let id: String
    public let timestamp: Date
    public let checksum: String
    
    public init(id: String, timestamp: Date, checksum: String) {
        self.id = id
        self.timestamp = timestamp
        self.checksum = checksum
    }
}

/// Distribution analysis results
public struct DistributionAnalysis: Sendable {
    public let dimensionality: Int
    public let density: Float
    public let clustering: ClusteringAnalysis
    public let outliers: [VectorID]
    public let statistics: DistributionStatistics
}

public struct ClusteringAnalysis: Sendable {
    public let estimatedClusters: Int
    public let silhouetteScore: Float
    public let inertia: Float
    public let clusterCenters: [[Float]]
}

public struct DistributionStatistics: Sendable {
    public let mean: [Float]
    public let variance: [Float]
    public let skewness: [Float]
    public let kurtosis: [Float]
}

/// Performance profile for different operations
public struct PerformanceProfile: Sendable {
    public let searchLatency: LatencyProfile
    public let insertLatency: LatencyProfile
    public let memoryUsage: MemoryProfile
    public let throughput: ThroughputProfile
}

public struct LatencyProfile: Sendable {
    public let p50: TimeInterval
    public let p90: TimeInterval
    public let p95: TimeInterval
    public let p99: TimeInterval
    public let max: TimeInterval
}

public struct MemoryProfile: Sendable {
    public let baseline: Int
    public let peak: Int
    public let average: Int
    public let efficiency: Float
}

public struct ThroughputProfile: Sendable {
    public let queriesPerSecond: Float
    public let insertsPerSecond: Float
    public let updatesPerSecond: Float
    public let deletesPerSecond: Float
}

/// Visualization data for research
public struct VisualizationData: Sendable {
    public let nodePositions: [[Float]]
    public let edges: [(Int, Int, Float)]
    public let nodeMetadata: [String: Any]
    public let layoutAlgorithm: String
}

/// Cache performance analysis
public struct CachePerformanceAnalysis: Sendable {
    public let hitRateOverTime: [(Date, Float)]
    public let memoryUtilization: Float
    public let evictionRate: Float
    public let optimalCacheSize: Int
    public let recommendations: [CacheRecommendation]
}

public struct CacheRecommendation: Sendable {
    public let type: RecommendationType
    public let description: String
    public let expectedImprovement: Float
}

public enum RecommendationType: String, Sendable {
    case sizeAdjustment = "size_adjustment"
    case policyChange = "policy_change"
    case prefetching = "prefetching"
    case partitioning = "partitioning"
}
// VectorStoreKit: Advanced Vector Store Implementation
//
// Research-grade vector store with sophisticated capabilities

import Foundation
import simd

// MARK: - Advanced Vector Store

/// Research-grade vector store with advanced capabilities
///
/// VectorStore provides a high-performance, feature-rich vector database
/// optimized for Apple platforms with cutting-edge algorithms and deep
/// hardware integration.
public actor VectorStore<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
>: Sendable 
where 
    Vector.Scalar: BinaryFloatingPoint,
    Index.Vector == Vector,
    Index.Metadata == Metadata,
    Cache.Vector == Vector 
{
    
    // MARK: - Core Properties
    
    /// Store configuration
    private let configuration: StoreConfiguration
    
    /// Primary index for vector operations
    private let primaryIndex: Index
    
    /// Storage backend for persistence
    private let storageBackend: Storage
    
    /// Cache for hot data
    private let cache: Cache
    
    /// Performance monitoring
    private let performanceMonitor: PerformanceMonitor
    
    /// Integrity manager
    private let integrityManager: IntegrityManager
    
    /// Access pattern analyzer
    private let accessAnalyzer: AccessPatternAnalyzer
    
    /// Current store state
    private var state: StoreState = .initializing
    
    // MARK: - Initialization
    
    /// Initialize with explicit components
    public init(
        index: Index,
        storage: Storage,
        cache: Cache,
        configuration: StoreConfiguration = .research
    ) async throws {
        self.configuration = configuration
        self.primaryIndex = index
        self.storageBackend = storage
        self.cache = cache
        
        self.performanceMonitor = PerformanceMonitor()
        self.integrityManager = IntegrityManager()
        self.accessAnalyzer = AccessPatternAnalyzer()
        
        try await initialize()
    }
    
    private func initialize() async throws {
        state = .initializing
        
        // Validate configuration
        try configuration.validate()
        
        // Initialize integrity checking
        await integrityManager.initialize()
        
        // Start performance monitoring
        await performanceMonitor.start()
        
        // Load existing data if any
        try await loadExistingData()
        
        state = .ready
    }
    
    // MARK: - Core Operations
    
    /// Add multiple vector entries with comprehensive tracking
    public func add(
        _ entries: [VectorEntry<Vector, Metadata>],
        options: InsertOptions = .default
    ) async throws -> DetailedInsertResult {
        try await ensureReady()
        
        let startTime = DispatchTime.now()
        let operation = Operation.bulkInsert(count: entries.count)
        
        await performanceMonitor.beginOperation(operation)
        
        // Validate entries
        try validateEntries(entries)
        
        var results: [VectorID: InsertResult] = [:]
        var totalInserted = 0
        var totalUpdated = 0
        var errors: [VectorStoreError] = []
        
        // Process entries concurrently using Swift 5.10's improved TaskGroup
        await withTaskGroup(of: (VectorID, Result<InsertResult, Error>).self) { group in
            // Add tasks for each entry
            for entry in entries {
                group.addTask {
                    do {
                        // Update access pattern
                        await self.accessAnalyzer.recordAccess(for: entry.id, type: .insert)
                        
                        // Insert into index
                        let indexResult = try await self.primaryIndex.insert(entry)
                        
                        // Store in persistence layer
                        try await self.storeEntry(entry, options: options)
                        
                        // Update cache if hot tier
                        if entry.tier == .hot {
                            await self.cache.set(id: entry.id, vector: entry.vector, priority: .normal)
                        }
                        
                        return (entry.id, .success(indexResult))
                    } catch {
                        return (entry.id, .failure(error))
                    }
                }
            }
            
            // Collect results
            for await (id, result) in group {
                switch result {
                case .success(let indexResult):
                    results[id] = indexResult
                    if indexResult.success {
                        totalInserted += 1
                    } else {
                        totalUpdated += 1
                    }
                case .failure(let error):
                    errors.append(VectorStoreError.insertion(id, error))
                }
            }
        }
        
        let duration = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        
        // End performance monitoring
        await performanceMonitor.endOperation(operation)
        
        return DetailedInsertResult(
            insertedCount: totalInserted,
            updatedCount: totalUpdated,
            errorCount: errors.count,
            errors: errors,
            individualResults: results,
            totalTime: TimeInterval(duration) / 1_000_000_000.0,
            performanceMetrics: await performanceMonitor.getMetrics(for: operation)
        )
    }
    
    /// Search for similar vectors with comprehensive analysis
    public func search(
        query: Vector,
        k: Int,
        strategy: SearchStrategy = .adaptive,
        filter: SearchFilter? = nil
    ) async throws -> ComprehensiveSearchResult<Metadata> {
        try await ensureReady()
        
        let startTime = DispatchTime.now()
        let operation = Operation.search(k: k, strategy: strategy)
        
        await performanceMonitor.beginOperation(operation)
        
        // Validate query
        try validateQuery(query)
        
        // Record access pattern
        let queryId = "query_\(UUID().uuidString)"
        await accessAnalyzer.recordAccess(for: queryId, type: .search)
        
        // Perform search
        let results = try await primaryIndex.search(
            query: query,
            k: k,
            strategy: strategy,
            filter: filter
        )
        
        // Update access patterns for found results
        for result in results {
            await accessAnalyzer.recordAccess(for: result.id, type: .access)
        }
        
        let duration = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        
        // End performance monitoring
        await performanceMonitor.endOperation(operation)
        
        return ComprehensiveSearchResult(
            results: results,
            queryTime: TimeInterval(duration) / 1_000_000_000.0,
            totalCandidates: await primaryIndex.count,
            strategy: strategy,
            performanceMetrics: await performanceMonitor.getMetrics(for: operation),
            qualityMetrics: await analyzeSearchQuality(results, query: query)
        )
    }
    
    /// Update an existing vector
    public func update(
        id: VectorID,
        vector: Vector? = nil,
        metadata: Metadata? = nil
    ) async throws -> UpdateResult {
        try await ensureReady()
        
        let startTime = DispatchTime.now()
        let operation = Operation.update(id: id)
        
        await performanceMonitor.beginOperation(operation)
        
        // Update access pattern
        await accessAnalyzer.recordAccess(for: id, type: .update)
        
        // Update in index
        let success = try await primaryIndex.update(id: id, vector: vector, metadata: metadata)
        
        if success {
            // Update in storage
            try await updateInStorage(id: id, vector: vector, metadata: metadata)
            
            // Update cache if present
            if let newVector = vector {
                await cache.set(id: id, vector: newVector, priority: .normal)
            }
        }
        
        let duration = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        
        await performanceMonitor.endOperation(operation)
        
        return UpdateResult(
            success: success,
            updateTime: TimeInterval(duration) / 1_000_000_000.0,
            performanceMetrics: await performanceMonitor.getMetrics(for: operation)
        )
    }
    
    /// Delete a vector
    public func delete(id: VectorID) async throws -> DeleteResult {
        try await ensureReady()
        
        let startTime = DispatchTime.now()
        let operation = Operation.delete(id: id)
        
        await performanceMonitor.beginOperation(operation)
        
        // Remove from index
        let success = try await primaryIndex.delete(id: id)
        
        if success {
            // Remove from storage
            try await storageBackend.delete(key: id)
            
            // Remove from cache
            await cache.remove(id: id)
            
            // Update access patterns
            await accessAnalyzer.recordDeletion(id: id)
        }
        
        let duration = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        
        await performanceMonitor.endOperation(operation)
        
        return DeleteResult(
            success: success,
            deleteTime: TimeInterval(duration) / 1_000_000_000.0,
            performanceMetrics: await performanceMonitor.getMetrics(for: operation)
        )
    }
    
    // MARK: - Advanced Operations
    
    /// Optimize the vector store for better performance
    public func optimize(strategy: OptimizationStrategy = .adaptive) async throws {
        try await ensureReady()
        
        let operation = Operation.optimization(strategy: strategy)
        await performanceMonitor.beginOperation(operation)
        
        // Optimize index
        try await primaryIndex.optimize(strategy: strategy)
        
        // Compact storage
        try await storageBackend.compact()
        
        // Optimize cache based on access patterns
        let patterns = await accessAnalyzer.getAccessPatterns()
        await cache.optimize()
        
        // Prefetch frequently accessed vectors
        let predictions = patterns.topAccessed(count: 100)
            .reduce(into: [:]) { dict, pattern in
                dict[pattern.id] = pattern.accessFrequency
            }
        await cache.prefetch(predictions)
        
        await performanceMonitor.endOperation(operation)
    }
    
    /// Get comprehensive statistics
    public func statistics() async -> StoreStatistics {
        let indexStats = await primaryIndex.statistics()
        let storageStats = await storageBackend.statistics()
        let cacheStats = await cache.statistics()
        let performanceStats = await performanceMonitor.overallStatistics()
        let accessStats = await accessAnalyzer.statistics()
        
        let stats = StoreStatistics(
            vectorCount: await primaryIndex.count,
            memoryUsage: await totalMemoryUsage(),
            diskUsage: await storageBackend.size,
            performanceStatistics: performanceStats,
            accessStatistics: accessStats,
            indexStatisticsSnapshot: IndexStatisticsSnapshot(from: indexStats),
            storageStatisticsSnapshot: StorageStatisticsSnapshot(from: storageStats),
            cacheStatisticsSnapshot: CacheStatisticsSnapshot(from: cacheStats)
        )
        return stats
    }
    
    /// Export store for backup or analysis
    public func export(format: ExportFormat = .binary) async throws -> Data {
        try await ensureReady()
        
        let operation = Operation.export(format: format)
        await performanceMonitor.beginOperation(operation)
        
        // Export index data
        let indexData = try await primaryIndex.export(format: format)
        
        // Export metadata
        let metadata = StoreMetadata(
            version: "1.0",
            vectorType: String(describing: Vector.self),
            metadataType: String(describing: Metadata.self),
            configuration: configuration,
            statistics: await statistics()
        )
        
        // Combine into final export
        let result = try combineExport(indexData: indexData, metadata: metadata, format: format)
        
        await performanceMonitor.endOperation(operation)
        
        return result
    }
    
    // MARK: - Private Helpers
    
    private func ensureReady() async throws {
        guard state == .ready else {
            throw VectorStoreError.notReady(state)
        }
    }
    
    private func validateEntries(_ entries: [VectorEntry<Vector, Metadata>]) throws {
        guard !entries.isEmpty else {
            throw VectorStoreError.validation("Empty entries array")
        }
        
        let expectedDimension = Vector.scalarCount
        for entry in entries {
            if entry.vector.scalarCount != expectedDimension {
                throw VectorStoreError.dimensionMismatch(
                    expected: expectedDimension,
                    actual: entry.vector.scalarCount
                )
            }
        }
    }
    
    private func validateQuery(_ query: Vector) throws {
        let expectedDimension = Vector.scalarCount
        if query.scalarCount != expectedDimension {
            throw VectorStoreError.dimensionMismatch(
                expected: expectedDimension,
                actual: query.scalarCount
            )
        }
    }
    
    private func storeEntry(_ entry: VectorEntry<Vector, Metadata>, options: InsertOptions) async throws {
        let encoder = JSONEncoder()
        let data = try encoder.encode(entry)
        
        let storageOptions = StorageOptions(
            compression: options.useCompression ? .adaptive : .none,
            durability: options.durabilityLevel,
            priority: .normal
        )
        
        try await storageBackend.store(key: entry.id, data: data, options: storageOptions)
    }
    
    private func updateInStorage(id: VectorID, vector: Vector?, metadata: Metadata?) async throws {
        // Retrieve existing entry
        guard let existingData = try await storageBackend.retrieve(key: id) else {
            throw VectorStoreError.storageError("Entry not found in storage")
        }
        
        let decoder = JSONDecoder()
        var entry = try decoder.decode(VectorEntry<Vector, Metadata>.self, from: existingData)
        
        // Update fields
        if let newVector = vector {
            entry = VectorEntry(
                id: entry.id,
                vector: newVector,
                metadata: metadata ?? entry.metadata,
                tier: entry.tier
            )
        } else if let newMetadata = metadata {
            entry = VectorEntry(
                id: entry.id,
                vector: entry.vector,
                metadata: newMetadata,
                tier: entry.tier
            )
        }
        
        // Store updated entry
        let encoder = JSONEncoder()
        let updatedData = try encoder.encode(entry)
        try await storageBackend.store(key: id, data: updatedData, options: .default)
    }
    
    private func loadExistingData() async throws {
        // Implementation depends on storage backend capabilities
        // For now, we'll skip loading on initialization
    }
    
    private func totalMemoryUsage() async -> Int {
        let indexMemory = await primaryIndex.memoryUsage
        let cacheMemory = await cache.memoryUsage
        let monitorMemory = await performanceMonitor.memoryUsage
        return indexMemory + cacheMemory + monitorMemory
    }
    
    private func analyzeSearchQuality(_ results: [SearchResult<Metadata>], query: Vector) async -> SearchQualityMetrics {
        // Simplified quality analysis
        let relevanceScore = results.isEmpty ? 0.0 : 1.0 - results.first!.distance
        let diversityScore = calculateDiversityScore(results)
        
        return SearchQualityMetrics(
            relevanceScore: relevanceScore,
            diversityScore: diversityScore,
            coverageScore: Float(results.count) / Float(max(1, await primaryIndex.count)),
            consistencyScore: 0.95 // Placeholder
        )
    }
    
    private func calculateDiversityScore(_ results: [SearchResult<Metadata>]) -> Float {
        guard results.count > 1 else { return 1.0 }
        
        var totalDistance: Float = 0
        var comparisons = 0
        
        for i in 0..<results.count {
            for j in (i+1)..<min(i+5, results.count) { // Compare with next 4 results
                totalDistance += abs(results[i].distance - results[j].distance)
                comparisons += 1
            }
        }
        
        return comparisons > 0 ? totalDistance / Float(comparisons) : 0
    }
    
    private func combineExport(indexData: Data, metadata: StoreMetadata, format: ExportFormat) throws -> Data {
        switch format {
        case .binary:
            var combined = Data()
            
            // Write magic number
            combined.append(contentsOf: "VSTR".utf8)
            
            // Write version
            combined.append(contentsOf: [1, 0, 0, 0]) // Version 1.0.0.0
            
            // Write metadata size and data
            let encoder = JSONEncoder()
            let metadataData = try encoder.encode(metadata)
            var metadataSize = UInt64(metadataData.count)
            combined.append(Data(bytes: &metadataSize, count: 8))
            combined.append(metadataData)
            
            // Write index data
            combined.append(indexData)
            
            return combined
            
        case .json:
            let exportObject = ExportContainer(
                metadata: metadata,
                indexData: indexData.base64EncodedString()
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            return try encoder.encode(exportObject)
            
        default:
            throw VectorStoreError.exportError("Unsupported export format: \(format)")
        }
    }
}

// MARK: - Supporting Types

/// Store state enumeration
public enum StoreState: String, Sendable {
    case initializing
    case ready
    case optimizing
    case error
}

/// Operation types for monitoring
public enum Operation: Sendable, Hashable {
    case bulkInsert(count: Int)
    case search(k: Int, strategy: SearchStrategy)
    case update(id: VectorID)
    case delete(id: VectorID)
    case optimization(strategy: OptimizationStrategy)
    case export(format: ExportFormat)
}

/// Insert options
public struct InsertOptions: Sendable {
    public let useCompression: Bool
    public let durabilityLevel: DurabilityLevel
    public let validateIntegrity: Bool
    public let parallel: Bool
    
    public init(
        useCompression: Bool = true,
        durabilityLevel: DurabilityLevel = .standard,
        validateIntegrity: Bool = true,
        parallel: Bool = true
    ) {
        self.useCompression = useCompression
        self.durabilityLevel = durabilityLevel
        self.validateIntegrity = validateIntegrity
        self.parallel = parallel
    }
    
    public static let `default` = InsertOptions()
    public static let fast = InsertOptions(useCompression: false, durabilityLevel: .none, validateIntegrity: false)
    public static let safe = InsertOptions(useCompression: true, durabilityLevel: .strict, validateIntegrity: true)
}


// MARK: - Result Types

/// Detailed insert result with comprehensive metrics
public struct DetailedInsertResult: Sendable {
    public let insertedCount: Int
    public let updatedCount: Int
    public let errorCount: Int
    public let errors: [VectorStoreError]
    public let individualResults: [VectorID: InsertResult]
    public let totalTime: TimeInterval
    public let performanceMetrics: OperationMetrics
}

/// Comprehensive search result with detailed analysis
public struct ComprehensiveSearchResult<Metadata: Codable & Sendable>: Sendable {
    public let results: [SearchResult<Metadata>]
    public let queryTime: TimeInterval
    public let totalCandidates: Int
    public let strategy: SearchStrategy
    public let performanceMetrics: OperationMetrics
    public let qualityMetrics: SearchQualityMetrics
}

/// Update result with performance metrics
public struct UpdateResult: Sendable {
    public let success: Bool
    public let updateTime: TimeInterval
    public let performanceMetrics: OperationMetrics
}

/// Delete result with performance metrics  
public struct DeleteResult: Sendable {
    public let success: Bool
    public let deleteTime: TimeInterval
    public let performanceMetrics: OperationMetrics
}

/// Store metadata for export
private struct StoreMetadata: Codable, Sendable {
    let version: String
    let vectorType: String
    let metadataType: String
    let configuration: StoreConfiguration
    let statistics: StoreStatistics
}

/// Export container
private struct ExportContainer: Codable, Sendable {
    let metadata: StoreMetadata
    let indexData: String
}

/// Comprehensive store statistics
public struct StoreStatistics: Sendable, Codable {
    public let vectorCount: Int
    public let memoryUsage: Int
    public let diskUsage: Int
    public let performanceStatistics: PerformanceStatistics
    public let accessStatistics: AccessStatistics
    
    // Statistics are now stored as sendable snapshots instead of protocol existentials
    @CodableIgnored
    public var indexStatisticsSnapshot: IndexStatisticsSnapshot?
    @CodableIgnored
    public var storageStatisticsSnapshot: StorageStatisticsSnapshot?
    @CodableIgnored
    public var cacheStatisticsSnapshot: CacheStatisticsSnapshot?
    
    internal init(
        vectorCount: Int,
        memoryUsage: Int,
        diskUsage: Int,
        performanceStatistics: PerformanceStatistics,
        accessStatistics: AccessStatistics,
        indexStatisticsSnapshot: IndexStatisticsSnapshot? = nil,
        storageStatisticsSnapshot: StorageStatisticsSnapshot? = nil,
        cacheStatisticsSnapshot: CacheStatisticsSnapshot? = nil
    ) {
        self.vectorCount = vectorCount
        self.memoryUsage = memoryUsage
        self.diskUsage = diskUsage
        self.performanceStatistics = performanceStatistics
        self.accessStatistics = accessStatistics
        self.indexStatisticsSnapshot = indexStatisticsSnapshot
        self.storageStatisticsSnapshot = storageStatisticsSnapshot
        self.cacheStatisticsSnapshot = cacheStatisticsSnapshot
    }
    
    private enum CodingKeys: String, CodingKey {
        case vectorCount, memoryUsage, diskUsage, performanceStatistics, accessStatistics
    }
}

/// Property wrapper to exclude properties from Codable
@propertyWrapper
public struct CodableIgnored<T: Sendable>: Codable, Sendable {
    public var wrappedValue: T?
    
    public init(wrappedValue: T?) {
        self.wrappedValue = wrappedValue
    }
    
    public init(from decoder: Decoder) throws {
        self.wrappedValue = nil
    }
    
    public func encode(to encoder: Encoder) throws {
        // Don't encode anything
    }
}

/// Search quality metrics for research
public struct SearchQualityMetrics: Sendable {
    public let relevanceScore: Float
    public let diversityScore: Float
    public let coverageScore: Float
    public let consistencyScore: Float
}

// MARK: - Vector Store Error

/// Comprehensive error types for vector store operations
public enum VectorStoreError: Error, Sendable {
    case validation(String)
    case notReady(StoreState)
    case insertion(VectorID, Error)
    case deletion(VectorID, Error)
    case dimensionMismatch(expected: Int, actual: Int)
    case indexError(String)
    case storageError(String)
    case cacheError(String)
    case integrityError(String)
    case exportError(String)
    case storeNotAvailable
    case importError(String)
}

// MARK: - Supporting Types

/// Performance monitoring
public actor PerformanceMonitor: Sendable {
    private var operations: [Operation: OperationMetrics] = [:]
    private var startTimes: [Operation: DispatchTime] = [:]
    
    public init() {}
    
    public func start() async {
        // Initialize monitoring
    }
    
    public func beginOperation(_ operation: Operation) async {
        startTimes[operation] = DispatchTime.now()
    }
    
    public func endOperation(_ operation: Operation) async {
        guard let startTime = startTimes[operation] else { return }
        let duration = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        
        let metrics = OperationMetrics(
            duration: TimeInterval(duration) / 1_000_000_000.0,
            memoryUsed: 0, // Placeholder
            cpuUsage: 0,   // Placeholder
            timestamp: Date()
        )
        
        operations[operation] = metrics
        startTimes[operation] = nil
    }
    
    public func getMetrics(for operation: Operation) async -> OperationMetrics {
        operations[operation] ?? OperationMetrics(duration: 0, memoryUsed: 0, cpuUsage: 0, timestamp: Date())
    }
    
    public func overallStatistics() async -> PerformanceStatistics {
        PerformanceStatistics()
    }
    
    public var memoryUsage: Int {
        MemoryLayout<Self>.size + operations.count * MemoryLayout<OperationMetrics>.size
    }
}

/// Operation metrics
public struct OperationMetrics: Sendable {
    public let duration: TimeInterval
    public let memoryUsed: Int
    public let cpuUsage: Float
    public let timestamp: Date
}

public struct PerformanceStatistics: Sendable, Codable {}
public struct AccessStatistics: Sendable, Codable {}

/// Sendable snapshot of index statistics
public struct IndexStatisticsSnapshot: Sendable {
    public let vectorCount: Int
    public let memoryUsage: Int
    public let averageSearchLatency: TimeInterval
    public let description: String
    
    public init(from stats: any IndexStatistics) {
        self.vectorCount = stats.vectorCount
        self.memoryUsage = stats.memoryUsage
        self.averageSearchLatency = stats.averageSearchLatency
        self.description = String(describing: stats)
    }
}

/// Sendable snapshot of storage statistics  
public struct StorageStatisticsSnapshot: Sendable {
    public let totalSize: Int
    public let description: String
    
    public init(from stats: any StorageStatistics) {
        self.totalSize = stats.totalSize
        self.description = String(describing: stats)
    }
}

/// Sendable snapshot of cache statistics
public struct CacheStatisticsSnapshot: Sendable {
    public let hits: Int
    public let misses: Int
    public let hitRate: Float
    public let memoryUsage: Int
    public let description: String
    
    public init(from stats: any CacheStatistics) {
        self.hits = stats.hits
        self.misses = stats.misses
        self.hitRate = stats.hitRate
        self.memoryUsage = stats.memoryEfficiency > 0 ? Int(Float(stats.hits + stats.misses) * stats.memoryEfficiency) : 0
        self.description = String(describing: stats)
    }
}

/// Integrity management
public actor IntegrityManager: Sendable {
    public init() {}
    
    public func initialize() async {
        // Initialize integrity checking
    }
}

/// Access pattern analysis
public actor AccessPatternAnalyzer: Sendable {
    private var accessRecords: [VectorID: AccessRecord] = [:]
    
    public init() {}
    
    public func recordAccess(for id: VectorID, type: AccessType) async {
        if let existing = accessRecords[id] {
            accessRecords[id] = AccessRecord(
                id: id,
                count: existing.count + 1,
                lastAccess: Date(),
                type: type
            )
        } else {
            accessRecords[id] = AccessRecord(id: id, count: 1, lastAccess: Date(), type: type)
        }
    }
    
    public func recordDeletion(id: VectorID) async {
        accessRecords[id] = nil
    }
    
    public func getAccessPatterns() async -> AccessPatterns {
        AccessPatterns(records: Array(accessRecords.values))
    }
    
    public func statistics() async -> AccessStatistics {
        AccessStatistics()
    }
}

public enum AccessType: Sendable {
    case insert, search, update, access
}

fileprivate struct AccessRecord: Sendable {
    let id: VectorID
    let count: Int
    let lastAccess: Date
    let type: AccessType
}

public struct AccessPatterns: Sendable {
    fileprivate let records: [AccessRecord]
    
    func topAccessed(count: Int) -> [(id: VectorID, accessFrequency: Float)] {
        records
            .sorted { $0.count > $1.count }
            .prefix(count)
            .map { ($0.id, Float($0.count)) }
    }
}


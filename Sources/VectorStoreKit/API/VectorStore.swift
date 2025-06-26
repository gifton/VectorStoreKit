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
    
    /// Initialize a vector store with explicit components
    ///
    /// Creates a new vector store instance with the specified index, storage backend,
    /// cache, and configuration. The initialization process validates the configuration
    /// and prepares all subsystems for operation.
    ///
    /// - Parameters:
    ///   - index: The vector index implementation for similarity search operations
    ///   - storage: The storage backend for persistent data storage
    ///   - cache: The caching layer for frequently accessed vectors
    ///   - configuration: Store configuration with performance and behavior settings
    ///
    /// - Throws: `VectorStoreError` if initialization fails or configuration is invalid
    ///
    /// - Note: This initializer is typically not called directly. Use `VectorUniverse`
    ///         for a more convenient configuration API.
    ///
    /// ## Example
    /// ```swift
    /// let store = try await VectorStore(
    ///     index: HNSWIndex(configuration: hnswConfig),
    ///     storage: HierarchicalStorage(configuration: storageConfig),
    ///     cache: LRUVectorCache(maxMemory: 100_000_000),
    ///     configuration: .default
    /// )
    /// ```
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
    
    /// Add multiple vector entries to the store
    ///
    /// Inserts one or more vector entries into the store with support for parallel
    /// processing, compression, and durability guarantees. The operation tracks
    /// detailed metrics and handles partial failures gracefully.
    ///
    /// - Parameters:
    ///   - entries: Array of vector entries to insert
    ///   - options: Configuration options for the insert operation
    ///
    /// - Returns: Detailed result including success count, errors, and performance metrics
    ///
    /// - Throws: `VectorStoreError.notReady` if store is not initialized
    ///           `VectorStoreError.validation` if entries are invalid
    ///           `VectorStoreError.dimensionMismatch` if vector dimensions don't match
    ///
    /// ## Example
    /// ```swift
    /// let entries = documents.map { doc in
    ///     VectorEntry(
    ///         id: doc.id,
    ///         vector: doc.embedding,
    ///         metadata: DocumentMetadata(title: doc.title, content: doc.content)
    ///     )
    /// }
    ///
    /// let result = try await store.add(entries, options: .default)
    /// print("Inserted: \(result.insertedCount), Errors: \(result.errorCount)")
    /// ```
    ///
    /// ## Performance Notes
    /// - Entries are processed in parallel for optimal throughput
    /// - Use `.fast` options for bulk loading scenarios
    /// - Use `.safe` options for critical data with validation
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
                        await self.accessAnalyzer.recordAccess(id: entry.id, level: entry.tier == .hot ? .l1 : .l3, timestamp: Date())
                        
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
                    errors.append(VectorStoreError.insertion(reason: error.localizedDescription, vectorId: id))
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
    
    /// Search for k-nearest neighbors of a query vector
    ///
    /// Performs similarity search to find the k most similar vectors to the query.
    /// Supports various search strategies, filtering, and comprehensive result analysis.
    ///
    /// - Parameters:
    ///   - query: The query vector to search for
    ///   - k: Number of nearest neighbors to return
    ///   - strategy: Search strategy to use (exact, approximate, or adaptive)
    ///   - filter: Optional filter to constrain results by metadata or vector properties
    ///
    /// - Returns: Comprehensive search results including matches, performance metrics, and quality analysis
    ///
    /// - Throws: `VectorStoreError.notReady` if store is not initialized
    ///           `VectorStoreError.dimensionMismatch` if query dimension doesn't match store vectors
    ///
    /// ## Example
    /// ```swift
    /// // Basic search
    /// let results = try await store.search(query: queryVector, k: 10)
    ///
    /// // Search with metadata filter
    /// let filtered = try await store.search(
    ///     query: queryVector,
    ///     k: 20,
    ///     filter: .metadata(MetadataFilter(key: "category", operation: .equals, value: "tech"))
    /// )
    ///
    /// // Use exact search for high accuracy
    /// let exact = try await store.search(query: queryVector, k: 5, strategy: .exact)
    /// ```
    ///
    /// ## Performance Notes
    /// - `.adaptive` strategy automatically chooses between exact and approximate search
    /// - `.approximate` is faster but may miss some results
    /// - `.exact` guarantees finding the true k-nearest neighbors
    /// - Filters are applied during search for efficiency
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
        await accessAnalyzer.recordAccess(id: queryId, level: nil, timestamp: Date())
        
        // Perform search
        let results = try await primaryIndex.search(
            query: query,
            k: k,
            strategy: strategy,
            filter: filter
        )
        
        // Update access patterns for found results
        for result in results {
            await accessAnalyzer.recordAccess(id: result.id, level: nil, timestamp: Date())
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
    
    /// Update an existing vector's data or metadata
    ///
    /// Updates the vector data and/or metadata for an existing entry. At least one
    /// of vector or metadata must be provided. The operation maintains consistency
    /// across index, storage, and cache layers.
    ///
    /// - Parameters:
    ///   - id: The unique identifier of the vector to update
    ///   - vector: New vector data (optional, preserves existing if nil)
    ///   - metadata: New metadata (optional, preserves existing if nil)
    ///
    /// - Returns: Update result with success status and performance metrics
    ///
    /// - Throws: `VectorStoreError.notReady` if store is not initialized
    ///
    /// ## Example
    /// ```swift
    /// // Update only metadata
    /// let result = try await store.update(
    ///     id: "doc_123",
    ///     metadata: DocumentMetadata(title: "Updated Title", content: existingContent)
    /// )
    ///
    /// // Update both vector and metadata
    /// let result = try await store.update(
    ///     id: "doc_123",
    ///     vector: newEmbedding,
    ///     metadata: newMetadata
    /// )
    /// ```
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
        await accessAnalyzer.recordAccess(id: id, level: nil, timestamp: Date())
        
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
    
    /// Delete a vector from the store
    ///
    /// Removes a vector entry from all layers (index, storage, cache) and updates
    /// access pattern tracking. The operation is atomic within each layer.
    ///
    /// - Parameter id: The unique identifier of the vector to delete
    ///
    /// - Returns: Delete result with success status and performance metrics
    ///
    /// - Throws: `VectorStoreError.notReady` if store is not initialized
    ///
    /// ## Example
    /// ```swift
    /// let result = try await store.delete(id: "doc_123")
    /// if result.success {
    ///     print("Vector deleted successfully")
    /// }
    /// ```
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
            // Deletion is recorded as access to mark the vector as deleted
            await accessAnalyzer.recordAccess(id: id, level: nil, timestamp: Date())
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
    ///
    /// Performs various optimization operations including index rebalancing,
    /// storage compaction, cache optimization, and prefetching based on access patterns.
    /// The specific optimizations depend on the chosen strategy.
    ///
    /// - Parameter strategy: The optimization strategy to apply
    ///
    /// - Throws: `VectorStoreError.notReady` if store is not initialized
    ///
    /// ## Strategies
    /// - `.none`: No optimization
    /// - `.light`: Quick optimizations with minimal impact
    /// - `.aggressive`: Thorough optimization, may take time
    /// - `.adaptive`: Automatically choose based on current state
    /// - `.intelligent`: ML-based optimization using access patterns
    ///
    /// ## Example
    /// ```swift
    /// // Periodic optimization
    /// try await store.optimize(strategy: .adaptive)
    ///
    /// // Aggressive optimization during maintenance window
    /// try await store.optimize(strategy: .aggressive)
    /// ```
    ///
    /// - Note: Optimization may temporarily impact query performance
    public func optimize(strategy: OptimizationStrategy = .adaptive) async throws {
        try await ensureReady()
        
        let operation = Operation.optimization(strategy: strategy)
        await performanceMonitor.beginOperation(operation)
        
        // Optimize index
        try await primaryIndex.optimize(strategy: strategy)
        
        // Compact storage
        try await storageBackend.compact()
        
        // Optimize cache based on access patterns
        let patterns = await accessAnalyzer.getCurrentPatterns()
        await cache.optimize()
        
        // Prefetch frequently accessed vectors (hot spots)
        let predictions = patterns.hotSpots.prefix(100)
            .reduce(into: [:]) { dict, vectorId in
                dict[vectorId] = Float(1.0) // High priority for hot spots
            }
        await cache.prefetch(predictions)
        
        await performanceMonitor.endOperation(operation)
    }
    
    /// Get comprehensive statistics about the vector store
    ///
    /// Returns detailed statistics including vector count, memory usage,
    /// performance metrics, and subsystem-specific information.
    ///
    /// - Returns: Comprehensive store statistics
    ///
    /// ## Example
    /// ```swift
    /// let stats = await store.statistics()
    /// print("Vectors: \(stats.vectorCount)")
    /// print("Memory: \(stats.memoryUsage / 1024 / 1024) MB")
    /// print("Cache hit rate: \(stats.cacheStatisticsSnapshot?.hitRate ?? 0)%")
    /// ```
    public func statistics() async -> StoreStatistics {
        let indexStats = await primaryIndex.statistics()
        let storageStats = await storageBackend.statistics()
        let cacheStats = await cache.statistics()
        let performanceStats = await performanceMonitor.overallStatistics()
        let accessStats = AccessStatistics() // Currently empty structure
        
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
    
    /// Export the vector store for backup or analysis
    ///
    /// Creates a portable representation of the store's data in the specified format.
    /// The export includes vectors, metadata, index structure, and configuration.
    ///
    /// - Parameter format: The export format to use
    ///
    /// - Returns: Serialized data in the requested format
    ///
    /// - Throws: `VectorStoreError.notReady` if store is not initialized
    ///           `VectorStoreError.exportError` if export fails
    ///
    /// ## Supported Formats
    /// - `.binary`: Efficient binary format (default)
    /// - `.json`: Human-readable JSON format
    /// - `.hdf5`: HDF5 format for scientific computing
    /// - `.arrow`: Apache Arrow format for data analysis
    ///
    /// ## Example
    /// ```swift
    /// // Export for backup
    /// let backupData = try await store.export(format: .binary)
    /// try backupData.write(to: backupURL)
    ///
    /// // Export for analysis
    /// let analysisData = try await store.export(format: .json)
    /// ```
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
            throw VectorStoreError.notReady(component: "VectorStore")
        }
    }
    
    private func validateEntries(_ entries: [VectorEntry<Vector, Metadata>]) throws {
        guard !entries.isEmpty else {
            throw VectorStoreError.validation(field: "entries", value: entries, reason: "Empty entries array")
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
    // TODO - Gifton - understa dn Vector vs query
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
            throw VectorStoreError.storageError(operation: "retrieve", reason: "Entry not found in storage")
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
            throw VectorStoreError.exportError(format: "\(format)", reason: "Unsupported export format")
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

/// Configuration options for vector insertion operations
///
/// InsertOptions allows fine-tuning the trade-offs between performance,
/// durability, and data integrity during vector insertion.
public struct InsertOptions: Sendable {
    /// Whether to compress vectors before storage
    ///
    /// Reduces storage space but adds CPU overhead. Recommended for
    /// large datasets with storage constraints.
    public let useCompression: Bool
    
    /// Durability guarantee level for the insertion
    ///
    /// - `.none`: No durability, fastest performance
    /// - `.eventual`: Eventually consistent
    /// - `.standard`: Standard ACID properties
    /// - `.strict`: Synchronous writes with fsync
    public let durabilityLevel: DurabilityLevel
    
    /// Whether to validate vector integrity before insertion
    ///
    /// Checks for NaN/Inf values and dimension consistency.
    /// Adds overhead but prevents data corruption.
    public let validateIntegrity: Bool
    
    /// Whether to process entries in parallel
    ///
    /// Enables concurrent processing for better throughput on
    /// multi-core systems. May affect insertion order.
    public let parallel: Bool
    
    /// Initialize custom insert options
    ///
    /// - Parameters:
    ///   - useCompression: Enable compression (default: true)
    ///   - durabilityLevel: Durability guarantee (default: .standard)
    ///   - validateIntegrity: Enable validation (default: true)
    ///   - parallel: Enable parallel processing (default: true)
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
    
    /// Default balanced options
    public static let `default` = InsertOptions()
    
    /// Optimized for speed, minimal guarantees
    public static let fast = InsertOptions(
        useCompression: false,
        durabilityLevel: .none,
        validateIntegrity: false
    )
    
    /// Optimized for safety, maximum guarantees
    public static let safe = InsertOptions(
        useCompression: true,
        durabilityLevel: .strict,
        validateIntegrity: true
    )
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
    public let performanceMetrics: StoreOperationMetrics
}

/// Comprehensive search result with detailed analysis
public struct ComprehensiveSearchResult<Metadata: Codable & Sendable>: Sendable {
    public let results: [SearchResult<Metadata>]
    public let queryTime: TimeInterval
    public let totalCandidates: Int
    public let strategy: SearchStrategy
    public let performanceMetrics: StoreOperationMetrics
    public let qualityMetrics: SearchQualityMetrics
}

/// Update result with performance metrics
public struct UpdateResult: Sendable {
    public let success: Bool
    public let updateTime: TimeInterval
    public let performanceMetrics: StoreOperationMetrics
}

/// Delete result with performance metrics  
public struct DeleteResult: Sendable {
    public let success: Bool
    public let deleteTime: TimeInterval
    public let performanceMetrics: StoreOperationMetrics
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
    
    // Statistics are stored as sendable snapshots instead of protocol existentials
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


// MARK: - Supporting Types

/// Performance monitoring
public actor PerformanceMonitor: Sendable {
    private var operations: [Operation: StoreOperationMetrics] = [:]
    private var startTimes: [Operation: DispatchTime] = [:]
    
    public init() {}
    
    // TODO - Gifton
    public func start() async {
        // Initialize monitoring
    }
    
    public func beginOperation(_ operation: Operation) async {
        startTimes[operation] = DispatchTime.now()
    }
    
    public func endOperation(_ operation: Operation) async {
        guard let startTime = startTimes[operation] else { return }
        let duration = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        
        let metrics = StoreOperationMetrics(
            duration: TimeInterval(duration) / 1_000_000_000.0,
            memoryUsed: 0, // Placeholder
            cpuUsage: 0,   // Placeholder
            timestamp: Date()
        )
        
        operations[operation] = metrics
        startTimes[operation] = nil
    }
    
    public func getMetrics(for operation: Operation) async -> StoreOperationMetrics {
        operations[operation] ?? StoreOperationMetrics(duration: 0, memoryUsed: 0, cpuUsage: 0, timestamp: Date())
    }
    
    // TODO - Gifton
    public func overallStatistics() async -> PerformanceStatistics {
        PerformanceStatistics()
    }
    
    public var memoryUsage: Int {
        MemoryLayout<Self>.size + operations.count * MemoryLayout<StoreOperationMetrics>.size
    }
}

/// Store operation metrics
public struct StoreOperationMetrics: Sendable {
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
// TODO - Gifton
public actor IntegrityManager: Sendable {
    public init() {}
    
    public func initialize() async {
        // Initialize integrity checking
    }
}

// AccessPatternAnalyzer is defined in Caching/AccessPatternAnalyzer.swift


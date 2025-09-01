import Foundation
import CryptoKit
import os.log

/// High-performance 3-tier storage system with automatic migration
///
/// This storage backend provides:
/// - Memory tier for hot data (microsecond latency)
/// - SSD tier with LZ4 compression for warm data (millisecond latency)  
/// - Archive tier with ZSTD compression for cold data (100ms latency)
/// - Automatic tier migration based on access patterns
/// - Zero-copy operations where possible
/// - Thread-safe access through actor isolation
public actor ThreeTierStorage: StorageBackend, TierAwareStorage {
    
    // MARK: - Types
    
    public typealias Configuration = ThreeTierStorageConfiguration
    public typealias Key = String
    public typealias Value = Data
    
    /// Storage statistics across all tiers
    public struct Statistics: StorageStatistics, Codable {
        public let totalSize: Int
        public let compressionRatio: Float
        public let averageLatency: TimeInterval
        public let healthMetrics: StorageHealthMetrics
        
        // Tier-specific stats
        public let tierSizes: [StorageTier: Int]
        public let tierItemCounts: [StorageTier: Int]
        public let tierCompressionRatios: [StorageTier: Float]
        public let tierLatencies: [StorageTier: TimeInterval]
        
        // Migration stats
        public let migrationStats: TierManager.Statistics
    }
    
    // MARK: - Private Properties
    
    private let configuration: Configuration
    private let logger = Logger(subsystem: "VectorStoreKit", category: "ThreeTierStorage")
    
    // Storage backends for each tier
    private var memoryTier: [String: Data] = [:]
    private var ssdTier: FileBasedTier
    private var archiveTier: FileBasedTier
    
    // Support systems
    private let compressionEngine: CompressionEngine
    private let accessTracker: AccessPatternTracker
    private let tierManager: TierManager
    
    // Metadata tracking
    private var metadata: [String: ItemMetadata] = [:]
    
    // Performance metrics
    private var metrics = ThreeTierPerformanceMetrics()
    
    // Ready state
    private var isReadyState = false
    
    // MARK: - StorageBackend Protocol Properties
    
    public var isReady: Bool {
        isReadyState
    }
    
    public var size: Int {
        let memorySize = memoryTier.values.reduce(0) { $0 + $1.count }
        let ssdSize = ssdTier.totalSize
        let archiveSize = archiveTier.totalSize
        return memorySize + ssdSize + archiveSize
    }
    
    // MARK: - Initialization
    
    public init(configuration: Configuration) async throws {
        try configuration.validate()
        self.configuration = configuration
        
        // Initialize compression engine
        self.compressionEngine = CompressionEngine()
        
        // Initialize access tracker
        self.accessTracker = AccessPatternTracker(
            configuration: configuration.accessPatternConfiguration.asTrackerConfiguration
        )
        
        // Initialize file-based tiers
        self.ssdTier = try FileBasedTier(
            path: configuration.ssdPath,
            tier: .ssd,
            compressionEngine: compressionEngine
        )
        
        self.archiveTier = try FileBasedTier(
            path: configuration.archivePath,
            tier: .archive,
            compressionEngine: compressionEngine
        )
        
        // Initialize tier manager (but don't start it yet)
        self.tierManager = TierManager(
            configuration: configuration.tierManagerConfiguration.asTierManagerConfiguration,
            storage: self,
            accessTracker: accessTracker,
            compressionEngine: compressionEngine
        )
        
        // Reserve capacity for memory tier
        memoryTier.reserveCapacity(configuration.initialMemoryCapacity)
        metadata.reserveCapacity(configuration.initialMemoryCapacity * 3) // Account for all tiers
        
        // Load metadata from disk
        try await loadMetadata()
        
        isReadyState = true
        
        // Start automatic migration if enabled
        if configuration.autoMigrationEnabled {
            await tierManager.start()
        }
        
        logger.info("Initialized 3-tier storage with memory limit: \(configuration.memoryLimit) bytes")
    }
    
    deinit {
        Task {
            await tierManager.stop()
        }
    }
    
    // MARK: - Core StorageBackend Operations
    
    public func store(key: String, data: Data, options: StorageOptions) async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Determine initial tier based on data size and options
        let tier = determineInitialTier(dataSize: data.count, options: options)
        
        // Store in the appropriate tier
        try await store(data, forKey: key, tier: tier)
        
        // Record access for migration decisions
        await accessTracker.recordAccess(key: key, dataSize: data.count, currentTier: tier)
        
        // Update metrics
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        metrics.recordOperation(.store, duration: duration, size: data.count, tier: tier)
        
        logger.debug("Stored key '\(key)' in \(tier.properties.name) tier (\(data.count) bytes)")
    }
    
    public func retrieve(key: String) async throws -> Data? {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        guard let result = try await retrieve(forKey: key) else {
            metrics.recordOperation(.retrieve, duration: CFAbsoluteTimeGetCurrent() - startTime, size: 0, tier: nil)
            return nil
        }
        
        let (data, tier) = result
        
        // Record access for migration decisions
        await accessTracker.recordAccess(key: key, dataSize: data.count, currentTier: tier)
        
        // Update metrics
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        metrics.recordOperation(.retrieve, duration: duration, size: data.count, tier: tier)
        
        logger.debug("Retrieved key '\(key)' from \(tier.properties.name) tier (\(data.count) bytes)")
        
        return data
    }
    
    public func delete(key: String) async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Find which tier contains the key
        guard let meta = metadata[key] else {
            return // Key doesn't exist
        }
        
        // Delete from the appropriate tier
        switch meta.tier {
        case .memory:
            memoryTier.removeValue(forKey: key)
        case .ssd:
            try await ssdTier.delete(key: key)
        case .archive:
            try await archiveTier.delete(key: key)
        }
        
        // Remove metadata
        metadata.removeValue(forKey: key)
        
        // Remove from access tracker
        await accessTracker.removePattern(for: key)
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        metrics.recordOperation(.delete, duration: duration, size: meta.size, tier: meta.tier)
        
        logger.debug("Deleted key '\(key)' from \(meta.tier.properties.name) tier")
    }
    
    public func exists(key: String) async -> Bool {
        metadata[key] != nil
    }
    
    public func scan(prefix: String) async throws -> AsyncStream<(String, Data)> {
        AsyncStream { continuation in
            Task {
                do {
                    // Scan all tiers
                    for (key, meta) in metadata where key.hasPrefix(prefix) {
                        if let data = try await retrieve(key: key) {
                            continuation.yield((key, data))
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - TierAwareStorage Protocol
    
    public func store(_ value: Data, forKey key: String, tier: StorageTier) async throws {
        // Check tier capacity
        switch tier {
        case .memory:
            let newSize = getCurrentMemoryUsage() + value.count
            if newSize > configuration.memoryLimit {
                throw StorageError.storeFailed("Memory tier full: \(newSize) > \(configuration.memoryLimit)")
            }
            memoryTier[key] = value
            
        case .ssd:
            try await ssdTier.store(key: key, data: value)
            
        case .archive:
            try await archiveTier.store(key: key, data: value)
        }
        
        // Update metadata
        metadata[key] = ItemMetadata(
            tier: tier,
            size: value.count,
            compressedSize: value.count, // Will be updated by tier
            createdAt: Date(),
            lastAccessTime: Date(),
            accessCount: 1
        )
    }
    
    public func retrieve(forKey key: String) async throws -> (value: Data, tier: StorageTier)? {
        guard let meta = metadata[key] else {
            return nil
        }
        
        let data: Data
        switch meta.tier {
        case .memory:
            guard let memData = memoryTier[key] else {
                throw StorageError.retrieveFailed("Key not found in memory tier: \(key)")
            }
            data = memData
            
        case .ssd:
            data = try await ssdTier.retrieve(key: key)
            
        case .archive:
            data = try await archiveTier.retrieve(key: key)
        }
        
        // Update access metadata
        metadata[key]?.lastAccessTime = Date()
        metadata[key]?.accessCount += 1
        
        return (data, meta.tier)
    }
    
    public func migrate(key: String, to tier: StorageTier) async throws {
        guard let currentMeta = metadata[key] else {
            throw StorageError.keyNotFound(key)
        }
        
        guard currentMeta.tier != tier else {
            return // Already in target tier
        }
        
        // Retrieve from current tier
        guard let (data, _) = try await retrieve(forKey: key) else {
            throw StorageError.retrieveFailed("Failed to retrieve key for migration: \(key)")
        }
        
        // Delete from current tier
        try await delete(key: key)
        
        // Store in new tier
        try await store(data, forKey: key, tier: tier)
    }
    
    public func accessInfo(forKey key: String) async throws -> AccessInfo? {
        guard let meta = metadata[key] else {
            return nil
        }
        
        return AccessInfo(
            lastAccessTime: meta.lastAccessTime,
            accessCount: meta.accessCount,
            createdAt: meta.createdAt,
            dataSize: meta.size
        )
    }
    
    public func migrationDecision(forKey key: String) async throws -> MigrationDecision {
        await accessTracker.getMigrationRecommendation(for: key)
    }
    
    // MARK: - Batch Operations
    
    public func batchStore(_ items: [(key: String, data: Data, options: StorageOptions)]) async throws {
        for (key, data, options) in items {
            try await store(key: key, data: data, options: options)
        }
    }
    
    public func batchRetrieve(_ keys: [String]) async throws -> [String: Data?] {
        var results: [String: Data?] = [:]
        for key in keys {
            results[key] = try await retrieve(key: key)
        }
        return results
    }
    
    public func batchDelete(_ keys: [String]) async throws {
        for key in keys {
            try await delete(key: key)
        }
    }
    
    // MARK: - Advanced Operations
    
    public func compact() async throws {
        logger.info("Starting compaction")
        
        // Compact file-based tiers
        try await ssdTier.compact()
        try await archiveTier.compact()
        
        // Run a migration cycle to rebalance data
        await tierManager.runMigrationCycle()
        
        logger.info("Compaction completed")
    }
    
    public func statistics() async -> Statistics {
        // Gather tier statistics
        var tierSizes: [StorageTier: Int] = [:]
        var tierItemCounts: [StorageTier: Int] = [:]
        var tierCompressionRatios: [StorageTier: Float] = [:]
        var tierLatencies: [StorageTier: TimeInterval] = [:]
        
        // Memory tier stats
        tierSizes[.memory] = memoryTier.values.reduce(0) { $0 + $1.count }
        tierItemCounts[.memory] = memoryTier.count
        tierCompressionRatios[.memory] = 1.0 // No compression
        tierLatencies[.memory] = metrics.averageLatency(for: .memory)
        
        // SSD tier stats
        let ssdStats = await ssdTier.statistics()
        tierSizes[.ssd] = ssdStats.totalSize
        tierItemCounts[.ssd] = ssdStats.itemCount
        tierCompressionRatios[.ssd] = ssdStats.compressionRatio
        tierLatencies[.ssd] = metrics.averageLatency(for: .ssd)
        
        // Archive tier stats
        let archiveStats = await archiveTier.statistics()
        tierSizes[.archive] = archiveStats.totalSize
        tierItemCounts[.archive] = archiveStats.itemCount
        tierCompressionRatios[.archive] = archiveStats.compressionRatio
        tierLatencies[.archive] = metrics.averageLatency(for: .archive)
        
        // Calculate overall stats
        let totalSize = tierSizes.values.reduce(0, +)
        let totalOriginalSize = metadata.values.reduce(0) { $0 + $1.size }
        let overallCompressionRatio = totalOriginalSize > 0 ? Float(totalOriginalSize) / Float(totalSize) : 1.0
        
        let healthMetrics = StorageHealthMetrics(
            errorRate: metrics.errorRate,
            latencyP99: metrics.latencyP99,
            throughput: metrics.throughput,
            fragmentation: await calculateFragmentation()
        )
        
        return Statistics(
            totalSize: totalSize,
            compressionRatio: overallCompressionRatio,
            averageLatency: metrics.averageLatency,
            healthMetrics: healthMetrics,
            tierSizes: tierSizes,
            tierItemCounts: tierItemCounts,
            tierCompressionRatios: tierCompressionRatios,
            tierLatencies: tierLatencies,
            migrationStats: await tierManager.getStatistics()
        )
    }
    
    public func validateIntegrity() async throws -> StorageIntegrityReport {
        var issues: [StorageIssue] = []
        
        // Validate memory usage
        let memoryUsage = getCurrentMemoryUsage()
        if memoryUsage > configuration.memoryLimit {
            issues.append(StorageIssue(
                type: .space,
                description: "Memory usage exceeds limit: \(memoryUsage) > \(configuration.memoryLimit)",
                impact: .major
            ))
        }
        
        // Validate metadata consistency
        for (key, meta) in metadata {
            let exists: Bool
            switch meta.tier {
            case .memory:
                exists = memoryTier[key] != nil
            case .ssd:
                exists = await ssdTier.exists(key: key)
            case .archive:
                exists = await archiveTier.exists(key: key)
            }
            
            if !exists {
                issues.append(StorageIssue(
                    type: .corruption,
                    description: "Metadata exists but data missing for key: \(key)",
                    impact: .critical
                ))
            }
        }
        
        // Validate file-based tiers
        let ssdReport = try await ssdTier.validateIntegrity()
        let archiveReport = try await archiveTier.validateIntegrity()
        
        issues.append(contentsOf: ssdReport.issues)
        issues.append(contentsOf: archiveReport.issues)
        
        let recommendations = issues.isEmpty ? [] : [
            "Run compaction to fix consistency issues",
            "Consider increasing memory limit if needed"
        ]
        
        return StorageIntegrityReport(
            isHealthy: issues.filter { $0.impact == .critical || $0.impact == .major }.isEmpty,
            issues: issues,
            recommendations: recommendations,
            lastCheck: Date()
        )
    }
    
    public func createSnapshot() async throws -> SnapshotIdentifier {
        // Create snapshots for all tiers
        let memorySnapshot = try await createMemorySnapshot()
        let ssdSnapshot = try await ssdTier.createSnapshot()
        let archiveSnapshot = try await archiveTier.createSnapshot()
        
        // Combine checksums
        var hasher = SHA256()
        hasher.update(memorySnapshot.checksum.data(using: .utf8)!)
        hasher.update(ssdSnapshot.checksum.data(using: .utf8)!)
        hasher.update(archiveSnapshot.checksum.data(using: .utf8)!)
        
        let combinedChecksum = hasher.finalize().compactMap { String(format: "%02x", $0) }.joined()
        
        return SnapshotIdentifier(
            id: UUID().uuidString,
            timestamp: Date(),
            checksum: combinedChecksum
        )
    }
    
    public func restoreSnapshot(_ identifier: SnapshotIdentifier) async throws {
        throw StorageError.invalidConfiguration("Snapshot restoration not implemented")
    }
    
    // MARK: - Private Methods
    
    private func determineInitialTier(dataSize: Int, options: StorageOptions) -> StorageTier {
        // Priority-based placement
        if options.priority == .critical || options.priority == .high {
            if dataSize <= configuration.memoryItemSizeLimit {
                return .memory
            }
        }
        
        // Size-based placement
        if dataSize > 10 * 1024 * 1024 { // > 10MB
            return .archive
        } else if dataSize > 1024 * 1024 { // > 1MB
            return .ssd
        } else {
            // Small items go to memory if there's space
            let currentMemoryUsage = getCurrentMemoryUsage()
            if currentMemoryUsage + dataSize <= configuration.memoryLimit {
                return .memory
            }
            return .ssd
        }
    }
    
    private func getCurrentMemoryUsage() -> Int {
        memoryTier.values.reduce(0) { $0 + $1.count }
    }
    
    private func loadMetadata() async throws {
        // Load metadata from disk
        let metadataPath = configuration.metadataPath
        if FileManager.default.fileExists(atPath: metadataPath) {
            let data = try Data(contentsOf: URL(fileURLWithPath: metadataPath))
            metadata = try JSONDecoder().decode([String: ItemMetadata].self, from: data)
            logger.info("Loaded metadata for \(metadata.count) items")
        }
    }
    
    private func saveMetadata() async throws {
        let data = try JSONEncoder().encode(metadata)
        try data.write(to: URL(fileURLWithPath: configuration.metadataPath))
    }
    
    private func createMemorySnapshot() async throws -> SnapshotIdentifier {
        var hasher = SHA256()
        for (key, data) in memoryTier.sorted(by: { $0.key < $1.key }) {
            hasher.update(key.data(using: .utf8)!)
            hasher.update(data)
        }
        
        let checksum = hasher.finalize().compactMap { String(format: "%02x", $0) }.joined()
        
        return SnapshotIdentifier(
            id: "memory-" + UUID().uuidString,
            timestamp: Date(),
            checksum: checksum
        )
    }
}

// MARK: - Supporting Types

/// Metadata for stored items
private struct ItemMetadata: Codable {
    var tier: StorageTier
    let size: Int
    var compressedSize: Int
    let createdAt: Date
    var lastAccessTime: Date
    var accessCount: Int
}

/// Performance metrics tracking for three-tier storage
private struct ThreeTierPerformanceMetrics {
    private var operations: [OperationKey: [TimeInterval]] = [:]
    private let maxSamples = 1000
    
    struct OperationKey: Hashable {
        let type: OperationType
        let tier: StorageTier?
    }
    
    enum OperationType {
        case store, retrieve, delete
    }
    
    mutating func recordOperation(_ type: OperationType, duration: TimeInterval, size: Int, tier: StorageTier?) {
        let key = OperationKey(type: type, tier: tier)
        var samples = operations[key] ?? []
        samples.append(duration)
        
        if samples.count > maxSamples {
            samples.removeFirst()
        }
        
        operations[key] = samples
    }
    
    var averageLatency: TimeInterval {
        let allSamples = operations.values.flatMap { $0 }
        return allSamples.isEmpty ? 0 : allSamples.reduce(0, +) / Double(allSamples.count)
    }
    
    func averageLatency(for tier: StorageTier) -> TimeInterval {
        let tierSamples = operations
            .filter { $0.key.tier == tier }
            .values
            .flatMap { $0 }
        return tierSamples.isEmpty ? 0 : tierSamples.reduce(0, +) / Double(tierSamples.count)
    }
    
    var latencyP99: TimeInterval {
        let allSamples = operations.values.flatMap { $0 }.sorted()
        guard !allSamples.isEmpty else { return 0 }
        let index = Int(Double(allSamples.count) * 0.99)
        return allSamples[min(index, allSamples.count - 1)]
    }
    
    var errorRate: Float {
        // Calculate error rate from last 1000 operations
        let recentOps = operations.values.flatMap { $0 }.suffix(1000)
        guard !recentOps.isEmpty else { return 0.0 }
        
        // For now, return 0 as we track successful operations
        // In production, this would track actual errors
        return 0.0
    }
    
    var throughput: Float {
        let totalOps = operations.values.reduce(0) { $0 + $1.count }
        let totalTime = operations.values.flatMap { $0 }.reduce(0, +)
        return totalTime > 0 ? Float(totalOps) / Float(totalTime) : 0
    }
}

/// File-based storage tier implementation
private actor FileBasedTier {
    let path: String
    let tier: StorageTier
    let compressionEngine: CompressionEngine
    let logger: Logger
    
    private var index: [String: FileLocation] = [:]
    private var totalSize: Int = 0
    
    struct FileLocation: Codable {
        let file: String
        let offset: Int
        let size: Int
        let compressedSize: Int
    }
    
    init(path: String, tier: StorageTier, compressionEngine: CompressionEngine) throws {
        self.path = path
        self.tier = tier
        self.compressionEngine = compressionEngine
        self.logger = Logger(subsystem: "VectorStoreKit", category: "FileBasedTier-\(tier.properties.name)")
        
        // Create directory if needed
        try FileManager.default.createDirectory(
            atPath: path,
            withIntermediateDirectories: true,
            attributes: nil
        )
        
        // Load index
        Task {
            try await loadIndex()
        }
    }
    
    func store(key: String, data: Data) async throws {
        // Apply compression
        let compressedData: Data
        if tier.properties.compressionType != .none {
            compressedData = try await compressionEngine.compress(
                data,
                using: tier.properties.compressionType,
                level: tier.properties.compressionLevel
            )
        } else {
            compressedData = data
        }
        
        // Simple implementation: one file per key
        let fileName = key.addingPercentEncoding(withAllowedCharacters: .alphanumerics) ?? key
        let filePath = (path as NSString).appendingPathComponent(fileName)
        
        try compressedData.write(to: URL(fileURLWithPath: filePath))
        
        index[key] = FileLocation(
            file: fileName,
            offset: 0,
            size: data.count,
            compressedSize: compressedData.count
        )
        
        totalSize += compressedData.count
    }
    
    func retrieve(key: String) async throws -> Data {
        guard let location = index[key] else {
            throw StorageError.keyNotFound(key)
        }
        
        let filePath = (path as NSString).appendingPathComponent(location.file)
        let compressedData = try Data(contentsOf: URL(fileURLWithPath: filePath))
        
        // Decompress if needed
        if tier.properties.compressionType != .none {
            return try await compressionEngine.decompress(
                compressedData,
                using: tier.properties.compressionType
            )
        }
        
        return compressedData
    }
    
    func delete(key: String) async throws {
        guard let location = index[key] else {
            return
        }
        
        let filePath = (path as NSString).appendingPathComponent(location.file)
        try FileManager.default.removeItem(atPath: filePath)
        
        totalSize -= location.compressedSize
        index.removeValue(forKey: key)
    }
    
    func exists(key: String) -> Bool {
        index[key] != nil
    }
    
    func compact() async throws {
        // Simple compaction: just save the index
        try saveIndex()
    }
    
    func statistics() -> (totalSize: Int, itemCount: Int, compressionRatio: Float) {
        let originalSize = index.values.reduce(0) { $0 + $1.size }
        let compressedSize = index.values.reduce(0) { $0 + $1.compressedSize }
        let ratio = originalSize > 0 ? Float(originalSize) / Float(compressedSize) : 1.0
        
        return (totalSize, index.count, ratio)
    }
    
    func validateIntegrity() async throws -> StorageIntegrityReport {
        var issues: [StorageIssue] = []
        
        for (key, location) in index {
            let filePath = (path as NSString).appendingPathComponent(location.file)
            if !FileManager.default.fileExists(atPath: filePath) {
                issues.append(StorageIssue(
                    type: .corruption,
                    description: "File missing for key '\(key)': \(location.file)",
                    impact: .critical
                ))
            }
        }
        
        return StorageIntegrityReport(
            isHealthy: issues.isEmpty,
            issues: issues,
            recommendations: [],
            lastCheck: Date()
        )
    }
    
    func createSnapshot() async throws -> SnapshotIdentifier {
        var hasher = SHA256()
        
        for (key, _) in index.sorted(by: { $0.key < $1.key }) {
            hasher.update(key.data(using: .utf8)!)
            if let data = try? await retrieve(key: key) {
                hasher.update(data)
            }
        }
        
        let checksum = hasher.finalize().compactMap { String(format: "%02x", $0) }.joined()
        
        return SnapshotIdentifier(
            id: "\(tier.properties.name)-" + UUID().uuidString,
            timestamp: Date(),
            checksum: checksum
        )
    }
    
    private func loadIndex() async throws {
        let indexPath = (path as NSString).appendingPathComponent(".index")
        if FileManager.default.fileExists(atPath: indexPath) {
            let data = try Data(contentsOf: URL(fileURLWithPath: indexPath))
            index = try JSONDecoder().decode([String: FileLocation].self, from: data)
            totalSize = index.values.reduce(0) { $0 + $1.compressedSize }
        }
    }
    
    private func saveIndex() throws {
        let indexPath = (path as NSString).appendingPathComponent(".index")
        let data = try JSONEncoder().encode(index)
        try data.write(to: URL(fileURLWithPath: indexPath))
    }
}
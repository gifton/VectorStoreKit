// VectorStoreKit: Simple Memory Storage Backend
//
// A lightweight, memory-only storage backend optimized for development and testing.
// This replaces the complex 4-tier storage system with a straightforward dictionary-based approach.

import Foundation
import CryptoKit
import os.log

/// Simple memory-only storage backend using a dictionary for key-value storage
///
/// This storage backend provides:
/// - Fast in-memory operations with O(1) access time
/// - Thread-safe access through actor isolation
/// - Optional compression for memory efficiency
/// - Basic statistics and monitoring
///
/// **Performance Characteristics:**
/// - Access latency: <100ns
/// - Throughput: Limited only by memory bandwidth
/// - Capacity: Limited by available system memory
public actor SimpleStorage: StorageBackend {
    
    // MARK: - Configuration
    
    /// Configuration for simple storage
    public typealias Configuration = SimpleStorageConfiguration
    
    /// Simple storage statistics
    public struct Statistics: StorageStatistics, Codable {
        /// Total storage size in bytes
        public let totalSize: Int
        
        /// Compression ratio (1.0 = no compression)
        public let compressionRatio: Float
        
        /// Average access latency
        public let averageLatency: TimeInterval
        
        /// Storage health metrics
        public let healthMetrics: StorageHealthMetrics
        
        /// Number of stored items
        public let itemCount: Int
        
        /// Memory usage breakdown
        public let memoryUsage: MemoryUsage
        
        internal init(
            totalSize: Int,
            compressionRatio: Float,
            averageLatency: TimeInterval,
            healthMetrics: StorageHealthMetrics,
            itemCount: Int,
            memoryUsage: MemoryUsage
        ) {
            self.totalSize = totalSize
            self.compressionRatio = compressionRatio
            self.averageLatency = averageLatency
            self.healthMetrics = healthMetrics
            self.itemCount = itemCount
            self.memoryUsage = memoryUsage
        }
    }
    
    /// Memory usage breakdown
    public struct MemoryUsage: Codable, Sendable {
        public let dataSize: Int
        public let metadataSize: Int
        public let overheadSize: Int
        
        public var totalSize: Int {
            dataSize + metadataSize + overheadSize
        }
    }
    
    // MARK: - Private Properties
    
    /// Main storage dictionary
    private var storage: [String: StoredItem] = [:]
    
    /// Configuration
    public let configuration: Configuration
    
    /// Logger for debugging
    private let logger = Logger(subsystem: "VectorStoreKit", category: "SimpleStorage")
    
    /// Performance metrics
    private var metrics = PerformanceMetrics()
    
    /// Ready state
    private var isReadyState = true
    
    // MARK: - StorageBackend Protocol Properties
    
    public var isReady: Bool {
        isReadyState
    }
    
    public var size: Int {
        storage.values.reduce(0) { $0 + $1.data.count }
    }
    
    // MARK: - Initialization
    
    /// Initialize simple storage with configuration
    public init(configuration: Configuration) async throws {
        try configuration.validate()
        self.configuration = configuration
        
        // Reserve initial capacity if specified
        if configuration.initialCapacity > 0 {
            storage.reserveCapacity(configuration.initialCapacity)
        }
        
        logger.info("Initialized simple storage with capacity: \(configuration.memoryLimit) bytes")
    }
    
    // MARK: - Core StorageBackend Operations
    
    /// Store data with specified key
    public func store(key: String, data: Data, options: StorageOptions) async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Check memory limit
        let newSize = size + data.count
        if newSize > configuration.memoryLimit {
            // Apply eviction policy
            try await evictIfNeeded(requiredSpace: data.count)
        }
        
        // Optionally compress data
        let storedData: Data
        let compressionRatio: Float
        
        if configuration.enableCompression && data.count > configuration.compressionThreshold {
            if let compressed = compress(data) {
                storedData = compressed
                compressionRatio = Float(data.count) / Float(compressed.count)
            } else {
                storedData = data
                compressionRatio = 1.0
            }
        } else {
            storedData = data
            compressionRatio = 1.0
        }
        
        // Create stored item
        let item = StoredItem(
            data: storedData,
            originalSize: data.count,
            compressionRatio: compressionRatio,
            timestamp: Date(),
            accessCount: 0,
            lastAccess: Date(),
            options: options
        )
        
        // Store the item
        storage[key] = item
        
        // Update metrics
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        metrics.recordOperation(.store, duration: duration, size: data.count)
        
        logger.debug("Stored key '\(key)' with size \(data.count) bytes (compressed: \(storedData.count) bytes)")
    }
    
    /// Retrieve data by key
    public func retrieve(key: String) async throws -> Data? {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        guard var item = storage[key] else {
            metrics.recordOperation(.retrieve, duration: CFAbsoluteTimeGetCurrent() - startTime, size: 0)
            return nil
        }
        
        // Update access tracking
        item.accessCount += 1
        item.lastAccess = Date()
        storage[key] = item
        
        // Decompress if needed
        let data: Data
        if item.compressionRatio > 1.0 {
            if let decompressed = decompress(item.data, originalSize: item.originalSize) {
                data = decompressed
            } else {
                throw StorageError.retrieveFailed("Failed to decompress data for key '\(key)'")
            }
        } else {
            data = item.data
        }
        
        // Update metrics
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        metrics.recordOperation(.retrieve, duration: duration, size: data.count)
        
        logger.debug("Retrieved key '\(key)' with size \(data.count) bytes")
        
        return data
    }
    
    /// Delete data by key
    public func delete(key: String) async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let removedItem = storage.removeValue(forKey: key)
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        metrics.recordOperation(.delete, duration: duration, size: removedItem?.data.count ?? 0)
        
        if removedItem != nil {
            logger.debug("Deleted key '\(key)'")
        }
    }
    
    /// Check if key exists
    public func exists(key: String) async -> Bool {
        storage[key] != nil
    }
    
    /// Scan keys with prefix
    public func scan(prefix: String) async throws -> AsyncStream<(String, Data)> {
        AsyncStream { continuation in
            Task {
                do {
                    for (key, item) in storage where key.hasPrefix(prefix) {
                        // Decompress if needed
                        let data: Data
                        if item.compressionRatio > 1.0 {
                            if let decompressed = decompress(item.data, originalSize: item.originalSize) {
                                data = decompressed
                            } else {
                                continuation.finish(throwing: StorageError.retrieveFailed("Failed to decompress data for key '\(key)'"))
                                return
                            }
                        } else {
                            data = item.data
                        }
                        
                        continuation.yield((key, data))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Advanced Operations
    
    /// Compact storage (no-op for memory storage)
    public func compact() async throws {
        // Memory storage doesn't need compaction
        logger.info("Compact called on memory storage (no-op)")
    }
    
    /// Get storage statistics
    public func statistics() async -> Statistics {
        let dataSize = storage.values.reduce(0) { $0 + $1.data.count }
        let originalSize = storage.values.reduce(0) { $0 + $1.originalSize }
        let compressionRatio = originalSize > 0 ? Float(originalSize) / Float(dataSize) : 1.0
        
        let memoryUsage = MemoryUsage(
            dataSize: dataSize,
            metadataSize: storage.count * MemoryLayout<StoredItem>.size,
            overheadSize: storage.count * 32 // Estimated dictionary overhead per entry
        )
        
        let healthMetrics = StorageHealthMetrics(
            errorRate: metrics.errorRate,
            latencyP99: metrics.latencyP99,
            throughput: metrics.throughput,
            fragmentation: 0.0 // No fragmentation in memory storage
        )
        
        return Statistics(
            totalSize: dataSize,
            compressionRatio: compressionRatio,
            averageLatency: metrics.averageLatency,
            healthMetrics: healthMetrics,
            itemCount: storage.count,
            memoryUsage: memoryUsage
        )
    }
    
    /// Validate storage integrity
    public func validateIntegrity() async throws -> StorageIntegrityReport {
        var issues: [StorageIssue] = []
        
        // Check memory usage
        let currentSize = size
        if currentSize > configuration.memoryLimit {
            issues.append(StorageIssue(
                type: .space,
                description: "Memory usage exceeds limit: \(currentSize) > \(configuration.memoryLimit)",
                impact: .major
            ))
        }
        
        // Check for compression errors
        var compressionErrors = 0
        for (key, item) in storage {
            if item.compressionRatio > 1.0 {
                if decompress(item.data, originalSize: item.originalSize) == nil {
                    compressionErrors += 1
                    issues.append(StorageIssue(
                        type: .corruption,
                        description: "Compression corruption detected for key: \(key)",
                        impact: .critical
                    ))
                }
            }
        }
        
        let recommendations = issues.isEmpty ? [] : ["Consider increasing memory limit or enabling eviction"]
        
        return StorageIntegrityReport(
            isHealthy: issues.filter { $0.impact == .critical || $0.impact == .major }.isEmpty,
            issues: issues,
            recommendations: recommendations,
            lastCheck: Date()
        )
    }
    
    /// Create a snapshot (returns current state)
    public func createSnapshot() async throws -> SnapshotIdentifier {
        let snapshotId = UUID().uuidString
        let timestamp = Date()
        
        // Calculate checksum of all data
        var hasher = SHA256()
        for (key, item) in storage.sorted(by: { $0.key < $1.key }) {
            hasher.update(key.data(using: .utf8)!)
            hasher.update(item.data)
        }
        
        let checksum = hasher.finalize().compactMap { String(format: "%02x", $0) }.joined()
        
        return SnapshotIdentifier(
            id: snapshotId,
            timestamp: timestamp,
            checksum: checksum
        )
    }
    
    /// Restore from snapshot (not implemented for memory storage)
    public func restoreSnapshot(_ identifier: SnapshotIdentifier) async throws {
        throw StorageError.invalidConfiguration("Snapshot restoration not supported for memory storage")
    }
    
    // MARK: - Batch Operations
    
    /// Store multiple items in a batch
    public func batchStore(_ items: [(key: String, data: Data, options: StorageOptions)]) async throws {
        for (key, data, options) in items {
            try await store(key: key, data: data, options: options)
        }
    }
    
    /// Retrieve multiple items in a batch
    public func batchRetrieve(_ keys: [String]) async throws -> [String: Data?] {
        var results: [String: Data?] = [:]
        
        for key in keys {
            results[key] = try await retrieve(key: key)
        }
        
        return results
    }
    
    /// Delete multiple items in a batch
    public func batchDelete(_ keys: [String]) async throws {
        for key in keys {
            try await delete(key: key)
        }
    }
}

// MARK: - Private Implementation

private extension SimpleStorage {
    
    /// Stored item with metadata
    struct StoredItem {
        let data: Data
        let originalSize: Int
        let compressionRatio: Float
        let timestamp: Date
        var accessCount: Int
        var lastAccess: Date
        let options: StorageOptions
    }
    
    /// Performance metrics tracking
    struct PerformanceMetrics {
        private var operations: [OperationType: [TimeInterval]] = [:]
        private let maxSamples = 1000
        
        mutating func recordOperation(_ type: OperationType, duration: TimeInterval, size: Int) {
            var samples = operations[type] ?? []
            samples.append(duration)
            
            if samples.count > maxSamples {
                samples.removeFirst()
            }
            
            operations[type] = samples
        }
        
        var averageLatency: TimeInterval {
            let allSamples = operations.values.flatMap { $0 }
            return allSamples.isEmpty ? 0 : allSamples.reduce(0, +) / Double(allSamples.count)
        }
        
        var latencyP99: TimeInterval {
            let allSamples = operations.values.flatMap { $0 }.sorted()
            guard !allSamples.isEmpty else { return 0 }
            let index = Int(Double(allSamples.count) * 0.99)
            return allSamples[min(index, allSamples.count - 1)]
        }
        
        var errorRate: Float {
            0.0 // Simple storage doesn't track errors
        }
        
        var throughput: Float {
            let totalOps = operations.values.reduce(0) { $0 + $1.count }
            let totalTime = operations.values.flatMap { $0 }.reduce(0, +)
            return totalTime > 0 ? Float(totalOps) / Float(totalTime) : 0
        }
    }
    
    enum OperationType {
        case store, retrieve, delete
    }
    
    /// Evict items if needed to make space
    func evictIfNeeded(requiredSpace: Int) async throws {
        let currentSize = size
        let targetSize = configuration.memoryLimit - requiredSpace
        
        guard currentSize > targetSize else { return }
        
        let bytesToEvict = currentSize - targetSize
        var evictedBytes = 0
        
        // Sort by eviction policy
        let sortedKeys: [String]
        switch configuration.evictionPolicy {
        case .lru:
            sortedKeys = storage.sorted { $0.value.lastAccess < $1.value.lastAccess }.map { $0.key }
        case .lfu:
            sortedKeys = storage.sorted { $0.value.accessCount < $1.value.accessCount }.map { $0.key }
        case .fifo:
            sortedKeys = storage.sorted { $0.value.timestamp < $1.value.timestamp }.map { $0.key }
        default:
            sortedKeys = Array(storage.keys).shuffled() // Random eviction
        }
        
        // Evict until we have enough space
        for key in sortedKeys {
            guard evictedBytes < bytesToEvict else { break }
            
            if let item = storage.removeValue(forKey: key) {
                evictedBytes += item.data.count
                logger.debug("Evicted key '\(key)' to free \(item.data.count) bytes")
            }
        }
        
        if evictedBytes < bytesToEvict {
            throw StorageError.storeFailed("Unable to evict enough space: needed \(bytesToEvict), evicted \(evictedBytes)")
        }
    }
    
    /// Compress data using zlib
    func compress(_ data: Data) -> Data? {
        guard configuration.enableCompression else { return nil }
        
        return data.withUnsafeBytes { bytes in
            guard let compressed = try? (bytes.bindMemory(to: UInt8.self).compressed(using: .zlib)) else {
                return nil
            }
            return compressed
        }
    }
    
    /// Decompress data using zlib
    func decompress(_ data: Data, originalSize: Int) -> Data? {
        return data.withUnsafeBytes { bytes in
            guard let decompressed = try? (bytes.bindMemory(to: UInt8.self).decompressed(using: .zlib)) else {
                return nil
            }
            return decompressed
        }
    }
}

// MARK: - Compression Extensions

extension Sequence where Element == UInt8 {
    func compressed(using algorithm: NSData.CompressionAlgorithm) throws -> Data {
        return try (Data(self) as NSData).compressed(using: algorithm) as Data
    }
    
    func decompressed(using algorithm: NSData.CompressionAlgorithm) throws -> Data {
        return try (Data(self) as NSData).decompressed(using: algorithm) as Data
    }
}
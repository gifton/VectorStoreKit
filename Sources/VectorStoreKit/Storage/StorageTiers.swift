// VectorStoreKit: Storage Tier Implementations
//
// Concrete implementations of storage tiers for HierarchicalStorage

import Foundation
import Compression
import CryptoKit
import os.log

// MARK: - Hot Tier Storage

/// In-memory storage tier for frequently accessed data
actor HotTierStorage {
    private let memoryLimit: Int
    private var storage: [String: CacheEntry] = [:]
    private var accessOrder: [String] = []
    private let logger = Logger(subsystem: "VectorStoreKit", category: "HotTier")
    
    struct CacheEntry {
        let data: Data
        let timestamp: Date
        var accessCount: Int
        var lastAccessed: Date
    }
    
    init(memoryLimit: Int) {
        self.memoryLimit = memoryLimit
    }
    
    var size: Int {
        storage.values.reduce(0) { $0 + $1.data.count }
    }
    
    var allocatedSize: Int { size }
    
    func store(key: String, data: Data) {
        // Check if we need to evict
        var currentSize = size
        while currentSize + data.count > memoryLimit && !accessOrder.isEmpty {
            let evictKey = accessOrder.removeFirst()
            if let entry = storage.removeValue(forKey: evictKey) {
                currentSize -= entry.data.count
                logger.debug("Evicted key from hot tier: \(evictKey)")
            }
        }
        
        // Store new entry
        let entry = CacheEntry(
            data: data,
            timestamp: Date(),
            accessCount: 1,
            lastAccessed: Date()
        )
        storage[key] = entry
        accessOrder.append(key)
    }
    
    func retrieve(key: String) -> Data? {
        guard var entry = storage[key] else { return nil }
        
        // Update access info
        entry.accessCount += 1
        entry.lastAccessed = Date()
        storage[key] = entry
        
        // Move to end of access order
        if let index = accessOrder.firstIndex(of: key) {
            accessOrder.remove(at: index)
            accessOrder.append(key)
        }
        
        return entry.data
    }
    
    func delete(key: String) {
        storage.removeValue(forKey: key)
        accessOrder.removeAll { $0 == key }
    }
    
    func exists(key: String) -> Bool {
        storage[key] != nil
    }
    
    func hasCapacity(for size: Int) -> Bool {
        self.size + size <= memoryLimit
    }
    
    func scan(prefix: String, callback: (String, Data) -> Void) {
        for (key, entry) in storage where key.hasPrefix(prefix) {
            callback(key, entry.data)
        }
    }
    
    func statistics() -> TierStatistics {
        let totalSize = size
        let itemCount = storage.count
        let avgLatency: TimeInterval = 0.00001 // ~10 microseconds
        let hitRate: Float = itemCount > 0 ? 1.0 : 0.0
        
        return TierStatistics(
            size: totalSize,
            originalSize: totalSize,
            itemCount: itemCount,
            averageLatency: avgLatency,
            hitRate: hitRate
        )
    }
    
    func validateIntegrity() -> StorageIntegrityReport {
        var issues: [StorageIssue] = []
        
        // Check memory usage
        if size > memoryLimit {
            issues.append(StorageIssue(
                type: .space,
                description: "Hot tier exceeds memory limit",
                impact: .critical
            ))
        }
        
        // Check for orphaned entries in access order
        for key in accessOrder {
            if storage[key] == nil {
                issues.append(StorageIssue(
                    type: .corruption,
                    description: "Orphaned key in access order: \(key)",
                    impact: .minor
                ))
            }
        }
        
        return StorageIntegrityReport(
            isHealthy: issues.isEmpty,
            issues: issues,
            recommendations: issues.isEmpty ? [] : ["Run optimize() to fix issues"],
            lastCheck: Date()
        )
    }
    
    func createSnapshot(id: String) -> Data {
        // Create snapshot of current state
        let snapshot = storage.mapValues { $0.data }
        return (try? JSONEncoder().encode(snapshot)) ?? Data()
    }
    
    func restoreSnapshot(_ identifier: SnapshotIdentifier) throws {
        // Not implemented for hot tier - data is transient
    }
    
    func optimize() {
        // Clean up access order
        accessOrder = accessOrder.filter { storage[$0] != nil }
        
        // Evict least recently used if over limit
        var currentSize = size
        while currentSize > memoryLimit && !accessOrder.isEmpty {
            let evictKey = accessOrder.removeFirst()
            if let entry = storage.removeValue(forKey: evictKey) {
                currentSize -= entry.data.count
            }
        }
    }
}

// MARK: - Warm Tier Storage

/// Memory-mapped file storage for balanced performance
actor WarmTierStorage {
    private let baseDirectory: URL
    private let fileSizeLimit: Int
    private var fileHandles: [String: FileHandle] = [:]
    private var metadata: [String: FileMetadata] = [:]
    private let logger = Logger(subsystem: "VectorStoreKit", category: "WarmTier")
    
    struct FileMetadata {
        let size: Int
        let createdAt: Date
        var lastAccessed: Date
        var accessCount: Int
    }
    
    init(baseDirectory: URL, fileSizeLimit: Int) async throws {
        self.baseDirectory = baseDirectory
        self.fileSizeLimit = fileSizeLimit
        
        // Create directory if needed
        try FileManager.default.createDirectory(
            at: baseDirectory.appendingPathComponent("warm"),
            withIntermediateDirectories: true
        )
        
        // Load existing metadata
        try await loadMetadata()
    }
    
    var size: Int {
        metadata.values.reduce(0) { $0 + $1.size }
    }
    
    var allocatedSize: Int { size }
    
    func store(key: String, data: Data, options: StorageOptions) async throws {
        let fileURL = urlForKey(key)
        
        // Write data to file
        try data.write(to: fileURL, options: .atomic)
        
        // Update metadata
        metadata[key] = FileMetadata(
            size: data.count,
            createdAt: Date(),
            lastAccessed: Date(),
            accessCount: 1
        )
        
        logger.debug("Stored key in warm tier: \(key)")
    }
    
    func retrieve(key: String) async throws -> Data? {
        guard var meta = metadata[key] else { return nil }
        
        let fileURL = urlForKey(key)
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            metadata.removeValue(forKey: key)
            return nil
        }
        
        // Update access info
        meta.lastAccessed = Date()
        meta.accessCount += 1
        metadata[key] = meta
        
        // Read data
        return try Data(contentsOf: fileURL)
    }
    
    func delete(key: String) async throws {
        let fileURL = urlForKey(key)
        try? FileManager.default.removeItem(at: fileURL)
        metadata.removeValue(forKey: key)
        fileHandles.removeValue(forKey: key)?.closeFile()
    }
    
    func exists(key: String) async -> Bool {
        metadata[key] != nil && FileManager.default.fileExists(atPath: urlForKey(key).path)
    }
    
    func scan(prefix: String, callback: (String, Data) -> Void) async throws {
        for (key, _) in metadata where key.hasPrefix(prefix) {
            if let data = try await retrieve(key: key) {
                callback(key, data)
            }
        }
    }
    
    func compact() async throws {
        // Remove orphaned files
        let warmDir = baseDirectory.appendingPathComponent("warm")
        let contents = try FileManager.default.contentsOfDirectory(
            at: warmDir,
            includingPropertiesForKeys: [.fileSizeKey]
        )
        
        for fileURL in contents {
            let key = fileURL.deletingPathExtension().lastPathComponent
            if metadata[key] == nil {
                try FileManager.default.removeItem(at: fileURL)
                logger.debug("Removed orphaned file: \(fileURL.lastPathComponent)")
            }
        }
    }
    
    func statistics() async -> TierStatistics {
        return TierStatistics(
            size: size,
            originalSize: size,
            itemCount: metadata.count,
            averageLatency: 0.00005, // ~50 microseconds
            hitRate: 0.9 // Estimated
        )
    }
    
    func validateIntegrity() async -> StorageIntegrityReport {
        var issues: [StorageIssue] = []
        
        // Check for missing files
        for (key, _) in metadata {
            let fileURL = urlForKey(key)
            if !FileManager.default.fileExists(atPath: fileURL.path) {
                issues.append(StorageIssue(
                    type: .corruption,
                    description: "Missing file for key: \(key)",
                    impact: .major
                ))
            }
        }
        
        return StorageIntegrityReport(
            isHealthy: issues.isEmpty,
            issues: issues,
            recommendations: issues.isEmpty ? [] : ["Run compact() to fix issues"],
            lastCheck: Date()
        )
    }
    
    func createSnapshot(id: String) async throws -> Data {
        // Create metadata snapshot
        return try JSONEncoder().encode(metadata)
    }
    
    func restoreSnapshot(_ identifier: SnapshotIdentifier) async throws {
        // Not implemented for warm tier
    }
    
    func optimize() async throws {
        // Could implement file defragmentation here
        try await compact()
    }
    
    func batchStore(_ items: [(String, Data, StorageOptions)]) async throws {
        for (key, data, options) in items {
            try await store(key: key, data: data, options: options)
        }
    }
    
    // MARK: - Private Helpers
    
    private func urlForKey(_ key: String) -> URL {
        baseDirectory
            .appendingPathComponent("warm")
            .appendingPathComponent("\(key).dat")
    }
    
    private func loadMetadata() async throws {
        let warmDir = baseDirectory.appendingPathComponent("warm")
        guard FileManager.default.fileExists(atPath: warmDir.path) else { return }
        
        let contents = try FileManager.default.contentsOfDirectory(
            at: warmDir,
            includingPropertiesForKeys: [.fileSizeKey, .creationDateKey]
        )
        
        for fileURL in contents {
            let key = fileURL.deletingPathExtension().lastPathComponent
            let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
            
            metadata[key] = FileMetadata(
                size: attributes[.size] as? Int ?? 0,
                createdAt: attributes[.creationDate] as? Date ?? Date(),
                lastAccessed: Date(),
                accessCount: 0
            )
        }
    }
}

// MARK: - Cold Tier Storage

/// Compressed disk storage for infrequently accessed data
actor ColdTierStorage {
    private let baseDirectory: URL
    private let compression: CompressionAlgorithm
    private let encryptionEngine: EncryptionEngine?
    private var index: [String: IndexEntry] = [:]
    private let logger = Logger(subsystem: "VectorStoreKit", category: "ColdTier")
    
    struct IndexEntry {
        let fileOffset: Int
        let compressedSize: Int
        let originalSize: Int
        let checksum: String
        var lastAccessed: Date
    }
    
    init(baseDirectory: URL, compression: CompressionAlgorithm, encryptionEngine: EncryptionEngine?) async throws {
        self.baseDirectory = baseDirectory
        self.compression = compression
        self.encryptionEngine = encryptionEngine
        
        // Create directory if needed
        try FileManager.default.createDirectory(
            at: baseDirectory.appendingPathComponent("cold"),
            withIntermediateDirectories: true
        )
        
        // Load existing index
        try await loadIndex()
    }
    
    var size: Int {
        index.values.reduce(0) { $0 + $1.compressedSize }
    }
    
    private var _allocatedSize: Int {
        // Include index overhead
        size + (index.count * 128) // Estimated index entry size
    }
    
    func store(key: String, data: Data, options: StorageOptions) async throws {
        // Compress data
        let compressedData = try compressData(data)
        
        // Encrypt if needed
        let finalData = try encryptionEngine?.encrypt(compressedData) ?? compressedData
        
        // Calculate checksum
        let checksum = SHA256.hash(data: data).compactMap { 
            String(format: "%02x", $0) 
        }.joined()
        
        // Write to pack file
        let packFile = getCurrentPackFile()
        let fileHandle = try FileHandle(forWritingTo: packFile)
        defer { try? fileHandle.close() }
        
        let offset = try fileHandle.seekToEnd()
        try fileHandle.write(contentsOf: finalData)
        
        // Update index
        index[key] = IndexEntry(
            fileOffset: Int(offset),
            compressedSize: finalData.count,
            originalSize: data.count,
            checksum: checksum,
            lastAccessed: Date()
        )
        
        // Save index
        try await saveIndex()
        
        logger.debug("Stored key in cold tier: \(key), compressed: \(data.count) -> \(finalData.count)")
    }
    
    func retrieve(key: String) async throws -> Data? {
        guard var entry = index[key] else { return nil }
        
        let packFile = getCurrentPackFile()
        let fileHandle = try FileHandle(forReadingFrom: packFile)
        defer { try? fileHandle.close() }
        
        // Read compressed data
        try fileHandle.seek(toOffset: UInt64(entry.fileOffset))
        let compressedData = fileHandle.readData(ofLength: entry.compressedSize)
        
        // Decrypt if needed
        let decryptedData = try encryptionEngine?.decrypt(compressedData) ?? compressedData
        
        // Decompress
        let data = try decompressData(decryptedData)
        
        // Verify checksum
        let checksum = SHA256.hash(data: data).compactMap { 
            String(format: "%02x", $0) 
        }.joined()
        
        guard checksum == entry.checksum else {
            throw StorageError.corruptedData("Checksum mismatch for key: \(key)")
        }
        
        // Update access time
        entry.lastAccessed = Date()
        index[key] = entry
        
        return data
    }
    
    func delete(key: String) async throws {
        // Mark as deleted in index (actual deletion happens during compaction)
        index.removeValue(forKey: key)
        try await saveIndex()
    }
    
    func exists(key: String) async -> Bool {
        index[key] != nil
    }
    
    func scan(prefix: String, callback: (String, Data) -> Void) async throws {
        for key in index.keys where key.hasPrefix(prefix) {
            if let data = try await retrieve(key: key) {
                callback(key, data)
            }
        }
    }
    
    func compact() async throws {
        logger.info("Starting cold tier compaction")
        
        // Create new pack file
        let newPackFile = baseDirectory
            .appendingPathComponent("cold")
            .appendingPathComponent("pack-\(UUID().uuidString).dat")
        
        let newHandle = try FileHandle(forWritingTo: newPackFile)
        defer { try? newHandle.close() }
        
        var newIndex: [String: IndexEntry] = [:]
        
        // Copy all valid entries to new pack file
        for (key, entry) in index {
            if let data = try await retrieve(key: key) {
                let compressedData = try compressData(data)
                let finalData = try encryptionEngine?.encrypt(compressedData) ?? compressedData
                
                let offset = try newHandle.seekToEnd()
                try newHandle.write(contentsOf: finalData)
                
                newIndex[key] = IndexEntry(
                    fileOffset: Int(offset),
                    compressedSize: finalData.count,
                    originalSize: entry.originalSize,
                    checksum: entry.checksum,
                    lastAccessed: entry.lastAccessed
                )
            }
        }
        
        // Replace old pack file
        let oldPackFile = getCurrentPackFile()
        try FileManager.default.removeItem(at: oldPackFile)
        
        // Update index
        index = newIndex
        try await saveIndex()
        
        logger.info("Cold tier compaction completed")
    }
    
    func statistics() async -> TierStatistics {
        let originalSize = index.values.reduce(0) { $0 + $1.originalSize }
        let compressedSize = size
        
        return TierStatistics(
            size: compressedSize,
            originalSize: originalSize,
            itemCount: index.count,
            averageLatency: 0.001, // ~1ms
            hitRate: 0.7 // Estimated
        )
    }
    
    func validateIntegrity() async -> StorageIntegrityReport {
        var issues: [StorageIssue] = []
        
        // Check pack file exists
        let packFile = getCurrentPackFile()
        if !FileManager.default.fileExists(atPath: packFile.path) {
            issues.append(StorageIssue(
                type: .corruption,
                description: "Pack file missing",
                impact: .critical
            ))
        }
        
        // Verify a sample of entries
        let sampleSize = min(10, index.count)
        let sampleKeys = Array(index.keys.shuffled().prefix(sampleSize))
        
        for key in sampleKeys {
            do {
                _ = try await retrieve(key: key)
            } catch {
                issues.append(StorageIssue(
                    type: .corruption,
                    description: "Failed to retrieve key \(key): \(error)",
                    impact: .major
                ))
            }
        }
        
        return StorageIntegrityReport(
            isHealthy: issues.isEmpty,
            issues: issues,
            recommendations: issues.isEmpty ? [] : ["Run compact() to fix issues"],
            lastCheck: Date()
        )
    }
    
    func createSnapshot(id: String) async throws -> Data {
        // Create index snapshot
        return try JSONEncoder().encode(index)
    }
    
    func restoreSnapshot(_ identifier: SnapshotIdentifier) async throws {
        // Not implemented for cold tier
    }
    
    func optimize() async throws {
        // Compact if fragmentation is high
        let fragmentation = estimateFragmentation()
        if fragmentation > 0.3 {
            try await compact()
        }
    }
    
    func batchStore(_ items: [(String, Data, StorageOptions)]) async throws {
        for (key, data, options) in items {
            try await store(key: key, data: data, options: options)
        }
    }
    
    // MARK: - Private Helpers
    
    private func getCurrentPackFile() -> URL {
        let coldDir = baseDirectory.appendingPathComponent("cold")
        let packFiles = try? FileManager.default.contentsOfDirectory(
            at: coldDir,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "dat" && $0.lastPathComponent.hasPrefix("pack-") }
        
        if let existingPack = packFiles?.first {
            return existingPack
        } else {
            // Create new pack file
            let newPack = coldDir.appendingPathComponent("pack-\(UUID().uuidString).dat")
            FileManager.default.createFile(atPath: newPack.path, contents: nil)
            return newPack
        }
    }
    
    private func loadIndex() async throws {
        let indexFile = baseDirectory
            .appendingPathComponent("cold")
            .appendingPathComponent("index.json")
        
        guard FileManager.default.fileExists(atPath: indexFile.path) else { return }
        
        let data = try Data(contentsOf: indexFile)
        index = try JSONDecoder().decode([String: IndexEntry].self, from: data)
    }
    
    private func saveIndex() async throws {
        let indexFile = baseDirectory
            .appendingPathComponent("cold")
            .appendingPathComponent("index.json")
        
        let data = try JSONEncoder().encode(index)
        try data.write(to: indexFile, options: .atomic)
    }
    
    private func compressData(_ data: Data) throws -> Data {
        switch compression {
        case .none:
            return data
        case .lz4:
            return try (data as NSData).compressed(using: .lz4) as Data
        case .zstd:
            return try (data as NSData).compressed(using: .zlib) as Data // Using zlib as zstd proxy
        case .brotli:
            return try (data as NSData).compressed(using: .lzma) as Data // Using lzma as brotli proxy
        case .adaptive:
            // Use zlib for adaptive compression
            return try (data as NSData).compressed(using: .zlib) as Data
        case .quantization, .learned:
            // These compression types are not supported in cold tier
            return data
        }
    }
    
    private func decompressData(_ data: Data) throws -> Data {
        switch compression {
        case .none:
            return data
        case .lz4:
            return try (data as NSData).decompressed(using: .lz4) as Data
        case .zstd:
            return try (data as NSData).decompressed(using: .zlib) as Data
        case .brotli:
            return try (data as NSData).decompressed(using: .lzma) as Data
        case .adaptive:
            // Use zlib for adaptive compression
            return try (data as NSData).decompressed(using: .zlib) as Data
        case .quantization, .learned:
            // These compression types are not supported in cold tier
            return data
        }
    }
    
    private func estimateFragmentation() -> Float {
        // Simple fragmentation estimate based on deleted entries
        // In real implementation, would track actual space usage
        return 0.1
    }
}

// MARK: - Archive Tier Storage

/// Maximum compression storage for long-term archival
actor ArchiveTierStorage {
    private let baseDirectory: URL
    private let encryptionEngine: EncryptionEngine?
    private var archives: [String: ArchiveInfo] = [:]
    private let logger = Logger(subsystem: "VectorStoreKit", category: "ArchiveTier")
    
    struct ArchiveInfo {
        let archiveFile: String
        let originalSize: Int
        let compressedSize: Int
        let createdAt: Date
    }
    
    init(baseDirectory: URL, encryptionEngine: EncryptionEngine?) async throws {
        self.baseDirectory = baseDirectory
        self.encryptionEngine = encryptionEngine
        
        // Create directory if needed
        try FileManager.default.createDirectory(
            at: baseDirectory.appendingPathComponent("archive"),
            withIntermediateDirectories: true
        )
        
        // Load archive metadata
        try await loadMetadata()
    }
    
    var size: Int {
        archives.values.reduce(0) { $0 + $1.compressedSize }
    }
    
    var allocatedSize: Int { size }
    
    func store(key: String, data: Data, options: StorageOptions) async throws {
        // Use maximum compression
        let compressedData = try (data as NSData).compressed(using: .lzma) as Data
        
        // Encrypt if needed
        let finalData = try encryptionEngine?.encrypt(compressedData) ?? compressedData
        
        // Write to individual archive file
        let archiveFile = "\(key).archive"
        let archiveURL = baseDirectory
            .appendingPathComponent("archive")
            .appendingPathComponent(archiveFile)
        
        try finalData.write(to: archiveURL, options: .atomic)
        
        // Update metadata
        archives[key] = ArchiveInfo(
            archiveFile: archiveFile,
            originalSize: data.count,
            compressedSize: finalData.count,
            createdAt: Date()
        )
        
        try await saveMetadata()
        
        logger.debug("Archived key: \(key), compression ratio: \(Float(data.count) / Float(finalData.count))")
    }
    
    func retrieve(key: String) async throws -> Data? {
        guard let info = archives[key] else { return nil }
        
        let archiveURL = baseDirectory
            .appendingPathComponent("archive")
            .appendingPathComponent(info.archiveFile)
        
        guard FileManager.default.fileExists(atPath: archiveURL.path) else {
            archives.removeValue(forKey: key)
            return nil
        }
        
        // Read compressed data
        let compressedData = try Data(contentsOf: archiveURL)
        
        // Decrypt if needed
        let decryptedData = try encryptionEngine?.decrypt(compressedData) ?? compressedData
        
        // Decompress
        return try (decryptedData as NSData).decompressed(using: .lzma) as Data
    }
    
    func delete(key: String) async throws {
        guard let info = archives[key] else { return }
        
        let archiveURL = baseDirectory
            .appendingPathComponent("archive")
            .appendingPathComponent(info.archiveFile)
        
        try FileManager.default.removeItem(at: archiveURL)
        archives.removeValue(forKey: key)
        
        try await saveMetadata()
    }
    
    func exists(key: String) async -> Bool {
        archives[key] != nil
    }
    
    func scan(prefix: String, callback: (String, Data) -> Void) async throws {
        for key in archives.keys where key.hasPrefix(prefix) {
            if let data = try await retrieve(key: key) {
                callback(key, data)
            }
        }
    }
    
    func compact() async throws {
        // Remove orphaned files
        let archiveDir = baseDirectory.appendingPathComponent("archive")
        let contents = try FileManager.default.contentsOfDirectory(
            at: archiveDir,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == "archive" }
        
        for fileURL in contents {
            let fileName = fileURL.lastPathComponent
            let key = String(fileName.dropLast(8)) // Remove .archive extension
            
            if archives[key] == nil {
                try FileManager.default.removeItem(at: fileURL)
                logger.debug("Removed orphaned archive: \(fileName)")
            }
        }
    }
    
    func statistics() async -> TierStatistics {
        let originalSize = archives.values.reduce(0) { $0 + $1.originalSize }
        
        return TierStatistics(
            size: size,
            originalSize: originalSize,
            itemCount: archives.count,
            averageLatency: 0.1, // ~100ms
            hitRate: 0.3 // Estimated - archive tier has low hit rate
        )
    }
    
    func validateIntegrity() async -> StorageIntegrityReport {
        var issues: [StorageIssue] = []
        
        // Check for missing archive files
        for (key, info) in archives {
            let archiveURL = baseDirectory
                .appendingPathComponent("archive")
                .appendingPathComponent(info.archiveFile)
            
            if !FileManager.default.fileExists(atPath: archiveURL.path) {
                issues.append(StorageIssue(
                    type: .corruption,
                    description: "Missing archive file for key: \(key)",
                    impact: .major
                ))
            }
        }
        
        return StorageIntegrityReport(
            isHealthy: issues.isEmpty,
            issues: issues,
            recommendations: issues.isEmpty ? [] : ["Run compact() to fix issues"],
            lastCheck: Date()
        )
    }
    
    func createSnapshot(id: String) async throws -> Data {
        // Create metadata snapshot
        return try JSONEncoder().encode(archives)
    }
    
    func restoreSnapshot(_ identifier: SnapshotIdentifier) async throws {
        // Not implemented for archive tier
    }
    
    func optimize() async throws {
        try await compact()
    }
    
    func batchStore(_ items: [(String, Data, StorageOptions)]) async throws {
        for (key, data, options) in items {
            try await store(key: key, data: data, options: options)
        }
    }
    
    // MARK: - Private Helpers
    
    private func loadMetadata() async throws {
        let metadataFile = baseDirectory
            .appendingPathComponent("archive")
            .appendingPathComponent("metadata.json")
        
        guard FileManager.default.fileExists(atPath: metadataFile.path) else { return }
        
        let data = try Data(contentsOf: metadataFile)
        archives = try JSONDecoder().decode([String: ArchiveInfo].self, from: data)
    }
    
    private func saveMetadata() async throws {
        let metadataFile = baseDirectory
            .appendingPathComponent("archive")
            .appendingPathComponent("metadata.json")
        
        let data = try JSONEncoder().encode(archives)
        try data.write(to: metadataFile, options: .atomic)
    }
}

// MARK: - Supporting Types

// Note: StorageIssue and StorageIntegrityReport are defined in Core/Protocols.swift
// We'll use the existing types from there

// Make existing types conform to Codable/Sendable as needed
extension HotTierStorage.CacheEntry: Codable {}

extension WarmTierStorage.FileMetadata: Codable {}

extension ColdTierStorage.IndexEntry: Codable {}

extension ArchiveTierStorage.ArchiveInfo: Codable {}

/// Basic encryption engine implementation
final class EncryptionEngine: Sendable {
    private let key: SymmetricKey
    
    init(settings: EncryptionSettings) throws {
        // Generate or derive key based on settings
        self.key = SymmetricKey(size: .bits256)
    }
    
    func encrypt(_ data: Data) throws -> Data {
        let sealedBox = try AES.GCM.seal(data, using: key)
        return sealedBox.combined ?? Data()
    }
    
    func decrypt(_ data: Data) throws -> Data {
        let sealedBox = try AES.GCM.SealedBox(combined: data)
        return try AES.GCM.open(sealedBox, using: key)
    }
}

// Error extension
extension StorageError {
    static func corruptedData(_ message: String) -> StorageError {
        .retrieveFailed("Corrupted data: \(message)")
    }
}
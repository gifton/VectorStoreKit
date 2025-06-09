// VectorStoreKit: Cold Tier Storage Implementation
//
// Compressed disk storage for infrequently accessed data

import Foundation
import Compression
import CryptoKit
import os.log

// MARK: - Cold Tier Storage

/// Compressed disk storage for infrequently accessed data
actor ColdTierStorage {
    private let baseDirectory: URL
    private let compression: CompressionAlgorithm
    private let encryptionEngine: EncryptionEngine?
    private var index: [String: IndexEntry] = [:]
    private let logger = Logger(subsystem: "VectorStoreKit", category: "ColdTier")
    
    struct IndexEntry: Codable {
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
    
    var count: Int { index.count }
    
    func getAllKeys() async -> [String] {
        Array(index.keys)
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
        guard !index.isEmpty else { return 0.0 }
        
        // Get pack file size
        let packFile = getCurrentPackFile()
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: packFile.path),
              let fileSize = attributes[.size] as? Int else {
            return 0.0
        }
        
        // Calculate total indexed data size (compressed)
        let totalIndexedSize = index.values.reduce(0) { $0 + $1.compressedSize }
        
        // Calculate wasted space from file system overhead
        let wastedSpace = max(0, fileSize - totalIndexedSize)
        
        // Calculate index overhead (metadata stored separately)
        let indexOverhead = index.count * 128 // Estimated per-entry overhead
        
        // Calculate fragmentation factors:
        
        // 1. File space efficiency (how much of the file is actual data vs padding/deleted space)
        let spaceEfficiency = fileSize > 0 ? Float(totalIndexedSize) / Float(fileSize) : 1.0
        
        // 2. Index overhead ratio (metadata size vs data size)
        let indexOverheadRatio = totalIndexedSize > 0 ? Float(indexOverhead) / Float(totalIndexedSize) : 0.0
        
        // 3. Entry size variance (fragmentation due to variable compression ratios)
        let compressionRatios = index.values.map { entry in
            entry.originalSize > 0 ? Float(entry.compressedSize) / Float(entry.originalSize) : 1.0
        }
        let avgCompressionRatio = compressionRatios.reduce(0, +) / Float(compressionRatios.count)
        let compressionVariance = compressionRatios.reduce(0) { sum, ratio in
            let diff = ratio - avgCompressionRatio
            return sum + (diff * diff)
        } / Float(compressionRatios.count)
        let compressionFragmentation = min(sqrt(compressionVariance), 1.0)
        
        // 4. File system block alignment inefficiencies
        let blockSize: Int = 4096 // Typical file system block size
        let alignmentWaste = wastedSpace % blockSize
        let alignmentFragmentation = fileSize > 0 ? Float(alignmentWaste) / Float(fileSize) : 0.0
        
        // Combine fragmentation factors with weights
        let weightedFragmentation = 
            (1.0 - spaceEfficiency) * 0.4 +           // Space waste (highest weight)
            min(indexOverheadRatio, 1.0) * 0.3 +      // Index overhead
            compressionFragmentation * 0.2 +          // Compression variance
            alignmentFragmentation * 0.1              // Block alignment
        
        return max(0.0, min(1.0, weightedFragmentation))
    }
}

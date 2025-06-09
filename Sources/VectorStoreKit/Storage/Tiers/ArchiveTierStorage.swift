// VectorStoreKit: Archive Tier Storage Implementation
//
// Maximum compression storage for long-term archival

import Foundation
import Compression
import CryptoKit
import os.log

// MARK: - Archive Tier Storage

/// Maximum compression storage for long-term archival
actor ArchiveTierStorage {
    private let baseDirectory: URL
    private let encryptionEngine: EncryptionEngine?
    private var archives: [String: ArchiveInfo] = [:]
    private let logger = Logger(subsystem: "VectorStoreKit", category: "ArchiveTier")
    
    struct ArchiveInfo: Codable {
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
    
    var count: Int { archives.count }
    
    func getAllKeys() async -> [String] {
        Array(archives.keys)
    }
    
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
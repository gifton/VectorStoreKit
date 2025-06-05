// VectorStoreKit: Hierarchical Storage Configuration
//
// Configuration for multi-tier storage system

import Foundation

/// Configuration for hierarchical storage system
public struct HierarchicalStorageConfiguration: StorageConfiguration, Codable {
    /// Memory budget for hot tier (bytes)
    public let hotTierMemoryLimit: Int
    
    /// File size limit for warm tier memory mapping (bytes)
    public let warmTierFileSizeLimit: Int
    
    /// Compression algorithm for cold tier
    public let coldTierCompression: CompressionAlgorithm
    
    /// Encryption settings for data at rest
    public let encryptionSettings: EncryptionSettings
    
    /// Automatic tier migration settings
    public let migrationSettings: MigrationSettings
    
    /// Write-ahead log configuration
    public let walConfiguration: WALConfiguration
    
    /// Performance monitoring settings
    public let monitoringSettings: MonitoringSettings
    
    /// Base directory for storage files
    public let baseDirectory: URL
    
    public init(
        hotTierMemoryLimit: Int = 256 * 1024 * 1024, // 256 MB
        warmTierFileSizeLimit: Int = 1024 * 1024 * 1024, // 1 GB
        coldTierCompression: CompressionAlgorithm = .zstd,
        encryptionSettings: EncryptionSettings = .disabled,
        migrationSettings: MigrationSettings = .automatic,
        walConfiguration: WALConfiguration = .default,
        monitoringSettings: MonitoringSettings = .enabled,
        baseDirectory: URL
    ) {
        self.hotTierMemoryLimit = hotTierMemoryLimit
        self.warmTierFileSizeLimit = warmTierFileSizeLimit
        self.coldTierCompression = coldTierCompression
        self.encryptionSettings = encryptionSettings
        self.migrationSettings = migrationSettings
        self.walConfiguration = walConfiguration
        self.monitoringSettings = monitoringSettings
        self.baseDirectory = baseDirectory
    }
    
    public func validate() throws {
        guard hotTierMemoryLimit > 0 else {
            throw StorageError.invalidConfiguration("Hot tier memory limit must be positive")
        }
        guard warmTierFileSizeLimit > 0 else {
            throw StorageError.invalidConfiguration("Warm tier file size limit must be positive")
        }
        guard FileManager.default.fileExists(atPath: baseDirectory.path) else {
            throw StorageError.invalidConfiguration("Base directory does not exist: \(baseDirectory.path)")
        }
    }
    
    public func storageOverhead() -> Float {
        // Estimated overhead: WAL (10%) + Metadata (5%) + Compression headers (2%)
        return 0.17
    }
    
    public func compressionCapabilities() -> CompressionCapabilities {
        return CompressionCapabilities(
            algorithms: [.none, .lz4, .zstd, .brotli],
            maxRatio: coldTierCompression == .zstd ? 10.0 : 5.0,
            lossless: true
        )
    }
    
    /// Research-optimized configuration for maximum performance analysis
    public static func research(baseDirectory: URL) -> HierarchicalStorageConfiguration {
        return HierarchicalStorageConfiguration(
            hotTierMemoryLimit: 1024 * 1024 * 1024, // 1 GB
            warmTierFileSizeLimit: 4 * 1024 * 1024 * 1024, // 4 GB
            coldTierCompression: .zstd,
            encryptionSettings: .aes256,
            migrationSettings: .intelligent,
            walConfiguration: .highPerformance,
            monitoringSettings: .comprehensive,
            baseDirectory: baseDirectory
        )
    }
}
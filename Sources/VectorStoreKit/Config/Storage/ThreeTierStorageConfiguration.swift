import Foundation

/// Codable wrapper for AccessPatternTracker.Configuration
public struct AccessPatternTrackerConfiguration: Codable, Sendable {
    public let promotionThreshold: Double
    public let demotionThreshold: Double
    public let migrationCooldown: TimeInterval
    public let maxTrackedPatterns: Int
    public let rateWindowHours: Int
    
    public init(
        promotionThreshold: Double = 0.7,
        demotionThreshold: Double = 0.3,
        migrationCooldown: TimeInterval = 3600,
        maxTrackedPatterns: Int = 100_000,
        rateWindowHours: Int = 168
    ) {
        self.promotionThreshold = promotionThreshold
        self.demotionThreshold = demotionThreshold
        self.migrationCooldown = migrationCooldown
        self.maxTrackedPatterns = maxTrackedPatterns
        self.rateWindowHours = rateWindowHours
    }
    
    var asTrackerConfiguration: AccessPatternTracker.Configuration {
        AccessPatternTracker.Configuration(
            promotionThreshold: promotionThreshold,
            demotionThreshold: demotionThreshold,
            migrationCooldown: migrationCooldown,
            maxTrackedPatterns: maxTrackedPatterns,
            rateWindowHours: rateWindowHours
        )
    }
}

/// Codable wrapper for TierManager.Configuration
public struct TierManagerConfiguration: Codable, Sendable {
    public let autoMigrationEnabled: Bool
    public let migrationInterval: TimeInterval
    public let batchSize: Int
    public let maxConcurrentMigrations: Int
    public let memoryPressureThreshold: Double
    
    public init(
        autoMigrationEnabled: Bool = true,
        migrationInterval: TimeInterval = 300,
        batchSize: Int = 100,
        maxConcurrentMigrations: Int = 4,
        memoryPressureThreshold: Double = 0.8
    ) {
        self.autoMigrationEnabled = autoMigrationEnabled
        self.migrationInterval = migrationInterval
        self.batchSize = batchSize
        self.maxConcurrentMigrations = maxConcurrentMigrations
        self.memoryPressureThreshold = memoryPressureThreshold
    }
    
    var asTierManagerConfiguration: TierManager.Configuration {
        TierManager.Configuration(
            autoMigrationEnabled: autoMigrationEnabled,
            migrationInterval: migrationInterval,
            batchSize: batchSize,
            maxConcurrentMigrations: maxConcurrentMigrations,
            memoryPressureThreshold: memoryPressureThreshold
        )
    }
}

/// Configuration for the 3-tier storage system
public struct ThreeTierStorageConfiguration: StorageConfiguration, Codable, Sendable {
    
    // MARK: - Properties
    
    /// Maximum memory usage for the memory tier in bytes (default: 500MB)
    public let memoryLimit: Int
    
    /// Maximum size for individual items in memory tier (default: 10MB)
    public let memoryItemSizeLimit: Int
    
    /// Initial capacity to reserve for memory tier
    public let initialMemoryCapacity: Int
    
    /// Path for SSD tier storage
    public let ssdPath: String
    
    /// Path for archive tier storage
    public let archivePath: String
    
    /// Path for metadata storage
    public let metadataPath: String
    
    /// Enable automatic tier migration
    public let autoMigrationEnabled: Bool
    
    /// Configuration for access pattern tracking
    public let accessPatternConfiguration: AccessPatternTrackerConfiguration
    
    /// Configuration for tier management
    public let tierManagerConfiguration: TierManagerConfiguration
    
    // MARK: - Initialization
    
    /// Create a new 3-tier storage configuration
    /// - Parameters:
    ///   - memoryLimit: Maximum memory usage for memory tier (default: 500MB)
    ///   - memoryItemSizeLimit: Maximum size for items in memory (default: 10MB)
    ///   - initialMemoryCapacity: Initial capacity for memory tier (default: 1000)
    ///   - basePath: Base directory for file-based tiers (default: system temp)
    ///   - autoMigrationEnabled: Enable automatic tier migration (default: true)
    public init(
        memoryLimit: Int = 524_288_000, // 500MB
        memoryItemSizeLimit: Int = 10_485_760, // 10MB
        initialMemoryCapacity: Int = 1000,
        basePath: String? = nil,
        autoMigrationEnabled: Bool = true,
        accessPatternConfiguration: AccessPatternTrackerConfiguration = .init(),
        tierManagerConfiguration: TierManagerConfiguration = .init()
    ) {
        self.memoryLimit = memoryLimit
        self.memoryItemSizeLimit = memoryItemSizeLimit
        self.initialMemoryCapacity = initialMemoryCapacity
        self.autoMigrationEnabled = autoMigrationEnabled
        self.accessPatternConfiguration = accessPatternConfiguration
        self.tierManagerConfiguration = tierManagerConfiguration
        
        // Set up paths
        let base = basePath ?? NSTemporaryDirectory()
        let storageBase = (base as NSString).appendingPathComponent("VectorStoreKit")
        
        self.ssdPath = (storageBase as NSString).appendingPathComponent("ssd")
        self.archivePath = (storageBase as NSString).appendingPathComponent("archive")
        self.metadataPath = (storageBase as NSString).appendingPathComponent("metadata.json")
    }
    
    // MARK: - Presets
    
    /// Default configuration for most use cases
    public static let `default` = ThreeTierStorageConfiguration()
    
    /// Small configuration for testing or constrained environments
    public static let small = ThreeTierStorageConfiguration(
        memoryLimit: 104_857_600, // 100MB
        memoryItemSizeLimit: 5_242_880, // 5MB
        initialMemoryCapacity: 100,
        autoMigrationEnabled: true,
        accessPatternConfiguration: AccessPatternTrackerConfiguration(
            promotionThreshold: 0.8,
            demotionThreshold: 0.2,
            maxTrackedPatterns: 10_000
        ),
        tierManagerConfiguration: TierManagerConfiguration(
            migrationInterval: 60, // 1 minute
            batchSize: 50
        )
    )
    
    /// Large configuration for high-capacity applications
    public static let large = ThreeTierStorageConfiguration(
        memoryLimit: 2_147_483_648, // 2GB
        memoryItemSizeLimit: 52_428_800, // 50MB
        initialMemoryCapacity: 10_000,
        autoMigrationEnabled: true,
        accessPatternConfiguration: AccessPatternTrackerConfiguration(
            promotionThreshold: 0.6,
            demotionThreshold: 0.4,
            maxTrackedPatterns: 1_000_000
        ),
        tierManagerConfiguration: TierManagerConfiguration(
            migrationInterval: 600, // 10 minutes
            batchSize: 500,
            maxConcurrentMigrations: 8
        )
    )
    
    /// Performance-optimized configuration (aggressive caching)
    public static let performance = ThreeTierStorageConfiguration(
        memoryLimit: 1_073_741_824, // 1GB
        memoryItemSizeLimit: 104_857_600, // 100MB
        initialMemoryCapacity: 10_000,
        autoMigrationEnabled: true,
        accessPatternConfiguration: AccessPatternTrackerConfiguration(
            promotionThreshold: 0.5, // Promote more aggressively
            demotionThreshold: 0.1,  // Keep data hot longer
            migrationCooldown: 1800   // 30 minutes
        ),
        tierManagerConfiguration: TierManagerConfiguration(
            migrationInterval: 120, // 2 minutes
            batchSize: 200,
            maxConcurrentMigrations: 8
        )
    )
    
    /// Development configuration with aggressive migration for testing
    public static let development = ThreeTierStorageConfiguration(
        memoryLimit: 52_428_800, // 50MB
        memoryItemSizeLimit: 1_048_576, // 1MB
        initialMemoryCapacity: 100,
        autoMigrationEnabled: true,
        accessPatternConfiguration: AccessPatternTrackerConfiguration(
            promotionThreshold: 0.6,
            demotionThreshold: 0.4,
            migrationCooldown: 10, // 10 seconds
            maxTrackedPatterns: 1000
        ),
        tierManagerConfiguration: TierManagerConfiguration(
            migrationInterval: 30, // 30 seconds
            batchSize: 20,
            maxConcurrentMigrations: 2
        )
    )
    
    // MARK: - StorageConfiguration Protocol
    
    /// Validate configuration parameters
    public func validate() throws {
        if memoryLimit <= 0 {
            throw VectorStoreError.configurationInvalid("Memory limit must be positive: \(memoryLimit)")
        }
        
        if memoryItemSizeLimit <= 0 {
            throw VectorStoreError.configurationInvalid("Memory item size limit must be positive: \(memoryItemSizeLimit)")
        }
        
        if memoryItemSizeLimit > memoryLimit {
            throw VectorStoreError.configurationInvalid("Item size limit cannot exceed memory limit")
        }
        
        if initialMemoryCapacity < 0 {
            throw VectorStoreError.configurationInvalid("Initial capacity cannot be negative: \(initialMemoryCapacity)")
        }
        
        if memoryLimit < 10_485_760 { // 10MB minimum
            throw VectorStoreError.configurationInvalid("Memory limit too small: \(memoryLimit) bytes (minimum: 10MB)")
        }
        
        // Validate access pattern configuration
        if accessPatternConfiguration.promotionThreshold <= accessPatternConfiguration.demotionThreshold {
            throw VectorStoreError.configurationInvalid("Promotion threshold must be greater than demotion threshold")
        }
    }
    
    /// Get storage overhead estimate
    public func storageOverhead() -> Float {
        // Estimate overhead for each tier
        let memoryOverheadPerEntry: Float = 64 // Dictionary + metadata
        let fileOverheadPerEntry: Float = 256  // File system + index
        
        let estimatedMemoryEntries = Float(initialMemoryCapacity)
        let estimatedFileEntries = estimatedMemoryEntries * 10 // Assume 10x more in files
        
        let memoryOverhead = memoryOverheadPerEntry * estimatedMemoryEntries
        let fileOverhead = fileOverheadPerEntry * estimatedFileEntries
        
        let totalOverhead = memoryOverhead + fileOverhead
        let totalCapacity = Float(memoryLimit) + Float(Int.max) // Unbounded file storage
        
        return min(totalOverhead / totalCapacity, 0.1) // Cap at 10%
    }
    
    /// Get compression capabilities
    public func compressionCapabilities() -> CompressionCapabilities {
        return CompressionCapabilities(
            algorithms: [.none, .lz4, .zstd],
            maxRatio: 10.0, // Typical max compression ratio
            lossless: true
        )
    }
    
    // MARK: - Helpers
    
    /// Create a configuration with custom memory limit
    public func withMemoryLimit(_ limit: Int) -> ThreeTierStorageConfiguration {
        ThreeTierStorageConfiguration(
            memoryLimit: limit,
            memoryItemSizeLimit: memoryItemSizeLimit,
            initialMemoryCapacity: initialMemoryCapacity,
            basePath: (ssdPath as NSString).deletingLastPathComponent,
            autoMigrationEnabled: autoMigrationEnabled,
            accessPatternConfiguration: accessPatternConfiguration,
            tierManagerConfiguration: tierManagerConfiguration
        )
    }
    
    /// Create a configuration with custom paths
    public func withBasePath(_ path: String) -> ThreeTierStorageConfiguration {
        ThreeTierStorageConfiguration(
            memoryLimit: memoryLimit,
            memoryItemSizeLimit: memoryItemSizeLimit,
            initialMemoryCapacity: initialMemoryCapacity,
            basePath: path,
            autoMigrationEnabled: autoMigrationEnabled,
            accessPatternConfiguration: accessPatternConfiguration,
            tierManagerConfiguration: tierManagerConfiguration
        )
    }
    
    /// Create a configuration with custom migration settings
    public func withMigrationSettings(
        enabled: Bool,
        interval: TimeInterval? = nil,
        batchSize: Int? = nil
    ) -> ThreeTierStorageConfiguration {
        var tierConfig = tierManagerConfiguration
        if let interval = interval {
            tierConfig = TierManagerConfiguration(
                autoMigrationEnabled: enabled,
                migrationInterval: interval,
                batchSize: batchSize ?? tierConfig.batchSize,
                maxConcurrentMigrations: tierConfig.maxConcurrentMigrations,
                memoryPressureThreshold: tierConfig.memoryPressureThreshold
            )
        }
        
        return ThreeTierStorageConfiguration(
            memoryLimit: memoryLimit,
            memoryItemSizeLimit: memoryItemSizeLimit,
            initialMemoryCapacity: initialMemoryCapacity,
            basePath: (ssdPath as NSString).deletingLastPathComponent,
            autoMigrationEnabled: enabled,
            accessPatternConfiguration: accessPatternConfiguration,
            tierManagerConfiguration: tierConfig
        )
    }
}

// MARK: - Custom Types

extension ThreeTierStorageConfiguration {
    /// Description of the storage configuration
    public var description: String {
        """
        3-Tier Storage Configuration:
        - Memory Tier: \(memoryLimit / 1024 / 1024)MB (max item: \(memoryItemSizeLimit / 1024 / 1024)MB)
        - SSD Tier: LZ4 compression at \(ssdPath)
        - Archive Tier: ZSTD compression at \(archivePath)
        - Auto Migration: \(autoMigrationEnabled ? "Enabled" : "Disabled")
        - Migration Interval: \(tierManagerConfiguration.migrationInterval)s
        - Promotion Threshold: \(accessPatternConfiguration.promotionThreshold)
        - Demotion Threshold: \(accessPatternConfiguration.demotionThreshold)
        """
    }
}
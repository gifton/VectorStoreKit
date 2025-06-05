// VectorStoreKit: Storage Migration Engine
//
// Manages automatic data migration between storage tiers based on access patterns

import Foundation
import os.log

/// Manages automatic data migration between storage tiers
actor MigrationEngine {
    // MARK: - Properties
    
    private let settings: MigrationSettings
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MigrationEngine")
    
    // Migration statistics
    private var totalMigrations: UInt64 = 0
    private var successfulMigrations: UInt64 = 0
    private var migrationTimes: [TimeInterval] = []
    private var tierDistribution: [StorageTier: Int] = [:]
    
    // Migration policies
    private let hotToWarmThreshold: TimeInterval = 3600 // 1 hour
    private let warmToColdThreshold: TimeInterval = 86400 // 1 day
    private let coldToArchiveThreshold: TimeInterval = 604800 // 1 week
    
    // Batch configuration
    private let maxBatchSize = 100
    private let maxConcurrentMigrations = 4
    
    // MARK: - Initialization
    
    init(settings: MigrationSettings) {
        self.settings = settings
    }
    
    // MARK: - Public Methods
    
    /// Run a migration cycle
    func runMigrationCycle(
        accessAnalyzer: StorageAccessAnalyzer,
        tiers: (hot: HotTierStorage, warm: WarmTierStorage, cold: ColdTierStorage, archive: ArchiveTierStorage)
    ) async throws {
        let startTime = Date()
        
        logger.info("Starting migration cycle")
        
        // Check if migration is enabled
        if case .disabled = settings {
            logger.debug("Migration disabled, skipping cycle")
            return
        }
        
        // Get current tier distributions
        await updateTierDistribution(tiers: tiers)
        
        // Perform migrations based on access patterns
        // Run migrations sequentially to avoid concurrency issues
        do {
            // Hot to Warm migrations
            try await migrateHotToWarm(
                accessAnalyzer: accessAnalyzer,
                hotTier: tiers.hot,
                warmTier: tiers.warm
            )
            
            // Warm to Cold migrations
            try await migrateWarmToCold(
                accessAnalyzer: accessAnalyzer,
                warmTier: tiers.warm,
                coldTier: tiers.cold
            )
            
            // Cold to Archive migrations
            try await migrateColdToArchive(
                accessAnalyzer: accessAnalyzer,
                coldTier: tiers.cold,
                archiveTier: tiers.archive
            )
            
            // Promotions (Archive/Cold/Warm to higher tiers)
            try await performPromotions(
                accessAnalyzer: accessAnalyzer,
                tiers: tiers
            )
        } catch {
            logger.error("Migration cycle failed: \(error)")
            throw error
        }
        
        let duration = Date().timeIntervalSince(startTime)
        migrationTimes.append(duration)
        
        // Keep only recent migration times
        if migrationTimes.count > 100 {
            migrationTimes = Array(migrationTimes.suffix(50))
        }
        
        logger.info("Migration cycle completed in \(duration) seconds")
    }
    
    /// Get migration metrics
    func getMetrics() async -> MigrationMetrics {
        let successRate = totalMigrations > 0
            ? Float(successfulMigrations) / Float(totalMigrations)
            : 1.0
        
        let averageMigrationTime = migrationTimes.isEmpty
            ? 0
            : migrationTimes.reduce(0, +) / Double(migrationTimes.count)
        
        return MigrationMetrics(
            totalMigrations: totalMigrations,
            successRate: successRate,
            averageMigrationTime: averageMigrationTime,
            tierDistribution: tierDistribution
        )
    }
    
    // MARK: - Private Migration Methods
    
    private func migrateHotToWarm(
        accessAnalyzer: StorageAccessAnalyzer,
        hotTier: HotTierStorage,
        warmTier: WarmTierStorage
    ) async throws {
        // Get all keys in hot tier
        let hotKeys = await hotTier.getAllKeys()
        var migrationCount = 0
        
        // Check each key's access pattern
        for key in hotKeys {
            if migrationCount >= maxBatchSize {
                break
            }
            
            let pattern = await accessAnalyzer.getPattern(for: key)
            
            // If not frequently accessed or not recent, consider for migration
            if pattern.frequency != .high || !pattern.recentActivity {
                if let data = await hotTier.retrieve(key: key) {
                    do {
                        // Store in warm tier first
                        try await warmTier.store(key: key, data: data, options: .default)
                        
                        // Then delete from hot tier
                        await hotTier.delete(key: key)
                        
                        await recordMigration(success: true)
                        migrationCount += 1
                        
                        logger.debug("Migrated key \(key) from hot to warm tier")
                    } catch {
                        logger.error("Failed to migrate key \(key): \(error)")
                        await recordMigration(success: false)
                    }
                }
            }
        }
        
        if migrationCount > 0 {
            logger.info("Migrated \(migrationCount) items from hot to warm tier")
        }
    }
    
    private func migrateWarmToCold(
        accessAnalyzer: StorageAccessAnalyzer,
        warmTier: WarmTierStorage,
        coldTier: ColdTierStorage
    ) async throws {
        let warmKeys = await warmTier.getAllKeys()
        var migrationCount = 0
        
        for key in warmKeys {
            if migrationCount >= maxBatchSize {
                break
            }
            
            let pattern = await accessAnalyzer.getPattern(for: key)
            
            // Migrate if low frequency or rare
            if pattern.frequency == .low || pattern.frequency == .rare {
                if let data = try await warmTier.retrieve(key: key) {
                    do {
                        // Store in cold tier first
                        try await coldTier.store(key: key, data: data, options: .default)
                        
                        // Then delete from warm tier
                        try await warmTier.delete(key: key)
                        
                        await recordMigration(success: true)
                        migrationCount += 1
                        
                        logger.debug("Migrated key \(key) from warm to cold tier")
                    } catch {
                        logger.error("Failed to migrate key \(key): \(error)")
                        await recordMigration(success: false)
                    }
                }
            }
        }
        
        if migrationCount > 0 {
            logger.info("Migrated \(migrationCount) items from warm to cold tier")
        }
    }
    
    private func migrateColdToArchive(
        accessAnalyzer: StorageAccessAnalyzer,
        coldTier: ColdTierStorage,
        archiveTier: ArchiveTierStorage
    ) async throws {
        let coldKeys = await coldTier.getAllKeys()
        var migrationCount = 0
        
        for key in coldKeys {
            if migrationCount >= maxBatchSize {
                break
            }
            
            let pattern = await accessAnalyzer.getPattern(for: key)
            
            // Migrate if rare and not recently accessed
            if pattern.frequency == .rare && !pattern.recentActivity {
                if let data = try await coldTier.retrieve(key: key) {
                    do {
                        // Store in archive tier first
                        try await archiveTier.store(key: key, data: data, options: .default)
                        
                        // Then delete from cold tier
                        try await coldTier.delete(key: key)
                        
                        await recordMigration(success: true)
                        migrationCount += 1
                        
                        logger.debug("Migrated key \(key) from cold to archive tier")
                    } catch {
                        logger.error("Failed to migrate key \(key): \(error)")
                        await recordMigration(success: false)
                    }
                }
            }
        }
        
        if migrationCount > 0 {
            logger.info("Migrated \(migrationCount) items from cold to archive tier")
        }
    }
    
    private func performPromotions(
        accessAnalyzer: StorageAccessAnalyzer,
        tiers: (hot: HotTierStorage, warm: WarmTierStorage, cold: ColdTierStorage, archive: ArchiveTierStorage)
    ) async throws {
        // Get hot data candidates from analyzer
        let hotCandidates = await accessAnalyzer.getHotDataCandidates(limit: maxBatchSize)
        var promotionCount = 0
        
        // Check each tier for promotion candidates
        for key in hotCandidates {
            // Try to find the key in lower tiers and promote
            if await tiers.warm.exists(key: key) {
                if let data = try await tiers.warm.retrieve(key: key) {
                    do {
                        await tiers.hot.store(key: key, data: data)
                        try await tiers.warm.delete(key: key)
                        await recordMigration(success: true)
                        promotionCount += 1
                        logger.debug("Promoted key \(key) from warm to hot tier")
                    } catch {
                        logger.error("Failed to promote key \(key): \(error)")
                        await recordMigration(success: false)
                    }
                }
            } else if await tiers.cold.exists(key: key) {
                if let data = try await tiers.cold.retrieve(key: key) {
                    do {
                        await tiers.hot.store(key: key, data: data)
                        try await tiers.cold.delete(key: key)
                        await recordMigration(success: true)
                        promotionCount += 1
                        logger.debug("Promoted key \(key) from cold to hot tier")
                    } catch {
                        logger.error("Failed to promote key \(key): \(error)")
                        await recordMigration(success: false)
                    }
                }
            } else if await tiers.archive.exists(key: key) {
                if let data = try await tiers.archive.retrieve(key: key) {
                    do {
                        // Promote to warm instead of hot from archive
                        try await tiers.warm.store(key: key, data: data, options: .default)
                        try await tiers.archive.delete(key: key)
                        await recordMigration(success: true)
                        promotionCount += 1
                        logger.debug("Promoted key \(key) from archive to warm tier")
                    } catch {
                        logger.error("Failed to promote key \(key): \(error)")
                        await recordMigration(success: false)
                    }
                }
            }
        }
        
        if promotionCount > 0 {
            logger.info("Promoted \(promotionCount) items to higher tiers")
        }
    }
    
    private func updateTierDistribution(
        tiers: (hot: HotTierStorage, warm: WarmTierStorage, cold: ColdTierStorage, archive: ArchiveTierStorage)
    ) async {
        tierDistribution = [
            .hot: await tiers.hot.count,
            .warm: await tiers.warm.count,
            .cold: await tiers.cold.count,
            .frozen: await tiers.archive.count
        ]
    }
    
    private func recordMigration(success: Bool) async {
        totalMigrations += 1
        if success {
            successfulMigrations += 1
        }
    }
}

// MARK: - Migration Error

enum MigrationError: Error {
    case tierNotAvailable
    case migrationFailed(String)
}
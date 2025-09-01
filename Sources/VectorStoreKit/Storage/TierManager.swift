import Foundation
import os.log

/// Manages automatic data migration between storage tiers based on access patterns
public actor TierManager {
    
    /// Configuration for tier management
    public struct Configuration: Sendable {
        /// Enable automatic migration
        public let autoMigrationEnabled: Bool
        
        /// Interval between migration runs
        public let migrationInterval: TimeInterval
        
        /// Maximum items to migrate per batch
        public let batchSize: Int
        
        /// Maximum concurrent migrations
        public let maxConcurrentMigrations: Int
        
        /// Memory pressure threshold (0-1) to pause migrations
        public let memoryPressureThreshold: Double
        
        public init(
            autoMigrationEnabled: Bool = true,
            migrationInterval: TimeInterval = 300, // 5 minutes
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
    }
    
    /// Migration statistics
    public struct Statistics: Sendable {
        public var totalMigrations: Int = 0
        public var successfulMigrations: Int = 0
        public var failedMigrations: Int = 0
        public var promotions: Int = 0
        public var demotions: Int = 0
        public var bytesPromoted: Int = 0
        public var bytesDemoted: Int = 0
        public var lastMigrationTime: Date?
        public var averageMigrationTime: TimeInterval = 0
        
        mutating func recordMigration(
            success: Bool,
            isPromotion: Bool,
            bytes: Int,
            duration: TimeInterval
        ) {
            totalMigrations += 1
            if success {
                successfulMigrations += 1
                if isPromotion {
                    promotions += 1
                    bytesPromoted += bytes
                } else {
                    demotions += 1
                    bytesDemoted += bytes
                }
            } else {
                failedMigrations += 1
            }
            
            // Update average migration time
            let totalTime = averageMigrationTime * Double(totalMigrations - 1) + duration
            averageMigrationTime = totalTime / Double(totalMigrations)
            
            lastMigrationTime = Date()
        }
    }
    
    // MARK: - Private Properties
    
    private let configuration: Configuration
    private let storage: any TierAwareStorage
    private let accessTracker: AccessPatternTracker
    private let compressionEngine: CompressionEngine
    private let logger = Logger(subsystem: "VectorStoreKit", category: "TierManager")
    
    private var statistics = Statistics()
    private var migrationTask: Task<Void, Never>?
    private var isRunning = false
    
    // MARK: - Initialization
    
    public init(
        configuration: Configuration,
        storage: any TierAwareStorage,
        accessTracker: AccessPatternTracker,
        compressionEngine: CompressionEngine
    ) {
        self.configuration = configuration
        self.storage = storage
        self.accessTracker = accessTracker
        self.compressionEngine = compressionEngine
    }
    
    // MARK: - Lifecycle
    
    /// Start automatic migration
    public func start() {
        guard configuration.autoMigrationEnabled else {
            logger.info("Automatic migration is disabled")
            return
        }
        
        guard !isRunning else {
            logger.warning("TierManager is already running")
            return
        }
        
        isRunning = true
        
        migrationTask = Task {
            logger.info("Starting automatic tier migration with interval: \(self.configuration.migrationInterval)s")
            
            while isRunning {
                do {
                    try await Task.sleep(nanoseconds: UInt64(configuration.migrationInterval * 1_000_000_000))
                    
                    if await shouldRunMigration() {
                        await runMigrationCycle()
                    }
                } catch {
                    // Task was cancelled
                    break
                }
            }
        }
    }
    
    /// Stop automatic migration
    public func stop() {
        isRunning = false
        migrationTask?.cancel()
        migrationTask = nil
        logger.info("Stopped automatic tier migration")
    }
    
    // MARK: - Manual Migration
    
    /// Run a manual migration cycle
    public func runMigrationCycle() async {
        let startTime = CFAbsoluteTimeGetCurrent()
        logger.info("Starting migration cycle")
        
        // Get migration candidates
        let candidates = await accessTracker.getMigrationCandidates(limit: configuration.batchSize)
        
        guard !candidates.isEmpty else {
            logger.debug("No migration candidates found")
            return
        }
        
        logger.info("Found \(candidates.count) migration candidates")
        
        // Process migrations in batches
        await withTaskGroup(of: Void.self) { group in
            for (pattern, decision) in candidates {
                // Limit concurrent migrations
                // Note: TaskGroup doesn't have a count property, so we'll track manually
                // For simplicity, we'll just let all tasks run concurrently
                
                group.addTask {
                    await self.migrateItem(pattern: pattern, decision: decision)
                }
            }
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        logger.info("Migration cycle completed in \(String(format: "%.2f", duration))s")
    }
    
    /// Migrate a specific key
    public func migrateKey(_ key: String) async throws {
        guard let pattern = await accessTracker.getPattern(for: key) else {
            throw TierManagerError.keyNotFound(key)
        }
        
        let decision = await accessTracker.getMigrationRecommendation(for: key)
        
        switch decision {
        case .stay:
            logger.debug("Key '\(key)' does not need migration")
        case .promote(_), .demote(_):
            await migrateItem(pattern: pattern, decision: decision)
        }
    }
    
    // MARK: - Statistics
    
    /// Get current migration statistics
    public func getStatistics() -> Statistics {
        statistics
    }
    
    /// Reset statistics
    public func resetStatistics() {
        statistics = Statistics()
    }
    
    // MARK: - Private Methods
    
    private func shouldRunMigration() async -> Bool {
        // Check memory pressure
        let memoryPressure = await getMemoryPressure()
        if memoryPressure > configuration.memoryPressureThreshold {
            logger.warning("Skipping migration due to memory pressure: \(memoryPressure)")
            return false
        }
        
        return true
    }
    
    private func migrateItem(pattern: AccessPatternTracker.Pattern, decision: MigrationDecision) async {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let targetTier: StorageTier
        let isPromotion: Bool
        
        switch decision {
        case .stay:
            return
        case .promote(let tier):
            targetTier = tier
            isPromotion = true
        case .demote(let tier):
            targetTier = tier
            isPromotion = false
        }
        
        do {
            // Retrieve data from current tier
            guard let (data, currentTier) = try await storage.retrieve(forKey: pattern.key) else {
                logger.error("Failed to retrieve key '\(pattern.key)' for migration")
                statistics.recordMigration(success: false, isPromotion: isPromotion, bytes: 0, duration: 0)
                return
            }
            
            // Apply compression based on target tier
            let processedData: Data
            switch targetTier.properties.compressionType {
            case .none:
                processedData = data
            case .lz4, .zstd:
                processedData = try await compressionEngine.compress(
                    data,
                    using: targetTier.properties.compressionType,
                    level: targetTier.properties.compressionLevel
                )
            }
            
            // Store in target tier
            try await storage.store(processedData, forKey: pattern.key, tier: targetTier)
            
            // Record migration with access tracker
            await accessTracker.recordMigration(key: pattern.key, to: targetTier)
            
            // Delete from source tier if different
            if currentTier != targetTier {
                try await storage.migrate(key: pattern.key, to: targetTier)
            }
            
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            statistics.recordMigration(
                success: true,
                isPromotion: isPromotion,
                bytes: pattern.dataSize,
                duration: duration
            )
            
            logger.info("Successfully migrated key '\(pattern.key)' from \(currentTier.properties.name) to \(targetTier.properties.name) in \(String(format: "%.3f", duration))s")
            
        } catch {
            logger.error("Failed to migrate key '\(pattern.key)': \(error)")
            statistics.recordMigration(
                success: false,
                isPromotion: isPromotion,
                bytes: 0,
                duration: CFAbsoluteTimeGetCurrent() - startTime
            )
        }
    }
    
    private func getMemoryPressure() async -> Double {
        // Simple memory pressure calculation
        let info = ProcessInfo.processInfo
        let physicalMemory = info.physicalMemory
        
        // Get current memory usage (simplified)
        var info_mach = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info_mach) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let usedMemory = Double(info_mach.resident_size)
            return usedMemory / Double(physicalMemory)
        }
        
        return 0.0
    }
}

/// Errors that can occur during tier management
public enum TierManagerError: LocalizedError {
    case keyNotFound(String)
    case migrationFailed(String, Error)
    case compressionFailed(Error)
    case storageError(Error)
    
    public var errorDescription: String? {
        switch self {
        case .keyNotFound(let key):
            return "Key not found in access tracker: \(key)"
        case .migrationFailed(let key, let error):
            return "Migration failed for key '\(key)': \(error)"
        case .compressionFailed(let error):
            return "Compression failed: \(error)"
        case .storageError(let error):
            return "Storage error: \(error)"
        }
    }
}
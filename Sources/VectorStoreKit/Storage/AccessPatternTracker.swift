import Foundation
import os.log

/// Tracks access patterns for intelligent tier migration decisions
public actor AccessPatternTracker {
    
    /// Access pattern statistics for a specific key
    public struct Pattern: Sendable {
        public let key: String
        public var accessCount: Int
        public var lastAccessTime: Date
        public var createdAt: Date
        public var dataSize: Int
        public var currentTier: StorageTier
        
        /// Moving averages for access frequency
        public var hourlyAccessRate: Double
        public var dailyAccessRate: Double
        public var weeklyAccessRate: Double
        
        /// Calculate temperature score (0-1, higher = hotter)
        public var temperature: Double {
            let now = Date()
            let hoursSinceAccess = now.timeIntervalSince(lastAccessTime) / 3600
            let hoursSinceCreation = now.timeIntervalSince(createdAt) / 3600
            
            // Recency factor (exponential decay)
            let recencyScore = exp(-hoursSinceAccess / 24) // Half-life of 24 hours
            
            // Frequency factor
            let avgAccessesPerHour = Double(accessCount) / max(hoursSinceCreation, 1)
            let frequencyScore = min(avgAccessesPerHour / 10, 1.0) // Normalize to 10 accesses/hour
            
            // Size factor (smaller items are hotter)
            let sizeScore = 1.0 / (1.0 + Double(dataSize) / (10 * 1024 * 1024)) // 10MB baseline
            
            // Weighted combination
            return recencyScore * 0.5 + frequencyScore * 0.3 + sizeScore * 0.2
        }
    }
    
    /// Configuration for access pattern tracking
    public struct Configuration: Sendable {
        /// Promotion threshold (temperature above this = promote)
        public let promotionThreshold: Double
        
        /// Demotion threshold (temperature below this = demote)
        public let demotionThreshold: Double
        
        /// Minimum time between tier changes (prevents thrashing)
        public let migrationCooldown: TimeInterval
        
        /// Maximum patterns to track in memory
        public let maxTrackedPatterns: Int
        
        /// Time window for access rate calculations
        public let rateWindowHours: Int
        
        public init(
            promotionThreshold: Double = 0.7,
            demotionThreshold: Double = 0.3,
            migrationCooldown: TimeInterval = 3600, // 1 hour
            maxTrackedPatterns: Int = 100_000,
            rateWindowHours: Int = 168 // 1 week
        ) {
            self.promotionThreshold = promotionThreshold
            self.demotionThreshold = demotionThreshold
            self.migrationCooldown = migrationCooldown
            self.maxTrackedPatterns = maxTrackedPatterns
            self.rateWindowHours = rateWindowHours
        }
    }
    
    // MARK: - Private Properties
    
    private var patterns: [String: Pattern] = [:]
    private var lastMigration: [String: Date] = [:]
    private let configuration: Configuration
    private let logger = Logger(subsystem: "VectorStoreKit", category: "AccessPatternTracker")
    
    /// Time-bucketed access counts for rate calculation
    private var accessBuckets: [String: [Date: Int]] = [:]
    private let bucketDuration: TimeInterval = 3600 // 1 hour buckets
    
    // MARK: - Initialization
    
    public init(configuration: Configuration = Configuration()) {
        self.configuration = configuration
        patterns.reserveCapacity(min(configuration.maxTrackedPatterns, 10_000))
    }
    
    // MARK: - Access Tracking
    
    /// Record an access to a key
    public func recordAccess(
        key: String,
        dataSize: Int,
        currentTier: StorageTier
    ) {
        let now = Date()
        
        // Update or create pattern
        if var pattern = patterns[key] {
            pattern.accessCount += 1
            pattern.lastAccessTime = now
            pattern.currentTier = currentTier
            
            // Update access rates
            updateAccessRates(for: key, pattern: &pattern, at: now)
            
            patterns[key] = pattern
        } else {
            // Create new pattern
            let pattern = Pattern(
                key: key,
                accessCount: 1,
                lastAccessTime: now,
                createdAt: now,
                dataSize: dataSize,
                currentTier: currentTier,
                hourlyAccessRate: 0,
                dailyAccessRate: 0,
                weeklyAccessRate: 0
            )
            patterns[key] = pattern
            
            // Evict oldest patterns if needed
            if patterns.count > configuration.maxTrackedPatterns {
                evictOldestPatterns()
            }
        }
        
        // Update access buckets
        updateAccessBucket(for: key, at: now)
        
        logger.debug("Recorded access for key '\(key)' in tier \(currentTier.properties.name)")
    }
    
    /// Get migration recommendation for a key
    public func getMigrationRecommendation(for key: String) -> MigrationDecision {
        guard let pattern = patterns[key] else {
            return .stay
        }
        
        // Check cooldown
        if let lastMigrationTime = lastMigration[key] {
            let timeSinceMigration = Date().timeIntervalSince(lastMigrationTime)
            if timeSinceMigration < configuration.migrationCooldown {
                return .stay
            }
        }
        
        let temperature = pattern.temperature
        let currentTier = pattern.currentTier
        
        // Promotion logic
        if temperature > configuration.promotionThreshold {
            if let higherTier = currentTier.higherTier {
                logger.info("Recommending promotion for key '\(key)' from \(currentTier.properties.name) to \(higherTier.properties.name) (temperature: \(temperature))")
                return .promote(to: higherTier)
            }
        }
        
        // Demotion logic
        if temperature < configuration.demotionThreshold {
            if let lowerTier = currentTier.lowerTier {
                logger.info("Recommending demotion for key '\(key)' from \(currentTier.properties.name) to \(lowerTier.properties.name) (temperature: \(temperature))")
                return .demote(to: lowerTier)
            }
        }
        
        return .stay
    }
    
    /// Record a tier migration
    public func recordMigration(key: String, to tier: StorageTier) {
        lastMigration[key] = Date()
        
        if var pattern = patterns[key] {
            pattern.currentTier = tier
            patterns[key] = pattern
        }
        
        logger.debug("Recorded migration for key '\(key)' to tier \(tier.properties.name)")
    }
    
    /// Get access pattern for a key
    public func getPattern(for key: String) -> Pattern? {
        patterns[key]
    }
    
    /// Get all patterns sorted by temperature
    public func getHottestPatterns(limit: Int = 100) -> [Pattern] {
        Array(patterns.values
            .sorted { $0.temperature > $1.temperature }
            .prefix(limit))
    }
    
    /// Get migration candidates
    public func getMigrationCandidates(limit: Int = 100) -> [(Pattern, MigrationDecision)] {
        var candidates: [(Pattern, MigrationDecision)] = []
        
        for pattern in patterns.values {
            let decision = getMigrationRecommendation(for: pattern.key)
            if decision != .stay {
                candidates.append((pattern, decision))
            }
        }
        
        // Sort by temperature difference from thresholds
        return Array(candidates
            .sorted { abs($0.0.temperature - 0.5) > abs($1.0.temperature - 0.5) }
            .prefix(limit))
    }
    
    /// Clear tracking data for a key
    public func removePattern(for key: String) {
        patterns.removeValue(forKey: key)
        lastMigration.removeValue(forKey: key)
        accessBuckets.removeValue(forKey: key)
    }
    
    /// Get statistics about tracked patterns
    public func getStatistics() -> Statistics {
        let totalPatterns = patterns.count
        let temperatures = patterns.values.map { $0.temperature }
        let avgTemperature = temperatures.isEmpty ? 0 : temperatures.reduce(0, +) / Double(temperatures.count)
        
        let tierDistribution = Dictionary(grouping: patterns.values, by: { $0.currentTier })
            .mapValues { $0.count }
        
        return Statistics(
            totalPatterns: totalPatterns,
            averageTemperature: avgTemperature,
            tierDistribution: tierDistribution,
            hotPatterns: patterns.values.filter { $0.temperature > configuration.promotionThreshold }.count,
            coldPatterns: patterns.values.filter { $0.temperature < configuration.demotionThreshold }.count
        )
    }
    
    // MARK: - Private Methods
    
    private func updateAccessRates(for key: String, pattern: inout Pattern, at time: Date) {
        let buckets = accessBuckets[key] ?? [:]
        let now = time
        
        // Calculate rates for different time windows
        let oneHourAgo = now.addingTimeInterval(-3600)
        let oneDayAgo = now.addingTimeInterval(-86400)
        let oneWeekAgo = now.addingTimeInterval(-604800)
        
        var hourlyCount = 0
        var dailyCount = 0
        var weeklyCount = 0
        
        for (bucketTime, count) in buckets {
            if bucketTime >= oneHourAgo {
                hourlyCount += count
            }
            if bucketTime >= oneDayAgo {
                dailyCount += count
            }
            if bucketTime >= oneWeekAgo {
                weeklyCount += count
            }
        }
        
        pattern.hourlyAccessRate = Double(hourlyCount)
        pattern.dailyAccessRate = Double(dailyCount) / 24
        pattern.weeklyAccessRate = Double(weeklyCount) / 168
    }
    
    private func updateAccessBucket(for key: String, at time: Date) {
        let bucketTime = Date(timeIntervalSince1970: floor(time.timeIntervalSince1970 / bucketDuration) * bucketDuration)
        
        var buckets = accessBuckets[key] ?? [:]
        buckets[bucketTime, default: 0] += 1
        
        // Clean old buckets
        let cutoffTime = time.addingTimeInterval(-Double(configuration.rateWindowHours) * 3600)
        buckets = buckets.filter { $0.key >= cutoffTime }
        
        accessBuckets[key] = buckets
    }
    
    private func evictOldestPatterns() {
        let evictCount = patterns.count / 10 // Evict 10% of patterns
        
        let coldestPatterns = patterns.values
            .sorted { $0.temperature < $1.temperature }
            .prefix(evictCount)
        
        for pattern in coldestPatterns {
            removePattern(for: pattern.key)
        }
        
        logger.debug("Evicted \(evictCount) coldest patterns")
    }
    
    // MARK: - Statistics
    
    public struct Statistics: Sendable {
        public let totalPatterns: Int
        public let averageTemperature: Double
        public let tierDistribution: [StorageTier: Int]
        public let hotPatterns: Int
        public let coldPatterns: Int
    }
}
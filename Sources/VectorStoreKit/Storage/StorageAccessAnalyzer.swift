// VectorStoreKit: Storage Access Pattern Analyzer
//
// Analyzes and tracks access patterns for intelligent data migration

import Foundation
import os.log

/// Analyzes storage access patterns for intelligent tier management
actor StorageAccessAnalyzer {
    // MARK: - Types
    
    /// Access type for tracking
    enum StorageAccessType {
        case read
        case write
    }
    
    /// Access pattern information for a key
    struct AccessPattern {
        let frequency: AccessFrequency
        let recentActivity: Bool
        let lastAccess: Date
        let accessCount: Int
        let readWriteRatio: Float
        let averageSize: Int
        let accessTimes: [Date]
        
        static let `default` = AccessPattern(
            frequency: .medium,
            recentActivity: false,
            lastAccess: Date.distantPast,
            accessCount: 0,
            readWriteRatio: 1.0,
            averageSize: 0,
            accessTimes: []
        )
    }
    
    /// Access frequency levels
    enum AccessFrequency: Int, Comparable {
        case rare = 0
        case low = 1
        case medium = 2
        case high = 3
        
        static func < (lhs: AccessFrequency, rhs: AccessFrequency) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }
    
    // MARK: - Properties
    
    private var accessRecords: [String: AccessRecord] = [:]
    private var globalStatistics = GlobalAccessStatistics()
    private let logger = Logger(subsystem: "VectorStoreKit", category: "StorageAccessAnalyzer")
    
    // Configuration
    private let recentAccessThreshold: TimeInterval = 3600 // 1 hour
    private let highFrequencyThreshold = 100
    private let mediumFrequencyThreshold = 10
    private let maxAccessTimesStored = 100
    private let decayFactor: Float = 0.95 // For time-based decay
    
    // MARK: - Public Methods
    
    /// Record an access event
    func recordAccess(key: String, type: StorageAccessType, size: Int) async {
        let now = Date()
        
        // Update or create access record
        if var record = accessRecords[key] {
            record.recordAccess(type: type, size: size, time: now)
            accessRecords[key] = record
        } else {
            var record = AccessRecord(key: key)
            record.recordAccess(type: type, size: size, time: now)
            accessRecords[key] = record
        }
        
        // Update global statistics
        switch type {
        case .read:
            globalStatistics.totalReads += 1
        case .write:
            globalStatistics.totalWrites += 1
        }
        globalStatistics.totalAccesses += 1
        
        // Periodic cleanup of old records
        if globalStatistics.totalAccesses % 10000 == 0 {
            await performCleanup()
        }
    }
    
    /// Get access pattern for a specific key
    func getPattern(for key: String) async -> StorageAccessPattern {
        guard let record = accessRecords[key] else {
            return StorageAccessPattern.default
        }
        
        let pattern = analyzeRecord(record)
        
        // Convert to the expected format
        return StorageAccessPattern(
            frequency: pattern.frequency,
            recentActivity: pattern.recentActivity
        )
    }
    
    /// Remove tracking for a key
    func removeKey(_ key: String) async {
        accessRecords.removeValue(forKey: key)
    }
    
    /// Get global access statistics
    func getStatistics() async -> AccessPatternStatistics {
        // Calculate access distribution
        var distribution: [String: UInt64] = [:]
        
        for (_, record) in accessRecords {
            let pattern = analyzeRecord(record)
            switch pattern.frequency {
            case .high:
                distribution["hot", default: 0] += 1
            case .medium:
                distribution["warm", default: 0] += 1
            case .low:
                distribution["cold", default: 0] += 1
            case .rare:
                distribution["archive", default: 0] += 1
            }
        }
        
        let readWriteRatio = globalStatistics.totalReads > 0 
            ? Float(globalStatistics.totalReads) / Float(globalStatistics.totalWrites + 1)
            : 0
        
        let hotDataPercentage = Float(distribution["hot"] ?? 0) / Float(max(1, accessRecords.count))
        
        return AccessPatternStatistics(
            totalAccesses: globalStatistics.totalAccesses,
            readWriteRatio: readWriteRatio,
            hotDataPercentage: hotDataPercentage,
            accessDistribution: distribution
        )
    }
    
    /// Reset all tracking data
    func reset() async {
        accessRecords.removeAll()
        globalStatistics = GlobalAccessStatistics()
        logger.info("Access analyzer reset")
    }
    
    /// Optimize internal data structures
    func optimize() async {
        // Remove stale records
        await performCleanup()
        
        // Compact access times for frequently accessed keys
        for key in accessRecords.keys {
            if var record = accessRecords[key], record.accessTimes.count > maxAccessTimesStored {
                record.compactAccessTimes(maxCount: maxAccessTimesStored)
                accessRecords[key] = record
            }
        }
        
        logger.info("Access analyzer optimized")
    }
    
    /// Get keys by access frequency
    func getKeysByFrequency(_ frequency: AccessFrequency) async -> [String] {
        accessRecords.compactMap { key, record in
            let pattern = analyzeRecord(record)
            return pattern.frequency == frequency ? key : nil
        }
    }
    
    /// Get hot data candidates for promotion
    func getHotDataCandidates(limit: Int = 100) async -> [String] {
        let hotKeys = accessRecords
            .map { (key: $0.key, pattern: analyzeRecord($0.value)) }
            .filter { $0.pattern.frequency == .high && $0.pattern.recentActivity }
            .sorted { $0.pattern.accessCount > $1.pattern.accessCount }
            .prefix(limit)
            .map { $0.key }
        
        return Array(hotKeys)
    }
    
    /// Get cold data candidates for demotion
    func getColdDataCandidates(limit: Int = 100) async -> [String] {
        let now = Date()
        let coldThreshold = now.addingTimeInterval(-7 * 24 * 3600) // 7 days
        
        let coldKeys = accessRecords
            .map { (key: $0.key, pattern: analyzeRecord($0.value)) }
            .filter { 
                $0.pattern.frequency == .rare && 
                $0.pattern.lastAccess < coldThreshold 
            }
            .sorted { $0.pattern.lastAccess < $1.pattern.lastAccess }
            .prefix(limit)
            .map { $0.key }
        
        return Array(coldKeys)
    }
    
    // MARK: - Private Methods
    
    private func analyzeRecord(_ record: AccessRecord) -> AccessPattern {
        let now = Date()
        
        // Determine frequency based on access count and time
        let frequency = determineFrequency(record: record, now: now)
        
        // Check recent activity
        let recentActivity = record.lastAccess.timeIntervalSince(now) > -recentAccessThreshold
        
        // Calculate read/write ratio
        let readWriteRatio = record.writeCount > 0
            ? Float(record.readCount) / Float(record.writeCount)
            : Float(record.readCount)
        
        // Average size
        let averageSize = record.totalAccessCount > 0
            ? record.totalSize / record.totalAccessCount
            : 0
        
        return AccessPattern(
            frequency: frequency,
            recentActivity: recentActivity,
            lastAccess: record.lastAccess,
            accessCount: record.totalAccessCount,
            readWriteRatio: readWriteRatio,
            averageSize: averageSize,
            accessTimes: record.accessTimes
        )
    }
    
    private func determineFrequency(record: AccessRecord, now: Date) -> AccessFrequency {
        // Apply time decay to access count
        let hoursSinceLastAccess = -record.lastAccess.timeIntervalSince(now) / 3600
        let decayedCount = Float(record.totalAccessCount) * pow(decayFactor, Float(hoursSinceLastAccess))
        
        // Also consider access rate (accesses per hour)
        let hoursSinceFirst = max(1, -record.firstAccess.timeIntervalSince(now) / 3600)
        let accessRate = Float(record.totalAccessCount) / Float(hoursSinceFirst)
        
        // Combined score
        let score = decayedCount * (1 + accessRate)
        
        if score >= Float(highFrequencyThreshold) {
            return .high
        } else if score >= Float(mediumFrequencyThreshold) {
            return .medium
        } else if score >= 1 {
            return .low
        } else {
            return .rare
        }
    }
    
    private func performCleanup() async {
        let now = Date()
        let staleThreshold = now.addingTimeInterval(-30 * 24 * 3600) // 30 days
        
        var keysToRemove: [String] = []
        
        for (key, record) in accessRecords {
            // Remove records that haven't been accessed in 30 days and have low frequency
            if record.lastAccess < staleThreshold {
                let pattern = analyzeRecord(record)
                if pattern.frequency == .rare {
                    keysToRemove.append(key)
                }
            }
        }
        
        for key in keysToRemove {
            accessRecords.removeValue(forKey: key)
        }
        
        if !keysToRemove.isEmpty {
            logger.info("Cleaned up \(keysToRemove.count) stale access records")
        }
    }
}

// MARK: - Supporting Types

/// Access record for a single key
private struct AccessRecord {
    let key: String
    var firstAccess: Date
    var lastAccess: Date
    var readCount: Int = 0
    var writeCount: Int = 0
    var totalSize: Int = 0
    var accessTimes: [Date] = []
    
    var totalAccessCount: Int { readCount + writeCount }
    
    init(key: String) {
        self.key = key
        self.firstAccess = Date()
        self.lastAccess = Date()
    }
    
    mutating func recordAccess(type: StorageAccessAnalyzer.StorageAccessType, size: Int, time: Date) {
        lastAccess = time
        totalSize += size
        
        switch type {
        case .read:
            readCount += 1
        case .write:
            writeCount += 1
        }
        
        // Keep limited access times for pattern analysis
        accessTimes.append(time)
        if accessTimes.count > 1000 {
            // Keep only recent 500 when we hit 1000
            accessTimes = Array(accessTimes.suffix(500))
        }
    }
    
    mutating func compactAccessTimes(maxCount: Int) {
        guard accessTimes.count > maxCount else { return }
        
        // Keep a representative sample: first, last, and evenly distributed middle
        let first = accessTimes.first!
        let last = accessTimes.last!
        let step = accessTimes.count / (maxCount - 2)
        
        var compacted = [first]
        for i in stride(from: step, to: accessTimes.count - step, by: step) {
            compacted.append(accessTimes[i])
        }
        compacted.append(last)
        
        accessTimes = compacted
    }
}

/// Global access statistics
private struct GlobalAccessStatistics {
    var totalAccesses: UInt64 = 0
    var totalReads: UInt64 = 0
    var totalWrites: UInt64 = 0
}

/// Storage access pattern (matches the interface expected by HierarchicalStorage)
struct StorageAccessPattern {
    let frequency: StorageAccessAnalyzer.AccessFrequency
    let recentActivity: Bool
    
    static let `default` = StorageAccessPattern(
        frequency: .medium,
        recentActivity: false
    )
}

/// Access frequency (re-exported for HierarchicalStorage compatibility)
typealias AccessFrequency = StorageAccessAnalyzer.AccessFrequency
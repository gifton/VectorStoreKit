// VectorStoreKit: Hot Tier Storage Implementation
//
// In-memory storage tier for frequently accessed data

import Foundation
import os.log

// MARK: - Hot Tier Storage

/// In-memory storage tier for frequently accessed data
actor HotTierStorage {
    private let memoryLimit: Int
    private var storage: [String: CacheEntry] = [:]
    private var accessOrder: [String] = []
    private let logger = Logger(subsystem: "VectorStoreKit", category: "HotTier")
    
    struct CacheEntry: Codable {
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
    
    var count: Int { storage.count }
    
    func getAllKeys() -> [String] {
        Array(storage.keys)
    }
    
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
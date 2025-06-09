// VectorStoreKit: Cache Entry and Common Types
//
// Common types and helpers for vector cache implementations

import Foundation
import simd

// MARK: - Cache Entry

/// Cache entry wrapper with metadata
public struct CacheEntry<Vector: SIMD> where Vector.Scalar: BinaryFloatingPoint {
    public let vector: Vector
    public let priority: CachePriority
    public let timestamp: Date
    public let accessCount: Int
    public let metadata: CacheEntryMetadata?
    
    public init(
        vector: Vector,
        priority: CachePriority,
        timestamp: Date = Date(),
        accessCount: Int = 0,
        metadata: CacheEntryMetadata? = nil
    ) {
        self.vector = vector
        self.priority = priority
        self.timestamp = timestamp
        self.accessCount = accessCount
        self.metadata = metadata
    }
}

// MARK: - Cache Entry Metadata

/// Additional metadata for cache entries
public struct CacheEntryMetadata: Sendable {
    public let source: String?
    public let lastAccessed: Date
    public let size: Int
    public let compressionRatio: Float?
    
    public init(
        source: String? = nil,
        lastAccessed: Date = Date(),
        size: Int,
        compressionRatio: Float? = nil
    ) {
        self.source = source
        self.lastAccessed = lastAccessed
        self.size = size
        self.compressionRatio = compressionRatio
    }
}

// MARK: - Cache Statistics

/// Basic cache statistics implementation
public struct BasicCacheStatistics: CacheStatistics {
    public let hitCount: Int
    public let missCount: Int
    public let evictionCount: Int
    public let totalAccessTime: TimeInterval
    public let memoryUsage: Int
    
    // CacheStatistics protocol properties
    public var hits: Int { hitCount }
    public var misses: Int { missCount }
    public var hitRate: Float {
        let total = hitCount + missCount
        return total > 0 ? Float(hitCount) / Float(total) : 0.0
    }
    public var memoryEfficiency: Float {
        // Calculate memory efficiency based on hit rate and memory usage
        guard memoryUsage > 0 else { return 0.0 }
        let baseEfficiency = hitRate
        let memoryPenalty = min(1.0, Float(memoryUsage) / Float(100_000_000)) // 100MB reference
        return baseEfficiency * (1.0 - memoryPenalty * 0.2) // Max 20% penalty for memory usage
    }
    
    public init(
        hitCount: Int = 0,
        missCount: Int = 0,
        evictionCount: Int = 0,
        totalAccessTime: TimeInterval = 0,
        memoryUsage: Int = 0
    ) {
        self.hitCount = hitCount
        self.missCount = missCount
        self.evictionCount = evictionCount
        self.totalAccessTime = totalAccessTime
        self.memoryUsage = memoryUsage
    }
}

// MARK: - Cache Errors

/// Vector cache specific errors
public enum VectorCacheError: Error, Sendable {
    case invalidConfiguration(String)
    case memoryLimitExceeded
    case cacheCorrupted
    case entryNotFound(VectorID)
    case serializationFailed
    case deserializationFailed
    case storageAccessFailed(String)
}

// MARK: - Cache Performance Recommendation Extension

public extension CacheRecommendation {
    /// Create a size adjustment recommendation
    static func sizeAdjustment(
        currentSize: Int,
        recommendedSize: Int,
        expectedImprovement: Float
    ) -> CacheRecommendation {
        CacheRecommendation(
            type: .sizeAdjustment,
            description: "Adjust cache size from \(currentSize) to \(recommendedSize) bytes",
            expectedImprovement: expectedImprovement
        )
    }
    
    /// Create a policy change recommendation
    static func policyChange(
        from currentPolicy: EvictionPolicy,
        to recommendedPolicy: EvictionPolicy,
        expectedImprovement: Float
    ) -> CacheRecommendation {
        CacheRecommendation(
            type: .policyChange,
            description: "Change eviction policy from \(currentPolicy.rawValue) to \(recommendedPolicy.rawValue)",
            expectedImprovement: expectedImprovement
        )
    }
    
    /// Create a prefetching recommendation
    static func enablePrefetching(expectedImprovement: Float) -> CacheRecommendation {
        CacheRecommendation(
            type: .prefetching,
            description: "Enable predictive prefetching based on access patterns",
            expectedImprovement: expectedImprovement
        )
    }
}

// MARK: - Memory Estimation

/// Estimate memory size for a vector
public func estimateVectorMemorySize<Vector: SIMD>(_ vector: Vector) -> Int 
    where Vector.Scalar: BinaryFloatingPoint {
    // Base size: vector dimensions * size of scalar
    let vectorSize = vector.scalarCount * MemoryLayout<Vector.Scalar>.size
    
    // Add overhead for metadata and alignment
    let metadataOverhead = 64 // Conservative estimate for entry metadata
    let alignmentPadding = 16 // Memory alignment padding
    
    return vectorSize + metadataOverhead + alignmentPadding
}

/// Estimate total memory for cache entries
public func estimateCacheMemoryUsage<Vector: SIMD>(
    entries: [VectorID: CacheEntry<Vector>]
) -> Int where Vector.Scalar: BinaryFloatingPoint {
    var totalSize = 0
    
    for (_, entry) in entries {
        totalSize += estimateVectorMemorySize(entry.vector)
        // Add key storage overhead
        totalSize += MemoryLayout<VectorID>.size
        // Add dictionary overhead per entry
        totalSize += 48 // Conservative estimate for dictionary entry overhead
    }
    
    return totalSize
}

// MARK: - Cache Analysis Helpers

/// Analyze cache access patterns
public struct SimpleCacheAccessPattern: Sendable {
    public let frequentlyAccessed: Set<VectorID>
    public let rarelyAccessed: Set<VectorID>
    public let accessSequences: [[VectorID]]
    public let temporalLocality: Float
    public let spatialLocality: Float
    
    public init(
        frequentlyAccessed: Set<VectorID>,
        rarelyAccessed: Set<VectorID>,
        accessSequences: [[VectorID]],
        temporalLocality: Float,
        spatialLocality: Float
    ) {
        self.frequentlyAccessed = frequentlyAccessed
        self.rarelyAccessed = rarelyAccessed
        self.accessSequences = accessSequences
        self.temporalLocality = temporalLocality
        self.spatialLocality = spatialLocality
    }
}

/// Calculate temporal locality score
public func calculateTemporalLocality(accessHistory: [VectorID]) -> Float {
    guard accessHistory.count > 1 else { return 0.0 }
    
    var recentAccesses = Set<VectorID>()
    var localityScore: Float = 0.0
    let windowSize = 10
    
    for (index, vectorId) in accessHistory.enumerated() {
        if recentAccesses.contains(vectorId) {
            localityScore += 1.0
        }
        
        recentAccesses.insert(vectorId)
        
        // Maintain sliding window
        if index >= windowSize {
            let oldId = accessHistory[index - windowSize]
            if !accessHistory[(index - windowSize + 1)...index].contains(oldId) {
                recentAccesses.remove(oldId)
            }
        }
    }
    
    return localityScore / Float(accessHistory.count - 1)
}
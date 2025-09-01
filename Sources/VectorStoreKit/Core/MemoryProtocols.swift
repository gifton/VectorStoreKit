// MemoryProtocols.swift
// VectorStoreKit
//
// Protocol definitions for memory management components to avoid circular dependencies

import Foundation
import Metal

/// Protocol for buffer pools that can be managed by MemoryManager
public protocol MemoryManagedBufferPool: Sendable {
    /// Get current statistics about buffer pool usage
    func getStatistics() async -> BufferPoolStatistics
    
    /// Clear all cached buffers to free memory
    func clearAll() async
    
    /// Get current memory usage in bytes
    func getCurrentMemoryUsage() async -> Int
}

/// Protocol for buffer caches that can be managed by MemoryManager
public protocol MemoryManagedCache: Sendable {
    /// Evict least recently used buffers to free memory
    func evictLRU(targetSize: Int) async
    
    /// Clear all cached buffers
    func clearCache() async
    
    /// Get current memory usage in bytes
    func getCurrentMemoryUsage() async -> Int
    
    /// Get cache statistics
    func getStatistics() async -> BufferCacheStatistics
}

/// Protocol for components that can adjust batch sizes
public protocol BatchSizeManager: Sendable {
    /// Get current batch size
    func getCurrentBatchSize() async -> Int
    
    /// Reduce batch size by a factor
    func reduceBatchSize(by factor: Float) async
    
    /// Reset to default batch size
    func resetBatchSize() async
}

/// Protocol for components that maintain training history
public protocol TrainingHistoryManager: Sendable {
    /// Clear training history to free memory
    func clearHistory() async
    
    /// Get current history size in bytes
    func getHistorySize() async -> Int
}

/// Protocol for components that are aware of memory pressure
public protocol MemoryPressureAware: Sendable {
    /// Handle memory pressure at specified level
    func handleMemoryPressure(_ level: SystemMemoryPressure) async
    
    /// Get current memory usage in bytes
    func getCurrentMemoryUsage() async -> Int
    
    /// Get memory usage statistics
    func getMemoryStatistics() async -> MemoryComponentStatistics
}

/// Protocol for distance computation components that support memory pressure
public protocol MemoryPressureAwareDistanceComputation: MemoryPressureAware {
    /// Reduce computational batch size during memory pressure
    func reduceBatchSize(by factor: Float) async
    
    /// Reset to default batch size
    func resetBatchSize() async
    
    /// Get current batch size
    func getCurrentBatchSize() async -> Int
}

/// Statistics for buffer cache usage
public struct BufferCacheStatistics: Sendable {
    public let totalBuffers: Int
    public let totalMemory: Int
    public let hitRate: Float
    public let evictionCount: Int
    
    // Additional properties from BufferCache usage
    public let storeCount: Int
    public let totalBytesStored: Int
    public let missCount: Int
    public let hitCount: Int
    public let totalBytesEvicted: Int
    public let clearCount: Int
    public let currentBufferCount: Int
    public let currentMemoryUsage: Int
    public let memoryLimit: Int
    
    // Computed properties
    public var memoryUtilization: Double {
        guard memoryLimit > 0 else { return 0.0 }
        return Double(currentMemoryUsage) / Double(memoryLimit)
    }
    
    public init(
        totalBuffers: Int = 0,
        totalMemory: Int = 0,
        hitRate: Float = 0.0,
        evictionCount: Int = 0,
        storeCount: Int = 0,
        totalBytesStored: Int = 0,
        missCount: Int = 0,
        hitCount: Int = 0,
        totalBytesEvicted: Int = 0,
        clearCount: Int = 0,
        currentBufferCount: Int = 0,
        currentMemoryUsage: Int = 0,
        memoryLimit: Int = 0
    ) {
        self.totalBuffers = totalBuffers
        self.totalMemory = totalMemory
        self.hitRate = hitRate
        self.evictionCount = evictionCount
        self.storeCount = storeCount
        self.totalBytesStored = totalBytesStored
        self.missCount = missCount
        self.hitCount = hitCount
        self.totalBytesEvicted = totalBytesEvicted
        self.clearCount = clearCount
        self.currentBufferCount = currentBufferCount
        self.currentMemoryUsage = currentMemoryUsage
        self.memoryLimit = memoryLimit
    }
}

/// Statistics for memory pressure aware components
public struct MemoryComponentStatistics: Sendable {
    public let componentName: String
    public let currentMemoryUsage: Int
    public let peakMemoryUsage: Int
    public let pressureEventCount: Int
    public let lastPressureHandled: Date?
    public let averageResponseTime: TimeInterval
    
    public init(
        componentName: String,
        currentMemoryUsage: Int,
        peakMemoryUsage: Int,
        pressureEventCount: Int = 0,
        lastPressureHandled: Date? = nil,
        averageResponseTime: TimeInterval = 0
    ) {
        self.componentName = componentName
        self.currentMemoryUsage = currentMemoryUsage
        self.peakMemoryUsage = peakMemoryUsage
        self.pressureEventCount = pressureEventCount
        self.lastPressureHandled = lastPressureHandled
        self.averageResponseTime = averageResponseTime
    }
}
// BufferCache.swift
// VectorStoreKit
//
// Created for VectorStoreKit to manage Metal buffer memory with bounded growth
// and intelligent LRU eviction policy to prevent memory exhaustion.

import Foundation
@preconcurrency import Metal
import os.log

// Import the MetalBuffer type from Core/ML/Layer.swift
// MetalBuffer is defined there with shape and stride information

/// Metadata for cached buffers
private struct CachedBufferMetadata {
    let buffer: MetalBuffer
    let key: String
    let sizeInBytes: Int
    var lastAccessTime: Date
    var accessCount: Int
    
    /// Initialize with MetalBuffer from Core/ML/Layer.swift
    init(buffer: MetalBuffer, key: String) {
        self.buffer = buffer
        self.key = key
        self.sizeInBytes = buffer.buffer.length
        self.lastAccessTime = Date()
        self.accessCount = 1
    }
}

/// A thread-safe buffer cache that manages Metal buffer memory with bounded growth.
///
/// This actor implements an LRU (Least Recently Used) eviction policy to ensure
/// memory usage stays within defined limits. It prevents unbounded memory growth
/// that can lead to memory exhaustion in ML workloads.
///
/// ## Key Features
/// - Bounded memory growth with configurable limits (default 1GB)
/// - LRU eviction policy for intelligent memory management
/// - Usage tracking for cache efficiency monitoring
/// - Thread-safe operations via actor isolation
/// - Automatic eviction when memory pressure is detected
///
/// ## Usage Example
/// ```swift
/// let cache = BufferCache(device: metalDevice)
/// 
/// // Store a buffer
/// let buffer = await cache.getOrCreate(
///     key: "weights_layer1",
///     size: 1000,
///     create: { size in
///         // Create buffer if not cached
///         return createMetalBuffer(size: size)
///     }
/// )
/// 
/// // Manually evict if needed
/// await cache.evictIfNeeded()
/// ```
public actor BufferCache {
    // MARK: - Properties
    
    /// The Metal device used for buffer allocation
    private let device: MTLDevice
    
    /// Maximum memory allowed for the cache in bytes (default: 1GB)
    private let maxMemory: Int
    
    /// Current memory usage in bytes
    private var currentMemory: Int = 0
    
    /// Cache storage mapping keys to buffer metadata
    private var cache: [String: CachedBufferMetadata] = [:]
    
    /// Priority queue for LRU tracking (oldest access times first)
    private var accessOrder: [(key: String, lastAccess: Date)] = []
    
    /// Statistics for monitoring cache efficiency
    private var storeCount: Int = 0
    private var totalBytesStored: Int = 0
    private var missCount: Int = 0
    private var hitCount: Int = 0
    private var evictionCount: Int = 0
    private var clearCount: Int = 0
    
    /// Logger for debugging and monitoring
    private let logger = Logger(subsystem: "VectorStoreKit", category: "BufferCache")
    
    // MARK: - Initialization
    
    /// Creates a new buffer cache with specified memory limit
    /// - Parameters:
    ///   - device: The Metal device to use for buffer allocation
    ///   - maxMemory: Maximum memory in bytes (default: 1GB)
    public init(device: MTLDevice, maxMemory: Int = 1_073_741_824) {
        self.device = device
        self.maxMemory = maxMemory
        
        logger.info("BufferCache initialized with \(maxMemory / 1_048_576)MB limit")
    }
    
    // MARK: - Public Methods
    
    /// Stores a buffer in the cache with the given key
    /// - Parameters:
    ///   - buffer: The Metal buffer to cache
    ///   - key: Unique identifier for the buffer
    /// - Throws: `BufferCacheError` if the buffer is too large for the cache
    public func store(buffer: MetalBuffer, for key: String) async throws {
        let sizeInBytes = buffer.buffer.length
        
        // Check if buffer is too large for cache
        guard sizeInBytes <= maxMemory else {
            throw BufferCacheError.bufferTooLarge(size: sizeInBytes, maxSize: maxMemory)
        }
        
        // Remove existing buffer if present
        if let existing = cache[key] {
            currentMemory -= existing.sizeInBytes
            removeFromAccessOrder(key: key)
        }
        
        // Evict buffers if needed to make room
        while currentMemory + sizeInBytes > maxMemory {
            try await evictLeastRecentlyUsed()
        }
        
        // Store the new buffer
        let metadata = CachedBufferMetadata(
            buffer: buffer,
            key: key
        )
        
        cache[key] = metadata
        currentMemory += sizeInBytes
        accessOrder.append((key: key, lastAccess: metadata.lastAccessTime))
        
        // Update statistics
        storeCount += 1
        totalBytesStored += sizeInBytes
        
        logger.debug("Stored buffer '\(key)' (\(sizeInBytes / 1024)KB), current memory: \(self.currentMemory / 1_048_576)MB")
    }
    
    /// Retrieves a buffer from the cache if available
    /// - Parameter key: The key to look up
    /// - Returns: The cached buffer if found, nil otherwise
    public func retrieve(for key: String) async -> MetalBuffer? {
        guard var metadata = cache[key] else {
            missCount += 1
            return nil
        }
        
        // Update access metadata
        metadata.lastAccessTime = Date()
        metadata.accessCount += 1
        cache[key] = metadata
        
        // Update access order
        updateAccessOrder(key: key, newTime: metadata.lastAccessTime)
        
        // Update statistics
        hitCount += 1
        
        logger.debug("Retrieved buffer '\(key)', access count: \(metadata.accessCount)")
        
        return metadata.buffer
    }
    
    /// Gets a buffer from cache or creates it if not present
    /// - Parameters:
    ///   - key: The key to look up or store under
    ///   - create: Closure to create the buffer if not cached
    /// - Returns: The cached or newly created buffer
    /// - Throws: Errors from buffer creation or storage
    public func getOrCreate(
        key: String,
        create: () async throws -> MetalBuffer
    ) async throws -> MetalBuffer {
        if let cached = await retrieve(for: key) {
            return cached
        }
        
        let buffer = try await create()
        try await store(buffer: buffer, for: key)
        return buffer
    }
    
    /// Evicts buffers if current memory usage exceeds the limit
    public func evictIfNeeded() async throws {
        while currentMemory > maxMemory {
            try await evictLeastRecentlyUsed()
        }
    }
    
    /// Manually evicts the least recently used buffer
    /// - Throws: `BufferCacheError.cacheEmpty` if there are no buffers to evict
    public func evictLeastRecentlyUsed() async throws {
        // Sort access order by last access time
        accessOrder.sort { $0.lastAccess < $1.lastAccess }
        
        guard let oldest = accessOrder.first else {
            throw BufferCacheError.cacheEmpty
        }
        
        // Remove from cache
        guard let metadata = cache.removeValue(forKey: oldest.key) else {
            // Inconsistent state - remove from access order and continue
            accessOrder.removeFirst()
            return
        }
        
        // Update memory tracking
        currentMemory -= metadata.sizeInBytes
        accessOrder.removeFirst()
        
        // Update statistics
        evictionCount += 1
        
        logger.info("Evicted buffer '\(oldest.key)' (\(metadata.sizeInBytes / 1024)KB), freed memory: \(self.currentMemory / 1_048_576)MB")
    }
    
    /// Clears all cached buffers
    public func clear() async {
        let totalBuffers = cache.count
        let totalMemory = currentMemory
        
        cache.removeAll()
        accessOrder.removeAll()
        currentMemory = 0
        
        clearCount += 1
        
        logger.info("Cleared \(totalBuffers) buffers, freed \(totalMemory / 1_048_576)MB")
    }
    
    /// Returns current cache statistics
    public func getStatistics() async -> BufferCacheStatistics {
        let hitRate = (hitCount + missCount) > 0 ? Float(hitCount) / Float(hitCount + missCount) : 0.0
        return BufferCacheStatistics(
            totalBuffers: cache.count,
            totalMemory: currentMemory,
            hitRate: hitRate,
            evictionCount: evictionCount,
            storeCount: storeCount,
            totalBytesStored: totalBytesStored,
            missCount: missCount,
            hitCount: hitCount,
            totalBytesEvicted: 0, // Not currently tracked
            clearCount: clearCount,
            currentBufferCount: cache.count,
            currentMemoryUsage: currentMemory,
            memoryLimit: maxMemory
        )
    }
    
    /// Returns detailed cache state for debugging
    public func getCacheState() async -> [String: BufferInfo] {
        cache.mapValues { metadata in
            BufferInfo(
                sizeInBytes: metadata.sizeInBytes,
                lastAccessTime: metadata.lastAccessTime,
                accessCount: metadata.accessCount
            )
        }
    }
    
    // MARK: - Memory Pressure Handling
    
    /// Handles system memory pressure by aggressively evicting buffers
    public func handleMemoryPressure() async {
        logger.warning("Handling memory pressure, current usage: \(self.currentMemory / 1_048_576)MB")
        
        // Evict half of the cached buffers
        let targetMemory = maxMemory / 2
        while currentMemory > targetMemory && !cache.isEmpty {
            try? await evictLeastRecentlyUsed()
        }
        
        logger.info("Memory pressure handled, reduced to: \(self.currentMemory / 1_048_576)MB")
    }
    
    // MARK: - Private Methods
    
    /// Updates the access order for a key with a new access time
    private func updateAccessOrder(key: String, newTime: Date) {
        accessOrder.removeAll { $0.key == key }
        accessOrder.append((key: key, lastAccess: newTime))
    }
    
    /// Removes a key from the access order tracking
    private func removeFromAccessOrder(key: String) {
        accessOrder.removeAll { $0.key == key }
    }
}

// MARK: - Protocol Conformance

extension BufferCache: MemoryManagedCache {
    /// Evict least recently used buffers until target size is reached
    public func evictLRU(targetSize: Int) async {
        while currentMemory > targetSize && !cache.isEmpty {
            try? await evictLeastRecentlyUsed()
        }
    }
    
    /// Clear all cached buffers
    public func clearCache() async {
        await clear()
    }
    
    /// Get current memory usage in bytes
    public func getCurrentMemoryUsage() async -> Int {
        return currentMemory
    }
}

// MARK: - Supporting Types

// BufferCacheStatistics is defined in MemoryProtocols.swift

/// Information about a cached buffer
public struct BufferInfo: Sendable {
    public let sizeInBytes: Int
    public let lastAccessTime: Date
    public let accessCount: Int
}

/// Errors that can occur during buffer cache operations
public enum BufferCacheError: LocalizedError {
    case bufferTooLarge(size: Int, maxSize: Int)
    case cacheEmpty
    case allocationFailed(reason: String)
    
    public var errorDescription: String? {
        switch self {
        case .bufferTooLarge(let size, let maxSize):
            return "Buffer size (\(size) bytes) exceeds cache limit (\(maxSize) bytes)"
        case .cacheEmpty:
            return "Cannot evict from empty cache"
        case .allocationFailed(let reason):
            return "Buffer allocation failed: \(reason)"
        }
    }
}

// MARK: - Extensions

extension BufferCache {
    /// Creates a buffer cache with custom configuration
    /// - Parameters:
    ///   - device: The Metal device to use
    ///   - maxMemoryMB: Maximum memory in megabytes
    /// - Returns: A configured buffer cache
    public static func create(
        device: MTLDevice,
        maxMemoryMB: Int = 1024
    ) -> BufferCache {
        BufferCache(
            device: device,
            maxMemory: maxMemoryMB * 1_048_576
        )
    }
    
    /// Monitors cache efficiency and logs warnings if performance degrades
    public func monitorEfficiency() async {
        let stats = await getStatistics()
        
        if stats.hitRate < 50 && stats.hitCount + stats.missCount > 100 {
            logger.warning("Low cache hit rate: \(stats.hitRate, format: .fixed(precision: 1))%")
        }
        
        if stats.evictionCount > stats.storeCount / 2 && stats.storeCount > 100 {
            logger.warning("High eviction rate: \(stats.evictionCount) evictions for \(stats.storeCount) stores")
        }
        
        if stats.memoryUtilization > 90 {
            logger.warning("High memory utilization: \(stats.memoryUtilization, format: .fixed(precision: 1))%")
        }
    }
}
// VectorStoreKit: Storage Slot Abstraction
//
// Shared storage mechanism for memory-efficient multi-cache operation

import Foundation
import os

// MARK: - Storage Slot

/// Reference-counted storage slot for vectors
/// Enables sharing vectors between multiple cache implementations
public final class StorageSlot<Vector: SIMD & Sendable>: Sendable 
where Vector.Scalar: BinaryFloatingPoint {
    
    /// The stored vector
    public let vector: Vector
    
    /// Unique identifier for this slot
    public let slotId: UUID
    
    /// Reference count (atomic)
    private let _refCount: OSAllocatedUnfairLock<Int32>
    
    /// Creation timestamp
    public let createdAt: ContinuousClock.Instant
    
    /// Size in bytes
    public let sizeInBytes: Int
    
    public init(vector: Vector) {
        self.vector = vector
        self.slotId = UUID()
        self.createdAt = ContinuousClock.now
        self.sizeInBytes = MemoryLayout<Vector>.size
        self._refCount = OSAllocatedUnfairLock(initialState: 1)
    }
    
    /// Increment reference count
    @discardableResult
    public func retain() -> Int32 {
        _refCount.withLock { count in
            count += 1
            return count
        }
    }
    
    /// Decrement reference count
    /// Returns true if this was the last reference
    @discardableResult
    public func release() -> Bool {
        _refCount.withLock { count in
            count -= 1
            return count == 0
        }
    }
    
    /// Current reference count
    public var refCount: Int32 {
        _refCount.withLock { $0 }
    }
    
    /// Check if slot is shared (refCount > 1)
    public var isShared: Bool {
        refCount > 1
    }
}

// MARK: - Shared Storage Manager

/// Manages shared storage slots for multiple caches
public actor SharedStorageManager<Vector: SIMD & Sendable> 
where Vector.Scalar: BinaryFloatingPoint {
    
    /// All storage slots by vector ID
    private var slots: [VectorID: StorageSlot<Vector>] = [:]
    
    /// Total memory usage
    private var totalMemoryUsage: Int = 0
    
    /// Memory limit (optional)
    private let memoryLimit: Int?
    
    /// Statistics
    private var stats = SharedStorageStatistics()
    
    public init(memoryLimit: Int? = nil) {
        self.memoryLimit = memoryLimit
    }
    
    /// Store or retrieve a vector slot
    /// If vector already exists, returns existing slot with incremented refcount
    public func storeVector(id: VectorID, vector: Vector) throws -> StorageSlot<Vector> {
        // Check if slot already exists
        if let existingSlot = slots[id] {
            existingSlot.retain()
            stats.sharedHits += 1
            return existingSlot
        }
        
        // Create new slot
        let slot = StorageSlot(vector: vector)
        
        // Check memory limit
        if let limit = memoryLimit {
            let newUsage = totalMemoryUsage + slot.sizeInBytes
            if newUsage > limit {
                throw SharedStorageError.memoryLimitExceeded(
                    requested: slot.sizeInBytes,
                    available: limit - totalMemoryUsage
                )
            }
        }
        
        // Store slot
        slots[id] = slot
        totalMemoryUsage += slot.sizeInBytes
        stats.totalSlots += 1
        
        return slot
    }
    
    /// Release a reference to a slot
    /// Removes slot if reference count reaches zero
    public func releaseSlot(id: VectorID) {
        guard let slot = slots[id] else { return }
        
        if slot.release() {
            // Last reference released, remove slot
            slots.removeValue(forKey: id)
            totalMemoryUsage -= slot.sizeInBytes
            stats.totalSlots -= 1
            stats.releasedSlots += 1
        }
    }
    
    /// Get slot without changing reference count (for inspection)
    public func getSlot(id: VectorID) -> StorageSlot<Vector>? {
        slots[id]
    }
    
    /// Current memory usage
    public var memoryUsage: Int {
        totalMemoryUsage
    }
    
    /// Number of stored slots
    public var slotCount: Int {
        slots.count
    }
    
    /// Get storage statistics
    public func statistics() -> SharedStorageStatistics {
        var stats = self.stats
        stats.currentMemoryUsage = totalMemoryUsage
        stats.sharedSlots = slots.values.filter { $0.isShared }.count
        return stats
    }
    
    /// Clear all slots (dangerous - only use when all caches are cleared)
    public func clearAll() {
        slots.removeAll()
        totalMemoryUsage = 0
        stats = SharedStorageStatistics()
    }
}

// MARK: - Storage Statistics

public struct SharedStorageStatistics: Sendable {
    public var totalSlots: Int = 0
    public var sharedSlots: Int = 0
    public var releasedSlots: Int = 0
    public var sharedHits: Int = 0
    public var currentMemoryUsage: Int = 0
    
    public var sharingRatio: Float {
        guard totalSlots > 0 else { return 0 }
        return Float(sharedSlots) / Float(totalSlots)
    }
    
    public var hitRate: Float {
        let totalAccesses = totalSlots + sharedHits
        guard totalAccesses > 0 else { return 0 }
        return Float(sharedHits) / Float(totalAccesses)
    }
}

// MARK: - Storage Errors

public enum SharedStorageError: Error, Sendable {
    case memoryLimitExceeded(requested: Int, available: Int)
    case slotNotFound(id: VectorID)
    case invalidOperation(String)
}

// MARK: - Cache Storage Adapter

/// Adapter to use shared storage with existing cache implementations
public protocol SharedStorageAdapter: Actor {
    associatedtype Vector: SIMD & Sendable where Vector.Scalar: BinaryFloatingPoint
    
    /// The shared storage manager
    var storageManager: SharedStorageManager<Vector> { get }
    
    /// Store a vector through shared storage
    func storeShared(id: VectorID, vector: Vector) async throws -> StorageSlot<Vector>
    
    /// Release a shared storage reference
    func releaseShared(id: VectorID) async
}

// MARK: - Slot-Based Cache Entry

/// Cache entry that references a shared storage slot
public struct SlotBasedCacheEntry<Vector: SIMD & Sendable>: Sendable 
where Vector.Scalar: BinaryFloatingPoint {
    /// Reference to the storage slot
    public let slot: StorageSlot<Vector>
    
    /// Cache-specific metadata
    public let priority: CachePriority
    public let timestamp: Date
    public var accessCount: Int
    
    /// Additional metadata
    public let metadata: CacheEntryMetadata?
    
    public init(
        slot: StorageSlot<Vector>,
        priority: CachePriority,
        timestamp: Date = Date(),
        accessCount: Int = 0,
        metadata: CacheEntryMetadata? = nil
    ) {
        self.slot = slot
        self.priority = priority
        self.timestamp = timestamp
        self.accessCount = accessCount
        self.metadata = metadata
    }
    
    /// Get the vector from the slot
    public var vector: Vector {
        slot.vector
    }
}

// MARK: - Memory Accounting Helpers

/// Precise memory accounting for shared storage
public struct SharedMemoryAccounting: Sendable {
    /// Calculate actual memory usage considering shared slots
    public static func calculateMemoryUsage<Vector: SIMD>(
        slots: [StorageSlot<Vector>]
    ) -> Int where Vector.Scalar: BinaryFloatingPoint {
        // Count unique slots only
        let uniqueSlots = Set(slots.map { $0.slotId })
        return uniqueSlots.count * MemoryLayout<Vector>.size
    }
    
    /// Calculate memory saved through sharing
    public static func calculateMemorySaved<Vector: SIMD>(
        slots: [StorageSlot<Vector>]
    ) -> Int where Vector.Scalar: BinaryFloatingPoint {
        let totalIfNotShared = slots.count * MemoryLayout<Vector>.size
        let actualUsage = calculateMemoryUsage(slots: slots)
        return totalIfNotShared - actualUsage
    }
}
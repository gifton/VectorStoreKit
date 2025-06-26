// MemoryPoolManager.swift
// VectorStoreKit
//
// Enhanced memory pool management with size tracking, memory pressure integration,
// automatic resizing, and defragmentation strategies

import Foundation
@preconcurrency import Dispatch
import os.log

// MARK: - Memory Pool Protocol

/// Enhanced protocol for type-specific memory pools with size tracking
public protocol MemoryPool: Actor {
    associatedtype Element
    
    func acquire() async throws -> Element
    func release(_ element: Element) async
    func preAllocate(count: Int) async
    func clear() async
    func statistics() async -> PoolStatistics
    func defragment() async -> DefragmentationResult
}

// MARK: - Pool Statistics

public struct PoolStatistics: Sendable {
    public let totalAllocated: Int
    public let currentlyInUse: Int
    public let peakUsage: Int
    public let hitRate: Double
    public let averageAllocationTime: TimeInterval
    public let totalMemoryBytes: Int
    public let fragmentationRatio: Double
    public let lastDefragmentation: Date?
    
    public var utilizationRate: Double {
        guard totalAllocated > 0 else { return 0 }
        return Double(currentlyInUse) / Double(totalAllocated)
    }
}

// MARK: - Defragmentation Result

public struct DefragmentationResult: Sendable {
    public let bytesReclaimed: Int
    public let objectsCompacted: Int
    public let duration: TimeInterval
    public let fragmentationBefore: Double
    public let fragmentationAfter: Double
}

// MARK: - Allocation Info

private struct AllocationInfo: Sendable {
    let id: UUID
    let size: Int
    let timestamp: Date
    let alignment: Int
}

// MARK: - Enhanced Buffer Pool

/// Enhanced buffer pool with size tracking and defragmentation
public actor EnhancedBufferPool: MemoryPool, MemoryManagedBufferPool {
    public typealias Element = UnsafeMutableRawPointer
    
    private struct BufferEntry {
        let pointer: UnsafeMutableRawPointer
        let size: Int
        let alignment: Int
        let allocationTime: Date
        var lastAccessTime: Date
    }
    
    // Pools organized by size buckets
    private var availablePools: [Int: [BufferEntry]] = [:]
    private var allocatedBuffers: [UnsafeMutableRawPointer: AllocationInfo] = [:]
    
    // Configuration
    private let maxPoolSize: Int
    private let defaultAlignment: Int
    private let growthFactor: Double
    private let shrinkThreshold: Double
    private let defragmentationThreshold: Double
    
    // Statistics
    private var totalBytesAllocated: Int = 0
    private var currentBytesInUse: Int = 0
    private var peakBytesUsage: Int = 0
    private var allocationCount: Int = 0
    private var hitCount: Int = 0
    private var missCount: Int = 0
    private var totalAllocationTime: TimeInterval = 0
    private var lastDefragmentation: Date?
    
    // Memory pressure integration
    private weak var memoryManager: MemoryManager?
    private var pressureCallbacks: [(SystemMemoryPressure) -> Void] = []
    
    // Usage pattern tracking
    private var sizeBucketUsage: [Int: Int] = [:]
    private var lastResizeCheck: Date = Date()
    private let resizeCheckInterval: TimeInterval = 60.0
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "EnhancedBufferPool")
    
    public init(
        maxPoolSize: Int = 100,
        defaultAlignment: Int = 64,
        growthFactor: Double = 1.5,
        shrinkThreshold: Double = 0.25,
        defragmentationThreshold: Double = 0.3,
        memoryManager: MemoryManager? = nil
    ) {
        self.maxPoolSize = maxPoolSize
        self.defaultAlignment = defaultAlignment
        self.growthFactor = growthFactor
        self.shrinkThreshold = shrinkThreshold
        self.defragmentationThreshold = defragmentationThreshold
        self.memoryManager = memoryManager
    }
    
    // MARK: - Memory Pool Protocol
    
    public func acquire() async throws -> UnsafeMutableRawPointer {
        try await acquireBuffer(size: 4096, alignment: defaultAlignment)
    }
    
    public func acquireBuffer(size: Int, alignment: Int? = nil) async throws -> UnsafeMutableRawPointer {
        let startTime = CFAbsoluteTimeGetCurrent()
        let actualAlignment = alignment ?? defaultAlignment
        let alignedSize = (size + actualAlignment - 1) & ~(actualAlignment - 1)
        
        // Track size bucket usage
        sizeBucketUsage[alignedSize, default: 0] += 1
        
        // Check if we need to resize pools
        await checkAndResizePools()
        
        // Try to get from pool
        if let entry = await getFromPool(size: alignedSize) {
            currentBytesInUse += entry.size
            peakBytesUsage = max(peakBytesUsage, currentBytesInUse)
            hitCount += 1
            
            // Track allocation
            let info = AllocationInfo(
                id: UUID(),
                size: entry.size,
                timestamp: Date(),
                alignment: entry.alignment
            )
            allocatedBuffers[entry.pointer] = info
            
            totalAllocationTime += CFAbsoluteTimeGetCurrent() - startTime
            return entry.pointer
        }
        
        // Allocate new buffer
        missCount += 1
        
        // Check memory pressure before allocation
        if currentBytesInUse + alignedSize > totalBytesAllocated {
            await handleMemoryPressure(.warning)
        }
        
        guard let pointer = UnsafeMutableRawPointer.allocate(
            byteCount: alignedSize,
            alignment: actualAlignment
        ) else {
            throw VectorStoreError(
                category: .memoryAllocation,
                code: .allocationFailed,
                message: "Failed to allocate buffer of size \(alignedSize)",
                context: [
                    "size": alignedSize,
                    "reason": "Failed to allocate buffer",
                    "availableMemory": totalBytesAllocated - currentBytesInUse
                ]
            )
        }
        
        // Track allocation
        let info = AllocationInfo(
            id: UUID(),
            size: alignedSize,
            timestamp: Date(),
            alignment: actualAlignment
        )
        allocatedBuffers[pointer] = info
        
        totalBytesAllocated += alignedSize
        currentBytesInUse += alignedSize
        peakBytesUsage = max(peakBytesUsage, currentBytesInUse)
        allocationCount += 1
        
        totalAllocationTime += CFAbsoluteTimeGetCurrent() - startTime
        
        logger.debug("Allocated new buffer: size=\(alignedSize), total=\(self.totalBytesAllocated)")
        
        return pointer
    }
    
    public func release(_ element: UnsafeMutableRawPointer) async {
        guard let info = allocatedBuffers.removeValue(forKey: element) else {
            logger.warning("Attempted to release unknown buffer")
            return
        }
        
        await releaseBuffer(element, size: info.size, alignment: info.alignment)
    }
    
    public func releaseBuffer(_ pointer: UnsafeMutableRawPointer, size: Int, alignment: Int? = nil) async {
        let actualAlignment = alignment ?? defaultAlignment
        let alignedSize = (size + actualAlignment - 1) & ~(actualAlignment - 1)
        
        currentBytesInUse -= alignedSize
        allocatedBuffers.removeValue(forKey: pointer)
        
        let entry = BufferEntry(
            pointer: pointer,
            size: alignedSize,
            alignment: actualAlignment,
            allocationTime: Date(),
            lastAccessTime: Date()
        )
        
        // Check if we should keep this buffer
        if shouldKeepBuffer(size: alignedSize) {
            if availablePools[alignedSize] == nil {
                availablePools[alignedSize] = []
            }
            availablePools[alignedSize]?.append(entry)
        } else {
            // Deallocate immediately
            pointer.deallocate()
            totalBytesAllocated -= alignedSize
        }
        
        // Check for defragmentation need
        let fragmentation = await calculateFragmentation()
        if fragmentation > defragmentationThreshold {
            Task.detached { [weak self] in
                await self?.defragment()
            }
        }
    }
    
    public func preAllocate(count: Int) async {
        // Pre-allocate based on usage patterns
        let topSizes = sizeBucketUsage
            .sorted { $0.value > $1.value }
            .prefix(5)
            .map { $0.key }
        
        let perSize = max(1, count / topSizes.count)
        
        for size in topSizes {
            for _ in 0..<perSize {
                guard let pointer = UnsafeMutableRawPointer.allocate(
                    byteCount: size,
                    alignment: defaultAlignment
                ) else { continue }
                
                let entry = BufferEntry(
                    pointer: pointer,
                    size: size,
                    alignment: defaultAlignment,
                    allocationTime: Date(),
                    lastAccessTime: Date()
                )
                
                if availablePools[size] == nil {
                    availablePools[size] = []
                }
                availablePools[size]?.append(entry)
                totalBytesAllocated += size
            }
        }
        
        logger.info("Pre-allocated \(count) buffers across \(topSizes.count) size buckets")
    }
    
    public func clear() async {
        for (_, entries) in availablePools {
            for entry in entries {
                entry.pointer.deallocate()
                totalBytesAllocated -= entry.size
            }
        }
        availablePools.removeAll()
        allocatedBuffers.removeAll()
        currentBytesInUse = 0
        
        logger.info("Cleared all buffers")
    }
    
    public func statistics() async -> PoolStatistics {
        let fragmentation = await calculateFragmentation()
        let hitRate = Double(hitCount) / Double(hitCount + missCount)
        let avgTime = Double(allocationCount) > 0 ? totalAllocationTime / Double(allocationCount) : 0
        
        return PoolStatistics(
            totalAllocated: totalBytesAllocated,
            currentlyInUse: currentBytesInUse,
            peakUsage: peakBytesUsage,
            hitRate: hitRate,
            averageAllocationTime: avgTime,
            totalMemoryBytes: totalBytesAllocated,
            fragmentationRatio: fragmentation,
            lastDefragmentation: lastDefragmentation
        )
    }
    
    // MARK: - Defragmentation
    
    public func defragment() async -> DefragmentationResult {
        let startTime = Date()
        let fragmentationBefore = await calculateFragmentation()
        
        var bytesReclaimed = 0
        var objectsCompacted = 0
        
        // Remove old unused buffers
        let cutoffTime = Date().addingTimeInterval(-300) // 5 minutes
        
        for (size, entries) in availablePools {
            let (keep, remove) = entries.partitioned { $0.lastAccessTime > cutoffTime }
            
            for entry in remove {
                entry.pointer.deallocate()
                totalBytesAllocated -= entry.size
                bytesReclaimed += entry.size
                objectsCompacted += 1
            }
            
            if keep.isEmpty {
                availablePools.removeValue(forKey: size)
            } else {
                availablePools[size] = keep
            }
        }
        
        // Consolidate fragmented size buckets
        await consolidateSizeBuckets()
        
        let duration = Date().timeIntervalSince(startTime)
        let fragmentationAfter = await calculateFragmentation()
        lastDefragmentation = Date()
        
        logger.info("Defragmentation: reclaimed=\(bytesReclaimed) bytes, compacted=\(objectsCompacted) objects")
        
        return DefragmentationResult(
            bytesReclaimed: bytesReclaimed,
            objectsCompacted: objectsCompacted,
            duration: duration,
            fragmentationBefore: fragmentationBefore,
            fragmentationAfter: fragmentationAfter
        )
    }
    
    // MARK: - Memory Managed Buffer Pool Protocol
    
    public func getStatistics() async -> BufferPoolStatistics {
        let poolCount = availablePools.values.reduce(0) { $0 + $1.count }
        
        // Create size distribution
        var sizeDistribution: [Int: Int] = [:]
        for (size, entries) in availablePools {
            sizeDistribution[size] = entries.count
        }
        
        return BufferPoolStatistics(
            totalBuffers: poolCount + allocatedBuffers.count,
            memoryUsage: currentBytesInUse,
            sizeDistribution: sizeDistribution
        )
    }
    
    public func clearAll() async {
        await clear()
    }
    
    public func getCurrentMemoryUsage() async -> Int {
        return currentBytesInUse
    }
    
    // MARK: - Memory Pressure
    
    public func registerMemoryPressureCallback(_ callback: @escaping (SystemMemoryPressure) -> Void) {
        pressureCallbacks.append(callback)
    }
    
    private func handleMemoryPressure(_ level: SystemMemoryPressure) async {
        logger.warning("Handling memory pressure: \(level.rawValue)")
        
        switch level {
        case .normal:
            return
            
        case .warning:
            // Remove 50% of unused buffers
            for (size, entries) in availablePools {
                let toRemove = entries.count / 2
                for _ in 0..<toRemove {
                    if let entry = availablePools[size]?.popLast() {
                        entry.pointer.deallocate()
                        totalBytesAllocated -= entry.size
                    }
                }
            }
            
        case .critical:
            // Remove all unused buffers
            await clear()
        }
        
        // Notify callbacks
        for callback in pressureCallbacks {
            callback(level)
        }
    }
    
    // MARK: - Automatic Resizing
    
    private func checkAndResizePools() async {
        let now = Date()
        guard now.timeIntervalSince(lastResizeCheck) > resizeCheckInterval else { return }
        lastResizeCheck = now
        
        // Analyze usage patterns
        let totalUsage = sizeBucketUsage.values.reduce(0, +)
        guard totalUsage > 0 else { return }
        
        for (size, usage) in sizeBucketUsage {
            let usageRatio = Double(usage) / Double(totalUsage)
            let currentPoolSize = availablePools[size]?.count ?? 0
            
            if usageRatio > 0.2 && currentPoolSize < maxPoolSize {
                // High usage - grow pool
                let targetSize = min(maxPoolSize, Int(Double(currentPoolSize) * growthFactor))
                let toAdd = targetSize - currentPoolSize
                
                for _ in 0..<toAdd {
                    guard let pointer = UnsafeMutableRawPointer.allocate(
                        byteCount: size,
                        alignment: defaultAlignment
                    ) else { break }
                    
                    let entry = BufferEntry(
                        pointer: pointer,
                        size: size,
                        alignment: defaultAlignment,
                        allocationTime: Date(),
                        lastAccessTime: Date()
                    )
                    
                    if availablePools[size] == nil {
                        availablePools[size] = []
                    }
                    availablePools[size]?.append(entry)
                    totalBytesAllocated += size
                }
                
                logger.debug("Grew pool for size \(size): added \(toAdd) buffers")
                
            } else if usageRatio < shrinkThreshold && currentPoolSize > 10 {
                // Low usage - shrink pool
                let targetSize = max(10, currentPoolSize / 2)
                let toRemove = currentPoolSize - targetSize
                
                for _ in 0..<toRemove {
                    if let entry = availablePools[size]?.popLast() {
                        entry.pointer.deallocate()
                        totalBytesAllocated -= entry.size
                    }
                }
                
                logger.debug("Shrank pool for size \(size): removed \(toRemove) buffers")
            }
        }
        
        // Reset usage counters
        sizeBucketUsage.removeAll()
    }
    
    // MARK: - Helper Methods
    
    private func getFromPool(size: Int) async -> BufferEntry? {
        // Try exact size first
        if var entries = availablePools[size], !entries.isEmpty {
            var entry = entries.removeLast()
            entry.lastAccessTime = Date()
            availablePools[size] = entries
            return entry
        }
        
        // Try larger sizes
        let largerSizes = availablePools.keys.filter { $0 > size }.sorted()
        for largerSize in largerSizes {
            if var entries = availablePools[largerSize], !entries.isEmpty {
                var entry = entries.removeLast()
                entry.lastAccessTime = Date()
                availablePools[largerSize] = entries
                return entry
            }
        }
        
        return nil
    }
    
    private func shouldKeepBuffer(size: Int) -> Bool {
        let poolSize = availablePools[size]?.count ?? 0
        return poolSize < maxPoolSize
    }
    
    private func calculateFragmentation() async -> Double {
        let totalPoolMemory = availablePools.values.reduce(0) { total, entries in
            total + entries.reduce(0) { $0 + $1.size }
        }
        
        guard totalBytesAllocated > 0 else { return 0 }
        return Double(totalPoolMemory) / Double(totalBytesAllocated)
    }
    
    private func consolidateSizeBuckets() async {
        // Identify similar size buckets that can be merged
        let sortedSizes = availablePools.keys.sorted()
        var mergedBuckets: [Int: [BufferEntry]] = [:]
        
        for size in sortedSizes {
            guard let entries = availablePools[size] else { continue }
            
            // Find the best bucket to merge into
            let targetSize = sortedSizes.first { $0 >= size && $0 <= Int(Double(size) * 1.2) } ?? size
            
            if mergedBuckets[targetSize] == nil {
                mergedBuckets[targetSize] = []
            }
            mergedBuckets[targetSize]?.append(contentsOf: entries)
        }
        
        availablePools = mergedBuckets
    }
    
    deinit {
        // Clean up all allocated buffers
        for (_, entries) in availablePools {
            for entry in entries {
                entry.pointer.deallocate()
            }
        }
    }
}

// MARK: - Enhanced Memory Pool Manager

/// Centralized manager for all memory pools with advanced features
public actor MemoryPoolManager {
    // Pool registry
    private var pools: [String: any MemoryPool] = [:]
    private var enhancedBufferPools: [String: EnhancedBufferPool] = [:]
    
    // Memory management integration
    private weak var memoryManager: MemoryManager?
    
    // Configuration
    private let autoDefragmentationEnabled: Bool
    private let defragmentationInterval: TimeInterval
    private var defragmentationTask: Task<Void, Never>?
    
    // Statistics
    private var totalPoolOperations: Int = 0
    private var lastDefragmentation: Date?
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MemoryPoolManager")
    
    public init(
        memoryManager: MemoryManager? = nil,
        autoDefragmentationEnabled: Bool = true,
        defragmentationInterval: TimeInterval = 300 // 5 minutes
    ) {
        self.memoryManager = memoryManager
        self.autoDefragmentationEnabled = autoDefragmentationEnabled
        self.defragmentationInterval = defragmentationInterval
        
        if autoDefragmentationEnabled {
            startAutoDefragmentation()
        }
        
        logger.info("MemoryPoolManager initialized with auto-defragmentation: \(autoDefragmentationEnabled)")
    }
    
    deinit {
        defragmentationTask?.cancel()
    }
    
    // MARK: - Pool Registration
    
    /// Register an enhanced buffer pool
    public func registerEnhancedBufferPool(
        _ pool: EnhancedBufferPool,
        for key: String
    ) async {
        enhancedBufferPools[key] = pool
        pools[key] = pool
        
        // Register with memory manager if available
        if let memoryManager = memoryManager {
            await memoryManager.registerBufferPool(pool)
        }
        
        logger.debug("Registered enhanced buffer pool: \(key)")
    }
    
    /// Create and register a new enhanced buffer pool
    public func createEnhancedBufferPool(
        for key: String,
        maxPoolSize: Int = 100,
        defaultAlignment: Int = 64,
        growthFactor: Double = 1.5,
        shrinkThreshold: Double = 0.25,
        defragmentationThreshold: Double = 0.3
    ) async -> EnhancedBufferPool {
        let pool = EnhancedBufferPool(
            maxPoolSize: maxPoolSize,
            defaultAlignment: defaultAlignment,
            growthFactor: growthFactor,
            shrinkThreshold: shrinkThreshold,
            defragmentationThreshold: defragmentationThreshold,
            memoryManager: memoryManager
        )
        
        await registerEnhancedBufferPool(pool, for: key)
        return pool
    }
    
    /// Get an enhanced buffer pool
    public func getEnhancedBufferPool(for key: String) -> EnhancedBufferPool? {
        return enhancedBufferPools[key]
    }
    
    // MARK: - Memory Pressure Integration
    
    /// Handle system memory pressure
    public func handleMemoryPressure(_ level: SystemMemoryPressure) async {
        logger.warning("Handling memory pressure: \(level.rawValue)")
        
        switch level {
        case .normal:
            return
            
        case .warning:
            // Clear 50% of pools
            await clearPools(percentage: 0.5)
            
        case .critical:
            // Clear all pools and force defragmentation
            await clearAll()
            await defragmentAll()
        }
    }
    
    // MARK: - Defragmentation
    
    /// Defragment all pools
    public func defragmentAll() async -> [String: DefragmentationResult] {
        var results: [String: DefragmentationResult] = [:]
        
        for (key, pool) in pools {
            let result = await pool.defragment()
            results[key] = result
        }
        
        lastDefragmentation = Date()
        
        let totalReclaimed = results.values.reduce(0) { $0 + $1.bytesReclaimed }
        logger.info("Defragmentation complete: reclaimed \(totalReclaimed) bytes across \(results.count) pools")
        
        return results
    }
    
    /// Start automatic defragmentation
    private func startAutoDefragmentation() {
        defragmentationTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(defragmentationInterval * 1_000_000_000))
                
                if !Task.isCancelled {
                    await defragmentAll()
                }
            }
        }
    }
    
    // MARK: - Pool Management
    
    /// Clear a percentage of pools
    public func clearPools(percentage: Double) async {
        let targetPercentage = max(0, min(1, percentage))
        
        for pool in pools.values {
            let stats = await pool.statistics()
            let targetSize = Int(Double(stats.totalAllocated) * (1 - targetPercentage))
            
            if targetSize < stats.currentlyInUse {
                // Can't clear in-use memory, skip
                continue
            }
            
            // Clear excess capacity
            // Note: Individual pools should implement partial clearing
            if targetPercentage >= 0.9 {
                await pool.clear()
            }
        }
    }
    
    /// Clear all pools
    public func clearAll() async {
        for (key, pool) in pools {
            await pool.clear()
            logger.debug("Cleared pool: \(key)")
        }
    }
    
    // MARK: - Statistics
    
    /// Get statistics for all pools
    public func allStatistics() async -> [String: PoolStatistics] {
        var stats: [String: PoolStatistics] = [:]
        
        for (key, pool) in pools {
            stats[key] = await pool.statistics()
        }
        
        return stats
    }
    
    /// Get total memory usage across all pools
    public func totalMemoryUsage() async -> Int {
        var total = 0
        
        for pool in pools.values {
            let stats = await pool.statistics()
            total += stats.currentlyInUse
        }
        
        return total
    }
    
    /// Get comprehensive memory report
    public func memoryReport() async -> MemoryReport {
        let stats = await allStatistics()
        let totalMemory = stats.values.reduce(0) { $0 + $1.totalMemoryBytes }
        let usedMemory = stats.values.reduce(0) { $0 + $1.currentlyInUse }
        let avgHitRate = stats.values.reduce(0.0) { $0 + $1.hitRate } / Double(max(1, stats.count))
        let avgFragmentation = stats.values.reduce(0.0) { $0 + $1.fragmentationRatio } / Double(max(1, stats.count))
        
        return MemoryReport(
            totalPools: pools.count,
            totalMemoryAllocated: totalMemory,
            totalMemoryInUse: usedMemory,
            averageHitRate: avgHitRate,
            averageFragmentation: avgFragmentation,
            lastDefragmentation: lastDefragmentation,
            poolStatistics: stats
        )
    }
}

// MARK: - Supporting Types

/// Comprehensive memory report
public struct MemoryReport: Sendable {
    public let totalPools: Int
    public let totalMemoryAllocated: Int
    public let totalMemoryInUse: Int
    public let averageHitRate: Double
    public let averageFragmentation: Double
    public let lastDefragmentation: Date?
    public let poolStatistics: [String: PoolStatistics]
    
    public var utilizationRate: Double {
        guard totalMemoryAllocated > 0 else { return 0 }
        return Double(totalMemoryInUse) / Double(totalMemoryAllocated)
    }
}


// MARK: - Error Extensions

extension VectorStoreError {
    static func allocationFailed(
        size: Int,
        reason: String,
        availableMemory: Int,
        file: String = #file,
        line: Int = #line,
        function: String = #function
    ) -> VectorStoreError {
        VectorStoreError(
            category: .memoryAllocation,
            code: .allocationFailed,
            message: "Failed to allocate \(size) bytes: \(reason)",
            context: [
                "requestedSize": size,
                "availableMemory": availableMemory,
                "reason": reason
            ],
            file: file,
            line: line,
            function: function
        )
    }
}

// MARK: - Array Extension

extension Array {
    func partitioned(by predicate: (Element) -> Bool) -> (matching: [Element], notMatching: [Element]) {
        var matching: [Element] = []
        var notMatching: [Element] = []
        
        for element in self {
            if predicate(element) {
                matching.append(element)
            } else {
                notMatching.append(element)
            }
        }
        
        return (matching, notMatching)
    }
}
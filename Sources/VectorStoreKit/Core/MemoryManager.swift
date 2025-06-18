// MemoryManager.swift
// VectorStoreKit
//
// Created for VectorStoreKit to handle system memory pressure and manage
// resource constraints across the ML pipeline and buffer systems.

import Foundation
import os.log
import Metal

/// A centralized memory management actor that monitors system memory pressure
/// and coordinates resource cleanup across various subsystems.
///
/// This actor implements intelligent memory pressure handling by:
/// - Monitoring system memory notifications
/// - Coordinating buffer pool and cache cleanup
/// - Reducing batch sizes during high memory pressure
/// - Clearing training history and temporary data
/// - Providing memory usage statistics
///
/// ## Usage Example
/// ```swift
/// let memoryManager = MemoryManager()
/// 
/// // Register subsystems
/// await memoryManager.registerBufferPool(bufferPool)
/// await memoryManager.registerBufferCache(bufferCache)
/// 
/// // Manual memory pressure handling
/// await memoryManager.handleMemoryPressure()
/// 
/// // Get current memory statistics
/// let stats = await memoryManager.getMemoryStatistics()
/// ```
public actor MemoryManager {
    // MARK: - Properties
    
    /// Registered buffer pools that can release memory
    private var bufferPools: [MemoryManagedBufferPool] = []
    
    /// Registered buffer caches that can evict entries
    private var bufferCaches: [MemoryManagedCache] = []
    
    /// Batch size managers that can reduce batch sizes
    private var batchSizeManagers: [BatchSizeManager] = []
    
    /// Training history managers that can clear history
    private var trainingHistoryManagers: [TrainingHistoryManager] = []
    
    /// Memory pressure aware components
    private var pressureAwareComponents: [MemoryPressureAware] = []
    
    /// Distance computation components with memory awareness
    private var distanceComputeComponents: [MemoryPressureAwareDistanceComputation] = []
    
    /// Memory warning observer for macOS
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    
    /// Statistics tracking
    private var stats = MemoryManagerStatistics()
    
    /// Logger for debugging and monitoring
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MemoryManager")
    
    /// Current memory pressure level
    private var currentPressureLevel: SystemMemoryPressure = .normal
    
    // MARK: - Initialization
    
    /// Creates a new memory manager and starts monitoring system memory
    public init() {
        Task {
            await setupMemoryPressureMonitoring()
            logger.info("MemoryManager initialized")
        }
    }
    
    deinit {
        memoryPressureSource?.cancel()
    }
    
    // MARK: - Memory Pressure Monitoring
    
    /// Sets up system memory pressure monitoring
    private func setupMemoryPressureMonitoring() {
        // Create dispatch source for memory pressure events
        memoryPressureSource = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: DispatchQueue.global(qos: .userInitiated)
        )
        
        memoryPressureSource?.setEventHandler { [weak self] in
            Task { [weak self] in
                guard let self = self else { return }
                
                let pressureLevel: SystemMemoryPressure
                if await self.memoryPressureSource?.data.contains(.critical) == true {
                    pressureLevel = .critical
                } else if await self.memoryPressureSource?.data.contains(.warning) == true {
                    pressureLevel = .warning
                } else {
                    pressureLevel = .normal
                }
                
                await self.handleMemoryPressure(level: pressureLevel)
            }
        }
        
        memoryPressureSource?.resume()
        logger.info("Memory pressure monitoring started")
    }
    
    // MARK: - Subsystem Registration
    
    /// Registers a buffer pool for memory management
    /// - Parameter pool: The buffer pool to register
    public func registerBufferPool(_ pool: MemoryManagedBufferPool) {
        bufferPools.append(pool)
        logger.debug("Registered buffer pool, total pools: \(self.bufferPools.count)")
    }
    
    /// Registers a buffer cache for memory management
    /// - Parameter cache: The buffer cache to register
    public func registerBufferCache(_ cache: MemoryManagedCache) {
        bufferCaches.append(cache)
        logger.debug("Registered buffer cache, total caches: \(self.bufferCaches.count)")
    }
    
    /// Registers a batch size manager for memory management
    /// - Parameter manager: The batch size manager to register
    public func registerBatchSizeManager(_ manager: BatchSizeManager) {
        batchSizeManagers.append(manager)
        logger.debug("Registered batch size manager, total managers: \(self.batchSizeManagers.count)")
    }
    
    /// Registers a training history manager for memory management
    /// - Parameter manager: The training history manager to register
    public func registerTrainingHistoryManager(_ manager: TrainingHistoryManager) {
        trainingHistoryManagers.append(manager)
        logger.debug("Registered training history manager, total managers: \(self.trainingHistoryManagers.count)")
    }
    
    /// Registers a memory pressure aware component
    /// - Parameter component: The component to register
    public func registerMemoryPressureAwareComponent(_ component: MemoryPressureAware) {
        pressureAwareComponents.append(component)
        logger.debug("Registered memory pressure aware component, total components: \(self.pressureAwareComponents.count)")
    }
    
    /// Registers a distance computation component with memory awareness
    /// - Parameter component: The distance computation component to register
    public func registerDistanceComputeComponent(_ component: MemoryPressureAwareDistanceComputation) {
        distanceComputeComponents.append(component)
        pressureAwareComponents.append(component) // Also add to general components
        logger.debug("Registered distance compute component, total components: \(self.distanceComputeComponents.count)")
    }
    
    // MARK: - Memory Pressure Handling
    
    /// Handles memory pressure by coordinating cleanup across all subsystems
    /// - Parameter level: The memory pressure level (defaults to current level)
    public func handleMemoryPressure(level: SystemMemoryPressure? = nil) async {
        let pressureLevel = level ?? currentPressureLevel
        currentPressureLevel = pressureLevel
        
        stats.pressureEventCount += 1
        let startTime = Date()
        
        logger.warning("Handling memory pressure: \(pressureLevel.rawValue)")
        
        switch pressureLevel {
        case .normal:
            // Check if we're recovering from pressure
            if currentPressureLevel != .normal {
                await handleMemoryRecovery()
            }
            return
            
        case .warning:
            await handleWarningPressure()
            
        case .critical:
            await handleCriticalPressure()
        }
        
        let duration = Date().timeIntervalSince(startTime)
        stats.totalHandlingTime += duration
        stats.lastPressureHandled = Date()
        
        logger.info("Memory pressure handled in \(duration, format: .fixed(precision: 2))s")
    }
    
    /// Handles warning-level memory pressure
    private func handleWarningPressure() async {
        logger.info("Handling warning-level memory pressure")
        
        // 1. Clear buffer pools (keep 50% of buffers)
        await withTaskGroup(of: Int.self) { group in
            for pool in bufferPools {
                group.addTask {
                    let poolStats = await pool.getStatistics()
                    if poolStats.totalBuffers > 10 {
                        await pool.clearAll()
                        return poolStats.totalBuffers
                    }
                    return 0
                }
            }
            
            for await buffersReleased in group {
                stats.buffersReleased += buffersReleased
            }
        }
        
        // 2. Evict from caches (reduce to 75% capacity)
        await withTaskGroup(of: Int.self) { group in
            for cache in bufferCaches {
                group.addTask {
                    let currentMemory = await cache.getCurrentMemoryUsage()
                    let targetMemory = Int(Double(currentMemory) * 0.75)
                    await cache.evictLRU(targetSize: targetMemory)
                    let statsAfter = await cache.getStatistics()
                    return statsAfter.evictionCount
                }
            }
            
            for await evicted in group {
                stats.cacheEntriesEvicted += evicted
            }
        }
        
        // 3. Reduce batch sizes by 25%
        for manager in batchSizeManagers {
            await manager.reduceBatchSize(by: 0.25)
            stats.batchSizeReductions += 1
        }
        
        // 4. Notify all memory pressure aware components
        await withTaskGroup(of: Void.self) { group in
            for component in pressureAwareComponents {
                group.addTask {
                    await component.handleMemoryPressure(.warning)
                }
            }
            
            for await _ in group {
                // Component handled pressure
            }
        }
        
        // 5. Reduce distance computation batch sizes specifically
        for component in distanceComputeComponents {
            await component.reduceBatchSize(by: 0.25)
        }
    }
    
    /// Handles critical-level memory pressure
    private func handleCriticalPressure() async {
        logger.warning("Handling critical-level memory pressure")
        
        // 1. Clear all buffer pools
        await withTaskGroup(of: Int.self) { group in
            for pool in bufferPools {
                group.addTask {
                    let poolStats = await pool.getStatistics()
                    await pool.clearAll()
                    return poolStats.totalBuffers
                }
            }
            
            for await buffersReleased in group {
                stats.buffersReleased += buffersReleased
            }
        }
        
        // 2. Aggressively evict from caches (reduce to 50% capacity)
        await withTaskGroup(of: Int.self) { group in
            for cache in bufferCaches {
                group.addTask {
                    let currentMemory = await cache.getCurrentMemoryUsage()
                    let targetMemory = Int(Double(currentMemory) * 0.5)
                    await cache.evictLRU(targetSize: targetMemory)
                    let cacheStats = await cache.getStatistics()
                    return cacheStats.evictionCount
                }
            }
            
            for await evicted in group {
                stats.cacheEntriesEvicted += evicted
            }
        }
        
        // 3. Reduce batch sizes by 50%
        for manager in batchSizeManagers {
            await manager.reduceBatchSize(by: 0.5)
            stats.batchSizeReductions += 1
        }
        
        // 4. Clear all training history
        for manager in trainingHistoryManagers {
            await manager.clearHistory()
            stats.trainingHistoriesCleared += 1
        }
        
        // 5. Notify all memory pressure aware components of critical pressure
        await withTaskGroup(of: Void.self) { group in
            for component in pressureAwareComponents {
                group.addTask {
                    await component.handleMemoryPressure(.critical)
                }
            }
            
            for await _ in group {
                // Component handled critical pressure
            }
        }
        
        // 6. Aggressively reduce distance computation batch sizes
        for component in distanceComputeComponents {
            await component.reduceBatchSize(by: 0.75) // More aggressive reduction
        }
        
        // 7. Force garbage collection hint
        // Note: Swift doesn't have explicit GC, but we can hint to release autoreleased objects
        autoreleasepool { }
    }
    
    /// Handles recovery when memory pressure returns to normal
    private func handleMemoryRecovery() async {
        logger.info("Handling memory pressure recovery")
        
        // 1. Reset batch sizes to defaults
        for manager in batchSizeManagers {
            await manager.resetBatchSize()
        }
        
        // 2. Reset distance computation batch sizes
        for component in distanceComputeComponents {
            await component.resetBatchSize()
        }
        
        // 3. Notify all components of normal pressure
        await withTaskGroup(of: Void.self) { group in
            for component in pressureAwareComponents {
                group.addTask {
                    await component.handleMemoryPressure(.normal)
                }
            }
            
            for await _ in group {
                // Component handled recovery
            }
        }
        
        logger.info("Memory pressure recovery completed")
    }
    
    // MARK: - Memory Statistics
    
    /// Returns current memory usage statistics
    public func getMemoryStatistics() async -> MemoryManagerStatistics {
        var stats = self.stats
        
        // Aggregate current memory usage from all subsystems
        var totalMemoryUsed: Int = 0
        var totalMemoryLimit: Int = 0
        
        for cache in bufferCaches {
            let memoryUsage = await cache.getCurrentMemoryUsage()
            totalMemoryUsed += memoryUsage
            // Since we don't have memory limit in the protocol, we'll estimate based on usage
            totalMemoryLimit += memoryUsage * 2
        }
        
        // Add memory usage from all pressure-aware components
        for component in pressureAwareComponents {
            let memoryUsage = await component.getCurrentMemoryUsage()
            totalMemoryUsed += memoryUsage
        }
        
        stats.currentMemoryUsage = totalMemoryUsed
        stats.memoryLimit = totalMemoryLimit
        stats.currentPressureLevel = currentPressureLevel
        
        return stats
    }
    
    /// Get comprehensive component statistics
    public func getComponentStatistics() async -> ComponentMemoryStatistics {
        var componentStats: [MemoryComponentStatistics] = []
        
        // Collect statistics from all pressure-aware components
        for component in pressureAwareComponents {
            let stats = await component.getMemoryStatistics()
            componentStats.append(stats)
        }
        
        return ComponentMemoryStatistics(
            totalComponents: pressureAwareComponents.count,
            componentStats: componentStats,
            totalMemoryUsage: componentStats.reduce(0) { $0 + $1.currentMemoryUsage },
            totalPressureEvents: componentStats.reduce(0) { $0 + $1.pressureEventCount }
        )
    }
    
    /// Resets memory management statistics
    public func resetStatistics() {
        stats = MemoryManagerStatistics()
    }
    
    // MARK: - Manual Memory Management
    
    /// Manually triggers memory cleanup at specified level
    /// - Parameter level: The cleanup level to apply
    public func triggerCleanup(level: CleanupLevel) async {
        logger.info("Manual cleanup triggered: \(level.rawValue)")
        
        switch level {
        case .light:
            // Just clear empty pools
            for pool in bufferPools {
                await pool.clearAll()
            }
            
        case .moderate:
            await handleWarningPressure()
            
        case .aggressive:
            await handleCriticalPressure()
        }
    }
}

// MARK: - Supporting Types

/// System memory pressure levels
public enum SystemMemoryPressure: String, Sendable {
    case normal
    case warning
    case critical
}

/// Manual cleanup levels
public enum CleanupLevel: String, Sendable {
    case light
    case moderate
    case aggressive
}

/// Statistics for memory management
public struct MemoryManagerStatistics: Sendable {
    /// Number of memory pressure events handled
    public var pressureEventCount: Int = 0
    
    /// Total time spent handling memory pressure
    public var totalHandlingTime: TimeInterval = 0
    
    /// Number of buffers released from pools
    public var buffersReleased: Int = 0
    
    /// Number of cache entries evicted
    public var cacheEntriesEvicted: Int = 0
    
    /// Number of batch size reductions
    public var batchSizeReductions: Int = 0
    
    /// Number of training histories cleared
    public var trainingHistoriesCleared: Int = 0
    
    /// Current total memory usage
    public var currentMemoryUsage: Int = 0
    
    /// Total memory limit
    public var memoryLimit: Int = 0
    
    /// Current pressure level
    public var currentPressureLevel: SystemMemoryPressure = .normal
    
    /// Last time pressure was handled
    public var lastPressureHandled: Date?
    
    /// Average handling time
    public var averageHandlingTime: TimeInterval {
        guard pressureEventCount > 0 else { return 0 }
        return totalHandlingTime / Double(pressureEventCount)
    }
    
    /// Memory utilization percentage
    public var memoryUtilization: Double {
        guard memoryLimit > 0 else { return 0 }
        return Double(currentMemoryUsage) / Double(memoryLimit) * 100
    }
}

/// Statistics for all memory pressure aware components
public struct ComponentMemoryStatistics: Sendable {
    public let totalComponents: Int
    public let componentStats: [MemoryComponentStatistics]
    public let totalMemoryUsage: Int
    public let totalPressureEvents: Int
    
    public var averageMemoryPerComponent: Int {
        guard totalComponents > 0 else { return 0 }
        return totalMemoryUsage / totalComponents
    }
    
    public var averagePressureEventsPerComponent: Double {
        guard totalComponents > 0 else { return 0 }
        return Double(totalPressureEvents) / Double(totalComponents)
    }
}



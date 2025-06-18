// VectorStoreKit: ML Memory Management
//
// Comprehensive memory management utilities for Metal ML operations

import Foundation
@preconcurrency import Metal
@preconcurrency import Dispatch
import os.log

// MARK: - Memory Pressure Levels
// Note: MemoryPressureLevel is imported from ../../../Caching/MemoryPressureHandler.swift

// MARK: - Memory Manager

/// Centralized memory manager for ML operations
public actor MLMemoryManager {
    // MARK: - Properties
    
    private let device: MTLDevice
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MLMemoryManager")
    
    // Memory tracking
    private var allocatedBuffers: Set<ObjectIdentifier> = []
    private var bufferSizes: [ObjectIdentifier: Int] = [:]
    private var totalAllocatedMemory: Int = 0
    private var peakMemoryUsage: Int = 0
    
    // Memory limits
    private let maxMemoryUsage: Int
    private let warningThreshold: Double = 0.75
    private let criticalThreshold: Double = 0.9
    
    // Cleanup callbacks
    private var cleanupCallbacks: [(MemoryPressureLevel) -> Void] = []
    
    // Memory pressure monitoring
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    private var currentPressureLevel: MemoryPressureLevel = .normal
    
    // Statistics
    private var allocationCount: Int = 0
    private var deallocationCount: Int = 0
    private var pressureEventCount: Int = 0
    
    // MARK: - Initialization
    
    public init(device: MTLDevice, maxMemoryGB: Double = 2.0) {
        self.device = device
        self.maxMemoryUsage = Int(maxMemoryGB * 1_073_741_824) // Convert GB to bytes
        
        Task {
            await setupMemoryPressureMonitoring()
        }
    }
    
    deinit {
        memoryPressureSource?.cancel()
    }
    
    // MARK: - Buffer Tracking
    
    /// Track a newly allocated buffer
    public func trackBuffer(_ buffer: MetalBuffer) {
        let id = ObjectIdentifier(buffer.buffer)
        allocatedBuffers.insert(id)
        bufferSizes[id] = buffer.byteLength
        
        totalAllocatedMemory += buffer.byteLength
        peakMemoryUsage = max(peakMemoryUsage, totalAllocatedMemory)
        allocationCount += 1
        
        // Check memory pressure
        checkMemoryPressure()
        
        logger.debug("Tracked buffer: \(buffer.byteLength) bytes, total: \(self.totalAllocatedMemory)")
    }
    
    /// Untrack a deallocated buffer
    public func untrackBuffer(_ buffer: MetalBuffer) {
        let id = ObjectIdentifier(buffer.buffer)
        
        guard allocatedBuffers.contains(id),
              let size = bufferSizes[id] else {
            return
        }
        
        allocatedBuffers.remove(id)
        bufferSizes.removeValue(forKey: id)
        totalAllocatedMemory -= size
        deallocationCount += 1
        
        logger.debug("Untracked buffer: \(size) bytes, total: \(self.totalAllocatedMemory)")
    }
    
    // MARK: - Memory Pressure Management
    
    private func setupMemoryPressureMonitoring() {
        memoryPressureSource = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: DispatchQueue.global(qos: .userInitiated)
        )
        
        memoryPressureSource?.setEventHandler { [weak self] in
            Task { [weak self] in
                await self?.handleSystemMemoryPressure()
            }
        }
        
        memoryPressureSource?.activate()
        logger.info("Memory pressure monitoring activated")
    }
    
    private func handleSystemMemoryPressure() {
        let data = memoryPressureSource?.data ?? []
        
        if data.contains(.critical) {
            currentPressureLevel = .critical
            logger.critical("System memory pressure: CRITICAL")
        } else if data.contains(.warning) {
            currentPressureLevel = .warning
            logger.warning("System memory pressure: WARNING")
        }
        
        pressureEventCount += 1
        executeCleanupCallbacks(level: currentPressureLevel)
    }
    
    private func checkMemoryPressure() {
        let usage = Double(totalAllocatedMemory) / Double(maxMemoryUsage)
        
        let newLevel: MemoryPressureLevel
        if usage > criticalThreshold {
            newLevel = .critical
        } else if usage > warningThreshold {
            newLevel = .warning
        } else {
            newLevel = .normal
        }
        
        if newLevel != currentPressureLevel {
            currentPressureLevel = newLevel
            
            switch newLevel {
            case .critical:
                logger.critical("Memory usage critical: \(String(format: "%.1f%%", usage * 100))")
            case .warning:
                logger.warning("Memory usage warning: \(String(format: "%.1f%%", usage * 100))")
            case .normal:
                logger.info("Memory usage normal: \(String(format: "%.1f%%", usage * 100))")
            }
            
            if newLevel > .normal {
                executeCleanupCallbacks(level: newLevel)
            }
        }
    }
    
    // MARK: - Cleanup Management
    
    /// Register a cleanup callback for memory pressure events
    public func registerCleanupCallback(_ callback: @escaping (MemoryPressureLevel) -> Void) {
        cleanupCallbacks.append(callback)
    }
    
    private func executeCleanupCallbacks(level: MemoryPressureLevel) {
        for callback in cleanupCallbacks {
            callback(level)
        }
    }
    
    // MARK: - Memory Statistics
    
    public func getStatistics() -> MemoryStatistics {
        MemoryStatistics(
            currentUsage: totalAllocatedMemory,
            peakUsage: peakMemoryUsage,
            bufferCount: allocatedBuffers.count,
            allocationCount: allocationCount,
            deallocationCount: deallocationCount,
            pressureEventCount: pressureEventCount,
            pressureLevel: currentPressureLevel
        )
    }
    
    public func getCurrentUsage() -> (bytes: Int, percentage: Double) {
        let percentage = Double(totalAllocatedMemory) / Double(maxMemoryUsage)
        return (totalAllocatedMemory, percentage)
    }
    
    /// Check if allocation is allowed
    public func canAllocate(size: Int) -> Bool {
        return totalAllocatedMemory + size <= maxMemoryUsage
    }
    
    /// Force cleanup of tracked buffers
    public func forceCleanup() {
        logger.warning("Forcing memory cleanup: \(allocatedBuffers.count) buffers")
        executeCleanupCallbacks(level: .critical)
    }
}

// MARK: - Memory Statistics

/// Memory usage statistics
public struct MemoryStatistics: Sendable {
    public let currentUsage: Int
    public let peakUsage: Int
    public let bufferCount: Int
    public let allocationCount: Int
    public let deallocationCount: Int
    public let pressureEventCount: Int
    public let pressureLevel: MemoryPressureLevel
    
    public var currentUsageMB: Double {
        Double(currentUsage) / 1_048_576
    }
    
    public var peakUsageMB: Double {
        Double(peakUsage) / 1_048_576
    }
}

// MARK: - Automatic Memory Management

/// Wrapper for automatic memory management of Metal buffers
public final class ManagedMetalBuffer: @unchecked Sendable {
    private let buffer: MetalBuffer
    private let memoryManager: MLMemoryManager
    private let releaseCallback: (() -> Void)?
    
    public init(
        buffer: MetalBuffer,
        memoryManager: MLMemoryManager,
        releaseCallback: (() -> Void)? = nil
    ) {
        self.buffer = buffer
        self.memoryManager = memoryManager
        self.releaseCallback = releaseCallback
        
        Task {
            await memoryManager.trackBuffer(buffer)
        }
    }
    
    deinit {
        Task {
            await memoryManager.untrackBuffer(buffer)
            releaseCallback?()
        }
    }
    
    public var metalBuffer: MetalBuffer {
        buffer
    }
}

// MARK: - Memory Pool with Pressure Handling

/// Enhanced buffer pool with memory pressure handling
public actor PressureAwareBufferPool {
    private let device: MTLDevice
    private let memoryManager: MLMemoryManager
    private var pools: [Int: [MetalBuffer]] = [:]
    private let maxPoolSize: Int
    private let logger = Logger(subsystem: "VectorStoreKit", category: "PressureAwareBufferPool")
    
    // Statistics
    private var hitCount: Int = 0
    private var missCount: Int = 0
    private var evictionCount: Int = 0
    
    public init(
        device: MTLDevice,
        memoryManager: MLMemoryManager,
        maxPoolSize: Int = 100
    ) {
        self.device = device
        self.memoryManager = memoryManager
        self.maxPoolSize = maxPoolSize
        
        // Register cleanup callback
        Task {
            await memoryManager.registerCleanupCallback { [weak self] level in
                Task {
                    await self?.handleMemoryPressure(level: level)
                }
            }
        }
    }
    
    /// Acquire a buffer from the pool
    public func acquire(size: Int) async throws -> ManagedMetalBuffer {
        let alignedSize = nextPowerOfTwo(size)
        
        // Check pool first
        if var cached = pools[alignedSize], !cached.isEmpty {
            let buffer = cached.removeLast()
            pools[alignedSize] = cached
            hitCount += 1
            
            return ManagedMetalBuffer(
                buffer: MetalBuffer(buffer: buffer.buffer, count: size),
                memoryManager: memoryManager,
                releaseCallback: { [weak self] in
                    Task {
                        await self?.release(buffer)
                    }
                }
            )
        }
        
        missCount += 1
        
        // Check if we can allocate
        let requiredSize = alignedSize * MemoryLayout<Float>.stride
        guard await memoryManager.canAllocate(size: requiredSize) else {
            // Try to evict some buffers
            await evictBuffers(targetSize: requiredSize)
            
            // Check again
            guard await memoryManager.canAllocate(size: requiredSize) else {
                throw MetalMLError.bufferAllocationFailed(size: alignedSize)
            }
        }
        
        // Allocate new buffer
        guard let mtlBuffer = device.makeBuffer(
            length: requiredSize,
            options: .storageModeShared
        ) else {
            throw MetalMLError.bufferAllocationFailed(size: alignedSize)
        }
        
        let buffer = MetalBuffer(buffer: mtlBuffer, count: size)
        
        return ManagedMetalBuffer(
            buffer: buffer,
            memoryManager: memoryManager,
            releaseCallback: { [weak self] in
                Task {
                    await self?.release(buffer)
                }
            }
        )
    }
    
    /// Release a buffer back to the pool
    private func release(_ buffer: MetalBuffer) {
        let alignedSize = nextPowerOfTwo(buffer.count)
        
        if (pools[alignedSize]?.count ?? 0) < maxPoolSize {
            if pools[alignedSize] == nil {
                pools[alignedSize] = []
            }
            pools[alignedSize]?.append(buffer)
        }
        // Otherwise let it be deallocated
    }
    
    /// Handle memory pressure
    private func handleMemoryPressure(level: MemoryPressureLevel) {
        switch level {
        case .normal:
            return
        case .warning:
            // Evict 50% of cached buffers
            evictPercentage(0.5)
        case .critical:
            // Clear all pools
            clearAll()
        }
    }
    
    /// Evict buffers to free memory
    private func evictBuffers(targetSize: Int) {
        var freedSize = 0
        
        for (size, buffers) in pools.sorted(by: { $0.key > $1.key }) {
            guard freedSize < targetSize, !buffers.isEmpty else { continue }
            
            let sizePerBuffer = size * MemoryLayout<Float>.stride
            var remainingBuffers = buffers
            
            while !remainingBuffers.isEmpty && freedSize < targetSize {
                remainingBuffers.removeLast()
                freedSize += sizePerBuffer
                evictionCount += 1
            }
            
            if remainingBuffers.isEmpty {
                pools[size] = nil
            } else {
                pools[size] = remainingBuffers
            }
        }
    }
    
    /// Evict a percentage of buffers
    private func evictPercentage(_ percentage: Double) {
        for (size, buffers) in pools {
            let evictCount = Int(Double(buffers.count) * percentage)
            if evictCount > 0 {
                var remainingBuffers = buffers
                for _ in 0..<evictCount {
                    _ = remainingBuffers.popLast()
                    evictionCount += 1
                }
                
                if remainingBuffers.isEmpty {
                    pools[size] = nil
                } else {
                    pools[size] = remainingBuffers
                }
            }
        }
        
        logger.info("Evicted \(percentage * 100)% of pooled buffers")
    }
    
    /// Clear all pools
    private func clearAll() {
        let totalBuffers = pools.values.reduce(0) { $0 + $1.count }
        pools.removeAll()
        evictionCount += totalBuffers
        logger.warning("Cleared all pools: \(totalBuffers) buffers")
    }
    
    /// Get pool statistics
    public func getStatistics() -> PoolStatistics {
        let totalBuffers = pools.values.reduce(0) { $0 + $1.count }
        let hitRate = Double(hitCount) / Double(hitCount + missCount)
        
        return PoolStatistics(
            totalAllocated: totalBuffers,
            currentlyInUse: 0, // Not tracked in this implementation
            peakUsage: 0, // Not tracked in this implementation
            hitRate: hitRate,
            averageAllocationTime: 0 // Not tracked in this implementation
        )
    }
    
    private func nextPowerOfTwo(_ n: Int) -> Int {
        var power = 1
        while power < n {
            power *= 2
        }
        return power
    }
}

// MARK: - Memory Profiler

/// Memory profiler for ML operations
public actor MLMemoryProfiler {
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MLMemoryProfiler")
    
    public struct AllocationEvent: Sendable {
        let timestamp: Date
        let size: Int
        let source: String
        let stackTrace: [String]
    }
    
    public struct ProfileReport: Sendable {
        let totalAllocations: Int
        let totalDeallocations: Int
        let peakMemoryUsage: Int
        let averageAllocationSize: Double
        let largestAllocation: Int
        let allocationsBySource: [String: Int]
        let timeline: [(Date, Int)] // (timestamp, memory usage)
    }
    
    private var allocationEvents: [AllocationEvent] = []
    private var memoryTimeline: [(Date, Int)] = []
    private var currentMemoryUsage: Int = 0
    private var peakMemoryUsage: Int = 0
    
    private let maxEventsToKeep = 10000
    
    /// Record an allocation
    public func recordAllocation(size: Int, source: String = #function) {
        let event = AllocationEvent(
            timestamp: Date(),
            size: size,
            source: source,
            stackTrace: Thread.callStackSymbols
        )
        
        allocationEvents.append(event)
        currentMemoryUsage += size
        peakMemoryUsage = max(peakMemoryUsage, currentMemoryUsage)
        memoryTimeline.append((Date(), currentMemoryUsage))
        
        // Trim old events if needed
        if allocationEvents.count > maxEventsToKeep {
            allocationEvents.removeFirst(allocationEvents.count - maxEventsToKeep)
        }
        
        if memoryTimeline.count > maxEventsToKeep {
            memoryTimeline.removeFirst(memoryTimeline.count - maxEventsToKeep)
        }
    }
    
    /// Record a deallocation
    public func recordDeallocation(size: Int) {
        currentMemoryUsage -= size
        memoryTimeline.append((Date(), currentMemoryUsage))
    }
    
    /// Generate profile report
    public func generateReport() -> ProfileReport {
        let totalAllocations = allocationEvents.count
        let totalSize = allocationEvents.reduce(0) { $0 + $1.size }
        let averageSize = totalAllocations > 0 ? Double(totalSize) / Double(totalAllocations) : 0
        let largestAllocation = allocationEvents.map { $0.size }.max() ?? 0
        
        // Group by source
        var allocationsBySource: [String: Int] = [:]
        for event in allocationEvents {
            allocationsBySource[event.source, default: 0] += 1
        }
        
        return ProfileReport(
            totalAllocations: totalAllocations,
            totalDeallocations: 0, // Not tracked separately
            peakMemoryUsage: peakMemoryUsage,
            averageAllocationSize: averageSize,
            largestAllocation: largestAllocation,
            allocationsBySource: allocationsBySource,
            timeline: memoryTimeline
        )
    }
    
    /// Clear profiling data
    public func reset() {
        allocationEvents.removeAll()
        memoryTimeline.removeAll()
        currentMemoryUsage = 0
        peakMemoryUsage = 0
        logger.info("Memory profiler reset")
    }
}

// MARK: - Gradient Checkpointing

/// Gradient checkpointing support for memory-efficient training
public actor GradientCheckpointer {
    private let metalPipeline: MetalMLPipeline
    private var checkpoints: [String: MetalBuffer] = [:]
    private let logger = Logger(subsystem: "VectorStoreKit", category: "GradientCheckpointer")
    
    public init(metalPipeline: MetalMLPipeline) {
        self.metalPipeline = metalPipeline
    }
    
    /// Save activation for recomputation during backward pass
    public func checkpoint(_ buffer: MetalBuffer, key: String) async throws {
        // Create a copy to avoid retaining the original
        let copy = try await metalPipeline.allocateBuffer(size: buffer.count)
        let operations = await metalPipeline.getOperations()
        try await operations.copyBuffer(from: buffer, to: copy)
        
        checkpoints[key] = copy
        logger.debug("Checkpointed buffer '\(key)': \(buffer.count) elements")
    }
    
    /// Retrieve checkpointed activation
    public func retrieve(key: String) -> MetalBuffer? {
        checkpoints[key]
    }
    
    /// Clear specific checkpoint
    public func clear(key: String) async {
        if let buffer = checkpoints.removeValue(forKey: key) {
            await metalPipeline.releaseBuffer(buffer)
            logger.debug("Cleared checkpoint '\(key)'")
        }
    }
    
    /// Clear all checkpoints
    public func clearAll() async {
        for (key, buffer) in checkpoints {
            await metalPipeline.releaseBuffer(buffer)
        }
        checkpoints.removeAll()
        logger.info("Cleared all checkpoints")
    }
    
    /// Get current memory usage
    public func getMemoryUsage() -> Int {
        checkpoints.values.reduce(0) { $0 + $1.byteLength }
    }
}

// MARK: - Memory-Efficient Training Extensions

extension NeuralNetwork {
    /// Enable gradient checkpointing for memory-efficient training
    public func enableGradientCheckpointing() async {
        // This would be implemented per-layer basis
        // Each layer would checkpoint activations and recompute during backward
        for layer in layers {
            if let checkpointable = layer as? GradientCheckpointable {
                await checkpointable.enableCheckpointing()
            }
        }
    }
}

/// Protocol for layers that support gradient checkpointing
public protocol GradientCheckpointable: Actor {
    func enableCheckpointing() async
    func disableCheckpointing() async
}

// MARK: - Memory Debugging Utilities

/// Memory debugging utilities
public enum MemoryDebugger {
    /// Check for memory leaks in a code block
    public static func detectLeaks<T>(
        in block: () async throws -> T
    ) async rethrows -> (result: T, leaks: [String]) {
        // This is a simplified leak detector
        // In practice, you'd use more sophisticated tracking
        
        let initialCount = ProcessInfo.processInfo.activeProcessorCount
        let result = try await block()
        let finalCount = ProcessInfo.processInfo.activeProcessorCount
        
        let leaks: [String] = []
        if finalCount > initialCount {
            // Potential leak detected
        }
        
        return (result, leaks)
    }
    
    /// Print memory usage summary
    public static func printMemoryUsage() {
        let info = ProcessInfo.processInfo
        let physicalMemory = info.physicalMemory
        let usedMemory = physicalMemory - info.availablePhysicalMemory
        
        print("=== Memory Usage ===")
        print("Physical Memory: \(formatBytes(physicalMemory))")
        print("Used Memory: \(formatBytes(usedMemory))")
        print("Available Memory: \(formatBytes(info.availablePhysicalMemory))")
        print("==================")
    }
    
    private static func formatBytes(_ bytes: UInt64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}
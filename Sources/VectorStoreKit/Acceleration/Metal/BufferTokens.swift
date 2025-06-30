// BufferTokens.swift
// VectorStoreKit
//
// Token types for safe buffer management
// Provides ownership-based memory safety for Metal buffers

import Foundation
import Metal

// MARK: - Buffer Token for Safe Buffer Management

/// Token representing ownership of a Metal buffer
public final class BufferToken: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let pool: BufferPoolType
    public let size: Int
    private weak var manager: UnifiedBufferManager?
    
    init(buffer: MTLBuffer, pool: BufferPoolType, size: Int, manager: UnifiedBufferManager) {
        self.buffer = buffer
        self.pool = pool
        self.size = size
        self.manager = manager
    }
    
    /// Release buffer back to appropriate pool
    public func release() async {
        guard let manager = manager else { return }
        await manager.releaseBuffer(self)
    }
    
    /// Get the underlying Metal buffer for direct use
    public var metalBuffer: MTLBuffer {
        return buffer
    }
    
    /// Check if the buffer is still valid
    public var isValid: Bool {
        return manager != nil
    }
}

// MARK: - Batch Buffer Token

/// Token for batch buffer operations
public struct BatchBufferToken: @unchecked Sendable {
    public let buffer: MTLBuffer?
    public let individualBuffers: [BufferToken]?
    public let count: Int
    private weak var manager: UnifiedBufferManager?
    
    init(buffer: MTLBuffer, count: Int, manager: UnifiedBufferManager) {
        self.buffer = buffer
        self.individualBuffers = nil
        self.count = count
        self.manager = manager
    }
    
    init(individualBuffers: [BufferToken], manager: UnifiedBufferManager) {
        self.buffer = nil
        self.individualBuffers = individualBuffers
        self.count = individualBuffers.count
        self.manager = manager
    }
    
    /// Release all buffers in the batch
    public func releaseAll() async {
        if let individualBuffers = individualBuffers {
            for token in individualBuffers {
                await token.release()
            }
        } else if let buffer = buffer, let manager = manager {
            await manager.releaseBatchBuffer(buffer)
        }
    }
    
    /// Get buffer for specific index (for batch operations)
    public func buffer(at index: Int) -> MTLBuffer? {
        guard index < count else { return nil }
        
        if let buffer = buffer {
            // For single large buffer, calculate offset
            let offset = index * 2048 // Vector512 size
            return buffer // Note: Actual offset handling would be done by caller
        } else if let individualBuffers = individualBuffers, index < individualBuffers.count {
            return individualBuffers[index].buffer
        }
        
        return nil
    }
    
    /// Get all individual buffers (creates array for single buffer case)
    public func getAllBuffers() -> [MTLBuffer] {
        if let individualBuffers = individualBuffers {
            return individualBuffers.map { $0.buffer }
        } else if let buffer = buffer {
            return [buffer] // Single large buffer
        }
        return []
    }
}

// MARK: - Command Buffer Token

/// Token for command buffer operations
public struct CommandBufferToken: @unchecked Sendable {
    public let commandBuffer: MTLCommandBuffer
    private weak var manager: UnifiedBufferManager?
    
    init(commandBuffer: MTLCommandBuffer, manager: UnifiedBufferManager) {
        self.commandBuffer = commandBuffer
        self.manager = manager
    }
    
    /// Complete and release command buffer
    public func commit() async {
        commandBuffer.commit()
        guard let manager = manager else { return }
        await manager.releaseCommandBuffer(commandBuffer)
    }
    
    /// Complete command buffer and wait for completion
    public func commitAndWait() async throws {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // Check for errors
        if let error = commandBuffer.error {
            throw MetalComputeError.commandBufferExecutionFailed(error: error.localizedDescription)
        }
        
        guard let manager = manager else { return }
        await manager.releaseCommandBuffer(commandBuffer)
    }
    
    /// Get the underlying Metal command buffer
    public var metalCommandBuffer: MTLCommandBuffer {
        return commandBuffer
    }
    
    /// Check if the command buffer is still valid
    public var isValid: Bool {
        return manager != nil && commandBuffer.status != .error
    }
}

// MARK: - Support Classes

/// General buffer pool for arbitrary data
public actor GeneralBufferPool: Sendable {
    private let device: MTLDevice
    private let configuration: MetalBufferPoolConfiguration
    private var buffersBySize: [Int: [MTLBuffer]] = [:]
    private var inUseBuffers: Set<ObjectIdentifier> = []
    private var lastAccessTime: [ObjectIdentifier: Date] = [:]
    
    public init(device: MTLDevice, config: MetalBufferPoolConfiguration) {
        self.device = device
        self.configuration = config
    }
    
    /// Get buffer for arbitrary data
    public func getBuffer(for data: Data) async throws -> PooledBuffer {
        let size = data.count
        
        if let buffer = await allocateBuffer(size: size) {
            data.withUnsafeBytes { bytes in
                buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: size)
            }
            return PooledBuffer(buffer: buffer, isReused: false)
        } else {
            throw MetalBufferPoolError.allocationFailed
        }
    }
    
    /// Get buffer for specific size
    public func getBuffer(size: Int) async throws -> PooledBuffer {
        if let buffer = await allocateBuffer(size: size) {
            return PooledBuffer(buffer: buffer, isReused: false)
        } else {
            throw MetalBufferPoolError.allocationFailed
        }
    }
    
    /// Release buffer back to pool
    public func releaseBuffer(_ buffer: MTLBuffer) async {
        let id = ObjectIdentifier(buffer)
        guard inUseBuffers.remove(id) != nil else { return }
        
        lastAccessTime[id] = Date()
        let size = buffer.length
        
        if buffersBySize[size] == nil {
            buffersBySize[size] = []
        }
        
        if buffersBySize[size]!.count < configuration.maxBuffersPerSize / 4 {
            buffersBySize[size]!.append(buffer)
        }
    }
    
    /// Evict oldest buffers
    public func evictOldest(percentage: Float) async {
        for (size, buffers) in buffersBySize {
            let evictCount = Int(Float(buffers.count) * percentage)
            if evictCount > 0 {
                let remaining = Array(buffers.dropFirst(evictCount))
                if remaining.isEmpty {
                    buffersBySize.removeValue(forKey: size)
                } else {
                    buffersBySize[size] = remaining
                }
            }
        }
    }
    
    /// Get current statistics
    public func getStatistics() async -> BufferPoolStatistics {
        let totalBuffers = buffersBySize.values.reduce(0) { $0 + $1.count }
        let totalMemory = buffersBySize.flatMap { (size, buffers) in
            buffers.map { _ in size }
        }.reduce(0, +)
        
        let sizeDistribution = buffersBySize.mapValues { $0.count }
        
        return BufferPoolStatistics(
            poolName: "TokenBufferPool",
            totalAllocations: 0, // Would need to track this
            currentAllocations: totalBuffers,
            peakAllocations: totalBuffers, // Would need to track this
            totalBytesAllocated: totalMemory,
            currentBytesAllocated: totalMemory,
            peakBytesAllocated: totalMemory, // Would need to track this
            reuseCount: 0, // Would need to track this
            hitRate: 0.0, // Would need to track this
            averageAllocationSize: totalBuffers > 0 ? totalMemory / totalBuffers : 0,
            fragmentationRatio: 0.0 // Would need to calculate this
        )
    }
    
    /// Get current memory usage
    public func getCurrentMemoryUsage() async -> Int {
        return buffersBySize.reduce(0) { total, entry in
            total + (entry.key * entry.value.count)
        }
    }
    
    // MARK: - Private Implementation
    
    private func allocateBuffer(size: Int) async -> MTLBuffer? {
        // Try to reuse existing buffer
        if let existing = buffersBySize[size]?.popLast() {
            inUseBuffers.insert(ObjectIdentifier(existing))
            return existing
        }
        
        // Create new buffer
        guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
            return nil
        }
        
        inUseBuffers.insert(ObjectIdentifier(buffer))
        return buffer
    }
}

/// Buffer memory manager for tracking memory usage
public actor BufferMemoryManager: Sendable {
    private let configuration: MetalBufferPoolConfiguration
    private var peakMemoryUsage: Int = 0
    private var currentMemoryUsage: Int = 0
    private var memoryPressureObserver: NSObjectProtocol?
    
    public init(config: MetalBufferPoolConfiguration) {
        self.configuration = config
        registerMemoryPressureNotifications()
    }
    
    deinit {
        if let observer = memoryPressureObserver {
            NotificationCenter.default.removeObserver(observer)
        }
    }
    
    /// Record memory allocation
    public func recordAllocation(bytes: Int) async {
        currentMemoryUsage += bytes
        peakMemoryUsage = max(peakMemoryUsage, currentMemoryUsage)
    }
    
    /// Record memory deallocation
    public func recordDeallocation(bytes: Int) async {
        currentMemoryUsage = max(0, currentMemoryUsage - bytes)
    }
    
    /// Get peak memory usage
    public func getPeakMemoryUsage() async -> Int {
        return peakMemoryUsage
    }
    
    /// Check if memory pressure action needed
    public func shouldTriggerCleanup() async -> Bool {
        let memoryThreshold = configuration.maxMemoryUsage
        return currentMemoryUsage > Int(Float(memoryThreshold) * 0.8) // 80% threshold
    }
    
    /// Calculate suggested eviction percentage based on memory pressure
    public func calculateEvictionPercentage() async -> Float {
        let memoryThreshold = configuration.maxMemoryUsage
        let usageRatio = Float(currentMemoryUsage) / Float(memoryThreshold)
        
        switch usageRatio {
        case 0.8..<0.9:
            return 0.2 // Light cleanup
        case 0.9..<0.95:
            return 0.4 // Moderate cleanup
        case 0.95...:
            return 0.7 // Aggressive cleanup
        default:
            return 0.0 // No cleanup needed
        }
    }
    
    private nonisolated func registerMemoryPressureNotifications() {
        #if os(macOS)
        // macOS memory pressure source
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical])
        source.setEventHandler { [weak self] in
            Task {
                await self?.handleSystemMemoryPressure()
            }
        }
        source.resume()
        #endif
    }
    
    private func handleSystemMemoryPressure() async {
        // This would trigger buffer manager cleanup
        // Actual implementation would notify the UnifiedBufferManager
    }
}
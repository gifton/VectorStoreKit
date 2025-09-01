// UnifiedBufferManager.swift
// VectorStoreKit
//
// Unified buffer manager that handles all Metal buffer operations
// Provides token-based ownership, memory pressure handling, and performance optimization

import Foundation
import Metal

/// Unified buffer manager that handles all Metal buffer operations
public actor UnifiedBufferManager: Sendable {
    
    // MARK: - Core Properties
    
    private let device: MTLDevice
    private let configuration: MetalBufferPoolConfiguration
    
    // Multi-tier buffer pools for different use cases
    private let vectorBufferPool: TypedBufferPool<Vector512>
    private let quantizedBufferPool: TypedBufferPool<QuantizedVector>
    private let commandBufferPool: MetalCommandBufferPool
    private let generalBufferPool: GeneralBufferPool
    
    // Memory management
    private let memoryManager: BufferMemoryManager
    private let performanceProfiler: MetalProfiler?
    
    // Statistics tracking
    private var allocationCount: UInt64 = 0
    private var totalBytesAllocated: UInt64 = 0
    private var bufferReuseCount: UInt64 = 0
    private var memoryPressureEventCount: UInt64 = 0
    
    // Active buffer tracking for token-based ownership
    private var activeTokens: Set<ObjectIdentifier> = []
    private var tokenToBuffer: [ObjectIdentifier: MTLBuffer] = [:]
    
    public init(device: MTLDevice, configuration: MetalBufferPoolConfiguration = .research) async {
        self.device = device
        self.configuration = configuration
        
        // Initialize specialized pools
        self.vectorBufferPool = TypedBufferPool<Vector512>(device: device, config: configuration)
        self.quantizedBufferPool = TypedBufferPool<QuantizedVector>(device: device, config: configuration)
        self.commandBufferPool = MetalCommandBufferPool(device: device, maxPoolSize: configuration.maxBuffersPerSize)
        self.generalBufferPool = GeneralBufferPool(device: device, config: configuration)
        
        // Initialize memory management
        self.memoryManager = BufferMemoryManager(config: configuration)
        self.performanceProfiler = configuration.enableProfiling ? MetalProfiler(enabled: true) : nil
        
        // Register for memory pressure notifications
        await registerMemoryPressureHandling()
    }
    
    // MARK: - Unified Buffer Allocation Interface
    
    /// Get buffer for Vector512 with optimal performance
    public func getBuffer(for vector: Vector512) async throws -> BufferToken {
        let startTime = CFAbsoluteTimeGetCurrent()
        defer {
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            Task { await performanceProfiler?.recordOperation(.bufferTransfer, duration: duration, dataSize: 2048) }
        }
        
        do {
            let buffer = try await vectorBufferPool.getBuffer(for: vector)
            await updateStatistics(allocation: true, bytes: 2048, reused: buffer.isReused)
            let token = BufferToken(buffer: buffer.buffer, pool: .vector, size: 2048, manager: self)
            await registerToken(token)
            return token
        } catch {
            throw MetalComputeError.bufferAllocationFailed(size: 2048, error: error.localizedDescription)
        }
    }
    
    /// Get buffer for quantized vectors with compression-aware handling
    public func getBuffer(for quantized: QuantizedVector) async throws -> BufferToken {
        let startTime = CFAbsoluteTimeGetCurrent()
        defer {
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            Task { await performanceProfiler?.recordOperation(.bufferTransfer, duration: duration, dataSize: quantized.codes.count) }
        }
        
        do {
            let buffer = try await quantizedBufferPool.getBuffer(for: quantized)
            await updateStatistics(allocation: true, bytes: quantized.codes.count, reused: buffer.isReused)
            let token = BufferToken(buffer: buffer.buffer, pool: .quantized, size: quantized.codes.count, manager: self)
            await registerToken(token)
            return token
        } catch {
            throw MetalComputeError.bufferAllocationFailed(size: quantized.codes.count, error: error.localizedDescription)
        }
    }
    
    /// Get buffer for arbitrary data with automatic size optimization
    public func getBuffer(for data: Data) async throws -> BufferToken {
        let startTime = CFAbsoluteTimeGetCurrent()
        defer {
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            Task { await performanceProfiler?.recordOperation(.bufferTransfer, duration: duration, dataSize: data.count) }
        }
        
        // Check if we can handle this more efficiently with specialized pools
        if data.count == 2048 { // Vector512 size
            if let vector = try? data.withUnsafeBytes({ bytes in
                Array(bytes.bindMemory(to: Float.self))
            }), vector.count == 512 {
                let vector512 = Vector512(vector)
                return try await getBuffer(for: vector512)
            }
        }
        
        // Use general buffer pool
        do {
            let buffer = try await generalBufferPool.getBuffer(for: data)
            await updateStatistics(allocation: true, bytes: data.count, reused: buffer.isReused)
            let token = BufferToken(buffer: buffer.buffer, pool: .general, size: data.count, manager: self)
            await registerToken(token)
            return token
        } catch {
            throw MetalComputeError.bufferAllocationFailed(size: data.count, error: error.localizedDescription)
        }
    }
    
    /// Get command buffer with optimal queue management
    public func getCommandBuffer() async throws -> CommandBufferToken {
        let startTime = CFAbsoluteTimeGetCurrent()
        defer {
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            Task { await performanceProfiler?.recordOperation(.commandBufferSubmission, duration: duration, dataSize: 0) }
        }
        
        do {
            let commandBuffer = try await commandBufferPool.getCommandBuffer()
            return CommandBufferToken(commandBuffer: commandBuffer, manager: self)
        } catch {
            throw MetalComputeError.commandBufferCreationFailed
        }
    }
    
    // MARK: - Advanced Buffer Operations
    
    /// Batch allocate buffers for multiple vectors with optimal memory layout
    public func getBatchBuffers(for vectors: [Vector512]) async throws -> BatchBufferToken {
        guard !vectors.isEmpty else {
            throw MetalComputeError.emptyInput(parameter: "vectors")
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let totalSize = vectors.count * 2048
        
        defer {
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            Task { await performanceProfiler?.recordOperation(.bufferTransfer, duration: duration, dataSize: totalSize) }
        }
        
        // Try to allocate a single large buffer for better performance
        if let batchBuffer = try? await generalBufferPool.getBuffer(size: totalSize) {
            // Copy all vectors into the batch buffer
            batchBuffer.buffer.contents().withMemoryRebound(to: Float.self, capacity: vectors.count * 512) { ptr in
                for (index, vector) in vectors.enumerated() {
                    let vectorArray = vector.toArray()
                    vectorArray.withUnsafeBufferPointer { vectorPtr in
                        let destPtr = ptr.advanced(by: index * 512)
                        destPtr.update(from: vectorPtr.baseAddress!, count: 512)
                    }
                }
            }
            
            await updateStatistics(allocation: true, bytes: totalSize, reused: batchBuffer.isReused)
            return BatchBufferToken(buffer: batchBuffer.buffer, count: vectors.count, manager: self)
        } else {
            // Fall back to individual buffers
            var individualBuffers: [BufferToken] = []
            for vector in vectors {
                let token = try await getBuffer(for: vector)
                individualBuffers.append(token)
            }
            return BatchBufferToken(individualBuffers: individualBuffers, manager: self)
        }
    }
    
    /// Prefetch buffers for anticipated workload
    public func prefetchBuffers(sizes: [Int]) async {
        for size in sizes {
            do {
                let buffer = try await generalBufferPool.getBuffer(size: size)
                // Immediately return to pool for future use
                await generalBufferPool.releaseBuffer(buffer.buffer)
            } catch {
                // Prefetch failures are non-critical
                continue
            }
        }
    }
    
    // MARK: - Memory Management
    
    /// Handle memory pressure with intelligent buffer eviction
    public func handleMemoryPressure(_ level: SystemMemoryPressure) async {
        memoryPressureEventCount += 1
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        switch level {
        case .normal:
            // Light cleanup - evict oldest unused buffers
            await vectorBufferPool.evictOldest(percentage: 0.2)
            await quantizedBufferPool.evictOldest(percentage: 0.2)
            await generalBufferPool.evictOldest(percentage: 0.1)
            
        case .warning:
            // Moderate cleanup - more aggressive eviction
            await vectorBufferPool.evictOldest(percentage: 0.5)
            await quantizedBufferPool.evictOldest(percentage: 0.5)
            await generalBufferPool.evictOldest(percentage: 0.3)
            await commandBufferPool.clearPool()
            
        case .critical:
            // Aggressive cleanup - keep only essential buffers
            await vectorBufferPool.evictAll()
            await quantizedBufferPool.evictAll()
            await generalBufferPool.evictOldest(percentage: 0.8)
            await commandBufferPool.clearPool()
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        await performanceProfiler?.recordOperation(.optimization, duration: duration, dataSize: 0)
    }
    
    /// Get comprehensive memory statistics
    public func getMemoryStatistics() async -> UnifiedBufferStatistics {
        let vectorStats = await vectorBufferPool.getStatistics()
        let quantizedStats = await quantizedBufferPool.getStatistics()
        let generalStats = await generalBufferPool.getStatistics()
        let commandStats = await commandBufferPool.getStatistics()
        
        return UnifiedBufferStatistics(
            totalAllocations: allocationCount,
            totalBytesAllocated: totalBytesAllocated,
            bufferReuseCount: bufferReuseCount,
            memoryPressureEvents: memoryPressureEventCount,
            vectorPool: vectorStats,
            quantizedPool: quantizedStats,
            generalPool: generalStats,
            commandPool: commandStats,
            currentMemoryUsage: await getCurrentMemoryUsage(),
            peakMemoryUsage: await memoryManager.getPeakMemoryUsage()
        )
    }
    
    // MARK: - Token Management
    
    /// Release buffer token and return buffer to appropriate pool
    internal func releaseBuffer(_ token: BufferToken) async {
        guard activeTokens.contains(ObjectIdentifier(token)) else { return }
        
        activeTokens.remove(ObjectIdentifier(token))
        tokenToBuffer.removeValue(forKey: ObjectIdentifier(token))
        
        switch token.pool {
        case .vector:
            await vectorBufferPool.releaseBuffer(token.buffer)
        case .quantized:
            await quantizedBufferPool.releaseBuffer(token.buffer)
        case .general:
            await generalBufferPool.releaseBuffer(token.buffer)
        case .command:
            // Command buffers are handled differently
            break
        case .matrix:
            await generalBufferPool.releaseBuffer(token.buffer)
        }
    }
    
    /// Release batch buffer
    internal func releaseBatchBuffer(_ buffer: MTLBuffer) async {
        await generalBufferPool.releaseBuffer(buffer)
    }
    
    /// Release command buffer
    internal func releaseCommandBuffer(_ commandBuffer: MTLCommandBuffer) async {
        await commandBufferPool.releaseCommandBuffer(commandBuffer)
    }
    
    // MARK: - Private Implementation
    
    private func updateStatistics(allocation: Bool, bytes: Int, reused: Bool) async {
        if allocation {
            allocationCount += 1
            totalBytesAllocated += UInt64(bytes)
            if reused {
                bufferReuseCount += 1
            }
        }
        
        await memoryManager.recordAllocation(bytes: bytes)
    }
    
    private func getCurrentMemoryUsage() async -> Int {
        let vectorMemory = await vectorBufferPool.getCurrentMemoryUsage()
        let quantizedMemory = await quantizedBufferPool.getCurrentMemoryUsage()
        let generalMemory = await generalBufferPool.getCurrentMemoryUsage()
        let commandMemory = await commandBufferPool.getCurrentMemoryUsage()
        
        return vectorMemory + quantizedMemory + generalMemory + commandMemory
    }
    
    private func registerMemoryPressureHandling() async {
        // Register for system memory pressure notifications
        // Implementation would depend on platform (iOS/macOS)
    }
    
    private func registerToken(_ token: BufferToken) async {
        activeTokens.insert(ObjectIdentifier(token))
        tokenToBuffer[ObjectIdentifier(token)] = token.buffer
    }
}

// MARK: - Global Convenience Accessor

extension MTLDevice {
    /// Get or create unified buffer manager for this device
    var unifiedBufferManager: UnifiedBufferManager {
        get async {
            return await UnifiedBufferManager.shared(for: self)
        }
    }
}

extension UnifiedBufferManager {
    /// Get shared buffer manager for device
    static func shared(for device: MTLDevice) async -> UnifiedBufferManager {
        return await SharedManagerStore.shared.getManager(for: device)
    }
}

/// Actor for managing shared UnifiedBufferManager instances
private actor SharedManagerStore {
    static let shared = SharedManagerStore()
    
    private var managers: [ObjectIdentifier: UnifiedBufferManager] = [:]
    
    func getManager(for device: MTLDevice) async -> UnifiedBufferManager {
        let deviceId = ObjectIdentifier(device)
        
        if let existing = managers[deviceId] {
            return existing
        }
        
        let manager = await UnifiedBufferManager(device: device)
        managers[deviceId] = manager
        return manager
    }
}
// VectorStoreKit: Metal Command Buffer Pool
//
// Efficient command buffer pooling for reduced allocation overhead

import Foundation
@preconcurrency import Metal
import os.log

/// Pool for reusing Metal command buffers to reduce allocation overhead
public actor MetalCommandBufferPool {
    
    // MARK: - Properties
    
    private let device: MTLDevice
    private let maxPoolSize: Int
    private let profiler: MetalProfiler?
    
    // Pool management
    private var availableBuffers: [PooledCommandBuffer] = []
    private var activeBuffers: Set<ObjectIdentifier> = []
    private var activeCommandBuffers: Set<ObjectIdentifier> = []
    
    // Statistics
    private var totalAllocations: Int = 0
    private var poolHits: Int = 0
    private var poolMisses: Int = 0
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalCommandBufferPool")
    
    // MARK: - Initialization
    
    public init(
        device: MTLDevice,
        maxPoolSize: Int = 16,
        profiler: MetalProfiler? = nil
    ) {
        self.device = device
        self.maxPoolSize = maxPoolSize
        self.profiler = profiler
    }
    
    // MARK: - Buffer Management
    
    /// Get a command buffer from the pool or create a new one
    public func getCommandBuffer(label: String? = nil) async throws -> PooledCommandBuffer {
        // Try to get from pool first
        if let pooledBuffer = availableBuffers.popLast() {
            poolHits += 1
            logger.debug("Command buffer pool hit (size: \(self.availableBuffers.count))")
            
            // Reset and configure buffer
            pooledBuffer.reset()
            if let label = label {
                pooledBuffer.buffer.label = label
            }
            
            // Track as active
            activeBuffers.insert(ObjectIdentifier(pooledBuffer))
            
            await profiler?.recordEvent(.commandBufferPoolHit)
            return pooledBuffer
        }
        
        // Create new buffer
        poolMisses += 1
        totalAllocations += 1
        
        guard let buffer = await device.makeCommandBuffer() else {
            throw MetalComputeError.commandBufferCreationFailed
        }
        
        if let label = label {
            buffer.label = label
        }
        
        let pooledBuffer = PooledCommandBuffer(
            buffer: buffer,
            pool: self,
            creationTime: CFAbsoluteTimeGetCurrent()
        )
        
        // Track as active
        activeBuffers.insert(ObjectIdentifier(pooledBuffer))
        
        logger.debug("Created new command buffer (total allocations: \(self.totalAllocations))")
        await profiler?.recordEvent(.commandBufferPoolMiss)
        await profiler?.recordOperation(
            .commandBufferSubmission,
            duration: 0.0,
            dataSize: 0
        )
        
        return pooledBuffer
    }
    
    /// Return a buffer to the pool
    internal func returnBuffer(_ pooledBuffer: PooledCommandBuffer) async {
        // Remove from active set
        activeBuffers.remove(ObjectIdentifier(pooledBuffer))
        
        // Untrack the command buffer
        untrackCompletedBuffer(pooledBuffer.buffer)
        
        // Check if pool is full
        if availableBuffers.count >= maxPoolSize {
            logger.debug("Command buffer pool full, discarding buffer")
            return
        }
        
        // Reset buffer state
        pooledBuffer.reset()
        
        // Add back to pool
        availableBuffers.append(pooledBuffer)
        logger.debug("Returned buffer to pool (size: \(self.availableBuffers.count))")
    }
    
    /// Clear all pooled buffers
    public func clear() {
        availableBuffers.removeAll()
        activeBuffers.removeAll()
        activeCommandBuffers.removeAll()
        logger.info("Cleared command buffer pool")
    }
    
    /// Track scheduled command buffer
    internal func trackScheduledBuffer(_ buffer: MTLCommandBuffer) {
        activeCommandBuffers.insert(ObjectIdentifier(buffer))
    }
    
    /// Untrack completed command buffer  
    internal func untrackCompletedBuffer(_ buffer: MTLCommandBuffer) {
        activeCommandBuffers.remove(ObjectIdentifier(buffer))
    }
    
    // MARK: - Statistics
    
    public var statistics: CommandBufferPoolStatistics {
        CommandBufferPoolStatistics(
            totalAllocations: totalAllocations,
            poolHits: poolHits,
            poolMisses: poolMisses,
            currentPoolSize: availableBuffers.count,
            activeBuffers: activeBuffers.count,
            activeCommandBuffers: activeCommandBuffers.count,
            hitRate: poolHits + poolMisses > 0 ? 
                Float(poolHits) / Float(poolHits + poolMisses) : 0
        )
    }
    
    /// Pre-warm the pool with buffers
    public func prewarm(count: Int) async throws {
        let bufferCount = min(count, maxPoolSize)
        
        for _ in 0..<bufferCount {
            guard let buffer = await device.makeCommandBuffer() else {
                throw MetalComputeError.commandBufferCreationFailed
            }
            
            let pooledBuffer = PooledCommandBuffer(
                buffer: buffer,
                pool: self,
                creationTime: CFAbsoluteTimeGetCurrent()
            )
            
            availableBuffers.append(pooledBuffer)
            totalAllocations += 1
        }
        
        logger.info("Pre-warmed pool with \(bufferCount) command buffers")
    }
}

// MARK: - Pooled Command Buffer

/// Wrapper for pooled command buffers with automatic return to pool
public final class PooledCommandBuffer: @unchecked Sendable {
    public let buffer: MTLCommandBuffer
    private weak var pool: MetalCommandBufferPool?
    private let creationTime: CFAbsoluteTime
    private var completionHandlers: [(MTLCommandBuffer) -> Void] = []
    private let lock = NSLock()
    
    init(buffer: MTLCommandBuffer, pool: MetalCommandBufferPool, creationTime: CFAbsoluteTime) {
        self.buffer = buffer
        self.pool = pool
        self.creationTime = creationTime
        
        // Track when scheduled
        buffer.addScheduledHandler { [weak pool] commandBuffer in
            guard let pool = pool else { return }
            Task {
                await pool.trackScheduledBuffer(commandBuffer)
            }
        }
        
        // Set up automatic return to pool on completion
        buffer.addCompletedHandler { [weak self] _ in
            guard let self = self else { return }
            
            // Execute any custom completion handlers
            self.lock.lock()
            let handlers = self.completionHandlers
            self.completionHandlers.removeAll()
            self.lock.unlock()
            
            for handler in handlers {
                handler(buffer)
            }
            
            // Return to pool
            Task { [weak pool, weak self] in
                guard let pool = pool, let self = self else { return }
                await pool.returnBuffer(self)
            }
        }
    }
    
    /// Reset buffer state for reuse
    func reset() {
        lock.lock()
        completionHandlers.removeAll()
        lock.unlock()
    }
    
    /// Add a completion handler
    public func addCompletedHandler(_ handler: @escaping (MTLCommandBuffer) -> Void) {
        lock.lock()
        completionHandlers.append(handler)
        lock.unlock()
    }
    
    /// Commit the buffer
    public func commit() {
        buffer.commit()
    }
    
    /// Wait until completed
    public func waitUntilCompleted() {
        buffer.waitUntilCompleted()
    }
    
    /// Get buffer age
    public var age: TimeInterval {
        CFAbsoluteTimeGetCurrent() - creationTime
    }
}

// MARK: - Supporting Types

public struct CommandBufferPoolStatistics: Sendable {
    public let totalAllocations: Int
    public let poolHits: Int
    public let poolMisses: Int
    public let currentPoolSize: Int
    public let activeBuffers: Int
    public let activeCommandBuffers: Int
    public let hitRate: Float
}

// MARK: - Batch Command Buffer

/// Specialized command buffer for efficient batch operations
public actor BatchCommandBuffer {
    private let device: MTLDevice
    private let commandBuffer: PooledCommandBuffer
    private var encoders: [MTLComputeCommandEncoder] = []
    private let maxEncodersPerBuffer: Int
    
    public init(
        device: MTLDevice,
        commandBuffer: PooledCommandBuffer,
        maxEncodersPerBuffer: Int = 100
    ) {
        self.device = device
        self.commandBuffer = commandBuffer
        self.maxEncodersPerBuffer = maxEncodersPerBuffer
    }
    
    /// Add a compute operation to the batch
    public func addOperation(
        pipeline: MTLComputePipelineState,
        buffers: [(buffer: MTLBuffer, index: Int)],
        threadgroupSize: MTLSize,
        threadgroupCount: MTLSize
    ) throws {
        guard encoders.count < maxEncodersPerBuffer else {
            throw MetalComputeError.batchSizeExceeded(
                current: encoders.count,
                maximum: maxEncodersPerBuffer
            )
        }
        
        guard let encoder = commandBuffer.buffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        
        for (buffer, index) in buffers {
            encoder.setBuffer(buffer, offset: 0, index: index)
        }
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        encoders.append(encoder)
    }
    
    /// Commit all operations in the batch
    public func commit() async {
        commandBuffer.commit()
    }
    
    /// Wait for all operations to complete
    public func waitUntilCompleted() async {
        commandBuffer.waitUntilCompleted()
    }
}

// MARK: - Extensions

extension MetalComputeError {
    static func batchSizeExceeded(current: Int, maximum: Int) -> MetalComputeError {
        .invalidBufferSize(requested: current, maximum: maximum)
    }
}
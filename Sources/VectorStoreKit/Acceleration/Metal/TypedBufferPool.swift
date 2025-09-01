// TypedBufferPool.swift
// VectorStoreKit
//
// Specialized buffer pool for specific data types
// Provides optimal performance through type-specific optimizations

import Foundation
import Metal

/// Specialized buffer pool for specific data types
public actor TypedBufferPool<T>: Sendable {
    private let device: MTLDevice
    private let configuration: MetalBufferPoolConfiguration
    private var availableBuffers: [PooledBuffer] = []
    private var inUseBuffers: Set<ObjectIdentifier> = []
    private let maxPoolSize: Int
    private var lastAccessTime: [ObjectIdentifier: Date] = [:]
    
    // Performance tracking
    private var totalAllocations: UInt64 = 0
    private var cacheHits: UInt64 = 0
    private var cacheMisses: UInt64 = 0
    
    public init(device: MTLDevice, config: MetalBufferPoolConfiguration) {
        self.device = device
        self.configuration = config
        self.maxPoolSize = config.maxBuffersPerSize
        
        // Pre-allocate common sizes if specified
        Task {
            await preAllocateBuffers()
        }
    }
    
    /// Get buffer for specific type with optimal layout
    public func getBuffer(for data: T) async throws -> PooledBuffer {
        let requiredSize = calculateSize(for: data)
        totalAllocations += 1
        
        // Try to find reusable buffer
        if let reusableIndex = findReusableBuffer(size: requiredSize) {
            let buffer = availableBuffers.remove(at: reusableIndex)
            inUseBuffers.insert(ObjectIdentifier(buffer.buffer))
            cacheHits += 1
            
            // Copy data to buffer
            try copyData(data, to: buffer.buffer)
            
            return PooledBuffer(buffer: buffer.buffer, isReused: true)
        }
        
        // Create new buffer
        cacheMisses += 1
        guard let mtlBuffer = device.makeBuffer(length: requiredSize, options: .storageModeShared) else {
            throw MetalComputeError.bufferAllocationFailed(size: requiredSize, error: "Metal device allocation failed")
        }
        
        // Copy data to new buffer
        try copyData(data, to: mtlBuffer)
        
        inUseBuffers.insert(ObjectIdentifier(mtlBuffer))
        return PooledBuffer(buffer: mtlBuffer, isReused: false)
    }
    
    /// Get buffer for specific size without data copying
    public func getBuffer(size: Int) async throws -> PooledBuffer {
        totalAllocations += 1
        
        // Try to find reusable buffer
        if let reusableIndex = findReusableBuffer(size: size) {
            let buffer = availableBuffers.remove(at: reusableIndex)
            inUseBuffers.insert(ObjectIdentifier(buffer.buffer))
            cacheHits += 1
            return PooledBuffer(buffer: buffer.buffer, isReused: true)
        }
        
        // Create new buffer
        cacheMisses += 1
        guard let mtlBuffer = device.makeBuffer(length: size, options: .storageModeShared) else {
            throw MetalComputeError.bufferAllocationFailed(size: size, error: "Metal device allocation failed")
        }
        
        inUseBuffers.insert(ObjectIdentifier(mtlBuffer))
        return PooledBuffer(buffer: mtlBuffer, isReused: false)
    }
    
    /// Release buffer back to pool with LRU tracking
    public func releaseBuffer(_ buffer: MTLBuffer) async {
        let id = ObjectIdentifier(buffer)
        guard inUseBuffers.remove(id) != nil else { return }
        
        lastAccessTime[id] = Date()
        
        // Add to available buffers if pool not full
        if availableBuffers.count < maxPoolSize {
            availableBuffers.append(PooledBuffer(buffer: buffer, isReused: false))
        }
        // Otherwise buffer will be deallocated automatically
    }
    
    /// Evict oldest buffers by percentage
    public func evictOldest(percentage: Float) async {
        let evictCount = Int(Float(availableBuffers.count) * percentage)
        guard evictCount > 0 else { return }
        
        // Sort by last access time and remove oldest
        let sortedBuffers = availableBuffers.sorted { buffer1, buffer2 in
            let time1 = lastAccessTime[ObjectIdentifier(buffer1.buffer)] ?? Date.distantPast
            let time2 = lastAccessTime[ObjectIdentifier(buffer2.buffer)] ?? Date.distantPast
            return time1 < time2
        }
        
        // Remove oldest buffers
        for i in 0..<min(evictCount, sortedBuffers.count) {
            let id = ObjectIdentifier(sortedBuffers[i].buffer)
            lastAccessTime.removeValue(forKey: id)
        }
        
        availableBuffers = Array(sortedBuffers.dropFirst(evictCount))
    }
    
    /// Evict all cached buffers
    public func evictAll() async {
        availableBuffers.removeAll()
        lastAccessTime.removeAll()
    }
    
    /// Get current statistics
    public func getStatistics() async -> BufferPoolStatistics {
        let totalMemory = availableBuffers.reduce(0) { $0 + $1.buffer.length }
        let sizeDistribution = Dictionary(grouping: availableBuffers, by: { $0.buffer.length })
            .mapValues { $0.count }
        
        return BufferPoolStatistics(
            totalBuffers: availableBuffers.count,
            memoryUsage: totalMemory,
            sizeDistribution: sizeDistribution
        )
    }
    
    /// Get current memory usage
    public func getCurrentMemoryUsage() async -> Int {
        return availableBuffers.reduce(0) { $0 + $1.buffer.length }
    }
    
    /// Get cache hit rate
    public func getCacheHitRate() async -> Float {
        guard totalAllocations > 0 else { return 0.0 }
        return Float(cacheHits) / Float(totalAllocations)
    }
    
    // MARK: - Private Implementation
    
    private func calculateSize(for data: T) -> Int {
        if let vector = data as? Vector512 {
            return 512 * MemoryLayout<Float>.stride
        } else if let quantized = data as? QuantizedVector {
            return quantized.codes.count * MemoryLayout<UInt8>.stride
        } else {
            return MemoryLayout<T>.size
        }
    }
    
    private func copyData(_ data: T, to buffer: MTLBuffer) throws {
        if let vector = data as? Vector512 {
            vector.withUnsafeMetalBytes { bytes in
                buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
        } else if let quantized = data as? QuantizedVector {
            quantized.codes.withUnsafeBytes { bytes in
                buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
        } else {
            withUnsafeBytes(of: data) { bytes in
                buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: bytes.count)
            }
        }
    }
    
    private func findReusableBuffer(size: Int) -> Int? {
        return availableBuffers.firstIndex { $0.buffer.length >= size }
    }
    
    private func preAllocateBuffers() async {
        let typicalSizes: [Int]
        
        if T.self == Vector512.self {
            typicalSizes = [2048] // Vector512 size
        } else if T.self == QuantizedVector.self {
            typicalSizes = [64, 128, 256, 512] // Common quantized sizes
        } else {
            typicalSizes = configuration.preallocationSizes
        }
        
        for size in typicalSizes {
            guard availableBuffers.count < maxPoolSize / 4 else { break } // Pre-allocate up to 25% of pool
            
            if let buffer = device.makeBuffer(length: size, options: .storageModeShared) {
                availableBuffers.append(PooledBuffer(buffer: buffer, isReused: false))
            }
        }
    }
}

/// Wrapper for pooled buffers with metadata
public struct PooledBuffer: Sendable {
    public let buffer: MTLBuffer
    public let isReused: Bool
}
// VectorStoreKit: Metal Buffer Pool
//
// Efficient buffer management for Metal operations

import Foundation
import Metal

// BufferPoolStatistics is defined in MetalMLBufferPool.swift

/// Manages a pool of reusable Metal buffers
public actor MetalBufferPool {
    
    /// Configuration for buffer pool
    public typealias Configuration = MetalBufferPoolConfiguration
    
    private let device: MTLDevice
    private let configuration: Configuration
    private var availableBuffers: [Int: [MTLBuffer]] = [:]
    private var inUseBuffers: Set<ObjectIdentifier> = []
    private var totalBuffers = 0
    private var memoryUsage = 0
    private var sizeDistribution: [Int: Int] = [:]
    
    public init(device: MTLDevice, configuration: Configuration = .research) {
        self.device = device
        self.configuration = configuration
        
        // Preallocate buffers if specified
        for size in configuration.preallocationSizes {
            availableBuffers[size] = []
        }
    }
    
    /// Get statistics about buffer pool usage
    public func getStatistics() -> BufferPoolStatistics {
        return BufferPoolStatistics(
            totalBuffers: totalBuffers,
            memoryUsage: memoryUsage,
            sizeDistribution: sizeDistribution
        )
    }
    
    /// Allocate or reuse a buffer of the specified size
    public func allocateBuffer(size: Int) -> MTLBuffer? {
        // Simplified implementation
        if let existing = availableBuffers[size]?.popLast() {
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
    
    /// Return a buffer to the pool
    public func releaseBuffer(_ buffer: MTLBuffer) {
        let id = ObjectIdentifier(buffer)
        guard inUseBuffers.contains(id) else { return }
        
        inUseBuffers.remove(id)
        let size = buffer.length
        
        if availableBuffers[size] == nil {
            availableBuffers[size] = []
        }
        availableBuffers[size]?.append(buffer)
    }
    
    /// Get a buffer for the specified data
    public func getBuffer(for data: [Float]) throws -> MTLBuffer {
        let size = data.count * MemoryLayout<Float>.size
        
        if let buffer = allocateBuffer(size: size) {
            data.withUnsafeBytes { bytes in
                buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: size)
            }
            return buffer
        } else {
            throw MetalBufferPoolError.allocationFailed
        }
    }
    
    /// Get a buffer for generic SIMD vector
    public func getBuffer<T: SIMD>(for vector: T) throws -> MTLBuffer where T.Scalar: BinaryFloatingPoint {
        let size = vector.scalarCount * MemoryLayout<T.Scalar>.size
        
        if let buffer = allocateBuffer(size: size) {
            withUnsafeBytes(of: vector) { bytes in
                buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: size)
            }
            return buffer
        } else {
            throw MetalBufferPoolError.allocationFailed
        }
    }
    
    /// Get a buffer for array of SIMD vectors
    public func getBuffer<T: SIMD>(for vectors: [T]) throws -> MTLBuffer where T.Scalar: BinaryFloatingPoint {
        let elementSize = vectors.first?.scalarCount ?? 0
        let totalSize = vectors.count * elementSize * MemoryLayout<T.Scalar>.size
        
        if let buffer = allocateBuffer(size: totalSize) {
            vectors.withUnsafeBytes { bytes in
                buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: totalSize)
            }
            return buffer
        } else {
            throw MetalBufferPoolError.allocationFailed
        }
    }
    
    /// Get a buffer for a single value
    public func getBuffer<T>(for value: T) throws -> MTLBuffer {
        let size = MemoryLayout<T>.size
        
        if let buffer = allocateBuffer(size: size) {
            withUnsafeBytes(of: value) { bytes in
                buffer.contents().copyMemory(from: bytes.baseAddress!, byteCount: size)
            }
            return buffer
        } else {
            throw MetalBufferPoolError.allocationFailed
        }
    }
    
    /// Get a buffer of the specified size
    public func getBuffer(size: Int) throws -> MTLBuffer {
        if let buffer = allocateBuffer(size: size) {
            return buffer
        } else {
            throw MetalBufferPoolError.allocationFailed
        }
    }
    
    /// Return a buffer to the pool (alias for releaseBuffer)
    public func returnBuffer(_ buffer: MTLBuffer) {
        releaseBuffer(buffer)
    }
}

// MARK: - Protocol Conformance

extension MetalBufferPool: MemoryManagedBufferPool {
    /// Clear all cached buffers to free memory
    public func clearAll() async {
        // Clear all available buffers
        availableBuffers.removeAll()
        // Note: in-use buffers remain allocated until explicitly released
    }
    
    /// Get current memory usage in bytes
    public func getCurrentMemoryUsage() async -> Int {
        var totalMemory = 0
        
        // Calculate memory from available buffers
        for (size, buffers) in availableBuffers {
            totalMemory += size * buffers.count
        }
        
        // Note: We can't easily calculate in-use buffer sizes without tracking them
        // This is a limitation of the current implementation
        return totalMemory
    }
}

// MetalBufferPoolError is defined in Core/MetalAccelerationTypes.swift
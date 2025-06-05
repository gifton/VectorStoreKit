// VectorStoreKit: Metal Buffer Pool
//
// Efficient buffer management for Metal operations

import Foundation
import Metal

/// Statistics for buffer pool usage
public struct BufferPoolStatistics: Sendable {
    public let totalAllocated: Int
    public let totalInUse: Int
    public let hitRate: Float
    public let averageAllocationSize: Int
    
    public init(
        totalAllocated: Int = 0,
        totalInUse: Int = 0,
        hitRate: Float = 0.0,
        averageAllocationSize: Int = 0
    ) {
        self.totalAllocated = totalAllocated
        self.totalInUse = totalInUse
        self.hitRate = hitRate
        self.averageAllocationSize = averageAllocationSize
    }
}

/// Manages a pool of reusable Metal buffers
public actor MetalBufferPool {
    
    /// Configuration for buffer pool
    public typealias Configuration = MetalBufferPoolConfiguration
    
    private let device: MTLDevice
    private let configuration: Configuration
    private var availableBuffers: [Int: [MTLBuffer]] = [:]
    private var inUseBuffers: Set<ObjectIdentifier> = []
    private var statistics = BufferPoolStatistics()
    
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
        return statistics
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

/// Errors for buffer pool operations
public enum MetalBufferPoolError: Error {
    case allocationFailed
}
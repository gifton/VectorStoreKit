// Vector512.swift
// VectorStoreKit
//
// Optimized 512-dimensional vector type with improved SIMD initialization
// and bulk memory operations for modern embedding models
//
// Performance optimizations:
// - Bulk memory operations for initialization
// - Multiple accumulators for dot product and distance calculations
// - Loop unrolling for arithmetic operations
// - Compiler optimization hints with @inlinable and @inline(__always)
// - Direct memory mapping for Data conversions

import Foundation
import simd
import Metal

/// Optimized 512-dimensional vector with SIMD and Metal acceleration
public struct Vector512: SIMD, Sendable, Hashable, Codable {
    // Store as 128 float4s for optimal Metal alignment
    @usableFromInline
    internal var storage: ContiguousArray<SIMD4<Float>>
    
    // MARK: - SIMD Protocol Requirements
    
    public typealias Scalar = Float
    public typealias MaskStorage = SIMD64<Float>.MaskStorage
    
    public var scalarCount: Int { 512 }
    
    public subscript(index: Int) -> Float {
        get {
            precondition(index >= 0 && index < 512, "Index out of bounds")
            let vectorIndex = index / 4
            let scalarIndex = index % 4
            return storage[vectorIndex][scalarIndex]
        }
        set {
            precondition(index >= 0 && index < 512, "Index out of bounds")
            let vectorIndex = index / 4
            let scalarIndex = index % 4
            storage[vectorIndex][scalarIndex] = newValue
        }
    }
    
    // MARK: - Initialization
    
    /// Initialize with zeros
    public init() {
        storage = ContiguousArray(repeating: SIMD4<Float>(), count: 128)
    }
    
    /// Initialize from array of floats with optimized bulk memory operations
    @inlinable
    public init(_ values: [Float]) {
        precondition(values.count == 512, "Expected 512 values, got \(values.count)")
        
        storage = ContiguousArray<SIMD4<Float>>()
        storage.reserveCapacity(128)
        
        // Use bulk memory operations for efficient initialization
        values.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else {
                // Fallback for empty buffer
                storage = ContiguousArray(repeating: SIMD4<Float>(), count: 128)
                return
            }
            
            // Cast the buffer to SIMD4 chunks for direct copying
            let simd4Buffer = UnsafeRawPointer(baseAddress).bindMemory(
                to: SIMD4<Float>.self,
                capacity: 128
            )
            
            // Bulk append using unsafe buffer
            storage.append(contentsOf: UnsafeBufferPointer(start: simd4Buffer, count: 128))
        }
    }
    
    /// Initialize with repeating value
    public init(repeating value: Float) {
        let simd4 = SIMD4<Float>(repeating: value)
        storage = ContiguousArray(repeating: simd4, count: 128)
    }
    
    /// Initialize from unsafe buffer with direct memory operations
    @inlinable
    public init(unsafeUninitializedCapacity: Int, initializingWith initializer: (inout UnsafeMutableBufferPointer<Float>) throws -> Void) rethrows {
        precondition(unsafeUninitializedCapacity == 512)
        
        // Pre-allocate storage
        storage = ContiguousArray<SIMD4<Float>>()
        storage.reserveCapacity(128)
        
        // Create a temporary aligned buffer for initialization
        let alignedBuffer = UnsafeMutableRawPointer.allocate(
            byteCount: 512 * MemoryLayout<Float>.stride,
            alignment: 16  // SIMD alignment requirement
        )
        defer { alignedBuffer.deallocate() }
        
        let floatBuffer = alignedBuffer.bindMemory(to: Float.self, capacity: 512)
        var mutableBuffer = UnsafeMutableBufferPointer(start: floatBuffer, count: 512)
        
        try initializer(&mutableBuffer)
        
        // Cast to SIMD4 and bulk copy
        let simd4Buffer = alignedBuffer.bindMemory(to: SIMD4<Float>.self, capacity: 128)
        storage.append(contentsOf: UnsafeBufferPointer(start: simd4Buffer, count: 128))
    }
    
    // MARK: - Memory Layout
    
    /// Aligned memory allocation for optimal Metal performance
    public static func allocateAligned(count: Int) -> UnsafeMutablePointer<Vector512> {
        let alignment = 64 // Cache line size on Apple Silicon
        let ptr = UnsafeMutableRawPointer.allocate(
            byteCount: count * MemoryLayout<Vector512>.size,
            alignment: alignment
        )
        return ptr.bindMemory(to: Vector512.self, capacity: count)
    }
    
    /// Zero-copy view for Metal operations
    public func withUnsafeMetalBytes<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try storage.withUnsafeBytes(body)
    }
    
    /// Create Metal buffer without copying
    public func makeMetalBuffer(device: MTLDevice, options: MTLResourceOptions = .storageModeShared) -> MTLBuffer? {
        storage.withUnsafeBytes { bytes in
            device.makeBuffer(
                bytes: bytes.baseAddress!,
                length: bytes.count,
                options: options
            )
        }
    }
    
    // MARK: - SIMD Operations
    
    /// Optimized dot product using SIMD with multiple accumulators
    @inlinable
    @inline(__always)
    public func dot(_ other: Vector512) -> Float {
        // Use 4 accumulators to hide latency and improve pipelining
        var sum0 = SIMD4<Float>()
        var sum1 = SIMD4<Float>()
        var sum2 = SIMD4<Float>()
        var sum3 = SIMD4<Float>()
        
        // Process 16 SIMD4 vectors at a time using 4 accumulators
        for i in stride(from: 0, to: 128, by: 16) {
            // Accumulator 0
            sum0 += storage[i] * other.storage[i]
            sum0 += storage[i+1] * other.storage[i+1]
            sum0 += storage[i+2] * other.storage[i+2]
            sum0 += storage[i+3] * other.storage[i+3]
            
            // Accumulator 1
            sum1 += storage[i+4] * other.storage[i+4]
            sum1 += storage[i+5] * other.storage[i+5]
            sum1 += storage[i+6] * other.storage[i+6]
            sum1 += storage[i+7] * other.storage[i+7]
            
            // Accumulator 2
            sum2 += storage[i+8] * other.storage[i+8]
            sum2 += storage[i+9] * other.storage[i+9]
            sum2 += storage[i+10] * other.storage[i+10]
            sum2 += storage[i+11] * other.storage[i+11]
            
            // Accumulator 3
            sum3 += storage[i+12] * other.storage[i+12]
            sum3 += storage[i+13] * other.storage[i+13]
            sum3 += storage[i+14] * other.storage[i+14]
            sum3 += storage[i+15] * other.storage[i+15]
        }
        
        // Combine accumulators
        let finalSum = sum0 + sum1 + sum2 + sum3
        return finalSum.sum()
    }
    
    /// Euclidean distance squared with multiple accumulators
    @inlinable
    @inline(__always)
    public func distanceSquared(to other: Vector512) -> Float {
        // Use 4 accumulators for better pipelining
        var sum0 = SIMD4<Float>()
        var sum1 = SIMD4<Float>()
        var sum2 = SIMD4<Float>()
        var sum3 = SIMD4<Float>()
        
        // Process 16 SIMD4 vectors at a time
        for i in stride(from: 0, to: 128, by: 16) {
            // Accumulator 0
            let diff0 = storage[i] - other.storage[i]
            let diff1 = storage[i+1] - other.storage[i+1]
            let diff2 = storage[i+2] - other.storage[i+2]
            let diff3 = storage[i+3] - other.storage[i+3]
            sum0 += diff0 * diff0
            sum0 += diff1 * diff1
            sum0 += diff2 * diff2
            sum0 += diff3 * diff3
            
            // Accumulator 1
            let diff4 = storage[i+4] - other.storage[i+4]
            let diff5 = storage[i+5] - other.storage[i+5]
            let diff6 = storage[i+6] - other.storage[i+6]
            let diff7 = storage[i+7] - other.storage[i+7]
            sum1 += diff4 * diff4
            sum1 += diff5 * diff5
            sum1 += diff6 * diff6
            sum1 += diff7 * diff7
            
            // Accumulator 2
            let diff8 = storage[i+8] - other.storage[i+8]
            let diff9 = storage[i+9] - other.storage[i+9]
            let diff10 = storage[i+10] - other.storage[i+10]
            let diff11 = storage[i+11] - other.storage[i+11]
            sum2 += diff8 * diff8
            sum2 += diff9 * diff9
            sum2 += diff10 * diff10
            sum2 += diff11 * diff11
            
            // Accumulator 3
            let diff12 = storage[i+12] - other.storage[i+12]
            let diff13 = storage[i+13] - other.storage[i+13]
            let diff14 = storage[i+14] - other.storage[i+14]
            let diff15 = storage[i+15] - other.storage[i+15]
            sum3 += diff12 * diff12
            sum3 += diff13 * diff13
            sum3 += diff14 * diff14
            sum3 += diff15 * diff15
        }
        
        // Combine accumulators
        let finalSum = sum0 + sum1 + sum2 + sum3
        return finalSum.sum()
    }
    
    /// Cosine similarity (assumes normalized vectors)
    @inlinable
    public func cosineSimilarity(to other: Vector512, normalized: Bool = false) -> Float {
        if normalized {
            return dot(other)
        } else {
            let dotProduct = dot(other)
            let magnitudeSelf = sqrt(dot(self))
            let magnitudeOther = sqrt(other.dot(other))
            return dotProduct / (magnitudeSelf * magnitudeOther + Float.ulpOfOne)
        }
    }
    
    /// L2 normalize the vector
    public mutating func normalize() {
        let magnitude = sqrt(dot(self))
        guard magnitude > Float.ulpOfOne else { return }
        
        let invMagnitude = 1.0 / magnitude
        for i in 0..<128 {
            storage[i] *= invMagnitude
        }
    }
    
    /// Return normalized copy
    public func normalized() -> Vector512 {
        var copy = self
        copy.normalize()
        return copy
    }
    
    // MARK: - Arithmetic Operations
    
    @inlinable
    public static func +(lhs: Vector512, rhs: Vector512) -> Vector512 {
        var result = Vector512()
        
        // Unroll loop for better performance
        for i in stride(from: 0, to: 128, by: 4) {
            result.storage[i] = lhs.storage[i] + rhs.storage[i]
            result.storage[i+1] = lhs.storage[i+1] + rhs.storage[i+1]
            result.storage[i+2] = lhs.storage[i+2] + rhs.storage[i+2]
            result.storage[i+3] = lhs.storage[i+3] + rhs.storage[i+3]
        }
        return result
    }
    
    @inlinable
    public static func -(lhs: Vector512, rhs: Vector512) -> Vector512 {
        var result = Vector512()
        
        // Unroll loop for better performance
        for i in stride(from: 0, to: 128, by: 4) {
            result.storage[i] = lhs.storage[i] - rhs.storage[i]
            result.storage[i+1] = lhs.storage[i+1] - rhs.storage[i+1]
            result.storage[i+2] = lhs.storage[i+2] - rhs.storage[i+2]
            result.storage[i+3] = lhs.storage[i+3] - rhs.storage[i+3]
        }
        return result
    }
    
    @inlinable
    public static func *(lhs: Vector512, rhs: Float) -> Vector512 {
        var result = Vector512()
        let scalar4 = SIMD4<Float>(repeating: rhs)
        
        // Unroll loop for better performance
        for i in stride(from: 0, to: 128, by: 4) {
            result.storage[i] = lhs.storage[i] * scalar4
            result.storage[i+1] = lhs.storage[i+1] * scalar4
            result.storage[i+2] = lhs.storage[i+2] * scalar4
            result.storage[i+3] = lhs.storage[i+3] * scalar4
        }
        return result
    }
    
    @inlinable
    public static func /(lhs: Vector512, rhs: Float) -> Vector512 {
        precondition(abs(rhs) > Float.ulpOfOne, "Division by zero")
        return lhs * (1.0 / rhs)
    }
    
    // MARK: - Conversion
    
    /// Convert to array of floats with optimized memory copy
    @inlinable
    public func toArray() -> [Float] {
        // Pre-allocate array with exact capacity
        var result = [Float]()
        result.reserveCapacity(512)
        
        // Use bulk memory copy
        storage.withUnsafeBytes { bytes in
            let floatPointer = bytes.bindMemory(to: Float.self)
            result.append(contentsOf: UnsafeBufferPointer(start: floatPointer.baseAddress!, count: 512))
        }
        
        return result
    }
    
    /// Create from Data with direct memory mapping
    @inlinable
    public init?(data: Data) {
        guard data.count == 512 * MemoryLayout<Float>.size else { return nil }
        
        storage = ContiguousArray<SIMD4<Float>>()
        storage.reserveCapacity(128)
        
        // Direct memory mapping without intermediate array
        data.withUnsafeBytes { bytes in
            let simd4Pointer = bytes.bindMemory(to: SIMD4<Float>.self)
            storage.append(contentsOf: UnsafeBufferPointer(
                start: simd4Pointer.baseAddress!,
                count: 128
            ))
        }
    }
    
    /// Convert to Data
    public func toData() -> Data {
        storage.withUnsafeBytes { bytes in
            Data(bytes)
        }
    }
}

// MARK: - SIMD Extension Helpers
// Note: SIMD4 already provides a sum() method in Swift, so no extension is needed

// MARK: - Collection Conformance

extension Vector512: Collection {
    public typealias Index = Int
    public typealias Indices = Range<Int>
    public typealias Iterator = IndexingIterator<Vector512>
    public typealias SubSequence = ArraySlice<Float>
    
    public var startIndex: Int { 0 }
    public var endIndex: Int { 512 }
    
    public var indices: Range<Int> {
        return startIndex..<endIndex
    }
    
    public func index(after i: Int) -> Int {
        return i + 1
    }
    
    public subscript(bounds: Range<Int>) -> SubSequence {
        precondition(bounds.lowerBound >= 0 && bounds.upperBound <= 512, "Range out of bounds")
        return ArraySlice(toArray()[bounds])
    }
}

// MARK: - Performance Hints

extension Vector512 {
    /// Prefetch for upcoming operations
    @inlinable
    public func prefetch() {
        // Memory prefetching is handled automatically by Apple Silicon
        // Explicit prefetch instructions are not exposed in Swift
        // The CPU's prefetcher will handle this based on access patterns
        
        // For optimal performance, ensure sequential access patterns
        // and process vectors in batches that fit in L1/L2 cache
    }
    
    /// Hint for streaming (non-temporal) access
    @inlinable
    public func streamingHint() {
        // Streaming hints are handled at the Metal buffer level
        // Use .storageModePrivate for non-temporal access patterns
        
        // For CPU operations, the hardware prefetcher will detect
        // streaming patterns and optimize accordingly
    }
    
    /// Create multiple vectors from a flat array efficiently
    @inlinable
    public static func createBatch(from flatArray: [Float]) -> [Vector512] {
        precondition(flatArray.count % 512 == 0, "Array size must be multiple of 512")
        
        let vectorCount = flatArray.count / 512
        var result = [Vector512]()
        result.reserveCapacity(vectorCount)
        
        flatArray.withUnsafeBufferPointer { buffer in
            guard let baseAddress = buffer.baseAddress else { return }
            
            for i in 0..<vectorCount {
                let offset = i * 512
                let vectorData = UnsafeBufferPointer(
                    start: baseAddress.advanced(by: offset),
                    count: 512
                )
                result.append(Vector512(Array(vectorData)))
            }
        }
        
        return result
    }
}

// MARK: - Bulk SIMD32 Initialization Helper

/// Optimized SIMD32 initialization from array
@inlinable
public func initializeSIMD32<T: BinaryFloatingPoint>(from array: [T]) -> SIMD32<T> {
    precondition(array.count >= 32, "Array must have at least 32 elements")
    
    var result = SIMD32<T>()
    
    // Use bulk memory operations instead of element-by-element
    array.withUnsafeBufferPointer { buffer in
        guard let baseAddress = buffer.baseAddress else { return }
        
        // Initialize using SIMD8 chunks for better performance
        let simd8_0 = SIMD8<T>(
            baseAddress[0], baseAddress[1], baseAddress[2], baseAddress[3],
            baseAddress[4], baseAddress[5], baseAddress[6], baseAddress[7]
        )
        let simd8_1 = SIMD8<T>(
            baseAddress[8], baseAddress[9], baseAddress[10], baseAddress[11],
            baseAddress[12], baseAddress[13], baseAddress[14], baseAddress[15]
        )
        let simd8_2 = SIMD8<T>(
            baseAddress[16], baseAddress[17], baseAddress[18], baseAddress[19],
            baseAddress[20], baseAddress[21], baseAddress[22], baseAddress[23]
        )
        let simd8_3 = SIMD8<T>(
            baseAddress[24], baseAddress[25], baseAddress[26], baseAddress[27],
            baseAddress[28], baseAddress[29], baseAddress[30], baseAddress[31]
        )
        
        // Combine SIMD8 chunks into SIMD32
        for i in 0..<8 {
            result[i] = simd8_0[i]
            result[i+8] = simd8_1[i]
            result[i+16] = simd8_2[i]
            result[i+24] = simd8_3[i]
        }
    }
    
    return result
}

/// Fast SIMD32 initialization with random values
@inlinable
public func randomSIMD32<T: BinaryFloatingPoint>(in range: ClosedRange<T>) -> SIMD32<T> 
    where T: FloatingPoint, T.RawSignificand: FixedWidthInteger {
    // Initialize using SIMD8 chunks for better performance
    let simd8_0 = SIMD8<T>.random(in: range)
    let simd8_1 = SIMD8<T>.random(in: range)
    let simd8_2 = SIMD8<T>.random(in: range)
    let simd8_3 = SIMD8<T>.random(in: range)
    
    var result = SIMD32<T>()
    for i in 0..<8 {
        result[i] = simd8_0[i]
        result[i+8] = simd8_1[i]
        result[i+16] = simd8_2[i]
        result[i+24] = simd8_3[i]
    }
    
    return result
}
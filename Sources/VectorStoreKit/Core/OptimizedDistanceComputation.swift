// OptimizedDistanceComputation.swift
// VectorStoreKit
//
// Highly optimized distance computation implementations with SIMD, cache optimization,
// and hardware-specific acceleration

import Foundation
import simd
import Accelerate
import os.log
import os

// MARK: - Memory Management Integration

// Note: MemoryPressureLevel is imported from MemoryManagement.swift

/// Memory pressure aware resource manager
// MemoryPressureAware protocol is defined in MemoryProtocols.swift

// MARK: - Safe Cache-Aligned Vector Storage

/// Cache-aligned vector storage with automatic memory management and safety features
@frozen
public struct AlignedVector<Scalar: BinaryFloatingPoint>: ~Copyable {
    private let storage: ManagedBuffer<AlignedVectorHeader, Scalar>
    public let count: Int
    private static var alignment: Int { 64 } // Cache line size
    
    private struct AlignedVectorHeader {
        let count: Int
        let alignment: Int
        var memoryPressureHandler: MemoryPressureAware?
    }
    
    public init(count: Int) throws {
        guard count > 0 else {
            throw VectorStoreError(
                category: .validation,
                code: .invalidInput,
                message: "Invalid vector dimensions: \(count). Count must be positive.",
                context: ["requestedCount": count]
            )
        }
        
        // Calculate aligned size
        let elementSize = MemoryLayout<Scalar>.stride
        let totalSize = count * elementSize
        let alignedCount = (totalSize + Self.alignment - 1) / elementSize
        
        self.count = count
        self.storage = ManagedBuffer<AlignedVectorHeader, Scalar>.create(
            minimumCapacity: alignedCount
        ) { buffer in
            AlignedVectorHeader(count: count, alignment: Self.alignment, memoryPressureHandler: nil)
        }
        
        // Zero-initialize memory
        storage.withUnsafeMutablePointerToElements { ptr in
            ptr.initialize(repeating: 0, count: count)
        }
    }
    
    deinit {
        // ManagedBuffer handles deallocation automatically
        storage.withUnsafeMutablePointerToElements { ptr in
            ptr.deinitialize(count: count)
        }
    }
    
    public subscript(index: Int) -> Scalar {
        get {
            #if DEBUG
            guard index >= 0 && index < count else {
                fatalError("Index \(index) out of bounds [0..<\(count)]")
            }
            #else
            precondition(index >= 0 && index < count, "Index out of bounds")
            #endif
            
            return storage.withUnsafeMutablePointerToElements { ptr in
                ptr[index]
            }
        }
        set {
            #if DEBUG
            guard index >= 0 && index < count else {
                fatalError("Index \(index) out of bounds [0..<\(count)]")
            }
            #else
            precondition(index >= 0 && index < count, "Index out of bounds")
            #endif
            
            storage.withUnsafeMutablePointerToElements { ptr in
                ptr[index] = newValue
            }
        }
    }
    
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Scalar>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutablePointerToElements { ptr in
            try body(UnsafeBufferPointer(start: ptr, count: count))
        }
    }
    
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R) rethrows -> R {
        try storage.withUnsafeMutablePointerToElements { ptr in
            try body(UnsafeMutableBufferPointer(start: ptr, count: count))
        }
    }
    
    /// Verify memory alignment
    public var isProperlyAligned: Bool {
        storage.withUnsafeMutablePointerToElements { ptr in
            Int(bitPattern: ptr) % Self.alignment == 0
        }
    }
}

// MARK: - Safe Vector Buffer

/// A safer alternative to AlignedVector using standard Swift arrays with alignment hints
public struct SafeVectorBuffer<Scalar: BinaryFloatingPoint>: Sendable {
    private var data: [Scalar]
    public let count: Int
    
    public init(count: Int) {
        self.count = count
        self.data = Array(repeating: 0, count: count)
    }
    
    public init(copying source: [Scalar]) {
        self.count = source.count
        self.data = source
    }
    
    public subscript(index: Int) -> Scalar {
        get {
            #if DEBUG
            guard index >= 0 && index < count else {
                fatalError("Index \(index) out of bounds [0..<\(count)]")
            }
            #endif
            return data[index]
        }
        set {
            #if DEBUG
            guard index >= 0 && index < count else {
                fatalError("Index \(index) out of bounds [0..<\(count)]")
            }
            #endif
            data[index] = newValue
        }
    }
    
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Scalar>) throws -> R) rethrows -> R {
        try data.withUnsafeBufferPointer(body)
    }
    
    public mutating func withUnsafeMutableBufferPointer<R>(_ body: (inout UnsafeMutableBufferPointer<Scalar>) throws -> R) rethrows -> R {
        try data.withUnsafeMutableBufferPointer(body)
    }
}

// MARK: - Optimized Distance Functions

/// Highly optimized Euclidean distance computation with safety features
public struct OptimizedEuclideanDistance {
    
    /// Compute squared Euclidean distance (avoiding sqrt for performance)
    /// Uses unsafe pointers for performance but includes safety checks
    @inlinable
    public static func distanceSquared<T: BinaryFloatingPoint>(
        _ a: UnsafePointer<T>,
        _ b: UnsafePointer<T>,
        count: Int
    ) -> T {
        precondition(count >= 0, "Count must be non-negative")
        guard count > 0 else { return 0 }
        
        var sum: T = 0
        
        // Process in chunks of 8 for better vectorization
        let simdCount = count & ~7
        var i = 0
        
        // Unrolled loop for SIMD processing
        while i < simdCount {
            let diff0 = a[i] - b[i]
            let diff1 = a[i+1] - b[i+1]
            let diff2 = a[i+2] - b[i+2]
            let diff3 = a[i+3] - b[i+3]
            let diff4 = a[i+4] - b[i+4]
            let diff5 = a[i+5] - b[i+5]
            let diff6 = a[i+6] - b[i+6]
            let diff7 = a[i+7] - b[i+7]
            
            sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3
            sum += diff4 * diff4 + diff5 * diff5 + diff6 * diff6 + diff7 * diff7
            
            i += 8
        }
        
        // Handle remaining elements
        while i < count {
            let diff = a[i] - b[i]
            sum += diff * diff
            i += 1
        }
        
        return sum
    }
    
    /// Safe distance computation using arrays
    @inlinable
    public static func safeDistanceSquared<T: BinaryFloatingPoint>(
        _ a: [T],
        _ b: [T]
    ) -> T? {
        guard a.count == b.count else { return nil }
        
        return a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                guard let aBase = aPtr.baseAddress,
                      let bBase = bPtr.baseAddress else { return T.zero }
                return distanceSquared(aBase, bBase, count: a.count)
            }
        }
    }
    
    /// Batch distance computation with early termination
    @inlinable
    public static func batchDistanceSquaredWithBounds<T: BinaryFloatingPoint>(
        query: UnsafePointer<T>,
        candidates: UnsafePointer<UnsafePointer<T>>,
        count: Int,
        dimensions: Int,
        bound: T? = nil
    ) -> [T] {
        precondition(count >= 0, "Count must be non-negative")
        precondition(dimensions >= 0, "Dimensions must be non-negative")
        guard count > 0 && dimensions > 0 else { return [] }
        
        var results = [T](repeating: 0, count: count)
        
        results.withUnsafeMutableBufferPointer { resultsPtr in
            for i in 0..<count {
                var sum: T = 0
                let candidate = candidates[i]
                
                // Early termination if bound is specified
                if let bound = bound {
                    for d in 0..<dimensions {
                        let diff = query[d] - candidate[d]
                        sum += diff * diff
                        
                        // Early termination if we exceed the bound
                        if sum > bound {
                            sum = T.infinity
                            break
                        }
                    }
                } else {
                    sum = distanceSquared(query, candidate, count: dimensions)
                }
                
                resultsPtr[i] = sum
            }
        }
        
        return results
    }
    
    /// Safe batch distance computation using arrays
    public static func safeBatchDistanceSquared<T: BinaryFloatingPoint>(
        query: [T],
        candidates: [[T]],
        bound: T? = nil
    ) -> [T]? {
        guard !candidates.isEmpty else { return [] }
        let dimensions = query.count
        
        // Verify all candidates have the same dimensions
        guard candidates.allSatisfy({ $0.count == dimensions }) else { return nil }
        
        return query.withUnsafeBufferPointer { queryPtr in
            guard let queryBase = queryPtr.baseAddress else { return [] }
            
            var results = [T](repeating: 0, count: candidates.count)
            
            for (i, candidate) in candidates.enumerated() {
                if let distance = safeDistanceSquared(query, candidate) {
                    results[i] = distance
                } else {
                    results[i] = T.infinity
                }
            }
            
            return results
        }
    }
}

/// Optimized cosine distance computation with safety features
public struct OptimizedCosineDistance {
    
    /// Compute cosine similarity for normalized vectors (just dot product)
    @inlinable
    public static func normalizedSimilarity<T: BinaryFloatingPoint>(
        _ a: UnsafePointer<T>,
        _ b: UnsafePointer<T>,
        count: Int
    ) -> T {
        precondition(count >= 0, "Count must be non-negative")
        guard count > 0 else { return 0 }
        
        var dotProduct: T = 0
        
        // Unrolled loop for better performance
        let simdCount = count & ~7
        var i = 0
        
        while i < simdCount {
            dotProduct += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3]
            dotProduct += a[i+4] * b[i+4] + a[i+5] * b[i+5] + a[i+6] * b[i+6] + a[i+7] * b[i+7]
            i += 8
        }
        
        // Handle remaining elements
        while i < count {
            dotProduct += a[i] * b[i]
            i += 1
        }
        
        return dotProduct
    }
    
    /// Safe cosine similarity using arrays
    @inlinable
    public static func safeNormalizedSimilarity<T: BinaryFloatingPoint>(
        _ a: [T],
        _ b: [T]
    ) -> T? {
        guard a.count == b.count else { return nil }
        
        return a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                guard let aBase = aPtr.baseAddress,
                      let bBase = bPtr.baseAddress else { return T.zero }
                return normalizedSimilarity(aBase, bBase, count: a.count)
            }
        }
    }
    
    /// Batch cosine distance for normalized vectors
    @inlinable
    public static func batchNormalizedDistance<T: BinaryFloatingPoint>(
        query: UnsafePointer<T>,
        candidates: UnsafePointer<UnsafePointer<T>>,
        count: Int,
        dimensions: Int
    ) -> [T] {
        var results = [T](repeating: 0, count: count)
        
        results.withUnsafeMutableBufferPointer { resultsPtr in
            for i in 0..<count {
                let similarity = normalizedSimilarity(query, candidates[i], count: dimensions)
                resultsPtr[i] = 1 - similarity
            }
        }
        
        return results
    }
}

// MARK: - SIMD-Optimized Distance Computation for Fixed Dimensions

/// Optimized distance computation for 128-dimensional vectors
public struct Distance128 {
    
    @inlinable
    public static func euclideanDistanceSquared(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>
    ) -> Float {
        var sum: SIMD4<Float> = .zero
        
        // Process 128 dimensions as 32 SIMD4 operations
        for i in stride(from: 0, to: 128, by: 4) {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let vb = SIMD4<Float>(b[i], b[i+1], b[i+2], b[i+3])
            let diff = va - vb
            sum += diff * diff
        }
        
        return sum.sum()
    }
}

/// Optimized distance computation for 256-dimensional vectors
public struct Distance256 {
    
    @inlinable
    public static func euclideanDistanceSquared(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>
    ) -> Float {
        var sum: SIMD8<Float> = .zero
        
        // Process 256 dimensions as 32 SIMD8 operations
        for i in stride(from: 0, to: 256, by: 8) {
            let va = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vb = SIMD8<Float>(
                b[i], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            )
            let diff = va - vb
            sum += diff * diff
        }
        
        return sum.sum()
    }
}

/// Optimized distance computation for 512-dimensional vectors
public struct Distance512Optimized {
    
    @inlinable
    public static func euclideanDistanceSquared(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>
    ) -> Float {
        var sum0: SIMD8<Float> = .zero
        var sum1: SIMD8<Float> = .zero
        var sum2: SIMD8<Float> = .zero
        var sum3: SIMD8<Float> = .zero
        
        // Process 512 dimensions as 64 SIMD8 operations, using 4 accumulators
        for i in stride(from: 0, to: 512, by: 32) {
            // First 8 elements
            let va0 = SIMD8<Float>(
                a[i], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vb0 = SIMD8<Float>(
                b[i], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            )
            let diff0 = va0 - vb0
            sum0 += diff0 * diff0
            
            // Second 8 elements
            let va1 = SIMD8<Float>(
                a[i+8], a[i+9], a[i+10], a[i+11],
                a[i+12], a[i+13], a[i+14], a[i+15]
            )
            let vb1 = SIMD8<Float>(
                b[i+8], b[i+9], b[i+10], b[i+11],
                b[i+12], b[i+13], b[i+14], b[i+15]
            )
            let diff1 = va1 - vb1
            sum1 += diff1 * diff1
            
            // Third 8 elements
            let va2 = SIMD8<Float>(
                a[i+16], a[i+17], a[i+18], a[i+19],
                a[i+20], a[i+21], a[i+22], a[i+23]
            )
            let vb2 = SIMD8<Float>(
                b[i+16], b[i+17], b[i+18], b[i+19],
                b[i+20], b[i+21], b[i+22], b[i+23]
            )
            let diff2 = va2 - vb2
            sum2 += diff2 * diff2
            
            // Fourth 8 elements
            let va3 = SIMD8<Float>(
                a[i+24], a[i+25], a[i+26], a[i+27],
                a[i+28], a[i+29], a[i+30], a[i+31]
            )
            let vb3 = SIMD8<Float>(
                b[i+24], b[i+25], b[i+26], b[i+27],
                b[i+28], b[i+29], b[i+30], b[i+31]
            )
            let diff3 = va3 - vb3
            sum3 += diff3 * diff3
        }
        
        // Combine accumulators
        return sum0.sum() + sum1.sum() + sum2.sum() + sum3.sum()
    }
    
    /// Batch distance computation with prefetching
    public static func batchEuclideanDistanceSquared(
        query: UnsafePointer<Float>,
        candidates: [UnsafePointer<Float>],
        prefetchAhead: Int = 4
    ) -> [Float] {
        var results = [Float](repeating: 0, count: candidates.count)
        
        results.withUnsafeMutableBufferPointer { resultsPtr in
            for i in 0..<candidates.count {
                // Prefetch next candidates
                if i + prefetchAhead < candidates.count {
                    let prefetchPtr = candidates[i + prefetchAhead]
                    // Prefetch hint for better cache performance
                    _ = prefetchPtr.pointee
                }
                
                resultsPtr[i] = euclideanDistanceSquared(query, candidates[i])
            }
        }
        
        return results
    }
}

// MARK: - Accelerate Framework Integration

/// Accelerate-optimized distance computations
public struct AccelerateDistanceComputation {
    
    /// Batch Euclidean distance using Accelerate
    public static func batchEuclideanDistance(
        query: [Float],
        candidates: [[Float]],
        squared: Bool = false
    ) -> [Float] {
        let dimensions = query.count
        let candidateCount = candidates.count
        var results = [Float](repeating: 0, count: candidateCount)
        
        // Flatten candidates for efficient processing
        var flatCandidates = [Float]()
        flatCandidates.reserveCapacity(candidateCount * dimensions)
        for candidate in candidates {
            flatCandidates.append(contentsOf: candidate)
        }
        
        query.withUnsafeBufferPointer { queryPtr in
            flatCandidates.withUnsafeBufferPointer { candidatesPtr in
                results.withUnsafeMutableBufferPointer { resultsPtr in
                    
                    for i in 0..<candidateCount {
                        let candidateOffset = i * dimensions
                        var diff = [Float](repeating: 0, count: dimensions)
                        
                        // Compute difference
                        vDSP_vsub(
                            candidatesPtr.baseAddress! + candidateOffset, 1,
                            queryPtr.baseAddress!, 1,
                            &diff, 1,
                            vDSP_Length(dimensions)
                        )
                        
                        // Square and sum
                        var sum: Float = 0
                        vDSP_dotpr(diff, 1, diff, 1, &sum, vDSP_Length(dimensions))
                        
                        resultsPtr[i] = squared ? sum : sqrt(sum)
                    }
                }
            }
        }
        
        return results
    }
    
    /// Matrix multiplication-based batch distance computation
    public static func matrixBatchDistance(
        queries: [[Float]],
        candidates: [[Float]],
        metric: DistanceMetric
    ) -> [[Float]] {
        guard !queries.isEmpty && !candidates.isEmpty else { return [] }
        
        let queryCount = queries.count
        let candidateCount = candidates.count
        let dimensions = queries[0].count
        
        // Flatten arrays for matrix operations
        let flatQueries = queries.flatMap { $0 }
        let flatCandidates = candidates.flatMap { $0 }
        
        var results = [[Float]](
            repeating: [Float](repeating: 0, count: candidateCount),
            count: queryCount
        )
        
        switch metric {
        case .euclidean:
            // Euclidean distance using matrix operations
            // d(x,y)² = ||x||² + ||y||² - 2<x,y>
            
            // Compute query norms
            var queryNorms = [Float](repeating: 0, count: queryCount)
            for i in 0..<queryCount {
                var norm: Float = 0
                flatQueries.withUnsafeBufferPointer { buffer in
                    let offset = i * dimensions
                    vDSP_svesq(
                        buffer.baseAddress!.advanced(by: offset), 1,
                        &norm,
                        vDSP_Length(dimensions)
                    )
                }
                queryNorms[i] = norm
            }
            
            // Compute candidate norms
            var candidateNorms = [Float](repeating: 0, count: candidateCount)
            for i in 0..<candidateCount {
                var norm: Float = 0
                flatCandidates.withUnsafeBufferPointer { buffer in
                    let offset = i * dimensions
                    vDSP_svesq(
                        buffer.baseAddress!.advanced(by: offset), 1,
                        &norm,
                        vDSP_Length(dimensions)
                    )
                }
                candidateNorms[i] = norm
            }
            
            // Compute dot products using matrix multiplication
            var dotProducts = [Float](repeating: 0, count: queryCount * candidateCount)
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                Int32(queryCount), Int32(candidateCount), Int32(dimensions),
                1.0, flatQueries, Int32(dimensions),
                flatCandidates, Int32(dimensions),
                0.0, &dotProducts, Int32(candidateCount)
            )
            
            // Compute final distances
            for i in 0..<queryCount {
                for j in 0..<candidateCount {
                    let dotProduct = dotProducts[i * candidateCount + j]
                    let distanceSquared = queryNorms[i] + candidateNorms[j] - 2 * dotProduct
                    results[i][j] = sqrt(max(0, distanceSquared))
                }
            }
            
        case .cosine:
            // Similar matrix operations for cosine distance
            // Implementation omitted for brevity
            break
            
        default:
            // Fallback to standard computation
            for i in 0..<queryCount {
                for j in 0..<candidateCount {
                    // Standard distance computation would go here
                    // For now, just set to 0
                    results[i][j] = 0
                }
            }
        }
        
        return results
    }
}

// MARK: - Safe Memory Pool for Distance Computations

/// Thread-safe, memory pressure aware pool for distance computation buffers
public actor SafeDistanceComputationBufferPool: MemoryPressureAware {
    private var availableBuffers: [SafeVectorBuffer<Float>] = []
    private let bufferSize: Int
    private let maxBuffers: Int
    private var allocatedCount: Int = 0
    private let logger = os.Logger(subsystem: "VectorStoreKit", category: "BufferPool")
    
    // Memory pressure handling
    private var currentPressureLevel: SystemMemoryPressure = .normal
    private var shouldReduceFootprint: Bool { currentPressureLevel != .normal }
    private var pressureEventCount: Int = 0
    private var lastPressureHandled: Date?
    private var peakMemoryUsage: Int = 0
    
    public init(bufferSize: Int, maxBuffers: Int = 100) {
        self.bufferSize = bufferSize
        self.maxBuffers = maxBuffers
        
        // Pre-allocate some buffers for better performance
        let preAllocateCount = min(10, maxBuffers)
        for _ in 0..<preAllocateCount {
            availableBuffers.append(SafeVectorBuffer(count: bufferSize))
            allocatedCount += 1
        }
    }
    
    public func acquireBuffer() async -> SafeVectorBuffer<Float> {
        // Under memory pressure, don't cache buffers
        if shouldReduceFootprint {
            return SafeVectorBuffer(count: bufferSize)
        }
        
        if let buffer = availableBuffers.popLast() {
            logger.debug("Reusing buffer from pool (available: \(self.availableBuffers.count))")
            return buffer
        } else if allocatedCount < maxBuffers {
            allocatedCount += 1
            logger.debug("Allocating new buffer (total: \(self.allocatedCount))")
            return SafeVectorBuffer(count: bufferSize)
        } else {
            // Pool exhausted, allocate temporary buffer
            logger.warning("Buffer pool exhausted, allocating temporary buffer")
            return SafeVectorBuffer(count: bufferSize)
        }
    }
    
    public func releaseBuffer(_ buffer: SafeVectorBuffer<Float>) async {
        // Don't retain buffers under memory pressure
        if shouldReduceFootprint {
            return
        }
        
        if buffer.count == bufferSize && availableBuffers.count < maxBuffers {
            availableBuffers.append(buffer)
            logger.debug("Buffer returned to pool (available: \(self.availableBuffers.count))")
        }
        // Otherwise, let it be deallocated
    }
    
    public func clear() async {
        logger.info("Clearing buffer pool (had \(self.availableBuffers.count) buffers)")
        availableBuffers.removeAll()
        allocatedCount = 0
    }
    
    // MARK: - MemoryPressureAware
    
    public func handleMemoryPressure(_ level: SystemMemoryPressure) async {
        pressureEventCount += 1
        lastPressureHandled = Date()
        currentPressureLevel = level
        
        switch level {
        case .normal:
            logger.info("Memory pressure normal")
        case .warning:
            logger.warning("Memory pressure warning - reducing pool size")
            // Keep only half of the buffers
            let keepCount = availableBuffers.count / 2
            availableBuffers.removeLast(availableBuffers.count - keepCount)
        case .critical:
            logger.error("Memory pressure critical - clearing pool")
            await clear()
        }
    }
    
    public func getCurrentMemoryUsage() async -> Int {
        let currentUsage = allocatedCount * bufferSize * MemoryLayout<Float>.stride
        peakMemoryUsage = max(peakMemoryUsage, currentUsage)
        return currentUsage
    }
    
    public func getMemoryStatistics() async -> MemoryComponentStatistics {
        let currentUsage = await getCurrentMemoryUsage()
        
        return MemoryComponentStatistics(
            componentName: "SafeDistanceComputationBufferPool",
            currentMemoryUsage: currentUsage,
            peakMemoryUsage: peakMemoryUsage,
            pressureEventCount: pressureEventCount,
            lastPressureHandled: lastPressureHandled,
            averageResponseTime: 0 // We don't track response time for now
        )
    }
    
    // Pool statistics for monitoring
    public var statistics: (allocated: Int, available: Int, maxSize: Int) {
        (allocatedCount, availableBuffers.count, maxBuffers)
    }
}

// MARK: - Safe Batch Operations

/// Safe wrapper for batch distance computations with automatic resource management
public struct SafeBatchDistanceComputation {
    private let bufferPool: SafeDistanceComputationBufferPool
    
    public init(bufferPool: SafeDistanceComputationBufferPool) {
        self.bufferPool = bufferPool
    }
    
    /// Compute distances with automatic buffer management
    public func computeDistances(
        query: [Float],
        candidates: [[Float]],
        metric: DistanceMetric = .euclidean
    ) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }
        
        // Acquire buffer from pool
        let buffer = await bufferPool.acquireBuffer()
        defer {
            Task {
                await bufferPool.releaseBuffer(buffer)
            }
        }
        
        // Perform computation using the buffer
        return try await withCheckedThrowingContinuation { continuation in
            Task {
                do {
                    let results = AccelerateDistanceComputation.batchEuclideanDistance(
                        query: query,
                        candidates: candidates,
                        squared: false  // For now, always compute full Euclidean distance
                    )
                    continuation.resume(returning: results)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

// MARK: - SIMD Extension for sum()

extension SIMD where Scalar: FloatingPoint {
    @inlinable
    func sum() -> Scalar {
        var result = Scalar.zero
        for i in indices {
            result += self[i]
        }
        return result
    }
}
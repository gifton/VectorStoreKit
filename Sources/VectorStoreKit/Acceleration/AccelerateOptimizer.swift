// VectorStoreKit: Accelerate Framework Optimizer
//
// CPU-optimized operations using Apple's Accelerate framework
// Provides SIMD-optimized implementations for vector and matrix operations
//
// Design principles:
// - Maximum utilization of Accelerate/vDSP functions
// - Cache-friendly memory access patterns
// - Vectorized operations using SIMD
// - Minimal memory allocations

import Foundation
import Accelerate
import simd

/// CPU-optimized operations using Accelerate framework
public actor AccelerateOptimizer {
    
    // MARK: - Properties
    
    private let simdLanes: Int
    private var scratchBuffers: ScratchBufferPool
    
    // MARK: - Initialization
    
    public init() {
        // Detect SIMD capabilities
        #if arch(arm64)
        self.simdLanes = 4 // NEON on Apple Silicon
        #else
        self.simdLanes = 8 // AVX on Intel
        #endif
        
        self.scratchBuffers = ScratchBufferPool()
    }
    
    // MARK: - Distance Computations
    
    /// Compute distances using optimized Accelerate functions
    public func computeDistances<Vector: SIMD & Sendable>(
        query: Vector,
        candidates: [Vector],
        metric: DistanceMetric
    ) async -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        
        let dimension = Vector.scalarCount
        var results = [Float](repeating: 0, count: candidates.count)
        
        // Convert query to Float array for Accelerate
        let queryFloats = vectorToFloatArray(query)
        
        switch metric {
        case .euclidean:
            await computeEuclideanDistances(
                query: queryFloats,
                candidates: candidates,
                dimension: dimension,
                results: &results
            )
            
        case .cosine:
            await computeCosineDistances(
                query: queryFloats,
                candidates: candidates,
                dimension: dimension,
                results: &results
            )
            
        case .manhattan:
            await computeManhattanDistances(
                query: queryFloats,
                candidates: candidates,
                dimension: dimension,
                results: &results
            )
            
        case .dotProduct:
            computeDotProductDistances(
                query: queryFloats,
                candidates: candidates,
                dimension: dimension,
                results: &results
            )
            
        default:
            // Fallback to basic implementation
            for (index, candidate) in candidates.enumerated() {
                results[index] = computeSingleDistance(query, candidate, metric: metric)
            }
        }
        
        return results
    }
    
    /// Batch compute distances with parallel processing
    public func batchComputeDistances<Vector: SIMD & Sendable>(
        queries: [Vector],
        candidates: [Vector],
        metric: DistanceMetric,
        topK: Int?
    ) async -> [[Float]] where Vector.Scalar: BinaryFloatingPoint {
        
        var results: [[Float]] = []
        results.reserveCapacity(queries.count)
        
        for query in queries {
            let distances = await computeDistances(
                query: query,
                candidates: candidates,
                metric: metric
            )
            
            if let k = topK {
                results.append(selectTopK(distances: distances, k: k))
            } else {
                results.append(distances)
            }
        }
        
        return results
    }
    
    // MARK: - Top-K Selection
    
    /// Select top-K smallest distances efficiently
    public func selectTopK(distances: [Float], k: Int) -> [Float] {
        guard k < distances.count else { return distances }
        
        // For small k, use partial sort
        if k < 10 {
            var indices = Array(0..<distances.count)
            indices.sort { distances[$0] < distances[$1] }
            return indices.prefix(k).map { distances[$0] }
        }
        
        // For larger k, use heap-based selection
        var heap = MinHeap<(Float, Int)>(capacity: k) { $0.0 < $1.0 }
        
        for (index, distance) in distances.enumerated() {
            if heap.count < k {
                heap.insert((distance, index))
            } else if distance < heap.peek()!.0 {
                heap.extractMin()
                heap.insert((distance, index))
            }
        }
        
        return heap.sorted().map { $0.0 }
    }
    
    // MARK: - Matrix Operations
    
    /// Matrix multiplication using Accelerate's BLAS
    public func matrixMultiply(
        matrixA: [[Float]],
        matrixB: [[Float]]
    ) -> [[Float]] {
        
        let m = matrixA.count
        let k = matrixA[0].count
        let n = matrixB[0].count
        
        // Flatten matrices for BLAS (column-major order)
        var flatA = [Float]()
        var flatB = [Float]()
        flatA.reserveCapacity(m * k)
        flatB.reserveCapacity(k * n)
        
        // Convert to column-major for BLAS
        for col in 0..<k {
            for row in 0..<m {
                flatA.append(matrixA[row][col])
            }
        }
        
        for col in 0..<n {
            for row in 0..<k {
                flatB.append(matrixB[row][col])
            }
        }
        
        var flatC = [Float](repeating: 0, count: m * n)
        
        // Use BLAS gemm for matrix multiplication
        // C = alpha * A * B + beta * C
        cblas_sgemm(
            CblasColMajor,      // Column-major storage
            CblasNoTrans,       // Don't transpose A
            CblasNoTrans,       // Don't transpose B
            Int32(m),           // Rows of A and C
            Int32(n),           // Columns of B and C
            Int32(k),           // Columns of A, rows of B
            1.0,                // alpha
            flatA,              // Matrix A
            Int32(m),           // Leading dimension of A
            flatB,              // Matrix B
            Int32(k),           // Leading dimension of B
            0.0,                // beta
            &flatC,             // Matrix C (output)
            Int32(m)            // Leading dimension of C
        )
        
        // Convert back to row-major 2D array
        var result = [[Float]](repeating: [Float](repeating: 0, count: n), count: m)
        for row in 0..<m {
            for col in 0..<n {
                result[row][col] = flatC[col * m + row]
            }
        }
        
        return result
    }
    
    // MARK: - Quantization
    
    /// Quantize vectors using SIMD operations
    public func quantizeVectors<Vector: SIMD & Sendable>(
        vectors: [Vector],
        scheme: QuantizationScheme,
        parameters: QuantizationParameters
    ) -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        switch scheme {
        case .scalar:
            return scalarQuantize(vectors: vectors, parameters: parameters)
            
        case .product:
            return productQuantize(vectors: vectors, parameters: parameters)
            
        case .binary:
            return binaryQuantize(vectors: vectors, parameters: parameters)
            
        default:
            // Basic quantization fallback
            return vectors.map { vector in
                let data = vectorToData(vector)
                return QuantizedVector(
                    codes: Array(data),
                    metadata: QuantizationMetadata(
                        scheme: scheme,
                        originalDimensions: vector.scalarCount,
                        compressionRatio: 1.0
                    )
                )
            }
        }
    }
    
    // MARK: - Private Distance Implementations
    
    private func computeEuclideanDistances<Vector: SIMD>(
        query: [Float],
        candidates: [Vector],
        dimension: Int,
        results: inout [Float]
    ) async where Vector.Scalar: BinaryFloatingPoint {
        
        let scratchBuffer = await scratchBuffers.getBuffer(size: dimension)
        
        for (index, candidate) in candidates.enumerated() {
            let candidateFloats = vectorToFloatArray(candidate)
            
            // Compute difference: scratch = candidate - query
            vDSP_vsub(
                query, 1,
                candidateFloats, 1,
                scratchBuffer.baseAddress!, 1,
                vDSP_Length(dimension)
            )
            
            // Compute squared distance
            var squaredDistance: Float = 0
            vDSP_svesq(
                scratchBuffer.baseAddress!, 1,
                &squaredDistance,
                vDSP_Length(dimension)
            )
            
            results[index] = sqrt(squaredDistance)
        }
        
        await scratchBuffers.returnBuffer(scratchBuffer)
    }
    
    private func computeCosineDistances<Vector: SIMD>(
        query: [Float],
        candidates: [Vector],
        dimension: Int,
        results: inout [Float]
    ) async where Vector.Scalar: BinaryFloatingPoint {
        
        // Precompute query magnitude
        var queryMagnitude: Float = 0
        vDSP_svesq(query, 1, &queryMagnitude, vDSP_Length(dimension))
        queryMagnitude = sqrt(queryMagnitude)
        
        for (index, candidate) in candidates.enumerated() {
            let candidateFloats = vectorToFloatArray(candidate)
            
            // Compute dot product
            var dotProduct: Float = 0
            vDSP_dotpr(
                query, 1,
                candidateFloats, 1,
                &dotProduct,
                vDSP_Length(dimension)
            )
            
            // Compute candidate magnitude
            var candidateMagnitude: Float = 0
            vDSP_svesq(candidateFloats, 1, &candidateMagnitude, vDSP_Length(dimension))
            candidateMagnitude = sqrt(candidateMagnitude)
            
            // Cosine similarity = dot / (||a|| * ||b||)
            let similarity = dotProduct / (queryMagnitude * candidateMagnitude + 1e-8)
            
            // Convert to distance
            results[index] = 1.0 - similarity
        }
    }
    
    private func computeManhattanDistances<Vector: SIMD>(
        query: [Float],
        candidates: [Vector],
        dimension: Int,
        results: inout [Float]
    ) async where Vector.Scalar: BinaryFloatingPoint {
        
        let scratchBuffer = await scratchBuffers.getBuffer(size: dimension)
        
        for (index, candidate) in candidates.enumerated() {
            let candidateFloats = vectorToFloatArray(candidate)
            
            // Compute absolute differences
            vDSP_vsub(
                query, 1,
                candidateFloats, 1,
                scratchBuffer.baseAddress!, 1,
                vDSP_Length(dimension)
            )
            
            vDSP_vabs(
                scratchBuffer.baseAddress!, 1,
                scratchBuffer.baseAddress!, 1,
                vDSP_Length(dimension)
            )
            
            // Sum absolute differences
            var manhattanDistance: Float = 0
            vDSP_sve(
                scratchBuffer.baseAddress!, 1,
                &manhattanDistance,
                vDSP_Length(dimension)
            )
            
            results[index] = manhattanDistance
        }
        
        await scratchBuffers.returnBuffer(scratchBuffer)
    }
    
    private func computeDotProductDistances<Vector: SIMD>(
        query: [Float],
        candidates: [Vector],
        dimension: Int,
        results: inout [Float]
    ) where Vector.Scalar: BinaryFloatingPoint {
        
        for (index, candidate) in candidates.enumerated() {
            let candidateFloats = vectorToFloatArray(candidate)
            
            var dotProduct: Float = 0
            vDSP_dotpr(
                query, 1,
                candidateFloats, 1,
                &dotProduct,
                vDSP_Length(dimension)
            )
            
            // Negative for similarity to distance conversion
            results[index] = -dotProduct
        }
    }
    
    // MARK: - Quantization Implementations
    
    private func scalarQuantize<Vector: SIMD>(
        vectors: [Vector],
        parameters: QuantizationParameters
    ) -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        let bits = parameters.precision
        let maxValue = Float(1 << bits) - 1
        
        return vectors.map { vector in
            let floats = vectorToFloatArray(vector)
            
            // Find min/max for normalization
            var minValue: Float = 0
            var maxValue: Float = 0
            vDSP_minv(floats, 1, &minValue, vDSP_Length(floats.count))
            vDSP_maxv(floats, 1, &maxValue, vDSP_Length(floats.count))
            
            let range = maxValue - minValue
            let scale = range > 0 ? Float(1 << bits - 1) / range : 1.0
            
            // Quantize values
            var quantized = [UInt8](repeating: 0, count: floats.count)
            
            for (i, value) in floats.enumerated() {
                let normalized = (value - minValue) * scale
                quantized[i] = UInt8(min(max(normalized, 0), 255))
            }
            
            // Store scale and offset for dequantization
            var metadata = Data()
            metadata.append(contentsOf: withUnsafeBytes(of: scale) { Data($0) })
            metadata.append(contentsOf: withUnsafeBytes(of: minValue) { Data($0) })
            metadata.append(contentsOf: quantized)
            
            return QuantizedVector(
                codes: quantized,
                metadata: QuantizationMetadata(
                    scheme: .scalar(bits: parameters.precision),
                    originalDimensions: vector.scalarCount,
                    compressionRatio: Float(vector.scalarCount * MemoryLayout<Float>.size) / Float(quantized.count + 8)
                )
            )
        }
    }
    
    private func productQuantize<Vector: SIMD>(
        vectors: [Vector],
        parameters: QuantizationParameters
    ) -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        // Product quantization implementation
        // This is a placeholder - full implementation would include:
        // 1. Training codebooks using k-means
        // 2. Encoding vectors using nearest codebook entries
        // 3. Storing codebook indices
        
        // For now, fall back to scalar quantization
        return scalarQuantize(vectors: vectors, parameters: parameters)
    }
    
    private func binaryQuantize<Vector: SIMD>(
        vectors: [Vector],
        parameters: QuantizationParameters
    ) -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        return vectors.map { vector in
            let floats = vectorToFloatArray(vector)
            let dimension = floats.count
            
            // Binary quantization: 1 bit per dimension
            let bytesNeeded = (dimension + 7) / 8
            var binaryData = Data(count: bytesNeeded)
            
            binaryData.withUnsafeMutableBytes { bytes in
                let bytePtr = bytes.bindMemory(to: UInt8.self).baseAddress!
                
                for i in 0..<dimension {
                    if floats[i] > 0 {
                        let byteIndex = i / 8
                        let bitIndex = i % 8
                        bytePtr[byteIndex] |= (1 << bitIndex)
                    }
                }
            }
            
            return QuantizedVector(
                codes: Array(binaryData),
                metadata: QuantizationMetadata(
                    scheme: .binary,
                    originalDimensions: dimension,
                    compressionRatio: Float(dimension * MemoryLayout<Float>.size) / Float(binaryData.count)
                )
            )
        }
    }
    
    // MARK: - Helper Methods
    
    private func vectorToFloatArray<Vector: SIMD>(_ vector: Vector) -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        var result = [Float](repeating: 0, count: vector.scalarCount)
        for i in 0..<vector.scalarCount {
            result[i] = Float(vector[i])
        }
        return result
    }
    
    private func vectorToData<Vector: SIMD>(_ vector: Vector) -> Data where Vector.Scalar: BinaryFloatingPoint {
        var data = Data()
        for i in 0..<vector.scalarCount {
            let value = Float(vector[i])
            data.append(contentsOf: withUnsafeBytes(of: value) { Data($0) })
        }
        return data
    }
    
    private func computeSingleDistance<Vector: SIMD>(
        _ a: Vector,
        _ b: Vector,
        metric: DistanceMetric
    ) -> Float where Vector.Scalar: BinaryFloatingPoint {
        
        switch metric {
        case .euclidean:
            let diff = a - b
            return Float(sqrt((diff * diff).sum()))
            
        case .cosine:
            let dot = (a * b).sum()
            let magA = sqrt((a * a).sum())
            let magB = sqrt((b * b).sum())
            return 1.0 - Float(dot) / (Float(magA) * Float(magB) + 1e-8)
            
        case .manhattan:
            return (0..<a.scalarCount).reduce(Float(0)) { sum, i in
                sum + abs(Float(a[i]) - Float(b[i]))
            }
            
        case .dotProduct:
            return Float(-(a * b).sum())
            
        default:
            return Float.infinity
        }
    }
}

// MARK: - Scratch Buffer Pool

/// Pool of reusable scratch buffers for Accelerate operations
private actor ScratchBufferPool {
    private var availableBuffers: [Int: [UnsafeMutableBufferPointer<Float>]] = [:]
    private let maxBuffersPerSize = 10
    
    func getBuffer(size: Int) -> UnsafeMutableBufferPointer<Float> {
        if let buffer = availableBuffers[size]?.popLast() {
            return buffer
        }
        
        // Allocate new buffer
        let pointer = UnsafeMutablePointer<Float>.allocate(capacity: size)
        return UnsafeMutableBufferPointer(start: pointer, count: size)
    }
    
    func returnBuffer(_ buffer: UnsafeMutableBufferPointer<Float>) {
        let size = buffer.count
        
        if availableBuffers[size] == nil {
            availableBuffers[size] = []
        }
        
        if availableBuffers[size]!.count < maxBuffersPerSize {
            availableBuffers[size]!.append(buffer)
        } else {
            // Deallocate excess buffer
            buffer.baseAddress?.deallocate()
        }
    }
    
    deinit {
        // Clean up all buffers
        for (_, buffers) in availableBuffers {
            for buffer in buffers {
                buffer.baseAddress?.deallocate()
            }
        }
    }
}

// MARK: - SIMD Extensions

extension SIMD where Scalar: BinaryFloatingPoint {
    /// Sum all elements in the vector
    func sum() -> Scalar {
        var result: Scalar = 0
        for i in 0..<scalarCount {
            result += self[i]
        }
        return result
    }
}

// MARK: - Min Heap for Top-K Selection

private struct MinHeap<T> {
    private var elements: [T]
    private let capacity: Int
    private let comparator: (T, T) -> Bool
    
    var count: Int { elements.count }
    
    init(capacity: Int, comparator: @escaping (T, T) -> Bool) {
        self.capacity = capacity
        self.comparator = comparator
        self.elements = []
        elements.reserveCapacity(capacity)
    }
    
    mutating func insert(_ element: T) {
        elements.append(element)
        heapifyUp(from: elements.count - 1)
    }
    
    mutating func extractMin() -> T? {
        guard !elements.isEmpty else { return nil }
        
        if elements.count == 1 {
            return elements.removeLast()
        }
        
        let min = elements[0]
        elements[0] = elements.removeLast()
        heapifyDown(from: 0)
        
        return min
    }
    
    func peek() -> T? {
        elements.first
    }
    
    func sorted() -> [T] {
        elements.sorted(by: comparator)
    }
    
    private mutating func heapifyUp(from index: Int) {
        var childIndex = index
        var parentIndex = (childIndex - 1) / 2
        
        while childIndex > 0 && comparator(elements[childIndex], elements[parentIndex]) {
            elements.swapAt(childIndex, parentIndex)
            childIndex = parentIndex
            parentIndex = (childIndex - 1) / 2
        }
    }
    
    private mutating func heapifyDown(from index: Int) {
        var parentIndex = index
        
        while true {
            let leftChild = 2 * parentIndex + 1
            let rightChild = 2 * parentIndex + 2
            var candidateIndex = parentIndex
            
            if leftChild < elements.count && comparator(elements[leftChild], elements[candidateIndex]) {
                candidateIndex = leftChild
            }
            
            if rightChild < elements.count && comparator(elements[rightChild], elements[candidateIndex]) {
                candidateIndex = rightChild
            }
            
            if candidateIndex == parentIndex {
                return
            }
            
            elements.swapAt(parentIndex, candidateIndex)
            parentIndex = candidateIndex
        }
    }
}
// VectorStoreKit: Product Quantization
//
// Memory-efficient vector compression using product quantization

import Foundation
import simd
import Accelerate


/// Product Quantizer for vector compression
public actor ProductQuantizer {
    
    // MARK: - Properties
    
    public let config: ProductQuantizationConfig
    private var codebooks: [Codebook]
    private var trained: Bool = false
    private let metalCompute: MetalCompute?
    
    /// Single codebook for a segment
    private struct Codebook: Sendable {
        let segmentIndex: Int
        let centroids: [[Float]]  // [codebookSize x segmentDimensions]
        
        func encode(segment: [Float]) -> UInt8 {
            var minDistance = Float.infinity
            var bestCode: UInt8 = 0
            
            for (code, centroid) in centroids.enumerated() {
                var distance: Float = 0
                for i in 0..<segment.count {
                    let diff = segment[i] - centroid[i]
                    distance += diff * diff
                }
                
                if distance < minDistance {
                    minDistance = distance
                    bestCode = UInt8(code)
                }
            }
            
            return bestCode
        }
        
        func decode(code: UInt8) -> [Float] {
            return centroids[Int(code)]
        }
    }
    
    // MARK: - Initialization
    
    public init(
        config: ProductQuantizationConfig,
        metalCompute: MetalCompute? = nil
    ) throws {
        try config.validate()
        self.config = config
        self.codebooks = []
        self.metalCompute = metalCompute
    }
    
    // MARK: - Training
    
    public func train(on vectors: [[Float]]) async throws {
        guard vectors.count >= config.codebookSize * config.segments else {
            throw QuantizationError.insufficientTrainingData(
                required: config.codebookSize * config.segments,
                provided: vectors.count
            )
        }
        
        guard vectors.first?.count == config.dimensions else {
            throw QuantizationError.dimensionMismatch
        }
        
        // Train codebook for each segment
        codebooks = []
        
        for segmentIdx in 0..<config.segments {
            let segmentVectors = extractSegment(
                from: vectors,
                segmentIndex: segmentIdx
            )
            
            let codebook = try await trainSegmentCodebook(
                segmentVectors: segmentVectors,
                segmentIndex: segmentIdx
            )
            
            codebooks.append(codebook)
        }
        
        trained = true
    }
    
    // MARK: - Encoding
    
    public func encode(_ vector: [Float]) async throws -> QuantizedVector {
        guard trained else {
            throw QuantizationError.notTrained
        }
        
        guard vector.count == config.dimensions else {
            throw QuantizationError.dimensionMismatch
        }
        
        // Calculate original norm for reconstruction
        let originalNorm = calculateNorm(vector)
        
        var codes: [UInt8] = []
        
        if metalCompute != nil {
            // GPU-accelerated encoding
            codes = try await metalEncode(vector)
        } else {
            // CPU encoding
            for segmentIdx in 0..<config.segments {
                let segment = extractSegment(from: vector, segmentIndex: segmentIdx)
                let code = codebooks[segmentIdx].encode(segment: segment)
                codes.append(code)
            }
        }
        
        // Store norm as the first 4 bytes of custom data
        var customData = Data()
        customData.append(contentsOf: withUnsafeBytes(of: originalNorm) { Data($0) })
        
        // Create quantization parameters with norm data
        let parameters = QuantizationParameters(
            precision: config.codeSize,
            centroids: config.codebookSize,
            subvectors: config.segments,
            customData: customData
        )
        
        return QuantizedVector(
            originalDimensions: vector.count,
            quantizedData: Data(codes),
            scheme: .product(segments: config.segments, bits: config.codeSize),
            parameters: parameters
        )
    }
    
    public func encodeBatch(_ vectors: [[Float]]) async throws -> [QuantizedVector] {
        guard trained else {
            throw QuantizationError.notTrained
        }
        
        if metalCompute != nil {
            // GPU batch encoding
            return try await metalEncodeBatch(vectors)
        } else {
            // Parallel CPU encoding
            return try await withThrowingTaskGroup(of: QuantizedVector.self) { group in
                for vector in vectors {
                    group.addTask {
                        try await self.encode(vector)
                    }
                }
                
                var results: [QuantizedVector] = []
                for try await result in group {
                    results.append(result)
                }
                return results
            }
        }
    }
    
    // MARK: - Decoding
    
    public func decode(_ quantized: QuantizedVector) async throws -> [Float] {
        guard trained else {
            throw QuantizationError.notTrained
        }
        
        guard quantized.codes.count == config.segments else {
            throw QuantizationError.invalidQuantizedVector
        }
        
        var reconstructed = Array(repeating: Float(0), count: config.dimensions)
        
        for segmentIdx in 0..<config.segments {
            let code = quantized.codes[segmentIdx]
            let segmentVector = codebooks[segmentIdx].decode(code: code)
            
            let startIdx = segmentIdx * config.segmentDimensions
            for i in 0..<config.segmentDimensions {
                reconstructed[startIdx + i] = segmentVector[i]
            }
        }
        
        // Apply norm correction if available
        if let customData = quantized.parameters.customData,
           customData.count >= MemoryLayout<Float>.size {
            // Extract original norm from custom data
            let originalNorm = customData.withUnsafeBytes { $0.load(as: Float.self) }
            
            // Calculate current norm
            let currentNorm = calculateNorm(reconstructed)
            
            // Apply norm correction if current norm is non-zero
            if currentNorm > 1e-6 {
                let scale = originalNorm / currentNorm
                for i in 0..<reconstructed.count {
                    reconstructed[i] *= scale
                }
            }
        }
        
        return reconstructed
    }
    
    // MARK: - Distance Computation
    
    /// Compute distance between query and quantized vector (asymmetric)
    public func computeDistance(
        query: [Float],
        quantized: QuantizedVector
    ) async throws -> Float {
        guard trained else {
            throw QuantizationError.notTrained
        }
        
        if config.useAsymmetricDistance {
            // Asymmetric distance: query vs reconstructed
            return try await computeAsymmetricDistance(
                query: query,
                codes: quantized.codes
            )
        } else {
            // Symmetric distance: reconstruct then compute
            let reconstructed = try await decode(quantized)
            return euclideanDistance(query, reconstructed)
        }
    }
    
    /// Precompute distance tables for efficient search
    public func precomputeDistanceTables(query: [Float]) async throws -> DistanceTables {
        guard trained else {
            throw QuantizationError.notTrained
        }
        
        guard query.count == config.dimensions else {
            throw QuantizationError.dimensionMismatch
        }
        
        var tables: [[Float]] = []
        
        for segmentIdx in 0..<config.segments {
            let querySegment = extractSegment(from: query, segmentIndex: segmentIdx)
            var segmentTable = Array(repeating: Float(0), count: config.codebookSize)
            
            // Compute distance from query segment to each codeword
            for (code, centroid) in codebooks[segmentIdx].centroids.enumerated() {
                segmentTable[code] = euclideanDistanceSquared(querySegment, centroid)
            }
            
            tables.append(segmentTable)
        }
        
        return DistanceTables(tables: tables)
    }
    
    // MARK: - Analysis
    
    public func analyzeQuantizationError(testVectors: [[Float]]) async throws -> QuantizationAnalysis {
        guard trained else {
            throw QuantizationError.notTrained
        }
        
        var totalError: Float = 0
        var maxError: Float = 0
        var errors: [Float] = []
        
        for vector in testVectors {
            let quantized = try await encode(vector)
            let reconstructed = try await decode(quantized)
            
            let error = euclideanDistance(vector, reconstructed)
            totalError += error
            maxError = max(maxError, error)
            errors.append(error)
        }
        
        let avgError = totalError / Float(testVectors.count)
        let variance = errors.map { pow($0 - avgError, 2) }.reduce(0, +) / Float(errors.count)
        
        return QuantizationAnalysis(
            averageError: avgError,
            maxError: maxError,
            errorVariance: variance,
            compressionRatio: config.compressionRatio,
            memoryUsage: estimateMemoryUsage()
        )
    }
    
    // MARK: - Private Methods
    
    private func trainSegmentCodebook(
        segmentVectors: [[Float]],
        segmentIndex: Int
    ) async throws -> Codebook {
        // Configure k-means clustering for this segment
        let kmeansConfig = KMeansClusteringConfiguration(
            maxIterations: config.trainingIterations,
            tolerance: 1e-4,
            initMethod: .kMeansPlusPlus,
            seed: UInt64(segmentIndex), // Use segment index as seed for reproducibility
            useMetalAcceleration: metalCompute != nil
        )
        
        let kmeans = try await KMeansClustering(configuration: kmeansConfig)
        
        // For very small segment vectors, use mini-batch k-means
        let result: ClusteringResult
        if segmentVectors.count > 10000 {
            // Use mini-batch k-means for large datasets
            result = try await kmeans.miniBatchCluster(
                vectors: segmentVectors,
                k: config.codebookSize,
                batchSize: min(1000, segmentVectors.count / 10)
            )
        } else {
            // Use standard k-means for smaller datasets
            result = try await kmeans.cluster(
                vectors: segmentVectors,
                k: config.codebookSize
            )
        }
        
        // Validate clustering result
        guard result.centroids.count == config.codebookSize else {
            throw QuantizationError.invalidCodeSize(result.centroids.count)
        }
        
        // Verify centroid quality
        for (idx, centroid) in result.centroids.enumerated() {
            guard centroid.count == config.segmentDimensions else {
                throw QuantizationError.dimensionMismatch
            }
            
            // Check for degenerate centroids (all zeros)
            let magnitude = sqrt(centroid.map { $0 * $0 }.reduce(0, +))
            if magnitude < 1e-6 {
                print("Warning: Centroid \(idx) in segment \(segmentIndex) has near-zero magnitude")
            }
        }
        
        return Codebook(
            segmentIndex: segmentIndex,
            centroids: result.centroids
        )
    }
    
    private func extractSegment(from vectors: [[Float]], segmentIndex: Int) -> [[Float]] {
        let startIdx = segmentIndex * config.segmentDimensions
        let endIdx = startIdx + config.segmentDimensions
        
        return vectors.map { vector in
            Array(vector[startIdx..<endIdx])
        }
    }
    
    private func extractSegment(from vector: [Float], segmentIndex: Int) -> [Float] {
        let startIdx = segmentIndex * config.segmentDimensions
        let endIdx = startIdx + config.segmentDimensions
        return Array(vector[startIdx..<endIdx])
    }
    
    private func computeAsymmetricDistance(
        query: [Float],
        codes: [UInt8]
    ) async throws -> Float {
        var distance: Float = 0
        
        for segmentIdx in 0..<config.segments {
            let querySegment = extractSegment(from: query, segmentIndex: segmentIdx)
            let code = codes[segmentIdx]
            let centroid = codebooks[segmentIdx].centroids[Int(code)]
            
            distance += euclideanDistanceSquared(querySegment, centroid)
        }
        
        return sqrt(distance)
    }
    
    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        sqrt(euclideanDistanceSquared(a, b))
    }
    
    private func euclideanDistanceSquared(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<min(a.count, b.count) {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sum
    }
    
    private func calculateNorm(_ vector: [Float]) -> Float {
        sqrt(vector.map { $0 * $0 }.reduce(0, +))
    }
    
    private func estimateMemoryUsage() -> Int {
        let codebookMemory = config.segments * config.codebookSize * config.segmentDimensions * MemoryLayout<Float>.size
        let overhead = 1024 // Rough estimate
        return codebookMemory + overhead
    }
    
    // MARK: - Metal Acceleration
    
    private func metalEncode(_ vector: [Float]) async throws -> [UInt8] {
        guard let metalCompute = metalCompute else {
            // Fallback to CPU if Metal not available
            var codes: [UInt8] = []
            for segmentIdx in 0..<config.segments {
                let segment = extractSegment(from: vector, segmentIndex: segmentIdx)
                let code = codebooks[segmentIdx].encode(segment: segment)
                codes.append(code)
            }
            return codes
        }
        
        // Process all segments in parallel using Metal's matrix operations
        var codes = Array(repeating: UInt8(0), count: config.segments)
        
        // Batch process all segments together for better GPU efficiency
        await withTaskGroup(of: (Int, UInt8).self) { group in
            for segmentIdx in 0..<config.segments {
                group.addTask {
                    let segment = await self.extractSegment(from: vector, segmentIndex: segmentIdx)
                    let centroids = await self.codebooks[segmentIdx].centroids
                    
                    // Compute distances using Metal matrix operations
                    // Create matrices for batch distance computation
                    let segmentMatrix = [segment] // 1 x segmentDimensions
                    let centroidsMatrix = centroids // codebookSize x segmentDimensions
                    
                    do {
                        // Compute pairwise distances using matrix operations
                        // ||a - b||² = ||a||² + ||b||² - 2a·b
                        
                        // First compute dot products between segment and all centroids
                        let (dotProducts, _) = try await metalCompute.matrixMultiply(
                            matrixA: segmentMatrix,
                            matrixB: self.transposeMatrix(centroidsMatrix)
                        )
                        
                        // Compute squared norms
                        let segmentNormSquared = segment.map { $0 * $0 }.reduce(0, +)
                        let centroidNormsSquared = centroids.map { centroid in
                            centroid.map { $0 * $0 }.reduce(0, +)
                        }
                        
                        // Compute squared distances
                        var distances: [Float] = []
                        for (idx, centroidNorm) in centroidNormsSquared.enumerated() {
                            let distance = segmentNormSquared + centroidNorm - 2 * dotProducts[0][idx]
                            distances.append(distance)
                        }
                        
                        // Find minimum distance
                        var minIdx = 0
                        var minDist = distances[0]
                        for (idx, dist) in distances.enumerated().dropFirst() {
                            if dist < minDist {
                                minDist = dist
                                minIdx = idx
                            }
                        }
                        
                        return (segmentIdx, UInt8(minIdx))
                    } catch {
                        // Fallback to CPU for this segment
                        let code = await self.codebooks[segmentIdx].encode(segment: segment)
                        return (segmentIdx, code)
                    }
                }
            }
            
            // Collect results
            for await (segmentIdx, code) in group {
                codes[segmentIdx] = code
            }
        }
        
        return codes
    }
    
    private func transposeMatrix(_ matrix: [[Float]]) -> [[Float]] {
        guard !matrix.isEmpty else { return [] }
        
        let rows = matrix.count
        let cols = matrix[0].count
        var transposed = Array(repeating: Array(repeating: Float(0), count: rows), count: cols)
        
        for i in 0..<rows {
            for j in 0..<cols {
                transposed[j][i] = matrix[i][j]
            }
        }
        
        return transposed
    }
    
    private func metalEncodeBatch(_ vectors: [[Float]]) async throws -> [QuantizedVector] {
        guard let metalCompute = metalCompute else {
            // Fallback to CPU batch encoding
            return try await encodeBatch(vectors)
        }
        
        // Process vectors in batches optimized for GPU memory
        let batchSize = 256 // Optimal batch size for GPU processing
        var results: [QuantizedVector] = []
        results.reserveCapacity(vectors.count)
        
        for batchStart in stride(from: 0, to: vectors.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, vectors.count)
            let batch = Array(vectors[batchStart..<batchEnd])
            
            // Initialize results for this batch
            var batchCodes: [[UInt8]] = Array(repeating: [], count: batch.count)
            
            // Process each segment across all vectors in the batch
            for segmentIdx in 0..<config.segments {
                // Extract all segments for this segment index
                let segments = batch.map { vector in
                    extractSegment(from: vector, segmentIndex: segmentIdx)
                }
                
                let centroids = codebooks[segmentIdx].centroids
                
                // Batch matrix multiplication for all segments at once
                // This is the key optimization - compute all distances in one GPU operation
                
                // Create segment matrix: batchSize x segmentDimensions
                let segmentMatrix = segments
                
                // Transpose centroids for multiplication: segmentDimensions x codebookSize
                let centroidsTransposed = transposeMatrix(centroids)
                
                do {
                    // Compute dot products: batchSize x codebookSize
                    let (dotProducts, _) = try await metalCompute.matrixMultiply(
                        matrixA: segmentMatrix,
                        matrixB: centroidsTransposed
                    )
                    
                    // Precompute centroid norms squared
                    let centroidNormsSquared = centroids.map { centroid in
                        centroid.map { $0 * $0 }.reduce(0, +)
                    }
                    
                    // Find nearest centroid for each segment
                    for (vecIdx, segment) in segments.enumerated() {
                        let segmentNormSquared = segment.map { $0 * $0 }.reduce(0, +)
                        
                        var minIdx = 0
                        var minDist = Float.infinity
                        
                        // Compute distances using precomputed dot products
                        for (centroidIdx, centroidNorm) in centroidNormsSquared.enumerated() {
                            let distance = segmentNormSquared + centroidNorm - 2 * dotProducts[vecIdx][centroidIdx]
                            if distance < minDist {
                                minDist = distance
                                minIdx = centroidIdx
                            }
                        }
                        
                        batchCodes[vecIdx].append(UInt8(minIdx))
                    }
                } catch {
                    // Fallback to CPU for this segment if GPU fails
                    for (vecIdx, segment) in segments.enumerated() {
                        let code = codebooks[segmentIdx].encode(segment: segment)
                        batchCodes[vecIdx].append(code)
                    }
                }
            }
            
            // Create QuantizedVector objects for this batch
            for (vecIdx, vector) in batch.enumerated() {
                let originalNorm = calculateNorm(vector)
                var customData = Data()
                customData.append(contentsOf: withUnsafeBytes(of: originalNorm) { Data($0) })
                
                let parameters = QuantizationParameters(
                    precision: config.codeSize,
                    centroids: config.codebookSize,
                    subvectors: config.segments,
                    customData: customData
                )
                
                results.append(QuantizedVector(
                    originalDimensions: vector.count,
                    quantizedData: Data(batchCodes[vecIdx]),
                    scheme: .product(segments: config.segments, bits: config.codeSize),
                    parameters: parameters
                ))
            }
        }
        
        return results
    }
}

// MARK: - Supporting Types

// Use QuantizedVector from VectorTypes.swift instead of defining duplicate

/// Precomputed distance tables for fast search
public struct DistanceTables: Sendable {
    public let tables: [[Float]]  // [segments x codebookSize]
    
    /// Compute distance using precomputed tables
    public func computeDistance(codes: [UInt8]) -> Float {
        var distance: Float = 0
        for (segmentIdx, code) in codes.enumerated() {
            distance += tables[segmentIdx][Int(code)]
        }
        return sqrt(distance)
    }
}

/// Quantization analysis results
public struct QuantizationAnalysis: Sendable {
    public let averageError: Float
    public let maxError: Float
    public let errorVariance: Float
    public let compressionRatio: Float
    public let memoryUsage: Int
}

/// Quantization errors
public enum QuantizationError: LocalizedError {
    case notTrained
    case insufficientTrainingData(required: Int, provided: Int)
    case dimensionMismatch
    case invalidSegmentation(dimensions: Int, segments: Int)
    case invalidCodeSize(Int)
    case invalidQuantizedVector
    
    public var errorDescription: String? {
        switch self {
        case .notTrained:
            return "Product quantizer must be trained before use"
        case .insufficientTrainingData(let required, let provided):
            return "Insufficient training data: \(provided) vectors provided, \(required) required"
        case .dimensionMismatch:
            return "Vector dimensions do not match configuration"
        case .invalidSegmentation(let dimensions, let segments):
            return "Dimensions \(dimensions) not divisible by \(segments) segments"
        case .invalidCodeSize(let size):
            return "Invalid code size: \(size). Must be between 4 and 16"
        case .invalidQuantizedVector:
            return "Invalid quantized vector format"
        }
    }
}

// MARK: - Optimized Product Quantization

/// Optimized PQ with rotation for better quantization
public actor OptimizedProductQuantizer {
    private let baseQuantizer: ProductQuantizer
    private var rotationMatrix: [[Float]]?
    
    public var config: ProductQuantizationConfig {
        get async { baseQuantizer.config }
    }
    
    public init(configuration: ProductQuantizationConfig) async throws {
        self.baseQuantizer = try ProductQuantizer(config: configuration)
    }
    
    /// Learn optimal rotation for better quantization
    public func learnRotation(vectors: [[Float]]) async throws {
        // Simplified OPQ implementation
        // In practice, would learn rotation matrix that minimizes quantization error
        let dimensions = vectors.first?.count ?? 0
        rotationMatrix = Array(
            repeating: Array(repeating: Float(0), count: dimensions),
            count: dimensions
        )
        
        // Identity matrix for now
        for i in 0..<dimensions {
            rotationMatrix![i][i] = 1.0
        }
    }
    
    private func applyRotation(_ vector: [Float]) -> [Float] {
        guard let matrix = rotationMatrix else { return vector }
        
        var rotated = Array(repeating: Float(0), count: vector.count)
        for i in 0..<vector.count {
            for j in 0..<vector.count {
                rotated[i] += matrix[i][j] * vector[j]
            }
        }
        return rotated
    }
}
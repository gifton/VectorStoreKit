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
    internal var codebooks: [Codebook]
    private var trained: Bool = false
    internal let metalCompute: MetalCompute?
    
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
    
    // MARK: - Internal Methods
    
    internal func trainSegmentCodebook(
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
    
    internal func extractSegment(from vectors: [[Float]], segmentIndex: Int) -> [[Float]] {
        let startIdx = segmentIndex * config.segmentDimensions
        let endIdx = startIdx + config.segmentDimensions
        
        return vectors.map { vector in
            Array(vector[startIdx..<endIdx])
        }
    }
    
    internal func extractSegment(from vector: [Float], segmentIndex: Int) -> [Float] {
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
    
    internal func calculateNorm(_ vector: [Float]) -> Float {
        sqrt(vector.map { $0 * $0 }.reduce(0, +))
    }
    
    private func estimateMemoryUsage() -> Int {
        let codebookMemory = config.segments * config.codebookSize * config.segmentDimensions * MemoryLayout<Float>.size
        let overhead = 1024 // Rough estimate
        return codebookMemory + overhead
    }
}
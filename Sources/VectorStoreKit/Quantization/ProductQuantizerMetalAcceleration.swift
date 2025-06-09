// VectorStoreKit: Product Quantizer Metal Acceleration
//
// GPU-accelerated methods for product quantization

import Foundation

// MARK: - Metal Acceleration Extension

extension ProductQuantizer {
    
    internal func metalEncode(_ vector: [Float]) async throws -> [UInt8] {
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
    
    internal func metalEncodeBatch(_ vectors: [[Float]]) async throws -> [QuantizedVector] {
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
    
    internal func transposeMatrix(_ matrix: [[Float]]) -> [[Float]] {
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
}
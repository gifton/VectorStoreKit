// VectorStoreKit: Metal Distance Computation
//
// Hardware-accelerated distance computations

import Foundation
@preconcurrency import Metal
import simd
import os.log

/// Hardware-accelerated distance computation engine
public actor MetalDistanceCompute {
    
    // MARK: - Properties
    
    private let device: MetalDevice
    private let bufferPool: MetalBufferPool
    private let pipelineManager: MetalPipelineManager
    private let profiler: MetalProfiler?
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalDistanceCompute")
    
    // MARK: - Initialization
    
    public init(
        device: MetalDevice,
        bufferPool: MetalBufferPool,
        pipelineManager: MetalPipelineManager,
        profiler: MetalProfiler? = nil
    ) {
        self.device = device
        self.bufferPool = bufferPool
        self.pipelineManager = pipelineManager
        self.profiler = profiler
    }
    
    // MARK: - Distance Computation
    
    /// Compute distances between a query vector and multiple candidates
    public func computeDistances<Vector: SIMD & Sendable>(
        query: Vector,
        candidates: [Vector],
        metric: DistanceMetric
    ) async throws -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        guard !candidates.isEmpty else {
            throw MetalComputeError.emptyInput(parameter: "candidates")
        }
        
        logger.debug("Computing distances for \(candidates.count) vectors")
        
        // Prepare buffers first
        let queryBuffer = try await bufferPool.getBuffer(for: query)
        let candidatesBuffer = try await bufferPool.getBuffer(for: candidates)
        let outputSize = candidates.count * MemoryLayout<Float>.size
        
        // Validate buffer size
        let maxBufferSize = await device.device.maxBufferLength
        if outputSize > maxBufferSize {
            throw MetalComputeError.invalidBufferSize(requested: outputSize, maximum: maxBufferSize)
        }
        
        let resultsBuffer = try await bufferPool.getBuffer(size: outputSize)
        
        defer {
            Task {
                await bufferPool.returnBuffer(queryBuffer)
                await bufferPool.returnBuffer(candidatesBuffer)
                await bufferPool.returnBuffer(resultsBuffer)
            }
        }
        
        // Execute computation with metric instead of pipeline
        let distances = try await executeDistanceComputationWithMetric(
            metric: metric,
            queryBuffer: queryBuffer,
            candidatesBuffer: candidatesBuffer,
            resultsBuffer: resultsBuffer,
            candidateCount: candidates.count,
            vectorDimension: Vector.scalarCount
        )
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        await profiler?.recordOperation(.distanceComputation, duration: duration, dataSize: candidates.count)
        
        return distances
    }
    
    /// Compute multiple distance metrics simultaneously
    public func computeMultipleDistances<Vector: SIMD & Sendable>(
        query: Vector,
        candidates: [Vector],
        metrics: Set<DistanceMetric>
    ) async throws -> [DistanceMetric: [Float]] where Vector.Scalar: BinaryFloatingPoint {
        
        guard !candidates.isEmpty else {
            throw MetalComputeError.emptyInput(parameter: "candidates")
        }
        
        guard !metrics.isEmpty else {
            throw MetalComputeError.invalidInputData(reason: "No distance metrics specified")
        }
        
        // Prepare shared buffers
        let queryBuffer = try await bufferPool.getBuffer(for: query)
        let candidatesBuffer = try await bufferPool.getBuffer(for: candidates)
        
        defer {
            Task {
                await bufferPool.returnBuffer(queryBuffer)
                await bufferPool.returnBuffer(candidatesBuffer)
            }
        }
        
        var results: [DistanceMetric: [Float]] = [:]
        
        // Compute each metric
        // In a more optimized version, we could use a single kernel that computes multiple metrics
        for metric in metrics {
            let resultsBuffer = try await bufferPool.getBuffer(
                size: candidates.count * MemoryLayout<Float>.size
            )
            
            let distances = try await executeDistanceComputationWithMetric(
                metric: metric,
                queryBuffer: queryBuffer,
                candidatesBuffer: candidatesBuffer,
                resultsBuffer: resultsBuffer,
                candidateCount: candidates.count,
                vectorDimension: Vector.scalarCount
            )
            
            results[metric] = distances
            await bufferPool.returnBuffer(resultsBuffer)
        }
        
        return results
    }
    
    // MARK: - Batch Operations
    
    /// Compute distances for multiple queries in parallel using optimized batch processing
    public func batchComputeDistances<Vector: SIMD & Sendable>(
        queries: [Vector],
        candidates: [Vector],
        metric: DistanceMetric
    ) async throws -> [[Float]] where Vector.Scalar: BinaryFloatingPoint {
        
        guard !queries.isEmpty && !candidates.isEmpty else {
            throw MetalComputeError.emptyInput(parameter: "queries or candidates")
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let numQueries = queries.count
        let numCandidates = candidates.count
        
        // Use GPU batch kernel for large workloads
        if numQueries * numCandidates > 10000 {
            return try await batchComputeDistancesGPU(
                queries: queries,
                candidates: candidates,
                metric: metric
            )
        }
        
        // For smaller batches, use concurrent processing
        return try await withThrowingTaskGroup(of: (Int, [Float]).self) { group in
            // Process queries in chunks for better memory usage
            let chunkSize = min(10, numQueries)
            
            for (index, query) in queries.enumerated() {
                group.addTask {
                    let distances = try await self.computeDistances(
                        query: query,
                        candidates: candidates,
                        metric: metric
                    )
                    return (index, distances)
                }
                
                // Limit concurrent tasks
                if (index + 1) % chunkSize == 0 {
                    for try await _ in group.prefix(chunkSize) { }
                }
            }
            
            var results = Array(repeating: [Float](), count: queries.count)
            for try await (index, distances) in group {
                results[index] = distances
            }
            
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            await profiler?.recordOperation(
                .distanceComputation,
                duration: duration,
                dataSize: numQueries * numCandidates
            )
            
            return results
        }
    }
    
    /// GPU-optimized batch distance computation
    private func batchComputeDistancesGPU<Vector: SIMD & Sendable>(
        queries: [Vector],
        candidates: [Vector],
        metric: DistanceMetric
    ) async throws -> [[Float]] where Vector.Scalar: BinaryFloatingPoint {
        
        let numQueries = queries.count
        let numCandidates = candidates.count
        let dimension = Vector.scalarCount
        
        // Get batch pipeline
        let basePipelineName: String
        switch metric {
        case .euclidean:
            basePipelineName = "euclideanDistance"
        case .cosine:
            basePipelineName = "cosineDistance"
        case .manhattan:
            basePipelineName = "manhattanDistance"
        case .dotProduct:
            basePipelineName = "dotProduct"
        default:
            throw MetalComputeError.unsupportedOperation(operation: "Distance metric", reason: "Metric \(metric) not supported")
        }
        
        let batchPipelineName = "batch" + basePipelineName.capitalized
        let pipeline = try await pipelineManager.getPipeline(functionName: batchPipelineName)
        
        // Flatten arrays for GPU processing
        var flatQueries: [Float] = []
        var flatCandidates: [Float] = []
        
        for query in queries {
            for i in 0..<dimension {
                flatQueries.append(Float(query[i]))
            }
        }
        
        for candidate in candidates {
            for i in 0..<dimension {
                flatCandidates.append(Float(candidate[i]))
            }
        }
        
        // Allocate buffers
        let queriesBuffer = try await bufferPool.getBuffer(for: flatQueries)
        let candidatesBuffer = try await bufferPool.getBuffer(for: flatCandidates)
        let outputSize = numQueries * numCandidates * MemoryLayout<Float>.size
        
        // Validate buffer size
        let maxBufferSize = await device.device.maxBufferLength
        if outputSize > maxBufferSize {
            // Split into smaller batches
            return try await batchComputeDistancesChunked(
                queries: queries,
                candidates: candidates,
                metric: metric,
                maxBufferSize: maxBufferSize
            )
        }
        
        let resultsBuffer = try await bufferPool.getBuffer(size: outputSize)
        
        defer {
            Task {
                await bufferPool.returnBuffer(queriesBuffer)
                await bufferPool.returnBuffer(candidatesBuffer)
                await bufferPool.returnBuffer(resultsBuffer)
            }
        }
        
        // Execute batch computation
        guard let commandBuffer = await device.makeCommandBuffer() else {
            throw MetalComputeError.commandBufferCreationFailed
        }
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queriesBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidatesBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
        
        var params = [UInt32(dimension), UInt32(numQueries), UInt32(numCandidates)]
        encoder.setBytes(&params, length: MemoryLayout<UInt32>.size * 3, index: 3)
        
        // 2D thread configuration for batch processing
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroups = MTLSize(
            width: (numCandidates + 15) / 16,
            height: (numQueries + 15) / 16,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        // Execute
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        if let error = commandBuffer.error {
            throw MetalComputeError.commandBufferExecutionFailed(error: error.localizedDescription)
        }
        
        // Extract results
        let resultsPointer = resultsBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numQueries * numCandidates
        )
        
        var results: [[Float]] = []
        for q in 0..<numQueries {
            let startIdx = q * numCandidates
            let distances = Array(UnsafeBufferPointer(
                start: resultsPointer.advanced(by: startIdx),
                count: numCandidates
            ))
            results.append(distances)
        }
        
        return results
    }
    
    /// Process large batches in chunks when they exceed GPU memory limits
    private func batchComputeDistancesChunked<Vector: SIMD & Sendable>(
        queries: [Vector],
        candidates: [Vector],
        metric: DistanceMetric,
        maxBufferSize: Int
    ) async throws -> [[Float]] where Vector.Scalar: BinaryFloatingPoint {
        
        let elementSize = MemoryLayout<Float>.size
        let maxElementsPerBatch = maxBufferSize / elementSize / 2 // Divide by 2 for safety
        let maxQueriesPerBatch = max(1, maxElementsPerBatch / candidates.count)
        
        var allResults: [[Float]] = []
        
        for queryChunk in queries.chunked(into: maxQueriesPerBatch) {
            let chunkResults = try await batchComputeDistancesGPU(
                queries: queryChunk,
                candidates: candidates,
                metric: metric
            )
            allResults.append(contentsOf: chunkResults)
        }
        
        return allResults
    }
    
    // MARK: - Private Methods
    
    
    private func executeDistanceComputationWithMetric(
        metric: DistanceMetric,
        queryBuffer: MTLBuffer,
        candidatesBuffer: MTLBuffer,
        resultsBuffer: MTLBuffer,
        candidateCount: Int,
        vectorDimension: Int
    ) async throws -> [Float] {
        
        // Get pipeline name first
        let pipelineName: MetalPipelineManager.StandardPipeline
        
        switch metric {
        case .euclidean:
            pipelineName = .euclideanDistance
        case .cosine:
            pipelineName = .cosineDistance
        case .manhattan:
            pipelineName = .manhattanDistance
        case .dotProduct:
            pipelineName = .dotProduct
        default:
            throw MetalComputeError.unsupportedOperation(operation: "Distance metric", reason: "Metric \(metric) not supported")
        }
        
        // Get pipeline directly within the same context
        let pipeline = try await pipelineManager.getPipeline(for: pipelineName)
        
        // Execute the computation inline to avoid passing pipeline
        guard let commandBuffer = await device.makeCommandBuffer() else {
            throw MetalComputeError.commandBufferCreationFailed
        }
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        // Set up compute command
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidatesBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
        
        // Set vector dimension as constant
        var dimension = Int32(vectorDimension)
        encoder.setBytes(&dimension, length: MemoryLayout<Int32>.size, index: 3)
        
        // Calculate thread configuration
        let (threadgroups, threadsPerGroup) = await pipelineManager.getThreadConfiguration(
            for: pipeline,
            workSize: candidateCount
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        // Execute and wait with error handling
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { buffer in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        // Check for execution errors
        if let error = commandBuffer.error {
            throw MetalComputeError.commandBufferExecutionFailed(error: error.localizedDescription)
        }
        
        // Extract results
        let resultsPointer = resultsBuffer.contents().bindMemory(
            to: Float.self,
            capacity: candidateCount
        )
        return Array(UnsafeBufferPointer(start: resultsPointer, count: candidateCount))
    }
    
    private func executeDistanceComputation(
        pipeline: MTLComputePipelineState,
        queryBuffer: MTLBuffer,
        candidatesBuffer: MTLBuffer,
        resultsBuffer: MTLBuffer,
        candidateCount: Int,
        vectorDimension: Int
    ) async throws -> [Float] {
        
        guard let commandBuffer = await device.makeCommandBuffer() else {
            throw MetalComputeError.commandBufferCreationFailed
        }
        
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        // Set up compute command
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidatesBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
        
        // Set vector dimension as constant
        var dimension = Int32(vectorDimension)
        encoder.setBytes(&dimension, length: MemoryLayout<Int32>.size, index: 3)
        
        // Calculate thread configuration
        let (threadgroups, threadsPerGroup) = await pipelineManager.getThreadConfiguration(
            for: pipeline,
            workSize: candidateCount
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        
        // Execute and wait with error handling
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { buffer in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        // Check for execution errors
        if let error = commandBuffer.error {
            throw MetalComputeError.commandBufferExecutionFailed(error: error.localizedDescription)
        }
        
        // Extract results
        let resultsPointer = resultsBuffer.contents().bindMemory(
            to: Float.self,
            capacity: candidateCount
        )
        return Array(UnsafeBufferPointer(start: resultsPointer, count: candidateCount))
    }
}

// MARK: - Supporting Types

/// Distance computation specific errors
public enum MetalDistanceError: Error, LocalizedError {
    case unsupportedMetric(DistanceMetric)
    case noMetricsSpecified
    
    public var errorDescription: String? {
        switch self {
        case .unsupportedMetric(let metric):
            return "Unsupported distance metric: \(metric)"
        case .noMetricsSpecified:
            return "No distance metrics specified"
        }
    }
}

// MARK: - Array Extension

extension Array {
    /// Split array into chunks of specified size
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}

// MARK: - CPU Fallback

extension MetalDistanceCompute {
    
    /// CPU fallback for small batches or unsupported operations
    public func computeDistancesCPU<Vector: SIMD & Sendable>(
        query: Vector,
        candidates: [Vector],
        metric: DistanceMetric
    ) async -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        
        return candidates.map { candidate in
            switch metric {
            case .euclidean:
                let diff = query - candidate
                return Float(sqrt((diff * diff).sum()))
                
            case .cosine:
                let dot = (query * candidate).sum()
                let queryMag = sqrt((query * query).sum())
                let candidateMag = sqrt((candidate * candidate).sum())
                return 1.0 - Float(dot) / (Float(queryMag) * Float(candidateMag))
                
            case .manhattan:
                return (0..<query.scalarCount).reduce(Float(0)) { sum, i in
                    sum + abs(Float(query[i]) - Float(candidate[i]))
                }
                
            case .dotProduct:
                return Float(-(query * candidate).sum()) // Negative for similarity
                
            default:
                return Float.infinity
            }
        }
    }
}
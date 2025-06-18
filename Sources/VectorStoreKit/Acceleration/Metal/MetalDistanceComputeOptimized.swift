// VectorStoreKit: Optimized Metal Distance Computation
//
// Hardware-accelerated distance computations with thread configuration optimization
//

import Foundation
@preconcurrency import Metal
import simd
import os.log

/// Optimized hardware-accelerated distance computation engine
public actor MetalDistanceComputeOptimized {
    
    // MARK: - Properties
    
    private let device: MetalDevice
    private let bufferPool: MetalBufferPool
    private let pipelineManager: MetalPipelineManager
    private let threadOptimizer: MetalThreadConfigurationOptimizer
    private let performanceProfiler: MetalPerformanceProfiler?
    private let commandBufferPool: MetalCommandBufferPool
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalDistanceComputeOptimized")
    
    // MARK: - Initialization
    
    public init(
        device: MetalDevice,
        bufferPool: MetalBufferPool,
        pipelineManager: MetalPipelineManager,
        threadOptimizer: MetalThreadConfigurationOptimizer? = nil,
        performanceProfiler: MetalPerformanceProfiler? = nil,
        commandBufferPool: MetalCommandBufferPool? = nil
    ) async {
        self.device = device
        self.bufferPool = bufferPool
        self.pipelineManager = pipelineManager
        if let optimizer = threadOptimizer {
            self.threadOptimizer = optimizer
        } else {
            self.threadOptimizer = await MetalThreadConfigurationOptimizer(device: device)
        }
        self.performanceProfiler = performanceProfiler
        self.commandBufferPool = commandBufferPool ?? MetalCommandBufferPool(device: device.device, profiler: nil)
    }
    
    // MARK: - Distance Computation
    
    /// Compute distances with optimized thread configuration
    public func computeDistances<Vector: SIMD & Sendable>(
        query: Vector,
        candidates: [Vector],
        metric: DistanceMetric
    ) async throws -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        
        guard !candidates.isEmpty else {
            throw MetalComputeError.emptyInput(parameter: "candidates")
        }
        
        // Begin profiling
        let operationHandle = await performanceProfiler?.beginOperation(
            "DistanceComputation",
            category: .distanceComputation,
            metadata: [
                "metric": metric.rawValue,
                "candidates": candidates.count,
                "dimension": Vector.scalarCount
            ]
        )
        
        defer {
            if let handle = operationHandle {
                Task {
                    await performanceProfiler?.endOperation(handle)
                }
            }
        }
        
        logger.debug("Computing distances for \(candidates.count) vectors with dimension \(Vector.scalarCount)")
        
        // Prepare buffers
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
        
        // Execute optimized computation
        let distances = try await executeOptimizedDistanceComputation(
            metric: metric,
            queryBuffer: queryBuffer,
            candidatesBuffer: candidatesBuffer,
            resultsBuffer: resultsBuffer,
            candidateCount: candidates.count,
            vectorDimension: Vector.scalarCount
        )
        
        return distances
    }
    
    /// Compute distances for 512-dimensional vectors with maximum optimization
    public func computeDistances512(
        query: Vector512,
        candidates: [Vector512],
        metric: DistanceMetric,
        normalized: Bool = false
    ) async throws -> [Float] {
        
        guard !candidates.isEmpty else {
            throw MetalComputeError.emptyInput(parameter: "candidates")
        }
        
        // Begin profiling
        let operationHandle = await performanceProfiler?.beginOperation(
            "DistanceComputation512",
            category: .distanceComputation,
            metadata: [
                "metric": metric.rawValue,
                "candidates": candidates.count,
                "normalized": normalized
            ]
        )
        
        defer {
            if let handle = operationHandle {
                Task {
                    await performanceProfiler?.endOperation(handle)
                }
            }
        }
        
        logger.debug("Computing distances for \(candidates.count) 512-dim vectors")
        
        // Get optimized pipeline
        let pipelineName: String
        switch metric {
        case .euclidean:
            pipelineName = "euclideanDistance512_simd"
        case .cosine:
            pipelineName = normalized ? "cosineDistance512_normalized" : "cosineDistance512_simd"
        default:
            // Fall back to generic implementation
            return try await computeDistances(query: query, candidates: candidates, metric: metric)
        }
        
        let pipeline: MTLComputePipelineState
        do {
            pipeline = try await pipelineManager.getPipeline(functionName: pipelineName)
        } catch {
            logger.warning("Optimized shader \(pipelineName) not found, falling back to generic")
            return try await computeDistances(query: query, candidates: candidates, metric: metric)
        }
        
        // Prepare buffers
        let queryData = query.withUnsafeMetalBytes { Data($0) }
        let queryBuffer = try await bufferPool.getBuffer(for: queryData)
        
        var candidatesData = Data()
        candidatesData.reserveCapacity(candidates.count * 512 * MemoryLayout<Float>.size)
        for candidate in candidates {
            candidate.withUnsafeMetalBytes { bytes in
                candidatesData += bytes
            }
        }
        
        let candidatesBuffer = try await bufferPool.getBuffer(for: candidatesData)
        let resultsBuffer = try await bufferPool.getBuffer(size: candidates.count * MemoryLayout<Float>.size)
        
        defer {
            Task {
                await bufferPool.returnBuffer(queryBuffer)
                await bufferPool.returnBuffer(candidatesBuffer)
                await bufferPool.returnBuffer(resultsBuffer)
            }
        }
        
        // Get optimal thread configuration for 512-dim vectors
        let config = await threadOptimizer.getOptimalConfiguration(
            for: .distanceComputation,
            workSize: candidates.count,
            vectorDimension: 512
        )
        
        // Execute with profiling
        let commandBuffer = try await commandBufferPool.getCommandBuffer(label: "Distance512:\(metric)")
        
        guard let encoder = commandBuffer.buffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        // Add timestamp markers if profiling
        let timestampBegin = await performanceProfiler?.addTimestampMarker(
            to: encoder,
            label: "Distance512Begin",
            at: .begin
        )
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidatesBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
        
        // Use optimized thread configuration
        encoder.dispatchThreadgroups(
            config.threadgroupsPerGrid,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
        
        // Add end timestamp
        if timestampBegin != nil {
            await performanceProfiler?.addTimestampMarker(
                to: encoder,
                label: "Distance512End",
                at: .end
            )
        }
        
        encoder.endEncoding()
        
        // Profile command buffer execution
        let profile = await performanceProfiler?.profileCommandBuffer(
            commandBuffer.buffer,
            label: "Distance512:\(metric)",
            kernels: [
                KernelDescriptor(
                    name: pipelineName,
                    totalThreads: candidates.count
                )
            ]
        )
        
        if let error = commandBuffer.buffer.error {
            throw MetalComputeError.commandBufferExecutionFailed(error: error.localizedDescription)
        }
        
        // Log performance metrics
        if let profile = profile {
            logger.info("""
                512-dim distance computation: \
                CPU=\(profile.cpuTime * 1000)ms, \
                GPU=\(profile.gpuTime * 1000)ms, \
                throughput=\(Double(candidates.count) / profile.gpuTime) vectors/sec
                """)
        }
        
        // Extract results
        let resultsPointer = resultsBuffer.contents().bindMemory(
            to: Float.self,
            capacity: candidates.count
        )
        return Array(UnsafeBufferPointer(start: resultsPointer, count: candidates.count))
    }
    
    /// Batch compute distances with 2D optimization
    public func batchComputeDistances<Vector: SIMD & Sendable>(
        queries: [Vector],
        candidates: [Vector],
        metric: DistanceMetric
    ) async throws -> [[Float]] where Vector.Scalar: BinaryFloatingPoint {
        
        guard !queries.isEmpty && !candidates.isEmpty else {
            throw MetalComputeError.emptyInput(parameter: "queries or candidates")
        }
        
        let numQueries = queries.count
        let numCandidates = candidates.count
        let dimension = Vector.scalarCount
        
        // Begin profiling
        let operationHandle = await performanceProfiler?.beginOperation(
            "BatchDistanceComputation",
            category: .distanceComputation,
            metadata: [
                "metric": metric.rawValue,
                "queries": numQueries,
                "candidates": numCandidates,
                "dimension": dimension
            ]
        )
        
        defer {
            if let handle = operationHandle {
                Task {
                    await performanceProfiler?.endOperation(handle)
                }
            }
        }
        
        // Get batch pipeline
        let pipelineName = "batch" + getStandardPipelineName(for: metric).capitalized
        let pipeline = try await pipelineManager.getPipeline(functionName: pipelineName)
        
        // Prepare data
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
        
        // Check buffer size limits
        let maxBufferSize = await device.device.maxBufferLength
        if outputSize > maxBufferSize {
            // Use chunked processing
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
        
        // Get optimal 2D configuration
        let config = await threadOptimizer.getOptimal2DConfiguration(
            for: .distanceComputation,
            rows: numQueries,
            columns: numCandidates,
            tileSize: 16
        )
        
        // Execute batch computation
        let commandBuffer = try await commandBufferPool.getCommandBuffer(label: "BatchDistance:\(metric)")
        
        guard let encoder = commandBuffer.buffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queriesBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidatesBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
        
        var params = [UInt32(dimension), UInt32(numQueries), UInt32(numCandidates)]
        encoder.setBytes(&params, length: MemoryLayout<UInt32>.size * 3, index: 3)
        
        // Use optimized 2D configuration
        encoder.dispatchThreadgroups(
            config.threadgroupsPerGrid,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
        
        encoder.endEncoding()
        
        // Profile execution
        let profile = await performanceProfiler?.profileCommandBuffer(
            commandBuffer.buffer,
            label: "BatchDistance:\(metric)",
            kernels: [
                KernelDescriptor(
                    name: pipelineName,
                    totalThreads: numQueries * numCandidates
                )
            ]
        )
        
        if let error = commandBuffer.buffer.error {
            throw MetalComputeError.commandBufferExecutionFailed(error: error.localizedDescription)
        }
        
        // Log performance
        if let profile = profile {
            let throughput = Double(numQueries * numCandidates) / profile.gpuTime
            logger.info("""
                Batch computation (\(numQueries)x\(numCandidates)): \
                GPU=\(profile.gpuTime * 1000)ms, \
                throughput=\(throughput) distances/sec, \
                occupancy=\(config.estimatedOccupancy)
                """)
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
    
    // MARK: - Private Methods
    
    private func executeOptimizedDistanceComputation(
        metric: DistanceMetric,
        queryBuffer: MTLBuffer,
        candidatesBuffer: MTLBuffer,
        resultsBuffer: MTLBuffer,
        candidateCount: Int,
        vectorDimension: Int
    ) async throws -> [Float] {
        
        // Get optimal configuration
        let config = await threadOptimizer.getOptimalConfiguration(
            for: .distanceComputation,
            workSize: candidateCount,
            vectorDimension: vectorDimension
        )
        
        // Check for optimized kernels
        let use512Optimization = vectorDimension == 512
        let useOptimizedKernel = vectorDimension % 4 == 0
        
        let pipelineName: String
        if use512Optimization {
            switch metric {
            case .euclidean:
                pipelineName = "euclideanDistance512_simd"
            case .cosine:
                pipelineName = "cosineDistance512_simd"
            default:
                pipelineName = getStandardPipelineName(for: metric)
            }
        } else if useOptimizedKernel && metric == .euclidean {
            pipelineName = "euclideanDistanceWarpOptimized"
        } else {
            pipelineName = getStandardPipelineName(for: metric)
        }
        
        // Get pipeline with fallback
        let pipeline: MTLComputePipelineState
        do {
            pipeline = try await pipelineManager.getPipeline(functionName: pipelineName)
        } catch {
            if pipelineName.contains("Optimized") || pipelineName.contains("512") {
                logger.debug("Optimized shader \(pipelineName) not available, falling back to standard")
                let fallbackName = getStandardPipelineName(for: metric)
                pipeline = try await pipelineManager.getPipeline(functionName: fallbackName)
            } else {
                throw error
            }
        }
        
        // Execute with profiling
        let commandBuffer = try await commandBufferPool.getCommandBuffer(label: "Distance:\(metric)")
        
        guard let encoder = commandBuffer.buffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidatesBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
        
        var dimension = Int32(vectorDimension)
        encoder.setBytes(&dimension, length: MemoryLayout<Int32>.size, index: 3)
        
        // Use optimized thread configuration
        encoder.dispatchThreadgroups(
            config.threadgroupsPerGrid,
            threadsPerThreadgroup: config.threadsPerThreadgroup
        )
        
        encoder.endEncoding()
        
        // Profile execution
        let profile = await performanceProfiler?.profileCommandBuffer(
            commandBuffer.buffer,
            label: "Distance:\(metric)",
            kernels: [
                KernelDescriptor(
                    name: pipelineName,
                    totalThreads: candidateCount
                )
            ]
        )
        
        if let error = commandBuffer.buffer.error {
            throw MetalComputeError.commandBufferExecutionFailed(error: error.localizedDescription)
        }
        
        // Log optimization metrics
        if let profile = profile {
            logger.debug("""
                Distance computation optimized: \
                config=\(config.threadsPerThreadgroup.width) threads, \
                occupancy=\(config.estimatedOccupancy), \
                GPU time=\(profile.gpuTime * 1000)ms
                """)
        }
        
        // Extract results
        let resultsPointer = resultsBuffer.contents().bindMemory(
            to: Float.self,
            capacity: candidateCount
        )
        return Array(UnsafeBufferPointer(start: resultsPointer, count: candidateCount))
    }
    
    private func getStandardPipelineName(for metric: DistanceMetric) -> String {
        switch metric {
        case .euclidean:
            return "euclideanDistance"
        case .cosine:
            return "cosineDistance"
        case .manhattan:
            return "manhattanDistance"
        case .dotProduct:
            return "dotProduct"
        default:
            return "euclideanDistance"
        }
    }
    
    private func batchComputeDistancesChunked<Vector: SIMD & Sendable>(
        queries: [Vector],
        candidates: [Vector],
        metric: DistanceMetric,
        maxBufferSize: Int
    ) async throws -> [[Float]] where Vector.Scalar: BinaryFloatingPoint {
        
        let elementSize = MemoryLayout<Float>.size
        let maxElementsPerBatch = maxBufferSize / elementSize / 2
        let maxQueriesPerBatch = max(1, maxElementsPerBatch / candidates.count)
        
        var allResults: [[Float]] = []
        
        for queryChunk in queries.chunked(into: maxQueriesPerBatch) {
            let chunkResults = try await batchComputeDistances(
                queries: queryChunk,
                candidates: candidates,
                metric: metric
            )
            allResults.append(contentsOf: chunkResults)
        }
        
        return allResults
    }
}

// MARK: - Extensions

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}
// VectorStoreKit: Metal Distance Matrix Computation
//
// GPU-accelerated distance matrix computation with intelligent CPU/GPU switching
//

import Foundation
@preconcurrency import Metal
import simd
import os.log

/// GPU-accelerated distance matrix computation engine
public actor MetalDistanceMatrix {
    
    // MARK: - Properties
    
    private let device: MetalDevice
    private let bufferPool: MetalBufferPool
    private let pipelineManager: MetalPipelineManager
    private let profiler: MetalProfiler?
    private let commandBufferPool: MetalCommandBufferPool
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalDistanceMatrix")
    
    // Heuristic thresholds
    private let gpuThresholdElements = 10_000  // Use GPU for matrices > 10k elements
    private let streamingThreshold = 100_000_000  // Use streaming for > 100M elements
    private let tileSize: Int = 32
    
    // MARK: - Initialization
    
    public init(
        device: MetalDevice,
        bufferPool: MetalBufferPool,
        pipelineManager: MetalPipelineManager,
        profiler: MetalProfiler? = nil,
        commandBufferPool: MetalCommandBufferPool? = nil
    ) {
        self.device = device
        self.bufferPool = bufferPool
        self.pipelineManager = pipelineManager
        self.profiler = profiler
        let mtlDevice = device.device
        self.commandBufferPool = commandBufferPool ?? MetalCommandBufferPool(device: mtlDevice, profiler: profiler)
    }
    
    // MARK: - Public API
    
    /// Compute distance matrix between two sets of vectors with intelligent CPU/GPU switching
    public func computeDistanceMatrix(
        vectorsA: [Vector512],
        vectorsB: [Vector512]? = nil,  // If nil, compute symmetric matrix for vectorsA
        metric: DistanceMetric = .euclidean
    ) async throws -> [[Float]] {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let isSymmetric = vectorsB == nil
        let actualVectorsB = vectorsB ?? vectorsA
        let numVectorsA = vectorsA.count
        let numVectorsB = actualVectorsB.count
        let totalElements = numVectorsA * numVectorsB
        
        logger.info("Computing distance matrix: \(numVectorsA)x\(numVectorsB), metric: \(String(describing: metric))")
        
        // Decide computation strategy
        let strategy: ComputationStrategy = selectComputationStrategy(
            numVectorsA: numVectorsA,
            numVectorsB: numVectorsB,
            dimension: 512,
            isSymmetric: isSymmetric
        )
        
        logger.debug("Selected strategy: \(String(describing: strategy))")
        
        let flatMatrix: [Float]
        
        switch strategy {
        case .cpu:
            flatMatrix = await computeDistanceMatrixCPU(
                vectorsA: vectorsA,
                vectorsB: actualVectorsB,
                metric: metric,
                isSymmetric: isSymmetric
            )
            
        case .gpu:
            flatMatrix = try await computeDistanceMatrixGPU(
                vectorsA: vectorsA,
                vectorsB: actualVectorsB,
                metric: metric,
                isSymmetric: isSymmetric
            )
            
        case .streaming:
            flatMatrix = try await computeDistanceMatrixStreaming(
                vectorsA: vectorsA,
                vectorsB: actualVectorsB,
                metric: metric,
                isSymmetric: isSymmetric
            )
        }
        
        // Convert flat array to 2D matrix
        var matrix = [[Float]]()
        matrix.reserveCapacity(numVectorsA)
        
        for i in 0..<numVectorsA {
            let startIdx = i * numVectorsB
            let row = Array(flatMatrix[startIdx..<startIdx + numVectorsB])
            matrix.append(row)
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        await profiler?.recordOperation(.distanceComputation, duration: duration, dataSize: totalElements)
        
        logger.info("Distance matrix computation completed in \(duration)s")
        
        return matrix
    }
    
    /// Async computation pipeline for overlapping CPU/GPU work
    public func computeDistanceMatrixAsync(
        vectorsA: [Vector512],
        vectorsB: [Vector512]? = nil,
        metric: DistanceMetric = .euclidean,
        completion: @escaping ([[Float]]) -> Void
    ) async throws {
        
        let isSymmetric = vectorsB == nil
        let actualVectorsB = vectorsB ?? vectorsA
        
        // Create async pipeline
        let pipeline = try await createAsyncPipeline(
            vectorsA: vectorsA,
            vectorsB: actualVectorsB,
            metric: metric,
            isSymmetric: isSymmetric
        )
        
        // Start async computation
        Task {
            let result = try await pipeline.execute()
            completion(result)
        }
    }
    
    // MARK: - Computation Strategies
    
    private enum ComputationStrategy {
        case cpu
        case gpu
        case streaming
    }
    
    private func selectComputationStrategy(
        numVectorsA: Int,
        numVectorsB: Int,
        dimension: Int,
        isSymmetric: Bool
    ) -> ComputationStrategy {
        
        let totalElements = numVectorsA * numVectorsB
        let memoryRequired = totalElements * MemoryLayout<Float>.size
        let availableMemory = device.device.recommendedMaxWorkingSetSize
        
        // Use streaming for very large matrices that won't fit in GPU memory
        if memoryRequired > availableMemory || totalElements > streamingThreshold {
            return .streaming
        }
        
        // Use CPU for small matrices where GPU overhead isn't worth it
        if totalElements < gpuThresholdElements {
            return .cpu
        }
        
        // Use GPU for medium to large matrices that fit in memory
        return .gpu
    }
    
    // MARK: - GPU Implementation
    
    private func computeDistanceMatrixGPU(
        vectorsA: [Vector512],
        vectorsB: [Vector512],
        metric: DistanceMetric,
        isSymmetric: Bool
    ) async throws -> [Float] {
        
        let numVectorsA = vectorsA.count
        let numVectorsB = vectorsB.count
        
        // Prepare data for GPU
        let vectorsAData = packVectorsForGPU(vectorsA)
        let vectorsBData = packVectorsForGPU(vectorsB)
        
        // Allocate buffers
        let bufferA = try await bufferPool.getBuffer(for: vectorsAData)
        let bufferB = try await bufferPool.getBuffer(for: vectorsBData)
        let outputSize = numVectorsA * numVectorsB * MemoryLayout<Float>.size
        let outputBuffer = try await bufferPool.getBuffer(size: outputSize)
        
        defer {
            Task {
                await bufferPool.returnBuffer(bufferA)
                await bufferPool.returnBuffer(bufferB)
                await bufferPool.returnBuffer(outputBuffer)
            }
        }
        
        // Select appropriate kernel
        let kernelName: String
        if isSymmetric {
            kernelName = "distanceMatrixSymmetric"
        } else if metric == .euclidean {
            kernelName = "distanceMatrix512_euclidean"
        } else {
            kernelName = "distanceMatrixBlocked"
        }
        
        let pipeline = try await pipelineManager.getPipeline(functionName: kernelName)
        
        // Execute computation
        let commandBuffer = try await commandBufferPool.getCommandBuffer(label: "DistanceMatrix")
        
        guard let encoder = commandBuffer.buffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.computeEncoderCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        
        if isSymmetric {
            var dimension = UInt32(512)
            var numVectors = UInt32(numVectorsA)
            var metricValue = UInt32(metricToInt(metric))
            
            encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.size, index: 2)
            encoder.setBytes(&numVectors, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&metricValue, length: MemoryLayout<UInt32>.size, index: 4)
        } else {
            var numA = UInt32(numVectorsA)
            var numB = UInt32(numVectorsB)
            
            encoder.setBytes(&numA, length: MemoryLayout<UInt32>.size, index: 3)
            encoder.setBytes(&numB, length: MemoryLayout<UInt32>.size, index: 4)
        }
        
        // Configure thread execution
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroups = MTLSize(
            width: (numVectorsB + 15) / 16,
            height: (numVectorsA + 15) / 16,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        // Execute and wait
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        if let error = commandBuffer.buffer.error {
            throw MetalComputeError.commandBufferExecutionFailed(error: error.localizedDescription)
        }
        
        // Extract results
        let resultsPointer = outputBuffer.contents().bindMemory(
            to: Float.self,
            capacity: numVectorsA * numVectorsB
        )
        
        return Array(UnsafeBufferPointer(start: resultsPointer, count: numVectorsA * numVectorsB))
    }
    
    // MARK: - Streaming Implementation
    
    private func computeDistanceMatrixStreaming(
        vectorsA: [Vector512],
        vectorsB: [Vector512],
        metric: DistanceMetric,
        isSymmetric: Bool
    ) async throws -> [Float] {
        
        let numVectorsA = vectorsA.count
        let numVectorsB = vectorsB.count
        let batchSize = 1000  // Process 1000x1000 tiles at a time
        
        var fullMatrix = [Float](repeating: 0, count: numVectorsA * numVectorsB)
        
        // Process in tiles
        for batchStartA in stride(from: 0, to: numVectorsA, by: batchSize) {
            let batchEndA = min(batchStartA + batchSize, numVectorsA)
            let batchVectorsA = Array(vectorsA[batchStartA..<batchEndA])
            
            for batchStartB in stride(from: 0, to: numVectorsB, by: batchSize) {
                let batchEndB = min(batchStartB + batchSize, numVectorsB)
                let batchVectorsB = Array(vectorsB[batchStartB..<batchEndB])
                
                // Skip lower triangle for symmetric matrices
                if isSymmetric && batchStartB < batchStartA {
                    continue
                }
                
                // Compute tile
                let tileResult = try await computeDistanceMatrixGPU(
                    vectorsA: batchVectorsA,
                    vectorsB: batchVectorsB,
                    metric: metric,
                    isSymmetric: false  // Always compute full tile
                )
                
                // Copy tile results to full matrix
                for (i, vecAIdx) in (batchStartA..<batchEndA).enumerated() {
                    for (j, vecBIdx) in (batchStartB..<batchEndB).enumerated() {
                        let tileIdx = i * batchVectorsB.count + j
                        let matrixIdx = vecAIdx * numVectorsB + vecBIdx
                        fullMatrix[matrixIdx] = tileResult[tileIdx]
                        
                        // Mirror for symmetric matrices
                        if isSymmetric && vecAIdx != vecBIdx {
                            fullMatrix[vecBIdx * numVectorsB + vecAIdx] = tileResult[tileIdx]
                        }
                    }
                }
            }
        }
        
        return fullMatrix
    }
    
    // MARK: - CPU Implementation
    
    private func computeDistanceMatrixCPU(
        vectorsA: [Vector512],
        vectorsB: [Vector512],
        metric: DistanceMetric,
        isSymmetric: Bool
    ) async -> [Float] {
        
        let numVectorsA = vectorsA.count
        let numVectorsB = vectorsB.count
        var matrix = [Float](repeating: 0, count: numVectorsA * numVectorsB)
        
        await withTaskGroup(of: (Int, Int, Float).self) { group in
            for i in 0..<numVectorsA {
                for j in (isSymmetric ? i : 0)..<numVectorsB {
                    group.addTask {
                        let distance: Float
                        switch metric {
                        case .euclidean:
                            distance = sqrt(vectorsA[i].distanceSquared(to: vectorsB[j]))
                        case .cosine:
                            distance = 1.0 - vectorsA[i].cosineSimilarity(to: vectorsB[j])
                        case .manhattan:
                            // Manhattan distance implementation
                            var sum: Float = 0
                            for k in 0..<512 {
                                sum += abs(vectorsA[i][k] - vectorsB[j][k])
                            }
                            distance = sum
                        default:
                            distance = sqrt(vectorsA[i].distanceSquared(to: vectorsB[j]))
                        }
                        return (i, j, distance)
                    }
                }
            }
            
            for await (i, j, distance) in group {
                matrix[i * numVectorsB + j] = distance
                if isSymmetric && i != j {
                    matrix[j * numVectorsB + i] = distance
                }
            }
        }
        
        return matrix
    }
    
    // MARK: - Async Pipeline
    
    private struct AsyncPipeline {
        let vectorsA: [Vector512]
        let vectorsB: [Vector512]
        let metric: DistanceMetric
        let isSymmetric: Bool
        let computation: () async throws -> [[Float]]
        
        func execute() async throws -> [[Float]] {
            try await computation()
        }
    }
    
    private func createAsyncPipeline(
        vectorsA: [Vector512],
        vectorsB: [Vector512],
        metric: DistanceMetric,
        isSymmetric: Bool
    ) async throws -> AsyncPipeline {
        
        return AsyncPipeline(
            vectorsA: vectorsA,
            vectorsB: vectorsB,
            metric: metric,
            isSymmetric: isSymmetric
        ) { [weak self] in
            guard let self = self else { throw MetalComputeError.deviceNotAvailable }
            return try await self.computeDistanceMatrix(
                vectorsA: vectorsA,
                vectorsB: isSymmetric ? nil : vectorsB,
                metric: metric
            )
        }
    }
    
    // MARK: - Helper Methods
    
    private func packVectorsForGPU(_ vectors: [Vector512]) -> Data {
        var data = Data()
        data.reserveCapacity(vectors.count * 512 * MemoryLayout<Float>.size)
        
        for vector in vectors {
            vector.withUnsafeMetalBytes { bytes in
                data.append(contentsOf: bytes)
            }
        }
        
        return data
    }
    
    private func metricToInt(_ metric: DistanceMetric) -> Int {
        switch metric {
        case .euclidean: return 0
        case .cosine: return 1
        case .manhattan: return 2
        default: return 0
        }
    }
}

// MARK: - Performance Benchmarking

extension MetalDistanceMatrix {
    
    /// Benchmark distance matrix computation comparing CPU vs GPU
    public func benchmark(
        sizes: [Int] = [100, 500, 1000, 2000, 5000],
        metric: DistanceMetric = .euclidean
    ) async throws -> DistanceMatrixBenchmarkResults {
        
        var results = DistanceMatrixBenchmarkResults()
        
        for size in sizes {
            logger.info("Benchmarking size: \(size)x\(size)")
            
            // Generate random vectors
            let vectors = (0..<size).map { _ in
                Vector512.random(in: -1...1)
            }
            
            // CPU benchmark
            let cpuStart = CFAbsoluteTimeGetCurrent()
            let _ = await computeDistanceMatrixCPU(
                vectorsA: vectors,
                vectorsB: vectors,
                metric: metric,
                isSymmetric: true
            )
            let cpuTime = CFAbsoluteTimeGetCurrent() - cpuStart
            
            // GPU benchmark
            let gpuStart = CFAbsoluteTimeGetCurrent()
            let _ = try await computeDistanceMatrixGPU(
                vectorsA: vectors,
                vectorsB: vectors,
                metric: metric,
                isSymmetric: true
            )
            let gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
            
            results.addResult(
                size: size,
                cpuTime: cpuTime,
                gpuTime: gpuTime,
                speedup: cpuTime / gpuTime
            )
            
            logger.info("Size \(size): CPU=\(cpuTime)s, GPU=\(gpuTime)s, Speedup=\(cpuTime/gpuTime)x")
        }
        
        return results
    }
}

// MARK: - Benchmark Results

public struct DistanceMatrixBenchmarkResults {
    public struct Result {
        public let size: Int
        public let cpuTime: Double
        public let gpuTime: Double
        public let speedup: Double
    }
    
    public private(set) var results: [Result] = []
    
    mutating func addResult(size: Int, cpuTime: Double, gpuTime: Double, speedup: Double) {
        results.append(Result(size: size, cpuTime: cpuTime, gpuTime: gpuTime, speedup: speedup))
    }
    
    public var averageSpeedup: Double {
        guard !results.isEmpty else { return 0 }
        return results.reduce(0) { $0 + $1.speedup } / Double(results.count)
    }
    
    public func summary() -> String {
        var summary = "Distance Matrix Benchmark Results:\n"
        summary += "Size\tCPU Time\tGPU Time\tSpeedup\n"
        for result in results {
            summary += "\(result.size)\t\(String(format: "%.3f", result.cpuTime))s\t"
            summary += "\(String(format: "%.3f", result.gpuTime))s\t"
            summary += "\(String(format: "%.2f", result.speedup))x\n"
        }
        summary += "Average Speedup: \(String(format: "%.2f", averageSpeedup))x"
        return summary
    }
}
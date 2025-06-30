// VectorStoreKit: Unified Metal Acceleration Engine
//
// Central acceleration engine that intelligently routes operations between CPU and GPU
// Consolidates buffer management, distance computation, and optimization decisions
//
// Design principles:
// - Single source of truth for acceleration decisions
// - Unified buffer management with automatic lifecycle
// - Smart CPU/GPU routing based on workload characteristics
// - Comprehensive performance monitoring and adaptation

import Foundation
import Metal
import MetalPerformanceShaders
import Accelerate
import simd
import os.log

/// Unified acceleration engine that manages all hardware-accelerated operations
public actor MetalAccelerationEngine {
    
    // MARK: - Properties
    
    /// Singleton instance for the default Metal device
    public static let shared: MetalAccelerationEngine = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported on this device")
        }
        return MetalAccelerationEngine(device: device)
    }()
    
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let bufferManager: MetalBufferPool
    private let pipelineManager: MetalPipelineManager
    private let performanceMonitor: PerformanceMonitor
    private let cpuOptimizer: AccelerateOptimizer
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalAccelerationEngine")
    
    // Adaptive thresholds that adjust based on performance
    private var cpuGpuThreshold: CPUGPUThreshold
    
    // MARK: - Initialization
    
    public init(device: MTLDevice, configuration: AccelerationConfiguration = .default) {
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            fatalError("Failed to create Metal command queue")
        }
        self.commandQueue = queue
        
        // Initialize subsystems
        self.bufferManager = MetalBufferPool(device: device, configuration: configuration.bufferConfig)
        // For now, create a new MetalDevice - this should be refactored to share instances
        do {
            let metalDevice = try MetalDevice()
            self.pipelineManager = try MetalPipelineManager(device: metalDevice)
        } catch {
            fatalError("Failed to create MetalDevice: \(error)")
        }
        self.performanceMonitor = PerformanceMonitor()
        self.cpuOptimizer = AccelerateOptimizer()
        self.cpuGpuThreshold = CPUGPUThreshold(configuration: configuration)
        
        // Pre-warm common pipelines
        Task {
            await pipelineManager.precompileStandardPipelines()
        }
    }
    
    // MARK: - Unified Distance Computation
    
    /// Compute distances with automatic CPU/GPU selection
    public func computeDistances<Vector: SIMD & Sendable>(
        query: Vector,
        candidates: [Vector],
        metric: DistanceMetric
    ) async throws -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Decide execution path based on workload characteristics
        let executionPath = await decideExecutionPath(
            vectorCount: candidates.count,
            vectorDimension: Vector.scalarCount,
            operation: .distanceComputation
        )
        
        logger.debug("Computing distances for \(candidates.count) vectors using \(executionPath)")
        
        let distances: [Float]
        
        switch executionPath {
        case .gpu:
            distances = try await computeDistancesGPU(
                query: query,
                candidates: candidates,
                metric: metric
            )
            
        case .cpu:
            distances = await computeDistancesCPU(
                query: query,
                candidates: candidates,
                metric: metric
            )
            
        case .hybrid:
            // Split workload between CPU and GPU for optimal throughput
            distances = try await computeDistancesHybrid(
                query: query,
                candidates: candidates,
                metric: metric
            )
        }
        
        // Record performance for adaptive threshold adjustment
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        await performanceMonitor.recordOperation(
            type: .distanceComputation,
            path: executionPath,
            duration: duration,
            dataSize: candidates.count * Vector.scalarCount
        )
        
        // Adjust thresholds based on performance
        await cpuGpuThreshold.updateFromPerformance(
            operation: .distanceComputation,
            path: executionPath,
            throughput: Double(candidates.count) / duration
        )
        
        return distances
    }
    
    /// Batch compute distances with optimized execution
    public func batchComputeDistances<Vector: SIMD & Sendable>(
        queries: [Vector],
        candidates: [Vector],
        metric: DistanceMetric,
        topK: Int? = nil
    ) async throws -> [[Float]] where Vector.Scalar: BinaryFloatingPoint {
        
        let totalOperations = queries.count * candidates.count
        let executionPath = await decideExecutionPath(
            vectorCount: totalOperations,
            vectorDimension: Vector.scalarCount,
            operation: .batchDistanceComputation
        )
        
        logger.debug("Batch computing \(totalOperations) distances using \(executionPath)")
        
        switch executionPath {
        case .gpu:
            return try await batchComputeDistancesGPU(
                queries: queries,
                candidates: candidates,
                metric: metric,
                topK: topK
            )
            
        case .cpu:
            return await batchComputeDistancesCPU(
                queries: queries,
                candidates: candidates,
                metric: metric,
                topK: topK
            )
            
        case .hybrid:
            // Use GPU for large batches, CPU for final top-k selection
            let allDistances = try await batchComputeDistancesGPU(
                queries: queries,
                candidates: candidates,
                metric: metric,
                topK: nil
            )
            
            if let k = topK {
                // CPU is often faster for sorting/selection on smaller sets
                return await withTaskGroup(of: (Int, [Float]).self) { group in
                    for (index, distances) in allDistances.enumerated() {
                        group.addTask {
                            let topK = await self.cpuOptimizer.selectTopK(
                                distances: distances,
                                k: k
                            )
                            return (index, topK)
                        }
                    }
                    
                    var results = [[Float]](repeating: [], count: queries.count)
                    for await (index, topKDistances) in group {
                        results[index] = topKDistances
                    }
                    return results
                }
            } else {
                return allDistances
            }
        }
    }
    
    // MARK: - Matrix Operations
    
    /// Perform matrix multiplication with automatic optimization
    public func matrixMultiply(
        matrixA: [[Float]],
        matrixB: [[Float]]
    ) async throws -> [[Float]] {
        
        let totalElements = matrixA.count * matrixA[0].count * matrixB[0].count
        let executionPath = await decideExecutionPath(
            vectorCount: totalElements,
            vectorDimension: 1,
            operation: .matrixMultiplication
        )
        
        switch executionPath {
        case .gpu:
            return try await matrixMultiplyGPU(matrixA: matrixA, matrixB: matrixB)
            
        case .cpu, .hybrid:
            // Use Accelerate's optimized BLAS for CPU path
            return await cpuOptimizer.matrixMultiply(matrixA: matrixA, matrixB: matrixB)
        }
    }
    
    // MARK: - Quantization Operations
    
    /// Quantize vectors with optimal hardware utilization
    public func quantizeVectors<Vector: SIMD & Sendable>(
        vectors: [Vector],
        scheme: QuantizationScheme,
        parameters: QuantizationParameters
    ) async throws -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        let executionPath = await decideExecutionPath(
            vectorCount: vectors.count,
            vectorDimension: Vector.scalarCount,
            operation: .quantization
        )
        
        switch executionPath {
        case .gpu:
            return try await quantizeVectorsGPU(
                vectors: vectors,
                scheme: scheme,
                parameters: parameters
            )
            
        case .cpu, .hybrid:
            // CPU quantization using SIMD operations
            return await cpuOptimizer.quantizeVectors(
                vectors: vectors,
                scheme: scheme,
                parameters: parameters
            )
        }
    }
    
    // MARK: - Execution Path Decision
    
    private func decideExecutionPath(
        vectorCount: Int,
        vectorDimension: Int,
        operation: OperationType
    ) async -> ExecutionPath {
        
        // Get current system state
        let memoryPressure = await performanceMonitor.getCurrentMemoryPressure()
        let gpuUtilization = await performanceMonitor.getCurrentGPUUtilization()
        let cpuUtilization = await performanceMonitor.getCurrentCPUUtilization()
        
        // Check if GPU is available and not overloaded
        guard memoryPressure != .critical,
              gpuUtilization < 0.9 else {
            return .cpu
        }
        
        // Get adaptive threshold for this operation
        let threshold = await cpuGpuThreshold.getThreshold(for: operation)
        
        // Calculate workload size
        let workloadSize = vectorCount * vectorDimension
        
        // Decision logic
        if workloadSize < threshold.minGPUWorkload {
            // Too small for GPU overhead
            return .cpu
        } else if workloadSize > threshold.hybridThreshold && cpuUtilization < 0.5 {
            // Large workload with available CPU - use both
            return .hybrid
        } else if workloadSize > threshold.minGPUWorkload {
            // Good fit for GPU
            return .gpu
        } else {
            // Default to CPU for predictable performance
            return .cpu
        }
    }
    
    // MARK: - GPU Implementations
    
    private func computeDistancesGPU<Vector: SIMD & Sendable>(
        query: Vector,
        candidates: [Vector],
        metric: DistanceMetric
    ) async throws -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        
        // Get optimized compute pipeline for dimension and type
        let basePipelineName = getPipelineName(for: metric, dimension: Vector.scalarCount)
        let pipeline = try await pipelineManager.getOptimizedPipeline(
            baseFunctionName: basePipelineName,
            dimensions: Vector.scalarCount,
            dataType: .float32
        )
        
        // Allocate buffers through unified manager
        let queryBuffer = try await bufferManager.getBuffer(for: query)
        let candidatesBuffer = try await bufferManager.getBuffer(for: candidates)
        let resultsBuffer = try await bufferManager.getBuffer(
            size: candidates.count * MemoryLayout<Float>.size
        )
        
        defer {
            Task {
                await bufferManager.returnBuffer(queryBuffer)
                await bufferManager.returnBuffer(candidatesBuffer)
                await bufferManager.returnBuffer(resultsBuffer)
            }
        }
        
        // Create and execute command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.commandBufferCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queryBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidatesBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
        
        var dimension = Int32(Vector.scalarCount)
        encoder.setBytes(&dimension, length: MemoryLayout<Int32>.size, index: 3)
        
        // Optimal thread configuration
        let threadgroupSize = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (candidates.count + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        // Execute and wait
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        // Extract results
        let resultsPointer = resultsBuffer.contents().bindMemory(
            to: Float.self,
            capacity: candidates.count
        )
        return Array(UnsafeBufferPointer(start: resultsPointer, count: candidates.count))
    }
    
    private func batchComputeDistancesGPU<Vector: SIMD & Sendable>(
        queries: [Vector],
        candidates: [Vector],
        metric: DistanceMetric,
        topK: Int?
    ) async throws -> [[Float]] where Vector.Scalar: BinaryFloatingPoint {
        
        // Use optimized batch kernel
        let basePipelineName = "batch" + getPipelineName(for: metric, dimension: Vector.scalarCount)
        let pipeline = try await pipelineManager.getOptimizedPipeline(
            baseFunctionName: basePipelineName,
            dimensions: Vector.scalarCount,
            dataType: .float32
        )
        
        // Flatten data for GPU processing
        let flatQueries = queries.flatMap { vector in
            (0..<vector.scalarCount).map { Float(vector[$0]) }
        }
        let flatCandidates = candidates.flatMap { vector in
            (0..<vector.scalarCount).map { Float(vector[$0]) }
        }
        
        // Allocate buffers
        let queriesBuffer = try await bufferManager.getBuffer(for: flatQueries)
        let candidatesBuffer = try await bufferManager.getBuffer(for: flatCandidates)
        let resultsBuffer = try await bufferManager.getBuffer(
            size: queries.count * candidates.count * MemoryLayout<Float>.size
        )
        
        defer {
            Task {
                await bufferManager.returnBuffer(queriesBuffer)
                await bufferManager.returnBuffer(candidatesBuffer)
                await bufferManager.returnBuffer(resultsBuffer)
            }
        }
        
        // Execute batch computation
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.commandBufferCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(queriesBuffer, offset: 0, index: 0)
        encoder.setBuffer(candidatesBuffer, offset: 0, index: 1)
        encoder.setBuffer(resultsBuffer, offset: 0, index: 2)
        
        var params: [UInt32] = [
            UInt32(Vector.scalarCount),
            UInt32(queries.count),
            UInt32(candidates.count)
        ]
        encoder.setBytes(&params, length: MemoryLayout<UInt32>.size * 3, index: 3)
        
        // 2D thread configuration for batch processing
        let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupCount = MTLSize(
            width: (candidates.count + 15) / 16,
            height: (queries.count + 15) / 16,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        // Extract results
        let resultsPointer = resultsBuffer.contents().bindMemory(
            to: Float.self,
            capacity: queries.count * candidates.count
        )
        
        var results: [[Float]] = []
        for q in 0..<queries.count {
            let startIdx = q * candidates.count
            let distances = Array(UnsafeBufferPointer(
                start: resultsPointer.advanced(by: startIdx),
                count: candidates.count
            ))
            
            if let k = topK {
                // GPU-based top-k selection if available
                let topKDistances = Array(distances.enumerated()
                    .sorted { $0.element < $1.element }
                    .prefix(k)
                    .map { $0.element })
                results.append(topKDistances)
            } else {
                results.append(distances)
            }
        }
        
        return results
    }
    
    private func matrixMultiplyGPU(
        matrixA: [[Float]],
        matrixB: [[Float]]
    ) async throws -> [[Float]] {
        
        // Use Metal Performance Shaders for optimized matrix multiplication
        let rowsA = matrixA.count
        let colsA = matrixA[0].count
        let rowsB = matrixB.count
        let colsB = matrixB[0].count
        
        guard colsA == rowsB else {
            throw MetalMatrixError.dimensionMismatch(
                a: (rowsA, colsA),
                b: (rowsB, colsB)
            )
        }
        
        // Flatten matrices for GPU
        let flatA = matrixA.flatMap { $0 }
        let flatB = matrixB.flatMap { $0 }
        
        // Create MPS matrices
        let matrixDescriptorA = MPSMatrixDescriptor(
            rows: rowsA,
            columns: colsA,
            rowBytes: colsA * MemoryLayout<Float>.size,
            dataType: .float32
        )
        
        let matrixDescriptorB = MPSMatrixDescriptor(
            rows: rowsB,
            columns: colsB,
            rowBytes: colsB * MemoryLayout<Float>.size,
            dataType: .float32
        )
        
        let matrixDescriptorC = MPSMatrixDescriptor(
            rows: rowsA,
            columns: colsB,
            rowBytes: colsB * MemoryLayout<Float>.size,
            dataType: .float32
        )
        
        // Allocate buffers
        let bufferA = try await bufferManager.getBuffer(for: flatA)
        let bufferB = try await bufferManager.getBuffer(for: flatB)
        let bufferC = try await bufferManager.getBuffer(
            size: rowsA * colsB * MemoryLayout<Float>.size
        )
        
        defer {
            Task {
                await bufferManager.returnBuffer(bufferA)
                await bufferManager.returnBuffer(bufferB)
                await bufferManager.returnBuffer(bufferC)
            }
        }
        
        // Create MPS matrices
        let mpsA = MPSMatrix(buffer: bufferA, descriptor: matrixDescriptorA)
        let mpsB = MPSMatrix(buffer: bufferB, descriptor: matrixDescriptorB)
        let mpsC = MPSMatrix(buffer: bufferC, descriptor: matrixDescriptorC)
        
        // Create matrix multiplication kernel
        let matrixMultiplication = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: rowsA,
            resultColumns: colsB,
            interiorColumns: colsA,
            alpha: 1.0,
            beta: 0.0
        )
        
        // Execute
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MetalComputeError.commandBufferCreationFailed
        }
        
        matrixMultiplication.encode(
            commandBuffer: commandBuffer,
            leftMatrix: mpsA,
            rightMatrix: mpsB,
            resultMatrix: mpsC
        )
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        // Extract results
        let resultsPointer = bufferC.contents().bindMemory(
            to: Float.self,
            capacity: rowsA * colsB
        )
        
        var result: [[Float]] = []
        for row in 0..<rowsA {
            let startIdx = row * colsB
            let rowData = Array(UnsafeBufferPointer(
                start: resultsPointer.advanced(by: startIdx),
                count: colsB
            ))
            result.append(rowData)
        }
        
        return result
    }
    
    private func quantizeVectorsGPU<Vector: SIMD & Sendable>(
        vectors: [Vector],
        scheme: QuantizationScheme,
        parameters: QuantizationParameters
    ) async throws -> [QuantizedVector] where Vector.Scalar: BinaryFloatingPoint {
        
        // Get quantization pipeline with proper constants
        var constants = ShaderFunctionConstants()
        constants.setValue(.uint32(UInt32(Vector.scalarCount)), at: 0)
        constants.setValue(.uint32(UInt32(parameters.numberOfClusters)), at: 1)
        
        let pipelineName = "quantize_\(scheme.rawValue)"
        let pipeline = try await pipelineManager.getPipelineWithConstants(
            functionName: pipelineName,
            constants: constants
        )
        
        // Implement GPU quantization
        // This would use custom Metal kernels for different quantization schemes
        // For now, fall back to CPU implementation
        return await cpuOptimizer.quantizeVectors(
            vectors: vectors,
            scheme: scheme,
            parameters: parameters
        )
    }
    
    // MARK: - CPU Implementations
    
    private func computeDistancesCPU<Vector: SIMD & Sendable>(
        query: Vector,
        candidates: [Vector],
        metric: DistanceMetric
    ) async -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        
        return await cpuOptimizer.computeDistances(
            query: query,
            candidates: candidates,
            metric: metric
        )
    }
    
    private func batchComputeDistancesCPU<Vector: SIMD & Sendable>(
        queries: [Vector],
        candidates: [Vector],
        metric: DistanceMetric,
        topK: Int?
    ) async -> [[Float]] where Vector.Scalar: BinaryFloatingPoint {
        
        return await cpuOptimizer.batchComputeDistances(
            queries: queries,
            candidates: candidates,
            metric: metric,
            topK: topK
        )
    }
    
    // MARK: - Hybrid Implementation
    
    private func computeDistancesHybrid<Vector: SIMD & Sendable>(
        query: Vector,
        candidates: [Vector],
        metric: DistanceMetric
    ) async throws -> [Float] where Vector.Scalar: BinaryFloatingPoint {
        
        // Split workload between CPU and GPU
        let splitPoint = candidates.count / 2
        let gpuCandidates = Array(candidates[..<splitPoint])
        let cpuCandidates = Array(candidates[splitPoint...])
        
        // Execute in parallel
        async let gpuDistances = computeDistancesGPU(
            query: query,
            candidates: gpuCandidates,
            metric: metric
        )
        
        async let cpuDistances = computeDistancesCPU(
            query: query,
            candidates: cpuCandidates,
            metric: metric
        )
        
        // Combine results
        let results = try await gpuDistances + cpuDistances
        return results
    }
    
    // MARK: - Helper Methods
    
    private func getPipelineName(for metric: DistanceMetric, dimension: Int) -> String {
        let baseName: String
        
        switch metric {
        case .euclidean:
            baseName = "euclideanDistance"
        case .cosine:
            baseName = "cosineDistance"
        case .manhattan:
            baseName = "manhattanDistance"
        case .dotProduct:
            baseName = "dotProduct"
        default:
            baseName = "euclideanDistance"
        }
        
        return baseName
    }
    
    // MARK: - Tensor Operations for ML
    
    /// Subtract two tensors element-wise
    public func subtractTensors(_ a: MetalTensor, _ b: MetalTensor) async throws -> MetalTensor {
        guard a.shape == b.shape else {
            throw MetalTensorError.shapeMismatch(a: a.shape, b: b.shape)
        }
        
        let totalElements = a.shape.reduce(1, *)
        let executionPath = await decideExecutionPath(
            vectorCount: totalElements,
            vectorDimension: 1,
            operation: .neuralInference
        )
        
        switch executionPath {
        case .gpu:
            return try await tensorOperationGPU(a, b, operation: "subtract")
        case .cpu, .hybrid:
            return await tensorOperationCPU(a, b) { $0 - $1 }
        }
    }
    
    /// Square tensor elements
    public func squareTensor(_ tensor: MetalTensor) async throws -> MetalTensor {
        let totalElements = tensor.shape.reduce(1, *)
        let executionPath = await decideExecutionPath(
            vectorCount: totalElements,
            vectorDimension: 1,
            operation: .neuralInference
        )
        
        switch executionPath {
        case .gpu:
            return try await tensorUnaryOperationGPU(tensor, operation: "square")
        case .cpu, .hybrid:
            return await tensorUnaryOperationCPU(tensor) { $0 * $0 }
        }
    }
    
    /// Reduce sum tensor elements
    public func reduceSumTensor(_ tensor: MetalTensor) async throws -> Float {
        let totalElements = tensor.shape.reduce(1, *)
        let executionPath = await decideExecutionPath(
            vectorCount: totalElements,
            vectorDimension: 1,
            operation: .neuralInference
        )
        
        switch executionPath {
        case .gpu:
            return try await tensorReductionGPU(tensor, operation: "sum")
        case .cpu, .hybrid:
            let pointer = tensor.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
            var sum: Float = 0
            vDSP_sve(pointer, 1, &sum, vDSP_Length(totalElements))
            return sum
        }
    }
    
    /// Scale tensor by scalar
    public func scaleTensor(_ tensor: MetalTensor, by scalar: Float) async throws -> MetalTensor {
        let totalElements = tensor.shape.reduce(1, *)
        let executionPath = await decideExecutionPath(
            vectorCount: totalElements,
            vectorDimension: 1,
            operation: .neuralInference
        )
        
        switch executionPath {
        case .gpu:
            return try await tensorScalarOperationGPU(tensor, scalar: scalar, operation: "multiply")
        case .cpu, .hybrid:
            return await tensorScalarOperationCPU(tensor, scalar: scalar) { $0 * $1 }
        }
    }
    
    // MARK: - Private Tensor Operation Implementations
    
    private func tensorOperationGPU(_ a: MetalTensor, _ b: MetalTensor, operation: String) async throws -> MetalTensor {
        let pipeline = try await pipelineManager.getPipeline(
            functionName: "tensor_\(operation)"
        )
        
        let totalElements = a.shape.reduce(1, *)
        let resultBuffer = try await bufferManager.getBuffer(
            size: totalElements * MemoryLayout<Float>.size
        )
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.commandBufferCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(a.buffer, offset: 0, index: 0)
        encoder.setBuffer(b.buffer, offset: 0, index: 1)
        encoder.setBuffer(resultBuffer, offset: 0, index: 2)
        
        var count = UInt32(totalElements)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        
        let threadgroupSize = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (totalElements + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        return MetalTensor(buffer: resultBuffer, shape: a.shape, device: device)
    }
    
    private func tensorUnaryOperationGPU(_ tensor: MetalTensor, operation: String) async throws -> MetalTensor {
        let pipeline = try await pipelineManager.getPipeline(
            functionName: "tensor_\(operation)"
        )
        
        let totalElements = tensor.shape.reduce(1, *)
        let resultBuffer = try await bufferManager.getBuffer(
            size: totalElements * MemoryLayout<Float>.size
        )
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.commandBufferCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(tensor.buffer, offset: 0, index: 0)
        encoder.setBuffer(resultBuffer, offset: 0, index: 1)
        
        var count = UInt32(totalElements)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
        
        let threadgroupSize = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (totalElements + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        return MetalTensor(buffer: resultBuffer, shape: tensor.shape, device: device)
    }
    
    private func tensorReductionGPU(_ tensor: MetalTensor, operation: String) async throws -> Float {
        // For simplicity, fall back to CPU for reductions
        // A proper GPU implementation would use parallel reduction
        let totalElements = tensor.shape.reduce(1, *)
        let pointer = tensor.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        
        switch operation {
        case "sum":
            var sum: Float = 0
            vDSP_sve(pointer, 1, &sum, vDSP_Length(totalElements))
            return sum
        case "max":
            var max: Float = 0
            vDSP_maxv(pointer, 1, &max, vDSP_Length(totalElements))
            return max
        case "min":
            var min: Float = 0
            vDSP_minv(pointer, 1, &min, vDSP_Length(totalElements))
            return min
        default:
            throw MetalTensorError.unsupportedOperation(operation)
        }
    }
    
    private func tensorScalarOperationGPU(_ tensor: MetalTensor, scalar: Float, operation: String) async throws -> MetalTensor {
        let pipeline = try await pipelineManager.getPipeline(
            functionName: "tensor_scalar_\(operation)"
        )
        
        let totalElements = tensor.shape.reduce(1, *)
        let resultBuffer = try await bufferManager.getBuffer(
            size: totalElements * MemoryLayout<Float>.size
        )
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalComputeError.commandBufferCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(tensor.buffer, offset: 0, index: 0)
        encoder.setBuffer(resultBuffer, offset: 0, index: 1)
        encoder.setBytes(&scalar, length: MemoryLayout<Float>.size, index: 2)
        
        var count = UInt32(totalElements)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        
        let threadgroupSize = MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (totalElements + threadgroupSize.width - 1) / threadgroupSize.width,
            height: 1,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        return MetalTensor(buffer: resultBuffer, shape: tensor.shape, device: device)
    }
    
    private func tensorOperationCPU(_ a: MetalTensor, _ b: MetalTensor, operation: (Float, Float) -> Float) async -> MetalTensor {
        let totalElements = a.shape.reduce(1, *)
        let aPointer = a.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        let bPointer = b.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        
        let resultBuffer = device.makeBuffer(
            length: totalElements * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        
        // Use vDSP for vectorized operations
        vDSP_vsub(bPointer, 1, aPointer, 1, resultPointer, 1, vDSP_Length(totalElements))
        
        return MetalTensor(buffer: resultBuffer, shape: a.shape, device: device)
    }
    
    private func tensorUnaryOperationCPU(_ tensor: MetalTensor, operation: (Float) -> Float) async -> MetalTensor {
        let totalElements = tensor.shape.reduce(1, *)
        let pointer = tensor.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        
        let resultBuffer = device.makeBuffer(
            length: totalElements * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        
        // Use vDSP for square operation
        vDSP_vsq(pointer, 1, resultPointer, 1, vDSP_Length(totalElements))
        
        return MetalTensor(buffer: resultBuffer, shape: tensor.shape, device: device)
    }
    
    private func tensorScalarOperationCPU(_ tensor: MetalTensor, scalar: Float, operation: (Float, Float) -> Float) async -> MetalTensor {
        let totalElements = tensor.shape.reduce(1, *)
        let pointer = tensor.buffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        
        let resultBuffer = device.makeBuffer(
            length: totalElements * MemoryLayout<Float>.size,
            options: .storageModeShared
        )!
        let resultPointer = resultBuffer.contents().bindMemory(to: Float.self, capacity: totalElements)
        
        // Use vDSP for scalar multiplication
        var scalarValue = scalar
        vDSP_vsmul(pointer, 1, &scalarValue, resultPointer, 1, vDSP_Length(totalElements))
        
        return MetalTensor(buffer: resultBuffer, shape: tensor.shape, device: device)
    }
    
    // MARK: - Performance Monitoring
    
    /// Get current performance statistics
    public func getPerformanceStatistics() async -> AccelerationStatistics {
        let bufferStats = await bufferManager.getStatistics()
        let performanceStats = await performanceMonitor.getStatistics()
        let pipelineStats = await pipelineManager.cacheStatistics
        
        return AccelerationStatistics(
            bufferStatistics: bufferStats,
            performanceStatistics: performanceStats,
            pipelineStatistics: pipelineStats,
            cpuGpuThresholds: await cpuGpuThreshold.getCurrentThresholds()
        )
    }
    
    /// Reset performance statistics and adaptive thresholds
    public func resetStatistics() async {
        await performanceMonitor.reset()
        await cpuGpuThreshold.reset()
    }
}

// MARK: - Supporting Types

/// Execution path for operations
public enum ExecutionPath: String, Sendable {
    case cpu = "CPU"
    case gpu = "GPU"
    case hybrid = "Hybrid"
}

/// Operation types for performance tracking
public enum OperationType: String, Sendable {
    case distanceComputation
    case batchDistanceComputation
    case matrixMultiplication
    case quantization
    case neuralInference
}

/// Configuration for the acceleration engine
public struct AccelerationConfiguration: Sendable {
    public let bufferConfig: MetalBufferPoolConfiguration
    public let enableAdaptiveThresholds: Bool
    public let enableProfiling: Bool
    public let maxMemoryUsage: Int
    
    public static let `default` = AccelerationConfiguration(
        bufferConfig: .research,
        enableAdaptiveThresholds: true,
        enableProfiling: true,
        maxMemoryUsage: 1_073_741_824 // 1GB
    )
    
    public static let production = AccelerationConfiguration(
        bufferConfig: MetalBufferPoolConfiguration(
            maxBuffersPerSize: 32,
            preallocationSizes: [1024, 4096, 16384, 65536, 262144, 1048576],
            maxMemoryUsage: 2 * 1024 * 1024 * 1024, // 2GB
            enableProfiling: false
        ),
        enableAdaptiveThresholds: true,
        enableProfiling: false,
        maxMemoryUsage: 2_147_483_648 // 2GB
    )
}

/// Adaptive CPU/GPU threshold management
public actor CPUGPUThreshold {
    private var thresholds: [OperationType: OperationThreshold]
    private let configuration: AccelerationConfiguration
    
    struct OperationThreshold {
        var minGPUWorkload: Int
        var hybridThreshold: Int
        let baseMinGPU: Int
        let baseHybrid: Int
        
        mutating func adjust(factor: Double) {
            minGPUWorkload = Int(Double(baseMinGPU) * factor)
            hybridThreshold = Int(Double(baseHybrid) * factor)
        }
    }
    
    init(configuration: AccelerationConfiguration) {
        self.configuration = configuration
        
        // Initialize with default thresholds
        self.thresholds = [
            .distanceComputation: OperationThreshold(
                minGPUWorkload: 1000,
                hybridThreshold: 10000,
                baseMinGPU: 1000,
                baseHybrid: 10000
            ),
            .batchDistanceComputation: OperationThreshold(
                minGPUWorkload: 5000,
                hybridThreshold: 50000,
                baseMinGPU: 5000,
                baseHybrid: 50000
            ),
            .matrixMultiplication: OperationThreshold(
                minGPUWorkload: 10000,
                hybridThreshold: 100000,
                baseMinGPU: 10000,
                baseHybrid: 100000
            ),
            .quantization: OperationThreshold(
                minGPUWorkload: 1000,
                hybridThreshold: 10000,
                baseMinGPU: 1000,
                baseHybrid: 10000
            ),
            .neuralInference: OperationThreshold(
                minGPUWorkload: 100,
                hybridThreshold: 1000,
                baseMinGPU: 100,
                baseHybrid: 1000
            )
        ]
    }
    
    func getThreshold(for operation: OperationType) -> OperationThreshold {
        thresholds[operation] ?? OperationThreshold(
            minGPUWorkload: 1000,
            hybridThreshold: 10000,
            baseMinGPU: 1000,
            baseHybrid: 10000
        )
    }
    
    func updateFromPerformance(
        operation: OperationType,
        path: ExecutionPath,
        throughput: Double
    ) {
        guard configuration.enableAdaptiveThresholds else { return }
        
        // Simple adaptive algorithm - can be made more sophisticated
        // If GPU throughput is significantly better, lower threshold
        // If CPU throughput is competitive, raise threshold
        
        // This is a placeholder for more sophisticated adaptation
        // In production, would use historical data and ML models
    }
    
    func getCurrentThresholds() -> [String: Any] {
        thresholds.mapValues { threshold in
            [
                "minGPUWorkload": threshold.minGPUWorkload,
                "hybridThreshold": threshold.hybridThreshold
            ]
        }
    }
    
    func reset() {
        for (operation, var threshold) in thresholds {
            threshold.adjust(factor: 1.0)
            thresholds[operation] = threshold
        }
    }
}

/// Performance monitoring for adaptive optimization
public actor PerformanceMonitor {
    private var operationHistory: [OperationRecord] = []
    private let maxHistorySize = 1000
    
    struct OperationRecord {
        let timestamp: Date
        let type: OperationType
        let path: ExecutionPath
        let duration: TimeInterval
        let dataSize: Int
        let throughput: Double
    }
    
    func recordOperation(
        type: OperationType,
        path: ExecutionPath,
        duration: TimeInterval,
        dataSize: Int
    ) {
        let throughput = Double(dataSize) / duration
        
        let record = OperationRecord(
            timestamp: Date(),
            type: type,
            path: path,
            duration: duration,
            dataSize: dataSize,
            throughput: throughput
        )
        
        operationHistory.append(record)
        
        // Maintain history size
        if operationHistory.count > maxHistorySize {
            operationHistory.removeFirst()
        }
    }
    
    func getCurrentMemoryPressure() -> SystemMemoryPressure {
        // Placeholder - would integrate with system memory monitoring
        return .normal
    }
    
    func getCurrentGPUUtilization() -> Double {
        // Placeholder - would integrate with Metal performance counters
        return 0.5
    }
    
    func getCurrentCPUUtilization() -> Double {
        // Placeholder - would integrate with system monitoring
        return 0.3
    }
    
    func getStatistics() -> PerformanceStatistics {
        let recentOperations = operationHistory.suffix(100)
        
        var stats = PerformanceStatistics()
        
        // Calculate average throughput by operation type and path
        for record in recentOperations {
            let key = "\(record.type.rawValue)_\(record.path.rawValue)"
            stats.averageThroughput[key] = (stats.averageThroughput[key] ?? 0) + record.throughput
            stats.operationCounts[key] = (stats.operationCounts[key] ?? 0) + 1
        }
        
        // Normalize averages
        for key in stats.averageThroughput.keys {
            if let count = stats.operationCounts[key], count > 0 {
                stats.averageThroughput[key]! /= Double(count)
            }
        }
        
        return stats
    }
    
    func reset() {
        operationHistory.removeAll()
    }
}

/// Acceleration statistics
public struct AccelerationStatistics: Sendable {
    public let bufferStatistics: BufferPoolStatistics
    public let performanceStatistics: PerformanceStatistics
    public let pipelineStatistics: PipelineCacheStatistics
    public let cpuGpuThresholds: [String: Any]
}

/// Performance statistics
public struct PerformanceStatistics: Sendable {
    public var averageThroughput: [String: Double] = [:]
    public var operationCounts: [String: Int] = [:]
}

/// System memory pressure levels
public enum SystemMemoryPressure: Sendable {
    case normal
    case warning
    case critical
}

/// Metal tensor wrapper for ML operations
public struct MetalTensor: Sendable {
    public let buffer: MTLBuffer
    public let shape: [Int]
    public let device: MTLDevice
    
    public init(buffer: MTLBuffer, shape: [Int], device: MTLDevice) {
        self.buffer = buffer
        self.shape = shape
        self.device = device
    }
}

/// Metal tensor errors
public enum MetalTensorError: Error {
    case shapeMismatch(a: [Int], b: [Int])
    case unsupportedOperation(String)
    case bufferCreationFailed
}
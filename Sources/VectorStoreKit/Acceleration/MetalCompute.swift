// VectorStoreKit: Metal Compute Orchestrator
//
// Simplified orchestrator for Metal compute operations

import Foundation
@preconcurrency import Metal
import simd
import os.log

/// Orchestrates Metal compute operations through specialized components
public actor MetalCompute {
    
    // MARK: - Configuration
    
    /// Configuration for Metal compute operations
    public typealias Configuration = MetalComputeConfiguration
    
    // MARK: - Properties
    
    private let configuration: Configuration
    private let device: MetalDevice
    private let bufferPool: MetalBufferPool
    private let pipelineManager: MetalPipelineManager
    private let profiler: MetalProfiler?
    
    // Specialized compute engines
    private let distanceCompute: MetalDistanceCompute
    private let matrixCompute: MetalMatrixCompute
    private let quantizationCompute: MetalQuantizationCompute
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalCompute")
    
    // MARK: - Initialization
    
    public init(configuration: Configuration = MetalComputeConfiguration.research) async throws {
        self.configuration = configuration
        
        // Initialize core components
        self.device = try MetalDevice()
        self.bufferPool = MetalBufferPool(device: await device.device, configuration: MetalBufferPool.Configuration(
            maxBuffersPerSize: configuration.bufferPoolConfig.maxBuffersPerSize,
            preallocationSizes: configuration.bufferPoolConfig.preallocationSizes
        ))
        self.pipelineManager = MetalPipelineManager(device: device)
        self.profiler = configuration.enableProfiling ? MetalProfiler() : nil
        
        // Initialize compute engines
        self.distanceCompute = MetalDistanceCompute(
            device: device,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager,
            profiler: profiler
        )
        
        self.matrixCompute = await MetalMatrixCompute(
            device: device,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager,
            profiler: profiler
        )
        
        self.quantizationCompute = MetalQuantizationCompute(
            device: device,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager,
            profiler: profiler
        )
        
        // Precompile standard pipelines
        await pipelineManager.precompileStandardPipelines()
        
        let deviceName = device.capabilities.deviceName
        logger.info("MetalCompute initialized with \(deviceName)")
    }
    
    // MARK: - Distance Computation
    
    /// Compute distances with automatic GPU/CPU selection
    public func computeDistances<Vector: SIMD & Sendable>(
        query: Vector,
        candidates: [Vector],
        metric: DistanceMetric
    ) async throws -> (distances: [Float], metrics: MetalPerformanceMetrics)
    where Vector.Scalar: BinaryFloatingPoint {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Choose execution path
        let supportsMetric = device.capabilities.supportsMetric(metric)
        let useGPU = candidates.count >= configuration.minBatchSizeForGPU && supportsMetric
        
        let distances: [Float]
        var gpuTime: TimeInterval = 0
        
        if useGPU {
            let gpuStart = CFAbsoluteTimeGetCurrent()
            distances = try await distanceCompute.computeDistances(
                query: query,
                candidates: candidates,
                metric: metric
            )
            gpuTime = CFAbsoluteTimeGetCurrent() - gpuStart
        } else {
            distances = await distanceCompute.computeDistancesCPU(
                query: query,
                candidates: candidates,
                metric: metric
            )
        }
        
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        
        let metrics = MetalPerformanceMetrics(
            gpuTime: gpuTime,
            cpuOverhead: totalTime - gpuTime,
            memoryBandwidth: calculateMemoryBandwidth(
                batchSize: candidates.count,
                vectorSize: MemoryLayout<Vector>.size,
                time: totalTime
            ),
            operationsPerSecond: Float(candidates.count) / Float(totalTime),
            gpuUtilization: useGPU ? (await profiler?.lastGPUUtilization ?? 0) : 0,
            memoryUtilization: await profiler?.lastMemoryUtilization ?? 0,
            usedGPU: useGPU,
            hardwareMetrics: [:]
        )
        
        return (distances: distances, metrics: metrics)
    }
    
    /// Compute multiple distance metrics
    public func computeMultipleDistances<Vector: SIMD & Sendable>(
        query: Vector,
        candidates: [Vector],
        metrics: Set<DistanceMetric>
    ) async throws -> (results: [DistanceMetric: [Float]], metrics: MetalPerformanceMetrics)
    where Vector.Scalar: BinaryFloatingPoint {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let results = try await distanceCompute.computeMultipleDistances(
            query: query,
            candidates: candidates,
            metrics: metrics
        )
        
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        
        let performanceMetrics = MetalPerformanceMetrics(
            gpuTime: totalTime * 0.8, // Estimate
            cpuOverhead: totalTime * 0.2,
            memoryBandwidth: calculateMemoryBandwidth(
                batchSize: candidates.count * metrics.count,
                vectorSize: MemoryLayout<Vector>.size,
                time: totalTime
            ),
            operationsPerSecond: Float(candidates.count * metrics.count) / Float(totalTime),
            gpuUtilization: await profiler?.lastGPUUtilization ?? 0,
            memoryUtilization: await profiler?.lastMemoryUtilization ?? 0,
            usedGPU: true,
            hardwareMetrics: [:]
        )
        
        return (results: results, metrics: performanceMetrics)
    }
    
    // MARK: - Matrix Operations
    
    /// Perform matrix multiplication
    public func matrixMultiply(
        matrixA: [[Float]],
        matrixB: [[Float]],
        useAMX: Bool = true
    ) async throws -> (result: [[Float]], metrics: MetalPerformanceMetrics) {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let hasAMX = device.capabilities.hasAMX
        let result = try await matrixCompute.matrixMultiply(
            matrixA: matrixA,
            matrixB: matrixB,
            useAMX: useAMX && hasAMX
        )
        
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        
        let metrics = MetalPerformanceMetrics(
            gpuTime: totalTime * 0.9,
            cpuOverhead: totalTime * 0.1,
            memoryBandwidth: calculateMatrixMemoryBandwidth(
                rowsA: matrixA.count,
                colsA: matrixA[0].count,
                colsB: matrixB[0].count,
                time: totalTime
            ),
            operationsPerSecond: Float(matrixA.count * matrixA[0].count * matrixB[0].count * 2) / Float(totalTime),
            gpuUtilization: await profiler?.lastGPUUtilization ?? 0,
            memoryUtilization: await profiler?.lastMemoryUtilization ?? 0,
            usedGPU: true,
            hardwareMetrics: [:]
        )
        
        return (result: result, metrics: metrics)
    }
    
    // MARK: - Quantization Operations
    
    /// Quantize vectors
    public func quantizeVectors<Vector: SIMD & Sendable>(
        vectors: [Vector],
        scheme: QuantizationScheme,
        parameters: QuantizationParameters
    ) async throws -> (quantized: [QuantizedVector], metrics: MetalPerformanceMetrics)
    where Vector.Scalar: BinaryFloatingPoint {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let quantizedVectors = try await quantizationCompute.quantizeVectors(
            vectors: vectors,
            scheme: scheme,
            parameters: parameters
        )
        
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        
        // Calculate compression ratio
        let originalSize = vectors.count * MemoryLayout<Vector>.size
        let compressedSize = quantizedVectors.reduce(0) { $0 + $1.codes.count }
        let compressionRatio = Float(originalSize) / Float(compressedSize)
        
        let metrics = MetalPerformanceMetrics(
            gpuTime: totalTime * 0.8, // Estimate GPU portion
            cpuOverhead: totalTime * 0.2,
            memoryBandwidth: calculateMemoryBandwidth(
                batchSize: vectors.count,
                vectorSize: MemoryLayout<Vector>.size,
                time: totalTime
            ),
            operationsPerSecond: Float(vectors.count) / Float(totalTime),
            gpuUtilization: await profiler?.lastGPUUtilization ?? 0,
            memoryUtilization: await profiler?.lastMemoryUtilization ?? 0,
            usedGPU: vectors.count >= configuration.minBatchSizeForGPU,
            hardwareMetrics: ["compressionRatio": compressionRatio]
        )
        
        return (quantized: quantizedVectors, metrics: metrics)
    }
    
    // MARK: - Performance Analysis
    
    /// Get performance statistics
    public func getPerformanceStatistics() async -> MetalPerformanceStatistics {
        let bufferStats = await bufferPool.getStatistics()
        let _ = await pipelineManager.cacheStatistics
        let profilerStats = await profiler?.getStatistics()
        
        return MetalPerformanceStatistics(
            totalOperations: profilerStats?.totalOperations ?? 0,
            averageGPUUtilization: profilerStats?.averageGPUUtilization ?? 0,
            averageMemoryUtilization: Float(profilerStats?.averageLatency ?? 0),
            peakMemoryBandwidth: profilerStats?.peakMemoryBandwidth ?? 0,
            averageLatency: profilerStats?.averageLatency ?? 0,
            hardwareCapabilities: device.capabilities,
            bufferPoolStatistics: bufferStats
        )
    }
    
    /// Reset performance counters
    public func resetPerformanceCounters() async {
        await profiler?.reset()
    }
    
    // MARK: - Private Helpers
    
    private func calculateMemoryBandwidth(batchSize: Int, vectorSize: Int, time: TimeInterval) -> Float {
        let bytesTransferred = batchSize * vectorSize * 2 // Read + write
        return Float(bytesTransferred) / Float(time) / (1024 * 1024 * 1024) // GB/s
    }
    
    private func calculateMatrixMemoryBandwidth(rowsA: Int, colsA: Int, colsB: Int, time: TimeInterval) -> Float {
        let bytesTransferred = (rowsA * colsA + colsA * colsB + rowsA * colsB) * MemoryLayout<Float>.size
        return Float(bytesTransferred) / Float(time) / (1024 * 1024 * 1024) // GB/s
    }
}

// MARK: - Supporting Types

/// Performance statistics for Metal operations
public struct MetalPerformanceStatistics: Sendable {
    public let totalOperations: UInt64
    public let averageGPUUtilization: Float
    public let averageMemoryUtilization: Float
    public let peakMemoryBandwidth: Float
    public let averageLatency: TimeInterval
    public let hardwareCapabilities: HardwareCapabilities
    public let bufferPoolStatistics: BufferPoolStatistics
}

// Note: MetalComputeError is defined in MetalErrors.swift

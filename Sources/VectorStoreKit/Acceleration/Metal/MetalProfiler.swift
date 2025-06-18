// VectorStoreKit: Metal Performance Profiler
//
// Performance monitoring and profiling for Metal operations

import Foundation
import os.log

/// Performance profiler for Metal operations
public actor MetalProfiler {
    
    // MARK: - Properties
    
    private let enabled: Bool
    private var operationHistory: [OperationRecord] = []
    private var totalOperations: UInt64 = 0
    
    // Performance metrics
    private var gpuUtilizationHistory: [Float] = []
    private var memoryUtilizationHistory: [Float] = []
    private var latencyHistory: [TimeInterval] = []
    private var bandwidthHistory: [Float] = []
    
    // Hardware metrics
    private var lastHardwareMetrics: [String: Float] = [:]
    
    // GPU-specific metrics
    private var commandBufferPoolHits: Int = 0
    private var commandBufferPoolMisses: Int = 0
    private var totalGPUMemoryAllocated: Int = 0
    private var currentGPUMemoryUsage: Int = 0
    private var peakGPUMemoryUsage: Int = 0
    private var kernelDispatches: Int = 0
    private var totalThreadsDispatched: Int = 0
    private var kernelExecutionCounts: [String: Int] = [:]
    private var commandBufferSubmissions: Int = 0
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalProfiler")
    
    // MARK: - Operation Types
    
    public enum OperationType: String, CaseIterable, Sendable {
        case distanceComputation = "distance"
        case matrixMultiplication = "matrix"
        case quantization = "quantization"
        case dequantization = "dequantization"
        case bufferTransfer = "transfer"
        case pipelineCompilation = "compilation"
        case optimization = "optimization"
        case commandBufferSubmission = "commandBuffer"
        case kernelExecution = "kernel"
    }
    
    public enum ProfileEvent {
        case commandBufferPoolHit
        case commandBufferPoolMiss
        case gpuMemoryAllocation(size: Int)
        case gpuMemoryDeallocation(size: Int)
        case kernelDispatch(name: String, threads: Int)
        case pipelineCreation(name: String)
    }
    
    // MARK: - Initialization
    
    public init(enabled: Bool = true) {
        self.enabled = enabled
        if enabled {
            logger.info("Metal profiler enabled")
        }
    }
    
    // MARK: - Recording Operations
    
    /// Record an operation with its metrics
    public func recordOperation(
        _ type: OperationType,
        duration: TimeInterval,
        dataSize: Int,
        gpuUtilization: Float? = nil,
        memoryBandwidth: Float? = nil
    ) {
        guard enabled else { return }
        
        totalOperations += 1
        
        let record = OperationRecord(
            type: type,
            timestamp: Date(),
            duration: duration,
            dataSize: dataSize,
            gpuUtilization: gpuUtilization ?? estimateGPUUtilization(duration: duration, dataSize: dataSize),
            memoryBandwidth: memoryBandwidth ?? estimateMemoryBandwidth(duration: duration, dataSize: dataSize)
        )
        
        operationHistory.append(record)
        
        // Update metrics
        if let gpu = gpuUtilization {
            gpuUtilizationHistory.append(gpu)
        }
        latencyHistory.append(duration)
        if let bandwidth = memoryBandwidth {
            bandwidthHistory.append(bandwidth)
        }
        
        // Maintain history size
        maintainHistoryLimits()
        
        logger.debug("\(type.rawValue): \(duration * 1000)ms, \(dataSize) items")
    }
    
    /// Update hardware metrics
    public func updateHardwareMetrics(_ metrics: [String: Float]) {
        guard enabled else { return }
        lastHardwareMetrics = metrics
    }
    
    /// Record a profiling event
    public func recordEvent(_ event: ProfileEvent) {
        guard enabled else { return }
        
        switch event {
        case .commandBufferPoolHit:
            commandBufferPoolHits += 1
        case .commandBufferPoolMiss:
            commandBufferPoolMisses += 1
        case .gpuMemoryAllocation(let size):
            totalGPUMemoryAllocated += size
            currentGPUMemoryUsage += size
            peakGPUMemoryUsage = max(peakGPUMemoryUsage, currentGPUMemoryUsage)
        case .gpuMemoryDeallocation(let size):
            currentGPUMemoryUsage = max(0, currentGPUMemoryUsage - size)
        case .kernelDispatch(let name, let threads):
            kernelDispatches += 1
            totalThreadsDispatched += threads
            kernelExecutionCounts[name, default: 0] += 1
        case .pipelineCreation(let name):
            logger.debug("Pipeline created: \(name)")
        }
    }
    
    // MARK: - Statistics
    
    /// Get overall performance statistics
    public func getStatistics() -> ProfilerStatistics {
        let operationCounts = Dictionary(
            grouping: operationHistory,
            by: { $0.type }
        ).mapValues { $0.count }
        
        // Calculate GPU-specific metrics
        let totalCommandBuffers = commandBufferPoolHits + commandBufferPoolMisses
        let commandBufferPoolHitRate = totalCommandBuffers > 0 ?
            Float(commandBufferPoolHits) / Float(totalCommandBuffers) : 0
        
        let avgThreadsPerDispatch = kernelDispatches > 0 ?
            totalThreadsDispatched / kernelDispatches : 0
        
        let topKernels = kernelExecutionCounts
            .sorted { $0.value > $1.value }
            .prefix(5)
            .map { ($0.key, $0.value) }
        
        return ProfilerStatistics(
            totalOperations: totalOperations,
            operationCounts: operationCounts,
            averageGPUUtilization: average(gpuUtilizationHistory),
            peakGPUUtilization: gpuUtilizationHistory.max() ?? 0,
            averageLatency: average(latencyHistory),
            latencyPercentiles: calculatePercentiles(latencyHistory),
            averageMemoryBandwidth: average(bandwidthHistory),
            peakMemoryBandwidth: bandwidthHistory.max() ?? 0,
            hardwareMetrics: lastHardwareMetrics,
            gpuMetrics: GPUSpecificMetrics(
                commandBufferPoolHitRate: commandBufferPoolHitRate,
                totalGPUMemoryAllocated: totalGPUMemoryAllocated,
                peakGPUMemoryUsage: peakGPUMemoryUsage,
                currentGPUMemoryUsage: currentGPUMemoryUsage,
                kernelDispatches: kernelDispatches,
                averageThreadsPerDispatch: avgThreadsPerDispatch,
                topKernels: topKernels,
                commandBufferSubmissions: commandBufferSubmissions
            )
        )
    }
    
    /// Get detailed metrics for a specific operation type
    public func getMetrics(for operationType: OperationType) -> ProfilerOperationMetrics? {
        let operations = operationHistory.filter { $0.type == operationType }
        guard !operations.isEmpty else { return nil }
        
        let durations = operations.map { $0.duration }
        let dataSizes = operations.map { $0.dataSize }
        
        return ProfilerOperationMetrics(
            count: operations.count,
            totalTime: durations.reduce(0, +),
            averageTime: average(durations),
            minTime: durations.min() ?? 0,
            maxTime: durations.max() ?? 0,
            percentiles: calculatePercentiles(durations),
            averageDataSize: average(dataSizes.map { Float($0) }),
            throughput: calculateThroughput(operations)
        )
    }
    
    // MARK: - Computed Properties
    
    public var lastGPUUtilization: Float {
        gpuUtilizationHistory.last ?? 0
    }
    
    public var lastMemoryUtilization: Float {
        // Estimate based on bandwidth usage
        let maxBandwidth: Float = 400.0 // GB/s typical for M-series
        let currentBandwidth = bandwidthHistory.last ?? 0
        return min(currentBandwidth / maxBandwidth, 1.0)
    }
    
    public var averageGPUUtilization: Float {
        average(gpuUtilizationHistory)
    }
    
    public var averageMemoryUtilization: Float {
        let maxBandwidth: Float = 400.0
        let avgBandwidth = average(bandwidthHistory)
        return min(avgBandwidth / maxBandwidth, 1.0)
    }
    
    public var peakMemoryBandwidth: Float {
        bandwidthHistory.max() ?? 0
    }
    
    public var averageLatency: TimeInterval {
        average(latencyHistory)
    }
    
    // MARK: - Reset
    
    /// Reset all profiling data
    public func reset() {
        operationHistory.removeAll()
        totalOperations = 0
        gpuUtilizationHistory.removeAll()
        memoryUtilizationHistory.removeAll()
        latencyHistory.removeAll()
        bandwidthHistory.removeAll()
        lastHardwareMetrics.removeAll()
        
        // Reset GPU metrics
        commandBufferPoolHits = 0
        commandBufferPoolMisses = 0
        totalGPUMemoryAllocated = 0
        currentGPUMemoryUsage = 0
        peakGPUMemoryUsage = 0
        kernelDispatches = 0
        totalThreadsDispatched = 0
        kernelExecutionCounts.removeAll()
        commandBufferSubmissions = 0
        
        logger.info("Profiler reset")
    }
    
    // MARK: - Private Methods
    
    private func maintainHistoryLimits() {
        let maxHistory = 1000
        
        if operationHistory.count > maxHistory {
            operationHistory.removeFirst(operationHistory.count - maxHistory)
        }
        
        if gpuUtilizationHistory.count > maxHistory {
            gpuUtilizationHistory.removeFirst(gpuUtilizationHistory.count - maxHistory)
        }
        
        if latencyHistory.count > maxHistory {
            latencyHistory.removeFirst(latencyHistory.count - maxHistory)
        }
        
        if bandwidthHistory.count > maxHistory {
            bandwidthHistory.removeFirst(bandwidthHistory.count - maxHistory)
        }
    }
    
    private func estimateGPUUtilization(duration: TimeInterval, dataSize: Int) -> Float {
        // Simple estimation based on duration and data size
        let expectedDuration = TimeInterval(dataSize) / 1_000_000.0 // 1M items/sec baseline
        return Float(min(expectedDuration / duration, 1.0))
    }
    
    private func estimateMemoryBandwidth(duration: TimeInterval, dataSize: Int) -> Float {
        // Estimate bandwidth in GB/s
        let bytesTransferred = dataSize * MemoryLayout<Float>.size * 2 // Read + write
        let gbTransferred = Float(bytesTransferred) / (1024 * 1024 * 1024)
        return gbTransferred / Float(duration)
    }
    
    private func average<T: BinaryFloatingPoint>(_ values: [T]) -> T {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / T(values.count)
    }
    
    private func calculatePercentiles<T: BinaryFloatingPoint & Comparable>(_ values: [T]) -> PercentileData<T> {
        guard !values.isEmpty else {
            return PercentileData(p50: 0, p90: 0, p95: 0, p99: 0)
        }
        
        let sorted = values.sorted()
        let count = sorted.count
        
        return PercentileData(
            p50: sorted[count * 50 / 100],
            p90: sorted[min(count * 90 / 100, count - 1)],
            p95: sorted[min(count * 95 / 100, count - 1)],
            p99: sorted[min(count * 99 / 100, count - 1)]
        )
    }
    
    private func calculateThroughput(_ operations: [OperationRecord]) -> Float {
        let totalTime = operations.map { $0.duration }.reduce(0, +)
        let totalData = operations.map { $0.dataSize }.reduce(0, +)
        guard totalTime > 0 else { return 0 }
        return Float(totalData) / Float(totalTime)
    }
}

// MARK: - Supporting Types

/// Record of a single operation
private struct OperationRecord {
    let type: MetalProfiler.OperationType
    let timestamp: Date
    let duration: TimeInterval
    let dataSize: Int
    let gpuUtilization: Float
    let memoryBandwidth: Float
}

/// GPU-specific metrics
public struct GPUSpecificMetrics: Sendable {
    public let commandBufferPoolHitRate: Float
    public let totalGPUMemoryAllocated: Int
    public let peakGPUMemoryUsage: Int
    public let currentGPUMemoryUsage: Int
    public let kernelDispatches: Int
    public let averageThreadsPerDispatch: Int
    public let topKernels: [(name: String, count: Int)]
    public let commandBufferSubmissions: Int
}

/// Profiler performance statistics
public struct ProfilerStatistics: Sendable {
    public let totalOperations: UInt64
    public let operationCounts: [MetalProfiler.OperationType: Int]
    public let averageGPUUtilization: Float
    public let peakGPUUtilization: Float
    public let averageLatency: TimeInterval
    public let latencyPercentiles: PercentileData<TimeInterval>
    public let averageMemoryBandwidth: Float
    public let peakMemoryBandwidth: Float
    public let hardwareMetrics: [String: Float]
    public let gpuMetrics: GPUSpecificMetrics
}

/// Metrics for a specific operation type
public struct ProfilerOperationMetrics: Sendable {
    public let count: Int
    public let totalTime: TimeInterval
    public let averageTime: TimeInterval
    public let minTime: TimeInterval
    public let maxTime: TimeInterval
    public let percentiles: PercentileData<TimeInterval>
    public let averageDataSize: Float
    public let throughput: Float // items/second
}

/// Percentile data
public struct PercentileData<T: Sendable>: Sendable {
    public let p50: T
    public let p90: T
    public let p95: T
    public let p99: T
}

/// Performance metrics for external reporting
public struct MetalPerformanceMetrics: Sendable {
    public let gpuTime: TimeInterval
    public let cpuOverhead: TimeInterval
    public let memoryBandwidth: Float
    public let operationsPerSecond: Float
    public let gpuUtilization: Float
    public let memoryUtilization: Float
    public let usedGPU: Bool
    public let hardwareMetrics: [String: Float]
}
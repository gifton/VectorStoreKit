// VectorStoreKit: Metal Batch Optimizer
//
// Adaptive batch sizing and optimization for GPU operations

import Foundation
@preconcurrency import Metal
import os.log

/// Optimizes batch sizes based on GPU capabilities and workload
public actor MetalBatchOptimizer {
    
    // MARK: - Properties
    
    private let device: MTLDevice
    private let profiler: MetalProfiler?
    
    // Optimization parameters
    private var optimalBatchSizes: [BatchKey: Int] = [:]
    private var performanceHistory: [BatchKey: [PerformanceSample]] = [:]
    
    // Hardware limits
    private let maxThreadgroupMemorySize: Int
    private let maxThreadsPerThreadgroup: Int
    private let recommendedMaxWorkingSetSize: Int
    
    // Dynamic state
    private var currentGPUUtilization: Float = 0.0
    private var memoryPressure: Float = 0.0
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalBatchOptimizer")
    
    // MARK: - Initialization
    
    public init(device: MTLDevice, profiler: MetalProfiler? = nil) async {
        self.device = device
        self.profiler = profiler
        
        // Query device capabilities
        self.maxThreadgroupMemorySize = device.maxThreadgroupMemoryLength
        self.maxThreadsPerThreadgroup = device.maxThreadsPerThreadgroup.width
        self.recommendedMaxWorkingSetSize = Int(device.recommendedMaxWorkingSetSize)
        
        logger.info("Batch optimizer initialized - Max threadgroup memory: \(self.maxThreadgroupMemorySize), Max threads: \(self.maxThreadsPerThreadgroup)")
    }
    
    // MARK: - Batch Size Optimization
    
    /// Calculate optimal batch size for a given operation
    public func getOptimalBatchSize(
        operation: MetalBatchOperation,
        dataType: MetalDataType,
        vectorDimension: Int,
        totalElements: Int
    ) async -> BatchConfiguration {
        
        let key = BatchKey(
            operation: operation,
            dataType: dataType,
            vectorDimension: vectorDimension
        )
        
        // Check if we have a cached optimal size
        if let cachedSize = optimalBatchSizes[key] {
            logger.debug("Using cached batch size: \(cachedSize) for \(operation)")
            return await createConfiguration(
                batchSize: cachedSize,
                operation: operation,
                vectorDimension: vectorDimension,
                totalElements: totalElements
            )
        }
        
        // Calculate initial batch size based on hardware limits
        let initialSize = await calculateInitialBatchSize(
            operation: operation,
            dataType: dataType,
            vectorDimension: vectorDimension
        )
        
        // Adjust based on current GPU state
        let adjustedSize = await adjustForGPUState(
            baseSize: initialSize,
            operation: operation
        )
        
        // Create configuration
        return await createConfiguration(
            batchSize: adjustedSize,
            operation: operation,
            vectorDimension: vectorDimension,
            totalElements: totalElements
        )
    }
    
    /// Update performance metrics after batch execution
    public func recordBatchPerformance(
        operation: MetalBatchOperation,
        dataType: MetalDataType,
        vectorDimension: Int,
        batchSize: Int,
        executionTime: TimeInterval,
        throughput: Double
    ) async {
        
        let key = BatchKey(
            operation: operation,
            dataType: dataType,
            vectorDimension: vectorDimension
        )
        
        let sample = PerformanceSample(
            batchSize: batchSize,
            executionTime: executionTime,
            throughput: throughput,
            timestamp: Date()
        )
        
        // Add to history
        if performanceHistory[key] == nil {
            performanceHistory[key] = []
        }
        performanceHistory[key]?.append(sample)
        
        // Keep only recent samples
        performanceHistory[key] = performanceHistory[key]?.suffix(100)
        
        // Update optimal batch size if we have enough data
        if let history = performanceHistory[key], history.count >= 10 {
            await updateOptimalBatchSize(key: key, history: history)
        }
    }
    
    // MARK: - Dynamic GPU State
    
    /// Update current GPU utilization metrics
    public func updateGPUState(utilization: Float, memoryPressure: Float) {
        self.currentGPUUtilization = utilization
        self.memoryPressure = memoryPressure
        
        logger.debug("GPU state updated - Utilization: \(utilization), Memory pressure: \(memoryPressure)")
    }
    
    // MARK: - Private Methods
    
    private func calculateInitialBatchSize(
        operation: MetalBatchOperation,
        dataType: MetalDataType,
        vectorDimension: Int
    ) async -> Int {
        
        let elementSize = dataType.byteSize * vectorDimension
        
        switch operation {
        case .distanceComputation:
            // For distance computation, we need space for query + candidates + results
            let maxCandidatesInMemory = recommendedMaxWorkingSetSize / (elementSize + MemoryLayout<Float>.size)
            let threadgroupLimit = maxThreadgroupMemorySize / elementSize
            
            // Use the smaller limit
            let baseSize = min(maxCandidatesInMemory, threadgroupLimit * 4)
            
            // Round to multiple of warp size for efficiency
            return (baseSize / 32) * 32
            
        case .matrixMultiplication:
            // For matrix ops, consider tiling requirements
            let tileSize = 16 // Common tile size
            let tilesPerDimension = Int(sqrt(Double(recommendedMaxWorkingSetSize / elementSize)))
            return tilesPerDimension * tileSize
            
        case .quantization:
            // Quantization typically has lower memory requirements
            return recommendedMaxWorkingSetSize / (elementSize / 4)
            
        case .indexing:
            // Indexing operations are usually memory-bound
            return recommendedMaxWorkingSetSize / (elementSize * 2)
            
        case .normalization:
            // Normalization is typically compute-bound
            return maxThreadsPerThreadgroup * 16
        }
    }
    
    private func adjustForGPUState(
        baseSize: Int,
        operation: MetalBatchOperation
    ) async -> Int {
        
        // Reduce batch size under high GPU utilization
        var scaleFactor: Float = 1.0
        
        if currentGPUUtilization > 0.9 {
            scaleFactor *= 0.7
        } else if currentGPUUtilization > 0.7 {
            scaleFactor *= 0.85
        }
        
        // Reduce batch size under memory pressure
        if memoryPressure > 0.8 {
            scaleFactor *= 0.6
        } else if memoryPressure > 0.6 {
            scaleFactor *= 0.8
        }
        
        // Apply operation-specific adjustments
        switch operation {
        case .distanceComputation, .matrixMultiplication:
            // These are memory-intensive, be more conservative
            scaleFactor *= 0.9
        case .normalization:
            // Compute-bound, can be more aggressive
            scaleFactor *= 1.1
        default:
            break
        }
        
        let adjustedSize = Int(Float(baseSize) * scaleFactor)
        
        // Ensure minimum viable batch size
        let minSize = operation.minimumBatchSize
        return max(adjustedSize, minSize)
    }
    
    private func createConfiguration(
        batchSize: Int,
        operation: MetalBatchOperation,
        vectorDimension: Int,
        totalElements: Int
    ) async -> BatchConfiguration {
        
        // Calculate number of batches
        let numBatches = (totalElements + batchSize - 1) / batchSize
        
        // Determine thread configuration
        let threadsPerThreadgroup: Int
        let threadgroupsPerBatch: Int
        
        switch operation {
        case .distanceComputation:
            threadsPerThreadgroup = min(256, batchSize)
            threadgroupsPerBatch = (batchSize + threadsPerThreadgroup - 1) / threadsPerThreadgroup
            
        case .matrixMultiplication:
            let tileSize = 16
            threadsPerThreadgroup = tileSize * tileSize
            let tilesPerDimension = Int(sqrt(Double(batchSize)))
            threadgroupsPerBatch = tilesPerDimension * tilesPerDimension
            
        default:
            threadsPerThreadgroup = min(256, maxThreadsPerThreadgroup)
            threadgroupsPerBatch = (batchSize + threadsPerThreadgroup - 1) / threadsPerThreadgroup
        }
        
        // Estimate memory usage
        let memoryPerBatch = operation.estimateMemoryUsage(
            batchSize: batchSize,
            vectorDimension: vectorDimension
        )
        
        return BatchConfiguration(
            batchSize: batchSize,
            numBatches: numBatches,
            threadsPerThreadgroup: threadsPerThreadgroup,
            threadgroupsPerBatch: threadgroupsPerBatch,
            estimatedMemoryUsage: memoryPerBatch,
            useDoubleBUffering: memoryPerBatch * 2 < recommendedMaxWorkingSetSize,
            prefetchNextBatch: currentGPUUtilization < 0.5
        )
    }
    
    private func updateOptimalBatchSize(key: BatchKey, history: [PerformanceSample]) async {
        // Find the batch size with best throughput
        var bestSize = 0
        var bestThroughput = 0.0
        
        // Group samples by batch size
        let sizeGroups = Dictionary(grouping: history) { $0.batchSize }
        
        for (size, samples) in sizeGroups {
            let avgThroughput = samples.map { $0.throughput }.reduce(0, +) / Double(samples.count)
            if avgThroughput > bestThroughput {
                bestThroughput = avgThroughput
                bestSize = size
            }
        }
        
        if bestSize > 0 {
            optimalBatchSizes[key] = bestSize
            logger.info("Updated optimal batch size for \(key.operation): \(bestSize) (throughput: \(bestThroughput))")
        }
    }
}

// MARK: - Supporting Types

/// Metal batch operation types
public enum MetalBatchOperation: Hashable, CustomStringConvertible {
    case distanceComputation
    case matrixMultiplication
    case quantization
    case indexing
    case normalization
    
    var minimumBatchSize: Int {
        switch self {
        case .distanceComputation: return 32
        case .matrixMultiplication: return 16
        case .quantization: return 64
        case .indexing: return 128
        case .normalization: return 256
        }
    }
    
    func estimateMemoryUsage(batchSize: Int, vectorDimension: Int) -> Int {
        let elementSize = MemoryLayout<Float>.size * vectorDimension
        
        switch self {
        case .distanceComputation:
            // Query + candidates + results
            return elementSize + (batchSize * elementSize) + (batchSize * MemoryLayout<Float>.size)
        case .matrixMultiplication:
            // Two matrices + result
            return 3 * batchSize * elementSize
        case .quantization:
            // Input + quantized output
            return batchSize * elementSize + (batchSize * elementSize / 4)
        case .indexing:
            // Vectors + metadata
            return batchSize * (elementSize + 64)
        case .normalization:
            // Input + output (in-place possible)
            return batchSize * elementSize
        }
    }
    
    public var description: String {
        switch self {
        case .distanceComputation: return "DistanceComputation"
        case .matrixMultiplication: return "MatrixMultiplication"
        case .quantization: return "Quantization"
        case .indexing: return "Indexing"
        case .normalization: return "Normalization"
        }
    }
}

/// Metal data type information
public enum MetalDataType: Hashable {
    case float32
    case float16
    case int8
    case uint8
    
    var byteSize: Int {
        switch self {
        case .float32: return 4
        case .float16: return 2
        case .int8, .uint8: return 1
        }
    }
}

/// Batch configuration
public struct BatchConfiguration: Sendable {
    public let batchSize: Int
    public let numBatches: Int
    public let threadsPerThreadgroup: Int
    public let threadgroupsPerBatch: Int
    public let estimatedMemoryUsage: Int
    public let useDoubleBUffering: Bool
    public let prefetchNextBatch: Bool
}

/// Key for batch optimization cache
private struct BatchKey: Hashable {
    let operation: MetalBatchOperation
    let dataType: MetalDataType
    let vectorDimension: Int
}

/// Performance sample for optimization
private struct PerformanceSample {
    let batchSize: Int
    let executionTime: TimeInterval
    let throughput: Double
    let timestamp: Date
}

// MARK: - Extensions

extension MetalBatchOptimizer {
    
    /// Create optimized batch iterator
    public func createBatchIterator<T>(
        data: [T],
        operation: MetalBatchOperation,
        vectorDimension: Int
    ) async -> BatchIterator<T> {
        
        let config = await getOptimalBatchSize(
            operation: operation,
            dataType: .float32,
            vectorDimension: vectorDimension,
            totalElements: data.count
        )
        
        return BatchIterator(
            data: data,
            configuration: config
        )
    }
}

/// Iterator for processing data in optimized batches
public struct BatchIterator<T>: AsyncSequence {
    public typealias Element = (batchIndex: Int, batch: ArraySlice<T>)
    
    private let data: [T]
    private let configuration: BatchConfiguration
    
    init(data: [T], configuration: BatchConfiguration) {
        self.data = data
        self.configuration = configuration
    }
    
    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(
            data: data,
            batchSize: configuration.batchSize
        )
    }
    
    public struct AsyncIterator: AsyncIteratorProtocol {
        private let data: [T]
        private let batchSize: Int
        private var currentIndex = 0
        private var batchIndex = 0
        
        init(data: [T], batchSize: Int) {
            self.data = data
            self.batchSize = batchSize
        }
        
        public mutating func next() async -> Element? {
            guard currentIndex < data.count else { return nil }
            
            let endIndex = Swift.min(currentIndex + batchSize, data.count)
            let batch = data[currentIndex..<endIndex]
            
            let result = (batchIndex, batch)
            currentIndex = endIndex
            batchIndex += 1
            
            return result
        }
    }
}
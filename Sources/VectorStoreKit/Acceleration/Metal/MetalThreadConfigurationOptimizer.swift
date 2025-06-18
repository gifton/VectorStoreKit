// VectorStoreKit: Metal Thread Configuration Optimizer
//
// Optimizes thread group sizes and grid configurations for Metal compute operations
//

import Foundation
@preconcurrency import Metal
import os.log

/// Optimizes thread configurations for Metal compute operations based on device capabilities
public actor MetalThreadConfigurationOptimizer {
    
    // MARK: - Properties
    
    private let device: MetalDevice
    private let profiler: MetalProfiler?
    private var configurationCache: [ConfigurationKey: OptimalConfiguration] = [:]
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "ThreadOptimizer")
    
    // Device-specific constants
    private let wavefrontSize: Int
    private let maxThreadsPerThreadgroup: Int
    private let maxThreadgroupMemoryLength: Int
    
    // MARK: - Initialization
    
    public init(device: MetalDevice, profiler: MetalProfiler? = nil) async {
        self.device = device
        self.profiler = profiler
        
        // Extract device capabilities
        let capabilities = await device.capabilities
        self.maxThreadsPerThreadgroup = capabilities.maxThreadsPerThreadgroup
        self.maxThreadgroupMemoryLength = capabilities.maxThreadgroupMemoryLength
        
        // Determine wavefront size based on device
        if capabilities.deviceName.contains("Apple") {
            self.wavefrontSize = 32 // Apple Silicon uses 32-thread warps
        } else {
            self.wavefrontSize = 64 // AMD/Intel typically use 64
        }
        
        logger.info("Initialized thread optimizer - wavefront: \(self.wavefrontSize), max threads: \(self.maxThreadsPerThreadgroup)")
    }
    
    // MARK: - Public Methods
    
    /// Get optimal thread configuration for a specific operation
    public func getOptimalConfiguration(
        for operation: OperationType,
        workSize: Int,
        vectorDimension: Int? = nil,
        sharedMemoryPerThread: Int = 0
    ) async -> OptimalConfiguration {
        
        let key = ConfigurationKey(
            operation: operation,
            workSize: workSize,
            vectorDimension: vectorDimension ?? 0,
            sharedMemoryPerThread: sharedMemoryPerThread
        )
        
        // Check cache first
        if let cached = configurationCache[key] {
            logger.debug("Using cached configuration for \(operation.rawValue)")
            return cached
        }
        
        // Calculate optimal configuration
        let config = await calculateOptimalConfiguration(
            operation: operation,
            workSize: workSize,
            vectorDimension: vectorDimension,
            sharedMemoryPerThread: sharedMemoryPerThread
        )
        
        // Cache the result
        configurationCache[key] = config
        
        // Record profiling event
        await profiler?.recordEvent(.pipelineCreation(name: "ThreadConfig:\(operation)"))
        
        return config
    }
    
    /// Get optimal 2D thread configuration for matrix operations
    public func getOptimal2DConfiguration(
        for operation: OperationType,
        rows: Int,
        columns: Int,
        tileSize: Int = 16
    ) async -> Optimal2DConfiguration {
        
        // Ensure tile size is aligned to wavefront
        let alignedTileSize = ((tileSize + wavefrontSize - 1) / wavefrontSize) * wavefrontSize
        
        // Calculate threadgroup size
        let threadsPerThreadgroup = MTLSize(
            width: min(alignedTileSize, maxThreadsPerThreadgroup),
            height: min(alignedTileSize, maxThreadsPerThreadgroup / alignedTileSize),
            depth: 1
        )
        
        // Calculate grid size
        let threadgroupsPerGrid = MTLSize(
            width: (columns + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: (rows + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            depth: 1
        )
        
        // Calculate memory requirements
        let sharedMemoryRequired = alignedTileSize * alignedTileSize * MemoryLayout<Float>.size * 2
        let occupancy = estimateOccupancy(
            threadsPerThreadgroup: threadsPerThreadgroup.width * threadsPerThreadgroup.height,
            sharedMemory: sharedMemoryRequired
        )
        
        logger.debug("""
            2D config for \(operation.rawValue): \
            tile=\(alignedTileSize)x\(alignedTileSize), \
            threads=\(threadsPerThreadgroup.width)x\(threadsPerThreadgroup.height), \
            occupancy=\(occupancy)
            """)
        
        return Optimal2DConfiguration(
            threadsPerThreadgroup: threadsPerThreadgroup,
            threadgroupsPerGrid: threadgroupsPerGrid,
            tileSize: alignedTileSize,
            sharedMemoryRequired: sharedMemoryRequired,
            estimatedOccupancy: occupancy
        )
    }
    
    /// Benchmark different configurations to find the best one
    public func benchmarkConfigurations(
        for pipeline: MTLComputePipelineState,
        workSize: Int,
        testIterations: Int = 10
    ) async throws -> ThreadConfigurationBenchmarkResults {
        
        let configurations = generateTestConfigurations(for: workSize)
        var results: [ConfigurationBenchmark] = []
        
        for config in configurations {
            let timing = try await measureConfiguration(
                pipeline: pipeline,
                configuration: config,
                iterations: testIterations
            )
            
            results.append(ConfigurationBenchmark(
                configuration: config,
                averageTime: timing.average,
                minTime: timing.min,
                maxTime: timing.max
            ))
        }
        
        // Find best configuration
        let best = results.min(by: { $0.averageTime < $1.averageTime })!
        
        logger.info("""
            Benchmark complete - best config: \
            \(best.configuration.threadsPerThreadgroup.width) threads, \
            \(best.averageTime * 1000)ms avg
            """)
        
        return ThreadConfigurationBenchmarkResults(
            configurations: results,
            bestConfiguration: best.configuration,
            speedupFactor: results.last!.averageTime / best.averageTime
        )
    }
    
    // MARK: - Private Methods
    
    private func calculateOptimalConfiguration(
        operation: OperationType,
        workSize: Int,
        vectorDimension: Int?,
        sharedMemoryPerThread: Int
    ) async -> OptimalConfiguration {
        
        // Get operation-specific parameters
        let params = getOperationParameters(for: operation, vectorDimension: vectorDimension)
        
        // Calculate optimal threadgroup size
        var threadsPerThreadgroup = params.preferredThreadgroupSize
        
        // Adjust for shared memory constraints
        if sharedMemoryPerThread > 0 {
            let maxThreadsByMemory = maxThreadgroupMemoryLength / sharedMemoryPerThread
            threadsPerThreadgroup = min(threadsPerThreadgroup, maxThreadsByMemory)
        }
        
        // Ensure alignment to wavefront
        threadsPerThreadgroup = ((threadsPerThreadgroup + wavefrontSize - 1) / wavefrontSize) * wavefrontSize
        
        // Ensure we don't exceed device limits
        threadsPerThreadgroup = min(threadsPerThreadgroup, maxThreadsPerThreadgroup)
        threadsPerThreadgroup = min(threadsPerThreadgroup, workSize)
        
        // Calculate grid size
        let threadgroupsPerGrid = (workSize + threadsPerThreadgroup - 1) / threadsPerThreadgroup
        
        // Estimate occupancy
        let occupancy = estimateOccupancy(
            threadsPerThreadgroup: threadsPerThreadgroup,
            sharedMemory: sharedMemoryPerThread * threadsPerThreadgroup,
            registersPerThread: params.estimatedRegistersPerThread
        )
        
        logger.debug("""
            Config for \(operation.rawValue): \
            threads=\(threadsPerThreadgroup), \
            groups=\(threadgroupsPerGrid), \
            occupancy=\(occupancy)
            """)
        
        return OptimalConfiguration(
            threadsPerThreadgroup: MTLSize(width: threadsPerThreadgroup, height: 1, depth: 1),
            threadgroupsPerGrid: MTLSize(width: threadgroupsPerGrid, height: 1, depth: 1),
            sharedMemoryRequired: sharedMemoryPerThread * threadsPerThreadgroup,
            estimatedOccupancy: occupancy,
            useNonUniformThreadgroups: workSize % threadsPerThreadgroup != 0
        )
    }
    
    private func getOperationParameters(
        for operation: OperationType,
        vectorDimension: Int?
    ) -> OperationParameters {
        
        switch operation {
        case .distanceComputation:
            // Distance computation is memory bandwidth limited
            if let dim = vectorDimension, dim == 512 {
                // Special case for 512-dim vectors
                return OperationParameters(
                    preferredThreadgroupSize: 64,
                    estimatedRegistersPerThread: 32,
                    memoryIntensive: true
                )
            } else {
                return OperationParameters(
                    preferredThreadgroupSize: 128,
                    estimatedRegistersPerThread: 16,
                    memoryIntensive: true
                )
            }
            
        case .matrixMultiplication:
            // Matrix multiplication benefits from larger threadgroups
            return OperationParameters(
                preferredThreadgroupSize: 256,
                estimatedRegistersPerThread: 64,
                memoryIntensive: false
            )
            
        case .quantization:
            // Quantization is compute intensive
            return OperationParameters(
                preferredThreadgroupSize: 256,
                estimatedRegistersPerThread: 32,
                memoryIntensive: false
            )
            
        case .vectorNormalization:
            // Normalization has two passes
            return OperationParameters(
                preferredThreadgroupSize: 128,
                estimatedRegistersPerThread: 8,
                memoryIntensive: true
            )
            
        case .clustering:
            // Clustering needs balanced approach
            return OperationParameters(
                preferredThreadgroupSize: 128,
                estimatedRegistersPerThread: 24,
                memoryIntensive: true
            )
            
        default:
            // Default configuration
            return OperationParameters(
                preferredThreadgroupSize: 128,
                estimatedRegistersPerThread: 16,
                memoryIntensive: true
            )
        }
    }
    
    private func estimateOccupancy(
        threadsPerThreadgroup: Int,
        sharedMemory: Int,
        registersPerThread: Int = 16
    ) -> Float {
        
        // Simplified occupancy model based on Apple Silicon characteristics
        // In reality, this would need profiling data
        
        // Base occupancy from thread count
        let threadOccupancy = Float(threadsPerThreadgroup) / Float(maxThreadsPerThreadgroup)
        
        // Memory pressure factor
        let memoryFactor: Float
        if sharedMemory > 0 {
            let memoryUsage = Float(sharedMemory) / Float(maxThreadgroupMemoryLength)
            memoryFactor = 1.0 - (memoryUsage * 0.5) // 50% penalty at max memory
        } else {
            memoryFactor = 1.0
        }
        
        // Register pressure factor (simplified)
        let registerFactor = max(0.5, 1.0 - Float(registersPerThread) / 128.0)
        
        // Combined occupancy estimate
        return threadOccupancy * memoryFactor * registerFactor
    }
    
    private func generateTestConfigurations(for workSize: Int) -> [OptimalConfiguration] {
        var configurations: [OptimalConfiguration] = []
        
        // Test powers of 2 and multiples of wavefront size
        let testSizes = [32, 64, 128, 256, 512, 1024].filter { $0 <= maxThreadsPerThreadgroup }
        
        for size in testSizes {
            let threadgroupsPerGrid = (workSize + size - 1) / size
            
            configurations.append(OptimalConfiguration(
                threadsPerThreadgroup: MTLSize(width: size, height: 1, depth: 1),
                threadgroupsPerGrid: MTLSize(width: threadgroupsPerGrid, height: 1, depth: 1),
                sharedMemoryRequired: 0,
                estimatedOccupancy: estimateOccupancy(threadsPerThreadgroup: size, sharedMemory: 0),
                useNonUniformThreadgroups: workSize % size != 0
            ))
        }
        
        return configurations
    }
    
    private func measureConfiguration(
        pipeline: MTLComputePipelineState,
        configuration: OptimalConfiguration,
        iterations: Int
    ) async throws -> (average: TimeInterval, min: TimeInterval, max: TimeInterval) {
        
        var timings: [TimeInterval] = []
        
        for _ in 0..<iterations {
            let startTime = CFAbsoluteTimeGetCurrent()
            
            // Create dummy command buffer for timing
            guard let commandBuffer = await device.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalComputeError.computeEncoderCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.dispatchThreadgroups(
                configuration.threadgroupsPerGrid,
                threadsPerThreadgroup: configuration.threadsPerThreadgroup
            )
            encoder.endEncoding()
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            timings.append(duration)
        }
        
        let average = timings.reduce(0, +) / Double(timings.count)
        let minTime = timings.min() ?? 0
        let maxTime = timings.max() ?? 0
        
        return (average, minTime, maxTime)
    }
}

// MARK: - Supporting Types

/// Operation types for thread configuration
public enum OperationType: String, CaseIterable {
    case distanceComputation = "distance"
    case matrixMultiplication = "matrix"
    case quantization = "quantization"
    case vectorNormalization = "normalization"
    case clustering = "clustering"
    case reduction = "reduction"
    case scan = "scan"
}

/// Optimal thread configuration for a compute operation
public struct OptimalConfiguration: Sendable {
    public let threadsPerThreadgroup: MTLSize
    public let threadgroupsPerGrid: MTLSize
    public let sharedMemoryRequired: Int
    public let estimatedOccupancy: Float
    public let useNonUniformThreadgroups: Bool
}

/// Optimal configuration for 2D operations
public struct Optimal2DConfiguration: Sendable {
    public let threadsPerThreadgroup: MTLSize
    public let threadgroupsPerGrid: MTLSize
    public let tileSize: Int
    public let sharedMemoryRequired: Int
    public let estimatedOccupancy: Float
}

/// Configuration cache key
private struct ConfigurationKey: Hashable {
    let operation: OperationType
    let workSize: Int
    let vectorDimension: Int
    let sharedMemoryPerThread: Int
}

/// Operation-specific parameters
private struct OperationParameters {
    let preferredThreadgroupSize: Int
    let estimatedRegistersPerThread: Int
    let memoryIntensive: Bool
}

/// Benchmark results for thread configurations
public struct ThreadConfigurationBenchmarkResults: Sendable {
    public let configurations: [ConfigurationBenchmark]
    public let bestConfiguration: OptimalConfiguration
    public let speedupFactor: Double
}

/// Individual configuration benchmark
public struct ConfigurationBenchmark: Sendable {
    public let configuration: OptimalConfiguration
    public let averageTime: TimeInterval
    public let minTime: TimeInterval
    public let maxTime: TimeInterval
}
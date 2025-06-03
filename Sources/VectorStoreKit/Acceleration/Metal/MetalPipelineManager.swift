// VectorStoreKit: Metal Pipeline Manager
//
// Manages compute pipeline state creation and caching

import Foundation
@preconcurrency import Metal
import os.log

/// Manages Metal compute pipeline states with caching
public actor MetalPipelineManager {
    
    // MARK: - Properties
    
    private let device: MetalDevice
    private var pipelineCache: [String: MTLComputePipelineState] = [:]
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalPipelineManager")
    
    // MARK: - Standard Pipeline Names
    
    public enum StandardPipeline: String, CaseIterable {
        case euclideanDistance = "euclideanDistance"
        case cosineDistance = "cosineDistance"
        case manhattanDistance = "manhattanDistance"
        case dotProduct = "dotProduct"
        case matrixMultiply = "matrixMultiply"
        case vectorQuantization = "vectorQuantization"
        case parallelSearch = "parallelSearch"
        case vectorNormalization = "vectorNormalization"
        case elementwiseOperations = "elementwiseOperations"
    }
    
    // MARK: - Initialization
    
    public init(device: MetalDevice) {
        self.device = device
    }
    
    // MARK: - Pipeline Management
    
    /// Get or create a pipeline state for a standard function
    public func getPipeline(for standardPipeline: StandardPipeline) async throws -> MTLComputePipelineState {
        return try await getPipeline(functionName: standardPipeline.rawValue)
    }
    
    /// Get or create a pipeline state for a named function
    public func getPipeline(functionName: String) async throws -> MTLComputePipelineState {
        // Check cache first
        if let cached = pipelineCache[functionName] {
            logger.debug("Using cached pipeline: \(functionName)")
            return cached
        }
        
        // Create new pipeline
        logger.info("Creating pipeline: \(functionName)")
        
        guard let function = await device.makeFunction(name: functionName) else {
            throw MetalPipelineError.functionNotFound(functionName)
        }
        
        let pipeline = try await device.makeComputePipelineState(function: function)
        
        // Cache for future use
        pipelineCache[functionName] = pipeline
        
        logger.info("Created pipeline: \(functionName), threads: \(pipeline.threadExecutionWidth)")
        
        return pipeline
    }
    
    /// Precompile all standard pipelines
    public func precompileStandardPipelines() async {
        logger.info("Precompiling standard pipelines")
        
        for pipeline in StandardPipeline.allCases {
            do {
                _ = try await getPipeline(for: pipeline)
            } catch {
                logger.warning("Failed to compile \(pipeline.rawValue): \(error)")
            }
        }
        
        logger.info("Precompilation complete: \(self.pipelineCache.count) pipelines cached")
    }
    
    /// Get optimal thread configuration for a pipeline and workload
    public func getThreadConfiguration(
        for pipeline: MTLComputePipelineState,
        workSize: Int
    ) -> (threadgroupsPerGrid: MTLSize, threadsPerThreadgroup: MTLSize) {
        let maxThreadsPerThreadgroup = pipeline.maxTotalThreadsPerThreadgroup
        let threadExecutionWidth = pipeline.threadExecutionWidth
        
        // Optimize threadgroup size based on pipeline characteristics
        let optimalThreadgroupSize = min(
            maxThreadsPerThreadgroup,
            ((workSize + threadExecutionWidth - 1) / threadExecutionWidth) * threadExecutionWidth
        )
        
        let threadsPerThreadgroup = MTLSize(
            width: min(optimalThreadgroupSize, workSize),
            height: 1,
            depth: 1
        )
        
        let threadgroupsPerGrid = MTLSize(
            width: (workSize + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )
        
        return (threadgroupsPerGrid, threadsPerThreadgroup)
    }
    
    /// Clear pipeline cache
    public func clearCache() {
        pipelineCache.removeAll()
        logger.info("Cleared pipeline cache")
    }
    
    /// Get cache statistics
    public var cacheStatistics: PipelineCacheStatistics {
        PipelineCacheStatistics(
            cachedPipelines: pipelineCache.count,
            pipelineNames: Array(pipelineCache.keys).sorted()
        )
    }
}

// MARK: - Supporting Types

/// Pipeline manager errors
public enum MetalPipelineError: Error, LocalizedError {
    case functionNotFound(String)
    case compilationFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .functionNotFound(let name):
            return "Metal function not found: \(name)"
        case .compilationFailed(let reason):
            return "Pipeline compilation failed: \(reason)"
        }
    }
}

/// Pipeline cache statistics
public struct PipelineCacheStatistics: Sendable {
    public let cachedPipelines: Int
    public let pipelineNames: [String]
}

// MARK: - Pipeline Descriptors

/// Descriptor for custom pipeline creation
public struct PipelineDescriptor: Sendable {
    public let functionName: String
    public let constantValues: MTLFunctionConstantValues?
    public let label: String?
    
    public init(
        functionName: String,
        constantValues: MTLFunctionConstantValues? = nil,
        label: String? = nil
    ) {
        self.functionName = functionName
        self.constantValues = constantValues
        self.label = label
    }
}

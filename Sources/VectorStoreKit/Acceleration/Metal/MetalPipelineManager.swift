// VectorStoreKit: Metal Pipeline Manager
//
// Manages compute pipeline state creation and caching
// Now uses standardized MetalShaderCompiler for all compilation

import Foundation
@preconcurrency import Metal
import os.log

/// Manages Metal compute pipeline states with advanced compilation and caching
public actor MetalPipelineManager {
    
    // MARK: - Properties
    
    private let device: MetalDevice
    private let shaderCompiler: MetalShaderCompiler
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
        
        // Optimized variants
        case euclideanDistance512 = "euclideanDistance512_simd"
        case cosineDistance512 = "cosineDistance512_simd"
        case euclideanDistance512Normalized = "cosineDistance512_normalized"
        case euclideanDistanceWarp = "euclideanDistanceWarpOptimized"
    }
    
    public enum MLPipeline: String, CaseIterable {
        case vaeReparameterization = "vae_reparameterization"
        case adamOptimizer = "adam_optimizer"
        case vaeLoss = "vae_loss"
        case mseLoss = "mse_loss"
        case klDivergence = "kl_divergence"
    }
    
    // MARK: - Initialization
    
    public init(device: MetalDevice, options: ShaderCompilationOptions = .default) throws {
        self.device = device
        let mtlDevice = device.device
        self.shaderCompiler = try MetalShaderCompiler(
            device: mtlDevice,
            options: options
        )
    }
    
    // MARK: - Pipeline Management
    
    /// Get or create a pipeline state for a standard function
    public func getPipeline(for standardPipeline: StandardPipeline) async throws -> MTLComputePipelineState {
        return try await shaderCompiler.compilePipeline(functionName: standardPipeline.rawValue)
    }
    
    /// Get or create a pipeline state for a named function
    public func getPipeline(functionName: String) async throws -> MTLComputePipelineState {
        return try await shaderCompiler.compilePipeline(functionName: functionName)
    }
    
    /// Get or create a pipeline state for an ML function
    public func getMLPipeline(_ mlPipeline: MLPipeline) async throws -> MTLComputePipelineState {
        return try await shaderCompiler.compilePipeline(functionName: mlPipeline.rawValue)
    }
    
    /// Get optimized pipeline for specific parameters
    public func getOptimizedPipeline(
        baseFunctionName: String,
        dimensions: Int,
        dataType: ShaderDataType = .float32,
        precision: ShaderComputePrecision = .full
    ) async throws -> MTLComputePipelineState {
        return try await shaderCompiler.getOptimalPipeline(
            baseFunctionName: baseFunctionName,
            dimensions: dimensions,
            dataType: dataType,
            precision: precision
        )
    }
    
    /// Compile pipeline with function constants
    public func getPipelineWithConstants(
        functionName: String,
        constants: ShaderFunctionConstants
    ) async throws -> MTLComputePipelineState {
        return try await shaderCompiler.compilePipeline(
            functionName: functionName,
            constants: constants
        )
    }
    
    /// Precompile all standard pipelines
    public func precompileStandardPipelines() async {
        logger.info("Precompiling standard pipelines")
        
        // Compile basic variants
        for pipeline in StandardPipeline.allCases {
            do {
                _ = try await getPipeline(for: pipeline)
            } catch {
                logger.warning("Failed to compile \(pipeline.rawValue): \(error)")
            }
        }
    }
    
    /// Load ML optimization shaders
    public func loadMLOptimizationShaders() async throws {
        logger.info("Loading ML optimization shaders")
        
        // Pre-compile ML pipelines
        for pipeline in MLPipeline.allCases {
            do {
                _ = try await getMLPipeline(pipeline)
            } catch {
                logger.warning("Failed to compile ML pipeline \(pipeline.rawValue): \(error)")
            }
        }
        
        // Compile common dimension variants
        await shaderCompiler.precompileCommonVariants()
        
        let stats = await shaderCompiler.getCacheStatistics()
        logger.info("Precompilation complete: \(stats.memoryCachedPipelines) pipelines cached")
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
    public func clearCache() async {
        await shaderCompiler.clearCache()
        logger.info("Cleared pipeline cache")
    }
    
    /// Get cache statistics
    public var cacheStatistics: PipelineCacheStatistics {
        get async {
            let stats = await shaderCompiler.getCacheStatistics()
            return PipelineCacheStatistics(
                cachedPipelines: stats.memoryCachedPipelines,
                pipelineNames: await shaderCompiler.availableFunctions(),
                diskCacheSize: stats.diskCacheSize,
                compilationMetrics: stats.compilationMetrics
            )
        }
    }
    
    /// Load custom shader library
    public func loadShaderLibrary(name: String, source: String? = nil, url: URL? = nil) async throws {
        try await shaderCompiler.loadLibrary(name: name, source: source, url: url)
        logger.info("Loaded shader library: \(name)")
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
    public let diskCacheSize: Int
    public let compilationMetrics: [ShaderCompilationMetric]
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

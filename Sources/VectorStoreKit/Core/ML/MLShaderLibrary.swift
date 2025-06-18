// VectorStoreKit: ML Shader Library
//
// Manages Metal compute pipelines for ML operations

import Foundation
@preconcurrency import Metal

/// ML Shader Library managing compute pipelines
public final class MLShaderLibrary: @unchecked Sendable {
    private let device: MTLDevice
    private let library: MTLLibrary
    private let pipelineCache = NSCache<NSString, MTLComputePipelineState>()
    private let pipelineLock = NSLock()
    
    public init(device: MTLDevice) throws {
        self.device = device
        
        // Try to load compiled library first
        if let library = device.makeDefaultLibrary() {
            self.library = library
        } else {
            // Compile from source
            let shaderSource = try MLShaderLibrary.loadShaderSource()
            guard let library = try? device.makeLibrary(source: shaderSource, options: nil) else {
                throw MetalMLError.shaderCompilationFailed("Failed to create Metal library")
            }
            self.library = library
        }
        
        // Pre-compile commonly used pipelines
        try precompilePipelines()
    }
    
    // MARK: - Pipeline Management
    
    /// Get or create a compute pipeline for the specified function
    public func pipeline(for functionName: String) throws -> MTLComputePipelineState {
        pipelineLock.lock()
        defer { pipelineLock.unlock() }
        
        let key = functionName as NSString
        if let cachedPipeline = pipelineCache.object(forKey: key) {
            return cachedPipeline
        }
        
        guard let function = library.makeFunction(name: functionName) else {
            throw MetalMLError.shaderCompilationFailed("Function '\(functionName)' not found")
        }
        
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelineCache.setObject(pipeline, forKey: key)
        return pipeline
    }
    
    /// Create a compute pipeline for the specified function name
    /// This method provides async compatibility for the pipeline creation process
    public func makeComputePipeline(for functionName: String) async throws -> MTLComputePipelineState {
        return try pipeline(for: functionName)
    }
    
    /// Create a compute pipeline with custom configuration
    public func makeComputePipeline(
        for functionName: String,
        constants: MTLFunctionConstantValues? = nil
    ) async throws -> MTLComputePipelineState {
        pipelineLock.lock()
        defer { pipelineLock.unlock() }
        
        // Create unique cache key if constants are provided
        let cacheKey: NSString
        if constants != nil {
            cacheKey = "\(functionName)_custom" as NSString
        } else {
            cacheKey = functionName as NSString
        }
        
        // Check cache first
        if let cachedPipeline = pipelineCache.object(forKey: cacheKey) {
            return cachedPipeline
        }
        
        // Create function with constants if provided
        let function: MTLFunction
        if let constants = constants {
            guard let baseFunction = library.makeFunction(name: functionName) else {
                throw MetalMLError.shaderCompilationFailed("Function '\(functionName)' not found in library")
            }
            function = try baseFunction.makeFunction(constantValues: constants)
        } else {
            guard let f = library.makeFunction(name: functionName) else {
                throw MetalMLError.shaderCompilationFailed("Function '\(functionName)' not found in library")
            }
            function = f
        }
        
        // Create pipeline state
        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            pipelineCache.setObject(pipeline, forKey: cacheKey)
            return pipeline
        } catch {
            throw MetalMLError.pipelineCreationFailed(function: functionName)
        }
    }
    
    /// Clear the pipeline cache to free memory
    public func clearPipelineCache() {
        pipelineLock.lock()
        defer { pipelineLock.unlock() }
        pipelineCache.removeAllObjects()
    }
    
    /// Get the number of cached pipelines
    public var cachedPipelineCount: Int {
        pipelineLock.lock()
        defer { pipelineLock.unlock() }
        return pipelineCache.countLimit
    }
    
    // MARK: - Shader Functions
    
    public enum MatrixOperation: String, CaseIterable {
        case matmulForward = "matmul_forward"
        case matmulTiled = "matmul_tiled"
        case matvecForward = "matvec_forward"
        case matmulBackwardA = "matmul_backward_A"
        case matmulBackwardB = "matmul_backward_B"
        case addBias = "add_bias"
        case reduceBiasGradient = "reduce_bias_gradient"
        case transpose = "transpose"
        case copyMatrix = "copy_matrix"
        case outerProduct = "outer_product"
        case outerProductAccumulate = "outer_product_accumulate"
        // LSTM operations
        case extractTimestep = "extract_timestep"
        case concatenateSequence = "concatenate_sequence"
        case extractTimestepGrad = "extract_timestep_grad"
    }
    
    public enum ElementwiseOperation: String, CaseIterable {
        case elementAdd = "element_add"
        case elementMultiply = "element_multiply"
        case oneMinus = "one_minus"
        case extractSlice = "extract_slice"
        case copyToOffset = "copy_to_offset"
        case accumulateAtOffset = "accumulate_at_offset"
        case scaleBuffer = "scale_buffer"
        case sumBuffers = "sum_buffers"
        case concatenateBuffers = "concatenate_buffers"
        case splitBuffer = "split_buffer"
        case maxBuffers = "max_buffers"
        case addBuffers = "add_buffers"
        case dropoutForward = "dropout_forward"
        case dropoutBackward = "dropout_backward"
    }
    
    public enum ActivationFunction: String, CaseIterable {
        // Forward
        case reluForward = "relu_forward"
        case leakyReluForward = "leaky_relu_forward"
        case sigmoidForward = "sigmoid_forward"
        case tanhForward = "tanh_forward"
        case softmaxForward = "softmax_forward"
        case geluForward = "gelu_forward"
        case swishForward = "swish_forward"
        case eluForward = "elu_forward"
        case mishForward = "mish_forward"
        case linearForward = "linear_forward"
        
        // Backward
        case reluBackward = "relu_backward"
        case leakyReluBackward = "leaky_relu_backward"
        case sigmoidBackward = "sigmoid_backward"
        case tanhBackward = "tanh_backward"
        case softmaxBackward = "softmax_backward"
        case geluBackward = "gelu_backward"
        case swishBackward = "swish_backward"
        case eluBackward = "elu_backward"
        case mishBackward = "mish_backward"
        case linearBackward = "linear_backward"
    }
    
    public enum NormalizationFunction: String, CaseIterable {
        case batchNormComputeStats = "batch_norm_compute_stats"
        case batchNormForward = "batch_norm_forward"
        case batchNormUpdateRunningStats = "batch_norm_update_running_stats"
        case batchNormBackward = "batch_norm_backward"
        case layerNormForward = "layer_norm_forward"
        case layerNormBackward = "layer_norm_backward"
        case dropoutGenerateMask = "dropout_generate_mask"
        case dropoutForward = "dropout_forward"
        case dropoutBackward = "dropout_backward"
    }
    
    public enum LossFunction: String, CaseIterable {
        case mseLoss = "mse_loss"
        case mseGradient = "mse_gradient"
        case crossEntropyLoss = "cross_entropy_loss"
        case crossEntropyGradient = "cross_entropy_gradient"
        case reduceSum = "reduce_sum"
        case reduceMean = "reduce_mean"
    }
    
    public enum ActivationGradient: String, CaseIterable {
        case sigmoidGradient = "sigmoid_gradient_gated"
        case tanhGradient = "tanh_gradient_gated"
    }
    
    public enum OptimizationOperation: String, CaseIterable {
        case sgdUpdate = "sgd_update"
        case sgdMomentumUpdate = "sgd_momentum_update"
        case adamUpdate = "adam_update"
        case rmspropUpdate = "rmsprop_update"
        case adagradUpdate = "adagrad_update"
        case clipGradientsByNorm = "clip_gradients_by_norm"
        case clipGradientsByValue = "clip_gradients_by_value"
        case zeroGradients = "zero_gradients"
        case accumulateGradients = "accumulate_gradients"
        case scaleGradients = "scale_gradients"
    }
    
    // MARK: - Private Methods
    
    private func precompilePipelines() throws {
        // Precompile frequently used operations
        let priorityFunctions = [
            MatrixOperation.matmulForward.rawValue,
            MatrixOperation.addBias.rawValue,
            ActivationFunction.reluForward.rawValue,
            ActivationFunction.reluBackward.rawValue,
            NormalizationFunction.dropoutForward.rawValue
        ]
        
        var failedCompilations: [String] = []
        
        for functionName in priorityFunctions {
            do {
                _ = try pipeline(for: functionName)
            } catch {
                failedCompilations.append(functionName)
            }
        }
        
        // If critical functions failed to compile, throw error
        if failedCompilations.contains(MatrixOperation.matmulForward.rawValue) ||
           failedCompilations.contains(ActivationFunction.reluForward.rawValue) {
            throw MetalMLError.shaderCompilationFailed(
                "Failed to compile critical shaders: \(failedCompilations.joined(separator: ", "))"
            )
        }
    }
    
    /// Validate that all required shader functions are available
    public func validateShaderFunctions() throws {
        var missingFunctions: [String] = []
        
        // Check matrix operations
        for op in MatrixOperation.allCases {
            if library.makeFunction(name: op.rawValue) == nil {
                missingFunctions.append(op.rawValue)
            }
        }
        
        // Check activation functions
        for op in ActivationFunction.allCases {
            if library.makeFunction(name: op.rawValue) == nil {
                missingFunctions.append(op.rawValue)
            }
        }
        
        // Check critical functions only
        let criticalFunctions = [
            MatrixOperation.matmulForward.rawValue,
            ActivationFunction.reluForward.rawValue,
            ElementwiseOperation.elementAdd.rawValue
        ]
        
        let missingCritical = criticalFunctions.filter { functionName in
            library.makeFunction(name: functionName) == nil
        }
        
        if !missingCritical.isEmpty {
            throw MetalMLError.shaderCompilationFailed(
                "Missing critical shader functions: \(missingCritical.joined(separator: ", "))"
            )
        }
    }
    
    private static func loadShaderSource() throws -> String {
        // Combine all shader sources
        let shaderFiles = [
            "MatrixOperations.metal",
            "Activations.metal", 
            "Normalization.metal",
            "NormalizationShaders.metal",
            "LossOperations.metal",
            "ElementwiseOperations.metal",
            "OptimizationShaders.metal"
        ]
        
        var combinedSource = "#include <metal_stdlib>\nusing namespace metal;\n\n"
        
        // Load shader files from bundle or file system
        let bundle = Bundle.module
        let fileManager = FileManager.default
        
        for fileName in shaderFiles {
            var shaderContent: String?
            
            // Try to load from bundle first
            if let resourcePath = bundle.path(forResource: fileName.replacingOccurrences(of: ".metal", with: ""), 
                                              ofType: "metal") {
                shaderContent = try? String(contentsOfFile: resourcePath)
            }
            
            // If not in bundle, try relative path
            if shaderContent == nil {
                // Get current file URL and construct path to Shaders directory
                let currentFile = URL(fileURLWithPath: #file)
                let shadersDir = currentFile
                    .deletingLastPathComponent()
                    .appendingPathComponent("Shaders")
                    .appendingPathComponent(fileName)
                
                if fileManager.fileExists(atPath: shadersDir.path) {
                    shaderContent = try? String(contentsOfFile: shadersDir.path)
                }
            }
            
            if let content = shaderContent {
                // Remove redundant includes from individual files
                let cleanedContent = content
                    .replacingOccurrences(of: "#include <metal_stdlib>", with: "")
                    .replacingOccurrences(of: "using namespace metal;", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                
                combinedSource += "\n// ===== \(fileName) =====\n"
                combinedSource += cleanedContent
                combinedSource += "\n\n"
            }
        }
        
        // If no files loaded successfully, use fallback implementation
        if combinedSource.count < 200 {
            combinedSource += """
            // Fallback Metal implementation
            kernel void relu_forward(
                constant float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                constant uint& size [[buffer(2)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= size) return;
                output[gid] = max(0.0f, input[gid]);
            }
            
            kernel void matmul_forward(
                constant float* A [[buffer(0)]],
                constant float* B [[buffer(1)]],
                device float* C [[buffer(2)]],
                constant uint& M [[buffer(3)]],
                constant uint& N [[buffer(4)]],
                constant uint& K [[buffer(5)]],
                uint2 gid [[thread_position_in_grid]]
            ) {
                uint row = gid.y;
                uint col = gid.x;
                
                if (row >= M || col >= N) return;
                
                float sum = 0.0f;
                for (uint k = 0; k < K; k++) {
                    sum += A[row * K + k] * B[k * N + col];
                }
                
                C[row * N + col] = sum;
            }
            
            kernel void add_bias(
                constant float* input [[buffer(0)]],
                constant float* bias [[buffer(1)]],
                device float* output [[buffer(2)]],
                constant uint& size [[buffer(3)]],
                constant uint& bias_size [[buffer(4)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid >= size) return;
                uint bias_idx = gid % bias_size;
                output[gid] = input[gid] + bias[bias_idx];
            }
            """
        }
        
        return combinedSource
    }
}

// MARK: - Compute Helpers

extension MLShaderLibrary {
    
    /// Calculate optimal thread configuration for a given workload
    public func threadConfiguration(
        for pipeline: MTLComputePipelineState,
        workSize: MTLSize
    ) -> (threadgroupSize: MTLSize, threadgroupCount: MTLSize) {
        let maxThreadsPerThreadgroup = pipeline.maxTotalThreadsPerThreadgroup
        let threadExecutionWidth = pipeline.threadExecutionWidth
        
        // For 1D workloads
        if workSize.height == 1 && workSize.depth == 1 {
            let optimalWidth = min(workSize.width, maxThreadsPerThreadgroup)
            let threadgroupSize = MTLSize(width: optimalWidth, height: 1, depth: 1)
            let threadgroupCount = MTLSize(
                width: (workSize.width + optimalWidth - 1) / optimalWidth,
                height: 1,
                depth: 1
            )
            return (threadgroupSize, threadgroupCount)
        }
        
        // For 2D workloads (matrix operations)
        let tileSize = 16 // Common tile size for matrix operations
        let threadgroupSize = MTLSize(width: tileSize, height: tileSize, depth: 1)
        let threadgroupCount = MTLSize(
            width: (workSize.width + tileSize - 1) / tileSize,
            height: (workSize.height + tileSize - 1) / tileSize,
            depth: 1
        )
        
        return (threadgroupSize, threadgroupCount)
    }
    
    /// Calculate GPU-adaptive thread configuration based on workload and device capabilities
    public func adaptiveThreadConfiguration(
        for pipeline: MTLComputePipelineState,
        functionName: String,
        workSize: MTLSize,
        preferBatching: Bool = false
    ) -> (threadgroupSize: MTLSize, threadgroupCount: MTLSize) {
        let maxThreadsPerThreadgroup = pipeline.maxTotalThreadsPerThreadgroup
        let threadExecutionWidth = pipeline.threadExecutionWidth
        
        // Determine optimal configuration based on operation type
        if workSize.height == 1 && workSize.depth == 1 {
            // 1D workload optimization
            return optimize1DConfiguration(
                workSize: workSize.width,
                maxThreads: maxThreadsPerThreadgroup,
                warpSize: threadExecutionWidth,
                functionName: functionName,
                preferBatching: preferBatching
            )
        } else {
            // 2D workload optimization (matrix operations)
            return optimize2DConfiguration(
                workSize: workSize,
                maxThreads: maxThreadsPerThreadgroup,
                warpSize: threadExecutionWidth,
                functionName: functionName
            )
        }
    }
    
    private func optimize1DConfiguration(
        workSize: Int,
        maxThreads: Int,
        warpSize: Int,
        functionName: String,
        preferBatching: Bool
    ) -> (threadgroupSize: MTLSize, threadgroupCount: MTLSize) {
        var optimalThreads: Int
        
        // Determine optimal thread count based on workload characteristics
        if workSize <= warpSize {
            // Very small workload - use single warp
            optimalThreads = warpSize
        } else if workSize <= 256 {
            // Small workload - use multiple of warp size
            optimalThreads = ((workSize + warpSize - 1) / warpSize) * warpSize
            optimalThreads = min(optimalThreads, 256)
        } else if isMemoryBoundOperation(functionName) {
            // Memory-bound operations benefit from smaller thread groups
            optimalThreads = 128
        } else if isComputeIntensiveOperation(functionName) {
            // Compute-intensive operations can use larger thread groups
            optimalThreads = min(512, maxThreads)
        } else {
            // Default balanced configuration
            optimalThreads = 256
        }
        
        // Adjust for batching preference
        if preferBatching && workSize > 1024 {
            // For batch operations, prefer power-of-2 sizes for better memory access
            optimalThreads = min(optimalThreads, 256)
        }
        
        // Ensure alignment to warp size
        optimalThreads = ((optimalThreads + warpSize - 1) / warpSize) * warpSize
        optimalThreads = min(optimalThreads, maxThreads)
        
        let threadgroupSize = MTLSize(width: optimalThreads, height: 1, depth: 1)
        let threadgroupCount = MTLSize(
            width: (workSize + optimalThreads - 1) / optimalThreads,
            height: 1,
            depth: 1
        )
        
        return (threadgroupSize, threadgroupCount)
    }
    
    private func optimize2DConfiguration(
        workSize: MTLSize,
        maxThreads: Int,
        warpSize: Int,
        functionName: String
    ) -> (threadgroupSize: MTLSize, threadgroupCount: MTLSize) {
        var tileWidth: Int
        var tileHeight: Int
        
        // Determine tile size based on operation type and GPU capabilities
        if functionName.contains("matmul") || functionName.contains("transpose") {
            // Matrix operations benefit from square tiles
            if maxThreads >= 1024 {
                // High-end GPU - use larger tiles
                tileWidth = 32
                tileHeight = 32
            } else if maxThreads >= 512 {
                // Mid-range GPU
                tileWidth = 16
                tileHeight = 16
            } else {
                // Lower-end GPU
                tileWidth = 8
                tileHeight = 8
            }
        } else if functionName.contains("conv") {
            // Convolution operations may benefit from rectangular tiles
            tileWidth = 16
            tileHeight = 8
        } else {
            // Default balanced tile size
            tileWidth = 16
            tileHeight = 16
        }
        
        // Adjust for small workloads
        if workSize.width < tileWidth || workSize.height < tileHeight {
            tileWidth = min(workSize.width, tileWidth)
            tileHeight = min(workSize.height, tileHeight)
        }
        
        // Ensure total threads don't exceed maximum
        while tileWidth * tileHeight > maxThreads {
            if tileWidth > tileHeight {
                tileWidth /= 2
            } else {
                tileHeight /= 2
            }
        }
        
        let threadgroupSize = MTLSize(width: tileWidth, height: tileHeight, depth: 1)
        let threadgroupCount = MTLSize(
            width: (workSize.width + tileWidth - 1) / tileWidth,
            height: (workSize.height + tileHeight - 1) / tileHeight,
            depth: 1
        )
        
        return (threadgroupSize, threadgroupCount)
    }
    
    private func isMemoryBoundOperation(_ functionName: String) -> Bool {
        let memoryBoundOps = [
            "copy", "extract", "split", "concatenate",
            "transpose", "reduce_bias_gradient"
        ]
        return memoryBoundOps.contains { functionName.contains($0) }
    }
    
    private func isComputeIntensiveOperation(_ functionName: String) -> Bool {
        let computeIntensiveOps = [
            "matmul", "conv", "gelu", "swish", "mish",
            "softmax", "layer_norm", "batch_norm"
        ]
        return computeIntensiveOps.contains { functionName.contains($0) }
    }
}
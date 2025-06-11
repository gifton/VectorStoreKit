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
    }
    
    public enum ElementwiseOperation: String, CaseIterable {
        case elementAdd = "element_add"
        case elementMultiply = "element_multiply"
        case oneMinus = "one_minus"
        case extractSlice = "extract_slice"
        case copyToOffset = "copy_to_offset"
    }
    
    public enum ActivationFunction: String, CaseIterable {
        // Forward
        case reluForward = "relu_forward"
        case leakyReluForward = "leaky_relu_forward"
        case sigmoidForward = "sigmoid_forward"
        case tanhForward = "tanh_forward"
        case eluForward = "elu_forward"
        case seluForward = "selu_forward"
        case geluForward = "gelu_forward"
        case swishForward = "swish_forward"
        case softmaxForward = "softmax_forward"
        
        // Backward
        case reluBackward = "relu_backward"
        case leakyReluBackward = "leaky_relu_backward"
        case sigmoidBackward = "sigmoid_backward"
        case tanhBackward = "tanh_backward"
        case eluBackward = "elu_backward"
        case geluBackward = "gelu_backward"
        case swishBackward = "swish_backward"
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
        case maeLoss = "mae_loss"
        case maeGradient = "mae_gradient"
        case crossEntropyLoss = "cross_entropy_loss"
        case crossEntropyGradient = "cross_entropy_gradient"
        case binaryCrossEntropyLoss = "binary_cross_entropy_loss"
        case binaryCrossEntropyGradient = "binary_cross_entropy_gradient"
        case huberLoss = "huber_loss"
        case huberGradient = "huber_gradient"
        case reduceSum = "reduce_sum"
        case reduceMean = "reduce_mean"
    }
    
    public enum ConvolutionFunction: String, CaseIterable {
        case conv2dForward = "conv2d_forward"
        case conv2dBackwardInput = "conv2d_backward_input"
        case conv2dBackwardWeight = "conv2d_backward_weight"
        case conv2dBackwardBias = "conv2d_backward_bias"
    }
    
    public enum PoolingFunction: String, CaseIterable {
        case maxPool2dForward = "maxpool2d_forward"
        case maxPool2dBackward = "maxpool2d_backward"
        case avgPool2dForward = "avgpool2d_forward"
        case avgPool2dBackward = "avgpool2d_backward"
        case globalAvgPool2dForward = "global_avgpool2d_forward"
        case globalAvgPool2dBackward = "global_avgpool2d_backward"
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
        
        for functionName in priorityFunctions {
            _ = try? pipeline(for: functionName)
        }
    }
    
    private static func loadShaderSource() throws -> String {
        // Combine all shader sources
        let _ = [
            "MatrixOperations.metal",
            "Activations.metal", 
            "Normalization.metal",
            "LossOperations.metal",
            "ElementwiseOperations.metal",
            "ConvolutionShaders.metal",
            "PoolingShaders.metal"
        ]
        
        var combinedSource = "#include <metal_stdlib>\nusing namespace metal;\n\n"
        
        // For now, skip loading from files and use the minimal implementation below
        
        // If no files found, include minimal implementation
        if combinedSource.count < 100 {
            combinedSource += """
            // Minimal Metal implementation
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
        let _ = pipeline.threadExecutionWidth
        
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
}
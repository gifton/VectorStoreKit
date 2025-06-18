// VectorStoreKit: Mixed Precision Extensions for Metal ML Operations
//
// Extensions to MetalMLOperations for mixed precision compute

import Foundation
@preconcurrency import Metal

// MARK: - Mixed Precision Extensions

/// Mixed precision support for MetalMLOperations
public extension MetalMLOperations {
    
    // MARK: - Mixed Precision Matrix Operations
    
    /// Perform matrix multiplication with mixed precision
    /// Uses FP16 for compute and FP32 for accumulation to balance speed and accuracy
    func matmulMixedPrecision(
        _ a: MixedPrecisionBuffer,      // [M x K]
        _ b: MixedPrecisionBuffer,      // [K x N]
        output: MetalBuffer,            // [M x N] - always FP32
        m: Int, n: Int, k: Int
    ) async throws {
        // Validate dimensions
        guard m > 0 && n > 0 && k > 0 else {
            throw MetalMLError.invalidBufferSize("Matrix dimensions must be positive: m=\(m), n=\(n), k=\(k)")
        }
        
        // Validate buffer sizes
        let expectedASize = m * k
        let expectedBSize = k * n
        let expectedOutputSize = m * n
        
        guard a.buffer.count >= expectedASize else {
            throw MetalMLError.incompatibleBufferSize(expected: expectedASize, actual: a.buffer.count)
        }
        guard b.buffer.count >= expectedBSize else {
            throw MetalMLError.incompatibleBufferSize(expected: expectedBSize, actual: b.buffer.count)
        }
        guard output.count >= expectedOutputSize else {
            throw MetalMLError.incompatibleBufferSize(expected: expectedOutputSize, actual: output.count)
        }
        
        // Check for NaN/Inf values
        if a.buffer.containsNaNOrInf() {
            throw MetalMLError.numericalInstability("NaN or Inf detected in matrix A for mixed precision matmul")
        }
        if b.buffer.containsNaNOrInf() {
            throw MetalMLError.numericalInstability("NaN or Inf detected in matrix B for mixed precision matmul")
        }
        // Ensure inputs are in FP16 for compute efficiency
        let converter = try PrecisionConverter(device: device)
        
        let aFP16: MetalBuffer
        let bFP16: MetalBuffer
        
        if a.precision == .fp32 {
            let fp16Buffer = try a.buffer.withPrecision(.fp16, device: device)
            try await converter.convertFP32ToFP16(a.buffer, output: fp16Buffer.buffer)
            aFP16 = fp16Buffer.buffer
        } else {
            aFP16 = a.buffer
        }
        
        if b.precision == .fp32 {
            let fp16Buffer = try b.buffer.withPrecision(.fp16, device: device)
            try await converter.convertFP32ToFP16(b.buffer, output: fp16Buffer.buffer)
            bFP16 = fp16Buffer.buffer
        } else {
            bFP16 = b.buffer
        }
        
        // Use mixed precision kernel
        let pipeline = try shaderLibrary.pipeline(for: "matmul_mixed_precision")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(aFP16.buffer, offset: 0, index: 0)
        encoder.setBuffer(bFP16.buffer, offset: 0, index: 1)
        encoder.setBuffer(output.buffer, offset: 0, index: 2)
        
        var mVal = UInt32(m)
        var nVal = UInt32(n)
        var kVal = UInt32(k)
        encoder.setBytes(&mVal, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&kVal, length: MemoryLayout<UInt32>.size, index: 5)
        
        let workSize = MTLSize(width: n, height: m, depth: 1)
        let (threadgroupSize, threadgroupCount) = shaderLibrary.threadConfiguration(
            for: pipeline,
            workSize: workSize
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        // Use async execution
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
    }
    
    /// Perform linear layer with mixed precision (matmul + bias + activation)
    func linearMixedPrecision(
        input: MixedPrecisionBuffer,
        weights: MixedPrecisionBuffer,
        bias: MetalBuffer,              // Bias stays in FP32
        output: MixedPrecisionBuffer,
        batchSize: Int,
        inputSize: Int,
        outputSize: Int,
        activation: Activation = .linear
    ) async throws {
        let pipeline = try shaderLibrary.pipeline(for: "linear_mixed_precision")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        // Map activation to integer
        var activationType: UInt32
        switch activation {
        case .linear: activationType = 0
        case .relu: activationType = 1
        default: activationType = 0 // Fallback to linear
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(weights.buffer.buffer, offset: 0, index: 1)
        encoder.setBuffer(bias.buffer, offset: 0, index: 2)
        encoder.setBuffer(output.buffer.buffer, offset: 0, index: 3)
        
        var batchSizeVal = UInt32(batchSize)
        var inputSizeVal = UInt32(inputSize)
        var outputSizeVal = UInt32(outputSize)
        encoder.setBytes(&batchSizeVal, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&inputSizeVal, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&outputSizeVal, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&activationType, length: MemoryLayout<UInt32>.size, index: 7)
        
        let workSize = MTLSize(width: outputSize, height: batchSize, depth: 1)
        let (threadgroupSize, threadgroupCount) = shaderLibrary.threadConfiguration(
            for: pipeline,
            workSize: workSize
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
    }
    
    // MARK: - Loss Scaling Operations
    
    /// Scale gradients for mixed precision training
    func scaleGradients(
        _ gradients: MetalBuffer,
        scale: Float,
        precision: Precision = .fp32
    ) async throws {
        let functionName = precision == .fp32 ? "scale_gradients" : "scale_gradients_fp16"
        let pipeline = try shaderLibrary.pipeline(for: functionName)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradients.buffer, offset: 0, index: 0)
        
        var scaleVal = scale
        var count = UInt32(gradients.count)
        encoder.setBytes(&scaleVal, length: MemoryLayout<Float>.size, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (gradients.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
    }
    
    /// Unscale gradients after mixed precision training
    func unscaleGradients(
        _ gradients: MetalBuffer,
        scale: Float,
        precision: Precision = .fp32
    ) async throws {
        let functionName = precision == .fp32 ? "unscale_gradients" : "unscale_gradients_fp16"
        let pipeline = try shaderLibrary.pipeline(for: functionName)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradients.buffer, offset: 0, index: 0)
        
        var scaleVal = scale
        var count = UInt32(gradients.count)
        encoder.setBytes(&scaleVal, length: MemoryLayout<Float>.size, index: 1)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (gradients.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
    }
    
    // MARK: - Mixed Precision Optimizer Updates
    
    /// SGD update with mixed precision (FP32 master weights, FP16 model weights)
    func sgdUpdateMixedPrecision(
        masterWeights: MetalBuffer,     // FP32
        modelWeights: MetalBuffer,      // FP16
        gradients: MetalBuffer,         // FP32
        learningRate: Float
    ) async throws {
        let pipeline = try shaderLibrary.pipeline(for: "sgd_update_mixed_precision")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(masterWeights.buffer, offset: 0, index: 0)
        encoder.setBuffer(modelWeights.buffer, offset: 0, index: 1)
        encoder.setBuffer(gradients.buffer, offset: 0, index: 2)
        
        var lr = learningRate
        var count = UInt32(masterWeights.count)
        encoder.setBytes(&lr, length: MemoryLayout<Float>.size, index: 3)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 4)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (masterWeights.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
    }
    
    /// Adam update with mixed precision
    func adamUpdateMixedPrecision(
        masterWeights: MetalBuffer,     // FP32
        modelWeights: MetalBuffer,      // FP16
        gradients: MetalBuffer,         // FP32
        m: MetalBuffer,                 // FP32 first moment
        v: MetalBuffer,                 // FP32 second moment
        learningRate: Float,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        epsilon: Float = 1e-8,
        timestep: Int
    ) async throws {
        let pipeline = try shaderLibrary.pipeline(for: "adam_update_mixed_precision")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(masterWeights.buffer, offset: 0, index: 0)
        encoder.setBuffer(modelWeights.buffer, offset: 0, index: 1)
        encoder.setBuffer(gradients.buffer, offset: 0, index: 2)
        encoder.setBuffer(m.buffer, offset: 0, index: 3)
        encoder.setBuffer(v.buffer, offset: 0, index: 4)
        
        var lr = learningRate
        var b1 = beta1
        var b2 = beta2
        var eps = epsilon
        var t = UInt32(timestep)
        var count = UInt32(masterWeights.count)
        
        encoder.setBytes(&lr, length: MemoryLayout<Float>.size, index: 5)
        encoder.setBytes(&b1, length: MemoryLayout<Float>.size, index: 6)
        encoder.setBytes(&b2, length: MemoryLayout<Float>.size, index: 7)
        encoder.setBytes(&eps, length: MemoryLayout<Float>.size, index: 8)
        encoder.setBytes(&t, length: MemoryLayout<UInt32>.size, index: 9)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 10)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (masterWeights.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
    }
    
    // MARK: - Batch Normalization with Mixed Precision
    
    /// Batch normalization with mixed precision
    func batchNormMixedPrecision(
        input: MixedPrecisionBuffer,
        runningMean: MetalBuffer,       // Always FP32 for stability
        runningVar: MetalBuffer,        // Always FP32 for stability
        gamma: MixedPrecisionBuffer,
        beta: MixedPrecisionBuffer,
        output: MixedPrecisionBuffer,
        epsilon: Float = 1e-5,
        batchSize: Int,
        channels: Int
    ) async throws {
        let pipeline = try shaderLibrary.pipeline(for: "batchnorm_mixed_precision")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(runningMean.buffer, offset: 0, index: 1)
        encoder.setBuffer(runningVar.buffer, offset: 0, index: 2)
        encoder.setBuffer(gamma.buffer.buffer, offset: 0, index: 3)
        encoder.setBuffer(beta.buffer.buffer, offset: 0, index: 4)
        encoder.setBuffer(output.buffer.buffer, offset: 0, index: 5)
        
        var eps = epsilon
        var batchSizeVal = UInt32(batchSize)
        var channelsVal = UInt32(channels)
        
        encoder.setBytes(&eps, length: MemoryLayout<Float>.size, index: 6)
        encoder.setBytes(&batchSizeVal, length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&channelsVal, length: MemoryLayout<UInt32>.size, index: 8)
        
        let workSize = MTLSize(width: channels, height: batchSize, depth: 1)
        let (threadgroupSize, threadgroupCount) = shaderLibrary.threadConfiguration(
            for: pipeline,
            workSize: workSize
        )
        
        encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
    }
    
    // MARK: - Reduction with Mixed Precision
    
    /// Reduce sum with mixed precision (FP16 input, FP32 accumulation)
    func reduceSumMixedPrecision(
        _ input: MixedPrecisionBuffer
    ) async throws -> Float {
        let pipeline = try shaderLibrary.pipeline(for: "reduce_sum_mixed_precision")
        
        var currentSize = input.elementCount
        var inputBuffer = input.buffer
        
        // Perform reduction in multiple passes
        while currentSize > 1 {
            let outputSize = (currentSize + 255) / 256
            guard let outputMTLBuffer = device.makeBuffer(
                length: outputSize * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else {
                throw MetalMLError.bufferAllocationFailed(size: outputSize)
            }
            let outputBuffer = MetalBuffer(buffer: outputMTLBuffer, count: outputSize)
            
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputBuffer.buffer, offset: 0, index: 0)
            encoder.setBuffer(outputBuffer.buffer, offset: 0, index: 1)
            
            var sizeVal = UInt32(currentSize)
            encoder.setBytes(&sizeVal, length: MemoryLayout<UInt32>.size, index: 2)
            
            let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
            let threadgroups = MTLSize(width: outputSize, height: 1, depth: 1)
            
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()
            
            await withCheckedContinuation { continuation in
                commandBuffer.addCompletedHandler { _ in
                    continuation.resume()
                }
                commandBuffer.commit()
            }
            
            inputBuffer = outputBuffer
            currentSize = outputSize
        }
        
        // Read the final result
        let result = inputBuffer.toArray()
        return result[0]
    }
}
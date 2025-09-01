// VectorStoreKit: Mixed Precision Training Support
//
// Phase 4.2 implementation for memory-efficient neural network training
// with automatic loss scaling and precision management

import Foundation
@preconcurrency import Metal

// MARK: - Mixed Precision Configuration

/// Configuration for mixed precision training
/// Enables FP16 compute while maintaining FP32 for critical operations
public struct MixedPrecisionConfig: Sendable {
    /// Whether to use FP16 for compute operations (forward/backward passes)
    public let useFP16Compute: Bool
    
    /// Precision to use for loss computation (should be FP32 for numerical stability)
    public let lossPrecision: Precision
    
    /// Precision to use for gradient accumulation (should be FP32 to prevent underflow)
    public let gradientPrecision: Precision
    
    /// Initial loss scale factor to prevent gradient underflow in FP16
    public let initialLossScale: Float
    
    /// Growth factor for dynamic loss scaling
    public let lossScaleGrowthFactor: Float
    
    /// Backoff factor when overflow is detected
    public let lossScaleBackoffFactor: Float
    
    /// Number of iterations to wait before increasing loss scale
    public let lossScaleGrowthInterval: Int
    
    /// Minimum allowed loss scale
    public let minLossScale: Float
    
    /// Maximum allowed loss scale
    public let maxLossScale: Float
    
    public init(
        useFP16Compute: Bool = true,
        lossPrecision: Precision = .fp32,
        gradientPrecision: Precision = .fp32,
        initialLossScale: Float = 65536.0, // 2^16
        lossScaleGrowthFactor: Float = 2.0,
        lossScaleBackoffFactor: Float = 0.5,
        lossScaleGrowthInterval: Int = 2000,
        minLossScale: Float = 1.0,
        maxLossScale: Float = 65536.0 * 65536.0 // 2^32
    ) {
        self.useFP16Compute = useFP16Compute
        self.lossPrecision = lossPrecision
        self.gradientPrecision = gradientPrecision
        self.initialLossScale = initialLossScale
        self.lossScaleGrowthFactor = lossScaleGrowthFactor
        self.lossScaleBackoffFactor = lossScaleBackoffFactor
        self.lossScaleGrowthInterval = lossScaleGrowthInterval
        self.minLossScale = minLossScale
        self.maxLossScale = maxLossScale
    }
    
    /// Default configuration for mixed precision training
    public static let `default` = MixedPrecisionConfig()
    
    /// Conservative configuration with smaller initial loss scale
    public static let conservative = MixedPrecisionConfig(
        initialLossScale: 1024.0,
        lossScaleGrowthInterval: 1000
    )
    
    /// Aggressive configuration for maximum memory savings
    public static let aggressive = MixedPrecisionConfig(
        useFP16Compute: true,
        lossPrecision: .fp32,
        gradientPrecision: .fp16,
        initialLossScale: 32768.0
    )
}

/// Precision types supported by Metal
public enum Precision: String, Codable, Sendable {
    case fp32 = "float32"
    case fp16 = "float16"
    
    /// Size in bytes for this precision
    public var byteSize: Int {
        switch self {
        case .fp32: return 4
        case .fp16: return 2
        }
    }
    
    /// Metal data type for this precision
    public var metalDataType: MTLDataType {
        switch self {
        case .fp32: return .float
        case .fp16: return .half
        }
    }
}

// MARK: - Precision Conversion Utilities

/// Utilities for converting between FP32 and FP16 precision
public actor PrecisionConverter {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var pipelineCache: [String: MTLComputePipelineState] = [:]
    
    public init(device: MTLDevice) throws {
        self.device = device
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
    }
    
    /// Convert FP32 buffer to FP16
    /// - Parameters:
    ///   - input: Input buffer containing FP32 values
    ///   - output: Output buffer to store FP16 values (must be half the byte size)
    /// - Note: This reduces memory usage by 50% but may lose some precision
    public func convertFP32ToFP16(
        _ input: MetalBuffer,
        output: MetalBuffer
    ) async throws {
        // Validate buffer sizes
        guard output.byteLength == input.count * 2 else {
            throw MetalMLError.incompatibleBufferSize(
                expected: input.count * 2,
                actual: output.byteLength
            )
        }
        
        let pipeline = try await getPipeline(functionName: "convert_fp32_to_fp16")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        
        var count = UInt32(input.count)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (input.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        // Use async execution instead of blocking
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
    }
    
    /// Convert FP16 buffer to FP32
    /// - Parameters:
    ///   - input: Input buffer containing FP16 values
    ///   - output: Output buffer to store FP32 values (must be double the byte size)
    /// - Note: This restores full precision for critical operations
    public func convertFP16ToFP32(
        _ input: MetalBuffer,
        output: MetalBuffer
    ) async throws {
        // Validate buffer sizes
        guard input.byteLength * 2 == output.byteLength else {
            throw MetalMLError.incompatibleBufferSize(
                expected: input.byteLength * 2,
                actual: output.byteLength
            )
        }
        
        let pipeline = try await getPipeline(functionName: "convert_fp16_to_fp32")
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(input.buffer, offset: 0, index: 0)
        encoder.setBuffer(output.buffer, offset: 0, index: 1)
        
        var count = UInt32(output.count)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (output.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
    }
    
    /// Check buffer for overflow/underflow conditions
    /// - Parameter buffer: Buffer to check
    /// - Returns: Tuple indicating (hasOverflow, hasUnderflow)
    public func checkNumericStability(_ buffer: MetalBuffer) async throws -> (overflow: Bool, underflow: Bool) {
        let pipeline = try await getPipeline(functionName: "check_numeric_stability")
        
        // Create result buffer for overflow/underflow flags
        guard let resultBuffer = device.makeBuffer(length: 8, options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: 8)
        }
        
        // Initialize to zero
        resultBuffer.contents().bindMemory(to: UInt32.self, capacity: 2).initialize(repeating: 0, count: 2)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(buffer.buffer, offset: 0, index: 0)
        encoder.setBuffer(resultBuffer, offset: 0, index: 1)
        
        var count = UInt32(buffer.count)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroups = MTLSize(width: (buffer.count + 255) / 256, height: 1, depth: 1)
        
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        // Read results
        let results = resultBuffer.contents().bindMemory(to: UInt32.self, capacity: 2)
        return (overflow: results[0] > 0, underflow: results[1] > 0)
    }
    
    func getPipeline(functionName: String) async throws -> MTLComputePipelineState {
        if let cached = pipelineCache[functionName] {
            return cached
        }
        
        guard let library = device.makeDefaultLibrary(),
              let function = library.makeFunction(name: functionName) else {
            throw MetalMLError.shaderCompilationFailed("Function '\(functionName)' not found")
        }
        
        let pipeline = try await device.makeComputePipelineState(function: function)
        pipelineCache[functionName] = pipeline
        return pipeline
    }
    
    func getDevice() -> MTLDevice {
        return device
    }
    
    func getCommandQueue() -> MTLCommandQueue {
        return commandQueue
    }
}

// MARK: - Dynamic Loss Scaler

/// Manages dynamic loss scaling to prevent gradient underflow in FP16 training
public actor DynamicLossScaler {
    private let config: MixedPrecisionConfig
    private var currentScale: Float
    private var growthTracker: Int = 0
    private let converter: PrecisionConverter
    
    public private(set) var overflowCount: Int = 0
    public private(set) var updateCount: Int = 0
    
    public init(config: MixedPrecisionConfig, device: MTLDevice) throws {
        self.config = config
        self.currentScale = config.initialLossScale
        self.converter = try PrecisionConverter(device: device)
    }
    
    /// Get current loss scale
    public var scale: Float {
        currentScale
    }
    
    /// Scale gradients before backward pass
    public func scaleGradients(_ gradients: MetalBuffer) async throws {
        let pipeline = try await converter.getPipeline(functionName: "scale_gradients")
        let commandQueue = await converter.getCommandQueue()
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradients.buffer, offset: 0, index: 0)
        
        var scale = currentScale
        encoder.setBytes(&scale, length: MemoryLayout<Float>.size, index: 1)
        
        var count = UInt32(gradients.count)
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
    
    /// Unscale gradients after backward pass
    public func unscaleGradients(_ gradients: MetalBuffer) async throws {
        let pipeline = try await converter.getPipeline(functionName: "unscale_gradients")
        let commandQueue = await converter.getCommandQueue()
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradients.buffer, offset: 0, index: 0)
        
        var invScale = 1.0 / currentScale
        encoder.setBytes(&invScale, length: MemoryLayout<Float>.size, index: 1)
        
        var count = UInt32(gradients.count)
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
    
    /// Check if scaling gradients would cause overflow
    private func checkGradientOverflow(_ gradients: MetalBuffer, scale: Float) async throws -> Bool {
        let pipeline = try await converter.getPipeline(functionName: "check_gradient_overflow")
        let commandQueue = await converter.getCommandQueue()
        let device = await converter.getDevice()
        
        // Create result buffer for overflow flag
        guard let overflowBuffer = device.makeBuffer(length: 4, options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: 4)
        }
        
        // Initialize to zero
        overflowBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).initialize(to: 0)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalMLError.commandQueueCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(gradients.buffer, offset: 0, index: 0)
        
        var scaleValue = scale
        encoder.setBytes(&scaleValue, length: MemoryLayout<Float>.size, index: 1)
        encoder.setBuffer(overflowBuffer, offset: 0, index: 2)
        
        var count = UInt32(gradients.count)
        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
        
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
        
        // Read result
        let overflow = overflowBuffer.contents().bindMemory(to: UInt32.self, capacity: 1).pointee > 0
        return overflow
    }
    
    /// Update scale based on gradient stability
    /// - Parameter gradients: Gradients to check for overflow/underflow
    /// - Returns: Whether the update was successful (no overflow)
    public func update(_ gradients: MetalBuffer) async throws -> Bool {
        updateCount += 1
        
        // First check if scaling would cause overflow
        let wouldOverflow = try await checkGradientOverflow(gradients, scale: currentScale)
        
        if wouldOverflow {
            // Backoff the scale
            currentScale *= config.lossScaleBackoffFactor
            currentScale = max(currentScale, config.minLossScale)
            growthTracker = 0
            overflowCount += 1
            return false
        }
        
        // Then check for general numeric stability
        let (overflow, underflow) = try await converter.checkNumericStability(gradients)
        
        if overflow {
            // Backoff the scale
            currentScale *= config.lossScaleBackoffFactor
            currentScale = max(currentScale, config.minLossScale)
            growthTracker = 0
            overflowCount += 1
            return false
        }
        
        // Increment growth tracker
        growthTracker += 1
        
        // Check if we should increase scale
        if growthTracker >= config.lossScaleGrowthInterval {
            currentScale *= config.lossScaleGrowthFactor
            currentScale = min(currentScale, config.maxLossScale)
            growthTracker = 0
        }
        
        return true
    }
    
    /// Reset scaler to initial state
    public func reset() {
        currentScale = config.initialLossScale
        growthTracker = 0
        overflowCount = 0
        updateCount = 0
    }
    
    /// Get scaler statistics
    public func getStatistics() -> LossScalerStatistics {
        LossScalerStatistics(
            currentScale: currentScale,
            overflowCount: overflowCount,
            updateCount: updateCount,
            overflowRate: updateCount > 0 ? Float(overflowCount) / Float(updateCount) : 0
        )
    }
}

/// Statistics for loss scaler performance
public struct LossScalerStatistics: Sendable {
    public let currentScale: Float
    public let overflowCount: Int
    public let updateCount: Int
    public let overflowRate: Float
}

// MARK: - Mixed Precision Buffer

/// Extended MetalBuffer with precision tracking
public struct MixedPrecisionBuffer: Sendable {
    public let buffer: MetalBuffer
    public let precision: Precision
    
    public init(buffer: MetalBuffer, precision: Precision) {
        self.buffer = buffer
        self.precision = precision
    }
    
    /// Element count (accounts for precision)
    public var elementCount: Int {
        buffer.byteLength / precision.byteSize
    }
    
    /// Create a new buffer with different precision
    public func withPrecision(_ newPrecision: Precision, device: MTLDevice) throws -> MixedPrecisionBuffer {
        guard newPrecision != precision else { return self }
        
        let newByteLength = elementCount * newPrecision.byteSize
        guard let newMTLBuffer = device.makeBuffer(length: newByteLength, options: .storageModeShared) else {
            throw MetalMLError.bufferAllocationFailed(size: newByteLength)
        }
        
        let newBuffer = MetalBuffer(
            buffer: newMTLBuffer,
            shape: buffer.shape,
            stride: newPrecision.byteSize
        )
        
        return MixedPrecisionBuffer(buffer: newBuffer, precision: newPrecision)
    }
}

// MARK: - Precision-Aware Compute Graph Node

/// Compute graph node with precision tracking
public struct PrecisionAwareNode: Sendable {
    public let id: UUID
    public let operation: String
    public let inputPrecisions: [Precision]
    public let outputPrecision: Precision
    public let requiresFP32: Bool // For critical operations like loss
    
    public init(
        operation: String,
        inputPrecisions: [Precision],
        outputPrecision: Precision,
        requiresFP32: Bool = false
    ) {
        self.id = UUID()
        self.operation = operation
        self.inputPrecisions = inputPrecisions
        self.outputPrecision = outputPrecision
        self.requiresFP32 = requiresFP32
    }
}

// MARK: - Memory Usage Calculator

/// Utilities for calculating memory usage with mixed precision
public struct MixedPrecisionMemoryCalculator {
    /// Calculate memory savings from using mixed precision
    /// - Parameters:
    ///   - modelSize: Number of parameters in the model
    ///   - batchSize: Training batch size
    ///   - sequenceLength: Sequence length (for transformers, etc.)
    /// - Returns: Memory usage comparison
    public static func calculateMemorySavings(
        modelSize: Int,
        batchSize: Int,
        sequenceLength: Int = 1
    ) -> MemoryUsageComparison {
        // FP32 memory usage
        let fp32ModelMemory = modelSize * 4 // 4 bytes per float
        let fp32ActivationMemory = estimateActivationMemory(
            modelSize: modelSize,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            precision: .fp32
        )
        let fp32TotalMemory = fp32ModelMemory + fp32ActivationMemory
        
        // Mixed precision memory usage
        let fp16ModelMemory = modelSize * 2 // 2 bytes per half
        let fp16ActivationMemory = estimateActivationMemory(
            modelSize: modelSize,
            batchSize: batchSize,
            sequenceLength: sequenceLength,
            precision: .fp16
        )
        // Add FP32 master weights for optimization
        let masterWeightsMemory = modelSize * 4
        let mixedTotalMemory = fp16ModelMemory + fp16ActivationMemory + masterWeightsMemory
        
        let savings = Float(fp32TotalMemory - mixedTotalMemory) / Float(fp32TotalMemory)
        
        return MemoryUsageComparison(
            fp32ModelMemory: fp32ModelMemory,
            fp32ActivationMemory: fp32ActivationMemory,
            fp32TotalMemory: fp32TotalMemory,
            mixedModelMemory: fp16ModelMemory,
            mixedActivationMemory: fp16ActivationMemory,
            mixedMasterWeightsMemory: masterWeightsMemory,
            mixedTotalMemory: mixedTotalMemory,
            memorySavingsPercent: savings * 100
        )
    }
    
    private static func estimateActivationMemory(
        modelSize: Int,
        batchSize: Int,
        sequenceLength: Int,
        precision: Precision
    ) -> Int {
        // Rough estimate: 3x model size for activations
        let activationSize = modelSize * 3 * batchSize * sequenceLength
        return activationSize * precision.byteSize
    }
}

/// Memory usage comparison between FP32 and mixed precision
public struct MemoryUsageComparison: Sendable {
    public let fp32ModelMemory: Int
    public let fp32ActivationMemory: Int
    public let fp32TotalMemory: Int
    public let mixedModelMemory: Int
    public let mixedActivationMemory: Int
    public let mixedMasterWeightsMemory: Int
    public let mixedTotalMemory: Int
    public let memorySavingsPercent: Float
    
    public var description: String {
        """
        Memory Usage Comparison:
        FP32:
          - Model: \(formatBytes(fp32ModelMemory))
          - Activations: \(formatBytes(fp32ActivationMemory))
          - Total: \(formatBytes(fp32TotalMemory))
        
        Mixed Precision:
          - Model (FP16): \(formatBytes(mixedModelMemory))
          - Activations (FP16): \(formatBytes(mixedActivationMemory))
          - Master Weights (FP32): \(formatBytes(mixedMasterWeightsMemory))
          - Total: \(formatBytes(mixedTotalMemory))
        
        Memory Savings: \(String(format: "%.1f%%", memorySavingsPercent))
        """
    }
    
    private func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}
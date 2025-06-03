// VectorStoreKit: Metal Error Handling
//
// Comprehensive error types and recovery strategies for Metal operations

import Foundation

// MARK: - Error Types

/// Comprehensive error types for Metal operations
public enum MetalComputeError: Error, LocalizedError {
    // Device errors
    case deviceNotAvailable
    case deviceOutOfMemory(required: Int, available: Int)
    case unsupportedOperation(operation: String, reason: String)
    
    // Pipeline errors
    case pipelineCompilationFailed(function: String, error: String)
    case pipelineNotFound(function: String)
    case shaderCompilationFailed(shader: String, error: String)
    
    // Buffer errors
    case bufferAllocationFailed(size: Int, error: String)
    case bufferPoolExhausted(requestedSize: Int)
    case invalidBufferSize(requested: Int, maximum: Int)
    
    // Computation errors
    case computeEncoderCreationFailed
    case commandBufferCreationFailed
    case commandBufferExecutionFailed(error: String)
    case commandBufferTimeout(duration: TimeInterval)
    
    // Data errors
    case invalidInputData(reason: String)
    case dimensionMismatch(expected: Int, actual: Int)
    case emptyInput(parameter: String)
    
    // Performance errors
    case performanceThresholdExceeded(metric: String, value: Double, threshold: Double)
    case gpuOverheated(temperature: Float)
    
    public var errorDescription: String? {
        switch self {
        case .deviceNotAvailable:
            return "Metal device is not available on this system"
        case .deviceOutOfMemory(let required, let available):
            return "Device out of memory: required \(required) bytes, available \(available) bytes"
        case .unsupportedOperation(let operation, let reason):
            return "Operation '\(operation)' is not supported: \(reason)"
        case .pipelineCompilationFailed(let function, let error):
            return "Failed to compile pipeline for function '\(function)': \(error)"
        case .pipelineNotFound(let function):
            return "Pipeline not found for function '\(function)'"
        case .shaderCompilationFailed(let shader, let error):
            return "Failed to compile shader '\(shader)': \(error)"
        case .bufferAllocationFailed(let size, let error):
            return "Failed to allocate buffer of size \(size): \(error)"
        case .bufferPoolExhausted(let size):
            return "Buffer pool exhausted, cannot allocate \(size) bytes"
        case .invalidBufferSize(let requested, let maximum):
            return "Invalid buffer size: requested \(requested), maximum \(maximum)"
        case .computeEncoderCreationFailed:
            return "Failed to create compute command encoder"
        case .commandBufferCreationFailed:
            return "Failed to create command buffer"
        case .commandBufferExecutionFailed(let error):
            return "Command buffer execution failed: \(error)"
        case .commandBufferTimeout(let duration):
            return "Command buffer timed out after \(duration) seconds"
        case .invalidInputData(let reason):
            return "Invalid input data: \(reason)"
        case .dimensionMismatch(let expected, let actual):
            return "Dimension mismatch: expected \(expected), got \(actual)"
        case .emptyInput(let parameter):
            return "Empty input for parameter '\(parameter)'"
        case .performanceThresholdExceeded(let metric, let value, let threshold):
            return "Performance threshold exceeded for \(metric): \(value) > \(threshold)"
        case .gpuOverheated(let temperature):
            return "GPU overheated: \(temperature)°C"
        }
    }
    
    public var isRecoverable: Bool {
        switch self {
        case .deviceOutOfMemory, .bufferPoolExhausted, .commandBufferTimeout:
            return true
        case .deviceNotAvailable, .unsupportedOperation, .shaderCompilationFailed:
            return false
        default:
            return true
        }
    }
    
    public var suggestedRecoveryAction: RecoveryAction? {
        switch self {
        case .deviceOutOfMemory, .bufferPoolExhausted:
            return .reduceWorkload
        case .commandBufferTimeout:
            return .retry(maxAttempts: 3)
        case .gpuOverheated:
            return .cooldown(duration: 5.0)
        default:
            return nil
        }
    }
}

// MARK: - Recovery Actions

/// Suggested recovery actions for errors
public enum RecoveryAction {
    case retry(maxAttempts: Int)
    case reduceWorkload
    case fallbackToCPU
    case cooldown(duration: TimeInterval)
    case clearCache
    case splitBatch(factor: Int)
}

// MARK: - Error Recovery Protocol

/// Protocol for components that can recover from errors
public protocol ErrorRecoverable {
    func attemptRecovery(from error: MetalComputeError, action: RecoveryAction) async throws
}

// MARK: - Error Context

/// Context information for debugging errors
public struct MetalErrorContext {
    public let timestamp: Date
    public let operation: String
    public let parameters: [String: Any]
    public let deviceState: DeviceState
    
    public struct DeviceState {
        public let availableMemory: Int
        public let gpuUtilization: Float
        public let temperature: Float?
        public let activeOperations: Int
    }
}

// MARK: - Error Handler

/// Centralized error handler for Metal operations
public actor MetalErrorHandler {
    private var errorHistory: [MetalErrorContext] = []
    private let maxHistorySize = 100
    
    public init() {}
    
    /// Record an error with context
    public func recordError(_ error: MetalComputeError, context: MetalErrorContext) {
        errorHistory.append(context)
        
        // Maintain history size limit
        if errorHistory.count > maxHistorySize {
            errorHistory.removeFirst()
        }
    }
    
    /// Analyze error patterns
    public func analyzeErrorPatterns() -> ErrorAnalysis {
        let recentErrors = errorHistory.suffix(20)
        
        // Count error types
        var errorCounts: [String: Int] = [:]
        for context in recentErrors {
            let key = context.operation
            errorCounts[key, default: 0] += 1
        }
        
        // Find most common error
        let mostCommonError = errorCounts.max { $0.value < $1.value }
        
        // Calculate error rate
        let errorRate = Float(recentErrors.count) / Float(maxHistorySize)
        
        return ErrorAnalysis(
            totalErrors: errorHistory.count,
            recentErrorRate: errorRate,
            mostCommonOperation: mostCommonError?.key,
            recommendations: generateRecommendations(from: recentErrors)
        )
    }
    
    /// Clear error history
    public func clearHistory() {
        errorHistory.removeAll()
    }
    
    private func generateRecommendations(from errors: ArraySlice<MetalErrorContext>) -> [String] {
        var recommendations: [String] = []
        
        // Check for memory issues
        let memoryErrors = errors.filter { $0.deviceState.availableMemory < 100_000_000 }
        if memoryErrors.count > 3 {
            recommendations.append("Consider reducing batch sizes to avoid memory pressure")
        }
        
        // Check for high GPU utilization
        let highUtilization = errors.filter { $0.deviceState.gpuUtilization > 0.9 }
        if highUtilization.count > 5 {
            recommendations.append("GPU is consistently at high utilization, consider load balancing")
        }
        
        return recommendations
    }
}

// MARK: - Error Analysis

/// Analysis of error patterns
public struct ErrorAnalysis {
    public let totalErrors: Int
    public let recentErrorRate: Float
    public let mostCommonOperation: String?
    public let recommendations: [String]
}
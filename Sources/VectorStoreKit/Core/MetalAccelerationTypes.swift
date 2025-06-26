// MetalAccelerationTypes.swift
// VectorStoreKit
//
// Unified exports for Metal acceleration types
// Ensures all acceleration types are properly accessible and eliminates import issues

import Foundation
import Metal

// MARK: - Core Vector Types
// Note: Types are defined in their respective files within the same module

// MARK: - Additional Type Aliases for Compatibility

/// Compatibility aliases for Metal integration
public typealias MetalDistanceMetric = DistanceMetric
public typealias MetalQuantizationScheme = QuantizationScheme
public typealias MetalQuantizationParameters = QuantizationParameters
public typealias MetalVector512 = Vector512
public typealias MetalQuantizedVector = QuantizedVector

// MARK: - Missing Error Types

/// Buffer pool specific errors that were missing
public enum MetalBufferPoolError: Error, LocalizedError {
    case allocationFailed
    case poolExhausted
    case invalidSize(Int)
    case deviceError(String)
    case incompatibleBuffer
    case memoryPressureExceeded
    
    public var errorDescription: String? {
        switch self {
        case .allocationFailed:
            return "Failed to allocate Metal buffer"
        case .poolExhausted:
            return "Buffer pool has been exhausted"
        case .invalidSize(let size):
            return "Invalid buffer size: \(size)"
        case .deviceError(let message):
            return "Metal device error: \(message)"
        case .incompatibleBuffer:
            return "Buffer is incompatible with requested operation"
        case .memoryPressureExceeded:
            return "Memory pressure exceeded safe limits"
        }
    }
    
    public var isRecoverable: Bool {
        switch self {
        case .allocationFailed, .poolExhausted, .memoryPressureExceeded:
            return true
        case .invalidSize, .deviceError, .incompatibleBuffer:
            return false
        }
    }
}

// MARK: - System Memory Pressure Types
// SystemMemoryPressure enum is defined in MemoryManager.swift

// MARK: - Buffer Pool Types

/// Types of buffer pools for specialized handling
public enum BufferPoolType: String, Sendable, CaseIterable {
    case vector = "vector"
    case quantized = "quantized"
    case general = "general"
    case command = "command"
    case matrix = "matrix"
    
    /// Expected data alignment for this pool type
    public var dataAlignment: Int {
        switch self {
        case .vector:
            return 16 // SIMD alignment
        case .quantized:
            return 4  // UInt8 alignment
        case .general:
            return 8  // General alignment
        case .command:
            return 1  // No specific alignment
        case .matrix:
            return 16 // SIMD alignment for matrix operations
        }
    }
    
    /// Typical buffer sizes for pre-allocation
    public var typicalSizes: [Int] {
        switch self {
        case .vector:
            return [2048] // Vector512 size
        case .quantized:
            return [64, 128, 256, 512] // Common quantized sizes
        case .general:
            return [1024, 4096, 16384, 65536] // Power of 2 sizes
        case .command:
            return [] // Command buffers don't have typical sizes
        case .matrix:
            return [4096, 16384, 65536] // Common matrix sizes
        }
    }
}

// Note: CompressionLevel and ComputationalComplexity are already defined in Protocols.swift

// MARK: - Global Type Validation

/// Validate that all required types are accessible
public func validateMetalAccelerationTypes() -> Bool {
    // Test that all new types can be instantiated/referenced
    let _: MetalBufferPoolError.Type = MetalBufferPoolError.self
    let _: SystemMemoryPressure.Type = SystemMemoryPressure.self
    let _: BufferPoolType.Type = BufferPoolType.self
    
    return true
}

// MARK: - Configuration Validation Extensions

// Configuration extensions will be defined in their respective type files
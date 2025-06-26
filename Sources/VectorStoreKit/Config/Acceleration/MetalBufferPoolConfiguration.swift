// VectorStoreKit: Metal Buffer Pool Configuration
//
// Configuration for Metal buffer management

import Foundation

/// Configuration for buffer pool
public struct MetalBufferPoolConfiguration: Sendable {
    public let maxBuffersPerSize: Int
    public let preallocationSizes: [Int]
    public let maxMemoryUsage: Int
    public let enableProfiling: Bool
    
    public init(
        maxBuffersPerSize: Int = 10,
        preallocationSizes: [Int] = [],
        maxMemoryUsage: Int = 1024 * 1024 * 1024, // 1GB default
        enableProfiling: Bool = false
    ) {
        self.maxBuffersPerSize = maxBuffersPerSize
        self.preallocationSizes = preallocationSizes
        self.maxMemoryUsage = maxMemoryUsage
        self.enableProfiling = enableProfiling
    }
    
    public static let research = MetalBufferPoolConfiguration(
        maxBuffersPerSize: 20,
        preallocationSizes: [1024, 4096, 16384, 65536],
        maxMemoryUsage: 2 * 1024 * 1024 * 1024, // 2GB for research
        enableProfiling: true // Enable profiling for research
    )
    
    public static let efficient = MetalBufferPoolConfiguration(
        maxBuffersPerSize: 5,
        preallocationSizes: [],
        maxMemoryUsage: 512 * 1024 * 1024, // 512MB for efficiency
        enableProfiling: false // Disable profiling for efficiency
    )
    
    public static let appleSilicon = MetalBufferPoolConfiguration(
        maxBuffersPerSize: 15,
        preallocationSizes: [1024, 4096, 16384, 65536, 262144],
        maxMemoryUsage: 1024 * 1024 * 1024, // 1GB for Apple Silicon
        enableProfiling: false // Disable profiling by default
    )
    
    /// Validate the configuration
    public func validate() throws {
        if maxBuffersPerSize <= 0 {
            throw VectorStoreError.configurationInvalid("maxBuffersPerSize must be positive")
        }
        if maxMemoryUsage <= 0 {
            throw VectorStoreError.configurationInvalid("maxMemoryUsage must be positive")
        }
    }
}
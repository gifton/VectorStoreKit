// VectorStoreKit: Metal Compute Configuration
//
// Configuration for Metal GPU acceleration

import Foundation

/// Configuration for Metal compute operations
public struct MetalComputeConfiguration: Sendable {
    public let minBatchSizeForGPU: Int
    public let enableProfiling: Bool
    public let bufferPoolConfig: MetalBufferPoolConfiguration
    
    public init(
        minBatchSizeForGPU: Int = 1000,
        enableProfiling: Bool = true,
        bufferPoolConfig: MetalBufferPoolConfiguration = .research
    ) {
        self.minBatchSizeForGPU = minBatchSizeForGPU
        self.enableProfiling = enableProfiling
        self.bufferPoolConfig = bufferPoolConfig
    }
    
    public static let research = MetalComputeConfiguration(
        minBatchSizeForGPU: 500,
        enableProfiling: true,
        bufferPoolConfig: .research
    )
    
    public static let efficient = MetalComputeConfiguration(
        minBatchSizeForGPU: 2000,
        enableProfiling: false,
        bufferPoolConfig: .efficient
    )
}
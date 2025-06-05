// VectorStoreKit: Metal Buffer Pool Configuration
//
// Configuration for Metal buffer management

import Foundation

/// Configuration for buffer pool
public struct MetalBufferPoolConfiguration: Sendable {
    public let maxBuffersPerSize: Int
    public let preallocationSizes: [Int]
    
    public init(
        maxBuffersPerSize: Int = 10,
        preallocationSizes: [Int] = []
    ) {
        self.maxBuffersPerSize = maxBuffersPerSize
        self.preallocationSizes = preallocationSizes
    }
    
    public static let research = MetalBufferPoolConfiguration(
        maxBuffersPerSize: 20,
        preallocationSizes: [1024, 4096, 16384, 65536]
    )
    
    public static let efficient = MetalBufferPoolConfiguration(
        maxBuffersPerSize: 5,
        preallocationSizes: []
    )
}
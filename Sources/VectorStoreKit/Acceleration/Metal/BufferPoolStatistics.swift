// VectorStoreKit: Buffer Pool Statistics
//
// Statistics for Metal buffer pool management

import Foundation

/// Statistics for Metal buffer pool operations
public struct BufferPoolStatistics: Sendable {
    public let poolName: String
    public let totalAllocations: Int
    public let currentAllocations: Int
    public let peakAllocations: Int
    public let totalBytesAllocated: Int
    public let currentBytesAllocated: Int
    public let peakBytesAllocated: Int
    public let reuseCount: Int
    public let hitRate: Float
    public let averageAllocationSize: Int
    public let fragmentationRatio: Float
    
    public init(
        poolName: String = "default",
        totalAllocations: Int = 0,
        currentAllocations: Int = 0,
        peakAllocations: Int = 0,
        totalBytesAllocated: Int = 0,
        currentBytesAllocated: Int = 0,
        peakBytesAllocated: Int = 0,
        reuseCount: Int = 0,
        hitRate: Float = 0.0,
        averageAllocationSize: Int = 0,
        fragmentationRatio: Float = 0.0
    ) {
        self.poolName = poolName
        self.totalAllocations = totalAllocations
        self.currentAllocations = currentAllocations
        self.peakAllocations = peakAllocations
        self.totalBytesAllocated = totalBytesAllocated
        self.currentBytesAllocated = currentBytesAllocated
        self.peakBytesAllocated = peakBytesAllocated
        self.reuseCount = reuseCount
        self.hitRate = hitRate
        self.averageAllocationSize = averageAllocationSize
        self.fragmentationRatio = fragmentationRatio
    }
    
    /// Create empty statistics
    public static func empty(poolName: String = "default") -> BufferPoolStatistics {
        BufferPoolStatistics(poolName: poolName)
    }
}
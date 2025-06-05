// VectorStoreKit: FIFO Cache Configuration
//
// Configuration for First In First Out cache

import Foundation

/// FIFO cache configuration
public struct FIFOCacheConfiguration: CacheConfiguration {
    public let maxMemory: Int
    
    public init(maxMemory: Int) {
        self.maxMemory = maxMemory
    }
    
    public func validate() throws {
        guard maxMemory > 0 else {
            throw ConfigurationError.invalidValue("maxMemory must be positive")
        }
    }
    
    public func estimatedOverhead() -> Float {
        return 0.05 // 5% overhead for FIFO
    }
    
    public func optimalEvictionPolicy() -> EvictionPolicy {
        return .fifo
    }
    
    public func memoryBudget() -> Int {
        return maxMemory
    }
    
    public func evictionPolicy() -> EvictionPolicy {
        return .fifo
    }
}
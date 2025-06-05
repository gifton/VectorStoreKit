// VectorStoreKit: LFU Cache Configuration
//
// Configuration for Least Frequently Used cache

import Foundation

/// LFU cache configuration
public struct LFUCacheConfiguration: CacheConfiguration {
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
        return 0.15 // 15% overhead for frequency tracking
    }
    
    public func optimalEvictionPolicy() -> EvictionPolicy {
        return .lfu
    }
    
    public func memoryBudget() -> Int {
        return maxMemory
    }
    
    public func evictionPolicy() -> EvictionPolicy {
        return .lfu
    }
}
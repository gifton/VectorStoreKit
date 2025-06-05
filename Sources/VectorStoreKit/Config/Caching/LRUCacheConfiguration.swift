// VectorStoreKit: LRU Cache Configuration
//
// Configuration for Least Recently Used cache

import Foundation

/// LRU cache configuration
public struct LRUCacheConfiguration: CacheConfiguration {
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
        return 0.1 // 10% overhead for LRU tracking
    }
    
    public func optimalEvictionPolicy() -> EvictionPolicy {
        return .lru
    }
    
    public func memoryBudget() -> Int {
        return maxMemory
    }
    
    public func evictionPolicy() -> EvictionPolicy {
        return .lru
    }
}
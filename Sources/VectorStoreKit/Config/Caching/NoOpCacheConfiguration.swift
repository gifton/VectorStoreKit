// VectorStoreKit: NoOp Cache Configuration
//
// Configuration for no-operation cache

import Foundation

/// No-op cache configuration
public struct NoOpCacheConfiguration: CacheConfiguration {
    public var maxMemory: Int = 0
    
    public init() {}
    
    public func validate() throws {
        // No validation needed for no-op cache
    }
    
    public func estimatedOverhead() -> Float {
        return 0.0
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
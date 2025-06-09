// VectorStoreKit: No-Operation Cache Implementation
//
// A cache implementation that doesn't actually cache anything - useful for testing and benchmarking

import Foundation
import simd

// MARK: - NoOpVectorCache

/// A no-operation cache that doesn't actually cache anything
/// Useful for testing, benchmarking, and as a baseline for performance comparisons
public actor NoOpVectorCache<Vector: SIMD & Sendable>: VectorCache where Vector.Scalar: BinaryFloatingPoint {
    public typealias Configuration = NoOpCacheConfiguration
    public typealias Statistics = BasicCacheStatistics
    
    public let configuration: Configuration
    
    private let performanceAnalyzer = CachePerformanceAnalyzer<Vector>()
    
    public init() {
        self.configuration = NoOpCacheConfiguration()
    }
    
    // MARK: - Core Properties
    
    public var count: Int { 0 }
    public var size: Int { 0 }
    public var memoryUsage: Int { 0 }
    public var hitRate: Float { 0.0 }
    
    // MARK: - Core Operations
    
    public func get(id: VectorID) async -> Vector? {
        // Always returns nil (cache miss)
        nil
    }
    
    public func set(id: VectorID, vector: Vector, priority: CachePriority) async {
        // No-op: doesn't store anything
    }
    
    public func remove(id: VectorID) async {
        // No-op: nothing to remove
    }
    
    public func clear() async {
        // No-op: nothing to clear
    }
    
    public func contains(id: VectorID) async -> Bool {
        // Always returns false
        false
    }
    
    // MARK: - Advanced Operations
    
    public func preload(ids: [VectorID]) async {
        // No-op: can't preload into a no-op cache
    }
    
    public func prefetch(_ predictions: [VectorID: Float]) async {
        // No-op: can't prefetch into a no-op cache
        // However, we can still analyze the predictions for testing purposes
    }
    
    public func optimize() async {
        // No-op: nothing to optimize
    }
    
    public func statistics() async -> Statistics {
        BasicCacheStatistics(
            hitCount: 0,
            missCount: 0,
            evictionCount: 0,
            totalAccessTime: 0,
            memoryUsage: 0
        )
    }
    
    public func performanceAnalysis() async -> CachePerformanceAnalysis {
        await performanceAnalyzer.analyze(
            currentHitRate: 0.0,
            currentMemoryUsage: 0,
            maxMemory: 0
        )
    }
}

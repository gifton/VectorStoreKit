// VectorStoreKit: HNSW Index Configuration
//
// Configuration for Hierarchical Navigable Small World (HNSW) index

import Foundation

/// HNSW-specific configuration parameters
public struct HNSWConfiguration: IndexConfiguration {
    /// Maximum number of bidirectional links for each node during construction
    /// Higher values improve recall but increase memory usage and construction time
    /// **Typical range:** 8-64, **Default:** 16
    public let maxConnections: Int
    
    /// Size of dynamic candidate list during construction
    /// Controls the quality vs speed tradeoff during index building
    /// **Typical range:** 100-800, **Default:** 200
    public let efConstruction: Int
    
    /// Maximum layer level multiplier for probabilistic layer assignment
    /// Controls the height of the hierarchical structure
    /// **Formula:** layer = floor(-ln(uniform()) * mL)
    /// **Typical range:** 1/ln(2) to 1/ln(4), **Default:** 1/ln(2)
    public let levelMultiplier: Float
    
    /// Distance metric for similarity computation
    /// Determines how vector similarity is calculated
    public let distanceMetric: DistanceMetric
    
    /// Whether to use adaptive parameter tuning during construction
    /// Enables ML-driven optimization of index parameters
    public let useAdaptiveTuning: Bool
    
    /// Maximum number of nodes before triggering optimization
    /// Controls when to rebalance the index structure
    public let optimizationThreshold: Int
    
    /// Whether to enable comprehensive analytics tracking
    /// Impacts performance but provides detailed insights
    public let enableAnalytics: Bool
    
    public init(
        maxConnections: Int = 16,
        efConstruction: Int = 200,
        levelMultiplier: Float = 1.0 / log(2.0),
        distanceMetric: DistanceMetric = .euclidean,
        useAdaptiveTuning: Bool = true,
        optimizationThreshold: Int = 100_000,
        enableAnalytics: Bool = true
    ) {
        self.maxConnections = maxConnections
        self.efConstruction = efConstruction
        self.levelMultiplier = levelMultiplier
        self.distanceMetric = distanceMetric
        self.useAdaptiveTuning = useAdaptiveTuning
        self.optimizationThreshold = optimizationThreshold
        self.enableAnalytics = enableAnalytics
    }
    
    public func validate() throws {
        guard maxConnections > 0 && maxConnections <= 256 else {
            throw HNSWError.invalidConfiguration("maxConnections must be between 1 and 256")
        }
        guard efConstruction > maxConnections else {
            throw HNSWError.invalidConfiguration("efConstruction must be greater than maxConnections")
        }
        guard levelMultiplier > 0 && levelMultiplier <= 2.0 else {
            throw HNSWError.invalidConfiguration("levelMultiplier must be between 0 and 2.0")
        }
        guard optimizationThreshold > 1000 else {
            throw HNSWError.invalidConfiguration("optimizationThreshold must be at least 1000")
        }
    }
    
    public func estimatedMemoryUsage(for vectorCount: Int) -> Int {
        // Base memory per node: vector + metadata + connections
        let baseMemory = 512 + 256 // Estimated vector + metadata size
        let connectionMemory = maxConnections * MemoryLayout<String>.size
        let avgLayers = Int(1.0 / levelMultiplier) + 1
        
        return vectorCount * (baseMemory + connectionMemory * avgLayers)
    }
    
    public func computationalComplexity() -> ComputationalComplexity {
        return .logarithmic // O(log n) search, O(n log n) construction
    }
}
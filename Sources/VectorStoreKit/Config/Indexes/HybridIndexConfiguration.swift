// VectorStoreKit: Hybrid Index Configuration
//
// Configuration for Hybrid Index combining IVF and Learned approaches

import Foundation

/// Configuration for Hybrid Index
public struct HybridIndexConfiguration: IndexConfiguration {
    public let dimensions: Int
    public let ivfConfig: IVFConfiguration
    public let learnedConfig: LearnedIndexConfiguration
    public let routingStrategy: RoutingStrategy
    public let adaptiveThreshold: Float
    
    public enum RoutingStrategy: Sendable, Codable {
        case fixed(ivfWeight: Float) // 0.0 = all learned, 1.0 = all IVF
        case adaptive // Automatically choose based on query
        case ensemble // Use both and merge results
        case hierarchical // Use learned for coarse, IVF for fine
    }
    
    public init(
        dimensions: Int,
        ivfConfig: IVFConfiguration? = nil,
        learnedConfig: LearnedIndexConfiguration? = nil,
        routingStrategy: RoutingStrategy = .adaptive,
        adaptiveThreshold: Float = 0.5
    ) {
        self.dimensions = dimensions
        self.ivfConfig = ivfConfig ?? IVFConfiguration(
            dimensions: dimensions,
            numberOfCentroids: 256,
            numberOfProbes: 10
        )
        self.learnedConfig = learnedConfig ?? LearnedIndexConfiguration(
            dimensions: dimensions,
            modelArchitecture: .mlp(hiddenSizes: [64, 32]),
            bucketSize: 200
        )
        self.routingStrategy = routingStrategy
        self.adaptiveThreshold = adaptiveThreshold
    }
    
    public func validate() throws {
        guard dimensions > 0 else {
            throw HybridIndexError.invalidDimensions(dimensions)
        }
        try ivfConfig.validate()
        try learnedConfig.validate()
    }
    
    public func estimatedMemoryUsage(for vectorCount: Int) -> Int {
        // Combined memory usage of both indexes
        return ivfConfig.estimatedMemoryUsage(for: vectorCount) +
               learnedConfig.estimatedMemoryUsage(for: vectorCount)
    }
    
    public func computationalComplexity() -> ComputationalComplexity {
        // Best case is logarithmic (learned), worst case is linearithmic (IVF)
        return .linearithmic
    }
}
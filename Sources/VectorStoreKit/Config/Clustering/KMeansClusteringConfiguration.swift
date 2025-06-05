// VectorStoreKit: K-means Clustering Configuration
//
// Configuration for K-means clustering algorithm

import Foundation

/// Configuration for K-means clustering
public struct KMeansClusteringConfiguration: Sendable, Codable {
    public let maxIterations: Int
    public let tolerance: Float
    public let initMethod: InitializationMethod
    public let seed: UInt64?
    public let useMetalAcceleration: Bool
    
    public enum InitializationMethod: Sendable, Codable {
        case random
        case kMeansPlusPlus
        case custom([[Float]])
    }
    
    public init(
        maxIterations: Int = 100,
        tolerance: Float = 1e-4,
        initMethod: InitializationMethod = .kMeansPlusPlus,
        seed: UInt64? = nil,
        useMetalAcceleration: Bool = true
    ) {
        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.initMethod = initMethod
        self.seed = seed
        self.useMetalAcceleration = useMetalAcceleration
    }
    
    public static let `default` = KMeansClusteringConfiguration()
    
    public static let fast = KMeansClusteringConfiguration(
        maxIterations: 50,
        tolerance: 1e-3,
        initMethod: .random,
        useMetalAcceleration: true
    )
    
    public static let accurate = KMeansClusteringConfiguration(
        maxIterations: 200,
        tolerance: 1e-5,
        initMethod: .kMeansPlusPlus,
        useMetalAcceleration: true
    )
}
// VectorStoreKit: K-means Configuration
//
// Configuration for K-means clustering algorithm

import Foundation

/// Configuration for K-means clustering
public struct KMeansConfiguration: Codable, Sendable {
    public let maxIterations: Int
    public let convergenceThreshold: Float
    public let initMethod: InitializationMethod
    public let useMiniPatch: Bool
    public let miniBatchSize: Int
    
    public enum InitializationMethod: String, Codable, Sendable {
        case random = "random"
        case kMeansPlusPlus = "kmeans++"
        case deterministic = "deterministic"
    }
    
    public init(
        maxIterations: Int = 25,
        convergenceThreshold: Float = 0.001,
        initMethod: InitializationMethod = .kMeansPlusPlus,
        useMiniPatch: Bool = false,
        miniBatchSize: Int = 256
    ) {
        self.maxIterations = maxIterations
        self.convergenceThreshold = convergenceThreshold
        self.initMethod = initMethod
        self.useMiniPatch = useMiniPatch
        self.miniBatchSize = miniBatchSize
    }
    
    /// Default configuration
    public static let `default` = KMeansConfiguration()
    
    /// Fast configuration for approximate results
    public static let fast = KMeansConfiguration(
        maxIterations: 10,
        convergenceThreshold: 0.01,
        initMethod: .random,
        useMiniPatch: true,
        miniBatchSize: 512
    )
    
    /// High quality configuration for better accuracy
    public static let highQuality = KMeansConfiguration(
        maxIterations: 50,
        convergenceThreshold: 0.0001,
        initMethod: .kMeansPlusPlus,
        useMiniPatch: false,
        miniBatchSize: 256
    )
}
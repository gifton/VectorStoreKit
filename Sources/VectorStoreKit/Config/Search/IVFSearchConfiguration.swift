// VectorStoreKit: IVF Search Configuration
//
// Configuration for IVF search operations

import Foundation

/// Configuration for IVF search
public struct IVFSearchConfiguration: Sendable {
    public let multiProbeConfig: MultiProbeConfiguration?
    public let adaptiveConfig: AdaptiveSearchConfiguration?
    public let reranking: RerankingStrategy
    public let useGPUAcceleration: Bool
    
    public init(
        multiProbeConfig: MultiProbeConfiguration? = MultiProbeConfiguration.default,
        adaptiveConfig: AdaptiveSearchConfiguration? = AdaptiveSearchConfiguration.default,
        reranking: RerankingStrategy = .none,
        useGPUAcceleration: Bool = true
    ) {
        self.multiProbeConfig = multiProbeConfig
        self.adaptiveConfig = adaptiveConfig
        self.reranking = reranking
        self.useGPUAcceleration = useGPUAcceleration
    }
}

/// Multi-probe configuration for improved recall
public struct MultiProbeConfiguration: Sendable {
    public let baseProbes: Int
    public let expansionFactor: Float
    public let maxProbes: Int
    public let perturbationStrategy: PerturbationStrategy
    
    public enum PerturbationStrategy: Sendable {
        case adjacent           // Probe adjacent Voronoi cells
        case hierarchical      // Use hierarchical structure
        case learned(model: String)  // ML-based probe selection
        case adaptive         // Adapt based on query difficulty
    }
    
    public init(
        baseProbes: Int = 10,
        expansionFactor: Float = 1.5,
        maxProbes: Int = 100,
        perturbationStrategy: PerturbationStrategy = .adjacent
    ) {
        self.baseProbes = baseProbes
        self.expansionFactor = expansionFactor
        self.maxProbes = maxProbes
        self.perturbationStrategy = perturbationStrategy
    }
    
    public static let `default` = MultiProbeConfiguration()
    
    public static let aggressive = MultiProbeConfiguration(
        baseProbes: 20,
        expansionFactor: 2.0,
        maxProbes: 200,
        perturbationStrategy: .hierarchical
    )
}

/// Adaptive search configuration
public struct AdaptiveSearchConfiguration: Sendable {
    public let confidenceThreshold: Float
    public let earlyTermination: Bool
    public let dynamicProbing: Bool
    public let queryAnalysis: QueryAnalysisConfig
    
    public struct QueryAnalysisConfig: Sendable {
        public let analyzeDistribution: Bool
        public let analyzeDifficulty: Bool
        public let useHistoricalData: Bool
        
        public init(
            analyzeDistribution: Bool = true,
            analyzeDifficulty: Bool = true,
            useHistoricalData: Bool = false
        ) {
            self.analyzeDistribution = analyzeDistribution
            self.analyzeDifficulty = analyzeDifficulty
            self.useHistoricalData = useHistoricalData
        }
    }
    
    public init(
        confidenceThreshold: Float = 0.95,
        earlyTermination: Bool = true,
        dynamicProbing: Bool = true,
        queryAnalysis: QueryAnalysisConfig = QueryAnalysisConfig()
    ) {
        self.confidenceThreshold = confidenceThreshold
        self.earlyTermination = earlyTermination
        self.dynamicProbing = dynamicProbing
        self.queryAnalysis = queryAnalysis
    }
    
    public static let `default` = AdaptiveSearchConfiguration()
}

/// Reranking strategies for search results
public enum RerankingStrategy: Sendable {
    case none
    case exact(top: Int)           // Exact reranking of top results
    case cascade(stages: [Int])    // Multi-stage cascade
    case learned(model: String)    // ML-based reranking
}
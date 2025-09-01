// VectorStoreKit: Advanced Search Types
//
// Sophisticated search result types with research-grade metrics and analysis

import Foundation
import simd

// MARK: - Advanced Search Result

/// Comprehensive search result with detailed similarity metrics and provenance
public struct SearchResult<Metadata: Codable & Sendable>: Codable, Sendable {
    
    // MARK: - Core Properties
    
    /// Vector identifier
    public let id: VectorID
    
    /// Primary similarity distance
    public let distance: Distance
    
    /// Associated metadata
    public let metadata: Metadata
    
    /// Storage tier where vector was found
    public let tier: StorageTier
    
    /// Detailed similarity analysis
    public let similarityAnalysis: SimilarityAnalysis
    
    /// Search provenance for research and debugging
    public let provenance: SearchProvenance
    
    /// Confidence score for this result
    public let confidence: Float
    
    // MARK: - Initialization
    
    public init(
        id: VectorID,
        distance: Distance,
        metadata: Metadata,
        tier: StorageTier,
        similarityAnalysis: SimilarityAnalysis,
        provenance: SearchProvenance,
        confidence: Float = 1.0
    ) {
        self.id = id
        self.distance = distance
        self.metadata = metadata
        self.tier = tier
        self.similarityAnalysis = similarityAnalysis
        self.provenance = provenance
        self.confidence = confidence
    }
}

// MARK: - Similarity Analysis

/// Comprehensive similarity analysis for research applications
public struct SimilarityAnalysis: Codable, Sendable {
    
    /// Primary distance metric used
    public let primaryMetric: DistanceMetric
    
    /// Alternative distance calculations for comparison
    public let alternativeDistances: [DistanceMetric: Distance]
    
    /// Dimensional contribution analysis
    public let dimensionalContributions: [Float]
    
    /// Similarity confidence interval
    public let confidenceInterval: ClosedRange<Float>
    
    /// Angular similarity (cosine-based)
    public let angularSimilarity: Float
    
    /// Magnitude similarity
    public let magnitudeSimilarity: Float
    
    /// Geometric interpretation
    public let geometricProperties: GeometricProperties
    
    public init(
        primaryMetric: DistanceMetric,
        alternativeDistances: [DistanceMetric: Distance] = [:],
        dimensionalContributions: [Float] = [],
        confidenceInterval: ClosedRange<Float> = 0.0...1.0,
        angularSimilarity: Float,
        magnitudeSimilarity: Float,
        geometricProperties: GeometricProperties
    ) {
        self.primaryMetric = primaryMetric
        self.alternativeDistances = alternativeDistances
        self.dimensionalContributions = dimensionalContributions
        self.confidenceInterval = confidenceInterval
        self.angularSimilarity = angularSimilarity
        self.magnitudeSimilarity = magnitudeSimilarity
        self.geometricProperties = geometricProperties
    }
}

// MARK: - Distance Metrics

/// Advanced distance metrics for research applications
public enum DistanceMetric: String, Codable, Sendable, CaseIterable {
    case euclidean = "euclidean"
    case cosine = "cosine"
    case dotProduct = "dotProduct"     // Dot product similarity
    case manhattan = "manhattan"
    case chebyshev = "chebyshev"
    case minkowski = "minkowski"
    case hamming = "hamming"
    case jaccard = "jaccard"
    case mahalanobis = "mahalanobis"
    case earth_mover = "earth_mover"
    case learned = "learned"           // ML-learned distance function
    case adaptive = "adaptive"         // Context-adaptive distance
    
    /// Computational complexity for this metric
    public var complexity: ComputationalComplexity {
        switch self {
        case .euclidean, .cosine, .dotProduct, .manhattan:
            return .linear
        case .chebyshev, .hamming, .jaccard:
            return .linear
        case .minkowski, .mahalanobis:
            return .quadratic
        case .earth_mover:
            return .cubic
        case .learned, .adaptive:
            return .variable
        }
    }
    
    /// Whether this metric preserves triangle inequality
    public var isMetric: Bool {
        switch self {
        case .euclidean, .manhattan, .chebyshev, .minkowski, .hamming, .mahalanobis, .earth_mover:
            return true
        case .cosine, .dotProduct, .jaccard, .learned, .adaptive:
            return false
        }
    }
}


// MARK: - Geometric Properties

/// Geometric properties of the similarity relationship
public struct GeometricProperties: Codable, Sendable {
    
    /// Whether vectors are in the same orthant
    public let sameOrthant: Bool
    
    /// Angle between vectors (in radians)
    public let angle: Float
    
    /// Relative magnitude ratio
    public let magnitudeRatio: Float
    
    /// Geometric center point
    public let centerPoint: [Float]
    
    /// Bounding box containing both vectors
    public let boundingBox: BoundingBox
    
    /// Topological relationship
    public let topology: Topology
}

public struct BoundingBox: Codable, Sendable {
    public let min: [Float]
    public let max: [Float]
    public let volume: Float
}

public enum TopologicalRelation: String, Codable, Sendable {
    case coincident = "coincident"     // Same point
    case collinear = "collinear"       // On same line
    case coplanar = "coplanar"         // In same plane
    case orthogonal = "orthogonal"     // Perpendicular
    case acute = "acute"               // Acute angle
    case obtuse = "obtuse"             // Obtuse angle
    case general = "general"           // General position
}

/// Simplified topology type for IVF search
public enum Topology: String, Codable, Sendable {
    case dense = "dense"
    case sparse = "sparse"
    case general = "general"
}

// MARK: - Search Provenance

/// Detailed provenance information for search results
public struct SearchProvenance: Codable, Sendable {
    
    /// Which index algorithm found this result
    public let indexAlgorithm: String
    
    /// Search path taken through the index
    public let searchPath: SearchPath
    
    /// Computational cost metrics
    public let computationalCost: ComputationalCost
    
    /// Approximation quality metrics
    public let approximationQuality: ApproximationQuality
    
    /// Timestamp when result was found
    public let timestamp: Timestamp
    
    /// Search strategy used
    public let strategy: SearchStrategy
}

/// Path taken through the search index
public struct SearchPath: Codable, Sendable {
    
    /// Nodes visited during search
    public let nodesVisited: [String]
    
    /// Pruning decisions made
    public let pruningDecisions: [PruningDecision]
    
    /// Backtracking events
    public let backtrackingEvents: [BacktrackingEvent]
    
    /// Final convergence point
    public let convergencePoint: String
}

public struct PruningDecision: Codable, Sendable {
    public let nodeId: String
    public let reason: PruningReason
    public let savedComputations: Int
}

public enum PruningReason: String, Codable, Sendable {
    case distanceBound = "distance_bound"
    case heuristic = "heuristic"
    case learned = "learned"
    case timeout = "timeout"
}

public struct BacktrackingEvent: Codable, Sendable {
    public let fromNode: String
    public let toNode: String
    public let reason: String
    public let cost: Int
}

/// Computational cost tracking
public struct ComputationalCost: Codable, Sendable {
    
    /// Number of distance calculations performed
    public let distanceCalculations: Int
    
    /// CPU cycles consumed (estimate)
    public let cpuCycles: UInt64
    
    /// Memory accessed (bytes)
    public let memoryAccessed: Int
    
    /// Cache hits/misses
    public let cacheStatistics: ComputeCacheStatistics
    
    /// GPU/Metal computations used
    public let gpuComputations: Int
    
    /// Total wall-clock time (nanoseconds)
    public let wallClockTime: UInt64
}

public struct ComputeCacheStatistics: Codable, Sendable {
    public let hits: Int
    public let misses: Int
    public let hitRate: Float
    
    public init(hits: Int, misses: Int, hitRate: Float) {
        self.hits = hits
        self.misses = misses
        self.hitRate = hitRate
    }
}

/// Quality metrics for approximate search results
public struct ApproximationQuality: Codable, Sendable {
    
    /// Estimated recall (what fraction of true neighbors we found)
    public let estimatedRecall: Float
    
    /// Distance error bounds
    public let distanceErrorBounds: ClosedRange<Float>
    
    /// Confidence in the approximation
    public let confidence: Float
    
    /// Whether exact computation was used
    public let isExact: Bool
    
    /// Quality guarantees provided
    public let qualityGuarantees: [QualityGuarantee]
}

public struct QualityGuarantee: Codable, Sendable {
    public let type: GuaranteeType
    public let value: Float
    public let confidence: Float
}

public enum GuaranteeType: String, Codable, Sendable {
    case recall = "recall"
    case precision = "precision"
    case distanceError = "distance_error"
    case ranking = "ranking"
}

// MARK: - Search Strategy

/// Search strategies with different performance and accuracy trade-offs
///
/// SearchStrategy determines how the vector search is performed, allowing
/// you to optimize for speed, accuracy, or adapt to your specific use case.
public enum SearchStrategy: String, Codable, Sendable, CaseIterable {
    /// Exact k-nearest neighbor search
    ///
    /// Guarantees finding the true nearest neighbors by comparing against
    /// all vectors. Slowest but most accurate option.
    /// - Complexity: O(n) where n is the number of vectors
    /// - Use when: Accuracy is critical and dataset is small
    case exact = "exact"
    
    /// Fast approximate nearest neighbor search
    ///
    /// Uses index structures (HNSW, IVF, etc.) to find approximate neighbors
    /// quickly. May miss some true neighbors but is much faster.
    /// - Complexity: O(log n) typical
    /// - Use when: Speed is important and >90% recall is acceptable
    case approximate = "approximate"
    
    /// Adaptive strategy that chooses based on query characteristics
    ///
    /// Automatically selects between exact and approximate search based on
    /// factors like query complexity, dataset size, and available resources.
    /// - Complexity: Variable
    /// - Use when: You want the system to optimize automatically
    case adaptive = "adaptive"
    
    /// Machine learning-guided search
    ///
    /// Uses trained models to optimize search paths and improve accuracy
    /// while maintaining speed. Requires training on your data.
    /// - Complexity: O(log n) with better constants
    /// - Use when: You have query patterns and can train the model
    case learned = "learned"
    
    /// Combines multiple search strategies
    ///
    /// Runs multiple strategies in parallel or sequence and merges results
    /// for improved recall at the cost of some performance.
    /// - Complexity: Depends on strategies used
    /// - Use when: Maximum recall is needed
    case hybrid = "hybrid"
    
    /// Progressive search that improves results over time
    ///
    /// Returns initial results quickly and continues refining them,
    /// useful for interactive applications.
    /// - Complexity: O(log n) initial, O(n) worst case
    /// - Use when: You need fast initial results with refinement
    case anytime = "anytime"
    
    /// Cross-modal search across different data types
    ///
    /// Specialized strategy for searching across different modalities
    /// (text, image, audio) with appropriate transformations.
    /// - Complexity: Variable based on modalities
    /// - Use when: Searching across different data types
    case multimodal = "multimodal"
    
    /// Expected quality characteristics
    public var characteristics: StrategyCharacteristics {
        switch self {
        case .exact:
            return StrategyCharacteristics(
                speed: .slow,
                accuracy: .perfect,
                resourceUsage: .high,
                scalability: .poor
            )
        case .approximate:
            return StrategyCharacteristics(
                speed: .fast,
                accuracy: .good,
                resourceUsage: .low,
                scalability: .excellent
            )
        case .adaptive:
            return StrategyCharacteristics(
                speed: .variable,
                accuracy: .variable,
                resourceUsage: .variable,
                scalability: .good
            )
        case .learned:
            return StrategyCharacteristics(
                speed: .fast,
                accuracy: .excellent,
                resourceUsage: .medium,
                scalability: .excellent
            )
        case .hybrid:
            return StrategyCharacteristics(
                speed: .balanced,
                accuracy: .excellent,
                resourceUsage: .medium,
                scalability: .good
            )
        case .anytime:
            return StrategyCharacteristics(
                speed: .progressive,
                accuracy: .improving,
                resourceUsage: .adaptive,
                scalability: .good
            )
        case .multimodal:
            return StrategyCharacteristics(
                speed: .medium,
                accuracy: .contextual,
                resourceUsage: .high,
                scalability: .medium
            )
        }
    }
}

public struct StrategyCharacteristics: Sendable {
    public let speed: PerformanceLevel
    public let accuracy: AccuracyLevel
    public let resourceUsage: ResourceLevel
    public let scalability: ScalabilityLevel
}

public enum PerformanceLevel: Sendable {
    case slow, medium, fast, progressive, variable, balanced
}

public enum AccuracyLevel: Sendable {
    case poor, good, excellent, perfect, improving, variable, contextual
}

public enum ResourceLevel: Sendable {
    case low, medium, high, adaptive, variable
}


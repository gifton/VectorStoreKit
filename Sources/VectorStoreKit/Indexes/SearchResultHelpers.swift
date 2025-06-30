// VectorStoreKit: Search Result Helpers
//
// Helper functions to create search results for the new indexes

import Foundation

/// Creates a simplified search result for index implementations
public func createSimpleSearchResult<Metadata: Codable & Sendable>(
    id: String,
    distance: Float,
    metadata: Metadata,
    indexAlgorithm: String = "Unknown"
) -> SearchResult<Metadata> {
    let similarityAnalysis = SimilarityAnalysis(
        primaryMetric: .euclidean,
        angularSimilarity: max(0, 1.0 - (distance / 2.0)),
        magnitudeSimilarity: 1.0,
        geometricProperties: GeometricProperties(
            sameOrthant: true,
            angle: 0,
            magnitudeRatio: 1.0,
            centerPoint: [],
            boundingBox: BoundingBox(
                min: [],
                max: [],
                volume: 0
            ),
            topology: .general
        )
    )
    
    let provenance = SearchProvenance(
        indexAlgorithm: indexAlgorithm,
        searchPath: SearchPath(
            nodesVisited: [],
            pruningDecisions: [],
            backtrackingEvents: [],
            convergencePoint: ""
        ),
        computationalCost: ComputationalCost(
            distanceCalculations: 1,
            cpuCycles: 1000,
            memoryAccessed: 100,
            cacheStatistics: ComputeCacheStatistics(
                hits: 0,
                misses: 1,
                hitRate: 0.0
            ),
            gpuComputations: 0,
            wallClockTime: 1000000 // 1ms in nanoseconds
        ),
        approximationQuality: ApproximationQuality(
            estimatedRecall: 0.95,
            distanceErrorBounds: 0.0...0.05,
            confidence: 0.95,
            isExact: false,
            qualityGuarantees: []
        ),
        timestamp: Timestamp(),
        strategy: .adaptive
    )
    
    return SearchResult(
        id: id,
        distance: distance,
        metadata: metadata,
        tier: .memory,
        similarityAnalysis: similarityAnalysis,
        provenance: provenance,
        confidence: max(0, min(1, 1.0 - (distance / 10.0)))
    )
}
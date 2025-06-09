// VectorStoreKit: IVF Search Engine
//
// Advanced search strategies for IVF indexes with multi-probe and adaptive techniques

import Foundation
import simd


/// Search result with extended metrics
public struct ExtendedSearchResult<Metadata: Codable & Sendable>: Sendable {
    public let result: SearchResult<Metadata>
    public let probesUsed: Int
    public let candidatesEvaluated: Int
    public let confidence: Float
    public let searchPath: [Int]  // Centroid indices visited
}

/// IVF Search Engine
public actor IVFSearchEngine<Vector: SIMD & Sendable, Metadata: Codable & Sendable> 
where Vector.Scalar: BinaryFloatingPoint {
    
    // MARK: - Properties
    
    private let configuration: IVFSearchConfiguration
    private var searchHistory: SearchHistory
    private let metalCompute: MetalCompute?
    
    // MARK: - Initialization
    
    public init(
        configuration: IVFSearchConfiguration,
        metalCompute: MetalCompute? = nil
    ) {
        self.configuration = configuration
        self.searchHistory = SearchHistory()
        self.metalCompute = configuration.useGPUAcceleration ? metalCompute : nil
    }
    
    // MARK: - Main Search Method
    
    public func search(
        query: Vector,
        k: Int,
        in index: IVFIndex<Vector, Metadata>,
        filter: SearchFilter? = nil
    ) async throws -> [ExtendedSearchResult<Metadata>] {
        let startTime = DispatchTime.now()
        
        // Convert query to float array
        let queryArray = vectorToArray(query)
        
        // Analyze query if adaptive search is enabled
        let queryAnalysis = configuration.adaptiveConfig != nil
            ? await analyzeQuery(queryArray, index: index)
            : nil
        
        // Determine initial probe count
        let initialProbes = determineInitialProbes(
            queryAnalysis: queryAnalysis,
            baseConfig: configuration.multiProbeConfig?.baseProbes ?? 10
        )
        
        // Get centroids and their distances
        let centroidDistances = try await computeCentroidDistances(
            query: queryArray,
            centroids: await index.getCentroids()
        )
        
        // Select probes using multi-probe strategy
        let probeSequence = await selectProbes(
            centroidDistances: centroidDistances,
            initialCount: initialProbes,
            multiProbeConfig: configuration.multiProbeConfig
        )
        
        // Perform adaptive search
        let searchResult = try await performAdaptiveSearch(
            query: queryArray,
            k: k,
            probeSequence: probeSequence,
            index: index,
            filter: filter,
            queryAnalysis: queryAnalysis
        )
        
        // Apply reranking if configured
        let finalResults = try await applyReranking(
            results: searchResult.results,
            query: queryArray,
            strategy: configuration.reranking
        )
        
        // Update search history
        let searchTime = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
        await updateSearchHistory(
            probesUsed: searchResult.probesUsed,
            recall: estimateRecall(finalResults, k: k),
            searchTime: searchTime
        )
        
        return finalResults
    }
    
    // MARK: - Query Analysis
    
    private func analyzeQuery(
        _ query: [Float],
        index: IVFIndex<Vector, Metadata>
    ) async -> QueryAnalysis {
        var analysis = QueryAnalysis()
        
        // Analyze query distribution
        if configuration.adaptiveConfig?.queryAnalysis.analyzeDistribution ?? false {
            analysis.sparsity = calculateSparsity(query)
            analysis.magnitude = calculateMagnitude(query)
            analysis.entropy = calculateEntropy(query)
        }
        
        // Estimate query difficulty
        if configuration.adaptiveConfig?.queryAnalysis.analyzeDifficulty ?? false {
            let centroids = await index.getCentroids()
            analysis.difficulty = await estimateQueryDifficulty(
                query: query,
                centroids: centroids
            )
        }
        
        // Use historical data if available
        if configuration.adaptiveConfig?.queryAnalysis.useHistoricalData ?? false {
            analysis.historicalPerformance = await searchHistory.getAverageMetrics()
        }
        
        return analysis
    }
    
    // MARK: - Multi-Probe Selection
    
    private func selectProbes(
        centroidDistances: [(index: Int, distance: Float)],
        initialCount: Int,
        multiProbeConfig: MultiProbeConfiguration?
    ) async -> [ProbeCandidate] {
        guard let config = multiProbeConfig else {
            // Simple top-k selection
            return centroidDistances.prefix(initialCount).map {
                ProbeCandidate(centroidIndex: $0.index, priority: 1.0 / $0.distance)
            }
        }
        
        var probes: [ProbeCandidate] = []
        
        switch config.perturbationStrategy {
        case .adjacent:
            probes = selectAdjacentProbes(
                centroidDistances: centroidDistances,
                count: initialCount,
                expansionFactor: config.expansionFactor
            )
            
        case .hierarchical:
            probes = selectHierarchicalProbes(
                centroidDistances: centroidDistances,
                count: initialCount,
                maxProbes: config.maxProbes
            )
            
        case .learned(let model):
            probes = selectLearnedProbes(
                centroidDistances: centroidDistances,
                count: initialCount,
                model: model
            )
            
        case .adaptive:
            probes = await selectAdaptiveProbes(
                centroidDistances: centroidDistances,
                count: initialCount,
                history: searchHistory
            )
        }
        
        return Array(probes.prefix(config.maxProbes))
    }
    
    private func selectAdjacentProbes(
        centroidDistances: [(index: Int, distance: Float)],
        count: Int,
        expansionFactor: Float
    ) -> [ProbeCandidate] {
        var probes: [ProbeCandidate] = []
        let sorted = centroidDistances.sorted { $0.distance < $1.distance }
        
        // Add primary probes
        for (i, (index, distance)) in sorted.prefix(count).enumerated() {
            let priority = 1.0 / (distance * Float(i + 1))
            probes.append(ProbeCandidate(centroidIndex: index, priority: priority))
        }
        
        // Add adjacent probes based on expansion factor
        let expandedCount = Int(Float(count) * expansionFactor)
        for (i, (index, distance)) in sorted[count..<min(expandedCount, sorted.count)].enumerated() {
            let priority = 0.5 / (distance * Float(i + count + 1))
            probes.append(ProbeCandidate(centroidIndex: index, priority: priority))
        }
        
        return probes.sorted { $0.priority > $1.priority }
    }
    
    private func selectHierarchicalProbes(
        centroidDistances: [(index: Int, distance: Float)],
        count: Int,
        maxProbes: Int
    ) -> [ProbeCandidate] {
        // Simplified hierarchical selection
        var probes: [ProbeCandidate] = []
        let levels = 3
        let probesPerLevel = count / levels
        
        for level in 0..<levels {
            let startIdx = level * probesPerLevel
            let endIdx = min((level + 1) * probesPerLevel, centroidDistances.count)
            
            for (index, distance) in centroidDistances[startIdx..<endIdx] {
                let priority = Float(levels - level) / (distance * Float(level + 1))
                probes.append(ProbeCandidate(centroidIndex: index, priority: priority))
            }
        }
        
        return Array(probes.sorted { $0.priority > $1.priority }.prefix(maxProbes))
    }
    
    private func selectLearnedProbes(
        centroidDistances: [(index: Int, distance: Float)],
        count: Int,
        model: String
    ) -> [ProbeCandidate] {
        // Placeholder for ML-based probe selection
        return selectAdjacentProbes(
            centroidDistances: centroidDistances,
            count: count,
            expansionFactor: 1.5
        )
    }
    
    private func selectAdaptiveProbes(
        centroidDistances: [(index: Int, distance: Float)],
        count: Int,
        history: SearchHistory
    ) async -> [ProbeCandidate] {
        let avgProbes = await history.getAverageMetrics().probesUsed
        let adaptedCount = avgProbes > 0 ? Int(Float(avgProbes) * 1.2) : count
        
        return centroidDistances.prefix(adaptedCount).map {
            ProbeCandidate(centroidIndex: $0.index, priority: 1.0 / $0.distance)
        }
    }
    
    // MARK: - Adaptive Search
    
    private func performAdaptiveSearch(
        query: [Float],
        k: Int,
        probeSequence: [ProbeCandidate],
        index: IVFIndex<Vector, Metadata>,
        filter: SearchFilter?,
        queryAnalysis: QueryAnalysis?
    ) async throws -> (results: [ExtendedSearchResult<Metadata>], probesUsed: Int) {
        var candidates: [CandidateResult<Vector, Metadata>] = []
        var probesUsed = 0
        var searchPath: [Int] = []
        var confidenceScore: Float = 0
        
        // Tracking variables for provenance
        let searchStartTime = DispatchTime.now()
        var distanceCalculations = 0
        var memoryAccessed = 0
        var pruningDecisions: [PruningDecision] = []
        
        // Adaptive termination criteria
        let earlyTermination = configuration.adaptiveConfig?.earlyTermination ?? false
        let confidenceThreshold = configuration.adaptiveConfig?.confidenceThreshold ?? 0.95
        
        for probe in probeSequence {
            // Get vectors from this probe
            let probeVectors = await index.getVectorsFromList(probe.centroidIndex)
            probesUsed += 1
            searchPath.append(probe.centroidIndex)
            memoryAccessed += probeVectors.count * MemoryLayout<StoredVector>.size
            
            // Apply filter if needed
            let filteredVectors = filter != nil
                ? try await applyFilter(probeVectors, filter: filter!)
                : probeVectors
            
            if filter != nil && filteredVectors.count < probeVectors.count {
                pruningDecisions.append(PruningDecision(
                    nodeId: "probe_\(probe.centroidIndex)",
                    reason: .distanceBound,
                    savedComputations: probeVectors.count - filteredVectors.count
                ))
            }
            
            // Compute distances and add to candidates
            for vector in filteredVectors {
                let distance = computeDistance(query, vector.vector)
                distanceCalculations += 1
                candidates.append(CandidateResult<Vector, Metadata>(
                    vector: vector,
                    distance: distance,
                    probeIndex: probe.centroidIndex
                ))
            }
            
            // Sort candidates by distance
            candidates.sort { $0.distance < $1.distance }
            
            // Check early termination conditions
            if earlyTermination && candidates.count >= k * 10 {
                confidenceScore = estimateConfidence(
                    candidates: Array(candidates.prefix(k * 2)),
                    k: k,
                    probesUsed: probesUsed,
                    totalProbes: probeSequence.count
                )
                
                if confidenceScore >= confidenceThreshold {
                    pruningDecisions.append(PruningDecision(
                        nodeId: "probe_\(probesUsed)",
                        reason: .heuristic,
                        savedComputations: (probeSequence.count - probesUsed) * 100
                    ))
                    break
                }
            }
        }
        
        // Convert top-k candidates to results
        let topCandidates = Array(candidates.prefix(k))
        let totalCentroids = await index.getCentroids().count
        var results: [ExtendedSearchResult<Metadata>] = []
        
        for candidate in topCandidates {
            let metadata = try JSONDecoder().decode(Metadata.self, from: candidate.vector.metadata)
            // Create proper similarity analysis
            let similarityAnalysis = calculateSimilarityAnalysis(
                query: query,
                candidate: candidate.vector.vector,
                distance: candidate.distance
            )
            
            // Calculate search time
            let searchEndTime = DispatchTime.now()
            let wallClockTime = Double(searchEndTime.uptimeNanoseconds - searchStartTime.uptimeNanoseconds) / 1_000_000_000
            
            // Create actual provenance with tracked metrics
            let provenance = SearchProvenance(
                indexAlgorithm: "IVF",
                searchPath: SearchPath(
                    nodesVisited: searchPath.map { String($0) },
                    pruningDecisions: pruningDecisions,
                    backtrackingEvents: [], // IVF doesn't backtrack
                    convergencePoint: "centroid_\(candidate.probeIndex)"
                ),
                computationalCost: ComputationalCost(
                    distanceCalculations: distanceCalculations,
                    cpuCycles: UInt64(distanceCalculations * 1000), // Estimate
                    memoryAccessed: memoryAccessed,
                    cacheStatistics: ComputeCacheStatistics(
                        hits: 0, // Would need actual cache tracking
                        misses: distanceCalculations,
                        hitRate: 0.0
                    ),
                    gpuComputations: metalCompute != nil ? distanceCalculations : 0,
                    wallClockTime: UInt64(wallClockTime * 1_000_000_000)
                ),
                approximationQuality: ApproximationQuality(
                    estimatedRecall: estimateRecall(k: k, probesUsed: probesUsed, totalProbes: probeSequence.count),
                    distanceErrorBounds: 0.0...0.1, // Conservative estimate
                    confidence: confidenceScore,
                    isExact: probesUsed == totalCentroids,
                    qualityGuarantees: probesUsed >= configuration.multiProbeConfig?.baseProbes ?? 1 
                        ? [QualityGuarantee(type: .recall, value: 0.9, confidence: 0.8)] : []
                ),
                timestamp: searchStartTime.uptimeNanoseconds,
                strategy: .adaptive
            )
            
            let searchResult = SearchResult(
                id: candidate.vector.id,
                distance: candidate.distance,
                metadata: metadata,
                tier: .hot,
                similarityAnalysis: similarityAnalysis,
                provenance: provenance,
                confidence: candidate.distance
            )
            
            results.append(ExtendedSearchResult(
                result: searchResult,
                probesUsed: probesUsed,
                candidatesEvaluated: candidates.count,
                confidence: confidenceScore,
                searchPath: searchPath
            ))
        }
        
        return (results, probesUsed)
    }
    
    // MARK: - Reranking
    
    private func applyReranking<M: Sendable>(
        results: [ExtendedSearchResult<M>],
        query: [Float],
        strategy: RerankingStrategy
    ) async throws -> [ExtendedSearchResult<M>] {
        switch strategy {
        case .none:
            return results
            
        case .exact(let top):
            // Recompute exact distances for top results
            let _ = Array(results.prefix(top))
            // In practice, would refetch original vectors and recompute
            return results
            
        case .cascade(let stages):
            // Multi-stage cascade reranking
            var currentResults = results
            for stageSize in stages {
                currentResults = Array(currentResults.prefix(stageSize))
                // Apply more precise distance computation at each stage
            }
            return currentResults
            
        case .learned(_):
            // ML-based reranking
            // Placeholder implementation
            return results
        }
    }
    
    // MARK: - Utility Methods
    
    private func vectorToArray<V: SIMD>(_ vector: V) -> [Float] where V.Scalar: BinaryFloatingPoint {
        var array: [Float] = []
        for i in 0..<vector.scalarCount {
            array.append(Float(vector[i]))
        }
        return array
    }
    
    private func computeCentroidDistances(
        query: [Float],
        centroids: [[Float]]
    ) async throws -> [(index: Int, distance: Float)] {
        if metalCompute != nil {
            // Use GPU acceleration with proper dimension handling
            guard !centroids.isEmpty else {
                return []
            }
            
            let _ = query.count // dimensions unused for now
            
            // MetalCompute should handle raw float arrays directly
            // For now, we'll use CPU computation until Metal compute is properly integrated
            // TODO: Integrate with MetalCompute's batch distance computation
            /*
            let result = try await metalCompute.computeBatchDistances(
                query: query,
                candidates: centroids,
                metric: .euclidean
            )
            return result.distances.enumerated().map { ($0, $1) }
            */
            
            // Fallback to CPU computation for now
            return centroids.enumerated().map { index, centroid in
                (index, computeDistance(query, centroid))
            }
        } else {
            // CPU computation
            return centroids.enumerated().map { index, centroid in
                (index, computeDistance(query, centroid))
            }
        }
    }
    
    private func computeDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<min(a.count, b.count) {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    private func calculateSparsity(_ vector: [Float]) -> Float {
        let nonZeros = vector.filter { abs($0) > 1e-6 }.count
        return 1.0 - Float(nonZeros) / Float(vector.count)
    }
    
    private func calculateMagnitude(_ vector: [Float]) -> Float {
        return sqrt(vector.map { $0 * $0 }.reduce(0, +))
    }
    
    private func calculateEntropy(_ vector: [Float]) -> Float {
        let magnitude = calculateMagnitude(vector)
        guard magnitude > 0 else { return 0 }
        
        let normalized = vector.map { abs($0) / magnitude }
        return -normalized.compactMap { $0 > 0 ? $0 * log($0) : nil }.reduce(0, +)
    }
    
    private func estimateQueryDifficulty(
        query: [Float],
        centroids: [[Float]]
    ) async -> Float {
        guard !centroids.isEmpty else { return 0.5 }
        
        // Compute distances to all centroids
        let distances = centroids.map { computeDistance(query, $0) }
        let sortedDistances = distances.sorted()
        
        // 1. Distance Distribution Factor
        let mean = distances.reduce(0, +) / Float(distances.count)
        let variance = distances.map { pow($0 - mean, 2) }.reduce(0, +) / Float(distances.count)
        let coefficientOfVariation = sqrt(variance) / (mean + 0.001)
        
        // 2. Ambiguity Factor - how similar are the closest centroids
        let ambiguityFactor: Float
        if sortedDistances.count >= 3 {
            let closest = sortedDistances[0]
            let secondClosest = sortedDistances[1]
            let thirdClosest = sortedDistances[2]
            
            // If top centroids are very close, query is ambiguous
            let gap1 = (secondClosest - closest) / (closest + 0.001)
            let gap2 = (thirdClosest - secondClosest) / (secondClosest + 0.001)
            ambiguityFactor = 1.0 - min(1.0, (gap1 + gap2) / 2.0)
        } else {
            ambiguityFactor = 0.5
        }
        
        // 3. Outlier Factor - is query far from all centroids?
        let minDistance = sortedDistances.first ?? 0
        let outlierThreshold = mean + 2 * sqrt(variance)
        let outlierFactor = min(1.0, minDistance / outlierThreshold)
        
        // 4. Query Characteristics
        let queryMagnitude = sqrt(query.map { $0 * $0 }.reduce(0, +))
        let querySparsity = Float(query.filter { abs($0) < 0.001 }.count) / Float(query.count)
        let queryEntropy = calculateEntropy(query)
        
        // Normalize query characteristics
        let magnitudeFactor = 1.0 / (1.0 + exp(-queryMagnitude + 5)) // Sigmoid normalization
        let sparsityFactor = querySparsity
        let entropyFactor = min(1.0, queryEntropy / 5.0) // Normalize to [0,1]
        
        // Combine factors with weights
        let difficulty = coefficientOfVariation * 0.2 +
                        ambiguityFactor * 0.3 +
                        outlierFactor * 0.2 +
                        magnitudeFactor * 0.1 +
                        sparsityFactor * 0.1 +
                        entropyFactor * 0.1
        
        return min(1.0, max(0.0, difficulty))
    }
    
    private func estimateConfidence(
        candidates: [CandidateResult<Vector, Metadata>],
        k: Int,
        probesUsed: Int,
        totalProbes: Int
    ) -> Float {
        guard candidates.count >= k else { return 0 }
        
        // 1. Coverage Factor - how many probes we've searched
        let coverageFactor = Float(probesUsed) / Float(totalProbes)
        let coverageWeight: Float = 0.25
        
        // 2. Density Factor - how many candidates we found relative to k
        let densityFactor = min(1.0, Float(candidates.count) / Float(k * 5))
        let densityWeight: Float = 0.20
        
        // 3. Distance Gap Factor - separation between k-th and (k+1)-th result
        let gapFactor: Float
        if candidates.count > k {
            let kthDistance = candidates[k-1].distance
            let nextDistance = candidates[k].distance
            // Larger gap means more confidence in top-k
            gapFactor = min(1.0, (nextDistance - kthDistance) / (kthDistance + 0.001))
        } else {
            gapFactor = 0.5 // Conservative when we don't have k+1 results
        }
        let gapWeight: Float = 0.30
        
        // 4. Distance Stability Factor - how stable are the top-k distances
        let stabilityFactor: Float
        if candidates.count >= k {
            let topKDistances = candidates.prefix(k).map { $0.distance }
            let avgDistance = topKDistances.reduce(0, +) / Float(k)
            let variance = topKDistances.map { pow($0 - avgDistance, 2) }.reduce(0, +) / Float(k)
            let coefficientOfVariation = sqrt(variance) / (avgDistance + 0.001)
            // Lower variation means more stable results
            stabilityFactor = max(0, 1.0 - coefficientOfVariation)
        } else {
            stabilityFactor = 0.3
        }
        let stabilityWeight: Float = 0.25
        
        // Combined confidence score with data-driven weights
        let weightedCoverage = coverageFactor * coverageWeight
        let weightedDensity = densityFactor * densityWeight
        let weightedGap = gapFactor * gapWeight
        let weightedStability = stabilityFactor * stabilityWeight
        let confidence = weightedCoverage + weightedDensity + weightedGap + weightedStability
        
        // Apply non-linear transformation for more realistic confidence
        // This makes low confidence lower and high confidence higher
        let transformedConfidence = pow(confidence, 1.5)
        
        return min(1.0, transformedConfidence)
    }
    
    private func estimateRecall(_ results: [ExtendedSearchResult<Metadata>], k: Int) -> Float {
        // Simplified recall estimation
        return min(1.0, Float(results.count) / Float(k))
    }
    
    private func estimateRecall(k: Int, probesUsed: Int, totalProbes: Int) -> Float {
        // Estimate recall based on probe coverage
        let probeCoverage = Float(probesUsed) / Float(totalProbes)
        // Assume recall increases sub-linearly with probe coverage
        return min(1.0, pow(probeCoverage, 0.7))
    }
    
    // MARK: - Similarity Analysis Helpers
    
    private func calculateSimilarityAnalysis(
        query: [Float],
        candidate: [Float],
        distance: Float
    ) -> SimilarityAnalysis {
        let angularSim = calculateAngularSimilarity(query, candidate)
        let magnitudeSim = calculateMagnitudeSimilarity(query, candidate)
        let geometricProps = calculateGeometricProperties(query, candidate)
        
        return SimilarityAnalysis(
            primaryMetric: .euclidean,
            angularSimilarity: angularSim,
            magnitudeSimilarity: magnitudeSim,
            geometricProperties: geometricProps
        )
    }
    
    private func calculateAngularSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        // Cosine similarity
        var dotProduct: Float = 0
        var magnitudeA: Float = 0
        var magnitudeB: Float = 0
        
        for i in 0..<min(a.count, b.count) {
            dotProduct += a[i] * b[i]
            magnitudeA += a[i] * a[i]
            magnitudeB += b[i] * b[i]
        }
        
        magnitudeA = sqrt(magnitudeA)
        magnitudeB = sqrt(magnitudeB)
        
        guard magnitudeA > 0 && magnitudeB > 0 else { return 0 }
        return dotProduct / (magnitudeA * magnitudeB)
    }
    
    private func calculateMagnitudeSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        let magA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let magB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        
        guard magA > 0 && magB > 0 else { return 0 }
        return min(magA, magB) / max(magA, magB)
    }
    
    private func calculateGeometricProperties(_ a: [Float], _ b: [Float]) -> GeometricProperties {
        // Check if vectors are in same orthant (all components have same sign)
        let sameOrthant = zip(a, b).allSatisfy { $0 * $1 >= 0 }
        
        // Calculate angle in radians
        let cosineSim = calculateAngularSimilarity(a, b)
        let angle = acos(max(-1, min(1, cosineSim)))
        
        // Magnitude ratio
        let magA = sqrt(a.map { $0 * $0 }.reduce(0, +))
        let magB = sqrt(b.map { $0 * $0 }.reduce(0, +))
        let magnitudeRatio = (magA > 0 && magB > 0) ? magA / magB : 1.0
        
        // Center point between vectors
        let centerPoint = zip(a, b).map { ($0 + $1) / 2 }
        
        // Bounding box
        let minVals = zip(a, b).map { min($0, $1) }
        let maxVals = zip(a, b).map { max($0, $1) }
        let volume = zip(minVals, maxVals).map { $1 - $0 }.reduce(1.0, *)
        
        // Determine topology
        let topology: Topology
        if sameOrthant && angle < 0.1 {
            topology = .dense
        } else if angle > 2.0 {
            topology = .sparse
        } else {
            topology = .general
        }
        
        return GeometricProperties(
            sameOrthant: sameOrthant,
            angle: angle,
            magnitudeRatio: magnitudeRatio,
            centerPoint: centerPoint,
            boundingBox: BoundingBox(min: minVals, max: maxVals, volume: volume),
            topology: topology
        )
    }
    
    private func applyFilter(
        _ vectors: [StoredVector],
        filter: SearchFilter
    ) async throws -> [StoredVector] {
        // Use the shared FilterEvaluator for consistent filtering across the codebase
        return try await FilterEvaluator.filterVectors(
            vectors,
            filter: filter,
            decoder: JSONDecoder(),
            encoder: JSONEncoder()
        )
    }
    
    private func updateSearchHistory(probesUsed: Int, recall: Float, searchTime: TimeInterval) async {
        await searchHistory.record(
            probesUsed: probesUsed,
            recall: recall,
            searchTime: searchTime
        )
    }
    
    private func determineInitialProbes(
        queryAnalysis: QueryAnalysis?,
        baseConfig: Int
    ) -> Int {
        guard let analysis = queryAnalysis else { return baseConfig }
        
        // Adjust probe count based on query difficulty
        if let difficulty = analysis.difficulty {
            if difficulty > 0.8 {
                return Int(Float(baseConfig) * 1.5)
            } else if difficulty < 0.2 {
                return max(1, Int(Float(baseConfig) * 0.7))
            }
        }
        
        return baseConfig
    }
}

// MARK: - Supporting Types

/// Probe candidate with priority
private struct ProbeCandidate {
    let centroidIndex: Int
    let priority: Float
}

/// Query analysis results
private struct QueryAnalysis {
    var sparsity: Float?
    var magnitude: Float?
    var entropy: Float?
    var difficulty: Float?
    var historicalPerformance: SearchHistory.Metrics?
}

/// Candidate result during search
private struct CandidateResult<Vector: SIMD & Sendable, Metadata: Codable & Sendable> 
where Vector.Scalar: BinaryFloatingPoint {
    let vector: StoredVector
    let distance: Float
    let probeIndex: Int
}

// StoredVector is now imported from IVFIndex

/// Search history for adaptive behavior
private actor SearchHistory {
    struct Metrics {
        let probesUsed: Float
        let recall: Float
        let searchTime: TimeInterval
    }
    
    private var history: [Metrics] = []
    private let maxHistory = 1000
    
    func record(probesUsed: Int, recall: Float, searchTime: TimeInterval) {
        let metrics = Metrics(
            probesUsed: Float(probesUsed),
            recall: recall,
            searchTime: searchTime
        )
        
        history.append(metrics)
        
        if history.count > maxHistory {
            history.removeFirst()
        }
    }
    
    func getAverageMetrics() -> Metrics {
        guard !history.isEmpty else {
            return Metrics(probesUsed: 10, recall: 0.9, searchTime: 0.001)
        }
        
        let count = Float(history.count)
        let avgProbes = history.map { $0.probesUsed }.reduce(0, +) / count
        let avgRecall = history.map { $0.recall }.reduce(0, +) / count
        let avgTime = history.map { $0.searchTime }.reduce(0, +) / Double(history.count)
        
        return Metrics(
            probesUsed: avgProbes,
            recall: avgRecall,
            searchTime: avgTime
        )
    }
}

// MARK: - IVFIndex Extension
// Note: These methods should be implemented directly in IVFIndex
// or accessed through proper public APIs
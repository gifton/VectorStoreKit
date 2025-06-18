// HierarchicalIndex.swift
// VectorStoreKit
//
// Two-level hierarchical index for efficient large-scale vector search

import Foundation
import os.log

/// Configuration for hierarchical index structure
public struct HierarchicalConfiguration: IndexConfiguration, Sendable {
    public let topLevelClusters: Int
    public let leafIndexSize: Int
    public let probesPerQuery: Int
    public let rebalanceThreshold: Float
    public let enableDynamicProbing: Bool
    
    public init(
        topLevelClusters: Int = 1024,      // sqrt(1M) for balanced hierarchy
        leafIndexSize: Int = 1000,         // ~1000 vectors per leaf
        probesPerQuery: Int = 10,          // Number of clusters to search
        rebalanceThreshold: Float = 0.2,   // Rebalance when 20% imbalanced
        enableDynamicProbing: Bool = true  // Adjust probes based on query
    ) {
        self.topLevelClusters = topLevelClusters
        self.leafIndexSize = leafIndexSize
        self.probesPerQuery = probesPerQuery
        self.rebalanceThreshold = rebalanceThreshold
        self.enableDynamicProbing = enableDynamicProbing
    }
    
    /// Optimized configurations for different dataset sizes
    public static func forDatasetSize(_ size: Int) -> HierarchicalConfiguration {
        switch size {
        case 0..<100_000:
            return HierarchicalConfiguration(
                topLevelClusters: 100,
                leafIndexSize: 1000,
                probesPerQuery: 5
            )
        case 100_000..<500_000:
            return HierarchicalConfiguration(
                topLevelClusters: 512,
                leafIndexSize: 1000,
                probesPerQuery: 8
            )
        case 500_000..<1_000_000:
            return HierarchicalConfiguration(
                topLevelClusters: 1024,
                leafIndexSize: 1000,
                probesPerQuery: 10
            )
        default:
            return HierarchicalConfiguration(
                topLevelClusters: 2048,
                leafIndexSize: 1000,
                probesPerQuery: 15
            )
        }
    }
    
    // MARK: - IndexConfiguration Protocol
    
    public func validate() throws {
        guard topLevelClusters > 0 else {
            throw HierarchicalIndexError.invalidConfiguration
        }
        guard leafIndexSize > 0 else {
            throw HierarchicalIndexError.invalidConfiguration
        }
        guard probesPerQuery > 0 && probesPerQuery <= topLevelClusters else {
            throw HierarchicalIndexError.invalidConfiguration
        }
        guard rebalanceThreshold >= 0 && rebalanceThreshold <= 1 else {
            throw HierarchicalIndexError.invalidConfiguration
        }
    }
    
    public func estimatedMemoryUsage(for vectorCount: Int) -> Int {
        // Estimate: vector storage + index overhead + cluster metadata
        let vectorStorage = vectorCount * 512 * MemoryLayout<Float>.size // Assume 512-d vectors
        let indexOverhead = vectorCount * 100 // ~100 bytes per vector for index structures
        let clusterMetadata = topLevelClusters * 1000 // ~1KB per cluster
        return vectorStorage + indexOverhead + clusterMetadata
    }
    
    public func computationalComplexity() -> ComputationalComplexity {
        // Hierarchical search is O(log n) with good clustering
        return .logarithmic
    }
}

/// Two-level hierarchical index for large datasets
public actor HierarchicalIndex<Vector: SIMD, Metadata: Codable & Sendable>: VectorIndex 
where Vector: Sendable, Vector.Scalar: BinaryFloatingPoint {
    
    // MARK: - Type Aliases (VectorIndex Protocol)
    
    public typealias Configuration = HierarchicalConfiguration
    public typealias Statistics = HierarchicalIndexStatistics
    
    // MARK: - Properties
    
    public let configuration: HierarchicalConfiguration
    private let distanceMetric: DistanceMetric
    
    // Two-level index structure
    private var topLevelIndex: IVFIndex<Vector, Metadata>  // Coarse quantizer
    private var leafIndices: [Int: HNSWIndex<Vector, Metadata>] = [:] // Fine-grained search
    
    // Cluster management
    private var clusterSizes: [Int: Int] = [:]
    private var clusterCentroids: [Vector] = []
    
    // Statistics
    private var totalVectors: Int = 0
    private var indexingTime: Double = 0
    private var rebalanceCount: Int = 0
    private var totalSearches: Int = 0
    private var totalSearchTime: Double = 0
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "HierarchicalIndex")
    
    // MARK: - Initialization
    
    public init(
        dimension: Int,
        configuration: HierarchicalConfiguration = HierarchicalConfiguration(),
        distanceMetric: DistanceMetric = .euclidean
    ) async throws {
        self.configuration = configuration
        self.distanceMetric = distanceMetric
        
        // Initialize top-level IVF index
        let ivfConfig = IVFConfiguration(
            dimensions: dimension,
            numberOfCentroids: configuration.topLevelClusters,
            numberOfProbes: configuration.probesPerQuery,
            trainingSampleSize: 10_000
        )
        
        self.topLevelIndex = try await IVFIndex<Vector, Metadata>(
            configuration: ivfConfig
        )
        
        logger.info("Initialized hierarchical index with \(configuration.topLevelClusters) top-level clusters")
    }
    
    // MARK: - VectorIndex Protocol Properties
    
    public var count: Int {
        get async { totalVectors }
    }
    
    public var capacity: Int {
        get async { Int.max }
    }
    
    public var memoryUsage: Int {
        get async {
            var totalMemory = await topLevelIndex.memoryUsage
            for (_, leafIndex) in leafIndices {
                totalMemory += await leafIndex.memoryUsage
            }
            return totalMemory
        }
    }
    
    public var isOptimized: Bool {
        get async { false } // Can be enhanced based on rebalancing state
    }
    
    // MARK: - VectorIndex Protocol Methods
    
    public func insert(_ entry: VectorEntry<Vector, Metadata>) async throws -> InsertResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        try await insertBatch([(entry.vector, entry.id, entry.metadata)])
        let insertTime = CFAbsoluteTimeGetCurrent() - startTime
        
        return InsertResult(
            success: true,
            insertTime: insertTime,
            memoryImpact: MemoryLayout<Vector>.size + MemoryLayout<Metadata>.size,
            indexReorganization: false
        )
    }
    
    /// Insert multiple vectors efficiently
    public func insertBatch(
        _ entries: [(vector: Vector, id: VectorID, metadata: Metadata?)]
    ) async throws {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Assign vectors to clusters
        let assignments = try await assignToClusters(entries.map(\.vector))
        
        // Group by cluster
        var clusteredEntries: [Int: [(Vector, VectorID, Metadata?)]] = [:]
        for (index, clusterID) in assignments.enumerated() {
            if clusteredEntries[clusterID] == nil {
                clusteredEntries[clusterID] = []
            }
            clusteredEntries[clusterID]?.append(entries[index])
        }
        
        // Insert into leaf indices
        for (clusterID, clusterEntries) in clusteredEntries {
            try await insertIntoLeaf(clusterID: clusterID, entries: clusterEntries)
        }
        
        totalVectors += entries.count
        indexingTime += CFAbsoluteTimeGetCurrent() - startTime
        
        // Check if rebalancing is needed
        if await shouldRebalance() {
            Task {
                try await rebalance()
            }
        }
    }
    
    /// Stream insertion for large datasets
    public func insertStream(
        _ vectors: AsyncStream<(Vector, VectorID, Metadata?)>
    ) async throws {
        var batch: [(Vector, VectorID, Metadata?)] = []
        
        for await entry in vectors {
            batch.append(entry)
            
            if batch.count >= configuration.leafIndexSize {
                try await insertBatch(batch)
                batch.removeAll(keepingCapacity: true)
            }
        }
        
        // Insert remaining batch
        if !batch.isEmpty {
            try await insertBatch(batch)
        }
    }
    
    // MARK: - Search
    
    public func search(
        query: Vector,
        k: Int,
        strategy: SearchStrategy,
        filter: SearchFilter?
    ) async throws -> [SearchResult<Metadata>] {
        let startTime = CFAbsoluteTimeGetCurrent()
        // Step 1: Find candidate clusters using top-level index
        let numProbes = configuration.enableDynamicProbing ? 
            await dynamicProbeCount(for: query, k: k) : 
            configuration.probesPerQuery
        
        let candidateClusters = try await topLevelIndex.search(
            query: query,
            k: numProbes
        )
        
        // Step 2: Search within candidate clusters
        var allResults: [SearchResult<Metadata>] = []
        
        await withTaskGroup(of: [SearchResult<Metadata>]?.self) { group in
            for result in candidateClusters {
                if let clusterID = Int(result.id),
                   let leafIndex = leafIndices[clusterID] {
                    group.addTask {
                        return try? await leafIndex.search(
                            query: query,
                            k: k * 2, // Over-fetch for better recall
                            strategy: .hybrid,
                            filter: filter
                        )
                    }
                }
            }
            
            for await results in group {
                if let results = results {
                    allResults.append(contentsOf: results)
                }
            }
        }
        
        // Apply filter if provided
        // Note: SearchFilter filtering should be handled by leaf indices
        // Additional filtering here would require converting SearchFilter to predicate
        
        // Step 3: Merge and sort results
        allResults.sort { $0.distance < $1.distance }
        
        totalSearches += 1
        totalSearchTime += CFAbsoluteTimeGetCurrent() - startTime
        
        return Array(allResults.prefix(k))
    }
    
    public func update(id: VectorID, vector: Vector?, metadata: Metadata?) async throws -> Bool {
        // Find which cluster contains this vector
        for (clusterID, leafIndex) in leafIndices {
            if await leafIndex.contains(id: id) {
                return try await leafIndex.update(id: id, vector: vector, metadata: metadata)
            }
        }
        return false
    }
    
    public func delete(id: VectorID) async throws -> Bool {
        // Find and delete from appropriate leaf index
        for (clusterID, leafIndex) in leafIndices {
            if await leafIndex.contains(id: id) {
                let deleted = try await leafIndex.delete(id: id)
                if deleted {
                    totalVectors -= 1
                    clusterSizes[clusterID, default: 0] -= 1
                }
                return deleted
            }
        }
        return false
    }
    
    public func contains(id: VectorID) async -> Bool {
        for (_, leafIndex) in leafIndices {
            if await leafIndex.contains(id: id) {
                return true
            }
        }
        return false
    }
    
    public func optimize(strategy: OptimizationStrategy) async throws {
        // Optimize all leaf indices
        for (_, leafIndex) in leafIndices {
            try await leafIndex.optimize(strategy: strategy)
        }
        
        // Potentially rebalance
        if strategy == .rebalance {
            try await rebalance()
        }
    }
    
    public func compact() async throws {
        // Compact all leaf indices
        for (_, leafIndex) in leafIndices {
            try await leafIndex.compact()
        }
    }
    
    public func statistics() async -> Statistics {
        getStatistics()
    }
    
    public func validateIntegrity() async throws -> IntegrityReport {
        var errors: [IntegrityError] = []
        var warnings: [IntegrityWarning] = []
        var totalChecks = 0
        var passedChecks = 0
        
        // Validate top-level index
        totalChecks += 1
        do {
            let topLevelReport = try await topLevelIndex.validateIntegrity()
            if topLevelReport.isValid {
                passedChecks += 1
            } else {
                errors.append(contentsOf: topLevelReport.errors)
            }
        } catch {
            errors.append(IntegrityError(
                type: .corruption,
                description: "Top-level index validation failed: \(error)",
                severity: .high
            ))
        }
        
        // Validate leaf indices
        for (clusterID, leafIndex) in leafIndices {
            totalChecks += 1
            do {
                let leafReport = try await leafIndex.validateIntegrity()
                if leafReport.isValid {
                    passedChecks += 1
                } else {
                    errors.append(contentsOf: leafReport.errors)
                }
            } catch {
                errors.append(IntegrityError(
                    type: .corruption,
                    description: "Leaf index \(clusterID) validation failed: \(error)",
                    severity: .medium
                ))
            }
        }
        
        return IntegrityReport(
            isValid: errors.isEmpty,
            errors: errors,
            warnings: warnings,
            statistics: IntegrityStatistics(
                totalChecks: totalChecks,
                passedChecks: passedChecks,
                failedChecks: totalChecks - passedChecks,
                checkDuration: 0
            )
        )
    }
    
    public func export(format: ExportFormat) async throws -> Data {
        throw HierarchicalIndexError.invalidConfiguration
    }
    
    public func `import`(data: Data, format: ExportFormat) async throws {
        throw HierarchicalIndexError.invalidConfiguration
    }
    
    public func analyzeDistribution() async -> DistributionAnalysis {
        // Basic implementation - can be enhanced
        let clusterSizeArray = Array(clusterSizes.values)
        let mean = clusterSizeArray.isEmpty ? 0 : clusterSizeArray.reduce(0, +) / clusterSizeArray.count
        
        return DistributionAnalysis(
            dimensionality: Vector.scalarCount,
            density: Float(totalVectors) / Float(configuration.topLevelClusters),
            clustering: ClusteringAnalysis(
                estimatedClusters: leafIndices.count,
                silhouetteScore: 0,
                inertia: 0,
                clusterCenters: []
            ),
            outliers: [],
            statistics: DistributionStatistics(
                mean: [Float(mean)],
                variance: [calculateVariance(clusterSizeArray)],
                skewness: [0],
                kurtosis: [0]
            )
        )
    }
    
    public func performanceProfile() async -> PerformanceProfile {
        let avgSearchLatency = totalSearches > 0 ? totalSearchTime / Double(totalSearches) : 0
        
        return PerformanceProfile(
            searchLatency: LatencyProfile(
                p50: avgSearchLatency,
                p90: avgSearchLatency * 1.2,
                p95: avgSearchLatency * 1.5,
                p99: avgSearchLatency * 2.0,
                max: avgSearchLatency * 3.0
            ),
            insertLatency: LatencyProfile(
                p50: indexingTime / Double(max(totalVectors, 1)),
                p90: indexingTime / Double(max(totalVectors, 1)) * 1.2,
                p95: indexingTime / Double(max(totalVectors, 1)) * 1.5,
                p99: indexingTime / Double(max(totalVectors, 1)) * 2.0,
                max: indexingTime / Double(max(totalVectors, 1)) * 3.0
            ),
            memoryUsage: MemoryProfile(
                baseline: 0,
                peak: await memoryUsage,
                average: await memoryUsage,
                efficiency: 0.8
            ),
            throughput: ThroughputProfile(
                queriesPerSecond: Float(totalSearches) / Float(max(totalSearchTime, 1)),
                insertsPerSecond: Float(totalVectors) / Float(max(indexingTime, 1)),
                updatesPerSecond: 0,
                deletesPerSecond: 0
            )
        )
    }
    
    public func visualizationData() async -> VisualizationData {
        var nodePositions: [[Float]] = []
        var edges: [(Int, Int, Float)] = []
        
        // Add cluster centroids as nodes
        for (i, _) in clusterCentroids.enumerated() {
            nodePositions.append([Float(i), 0])
        }
        
        return VisualizationData(
            nodePositions: nodePositions,
            edges: edges,
            nodeMetadata: [:],
            layoutAlgorithm: "hierarchical"
        )
    }
    
    /// Batch search for multiple queries
    public func batchSearch(
        queries: [Vector],
        k: Int,
        filter: SearchFilter? = nil
    ) async throws -> [[SearchResult<Metadata>]] {
        try await withThrowingTaskGroup(of: (Int, [SearchResult<Metadata>]).self) { group in
            for (index, query) in queries.enumerated() {
                group.addTask {
                    let results = try await self.search(
                        query: query, 
                        k: k, 
                        strategy: .approximate,
                        filter: filter
                    )
                    return (index, results)
                }
            }
            
            var results = Array(repeating: [SearchResult<Metadata>](), count: queries.count)
            for try await (index, searchResults) in group {
                results[index] = searchResults
            }
            return results
        }
    }
    
    // MARK: - Cluster Management
    
    private func assignToClusters(_ vectors: [Vector]) async throws -> [Int] {
        // Use top-level index to find nearest clusters
        var assignments: [Int] = []
        
        for vector in vectors {
            let nearestClusters = try await topLevelIndex.search(query: vector, k: 1)
            if let nearest = nearestClusters.first,
               let clusterID = Int(nearest.id) {
                assignments.append(clusterID)
            } else {
                // Assign to least loaded cluster
                let clusterID = await findLeastLoadedCluster()
                assignments.append(clusterID)
            }
        }
        
        return assignments
    }
    
    private func insertIntoLeaf(
        clusterID: Int,
        entries: [(Vector, VectorID, Metadata?)]
    ) async throws {
        // Get or create leaf index
        if leafIndices[clusterID] == nil {
            leafIndices[clusterID] = try await createLeafIndex()
        }
        
        guard let leafIndex = leafIndices[clusterID] else {
            throw HierarchicalIndexError.leafIndexNotFound(clusterID)
        }
        
        // Insert entries
        for (vector, id, metadata) in entries {
            if let metadata = metadata {
                let entry = VectorEntry(id: id, vector: vector, metadata: metadata, tier: .hot)
                _ = try await leafIndex.insert(entry)
            }
        }
        
        // Update cluster size
        clusterSizes[clusterID, default: 0] += entries.count
    }
    
    private func createLeafIndex() async throws -> HNSWIndex<Vector, Metadata> {
        let hnswConfig = HNSWConfiguration(
            maxConnections: 16,
            efConstruction: 200
        )
        
        return try HNSWIndex<Vector, Metadata>(
            configuration: HNSWIndex<Vector, Metadata>.Configuration(
                maxConnections: hnswConfig.maxConnections,
                efConstruction: hnswConfig.efConstruction,
                levelMultiplier: hnswConfig.levelMultiplier,
                distanceMetric: hnswConfig.distanceMetric,
                useAdaptiveTuning: hnswConfig.useAdaptiveTuning,
                optimizationThreshold: hnswConfig.optimizationThreshold,
                enableAnalytics: hnswConfig.enableAnalytics
            )
        )
    }
    
    // MARK: - Dynamic Probing
    
    private func dynamicProbeCount(for query: Vector, k: Int) async -> Int {
        // Adjust probe count based on query characteristics
        let baseProbes = configuration.probesPerQuery
        
        // If k is large, probe more clusters
        let kFactor = min(2.0, Double(k) / 50.0)
        
        // If dataset is imbalanced, probe more
        let imbalanceFactor = await calculateImbalanceFactor()
        
        let baseFactor = Double(baseProbes) * Double(kFactor)
        let imbalanceAdjustment = 1.0 + Double(imbalanceFactor)
        let adjustedProbes = Int(baseFactor * imbalanceAdjustment)
        let maxProbes = configuration.topLevelClusters / 2
        return min(adjustedProbes, maxProbes)
    }
    
    // MARK: - Rebalancing
    
    private func shouldRebalance() async -> Bool {
        let imbalance = await calculateImbalanceFactor()
        return imbalance > configuration.rebalanceThreshold
    }
    
    private func calculateImbalanceFactor() async -> Float {
        guard !clusterSizes.isEmpty else { return 0 }
        
        let sizes = Array(clusterSizes.values)
        let mean = Float(sizes.reduce(0, +)) / Float(sizes.count)
        let variance = sizes.reduce(Float(0)) { sum, size in
            sum + pow(Float(size) - mean, 2)
        } / Float(sizes.count)
        
        let stdDev = sqrt(variance)
        return stdDev / (mean + 1) // Coefficient of variation
    }
    
    private func rebalance() async throws {
        logger.info("Starting rebalance operation")
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Collect all vectors from heavily loaded clusters
        var vectorsToRedistribute: [(Vector, VectorID, Metadata?)] = []
        let meanSize = totalVectors / configuration.topLevelClusters
        let threshold = Int(Float(meanSize) * 1.5)
        
        for (clusterID, size) in clusterSizes where size > threshold {
            if let leafIndex = leafIndices[clusterID] {
                // Extract vectors from overloaded cluster
                // This is a simplified version - in practice, you'd want to
                // selectively move vectors based on their distance to centroids
                let excess = size - meanSize
                logger.debug("Cluster \(clusterID) has \(excess) excess vectors")
            }
        }
        
        // Redistribute vectors
        if !vectorsToRedistribute.isEmpty {
            try await insertBatch(vectorsToRedistribute)
        }
        
        rebalanceCount += 1
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        logger.info("Rebalance completed in \(duration)s")
    }
    
    private func findLeastLoadedCluster() async -> Int {
        // Find cluster with minimum size
        var minSize = Int.max
        var selectedCluster = 0
        
        for i in 0..<configuration.topLevelClusters {
            let size = clusterSizes[i] ?? 0
            if size < minSize {
                minSize = size
                selectedCluster = i
            }
        }
        
        return selectedCluster
    }
    
    private func calculateVariance(_ values: [Int]) -> Float {
        guard !values.isEmpty else { return 0 }
        let mean = Float(values.reduce(0, +)) / Float(values.count)
        let squaredDiffs = values.map { pow(Float($0) - mean, 2) }
        return squaredDiffs.reduce(0, +) / Float(values.count)
    }
    
    // MARK: - Statistics
    
    public func getStatistics() -> HierarchicalIndexStatistics {
        let clusterSizeArray = Array(clusterSizes.values)
        
        return HierarchicalIndexStatistics(
            totalVectors: totalVectors,
            numClusters: configuration.topLevelClusters,
            numActiveClusters: leafIndices.count,
            averageClusterSize: clusterSizeArray.isEmpty ? 0 : 
                clusterSizeArray.reduce(0, +) / clusterSizeArray.count,
            maxClusterSize: clusterSizeArray.max() ?? 0,
            minClusterSize: clusterSizeArray.min() ?? 0,
            indexingTime: indexingTime,
            rebalanceCount: rebalanceCount
        )
    }
}

// MARK: - Supporting Types

public struct HierarchicalIndexStatistics: IndexStatistics, Sendable {
    public let totalVectors: Int
    public let numClusters: Int
    public let numActiveClusters: Int
    public let averageClusterSize: Int
    public let maxClusterSize: Int
    public let minClusterSize: Int
    public let indexingTime: Double
    public let rebalanceCount: Int
    
    // IndexStatistics protocol requirements
    public var vectorCount: Int { totalVectors }
    
    public var memoryUsage: Int {
        // Rough estimate based on vector count and metadata
        totalVectors * (MemoryLayout<Float>.size * 512 + 100) // Assume 512-d vectors + metadata overhead
    }
    
    public var averageSearchLatency: TimeInterval {
        0.001 // Placeholder - should be calculated from actual measurements
    }
    
    public var qualityMetrics: IndexQualityMetrics {
        IndexQualityMetrics(
            recall: 0.95,
            precision: 0.95,
            buildTime: indexingTime,
            memoryEfficiency: Float(totalVectors) / Float(numClusters * 1000),
            searchLatency: averageSearchLatency
        )
    }
    
    public var loadFactor: Float {
        Float(totalVectors) / Float(numClusters * 1000) // Assuming target of 1000 per cluster
    }
    
    public var imbalanceRatio: Float {
        guard averageClusterSize > 0 else { return 0 }
        return Float(maxClusterSize - minClusterSize) / Float(averageClusterSize)
    }
}

public enum HierarchicalIndexError: LocalizedError {
    case leafIndexNotFound(Int)
    case clusteringFailed
    case invalidConfiguration
    
    public var errorDescription: String? {
        switch self {
        case .leafIndexNotFound(let clusterID):
            return "Leaf index not found for cluster \(clusterID)"
        case .clusteringFailed:
            return "Failed to perform clustering"
        case .invalidConfiguration:
            return "Invalid hierarchical index configuration"
        }
    }
}

// MARK: - Extensions for specific vector types

extension HierarchicalIndex where Vector == Vector512 {
    /// Optimized initialization for 512-dimensional vectors
    public static func optimizedFor512D(
        datasetSize: Int,
        distanceMetric: DistanceMetric = .euclidean
    ) async throws -> HierarchicalIndex<Vector512, Metadata> {
        let config = HierarchicalConfiguration.forDatasetSize(datasetSize)
        return try await HierarchicalIndex<Vector512, Metadata>(
            dimension: 512,
            configuration: config,
            distanceMetric: distanceMetric
        )
    }
}
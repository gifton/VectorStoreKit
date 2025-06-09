// VectorStoreKit: Hybrid Index
//
// Combines IVF and Learned Index approaches for optimal performance

import Foundation
import simd

/// Query routing decision
private struct RoutingDecision: Sendable {
    let useIVF: Bool
    let useLearnedIndex: Bool
    let ivfWeight: Float
    let reason: String
}

/// Hybrid Index implementation
public actor HybridIndex<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: VectorIndex
where Vector.Scalar: BinaryFloatingPoint {
    
    // MARK: - Properties
    
    public let configuration: HybridIndexConfiguration
    private let ivfIndex: IVFIndex<Vector, Metadata>
    private let learnedIndex: LearnedIndex<Vector, Metadata>
    private var queryStatistics: QueryStatistics
    private var routingModel: RoutingModel?
    
    private struct QueryStatistics: Sendable {
        var totalQueries: Int = 0
        var ivfQueries: Int = 0
        var learnedQueries: Int = 0
        var hybridQueries: Int = 0
        var averageIVFLatency: Double = 0
        var averageLearnedLatency: Double = 0
    }
    
    private actor RoutingModel {
        private var ivfPerformance: RoutingPerformanceMetrics
        private var learnedPerformance: RoutingPerformanceMetrics
        
        struct RoutingPerformanceMetrics: Sendable {
            var latencies: [Double] = []
            var recalls: [Float] = []
            var averageLatency: Double { latencies.isEmpty ? 0 : latencies.reduce(0, +) / Double(latencies.count) }
            var averageRecall: Float { recalls.isEmpty ? 0 : recalls.reduce(0, +) / Float(recalls.count) }
        }
        
        init() {
            self.ivfPerformance = RoutingPerformanceMetrics()
            self.learnedPerformance = RoutingPerformanceMetrics()
        }
        
        func recordPerformance(
            forIVF: Bool,
            latency: Double,
            recall: Float?
        ) {
            if forIVF {
                ivfPerformance.latencies.append(latency)
                if let recall = recall {
                    ivfPerformance.recalls.append(recall)
                }
            } else {
                learnedPerformance.latencies.append(latency)
                if let recall = recall {
                    learnedPerformance.recalls.append(recall)
                }
            }
            
            // Keep only recent measurements
            let maxHistory = 1000
            if ivfPerformance.latencies.count > maxHistory {
                ivfPerformance.latencies.removeFirst()
            }
            if learnedPerformance.latencies.count > maxHistory {
                learnedPerformance.latencies.removeFirst()
            }
        }
        
        func recommendRouting() -> RoutingDecision {
            let ivfScore = computeScore(ivfPerformance)
            let learnedScore = computeScore(learnedPerformance)
            
            if ivfScore > learnedScore * 1.2 {
                return RoutingDecision(
                    useIVF: true,
                    useLearnedIndex: false,
                    ivfWeight: 1.0,
                    reason: "IVF performing better"
                )
            } else if learnedScore > ivfScore * 1.2 {
                return RoutingDecision(
                    useIVF: false,
                    useLearnedIndex: true,
                    ivfWeight: 0.0,
                    reason: "Learned index performing better"
                )
            } else {
                return RoutingDecision(
                    useIVF: true,
                    useLearnedIndex: true,
                    ivfWeight: 0.5,
                    reason: "Similar performance, using both"
                )
            }
        }
        
        private func computeScore(_ metrics: RoutingPerformanceMetrics) -> Double {
            // Score based on latency (lower is better) and recall (higher is better)
            let latencyScore = metrics.averageLatency > 0 ? 1.0 / metrics.averageLatency : 0
            let recallScore = Double(metrics.averageRecall)
            return latencyScore * recallScore
        }
    }
    
    // MARK: - Initialization
    
    public init(configuration: HybridIndexConfiguration) async throws {
        self.configuration = configuration
        self.ivfIndex = try await IVFIndex(configuration: configuration.ivfConfig)
        self.learnedIndex = try await LearnedIndex(configuration: configuration.learnedConfig)
        self.queryStatistics = QueryStatistics()
        self.routingModel = RoutingModel()
    }
    
    // MARK: - VectorIndex Protocol
    
    public var count: Int {
        get async {
            await ivfIndex.count // Both indexes should have same count
        }
    }
    
    public var capacity: Int {
        Int.max
    }
    
    public var memoryUsage: Int {
        get async {
            let ivfMemory = await ivfIndex.memoryUsage
            let learnedMemory = await learnedIndex.memoryUsage
            return ivfMemory + learnedMemory
        }
    }
    
    public var isOptimized: Bool {
        get async {
            let ivfOptimized = await ivfIndex.isOptimized
            let learnedOptimized = await learnedIndex.isOptimized
            return ivfOptimized && learnedOptimized
        }
    }
    
    public func insert(_ entry: VectorEntry<Vector, Metadata>) async throws -> InsertResult {
        // Insert into both indexes
        async let ivfResult = ivfIndex.insert(entry)
        async let learnedResult = learnedIndex.insert(entry)
        
        let results = try await [ivfResult, learnedResult]
        
        // Return combined result
        return InsertResult(
            success: results.allSatisfy { $0.success },
            insertTime: results.map { $0.insertTime }.max() ?? 0,
            memoryImpact: results.reduce(0) { $0 + $1.memoryImpact },
            indexReorganization: results.contains { $0.indexReorganization }
        )
    }
    
    public func search(
        query: Vector,
        k: Int,
        strategy: SearchStrategy = .adaptive,
        filter: SearchFilter? = nil
    ) async throws -> [SearchResult<Metadata>] {
        let startTime = DispatchTime.now()
        
        // Decide routing based on configuration
        let routing = await decideRouting(query: query, strategy: strategy)
        
        var results: [SearchResult<Metadata>] = []
        
        if routing.useIVF && routing.useLearnedIndex {
            // Use both indexes and merge results
            async let ivfResults = ivfIndex.search(
                query: query,
                k: k * 2, // Get more candidates
                strategy: strategy,
                filter: filter
            )
            async let learnedResults = learnedIndex.search(
                query: query,
                k: k * 2,
                strategy: strategy,
                filter: filter
            )
            
            let allResults = try await ivfResults + learnedResults
            results = mergeResults(allResults, k: k, ivfWeight: routing.ivfWeight)
            
            queryStatistics.hybridQueries += 1
            
        } else if routing.useIVF {
            // Use only IVF
            results = try await ivfIndex.search(
                query: query,
                k: k,
                strategy: strategy,
                filter: filter
            )
            queryStatistics.ivfQueries += 1
            
        } else {
            // Use only learned index
            results = try await learnedIndex.search(
                query: query,
                k: k,
                strategy: strategy,
                filter: filter
            )
            queryStatistics.learnedQueries += 1
        }
        
        queryStatistics.totalQueries += 1
        
        // Record performance for adaptive routing
        let latency = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
        await routingModel?.recordPerformance(
            forIVF: routing.useIVF,
            latency: latency,
            recall: nil // Would need ground truth to compute
        )
        
        return results
    }
    
    public func update(id: String, vector: Vector?, metadata: Metadata?) async throws -> Bool {
        // Neither IVF nor Learned index support efficient updates
        return false
    }
    
    public func delete(id: String) async throws -> Bool {
        async let ivfDelete = ivfIndex.delete(id: id)
        async let learnedDelete = learnedIndex.delete(id: id)
        
        let results = try await [ivfDelete, learnedDelete]
        return results.allSatisfy { $0 }
    }
    
    public func contains(id: String) async -> Bool {
        await ivfIndex.contains(id: id)
    }
    
    public func optimize(strategy: OptimizationStrategy) async throws {
        async let ivfOptimize: Void = ivfIndex.optimize(strategy: strategy)
        async let learnedOptimize: Void = learnedIndex.optimize(strategy: strategy)
        
        _ = try await [ivfOptimize, learnedOptimize]
    }
    
    public func compact() async throws {
        try await ivfIndex.compact()
        try await learnedIndex.compact()
    }
    
    public func statistics() async -> HybridIndexStatistics {
        async let ivfStats = ivfIndex.statistics()
        async let learnedStats = learnedIndex.statistics()
        
        let stats = await (ivfStats, learnedStats)
        
        return HybridIndexStatistics(
            vectorCount: await count,
            memoryUsage: await memoryUsage,
            averageSearchLatency: 0.0005,
            ivfStatistics: stats.0,
            learnedStatistics: stats.1,
            routingStatistics: RoutingStatistics(
                totalQueries: queryStatistics.totalQueries,
                ivfQueries: queryStatistics.ivfQueries,
                learnedQueries: queryStatistics.learnedQueries,
                hybridQueries: queryStatistics.hybridQueries
            )
        )
    }
    
    public func validateIntegrity() async throws -> IntegrityReport {
        async let ivfReport = ivfIndex.validateIntegrity()
        async let learnedReport = learnedIndex.validateIntegrity()
        
        let reports = try await [ivfReport, learnedReport]
        
        // Combine reports
        let allErrors = reports.flatMap { $0.errors }
        let allWarnings = reports.flatMap { $0.warnings }
        
        return IntegrityReport(
            isValid: allErrors.isEmpty,
            errors: allErrors,
            warnings: allWarnings,
            statistics: IntegrityStatistics(
                totalChecks: reports.reduce(0) { $0 + $1.statistics.totalChecks },
                passedChecks: reports.reduce(0) { $0 + $1.statistics.passedChecks },
                failedChecks: reports.reduce(0) { $0 + $1.statistics.failedChecks },
                checkDuration: reports.map { $0.statistics.checkDuration }.max() ?? 0
            )
        )
    }
    
    public func export(format: ExportFormat) async throws -> Data {
        async let ivfData = ivfIndex.export(format: format)
        async let learnedData = learnedIndex.export(format: format)
        
        let exportData = HybridExportData(
            configuration: configuration,
            ivfData: try await ivfData,
            learnedData: try await learnedData
        )
        
        switch format {
        case .json:
            return try JSONEncoder().encode(exportData)
        case .binary:
            return try PropertyListEncoder().encode(exportData)
        default:
            throw ExportError.unsupportedFormat(format)
        }
    }
    
    public func `import`(data: Data, format: ExportFormat) async throws {
        let importData: HybridExportData
        
        switch format {
        case .json:
            importData = try JSONDecoder().decode(HybridExportData.self, from: data)
        case .binary:
            importData = try PropertyListDecoder().decode(HybridExportData.self, from: data)
        default:
            throw ImportError.unsupportedFormat(format)
        }
        
        try await ivfIndex.import(data: importData.ivfData, format: format)
        try await learnedIndex.import(data: importData.learnedData, format: format)
    }
    
    public func analyzeDistribution() async -> DistributionAnalysis {
        // Use IVF's distribution analysis as primary
        await ivfIndex.analyzeDistribution()
    }
    
    public func performanceProfile() async -> PerformanceProfile {
        async let ivfProfile = ivfIndex.performanceProfile()
        async let learnedProfile = learnedIndex.performanceProfile()
        
        let profiles = await (ivfProfile, learnedProfile)
        
        // Return best-case performance from both
        return PerformanceProfile(
            searchLatency: LatencyProfile(
                p50: min(profiles.0.searchLatency.p50, profiles.1.searchLatency.p50),
                p90: min(profiles.0.searchLatency.p90, profiles.1.searchLatency.p90),
                p95: min(profiles.0.searchLatency.p95, profiles.1.searchLatency.p95),
                p99: min(profiles.0.searchLatency.p99, profiles.1.searchLatency.p99),
                max: min(profiles.0.searchLatency.max, profiles.1.searchLatency.max)
            ),
            insertLatency: profiles.0.insertLatency, // Use IVF's insert latency
            memoryUsage: MemoryProfile(
                baseline: profiles.0.memoryUsage.baseline + profiles.1.memoryUsage.baseline,
                peak: profiles.0.memoryUsage.peak + profiles.1.memoryUsage.peak,
                average: profiles.0.memoryUsage.average + profiles.1.memoryUsage.average,
                efficiency: (profiles.0.memoryUsage.efficiency + profiles.1.memoryUsage.efficiency) / 2
            ),
            throughput: ThroughputProfile(
                queriesPerSecond: max(profiles.0.throughput.queriesPerSecond, profiles.1.throughput.queriesPerSecond),
                insertsPerSecond: min(profiles.0.throughput.insertsPerSecond, profiles.1.throughput.insertsPerSecond),
                updatesPerSecond: 0,
                deletesPerSecond: min(profiles.0.throughput.deletesPerSecond, profiles.1.throughput.deletesPerSecond)
            )
        )
    }
    
    public func visualizationData() async -> VisualizationData {
        // Combine visualization data from both indexes
        async let ivfViz = ivfIndex.visualizationData()
        async let learnedViz = learnedIndex.visualizationData()
        
        let vizData = await (ivfViz, learnedViz)
        
        return VisualizationData(
            nodePositions: vizData.0.nodePositions + vizData.1.nodePositions,
            edges: vizData.0.edges + vizData.1.edges,
            nodeMetadata: vizData.0.nodeMetadata.merging(vizData.1.nodeMetadata) { _, new in new },
            layoutAlgorithm: "hybrid"
        )
    }
    
    // MARK: - Hybrid-specific Methods
    
    /// Train both indexes
    public func train(on samples: [[Float]]) async throws {
        // Split samples for training
        let ivfSamples = Array(samples.prefix(samples.count / 2))
        let learnedSamples = Array(samples.suffix(samples.count / 2))
        
        async let ivfTrain: Void = ivfIndex.train(on: ivfSamples)
        async let learnedTrain: Void = learnedIndex.train(on: learnedSamples)
        
        _ = try await [ivfTrain, learnedTrain]
    }
    
    /// Get routing statistics
    public func getRoutingStatistics() async -> RoutingStatistics {
        RoutingStatistics(
            totalQueries: queryStatistics.totalQueries,
            ivfQueries: queryStatistics.ivfQueries,
            learnedQueries: queryStatistics.learnedQueries,
            hybridQueries: queryStatistics.hybridQueries
        )
    }
    
    // MARK: - Private Methods
    
    private func decideRouting(
        query: Vector,
        strategy: SearchStrategy
    ) async -> RoutingDecision {
        switch configuration.routingStrategy {
        case .fixed(let ivfWeight):
            return RoutingDecision(
                useIVF: ivfWeight > 0,
                useLearnedIndex: ivfWeight < 1,
                ivfWeight: ivfWeight,
                reason: "Fixed routing strategy"
            )
            
        case .adaptive:
            return await routingModel?.recommendRouting() ?? RoutingDecision(
                useIVF: true,
                useLearnedIndex: true,
                ivfWeight: 0.5,
                reason: "Default adaptive routing"
            )
            
        case .ensemble:
            return RoutingDecision(
                useIVF: true,
                useLearnedIndex: true,
                ivfWeight: 0.5,
                reason: "Ensemble strategy"
            )
            
        case .hierarchical:
            // Use learned for coarse search, IVF for refinement
            return RoutingDecision(
                useIVF: true,
                useLearnedIndex: true,
                ivfWeight: 0.7,
                reason: "Hierarchical strategy"
            )
        }
    }
    
    private func mergeResults(
        _ results: [SearchResult<Metadata>],
        k: Int,
        ivfWeight: Float
    ) -> [SearchResult<Metadata>] {
        // Deduplicate by ID
        var uniqueResults: [String: SearchResult<Metadata>] = [:]
        
        for result in results {
            if let existing = uniqueResults[result.id] {
                // Weighted average of distances
                let weightedDistance = existing.distance * ivfWeight + result.distance * (1 - ivfWeight)
                uniqueResults[result.id] = SearchResult(
                    id: result.id,
                    distance: weightedDistance,
                    metadata: result.metadata,
                    tier: result.tier,
                    similarityAnalysis: result.similarityAnalysis,
                    provenance: result.provenance,
                    confidence: result.confidence
                )
            } else {
                uniqueResults[result.id] = result
            }
        }
        
        // Sort by distance and take top k
        return Array(uniqueResults.values
            .sorted { $0.distance < $1.distance }
            .prefix(k))
    }
}

// MARK: - Supporting Types

/// Statistics for hybrid index
public struct HybridIndexStatistics: IndexStatistics {
    public let vectorCount: Int
    public let memoryUsage: Int
    public let averageSearchLatency: TimeInterval
    
    public let ivfStatistics: IVFStatistics
    public let learnedStatistics: LearnedIndexStatistics
    public let routingStatistics: RoutingStatistics
    
    public var qualityMetrics: IndexQualityMetrics {
        // Use best metrics from both indexes
        let ivfMetrics = ivfStatistics.qualityMetrics
        let learnedMetrics = learnedStatistics.qualityMetrics
        
        return IndexQualityMetrics(
            recall: max(ivfMetrics.recall, learnedMetrics.recall),
            precision: max(ivfMetrics.precision, learnedMetrics.precision),
            buildTime: ivfMetrics.buildTime + learnedMetrics.buildTime,
            memoryEfficiency: (ivfMetrics.memoryEfficiency + learnedMetrics.memoryEfficiency) / 2,
            searchLatency: averageSearchLatency
        )
    }
}

/// Routing statistics
public struct RoutingStatistics: Sendable, Codable {
    public let totalQueries: Int
    public let ivfQueries: Int
    public let learnedQueries: Int
    public let hybridQueries: Int
    
    public var ivfRatio: Float {
        totalQueries > 0 ? Float(ivfQueries) / Float(totalQueries) : 0
    }
    
    public var learnedRatio: Float {
        totalQueries > 0 ? Float(learnedQueries) / Float(totalQueries) : 0
    }
    
    public var hybridRatio: Float {
        totalQueries > 0 ? Float(hybridQueries) / Float(totalQueries) : 0
    }
}

/// Export data for hybrid index
private struct HybridExportData: Codable {
    let configuration: HybridIndexConfiguration
    let ivfData: Data
    let learnedData: Data
}

/// Errors specific to hybrid index
public enum HybridIndexError: LocalizedError {
    case invalidDimensions(Int)
    
    public var errorDescription: String? {
        switch self {
        case .invalidDimensions(let dimensions):
            return "Invalid dimensions: \(dimensions)"
        }
    }
}
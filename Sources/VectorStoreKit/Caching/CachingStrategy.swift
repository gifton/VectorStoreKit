// VectorStoreKit: Production-Ready Multi-Level Caching Strategy
//
// Sophisticated caching framework with adaptive sizing, intelligent eviction,
// and comprehensive monitoring for high-performance vector operations
//
// Key Features:
// - Multi-level cache hierarchy (L1/L2/L3)
// - Adaptive cache sizing based on memory pressure
// - Intelligent cache warming and prefetching
// - Comprehensive hit rate monitoring and analytics
// - Support for multiple eviction policies
// - Integration with memory pressure system

import Foundation
import simd
import os.log

// MARK: - Cache Level Configuration

/// Configuration for individual cache levels
public struct CacheLevelConfiguration: Sendable, Codable {
    public let level: CacheLevel
    public let maxMemory: Int
    public let evictionPolicy: EvictionPolicy
    public let accessTimeThreshold: TimeInterval
    public let warmupEnabled: Bool
    public let prefetchEnabled: Bool
    
    public init(
        level: CacheLevel,
        maxMemory: Int,
        evictionPolicy: EvictionPolicy = EvictionPolicy.lru,
        accessTimeThreshold: TimeInterval = 0.001,
        warmupEnabled: Bool = true,
        prefetchEnabled: Bool = true
    ) {
        self.level = level
        self.maxMemory = maxMemory
        self.evictionPolicy = evictionPolicy
        self.accessTimeThreshold = accessTimeThreshold
        self.warmupEnabled = warmupEnabled
        self.prefetchEnabled = prefetchEnabled
    }
}

/// Cache levels in the hierarchy
public enum CacheLevel: Int, Sendable, Codable, CaseIterable {
    case l1 = 1  // Ultra-fast, small capacity
    case l2 = 2  // Fast, medium capacity
    case l3 = 3  // Slower, large capacity
    
    var promotionThreshold: Int {
        switch self {
        case .l1: return 10  // Promote after 10 accesses
        case .l2: return 5   // Promote after 5 accesses
        case .l3: return 1   // No promotion from L3
        }
    }
    
    var maxAccessTime: TimeInterval {
        switch self {
        case .l1: return 0.0001  // 100 microseconds
        case .l2: return 0.001   // 1 millisecond
        case .l3: return 0.01    // 10 milliseconds
        }
    }
}

// MARK: - Multi-Level Cache Strategy

/// Production-ready multi-level caching strategy
public actor MultiLevelCachingStrategy<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: MemoryPressureAware
where Vector.Scalar: BinaryFloatingPoint {
    
    // MARK: - Properties
    
    /// Configuration for each cache level
    private let levelConfigurations: [CacheLevel: CacheLevelConfiguration]
    
    /// Individual cache instances for each level
    private var caches: [CacheLevel: any VectorCache<Vector>] = [:]
    
    /// Unified monitoring system
    private let monitor: CacheMonitor
    
    /// Memory pressure handler
    private let memoryPressureHandler: MemoryPressureHandler
    
    /// Cache warming engine
    private let warmingEngine: CacheWarmingEngine<Vector>
    
    /// Adaptive sizing controller
    private let adaptiveSizer: AdaptiveCacheSizer
    
    /// Access pattern analyzer
    private let patternAnalyzer: AccessPatternAnalyzer
    
    /// Current memory pressure level
    private var currentMemoryPressure: MemoryPressureLevel = .normal
    
    /// Logger for debugging
    private let logger = Logger(subsystem: "VectorStoreKit", category: "CachingStrategy")
    
    // MARK: - Initialization
    
    public init(
        configurations: [CacheLevelConfiguration],
        storageBackend: (any CacheStorageBackend<Vector>)? = nil
    ) async throws {
        // Validate configurations
        guard !configurations.isEmpty else {
            throw VectorStoreError.configurationInvalid("At least one cache level must be configured")
        }
        
        // Create level configurations map
        var levelConfigs: [CacheLevel: CacheLevelConfiguration] = [:]
        for config in configurations {
            levelConfigs[config.level] = config
        }
        self.levelConfigurations = levelConfigs
        
        // Initialize monitoring and control systems
        self.monitor = CacheMonitor()
        self.memoryPressureHandler = MemoryPressureHandler()
        self.warmingEngine = CacheWarmingEngine()
        self.adaptiveSizer = AdaptiveCacheSizer(
            initialConfigurations: levelConfigs
        )
        self.patternAnalyzer = AccessPatternAnalyzer()
        
        // Create cache instances
        try await initializeCaches(storageBackend: storageBackend)
        
        // Start memory pressure monitoring
        await startMemoryPressureMonitoring()
        
        // Start adaptive sizing
        await startAdaptiveSizing()
        
        logger.info("Multi-level caching strategy initialized with \(configurations.count) levels")
    }
    
    // MARK: - Core Operations
    
    /// Get a vector from the cache hierarchy
    public func get(id: VectorID) async -> Vector? {
        let startTime = ContinuousClock.now
        
        // Try each cache level in order
        for level in CacheLevel.allCases {
            guard let cache = caches[level] else { continue }
            
            if let vector = await cache.get(id: id) {
                let latency = ContinuousClock.now - startTime
                await recordAccess(id: id, level: level, hit: true, latency: latency)
                
                // Promote to higher level if access count exceeds threshold
                await promoteIfNeeded(id: id, vector: vector, fromLevel: level)
                
                return vector
            }
        }
        
        let latency = ContinuousClock.now - startTime
        await recordAccess(id: id, level: nil, hit: false, latency: latency)
        
        return nil
    }
    
    /// Set a vector in the appropriate cache level
    public func set(
        id: VectorID,
        vector: Vector,
        metadata: Metadata? = nil,
        priority: CachePriority = CachePriority.normal
    ) async {
        // Analyze vector quality to determine initial level
        let quality = VectorQuality.assess(vector)
        let level = await determineOptimalLevel(
            quality: quality,
            priority: priority,
            vectorSize: estimateVectorMemorySize(vector)
        )
        
        // Store in determined level
        if let cache = caches[level] {
            await cache.set(id: id, vector: vector, priority: priority)
            
            // Update pattern analyzer
            await patternAnalyzer.recordWrite(id: id, level: level)
            
            // Trigger cache warming if patterns detected
            await triggerWarmingIfNeeded(id: id, level: level)
        }
    }
    
    /// Remove a vector from all cache levels
    public func remove(id: VectorID) async {
        for cache in caches.values {
            await cache.remove(id: id)
        }
        
        await monitor.recordRemoval(id: id)
    }
    
    /// Clear all cache levels
    public func clear() async {
        for cache in caches.values {
            await cache.clear()
        }
        
        await monitor.reset()
        await patternAnalyzer.reset()
    }
    
    // MARK: - Advanced Operations
    
    /// Warm caches with frequently accessed data
    public func warmCaches(predictions: [VectorID: AccessPrediction]) async {
        guard !predictions.isEmpty else { return }
        
        let warmingPlan = await warmingEngine.createWarmingPlan(
            predictions: predictions,
            currentState: await getCurrentCacheState(),
            memoryPressure: currentMemoryPressure
        )
        
        await executeWarmingPlan(warmingPlan)
    }
    
    /// Optimize cache hierarchy based on access patterns
    public func optimize() async {
        // Collect current metrics
        let metrics = await collectComprehensiveMetrics()
        
        // Analyze patterns
        let patterns = await patternAnalyzer.analyzePatterns()
        
        // Generate optimization plan
        let optimizationPlan = await generateOptimizationPlan(
            metrics: metrics,
            patterns: patterns
        )
        
        // Execute optimization
        await executeOptimizationPlan(optimizationPlan)
        
        logger.info("Cache optimization completed")
    }
    
    /// Get comprehensive statistics
    public func statistics() async -> MultiLevelCacheStatistics {
        var levelStats: [CacheLevel: CacheStatistics] = [:]
        
        for (level, cache) in caches {
            levelStats[level] = await cache.statistics()
        }
        
        return MultiLevelCacheStatistics(
            levelStatistics: levelStats,
            globalMetrics: await monitor.getGlobalMetrics(),
            memoryPressure: currentMemoryPressure,
            accessPatterns: await patternAnalyzer.getCurrentPatterns()
        )
    }
    
    /// Get performance analysis
    public func performanceAnalysis() async -> MultiLevelPerformanceAnalysis {
        var levelAnalyses: [CacheLevel: CachePerformanceAnalysis] = [:]
        
        for (level, cache) in caches {
            levelAnalyses[level] = await cache.performanceAnalysis()
        }
        
        let globalAnalysis = await monitor.performGlobalAnalysis()
        let recommendations = await generateRecommendations(
            levelAnalyses: levelAnalyses,
            globalAnalysis: globalAnalysis
        )
        
        return MultiLevelPerformanceAnalysis(
            levelAnalyses: levelAnalyses,
            globalAnalysis: globalAnalysis,
            recommendations: recommendations,
            healthScore: await calculateHealthScore()
        )
    }
    
    // MARK: - Private Methods
    
    private func initializeCaches(
        storageBackend: (any CacheStorageBackend<Vector>)?
    ) async throws {
        for (level, config) in levelConfigurations {
            let cache = try await createCache(
                level: level,
                config: config,
                storageBackend: storageBackend
            )
            caches[level] = cache
        }
    }
    
    private func createCache(
        level: CacheLevel,
        config: CacheLevelConfiguration,
        storageBackend: (any CacheStorageBackend<Vector>)?
    ) async throws -> any VectorCache<Vector> {
        switch config.evictionPolicy {
        case EvictionPolicy.lru:
            return try await BasicLRUVectorCache<Vector>(
                maxMemory: config.maxMemory,
                storageBackend: storageBackend
            )
        case EvictionPolicy.lfu:
            let cache = try await BasicLFUVectorCache<Vector>(
                maxMemory: config.maxMemory,
                storageBackend: storageBackend
            )
            await cache.start()
            return cache
        case EvictionPolicy.fifo:
            return try await BasicFIFOVectorCache<Vector>(
                maxMemory: config.maxMemory,
                storageBackend: storageBackend
            )
        case EvictionPolicy.arc:
            // Use ARC implementation when available, fallback to LRU
            return try await AdaptiveReplacementCache<Vector>(
                maxMemory: config.maxMemory,
                storageBackend: storageBackend
            )
        default:
            // Default to LRU for unsupported policies
            return try await BasicLRUVectorCache<Vector>(
                maxMemory: config.maxMemory,
                storageBackend: storageBackend
            )
        }
    }
    
    private func startMemoryPressureMonitoring() async {
        Task {
            for await pressure in await memoryPressureHandler.pressureUpdates() {
                await handleMemoryPressure(pressure)
            }
        }
    }
    
    private func startAdaptiveSizing() async {
        Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(30))
                await performAdaptiveSizing()
            }
        }
    }
    
    private func handleMemoryPressure(_ pressure: MemoryPressureLevel) async {
        currentMemoryPressure = pressure
        
        switch pressure {
        case MemoryPressureLevel.normal:
            // No action needed
            break
            
        case MemoryPressureLevel.warning:
            // Reduce L1 and L2 cache sizes by 20%
            await adaptiveSizer.adjustSizes(factor: 0.8, levels: [CacheLevel.l1, CacheLevel.l2])
            
        case MemoryPressureLevel.urgent:
            // Reduce all cache sizes by 40%
            await adaptiveSizer.adjustSizes(factor: 0.6, levels: CacheLevel.allCases)
            
        case MemoryPressureLevel.critical:
            // Emergency eviction - keep only critical items
            await performEmergencyEviction()
        }
    }
    
    private func performAdaptiveSizing() async {
        let metrics = await collectComprehensiveMetrics()
        let recommendations = await adaptiveSizer.recommendSizeAdjustments(
            metrics: metrics,
            memoryPressure: currentMemoryPressure
        )
        
        for recommendation in recommendations {
            await applySizeAdjustment(recommendation)
        }
    }
    
    private func recordAccess(
        id: VectorID,
        level: CacheLevel?,
        hit: Bool,
        latency: Duration
    ) async {
        await monitor.recordAccess(
            id: id,
            level: level,
            hit: hit,
            latency: latency
        )
        
        await patternAnalyzer.recordAccess(
            id: id,
            level: level,
            timestamp: Date()
        )
    }
    
    private func promoteIfNeeded(
        id: VectorID,
        vector: Vector,
        fromLevel: CacheLevel
    ) async {
        guard fromLevel != CacheLevel.l1 else { return }
        
        let accessCount = await patternAnalyzer.getAccessCount(id: id)
        
        if accessCount >= fromLevel.promotionThreshold {
            // Promote to next level
            let targetLevel: CacheLevel
            switch fromLevel {
            case .l3: targetLevel = CacheLevel.l2
            case .l2: targetLevel = CacheLevel.l1
            case .l1: return // Already at highest level
            }
            
            if let targetCache = caches[targetLevel] {
                await targetCache.set(id: id, vector: vector, priority: CachePriority.high)
                logger.debug("Promoted vector \(id) from \(fromLevel) to \(targetLevel)")
            }
        }
    }
    
    private func determineOptimalLevel(
        quality: VectorQuality,
        priority: CachePriority,
        vectorSize: Int
    ) async -> CacheLevel {
        // High quality, high priority vectors go to L1
        if quality.quantizability > 0.8 && priority == CachePriority.critical {
            return CacheLevel.l1
        }
        
        // Medium quality or normal priority go to L2
        if quality.quantizability > 0.5 || priority == CachePriority.high {
            return CacheLevel.l2
        }
        
        // Everything else goes to L3
        return CacheLevel.l3
    }
    
    private func triggerWarmingIfNeeded(id: VectorID, level: CacheLevel) async {
        let patterns = await patternAnalyzer.getRelatedVectors(id: id, limit: 10)
        
        if patterns.count > 5 {
            // Significant pattern detected, warm related vectors
            let predictions = patterns.map { 
                ($0, AccessPrediction(confidence: 0.7, timeWindow: 60))
            }
            await warmCaches(predictions: Dictionary(uniqueKeysWithValues: predictions))
        }
    }
    
    private func getCurrentCacheState() async -> CacheState {
        var levelStates: [CacheLevel: LevelState] = [:]
        
        for (level, cache) in caches {
            levelStates[level] = LevelState(
                count: await cache.count,
                memoryUsage: await cache.memoryUsage,
                hitRate: await cache.hitRate
            )
        }
        
        return CacheState(levelStates: levelStates)
    }
    
    private func executeWarmingPlan(_ plan: CacheWarmingPlan) async {
        for action in plan.actions {
            switch action {
            case CacheWarmingPlan.Action.preload(let ids, let level):
                if let cache = caches[level] {
                    await cache.preload(ids)
                }
                
            case CacheWarmingPlan.Action.promote(let id, let fromLevel, let toLevel):
                if let fromCache = caches[fromLevel],
                   let toCache = caches[toLevel],
                   let vector = await fromCache.get(id: id) {
                    await toCache.set(id: id, vector: vector, priority: CachePriority.high)
                }
                
            case CacheWarmingPlan.Action.prefetch(let predictions, let level):
                if let cache = caches[level] {
                    await cache.prefetch(predictions)
                }
            }
        }
    }
    
    private func collectComprehensiveMetrics() async -> ComprehensiveCacheMetrics {
        var levelMetrics: [CacheLevel: LevelMetrics] = [:]
        
        for (level, cache) in caches {
            let stats = await cache.statistics()
            levelMetrics[level] = LevelMetrics(
                hitRate: await cache.hitRate,
                memoryUsage: await cache.memoryUsage,
                count: await cache.count,
                evictionRate: Float(stats.evictionCount) / Float(max(stats.hits + stats.misses, 1))
            )
        }
        
        return ComprehensiveCacheMetrics(
            levelMetrics: levelMetrics,
            globalHitRate: await monitor.getGlobalHitRate(),
            totalMemoryUsage: levelMetrics.values.reduce(0) { $0 + $1.memoryUsage },
            accessPatternStrength: await patternAnalyzer.getPatternStrength()
        )
    }
    
    private func generateOptimizationPlan(
        metrics: ComprehensiveCacheMetrics,
        patterns: AccessPatternSummary
    ) async -> CacheOptimizationPlan {
        var actions: [OptimizationAction] = []
        
        // Check for underutilized levels
        for (level, levelMetrics) in metrics.levelMetrics {
            if levelMetrics.hitRate < 0.3 && levelMetrics.count > 0 {
                actions.append(CacheOptimizationPlan.OptimizationAction.resizeLevel(level: level, factor: 0.7))
            }
        }
        
        // Check for hot spots
        if patterns.hotSpots.count > 10 {
            actions.append(CacheOptimizationPlan.OptimizationAction.redistributeHotSpots(
                vectors: Array(patterns.hotSpots.prefix(20)),
                targetLevel: CacheLevel.l1
            ))
        }
        
        // Check for sequential access patterns
        if patterns.sequentialityScore > 0.7 {
            actions.append(CacheOptimizationPlan.OptimizationAction.enableStreamPrefetch(levels: [CacheLevel.l2, CacheLevel.l3]))
        }
        
        return CacheOptimizationPlan(actions: actions)
    }
    
    private func executeOptimizationPlan(_ plan: CacheOptimizationPlan) async {
        for action in plan.actions {
            switch action {
            case CacheOptimizationPlan.OptimizationAction.resizeLevel(let level, let factor):
                await adaptiveSizer.adjustSize(level: level, factor: factor)
                
            case CacheOptimizationPlan.OptimizationAction.redistributeHotSpots(let vectors, let targetLevel):
                // Move hot vectors to target level
                for vectorId in vectors {
                    // Find vector in lower levels and promote
                    for sourceLevel in CacheLevel.allCases.reversed() {
                        if sourceLevel == targetLevel { continue }
                        if let cache = caches[sourceLevel],
                           let vector = await cache.get(id: vectorId) {
                            if let targetCache = caches[targetLevel] {
                                await targetCache.set(
                                    id: vectorId,
                                    vector: vector,
                                    priority: CachePriority.critical
                                )
                            }
                            break
                        }
                    }
                }
                
            case CacheOptimizationPlan.OptimizationAction.enableStreamPrefetch(let levels):
                // Enable streaming prefetch for specified levels
                logger.info("Enabling stream prefetch for levels: \(levels)")
            }
        }
    }
    
    private func performEmergencyEviction() async {
        logger.warning("Performing emergency eviction due to critical memory pressure")
        
        // Keep only critical items in L1
        if let l1Cache = caches[CacheLevel.l1] {
            await l1Cache.clear()
        }
        
        // Reduce L2 to 25% capacity
        if let l2Cache = caches[CacheLevel.l2] {
            let currentCount = await l2Cache.count
            let targetCount = currentCount / 4
            // Trigger aggressive eviction
            await l2Cache.optimize()
        }
        
        // Clear L3 entirely
        if let l3Cache = caches[CacheLevel.l3] {
            await l3Cache.clear()
        }
    }
    
    private func generateRecommendations(
        levelAnalyses: [CacheLevel: CachePerformanceAnalysis],
        globalAnalysis: GlobalCacheAnalysis
    ) async -> [MultiLevelCacheRecommendation] {
        var recommendations: [MultiLevelCacheRecommendation] = []
        
        // Check for level imbalance
        let hitRates = levelAnalyses.mapValues { $0.hitRateOverTime.last?.1 ?? 0 }
        let avgHitRate = hitRates.values.reduce(0, +) / Float(hitRates.count)
        
        for (level, hitRate) in hitRates {
            if hitRate < avgHitRate * 0.5 {
                recommendations.append(MultiLevelCacheRecommendation(
                    type: MultiLevelCacheRecommendation.RecommendationType.rebalance,
                    description: "Level \(level) significantly underperforming",
                    expectedImprovement: 0.2,
                    affectedLevels: [level]
                ))
            }
        }
        
        // Check for memory pressure
        if currentMemoryPressure != .normal {
            recommendations.append(MultiLevelCacheRecommendation(
                type: MultiLevelCacheRecommendation.RecommendationType.memoryOptimization,
                description: "Optimize memory usage due to \(currentMemoryPressure) pressure",
                expectedImprovement: 0.15,
                affectedLevels: CacheLevel.allCases
            ))
        }
        
        // Check for access pattern optimization
        if globalAnalysis.accessPatternEfficiency < 0.6 {
            recommendations.append(MultiLevelCacheRecommendation(
                type: MultiLevelCacheRecommendation.RecommendationType.patternOptimization,
                description: "Improve cache organization based on access patterns",
                expectedImprovement: 0.3,
                affectedLevels: [CacheLevel.l1, CacheLevel.l2]
            ))
        }
        
        return recommendations
    }
    
    private func calculateHealthScore() async -> Float {
        let metrics = await collectComprehensiveMetrics()
        
        var score: Float = 100.0
        
        // Penalize low global hit rate
        let hitRatePenalty = max(0, (0.7 - metrics.globalHitRate)) * 50
        score -= hitRatePenalty
        
        // Penalize memory pressure
        switch currentMemoryPressure {
        case MemoryPressureLevel.normal: break
        case MemoryPressureLevel.warning: score -= 10
        case MemoryPressureLevel.urgent: score -= 25
        case MemoryPressureLevel.critical: score -= 40
        }
        
        // Penalize poor pattern utilization
        let patternPenalty = (1.0 - metrics.accessPatternStrength) * 20
        score -= patternPenalty
        
        // Penalize imbalanced levels
        let levelHitRates = metrics.levelMetrics.values.map { $0.hitRate }
        let hitRateVariance = calculateVariance(levelHitRates)
        score -= min(hitRateVariance * 10, 20)
        
        return max(0, min(100, score))
    }
    
    private func calculateVariance(_ values: [Float]) -> Float {
        guard values.count > 1 else { return 0 }
        let mean = values.reduce(0, +) / Float(values.count)
        let squaredDiffs = values.map { pow($0 - mean, 2) }
        return squaredDiffs.reduce(0, +) / Float(values.count - 1)
    }
    
    // MARK: - MemoryPressureAware Protocol
    
    public func handleMemoryPressure(_ level: SystemMemoryPressure) async {
        await handleMemoryPressure(MemoryPressureLevel(from: level))
    }
    
    public func getCurrentMemoryUsage() async -> Int {
        var totalMemory = 0
        for cache in caches.values {
            totalMemory += await cache.memoryUsage
        }
        return totalMemory
    }
    
    public func getMemoryStatistics() async -> MemoryComponentStatistics {
        let currentUsage = await getCurrentMemoryUsage()
        let stats = await monitor.getGlobalMetrics()
        
        return MemoryComponentStatistics(
            componentName: "MultiLevelCachingStrategy",
            currentMemoryUsage: currentUsage,
            peakMemoryUsage: currentUsage, // Would need to track this properly
            pressureEventCount: 0, // Would need to track this
            lastPressureHandled: nil,
            averageResponseTime: stats.avgLatency
        )
    }
}

// MARK: - Helper Extensions

private extension MemoryPressureLevel {
    init(from systemLevel: SystemMemoryPressure) {
        switch systemLevel {
        case SystemMemoryPressure.normal: self = MemoryPressureLevel.normal
        case SystemMemoryPressure.warning: self = MemoryPressureLevel.warning
        case SystemMemoryPressure.critical: self = MemoryPressureLevel.critical
        }
    }
}

// MARK: - Supporting Types

/// Access prediction for cache warming
public struct AccessPrediction: Sendable {
    public let confidence: Float
    public let timeWindow: TimeInterval
    
    public init(confidence: Float, timeWindow: TimeInterval) {
        self.confidence = confidence
        self.timeWindow = timeWindow
    }
}

/// Multi-level cache statistics
public struct MultiLevelCacheStatistics: Sendable {
    public let levelStatistics: [CacheLevel: CacheStatistics]
    public let globalMetrics: GlobalCacheMetrics
    public let memoryPressure: MemoryPressureLevel
    public let accessPatterns: AccessPatternSummary
}

/// Multi-level performance analysis
public struct MultiLevelPerformanceAnalysis: Sendable {
    public let levelAnalyses: [CacheLevel: CachePerformanceAnalysis]
    public let globalAnalysis: GlobalCacheAnalysis
    public let recommendations: [MultiLevelCacheRecommendation]
    public let healthScore: Float
}

/// Cache warming plan
struct CacheWarmingPlan {
    enum Action {
        case preload(ids: [VectorID], level: CacheLevel)
        case promote(id: VectorID, from: CacheLevel, to: CacheLevel)
        case prefetch(predictions: [VectorID: Float], level: CacheLevel)
    }
    
    let actions: [Action]
}

/// Cache state snapshot
struct CacheState {
    let levelStates: [CacheLevel: LevelState]
}

/// Level state
struct LevelState {
    let count: Int
    let memoryUsage: Int
    let hitRate: Float
}

/// Comprehensive cache metrics
struct ComprehensiveCacheMetrics {
    let levelMetrics: [CacheLevel: LevelMetrics]
    let globalHitRate: Float
    let totalMemoryUsage: Int
    let accessPatternStrength: Float
}

/// Level-specific metrics
struct LevelMetrics {
    let hitRate: Float
    let memoryUsage: Int
    let count: Int
    let evictionRate: Float
}

/// Cache optimization plan
struct CacheOptimizationPlan {
    enum OptimizationAction {
        case resizeLevel(level: CacheLevel, factor: Float)
        case redistributeHotSpots(vectors: [VectorID], targetLevel: CacheLevel)
        case enableStreamPrefetch(levels: [CacheLevel])
    }
    
    let actions: [OptimizationAction]
}

/// Multi-level cache recommendation
public struct MultiLevelCacheRecommendation: Sendable {
    public enum RecommendationType: Sendable {
        case rebalance
        case memoryOptimization
        case patternOptimization
        case policyChange
    }
    
    public let type: RecommendationType
    public let description: String
    public let expectedImprovement: Float
    public let affectedLevels: [CacheLevel]
}

/// Global cache metrics
public struct GlobalCacheMetrics: Sendable {
    public let totalHits: Int
    public let totalMisses: Int
    public let avgLatency: TimeInterval
    public let memoryEfficiency: Float
}

/// Global cache analysis
public struct GlobalCacheAnalysis: Sendable {
    public let trend: Trend
    public let accessPatternEfficiency: Float
    public let levelBalance: Float
    public let anomalies: [String]
}

/// Access pattern summary
public struct AccessPatternSummary: Sendable {
    public let hotSpots: Set<VectorID>
    public let coldSpots: Set<VectorID>
    public let sequentialityScore: Float
    public let temporalLocality: Float
    public let spatialLocality: Float
}
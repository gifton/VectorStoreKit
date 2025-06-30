// VectorStoreKit: Adaptive Cache Sizer
//
// Dynamic cache size adjustment based on performance and memory pressure

import Foundation
import os.log

// MARK: - Adaptive Cache Sizer

/// Dynamically adjusts cache sizes based on performance metrics
public actor AdaptiveCacheSizer {
    
    // MARK: - Properties
    
    private var configurations: [CacheLevel: CacheLevelConfiguration]
    private var sizeHistory: [CacheLevel: [SizeAdjustment]] = [:]
    private var performanceHistory: [CacheLevel: [PerformanceSnapshot]] = [:]
    
    // Adjustment parameters
    private let minAdjustmentFactor: Float = 0.5
    private let maxAdjustmentFactor: Float = 2.0
    private let adjustmentStepSize: Float = 0.1
    
    // Performance thresholds
    private let targetHitRate: Float = 0.8
    private let minAcceptableHitRate: Float = 0.5
    
    // Logger
    private let logger = Logger(subsystem: "VectorStoreKit", category: "AdaptiveSizer")
    
    // MARK: - Types
    
    private struct SizeAdjustment {
        let timestamp: Date
        let oldSize: Int
        let newSize: Int
        let reason: AdjustmentReason
        let factor: Float
    }
    
    private enum AdjustmentReason {
        case memoryPressure
        case lowHitRate
        case highHitRate
        case rebalancing
        case manual
    }
    
    private struct PerformanceSnapshot {
        let timestamp: Date
        let hitRate: Float
        let memoryUsage: Int
        let evictionRate: Float
    }
    
    // MARK: - Initialization
    
    public init(initialConfigurations: [CacheLevel: CacheLevelConfiguration]) {
        self.configurations = initialConfigurations
        
        // Initialize history tracking
        for level in CacheLevel.allCases {
            sizeHistory[level] = []
            performanceHistory[level] = []
        }
    }
    
    // MARK: - Public Methods
    
    /// Recommend size adjustments based on metrics
    public func recommendSizeAdjustments(
        metrics: ComprehensiveCacheMetrics,
        memoryPressure: MemoryPressureLevel
    ) async -> [SizeRecommendation] {
        var recommendations: [SizeRecommendation] = []
        
        // Record current performance
        for (level, levelMetrics) in metrics.levelMetrics {
            let snapshot = PerformanceSnapshot(
                timestamp: Date(),
                hitRate: levelMetrics.hitRate,
                memoryUsage: levelMetrics.memoryUsage,
                evictionRate: levelMetrics.evictionRate
            )
            performanceHistory[level]?.append(snapshot)
            
            // Keep bounded history
            if performanceHistory[level]?.count ?? 0 > 100 {
                performanceHistory[level]?.removeFirst()
            }
        }
        
        // Handle memory pressure first
        if memoryPressure != .normal {
            recommendations.append(contentsOf: handleMemoryPressure(
                metrics: metrics,
                pressure: memoryPressure
            ))
        }
        
        // Analyze individual level performance
        for (level, levelMetrics) in metrics.levelMetrics {
            let recommendation = analyzeLevel(
                level: level,
                metrics: levelMetrics,
                globalMetrics: metrics
            )
            
            if let rec = recommendation {
                recommendations.append(rec)
            }
        }
        
        // Check for rebalancing opportunities
        if let rebalancing = checkRebalancing(metrics: metrics) {
            recommendations.append(rebalancing)
        }
        
        return recommendations
    }
    
    /// Adjust cache sizes based on a factor
    public func adjustSizes(factor: Float, levels: [CacheLevel]) async {
        for level in levels {
            await adjustSize(level: level, factor: factor)
        }
    }
    
    /// Adjust a specific cache level size
    public func adjustSize(level: CacheLevel, factor: Float) async {
        guard var config = configurations[level] else { return }
        
        let clampedFactor = max(minAdjustmentFactor, min(maxAdjustmentFactor, factor))
        let oldSize = config.maxMemory
        let newSize = Int(Float(oldSize) * clampedFactor)
        
        // Update configuration
        config = CacheLevelConfiguration(
            level: config.level,
            maxMemory: newSize,
            evictionPolicy: config.evictionPolicy,
            accessTimeThreshold: config.accessTimeThreshold,
            warmupEnabled: config.warmupEnabled,
            prefetchEnabled: config.prefetchEnabled
        )
        configurations[level] = config
        
        // Record adjustment
        let adjustment = SizeAdjustment(
            timestamp: Date(),
            oldSize: oldSize,
            newSize: newSize,
            reason: factor < 1.0 ? .memoryPressure : .lowHitRate,
            factor: clampedFactor
        )
        sizeHistory[level]?.append(adjustment)
        
        logger.info("Adjusted \(level.rawValue) cache size: \(oldSize) -> \(newSize) (factor: \(clampedFactor))")
    }
    
    /// Get current configuration for a level
    public func getConfiguration(for level: CacheLevel) async -> CacheLevelConfiguration? {
        configurations[level]
    }
    
    /// Apply a size recommendation
    public func applySizeRecommendation(_ recommendation: SizeRecommendation) async {
        switch recommendation.type {
        case .increase:
            await adjustSize(level: recommendation.level, factor: recommendation.factor)
            
        case .decrease:
            await adjustSize(level: recommendation.level, factor: recommendation.factor)
            
        case .rebalance:
            // Rebalance across multiple levels
            for (level, factor) in recommendation.rebalanceFactors ?? [:] {
                await adjustSize(level: level, factor: factor)
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func handleMemoryPressure(
        metrics: ComprehensiveCacheMetrics,
        pressure: MemoryPressureLevel
    ) -> [SizeRecommendation] {
        var recommendations: [SizeRecommendation] = []
        
        switch pressure {
        case .warning:
            // Reduce L1 and L2 by 20%
            recommendations.append(SizeRecommendation(
                level: .l1,
                type: .decrease,
                factor: 0.8,
                reason: "Memory pressure: warning level"
            ))
            recommendations.append(SizeRecommendation(
                level: .l2,
                type: .decrease,
                factor: 0.8,
                reason: "Memory pressure: warning level"
            ))
            
        case .urgent:
            // Reduce all levels by 40%
            for level in CacheLevel.allCases {
                recommendations.append(SizeRecommendation(
                    level: level,
                    type: .decrease,
                    factor: 0.6,
                    reason: "Memory pressure: urgent level"
                ))
            }
            
        case .critical:
            // Emergency reduction - keep only essential
            for level in CacheLevel.allCases {
                let factor: Float = level == .l1 ? 0.3 : 0.2
                recommendations.append(SizeRecommendation(
                    level: level,
                    type: .decrease,
                    factor: factor,
                    reason: "Memory pressure: critical level"
                ))
            }
            
        case .normal:
            break
        }
        
        return recommendations
    }
    
    private func analyzeLevel(
        level: CacheLevel,
        metrics: LevelMetrics,
        globalMetrics: ComprehensiveCacheMetrics
    ) -> SizeRecommendation? {
        // Skip if no data
        guard metrics.count > 0 else { return nil }
        
        // Check hit rate
        if metrics.hitRate < minAcceptableHitRate {
            // Low hit rate - might need more space
            if metrics.evictionRate > 0.1 {
                // High eviction rate confirms size issue
                return SizeRecommendation(
                    level: level,
                    type: .increase,
                    factor: 1.5,
                    reason: "Low hit rate (\(metrics.hitRate)) with high evictions"
                )
            }
        } else if metrics.hitRate > 0.95 && metrics.memoryUsage < configurations[level]!.maxMemory / 2 {
            // Very high hit rate with low usage - might be oversized
            return SizeRecommendation(
                level: level,
                type: .decrease,
                factor: 0.8,
                reason: "High hit rate with low memory utilization"
            )
        }
        
        // Check for thrashing
        if let history = performanceHistory[level], history.count >= 10 {
            let recentEvictionRates = history.suffix(10).map { $0.evictionRate }
            let avgEvictionRate = recentEvictionRates.reduce(0, +) / Float(recentEvictionRates.count)
            
            if avgEvictionRate > 0.2 {
                return SizeRecommendation(
                    level: level,
                    type: .increase,
                    factor: 1.3,
                    reason: "Cache thrashing detected (high eviction rate)"
                )
            }
        }
        
        return nil
    }
    
    private func checkRebalancing(metrics: ComprehensiveCacheMetrics) -> SizeRecommendation? {
        // Check if cache levels are imbalanced
        let hitRates = metrics.levelMetrics.mapValues { $0.hitRate }
        
        guard !hitRates.isEmpty else { return nil }
        
        let avgHitRate = hitRates.values.reduce(0, +) / Float(hitRates.count)
        var imbalanced = false
        var rebalanceFactors: [CacheLevel: Float] = [:]
        
        for (level, hitRate) in hitRates {
            let deviation = abs(hitRate - avgHitRate)
            
            if deviation > 0.2 {
                imbalanced = true
                
                // Calculate rebalancing factor
                if hitRate < avgHitRate {
                    // This level needs more space
                    rebalanceFactors[level] = 1.0 + (avgHitRate - hitRate)
                } else {
                    // This level can give up some space
                    rebalanceFactors[level] = 1.0 - (hitRate - avgHitRate) * 0.5
                }
            }
        }
        
        if imbalanced {
            return SizeRecommendation(
                level: .l1, // Primary level for reporting
                type: .rebalance,
                factor: 1.0,
                reason: "Cache levels are imbalanced",
                rebalanceFactors: rebalanceFactors
            )
        }
        
        return nil
    }
}

// MARK: - Supporting Types

/// Size adjustment recommendation
public struct SizeRecommendation: Sendable {
    public enum AdjustmentType: Sendable {
        case increase
        case decrease
        case rebalance
    }
    
    public let level: CacheLevel
    public let type: AdjustmentType
    public let factor: Float
    public let reason: String
    public let rebalanceFactors: [CacheLevel: Float]?
    
    public init(
        level: CacheLevel,
        type: AdjustmentType,
        factor: Float,
        reason: String,
        rebalanceFactors: [CacheLevel: Float]? = nil
    ) {
        self.level = level
        self.type = type
        self.factor = factor
        self.reason = reason
        self.rebalanceFactors = rebalanceFactors
    }
}
// VectorStoreKit: Cache Utilities
//
// Shared utilities and helpers for cache implementations

import Foundation
import simd

// MARK: - Cache Storage Integration

/// Protocol for cache storage backends
public protocol CacheStorageBackend<Vector>: Actor {
    associatedtype Vector: SIMD where Vector.Scalar: BinaryFloatingPoint
    
    /// Fetch vectors from storage
    func fetchVectors(ids: [VectorID]) async throws -> [VectorID: Vector]
    
    /// Store vectors to persistent storage
    func storeVectors(_ vectors: [VectorID: Vector]) async throws
    
    /// Check if vectors exist in storage
    func checkExistence(ids: [VectorID]) async -> [VectorID: Bool]
}

// MARK: - Eviction Strategies

/// Base protocol for eviction strategies
public protocol EvictionStrategy: Sendable {
    associatedtype Vector: SIMD where Vector.Scalar: BinaryFloatingPoint
    
    /// Select entries to evict
    func selectForEviction(
        from entries: [VectorID: CacheEntry<Vector>],
        targetSize: Int,
        currentSize: Int
    ) -> [VectorID]
}

/// Priority-based eviction strategy
public struct PriorityEvictionStrategy<Vector: SIMD>: EvictionStrategy 
    where Vector.Scalar: BinaryFloatingPoint {
    
    public func selectForEviction(
        from entries: [VectorID: CacheEntry<Vector>],
        targetSize: Int,
        currentSize: Int
    ) -> [VectorID] {
        guard currentSize > targetSize else { return [] }
        
        let sizeToEvict = currentSize - targetSize
        var evictedSize = 0
        var toEvict: [VectorID] = []
        
        // Sort by priority (lowest first) and timestamp (oldest first)
        let sortedEntries = entries.sorted { lhs, rhs in
            if lhs.value.priority.rawValue == rhs.value.priority.rawValue {
                return lhs.value.timestamp < rhs.value.timestamp
            }
            return lhs.value.priority.rawValue < rhs.value.priority.rawValue
        }
        
        for (id, entry) in sortedEntries {
            if evictedSize >= sizeToEvict { break }
            
            toEvict.append(id)
            evictedSize += estimateVectorMemorySize(entry.vector)
        }
        
        return toEvict
    }
}

// MARK: - Performance Analysis

/// Enhanced performance analyzer for caches
public actor CachePerformanceAnalyzer<Vector: SIMD> where Vector.Scalar: BinaryFloatingPoint {
    private var hitRateHistory: [(Date, Float)] = []
    private var memoryUsageHistory: [(Date, Int)] = []
    private var evictionHistory: [(Date, Int)] = []
    private let maxHistorySize = 1000
    
    public func recordHit() async {
        // Implementation handled by individual caches
    }
    
    public func recordMiss() async {
        // Implementation handled by individual caches
    }
    
    public func recordEviction(count: Int) async {
        let now = Date()
        evictionHistory.append((now, count))
        
        // Trim history if needed
        if evictionHistory.count > maxHistorySize {
            evictionHistory.removeFirst(evictionHistory.count - maxHistorySize)
        }
    }
    
    public func recordMemoryUsage(_ usage: Int) async {
        let now = Date()
        memoryUsageHistory.append((now, usage))
        
        // Trim history if needed
        if memoryUsageHistory.count > maxHistorySize {
            memoryUsageHistory.removeFirst(memoryUsageHistory.count - maxHistorySize)
        }
    }
    
    public func analyze(
        currentHitRate: Float,
        currentMemoryUsage: Int,
        maxMemory: Int
    ) async -> CachePerformanceAnalysis {
        let now = Date()
        
        // Record current hit rate
        hitRateHistory.append((now, currentHitRate))
        if hitRateHistory.count > maxHistorySize {
            hitRateHistory.removeFirst(hitRateHistory.count - maxHistorySize)
        }
        
        // Calculate metrics
        let memoryUtilization = Float(currentMemoryUsage) / Float(maxMemory)
        let evictionRate = calculateEvictionRate()
        let optimalSize = estimateOptimalCacheSize(
            currentSize: maxMemory,
            hitRate: currentHitRate,
            memoryUtilization: memoryUtilization
        )
        
        // Generate recommendations
        let recommendations = generateRecommendations(
            hitRate: currentHitRate,
            memoryUtilization: memoryUtilization,
            evictionRate: evictionRate,
            currentSize: maxMemory,
            optimalSize: optimalSize
        )
        
        return CachePerformanceAnalysis(
            hitRateOverTime: hitRateHistory,
            memoryUtilization: memoryUtilization,
            evictionRate: evictionRate,
            optimalCacheSize: optimalSize,
            recommendations: recommendations
        )
    }
    
    private func calculateEvictionRate() -> Float {
        guard evictionHistory.count > 1 else { return 0.0 }
        
        let recentWindow = Date().addingTimeInterval(-3600) // Last hour
        let recentEvictions = evictionHistory.filter { $0.0 > recentWindow }
        
        guard !recentEvictions.isEmpty else { return 0.0 }
        
        let totalEvictions = recentEvictions.reduce(0) { $0 + $1.1 }
        let timeSpan = recentEvictions.last!.0.timeIntervalSince(recentEvictions.first!.0)
        
        return timeSpan > 0 ? Float(totalEvictions) / Float(timeSpan) : 0.0
    }
    
    private func estimateOptimalCacheSize(
        currentSize: Int,
        hitRate: Float,
        memoryUtilization: Float
    ) -> Int {
        // Simple heuristic for optimal cache size
        if hitRate > 0.9 && memoryUtilization < 0.7 {
            // High hit rate with low utilization - cache might be oversized
            return Int(Float(currentSize) * 0.8)
        } else if hitRate < 0.5 && memoryUtilization > 0.9 {
            // Low hit rate with high utilization - cache might be undersized
            return Int(Float(currentSize) * 1.5)
        } else {
            // Cache size seems appropriate
            return currentSize
        }
    }
    
    private func generateRecommendations(
        hitRate: Float,
        memoryUtilization: Float,
        evictionRate: Float,
        currentSize: Int,
        optimalSize: Int
    ) -> [CacheRecommendation] {
        var recommendations: [CacheRecommendation] = []
        
        // Size adjustment recommendation
        if abs(optimalSize - currentSize) > currentSize / 10 { // More than 10% difference
            let improvement = abs(hitRate - 0.8) * 0.2 // Estimate improvement
            recommendations.append(.sizeAdjustment(
                currentSize: currentSize,
                recommendedSize: optimalSize,
                expectedImprovement: improvement
            ))
        }
        
        // Prefetching recommendation
        if hitRate < 0.6 && evictionRate < 0.1 {
            recommendations.append(.enablePrefetching(expectedImprovement: 0.15))
        }
        
        // Policy change recommendation
        if hitRate < 0.4 {
            recommendations.append(CacheRecommendation(
                type: .policyChange,
                description: "Consider switching to an adaptive eviction policy",
                expectedImprovement: 0.1
            ))
        }
        
        return recommendations
    }
}

// MARK: - Batch Operations

/// Helper for batch cache operations
public struct CacheBatchOperations<Vector: SIMD> where Vector.Scalar: BinaryFloatingPoint {
    
    /// Process batch insertions efficiently
    public static func batchInsert<Strategy: EvictionStrategy>(
        entries: [(VectorID, Vector, CachePriority)],
        into cache: inout [VectorID: CacheEntry<Vector>],
        currentMemoryUsage: inout Int,
        maxMemory: Int,
        evictionStrategy: Strategy
    ) -> (inserted: Int, evicted: Int) where Strategy.Vector == Vector {
        var inserted = 0
        var evicted = 0
        
        // Calculate total size needed
        let totalSizeNeeded = entries.reduce(0) { sum, entry in
            sum + estimateVectorMemorySize(entry.1)
        }
        
        // Check if we need to evict
        if currentMemoryUsage + totalSizeNeeded > maxMemory {
            let targetSize = maxMemory - totalSizeNeeded
            let toEvict = evictionStrategy.selectForEviction(
                from: cache,
                targetSize: targetSize,
                currentSize: currentMemoryUsage
            )
            
            for id in toEvict {
                if let entry = cache.removeValue(forKey: id) {
                    currentMemoryUsage -= estimateVectorMemorySize(entry.vector)
                    evicted += 1
                }
            }
        }
        
        // Insert new entries
        for (id, vector, priority) in entries {
            let entry = CacheEntry(
                vector: vector,
                priority: priority,
                timestamp: Date()
            )
            
            let entrySize = estimateVectorMemorySize(vector)
            if currentMemoryUsage + entrySize <= maxMemory {
                cache[id] = entry
                currentMemoryUsage += entrySize
                inserted += 1
            }
        }
        
        return (inserted, evicted)
    }
}

// MARK: - Cache Warming

/// Helper for cache warming operations
public struct CacheWarming<Vector: SIMD> where Vector.Scalar: BinaryFloatingPoint {
    
    /// Warm cache with frequently accessed vectors
    public static func warmCache(
        with vectors: [(VectorID, Vector, Float)], // (id, vector, access_frequency)
        cache: inout [VectorID: CacheEntry<Vector>],
        currentMemoryUsage: inout Int,
        maxMemory: Int
    ) -> Int {
        // Sort by access frequency
        let sorted = vectors.sorted { $0.2 > $1.2 }
        
        var warmed = 0
        for (id, vector, frequency) in sorted {
            let entrySize = estimateVectorMemorySize(vector)
            
            if currentMemoryUsage + entrySize > maxMemory {
                break
            }
            
            // Determine priority based on frequency
            let priority: CachePriority = frequency > 0.8 ? .critical :
                                         frequency > 0.6 ? .high :
                                         frequency > 0.3 ? .normal : .low
            
            cache[id] = CacheEntry(
                vector: vector,
                priority: priority,
                timestamp: Date()
            )
            
            currentMemoryUsage += entrySize
            warmed += 1
        }
        
        return warmed
    }
}

// MARK: - Predictive Prefetching

/// Predictive prefetching engine for cache optimization
public actor PredictivePrefetchEngine<Vector: SIMD> where Vector.Scalar: BinaryFloatingPoint {
    
    // Access pattern tracking
    private var accessSequences: [[VectorID]] = []
    private var transitionMatrix: [VectorID: [VectorID: Float]] = [:] // From -> To -> Probability
    private var accessFrequency: [VectorID: Int] = [:]
    private var lastAccessTime: [VectorID: Date] = [:]
    
    // Prefetch configuration
    private let maxSequenceLength = 10
    private let maxSequences = 1000
    private let minConfidence: Float = 0.3
    private let maxPrefetchItems = 20
    
    // Current sequence being tracked
    private var currentSequence: [VectorID] = []
    
    public init() {}
    
    /// Record an access for pattern learning
    public func recordAccess(id: VectorID) async {
        // Update frequency and time
        accessFrequency[id, default: 0] += 1
        lastAccessTime[id] = Date()
        
        // Add to current sequence
        currentSequence.append(id)
        
        // Update transition matrix
        if currentSequence.count >= 2 {
            let prev = currentSequence[currentSequence.count - 2]
            let current = id
            
            if transitionMatrix[prev] == nil {
                transitionMatrix[prev] = [:]
            }
            
            let currentCount = transitionMatrix[prev]![current, default: 0]
            transitionMatrix[prev]![current] = currentCount + 1
        }
        
        // Manage sequence length
        if currentSequence.count > maxSequenceLength {
            currentSequence.removeFirst()
        }
        
        // Store completed sequences
        if currentSequence.count == maxSequenceLength {
            accessSequences.append(currentSequence)
            if accessSequences.count > maxSequences {
                accessSequences.removeFirst()
            }
        }
    }
    
    /// Generate prefetch predictions based on current access patterns
    public func generatePredictions(
        currentId: VectorID,
        predictions: [VectorID: Float]
    ) async -> [VectorID: Float] {
        var results: [VectorID: Float] = [:]
        
        // 1. Transition-based predictions
        if let transitions = transitionMatrix[currentId] {
            let total = transitions.values.reduce(0, +)
            
            for (nextId, count) in transitions {
                let probability = count / total
                if probability >= minConfidence {
                    results[nextId] = max(results[nextId, default: 0], probability)
                }
            }
        }
        
        // 2. Sequence pattern matching
        let sequenceResults = await findSequencePatterns(currentId: currentId)
        for (id, confidence) in sequenceResults {
            results[id] = max(results[id, default: 0], confidence)
        }
        
        // 3. Incorporate provided predictions
        for (id, confidence) in predictions {
            results[id] = max(results[id, default: 0], confidence * 0.8) // Slightly discount external predictions
        }
        
        // 4. Boost based on access frequency
        for (id, baseConfidence) in results {
            if let frequency = accessFrequency[id] {
                let frequencyBoost = min(0.2, Float(frequency) / 1000.0)
                results[id] = min(1.0, baseConfidence + frequencyBoost)
            }
        }
        
        // 5. Apply time decay
        let now = Date()
        for (id, confidence) in results {
            if let lastAccess = lastAccessTime[id] {
                let timeSinceAccess = now.timeIntervalSince(lastAccess)
                let decayFactor = exp(-timeSinceAccess / 3600.0) // Decay over hours
                results[id] = confidence * Float(decayFactor)
            }
        }
        
        // Return top predictions
        let sortedResults = results
            .sorted { $0.value > $1.value }
            .prefix(maxPrefetchItems)
            .map { ($0.key, $0.value) }
        
        return Dictionary(uniqueKeysWithValues: sortedResults)
    }
    
    /// Find patterns in access sequences
    private func findSequencePatterns(currentId: VectorID) async -> [VectorID: Float] {
        var predictions: [VectorID: Float] = [:]
        
        // Look for the current ID in historical sequences
        for sequence in accessSequences {
            if let index = sequence.firstIndex(of: currentId),
               index < sequence.count - 1 {
                // Found current ID in a sequence
                let nextId = sequence[index + 1]
                predictions[nextId, default: 0] += 1
                
                // Also look ahead further with decreasing confidence
                for offset in 2..<min(4, sequence.count - index) {
                    let futureId = sequence[index + offset]
                    predictions[futureId, default: 0] += 1.0 / Float(offset)
                }
            }
        }
        
        // Normalize to probabilities
        let total = predictions.values.reduce(0, +)
        if total > 0 {
            for (id, count) in predictions {
                predictions[id] = count / total
            }
        }
        
        return predictions
    }
    
    /// Analyze effectiveness of predictions
    public func analyzeEffectiveness() async -> PrefetchEffectiveness {
        let totalPredictions = transitionMatrix.values
            .flatMap { $0.values }
            .reduce(0, +)
        
        let uniqueItems = Set(accessFrequency.keys).count
        let totalAccesses = accessFrequency.values.reduce(0, +)
        
        let avgSequenceLength = accessSequences.isEmpty ? 0 :
            Float(accessSequences.map { $0.count }.reduce(0, +)) / Float(accessSequences.count)
        
        return PrefetchEffectiveness(
            predictionAccuracy: 0.0, // Would need to track hits/misses
            coverageRatio: Float(uniqueItems) / Float(max(totalAccesses, 1)),
            avgConfidence: totalPredictions > 0 ? Float(totalPredictions) / Float(uniqueItems) : 0,
            patternStrength: avgSequenceLength / Float(maxSequenceLength)
        )
    }
}

/// Prefetch effectiveness metrics
public struct PrefetchEffectiveness: Sendable {
    public let predictionAccuracy: Float
    public let coverageRatio: Float
    public let avgConfidence: Float
    public let patternStrength: Float
}

// MARK: - Enhanced Performance Analysis

/// Enhanced cache performance analyzer with historical tracking and trend analysis
public actor EnhancedCachePerformanceAnalyzer<Vector: SIMD> where Vector.Scalar: BinaryFloatingPoint {
    
    // Detailed metrics tracking
    private struct PerformanceSnapshot {
        let timestamp: Date
        let hitRate: Float
        let missRate: Float
        let evictionRate: Float
        let memoryUsage: Int
        let accessLatency: TimeInterval
        let cacheSize: Int
    }
    
    // Historical data
    private var snapshots: [PerformanceSnapshot] = []
    private var accessLatencies: [TimeInterval] = []
    private var hitMissSequence: [Bool] = [] // true = hit, false = miss
    
    // Anomaly detection
    private var baselineMetrics: BaselineMetrics?
    private let anomalyThreshold: Float = 2.5 // Standard deviations
    
    // Configuration
    private let maxSnapshots = 10000
    private let maxLatencySamples = 1000
    private let analysisWindowSize = 100
    
    // Trend analysis
    private var trendAnalyzer = TrendAnalyzer()
    
    public init() {}
    
    /// Record a performance snapshot
    public func recordSnapshot(
        hitRate: Float,
        missRate: Float,
        evictionRate: Float,
        memoryUsage: Int,
        accessLatency: TimeInterval,
        cacheSize: Int
    ) async {
        let snapshot = PerformanceSnapshot(
            timestamp: Date(),
            hitRate: hitRate,
            missRate: missRate,
            evictionRate: evictionRate,
            memoryUsage: memoryUsage,
            accessLatency: accessLatency,
            cacheSize: cacheSize
        )
        
        snapshots.append(snapshot)
        if snapshots.count > maxSnapshots {
            snapshots.removeFirst(snapshots.count - maxSnapshots)
        }
        
        // Update baseline if needed
        if baselineMetrics == nil && snapshots.count > 100 {
            await establishBaseline()
        }
    }
    
    /// Record individual access latency
    public func recordAccessLatency(_ latency: TimeInterval) async {
        accessLatencies.append(latency)
        if accessLatencies.count > maxLatencySamples {
            accessLatencies.removeFirst()
        }
    }
    
    /// Record hit/miss sequence
    public func recordAccess(hit: Bool) async {
        hitMissSequence.append(hit)
        if hitMissSequence.count > maxSnapshots {
            hitMissSequence.removeFirst()
        }
    }
    
    /// Perform comprehensive analysis
    public func performComprehensiveAnalysis(
        currentHitRate: Float,
        currentMemoryUsage: Int,
        maxMemory: Int,
        cacheType: String
    ) async -> EnhancedCacheAnalysis {
        // Record current state
        await recordSnapshot(
            hitRate: currentHitRate,
            missRate: 1.0 - currentHitRate,
            evictionRate: calculateRecentEvictionRate(),
            memoryUsage: currentMemoryUsage,
            accessLatency: calculateAverageLatency(),
            cacheSize: maxMemory
        )
        
        // Perform analyses
        let trends = await analyzeTrends()
        let anomalies = await detectAnomalies()
        let patterns = await analyzeAccessPatterns()
        let forecast = await forecastPerformance()
        let recommendations = await generateAdvancedRecommendations(
            trends: trends,
            anomalies: anomalies,
            patterns: patterns,
            currentMemoryUsage: currentMemoryUsage,
            maxMemory: maxMemory,
            cacheType: cacheType
        )
        
        return EnhancedCacheAnalysis(
            currentMetrics: getCurrentMetrics(),
            historicalTrends: trends,
            anomalies: anomalies,
            accessPatterns: patterns,
            forecast: forecast,
            recommendations: recommendations,
            healthScore: calculateHealthScore()
        )
    }
    
    // MARK: - Private Analysis Methods
    
    private func establishBaseline() async {
        guard snapshots.count >= 100 else { return }
        
        let recentSnapshots = Array(snapshots.suffix(100))
        
        baselineMetrics = BaselineMetrics(
            avgHitRate: recentSnapshots.map { $0.hitRate }.reduce(0, +) / Float(recentSnapshots.count),
            stdDevHitRate: calculateStandardDeviation(recentSnapshots.map { $0.hitRate }),
            avgLatency: recentSnapshots.map { $0.accessLatency }.reduce(0, +) / Double(recentSnapshots.count),
            stdDevLatency: calculateStandardDeviation(recentSnapshots.map { Float($0.accessLatency) }),
            avgMemoryUsage: recentSnapshots.map { $0.memoryUsage }.reduce(0, +) / recentSnapshots.count,
            stdDevMemoryUsage: calculateStandardDeviation(recentSnapshots.map { Float($0.memoryUsage) })
        )
    }
    
    private func analyzeTrends() async -> TrendAnalysis {
        guard snapshots.count > analysisWindowSize else {
            return TrendAnalysis(hitRateTrend: .stable, memoryTrend: .stable, performanceTrend: .stable)
        }
        
        let recentSnapshots = Array(snapshots.suffix(analysisWindowSize))
        
        // Analyze hit rate trend
        let hitRates = recentSnapshots.map { $0.hitRate }
        let hitRateTrend = trendAnalyzer.analyzeTrend(values: hitRates)
        
        // Analyze memory trend
        let memoryUsages = recentSnapshots.map { Float($0.memoryUsage) }
        let memoryTrend = trendAnalyzer.analyzeTrend(values: memoryUsages)
        
        // Analyze performance trend (based on latency)
        let latencies = recentSnapshots.map { Float($0.accessLatency) }
        let performanceTrend = trendAnalyzer.analyzeTrend(values: latencies, inverted: true)
        
        return TrendAnalysis(
            hitRateTrend: hitRateTrend,
            memoryTrend: memoryTrend,
            performanceTrend: performanceTrend
        )
    }
    
    private func detectAnomalies() async -> [PerformanceAnomaly] {
        guard let baseline = baselineMetrics, snapshots.count > 10 else { return [] }
        
        var anomalies: [PerformanceAnomaly] = []
        let recentSnapshots = Array(snapshots.suffix(10))
        
        for snapshot in recentSnapshots {
            // Check hit rate anomaly
            let hitRateDeviation = abs(snapshot.hitRate - baseline.avgHitRate) / baseline.stdDevHitRate
            if hitRateDeviation > anomalyThreshold {
                anomalies.append(PerformanceAnomaly(
                    timestamp: snapshot.timestamp,
                    type: .hitRateAnomaly,
                    severity: hitRateDeviation > anomalyThreshold * 2 ? .high : .medium,
                    description: "Hit rate deviation: \(hitRateDeviation) std devs"
                ))
            }
            
            // Check latency anomaly
            let latencyDeviation = abs(Float(snapshot.accessLatency) - Float(baseline.avgLatency)) / baseline.stdDevLatency
            if latencyDeviation > anomalyThreshold {
                anomalies.append(PerformanceAnomaly(
                    timestamp: snapshot.timestamp,
                    type: .latencyAnomaly,
                    severity: latencyDeviation > anomalyThreshold * 2 ? .high : .medium,
                    description: "Latency deviation: \(latencyDeviation) std devs"
                ))
            }
        }
        
        return anomalies
    }
    
    private func analyzeAccessPatterns() async -> CacheAccessPatternAnalysis {
        guard hitMissSequence.count > 100 else {
            return CacheAccessPatternAnalysis(
                sequentialityScore: 0,
                burstiness: 0,
                periodicityScore: 0,
                workingSetSize: 0
            )
        }
        
        // Calculate sequentiality (consecutive hits)
        var consecutiveHits = 0
        var maxConsecutive = 0
        for hit in hitMissSequence {
            if hit {
                consecutiveHits += 1
                maxConsecutive = max(maxConsecutive, consecutiveHits)
            } else {
                consecutiveHits = 0
            }
        }
        let sequentialityScore = Float(maxConsecutive) / Float(hitMissSequence.count)
        
        // Calculate burstiness
        let windowSize = 10
        var accessCounts: [Int] = []
        for i in stride(from: 0, to: hitMissSequence.count - windowSize, by: windowSize) {
            let window = hitMissSequence[i..<i+windowSize]
            accessCounts.append(window.filter { $0 }.count)
        }
        let avgAccess = Float(accessCounts.reduce(0, +)) / Float(accessCounts.count)
        let variance = accessCounts.map { pow(Float($0) - avgAccess, 2) }.reduce(0, +) / Float(accessCounts.count)
        let burstiness = sqrt(variance) / max(avgAccess, 1)
        
        return CacheAccessPatternAnalysis(
            sequentialityScore: sequentialityScore,
            burstiness: burstiness,
            periodicityScore: 0, // Would need FFT for proper periodicity
            workingSetSize: estimateWorkingSetSize()
        )
    }
    
    private func forecastPerformance() async -> PerformanceForecast {
        guard snapshots.count > 50 else {
            return PerformanceForecast(
                predictedHitRate: snapshots.last?.hitRate ?? 0,
                predictedMemoryUsage: snapshots.last?.memoryUsage ?? 0,
                confidence: 0.0
            )
        }
        
        // Simple linear regression for hit rate
        let recentHitRates = snapshots.suffix(50).map { $0.hitRate }
        let prediction = trendAnalyzer.predictNextValue(values: recentHitRates)
        
        // Memory usage prediction
        let recentMemory = snapshots.suffix(50).map { Float($0.memoryUsage) }
        let memoryPrediction = trendAnalyzer.predictNextValue(values: recentMemory)
        
        return PerformanceForecast(
            predictedHitRate: prediction.value,
            predictedMemoryUsage: Int(memoryPrediction.value),
            confidence: min(prediction.confidence, memoryPrediction.confidence)
        )
    }
    
    private func generateAdvancedRecommendations(
        trends: TrendAnalysis,
        anomalies: [PerformanceAnomaly],
        patterns: CacheAccessPatternAnalysis,
        currentMemoryUsage: Int,
        maxMemory: Int,
        cacheType: String
    ) async -> [CacheRecommendation] {
        var recommendations: [CacheRecommendation] = []
        
        // Trend-based recommendations
        if trends.hitRateTrend == .declining {
            recommendations.append(CacheRecommendation(
                type: .sizeAdjustment,
                description: "Hit rate declining - consider increasing cache size by 50%",
                expectedImprovement: 0.15
            ))
        }
        
        // Anomaly-based recommendations
        if anomalies.contains(where: { $0.type == .latencyAnomaly && $0.severity == .high }) {
            recommendations.append(CacheRecommendation(
                type: .indexOptimization,
                description: "High latency anomalies detected - optimize internal structures",
                expectedImprovement: 0.2
            ))
        }
        
        // Pattern-based recommendations
        if patterns.burstiness > 2.0 {
            recommendations.append(CacheRecommendation(
                type: .prefetching,
                description: "Bursty access pattern detected - enable aggressive prefetching",
                expectedImprovement: 0.25
            ))
        }
        
        if patterns.sequentialityScore > 0.7 {
            recommendations.append(CacheRecommendation(
                type: .policyChange,
                description: "Sequential access pattern - consider FIFO policy",
                expectedImprovement: 0.1
            ))
        }
        
        // Memory efficiency
        let memoryUtilization = Float(currentMemoryUsage) / Float(maxMemory)
        if memoryUtilization < 0.5 && trends.hitRateTrend != .improving {
            recommendations.append(CacheRecommendation(
                type: .sizeAdjustment,
                description: "Low memory utilization - reduce cache size to improve efficiency",
                expectedImprovement: 0.05
            ))
        }
        
        return recommendations
    }
    
    // MARK: - Helper Methods
    
    private func calculateRecentEvictionRate() -> Float {
        // Simplified - would need actual eviction tracking
        return 0.0
    }
    
    private func calculateAverageLatency() -> TimeInterval {
        guard !accessLatencies.isEmpty else { return 0 }
        return accessLatencies.reduce(0, +) / Double(accessLatencies.count)
    }
    
    private func calculateStandardDeviation(_ values: [Float]) -> Float {
        guard values.count > 1 else { return 0 }
        let mean = values.reduce(0, +) / Float(values.count)
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Float(values.count - 1)
        return sqrt(variance)
    }
    
    private func estimateWorkingSetSize() -> Int {
        // Estimate based on hit rate patterns
        guard !snapshots.isEmpty else { return 0 }
        let avgHitRate = snapshots.suffix(50).map { $0.hitRate }.reduce(0, +) / Float(min(snapshots.count, 50))
        let avgCacheSize = snapshots.suffix(50).map { $0.cacheSize }.reduce(0, +) / min(snapshots.count, 50)
        return Int(Float(avgCacheSize) * avgHitRate)
    }
    
    private func getCurrentMetrics() -> CurrentCacheMetrics {
        let recent = snapshots.last
        return CurrentCacheMetrics(
            hitRate: recent?.hitRate ?? 0,
            missRate: recent?.missRate ?? 0,
            evictionRate: recent?.evictionRate ?? 0,
            avgLatency: calculateAverageLatency(),
            memoryUsage: recent?.memoryUsage ?? 0
        )
    }
    
    private func calculateHealthScore() -> Float {
        guard let recent = snapshots.last else { return 0 }
        
        var score: Float = 100.0
        
        // Deduct for low hit rate
        if recent.hitRate < 0.5 {
            score -= (0.5 - recent.hitRate) * 50
        }
        
        // Deduct for high latency
        if recent.accessLatency > 0.001 { // 1ms threshold
            score -= Float(recent.accessLatency * 10000)
        }
        
        // Deduct for high eviction rate
        if recent.evictionRate > 0.1 {
            score -= recent.evictionRate * 20
        }
        
        return max(0, min(100, score))
    }
}

// MARK: - Supporting Types

private struct BaselineMetrics {
    let avgHitRate: Float
    let stdDevHitRate: Float
    let avgLatency: TimeInterval
    let stdDevLatency: Float
    let avgMemoryUsage: Int
    let stdDevMemoryUsage: Float
}

private struct TrendAnalyzer {
    func analyzeTrend(values: [Float], inverted: Bool = false) -> Trend {
        guard values.count > 2 else { return .stable }
        
        // Simple linear regression
        let n = Float(values.count)
        let indices = (0..<values.count).map { Float($0) }
        
        let sumX = indices.reduce(0, +)
        let sumY = values.reduce(0, +)
        let sumXY = zip(indices, values).map { $0 * $1 }.reduce(0, +)
        let sumXX = indices.map { $0 * $0 }.reduce(0, +)
        
        let slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
        
        let threshold: Float = 0.01
        
        if inverted {
            if slope < -threshold { return .improving }
            if slope > threshold { return .declining }
        } else {
            if slope > threshold { return .improving }
            if slope < -threshold { return .declining }
        }
        
        return .stable
    }
    
    func predictNextValue(values: [Float]) -> (value: Float, confidence: Float) {
        guard values.count > 2 else {
            return (values.last ?? 0, 0.0)
        }
        
        // Simple linear extrapolation
        let recent = Array(values.suffix(10))
        let avg = recent.reduce(0, +) / Float(recent.count)
        let trend = analyzeTrend(values: recent)
        
        var prediction = avg
        if trend == .improving {
            prediction *= 1.05
        } else if trend == .declining {
            prediction *= 0.95
        }
        
        // Confidence based on variance
        let variance = recent.map { pow($0 - avg, 2) }.reduce(0, +) / Float(recent.count)
        let confidence = 1.0 / (1.0 + variance)
        
        return (prediction, confidence)
    }
}

public enum Trend: Sendable {
    case improving
    case stable
    case declining
}

public struct TrendAnalysis: Sendable {
    public let hitRateTrend: Trend
    public let memoryTrend: Trend
    public let performanceTrend: Trend
}

public struct PerformanceAnomaly: Sendable {
    public let timestamp: Date
    public let type: AnomalyType
    public let severity: Severity
    public let description: String
}

public enum AnomalyType: Sendable {
    case hitRateAnomaly
    case latencyAnomaly
    case memoryAnomaly
    case evictionAnomaly
}

public struct CacheAccessPatternAnalysis: Sendable {
    public let sequentialityScore: Float
    public let burstiness: Float
    public let periodicityScore: Float
    public let workingSetSize: Int
}

public struct PerformanceForecast: Sendable {
    public let predictedHitRate: Float
    public let predictedMemoryUsage: Int
    public let confidence: Float
}

public struct CurrentCacheMetrics: Sendable {
    public let hitRate: Float
    public let missRate: Float
    public let evictionRate: Float
    public let avgLatency: TimeInterval
    public let memoryUsage: Int
}

public struct EnhancedCacheAnalysis: Sendable {
    public let currentMetrics: CurrentCacheMetrics
    public let historicalTrends: TrendAnalysis
    public let anomalies: [PerformanceAnomaly]
    public let accessPatterns: CacheAccessPatternAnalysis
    public let forecast: PerformanceForecast
    public let recommendations: [CacheRecommendation]
    public let healthScore: Float
}
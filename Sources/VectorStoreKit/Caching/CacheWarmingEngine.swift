// VectorStoreKit: Cache Warming Engine
//
// Intelligent cache warming based on access patterns and predictions

import Foundation
import simd

// MARK: - Cache Warming Engine

/// Engine for intelligent cache warming strategies
public actor CacheWarmingEngine<Vector: SIMD> where Vector.Scalar: BinaryFloatingPoint {
    
    // MARK: - Properties
    
    private var accessHistory: [VectorID: AccessHistory] = [:]
    private var correlationMatrix: [VectorID: [VectorID: Float]] = [:]
    private var temporalPatterns: TemporalPatternDetector
    private var spatialPatterns: SpatialPatternDetector
    
    // Configuration
    private let maxHistorySize = 1000
    private let correlationThreshold: Float = 0.3
    private let confidenceThreshold: Float = 0.5
    
    // MARK: - Types
    
    private struct AccessHistory {
        var timestamps: [Date] = []
        var levels: [CacheLevel] = []
        var contexts: [String] = []
        
        mutating func record(timestamp: Date, level: CacheLevel, context: String = "") {
            timestamps.append(timestamp)
            levels.append(level)
            contexts.append(context)
            
            // Keep bounded history
            if timestamps.count > 100 {
                timestamps.removeFirst()
                levels.removeFirst()
                contexts.removeFirst()
            }
        }
    }
    
    // MARK: - Initialization
    
    public init() {
        self.temporalPatterns = TemporalPatternDetector()
        self.spatialPatterns = SpatialPatternDetector()
    }
    
    // MARK: - Public Methods
    
    /// Create a warming plan based on predictions and current state
    public func createWarmingPlan(
        predictions: [VectorID: AccessPrediction],
        currentState: CacheState,
        memoryPressure: MemoryPressureLevel
    ) async -> CacheWarmingPlan {
        var actions: [CacheWarmingPlan.Action] = []
        
        // Analyze memory constraints
        let availableMemory = calculateAvailableMemory(
            currentState: currentState,
            memoryPressure: memoryPressure
        )
        
        // Sort predictions by confidence and time window
        let sortedPredictions = predictions
            .sorted { 
                let score1 = $0.value.confidence / Float($0.value.timeWindow)
                let score2 = $1.value.confidence / Float($1.value.timeWindow)
                return score1 > score2
            }
        
        var memoryUsed = 0
        
        // Process predictions
        for (vectorId, prediction) in sortedPredictions {
            guard prediction.confidence >= confidenceThreshold else { continue }
            
            // Check if already cached
            if isAlreadyCached(id: vectorId, state: currentState) {
                // Consider promotion if in lower tier
                if let promotionAction = considerPromotion(
                    id: vectorId,
                    state: currentState,
                    prediction: prediction
                ) {
                    actions.append(promotionAction)
                }
            } else {
                // Determine target level
                let targetLevel = determineTargetLevel(
                    prediction: prediction,
                    memoryPressure: memoryPressure
                )
                
                // Estimate memory requirement
                let estimatedSize = estimateVectorSize()
                
                if memoryUsed + estimatedSize <= availableMemory {
                    // Find correlated vectors
                    let correlated = await findCorrelatedVectors(
                        id: vectorId,
                        limit: 5,
                        minConfidence: 0.5
                    )
                    
                    // Add preload action
                    actions.append(.preload(
                        ids: [vectorId] + correlated,
                        level: targetLevel
                    ))
                    
                    memoryUsed += estimatedSize * (1 + correlated.count)
                }
            }
            
            // Check memory limit
            if memoryUsed >= availableMemory {
                break
            }
        }
        
        // Add temporal pattern-based prefetching
        let temporalPrefetch = await createTemporalPrefetch(
            currentState: currentState,
            remainingMemory: availableMemory - memoryUsed
        )
        actions.append(contentsOf: temporalPrefetch)
        
        return CacheWarmingPlan(actions: actions)
    }
    
    /// Record access for pattern learning
    public func recordAccess(
        id: VectorID,
        level: CacheLevel,
        timestamp: Date,
        context: String = ""
    ) async {
        // Update access history
        if accessHistory[id] == nil {
            accessHistory[id] = AccessHistory()
        }
        accessHistory[id]?.record(
            timestamp: timestamp,
            level: level,
            context: context
        )
        
        // Update correlation matrix
        await updateCorrelations(id: id)
        
        // Update pattern detectors
        await temporalPatterns.recordAccess(id: id, timestamp: timestamp)
        await spatialPatterns.recordAccess(id: id, context: context)
        
        // Prune old history
        if accessHistory.count > maxHistorySize {
            await pruneOldHistory()
        }
    }
    
    /// Find vectors correlated with the given ID
    public func findCorrelatedVectors(
        id: VectorID,
        limit: Int,
        minConfidence: Float
    ) async -> [VectorID] {
        guard let correlations = correlationMatrix[id] else { return [] }
        
        return correlations
            .filter { $0.value >= minConfidence }
            .sorted { $0.value > $1.value }
            .prefix(limit)
            .map { $0.key }
    }
    
    /// Analyze warming effectiveness
    public func analyzeEffectiveness(
        warmingActions: [CacheWarmingPlan.Action],
        actualAccesses: [VectorID]
    ) async -> WarmingEffectiveness {
        let warmedIds = extractWarmedIds(from: warmingActions)
        let actualSet = Set(actualAccesses)
        
        let hits = warmedIds.intersection(actualSet).count
        let misses = actualSet.subtracting(warmedIds).count
        let wasted = warmedIds.subtracting(actualSet).count
        
        let precision = warmedIds.isEmpty ? 0 : Float(hits) / Float(warmedIds.count)
        let recall = actualSet.isEmpty ? 0 : Float(hits) / Float(actualSet.count)
        
        return WarmingEffectiveness(
            precision: precision,
            recall: recall,
            wastedPrefetches: wasted,
            missedOpportunities: misses
        )
    }
    
    // MARK: - Private Methods
    
    private func calculateAvailableMemory(
        currentState: CacheState,
        memoryPressure: MemoryPressureLevel
    ) -> Int {
        let totalMemory = currentState.levelStates.values
            .map { $0.memoryUsage }
            .reduce(0, +)
        
        // Adjust based on memory pressure
        switch memoryPressure {
        case .normal:
            return totalMemory / 4  // Use 25% for warming
        case .warning:
            return totalMemory / 10 // Use 10% for warming
        case .urgent, .critical:
            return 0  // No warming under pressure
        }
    }
    
    private func isAlreadyCached(id: VectorID, state: CacheState) -> Bool {
        // This would need actual implementation to check cache state
        false
    }
    
    private func considerPromotion(
        id: VectorID,
        state: CacheState,
        prediction: AccessPrediction
    ) -> CacheWarmingPlan.Action? {
        // Simplified promotion logic
        if prediction.confidence > 0.8 {
            return .promote(id: id, from: .l3, to: .l2)
        }
        return nil
    }
    
    private func determineTargetLevel(
        prediction: AccessPrediction,
        memoryPressure: MemoryPressureLevel
    ) -> CacheLevel {
        switch memoryPressure {
        case .normal:
            if prediction.confidence > 0.9 && prediction.timeWindow < 30 {
                return .l1
            } else if prediction.confidence > 0.7 {
                return .l2
            }
            return .l3
            
        case .warning:
            return prediction.confidence > 0.8 ? .l2 : .l3
            
        case .urgent, .critical:
            return .l3
        }
    }
    
    private func estimateVectorSize() -> Int {
        // Estimate based on Vector type
        MemoryLayout<Vector>.size + 128  // Include metadata overhead
    }
    
    private func updateCorrelations(id: VectorID) async {
        // Look for access patterns in recent history
        let recentWindow: TimeInterval = 60  // 1 minute window
        let now = Date()
        
        // Find all vectors accessed near this one
        for (otherId, history) in accessHistory {
            guard otherId != id else { continue }
            
            // Count co-occurrences within time window
            let coOccurrences = history.timestamps.filter { timestamp in
                abs(timestamp.timeIntervalSince(now)) < recentWindow
            }.count
            
            if coOccurrences > 0 {
                // Update correlation score
                if correlationMatrix[id] == nil {
                    correlationMatrix[id] = [:]
                }
                
                let currentScore = correlationMatrix[id]?[otherId] ?? 0
                let newScore = (currentScore * 0.9) + (Float(coOccurrences) * 0.1)
                correlationMatrix[id]?[otherId] = min(1.0, newScore)
            }
        }
    }
    
    private func createTemporalPrefetch(
        currentState: CacheState,
        remainingMemory: Int
    ) async -> [CacheWarmingPlan.Action] {
        var actions: [CacheWarmingPlan.Action] = []
        
        // Get temporal predictions
        let predictions = await temporalPatterns.getPredictions(
            currentTime: Date(),
            limit: 20
        )
        
        var memoryUsed = 0
        let vectorSize = estimateVectorSize()
        
        for prediction in predictions {
            if memoryUsed + vectorSize > remainingMemory {
                break
            }
            
            // Create prefetch action
            let confidence = prediction.confidence
            let targetLevel: CacheLevel = confidence > 0.7 ? .l2 : .l3
            
            actions.append(.prefetch(
                predictions: [prediction.id: confidence],
                level: targetLevel
            ))
            
            memoryUsed += vectorSize
        }
        
        return actions
    }
    
    private func pruneOldHistory() async {
        // Remove oldest entries
        let sortedIds = accessHistory
            .compactMap { entry -> (VectorID, Date)? in
                guard let lastAccess = entry.value.timestamps.last else { return nil }
                return (entry.key, lastAccess)
            }
            .sorted { $0.1 < $1.1 }
        
        let toRemove = sortedIds.prefix(maxHistorySize / 10).map { $0.0 }
        
        for id in toRemove {
            accessHistory.removeValue(forKey: id)
            correlationMatrix.removeValue(forKey: id)
            
            // Also remove from correlation targets
            for (_, correlations) in correlationMatrix {
                correlations.removeValue(forKey: id)
            }
        }
    }
    
    private func extractWarmedIds(
        from actions: [CacheWarmingPlan.Action]
    ) -> Set<VectorID> {
        var ids = Set<VectorID>()
        
        for action in actions {
            switch action {
            case .preload(let preloadIds, _):
                ids.formUnion(preloadIds)
            case .promote(let id, _, _):
                ids.insert(id)
            case .prefetch(let predictions, _):
                ids.formUnion(predictions.keys)
            }
        }
        
        return ids
    }
}

// MARK: - Pattern Detectors

/// Temporal pattern detection for time-based prefetching
private actor TemporalPatternDetector {
    private var accessTimeSeries: [VectorID: [Date]] = [:]
    private var patterns: [TemporalPattern] = []
    
    struct TemporalPattern {
        let vectorId: VectorID
        let period: TimeInterval?  // For periodic patterns
        let timeOfDay: TimeInterval?  // For daily patterns
        let confidence: Float
    }
    
    func recordAccess(id: VectorID, timestamp: Date) async {
        if accessTimeSeries[id] == nil {
            accessTimeSeries[id] = []
        }
        accessTimeSeries[id]?.append(timestamp)
        
        // Detect patterns periodically
        if accessTimeSeries[id]?.count ?? 0 > 10 {
            await detectPatterns(for: id)
        }
    }
    
    func getPredictions(currentTime: Date, limit: Int) async -> [(id: VectorID, confidence: Float)] {
        var predictions: [(VectorID, Float)] = []
        
        for pattern in patterns {
            let confidence = calculateConfidence(
                pattern: pattern,
                currentTime: currentTime
            )
            
            if confidence > 0.3 {
                predictions.append((pattern.vectorId, confidence))
            }
        }
        
        return Array(predictions
            .sorted { $0.1 > $1.1 }
            .prefix(limit))
    }
    
    private func detectPatterns(for id: VectorID) async {
        guard let timestamps = accessTimeSeries[id],
              timestamps.count >= 10 else { return }
        
        // Simple periodicity detection
        let intervals = zip(timestamps.dropFirst(), timestamps)
            .map { $0.0.timeIntervalSince($0.1) }
        
        if let period = detectPeriod(in: intervals) {
            patterns.append(TemporalPattern(
                vectorId: id,
                period: period,
                timeOfDay: nil,
                confidence: 0.7
            ))
        }
    }
    
    private func detectPeriod(in intervals: [TimeInterval]) -> TimeInterval? {
        guard intervals.count > 2 else { return nil }
        
        let mean = intervals.reduce(0, +) / Double(intervals.count)
        let variance = intervals
            .map { pow($0 - mean, 2) }
            .reduce(0, +) / Double(intervals.count)
        
        // Low variance indicates periodicity
        if variance / mean < 0.2 {
            return mean
        }
        
        return nil
    }
    
    private func calculateConfidence(
        pattern: TemporalPattern,
        currentTime: Date
    ) -> Float {
        // Simplified confidence calculation
        pattern.confidence
    }
}

/// Spatial pattern detection for context-based prefetching
private actor SpatialPatternDetector {
    private var contextMap: [String: Set<VectorID>] = [:]
    private var transitionProbabilities: [String: [String: Float]] = [:]
    
    func recordAccess(id: VectorID, context: String) async {
        if contextMap[context] == nil {
            contextMap[context] = []
        }
        contextMap[context]?.insert(id)
    }
    
    func getRelatedVectors(context: String, limit: Int) async -> [VectorID] {
        guard let vectors = contextMap[context] else { return [] }
        return Array(vectors.prefix(limit))
    }
}

// MARK: - Supporting Types

/// Warming effectiveness metrics
public struct WarmingEffectiveness: Sendable {
    public let precision: Float       // Warmed vectors that were accessed
    public let recall: Float          // Accessed vectors that were warmed
    public let wastedPrefetches: Int  // Warmed but not accessed
    public let missedOpportunities: Int  // Accessed but not warmed
    
    public var f1Score: Float {
        guard precision + recall > 0 else { return 0 }
        return 2 * (precision * recall) / (precision + recall)
    }
}
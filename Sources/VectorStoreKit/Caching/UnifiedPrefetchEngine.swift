// VectorStoreKit: Unified Predictive Prefetch Engine
//
// Shared prefetch prediction engine for all cache types

import Foundation
import os

// MARK: - Unified Prefetch Engine

/// Shared predictive prefetch engine that generates predictions for all cache types
public actor UnifiedPrefetchEngine<Vector: SIMD & Sendable> 
where Vector.Scalar: BinaryFloatingPoint {
    
    // MARK: - Types
    
    public struct PrefetchPrediction: Sendable {
        public let vectorId: VectorID
        public let confidence: Float
        public let reason: PredictionReason
        public let suggestedCacheType: CacheType?
    }
    
    public enum PredictionReason: String, Sendable {
        case sequential = "sequential_access"
        case temporal = "temporal_locality"
        case frequency = "high_frequency"
        case relatedContent = "related_content"
        case userPattern = "user_pattern"
    }
    
    public enum CacheType: String, Sendable {
        case lru = "lru"
        case lfu = "lfu"
        case fifo = "fifo"
    }
    
    // MARK: - State
    
    /// Access history with timestamps
    private var accessHistory: [(id: VectorID, timestamp: ContinuousClock.Instant)] = []
    private let maxHistorySize = 10_000
    
    /// Access frequency tracking
    private var accessFrequency: [VectorID: Int] = [:]
    
    /// Sequential access detection
    private var sequentialPatterns: [SequentialPattern] = []
    
    /// Related item graph (simplified)
    private var relatedItems: [VectorID: Set<VectorID>] = [:]
    
    /// Performance metrics
    private var predictionStats = PredictionStatistics()
    
    /// Configuration
    private let config: PrefetchConfiguration
    
    // MARK: - Initialization
    
    public init(config: PrefetchConfiguration = .default) {
        self.config = config
    }
    
    // MARK: - Access Recording
    
    /// Record a vector access from any cache
    public func recordAccess(id: VectorID, from cacheType: CacheType? = nil) async {
        let timestamp = ContinuousClock.now
        
        // Update history
        accessHistory.append((id, timestamp))
        if accessHistory.count > maxHistorySize {
            accessHistory.removeFirst(accessHistory.count - maxHistorySize / 2)
        }
        
        // Update frequency
        accessFrequency[id, default: 0] += 1
        
        // Detect sequential patterns
        detectSequentialPattern(id: id, timestamp: timestamp)
        
        // Update related items
        updateRelatedItems(id: id)
        
        // Track which cache type is accessing
        if let cacheType = cacheType {
            predictionStats.accessesByCache[cacheType.rawValue, default: 0] += 1
        }
    }
    
    // MARK: - Prediction Generation
    
    /// Generate prefetch predictions based on current access patterns
    public func generatePredictions(
        currentId: VectorID? = nil,
        requestedPredictions: [VectorID: Float] = [:],
        excludeIds: Set<VectorID> = []
    ) async -> [VectorID: PrefetchPrediction] {
        var predictions: [VectorID: PrefetchPrediction] = [:]
        
        // 1. Sequential predictions
        let sequentialPreds = generateSequentialPredictions(currentId: currentId)
        for pred in sequentialPreds {
            if !excludeIds.contains(pred.vectorId) {
                predictions[pred.vectorId] = pred
            }
        }
        
        // 2. Temporal locality predictions
        let temporalPreds = generateTemporalPredictions(currentId: currentId)
        for pred in temporalPreds {
            if !excludeIds.contains(pred.vectorId) {
                // Merge with higher confidence
                if let existing = predictions[pred.vectorId] {
                    predictions[pred.vectorId] = pred.confidence > existing.confidence ? pred : existing
                } else {
                    predictions[pred.vectorId] = pred
                }
            }
        }
        
        // 3. Frequency-based predictions
        let frequencyPreds = generateFrequencyPredictions()
        for pred in frequencyPreds {
            if !excludeIds.contains(pred.vectorId) {
                predictions[pred.vectorId] = predictions[pred.vectorId] ?? pred
            }
        }
        
        // 4. Related content predictions
        if let currentId = currentId {
            let relatedPreds = generateRelatedPredictions(for: currentId)
            for pred in relatedPreds {
                if !excludeIds.contains(pred.vectorId) {
                    predictions[pred.vectorId] = predictions[pred.vectorId] ?? pred
                }
            }
        }
        
        // 5. Merge with requested predictions
        for (id, confidence) in requestedPredictions {
            if !excludeIds.contains(id) {
                let pred = PrefetchPrediction(
                    vectorId: id,
                    confidence: confidence,
                    reason: .userPattern,
                    suggestedCacheType: determineCacheType(for: id)
                )
                predictions[id] = pred
            }
        }
        
        // Limit to top predictions
        let topPredictions = predictions.values
            .sorted { $0.confidence > $1.confidence }
            .prefix(config.maxPredictions)
        
        return Dictionary(uniqueKeysWithValues: topPredictions.map { ($0.vectorId, $0) })
    }
    
    /// Get shared predictions for all caches
    public func getSharedPredictions(excludeIds: Set<VectorID> = []) async -> [PrefetchPrediction] {
        let predictions = await generatePredictions(excludeIds: excludeIds)
        return Array(predictions.values)
            .sorted { $0.confidence > $1.confidence }
            .prefix(config.maxSharedPredictions)
            .map { $0 }
    }
    
    // MARK: - Statistics
    
    public func statistics() -> PredictionStatistics {
        predictionStats
    }
    
    public func resetStatistics() {
        predictionStats = PredictionStatistics()
    }
    
    // MARK: - Private Helpers
    
    private func detectSequentialPattern(id: VectorID, timestamp: ContinuousClock.Instant) {
        // Simple sequential detection: check if ID follows a pattern
        guard accessHistory.count >= 2 else { return }
        
        let recent = accessHistory.suffix(10)
        
        // Look for numeric sequences in IDs
        if let currentNum = extractNumber(from: id) {
            var sequentialCount = 1
            
            for i in stride(from: recent.count - 2, through: 0, by: -1) {
                if let prevNum = extractNumber(from: recent[i].id) {
                    if currentNum - prevNum == recent.count - 1 - i {
                        sequentialCount += 1
                    } else {
                        break
                    }
                }
            }
            
            if sequentialCount >= 3 {
                sequentialPatterns.append(SequentialPattern(
                    baseId: id,
                    detectedAt: timestamp,
                    strength: Float(sequentialCount) / 10.0
                ))
                
                // Keep only recent patterns
                if sequentialPatterns.count > 100 {
                    sequentialPatterns.removeFirst(50)
                }
            }
        }
    }
    
    private func updateRelatedItems(id: VectorID) {
        // Track items accessed close in time as related
        let recentWindow = Duration.seconds(5)
        let now = ContinuousClock.now
        
        let recentIds = accessHistory
            .reversed()
            .prefix(20)
            .filter { now - $0.timestamp < recentWindow }
            .map { $0.id }
        
        for recentId in recentIds where recentId != id {
            relatedItems[id, default: []].insert(recentId)
            relatedItems[recentId, default: []].insert(id)
            
            // Limit related items per ID
            if relatedItems[id]?.count ?? 0 > 10 {
                relatedItems[id]?.removeFirst()
            }
        }
    }
    
    private func generateSequentialPredictions(currentId: VectorID?) -> [PrefetchPrediction] {
        var predictions: [PrefetchPrediction] = []
        
        // Check recent patterns
        for pattern in sequentialPatterns.suffix(5) {
            if let baseNum = extractNumber(from: pattern.baseId),
               let currentNum = extractNumber(from: currentId ?? "") {
                // Predict next in sequence
                let nextNum = currentNum + 1
                let nextId = pattern.baseId.replacingOccurrences(
                    of: "\(baseNum)",
                    with: "\(nextNum)"
                )
                
                predictions.append(PrefetchPrediction(
                    vectorId: nextId,
                    confidence: pattern.strength * 0.9,
                    reason: .sequential,
                    suggestedCacheType: .fifo
                ))
            }
        }
        
        return predictions
    }
    
    private func generateTemporalPredictions(currentId: VectorID?) -> [PrefetchPrediction] {
        guard let currentId = currentId else { return [] }
        
        var predictions: [PrefetchPrediction] = []
        
        // Find items frequently accessed together
        var cooccurrence: [VectorID: Int] = [:]
        
        for i in 0..<accessHistory.count {
            if accessHistory[i].id == currentId {
                // Look at nearby accesses
                for j in max(0, i-5)..<min(accessHistory.count, i+5) where i != j {
                    let nearbyId = accessHistory[j].id
                    let timeDiff = abs((accessHistory[i].timestamp - accessHistory[j].timestamp).components.seconds)
                    
                    if timeDiff < 1 { // Within 1 second
                        cooccurrence[nearbyId, default: 0] += 1
                    }
                }
            }
        }
        
        // Convert to predictions
        for (id, count) in cooccurrence where count > 1 {
            let confidence = min(Float(count) / 10.0, 0.9)
            predictions.append(PrefetchPrediction(
                vectorId: id,
                confidence: confidence,
                reason: .temporal,
                suggestedCacheType: .lru
            ))
        }
        
        return predictions
    }
    
    private func generateFrequencyPredictions() -> [PrefetchPrediction] {
        // Get top frequently accessed items
        let topFrequent = accessFrequency
            .sorted { $0.value > $1.value }
            .prefix(20)
        
        return topFrequent.map { (id, freq) in
            PrefetchPrediction(
                vectorId: id,
                confidence: min(Float(freq) / Float(accessHistory.count) * 10.0, 0.8),
                reason: .frequency,
                suggestedCacheType: .lfu
            )
        }
    }
    
    private func generateRelatedPredictions(for id: VectorID) -> [PrefetchPrediction] {
        guard let related = relatedItems[id] else { return [] }
        
        return related.prefix(5).map { relatedId in
            PrefetchPrediction(
                vectorId: relatedId,
                confidence: 0.6,
                reason: .relatedContent,
                suggestedCacheType: determineCacheType(for: relatedId)
            )
        }
    }
    
    private func determineCacheType(for id: VectorID) -> CacheType? {
        // Determine best cache type based on access pattern
        let freq = accessFrequency[id] ?? 0
        let avgFreq = accessFrequency.values.reduce(0, +) / max(accessFrequency.count, 1)
        
        // Check if part of sequential pattern
        let isSequential = sequentialPatterns.contains { pattern in
            extractNumber(from: pattern.baseId) != nil &&
            extractNumber(from: id) != nil
        }
        
        if isSequential {
            return .fifo
        } else if freq > avgFreq * 2 {
            return .lfu
        } else {
            return .lru
        }
    }
    
    private func extractNumber(from id: VectorID) -> Int? {
        // Extract trailing number from ID like "vec123" -> 123
        let digits = id.reversed().prefix(while: { $0.isNumber }).reversed()
        return Int(String(digits))
    }
}

// MARK: - Supporting Types

private struct SequentialPattern: Sendable {
    let baseId: VectorID
    let detectedAt: ContinuousClock.Instant
    let strength: Float
}

public struct PredictionStatistics: Sendable {
    public var totalPredictions: Int = 0
    public var successfulPredictions: Int = 0
    public var accessesByCache: [String: Int] = [:] // Using String for CacheType.rawValue
    
    public var hitRate: Float {
        guard totalPredictions > 0 else { return 0 }
        return Float(successfulPredictions) / Float(totalPredictions)
    }
}

public struct PrefetchConfiguration: Sendable {
    public let maxPredictions: Int
    public let maxSharedPredictions: Int
    public let enableSequentialDetection: Bool
    public let enableTemporalDetection: Bool
    public let enableFrequencyDetection: Bool
    public let enableRelatedDetection: Bool
    
    public init(
        maxPredictions: Int = 20,
        maxSharedPredictions: Int = 50,
        enableSequentialDetection: Bool = true,
        enableTemporalDetection: Bool = true,
        enableFrequencyDetection: Bool = true,
        enableRelatedDetection: Bool = true
    ) {
        self.maxPredictions = maxPredictions
        self.maxSharedPredictions = maxSharedPredictions
        self.enableSequentialDetection = enableSequentialDetection
        self.enableTemporalDetection = enableTemporalDetection
        self.enableFrequencyDetection = enableFrequencyDetection
        self.enableRelatedDetection = enableRelatedDetection
    }
    
    public static let `default` = PrefetchConfiguration()
    
    public static let aggressive = PrefetchConfiguration(
        maxPredictions: 50,
        maxSharedPredictions: 100
    )
    
    public static let conservative = PrefetchConfiguration(
        maxPredictions: 10,
        maxSharedPredictions: 20
    )
}
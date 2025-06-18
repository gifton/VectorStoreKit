// VectorStoreKit: Access Pattern Analyzer
//
// Analyzes cache access patterns for optimization and prediction

import Foundation

// MARK: - Access Pattern Analyzer

/// Analyzes access patterns to optimize cache behavior
public actor AccessPatternAnalyzer {
    
    // MARK: - Properties
    
    private var accessLog: [AccessEntry] = []
    private var accessCounts: [VectorID: Int] = [:]
    private var transitionMatrix: [VectorID: [VectorID: Int]] = [:]
    private var temporalPatterns: [VectorID: TemporalPattern] = [:]
    
    // Pattern detection parameters
    private let maxLogSize = 10000
    private let minPatternSupport = 5
    private let sequenceWindowSize = 10
    
    // Analysis results cache
    private var cachedPatterns: AccessPatternSummary?
    private var cacheInvalidated = true
    
    // MARK: - Types
    
    private struct AccessEntry {
        let id: VectorID
        let level: CacheLevel?
        let timestamp: Date
        let accessType: AccessType
    }
    
    private enum AccessType {
        case read
        case write
        case prefetch
    }
    
    private struct TemporalPattern {
        let avgInterval: TimeInterval
        let variance: TimeInterval
        let lastAccess: Date
        let accessTimes: [Date]
    }
    
    // MARK: - Public Methods
    
    /// Record a read access
    public func recordAccess(id: VectorID, level: CacheLevel?, timestamp: Date) async {
        let entry = AccessEntry(
            id: id,
            level: level,
            timestamp: timestamp,
            accessType: .read
        )
        
        await recordEntry(entry)
        
        // Update access count
        accessCounts[id, default: 0] += 1
        
        // Update transition matrix
        await updateTransitions(id: id)
        
        // Update temporal patterns
        await updateTemporalPattern(id: id, timestamp: timestamp)
    }
    
    /// Record a write access
    public func recordWrite(id: VectorID, level: CacheLevel) async {
        let entry = AccessEntry(
            id: id,
            level: level,
            timestamp: Date(),
            accessType: .write
        )
        
        await recordEntry(entry)
    }
    
    /// Get access count for a vector
    public func getAccessCount(id: VectorID) async -> Int {
        accessCounts[id] ?? 0
    }
    
    /// Get related vectors based on access patterns
    public func getRelatedVectors(id: VectorID, limit: Int) async -> [VectorID] {
        guard let transitions = transitionMatrix[id] else { return [] }
        
        return transitions
            .sorted { $0.value > $1.value }
            .prefix(limit)
            .map { $0.key }
    }
    
    /// Analyze access patterns
    public func analyzePatterns() async -> AccessPatternSummary {
        if !cacheInvalidated, let cached = cachedPatterns {
            return cached
        }
        
        let hotSpots = identifyHotSpots()
        let coldSpots = identifyColdSpots()
        let sequences = detectSequentialPatterns()
        let temporal = analyzeTemporalLocality()
        let spatial = analyzeSpatialLocality()
        
        let summary = AccessPatternSummary(
            hotSpots: hotSpots,
            coldSpots: coldSpots,
            sequentialityScore: calculateSequentialityScore(sequences),
            temporalLocality: temporal,
            spatialLocality: spatial
        )
        
        cachedPatterns = summary
        cacheInvalidated = false
        
        return summary
    }
    
    /// Get current access patterns
    public func getCurrentPatterns() async -> AccessPatternSummary {
        await analyzePatterns()
    }
    
    /// Get pattern strength score
    public func getPatternStrength() async -> Float {
        let patterns = await analyzePatterns()
        
        // Combine different pattern strengths
        let sequential = patterns.sequentialityScore
        let temporal = patterns.temporalLocality
        let spatial = patterns.spatialLocality
        
        // Weighted average
        return (sequential * 0.3 + temporal * 0.4 + spatial * 0.3)
    }
    
    /// Reset analyzer state
    public func reset() async {
        accessLog.removeAll()
        accessCounts.removeAll()
        transitionMatrix.removeAll()
        temporalPatterns.removeAll()
        cachedPatterns = nil
        cacheInvalidated = true
    }
    
    // MARK: - Private Methods
    
    private func recordEntry(_ entry: AccessEntry) async {
        accessLog.append(entry)
        cacheInvalidated = true
        
        // Maintain bounded log
        if accessLog.count > maxLogSize {
            let toRemove = accessLog.count - maxLogSize
            accessLog.removeFirst(toRemove)
        }
    }
    
    private func updateTransitions(id: VectorID) async {
        // Look at recent accesses to update transition probabilities
        let recentAccesses = accessLog.suffix(sequenceWindowSize)
        
        guard recentAccesses.count >= 2 else { return }
        
        // Find previous access
        var previousId: VectorID?
        for access in recentAccesses.dropLast().reversed() {
            if access.accessType == .read {
                previousId = access.id
                break
            }
        }
        
        if let prev = previousId, prev != id {
            // Update transition count
            if transitionMatrix[prev] == nil {
                transitionMatrix[prev] = [:]
            }
            transitionMatrix[prev]![id, default: 0] += 1
        }
    }
    
    private func updateTemporalPattern(id: VectorID, timestamp: Date) async {
        if var pattern = temporalPatterns[id] {
            // Update existing pattern
            var times = pattern.accessTimes
            times.append(timestamp)
            
            // Keep bounded history
            if times.count > 20 {
                times.removeFirst()
            }
            
            // Recalculate statistics
            let intervals = zip(times.dropFirst(), times).map {
                $0.0.timeIntervalSince($0.1)
            }
            
            if !intervals.isEmpty {
                let avgInterval = intervals.reduce(0, +) / Double(intervals.count)
                let variance = intervals
                    .map { pow($0 - avgInterval, 2) }
                    .reduce(0, +) / Double(intervals.count)
                
                temporalPatterns[id] = TemporalPattern(
                    avgInterval: avgInterval,
                    variance: variance,
                    lastAccess: timestamp,
                    accessTimes: times
                )
            }
        } else {
            // Create new pattern
            temporalPatterns[id] = TemporalPattern(
                avgInterval: 0,
                variance: 0,
                lastAccess: timestamp,
                accessTimes: [timestamp]
            )
        }
    }
    
    private func identifyHotSpots() -> Set<VectorID> {
        guard !accessCounts.isEmpty else { return [] }
        
        // Calculate threshold for hot spots (top 20%)
        let sortedCounts = accessCounts.values.sorted(by: >)
        let threshold = sortedCounts[min(sortedCounts.count / 5, sortedCounts.count - 1)]
        
        return Set(accessCounts.compactMap { 
            $0.value >= threshold ? $0.key : nil 
        })
    }
    
    private func identifyColdSpots() -> Set<VectorID> {
        guard !accessCounts.isEmpty else { return [] }
        
        // Vectors accessed only once or twice
        return Set(accessCounts.compactMap { 
            $0.value <= 2 ? $0.key : nil 
        })
    }
    
    private func detectSequentialPatterns() -> [[VectorID]] {
        var sequences: [[VectorID]] = []
        var currentSequence: [VectorID] = []
        
        // Look for sequential access patterns
        for i in 1..<accessLog.count {
            let current = accessLog[i]
            let previous = accessLog[i-1]
            
            // Check if sequential (close in time)
            let timeDiff = current.timestamp.timeIntervalSince(previous.timestamp)
            
            if timeDiff < 0.1 && current.accessType == .read && previous.accessType == .read {
                if currentSequence.isEmpty {
                    currentSequence.append(previous.id)
                }
                currentSequence.append(current.id)
            } else if !currentSequence.isEmpty {
                if currentSequence.count >= 3 {
                    sequences.append(currentSequence)
                }
                currentSequence = []
            }
        }
        
        // Add final sequence if any
        if currentSequence.count >= 3 {
            sequences.append(currentSequence)
        }
        
        return sequences
    }
    
    private func calculateSequentialityScore(_ sequences: [[VectorID]]) -> Float {
        guard !accessLog.isEmpty else { return 0 }
        
        let sequentialAccesses = sequences.reduce(0) { $0 + $1.count }
        let totalAccesses = accessLog.filter { $0.accessType == .read }.count
        
        return totalAccesses > 0 ? Float(sequentialAccesses) / Float(totalAccesses) : 0
    }
    
    private func analyzeTemporalLocality() -> Float {
        // Measure how often recently accessed items are accessed again
        let recentWindow: TimeInterval = 60 // 1 minute
        let now = Date()
        
        var recentIds = Set<VectorID>()
        var reuseCount = 0
        var totalAccesses = 0
        
        for access in accessLog.reversed() {
            guard access.accessType == .read else { continue }
            
            if now.timeIntervalSince(access.timestamp) > recentWindow {
                break
            }
            
            totalAccesses += 1
            
            if recentIds.contains(access.id) {
                reuseCount += 1
            } else {
                recentIds.insert(access.id)
            }
        }
        
        return totalAccesses > 0 ? Float(reuseCount) / Float(totalAccesses) : 0
    }
    
    private func analyzeSpatialLocality() -> Float {
        // Analyze how clustered accesses are
        // This is simplified - real implementation would consider vector relationships
        
        guard !transitionMatrix.isEmpty else { return 0 }
        
        // Calculate average transition probability
        var totalTransitions = 0
        var strongTransitions = 0
        
        for (_, transitions) in transitionMatrix {
            let total = transitions.values.reduce(0, +)
            for count in transitions.values {
                totalTransitions += 1
                if Float(count) / Float(total) > 0.2 {
                    strongTransitions += 1
                }
            }
        }
        
        return totalTransitions > 0 ? Float(strongTransitions) / Float(totalTransitions) : 0
    }
}

// MARK: - Access Ring Buffer

/// Efficient ring buffer for tracking recent accesses
public struct CacheAccessRingBuffer {
    private var buffer: [VectorID?]
    private var writeIndex = 0
    private var count = 0
    private let capacity: Int
    
    public init(capacity: Int) {
        self.capacity = capacity
        self.buffer = Array(repeating: nil, count: capacity)
    }
    
    public mutating func recordAccess(_ id: VectorID) {
        buffer[writeIndex] = id
        writeIndex = (writeIndex + 1) % capacity
        count = min(count + 1, capacity)
    }
    
    public func mostRecent(_ k: Int) -> [VectorID] {
        let k = min(k, count)
        var result: [VectorID] = []
        
        for i in 0..<k {
            let index = (writeIndex - 1 - i + capacity) % capacity
            if let id = buffer[index] {
                result.append(id)
            }
        }
        
        return result
    }
    
    public var recentAccesses: [VectorID] {
        var result: [VectorID] = []
        
        for i in 0..<count {
            let index = (writeIndex - count + i + capacity) % capacity
            if let id = buffer[index] {
                result.append(id)
            }
        }
        
        return result
    }
    
    public mutating func clear() {
        buffer = Array(repeating: nil, count: capacity)
        writeIndex = 0
        count = 0
    }
    
    public func analyzePatterns() -> (
        totalAccesses: Int,
        uniqueCount: Int,
        accessFrequency: [VectorID: Int]
    ) {
        var frequency: [VectorID: Int] = [:]
        
        for i in 0..<count {
            if let id = buffer[i] {
                frequency[id, default: 0] += 1
            }
        }
        
        return (
            totalAccesses: count,
            uniqueCount: frequency.count,
            accessFrequency: frequency
        )
    }
}
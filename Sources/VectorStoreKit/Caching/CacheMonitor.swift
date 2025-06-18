// VectorStoreKit: Cache Monitoring System
//
// Comprehensive monitoring for multi-level cache performance

import Foundation
import os.log

// MARK: - Cache Monitor

/// Unified monitoring system for cache operations
public actor CacheMonitor {
    
    // MARK: - Properties
    
    private var accessLog: CircularBuffer<AccessRecord>
    private var levelMetrics: [CacheLevel: LevelMetrics]
    private var globalMetrics: GlobalMetrics
    private let startTime: Date
    
    // Anomaly detection
    private var baselineEstablished = false
    private var baseline: BaselineMetrics?
    
    // Real-time metrics
    private var recentHitRates: CircularBuffer<Float>
    private var recentLatencies: CircularBuffer<TimeInterval>
    
    // Logger
    private let logger = Logger(subsystem: "VectorStoreKit", category: "CacheMonitor")
    
    // MARK: - Types
    
    private struct AccessRecord {
        let id: VectorID
        let level: CacheLevel?
        let hit: Bool
        let latency: TimeInterval
        let timestamp: Date
    }
    
    private struct LevelMetrics {
        var hits: Int = 0
        var misses: Int = 0
        var totalLatency: TimeInterval = 0
        var accessCount: Int = 0
        
        var hitRate: Float {
            let total = hits + misses
            return total > 0 ? Float(hits) / Float(total) : 0.0
        }
        
        var avgLatency: TimeInterval {
            accessCount > 0 ? totalLatency / Double(accessCount) : 0
        }
    }
    
    private struct GlobalMetrics {
        var totalHits: Int = 0
        var totalMisses: Int = 0
        var totalAccesses: Int = 0
        var totalLatency: TimeInterval = 0
        
        var hitRate: Float {
            totalAccesses > 0 ? Float(totalHits) / Float(totalAccesses) : 0.0
        }
        
        var avgLatency: TimeInterval {
            totalAccesses > 0 ? totalLatency / Double(totalAccesses) : 0
        }
    }
    
    private struct BaselineMetrics {
        let avgHitRate: Float
        let stdDevHitRate: Float
        let avgLatency: TimeInterval
        let stdDevLatency: Float
        let p95Latency: TimeInterval
        let p99Latency: TimeInterval
    }
    
    // MARK: - Initialization
    
    public init(bufferSize: Int = 10000) {
        self.accessLog = CircularBuffer(capacity: bufferSize)
        self.levelMetrics = [:]
        self.globalMetrics = GlobalMetrics()
        self.startTime = Date()
        self.recentHitRates = CircularBuffer(capacity: 100)
        self.recentLatencies = CircularBuffer(capacity: 1000)
        
        // Initialize level metrics
        for level in CacheLevel.allCases {
            levelMetrics[level] = LevelMetrics()
        }
    }
    
    // MARK: - Recording Methods
    
    /// Record a cache access
    public func recordAccess(
        id: VectorID,
        level: CacheLevel?,
        hit: Bool,
        latency: Duration
    ) async {
        let latencySeconds = Double(latency.components.seconds) + 
                           Double(latency.components.attoseconds) / 1e18
        
        let record = AccessRecord(
            id: id,
            level: level,
            hit: hit,
            latency: latencySeconds,
            timestamp: Date()
        )
        
        accessLog.append(record)
        
        // Update metrics
        if hit {
            globalMetrics.totalHits += 1
            if let level = level {
                levelMetrics[level]?.hits += 1
            }
        } else {
            globalMetrics.totalMisses += 1
            if let level = level {
                levelMetrics[level]?.misses += 1
            }
        }
        
        globalMetrics.totalAccesses += 1
        globalMetrics.totalLatency += latencySeconds
        
        if let level = level {
            levelMetrics[level]?.accessCount += 1
            levelMetrics[level]?.totalLatency += latencySeconds
        }
        
        // Update real-time metrics
        recentHitRates.append(hit ? 1.0 : 0.0)
        recentLatencies.append(latencySeconds)
        
        // Check for anomalies
        if baselineEstablished {
            await checkForAnomalies(latency: latencySeconds, hit: hit)
        } else if globalMetrics.totalAccesses > 1000 {
            await establishBaseline()
        }
    }
    
    /// Record a cache removal
    public func recordRemoval(id: VectorID) async {
        // Track removals for analysis
        logger.debug("Cache removal recorded for \(id)")
    }
    
    /// Reset all metrics
    public func reset() async {
        accessLog.clear()
        levelMetrics.removeAll()
        globalMetrics = GlobalMetrics()
        baselineEstablished = false
        baseline = nil
        recentHitRates.clear()
        recentLatencies.clear()
        
        // Reinitialize level metrics
        for level in CacheLevel.allCases {
            levelMetrics[level] = LevelMetrics()
        }
    }
    
    // MARK: - Analysis Methods
    
    /// Get global hit rate
    public func getGlobalHitRate() async -> Float {
        globalMetrics.hitRate
    }
    
    /// Get global metrics
    public func getGlobalMetrics() async -> GlobalCacheMetrics {
        GlobalCacheMetrics(
            totalHits: globalMetrics.totalHits,
            totalMisses: globalMetrics.totalMisses,
            avgLatency: globalMetrics.avgLatency,
            memoryEfficiency: calculateMemoryEfficiency()
        )
    }
    
    /// Perform global analysis
    public func performGlobalAnalysis() async -> GlobalCacheAnalysis {
        let trend = analyzeTrend()
        let efficiency = calculateAccessPatternEfficiency()
        let balance = calculateLevelBalance()
        let anomalies = await detectRecentAnomalies()
        
        return GlobalCacheAnalysis(
            trend: trend,
            accessPatternEfficiency: efficiency,
            levelBalance: balance,
            anomalies: anomalies
        )
    }
    
    /// Get detailed metrics for a specific level
    public func getLevelMetrics(level: CacheLevel) async -> LevelMetrics? {
        levelMetrics[level]
    }
    
    /// Get access history for analysis
    public func getAccessHistory(limit: Int = 1000) async -> [AccessRecord] {
        Array(accessLog.suffix(limit))
    }
    
    // MARK: - Private Methods
    
    private func establishBaseline() async {
        let recentLatencyArray = Array(recentLatencies)
        let recentHitRateArray = Array(recentHitRates)
        
        guard !recentLatencyArray.isEmpty && !recentHitRateArray.isEmpty else { return }
        
        // Calculate baseline metrics
        let avgHitRate = recentHitRateArray.reduce(0, +) / Float(recentHitRateArray.count)
        let avgLatency = recentLatencyArray.reduce(0, +) / Double(recentLatencyArray.count)
        
        let hitRateStdDev = calculateStandardDeviation(recentHitRateArray)
        let latencyStdDev = calculateStandardDeviation(recentLatencyArray.map { Float($0) })
        
        let sortedLatencies = recentLatencyArray.sorted()
        let p95Index = Int(Double(sortedLatencies.count) * 0.95)
        let p99Index = Int(Double(sortedLatencies.count) * 0.99)
        
        baseline = BaselineMetrics(
            avgHitRate: avgHitRate,
            stdDevHitRate: hitRateStdDev,
            avgLatency: avgLatency,
            stdDevLatency: latencyStdDev,
            p95Latency: sortedLatencies[safe: p95Index] ?? avgLatency,
            p99Latency: sortedLatencies[safe: p99Index] ?? avgLatency
        )
        
        baselineEstablished = true
        logger.info("Cache baseline established: hit rate=\(avgHitRate), latency=\(avgLatency)s")
    }
    
    private func checkForAnomalies(latency: TimeInterval, hit: Bool) async {
        guard let baseline = baseline else { return }
        
        // Check latency anomaly
        if latency > baseline.p99Latency * 2 {
            logger.warning("Latency anomaly detected: \(latency)s (baseline p99: \(baseline.p99Latency)s)")
        }
        
        // Check hit rate anomaly (using sliding window)
        let recentHits = Array(recentHitRates.suffix(100))
        if recentHits.count >= 100 {
            let recentHitRate = recentHits.reduce(0, +) / Float(recentHits.count)
            let deviation = abs(recentHitRate - baseline.avgHitRate) / baseline.stdDevHitRate
            
            if deviation > 3.0 {
                logger.warning("Hit rate anomaly detected: \(recentHitRate) (baseline: \(baseline.avgHitRate))")
            }
        }
    }
    
    private func analyzeTrend() -> Trend {
        let recentHitRateArray = Array(recentHitRates.suffix(50))
        guard recentHitRateArray.count >= 10 else { return .stable }
        
        // Simple linear regression
        let n = Float(recentHitRateArray.count)
        let indices = (0..<recentHitRateArray.count).map { Float($0) }
        
        let sumX = indices.reduce(0, +)
        let sumY = recentHitRateArray.reduce(0, +)
        let sumXY = zip(indices, recentHitRateArray).map { $0 * $1 }.reduce(0, +)
        let sumXX = indices.map { $0 * $0 }.reduce(0, +)
        
        let slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
        
        if slope > 0.01 { return .improving }
        if slope < -0.01 { return .declining }
        return .stable
    }
    
    private func calculateAccessPatternEfficiency() -> Float {
        // Analyze how well cache levels are utilized
        var totalEfficiency: Float = 0.0
        var levelCount = 0
        
        for (level, metrics) in levelMetrics {
            if metrics.accessCount > 0 {
                // Efficiency based on hit rate and latency target
                let hitRateScore = metrics.hitRate
                let latencyScore = Float(min(1.0, level.maxAccessTime / metrics.avgLatency))
                let efficiency = (hitRateScore + latencyScore) / 2.0
                
                totalEfficiency += efficiency
                levelCount += 1
            }
        }
        
        return levelCount > 0 ? totalEfficiency / Float(levelCount) : 0.0
    }
    
    private func calculateLevelBalance() -> Float {
        // Measure how well distributed accesses are across levels
        let accessCounts = levelMetrics.values.map { Float($0.accessCount) }
        guard !accessCounts.isEmpty else { return 0.0 }
        
        let total = accessCounts.reduce(0, +)
        guard total > 0 else { return 0.0 }
        
        // Calculate entropy as measure of balance
        var entropy: Float = 0.0
        for count in accessCounts {
            if count > 0 {
                let p = count / total
                entropy -= p * log2(p)
            }
        }
        
        // Normalize entropy
        let maxEntropy = log2(Float(CacheLevel.allCases.count))
        return maxEntropy > 0 ? entropy / maxEntropy : 0.0
    }
    
    private func calculateMemoryEfficiency() -> Float {
        // Efficiency based on hit rate per unit of memory
        // This is a simplified calculation
        globalMetrics.hitRate
    }
    
    private func detectRecentAnomalies() async -> [String] {
        var anomalies: [String] = []
        
        // Check for sudden hit rate drops
        let recentHitRateArray = Array(recentHitRates.suffix(20))
        if recentHitRateArray.count >= 20 {
            let firstHalf = Array(recentHitRateArray.prefix(10))
            let secondHalf = Array(recentHitRateArray.suffix(10))
            
            let firstAvg = firstHalf.reduce(0, +) / Float(firstHalf.count)
            let secondAvg = secondHalf.reduce(0, +) / Float(secondHalf.count)
            
            if firstAvg > 0.5 && secondAvg < firstAvg * 0.5 {
                anomalies.append("Sudden hit rate degradation detected")
            }
        }
        
        // Check for latency spikes
        let recentLatencyArray = Array(recentLatencies.suffix(100))
        if let baseline = baseline, recentLatencyArray.count >= 10 {
            let spikeCount = recentLatencyArray.filter { $0 > baseline.p95Latency * 3 }.count
            if spikeCount > 5 {
                anomalies.append("Multiple latency spikes detected")
            }
        }
        
        return anomalies
    }
    
    private func calculateStandardDeviation(_ values: [Float]) -> Float {
        guard values.count > 1 else { return 0 }
        
        let mean = values.reduce(0, +) / Float(values.count)
        let squaredDiffs = values.map { pow($0 - mean, 2) }
        let variance = squaredDiffs.reduce(0, +) / Float(values.count - 1)
        
        return sqrt(variance)
    }
}

// MARK: - Supporting Types

/// Circular buffer for efficient metric storage
private struct CircularBuffer<T> {
    private var buffer: [T?]
    private var writeIndex = 0
    private var count = 0
    private let capacity: Int
    
    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = Array(repeating: nil, count: capacity)
    }
    
    mutating func append(_ element: T) {
        buffer[writeIndex] = element
        writeIndex = (writeIndex + 1) % capacity
        count = min(count + 1, capacity)
    }
    
    mutating func clear() {
        buffer = Array(repeating: nil, count: capacity)
        writeIndex = 0
        count = 0
    }
    
    func suffix(_ k: Int) -> [T] {
        let k = min(k, count)
        var result: [T] = []
        
        for i in 0..<k {
            let index = (writeIndex - k + i + capacity) % capacity
            if let element = buffer[index] {
                result.append(element)
            }
        }
        
        return result
    }
}

// Array safe subscript extension
private extension Array {
    subscript(safe index: Int) -> Element? {
        guard index >= 0 && index < count else { return nil }
        return self[index]
    }
}
// VectorStoreKit: LFU Cache Implementation
//
// Least Frequently Used (LFU) cache with frequency decay and adaptive thresholds

import Foundation
import simd

// MARK: - Frequency Node

/// Node for frequency list in LFU implementation
private class FrequencyNode<Vector: SIMD> where Vector.Scalar: BinaryFloatingPoint {
    let frequency: Int
    var entries: Set<VectorID> = []
    var prev: FrequencyNode?
    var next: FrequencyNode?
    
    init(frequency: Int) {
        self.frequency = frequency
    }
}

// MARK: - LFU Entry

/// Enhanced cache entry for LFU with frequency tracking
private struct LFUEntry<Vector: SIMD> where Vector.Scalar: BinaryFloatingPoint {
    var entry: CacheEntry<Vector>
    var frequency: Float // Use Float for decay support
    var lastAccessTime: ContinuousClock.Instant
    var frequencyNode: FrequencyNode<Vector>?
    
    init(entry: CacheEntry<Vector>, frequency: Float = 1.0) {
        self.entry = entry
        self.frequency = frequency
        self.lastAccessTime = ContinuousClock.now
    }
}

// MARK: - BasicLFUVectorCache

/// Least Frequently Used (LFU) cache with frequency decay and enhanced features
public actor BasicLFUVectorCache<Vector: SIMD & Sendable>: VectorCache where Vector.Scalar: BinaryFloatingPoint {
    public typealias Configuration = LFUCacheConfiguration
    public typealias Statistics = BasicCacheStatistics
    
    public let configuration: Configuration
    
    // Main cache storage
    private var cache: [VectorID: LFUEntry<Vector>] = [:]
    private var currentMemoryUsage: Int = 0
    
    // Frequency tracking with decay
    private var minFrequency: Int = 0
    private var frequencyList: [Int: FrequencyNode<Vector>] = [:]
    private var frequencyHead: FrequencyNode<Vector>?
    
    // Decay parameters
    private let decayFactor: Float = 0.95
    private let decayInterval: TimeInterval = 300 // 5 minutes
    private var lastDecayTime = ContinuousClock.now
    
    // Scheduler tasks
    private var decayTask: Task<Void, Never>?
    private var thresholdTask: Task<Void, Never>?
    
    // Adaptive threshold
    private var adaptiveEvictionThreshold: Float = 2.0
    private let thresholdUpdateInterval: TimeInterval = 600 // 10 minutes
    private var lastThresholdUpdate = ContinuousClock.now
    
    // Statistics
    private var hitCount: Int = 0
    private var missCount: Int = 0
    private var evictionCount: Int = 0
    private var totalAccessTime: TimeInterval = 0
    
    // Access pattern tracking
    private var recentHitRates: [Float] = []
    private let maxHitRateHistory = 20
    
    // Performance analyzer
    private let performanceAnalyzer = CachePerformanceAnalyzer<Vector>()
    
    // Predictive prefetch engine
    private let prefetchEngine = PredictivePrefetchEngine<Vector>()
    
    // Storage backend
    private let storageBackend: (any CacheStorageBackend<Vector>)?
    
    public init(maxMemory: Int, storageBackend: (any CacheStorageBackend<Vector>)? = nil) throws {
        guard maxMemory > 0 else {
            throw VectorCacheError.invalidConfiguration("maxMemory must be positive")
        }
        self.configuration = LFUCacheConfiguration(maxMemory: maxMemory)
        self.storageBackend = storageBackend
        
        // Schedulers will be started in a non-isolated context
    }
    
    public func start() async {
        startDecayScheduler()
        startThresholdScheduler()
    }
    
    deinit {
        decayTask?.cancel()
        thresholdTask?.cancel()
    }
    
    // MARK: - Core Properties
    
    public var count: Int { cache.count }
    public var size: Int { currentMemoryUsage }
    public var memoryUsage: Int { currentMemoryUsage }
    public var hitRate: Float {
        let total = hitCount + missCount
        return total > 0 ? Float(hitCount) / Float(total) : 0.0
    }
    
    // MARK: - Core Operations
    
    public func get(id: VectorID) async -> Vector? {
        let startTime = ContinuousClock.now
        defer {
            let elapsed = ContinuousClock.now - startTime
            totalAccessTime += Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
        }
        
        guard var lfuEntry = cache[id] else {
            missCount += 1
            
            // Record miss synchronously
            await prefetchEngine.recordAccess(id: id)
            
            return nil
        }
        
        hitCount += 1
        
        // Update frequency with increment
        incrementFrequency(for: id, entry: &lfuEntry)
        cache[id] = lfuEntry
        
        // Record hit synchronously
        await prefetchEngine.recordAccess(id: id)
        
        return lfuEntry.entry.vector
    }
    
    public func set(id: VectorID, vector: Vector, priority: CachePriority) async {
        let entrySize = estimateVectorMemorySize(vector)
        
        // Update existing entry
        if var existingEntry = cache[id] {
            existingEntry.entry = CacheEntry(
                vector: vector,
                priority: priority,
                timestamp: Date(),
                accessCount: existingEntry.entry.accessCount + 1
            )
            existingEntry.lastAccessTime = ContinuousClock.now
            cache[id] = existingEntry
            return
        }
        
        // Check if we need to evict
        while currentMemoryUsage + entrySize > configuration.maxMemory && !cache.isEmpty {
            await evictLeastFrequentlyUsed()
        }
        
        // Add new entry
        let cacheEntry = CacheEntry(vector: vector, priority: priority)
        let lfuEntry = LFUEntry(entry: cacheEntry)
        
        cache[id] = lfuEntry
        currentMemoryUsage += entrySize
        
        // Add to frequency list
        addToFrequencyList(id: id, frequency: 1)
    }
    
    public func remove(id: VectorID) async {
        guard let lfuEntry = cache.removeValue(forKey: id) else { return }
        
        let entrySize = estimateVectorMemorySize(lfuEntry.entry.vector)
        currentMemoryUsage -= entrySize
        
        // Remove from frequency tracking
        removeFromFrequencyList(id: id, frequency: Int(lfuEntry.frequency))
    }
    
    public func clear() async {
        evictionCount += cache.count
        cache.removeAll()
        frequencyList.removeAll()
        frequencyHead = nil
        minFrequency = 0
        currentMemoryUsage = 0
    }
    
    public func contains(id: VectorID) async -> Bool {
        cache[id] != nil
    }
    
    // MARK: - Advanced Operations (Medium Complexity)
    
    public func preload(ids: [VectorID]) async {
        guard let storage = storageBackend else { return }
        
        let missingIds = ids.filter { cache[$0] == nil }
        guard !missingIds.isEmpty else { return }
        
        do {
            let vectors = try await storage.fetchVectors(ids: missingIds)
            
            for (id, vector) in vectors {
                // Preloaded items start with lower frequency
                await set(id: id, vector: vector, priority: .normal)
                
                // Adjust initial frequency based on position in request
                if let index = ids.firstIndex(of: id) {
                    let boostFactor = 1.0 - (Float(index) / Float(ids.count)) * 0.5
                    if var entry = cache[id] {
                        entry.frequency *= boostFactor
                        cache[id] = entry
                    }
                }
            }
        } catch {
            // Preload is best effort
        }
    }
    
    public func prefetch(_ predictions: [VectorID: Float]) async {
        guard let storage = storageBackend else { return }
        
        // Get current most accessed item for context
        let currentId = cache.max(by: { $0.value.frequency < $1.value.frequency })?.key
        
        // Get enhanced predictions
        let enhancedPredictions = await prefetchEngine.generatePredictions(
            currentId: currentId ?? "",
            predictions: predictions
        )
        
        // Filter candidates
        let candidateIds = enhancedPredictions
            .filter { cache[$0.key] == nil }
            .sorted { $0.value > $1.value }
            .prefix(8) // Smaller batch for LFU
            .map { $0.key }
        
        guard !candidateIds.isEmpty else { return }
        
        // Check memory constraints
        let avgEntrySize = currentMemoryUsage / max(cache.count, 1)
        let estimatedSize = avgEntrySize * candidateIds.count
        let freeSpace = configuration.maxMemory - currentMemoryUsage
        
        guard estimatedSize < Int(Double(freeSpace) * 0.25) else { return } // More conservative for LFU
        
        do {
            let vectors = try await storage.fetchVectors(ids: Array(candidateIds))
            
            for (id, vector) in vectors {
                let confidence = enhancedPredictions[id] ?? 0.3
                
                // Initial frequency based on confidence
                let initialFrequency = 1.0 + confidence * 2.0 // 1.0 to 3.0 range
                
                let priority: CachePriority = confidence > 0.6 ? .normal : .low
                
                // Check memory
                let entrySize = estimateVectorMemorySize(vector)
                if currentMemoryUsage + entrySize > configuration.maxMemory {
                    // Try to evict items with frequency below threshold
                    var evicted = false
                    for (cacheId, entry) in cache {
                        if entry.frequency < adaptiveEvictionThreshold * 0.5 &&
                           entry.entry.priority == .low {
                            await remove(id: cacheId)
                            evicted = true
                            break
                        }
                    }
                    
                    if !evicted {
                        break
                    }
                }
                
                // Create LFU entry with initial frequency
                let cacheEntry = CacheEntry(
                    vector: vector,
                    priority: priority,
                    timestamp: Date(),
                    metadata: CacheEntryMetadata(
                        source: "prefetch",
                        lastAccessed: Date(),
                        size: entrySize
                    )
                )
                
                let lfuEntry = LFUEntry(
                    entry: cacheEntry,
                    frequency: initialFrequency
                )
                
                cache[id] = lfuEntry
                currentMemoryUsage += entrySize
                
                // Add to frequency tracking
                addToFrequencyList(id: id, frequency: Int(initialFrequency))
            }
        } catch {
            // Silent failure for prefetch
        }
    }
    
    public func optimize() async {
        // Force immediate decay if needed
        let now = ContinuousClock.now
        let elapsed = now - lastDecayTime
        if elapsed.components.seconds > 60 { // Only if more than a minute since last decay
            await applyDecay()
        }
        
        // Remove entries below adaptive threshold
        var toRemove: [VectorID] = []
        
        for (id, entry) in cache {
            if entry.frequency < adaptiveEvictionThreshold && 
               entry.entry.priority == .low {
                toRemove.append(id)
            }
        }
        
        for id in toRemove {
            await remove(id: id)
        }
        
        // Update performance metrics
        await performanceAnalyzer.recordMemoryUsage(currentMemoryUsage)
        
        // Clean up empty frequency nodes
        cleanupFrequencyList()
    }
    
    public func statistics() async -> Statistics {
        BasicCacheStatistics(
            hitCount: hitCount,
            missCount: missCount,
            evictionCount: evictionCount,
            totalAccessTime: totalAccessTime,
            memoryUsage: currentMemoryUsage
        )
    }
    
    public func performanceAnalysis() async -> CachePerformanceAnalysis {
        let analysis = await performanceAnalyzer.analyze(
            currentHitRate: hitRate,
            currentMemoryUsage: currentMemoryUsage,
            maxMemory: configuration.maxMemory
        )
        
        var recommendations = analysis.recommendations
        
        // LFU-specific recommendations
        if adaptiveEvictionThreshold > 5.0 {
            recommendations.append(CacheRecommendation(
                type: .policyChange,
                description: "High eviction threshold detected - consider more aggressive caching",
                expectedImprovement: 0.1
            ))
        }
        
        // Check frequency distribution
        let freqDistribution = analyzeFrequencyDistribution()
        if freqDistribution.skewness > 2.0 {
            recommendations.append(CacheRecommendation(
                type: .partitioning,
                description: "Highly skewed access pattern - consider cache partitioning",
                expectedImprovement: 0.15
            ))
        }
        
        return CachePerformanceAnalysis(
            hitRateOverTime: analysis.hitRateOverTime,
            memoryUtilization: analysis.memoryUtilization,
            evictionRate: analysis.evictionRate,
            optimalCacheSize: analysis.optimalCacheSize,
            recommendations: recommendations
        )
    }
    
    // MARK: - Private Helpers
    
    private func startDecayScheduler() {
        decayTask = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    // Sleep for decay interval
                    try await Task.sleep(for: .seconds(self?.decayInterval ?? 300))
                    
                    // Apply decay if still active
                    guard let self else { return }
                    await self.applyDecay()
                } catch {
                    // Task cancelled
                    break
                }
            }
        }
    }
    
    private func startThresholdScheduler() {
        thresholdTask = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    // Sleep for threshold update interval
                    try await Task.sleep(for: .seconds(self?.thresholdUpdateInterval ?? 600))
                    
                    // Update threshold if still active
                    guard let self else { return }
                    await self.updateAdaptiveThreshold()
                } catch {
                    // Task cancelled
                    break
                }
            }
        }
    }
    
    private func incrementFrequency(for id: VectorID, entry: inout LFUEntry<Vector>) {
        let oldFreq = Int(entry.frequency)
        entry.frequency += 1.0
        entry.lastAccessTime = ContinuousClock.now
        
        // Update frequency list
        removeFromFrequencyList(id: id, frequency: oldFreq)
        addToFrequencyList(id: id, frequency: Int(entry.frequency))
    }
    
    private func addToFrequencyList(id: VectorID, frequency: Int) {
        if frequencyList[frequency] == nil {
            frequencyList[frequency] = FrequencyNode(frequency: frequency)
        }
        
        frequencyList[frequency]?.entries.insert(id)
        
        // Update min frequency
        if frequency < minFrequency || minFrequency == 0 {
            minFrequency = frequency
        }
    }
    
    private func removeFromFrequencyList(id: VectorID, frequency: Int) {
        guard let node = frequencyList[frequency] else { return }
        
        node.entries.remove(id)
        
        if node.entries.isEmpty {
            frequencyList.removeValue(forKey: frequency)
            
            // Update min frequency if needed - compute lazily
            if frequency == minFrequency {
                // Will be recomputed when needed in evictLeastFrequentlyUsed
                minFrequency = -1 // Mark as invalid
            }
        }
    }
    
    private func evictLeastFrequentlyUsed() async {
        // Recompute minFrequency if invalid
        if minFrequency < 0 || frequencyList[minFrequency] == nil {
            minFrequency = frequencyList.keys.min() ?? 0
        }
        
        // Find candidates with lowest frequency
        var candidateId: VectorID?
        var lowestFrequency = Float.greatestFiniteMagnitude
        var oldestAccess = ContinuousClock.now
        
        // Look for entries at or near min frequency
        let frequencyRange = minFrequency...(minFrequency + 2)
        
        for (id, entry) in cache {
            if frequencyRange.contains(Int(entry.frequency)) {
                // Consider priority and age for tie-breaking
                let currentFrequency = entry.frequency
                let currentPriority = entry.entry.priority.rawValue
                let currentAccessTime = entry.lastAccessTime
                
                let isCandidate = currentFrequency < lowestFrequency ||
                    (currentFrequency == lowestFrequency && 
                     (candidateId == nil || 
                      currentPriority < cache[candidateId!]!.entry.priority.rawValue ||
                      currentAccessTime < oldestAccess))
                
                if isCandidate {
                    candidateId = id
                    lowestFrequency = currentFrequency
                    oldestAccess = currentAccessTime
                }
            }
        }
        
        if let id = candidateId {
            await remove(id: id)
            evictionCount += 1
            await performanceAnalyzer.recordEviction(count: 1)
        }
    }
    
    // Removed applyDecayIfNeeded - now handled by scheduler
    
    private func applyDecay() async {
        let now = ContinuousClock.now
        
        for (id, var entry) in cache {
            // Apply time-based decay
            let timeSinceAccess = (now - entry.lastAccessTime).components.seconds
            let decayMultiplier = pow(decayFactor, Float(Double(timeSinceAccess) / decayInterval))
            
            let oldFreq = Int(entry.frequency)
            entry.frequency *= decayMultiplier
            
            // Ensure minimum frequency
            entry.frequency = max(1.0, entry.frequency)
            
            cache[id] = entry
            
            // Update frequency list if changed significantly
            if Int(entry.frequency) != oldFreq {
                removeFromFrequencyList(id: id, frequency: oldFreq)
                addToFrequencyList(id: id, frequency: Int(entry.frequency))
            }
        }
        
        lastDecayTime = now
    }
    
    private func updateAdaptiveThreshold() async {
        // Track recent hit rates
        recentHitRates.append(hitRate)
        if recentHitRates.count > maxHitRateHistory {
            recentHitRates.removeFirst()
        }
        
        // Adjust threshold based on performance
        let avgHitRate = recentHitRates.reduce(0, +) / Float(recentHitRates.count)
        
        if avgHitRate < 0.5 {
            // Poor hit rate - be less aggressive with eviction
            adaptiveEvictionThreshold *= 1.1
        } else if avgHitRate > 0.8 {
            // Good hit rate - can be more aggressive
            adaptiveEvictionThreshold *= 0.9
        }
        
        // Keep threshold in reasonable bounds
        adaptiveEvictionThreshold = max(1.0, min(10.0, adaptiveEvictionThreshold))
        
        lastThresholdUpdate = ContinuousClock.now
    }
    
    private func cleanupFrequencyList() {
        frequencyList = frequencyList.filter { !$0.value.entries.isEmpty }
        minFrequency = frequencyList.keys.min() ?? 0
    }
    
    private func analyzeFrequencyDistribution() -> (mean: Float, variance: Float, skewness: Float) {
        guard !cache.isEmpty else { return (0, 0, 0) }
        
        let frequencies = cache.values.map { $0.frequency }
        let count = Float(frequencies.count)
        
        // Calculate mean
        let mean = frequencies.reduce(0, +) / count
        
        // Calculate variance
        let variance = frequencies.map { pow($0 - mean, 2) }.reduce(0, +) / count
        
        // Calculate skewness
        let stdDev = sqrt(variance)
        let skewness = stdDev > 0 ? 
            frequencies.map { pow(($0 - mean) / stdDev, 3) }.reduce(0, +) / count : 0
        
        return (mean, variance, skewness)
    }
}
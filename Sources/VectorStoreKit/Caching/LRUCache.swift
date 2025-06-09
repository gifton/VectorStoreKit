// VectorStoreKit: LRU Cache Implementation
//
// Least Recently Used (LRU) cache with medium complexity features

import Foundation
import simd

// MARK: - LRU Node

/// Node for doubly linked list used in LRU implementation
private class LRUNode<Vector: SIMD> where Vector.Scalar: BinaryFloatingPoint {
    let id: VectorID
    var entry: CacheEntry<Vector>
    var prev: LRUNode?
    var next: LRUNode?
    
    init(id: VectorID, entry: CacheEntry<Vector>) {
        self.id = id
        self.entry = entry
    }
}

// MARK: - BasicLRUVectorCache

/// Least Recently Used (LRU) cache implementation with enhanced features
public actor BasicLRUVectorCache<Vector: SIMD & Sendable>: VectorCache where Vector.Scalar: BinaryFloatingPoint {
    public typealias Configuration = LRUCacheConfiguration
    public typealias Statistics = BasicCacheStatistics
    
    public let configuration: Configuration
    
    // Use doubly linked list for O(1) LRU operations
    private var cache: [VectorID: LRUNode<Vector>] = [:]
    private var head: LRUNode<Vector>?
    private var tail: LRUNode<Vector>?
    
    private var currentMemoryUsage: Int = 0
    
    // Statistics tracking
    private var hitCount: Int = 0
    private var missCount: Int = 0
    private var evictionCount: Int = 0
    private var totalAccessTime: TimeInterval = 0
    
    // Access history for pattern analysis
    private var accessHistory: [VectorID] = []
    private let maxHistorySize = 1000
    
    // Performance analyzer
    private let performanceAnalyzer = CachePerformanceAnalyzer<Vector>()
    private let enhancedAnalyzer = EnhancedCachePerformanceAnalyzer<Vector>()
    
    // Predictive prefetch engine
    private let prefetchEngine = PredictivePrefetchEngine<Vector>()
    
    // Storage backend for preloading
    private let storageBackend: (any CacheStorageBackend<Vector>)?
    
    public init(maxMemory: Int, storageBackend: (any CacheStorageBackend<Vector>)? = nil) throws {
        guard maxMemory > 0 else {
            throw VectorCacheError.invalidConfiguration("maxMemory must be positive")
        }
        self.configuration = LRUCacheConfiguration(maxMemory: maxMemory)
        self.storageBackend = storageBackend
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
        let startTime = Date()
        defer {
            let latency = Date().timeIntervalSince(startTime)
            totalAccessTime += latency
            recordAccess(id: id)
            Task {
                await prefetchEngine.recordAccess(id: id)
                await enhancedAnalyzer.recordAccessLatency(latency)
            }
        }
        
        if let node = cache[id] {
            hitCount += 1
            // Move to front (most recently used)
            moveToFront(node)
            Task {
                await enhancedAnalyzer.recordAccess(hit: true)
            }
            return node.entry.vector
        } else {
            missCount += 1
            Task {
                await enhancedAnalyzer.recordAccess(hit: false)
            }
            return nil
        }
    }
    
    public func set(id: VectorID, vector: Vector, priority: CachePriority) async {
        let entrySize = estimateVectorMemorySize(vector)
        
        // Check if updating existing entry
        if let existingNode = cache[id] {
            // Update existing entry
            existingNode.entry = CacheEntry(
                vector: vector,
                priority: priority,
                timestamp: Date(),
                accessCount: existingNode.entry.accessCount + 1
            )
            moveToFront(existingNode)
            return
        }
        
        // Check if we need to evict entries
        while currentMemoryUsage + entrySize > configuration.maxMemory && !cache.isEmpty {
            await evictLeastRecentlyUsed()
        }
        
        // Create new node
        let entry = CacheEntry(vector: vector, priority: priority)
        let newNode = LRUNode(id: id, entry: entry)
        
        // Add to cache
        cache[id] = newNode
        currentMemoryUsage += entrySize
        
        // Add to front of list
        addToFront(newNode)
    }
    
    public func remove(id: VectorID) async {
        guard let node = cache.removeValue(forKey: id) else { return }
        
        let entrySize = estimateVectorMemorySize(node.entry.vector)
        currentMemoryUsage -= entrySize
        
        // Remove from linked list
        removeNode(node)
    }
    
    public func clear() async {
        evictionCount += cache.count
        cache.removeAll()
        head = nil
        tail = nil
        currentMemoryUsage = 0
        accessHistory.removeAll()
    }
    
    public func contains(id: VectorID) async -> Bool {
        cache[id] != nil
    }
    
    // MARK: - Advanced Operations (Medium Complexity)
    
    public func preload(ids: [VectorID]) async {
        guard let storage = storageBackend else { return }
        
        // Filter out already cached IDs
        let missingIds = ids.filter { cache[$0] == nil }
        guard !missingIds.isEmpty else { return }
        
        do {
            // Fetch vectors from storage
            let vectors = try await storage.fetchVectors(ids: missingIds)
            
            // Batch insert with priority based on request order
            let entries = vectors.map { (id, vector) -> (VectorID, Vector, CachePriority) in
                // Earlier in the list = higher priority
                let index = ids.firstIndex(of: id) ?? ids.count
                let priority: CachePriority = index < ids.count / 3 ? .high : .normal
                return (id, vector, priority)
            }
            
            // Manual batch insertion for LRU cache
            var inserted = 0
            var evicted = 0
            
            for (id, vector, priority) in entries {
                let entrySize = estimateVectorMemorySize(vector)
                
                // Check if we need to evict
                while currentMemoryUsage + entrySize > configuration.maxMemory && !cache.isEmpty {
                    await evictLeastRecentlyUsed()
                    evicted += 1
                }
                
                // Insert if space available
                if currentMemoryUsage + entrySize <= configuration.maxMemory {
                    let entry = CacheEntry(vector: vector, priority: priority)
                    let node = LRUNode(id: id, entry: entry)
                    cache[id] = node
                    currentMemoryUsage += entrySize
                    addToFront(node)
                    inserted += 1
                }
            }
            
            evictionCount += evicted
        } catch {
            // Log error but don't fail - preload is best effort
        }
    }
    
    public func prefetch(_ predictions: [VectorID: Float]) async {
        guard let storage = storageBackend else { return }
        
        // Get enhanced predictions from the prefetch engine
        let currentId = accessHistory.last
        let enhancedPredictions = await prefetchEngine.generatePredictions(
            currentId: currentId ?? "",
            predictions: predictions
        )
        
        // Filter out already cached items and sort by confidence
        let candidateIds = enhancedPredictions
            .filter { cache[$0.key] == nil }
            .sorted { $0.value > $1.value }
            .prefix(10) // Limit prefetch batch size
            .map { $0.key }
        
        guard !candidateIds.isEmpty else { return }
        
        // Check available memory
        let avgEntrySize = currentMemoryUsage / max(cache.count, 1)
        let estimatedSizeNeeded = avgEntrySize * candidateIds.count
        let availableMemory = configuration.maxMemory - currentMemoryUsage
        
        // Only prefetch if we have reasonable space
        guard estimatedSizeNeeded < Int(Double(availableMemory) * 0.3) else { return } // Use max 30% of free space
        
        do {
            // Fetch vectors from storage
            let vectors = try await storage.fetchVectors(ids: Array(candidateIds))
            
            // Insert with priority based on confidence
            for (id, vector) in vectors {
                let confidence = enhancedPredictions[id] ?? 0.5
                let priority: CachePriority = confidence > 0.7 ? .high :
                                            confidence > 0.4 ? .normal : .low
                
                // Check memory before each insertion
                let entrySize = estimateVectorMemorySize(vector)
                if currentMemoryUsage + entrySize > configuration.maxMemory {
                    // Apply more aggressive eviction for prefetched items
                    var evicted = false
                    
                    // First try to evict other low-priority prefetched items
                    for node in cache.values {
                        if node.entry.priority == .low && 
                           node.entry.accessCount == 0 { // Never accessed
                            await remove(id: node.id)
                            evicted = true
                            break
                        }
                    }
                    
                    if !evicted {
                        break // Stop prefetching if no space
                    }
                }
                
                // Create and insert prefetched entry
                let entry = CacheEntry(
                    vector: vector,
                    priority: priority,
                    timestamp: Date(),
                    metadata: CacheEntryMetadata(
                        source: "prefetch",
                        lastAccessed: Date(),
                        size: entrySize
                    )
                )
                
                let node = LRUNode(id: id, entry: entry)
                cache[id] = node
                currentMemoryUsage += entrySize
                
                // Add to back of LRU list (least recently used)
                if tail != nil {
                    tail!.next = node
                    node.prev = tail
                    tail = node
                } else {
                    head = node
                    tail = node
                }
            }
        } catch {
            // Prefetch failures are silent - best effort only
        }
    }
    
    public func optimize() async {
        // Remove stale entries based on access patterns
        let analysisResult = analyzeAccessPatterns()
        
        // Remove rarely accessed entries
        for id in analysisResult.rarelyAccessed {
            if let node = cache[id], node.entry.priority == .low {
                await remove(id: id)
            }
        }
        
        // Compact access history
        if accessHistory.count > maxHistorySize {
            accessHistory = Array(accessHistory.suffix(maxHistorySize / 2))
        }
        
        // Update performance metrics
        await performanceAnalyzer.recordMemoryUsage(currentMemoryUsage)
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
        // Use enhanced analyzer for comprehensive analysis
        let enhancedAnalysis = await enhancedAnalyzer.performComprehensiveAnalysis(
            currentHitRate: hitRate,
            currentMemoryUsage: currentMemoryUsage,
            maxMemory: configuration.maxMemory,
            cacheType: "LRU"
        )
        
        // Convert to standard CachePerformanceAnalysis format
        let hitRateHistory = await performanceAnalyzer.analyze(
            currentHitRate: hitRate,
            currentMemoryUsage: currentMemoryUsage,
            maxMemory: configuration.maxMemory
        ).hitRateOverTime
        
        // Combine recommendations from enhanced analysis
        var recommendations = enhancedAnalysis.recommendations
        
        // Add LRU-specific insights based on enhanced analysis
        if enhancedAnalysis.accessPatterns.sequentialityScore > 0.8 {
            recommendations.append(CacheRecommendation(
                type: .prefetching,
                description: "Highly sequential access pattern detected - enable streaming prefetch",
                expectedImprovement: 0.3
            ))
        }
        
        if enhancedAnalysis.anomalies.contains(where: { $0.type == .evictionAnomaly }) {
            recommendations.append(CacheRecommendation(
                type: .sizeAdjustment,
                description: "Eviction anomalies detected - cache may be undersized for workload",
                expectedImprovement: 0.25
            ))
        }
        
        // Include health score in recommendations if poor
        if enhancedAnalysis.healthScore < 50 {
            recommendations.insert(CacheRecommendation(
                type: .indexOptimization,
                description: "Cache health score low (\(Int(enhancedAnalysis.healthScore))%) - comprehensive optimization needed",
                expectedImprovement: 0.4
            ), at: 0)
        }
        
        // Calculate metrics separately to avoid type-checking timeout
        let memoryUtilization = Float(currentMemoryUsage) / Float(configuration.maxMemory)
        let totalAccesses = max(hitCount + missCount, 1)
        let evictionRate = Float(evictionCount) / Float(totalAccesses)
        
        // Calculate optimal size based on forecast
        let sizeFactor: Float = enhancedAnalysis.forecast.confidence > 0.7 ? 
            Float(enhancedAnalysis.forecast.predictedMemoryUsage) / Float(max(currentMemoryUsage, 1)) : 1.0
        let optimalCacheSize = Int(Float(configuration.maxMemory) * sizeFactor)
        
        return CachePerformanceAnalysis(
            hitRateOverTime: hitRateHistory,
            memoryUtilization: memoryUtilization,
            evictionRate: evictionRate,
            optimalCacheSize: optimalCacheSize,
            recommendations: recommendations
        )
    }
    
    // MARK: - Private Helpers
    
    private func moveToFront(_ node: LRUNode<Vector>) {
        guard node !== head else { return }
        
        removeNode(node)
        addToFront(node)
    }
    
    private func addToFront(_ node: LRUNode<Vector>) {
        node.next = head
        node.prev = nil
        
        if let currentHead = head {
            currentHead.prev = node
        }
        
        head = node
        
        if tail == nil {
            tail = node
        }
    }
    
    private func removeNode(_ node: LRUNode<Vector>) {
        let prev = node.prev
        let next = node.next
        
        prev?.next = next
        next?.prev = prev
        
        if node === head {
            head = next
        }
        
        if node === tail {
            tail = prev
        }
        
        node.prev = nil
        node.next = nil
    }
    
    private func evictLeastRecentlyUsed() async {
        guard let tailNode = tail else { return }
        
        // Consider priority in eviction
        var nodeToEvict = tailNode
        var current = tailNode.prev
        var stepsBack = 0
        let maxStepsBack = 5 // Check last 5 entries
        
        // Look for lower priority items near the tail
        while let node = current, stepsBack < maxStepsBack {
            if node.entry.priority.rawValue < nodeToEvict.entry.priority.rawValue {
                nodeToEvict = node
            }
            current = node.prev
            stepsBack += 1
        }
        
        await remove(id: nodeToEvict.id)
        evictionCount += 1
        await performanceAnalyzer.recordEviction(count: 1)
    }
    
    private func recordAccess(id: VectorID) {
        accessHistory.append(id)
        if accessHistory.count > maxHistorySize {
            accessHistory.removeFirst()
        }
    }
    
    private func analyzeAccessPatterns() -> SimpleCacheAccessPattern {
        // Analyze access frequency
        var accessCounts: [VectorID: Int] = [:]
        for id in accessHistory {
            accessCounts[id, default: 0] += 1
        }
        
        let avgAccessCount = Float(accessHistory.count) / Float(max(accessCounts.count, 1))
        let frequentThreshold = avgAccessCount * 2
        let rareThreshold = avgAccessCount * 0.5
        
        let frequentlyAccessed = Set(accessCounts.compactMap { 
            Float($0.value) > frequentThreshold ? $0.key : nil 
        })
        let rarelyAccessed = Set(accessCounts.compactMap { 
            Float($0.value) < rareThreshold ? $0.key : nil 
        })
        
        // Calculate temporal locality
        let temporalLocality = calculateTemporalLocality(accessHistory: accessHistory)
        
        return SimpleCacheAccessPattern(
            frequentlyAccessed: frequentlyAccessed,
            rarelyAccessed: rarelyAccessed,
            accessSequences: [], // Not implemented for medium complexity
            temporalLocality: temporalLocality,
            spatialLocality: 0.0 // Not implemented for medium complexity
        )
    }
}
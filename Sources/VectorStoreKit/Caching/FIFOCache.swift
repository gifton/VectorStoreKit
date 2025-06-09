// VectorStoreKit: FIFO Cache Implementation
//
// First In First Out (FIFO) cache with enhanced memory management

import Foundation
import simd

// MARK: - BasicFIFOVectorCache

/// First In First Out (FIFO) cache implementation with priority support
public actor BasicFIFOVectorCache<Vector: SIMD & Sendable>: VectorCache where Vector.Scalar: BinaryFloatingPoint {
    public typealias Configuration = FIFOCacheConfiguration
    public typealias Statistics = BasicCacheStatistics
    
    public let configuration: Configuration
    
    // Cache storage with insertion order tracking
    private var cache: [VectorID: CacheEntry<Vector>] = [:]
    private var insertionOrder: [VectorID] = []
    private var currentMemoryUsage: Int = 0
    
    // Priority queue for high-priority items
    private var priorityQueue: [VectorID] = []
    private let priorityProtectionRatio: Float = 0.2 // Protect 20% for high priority
    
    // Statistics
    private var hitCount: Int = 0
    private var missCount: Int = 0
    private var evictionCount: Int = 0
    private var totalAccessTime: TimeInterval = 0
    
    // Batch operation queue
    private var pendingBatch: [(VectorID, Vector, CachePriority)] = []
    private let batchSize = 100
    
    // Performance tracking
    private let performanceAnalyzer = CachePerformanceAnalyzer<Vector>()
    private var insertionTimes: [Date] = []
    private let maxTimeTracking = 1000
    
    // Predictive prefetch engine
    private let prefetchEngine = PredictivePrefetchEngine<Vector>()
    
    // Storage backend
    private let storageBackend: (any CacheStorageBackend<Vector>)?
    
    public init(maxMemory: Int, storageBackend: (any CacheStorageBackend<Vector>)? = nil) throws {
        guard maxMemory > 0 else {
            throw VectorCacheError.invalidConfiguration("maxMemory must be positive")
        }
        self.configuration = FIFOCacheConfiguration(maxMemory: maxMemory)
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
            totalAccessTime += Date().timeIntervalSince(startTime)
            Task {
                await prefetchEngine.recordAccess(id: id)
            }
        }
        
        if let entry = cache[id] {
            hitCount += 1
            // Update access count for statistics
            var updatedEntry = entry
            updatedEntry = CacheEntry(
                vector: entry.vector,
                priority: entry.priority,
                timestamp: entry.timestamp,
                accessCount: entry.accessCount + 1
            )
            cache[id] = updatedEntry
            return entry.vector
        } else {
            missCount += 1
            return nil
        }
    }
    
    public func set(id: VectorID, vector: Vector, priority: CachePriority) async {
        let entrySize = estimateVectorMemorySize(vector)
        
        // Update existing entry
        if insertionOrder.contains(id) {
            let existingEntry = cache[id]!
            cache[id] = CacheEntry(
                vector: vector,
                priority: priority,
                timestamp: existingEntry.timestamp, // Keep original insertion time
                accessCount: existingEntry.accessCount
            )
            
            // Update priority queue if needed
            updatePriorityQueue(id: id, oldPriority: existingEntry.priority, newPriority: priority)
            return
        }
        
        // Add to batch for efficient processing
        pendingBatch.append((id, vector, priority))
        
        // Process batch if full
        if pendingBatch.count >= batchSize {
            await processBatch()
        } else {
            // Process single item if no batch pending
            await processSingleInsertion(id: id, vector: vector, priority: priority, entrySize: entrySize)
        }
    }
    
    public func remove(id: VectorID) async {
        guard let entry = cache.removeValue(forKey: id) else { return }
        
        let entrySize = estimateVectorMemorySize(entry.vector)
        currentMemoryUsage -= entrySize
        
        insertionOrder.removeAll { $0 == id }
        priorityQueue.removeAll { $0 == id }
    }
    
    public func clear() async {
        evictionCount += cache.count
        cache.removeAll()
        insertionOrder.removeAll()
        priorityQueue.removeAll()
        currentMemoryUsage = 0
        pendingBatch.removeAll()
        insertionTimes.removeAll()
    }
    
    public func contains(id: VectorID) async -> Bool {
        cache[id] != nil
    }
    
    // MARK: - Advanced Operations
    
    public func preload(ids: [VectorID]) async {
        guard let storage = storageBackend else { return }
        
        let missingIds = ids.filter { cache[$0] == nil }
        guard !missingIds.isEmpty else { return }
        
        do {
            let vectors = try await storage.fetchVectors(ids: missingIds)
            
            // Preload with normal priority by default
            for (id, vector) in vectors {
                pendingBatch.append((id, vector, .normal))
            }
            
            // Process all preloaded items
            await processBatch()
        } catch {
            // Preload is best effort
        }
    }
    
    public func prefetch(_ predictions: [VectorID: Float]) async {
        guard let storage = storageBackend else { return }
        
        // Use the most recent access for context
        let currentId = insertionOrder.last
        
        // Get enhanced predictions
        let enhancedPredictions = await prefetchEngine.generatePredictions(
            currentId: currentId ?? "",
            predictions: predictions
        )
        
        // Filter and sort candidates
        let candidateIds = enhancedPredictions
            .filter { cache[$0.key] == nil }
            .sorted { $0.value > $1.value }
            .prefix(12) // FIFO can handle more since order is simple
            .map { $0.key }
        
        guard !candidateIds.isEmpty else { return }
        
        // Memory check
        let avgSize = currentMemoryUsage / max(cache.count, 1)
        let neededSize = avgSize * candidateIds.count
        let available = configuration.maxMemory - currentMemoryUsage
        
        guard neededSize < Int(Double(available) * 0.4) else { return } // Use up to 40% of free space
        
        do {
            let vectors = try await storage.fetchVectors(ids: Array(candidateIds))
            
            // Create batch for efficient insertion
            var prefetchBatch: [(VectorID, Vector, CachePriority)] = []
            
            for (id, vector) in vectors {
                let confidence = enhancedPredictions[id] ?? 0.4
                
                // FIFO respects insertion order, so priority matters less
                let priority: CachePriority = confidence > 0.8 ? .high :
                                            confidence > 0.5 ? .normal : .low
                
                prefetchBatch.append((id, vector, priority))
            }
            
            // Add to pending batch for processing
            pendingBatch.append(contentsOf: prefetchBatch)
            
            // Process immediately if batch is large enough
            if pendingBatch.count >= batchSize {
                await processBatch()
            }
        } catch {
            // Silent failure for prefetch
        }
    }
    
    public func optimize() async {
        // Process any pending batch operations
        if !pendingBatch.isEmpty {
            await processBatch()
        }
        
        // Analyze age distribution
        let ageAnalysis = analyzeAgeDistribution()
        
        // Remove very old low-priority entries if cache is nearly full
        if Float(currentMemoryUsage) / Float(configuration.maxMemory) > 0.9 {
            let cutoffAge = ageAnalysis.percentile75
            var toRemove: [VectorID] = []
            
            for id in insertionOrder {
                if let entry = cache[id],
                   entry.priority == .low,
                   Date().timeIntervalSince(entry.timestamp) > cutoffAge {
                    toRemove.append(id)
                }
            }
            
            // Remove up to 10% of cache
            let removeCount = min(toRemove.count, cache.count / 10)
            for i in 0..<removeCount {
                await remove(id: toRemove[i])
            }
        }
        
        // Update performance metrics
        await performanceAnalyzer.recordMemoryUsage(currentMemoryUsage)
        
        // Trim tracking arrays
        if insertionTimes.count > maxTimeTracking {
            insertionTimes = Array(insertionTimes.suffix(maxTimeTracking / 2))
        }
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
        
        // FIFO-specific analysis
        let ageAnalysis = analyzeAgeDistribution()
        
        // Check for rapid turnover
        if ageAnalysis.averageAge < 60 { // Less than 1 minute average
            recommendations.append(CacheRecommendation(
                type: .sizeAdjustment,
                description: "High turnover rate detected - cache may be too small",
                expectedImprovement: 0.25
            ))
        }
        
        // Check priority distribution
        let priorityStats = analyzePriorityDistribution()
        if priorityStats.highPriorityRatio > 0.5 {
            recommendations.append(CacheRecommendation(
                type: .partitioning,
                description: "High ratio of priority items - consider dedicated priority cache",
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
    
    private func processSingleInsertion(
        id: VectorID,
        vector: Vector,
        priority: CachePriority,
        entrySize: Int
    ) async {
        // Check if we need to evict
        while currentMemoryUsage + entrySize > configuration.maxMemory && !cache.isEmpty {
            await evictOldest()
        }
        
        // Add new entry
        let entry = CacheEntry(vector: vector, priority: priority)
        cache[id] = entry
        insertionOrder.append(id)
        currentMemoryUsage += entrySize
        
        // Track insertion time
        insertionTimes.append(Date())
        if insertionTimes.count > maxTimeTracking {
            insertionTimes.removeFirst()
        }
        
        // Add to priority queue if high priority
        if priority == .high || priority == .critical {
            priorityQueue.append(id)
        }
    }
    
    private func processBatch() async {
        guard !pendingBatch.isEmpty else { return }
        
        // Calculate total size needed
        let totalSizeNeeded = pendingBatch.reduce(0) { sum, item in
            sum + estimateVectorMemorySize(item.1)
        }
        
        // Bulk eviction if needed
        if currentMemoryUsage + totalSizeNeeded > configuration.maxMemory {
            let targetSize = configuration.maxMemory - totalSizeNeeded
            await bulkEvict(targetSize: targetSize)
        }
        
        // Insert all pending items
        let now = Date()
        for (id, vector, priority) in pendingBatch {
            let entrySize = estimateVectorMemorySize(vector)
            
            // Skip if still not enough space
            if currentMemoryUsage + entrySize > configuration.maxMemory {
                continue
            }
            
            let entry = CacheEntry(vector: vector, priority: priority, timestamp: now)
            cache[id] = entry
            insertionOrder.append(id)
            currentMemoryUsage += entrySize
            
            if priority == .high || priority == .critical {
                priorityQueue.append(id)
            }
        }
        
        // Track batch insertion
        insertionTimes.append(now)
        
        pendingBatch.removeAll()
    }
    
    private func evictOldest() async {
        // Protect high-priority items
        var candidateIndex = 0
        
        // Find first non-protected item
        while candidateIndex < insertionOrder.count {
            let id = insertionOrder[candidateIndex]
            if let entry = cache[id],
               entry.priority != .high && entry.priority != .critical {
                break
            }
            candidateIndex += 1
        }
        
        // If all items are protected, evict the oldest regardless
        if candidateIndex >= insertionOrder.count {
            candidateIndex = 0
        }
        
        if candidateIndex < insertionOrder.count {
            let id = insertionOrder[candidateIndex]
            await remove(id: id)
            evictionCount += 1
            await performanceAnalyzer.recordEviction(count: 1)
        }
    }
    
    private func bulkEvict(targetSize: Int) async {
        var currentSize = currentMemoryUsage
        var evicted = 0
        
        // First pass: evict low priority items
        var i = 0
        while i < insertionOrder.count && currentSize > targetSize {
            let id = insertionOrder[i]
            if let entry = cache[id], entry.priority == .low {
                let entrySize = estimateVectorMemorySize(entry.vector)
                cache.removeValue(forKey: id)
                insertionOrder.remove(at: i)
                currentSize -= entrySize
                evicted += 1
            } else {
                i += 1
            }
        }
        
        // Second pass: evict normal priority items if needed
        i = 0
        while i < insertionOrder.count && currentSize > targetSize {
            let id = insertionOrder[i]
            if let entry = cache[id], entry.priority == .normal {
                let entrySize = estimateVectorMemorySize(entry.vector)
                cache.removeValue(forKey: id)
                insertionOrder.remove(at: i)
                currentSize -= entrySize
                evicted += 1
            } else {
                i += 1
            }
        }
        
        currentMemoryUsage = currentSize
        evictionCount += evicted
        await performanceAnalyzer.recordEviction(count: evicted)
    }
    
    private func updatePriorityQueue(id: VectorID, oldPriority: CachePriority, newPriority: CachePriority) {
        let wasHighPriority = oldPriority == .high || oldPriority == .critical
        let isHighPriority = newPriority == .high || newPriority == .critical
        
        if wasHighPriority && !isHighPriority {
            priorityQueue.removeAll { $0 == id }
        } else if !wasHighPriority && isHighPriority {
            priorityQueue.append(id)
        }
    }
    
    private func analyzeAgeDistribution() -> (averageAge: TimeInterval, median: TimeInterval, percentile75: TimeInterval) {
        let now = Date()
        let ages = insertionOrder.compactMap { id -> TimeInterval? in
            cache[id].map { now.timeIntervalSince($0.timestamp) }
        }.sorted()
        
        guard !ages.isEmpty else { return (0, 0, 0) }
        
        let average = ages.reduce(0, +) / Double(ages.count)
        let median = ages[ages.count / 2]
        let percentile75 = ages[min(Int(Double(ages.count) * 0.75), ages.count - 1)]
        
        return (average, median, percentile75)
    }
    
    private func analyzePriorityDistribution() -> (highPriorityRatio: Float, distribution: [CachePriority: Int]) {
        var distribution: [CachePriority: Int] = [:]
        
        for entry in cache.values {
            distribution[entry.priority, default: 0] += 1
        }
        
        let highPriorityCount = (distribution[.high] ?? 0) + (distribution[.critical] ?? 0)
        let highPriorityRatio = cache.isEmpty ? 0.0 : Float(highPriorityCount) / Float(cache.count)
        
        return (highPriorityRatio, distribution)
    }
}
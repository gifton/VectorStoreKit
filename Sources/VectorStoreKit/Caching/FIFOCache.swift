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
    
    // Two-queue model: protected (high priority) and open queues
    private var protectedQueue: [VectorID] = []  // High/Critical priority items
    private var openQueue: [VectorID] = []        // Normal/Low priority items
    private let protectedRatio: Float = 0.2       // 20% reserved for protected queue
    private var protectedMemoryUsage: Int = 0
    private var openMemoryUsage: Int = 0
    
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
    private var insertionTimes: [ContinuousClock.Instant] = []
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
        let startTime = ContinuousClock.now
        defer {
            let elapsed = ContinuousClock.now - startTime
            totalAccessTime += Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
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
            
            // Record access synchronously
            await prefetchEngine.recordAccess(id: id)
            
            return entry.vector
        } else {
            missCount += 1
            
            // Record access synchronously
            await prefetchEngine.recordAccess(id: id)
            
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
        protectedQueue.removeAll { $0 == id }
        openQueue.removeAll { $0 == id }
        
        // Update memory tracking
        if entry.priority == .high || entry.priority == .critical {
            protectedMemoryUsage -= entrySize
        } else {
            openMemoryUsage -= entrySize
        }
    }
    
    public func clear() async {
        evictionCount += cache.count
        cache.removeAll()
        insertionOrder.removeAll()
        protectedQueue.removeAll()
        openQueue.removeAll()
        protectedMemoryUsage = 0
        openMemoryUsage = 0
        currentMemoryUsage = 0
        pendingBatch.removeAll()
        insertionTimes.removeAll()
    }
    
    public func contains(id: VectorID) async -> Bool {
        cache[id] != nil
    }
    
    public func currentSize() async -> Int {
        cache.count
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
        
        // Two-queue analysis
        let protectedRatio = Float(protectedMemoryUsage) / Float(configuration.maxMemory)
        if protectedRatio < 0.1 {
            recommendations.append(CacheRecommendation(
                type: .configuration,
                description: "Protected queue underutilized - consider adjusting priority thresholds",
                expectedImprovement: 0.1
            ))
        } else if protectedRatio > 0.3 {
            recommendations.append(CacheRecommendation(
                type: .configuration,
                description: "Protected queue overloaded - consider increasing protectedRatio",
                expectedImprovement: 0.2
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
        insertionTimes.append(ContinuousClock.now)
        if insertionTimes.count > maxTimeTracking {
            insertionTimes.removeFirst()
        }
        
        // Add to appropriate queue based on priority
        if priority == .high || priority == .critical {
            protectedQueue.append(id)
            protectedMemoryUsage += entrySize
        } else {
            openQueue.append(id)
            openMemoryUsage += entrySize
        }
    }
    
    private func processBatch() async {
        guard !pendingBatch.isEmpty else { return }
        
        // Insert pending items one by one with proper eviction
        let now = Date()
        for (id, vector, priority) in pendingBatch {
            let entrySize = estimateVectorMemorySize(vector)
            
            // Evict until we have space for this entry
            while currentMemoryUsage + entrySize > configuration.maxMemory && !cache.isEmpty {
                await evictOldest()
            }
            
            // Skip if still not enough space (cache is empty but entry too large)
            if currentMemoryUsage + entrySize > configuration.maxMemory {
                continue
            }
            
            let entry = CacheEntry(vector: vector, priority: priority, timestamp: now)
            cache[id] = entry
            insertionOrder.append(id)
            currentMemoryUsage += entrySize
            
            if priority == .high || priority == .critical {
                protectedQueue.append(id)
                protectedMemoryUsage += entrySize
            } else {
                openQueue.append(id)
                openMemoryUsage += entrySize
            }
        }
        
        // Track batch insertion
        insertionTimes.append(ContinuousClock.now)
        
        pendingBatch.removeAll()
    }
    
    private func evictOldest() async {
        // Two-queue eviction policy
        let protectedLimit = Int(Float(configuration.maxMemory) * protectedRatio)
        
        // First, try to evict from open queue
        if !openQueue.isEmpty {
            let id = openQueue.removeFirst()
            if let entry = cache[id] {
                let entrySize = estimateVectorMemorySize(entry.vector)
                cache.removeValue(forKey: id)
                insertionOrder.removeAll { $0 == id }
                currentMemoryUsage -= entrySize
                openMemoryUsage -= entrySize
                evictionCount += 1
                await performanceAnalyzer.recordEviction(count: 1)
                return
            }
        }
        
        // If open queue is empty or protected queue exceeds limit, evict from protected
        if protectedMemoryUsage > protectedLimit && !protectedQueue.isEmpty {
            let id = protectedQueue.removeFirst()
            if let entry = cache[id] {
                let entrySize = estimateVectorMemorySize(entry.vector)
                cache.removeValue(forKey: id)
                insertionOrder.removeAll { $0 == id }
                currentMemoryUsage -= entrySize
                protectedMemoryUsage -= entrySize
                evictionCount += 1
                await performanceAnalyzer.recordEviction(count: 1)
            }
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
        
        if wasHighPriority != isHighPriority {
            // Get the entry size for memory tracking
            if let entry = cache[id] {
                let entrySize = estimateVectorMemorySize(entry.vector)
                
                if wasHighPriority && !isHighPriority {
                    // Move from protected to open queue
                    protectedQueue.removeAll { $0 == id }
                    openQueue.append(id)
                    protectedMemoryUsage -= entrySize
                    openMemoryUsage += entrySize
                } else if !wasHighPriority && isHighPriority {
                    // Move from open to protected queue
                    openQueue.removeAll { $0 == id }
                    protectedQueue.append(id)
                    openMemoryUsage -= entrySize
                    protectedMemoryUsage += entrySize
                }
            }
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
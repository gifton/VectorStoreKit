// VectorStoreKit: Adaptive Replacement Cache (ARC) Implementation
//
// ARC is a self-tuning, low-overhead cache replacement policy that
// dynamically balances between recency and frequency
//
// Based on: "ARC: A Self-Tuning, Low Overhead Replacement Cache"
// by Nimrod Megiddo and Dharmendra S. Modha

import Foundation
import simd

// MARK: - ARC Ghost List Entry

/// Ghost list entry for tracking evicted items
private struct GhostEntry {
    let id: VectorID
    let timestamp: Date
    let size: Int
}

// MARK: - Adaptive Replacement Cache

/// ARC implementation with dynamic adaptation between LRU and LFU
public actor AdaptiveReplacementCache<Vector: SIMD & Sendable>: VectorCache 
where Vector.Scalar: BinaryFloatingPoint {
    
    public typealias Configuration = ARCCacheConfiguration
    public typealias Statistics = BasicCacheStatistics
    
    // MARK: - Properties
    
    public let configuration: Configuration
    
    // Four lists: T1 (recent), T2 (frequent), B1 (ghost recent), B2 (ghost frequent)
    private var t1: OrderedSet<VectorID> = OrderedSet()  // Recent cache entries
    private var t2: OrderedSet<VectorID> = OrderedSet()  // Frequent cache entries
    private var b1: OrderedSet<VectorID> = OrderedSet()  // Ghost list for T1
    private var b2: OrderedSet<VectorID> = OrderedSet()  // Ghost list for T2
    
    // Actual cache storage
    private var cache: [VectorID: CacheEntry<Vector>] = [:]
    
    // Target size for T1 (dynamically adapted)
    private var p: Int = 0
    
    // Memory tracking
    private var currentMemoryUsage: Int = 0
    private var memoryMap: [VectorID: Int] = [:]
    
    // Statistics
    private var hitCount: Int = 0
    private var missCount: Int = 0
    private var evictionCount: Int = 0
    private var totalAccessTime: TimeInterval = 0
    
    // Ghost list metadata
    private var ghostMetadata: [VectorID: GhostEntry] = [:]
    
    // Performance tracking
    private let performanceAnalyzer = CachePerformanceAnalyzer<Vector>()
    private let enhancedAnalyzer = EnhancedCachePerformanceAnalyzer<Vector>()
    
    // Storage backend
    private let storageBackend: (any CacheStorageBackend<Vector>)?
    
    // MARK: - Initialization
    
    public init(
        maxMemory: Int,
        storageBackend: (any CacheStorageBackend<Vector>)? = nil
    ) throws {
        guard maxMemory > 0 else {
            throw VectorStoreError.configurationInvalid("maxMemory must be positive")
        }
        
        self.configuration = ARCCacheConfiguration(maxMemory: maxMemory)
        self.storageBackend = storageBackend
        self.p = maxMemory / 2  // Start with balanced split
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
            totalAccessTime += Double(elapsed.components.seconds) + 
                             Double(elapsed.components.attoseconds) / 1e18
        }
        
        // Case 1: Hit in T1 or T2
        if let entry = cache[id] {
            hitCount += 1
            
            // Move to T2 (mark as frequent)
            if t1.contains(id) {
                t1.remove(id)
                t2.append(id)
            } else if t2.contains(id) {
                // Move to MRU position in T2
                t2.remove(id)
                t2.append(id)
            }
            
            return entry.vector
        }
        
        missCount += 1
        
        // Case 2: Hit in ghost list B1
        if b1.contains(id) {
            // Adapt: increase p (favor recency)
            let delta = b2.count > b1.count ? b1.count / b2.count : 1
            p = min(p + delta, configuration.maxMemory)
            
            // Handle replacement and fetch
            await replace(id: id, inB2: false)
            
            // Try to fetch from storage
            if let vector = await fetchFromStorage(id: id) {
                await insertIntoT2(id: id, vector: vector)
                return vector
            }
        }
        
        // Case 3: Hit in ghost list B2
        else if b2.contains(id) {
            // Adapt: decrease p (favor frequency)
            let delta = b1.count > b2.count ? b2.count / b1.count : 1
            p = max(p - delta, 0)
            
            // Handle replacement and fetch
            await replace(id: id, inB2: true)
            
            // Try to fetch from storage
            if let vector = await fetchFromStorage(id: id) {
                await insertIntoT2(id: id, vector: vector)
                return vector
            }
        }
        
        return nil
    }
    
    public func set(id: VectorID, vector: Vector, priority: CachePriority) async {
        let entrySize = estimateVectorMemorySize(vector)
        
        // If already in cache, update it
        if cache[id] != nil {
            // Update entry
            cache[id] = CacheEntry(
                vector: vector,
                priority: priority,
                timestamp: Date()
            )
            return
        }
        
        // Check if we have space
        let totalSize = t1.count + t2.count
        let maxItems = configuration.maxMemory / averageEntrySize()
        
        // Case 4: Cache miss, not in ghost lists
        if !b1.contains(id) && !b2.contains(id) {
            // Case 4.1: L1 has space
            if totalSize < maxItems {
                // Insert into T1
                await insertIntoT1(id: id, vector: vector, priority: priority)
            } else {
                // Need to replace
                if t1.count < maxItems {
                    await replace(id: id, inB2: false)
                    await insertIntoT1(id: id, vector: vector, priority: priority)
                } else {
                    // Evict from T1
                    await evictFromT1()
                    await insertIntoT1(id: id, vector: vector, priority: priority)
                }
            }
        }
    }
    
    public func remove(id: VectorID) async {
        // Remove from all lists
        t1.remove(id)
        t2.remove(id)
        b1.remove(id)
        b2.remove(id)
        
        if let entry = cache.removeValue(forKey: id) {
            let size = memoryMap.removeValue(forKey: id) ?? 0
            currentMemoryUsage -= size
        }
        
        ghostMetadata.removeValue(forKey: id)
    }
    
    public func clear() async {
        evictionCount += cache.count
        
        t1.removeAll()
        t2.removeAll()
        b1.removeAll()
        b2.removeAll()
        cache.removeAll()
        memoryMap.removeAll()
        ghostMetadata.removeAll()
        currentMemoryUsage = 0
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
            
            for (id, vector) in vectors {
                await set(id: id, vector: vector, priority: .normal)
            }
        } catch {
            // Preload is best effort
        }
    }
    
    public func prefetch(_ predictions: [VectorID: Float]) async {
        guard let storage = storageBackend else { return }
        
        // Filter predictions not in cache
        let candidates = predictions
            .filter { cache[$0.key] == nil }
            .sorted { $0.value > $1.value }
            .prefix(10)
        
        guard !candidates.isEmpty else { return }
        
        do {
            let ids = candidates.map { $0.key }
            let vectors = try await storage.fetchVectors(ids: Array(ids))
            
            for (id, vector) in vectors {
                let confidence = predictions[id] ?? 0.5
                let priority: CachePriority = confidence > 0.7 ? .high : .normal
                await set(id: id, vector: vector, priority: priority)
            }
        } catch {
            // Prefetch is best effort
        }
    }
    
    public func optimize() async {
        // Clean up ghost lists if they grow too large
        let maxGhostSize = configuration.maxMemory / averageEntrySize()
        
        if b1.count > maxGhostSize {
            let toRemove = b1.count - maxGhostSize
            for _ in 0..<toRemove {
                if let id = b1.first {
                    b1.removeFirst()
                    ghostMetadata.removeValue(forKey: id)
                }
            }
        }
        
        if b2.count > maxGhostSize {
            let toRemove = b2.count - maxGhostSize
            for _ in 0..<toRemove {
                if let id = b2.first {
                    b2.removeFirst()
                    ghostMetadata.removeValue(forKey: id)
                }
            }
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
        let analysis = await performanceAnalyzer.analyze(
            currentHitRate: hitRate,
            currentMemoryUsage: currentMemoryUsage,
            maxMemory: configuration.maxMemory
        )
        
        var recommendations = analysis.recommendations
        
        // ARC-specific recommendations
        let t1Ratio = Float(t1.count) / Float(max(t1.count + t2.count, 1))
        if t1Ratio > 0.8 {
            recommendations.append(CacheRecommendation(
                type: .policyChange,
                description: "Workload is highly recency-focused, consider pure LRU",
                expectedImprovement: 0.05
            ))
        } else if t1Ratio < 0.2 {
            recommendations.append(CacheRecommendation(
                type: .policyChange,
                description: "Workload is highly frequency-focused, consider pure LFU",
                expectedImprovement: 0.05
            ))
        }
        
        // Check adaptation rate
        let adaptationRate = Float(abs(p - configuration.maxMemory / 2)) / 
                           Float(configuration.maxMemory / 2)
        if adaptationRate < 0.1 {
            recommendations.append(CacheRecommendation(
                type: .indexOptimization,
                description: "ARC adaptation is minimal - workload is well-balanced",
                expectedImprovement: 0.0
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
    
    // MARK: - Private Methods
    
    private func replace(id: VectorID, inB2: Bool) async {
        let t1Size = t1.count
        let t2Size = t2.count
        
        if t1Size > 0 && (t1Size > p || (inB2 && t1Size == p)) {
            // Replace from T1
            if let victim = t1.first {
                await evictFromT1(specificId: victim)
            }
        } else if t2Size > 0 {
            // Replace from T2
            if let victim = t2.first {
                await evictFromT2(specificId: victim)
            }
        }
    }
    
    private func insertIntoT1(
        id: VectorID,
        vector: Vector,
        priority: CachePriority
    ) async {
        let entry = CacheEntry(
            vector: vector,
            priority: priority,
            timestamp: Date()
        )
        
        cache[id] = entry
        t1.append(id)
        
        let size = estimateVectorMemorySize(vector)
        memoryMap[id] = size
        currentMemoryUsage += size
        
        // Remove from ghost lists if present
        b1.remove(id)
        b2.remove(id)
    }
    
    private func insertIntoT2(id: VectorID, vector: Vector) async {
        let entry = CacheEntry(
            vector: vector,
            priority: .normal,
            timestamp: Date()
        )
        
        cache[id] = entry
        t2.append(id)
        
        let size = estimateVectorMemorySize(vector)
        memoryMap[id] = size
        currentMemoryUsage += size
        
        // Remove from ghost lists if present
        b1.remove(id)
        b2.remove(id)
    }
    
    private func evictFromT1(specificId: VectorID? = nil) async {
        let victim = specificId ?? t1.first
        guard let id = victim else { return }
        
        t1.remove(id)
        b1.append(id)
        
        if let entry = cache.removeValue(forKey: id) {
            let size = memoryMap.removeValue(forKey: id) ?? 0
            currentMemoryUsage -= size
            evictionCount += 1
            
            // Store ghost metadata
            ghostMetadata[id] = GhostEntry(
                id: id,
                timestamp: Date(),
                size: size
            )
        }
    }
    
    private func evictFromT2(specificId: VectorID? = nil) async {
        let victim = specificId ?? t2.first
        guard let id = victim else { return }
        
        t2.remove(id)
        b2.append(id)
        
        if let entry = cache.removeValue(forKey: id) {
            let size = memoryMap.removeValue(forKey: id) ?? 0
            currentMemoryUsage -= size
            evictionCount += 1
            
            // Store ghost metadata
            ghostMetadata[id] = GhostEntry(
                id: id,
                timestamp: Date(),
                size: size
            )
        }
    }
    
    private func fetchFromStorage(id: VectorID) async -> Vector? {
        guard let storage = storageBackend else { return nil }
        
        do {
            let vectors = try await storage.fetchVectors(ids: [id])
            return vectors[id]
        } catch {
            return nil
        }
    }
    
    private func averageEntrySize() -> Int {
        guard !memoryMap.isEmpty else {
            // Estimate based on Vector type
            return MemoryLayout<Vector>.size + 128  // Vector + metadata overhead
        }
        
        let totalSize = memoryMap.values.reduce(0, +)
        return totalSize / memoryMap.count
    }
}

// MARK: - ARC Configuration

/// Configuration for Adaptive Replacement Cache
public struct ARCCacheConfiguration: CacheConfiguration {
    public let maxMemory: Int
    public let ghostListMultiplier: Float
    
    public init(
        maxMemory: Int,
        ghostListMultiplier: Float = 2.0
    ) {
        self.maxMemory = maxMemory
        self.ghostListMultiplier = ghostListMultiplier
    }
    
    public func validate() throws {
        guard maxMemory > 0 else {
            throw VectorStoreError.configurationInvalid("maxMemory must be positive")
        }
        guard ghostListMultiplier >= 1.0 else {
            throw VectorStoreError.configurationInvalid("ghostListMultiplier must be >= 1.0")
        }
    }
    
    public func estimatedMemoryUsage(for entryCount: Int) -> Int {
        // Account for ghost lists
        let baseUsage = maxMemory
        let ghostUsage = Int(Float(maxMemory) * (ghostListMultiplier - 1.0))
        return baseUsage + ghostUsage
    }
    
    public func computationalComplexity() -> ComputationalComplexity {
        ComputationalComplexity(
            timeComplexity: "O(1)",
            spaceComplexity: "O(n)",
            description: "Constant time operations with linear space"
        )
    }
}

// MARK: - Ordered Set Helper

/// Simple ordered set implementation for ARC lists
private struct OrderedSet<Element: Hashable> {
    private var array: [Element] = []
    private var set: Set<Element> = []
    
    var count: Int { array.count }
    var first: Element? { array.first }
    var isEmpty: Bool { array.isEmpty }
    
    mutating func append(_ element: Element) {
        guard !set.contains(element) else { return }
        array.append(element)
        set.insert(element)
    }
    
    mutating func remove(_ element: Element) {
        guard set.contains(element) else { return }
        array.removeAll { $0 == element }
        set.remove(element)
    }
    
    mutating func removeFirst() {
        guard let first = array.first else { return }
        array.removeFirst()
        set.remove(first)
    }
    
    mutating func removeAll() {
        array.removeAll()
        set.removeAll()
    }
    
    func contains(_ element: Element) -> Bool {
        set.contains(element)
    }
}
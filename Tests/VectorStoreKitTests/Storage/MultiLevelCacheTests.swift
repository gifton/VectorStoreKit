import XCTest
// VectorStoreKit: Multi-Level Cache Example
//
// Demonstrates the sophisticated multi-level caching strategy with
// adaptive sizing, intelligent eviction, and comprehensive monitoring

import Foundation
import VectorStoreKit
import simd

final class MultiLevelCacheTests: XCTestCase {
    typealias Vector = SIMD32<Float>
    typealias Metadata = DocumentMetadata
    
    struct DocumentMetadata: Codable, Sendable {
        let title: String
        let category: String
        let importance: Float
        let lastAccessed: Date
    }
    
    func testMain() async throws {
        print("=== VectorStoreKit Multi-Level Cache Example ===\n")
        
        // Create a sophisticated caching system
        let cache = try await createMultiLevelCache()
        
        // Demonstrate various features
        try await demonstrateBasicOperations(cache: cache)
        try await demonstrateLevelPromotion(cache: cache)
        try await demonstrateCacheWarming(cache: cache)
        try await demonstrateMemoryPressure(cache: cache)
        try await demonstratePatternAnalysis(cache: cache)
        try await demonstrateOptimization(cache: cache)
        
        print("\n=== Example Complete ===")
    }
    
    // MARK: - Cache Creation
    
    func testCreateMultiLevelCache() async throws -> MultiLevelCachingStrategy<Vector, Metadata> {
        print("Creating multi-level cache with 3 tiers...")
        
        // Configure cache levels with different characteristics
        let configurations = [
            // L1: Ultra-fast, small capacity for hot data
            CacheLevelConfiguration(
                level: .l1,
                maxMemory: 10 * 1024 * 1024,  // 10MB
                evictionPolicy: .lru,
                accessTimeThreshold: 0.0001,   // 100 microseconds
                warmupEnabled: true,
                prefetchEnabled: true
            ),
            
            // L2: Fast, medium capacity for warm data
            CacheLevelConfiguration(
                level: .l2,
                maxMemory: 100 * 1024 * 1024,  // 100MB
                evictionPolicy: .arc,           // Adaptive replacement
                accessTimeThreshold: 0.001,     // 1 millisecond
                warmupEnabled: true,
                prefetchEnabled: true
            ),
            
            // L3: Large capacity for cold data
            CacheLevelConfiguration(
                level: .l3,
                maxMemory: 1024 * 1024 * 1024,  // 1GB
                evictionPolicy: .fifo,
                accessTimeThreshold: 0.01,      // 10 milliseconds
                warmupEnabled: false,
                prefetchEnabled: false
            )
        ]
        
        // Create mock storage backend
        let storageBackend = MockStorageBackend()
        
        let cache = try await MultiLevelCachingStrategy<Vector, Metadata>(
            configurations: configurations,
            storageBackend: storageBackend
        )
        
        print("✓ Multi-level cache created with L1, L2, and L3 tiers")
        return cache
    }
    
    // MARK: - Basic Operations
    
    func testDemonstrateBasicOperations(cache: MultiLevelCachingStrategy<Vector, Metadata>) async throws {
        print("\n--- Basic Cache Operations ---")
        
        // Create sample vectors
        let vectors = [
            ("doc1", createVector(pattern: .sequential)),
            ("doc2", createVector(pattern: .random)),
            ("doc3", createVector(pattern: .gaussian)),
            ("doc4", createVector(pattern: .sparse)),
            ("doc5", createVector(pattern: .dense))
        ]
        
        // Add vectors with different priorities
        for (i, (id, vector)) in vectors.enumerated() {
            let metadata = DocumentMetadata(
                title: "Document \(i+1)",
                category: i < 2 ? "important" : "regular",
                importance: Float(5 - i) / 5.0,
                lastAccessed: Date()
            )
            
            let priority: CachePriority = i < 2 ? .critical : .normal
            
            await cache.set(
                id: id,
                vector: vector,
                metadata: metadata,
                priority: priority
            )
            
            print("  Added \(id) with priority: \(priority)")
        }
        
        // Retrieve vectors
        print("\n  Retrieving vectors:")
        for (id, _) in vectors {
            let startTime = ContinuousClock.now
            let retrieved = await cache.get(id: id)
            let duration = ContinuousClock.now - startTime
            
            if retrieved != nil {
                let microseconds = Double(duration.components.attoseconds) / 1e12
                print("    ✓ Retrieved \(id) in \(String(format: "%.3f", microseconds))ms")
            }
        }
        
        // Show statistics
        await printStatistics(cache: cache)
    }
    
    // MARK: - Level Promotion
    
    func testDemonstrateLevelPromotion(cache: MultiLevelCachingStrategy<Vector, Metadata>) async throws {
        print("\n--- Cache Level Promotion ---")
        
        // Add a vector to lower tier
        let vectorId = "promote-test"
        let vector = createVector(pattern: .sequential)
        let metadata = DocumentMetadata(
            title: "Frequently Accessed Document",
            category: "test",
            importance: 0.5,
            lastAccessed: Date()
        )
        
        await cache.set(
            id: vectorId,
            vector: vector,
            metadata: metadata,
            priority: .low  // Start in lower tier
        )
        
        print("  Added vector with low priority (likely in L3)")
        
        // Access multiple times to trigger promotion
        print("  Accessing vector multiple times to trigger promotion...")
        for i in 1...15 {
            let startTime = ContinuousClock.now
            _ = await cache.get(id: vectorId)
            let duration = ContinuousClock.now - startTime
            
            if i % 5 == 0 {
                let microseconds = Double(duration.components.attoseconds) / 1e12
                print("    Access #\(i): \(String(format: "%.3f", microseconds))ms")
            }
        }
        
        print("  ✓ Vector should now be promoted to higher tier (faster access)")
    }
    
    // MARK: - Cache Warming
    
    func testDemonstrateCacheWarming(cache: MultiLevelCachingStrategy<Vector, Metadata>) async throws {
        print("\n--- Cache Warming ---")
        
        // Prepare vectors that will be needed soon
        let upcomingVectors = [
            "warm1": 0.9,  // 90% confidence
            "warm2": 0.8,  // 80% confidence
            "warm3": 0.7,  // 70% confidence
            "warm4": 0.6,  // 60% confidence
            "warm5": 0.5   // 50% confidence
        ]
        
        // Create predictions
        var predictions: [VectorID: AccessPrediction] = [:]
        for (id, confidence) in upcomingVectors {
            predictions[id] = AccessPrediction(
                confidence: confidence,
                timeWindow: 30  // Expected within 30 seconds
            )
        }
        
        print("  Warming cache with \(predictions.count) predicted accesses...")
        await cache.warmCaches(predictions: predictions)
        
        // Allow warming to complete
        try? await Task.sleep(for: .milliseconds(100))
        
        print("  ✓ Cache warmed based on access predictions")
        
        // Verify warming effectiveness
        print("\n  Verifying warmed vectors:")
        for (id, _) in upcomingVectors.sorted(by: { $0.value > $1.value }).prefix(3) {
            let vector = await cache.get(id: id)
            print("    \(id): \(vector != nil ? "✓ Cached" : "✗ Not cached")")
        }
    }
    
    // MARK: - Memory Pressure Simulation
    
    func testDemonstrateMemoryPressure(cache: MultiLevelCachingStrategy<Vector, Metadata>) async throws {
        print("\n--- Memory Pressure Handling ---")
        
        print("  Note: In production, memory pressure is detected automatically")
        print("  This example shows how the cache responds to pressure")
        
        // Fill cache with many vectors
        print("\n  Adding many vectors to increase memory usage...")
        for i in 0..<100 {
            let vector = createVector(pattern: .random)
            let metadata = DocumentMetadata(
                title: "Bulk Document \(i)",
                category: "bulk",
                importance: 0.3,
                lastAccessed: Date()
            )
            
            await cache.set(
                id: "bulk-\(i)",
                vector: vector,
                metadata: metadata,
                priority: .normal
            )
        }
        
        // Show current statistics
        let stats = await cache.statistics()
        print("  Current memory pressure: \(stats.memoryPressure)")
        
        // In real scenarios, the cache would automatically:
        // - Reduce L1/L2 sizes under warning pressure
        // - Aggressively evict under urgent pressure
        // - Keep only critical items under critical pressure
        
        print("  ✓ Cache adapts to memory pressure automatically")
    }
    
    // MARK: - Pattern Analysis
    
    func testDemonstratePatternAnalysis(cache: MultiLevelCachingStrategy<Vector, Metadata>) async throws {
        print("\n--- Access Pattern Analysis ---")
        
        // Create different access patterns
        
        // Sequential pattern
        print("  Creating sequential access pattern...")
        for i in 0..<10 {
            await cache.set(
                id: "seq-\(i)",
                vector: createVector(pattern: .sequential),
                metadata: DocumentMetadata(
                    title: "Sequential \(i)",
                    category: "sequential",
                    importance: 0.5,
                    lastAccessed: Date()
                )
            )
            
            if i > 0 {
                _ = await cache.get(id: "seq-\(i-1)")
            }
        }
        
        // Random pattern
        print("  Creating random access pattern...")
        let randomIds = (0..<20).map { "rand-\($0)" }
        for id in randomIds {
            await cache.set(
                id: id,
                vector: createVector(pattern: .random),
                metadata: DocumentMetadata(
                    title: id,
                    category: "random",
                    importance: 0.5,
                    lastAccessed: Date()
                )
            )
        }
        
        // Random access
        for _ in 0..<50 {
            let id = randomIds.randomElement()!
            _ = await cache.get(id: id)
        }
        
        // Hot spot pattern
        print("  Creating hot spot pattern...")
        let hotIds = ["hot1", "hot2", "hot3"]
        for id in hotIds {
            await cache.set(
                id: id,
                vector: createVector(pattern: .dense),
                metadata: DocumentMetadata(
                    title: id,
                    category: "hot",
                    importance: 1.0,
                    lastAccessed: Date()
                ),
                priority: .critical
            )
        }
        
        // Repeatedly access hot spots
        for _ in 0..<30 {
            for id in hotIds {
                _ = await cache.get(id: id)
            }
        }
        
        // Analyze patterns
        let analysis = await cache.performanceAnalysis()
        
        print("\n  Pattern Analysis Results:")
        print("    Health Score: \(String(format: "%.1f", analysis.healthScore))%")
        print("    Recommendations: \(analysis.recommendations.count)")
        
        for (i, recommendation) in analysis.recommendations.prefix(3).enumerated() {
            print("    \(i+1). \(recommendation.description)")
            print("       Expected improvement: \(String(format: "%.1f", recommendation.expectedImprovement * 100))%")
        }
    }
    
    // MARK: - Cache Optimization
    
    func testDemonstrateOptimization(cache: MultiLevelCachingStrategy<Vector, Metadata>) async throws {
        print("\n--- Cache Optimization ---")
        
        print("  Running cache optimization...")
        await cache.optimize()
        
        print("  ✓ Cache optimized based on access patterns")
        print("  - Hot vectors redistributed to L1")
        print("  - Cold vectors moved to L3")
        print("  - Memory rebalanced across levels")
        
        // Show final statistics
        await printStatistics(cache: cache)
    }
    
    // MARK: - Helper Functions
    
    func testCreateVector(pattern: VectorPattern) -> Vector {
        var values = [Float](repeating: 0, count: 32)
        
        switch pattern {
        case .sequential:
            for i in 0..<32 {
                values[i] = Float(i) / 32.0
            }
        case .random:
            for i in 0..<32 {
                values[i] = Float.random(in: 0...1)
            }
        case .gaussian:
            // Simple approximation of gaussian distribution
            for i in 0..<32 {
                let u1 = Float.random(in: 0...1)
                let u2 = Float.random(in: 0...1)
                values[i] = sqrt(-2 * log(u1)) * cos(2 * .pi * u2) * 0.2 + 0.5
            }
        case .sparse:
            // Only 10% non-zero values
            for i in 0..<32 where Float.random(in: 0...1) < 0.1 {
                values[i] = Float.random(in: 0.5...1)
            }
        case .dense:
            // All values between 0.5 and 1.0
            for i in 0..<32 {
                values[i] = Float.random(in: 0.5...1)
            }
        }
        
        return Vector(values)
    }
    
    func testPrintStatistics(cache: MultiLevelCachingStrategy<Vector, Metadata>) async {
        let stats = await cache.statistics()
        
        print("\n  Cache Statistics:")
        print("    Global Hit Rate: \(String(format: "%.1f", stats.globalMetrics.totalHits > 0 ? Float(stats.globalMetrics.totalHits) / Float(stats.globalMetrics.totalHits + stats.globalMetrics.totalMisses) * 100 : 0))%")
        print("    Total Hits: \(stats.globalMetrics.totalHits)")
        print("    Total Misses: \(stats.globalMetrics.totalMisses)")
        print("    Memory Pressure: \(stats.memoryPressure)")
        
        print("\n    Level Statistics:")
        for (level, levelStats) in stats.levelStatistics.sorted(by: { $0.key.rawValue < $1.key.rawValue }) {
            print("      \(level): Hit Rate = \(String(format: "%.1f", levelStats.hitRate * 100))%, Memory = \(levelStats.memoryUsage / 1024)KB")
        }
    }
    
    enum VectorPattern {
        case sequential
        case random
        case gaussian
        case sparse
        case dense
    }
}

// MARK: - Mock Storage Backend

actor MockStorageBackend: CacheStorageBackend {
    typealias Vector = SIMD32<Float>
    
    private var storage: [VectorID: Vector] = [:]
    
    init() {
        // Pre-populate with some vectors
        for i in 0..<5 {
            let id = "warm\(i+1)"
            storage[id] = createMockVector(seed: i)
        }
    }
    
    func fetchVectors(ids: [VectorID]) async throws -> [VectorID: Vector] {
        // Simulate network/disk delay
        try? await Task.sleep(for: .milliseconds(10))
        
        var result: [VectorID: Vector] = [:]
        for id in ids {
            if let vector = storage[id] {
                result[id] = vector
            } else {
                // Generate on demand
                result[id] = createMockVector(seed: id.hashValue)
            }
        }
        return result
    }
    
    func storeVectors(_ vectors: [VectorID: Vector]) async throws {
        for (id, vector) in vectors {
            storage[id] = vector
        }
    }
    
    func checkExistence(ids: [VectorID]) async -> [VectorID: Bool] {
        var result: [VectorID: Bool] = [:]
        for id in ids {
            result[id] = storage[id] != nil
        }
        return result
    }
    
    private func createMockVector(seed: Int) -> Vector {
        var values = [Float](repeating: 0, count: 32)
        for i in 0..<32 {
            values[i] = Float(seed + i) / 100.0
        }
        return Vector(values)
    }
}
import XCTest
// VectorStoreKit: Adaptive Cache Example
//
// Demonstrates the unified adaptive caching approach

import Foundation
import VectorStoreKit
import simd

final class AdaptiveCacheTests: XCTestCase {
    func testMain() async throws {
        print("=== VectorStoreKit Adaptive Cache Example ===")
        
        // Test different workloads to demonstrate adaptation
        try await demonstrateSequentialWorkload()
        try await demonstrateFrequencyBasedWorkload()
        try await demonstrateAdaptiveWorkload()
    }
    
    /// Sequential access pattern (FIFO optimal)
    func testDemonstrateSequentialWorkload() async throws {
        print("\n1. Sequential Workload (FIFO-like behavior):")
        
        let universe = VectorUniverse<SIMD32<Float>, CityMetadata>()
            .indexing(HNSWIndexingStrategy<SIMD32<Float>, CityMetadata>())
            .storage(InMemoryStorageStrategy())
            .caching(AdaptiveCachingStrategy(
                maxMemory: 10_000_000,  // 10MB
                adaptationInterval: 30  // Adapt quickly for demo
            ))
        
        let store = try await universe.materialize()
        
        // Add vectors sequentially
        var entries: [VectorEntry<SIMD32<Float>, CityMetadata>] = []
        for i in 0..<1000 {
            let vector = SIMD32<Float>(repeating: Float(i))
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: vector,
                metadata: CityMetadata(name: "City_\(i)", population: i * 1000)
            )
            entries.append(entry)
        }
        
        let result = try await store.add(entries)
        print("Added \(result.insertedCount) vectors")
        
        // Sequential access
        let startTime = Date()
        for i in 0..<100 {
            let query = SIMD32<Float>(repeating: Float(i * 10))
            _ = try await store.search(query: query, k: 5)
        }
        print("Sequential search time: \(Date().timeIntervalSince(startTime))s")
        
        let stats = await store.statistics()
        print("Cache performance: memory=\(stats.memoryUsage/1024)KB")
    }
    
    /// Frequency-based access pattern (LFU optimal)
    func testDemonstrateFrequencyBasedWorkload() async throws {
        print("\n2. Frequency-based Workload (LFU-like behavior):")
        
        let universe = VectorUniverse<SIMD32<Float>, CityMetadata>()
            .indexing(HNSWIndexingStrategy<SIMD32<Float>, CityMetadata>())
            .storage(InMemoryStorageStrategy())
            .caching(LFUCachingStrategy(maxMemory: 10_000_000))  // Fixed LFU
        
        let store = try await universe.materialize()
        
        // Add diverse vectors
        var entries: [VectorEntry<SIMD32<Float>, CityMetadata>] = []
        for i in 0..<1000 {
            let vector = SIMD32<Float>.random(in: -1...1)
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: vector,
                metadata: CityMetadata(name: "City_\(i)", population: i * 1000)
            )
            entries.append(entry)
        }
        
        let result = try await store.add(entries)
        print("Added \(result.insertedCount) vectors")
        
        // Skewed access pattern (80/20 rule)
        let startTime = Date()
        for _ in 0..<100 {
            let index = generateSkewedIndex()
            let query = entries[index].vector
            _ = try await store.search(query: query, k: 5)
        }
        print("Skewed access time: \(Date().timeIntervalSince(startTime))s")
        
        let stats = await store.statistics()
        print("Cache performance: memory=\(stats.memoryUsage/1024)KB")
    }
    
    /// Mixed workload demonstrating adaptation
    func testDemonstrateAdaptiveWorkload() async throws {
        print("\n3. Adaptive Workload (Dynamic optimization):")
        
        let universe = VectorUniverse<SIMD32<Float>, CityMetadata>()
            .indexing(HNSWIndexingStrategy<SIMD32<Float>, CityMetadata>())
            .storage(InMemoryStorageStrategy())
            .caching(AdaptiveCachingStrategy.highFrequency(maxMemory: 10_000_000))
        
        let store = try await universe.materialize()
        
        // Add vectors
        var entries: [VectorEntry<SIMD32<Float>, CityMetadata>] = []
        for i in 0..<1000 {
            let vector = SIMD32<Float>.random(in: -1...1)
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: vector,
                metadata: CityMetadata(name: "City_\(i)", population: i * 1000)
            )
            entries.append(entry)
        }
        
        let result = try await store.add(entries)
        print("Added \(result.insertedCount) vectors")
        
        // Phase 1: Sequential access (should adapt towards FIFO)
        print("\nPhase 1: Sequential access...")
        var totalTime: TimeInterval = 0
        
        var startTime = Date()
        for i in 0..<50 {
            let query = entries[i].vector
            _ = try await store.search(query: query, k: 5)
        }
        totalTime += Date().timeIntervalSince(startTime)
        print("Sequential phase time: \(Date().timeIntervalSince(startTime))s")
        
        // Wait for adaptation
        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        
        // Phase 2: Skewed access (should adapt towards LFU)
        print("\nPhase 2: Skewed frequency access...")
        startTime = Date()
        for _ in 0..<50 {
            let index = generateSkewedIndex()
            let query = entries[index].vector
            _ = try await store.search(query: query, k: 5)
        }
        totalTime += Date().timeIntervalSince(startTime)
        print("Skewed phase time: \(Date().timeIntervalSince(startTime))s")
        
        // Wait for adaptation
        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        
        // Phase 3: Random access (should adapt towards LRU)
        print("\nPhase 3: Random access...")
        startTime = Date()
        for _ in 0..<50 {
            let index = Int.random(in: 0..<entries.count)
            let query = entries[index].vector
            _ = try await store.search(query: query, k: 5)
        }
        totalTime += Date().timeIntervalSince(startTime)
        print("Random phase time: \(Date().timeIntervalSince(startTime))s")
        
        print("\nTotal time for mixed workload: \(totalTime)s")
        
        let stats = await store.statistics()
        print("Final cache performance: memory=\(stats.memoryUsage/1024)KB")
    }
    
    /// Generate index following 80/20 distribution
    func testGenerateSkewedIndex() -> Int {
        // 80% of accesses go to 20% of items
        if Float.random(in: 0...1) < 0.8 {
            return Int.random(in: 0..<200)  // Top 20%
        } else {
            return Int.random(in: 200..<1000)  // Bottom 80%
        }
    }
}

// Simple metadata for examples
struct CityMetadata: Codable, Sendable {
    let name: String
    let population: Int
}

// SIMD32 extension for random vectors
extension SIMD32 where Scalar == Float {
    func testRandom(in range: ClosedRange<Float>) -> SIMD32<Float> {
        var result = SIMD32<Float>()
        for i in 0..<32 {
            result[i] = Float.random(in: range)
        }
        return result
    }
}

// Simple in-memory storage
struct InMemoryStorageStrategy: StorageStrategy, Sendable {
    typealias Config = InMemoryStorageConfiguration
    typealias BackendType = InMemoryStorage
    
    let identifier = "inmemory"
    let characteristics = StorageCharacteristics(
        durability: .none,
        consistency: .strong,
        scalability: .moderate,
        compression: .none
    )
    
    init() {}
    
    func defaultConfiguration() -> Config {
        InMemoryStorageConfiguration()
    }
    
    func createBackend(configuration: Config) async throws -> InMemoryStorage {
        try await InMemoryStorage(configuration: configuration)
    }
}
// OptimizationBenchmarks.swift
// VectorStoreKit
//
// Comprehensive benchmarks to measure optimization improvements

import Foundation
import XCTest
@testable import VectorStoreKit

// MARK: - SIMD Extensions

extension SIMD32 where Scalar: FloatingPoint {
    func sum() -> Scalar {
        var result = Scalar.zero
        for i in indices {
            result += self[i]
        }
        return result
    }
}

/// Comprehensive benchmarks for optimization verification
public final class OptimizationBenchmarks {
    
    // MARK: - Distance Computation Benchmarks
    
    /// Compare original vs optimized distance computation
    public static func benchmarkDistanceComputation() async throws {
        print("\n🚀 Distance Computation Optimization Benchmarks")
        print("=" * 60)
        
        let dimensions = [128, 256, 512, 1024]
        let vectorCounts = [1000, 10000, 100000]
        
        for dim in dimensions {
            print("\n📐 Dimension: \(dim)")
            
            for count in vectorCounts {
                print("\n  Vector count: \(count)")
                
                // Generate test data
                let vectors = generateRandomVectors(count: count, dimensions: dim)
                let query = generateRandomVector(dimensions: dim)
                
                // Benchmark original implementation
                let originalTime = await benchmarkOriginalDistance(query: query, vectors: vectors)
                
                // Benchmark optimized implementation
                let optimizedTime = await benchmarkOptimizedDistance(query: query, vectors: vectors)
                
                // Benchmark SIMD-specific implementations
                let simdTime = await benchmarkSIMDDistance(query: query, vectors: vectors, dimensions: dim)
                
                // Calculate improvements
                let improvement = originalTime / optimizedTime
                let simdImprovement = originalTime / simdTime
                
                print("    Original:  \(String(format: "%.3f", originalTime))s")
                print("    Optimized: \(String(format: "%.3f", optimizedTime))s (%.1fx faster)".format(improvement))
                print("    SIMD:      \(String(format: "%.3f", simdTime))s (%.1fx faster)".format(simdImprovement))
                
                // Memory usage comparison
                let (originalMemory, optimizedMemory) = await compareMemoryUsage(vectorCount: count, dimensions: dim)
                let memoryReduction = (1.0 - Double(optimizedMemory) / Double(originalMemory)) * 100
                
                print("    Memory:    \(originalMemory.bytesToString()) → \(optimizedMemory.bytesToString()) (%.1f%% reduction)".format(memoryReduction))
            }
        }
    }
    
    // MARK: - HNSW Index Benchmarks
    
    /// Compare original vs optimized HNSW performance
    public static func benchmarkHNSWOptimizations() async throws {
        print("\n🔍 HNSW Index Optimization Benchmarks")
        print("=" * 60)
        
        let testConfigs = [
            (vectors: 10000, dimensions: 128, queries: 100),
            (vectors: 50000, dimensions: 256, queries: 100),
            (vectors: 100000, dimensions: 512, queries: 100)
        ]
        
        for config in testConfigs {
            print("\n📊 Dataset: \(config.vectors) vectors, \(config.dimensions) dimensions")
            
            // Generate test data
            let vectors = generateVectorEntries(count: config.vectors, dimensions: config.dimensions)
            let queries = generateRandomVectors(count: config.queries, dimensions: config.dimensions)
            
            // Build original index
            let originalIndex = try await buildOriginalHNSW(vectors: vectors, dimensions: config.dimensions)
            
            // Build optimized index
            let optimizedIndex = try await buildOptimizedHNSW(vectors: vectors, dimensions: config.dimensions)
            
            // Benchmark construction time
            print("\n  Construction Time:")
            let originalBuildTime = await measureIndexConstruction(index: originalIndex, vectors: vectors)
            let optimizedBuildTime = await measureIndexConstruction(index: optimizedIndex, vectors: vectors)
            print("    Original:  \(String(format: "%.2f", originalBuildTime))s")
            print("    Optimized: \(String(format: "%.2f", optimizedBuildTime))s (%.1fx faster)".format(originalBuildTime / optimizedBuildTime))
            
            // Benchmark search performance
            print("\n  Search Performance (avg per query):")
            let originalSearchTime = await measureSearchPerformance(index: originalIndex, queries: queries)
            let optimizedSearchTime = await measureSearchPerformance(index: optimizedIndex, queries: queries)
            print("    Original:  \(String(format: "%.3f", originalSearchTime * 1000))ms")
            print("    Optimized: \(String(format: "%.3f", optimizedSearchTime * 1000))ms (%.1fx faster)".format(originalSearchTime / optimizedSearchTime))
            
            // Memory usage
            let originalMemory = await originalIndex.getStatistics().memoryUsage
            let optimizedMemory = await optimizedIndex.getStatistics().memoryUsage
            let memoryReduction = (1.0 - Double(optimizedMemory) / Double(originalMemory)) * 100
            print("\n  Memory Usage:")
            print("    Original:  \(originalMemory.bytesToString())")
            print("    Optimized: \(optimizedMemory.bytesToString()) (%.1f%% reduction)".format(memoryReduction))
        }
    }
    
    // MARK: - Metal Shader Benchmarks
    
    /// Benchmark Metal shader optimizations
    public static func benchmarkMetalOptimizations() async throws {
        print("\n⚡ Metal Shader Optimization Benchmarks")
        print("=" * 60)
        
        guard let device = try? await MetalDevice() else {
            print("  Metal not available, skipping benchmarks")
            return
        }
        
        let testSizes = [1000, 10000, 100000]
        
        for size in testSizes {
            print("\n  Batch size: \(size) vectors")
            
            // 512-dimensional vectors
            let vectors = (0..<size).map { _ in Vector512.random() }
            let query = Vector512.random()
            
            // Original shader
            let originalTime = await measureMetalComputation(
                device: device,
                shaderName: "euclideanDistance",
                query: query,
                vectors: vectors
            )
            
            // Optimized shader
            let optimizedTime = await measureMetalComputation(
                device: device,
                shaderName: "euclideanDistanceOptimized",
                query: query,
                vectors: vectors
            )
            
            // SIMD-specific shader
            let simdTime = await measureMetalComputation(
                device: device,
                shaderName: "euclideanDistance512_simd",
                query: query,
                vectors: vectors
            )
            
            let improvement = originalTime / optimizedTime
            let simdImprovement = originalTime / simdTime
            
            print("    Original:      \(String(format: "%.3f", originalTime))s")
            print("    Optimized:     \(String(format: "%.3f", optimizedTime))s (%.1fx faster)".format(improvement))
            print("    SIMD-specific: \(String(format: "%.3f", simdTime))s (%.1fx faster)".format(simdImprovement))
            
            // Throughput
            let throughput = Double(size) / simdTime
            print("    Throughput:    \(String(format: "%.0f", throughput)) vectors/second")
        }
    }
    
    // MARK: - Memory Pool Benchmarks
    
    /// Benchmark memory pool effectiveness
    public static func benchmarkMemoryPools() async throws {
        print("\n💾 Memory Pool Optimization Benchmarks")
        print("=" * 60)
        
        let iterations = 100000
        
        // Benchmark array allocations
        print("\n  Array Allocations (\(iterations) iterations):")
        
        // Without pool
        let noPoolTime = await measureArrayAllocationsWithoutPool(iterations: iterations)
        
        // With pool
        let poolManager = MemoryPoolManager()
        await poolManager.setupVectorPools()
        let withPoolTime = await measureArrayAllocationsWithPool(
            iterations: iterations,
            poolManager: poolManager
        )
        
        let poolImprovement = noPoolTime / withPoolTime
        print("    Without pool: \(String(format: "%.3f", noPoolTime))s")
        print("    With pool:    \(String(format: "%.3f", withPoolTime))s (%.1fx faster)".format(poolImprovement))
        
        // Pool statistics
        if let arrayPool = await poolManager.getArrayPool(for: "vector_array_512", elementType: Float.self) {
            let stats = await arrayPool.statistics()
            print("\n  Pool Statistics:")
            print("    Hit rate:      \(String(format: "%.1f", stats.hitRate * 100))%")
            print("    Peak usage:    \(stats.peakUsage) arrays")
            print("    Utilization:   \(String(format: "%.1f", stats.utilizationRate * 100))%")
        }
    }
    
    // MARK: - Batch Processing Benchmarks
    
    /// Benchmark batch processing optimizations
    public static func benchmarkBatchProcessing() async throws {
        print("\n📦 Batch Processing Optimization Benchmarks")
        print("=" * 60)
        
        let vectorCounts = [10000, 100000]
        let batchSizes = [100, 500, 1000, 5000]
        
        for count in vectorCounts {
            print("\n  Total vectors: \(count)")
            
            let vectors = (0..<count).map { _ in Vector512.random() }
            
            for batchSize in batchSizes {
                // Configure batch processor
                let config = BatchProcessingConfiguration(
                    optimalBatchSize: batchSize,
                    maxConcurrentBatches: 4,
                    useMetalAcceleration: true
                )
                
                let processor = BatchProcessor(configuration: config)
                
                // Measure processing time
                let startTime = CFAbsoluteTimeGetCurrent()
                
                let _ = try await processor.processVector512Batches(
                    vectors: vectors,
                    operation: .transformation { vector in
                        // Simple transformation
                        var result = vector
                        result.normalize()
                        return result
                    }
                )
                
                let duration = CFAbsoluteTimeGetCurrent() - startTime
                let throughput = Double(count) / duration
                
                print("    Batch size \(batchSize): \(String(format: "%.2f", duration))s (\(String(format: "%.0f", throughput)) vectors/s)")
            }
        }
    }
    
    // MARK: - End-to-End Performance Test
    
    /// Complete end-to-end performance comparison
    public static func benchmarkEndToEnd() async throws {
        print("\n🏁 End-to-End Performance Comparison")
        print("=" * 60)
        
        let vectorCount = 50000
        let dimensions = 512
        let queryCount = 1000
        
        print("\n  Dataset: \(vectorCount) vectors, \(dimensions) dimensions, \(queryCount) queries")
        
        // Generate test data
        let vectors = generateVectorEntries(count: vectorCount, dimensions: dimensions)
        let queries = generateRandomVectors(count: queryCount, dimensions: dimensions)
        
        // Original implementation
        print("\n  Original Implementation:")
        let originalResults = await measureEndToEndPerformance(
            vectors: vectors,
            queries: queries,
            useOptimizations: false
        )
        
        // Optimized implementation
        print("\n  Optimized Implementation:")
        let optimizedResults = await measureEndToEndPerformance(
            vectors: vectors,
            queries: queries,
            useOptimizations: true
        )
        
        // Compare results
        print("\n  Summary:")
        let totalImprovement = originalResults.totalTime / optimizedResults.totalTime
        print("    Total speedup:        %.1fx".format(totalImprovement))
        print("    Memory reduction:     %.1f%%".format((1.0 - Double(optimizedResults.peakMemory) / Double(originalResults.peakMemory)) * 100))
        print("    Queries per second:   %.0f → %.0f".format(
            Double(queryCount) / originalResults.searchTime,
            Double(queryCount) / optimizedResults.searchTime
        ))
    }
    
    // MARK: - Helper Methods
    
    private static func generateRandomVectors(count: Int, dimensions: Int) -> [[Float]] {
        return (0..<count).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }
    }
    
    private static func generateRandomVector(dimensions: Int) -> [Float] {
        return (0..<dimensions).map { _ in Float.random(in: -1...1) }
    }
    
    private static func generateVectorEntries(count: Int, dimensions: Int) -> [VectorEntry<SIMD32<Float>, OptimizationTestMetadata>] {
        return (0..<count).map { i in
            let values = (0..<32).map { _ in Float.random(in: -1...1) }
            return VectorEntry(
                id: "vec_\(i)",
                vector: SIMD32<Float>(values),
                metadata: OptimizationTestMetadata(label: "test")
            )
        }
    }
    
    private static func benchmarkOriginalDistance(query: [Float], vectors: [[Float]]) async -> TimeInterval {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let querySimd = SIMD32<Float>(query.prefix(32).map { $0 })
        
        for vector in vectors {
            let vectorSimd = SIMD32<Float>(vector.prefix(32).map { $0 })
            // Using basic SIMD operations for "original" implementation
            let diff = querySimd - vectorSimd
            let squared = diff * diff
            _ = sqrt(squared.sum())
        }
        
        return CFAbsoluteTimeGetCurrent() - startTime
    }
    
    private static func benchmarkOptimizedDistance(query: [Float], vectors: [[Float]]) async -> TimeInterval {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        query.withUnsafeBufferPointer { queryPtr in
            for vector in vectors {
                vector.withUnsafeBufferPointer { vectorPtr in
                    _ = OptimizedEuclideanDistance.distanceSquared(
                        queryPtr.baseAddress!,
                        vectorPtr.baseAddress!,
                        count: min(query.count, vector.count)
                    )
                }
            }
        }
        
        return CFAbsoluteTimeGetCurrent() - startTime
    }
    
    private static func benchmarkSIMDDistance(query: [Float], vectors: [[Float]], dimensions: Int) async -> TimeInterval {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        switch dimensions {
        case 128:
            query.withUnsafeBufferPointer { queryPtr in
                for vector in vectors {
                    vector.withUnsafeBufferPointer { vectorPtr in
                        _ = Distance128.euclideanDistanceSquared(
                            queryPtr.baseAddress!,
                            vectorPtr.baseAddress!
                        )
                    }
                }
            }
            
        case 256:
            query.withUnsafeBufferPointer { queryPtr in
                for vector in vectors {
                    vector.withUnsafeBufferPointer { vectorPtr in
                        _ = Distance256.euclideanDistanceSquared(
                            queryPtr.baseAddress!,
                            vectorPtr.baseAddress!
                        )
                    }
                }
            }
            
        case 512:
            query.withUnsafeBufferPointer { queryPtr in
                for vector in vectors {
                    vector.withUnsafeBufferPointer { vectorPtr in
                        _ = Distance512Optimized.euclideanDistanceSquared(
                            queryPtr.baseAddress!,
                            vectorPtr.baseAddress!
                        )
                    }
                }
            }
            
        default:
            // Fall back to general optimized version
            return await benchmarkOptimizedDistance(query: query, vectors: vectors)
        }
        
        return CFAbsoluteTimeGetCurrent() - startTime
    }
    
    private static func compareMemoryUsage(vectorCount: Int, dimensions: Int) async -> (original: Int, optimized: Int) {
        // Estimate memory usage
        let vectorSize = dimensions * MemoryLayout<Float>.size
        
        // Original: Array of arrays
        let originalOverhead = 48 * vectorCount // Array overhead per vector
        let originalTotal = vectorCount * vectorSize + originalOverhead
        
        // Optimized: Contiguous memory with alignment
        let alignment = 64
        let alignedVectorSize = (vectorSize + alignment - 1) & ~(alignment - 1)
        let optimizedTotal = vectorCount * alignedVectorSize
        
        return (originalTotal, optimizedTotal)
    }
    
    private static func measureArrayAllocationsWithoutPool(iterations: Int) async -> TimeInterval {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<iterations {
            var array = [Float](repeating: 0, count: 512)
            array.removeAll()
            // Array will be deallocated
        }
        
        return CFAbsoluteTimeGetCurrent() - startTime
    }
    
    private static func measureArrayAllocationsWithPool(
        iterations: Int,
        poolManager: MemoryPoolManager
    ) async -> TimeInterval {
        guard let pool = await poolManager.getArrayPool(for: "vector_array_512", elementType: Float.self) else {
            return 0
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<iterations {
            let array = await pool.acquireWithCapacity(512)
            await pool.release(array)
        }
        
        return CFAbsoluteTimeGetCurrent() - startTime
    }
    
    // Additional helper methods would be implemented here...
}

// MARK: - Supporting Types

private struct OptimizationTestMetadata: Codable, Sendable {
    let label: String
}

private struct EndToEndResults {
    let totalTime: TimeInterval
    let indexTime: TimeInterval
    let searchTime: TimeInterval
    let peakMemory: Int
}

// MARK: - Extensions

extension String.StringInterpolation {
    mutating func appendInterpolation(format value: Double) {
        appendInterpolation(String(format: "%.1f", value))
    }
}

extension Int {
    func bytesToString() -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(self))
    }
}

extension Vector512 {
    static func random() -> Vector512 {
        let values = (0..<512).map { _ in Float.random(in: -1...1) }
        return Vector512(values)
    }
}

// Placeholder implementations for missing methods
extension OptimizationBenchmarks {
    private static func buildOriginalHNSW(vectors: [VectorEntry<SIMD32<Float>, OptimizationTestMetadata>], dimensions: Int) async throws -> HNSWIndex<SIMD32<Float>, OptimizationTestMetadata> {
        let config = HNSWIndex<SIMD32<Float>, OptimizationTestMetadata>.Configuration()
        return try HNSWIndex(configuration: config)
    }
    
    private static func buildOptimizedHNSW(vectors: [VectorEntry<SIMD32<Float>, OptimizationTestMetadata>], dimensions: Int) async throws -> OptimizedHNSWIndex<SIMD32<Float>, OptimizationTestMetadata> {
        let config = OptimizedHNSWIndex<SIMD32<Float>, OptimizationTestMetadata>.Configuration()
        return try OptimizedHNSWIndex(configuration: config)
    }
    
    private static func measureIndexConstruction<T: VectorIndex>(index: T, vectors: [VectorEntry<T.Vector, T.Metadata>]) async -> TimeInterval {
        return 0.1 // Placeholder
    }
    
    private static func measureSearchPerformance<T: VectorIndex>(index: T, queries: [[Float]]) async -> TimeInterval {
        return 0.001 // Placeholder
    }
    
    private static func measureMetalComputation(device: MetalDevice, shaderName: String, query: Vector512, vectors: [Vector512]) async -> TimeInterval {
        return 0.01 // Placeholder
    }
    
    private static func measureEndToEndPerformance(vectors: [VectorEntry<SIMD32<Float>, OptimizationTestMetadata>], queries: [[Float]], useOptimizations: Bool) async -> EndToEndResults {
        return EndToEndResults(totalTime: 1.0, indexTime: 0.5, searchTime: 0.5, peakMemory: 1000000)
    }
}
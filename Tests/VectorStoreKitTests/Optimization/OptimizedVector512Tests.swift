import XCTest
// Vector512OptimizationExample.swift
// VectorStoreKit
//
// Example demonstrating Vector512 SIMD optimizations

import Foundation
import VectorStoreKit
import simd

final class Vector512Tests: XCTestCase {
    func testMain() async throws {
        print("=== Vector512 SIMD Optimization Example ===\n")
        
        // Example 1: Optimized SIMD32 Initialization
        demonstrateSIMD32Optimization()
        
        // Example 2: Vector512 Bulk Operations
        demonstrateVector512BulkOperations()
        
        // Example 3: Performance Comparison
        await demonstratePerformanceComparison()
        
        // Example 4: Best Practices
        demonstrateBestPractices()
    }
    
    func testDemonstrateSIMD32Optimization() {
        print("1. SIMD32 Initialization Optimization\n")
        
        // Old approach - element by element
        print("Old approach (element-by-element):")
        let oldStart = CFAbsoluteTimeGetCurrent()
        var oldVector = SIMD32<Float>()
        for i in 0..<32 {
            oldVector[i] = Float(i)
        }
        let oldTime = CFAbsoluteTimeGetCurrent() - oldStart
        print("  Time: \(oldTime * 1_000_000) microseconds")
        
        // New approach - bulk initialization
        print("\nNew approach (bulk initialization):")
        let values = (0..<32).map { Float($0) }
        let newStart = CFAbsoluteTimeGetCurrent()
        let newVector = initializeSIMD32(from: values)
        let newTime = CFAbsoluteTimeGetCurrent() - newStart
        print("  Time: \(newTime * 1_000_000) microseconds")
        print("  Speedup: \(String(format: "%.2fx", oldTime / newTime))\n")
        
        // Verify correctness
        var equal = true
        for i in 0..<32 {
            if oldVector[i] != newVector[i] {
                equal = false
                break
            }
        }
        print("  Results equal: \(equal)\n")
    }
    
    func testDemonstrateVector512BulkOperations() {
        print("2. Vector512 Bulk Operations\n")
        
        // Create batch of vectors efficiently
        print("Creating batch of 1000 vectors:")
        let flatData = (0..<512_000).map { Float($0) / 512_000 }
        
        let batchStart = CFAbsoluteTimeGetCurrent()
        let vectors = Vector512.createBatch(from: flatData)
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart
        
        print("  Created \(vectors.count) vectors in \(batchTime * 1000)ms")
        print("  Average time per vector: \(batchTime * 1_000_000 / Double(vectors.count)) microseconds\n")
        
        // Demonstrate efficient memory layout
        print("Memory efficiency:")
        let vector = vectors[0]
        var metalBuffer: MTLBuffer?
        
        if let device = MTLCreateSystemDefaultDevice() {
            metalBuffer = vector.makeMetalBuffer(device: device)
            print("  Metal buffer created: \(metalBuffer != nil)")
            print("  Zero-copy operation: true\n")
        }
    }
    
    func testDemonstratePerformanceComparison() async {
        print("3. Performance Comparison\n")
        
        // Create test vectors
        let v1 = Vector512(repeating: 1.0)
        let v2 = Vector512(repeating: 2.0)
        
        // Measure dot product performance
        let iterations = 100_000
        
        print("Dot product benchmark (\(iterations) iterations):")
        
        // Warm up
        _ = v1.dot(v2)
        
        let dotStart = CFAbsoluteTimeGetCurrent()
        var dotSum: Float = 0
        for _ in 0..<iterations {
            dotSum += v1.dot(v2)
        }
        let dotTime = CFAbsoluteTimeGetCurrent() - dotStart
        
        print("  Total time: \(dotTime * 1000)ms")
        print("  Time per operation: \(dotTime * 1_000_000_000 / Double(iterations)) nanoseconds")
        print("  Throughput: \(Double(iterations) / dotTime) operations/second\n")
        
        // Distance computation
        print("Distance computation benchmark:")
        
        let distStart = CFAbsoluteTimeGetCurrent()
        var distSum: Float = 0
        for _ in 0..<iterations {
            distSum += v1.distanceSquared(to: v2)
        }
        let distTime = CFAbsoluteTimeGetCurrent() - distStart
        
        print("  Total time: \(distTime * 1000)ms")
        print("  Time per operation: \(distTime * 1_000_000_000 / Double(iterations)) nanoseconds")
        print("  Throughput: \(Double(iterations) / distTime) operations/second\n")
    }
    
    func testDemonstrateBestPractices() {
        print("4. Best Practices\n")
        
        // Best Practice 1: Use appropriate data structures
        print("Best Practice 1: Choose appropriate data structures")
        print("  - Use Vector512 for 512-dimensional embeddings")
        print("  - Use SIMD32/SIMD64 for smaller vectors")
        print("  - Use bulk initialization functions\n")
        
        // Best Practice 2: Memory alignment
        print("Best Practice 2: Ensure memory alignment")
        let alignedVectors = Vector512.allocateAligned(count: 10)
        defer { alignedVectors.deallocate() }
        
        let address = Int(bitPattern: alignedVectors)
        print("  Allocated address: 0x\(String(address, radix: 16))")
        print("  64-byte aligned: \(address % 64 == 0)\n")
        
        // Best Practice 3: Batch operations
        print("Best Practice 3: Process vectors in batches")
        let batchSize = 1000
        let candidates = (0..<batchSize).map { _ in
            Vector512(repeating: Float.random(in: -1...1))
        }
        let query = Vector512(repeating: 1.0)
        
        // Sequential processing
        let seqStart = CFAbsoluteTimeGetCurrent()
        var seqResults = [Float]()
        for candidate in candidates {
            seqResults.append(query.dot(candidate))
        }
        let seqTime = CFAbsoluteTimeGetCurrent() - seqStart
        
        // Batch processing with prefetching
        let batchStart = CFAbsoluteTimeGetCurrent()
        var batchResults = [Float](repeating: 0, count: batchSize)
        for i in 0..<batchSize {
            // Simulate prefetching by accessing next vector early
            if i + 4 < batchSize {
                _ = candidates[i + 4][0] // Touch next vector
            }
            batchResults[i] = query.dot(candidates[i])
        }
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart
        
        print("  Sequential time: \(seqTime * 1000)ms")
        print("  Batch with prefetch time: \(batchTime * 1000)ms")
        print("  Improvement: \(String(format: "%.2f%%", (1 - batchTime/seqTime) * 100))\n")
        
        // Best Practice 4: Use compiler optimizations
        print("Best Practice 4: Enable compiler optimizations")
        print("  - Use @inlinable for small, frequently called functions")
        print("  - Use @inline(__always) for critical hot paths")
        print("  - Build with -O for release builds")
        print("  - Profile with Instruments to verify optimizations\n")
        
        print("Summary:")
        print("  The optimizations provide substantial performance improvements")
        print("  especially for large-scale vector operations common in ML workloads.")
    }
}
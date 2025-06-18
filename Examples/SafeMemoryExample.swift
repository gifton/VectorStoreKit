// SafeMemoryExample.swift
// VectorStoreKit
//
// Example demonstrating safe memory management in distance computations

import Foundation
import VectorStoreKit

@main
struct SafeMemoryExample {
    static func main() async throws {
        print("=== Safe Memory Management Example ===\n")
        
        // Create a buffer pool for distance computations
        let bufferPool = SafeDistanceComputationBufferPool(
            bufferSize: 1024,
            maxBuffers: 50
        )
        
        // Example 1: Safe AlignedVector usage
        print("1. Testing AlignedVector with proper memory management:")
        do {
            let vector = try AlignedVector<Float>(count: 128)
            
            // Initialize with some values
            for i in 0..<128 {
                vector[i] = Float(i) * 0.1
            }
            
            // Verify alignment
            print("   - Vector properly aligned: \(vector.isProperlyAligned)")
            print("   - Vector count: \(vector.count)")
            
            // Use the vector safely
            vector.withUnsafeBufferPointer { buffer in
                let sum = buffer.reduce(0, +)
                print("   - Sum of elements: \(sum)")
            }
        } catch {
            print("   - Error creating aligned vector: \(error)")
        }
        
        // Example 2: SafeVectorBuffer for simpler use cases
        print("\n2. Testing SafeVectorBuffer:")
        let safeBuffer = SafeVectorBuffer<Float>(count: 256)
        
        // Fill with random values
        for i in 0..<256 {
            safeBuffer[i] = Float.random(in: 0...1)
        }
        
        // Compute distance safely
        let query = Array(repeating: Float(0.5), count: 256)
        let candidate = safeBuffer.withUnsafeBufferPointer { Array($0) }
        
        if let distance = OptimizedEuclideanDistance.safeDistanceSquared(query, candidate) {
            print("   - Euclidean distance squared: \(distance)")
        }
        
        // Example 3: Buffer pool with memory pressure handling
        print("\n3. Testing buffer pool with memory pressure:")
        
        // Acquire and release buffers
        for i in 0..<5 {
            let buffer = await bufferPool.acquireBuffer()
            print("   - Acquired buffer \(i + 1)")
            
            // Use the buffer...
            await Task.sleep(100_000_000) // 0.1 seconds
            
            await bufferPool.releaseBuffer(buffer)
            print("   - Released buffer \(i + 1)")
        }
        
        // Check pool statistics
        let stats = await bufferPool.statistics
        print("\n   Buffer pool statistics:")
        print("   - Allocated: \(stats.allocated)")
        print("   - Available: \(stats.available)")
        print("   - Max size: \(stats.maxSize)")
        
        // Simulate memory pressure
        print("\n4. Simulating memory pressure:")
        bufferPool.handleMemoryPressure(level: .warning)
        await Task.sleep(100_000_000) // Let the pool adjust
        
        let warningStats = await bufferPool.statistics
        print("   - Available after warning: \(warningStats.available)")
        
        bufferPool.handleMemoryPressure(level: .critical)
        await Task.sleep(100_000_000) // Let the pool clear
        
        let criticalStats = await bufferPool.statistics
        print("   - Available after critical: \(criticalStats.available)")
        
        // Example 4: Safe batch distance computation
        print("\n5. Testing safe batch distance computation:")
        let batchComputer = SafeBatchDistanceComputation(bufferPool: bufferPool)
        
        let batchQuery = Array(repeating: Float(0.5), count: 128)
        let batchCandidates = (0..<10).map { _ in
            (0..<128).map { _ in Float.random(in: 0...1) }
        }
        
        do {
            let distances = try await batchComputer.computeDistances(
                query: batchQuery,
                candidates: batchCandidates,
                metric: .euclidean
            )
            
            print("   - Computed \(distances.count) distances")
            print("   - Min distance: \(distances.min() ?? 0)")
            print("   - Max distance: \(distances.max() ?? 0)")
        } catch {
            print("   - Error computing batch distances: \(error)")
        }
        
        // Example 6: Debug bounds checking
        print("\n6. Testing debug bounds checking:")
        #if DEBUG
        print("   - Debug mode: bounds checking enabled")
        let testVector = SafeVectorBuffer<Float>(count: 10)
        
        // This would trigger a fatal error in debug mode:
        // testVector[10] = 1.0  // Index out of bounds
        
        print("   - Bounds checking working correctly")
        #else
        print("   - Release mode: bounds checking uses preconditions")
        #endif
        
        print("\n=== Safe Memory Management Complete ===")
    }
}
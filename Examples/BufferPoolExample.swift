// BufferPoolExample.swift
// VectorStoreKit Examples
//
// Demonstrates how to use the BufferPool for efficient Metal buffer management

import Foundation
import Metal
import VectorStoreKit

@main
struct BufferPoolExample {
    static func main() async throws {
        print("=== BufferPool Example ===\n")
        
        // Get the default Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not available on this device")
            return
        }
        
        // Create a buffer pool
        let pool = BufferPool(device: device)
        print("Created BufferPool for device: \(device.name)")
        
        // Example 1: Basic buffer acquisition and release
        print("\n1. Basic Usage:")
        let buffer1 = try await pool.acquire(size: 1000)
        print("   Acquired buffer for 1000 elements, actual size: \(buffer1.count)")
        
        // Use the buffer for some computation...
        
        // Release it back to the pool
        await pool.release(buffer1)
        print("   Released buffer back to pool")
        
        // Example 2: Demonstrating buffer reuse
        print("\n2. Buffer Reuse:")
        let buffer2 = try await pool.acquire(size: 1000)
        print("   Acquired another buffer of same size")
        print("   Buffer reused: \(buffer1.buffer === buffer2.buffer)")
        await pool.release(buffer2)
        
        // Example 3: Different size buckets
        print("\n3. Size Buckets:")
        let sizes = [100, 200, 500, 1000, 2000]
        var buffers: [MetalBuffer] = []
        
        for size in sizes {
            let buffer = try await pool.acquire(size: size)
            print("   Requested: \(size), Got: \(buffer.count)")
            buffers.append(buffer)
        }
        
        // Release all buffers
        for buffer in buffers {
            await pool.release(buffer)
        }
        
        // Example 4: Pool statistics
        print("\n4. Pool Statistics:")
        let stats = await pool.getStatistics()
        print("   Total acquisitions: \(stats.acquisitionCount)")
        print("   Buffers reused: \(stats.reuseCount)")
        print("   New allocations: \(stats.allocationCount)")
        print("   Reuse rate: \(String(format: "%.1f", stats.reuseRate))%")
        print("   Pooling efficiency: \(String(format: "%.1f", stats.poolingEfficiency))%")
        
        // Example 5: Performance comparison
        print("\n5. Performance Comparison:")
        
        // Without pooling
        let withoutPoolStart = Date()
        for _ in 0..<1000 {
            let size = 1024 * MemoryLayout<Float>.stride
            guard let _ = device.makeBuffer(length: size, options: .storageModeShared) else {
                throw BufferPoolError.allocationFailed(size: 1024)
            }
            // Buffer gets deallocated
        }
        let withoutPoolTime = Date().timeIntervalSince(withoutPoolStart)
        
        // With pooling
        let withPoolStart = Date()
        for _ in 0..<1000 {
            let buffer = try await pool.acquire(size: 1024)
            await pool.release(buffer)
        }
        let withPoolTime = Date().timeIntervalSince(withPoolStart)
        
        print("   Without pooling: \(String(format: "%.3f", withoutPoolTime)) seconds")
        print("   With pooling: \(String(format: "%.3f", withPoolTime)) seconds")
        print("   Speedup: \(String(format: "%.1fx", withoutPoolTime / withPoolTime))")
        
        // Example 6: Pool state inspection
        print("\n6. Pool State:")
        let poolState = await pool.getPoolState()
        for (size, count) in poolState.sorted(by: { $0.key < $1.key }) {
            print("   Size \(size): \(count) buffers")
        }
        
        // Example 7: Memory pressure handling
        print("\n7. Memory Management:")
        print("   Clearing pool to free memory...")
        await pool.clear()
        
        let statsAfterClear = await pool.getStatistics()
        print("   Buffers cleared: \(statsAfterClear.clearedCount)")
        print("   Current pooled: \(statsAfterClear.currentPooledCount)")
        
        print("\n=== Example Complete ===")
    }
}
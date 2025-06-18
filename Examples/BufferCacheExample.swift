// BufferCacheExample.swift
// VectorStoreKit
//
// Example demonstrating BufferCache usage for managing Metal buffer memory
// with bounded growth and LRU eviction policy.

import Foundation
import Metal
import VectorStoreKit

@main
struct BufferCacheExample {
    static func main() async throws {
        print("=== BufferCache Example ===\n")
        
        // Create Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Error: Metal device not available")
            return
        }
        
        print("Metal device: \(device.name)")
        
        // Create cache with 100MB limit
        let cache = BufferCache(device: device, maxMemory: 104_857_600)
        print("Created BufferCache with 100MB memory limit\n")
        
        // Example 1: Basic store and retrieve
        print("Example 1: Basic Store and Retrieve")
        print("-" * 40)
        
        // Create a buffer for weights
        let weightsBuffer = try await createBuffer(
            device: device,
            name: "layer1_weights",
            size: 1000
        )
        
        // Store in cache
        try await cache.store(buffer: weightsBuffer, for: "layer1_weights")
        print("✓ Stored layer1_weights buffer (size: \(weightsBuffer.count))")
        
        // Retrieve from cache
        if let retrieved = await cache.retrieve(for: "layer1_weights") {
            print("✓ Retrieved layer1_weights from cache")
            print("  Buffer size: \(retrieved.count)")
        }
        
        // Example 2: Get or create pattern
        print("\nExample 2: Get or Create Pattern")
        print("-" * 40)
        
        var createCount = 0
        
        // First call creates the buffer
        let biasBuffer1 = try await cache.getOrCreate(key: "layer1_bias") {
            createCount += 1
            print("  Creating new bias buffer...")
            return try await createBuffer(
                device: device,
                name: "layer1_bias",
                size: 100
            )
        }
        print("✓ Got layer1_bias buffer (created: \(createCount == 1))")
        
        // Second call retrieves from cache
        createCount = 0
        let biasBuffer2 = try await cache.getOrCreate(key: "layer1_bias") {
            createCount += 1
            return try await createBuffer(
                device: device,
                name: "layer1_bias",
                size: 100
            )
        }
        print("✓ Got layer1_bias buffer again (created: \(createCount == 1))")
        
        // Example 3: Memory pressure and eviction
        print("\nExample 3: Memory Pressure and Eviction")
        print("-" * 40)
        
        // Fill cache with multiple buffers
        let bufferSizeMB = 10
        let bufferSize = bufferSizeMB * 1_048_576 / MemoryLayout<Float>.stride
        
        print("Filling cache with 10MB buffers...")
        for i in 0..<12 {
            let buffer = try await createBuffer(
                device: device,
                name: "large_buffer_\(i)",
                size: bufferSize
            )
            try await cache.store(buffer: buffer, for: "large_buffer_\(i)")
            
            let stats = await cache.getStatistics()
            print("  Buffer \(i): Memory usage = \(stats.currentMemoryUsage / 1_048_576)MB " +
                  "(\(Int(stats.memoryUtilization))% of limit)")
        }
        
        // Check what's still in cache
        print("\nChecking cache contents after eviction:")
        for i in 0..<12 {
            let exists = await cache.retrieve(for: "large_buffer_\(i)") != nil
            print("  large_buffer_\(i): \(exists ? "✓ cached" : "✗ evicted")")
        }
        
        // Example 4: LRU behavior
        print("\nExample 4: LRU Eviction Behavior")
        print("-" * 40)
        
        // Clear cache for clean test
        await cache.clear()
        print("Cleared cache")
        
        // Add buffers
        for i in 0..<5 {
            let buffer = try await createBuffer(
                device: device,
                name: "lru_buffer_\(i)",
                size: bufferSize
            )
            try await cache.store(buffer: buffer, for: "lru_buffer_\(i)")
        }
        
        // Access some buffers to make them "recently used"
        print("\nAccessing buffers 0, 1, 2 to mark them as recently used...")
        _ = await cache.retrieve(for: "lru_buffer_0")
        _ = await cache.retrieve(for: "lru_buffer_1")
        _ = await cache.retrieve(for: "lru_buffer_2")
        
        // Add more buffers to trigger eviction
        print("\nAdding more buffers to trigger eviction...")
        for i in 5..<8 {
            let buffer = try await createBuffer(
                device: device,
                name: "lru_buffer_\(i)",
                size: bufferSize
            )
            try await cache.store(buffer: buffer, for: "lru_buffer_\(i)")
        }
        
        // Check which buffers survived
        print("\nCache contents after LRU eviction:")
        for i in 0..<8 {
            let exists = await cache.retrieve(for: "lru_buffer_\(i)") != nil
            print("  lru_buffer_\(i): \(exists ? "✓ cached" : "✗ evicted")")
        }
        
        // Example 5: Statistics and monitoring
        print("\nExample 5: Statistics and Monitoring")
        print("-" * 40)
        
        let finalStats = await cache.getStatistics()
        print("Cache Statistics:")
        print("  Total hits: \(finalStats.hitCount)")
        print("  Total misses: \(finalStats.missCount)")
        print("  Hit rate: \(String(format: "%.1f", finalStats.hitRate))%")
        print("  Current buffers: \(finalStats.currentBufferCount)")
        print("  Memory usage: \(finalStats.currentMemoryUsage / 1_048_576)MB")
        print("  Memory utilization: \(String(format: "%.1f", finalStats.memoryUtilization))%")
        print("  Total evictions: \(finalStats.evictionCount)")
        
        // Example 6: Memory pressure handling
        print("\nExample 6: Memory Pressure Handling")
        print("-" * 40)
        
        print("Current memory usage: \(finalStats.currentMemoryUsage / 1_048_576)MB")
        print("Simulating memory pressure...")
        
        await cache.handleMemoryPressure()
        
        let pressureStats = await cache.getStatistics()
        print("After memory pressure handling:")
        print("  Memory usage: \(pressureStats.currentMemoryUsage / 1_048_576)MB")
        print("  Buffers remaining: \(pressureStats.currentBufferCount)")
        
        print("\n=== BufferCache Example Complete ===")
    }
    
    // Helper function to create a test buffer
    static func createBuffer(
        device: MTLDevice,
        name: String,
        size: Int
    ) async throws -> MetalBuffer {
        let byteSize = size * MemoryLayout<Float>.stride
        guard let mtlBuffer = device.makeBuffer(
            length: byteSize,
            options: .storageModeShared
        ) else {
            throw BufferCacheError.allocationFailed(
                reason: "Failed to create buffer '\(name)'"
            )
        }
        
        // Fill with sample data
        let pointer = mtlBuffer.contents().bindMemory(to: Float.self, capacity: size)
        for i in 0..<min(size, 10) {
            pointer[i] = Float(i) * 0.1
        }
        
        return MetalBuffer(buffer: mtlBuffer, count: size)
    }
}

// Helper to repeat a string
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}
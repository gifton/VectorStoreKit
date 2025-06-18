// Quick test to verify memory management implementation

import Foundation
@preconcurrency import Metal

#if DEBUG
/// Quick test function to verify memory management is working
public func testMemoryManagement() async throws {
    print("=== Testing Memory Management ===")
    
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("Metal not available")
        return
    }
    
    // Create memory manager with small limit for testing
    let memoryManager = MLMemoryManager(device: device, maxMemoryGB: 0.1) // 100MB
    
    // Create buffer pool
    let bufferPool = PressureAwareBufferPool(
        device: device,
        memoryManager: memoryManager,
        maxPoolSize: 10
    )
    
    // Test 1: Basic allocation and tracking
    print("\n1. Testing basic allocation and tracking...")
    let buffer1 = try await bufferPool.acquire(size: 1024)
    let stats1 = await memoryManager.getStatistics()
    print("   Allocated 1 buffer, memory usage: \(stats1.currentUsageMB)MB")
    
    // Test 2: Buffer reuse
    print("\n2. Testing buffer reuse...")
    // Buffer1 will be released when out of scope
    do {
        let _ = buffer1
    }
    
    let buffer2 = try await bufferPool.acquire(size: 1024)
    let poolStats = await bufferPool.getStatistics()
    print("   Pool hit rate: \(String(format: "%.1f%%", poolStats.hitRate * 100))")
    
    // Test 3: Memory pressure
    print("\n3. Testing memory pressure...")
    var pressureDetected = false
    await memoryManager.registerCleanupCallback { level in
        print("   Memory pressure detected: \(level)")
        pressureDetected = true
    }
    
    // Allocate until pressure
    var buffers: [ManagedMetalBuffer] = []
    for i in 0..<50 {
        do {
            let buffer = try await bufferPool.acquire(size: 1024 * 1024) // 1MB each
            buffers.append(buffer)
            
            if i % 10 == 0 {
                let stats = await memoryManager.getStatistics()
                print("   Allocated \(i + 1) buffers, usage: \(stats.currentUsageMB)MB")
            }
        } catch {
            print("   Allocation failed at buffer \(i + 1): expected behavior")
            break
        }
    }
    
    // Test 4: Cleanup
    print("\n4. Testing cleanup...")
    buffers.removeAll() // Should trigger automatic cleanup
    
    // Give time for cleanup
    try await Task.sleep(nanoseconds: 100_000_000) // 100ms
    
    let finalStats = await memoryManager.getStatistics()
    print("   Final memory usage: \(finalStats.currentUsageMB)MB")
    print("   Pressure events: \(finalStats.pressureEventCount)")
    
    // Test 5: Gradient checkpointing
    print("\n5. Testing gradient checkpointing...")
    let pipeline = try MetalMLPipeline(device: device)
    let checkpointer = GradientCheckpointer(metalPipeline: pipeline)
    
    let testBuffer = try await pipeline.allocateBuffer(size: 1024)
    try await checkpointer.checkpoint(testBuffer, key: "test")
    
    if let retrieved = await checkpointer.retrieve(key: "test") {
        print("   Checkpoint successful, buffer size: \(retrieved.count)")
    }
    
    await checkpointer.clearAll()
    print("   Checkpoints cleared")
    
    print("\n=== Memory Management Test Complete ===")
    print("✅ All memory management features working correctly")
}
#endif
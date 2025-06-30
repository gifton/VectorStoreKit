import XCTest
// VectorStoreKit Metal Resource Management Example
//
// Demonstrates proper Metal resource lifecycle management

import Foundation
import Metal
import VectorStoreKit

final class ResourceManagementTests: XCTestCase {
    func testMain() async throws {
        print("Metal Resource Management Example")
        print("=================================")
        
        // Get Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Error: Metal not supported on this device")
            return
        }
        
        print("Using Metal device: \(device.name)")
        print("Max buffer length: \(device.maxBufferLength / 1024 / 1024) MB")
        
        // Example 1: Proper buffer pool usage
        await demonstrateBufferPoolUsage(device: device)
        
        // Example 2: Command queue management
        await demonstrateCommandQueueManagement(device: device)
        
        // Example 3: Memory pressure handling
        await demonstrateMemoryPressureHandling(device: device)
        
        // Example 4: Resource cleanup
        await demonstrateResourceCleanup(device: device)
    }
    
    func testDemonstrateBufferPoolUsage(device: MTLDevice) async {
        print("\n1. Buffer Pool Usage")
        print("-------------------")
        
        let bufferPool = MetalMLBufferPool(device: device)
        
        // Allocate buffers of various sizes
        do {
            let sizes = [1024, 2048, 4096, 8192, 16384]
            var buffers: [MetalBuffer] = []
            
            // Allocate buffers
            for size in sizes {
                let buffer = try await bufferPool.acquire(size: size)
                buffers.append(buffer)
                print("Allocated buffer of size \(size) elements")
            }
            
            // Get statistics
            let stats = await bufferPool.getStatistics()
            print("\nBuffer pool statistics:")
            print("- Total buffers: \(stats.totalBuffers)")
            print("- Memory usage: \(stats.memoryUsage / 1024 / 1024) MB")
            
            // Release buffers back to pool
            for buffer in buffers {
                await bufferPool.release(buffer)
            }
            print("\nReleased all buffers back to pool")
            
            // Reacquire to demonstrate reuse
            let reusedBuffer = try await bufferPool.acquire(size: 1024)
            print("Reacquired buffer from pool (reuse demonstration)")
            await bufferPool.release(reusedBuffer)
            
        } catch {
            print("Error: \(error)")
        }
    }
    
    func testDemonstrateCommandQueueManagement(device: MTLDevice) async {
        print("\n2. Command Queue Management")
        print("--------------------------")
        
        let commandQueue = MetalCommandQueue(device: device, label: "ExampleQueue", maxConcurrentOperations: 4)
        
        // Submit multiple operations
        do {
            // Sequential execution
            print("Executing sequential operations...")
            let operations = (0..<3).map { i in
                return { (buffer: MTLCommandBuffer) async throws in
                    print("  - Operation \(i) executing")
                    // Simulate work
                    try await Task.sleep(nanoseconds: 100_000_000) // 0.1s
                }
            }
            
            let startSeq = Date()
            try await commandQueue.executeSequence(operations)
            let seqDuration = Date().timeIntervalSince(startSeq)
            print("Sequential execution completed in \(String(format: "%.2f", seqDuration))s")
            
            // Parallel execution
            print("\nExecuting parallel operations...")
            let startPar = Date()
            try await commandQueue.executeParallel(operations)
            let parDuration = Date().timeIntervalSince(startPar)
            print("Parallel execution completed in \(String(format: "%.2f", parDuration))s")
            
            // Check pending buffers
            let pendingCount = await commandQueue.pendingBufferCount
            print("\nPending buffers after execution: \(pendingCount)")
            
        } catch {
            print("Error: \(error)")
        }
    }
    
    func testDemonstrateMemoryPressureHandling(device: MTLDevice) async {
        print("\n3. Memory Pressure Handling")
        print("--------------------------")
        
        let bufferPool = MetalMLBufferPool(device: device)
        
        do {
            // Allocate many buffers to simulate memory pressure
            var buffers: [MetalBuffer] = []
            let largeSize = 1024 * 1024 // 1M elements (4MB each)
            
            print("Allocating large buffers to simulate memory pressure...")
            for i in 0..<10 {
                do {
                    let buffer = try await bufferPool.acquire(size: largeSize)
                    buffers.append(buffer)
                    
                    let stats = await bufferPool.getStatistics()
                    print("Buffer \(i): Memory usage = \(stats.memoryUsage / 1024 / 1024) MB")
                } catch {
                    print("Allocation failed at buffer \(i): \(error)")
                    break
                }
            }
            
            // Trigger manual memory pressure handling
            print("\nTriggering memory pressure cleanup...")
            await bufferPool.handleMemoryPressure()
            
            // Release buffers
            for buffer in buffers {
                await bufferPool.release(buffer)
            }
            
            let finalStats = await bufferPool.getStatistics()
            print("Final memory usage: \(finalStats.memoryUsage / 1024 / 1024) MB")
            
        } catch {
            print("Error: \(error)")
        }
    }
    
    func testDemonstrateResourceCleanup(device: MTLDevice) async {
        print("\n4. Resource Cleanup")
        print("------------------")
        
        // Create a scope to demonstrate cleanup
        do {
            let bufferPool = MetalMLBufferPool(device: device)
            let commandQueue = MetalCommandQueue(device: device)
            
            // Allocate resources
            let buffer1 = try await bufferPool.acquire(size: 1024)
            let buffer2 = try await bufferPool.acquire(size: 2048)
            
            // Use resources
            try await commandQueue.execute { cmdBuffer in
                // Simulate GPU work
                print("Executing GPU operations...")
            }
            
            // Explicit cleanup
            await bufferPool.release(buffer1)
            await bufferPool.release(buffer2)
            
            // Wait for all GPU work to complete
            try await commandQueue.waitForCompletion()
            
            print("All resources cleaned up properly")
            
            // Resources will be automatically cleaned up when going out of scope
        } catch {
            print("Error during cleanup: \(error)")
        }
        
        print("\nResource cleanup demonstration completed")
    }
}

// Helper extension for formatting
extension Int {
    var formatted: String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: self)) ?? "\(self)"
    }
}
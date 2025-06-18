// EnhancedMemoryPoolExample.swift
// VectorStoreKit Examples
//
// Demonstrates the enhanced memory pool manager with size tracking,
// memory pressure integration, automatic resizing, and defragmentation

import Foundation
import VectorStoreKit

@main
struct EnhancedMemoryPoolExample {
    static func main() async throws {
        print("=== Enhanced Memory Pool Manager Example ===\n")
        
        // Create memory manager
        let memoryManager = MemoryManager()
        
        // Create pool manager with auto-defragmentation
        let poolManager = MemoryPoolManager(
            memoryManager: memoryManager,
            autoDefragmentationEnabled: true,
            defragmentationInterval: 60 // 1 minute
        )
        
        // Example 1: Size-tracked allocations
        print("1. Size-Tracked Allocations")
        await demonstrateSizeTracking(poolManager: poolManager)
        print()
        
        // Example 2: Memory pressure handling
        print("2. Memory Pressure Handling")
        await demonstrateMemoryPressure(poolManager: poolManager)
        print()
        
        // Example 3: Automatic pool resizing
        print("3. Automatic Pool Resizing")
        await demonstrateAutoResizing(poolManager: poolManager)
        print()
        
        // Example 4: Defragmentation
        print("4. Defragmentation Strategy")
        await demonstrateDefragmentation(poolManager: poolManager)
        print()
        
        // Example 5: Usage pattern optimization
        print("5. Usage Pattern Optimization")
        await demonstrateUsagePatterns(poolManager: poolManager)
        print()
        
        // Final report
        print("=== Final Memory Report ===")
        await printMemoryReport(poolManager: poolManager)
    }
    
    // MARK: - Demonstrations
    
    static func demonstrateSizeTracking(poolManager: MemoryPoolManager) async {
        let pool = await poolManager.createEnhancedBufferPool(
            for: "size_tracking_demo",
            maxPoolSize: 50
        )
        
        // Allocate buffers of various sizes
        var buffers: [(UnsafeMutableRawPointer, Int)] = []
        
        for size in [512, 1024, 2048, 4096, 8192] {
            if let buffer = try? await pool.acquireBuffer(size: size) {
                buffers.append((buffer, size))
                print("  Allocated buffer of size: \(size) bytes")
            }
        }
        
        // Check statistics
        let stats = await pool.statistics()
        print("  Current memory in use: \(stats.currentlyInUse) bytes")
        print("  Total allocated: \(stats.totalAllocated) bytes")
        
        // Release buffers - size is tracked automatically
        for (buffer, _) in buffers {
            await pool.release(buffer)
        }
        
        let finalStats = await pool.statistics()
        print("  After release - memory in use: \(finalStats.currentlyInUse) bytes")
    }
    
    static func demonstrateMemoryPressure(poolManager: MemoryPoolManager) async {
        let pool = await poolManager.createEnhancedBufferPool(
            for: "pressure_demo",
            maxPoolSize: 100
        )
        
        // Pre-allocate many buffers
        await pool.preAllocate(count: 50)
        
        let beforeStats = await pool.statistics()
        print("  Before pressure - total allocated: \(beforeStats.totalAllocated) bytes")
        
        // Simulate memory pressure
        print("  Simulating warning-level memory pressure...")
        await poolManager.handleMemoryPressure(.warning)
        
        // Wait for cleanup
        try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
        
        let afterStats = await pool.statistics()
        print("  After pressure - total allocated: \(afterStats.totalAllocated) bytes")
        print("  Memory reduced by: \(beforeStats.totalAllocated - afterStats.totalAllocated) bytes")
    }
    
    static func demonstrateAutoResizing(poolManager: MemoryPoolManager) async {
        let pool = await poolManager.createEnhancedBufferPool(
            for: "auto_resize_demo",
            maxPoolSize: 200,
            growthFactor: 2.0,
            shrinkThreshold: 0.1
        )
        
        // Create heavy usage pattern for 1KB buffers
        print("  Creating usage pattern for 1KB buffers...")
        var buffers: [UnsafeMutableRawPointer] = []
        
        for i in 0..<30 {
            if let buffer = try? await pool.acquireBuffer(size: 1024) {
                buffers.append(buffer)
                if i % 10 == 9 {
                    print("    Allocated \(i + 1) buffers")
                }
            }
        }
        
        // Release all to return to pool
        for buffer in buffers {
            await pool.release(buffer)
        }
        
        // Pool should now have grown for 1KB size
        let stats = await pool.statistics()
        print("  Pool has adapted - hit rate: \(String(format: "%.1f%%", stats.hitRate * 100))")
        
        // Now allocate again - should have high hit rate
        buffers.removeAll()
        for _ in 0..<20 {
            if let buffer = try? await pool.acquireBuffer(size: 1024) {
                buffers.append(buffer)
            }
        }
        
        let newStats = await pool.statistics()
        print("  After reuse - hit rate: \(String(format: "%.1f%%", newStats.hitRate * 100))")
        
        // Cleanup
        for buffer in buffers {
            await pool.release(buffer)
        }
    }
    
    static func demonstrateDefragmentation(poolManager: MemoryPoolManager) async {
        let pool = await poolManager.createEnhancedBufferPool(
            for: "defrag_demo",
            defragmentationThreshold: 0.3
        )
        
        // Create fragmentation by allocating various sizes
        print("  Creating fragmented memory state...")
        var buffers: [(UnsafeMutableRawPointer, Int)] = []
        
        for i in 0..<20 {
            let size = (i % 5 + 1) * 1024 // 1KB to 5KB
            if let buffer = try? await pool.acquireBuffer(size: size) {
                buffers.append((buffer, size))
            }
        }
        
        // Release every other buffer to create fragmentation
        for (index, (buffer, _)) in buffers.enumerated() where index % 2 == 0 {
            await pool.release(buffer)
        }
        
        let beforeStats = await pool.statistics()
        print("  Fragmentation ratio: \(String(format: "%.1f%%", beforeStats.fragmentationRatio * 100))")
        
        // Force defragmentation
        print("  Running defragmentation...")
        let result = await pool.defragment()
        
        print("  Defragmentation results:")
        print("    - Bytes reclaimed: \(result.bytesReclaimed)")
        print("    - Objects compacted: \(result.objectsCompacted)")
        print("    - Duration: \(String(format: "%.3f", result.duration)) seconds")
        print("    - Fragmentation: \(String(format: "%.1f%%", result.fragmentationBefore * 100)) -> \(String(format: "%.1f%%", result.fragmentationAfter * 100))")
        
        // Cleanup
        for (index, (buffer, _)) in buffers.enumerated() where index % 2 == 1 {
            await pool.release(buffer)
        }
    }
    
    static func demonstrateUsagePatterns(poolManager: MemoryPoolManager) async {
        let pool = await poolManager.createEnhancedBufferPool(
            for: "pattern_demo"
        )
        
        print("  Creating diverse usage patterns...")
        
        // Pattern 1: Many small allocations (512 bytes)
        for _ in 0..<50 {
            if let buffer = try? await pool.acquireBuffer(size: 512) {
                await pool.release(buffer)
            }
        }
        
        // Pattern 2: Medium allocations (4KB)
        for _ in 0..<20 {
            if let buffer = try? await pool.acquireBuffer(size: 4096) {
                await pool.release(buffer)
            }
        }
        
        // Pattern 3: Large allocations (16KB)
        for _ in 0..<5 {
            if let buffer = try? await pool.acquireBuffer(size: 16384) {
                await pool.release(buffer)
            }
        }
        
        print("  Usage patterns established")
        
        // Pre-allocate based on patterns
        print("  Pre-allocating based on usage patterns...")
        await pool.preAllocate(count: 30)
        
        // Test hit rates for each pattern
        var smallHits = 0, mediumHits = 0, largeHits = 0
        let statsBefore = await pool.statistics()
        
        // Small allocations
        for _ in 0..<10 {
            if let buffer = try? await pool.acquireBuffer(size: 512) {
                await pool.release(buffer)
            }
        }
        let statsSmall = await pool.statistics()
        smallHits = Int((statsSmall.hitRate - statsBefore.hitRate) * 100)
        
        // Medium allocations
        for _ in 0..<5 {
            if let buffer = try? await pool.acquireBuffer(size: 4096) {
                await pool.release(buffer)
            }
        }
        let statsMedium = await pool.statistics()
        mediumHits = Int((statsMedium.hitRate - statsSmall.hitRate) * 100)
        
        // Large allocations
        for _ in 0..<2 {
            if let buffer = try? await pool.acquireBuffer(size: 16384) {
                await pool.release(buffer)
            }
        }
        let statsLarge = await pool.statistics()
        largeHits = Int((statsLarge.hitRate - statsMedium.hitRate) * 100)
        
        print("  Pattern-based pre-allocation results:")
        print("    - Small (512B): High priority in pre-allocation")
        print("    - Medium (4KB): Medium priority in pre-allocation")
        print("    - Large (16KB): Low priority in pre-allocation")
    }
    
    static func printMemoryReport(poolManager: MemoryPoolManager) async {
        let report = await poolManager.memoryReport()
        
        print("Total Pools: \(report.totalPools)")
        print("Total Memory Allocated: \(formatBytes(report.totalMemoryAllocated))")
        print("Total Memory In Use: \(formatBytes(report.totalMemoryInUse))")
        print("Utilization Rate: \(String(format: "%.1f%%", report.utilizationRate * 100))")
        print("Average Hit Rate: \(String(format: "%.1f%%", report.averageHitRate * 100))")
        print("Average Fragmentation: \(String(format: "%.1f%%", report.averageFragmentation * 100))")
        
        if let lastDefrag = report.lastDefragmentation {
            let formatter = DateFormatter()
            formatter.dateStyle = .none
            formatter.timeStyle = .medium
            print("Last Defragmentation: \(formatter.string(from: lastDefrag))")
        }
        
        print("\nPer-Pool Statistics:")
        for (name, stats) in report.poolStatistics {
            print("  \(name):")
            print("    - Memory: \(formatBytes(stats.totalMemoryBytes))")
            print("    - In Use: \(formatBytes(stats.currentlyInUse))")
            print("    - Hit Rate: \(String(format: "%.1f%%", stats.hitRate * 100))")
            print("    - Fragmentation: \(String(format: "%.1f%%", stats.fragmentationRatio * 100))")
        }
    }
    
    // MARK: - Helpers
    
    static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}
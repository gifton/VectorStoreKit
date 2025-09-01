import XCTest
// MemoryManagementExample.swift
// VectorStoreKit
//
// Example demonstrating memory management and gradient clipping functionality

import Foundation
import Metal
import VectorStoreKit

final class MemoryManagementTests: XCTestCase {
    func testMain() async throws {
        print("=== VectorStoreKit Memory Management Example ===\n")
        
        // Get Metal device
        guard let device = MTLDevice.default else {
            print("Error: Metal device not available")
            return
        }
        
        // Initialize components
        let bufferPool = BufferPool(device: device)
        let bufferCache = BufferCache(device: device, maxMemory: 512_000_000) // 512MB limit
        let memoryManager = MemoryManager()
        
        // Register components with memory manager
        await memoryManager.registerBufferPool(bufferPool)
        await memoryManager.registerBufferCache(bufferCache)
        
        print("1. Memory Manager Setup Complete")
        print("   - Buffer Pool registered")
        print("   - Buffer Cache registered (512MB limit)")
        
        // Demonstrate buffer pool usage
        print("\n2. Buffer Pool Usage:")
        
        // Acquire some buffers
        let buffer1 = try await bufferPool.acquire(size: 1000)
        let buffer2 = try await bufferPool.acquire(size: 2000)
        let buffer3 = try await bufferPool.acquire(size: 4000)
        
        var poolStats = await bufferPool.getStatistics()
        print("   - Acquired 3 buffers")
        print("   - Allocations: \(poolStats.allocationCount)")
        print("   - Reuse rate: \(poolStats.reuseRate)%")
        
        // Release buffers back to pool
        await bufferPool.release(buffer1)
        await bufferPool.release(buffer2)
        
        // Acquire again to demonstrate reuse
        let buffer4 = try await bufferPool.acquire(size: 1000)
        poolStats = await bufferPool.getStatistics()
        print("   - After reuse - Allocations: \(poolStats.allocationCount), Reuse rate: \(poolStats.reuseRate)%")
        
        // Demonstrate buffer cache usage
        print("\n3. Buffer Cache Usage:")
        
        // Cache some buffers
        try await bufferCache.store(buffer: buffer3, for: "weights_layer1")
        try await bufferCache.store(buffer: buffer4, for: "weights_layer2")
        
        var cacheStats = await bufferCache.getStatistics()
        print("   - Cached 2 buffers")
        print("   - Memory usage: \(cacheStats.currentMemoryUsage / 1024)KB / \(cacheStats.memoryLimit / 1_048_576)MB")
        print("   - Hit rate: \(cacheStats.hitRate)%")
        
        // Retrieve from cache
        if let cached = await bufferCache.retrieve(for: "weights_layer1") {
            print("   - Successfully retrieved 'weights_layer1' from cache")
        }
        
        // Demonstrate memory pressure handling
        print("\n4. Memory Pressure Simulation:")
        
        // Get initial statistics
        let initialStats = await memoryManager.getMemoryStatistics()
        print("   - Initial memory usage: \(initialStats.currentMemoryUsage / 1024)KB")
        
        // Simulate warning-level pressure
        await memoryManager.handleMemoryPressure(level: .warning)
        
        let afterWarningStats = await memoryManager.getMemoryStatistics()
        print("   - After warning pressure: \(afterWarningStats.currentMemoryUsage / 1024)KB")
        print("   - Buffers released: \(afterWarningStats.buffersReleased)")
        print("   - Cache entries evicted: \(afterWarningStats.cacheEntriesEvicted)")
        
        // Demonstrate gradient clipping
        print("\n5. Gradient Clipping Example:")
        
        guard let commandQueue = device.makeCommandQueue() else {
            print("Error: Could not create command queue")
            return
        }
        
        let shaderLibrary = try MLShaderLibrary(device: device)
        let mlOperations = MetalMLOperations(
            device: device,
            commandQueue: commandQueue,
            shaderLibrary: shaderLibrary
        )
        
        // Create a gradient buffer with some large values
        let gradientData: [Float] = [
            10.0, -15.0, 8.0, -12.0, 20.0,  // Large gradients
            2.0, -3.0, 1.0, -2.0, 4.0       // Normal gradients
        ]
        
        guard let gradientMTLBuffer = device.makeBuffer(
            bytes: gradientData,
            length: gradientData.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            print("Error: Could not create gradient buffer")
            return
        }
        
        let gradientBuffer = MetalBuffer(buffer: gradientMTLBuffer, count: gradientData.count)
        
        // Compute initial L2 norm
        let initialNorm = try await mlOperations.computeL2Norm(gradientBuffer)
        print("   - Initial gradient L2 norm: \(initialNorm)")
        
        // Apply gradient clipping with max norm of 10.0
        let maxNorm: Float = 10.0
        try await mlOperations.clipGradients(gradientBuffer, maxNorm: maxNorm)
        
        // Compute norm after clipping
        let clippedNorm = try await mlOperations.computeL2Norm(gradientBuffer)
        print("   - Clipped gradient L2 norm: \(clippedNorm)")
        print("   - Max allowed norm: \(maxNorm)")
        
        // Show clipped gradient values
        let clippedGradients = gradientBuffer.toArray()
        print("   - Clipped gradients: \(clippedGradients.map { String(format: "%.2f", $0) }.joined(separator: ", "))")
        
        // Demonstrate batch size management
        print("\n6. Batch Size Management:")
        
        let batchManager = DefaultBatchSizeManager(initialBatchSize: 128, minBatchSize: 16)
        await memoryManager.registerBatchSizeManager(batchManager)
        
        print("   - Initial batch size: \(await batchManager.getCurrentBatchSize())")
        
        // Simulate critical memory pressure
        await memoryManager.handleMemoryPressure(level: .critical)
        
        print("   - After critical pressure: \(await batchManager.getCurrentBatchSize())")
        
        // Final statistics
        print("\n7. Final Memory Statistics:")
        let finalStats = await memoryManager.getMemoryStatistics()
        print("   - Total pressure events: \(finalStats.pressureEventCount)")
        print("   - Average handling time: \(finalStats.averageHandlingTime)s")
        print("   - Total buffers released: \(finalStats.buffersReleased)")
        print("   - Total cache evictions: \(finalStats.cacheEntriesEvicted)")
        print("   - Batch size reductions: \(finalStats.batchSizeReductions)")
        
        print("\n=== Example Complete ===")
    }
}

// Helper to create MLShaderLibrary for the example
extension MLShaderLibrary {
    init(device: MTLDevice) throws {
        // This is a simplified initialization for the example
        // In practice, this would load the actual shader library
        guard let library = device.makeDefaultLibrary() else {
            throw MetalMLError.shaderCompilationFailed("Could not create default library")
        }
        
        // Initialize with empty pipeline cache for example
        self.init(library: library, pipelineCache: [:])
    }
    
    private init(library: MTLLibrary, pipelineCache: [String: MTLComputePipelineState]) {
        // This would be implemented in the actual MLShaderLibrary
        fatalError("This is a simplified example initialization")
    }
}
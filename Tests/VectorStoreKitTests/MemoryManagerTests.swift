// MemoryManagerTests.swift
// VectorStoreKit Tests
//
// Tests for MemoryManager and gradient clipping functionality

import XCTest
import Metal
@testable import VectorStoreKit

final class MemoryManagerTests: XCTestCase {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var memoryManager: MemoryManager!
    var bufferPool: BufferPool!
    var bufferCache: BufferCache!
    
    override func setUp() async throws {
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device not available")
        
        commandQueue = device.makeCommandQueue()
        XCTAssertNotNil(commandQueue, "Could not create command queue")
        
        memoryManager = MemoryManager()
        bufferPool = BufferPool(device: device)
        bufferCache = BufferCache(device: device, maxMemory: 10_485_760) // 10MB for testing
    }
    
    // MARK: - Memory Manager Tests
    
    func testMemoryManagerRegistration() async throws {
        // Register components
        await memoryManager.registerBufferPool(bufferPool)
        await memoryManager.registerBufferCache(bufferCache)
        
        // Get statistics
        let stats = await memoryManager.getMemoryStatistics()
        XCTAssertEqual(stats.pressureEventCount, 0)
        XCTAssertEqual(stats.currentPressureLevel, .normal)
    }
    
    func testWarningLevelPressureHandling() async throws {
        // Register components
        await memoryManager.registerBufferPool(bufferPool)
        await memoryManager.registerBufferCache(bufferCache)
        
        // Create some buffers in pool
        let buffers = try await (0..<10).asyncMap { i in
            try await bufferPool.acquire(size: 1024)
        }
        
        // Release them back to pool
        for buffer in buffers {
            await bufferPool.release(buffer)
        }
        
        // Cache some buffers
        for i in 0..<5 {
            let buffer = try await bufferPool.acquire(size: 1024)
            try await bufferCache.store(buffer: buffer, for: "test_\(i)")
        }
        
        // Initial statistics
        let initialPoolStats = await bufferPool.getStatistics()
        let initialCacheStats = await bufferCache.getStatistics()
        
        XCTAssertGreaterThan(initialPoolStats.currentPooledCount, 0)
        XCTAssertGreaterThan(initialCacheStats.currentBufferCount, 0)
        
        // Handle warning pressure
        await memoryManager.handleMemoryPressure(level: .warning)
        
        // Verify cleanup occurred
        let afterStats = await memoryManager.getMemoryStatistics()
        XCTAssertEqual(afterStats.pressureEventCount, 1)
        XCTAssertGreaterThan(afterStats.buffersReleased, 0)
        
        // Pool should be cleared
        let poolStats = await bufferPool.getStatistics()
        XCTAssertEqual(poolStats.currentPooledCount, 0)
    }
    
    func testCriticalLevelPressureHandling() async throws {
        // Register components
        await memoryManager.registerBufferPool(bufferPool)
        await memoryManager.registerBufferCache(bufferCache)
        
        let batchManager = DefaultBatchSizeManager(initialBatchSize: 128)
        await memoryManager.registerBatchSizeManager(batchManager)
        
        // Cache some buffers
        for i in 0..<10 {
            let buffer = try await bufferPool.acquire(size: 1024)
            try await bufferCache.store(buffer: buffer, for: "test_\(i)")
        }
        
        let initialBatchSize = await batchManager.getCurrentBatchSize()
        XCTAssertEqual(initialBatchSize, 128)
        
        // Handle critical pressure
        await memoryManager.handleMemoryPressure(level: .critical)
        
        // Verify aggressive cleanup
        let afterStats = await memoryManager.getMemoryStatistics()
        XCTAssertEqual(afterStats.pressureEventCount, 1)
        XCTAssertGreaterThan(afterStats.cacheEntriesEvicted, 0)
        XCTAssertEqual(afterStats.batchSizeReductions, 1)
        
        // Batch size should be reduced by 50%
        let newBatchSize = await batchManager.getCurrentBatchSize()
        XCTAssertEqual(newBatchSize, 64)
        
        // Cache should have reduced memory usage
        let cacheStats = await bufferCache.getStatistics()
        XCTAssertLessThanOrEqual(cacheStats.memoryUtilization, 50.0)
    }
    
    func testManualCleanup() async throws {
        await memoryManager.registerBufferPool(bufferPool)
        
        // Add buffers to pool
        let buffers = try await (0..<5).asyncMap { _ in
            try await bufferPool.acquire(size: 512)
        }
        for buffer in buffers {
            await bufferPool.release(buffer)
        }
        
        // Light cleanup
        await memoryManager.triggerCleanup(level: .light)
        
        let poolStats = await bufferPool.getStatistics()
        XCTAssertEqual(poolStats.currentPooledCount, 0)
    }
    
    // MARK: - Gradient Clipping Tests
    
    func testGradientClippingWithLargeNorm() async throws {
        let shaderLibrary = try createTestShaderLibrary()
        let mlOperations = MetalMLOperations(
            device: device,
            commandQueue: commandQueue!,
            shaderLibrary: shaderLibrary
        )
        
        // Create gradients with large norm
        let gradientData: [Float] = [3.0, 4.0, 0.0, 0.0, 0.0] // L2 norm = 5.0
        let gradientBuffer = try createMetalBuffer(from: gradientData)
        
        // Verify initial norm
        let initialNorm = try await mlOperations.computeL2Norm(gradientBuffer)
        XCTAssertEqual(initialNorm, 5.0, accuracy: 0.001)
        
        // Clip with max norm of 2.0
        try await mlOperations.clipGradients(gradientBuffer, maxNorm: 2.0)
        
        // Verify clipped norm
        let clippedNorm = try await mlOperations.computeL2Norm(gradientBuffer)
        XCTAssertEqual(clippedNorm, 2.0, accuracy: 0.001)
        
        // Verify values are scaled correctly
        let clippedValues = gradientBuffer.toArray()
        let expectedScale: Float = 2.0 / 5.0
        XCTAssertEqual(clippedValues[0], 3.0 * expectedScale, accuracy: 0.001)
        XCTAssertEqual(clippedValues[1], 4.0 * expectedScale, accuracy: 0.001)
    }
    
    func testGradientClippingWithSmallNorm() async throws {
        let shaderLibrary = try createTestShaderLibrary()
        let mlOperations = MetalMLOperations(
            device: device,
            commandQueue: commandQueue!,
            shaderLibrary: shaderLibrary
        )
        
        // Create gradients with small norm
        let gradientData: [Float] = [0.6, 0.8, 0.0, 0.0, 0.0] // L2 norm = 1.0
        let gradientBuffer = try createMetalBuffer(from: gradientData)
        
        // Store original values
        let originalValues = gradientBuffer.toArray()
        
        // Clip with max norm of 5.0 (larger than current norm)
        try await mlOperations.clipGradients(gradientBuffer, maxNorm: 5.0)
        
        // Values should remain unchanged
        let afterValues = gradientBuffer.toArray()
        for (original, after) in zip(originalValues, afterValues) {
            XCTAssertEqual(original, after, accuracy: 0.001)
        }
    }
    
    func testL2NormComputation() async throws {
        let shaderLibrary = try createTestShaderLibrary()
        let mlOperations = MetalMLOperations(
            device: device,
            commandQueue: commandQueue!,
            shaderLibrary: shaderLibrary
        )
        
        // Test various vectors
        let testCases: [([Float], Float)] = [
            ([3.0, 4.0], 5.0),
            ([1.0, 1.0, 1.0, 1.0], 2.0),
            ([2.0, 0.0, 0.0], 2.0),
            ([1.0], 1.0),
            (Array(repeating: 1.0, count: 100), 10.0)
        ]
        
        for (vector, expectedNorm) in testCases {
            let buffer = try createMetalBuffer(from: vector)
            let computedNorm = try await mlOperations.computeL2Norm(buffer)
            XCTAssertEqual(computedNorm, expectedNorm, accuracy: 0.01,
                          "L2 norm of \(vector) should be \(expectedNorm)")
        }
    }
    
    func testScaleOperation() async throws {
        let shaderLibrary = try createTestShaderLibrary()
        let mlOperations = MetalMLOperations(
            device: device,
            commandQueue: commandQueue!,
            shaderLibrary: shaderLibrary
        )
        
        let originalData: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let buffer = try createMetalBuffer(from: originalData)
        
        // Scale by 0.5
        try await mlOperations.scale(buffer, by: 0.5)
        
        let scaledValues = buffer.toArray()
        for (original, scaled) in zip(originalData, scaledValues) {
            XCTAssertEqual(scaled, original * 0.5, accuracy: 0.001)
        }
    }
    
    // MARK: - Batch Size Manager Tests
    
    func testBatchSizeReduction() async throws {
        let manager = DefaultBatchSizeManager(initialBatchSize: 100, minBatchSize: 10)
        
        XCTAssertEqual(await manager.getCurrentBatchSize(), 100)
        
        // Reduce by 25%
        let reduced1 = await manager.reduceBatchSize(by: 0.25)
        XCTAssertTrue(reduced1)
        XCTAssertEqual(await manager.getCurrentBatchSize(), 75)
        
        // Reduce by 50%
        let reduced2 = await manager.reduceBatchSize(by: 0.5)
        XCTAssertTrue(reduced2)
        XCTAssertEqual(await manager.getCurrentBatchSize(), 38) // 75 - 37.5, rounded
        
        // Try to reduce below minimum
        let reduced3 = await manager.reduceBatchSize(by: 0.9)
        XCTAssertTrue(reduced3)
        XCTAssertEqual(await manager.getCurrentBatchSize(), 10) // Min batch size
        
        // Further reduction should fail
        let reduced4 = await manager.reduceBatchSize(by: 0.5)
        XCTAssertFalse(reduced4)
        XCTAssertEqual(await manager.getCurrentBatchSize(), 10)
    }
    
    // MARK: - Helper Methods
    
    private func createMetalBuffer(from array: [Float]) throws -> MetalBuffer {
        guard let mtlBuffer = device.makeBuffer(
            bytes: array,
            length: array.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw MetalMLError.bufferAllocationFailed(size: array.count)
        }
        return MetalBuffer(buffer: mtlBuffer, count: array.count)
    }
    
    private func createTestShaderLibrary() throws -> MLShaderLibrary {
        // For testing, we'll create a mock shader library
        // In a real implementation, this would load actual Metal shaders
        return try TestMLShaderLibrary(device: device)
    }
}

// MARK: - Test Helpers

private final class TestMLShaderLibrary: MLShaderLibrary {
    let device: MTLDevice
    let library: MTLLibrary
    
    init(device: MTLDevice) throws {
        self.device = device
        
        // For testing, use the default library
        guard let lib = device.makeDefaultLibrary() else {
            throw MetalMLError.shaderCompilationFailed("Could not create test library")
        }
        self.library = lib
        
        // In real implementation, this would be properly initialized
        // For testing, we're using a simplified version
        super.init()
    }
    
    override func pipeline(for functionName: String) throws -> MTLComputePipelineState {
        // For testing, create mock pipelines
        // In real implementation, these would be actual Metal compute pipelines
        
        // Check if we have the actual function, otherwise create a mock
        if let function = library.makeFunction(name: functionName) {
            return try device.makeComputePipelineState(function: function)
        }
        
        // Create a simple passthrough function for testing
        let source = """
        #include <metal_stdlib>
        using namespace metal;
        
        kernel void test_kernel(device float* data [[buffer(0)]],
                               uint gid [[thread_position_in_grid]]) {
            // Mock kernel for testing
        }
        """
        
        let options = MTLCompileOptions()
        let testLibrary = try device.makeLibrary(source: source, options: options)
        guard let function = testLibrary.makeFunction(name: "test_kernel") else {
            throw MetalMLError.shaderCompilationFailed("Could not create test function")
        }
        
        return try device.makeComputePipelineState(function: function)
    }
}

// Helper for async operations
extension Sequence {
    func asyncMap<T>(_ transform: @escaping (Element) async throws -> T) async throws -> [T] {
        var results: [T] = []
        for element in self {
            try await results.append(transform(element))
        }
        return results
    }
}
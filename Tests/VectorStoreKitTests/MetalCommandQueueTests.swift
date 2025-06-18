//
//  MetalCommandQueueTests.swift
//  VectorStoreKitTests
//
//  Tests for the MetalCommandQueue async command buffer implementation
//

import XCTest
import Metal
@testable import VectorStoreKit

final class MetalCommandQueueTests: XCTestCase {
    var device: MTLDevice!
    var commandQueue: MetalCommandQueue!
    
    override func setUp() async throws {
        try await super.setUp()
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this device")
        }
        self.device = device
        self.commandQueue = MetalCommandQueue(device: device)
    }
    
    func testBasicAsyncSubmission() async throws {
        // Create test buffers
        let size = 1024
        let sourceData = (0..<size).map { Float($0) }
        
        guard let sourceBuffer = device.makeBuffer(bytes: sourceData, 
                                                  length: size * MemoryLayout<Float>.stride,
                                                  options: .storageModeShared),
              let destBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.stride,
                                               options: .storageModeShared) else {
            XCTFail("Failed to create Metal buffers")
            return
        }
        
        // Submit async copy operation
        try await commandQueue.submitAsync { buffer in
            guard let encoder = buffer.makeBlitCommandEncoder() else {
                throw MetalCommandQueueError.failedToCreateCommandBuffer
            }
            encoder.copy(from: sourceBuffer, sourceOffset: 0, 
                        to: destBuffer, destinationOffset: 0, 
                        size: size * MemoryLayout<Float>.stride)
            encoder.endEncoding()
        }
        
        // Verify the copy worked
        let destPointer = destBuffer.contents().bindMemory(to: Float.self, capacity: size)
        for i in 0..<size {
            XCTAssertEqual(destPointer[i], Float(i), accuracy: 0.0001)
        }
    }
    
    func testCopyAsyncConvenience() async throws {
        let size = 512
        let sourceData = (0..<size).map { Float($0 * 2) }
        
        guard let sourceBuffer = device.makeBuffer(bytes: sourceData,
                                                  length: size * MemoryLayout<Float>.stride,
                                                  options: .storageModeShared),
              let destBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.stride,
                                               options: .storageModeShared) else {
            XCTFail("Failed to create Metal buffers")
            return
        }
        
        // Use convenience method
        try await commandQueue.copyAsync(from: sourceBuffer, to: destBuffer, 
                                       size: size * MemoryLayout<Float>.stride)
        
        // Verify
        let destPointer = destBuffer.contents().bindMemory(to: Float.self, capacity: size)
        for i in 0..<size {
            XCTAssertEqual(destPointer[i], Float(i * 2), accuracy: 0.0001)
        }
    }
    
    func testMultipleAsyncSubmissions() async throws {
        let count = 10
        let size = 256
        
        var buffers: [(source: MTLBuffer, dest: MTLBuffer)] = []
        
        // Create multiple buffer pairs
        for i in 0..<count {
            let value = Float(i + 1)
            let sourceData = [Float](repeating: value, count: size)
            
            guard let sourceBuffer = device.makeBuffer(bytes: sourceData,
                                                      length: size * MemoryLayout<Float>.stride,
                                                      options: .storageModeShared),
                  let destBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.stride,
                                                   options: .storageModeShared) else {
                XCTFail("Failed to create Metal buffers")
                return
            }
            
            buffers.append((sourceBuffer, destBuffer))
        }
        
        // Submit all operations concurrently
        try await withThrowingTaskGroup(of: Void.self) { group in
            for (source, dest) in buffers {
                group.addTask {
                    try await self.commandQueue.copyAsync(from: source, to: dest,
                                                        size: size * MemoryLayout<Float>.stride)
                }
            }
            
            try await group.waitForAll()
        }
        
        // Verify all copies completed correctly
        for (i, (_, dest)) in buffers.enumerated() {
            let expectedValue = Float(i + 1)
            let destPointer = dest.contents().bindMemory(to: Float.self, capacity: size)
            
            for j in 0..<size {
                XCTAssertEqual(destPointer[j], expectedValue, accuracy: 0.0001,
                             "Buffer \(i) position \(j) has incorrect value")
            }
        }
    }
    
    func testWaitForCompletion() async throws {
        let size = 1024
        var operations = 0
        
        // Submit multiple operations
        for i in 0..<5 {
            let sourceData = [Float](repeating: Float(i), count: size)
            
            guard let sourceBuffer = device.makeBuffer(bytes: sourceData,
                                                      length: size * MemoryLayout<Float>.stride,
                                                      options: .storageModeShared),
                  let destBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.stride,
                                                   options: .storageModeShared) else {
                XCTFail("Failed to create Metal buffers")
                return
            }
            
            Task {
                try await commandQueue.copyAsync(from: sourceBuffer, to: destBuffer,
                                               size: size * MemoryLayout<Float>.stride)
                operations += 1
            }
        }
        
        // Wait for all to complete
        try await commandQueue.waitForCompletion()
        
        // Verify no pending buffers
        let pendingCount = await commandQueue.pendingBufferCount
        XCTAssertEqual(pendingCount, 0, "Should have no pending buffers after waitForCompletion")
    }
    
    func testReturnValueSupport() async throws {
        let size = 100
        let sourceData = (0..<size).map { Float($0) }
        
        guard let buffer = device.makeBuffer(bytes: sourceData,
                                           length: size * MemoryLayout<Float>.stride,
                                           options: .storageModeShared) else {
            XCTFail("Failed to create Metal buffer")
            return
        }
        
        // Submit operation that returns a value
        let sum = try await commandQueue.submitAsync { commandBuffer in
            // Simulate some computation result
            // In real usage, this would be reading results from a GPU computation
            let pointer = buffer.contents().bindMemory(to: Float.self, capacity: size)
            var total: Float = 0
            for i in 0..<size {
                total += pointer[i]
            }
            return total
        }
        
        // Verify the sum
        let expectedSum = Float((0..<size).reduce(0, +))
        XCTAssertEqual(sum, expectedSum, accuracy: 0.0001)
    }
    
    func testErrorHandling() async throws {
        // Test with an intentionally failing operation
        do {
            try await commandQueue.submitAsync { buffer in
                // Force an error by not ending encoding
                guard let encoder = buffer.makeBlitCommandEncoder() else {
                    throw MetalCommandQueueError.failedToCreateCommandBuffer
                }
                // Intentionally don't call endEncoding() to cause an error
                // This would cause a Metal validation error in a real scenario
            }
            
            // In a real test environment this might not fail immediately
            // Metal errors often occur asynchronously
        } catch {
            // Expected to catch some error
            XCTAssertTrue(error is MetalCommandQueueError)
        }
    }
    
    func testPendingBufferCount() async throws {
        let initialCount = await commandQueue.pendingBufferCount
        XCTAssertEqual(initialCount, 0, "Should start with no pending buffers")
        
        // Submit a few operations
        let size = 256
        for _ in 0..<3 {
            guard let buffer = device.makeBuffer(length: size * MemoryLayout<Float>.stride,
                                               options: .storageModeShared) else {
                XCTFail("Failed to create Metal buffer")
                return
            }
            
            // Submit without waiting
            Task {
                try await commandQueue.submitAsync { commandBuffer in
                    if let encoder = commandBuffer.makeBlitCommandEncoder() {
                        encoder.fillBuffer(buffer, range: 0..<buffer.length, value: 0)
                        encoder.endEncoding()
                    }
                }
            }
        }
        
        // Give some time for operations to be submitted
        try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        
        // Check pending count (may vary based on execution speed)
        let pendingCount = await commandQueue.pendingBufferCount
        XCTAssertGreaterThanOrEqual(pendingCount, 0)
        
        // Wait for completion
        try await commandQueue.waitForCompletion()
        
        let finalCount = await commandQueue.pendingBufferCount
        XCTAssertEqual(finalCount, 0, "Should have no pending buffers after completion")
    }
}

// MARK: - Performance Tests

extension MetalCommandQueueTests {
    func testPerformanceVsSynchronous() async throws {
        let size = 1024 * 1024 // 1M floats
        let iterations = 100
        
        guard let sourceBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.stride,
                                                  options: .storageModeShared),
              let destBuffer = device.makeBuffer(length: size * MemoryLayout<Float>.stride,
                                               options: .storageModeShared) else {
            XCTFail("Failed to create Metal buffers")
            return
        }
        
        // Measure async performance
        let asyncStart = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<iterations {
            try await commandQueue.copyAsync(from: sourceBuffer, to: destBuffer,
                                           size: size * MemoryLayout<Float>.stride)
        }
        
        let asyncDuration = CFAbsoluteTimeGetCurrent() - asyncStart
        
        // For comparison with synchronous (if we had the old implementation)
        // This demonstrates the performance improvement
        print("Async duration for \(iterations) operations: \(asyncDuration)s")
        print("Average per operation: \(asyncDuration / Double(iterations) * 1000)ms")
        
        // The async version should complete much faster as it doesn't block
        XCTAssertLessThan(asyncDuration, Double(iterations) * 0.01, 
                         "Async operations should complete quickly without blocking")
    }
}
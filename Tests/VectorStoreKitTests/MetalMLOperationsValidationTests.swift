// VectorStoreKit: MetalMLOperations Validation Tests
//
// Tests for the validation extension to ensure numerical stability

import XCTest
import Metal
@testable import VectorStoreKit

@available(macOS 13.0, iOS 16.0, *)
final class MetalMLOperationsValidationTests: XCTestCase {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var shaderLibrary: MLShaderLibrary!
    var metalOps: MetalMLOperations!
    
    override func setUp() async throws {
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
        shaderLibrary = try await MLShaderLibrary(device: device)
        metalOps = await MetalMLOperations(
            device: device,
            commandQueue: commandQueue,
            shaderLibrary: shaderLibrary
        )
    }
    
    func testValidateBufferSizes_EmptyBuffers() async throws {
        // Create empty buffers
        let emptyBuffer = device.makeBuffer(length: 0)!
        let buffer1 = MetalBuffer(buffer: emptyBuffer, count: 0)
        
        let normalBuffer = device.makeBuffer(length: 16)!
        let buffer2 = MetalBuffer(buffer: normalBuffer, count: 4)
        
        // Test validation with empty first buffer
        do {
            try await metalOps.validateBufferSizes(buffer1, buffer2, operation: "test")
            XCTFail("Should have thrown error for empty buffer")
        } catch let error as MetalMLError {
            switch error {
            case .invalidBufferSize(let message):
                XCTAssertTrue(message.contains("Empty buffer"))
                XCTAssertTrue(message.contains("test"))
            default:
                XCTFail("Wrong error type: \(error)")
            }
        }
    }
    
    func testValidateBufferSizes_NaNDetection() async throws {
        // Create buffer with NaN values
        var data: [Float] = [1.0, 2.0, Float.nan, 4.0]
        let nanBuffer = device.makeBuffer(
            bytes: &data,
            length: data.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let buffer1 = MetalBuffer(buffer: nanBuffer, count: data.count)
        
        // Create normal buffer
        var normalData: [Float] = [1.0, 2.0, 3.0, 4.0]
        let normalBuffer = device.makeBuffer(
            bytes: &normalData,
            length: normalData.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let buffer2 = MetalBuffer(buffer: normalBuffer, count: normalData.count)
        
        // Test validation with NaN
        do {
            try await metalOps.validateBufferSizes(buffer1, buffer2, operation: "matmul")
            XCTFail("Should have thrown error for NaN")
        } catch let error as MetalMLError {
            switch error {
            case .numericalInstability(let message):
                XCTAssertTrue(message.contains("NaN detected"))
                XCTAssertTrue(message.contains("matmul"))
            default:
                XCTFail("Wrong error type: \(error)")
            }
        }
    }
    
    func testValidateBufferSizes_ValidBuffers() async throws {
        // Create valid buffers
        var data1: [Float] = [1.0, 2.0, 3.0, 4.0]
        let buffer1 = device.makeBuffer(
            bytes: &data1,
            length: data1.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let metalBuffer1 = MetalBuffer(buffer: buffer1, count: data1.count)
        
        var data2: [Float] = [5.0, 6.0, 7.0, 8.0]
        let buffer2 = device.makeBuffer(
            bytes: &data2,
            length: data2.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let metalBuffer2 = MetalBuffer(buffer: buffer2, count: data2.count)
        
        // Should not throw
        try await metalOps.validateBufferSizes(metalBuffer1, metalBuffer2, operation: "add")
    }
    
    func testContainsNaN() async throws {
        // Test buffer without NaN
        var normalData: [Float] = [1.0, 2.0, 3.0, 4.0]
        let normalBuffer = device.makeBuffer(
            bytes: &normalData,
            length: normalData.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let metalBuffer1 = MetalBuffer(buffer: normalBuffer, count: normalData.count)
        XCTAssertFalse(metalBuffer1.containsNaN())
        
        // Test buffer with NaN
        var nanData: [Float] = [1.0, Float.nan, 3.0, 4.0]
        let nanBuffer = device.makeBuffer(
            bytes: &nanData,
            length: nanData.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let metalBuffer2 = MetalBuffer(buffer: nanBuffer, count: nanData.count)
        XCTAssertTrue(metalBuffer2.containsNaN())
        
        // Test buffer with infinity (should not be detected as NaN)
        var infData: [Float] = [1.0, Float.infinity, 3.0, 4.0]
        let infBuffer = device.makeBuffer(
            bytes: &infData,
            length: infData.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        let metalBuffer3 = MetalBuffer(buffer: infBuffer, count: infData.count)
        XCTAssertFalse(metalBuffer3.containsNaN())
    }
}
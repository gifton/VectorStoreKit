import XCTest
import Metal
@testable import VectorStoreKit

final class MetalBufferOptimizationTests: XCTestCase {
    var device: MTLDevice!
    
    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal is not supported on this device")
    }
    
    func testOptimizedArrayInitializer() throws {
        // Given
        let testArray: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        // When
        let metalBuffer = try MetalBuffer(device: device, array: testArray)
        
        // Then
        XCTAssertEqual(metalBuffer.count, testArray.count)
        XCTAssertEqual(metalBuffer.stride, MemoryLayout<Float>.stride)
        XCTAssertEqual(metalBuffer.byteLength, testArray.count * MemoryLayout<Float>.stride)
        
        // Verify contents
        let contents = metalBuffer.toArray()
        XCTAssertEqual(contents, testArray)
    }
    
    func testOptimizedArrayWithShapeInitializer() throws {
        // Given
        let testArray: [Float] = Array(repeating: 0, count: 12)
        let shape = TensorShape(3, 4)
        
        // When
        let metalBuffer = try MetalBuffer(device: device, array: testArray, shape: shape)
        
        // Then
        XCTAssertEqual(metalBuffer.count, testArray.count)
        XCTAssertEqual(metalBuffer.shape, shape)
        XCTAssertEqual(metalBuffer.shape.count, testArray.count)
    }
    
    func testGenericTypeInitializer() throws {
        // Given
        let intArray: [Int32] = [10, 20, 30, 40, 50]
        
        // When
        let metalBuffer = try MetalBuffer(device: device, array: intArray)
        
        // Then
        XCTAssertEqual(metalBuffer.count, intArray.count)
        XCTAssertEqual(metalBuffer.stride, MemoryLayout<Int32>.stride)
        
        // Verify contents
        let contents = metalBuffer.toArray(as: Int32.self)
        XCTAssertEqual(contents, intArray)
    }
    
    func testUnsafeBufferPointerInitializer() throws {
        // Given
        let testArray: [Float] = [1.5, 2.5, 3.5, 4.5]
        
        try testArray.withUnsafeBufferPointer { pointer in
            // When
            let metalBuffer = try MetalBuffer(device: device, pointer: pointer)
            
            // Then
            XCTAssertEqual(metalBuffer.count, testArray.count)
            
            // Verify contents
            let contents = metalBuffer.toArray()
            XCTAssertEqual(contents, testArray)
        }
    }
    
    func testEmptyArrayHandling() throws {
        // Given
        let emptyArray: [Float] = []
        
        // When/Then
        XCTAssertThrowsError(try MetalBuffer(device: device, array: emptyArray)) { error in
            guard let metalError = error as? MetalMLError,
                  case .bufferAllocationFailed(let size) = metalError else {
                XCTFail("Expected MetalMLError.bufferAllocationFailed")
                return
            }
            XCTAssertEqual(size, 0)
        }
    }
    
    func testShapeMismatch() throws {
        // Given
        let testArray: [Float] = [1, 2, 3, 4]
        let wrongShape = TensorShape(2, 3) // Total count = 6, not 4
        
        // When/Then
        XCTAssertThrowsError(try MetalBuffer(device: device, array: testArray, shape: wrongShape)) { error in
            guard let metalError = error as? MetalMLError,
                  case .incompatibleBufferSize(let expected, let actual) = metalError else {
                XCTFail("Expected MetalMLError.incompatibleBufferSize")
                return
            }
            XCTAssertEqual(expected, 6)
            XCTAssertEqual(actual, 4)
        }
    }
    
    func testPerformanceComparison() throws {
        let size = 1_000_000
        let testArray = Array(repeating: Float.random(in: 0...1), count: size)
        
        // Measure optimized initialization
        measure {
            do {
                _ = try MetalBuffer(device: device, array: testArray)
            } catch {
                XCTFail("Failed to create MetalBuffer: \(error)")
            }
        }
    }
}
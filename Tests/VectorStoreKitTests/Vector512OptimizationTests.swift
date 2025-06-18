// Vector512OptimizationTests.swift
// VectorStoreKitTests
//
// Tests for Vector512 SIMD optimizations

import XCTest
@testable import VectorStoreKit
import simd

final class Vector512OptimizationTests: XCTestCase {
    
    // MARK: - SIMD32 Optimization Tests
    
    func testSIMD32BulkInitialization() {
        let values = (0..<32).map { Float($0) }
        
        // Create using optimized function
        let optimized = initializeSIMD32(from: values)
        
        // Verify all values
        for i in 0..<32 {
            XCTAssertEqual(optimized[i], Float(i), "Value at index \(i) should be \(i)")
        }
    }
    
    func testSIMD32RandomInitialization() {
        let range: ClosedRange<Float> = -1.0...1.0
        let vector = randomSIMD32(in: range)
        
        // Verify all values are within range
        for i in 0..<32 {
            XCTAssertTrue(range.contains(vector[i]), 
                         "Value at index \(i) (\(vector[i])) should be in range \(range)")
        }
    }
    
    func testSIMD32InitializationPerformance() {
        let values = (0..<32).map { Float($0) }
        
        // Measure optimized initialization
        measure {
            for _ in 0..<10000 {
                _ = initializeSIMD32(from: values)
            }
        }
    }
    
    // MARK: - Vector512 Bulk Memory Tests
    
    func testVector512BulkArrayInitialization() {
        let values = (0..<512).map { Float($0) }
        let vector = Vector512(values)
        
        // Verify initialization
        for i in 0..<512 {
            XCTAssertEqual(vector[i], Float(i), accuracy: 0.0001)
        }
    }
    
    func testVector512ToArrayOptimization() {
        let original = (0..<512).map { Float($0) }
        let vector = Vector512(original)
        let array = vector.toArray()
        
        // Verify conversion
        XCTAssertEqual(array.count, 512)
        for i in 0..<512 {
            XCTAssertEqual(array[i], original[i], accuracy: 0.0001)
        }
    }
    
    func testVector512DataConversion() {
        let original = (0..<512).map { Float($0) }
        let vector1 = Vector512(original)
        
        // Convert to Data and back
        let data = vector1.toData()
        XCTAssertEqual(data.count, 512 * MemoryLayout<Float>.size)
        
        guard let vector2 = Vector512(data: data) else {
            XCTFail("Failed to create vector from data")
            return
        }
        
        // Verify round trip
        for i in 0..<512 {
            XCTAssertEqual(vector1[i], vector2[i], accuracy: 0.0001)
        }
    }
    
    // MARK: - Multiple Accumulator Tests
    
    func testDotProductCorrectness() {
        // Test with known values
        let v1 = Vector512(repeating: 2.0)
        let v2 = Vector512(repeating: 3.0)
        
        let result = v1.dot(v2)
        let expected: Float = 512 * 2.0 * 3.0
        
        XCTAssertEqual(result, expected, accuracy: 0.001)
    }
    
    func testDotProductWithMixedValues() {
        let values1 = (0..<512).map { Float($0) / 512.0 }
        let values2 = (0..<512).map { Float(511 - $0) / 512.0 }
        
        let v1 = Vector512(values1)
        let v2 = Vector512(values2)
        
        let result = v1.dot(v2)
        
        // Calculate expected value
        var expected: Float = 0
        for i in 0..<512 {
            expected += values1[i] * values2[i]
        }
        
        XCTAssertEqual(result, expected, accuracy: 0.001)
    }
    
    func testDistanceSquaredCorrectness() {
        let v1 = Vector512(repeating: 1.0)
        let v2 = Vector512(repeating: 3.0)
        
        let result = v1.distanceSquared(to: v2)
        let expected: Float = 512 * (3.0 - 1.0) * (3.0 - 1.0)
        
        XCTAssertEqual(result, expected, accuracy: 0.001)
    }
    
    // MARK: - Arithmetic Operation Tests
    
    func testOptimizedAddition() {
        let values1 = (0..<512).map { Float($0) }
        let values2 = (0..<512).map { Float($0 * 2) }
        
        let v1 = Vector512(values1)
        let v2 = Vector512(values2)
        let result = v1 + v2
        
        for i in 0..<512 {
            XCTAssertEqual(result[i], Float(i + i * 2), accuracy: 0.0001)
        }
    }
    
    func testOptimizedSubtraction() {
        let values1 = (0..<512).map { Float($0 * 3) }
        let values2 = (0..<512).map { Float($0) }
        
        let v1 = Vector512(values1)
        let v2 = Vector512(values2)
        let result = v1 - v2
        
        for i in 0..<512 {
            XCTAssertEqual(result[i], Float(i * 3 - i), accuracy: 0.0001)
        }
    }
    
    func testOptimizedScalarMultiplication() {
        let values = (0..<512).map { Float($0) }
        let vector = Vector512(values)
        let scalar: Float = 2.5
        
        let result = vector * scalar
        
        for i in 0..<512 {
            XCTAssertEqual(result[i], Float(i) * scalar, accuracy: 0.0001)
        }
    }
    
    // MARK: - Batch Operation Tests
    
    func testBatchVectorCreation() {
        let vectorCount = 10
        let flatArray = (0..<(vectorCount * 512)).map { Float($0) }
        
        let vectors = Vector512.createBatch(from: flatArray)
        
        XCTAssertEqual(vectors.count, vectorCount)
        
        // Verify each vector
        for v in 0..<vectorCount {
            for i in 0..<512 {
                let expectedValue = Float(v * 512 + i)
                XCTAssertEqual(vectors[v][i], expectedValue, accuracy: 0.0001)
            }
        }
    }
    
    // MARK: - Performance Tests
    
    func testDotProductPerformanceOptimized() {
        let v1 = Vector512(repeating: 1.0)
        let v2 = Vector512(repeating: 2.0)
        
        measure {
            var sum: Float = 0
            for _ in 0..<10000 {
                sum += v1.dot(v2)
            }
            // Prevent optimization
            XCTAssertGreaterThan(sum, 0)
        }
    }
    
    func testDistanceSquaredPerformanceOptimized() {
        let v1 = Vector512(repeating: 1.0)
        let v2 = Vector512(repeating: 2.0)
        
        measure {
            var sum: Float = 0
            for _ in 0..<10000 {
                sum += v1.distanceSquared(to: v2)
            }
            // Prevent optimization
            XCTAssertGreaterThan(sum, 0)
        }
    }
    
    func testBulkInitializationPerformance() {
        let values = (0..<512).map { Float($0) }
        
        measure {
            for _ in 0..<1000 {
                _ = Vector512(values)
            }
        }
    }
    
    func testToArrayPerformance() {
        let vector = Vector512(repeating: 1.0)
        
        measure {
            for _ in 0..<1000 {
                _ = vector.toArray()
            }
        }
    }
    
    // MARK: - Edge Case Tests
    
    func testEmptyDataInitialization() {
        let data = Data()
        let vector = Vector512(data: data)
        XCTAssertNil(vector)
    }
    
    func testIncorrectDataSizeInitialization() {
        let data = Data(repeating: 0, count: 100) // Wrong size
        let vector = Vector512(data: data)
        XCTAssertNil(vector)
    }
    
    func testLargeValueHandling() {
        var vector = Vector512()
        vector[0] = .infinity
        vector[1] = -.infinity
        vector[2] = .nan
        
        // Test normalization handles special values
        vector.normalize()
        XCTAssertFalse(vector[0].isNaN)
    }
    
    // MARK: - Correctness Verification
    
    func testOptimizationCorrectness() {
        // This test verifies that optimizations don't change behavior
        let values = (0..<512).map { _ in Float.random(in: -1...1) }
        
        // Create two vectors with same data
        let v1 = Vector512(values)
        let v2 = Vector512(values)
        
        // Test all operations produce same results
        XCTAssertEqual(v1.dot(v2), v2.dot(v1), accuracy: 0.0001)
        XCTAssertEqual(v1.distanceSquared(to: v2), 0.0, accuracy: 0.0001)
        
        let v3 = v1 + v2
        let v4 = v2 + v1
        for i in 0..<512 {
            XCTAssertEqual(v3[i], v4[i], accuracy: 0.0001)
        }
    }
}
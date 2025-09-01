// Vector512Tests.swift
// VectorStoreKitTests
//
// Tests for optimized 512-dimensional vector operations

import XCTest
@testable import VectorStoreKit
import simd

final class Vector512Tests: XCTestCase {
    
    // MARK: - Vector512 Type Tests
    
    func testVector512Initialization() {
        // Test default initialization
        let v1 = Vector512()
        XCTAssertEqual(v1.scalarCount, 512)
        XCTAssertEqual(v1[0], 0)
        XCTAssertEqual(v1[511], 0)
        
        // Test initialization from array
        let values = (0..<512).map { Float($0) }
        let v2 = Vector512(values)
        XCTAssertEqual(v2[0], 0)
        XCTAssertEqual(v2[511], 511)
        
        // Test repeating initialization
        let v3 = Vector512(repeating: 3.14)
        XCTAssertEqual(v3[0], 3.14)
        XCTAssertEqual(v3[255], 3.14)
        XCTAssertEqual(v3[511], 3.14)
    }
    
    func testVector512Subscript() {
        var vector = Vector512()
        
        // Test write
        vector[0] = 1.0
        vector[255] = 2.0
        vector[511] = 3.0
        
        // Test read
        XCTAssertEqual(vector[0], 1.0)
        XCTAssertEqual(vector[255], 2.0)
        XCTAssertEqual(vector[511], 3.0)
    }
    
    func testVector512MemoryAlignment() {
        // Test aligned allocation
        let count = 10
        let aligned = Vector512.allocateAligned(count: count)
        defer { aligned.deallocate() }
        
        // Check alignment (should be 64-byte aligned)
        let address = Int(bitPattern: aligned)
        XCTAssertEqual(address % 64, 0, "Memory should be 64-byte aligned")
    }
    
    // MARK: - SIMD Operation Tests
    
    func testVector512DotProduct() {
        let v1 = Vector512(repeating: 1.0)
        let v2 = Vector512(repeating: 2.0)
        
        let dot = v1.dot(v2)
        XCTAssertEqual(dot, 1024.0, accuracy: 0.001) // 512 * 1 * 2
    }
    
    func testVector512EuclideanDistance() {
        let v1 = Vector512(repeating: 1.0)
        let v2 = Vector512(repeating: 3.0)
        
        let distSquared = v1.distanceSquared(to: v2)
        XCTAssertEqual(distSquared, 2048.0, accuracy: 0.001) // 512 * (3-1)^2
        
        let dist = sqrt(distSquared)
        XCTAssertEqual(dist, sqrt(2048.0), accuracy: 0.001)
    }
    
    func testVector512CosineSimilarity() {
        // Test with normalized vectors
        var v1 = Vector512(repeating: 1.0)
        v1.normalize()
        
        var v2 = Vector512(repeating: 1.0)
        v2.normalize()
        
        let similarity = v1.cosineSimilarity(to: v2, normalized: true)
        XCTAssertEqual(similarity, 1.0, accuracy: 0.001)
        
        // Test with orthogonal vectors
        var v3 = Vector512()
        v3[0] = 1.0
        v3.normalize()
        
        var v4 = Vector512()
        v4[1] = 1.0
        v4.normalize()
        
        let orthogonalSimilarity = v3.cosineSimilarity(to: v4, normalized: true)
        XCTAssertEqual(orthogonalSimilarity, 0.0, accuracy: 0.001)
    }
    
    func testVector512Normalization() {
        var vector = Vector512(repeating: 2.0)
        vector.normalize()
        
        // Check magnitude is 1
        let magnitude = sqrt(vector.dot(vector))
        XCTAssertEqual(magnitude, 1.0, accuracy: 0.001)
    }
    
    // MARK: - Arithmetic Operation Tests
    
    func testVector512Addition() {
        let v1 = Vector512(repeating: 1.0)
        let v2 = Vector512(repeating: 2.0)
        
        let result = v1 + v2
        XCTAssertEqual(result[0], 3.0)
        XCTAssertEqual(result[255], 3.0)
        XCTAssertEqual(result[511], 3.0)
    }
    
    func testVector512Subtraction() {
        let v1 = Vector512(repeating: 5.0)
        let v2 = Vector512(repeating: 2.0)
        
        let result = v1 - v2
        XCTAssertEqual(result[0], 3.0)
        XCTAssertEqual(result[511], 3.0)
    }
    
    func testVector512ScalarMultiplication() {
        let vector = Vector512(repeating: 2.0)
        let result = vector * 3.0
        
        XCTAssertEqual(result[0], 6.0)
        XCTAssertEqual(result[511], 6.0)
    }
    
    // MARK: - Distance Computation Tests
    
    func testDistanceComputation512Euclidean() {
        let v1 = Vector512(repeating: 0.0)
        let v2 = Vector512(repeating: 1.0)
        
        let distance = DistanceComputation512.euclideanDistance(v1, v2)
        let expected = sqrt(512.0) // sqrt(sum of 512 ones)
        XCTAssertEqual(distance, expected, accuracy: 0.001)
    }
    
    func testDistanceComputation512Manhattan() {
        let v1 = Vector512(repeating: 0.0)
        let v2 = Vector512(repeating: 1.0)
        
        let distance = DistanceComputation512.manhattanDistance(v1, v2)
        XCTAssertEqual(distance, 512.0, accuracy: 0.001)
    }
    
    func testDistanceComputation512DotProduct() {
        let values1 = (0..<512).map { Float($0) / 512.0 }
        let values2 = (0..<512).map { Float(511 - $0) / 512.0 }
        
        let v1 = Vector512(values1)
        let v2 = Vector512(values2)
        
        let dotProduct = DistanceComputation512.dotProduct(v1, v2)
        XCTAssertGreaterThan(dotProduct, 0)
    }
    
    // MARK: - Batch Operation Tests
    
    func testBatchEuclideanDistance() {
        let query = Vector512(repeating: 0.0)
        let candidates = (0..<100).map { i in
            Vector512(repeating: Float(i))
        }
        
        let distances = DistanceComputation512.batchEuclideanDistance(
            query: query,
            candidates: candidates
        )
        
        XCTAssertEqual(distances.count, 100)
        XCTAssertEqual(distances[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(distances[1], sqrt(512.0), accuracy: 0.001)
    }
    
    func testBatchEuclideanDistanceAccelerate() {
        let query = Vector512(repeating: 0.0)
        let candidates = (0..<100).map { i in
            Vector512(repeating: Float(i))
        }
        
        let distances = DistanceComputation512.batchEuclideanDistanceAccelerate(
            query: query,
            candidates: candidates
        )
        
        XCTAssertEqual(distances.count, 100)
        XCTAssertEqual(distances[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(distances[1], sqrt(512.0), accuracy: 0.001)
    }
    
    // MARK: - Performance Tests
    
    func testVector512PerformanceDotProduct() {
        let v1 = Vector512(repeating: 1.0)
        let v2 = Vector512(repeating: 2.0)
        
        measure {
            for _ in 0..<10000 {
                _ = v1.dot(v2)
            }
        }
    }
    
    func testVector512PerformanceEuclideanDistance() {
        let v1 = Vector512(repeating: 1.0)
        let v2 = Vector512(repeating: 2.0)
        
        measure {
            for _ in 0..<10000 {
                _ = v1.distanceSquared(to: v2)
            }
        }
    }
    
    func testBatchDistancePerformanceComparison() {
        let query = Vector512(repeating: 0.0)
        let candidates = (0..<1000).map { i in
            Vector512(repeating: Float(i % 10))
        }
        
        // Measure SIMD performance
        let simdTime = measure {
            _ = DistanceComputation512.batchEuclideanDistance(
                query: query,
                candidates: candidates
            )
        }
        
        // Measure Accelerate performance
        let accelerateTime = measure {
            _ = DistanceComputation512.batchEuclideanDistanceAccelerate(
                query: query,
                candidates: candidates
            )
        }
        
        print("SIMD time: \(simdTime)")
        print("Accelerate time: \(accelerateTime)")
    }
    
    // MARK: - Metal Integration Tests
    
    func testVector512MetalBufferCreation() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTSkip("Metal not available")
            return
        }
        
        let vector = Vector512(repeating: 3.14)
        let buffer = vector.makeMetalBuffer(device: device)
        
        XCTAssertNotNil(buffer)
        XCTAssertEqual(buffer?.length, 512 * MemoryLayout<Float>.size)
    }
    
    // MARK: - Configuration Tests
    
    func testVectorDimensionConfiguration() {
        let fixed512 = VectorDimension.fixed512
        XCTAssertEqual(fixed512.dimension, 512)
        XCTAssertTrue(fixed512.isOptimized)
        XCTAssertEqual(fixed512.optimalBatchSize, 2048)
        
        let variable = VectorDimension.variable(768)
        XCTAssertEqual(variable.dimension, 768)
        XCTAssertFalse(variable.isOptimized)
    }
    
    func testDistanceComputeBackendSelection() {
        let backend = DistanceComputeBackend.auto
        
        // Large batch should select Metal
        let metalBackend = backend.selectOptimal(for: .fixed512, candidateCount: 2000)
        XCTAssertEqual(metalBackend, .metal)
        
        // Small batch with 512-dim should select SIMD
        let simdBackend = backend.selectOptimal(for: .fixed512, candidateCount: 100)
        XCTAssertEqual(simdBackend, .simd)
        
        // Variable dimension should select Accelerate
        let accelerateBackend = backend.selectOptimal(for: .variable(768), candidateCount: 100)
        XCTAssertEqual(accelerateBackend, .accelerate)
    }
    
    func testStoreConfiguration512() {
        let config = StoreConfiguration.optimized512
        XCTAssertEqual(config.vectorDimension, .fixed512)
        XCTAssertEqual(config.distanceComputeBackend, .auto)
    }
    
    // MARK: - Edge Case Tests
    
    func testVector512WithSpecialValues() {
        var vector = Vector512()
        
        // Test with infinity
        vector[0] = .infinity
        vector[1] = -.infinity
        
        // Test with NaN
        vector[2] = .nan
        
        // Normalization should handle special values
        vector.normalize()
        XCTAssertFalse(vector[0].isNaN)
    }
    
    func testEmptyCandidatesHandling() {
        let query = Vector512(repeating: 1.0)
        let emptyCandidates: [Vector512] = []
        
        let distances = DistanceComputation512.batchEuclideanDistance(
            query: query,
            candidates: emptyCandidates
        )
        
        XCTAssertTrue(distances.isEmpty)
    }
}

// MARK: - Test Helpers

extension XCTestCase {
    func measure(block: () -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        block()
        return CFAbsoluteTimeGetCurrent() - start
    }
}
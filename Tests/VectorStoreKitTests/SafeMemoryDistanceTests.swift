// SafeMemoryDistanceTests.swift
// VectorStoreKit
//
// Tests for safe memory management in distance computations

import XCTest
@testable import VectorStoreKit

final class SafeMemoryDistanceTests: XCTestCase {
    
    // MARK: - AlignedVector Tests
    
    func testAlignedVectorInitialization() throws {
        // Test valid initialization
        let vector = try AlignedVector<Float>(count: 128)
        XCTAssertEqual(vector.count, 128)
        XCTAssertTrue(vector.isProperlyAligned)
        
        // Test zero initialization
        vector.withUnsafeBufferPointer { buffer in
            for i in 0..<128 {
                XCTAssertEqual(buffer[i], 0.0)
            }
        }
    }
    
    func testAlignedVectorInvalidInitialization() {
        // Test invalid count
        XCTAssertThrowsError(try AlignedVector<Float>(count: 0)) { error in
            guard let vectorError = error as? VectorStoreError else {
                XCTFail("Expected VectorStoreError")
                return
            }
            XCTAssertEqual(vectorError.category, .validation)
            XCTAssertEqual(vectorError.code, .invalidInput)
        }
        
        XCTAssertThrowsError(try AlignedVector<Float>(count: -10)) { error in
            guard let vectorError = error as? VectorStoreError else {
                XCTFail("Expected VectorStoreError")
                return
            }
            XCTAssertEqual(vectorError.category, .validation)
            XCTAssertEqual(vectorError.code, .invalidInput)
        }
    }
    
    func testAlignedVectorBoundsChecking() throws {
        let vector = try AlignedVector<Float>(count: 10)
        
        // Valid access
        vector[0] = 1.0
        vector[9] = 2.0
        XCTAssertEqual(vector[0], 1.0)
        XCTAssertEqual(vector[9], 2.0)
        
        // In debug mode, this would trigger a fatal error
        // We can't test this directly without crashing the test
    }
    
    func testAlignedVectorMutableAccess() throws {
        var vector = try AlignedVector<Float>(count: 64)
        
        // Fill with test data
        vector.withUnsafeMutableBufferPointer { buffer in
            for i in 0..<64 {
                buffer[i] = Float(i)
            }
        }
        
        // Verify data
        vector.withUnsafeBufferPointer { buffer in
            for i in 0..<64 {
                XCTAssertEqual(buffer[i], Float(i))
            }
        }
    }
    
    // MARK: - SafeVectorBuffer Tests
    
    func testSafeVectorBuffer() {
        let buffer = SafeVectorBuffer<Float>(count: 256)
        XCTAssertEqual(buffer.count, 256)
        
        // Test initialization
        for i in 0..<256 {
            XCTAssertEqual(buffer[i], 0.0)
        }
        
        // Test mutation
        for i in 0..<256 {
            buffer[i] = Float(i) * 0.5
        }
        
        // Verify
        for i in 0..<256 {
            XCTAssertEqual(buffer[i], Float(i) * 0.5)
        }
    }
    
    func testSafeVectorBufferCopy() {
        let source = (0..<128).map { Float($0) }
        let buffer = SafeVectorBuffer(copying: source)
        
        XCTAssertEqual(buffer.count, 128)
        
        for i in 0..<128 {
            XCTAssertEqual(buffer[i], Float(i))
        }
    }
    
    // MARK: - Safe Distance Computation Tests
    
    func testSafeEuclideanDistance() {
        let a = [Float](repeating: 1.0, count: 128)
        let b = [Float](repeating: 2.0, count: 128)
        
        let distance = OptimizedEuclideanDistance.safeDistanceSquared(a, b)
        XCTAssertNotNil(distance)
        XCTAssertEqual(distance!, 128.0, accuracy: 0.0001)
    }
    
    func testSafeEuclideanDistanceMismatch() {
        let a = [Float](repeating: 1.0, count: 128)
        let b = [Float](repeating: 2.0, count: 64)
        
        let distance = OptimizedEuclideanDistance.safeDistanceSquared(a, b)
        XCTAssertNil(distance)
    }
    
    func testSafeBatchDistance() {
        let query = [Float](repeating: 0.5, count: 64)
        let candidates = (0..<10).map { i in
            [Float](repeating: Float(i) * 0.1, count: 64)
        }
        
        let distances = OptimizedEuclideanDistance.safeBatchDistanceSquared(
            query: query,
            candidates: candidates
        )
        
        XCTAssertNotNil(distances)
        XCTAssertEqual(distances?.count, 10)
        
        // Verify distances are increasing
        if let distances = distances {
            for i in 1..<distances.count {
                XCTAssertGreaterThan(distances[i], distances[i-1])
            }
        }
    }
    
    func testSafeCosineDistance() {
        let a = [Float](repeating: 1.0, count: 128)
        let b = [Float](repeating: 1.0, count: 128)
        
        let similarity = OptimizedCosineDistance.safeNormalizedSimilarity(a, b)
        XCTAssertNotNil(similarity)
        XCTAssertEqual(similarity!, 128.0, accuracy: 0.0001)
    }
    
    // MARK: - Buffer Pool Tests
    
    func testBufferPoolBasicOperations() async {
        let pool = SafeDistanceComputationBufferPool(bufferSize: 512, maxBuffers: 10)
        
        // Test acquiring buffers
        let buffer1 = await pool.acquireBuffer()
        XCTAssertEqual(buffer1.count, 512)
        
        let buffer2 = await pool.acquireBuffer()
        XCTAssertEqual(buffer2.count, 512)
        
        // Test releasing buffers
        await pool.releaseBuffer(buffer1)
        await pool.releaseBuffer(buffer2)
        
        // Check statistics
        let stats = await pool.statistics
        XCTAssertGreaterThan(stats.allocated, 0)
        XCTAssertLessThanOrEqual(stats.allocated, stats.maxSize)
    }
    
    func testBufferPoolMemoryPressure() async {
        let pool = SafeDistanceComputationBufferPool(bufferSize: 1024, maxBuffers: 20)
        
        // Pre-allocate some buffers
        var buffers: [SafeVectorBuffer<Float>] = []
        for _ in 0..<5 {
            buffers.append(await pool.acquireBuffer())
        }
        
        // Release them back to pool
        for buffer in buffers {
            await pool.releaseBuffer(buffer)
        }
        
        let initialStats = await pool.statistics
        XCTAssertGreaterThan(initialStats.available, 0)
        
        // Simulate memory pressure warning
        pool.handleMemoryPressure(level: .warning)
        await Task.sleep(100_000_000) // 0.1 seconds
        
        let warningStats = await pool.statistics
        XCTAssertLessThan(warningStats.available, initialStats.available)
        
        // Simulate critical memory pressure
        pool.handleMemoryPressure(level: .critical)
        await Task.sleep(100_000_000) // 0.1 seconds
        
        let criticalStats = await pool.statistics
        XCTAssertEqual(criticalStats.available, 0)
        XCTAssertEqual(criticalStats.allocated, 0)
    }
    
    func testSafeBatchComputation() async throws {
        let pool = SafeDistanceComputationBufferPool(bufferSize: 1024, maxBuffers: 10)
        let computer = SafeBatchDistanceComputation(bufferPool: pool)
        
        let query = [Float](repeating: 0.5, count: 128)
        let candidates = (0..<20).map { i in
            [Float](repeating: Float(i) * 0.05, count: 128)
        }
        
        let distances = try await computer.computeDistances(
            query: query,
            candidates: candidates,
            metric: .euclidean
        )
        
        XCTAssertEqual(distances.count, 20)
        
        // Verify distances are reasonable
        for distance in distances {
            XCTAssertGreaterThanOrEqual(distance, 0)
            XCTAssertFalse(distance.isNaN)
            XCTAssertFalse(distance.isInfinite)
        }
    }
    
    // MARK: - Performance Tests
    
    func testAlignedVectorPerformance() throws {
        measure {
            do {
                let vector = try AlignedVector<Float>(count: 1024 * 1024) // 1M elements
                
                // Write
                vector.withUnsafeMutableBufferPointer { buffer in
                    for i in 0..<buffer.count {
                        buffer[i] = Float(i)
                    }
                }
                
                // Read
                var sum: Float = 0
                vector.withUnsafeBufferPointer { buffer in
                    sum = buffer.reduce(0, +)
                }
                
                XCTAssertGreaterThan(sum, 0)
            } catch {
                XCTFail("Failed to create aligned vector: \(error)")
            }
        }
    }
    
    func testSafeDistancePerformance() {
        let a = [Float](repeating: 1.0, count: 1024)
        let b = [Float](repeating: 2.0, count: 1024)
        
        measure {
            for _ in 0..<1000 {
                _ = OptimizedEuclideanDistance.safeDistanceSquared(a, b)
            }
        }
    }
}
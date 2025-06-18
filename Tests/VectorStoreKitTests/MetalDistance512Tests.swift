// MetalDistance512Tests.swift
// VectorStoreKitTests
//
// Tests for Metal-accelerated 512-dimensional distance computation

import XCTest
@testable import VectorStoreKit
import Metal

final class MetalDistance512Tests: XCTestCase {
    
    var device: MetalDevice!
    var bufferPool: MetalBufferPool!
    var pipelineManager: MetalPipelineManager!
    var distanceCompute: MetalDistanceCompute!
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Setup Metal components
        guard let mtlDevice = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        
        device = try await MetalDevice(device: mtlDevice)
        
        let poolConfig = MetalBufferPoolConfiguration(
            initialSize: 100,
            maxSize: 1000,
            growthFactor: 2.0
        )
        bufferPool = try await MetalBufferPool(device: device, configuration: poolConfig)
        
        pipelineManager = try await MetalPipelineManager(device: device)
        
        distanceCompute = MetalDistanceCompute(
            device: device,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager
        )
    }
    
    override func tearDown() async throws {
        await bufferPool.shutdown()
        try await super.tearDown()
    }
    
    // MARK: - 512-Dimensional Optimized Tests
    
    func testOptimized512EuclideanDistance() async throws {
        let query = Vector512(repeating: 0.0)
        let candidates = (0..<100).map { i in
            Vector512(repeating: Float(i))
        }
        
        let distances = try await distanceCompute.computeDistances512(
            query: query,
            candidates: candidates,
            metric: .euclidean
        )
        
        XCTAssertEqual(distances.count, 100)
        XCTAssertEqual(distances[0], 0.0, accuracy: 0.001)
        XCTAssertEqual(distances[1], sqrt(512.0), accuracy: 0.001)
        XCTAssertEqual(distances[2], sqrt(512.0 * 4), accuracy: 0.001)
    }
    
    func testOptimized512CosineDistance() async throws {
        // Create normalized vectors
        var query = Vector512(repeating: 1.0)
        query.normalize()
        
        var same = Vector512(repeating: 1.0)
        same.normalize()
        
        var different = Vector512()
        different[0] = 1.0
        different.normalize()
        
        let candidates = [same, different]
        
        let distances = try await distanceCompute.computeDistances512(
            query: query,
            candidates: candidates,
            metric: .cosine,
            normalized: true
        )
        
        XCTAssertEqual(distances.count, 2)
        XCTAssertEqual(distances[0], 0.0, accuracy: 0.001) // Same vector
        XCTAssertGreaterThan(distances[1], 0.9) // Very different
    }
    
    func testOptimized512LargeBatch() async throws {
        let query = Vector512(repeating: 0.0)
        let candidates = (0..<10000).map { i in
            Vector512(repeating: Float(i % 100) / 100.0)
        }
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let distances = try await distanceCompute.computeDistances512(
            query: query,
            candidates: candidates,
            metric: .euclidean
        )
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        XCTAssertEqual(distances.count, 10000)
        print("512-dim optimized computation for 10K vectors took \(duration)s")
        
        // Verify some results
        for i in 0..<100 {
            let expected = sqrt(512.0 * pow(Float(i) / 100.0, 2))
            XCTAssertEqual(distances[i], expected, accuracy: 0.01)
        }
    }
    
    // MARK: - Generic vs Optimized Comparison
    
    func testGenericVsOptimized512Performance() async throws {
        let query = Vector512(repeating: 0.0)
        let candidates = (0..<1000).map { i in
            Vector512(repeating: Float(i % 10))
        }
        
        // Time generic implementation
        let genericStart = CFAbsoluteTimeGetCurrent()
        let genericDistances = try await distanceCompute.computeDistances(
            query: query,
            candidates: candidates,
            metric: .euclidean
        )
        let genericTime = CFAbsoluteTimeGetCurrent() - genericStart
        
        // Time optimized implementation
        let optimizedStart = CFAbsoluteTimeGetCurrent()
        let optimizedDistances = try await distanceCompute.computeDistances512(
            query: query,
            candidates: candidates,
            metric: .euclidean
        )
        let optimizedTime = CFAbsoluteTimeGetCurrent() - optimizedStart
        
        print("Generic time: \(genericTime)s")
        print("Optimized 512 time: \(optimizedTime)s")
        print("Speedup: \(genericTime / optimizedTime)x")
        
        // Results should be identical
        for i in 0..<genericDistances.count {
            XCTAssertEqual(genericDistances[i], optimizedDistances[i], accuracy: 0.001)
        }
    }
    
    // MARK: - Batch Operations
    
    func testBatch512DistanceComputation() async throws {
        let queries = (0..<10).map { i in
            Vector512(repeating: Float(i))
        }
        
        let candidates = (0..<100).map { i in
            Vector512(repeating: Float(i % 10))
        }
        
        // Use optimized batch computation
        let distanceMatrix = try await distanceCompute.batchComputeDistances(
            queries: queries,
            candidates: candidates,
            metric: .euclidean
        )
        
        XCTAssertEqual(distanceMatrix.count, 10)
        XCTAssertEqual(distanceMatrix[0].count, 100)
        
        // Verify some results
        for i in 0..<10 {
            for j in 0..<10 {
                let expected = sqrt(512.0 * pow(Float(i) - Float(j), 2))
                XCTAssertEqual(distanceMatrix[i][j], expected, accuracy: 0.01)
            }
        }
    }
    
    // MARK: - Shader Verification
    
    func testOptimized512ShaderExists() async throws {
        // Verify that optimized shaders are available
        let euclideanPipeline = try? await pipelineManager.getPipeline(functionName: "euclideanDistance512_simd")
        XCTAssertNotNil(euclideanPipeline, "Optimized Euclidean shader should exist")
        
        let cosinePipeline = try? await pipelineManager.getPipeline(functionName: "cosineDistance512_simd")
        XCTAssertNotNil(cosinePipeline, "Optimized Cosine shader should exist")
        
        let normalizedCosinePipeline = try? await pipelineManager.getPipeline(functionName: "cosineDistance512_normalized")
        XCTAssertNotNil(normalizedCosinePipeline, "Optimized normalized Cosine shader should exist")
    }
    
    // MARK: - Memory and Buffer Tests
    
    func testOptimized512BufferAlignment() async throws {
        let vector = Vector512(repeating: 3.14)
        let data = vector.withUnsafeMetalBytes { Data($0) }
        let buffer = try await bufferPool.getBuffer(data: data)
        
        defer {
            Task { await bufferPool.returnBuffer(buffer) }
        }
        
        // Check buffer size
        XCTAssertEqual(buffer.length, 512 * MemoryLayout<Float>.size)
        
        // Verify data integrity
        let contents = buffer.contents().bindMemory(to: Float.self, capacity: 512)
        for i in 0..<512 {
            XCTAssertEqual(contents[i], 3.14, accuracy: 0.001)
        }
    }
    
    // MARK: - Error Handling
    
    func testEmptyCandidatesError() async throws {
        let query = Vector512(repeating: 1.0)
        let emptyCandidates: [Vector512] = []
        
        do {
            _ = try await distanceCompute.computeDistances512(
                query: query,
                candidates: emptyCandidates,
                metric: .euclidean
            )
            XCTFail("Should throw error for empty candidates")
        } catch {
            // Expected error
            XCTAssertTrue(error is MetalComputeError)
        }
    }
    
    // MARK: - Performance Benchmarks
    
    func testPerformanceBenchmark512Euclidean() async throws {
        let query = Vector512(repeating: 0.0)
        let candidates = (0..<10000).map { i in
            Vector512(repeating: Float(i % 100) / 100.0)
        }
        
        measure {
            let expectation = self.expectation(description: "Distance computation")
            
            Task {
                _ = try await distanceCompute.computeDistances512(
                    query: query,
                    candidates: candidates,
                    metric: .euclidean
                )
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10.0)
        }
    }
    
    func testPerformanceBenchmark512Cosine() async throws {
        var query = Vector512(repeating: 1.0)
        query.normalize()
        
        let candidates = (0..<10000).map { i in
            var v = Vector512(repeating: Float(i % 100) / 100.0)
            v.normalize()
            return v
        }
        
        measure {
            let expectation = self.expectation(description: "Cosine computation")
            
            Task {
                _ = try await distanceCompute.computeDistances512(
                    query: query,
                    candidates: candidates,
                    metric: .cosine,
                    normalized: true
                )
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10.0)
        }
    }
}

// MARK: - Test Utilities

extension MetalDistance512Tests {
    
    func createRandomVector512() -> Vector512 {
        let values = (0..<512).map { _ in Float.random(in: -1...1) }
        return Vector512(values)
    }
    
    func createNormalizedRandomVector512() -> Vector512 {
        var vector = createRandomVector512()
        vector.normalize()
        return vector
    }
    
    func assertDistancesEqual(_ actual: [Float], _ expected: [Float], accuracy: Float = 0.001) {
        XCTAssertEqual(actual.count, expected.count)
        for i in 0..<actual.count {
            XCTAssertEqual(actual[i], expected[i], accuracy: accuracy, "Distance at index \(i) differs")
        }
    }
}
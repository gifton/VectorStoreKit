// Metal Distance Matrix Tests
//
// Tests for GPU-accelerated distance matrix computation
//

import XCTest
@testable import VectorStoreKit

final class MetalDistanceMatrixTests: XCTestCase {
    
    var metalMatrix: MetalDistanceMatrix!
    
    override func setUp() async throws {
        try await super.setUp()
        
        let device = try MetalDevice()
        let bufferPool = MetalBufferPool(device: device.device)
        let pipelineManager = try await MetalPipelineManager(device: device)
        
        metalMatrix = MetalDistanceMatrix(
            device: device,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager
        )
    }
    
    func testSymmetricDistanceMatrix() async throws {
        // Generate test vectors
        let vectors = (0..<100).map { i in
            Vector512(repeating: Float(i))
        }
        
        // Compute distance matrix
        let matrix = try await metalMatrix.computeDistanceMatrix(
            vectorsA: vectors,
            vectorsB: nil,
            metric: .euclidean
        )
        
        // Verify matrix properties
        XCTAssertEqual(matrix.count, vectors.count)
        XCTAssertEqual(matrix[0].count, vectors.count)
        
        // Check symmetry
        for i in 0..<vectors.count {
            for j in 0..<vectors.count {
                XCTAssertEqual(matrix[i][j], matrix[j][i], accuracy: 0.0001)
            }
        }
        
        // Check diagonal is zero
        for i in 0..<vectors.count {
            XCTAssertEqual(matrix[i][i], 0.0, accuracy: 0.0001)
        }
    }
    
    func testNonSymmetricDistanceMatrix() async throws {
        let vectorsA = (0..<50).map { i in
            Vector512(repeating: Float(i))
        }
        
        let vectorsB = (0..<75).map { i in
            Vector512(repeating: Float(i * 2))
        }
        
        let matrix = try await metalMatrix.computeDistanceMatrix(
            vectorsA: vectorsA,
            vectorsB: vectorsB,
            metric: .euclidean
        )
        
        XCTAssertEqual(matrix.count, vectorsA.count)
        XCTAssertEqual(matrix[0].count, vectorsB.count)
    }
    
    func testDifferentMetrics() async throws {
        let vectors = [
            Vector512(repeating: 1.0),
            Vector512(repeating: 2.0),
            Vector512(repeating: 3.0)
        ]
        
        // Test Euclidean
        let euclideanMatrix = try await metalMatrix.computeDistanceMatrix(
            vectorsA: vectors,
            vectorsB: nil,
            metric: .euclidean
        )
        
        // Expected: sqrt(512 * (2-1)^2) = sqrt(512) ≈ 22.627
        XCTAssertEqual(euclideanMatrix[0][1], sqrt(512), accuracy: 0.01)
        
        // Test Manhattan
        let manhattanMatrix = try await metalMatrix.computeDistanceMatrix(
            vectorsA: vectors,
            vectorsB: nil,
            metric: .manhattan
        )
        
        // Expected: 512 * |2-1| = 512
        XCTAssertEqual(manhattanMatrix[0][1], 512, accuracy: 0.01)
        
        // Test Cosine (all same direction, so cosine similarity = 1, distance = 0)
        let cosineMatrix = try await metalMatrix.computeDistanceMatrix(
            vectorsA: vectors,
            vectorsB: nil,
            metric: .cosine
        )
        
        XCTAssertEqual(cosineMatrix[0][1], 0.0, accuracy: 0.01)
    }
    
    func testGPUvsCPUAccuracy() async throws {
        let vectors = (0..<20).map { _ in
            let values = (0..<512).map { _ in Float.random(in: -1...1) }
            return Vector512(values)
        }
        
        // CPU computation
        let cpuMatrix = DistanceComputation512.distanceMatrixCPU(
            vectors: vectors,
            metric: .euclidean
        )
        
        // GPU computation
        let gpuMatrix = try await metalMatrix.computeDistanceMatrix(
            vectorsA: vectors,
            vectorsB: nil,
            metric: .euclidean
        )
        
        // Compare results
        for i in 0..<vectors.count {
            for j in 0..<vectors.count {
                XCTAssertEqual(cpuMatrix[i][j], gpuMatrix[i][j], accuracy: 0.001,
                             "Mismatch at [\(i)][\(j)]: CPU=\(cpuMatrix[i][j]), GPU=\(gpuMatrix[i][j])")
            }
        }
    }
    
    func testStreamingLargeMatrix() async throws {
        // Test streaming with a moderately large matrix
        let vectors = (0..<500).map { i in
            Vector512(repeating: Float(i) / 500.0)
        }
        
        let matrix = try await metalMatrix.computeDistanceMatrix(
            vectorsA: vectors,
            vectorsB: nil,
            metric: .euclidean
        )
        
        XCTAssertEqual(matrix.count, vectors.count)
        XCTAssertEqual(matrix[0].count, vectors.count)
        
        // Spot check some values
        XCTAssertEqual(matrix[0][0], 0.0, accuracy: 0.0001)
        XCTAssertGreaterThan(matrix[0][vectors.count-1], 0)
    }
    
    func testAsyncPipeline() async throws {
        let vectors = (0..<100).map { i in
            Vector512(repeating: Float(i))
        }
        
        let expectation = XCTestExpectation(description: "Async computation completes")
        var result: [[Float]]?
        
        try await metalMatrix.computeDistanceMatrixAsync(
            vectorsA: vectors,
            vectorsB: nil,
            metric: .euclidean
        ) { matrix in
            result = matrix
            expectation.fulfill()
        }
        
        await fulfillment(of: [expectation], timeout: 5.0)
        
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.count, vectors.count)
    }
    
    func testPerformanceSmallMatrix() async throws {
        let vectors = (0..<100).map { _ in
            let values = (0..<512).map { _ in Float.random(in: -1...1) }
            return Vector512(values)
        }
        
        // Measure GPU performance
        let gpuTime = await measure {
            _ = try? await metalMatrix.computeDistanceMatrix(
                vectorsA: vectors,
                vectorsB: nil,
                metric: .euclidean
            )
        }
        
        // Measure CPU performance
        let cpuTime = measure {
            _ = DistanceComputation512.distanceMatrixCPU(
                vectors: vectors,
                metric: .euclidean
            )
        }
        
        print("Small matrix (100x100):")
        print("  GPU time: \(gpuTime)s")
        print("  CPU time: \(cpuTime)s")
        print("  Speedup: \(cpuTime/gpuTime)x")
    }
    
    func testPerformanceLargeMatrix() async throws {
        let vectors = (0..<1000).map { _ in
            let values = (0..<512).map { _ in Float.random(in: -1...1) }
            return Vector512(values)
        }
        
        // Only measure GPU for large matrix (CPU would be too slow)
        let gpuTime = await measure {
            _ = try? await metalMatrix.computeDistanceMatrix(
                vectorsA: vectors,
                vectorsB: nil,
                metric: .euclidean
            )
        }
        
        print("Large matrix (1000x1000):")
        print("  GPU time: \(gpuTime)s")
        print("  Throughput: \(1_000_000.0/gpuTime) distances/sec")
    }
    
    func testBenchmarkResults() async throws {
        let results = try await metalMatrix.benchmark(
            sizes: [100, 500, 1000],
            metric: .euclidean
        )
        
        print(results.summary())
        
        XCTAssertGreaterThan(results.averageSpeedup, 1.0)
    }
    
    // Helper to measure async execution time
    func measure(_ block: () async throws -> Void) async -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        try? await block()
        return CFAbsoluteTimeGetCurrent() - start
    }
    
    // Helper to measure sync execution time
    func measure(_ block: () throws -> Void) -> TimeInterval {
        let start = CFAbsoluteTimeGetCurrent()
        try? block()
        return CFAbsoluteTimeGetCurrent() - start
    }
}
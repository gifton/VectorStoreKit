// VectorStoreKit: Vector ML Pipeline Tests
//
// Unit tests for VectorMLPipeline functionality

import XCTest
@testable import VectorStoreKit
@preconcurrency import Metal

final class VectorMLPipelineTests: XCTestCase {
    
    var pipeline: VectorMLPipeline!
    
    override func setUp() async throws {
        try await super.setUp()
        pipeline = try await VectorMLPipeline()
    }
    
    override func tearDown() async throws {
        pipeline = nil
        try await super.tearDown()
    }
    
    // MARK: - Encoding Tests
    
    func testVectorEncoding() async throws {
        // Generate test vectors
        let vectorCount = 100
        let dimensions = 64
        let targetDim = 16
        
        var vectors: [[Float]] = []
        for _ in 0..<vectorCount {
            let vector = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            vectors.append(vector)
        }
        
        // Train encoder
        let config = VectorEncoderConfig(
            hiddenLayers: [32],
            epochs: 10,  // Reduced for testing
            batchSize: 32
        )
        
        try await pipeline.trainEncoder(
            on: vectors,
            targetDimensions: targetDim,
            config: config
        )
        
        // Encode vectors
        let encoded = try await pipeline.encodeVectors(vectors)
        
        // Verify dimensions
        XCTAssertEqual(encoded.count, vectors.count)
        XCTAssertEqual(encoded.first?.count, targetDim)
        
        // Verify values are bounded
        for vector in encoded {
            for value in vector {
                XCTAssertTrue(value >= -1 && value <= 1, "Encoded values should be bounded by tanh activation")
            }
        }
    }
    
    // MARK: - Similarity Learning Tests
    
    func testSimilarityLearning() async throws {
        let dimensions = 32
        
        // Create test data
        var positives: [([Float], [Float])] = []
        var negatives: [([Float], [Float])] = []
        
        // Generate positive pairs (similar vectors)
        for _ in 0..<20 {
            let base = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            let similar = base.map { $0 + Float.random(in: -0.1...0.1) }
            positives.append((base, similar))
        }
        
        // Generate negative pairs (different vectors)
        for _ in 0..<20 {
            let vec1 = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            let vec2 = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            negatives.append((vec1, vec2))
        }
        
        // Train similarity model
        let config = SimilarityLearningConfig(
            hiddenSize: 64,
            epochs: 10,
            batchSize: 8
        )
        
        let model = try await pipeline.learnSimilarity(
            positives: positives,
            negatives: negatives,
            config: config
        )
        
        // Test similarity computation
        let testSimilar = positives[0]
        let testDifferent = negatives[0]
        
        let simScore = try await pipeline.computeSimilarity(
            testSimilar.0,
            testSimilar.1,
            using: model
        )
        
        let diffScore = try await pipeline.computeSimilarity(
            testDifferent.0,
            testDifferent.1,
            using: model
        )
        
        // Similar vectors should have higher similarity
        XCTAssertGreaterThan(simScore, 0.5, "Similar vectors should have high similarity")
        XCTAssertLessThan(diffScore, 0.5, "Different vectors should have low similarity")
    }
    
    // MARK: - Clustering Tests
    
    func testClusterOptimization() async throws {
        let vectorCount = 200
        let dimensions = 32
        let clusterCount = 8
        
        // Generate clustered data
        var vectors: [[Float]] = []
        
        // Create distinct clusters
        for cluster in 0..<clusterCount {
            // Generate cluster center
            let center = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            
            // Generate points around center
            for _ in 0..<(vectorCount / clusterCount) {
                let point = center.map { $0 + Float.random(in: -0.2...0.2) }
                vectors.append(point)
            }
        }
        
        // Shuffle vectors
        vectors.shuffle()
        
        // Optimize clusters
        let config = ClusterOptimizationConfig(
            epochs: 20,
            batchSize: 16,
            adaptiveProbing: true
        )
        
        let assignments = try await pipeline.optimizeClusters(
            vectors,
            clusterCount: clusterCount,
            config: config
        )
        
        // Verify results
        XCTAssertEqual(assignments.centroids.count, clusterCount)
        XCTAssertGreaterThan(assignments.assignments.count, 0)
        
        // Test probe prediction
        let query = vectors[0]
        let probes = try await pipeline.predictOptimalProbes(
            for: query,
            targetRecall: 0.95
        )
        
        XCTAssertGreaterThan(probes, 0)
        XCTAssertLessThanOrEqual(probes, clusterCount)
    }
    
    // MARK: - Dimension Reduction Tests
    
    func testDimensionReductionMethods() async throws {
        let vectors = generateTestVectors(count: 50, dimensions: 64)
        let targetDim = 16
        
        // Test different methods
        let methods: [DimensionReductionMethod] = [.pca, .randomProjection, .neuralPCA]
        
        for method in methods {
            let reduced = try await pipeline.reduceDimensions(
                vectors,
                targetDim: targetDim,
                method: method
            )
            
            XCTAssertEqual(reduced.count, vectors.count)
            XCTAssertEqual(reduced.first?.count, targetDim)
            
            // Verify values are finite
            for vector in reduced {
                for value in vector {
                    XCTAssertTrue(value.isFinite, "\(method) produced non-finite values")
                }
            }
        }
    }
    
    // MARK: - Performance Metrics Tests
    
    func testPerformanceMetrics() async throws {
        let vectors = generateTestVectors(count: 100, dimensions: 32)
        
        // Reset metrics
        await pipeline.resetMetrics()
        
        // Perform operations
        let config = VectorEncoderConfig(
            hiddenLayers: [16],
            epochs: 5,
            batchSize: 32
        )
        
        try await pipeline.trainEncoder(
            on: vectors,
            targetDimensions: 8,
            config: config
        )
        
        let _ = try await pipeline.encodeVectors(vectors)
        
        // Check metrics
        let metrics = await pipeline.getMetrics()
        
        XCTAssertGreaterThan(metrics.vectorsEncoded, 0)
        XCTAssertGreaterThan(metrics.encoderTrainingTime, 0)
    }
    
    // MARK: - Helper Methods
    
    private func generateTestVectors(count: Int, dimensions: Int) -> [[Float]] {
        (0..<count).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }
    }
    
    // MARK: - Integration Tests
    
    func testEndToEndVectorProcessing() async throws {
        // Generate high-dimensional vectors
        let originalVectors = generateTestVectors(count: 200, dimensions: 128)
        
        // 1. Reduce dimensions
        let reducedDim = 32
        try await pipeline.trainEncoder(
            on: originalVectors,
            targetDimensions: reducedDim,
            config: VectorEncoderConfig(epochs: 20)
        )
        
        let reducedVectors = try await pipeline.encodeVectors(originalVectors)
        
        // 2. Optimize clustering on reduced vectors
        let clusterCount = 16
        let assignments = try await pipeline.optimizeClusters(
            reducedVectors,
            clusterCount: clusterCount
        )
        
        // 3. Learn similarity on reduced space
        var positives: [([Float], [Float])] = []
        for i in 0..<10 {
            if i + 1 < reducedVectors.count {
                positives.append((reducedVectors[i], reducedVectors[i + 1]))
            }
        }
        
        var negatives: [([Float], [Float])] = []
        for i in 0..<10 {
            if i < reducedVectors.count / 2 {
                negatives.append((reducedVectors[i], reducedVectors[reducedVectors.count - i - 1]))
            }
        }
        
        let similarityModel = try await pipeline.learnSimilarity(
            positives: positives,
            negatives: negatives,
            config: SimilarityLearningConfig(epochs: 10)
        )
        
        // Verify complete pipeline
        XCTAssertEqual(reducedVectors.count, originalVectors.count)
        XCTAssertEqual(reducedVectors.first?.count, reducedDim)
        XCTAssertEqual(assignments.centroids.count, clusterCount)
        XCTAssertNotNil(similarityModel)
        
        // Test query processing
        let query = originalVectors[0]
        let encodedQuery = try await pipeline.encodeVectors([query])[0]
        let optimalProbes = try await pipeline.predictOptimalProbes(for: encodedQuery)
        
        XCTAssertGreaterThan(optimalProbes, 0)
        XCTAssertLessThanOrEqual(optimalProbes, clusterCount)
    }
}
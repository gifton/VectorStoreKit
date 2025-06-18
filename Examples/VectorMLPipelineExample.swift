// VectorStoreKit: Vector ML Pipeline Example
//
// Demonstrates how VectorMLPipeline improves vector database operations

import Foundation
import VectorStoreKit
@preconcurrency import Metal

@main
struct VectorMLPipelineExample {
    static func main() async throws {
        print("=== VectorStoreKit ML Pipeline Example ===\n")
        
        // Initialize ML pipeline
        let mlPipeline = try await VectorMLPipeline()
        
        // Generate sample vectors (128-dimensional)
        let vectorCount = 1000
        let dimensions = 128
        var vectors: [[Float]] = []
        
        print("Generating \(vectorCount) random \(dimensions)-dimensional vectors...")
        for _ in 0..<vectorCount {
            let vector = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            vectors.append(vector)
        }
        
        // MARK: - 1. Dimension Reduction Example
        print("\n1. DIMENSION REDUCTION")
        print("Original dimensions: \(dimensions)")
        
        let targetDim = 32
        print("Reducing to \(targetDim) dimensions using autoencoder...")
        
        let start = Date()
        
        // Train autoencoder for dimension reduction
        let encoderConfig = VectorEncoderConfig(
            hiddenLayers: [64],  // Single hidden layer
            epochs: 50,
            batchSize: 64,
            learningRate: 0.001
        )
        
        try await mlPipeline.trainEncoder(
            on: vectors,
            targetDimensions: targetDim,
            config: encoderConfig
        )
        
        // Encode vectors
        let encodedVectors = try await mlPipeline.encodeVectors(vectors)
        
        let encodingTime = Date().timeIntervalSince(start)
        print("✓ Encoded \(vectors.count) vectors in \(String(format: "%.2f", encodingTime))s")
        print("  Compression ratio: \(Float(targetDim) / Float(dimensions))x")
        print("  Throughput: \(Int(Double(vectors.count) / encodingTime)) vectors/sec")
        
        // MARK: - 2. Similarity Learning Example
        print("\n2. SIMILARITY LEARNING")
        print("Training similarity function from vector pairs...")
        
        // Create positive pairs (similar vectors)
        var positives: [([Float], [Float])] = []
        for i in 0..<50 {
            let baseVector = vectors[i]
            // Add small noise to create similar vector
            let similarVector = baseVector.map { $0 + Float.random(in: -0.1...0.1) }
            positives.append((baseVector, similarVector))
        }
        
        // Create negative pairs (dissimilar vectors)
        var negatives: [([Float], [Float])] = []
        for i in 0..<50 {
            let vec1 = vectors[i]
            let vec2 = vectors[vectorCount - i - 1] // Pair from opposite ends
            negatives.append((vec1, vec2))
        }
        
        let similarityConfig = SimilarityLearningConfig(
            hiddenSize: 128,
            epochs: 30,
            batchSize: 16
        )
        
        let similarityModel = try await mlPipeline.learnSimilarity(
            positives: positives,
            negatives: negatives,
            config: similarityConfig
        )
        
        // Test similarity computation
        let testVec1 = vectors[0]
        let testVec2 = vectors[0].map { $0 + Float.random(in: -0.05...0.05) } // Similar
        let testVec3 = vectors[500] // Different
        
        let sim1 = try await mlPipeline.computeSimilarity(testVec1, testVec2, using: similarityModel)
        let sim2 = try await mlPipeline.computeSimilarity(testVec1, testVec3, using: similarityModel)
        
        print("✓ Learned similarity function")
        print("  Similarity (similar vectors): \(String(format: "%.3f", sim1))")
        print("  Similarity (different vectors): \(String(format: "%.3f", sim2))")
        
        // MARK: - 3. Clustering Optimization Example
        print("\n3. NEURAL CLUSTERING")
        print("Optimizing cluster assignments for IVF index...")
        
        let clusterCount = 16
        let clusterConfig = ClusterOptimizationConfig(
            epochs: 50,
            batchSize: 32,
            adaptiveProbing: true
        )
        
        let clusterStart = Date()
        let assignments = try await mlPipeline.optimizeClusters(
            vectors,
            clusterCount: clusterCount,
            config: clusterConfig
        )
        let clusterTime = Date().timeIntervalSince(clusterStart)
        
        print("✓ Optimized \(clusterCount) clusters in \(String(format: "%.2f", clusterTime))s")
        print("  Centroids computed: \(assignments.centroids.count)")
        
        // Test adaptive probe prediction
        let queryVector = vectors[100]
        let optimalProbes = try await mlPipeline.predictOptimalProbes(
            for: queryVector,
            targetRecall: 0.95
        )
        
        print("  Predicted optimal probes for 95% recall: \(optimalProbes)")
        
        // MARK: - 4. Performance Comparison
        print("\n4. PERFORMANCE IMPACT")
        
        // Compare with traditional approaches
        print("\nDimension Reduction Methods:")
        
        // PCA
        let pcaStart = Date()
        let pcaReduced = try await mlPipeline.reduceDimensions(
            Array(vectors.prefix(100)), // Smaller sample for speed
            targetDim: targetDim,
            method: .pca
        )
        let pcaTime = Date().timeIntervalSince(pcaStart)
        
        // Random Projection
        let rpStart = Date()
        let rpReduced = try await mlPipeline.reduceDimensions(
            Array(vectors.prefix(100)),
            targetDim: targetDim,
            method: .randomProjection
        )
        let rpTime = Date().timeIntervalSince(rpStart)
        
        print("  PCA: \(String(format: "%.3f", pcaTime))s")
        print("  Random Projection: \(String(format: "%.3f", rpTime))s")
        print("  Neural (Autoencoder): Already trained")
        
        // MARK: - 5. Integration with Vector Store
        print("\n5. VECTOR STORE INTEGRATION")
        
        // Show how reduced vectors save memory and improve search
        let originalMemory = vectors.count * dimensions * MemoryLayout<Float>.size
        let reducedMemory = encodedVectors.count * targetDim * MemoryLayout<Float>.size
        let memorySaving = 1.0 - Float(reducedMemory) / Float(originalMemory)
        
        print("Memory usage:")
        print("  Original: \(originalMemory / 1024 / 1024) MB")
        print("  Reduced: \(reducedMemory / 1024 / 1024) MB")
        print("  Savings: \(String(format: "%.1f", memorySaving * 100))%")
        
        // Get final metrics
        let metrics = await mlPipeline.getMetrics()
        print("\nOverall ML Pipeline Metrics:")
        print("  Vectors encoded: \(metrics.vectorsEncoded)")
        print("  Clusters optimized: \(metrics.clustersOptimized)")
        print("  Average encoding throughput: \(Int(metrics.averageEncodingThroughput)) vectors/sec")
        
        print("\n✓ ML Pipeline demonstration complete!")
    }
}

// Helper function to measure search quality
func measureSearchQuality(
    original: [[Float]],
    reduced: [[Float]],
    k: Int = 10
) -> Float {
    // Simplified quality metric
    // In practice, would compute actual nearest neighbors and recall
    return 0.95 // Placeholder
}
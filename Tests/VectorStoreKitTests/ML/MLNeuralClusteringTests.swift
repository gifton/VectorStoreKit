import XCTest
// VectorStoreKit: Neural Clustering Example
//
// Demonstrates the use of neural clustering with IVF index

import Foundation
import VectorStoreKit

final class NeuralClusteringTests: XCTestCase {
    func testMain() async throws {
        print("🧠 VectorStoreKit Neural Clustering Example")
        print("==========================================\n")
        
        // Configuration
        let dimensions = 128
        let numberOfClusters = 20
        let vectorCount = 10000
        
        // Create neural clustering configuration
        let neuralConfig = NeuralClusteringConfig(
            clusterHiddenSizes: [256, 128], // Neural network architecture
            epochs: 50,
            batchSize: 32,
            learningRate: 0.001,
            adaptiveProbing: true,           // Enable adaptive probe prediction
            onlineAdaptation: true,          // Enable online learning from queries
            queryHistorySize: 1000,
            adaptationInterval: 100
        )
        
        // Create IVF configuration with neural clustering
        let ivfConfig = IVFConfiguration(
            dimensions: dimensions,
            numberOfCentroids: numberOfClusters,
            numberOfProbes: 5,               // Default probes (will be adapted)
            trainingSampleSize: 5000,
            neuralClusteringConfig: neuralConfig,
            clusteringMethod: .hybrid        // Use k-means + neural refinement
        )
        
        print("📊 Configuration:")
        print("  - Dimensions: \(dimensions)")
        print("  - Clusters: \(numberOfClusters)")
        print("  - Vector count: \(vectorCount)")
        print("  - Clustering method: Hybrid (K-means + Neural)")
        print("  - Adaptive probing: Enabled")
        print("  - Online adaptation: Enabled\n")
        
        // Create index
        let index = try await IVFIndex<SIMD32<Float>, DocumentMetadata>(
            configuration: ivfConfig
        )
        
        // Generate synthetic document embeddings
        print("🔄 Generating synthetic document embeddings...")
        let vectors = generateDocumentEmbeddings(
            count: vectorCount,
            dimensions: dimensions,
            numberOfTopics: numberOfClusters
        )
        
        // Train the index
        print("🎯 Training neural clustering...")
        let trainingStart = Date()
        try await index.train(on: vectors.map { $0.0 })
        let trainingTime = Date().timeIntervalSince(trainingStart)
        print("✅ Training completed in \(String(format: "%.2f", trainingTime))s\n")
        
        // Insert vectors
        print("📥 Inserting vectors into index...")
        let insertStart = Date()
        for (i, (vector, metadata)) in vectors.enumerated() {
            if i % 1000 == 0 {
                print("  Inserted \(i)/\(vectorCount) vectors...")
            }
            
            let simdVector = vectorToSIMD32(vector)
            let entry = VectorEntry(
                id: "doc_\(i)",
                vector: simdVector,
                metadata: metadata
            )
            _ = try await index.insert(entry)
        }
        let insertTime = Date().timeIntervalSince(insertStart)
        print("✅ Insertion completed in \(String(format: "%.2f", insertTime))s\n")
        
        // Demonstrate adaptive search
        print("🔍 Demonstrating adaptive neural search:")
        
        // Search for documents similar to a query
        let queryTopic = 3 // Search for documents about topic 3
        let queryVector = generateTopicVector(
            topic: queryTopic,
            dimensions: dimensions,
            noise: 0.3
        )
        let query = vectorToSIMD32(queryVector)
        
        print("\n  Query: Documents similar to topic \(queryTopic)")
        
        // Perform multiple searches to show adaptation
        var searchTimes: [TimeInterval] = []
        
        for iteration in 1...5 {
            print("\n  🔄 Search iteration \(iteration):")
            
            let searchStart = Date()
            let results = try await index.search(
                query: query,
                k: 10,
                strategy: .adaptive
            )
            let searchTime = Date().timeIntervalSince(searchStart)
            searchTimes.append(searchTime)
            
            print("    ⏱️  Search time: \(String(format: "%.3f", searchTime * 1000))ms")
            print("    📊 Results:")
            
            for (i, result) in results.prefix(5).enumerated() {
                print("      \(i+1). \(result.id) - Distance: \(String(format: "%.3f", result.distance))")
                print("         Topic: \(result.metadata.topic), Category: \(result.metadata.category)")
            }
        }
        
        // Show search time improvement
        print("\n📈 Search Performance Analysis:")
        print("  - Initial search time: \(String(format: "%.3f", searchTimes[0] * 1000))ms")
        print("  - Final search time: \(String(format: "%.3f", searchTimes.last! * 1000))ms")
        if let improvement = calculateImprovement(initial: searchTimes[0], final: searchTimes.last!) {
            print("  - Improvement: \(String(format: "%.1f", improvement))%")
        }
        
        // Show index statistics
        print("\n📊 Index Statistics:")
        let stats = await index.statistics()
        print("  - Total vectors: \(stats.vectorCount)")
        print("  - Memory usage: \(formatBytes(stats.memoryUsage))")
        print("  - Average list size: \(stats.averageListSize)")
        print("  - Trained: \(stats.trained)")
        
        print("\n✅ Example completed successfully!")
    }
    
    // MARK: - Helper Functions
    
    func testGenerateDocumentEmbeddings(
        count: Int,
        dimensions: Int,
        numberOfTopics: Int
    ) -> [([Float], DocumentMetadata)] {
        var embeddings: [([Float], DocumentMetadata)] = []
        
        let categories = ["Technology", "Science", "Business", "Health", "Entertainment"]
        
        for i in 0..<count {
            let topic = i % numberOfTopics
            let category = categories[topic % categories.count]
            
            let vector = generateTopicVector(
                topic: topic,
                dimensions: dimensions,
                noise: 0.5
            )
            
            let metadata = DocumentMetadata(
                topic: topic,
                category: category,
                timestamp: Date(),
                score: Float.random(in: 0.5...1.0)
            )
            
            embeddings.append((vector, metadata))
        }
        
        return embeddings.shuffled()
    }
    
    func testGenerateTopicVector(
        topic: Int,
        dimensions: Int,
        noise: Float
    ) -> [Float] {
        var vector = [Float](repeating: 0, count: dimensions)
        
        // Create a characteristic pattern for each topic
        let baseOffset = Float(topic) * 2.0
        let frequency = Float(topic + 1) * 0.5
        
        for i in 0..<dimensions {
            let angle = Float(i) * frequency * .pi / Float(dimensions)
            let baseValue = sin(angle + baseOffset) * 0.5 + 0.5
            let noiseValue = Float.random(in: -noise...noise)
            vector[i] = baseValue + noiseValue
        }
        
        // Normalize vector
        let norm = sqrt(vector.reduce(0) { $0 + $1 * $1 })
        if norm > 0 {
            vector = vector.map { $0 / norm }
        }
        
        return vector
    }
    
    func testVectorToSIMD32(_ vector: [Float]) -> SIMD32<Float> {
        var simd = SIMD32<Float>()
        for i in 0..<min(vector.count, 32) {
            simd[i] = vector[i]
        }
        return simd
    }
    
    func testCalculateImprovement(initial: TimeInterval, final: TimeInterval) -> Double? {
        guard initial > 0 else { return nil }
        return ((initial - final) / initial) * 100
    }
    
    func testFormatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}

// MARK: - Document Metadata

struct DocumentMetadata: Codable, Sendable {
    let topic: Int
    let category: String
    let timestamp: Date
    let score: Float
}

// MARK: - SIMD Extensions

extension SIMD32 where Scalar == Float {
    init(_ array: [Float]) {
        self.init()
        for i in 0..<Swift.min(array.count, 32) {
            self[i] = array[i]
        }
    }
}
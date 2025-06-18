// VectorStoreKit: Image Similarity Search Example
//
// Demonstrates image similarity search using feature embeddings

import Foundation
import VectorStoreKit
import CoreImage
import Vision
import Metal

@main
struct ImageSimilarityExample {
    
    struct ImageData {
        let id: String
        let filename: String
        let category: String
        let tags: [String]
    }
    
    static func main() async throws {
        print("🖼️ VectorStoreKit Image Similarity Search Example")
        print("=" * 50)
        
        // Initialize Metal-accelerated vector store
        let vectorStore = try await createImageVectorStore()
        
        // Load and index sample images
        print("\n📸 Indexing images...")
        let images = createSampleImageData()
        try await indexImages(images, in: vectorStore)
        
        // Demonstrate various similarity searches
        print("\n🔍 Performing similarity searches...")
        
        // 1. Find similar images
        await demonstrateSimilarImageSearch(vectorStore: vectorStore)
        
        // 2. Search by visual features
        await demonstrateFeatureBasedSearch(vectorStore: vectorStore)
        
        // 3. Combined visual and tag search
        await demonstrateCombinedSearch(vectorStore: vectorStore)
        
        // 4. Batch similarity analysis
        await demonstrateBatchAnalysis(vectorStore: vectorStore)
        
        // 5. Clustering similar images
        await demonstrateClustering(vectorStore: vectorStore)
        
        print("\n✅ Image similarity example completed!")
    }
    
    // MARK: - Store Creation
    
    static func createImageVectorStore() async throws -> VectorStore<SIMD256<Float>, ImageMetadata> {
        // Use HNSW for fast similarity search with larger embeddings
        let hnswConfig = HNSWConfiguration(
            dimensions: 256, // Typical for image features
            maxConnections: 32,
            efConstruction: 400,
            similarity: .cosine
        )
        
        // Configure Metal acceleration
        let metalConfig = MetalComputeConfiguration(
            device: nil, // Use default device
            useSharedMemory: true,
            enableProfiling: false
        )
        
        // Multi-level caching for performance
        let cacheConfig = AdaptiveCacheConfiguration<SIMD256<Float>>(
            hotCacheSize: 100,
            warmCacheSize: 500,
            coldCacheSize: 2000,
            adaptationInterval: 60
        )
        
        let config = StoreConfiguration(
            indexType: .hnsw(hnswConfig),
            cacheType: .adaptive(cacheConfig),
            metalConfiguration: metalConfig,
            persistenceURL: URL(fileURLWithPath: "/tmp/image_vectors")
        )
        
        return try await VectorStore<SIMD256<Float>, ImageMetadata>(
            configuration: config
        )
    }
    
    // MARK: - Image Indexing
    
    static func indexImages(
        _ images: [ImageData],
        in store: VectorStore<SIMD256<Float>, ImageMetadata>
    ) async throws {
        let featureExtractor = MockImageFeatureExtractor()
        
        // Batch process for efficiency
        let batchSize = 10
        for batch in images.chunked(into: batchSize) {
            var entries: [VectorEntry<SIMD256<Float>, ImageMetadata>] = []
            
            for image in batch {
                // Extract features (in real app, would process actual image)
                let features = featureExtractor.extractFeatures(from: image)
                
                let metadata = ImageMetadata(
                    filename: image.filename,
                    category: image.category,
                    tags: image.tags,
                    width: 1024,
                    height: 768,
                    format: "jpeg"
                )
                
                entries.append(VectorEntry(
                    id: image.id,
                    vector: features,
                    metadata: metadata
                ))
            }
            
            // Batch insert
            try await store.addBatch(
                entries,
                options: InsertOptions(
                    deduplication: .error,
                    background: false
                )
            )
        }
        
        print("✅ Indexed \(images.count) images")
        
        // Optimize for search
        try await store.optimize(.incremental)
    }
    
    // MARK: - Search Demonstrations
    
    static func demonstrateSimilarImageSearch(vectorStore: VectorStore<SIMD256<Float>, ImageMetadata>) async {
        print("\n1️⃣ Similar Image Search")
        print("-" * 30)
        
        // Simulate searching with a query image
        let queryImage = ImageData(
            id: "query",
            filename: "sunset_beach.jpg",
            category: "landscape",
            tags: ["sunset", "beach", "ocean"]
        )
        
        let featureExtractor = MockImageFeatureExtractor()
        let queryFeatures = featureExtractor.extractFeatures(from: queryImage)
        
        do {
            let results = try await vectorStore.search(
                query: queryFeatures,
                k: 5,
                strategy: .approximate(probes: 20) // Fast approximate search
            )
            
            print("Query image: \(queryImage.filename)")
            print("Similar images:")
            for (i, result) in results.enumerated() {
                let similarity = (1.0 + result.score) / 2.0 // Convert cosine to 0-1
                print("  \(i + 1). \(result.metadata?.filename ?? "Unknown") (similarity: \(String(format: "%.1f%%", similarity * 100)))")
                if let tags = result.metadata?.tags {
                    print("     Tags: \(tags.joined(separator: ", "))")
                }
            }
        } catch {
            print("Search error: \(error)")
        }
    }
    
    static func demonstrateFeatureBasedSearch(vectorStore: VectorStore<SIMD256<Float>, ImageMetadata>) async {
        print("\n2️⃣ Feature-Based Visual Search")
        print("-" * 30)
        
        // Create a feature vector emphasizing certain visual characteristics
        var featureVector = SIMD256<Float>(repeating: 0)
        
        // Simulate emphasis on certain features
        // Indices 0-63: Color features (emphasize warm colors)
        for i in 0..<64 {
            featureVector[i] = Float.random(in: 0.7...1.0)
        }
        
        // Indices 64-127: Texture features
        for i in 64..<128 {
            featureVector[i] = Float.random(in: 0.3...0.5)
        }
        
        // Indices 128-191: Shape features
        for i in 128..<192 {
            featureVector[i] = Float.random(in: 0.2...0.4)
        }
        
        // Normalize
        let magnitude = sqrt((0..<256).reduce(Float(0)) { $0 + featureVector[$1] * featureVector[$1] })
        featureVector /= magnitude
        
        do {
            let results = try await vectorStore.search(
                query: featureVector,
                k: 3,
                strategy: .exact // Exact search for precision
            )
            
            print("Searching for images with warm colors and soft textures")
            print("Results:")
            for (i, result) in results.enumerated() {
                print("  \(i + 1). \(result.metadata?.filename ?? "Unknown")")
                print("     Category: \(result.metadata?.category ?? "Unknown")")
            }
        } catch {
            print("Search error: \(error)")
        }
    }
    
    static func demonstrateCombinedSearch(vectorStore: VectorStore<SIMD256<Float>, ImageMetadata>) async {
        print("\n3️⃣ Combined Visual and Tag Search")
        print("-" * 30)
        
        // Visual query
        let queryImage = ImageData(
            id: "query",
            filename: "mountain_snow.jpg",
            category: "landscape",
            tags: ["mountain", "snow"]
        )
        
        let featureExtractor = MockImageFeatureExtractor()
        let queryFeatures = featureExtractor.extractFeatures(from: queryImage)
        
        // Tag filter
        let requiredTags = ["mountain", "landscape"]
        let tagFilter = MetadataFilter<ImageMetadata> { metadata in
            let imageTags = Set(metadata.tags)
            let required = Set(requiredTags)
            return !imageTags.intersection(required).isEmpty
        }
        
        do {
            let results = try await vectorStore.search(
                query: queryFeatures,
                k: 5,
                strategy: .auto,
                filter: tagFilter
            )
            
            print("Visual query: mountain landscape")
            print("Required tags: \(requiredTags.joined(separator: ", "))")
            print("Results:")
            for (i, result) in results.enumerated() {
                print("  \(i + 1). \(result.metadata?.filename ?? "Unknown")")
                if let tags = result.metadata?.tags {
                    print("     Tags: \(tags.joined(separator: ", "))")
                }
            }
        } catch {
            print("Search error: \(error)")
        }
    }
    
    static func demonstrateBatchAnalysis(vectorStore: VectorStore<SIMD256<Float>, ImageMetadata>) async {
        print("\n4️⃣ Batch Similarity Analysis")
        print("-" * 30)
        
        // Analyze similarity between multiple query images
        let queryBatch = [
            ("sunset1.jpg", "Find images similar to multiple sunsets"),
            ("sunset2.jpg", ""),
            ("sunset3.jpg", "")
        ]
        
        let featureExtractor = MockImageFeatureExtractor()
        var allResults: [String: (count: Int, avgScore: Float)] = [:]
        
        for (filename, _) in queryBatch {
            let image = ImageData(id: filename, filename: filename, category: "query", tags: [])
            let features = featureExtractor.extractFeatures(from: image)
            
            if let results = try? await vectorStore.search(
                query: features,
                k: 10,
                strategy: .approximate(probes: 15)
            ) {
                for result in results {
                    let existing = allResults[result.id] ?? (count: 0, avgScore: 0)
                    let newCount = existing.count + 1
                    let newAvg = (existing.avgScore * Float(existing.count) + result.score) / Float(newCount)
                    allResults[result.id] = (count: newCount, avgScore: newAvg)
                }
            }
        }
        
        // Find images similar to all queries
        let commonResults = allResults
            .filter { $0.value.count >= 2 }
            .sorted { $0.value.avgScore > $1.value.avgScore }
            .prefix(5)
        
        print("Images similar to multiple sunset queries:")
        for (i, (id, stats)) in commonResults.enumerated() {
            print("  \(i + 1). Image \(id)")
            print("     Appeared in \(stats.count)/\(queryBatch.count) searches")
            print("     Average similarity: \(String(format: "%.2f", stats.avgScore))")
        }
    }
    
    static func demonstrateClustering(vectorStore: VectorStore<SIMD256<Float>, ImageMetadata>) async {
        print("\n5️⃣ Image Clustering")
        print("-" * 30)
        
        do {
            // Get all vectors for clustering
            let allVectors = try await vectorStore.export(format: .native)
            
            print("Performing k-means clustering on \(allVectors.count) images...")
            
            // Use the neural clustering component
            let clusteringConfig = NeuralClusteringConfiguration(
                numberOfClusters: 5,
                dimensions: 256,
                maxIterations: 50,
                convergenceThreshold: 0.001
            )
            
            let clustering = NeuralClustering(configuration: clusteringConfig)
            
            // Prepare vectors for clustering
            let vectors = allVectors.map { $0.vector }
            let clusters = try await clustering.cluster(vectors)
            
            // Analyze clusters
            print("\nCluster analysis:")
            for (clusterId, cluster) in clusters.enumerated() {
                print("\nCluster \(clusterId + 1): \(cluster.count) images")
                
                // Find common characteristics
                var categoryCount: [String: Int] = [:]
                var commonTags: [String: Int] = [:]
                
                for vectorId in cluster {
                    if let entry = allVectors.first(where: { $0.id == vectorId }),
                       let metadata = entry.metadata {
                        categoryCount[metadata.category, default: 0] += 1
                        for tag in metadata.tags {
                            commonTags[tag, default: 0] += 1
                        }
                    }
                }
                
                // Show dominant category
                if let dominantCategory = categoryCount.max(by: { $0.value < $1.value }) {
                    print("  Dominant category: \(dominantCategory.key) (\(dominantCategory.value) images)")
                }
                
                // Show common tags
                let topTags = commonTags
                    .sorted { $0.value > $1.value }
                    .prefix(3)
                    .map { $0.key }
                
                if !topTags.isEmpty {
                    print("  Common tags: \(topTags.joined(separator: ", "))")
                }
            }
        } catch {
            print("Clustering error: \(error)")
        }
    }
    
    // MARK: - Helper Functions
    
    static func createSampleImageData() -> [ImageData] {
        [
            // Landscapes
            ImageData(id: "img001", filename: "mountain_sunrise.jpg", category: "landscape", 
                     tags: ["mountain", "sunrise", "nature", "landscape"]),
            ImageData(id: "img002", filename: "beach_sunset.jpg", category: "landscape", 
                     tags: ["beach", "sunset", "ocean", "landscape"]),
            ImageData(id: "img003", filename: "forest_path.jpg", category: "landscape", 
                     tags: ["forest", "trees", "nature", "path"]),
            ImageData(id: "img004", filename: "desert_dunes.jpg", category: "landscape", 
                     tags: ["desert", "sand", "dunes", "landscape"]),
            ImageData(id: "img005", filename: "lake_reflection.jpg", category: "landscape", 
                     tags: ["lake", "water", "reflection", "mountain"]),
            
            // Urban
            ImageData(id: "img006", filename: "city_skyline.jpg", category: "urban", 
                     tags: ["city", "buildings", "skyline", "urban"]),
            ImageData(id: "img007", filename: "street_night.jpg", category: "urban", 
                     tags: ["street", "night", "lights", "urban"]),
            ImageData(id: "img008", filename: "bridge_architecture.jpg", category: "urban", 
                     tags: ["bridge", "architecture", "engineering"]),
            
            // Nature
            ImageData(id: "img009", filename: "flower_macro.jpg", category: "nature", 
                     tags: ["flower", "macro", "nature", "botanical"]),
            ImageData(id: "img010", filename: "butterfly_garden.jpg", category: "nature", 
                     tags: ["butterfly", "insect", "garden", "nature"]),
            ImageData(id: "img011", filename: "waterfall_jungle.jpg", category: "nature", 
                     tags: ["waterfall", "jungle", "water", "nature"]),
            
            // Animals
            ImageData(id: "img012", filename: "lion_portrait.jpg", category: "animals", 
                     tags: ["lion", "wildlife", "africa", "animal"]),
            ImageData(id: "img013", filename: "bird_flight.jpg", category: "animals", 
                     tags: ["bird", "flight", "wings", "animal"]),
            ImageData(id: "img014", filename: "underwater_fish.jpg", category: "animals", 
                     tags: ["fish", "underwater", "ocean", "marine"]),
            
            // Abstract
            ImageData(id: "img015", filename: "color_splash.jpg", category: "abstract", 
                     tags: ["abstract", "color", "art", "paint"]),
            ImageData(id: "img016", filename: "geometric_pattern.jpg", category: "abstract", 
                     tags: ["geometric", "pattern", "abstract", "design"])
        ]
    }
}

// MARK: - Supporting Types

struct ImageMetadata: VectorMetadata {
    let filename: String
    let category: String
    let tags: [String]
    let width: Int
    let height: Int
    let format: String
}

// Mock feature extractor for demonstration
struct MockImageFeatureExtractor {
    func extractFeatures(from image: ImageData) -> SIMD256<Float> {
        // In a real implementation, this would use Vision framework or Core ML
        var features = SIMD256<Float>(repeating: 0)
        
        // Simulate feature extraction based on image properties
        let categoryHash = image.category.hashValue
        let tagHashes = image.tags.map { $0.hashValue }
        
        // Color features (0-63)
        for i in 0..<64 {
            features[i] = Float((categoryHash + i) % 100) / 100.0
        }
        
        // Texture features (64-127)
        for i in 64..<128 {
            let tagInfluence = tagHashes.isEmpty ? 0 : tagHashes[i % tagHashes.count]
            features[i] = Float((tagInfluence + i) % 100) / 100.0
        }
        
        // Shape features (128-191)
        for i in 128..<192 {
            features[i] = Float((image.filename.hashValue + i) % 100) / 100.0
        }
        
        // Semantic features (192-255)
        for i in 192..<256 {
            features[i] = Float.random(in: 0...1)
        }
        
        // Add specific features based on tags
        if image.tags.contains("sunset") || image.tags.contains("sunrise") {
            for i in 0..<32 { features[i] = max(features[i], 0.8) } // Warm colors
        }
        if image.tags.contains("ocean") || image.tags.contains("water") {
            for i in 32..<64 { features[i] = max(features[i], 0.7) } // Blue tones
        }
        if image.tags.contains("mountain") || image.tags.contains("landscape") {
            for i in 128..<160 { features[i] = max(features[i], 0.6) } // Horizontal lines
        }
        
        // Normalize
        let magnitude = sqrt((0..<256).reduce(Float(0)) { $0 + features[$1] * features[$1] })
        if magnitude > 0 {
            features /= magnitude
        }
        
        return features
    }
}

// Helper extension
extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}

extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
import Foundation
import VectorStoreKit

/// Example demonstrating semantic search capabilities
@main
struct SemanticSearchExample {
    static func main() async throws {
        print("🔍 VectorStoreKit Semantic Search Example")
        print("=========================================\n")
        
        // Initialize components
        let vectorStore = try await createVectorStore()
        let embeddingProvider = try await createEmbeddingProvider()
        
        // Create semantic search engine
        let searchConfig = SemanticSearchConfiguration(
            topK: 10,
            threshold: 0.5,
            useGPUAcceleration: true,
            normalizeQuery: true,
            strategy: .approximate
        )
        
        let semanticSearch = SemanticSearch(
            vectorStore: vectorStore,
            embeddingProvider: embeddingProvider,
            configuration: searchConfig
        )
        
        // Index sample data
        print("📚 Indexing sample data...")
        try await indexSampleData(vectorStore, embeddingProvider)
        
        // Perform searches
        print("\n🔎 Performing semantic searches...")
        try await performSearches(semanticSearch)
        
        // Demonstrate batch search
        print("\n🔍 Batch semantic search...")
        try await performBatchSearch(semanticSearch)
        
        // Show metadata filtering
        print("\n🏷️ Search with metadata filtering...")
        try await searchWithFilters(semanticSearch)
        
        print("\n✅ Semantic search example completed!")
    }
    
    static func createVectorStore() async throws -> any LLMVectorStore {
        // Create in-memory vector store
        return InMemoryVectorStore(metric: .cosine)
    }
    
    static func createEmbeddingProvider() async throws -> any EmbeddingProvider {
        // Create a mock local provider for the example
        return MockLocalEmbeddingProvider(dimensions: 384)
    }
    
    static func indexSampleData(
        _ vectorStore: any LLMVectorStore,
        _ embeddingProvider: any EmbeddingProvider
    ) async throws {
        let documents = [
            (
                id: "doc1",
                text: "The quick brown fox jumps over the lazy dog",
                metadata: ["category": "animals", "language": "english"]
            ),
            (
                id: "doc2",
                text: "Machine learning models can understand semantic similarity",
                metadata: ["category": "technology", "topic": "ml"]
            ),
            (
                id: "doc3",
                text: "Swift is a powerful programming language for Apple platforms",
                metadata: ["category": "technology", "topic": "programming"]
            ),
            (
                id: "doc4",
                text: "The dog chased the cat through the garden",
                metadata: ["category": "animals", "location": "garden"]
            ),
            (
                id: "doc5",
                text: "Natural language processing enables semantic search",
                metadata: ["category": "technology", "topic": "nlp"]
            ),
            (
                id: "doc6",
                text: "Cats are independent and curious animals",
                metadata: ["category": "animals", "trait": "independent"]
            ),
            (
                id: "doc7",
                text: "Vector databases enable fast similarity search",
                metadata: ["category": "technology", "topic": "databases"]
            ),
            (
                id: "doc8",
                text: "The Swift programming language emphasizes safety and performance",
                metadata: ["category": "technology", "topic": "programming", "language": "swift"]
            )
        ]
        
        // Embed and index documents
        for doc in documents {
            let embedding = try await embeddingProvider.embed(doc.text)
            try await vectorStore.add(
                id: doc.id,
                vector: embedding,
                metadata: doc.metadata + ["text": doc.text]
            )
            print("  ✓ Indexed: \(doc.text.prefix(40))...")
        }
    }
    
    static func performSearches(_ semanticSearch: SemanticSearch) async throws {
        let queries = [
            "Animals in nature",
            "Programming languages",
            "Semantic understanding",
            "Fast and agile creatures"
        ]
        
        for query in queries {
            print("\n🔍 Query: '\(query)'")
            
            let results = try await semanticSearch.search(query: query)
            
            for (index, result) in results.prefix(3).enumerated() {
                let text = result.metadata["text"] as? String ?? "No text"
                print("  \(index + 1). [\(String(format: "%.3f", result.score))] \(text.prefix(50))...")
                if let category = result.metadata["category"] as? String {
                    print("     Category: \(category)")
                }
            }
        }
    }
    
    static func performBatchSearch(_ semanticSearch: SemanticSearch) async throws {
        let batchQueries = [
            "Furry animals",
            "Modern programming",
            "Data storage solutions"
        ]
        
        print("\n📦 Batch queries: \(batchQueries)")
        
        let batchResults = try await semanticSearch.batchSearch(queries: batchQueries)
        
        for (queryIndex, results) in batchResults.enumerated() {
            print("\n  Results for '\(batchQueries[queryIndex])':")
            for result in results.prefix(2) {
                let text = result.metadata["text"] as? String ?? "No text"
                print("    - [\(String(format: "%.3f", result.score))] \(text.prefix(40))...")
            }
        }
    }
    
    static func searchWithFilters(_ semanticSearch: SemanticSearch) async throws {
        // Search only in technology category
        let techFilter = MetadataFilter(.equals("category", "technology"))
        
        print("\n🔍 Query: 'programming' (filtered to technology)")
        let techResults = try await semanticSearch.search(
            query: "programming",
            filter: techFilter
        )
        
        for result in techResults.prefix(3) {
            let text = result.metadata["text"] as? String ?? "No text"
            let topic = result.metadata["topic"] as? String ?? "unknown"
            print("  - [\(String(format: "%.3f", result.score))] Topic: \(topic)")
            print("    \(text.prefix(50))...")
        }
        
        // Search with complex filter
        let complexFilter = MetadataFilter(
            .and([
                .equals("category", "animals"),
                .or([
                    .contains("text", "dog"),
                    .contains("text", "cat")
                ])
            ])
        )
        
        print("\n🔍 Query: 'pets' (filtered to animals mentioning dog or cat)")
        let petResults = try await semanticSearch.search(
            query: "pets",
            filter: complexFilter
        )
        
        for result in petResults {
            let text = result.metadata["text"] as? String ?? "No text"
            print("  - [\(String(format: "%.3f", result.score))] \(text)")
        }
    }
}

// MARK: - Mock Implementation for Example

/// Mock local embedding provider for demonstration
actor MockLocalEmbeddingProvider: EmbeddingProvider {
    let configuration: EmbeddingConfiguration
    
    init(dimensions: Int) {
        self.configuration = EmbeddingConfiguration(
            model: "mock-local",
            dimensions: dimensions,
            maxTokens: 512,
            normalize: true,
            batchSize: 32
        )
    }
    
    func embed(_ texts: [String]) async throws -> [Vector] {
        // Create deterministic embeddings based on text content
        return texts.map { text in
            var vector = Vector(dimensions: configuration.dimensions)
            
            // Simple hash-based embedding for demonstration
            let words = text.lowercased().split(separator: " ")
            for (index, word) in words.enumerated() {
                let hash = word.hashValue
                let position = abs(hash) % configuration.dimensions
                vector.data[position] += Float(1.0 / Float(words.count))
                
                // Add some variety based on word position
                if index < configuration.dimensions {
                    vector.data[index] += 0.1
                }
            }
            
            // Normalize
            vector.normalize()
            return vector
        }
    }
    
    func embed(_ text: String) async throws -> Vector {
        let results = try await embed([text])
        return results[0]
    }
    
    func estimateTokens(for text: String) -> Int {
        text.split(separator: " ").count
    }
    
    func isAvailable() async -> Bool {
        true
    }
}
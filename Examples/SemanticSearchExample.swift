// VectorStoreKit: Semantic Search Example
//
// Demonstrates building a semantic search engine using vector embeddings

import Foundation
import VectorStoreKit
import NaturalLanguage

@main
struct SemanticSearchExample {
    
    // Mock document database
    struct Document {
        let id: String
        let title: String
        let content: String
        let category: String
        let timestamp: Date
    }
    
    static func main() async throws {
        print("🔍 VectorStoreKit Semantic Search Example")
        print("=" * 50)
        
        // Create sample documents
        let documents = createSampleDocuments()
        
        // Initialize vector store with hybrid index for best search quality
        let vectorStore = try await createSemanticSearchStore()
        
        // Generate and index embeddings
        print("\n📝 Indexing documents...")
        try await indexDocuments(documents, in: vectorStore)
        
        // Demonstrate various search scenarios
        print("\n🔎 Performing searches...")
        
        // 1. Simple semantic search
        await demonstrateSimpleSearch(vectorStore: vectorStore)
        
        // 2. Filtered search
        await demonstrateFilteredSearch(vectorStore: vectorStore)
        
        // 3. Hybrid search (semantic + keyword)
        await demonstrateHybridSearch(vectorStore: vectorStore)
        
        // 4. Multi-query search
        await demonstrateMultiQuerySearch(vectorStore: vectorStore)
        
        // 5. Explain search results
        await demonstrateExplainableSearch(vectorStore: vectorStore)
        
        print("\n✅ Semantic search example completed!")
    }
    
    // MARK: - Store Creation
    
    static func createSemanticSearchStore() async throws -> VectorStore<SIMD64<Float>, DocumentMetadata> {
        // Configure hybrid index for best search quality
        let indexConfig = HybridIndexConfiguration(
            dimensions: 64, // Using 64-dim embeddings for this example
            hnswConfig: HNSWConfiguration(
                dimensions: 64,
                maxConnections: 16,
                efConstruction: 200,
                similarity: .cosine
            ),
            ivfConfig: IVFConfiguration(
                dimensions: 64,
                numberOfCentroids: 100,
                searchConfiguration: IVFSearchConfiguration(nProbe: 10)
            ),
            routingStrategy: .adaptive,
            adaptiveThreshold: 0.85
        )
        
        // Configure caching for frequently accessed documents
        let cacheConfig = LRUCacheConfiguration<SIMD64<Float>>(maxCapacity: 1000)
        
        // Create vector store
        let vectorStore = try await VectorStore<SIMD64<Float>, DocumentMetadata>(
            configuration: StoreConfiguration(
                indexType: .hybrid(indexConfig),
                cacheType: .lru(cacheConfig),
                persistenceURL: nil // In-memory for demo
            )
        )
        
        return vectorStore
    }
    
    // MARK: - Document Indexing
    
    static func indexDocuments(
        _ documents: [Document],
        in store: VectorStore<SIMD64<Float>, DocumentMetadata>
    ) async throws {
        // Generate embeddings for documents
        let embedder = MockEmbedder()
        
        for document in documents {
            // Generate embedding from title and content
            let text = "\(document.title) \(document.content)"
            let embedding = embedder.embed(text: text)
            
            // Create metadata
            let metadata = DocumentMetadata(
                title: document.title,
                content: document.content,
                category: document.category,
                timestamp: document.timestamp
            )
            
            // Index document
            try await store.add(
                VectorEntry(
                    id: document.id,
                    vector: embedding,
                    metadata: metadata
                ),
                options: InsertOptions(
                    deduplication: .skip,
                    background: false
                )
            )
        }
        
        print("✅ Indexed \(documents.count) documents")
        
        // Optimize index for search
        try await store.optimize(.full)
    }
    
    // MARK: - Search Demonstrations
    
    static func demonstrateSimpleSearch(vectorStore: VectorStore<SIMD64<Float>, DocumentMetadata>) async {
        print("\n1️⃣ Simple Semantic Search")
        print("-" * 30)
        
        let query = "How to use machine learning for data analysis"
        let embedder = MockEmbedder()
        let queryEmbedding = embedder.embed(text: query)
        
        do {
            let results = try await vectorStore.search(
                query: queryEmbedding,
                k: 5,
                strategy: .auto
            )
            
            print("Query: '\(query)'")
            print("Results:")
            for (i, result) in results.enumerated() {
                print("  \(i + 1). \(result.metadata?.title ?? "Unknown") (score: \(String(format: "%.3f", result.score)))")
                if let content = result.metadata?.content {
                    print("     Preview: \(String(content.prefix(100)))...")
                }
            }
        } catch {
            print("Search error: \(error)")
        }
    }
    
    static func demonstrateFilteredSearch(vectorStore: VectorStore<SIMD64<Float>, DocumentMetadata>) async {
        print("\n2️⃣ Filtered Semantic Search")
        print("-" * 30)
        
        let query = "AI applications"
        let embedder = MockEmbedder()
        let queryEmbedding = embedder.embed(text: query)
        
        // Filter by category
        let filter = MetadataFilter<DocumentMetadata> { metadata in
            metadata.category == "Technology"
        }
        
        do {
            let results = try await vectorStore.search(
                query: queryEmbedding,
                k: 3,
                strategy: .exact,
                filter: filter
            )
            
            print("Query: '\(query)' (filtered by Technology category)")
            print("Results:")
            for (i, result) in results.enumerated() {
                print("  \(i + 1). \(result.metadata?.title ?? "Unknown") - \(result.metadata?.category ?? "")")
            }
        } catch {
            print("Search error: \(error)")
        }
    }
    
    static func demonstrateHybridSearch(vectorStore: VectorStore<SIMD64<Float>, DocumentMetadata>) async {
        print("\n3️⃣ Hybrid Search (Semantic + Keyword)")
        print("-" * 30)
        
        let semanticQuery = "artificial intelligence"
        let keywords = ["neural", "network"]
        
        let embedder = MockEmbedder()
        let queryEmbedding = embedder.embed(text: semanticQuery)
        
        // Create keyword filter
        let keywordFilter = MetadataFilter<DocumentMetadata> { metadata in
            let content = (metadata.title + " " + metadata.content).lowercased()
            return keywords.allSatisfy { content.contains($0) }
        }
        
        do {
            let results = try await vectorStore.search(
                query: queryEmbedding,
                k: 5,
                strategy: .approximate(probes: 10),
                filter: keywordFilter
            )
            
            print("Semantic query: '\(semanticQuery)'")
            print("Required keywords: \(keywords.joined(separator: ", "))")
            print("Results:")
            for (i, result) in results.enumerated() {
                print("  \(i + 1). \(result.metadata?.title ?? "Unknown")")
            }
        } catch {
            print("Search error: \(error)")
        }
    }
    
    static func demonstrateMultiQuerySearch(vectorStore: VectorStore<SIMD64<Float>, DocumentMetadata>) async {
        print("\n4️⃣ Multi-Query Search (Query Expansion)")
        print("-" * 30)
        
        // Original query and expanded variations
        let queries = [
            "machine learning applications",
            "AI use cases",
            "artificial intelligence implementation"
        ]
        
        let embedder = MockEmbedder()
        var allResults: [String: Float] = [:]
        
        // Search with each query
        for query in queries {
            let embedding = embedder.embed(text: query)
            
            if let results = try? await vectorStore.search(
                query: embedding,
                k: 10,
                strategy: .auto
            ) {
                // Aggregate scores
                for result in results {
                    let id = result.id
                    allResults[id] = max(allResults[id] ?? 0, result.score)
                }
            }
        }
        
        // Sort by aggregated score
        let sortedResults = allResults.sorted { $0.value > $1.value }.prefix(5)
        
        print("Expanded queries: \(queries.joined(separator: ", "))")
        print("Aggregated results:")
        for (i, (id, score)) in sortedResults.enumerated() {
            print("  \(i + 1). Document \(id) (aggregated score: \(String(format: "%.3f", score)))")
        }
    }
    
    static func demonstrateExplainableSearch(vectorStore: VectorStore<SIMD64<Float>, DocumentMetadata>) async {
        print("\n5️⃣ Explainable Search Results")
        print("-" * 30)
        
        let query = "deep learning for computer vision"
        let embedder = MockEmbedder()
        let queryEmbedding = embedder.embed(text: query)
        
        do {
            let results = try await vectorStore.search(
                query: queryEmbedding,
                k: 3,
                strategy: .exact
            )
            
            print("Query: '\(query)'")
            print("\nDetailed results with explanations:")
            
            for (i, result) in results.enumerated() {
                print("\n\(i + 1). \(result.metadata?.title ?? "Unknown")")
                print("   Score: \(String(format: "%.3f", result.score))")
                print("   Category: \(result.metadata?.category ?? "Unknown")")
                
                // Calculate term overlap for explanation
                let queryTerms = Set(query.lowercased().split(separator: " ").map(String.init))
                let contentTerms = Set((result.metadata?.content ?? "").lowercased().split(separator: " ").map(String.init))
                let overlap = queryTerms.intersection(contentTerms)
                
                print("   Matching terms: \(overlap.joined(separator: ", "))")
                print("   Relevance: High semantic similarity in vector space")
            }
        } catch {
            print("Search error: \(error)")
        }
    }
    
    // MARK: - Helper Functions
    
    static func createSampleDocuments() -> [Document] {
        [
            Document(
                id: "doc1",
                title: "Introduction to Machine Learning",
                content: "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
                category: "Technology",
                timestamp: Date()
            ),
            Document(
                id: "doc2",
                title: "Deep Learning and Neural Networks",
                content: "Deep learning is a machine learning technique that teaches computers to do what comes naturally to humans: learn by example. Deep learning is a key technology behind driverless cars, enabling them to recognize a stop sign or distinguish pedestrians.",
                category: "Technology",
                timestamp: Date()
            ),
            Document(
                id: "doc3",
                title: "Natural Language Processing Fundamentals",
                content: "Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines including computational linguistics.",
                category: "Technology",
                timestamp: Date()
            ),
            Document(
                id: "doc4",
                title: "Computer Vision Applications",
                content: "Computer vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can identify and classify objects.",
                category: "Technology",
                timestamp: Date()
            ),
            Document(
                id: "doc5",
                title: "Data Science Best Practices",
                content: "Data science combines domain expertise, programming skills, and knowledge of mathematics and statistics to extract meaningful insights from data. Machine learning is a crucial tool for data scientists.",
                category: "Data Science",
                timestamp: Date()
            ),
            Document(
                id: "doc6",
                title: "AI in Healthcare",
                content: "Artificial intelligence in healthcare is the use of complex algorithms and software to emulate human cognition in the analysis, interpretation, and comprehension of complicated medical and healthcare data.",
                category: "Healthcare",
                timestamp: Date()
            ),
            Document(
                id: "doc7",
                title: "Reinforcement Learning Explained",
                content: "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.",
                category: "Technology",
                timestamp: Date()
            ),
            Document(
                id: "doc8",
                title: "Big Data Analytics",
                content: "Big data analytics examines large amounts of data to uncover hidden patterns, correlations and other insights. With today's technology, it's possible to analyze your data and get answers from it immediately.",
                category: "Data Science",
                timestamp: Date()
            )
        ]
    }
}

// MARK: - Supporting Types

struct DocumentMetadata: VectorMetadata {
    let title: String
    let content: String
    let category: String
    let timestamp: Date
}

// Mock embedder for demonstration
struct MockEmbedder {
    func embed(text: String) -> SIMD64<Float> {
        // In a real implementation, this would use a proper embedding model
        // For demo, we create a deterministic embedding based on text features
        var embedding = SIMD64<Float>(repeating: 0)
        
        let words = text.lowercased().split(separator: " ")
        
        // Simple feature extraction
        for (i, word) in words.prefix(64).enumerated() {
            let hash = word.hashValue
            embedding[i] = Float(hash % 100) / 100.0
        }
        
        // Add some semantic features
        if text.contains("machine learning") { embedding[0] = 0.9 }
        if text.contains("artificial intelligence") { embedding[1] = 0.9 }
        if text.contains("deep learning") { embedding[2] = 0.9 }
        if text.contains("neural network") { embedding[3] = 0.9 }
        if text.contains("computer vision") { embedding[4] = 0.9 }
        if text.contains("nlp") || text.contains("natural language") { embedding[5] = 0.9 }
        
        // Normalize
        let magnitude = sqrt(embedding.sum { $0 * $0 })
        if magnitude > 0 {
            embedding /= magnitude
        }
        
        return embedding
    }
}

// Helper extension
extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
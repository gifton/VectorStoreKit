import Foundation
import VectorStoreKit

/// Example demonstrating RAG (Retrieval-Augmented Generation) pipeline
@main
struct RAGExample {
    static func main() async throws {
        print("🚀 VectorStoreKit RAG Example")
        print("=============================\n")
        
        // Initialize components
        let vectorStore = try await createVectorStore()
        let embeddingProvider = try await createEmbeddingProvider()
        
        // Create RAG pipeline
        let ragConfig = RAGConfiguration(
            searchConfig: SemanticSearchConfiguration(
                topK: 5,
                threshold: 0.7,
                useGPUAcceleration: true
            ),
            chunkingConfig: ChunkingConfiguration(
                maxChunkSize: 500,
                overlapSize: 50,
                respectSentences: true
            ),
            contextConfig: ContextConfiguration(
                maxTokens: 4096,
                systemPromptTokens: 200,
                queryTokens: 100,
                responseTokens: 1000
            )
        )
        
        let ragPipeline = RAGPipeline(
            vectorStore: vectorStore,
            embeddingProvider: embeddingProvider,
            configuration: ragConfig
        )
        
        // Index some documents
        print("📚 Indexing documents...")
        try await indexSampleDocuments(ragPipeline)
        
        // Perform RAG queries
        print("\n🔍 Performing RAG queries...")
        try await performQueries(ragPipeline)
        
        // Demonstrate context management
        print("\n📝 Demonstrating context management...")
        try await demonstrateContextManagement()
        
        // Show chunking strategies
        print("\n✂️ Demonstrating chunking strategies...")
        demonstrateChunking()
        
        print("\n✅ RAG example completed!")
    }
    
    static func createVectorStore() async throws -> any LLMVectorStore {
        // Create in-memory vector store for the example
        return InMemoryVectorStore(metric: .cosine)
    }
    
    static func createEmbeddingProvider() async throws -> any EmbeddingProvider {
        // Try to create OpenAI provider from environment
        if let openAIProvider = await OpenAIEmbeddingProvider.fromEnvironment() {
            print("✅ Using OpenAI embeddings")
            return openAIProvider
        }
        
        // Fall back to local provider
        print("📱 Using local embeddings")
        return try await LocalEmbeddingProvider.sentenceTransformer()
    }
    
    static func indexSampleDocuments(_ pipeline: RAGPipeline) async throws {
        let documents = [
            (
                content: """
                VectorStoreKit is a high-performance vector database optimized for Apple Silicon. 
                It leverages Metal GPU acceleration for fast similarity search and supports various 
                indexing strategies including HNSW, IVF, and learned indexes. The framework is 
                designed with a focus on zero-copy operations and unified memory architecture.
                """,
                metadata: ["source": "README", "category": "overview"]
            ),
            (
                content: """
                Metal acceleration in VectorStoreKit provides significant performance improvements 
                for distance computations. The framework automatically routes operations between 
                CPU and GPU based on workload characteristics. For large-scale operations, GPU 
                acceleration can provide 10-100x speedup compared to CPU-only implementations.
                """,
                metadata: ["source": "docs", "category": "performance"]
            ),
            (
                content: """
                The RAG pipeline in VectorStoreKit supports semantic search with automatic chunking 
                and context management. Documents are split into chunks using configurable strategies, 
                embedded using various providers, and indexed for fast retrieval. The system handles 
                token budgeting and context assembly for LLM interactions.
                """,
                metadata: ["source": "docs", "category": "rag"]
            )
        ]
        
        for (index, doc) in documents.enumerated() {
            let document = try await pipeline.indexDocument(
                doc.content,
                metadata: doc.metadata,
                documentId: "doc-\(index)"
            )
            print("  ✓ Indexed document \(document.id) with \(document.chunks.count) chunks")
        }
    }
    
    static func performQueries(_ pipeline: RAGPipeline) async throws {
        let queries = [
            "What indexing strategies does VectorStoreKit support?",
            "How does Metal acceleration improve performance?",
            "Explain the RAG pipeline architecture"
        ]
        
        for query in queries {
            print("\n❓ Query: \(query)")
            
            // Retrieve context
            let context = try await pipeline.retrieveContext(for: query)
            
            print("📊 Retrieved \(context.chunks.count) relevant chunks")
            print("📈 Average relevance: \(String(format: "%.2f", context.metadata.averageRelevance))")
            print("⏱️ Retrieval time: \(String(format: "%.3f", context.metadata.retrievalTime))s")
            
            // Build prompt
            let (prompt, _) = try await pipeline.process(
                query: query,
                systemPrompt: "You are a helpful assistant with knowledge about VectorStoreKit."
            )
            
            print("💬 Generated prompt (\(context.tokenCount) tokens):")
            print(prompt.prefix(200) + "...")
        }
    }
    
    static func demonstrateContextManagement() async throws {
        let contextAssembler = ContextAssembler(
            configuration: ContextConfiguration(
                maxTokens: 4096,
                systemPromptTokens: 200,
                queryTokens: 100,
                responseTokens: 1000
            )
        )
        
        // Create sample conversation history
        let history = [
            ConversationTurn(role: .user, content: "What is VectorStoreKit?"),
            ConversationTurn(role: .assistant, content: "VectorStoreKit is a vector database for Apple Silicon."),
            ConversationTurn(role: .user, content: "What makes it fast?")
        ]
        
        // Create sample chunks
        let chunks = [
            (content: "Metal GPU acceleration provides fast computations.", metadata: ["source": "docs"]),
            (content: "Zero-copy operations reduce memory overhead.", metadata: ["source": "README"])
        ]
        
        // Assemble context
        let assembledContext = try await contextAssembler.assembleContext(
            systemPrompt: "You are an expert on VectorStoreKit.",
            userQuery: "Explain the performance optimizations",
            retrievedChunks: chunks,
            conversationHistory: history
        )
        
        print("📦 Assembled context:")
        print("  - System tokens: \(assembledContext.metadata.systemTokens)")
        print("  - Query tokens: \(assembledContext.metadata.queryTokens)")
        print("  - Context tokens: \(assembledContext.metadata.contextTokens)")
        print("  - Total tokens: \(assembledContext.metadata.totalTokensUsed)")
        
        // Format prompt
        let formattedPrompt = await contextAssembler.formatPrompt(
            assembledContext,
            template: .markdown
        )
        print("\n📄 Formatted prompt preview:")
        print(formattedPrompt.prefix(300) + "...")
    }
    
    static func demonstrateChunking() {
        let sampleText = """
        VectorStoreKit is designed for high performance. It uses Metal acceleration 
        for GPU compute operations.
        
        The framework supports multiple indexing strategies. HNSW provides excellent 
        recall with fast search times. IVF offers a good balance between memory usage 
        and search quality.
        
        For large datasets, hierarchical storage is employed. Hot data stays in memory 
        for fast access. Warm data uses memory-mapped files. Cold data is compressed 
        and stored on disk.
        """
        
        let config = ChunkingConfiguration(
            maxChunkSize: 100,
            overlapSize: 20,
            respectSentences: true
        )
        
        // Test different strategies
        let strategies: [(name: String, strategy: ChunkingStrategyFactory.Strategy)] = [
            ("Fixed Size", .fixedSize),
            ("Semantic", .semantic),
            ("Recursive", .recursive)
        ]
        
        for (name, strategy) in strategies {
            print("\n🔧 \(name) Chunking:")
            let chunker = ChunkingStrategyFactory.create(strategy)
            let chunks = chunker.chunk(sampleText, configuration: config)
            
            for (index, chunk) in chunks.enumerated() {
                print("  Chunk \(index + 1): \(chunk.characterCount) chars")
                print("    Preview: \(chunk.content.prefix(50))...")
            }
        }
    }
}


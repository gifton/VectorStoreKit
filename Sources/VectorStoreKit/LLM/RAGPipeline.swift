import Foundation

/// Configuration for RAG pipeline
public struct RAGConfiguration {
    /// Search configuration for retrieval
    public let searchConfig: SemanticSearchConfiguration
    
    /// Chunking configuration for documents
    public let chunkingConfig: ChunkingConfiguration
    
    /// Context window configuration
    public let contextConfig: ContextConfiguration
    
    /// Whether to include metadata in context
    public let includeMetadata: Bool
    
    /// Whether to deduplicate retrieved chunks
    public let deduplicateChunks: Bool
    
    /// Minimum relevance score for inclusion
    public let minRelevanceScore: Float
    
    public init(
        searchConfig: SemanticSearchConfiguration = SemanticSearchConfiguration(),
        chunkingConfig: ChunkingConfiguration = ChunkingConfiguration(),
        contextConfig: ContextConfiguration = ContextConfiguration(),
        includeMetadata: Bool = true,
        deduplicateChunks: Bool = true,
        minRelevanceScore: Float = 0.7
    ) {
        self.searchConfig = searchConfig
        self.chunkingConfig = chunkingConfig
        self.contextConfig = contextConfig
        self.includeMetadata = includeMetadata
        self.deduplicateChunks = deduplicateChunks
        self.minRelevanceScore = minRelevanceScore
    }
}

/// Configuration for context window management
public struct ContextConfiguration {
    /// Maximum tokens in context
    public let maxTokens: Int
    
    /// Reserved tokens for system prompt
    public let systemPromptTokens: Int
    
    /// Reserved tokens for user query
    public let queryTokens: Int
    
    /// Reserved tokens for response
    public let responseTokens: Int
    
    /// Strategy for selecting chunks when over limit
    public let selectionStrategy: SelectionStrategy
    
    public enum SelectionStrategy {
        case topK              // Keep top K most relevant
        case diverse           // Maximize diversity
        case recency           // Prefer recent chunks
        case hybrid            // Combination of strategies
    }
    
    /// Available tokens for context
    public var availableContextTokens: Int {
        maxTokens - systemPromptTokens - queryTokens - responseTokens
    }
    
    public init(
        maxTokens: Int = 4096,
        systemPromptTokens: Int = 500,
        queryTokens: Int = 200,
        responseTokens: Int = 1000,
        selectionStrategy: SelectionStrategy = .hybrid
    ) {
        self.maxTokens = maxTokens
        self.systemPromptTokens = systemPromptTokens
        self.queryTokens = queryTokens
        self.responseTokens = responseTokens
        self.selectionStrategy = selectionStrategy
    }
}

/// Result from RAG retrieval
public struct RAGContext {
    /// Retrieved chunks with relevance scores
    public let chunks: [(chunk: TextChunk, score: Float)]
    
    /// Formatted context string
    public let formattedContext: String
    
    /// Total token count
    public let tokenCount: Int
    
    /// Metadata about retrieval
    public let metadata: RAGMetadata
    
    public struct RAGMetadata {
        public let totalChunksRetrieved: Int
        public let chunksIncluded: Int
        public let averageRelevance: Float
        public let sources: Set<String>
        public let retrievalTime: TimeInterval
    }
}

/// RAG pipeline for retrieval-augmented generation
public actor RAGPipeline {
    private let vectorStore: any LLMVectorStore
    private let embeddingProvider: any EmbeddingProvider
    private let semanticSearch: SemanticSearch
    private let configuration: RAGConfiguration
    
    /// Document storage for full text retrieval
    private var documentStore: [String: Document] = [:]
    
    public struct Document {
        public let id: String
        public let content: String
        public let metadata: [String: Any]
        public let chunks: [TextChunk]
        public let embeddings: [EmbeddingVector]?
        
        public init(
            id: String = UUID().uuidString,
            content: String,
            metadata: [String: Any] = [:],
            chunks: [TextChunk] = [],
            embeddings: [EmbeddingVector]? = nil
        ) {
            self.id = id
            self.content = content
            self.metadata = metadata
            self.chunks = chunks
            self.embeddings = embeddings
        }
    }
    
    public init(
        vectorStore: any LLMVectorStore,
        embeddingProvider: any EmbeddingProvider,
        configuration: RAGConfiguration = RAGConfiguration()
    ) {
        self.vectorStore = vectorStore
        self.embeddingProvider = embeddingProvider
        self.configuration = configuration
        self.semanticSearch = SemanticSearch(
            vectorStore: vectorStore,
            embeddingProvider: embeddingProvider,
            configuration: configuration.searchConfig
        )
    }
    
    /// Index a document for RAG
    public func indexDocument(
        _ content: String,
        metadata: [String: Any] = [:],
        documentId: String? = nil,
        chunkingStrategy: ChunkingStrategyFactory.Strategy = .recursive
    ) async throws -> Document {
        let docId = documentId ?? UUID().uuidString
        
        // Chunk the document
        let chunker = ChunkingStrategyFactory.create(chunkingStrategy)
        var chunks = chunker.chunk(content, configuration: configuration.chunkingConfig)
        
        // Add document metadata to each chunk
        chunks = chunks.map { chunk in
            var updatedChunk = chunk
            updatedChunk.metadata["document_id"] = docId
            updatedChunk.metadata["chunk_index"] = chunk.chunkIndex
            updatedChunk.metadata["total_chunks"] = chunk.totalChunks
            for (key, value) in metadata {
                updatedChunk.metadata[key] = value
            }
            return updatedChunk
        }
        
        // Generate embeddings for chunks
        let chunkTexts = chunks.map { $0.content }
        let embeddings = try await embeddingProvider.embed(chunkTexts)
        
        // Store in vector database
        for (index, (chunk, embedding)) in zip(chunks, embeddings).enumerated() {
            try await vectorStore.add(
                id: chunk.id,
                vector: embedding,
                metadata: chunk.metadata
            )
        }
        
        // Create and store document
        let document = Document(
            id: docId,
            content: content,
            metadata: metadata,
            chunks: chunks,
            embeddings: embeddings
        )
        documentStore[docId] = document
        
        return document
    }
    
    /// Retrieve context for a query
    public func retrieveContext(
        for query: String,
        filter: LLMMetadataFilter? = nil
    ) async throws -> RAGContext {
        let startTime = Date()
        
        // Search for relevant chunks
        let searchResults = try await semanticSearch.search(
            query: query,
            filter: filter,
            configuration: configuration.searchConfig
        )
        
        // Filter by minimum relevance
        let relevantResults = searchResults.filter { $0.score >= configuration.minRelevanceScore }
        
        // Retrieve full chunk information
        var retrievedChunks: [(chunk: TextChunk, score: Float)] = []
        for result in relevantResults {
            if let chunkMetadata = result.metadata["chunk_id"] as? String,
               let documentId = result.metadata["document_id"] as? String,
               let document = documentStore[documentId],
               let chunk = document.chunks.first(where: { $0.id == chunkMetadata }) {
                retrievedChunks.append((chunk: chunk, score: result.score))
            }
        }
        
        // Deduplicate if requested
        if configuration.deduplicateChunks {
            retrievedChunks = deduplicateChunks(retrievedChunks)
        }
        
        // Select chunks based on context window
        let selectedChunks = try selectChunksForContext(
            chunks: retrievedChunks,
            availableTokens: configuration.contextConfig.availableContextTokens
        )
        
        // Format context
        let formattedContext = formatContext(selectedChunks)
        let tokenCount = await estimateTokens(for: formattedContext)
        
        // Calculate metadata
        let averageRelevance = selectedChunks.isEmpty ? 0 :
            selectedChunks.reduce(0) { $0 + $1.score } / Float(selectedChunks.count)
        
        let sources = Set(selectedChunks.compactMap { chunk in
            chunk.chunk.metadata["source"] as? String ??
            chunk.chunk.metadata["document_id"] as? String
        })
        
        let metadata = RAGContext.RAGMetadata(
            totalChunksRetrieved: relevantResults.count,
            chunksIncluded: selectedChunks.count,
            averageRelevance: averageRelevance,
            sources: sources,
            retrievalTime: Date().timeIntervalSince(startTime)
        )
        
        return RAGContext(
            chunks: selectedChunks,
            formattedContext: formattedContext,
            tokenCount: tokenCount,
            metadata: metadata
        )
    }
    
    /// Build a prompt with retrieved context
    public func buildPrompt(
        systemPrompt: String,
        query: String,
        context: RAGContext,
        promptTemplate: String? = nil
    ) -> String {
        let template = promptTemplate ?? """
        You are a helpful assistant with access to relevant context information.
        
        Context:
        {context}
        
        Question: {query}
        
        Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question, say so.
        """
        
        return template
            .replacingOccurrences(of: "{context}", with: context.formattedContext)
            .replacingOccurrences(of: "{query}", with: query)
            .replacingOccurrences(of: "{system}", with: systemPrompt)
    }
    
    /// Complete RAG flow: retrieve context and build prompt
    public func process(
        query: String,
        systemPrompt: String = "",
        filter: LLMMetadataFilter? = nil,
        promptTemplate: String? = nil
    ) async throws -> (prompt: String, context: RAGContext) {
        // Retrieve context
        let context = try await retrieveContext(for: query, filter: filter)
        
        // Build prompt
        let prompt = buildPrompt(
            systemPrompt: systemPrompt,
            query: query,
            context: context,
            promptTemplate: promptTemplate
        )
        
        return (prompt: prompt, context: context)
    }
    
    // MARK: - Private Methods
    
    private func deduplicateChunks(
        _ chunks: [(chunk: TextChunk, score: Float)]
    ) -> [(chunk: TextChunk, score: Float)] {
        var seen = Set<String>()
        var deduplicated: [(chunk: TextChunk, score: Float)] = []
        
        for item in chunks {
            // Create a normalized version for comparison
            let normalized = item.chunk.content
                .lowercased()
                .trimmingCharacters(in: .whitespacesAndNewlines)
            
            // Check for substantial overlap
            var isDuplicate = false
            for seenContent in seen {
                let similarity = calculateSimilarity(normalized, seenContent)
                if similarity > 0.9 {  // 90% similarity threshold
                    isDuplicate = true
                    break
                }
            }
            
            if !isDuplicate {
                seen.insert(normalized)
                deduplicated.append(item)
            }
        }
        
        return deduplicated
    }
    
    private func calculateSimilarity(_ text1: String, _ text2: String) -> Float {
        // Simple Jaccard similarity
        let words1 = Set(text1.split(separator: " "))
        let words2 = Set(text2.split(separator: " "))
        
        let intersection = words1.intersection(words2).count
        let union = words1.union(words2).count
        
        return union > 0 ? Float(intersection) / Float(union) : 0
    }
    
    private func selectChunksForContext(
        chunks: [(chunk: TextChunk, score: Float)],
        availableTokens: Int
    ) throws -> [(chunk: TextChunk, score: Float)] {
        switch configuration.contextConfig.selectionStrategy {
        case .topK:
            return selectTopKChunks(chunks: chunks, availableTokens: availableTokens)
            
        case .diverse:
            return selectDiverseChunks(chunks: chunks, availableTokens: availableTokens)
            
        case .recency:
            return selectRecentChunks(chunks: chunks, availableTokens: availableTokens)
            
        case .hybrid:
            return selectHybridChunks(chunks: chunks, availableTokens: availableTokens)
        }
    }
    
    private func selectTopKChunks(
        chunks: [(chunk: TextChunk, score: Float)],
        availableTokens: Int
    ) -> [(chunk: TextChunk, score: Float)] {
        var selected: [(chunk: TextChunk, score: Float)] = []
        var currentTokens = 0
        
        // Sort by relevance score
        let sorted = chunks.sorted { $0.score > $1.score }
        
        for item in sorted {
            let chunkTokens = item.chunk.estimatedTokens
            if currentTokens + chunkTokens <= availableTokens {
                selected.append(item)
                currentTokens += chunkTokens
            }
        }
        
        return selected
    }
    
    private func selectDiverseChunks(
        chunks: [(chunk: TextChunk, score: Float)],
        availableTokens: Int
    ) -> [(chunk: TextChunk, score: Float)] {
        // Implement diversity-based selection
        // This is a simplified version; could use more sophisticated methods
        var selected: [(chunk: TextChunk, score: Float)] = []
        var currentTokens = 0
        var coveredDocuments = Set<String>()
        
        // Sort by score but prioritize different documents
        let sorted = chunks.sorted { $0.score > $1.score }
        
        for item in sorted {
            let docId = item.chunk.metadata["document_id"] as? String ?? ""
            let chunkTokens = item.chunk.estimatedTokens
            
            if currentTokens + chunkTokens <= availableTokens {
                // Prioritize chunks from new documents
                if !coveredDocuments.contains(docId) || selected.count < 3 {
                    selected.append(item)
                    currentTokens += chunkTokens
                    coveredDocuments.insert(docId)
                }
            }
        }
        
        return selected
    }
    
    private func selectRecentChunks(
        chunks: [(chunk: TextChunk, score: Float)],
        availableTokens: Int
    ) -> [(chunk: TextChunk, score: Float)] {
        // Sort by recency (if timestamp metadata available) then by score
        let sorted = chunks.sorted { lhs, rhs in
            if let time1 = lhs.chunk.metadata["timestamp"] as? Date,
               let time2 = rhs.chunk.metadata["timestamp"] as? Date {
                return time1 > time2
            }
            return lhs.score > rhs.score
        }
        
        return selectTopKChunks(chunks: sorted, availableTokens: availableTokens)
    }
    
    private func selectHybridChunks(
        chunks: [(chunk: TextChunk, score: Float)],
        availableTokens: Int
    ) -> [(chunk: TextChunk, score: Float)] {
        // Combine multiple strategies
        let third = availableTokens / 3
        
        let topK = selectTopKChunks(chunks: chunks, availableTokens: third)
        let diverse = selectDiverseChunks(
            chunks: chunks.filter { item in !topK.contains(where: { $0.chunk.id == item.chunk.id }) },
            availableTokens: third
        )
        let recent = selectRecentChunks(
            chunks: chunks.filter { item in
                !topK.contains(where: { $0.chunk.id == item.chunk.id }) &&
                !diverse.contains(where: { $0.chunk.id == item.chunk.id })
            },
            availableTokens: availableTokens - (topK.count + diverse.count) * 100
        )
        
        return topK + diverse + recent
    }
    
    private func formatContext(_ chunks: [(chunk: TextChunk, score: Float)]) -> String {
        if chunks.isEmpty {
            return "No relevant context found."
        }
        
        var formatted = ""
        
        for (index, item) in chunks.enumerated() {
            if configuration.includeMetadata {
                // Add source information
                if let source = item.chunk.metadata["source"] as? String {
                    formatted += "[Source: \(source)]\n"
                }
                formatted += "[Relevance: \(String(format: "%.2f", item.score))]\n"
            }
            
            formatted += item.chunk.content
            
            if index < chunks.count - 1 {
                formatted += "\n\n---\n\n"
            }
        }
        
        return formatted
    }
    
    private func estimateTokens(for text: String) async -> Int {
        await embeddingProvider.estimateTokens(for: text)
    }
}

// MARK: - Context Manager

/// Manages context windows for different LLM providers
public struct ContextManager {
    public enum LLMProvider {
        case openAI(model: String)
        case anthropic(model: String)
        case custom(maxTokens: Int)
        
        public var maxTokens: Int {
            switch self {
            case .openAI(let model):
                switch model {
                case "gpt-4-turbo", "gpt-4-turbo-preview":
                    return 128000
                case "gpt-4", "gpt-4-0613":
                    return 8192
                case "gpt-3.5-turbo-16k":
                    return 16384
                case "gpt-3.5-turbo":
                    return 4096
                default:
                    return 4096
                }
                
            case .anthropic(let model):
                switch model {
                case "claude-3-opus", "claude-3-sonnet":
                    return 200000
                case "claude-2.1":
                    return 100000
                case "claude-2", "claude-instant":
                    return 100000
                default:
                    return 100000
                }
                
            case .custom(let maxTokens):
                return maxTokens
            }
        }
    }
    
    public static func optimalConfiguration(
        for provider: LLMProvider,
        systemPromptLength: Int = 500
    ) -> ContextConfiguration {
        let maxTokens = provider.maxTokens
        
        // Reserve ~25% for response
        let responseTokens = maxTokens / 4
        
        // Calculate available context tokens
        let contextTokens = maxTokens - systemPromptLength - 200 - responseTokens
        
        return ContextConfiguration(
            maxTokens: maxTokens,
            systemPromptTokens: systemPromptLength,
            queryTokens: 200,
            responseTokens: responseTokens,
            selectionStrategy: .hybrid
        )
    }
}


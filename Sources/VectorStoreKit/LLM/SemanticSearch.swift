import Foundation

/// Configuration for semantic search operations
public struct SemanticSearchConfiguration {
    /// Number of results to return
    public let topK: Int
    
    /// Similarity threshold (0-1, where 1 is identical)
    public let threshold: Float?
    
    /// Whether to use GPU acceleration
    public let useGPUAcceleration: Bool
    
    /// Maximum number of candidates to evaluate
    public let maxCandidates: Int
    
    /// Whether to normalize query vectors
    public let normalizeQuery: Bool
    
    /// Search strategy
    public let strategy: SearchStrategy
    
    public enum SearchStrategy {
        case exact              // Brute force search
        case approximate        // Use indexing (HNSW, IVF, etc.)
        case hybrid            // Combine exact and approximate
    }
    
    public init(
        topK: Int = 10,
        threshold: Float? = nil,
        useGPUAcceleration: Bool = true,
        maxCandidates: Int = 1000,
        normalizeQuery: Bool = true,
        strategy: SearchStrategy = .approximate
    ) {
        self.topK = topK
        self.threshold = threshold
        self.useGPUAcceleration = useGPUAcceleration
        self.maxCandidates = maxCandidates
        self.normalizeQuery = normalizeQuery
        self.strategy = strategy
    }
}

/// Result from semantic search
public struct SemanticSearchResult {
    /// Document/vector ID
    public let id: String
    
    /// Similarity score (0-1, where 1 is most similar)
    public let score: Float
    
    /// Associated metadata
    public let metadata: [String: Any]
    
    /// Original text (if available)
    public let text: String?
    
    /// Distance metric used
    public let metric: EmbeddingDistanceMetric
    
    public init(
        id: String,
        score: Float,
        metadata: [String: Any] = [:],
        text: String? = nil,
        metric: EmbeddingDistanceMetric = .cosine
    ) {
        self.id = id
        self.score = score
        self.metadata = metadata
        self.text = text
        self.metric = metric
    }
}

/// Semantic search engine with Metal acceleration
public actor SemanticSearch {
    private let vectorStore: any LLMVectorStore
    private let embeddingProvider: any EmbeddingProvider
    private let configuration: SemanticSearchConfiguration
    private let accelerationEngine: MetalAccelerationEngine
    
    /// Cache for query embeddings
    private var queryCache: [String: EmbeddingVector] = [:]
    private let maxCacheSize = 1000
    
    public init(
        vectorStore: any LLMVectorStore,
        embeddingProvider: any EmbeddingProvider,
        configuration: SemanticSearchConfiguration = SemanticSearchConfiguration()
    ) {
        self.vectorStore = vectorStore
        self.embeddingProvider = embeddingProvider
        self.configuration = configuration
        self.accelerationEngine = MetalAccelerationEngine.shared
    }
    
    /// Search for similar documents using text query
    public func search(
        query: String,
        filter: LLMMetadataFilter? = nil,
        configuration: SemanticSearchConfiguration? = nil
    ) async throws -> [SemanticSearchResult] {
        let config = configuration ?? self.configuration
        
        // Get or compute query embedding
        let queryVector = try await getQueryEmbedding(query)
        
        // Perform vector search
        return try await searchByVector(
            queryVector: queryVector,
            filter: filter,
            configuration: config
        )
    }
    
    /// Search for similar documents using vector
    public func searchByVector(
        queryVector: EmbeddingVector,
        filter: LLMMetadataFilter? = nil,
        configuration: SemanticSearchConfiguration? = nil
    ) async throws -> [SemanticSearchResult] {
        let config = configuration ?? self.configuration
        
        // Normalize query if requested
        var normalizedQuery = queryVector
        if config.normalizeQuery {
            normalizedQuery.normalize()
        }
        
        // Get candidates based on strategy
        let candidates = try await getCandidates(
            queryVector: normalizedQuery,
            filter: filter,
            configuration: config
        )
        
        // Compute similarities using Metal acceleration
        let similarities = try await computeSimilarities(
            queryVector: normalizedQuery,
            candidates: candidates,
            useGPU: config.useGPUAcceleration
        )
        
        // Filter by threshold if specified
        let filtered = filterByThreshold(
            similarities: similarities,
            threshold: config.threshold
        )
        
        // Sort and take top K
        let topResults = Array(
            filtered
                .sorted { $0.score > $1.score }
                .prefix(config.topK)
        )
        
        return topResults
    }
    
    /// Batch search for multiple queries
    public func batchSearch(
        queries: [String],
        filter: LLMMetadataFilter? = nil,
        configuration: SemanticSearchConfiguration? = nil
    ) async throws -> [[SemanticSearchResult]] {
        let config = configuration ?? self.configuration
        
        // Embed all queries in batch
        let queryVectors = try await embeddingProvider.embed(queries)
        
        // Search for each query
        return try await withThrowingTaskGroup(of: (Int, [SemanticSearchResult]).self) { group in
            for (index, queryVector) in queryVectors.enumerated() {
                group.addTask {
                    let results = try await self.searchByVector(
                        queryVector: queryVector,
                        filter: filter,
                        configuration: config
                    )
                    return (index, results)
                }
            }
            
            // Collect results in order
            var results = Array(repeating: [SemanticSearchResult](), count: queries.count)
            for try await (index, searchResults) in group {
                results[index] = searchResults
            }
            return results
        }
    }
    
    /// Re-rank results using a different embedding model or strategy
    public func rerank(
        query: String,
        results: [SemanticSearchResult],
        embeddingProvider: (any EmbeddingProvider)? = nil
    ) async throws -> [SemanticSearchResult] {
        let provider = embeddingProvider ?? self.embeddingProvider
        
        // Get query embedding
        let queryVector = try await provider.embed(query)
        
        // Get vectors for results
        let resultVectors = try await getResultVectors(results)
        
        // Recompute similarities
        let candidatesWithMetadata = zip(resultVectors, results).map { (vectorPair, result) in
            (id: vectorPair.0, vector: vectorPair.1, metadata: result.metadata)
        }
        let similarities = try await computeSimilarities(
            queryVector: queryVector,
            candidates: candidatesWithMetadata,
            useGPU: configuration.useGPUAcceleration
        )
        
        // Create reranked results
        return similarities.sorted { $0.score > $1.score }
    }
    
    // MARK: - Private Methods
    
    private func getQueryEmbedding(_ query: String) async throws -> EmbeddingVector {
        // Check cache
        if let cached = queryCache[query] {
            return cached
        }
        
        // Compute embedding
        let embedding = try await embeddingProvider.embed(query)
        
        // Update cache (with size limit)
        if queryCache.count >= maxCacheSize {
            // Remove oldest entries (simple FIFO)
            queryCache.removeAll()
        }
        queryCache[query] = embedding
        
        return embedding
    }
    
    private func getCandidates(
        queryVector: EmbeddingVector,
        filter: LLMMetadataFilter?,
        configuration: SemanticSearchConfiguration
    ) async throws -> [(id: String, vector: Vector, metadata: [String: Any])] {
        switch configuration.strategy {
        case .exact:
            // Get all vectors matching filter
            let results = try await vectorStore.getAllVectors(filter: filter)
            return results.map { (id: $0.id, vector: $0.vector.data, metadata: $0.metadata) }
            
        case .approximate:
            // Use index for approximate search
            let searchResults = try await vectorStore.search(
                vector: queryVector,
                k: configuration.maxCandidates,
                filter: filter
            )
            return searchResults.map { result in
                (id: result.id, vector: result.vector.data, metadata: result.metadata)
            }
            
        case .hybrid:
            // Combine exact and approximate
            let approximate = try await vectorStore.search(
                vector: queryVector,
                k: configuration.maxCandidates / 2,
                filter: filter
            )
            
            let exact = try await vectorStore.getAllVectors(filter: filter)
                .prefix(configuration.maxCandidates / 2)
            
            // Merge results
            var combined = approximate.map { result in
                (id: result.id, vector: result.vector.data, metadata: result.metadata)
            }
            combined.append(contentsOf: exact.map { 
                (id: $0.id, vector: $0.vector.data, metadata: $0.metadata) 
            })
            
            return combined
        }
    }
    
    private func computeSimilarities(
        queryVector: EmbeddingVector,
        candidates: [(id: String, vector: Vector, metadata: [String: Any])],
        useGPU: Bool
    ) async throws -> [SemanticSearchResult] {
        guard !candidates.isEmpty else { return [] }
        
        // Prepare candidate vectors
        let candidateVectors = candidates.map { $0.vector }
        
        // Compute similarities using acceleration engine
        let scores: [Float]
        // Always use GPU if requested and available
        let shouldUseGPU = useGPU
        
        if useGPU && shouldUseGPU {
            // Use GPU acceleration
            // Convert vectors to SIMD32<Float> for GPU computation
            let simdQuery = SIMD32<Float>(queryVector.data)
            let simdCandidates = candidateVectors.map { SIMD32<Float>($0) }
            let distances = try await accelerationEngine.computeDistances(
                query: simdQuery,
                candidates: simdCandidates,
                metric: .cosine
            )
            scores = distances.map { 1.0 - $0 }  // Convert distance to similarity
        } else {
            // Use UnifiedDistanceComputation for CPU path
            let unifiedCompute = UnifiedDistanceComputation(preferGPU: false)
            let distances = try await unifiedCompute.computeDistances(
                query: queryVector.data,
                candidates: candidateVectors,
                metric: .cosine
            )
            scores = distances.map { 1.0 - $0 }  // Convert distance to similarity
        }
        
        // Create results
        return zip(candidates, scores).map { candidate, score in
            SemanticSearchResult(
                id: candidate.id,
                score: score,
                metadata: candidate.metadata,
                metric: .cosine
            )
        }
    }
    
    private func filterByThreshold(
        similarities: [SemanticSearchResult],
        threshold: Float?
    ) -> [SemanticSearchResult] {
        guard let threshold = threshold else { return similarities }
        return similarities.filter { $0.score >= threshold }
    }
    
    private func getResultVectors(
        _ results: [SemanticSearchResult]
    ) async throws -> [(String, Vector)] {
        // Fetch vectors for results
        let ids = results.map { $0.id }
        let vectors = try await vectorStore.getVectors(ids: ids)
        
        return zip(ids, vectors).map { ($0, $1.data) }
    }
}

// MARK: - Metadata Filtering

/// Filter for metadata-based search
public struct LLMMetadataFilter {
    public enum Condition {
        case equals(String, Any)
        case notEquals(String, Any)
        case contains(String, String)
        case greaterThan(String, Double)
        case lessThan(String, Double)
        case inList(String, [Any])
        case and([Condition])
        case or([Condition])
    }
    
    public let condition: Condition
    
    public init(_ condition: Condition) {
        self.condition = condition
    }
    
    /// Check if metadata matches filter
    public func matches(_ metadata: [String: Any]) -> Bool {
        return evaluateCondition(condition, metadata: metadata)
    }
    
    private func evaluateCondition(_ condition: Condition, metadata: [String: Any]) -> Bool {
        switch condition {
        case .equals(let key, let value):
            return metadata[key] as? String == value as? String
            
        case .notEquals(let key, let value):
            return metadata[key] as? String != value as? String
            
        case .contains(let key, let substring):
            guard let value = metadata[key] as? String else { return false }
            return value.contains(substring)
            
        case .greaterThan(let key, let threshold):
            guard let value = metadata[key] as? Double else { return false }
            return value > threshold
            
        case .lessThan(let key, let threshold):
            guard let value = metadata[key] as? Double else { return false }
            return value < threshold
            
        case .inList(let key, let list):
            guard let value = metadata[key] else { return false }
            return list.contains { "\($0)" == "\(value)" }
            
        case .and(let conditions):
            return conditions.allSatisfy { evaluateCondition($0, metadata: metadata) }
            
        case .or(let conditions):
            return conditions.contains { evaluateCondition($0, metadata: metadata) }
        }
    }
}


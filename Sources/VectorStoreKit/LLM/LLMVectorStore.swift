import Foundation

/// Simple vector store protocol for LLM components
public protocol LLMVectorStore: Actor {
    /// Add a vector with metadata
    func add(id: String, vector: EmbeddingVector, metadata: [String: Any]) async throws
    
    /// Search for similar vectors
    func search(
        vector: EmbeddingVector,
        k: Int,
        filter: LLMMetadataFilter?
    ) async throws -> [LLMSearchResult]
    
    /// Get all vectors matching a filter
    func getAllVectors(
        filter: LLMMetadataFilter?
    ) async throws -> [(id: String, vector: EmbeddingVector, metadata: [String: Any])]
    
    /// Get vectors by IDs
    func getVectors(ids: [String]) async throws -> [EmbeddingVector]
    
    /// Delete vectors by IDs
    func delete(ids: [String]) async throws
    
    /// Get store statistics
    func statistics() async -> VectorStoreStatistics
}

/// Search result from vector store
public struct LLMSearchResult {
    public let id: String
    public let score: Float
    public let vector: EmbeddingVector
    public let metadata: [String: Any]
    
    public init(
        id: String,
        score: Float,
        vector: EmbeddingVector,
        metadata: [String: Any]
    ) {
        self.id = id
        self.score = score
        self.vector = vector
        self.metadata = metadata
    }
}

/// Vector store statistics
public struct VectorStoreStatistics {
    public let vectorCount: Int
    public let dimensions: Int
    public let indexType: String
    public let memoryUsage: Int
    
    public init(
        vectorCount: Int,
        dimensions: Int,
        indexType: String,
        memoryUsage: Int
    ) {
        self.vectorCount = vectorCount
        self.dimensions = dimensions
        self.indexType = indexType
        self.memoryUsage = memoryUsage
    }
}

/// In-memory vector store implementation for examples and testing
public actor InMemoryVectorStore: LLMVectorStore {
    private var vectors: [String: (EmbeddingVector, [String: Any])] = [:]
    private let metric: EmbeddingDistanceMetric
    
    public init(metric: EmbeddingDistanceMetric = .cosine) {
        self.metric = metric
    }
    
    public func add(id: String, vector: EmbeddingVector, metadata: [String: Any]) async throws {
        vectors[id] = (vector, metadata)
    }
    
    public func search(
        vector: EmbeddingVector,
        k: Int,
        filter: LLMMetadataFilter?
    ) async throws -> [LLMSearchResult] {
        var results: [(String, Float, EmbeddingVector, [String: Any])] = []
        
        for (id, (storedVector, metadata)) in vectors {
            // Apply filter if provided
            if let filter = filter, !filter.matches(metadata) {
                continue
            }
            
            // Compute similarity
            let distance = metric.distance(vector, storedVector)
            let similarity = metric.toSimilarity(distance)
            
            results.append((id, similarity, storedVector, metadata))
        }
        
        // Sort by similarity and take top k
        results.sort { $0.1 > $1.1 }
        
        return results.prefix(k).map { result in
            LLMSearchResult(
                id: result.0,
                score: result.1,
                vector: result.2,
                metadata: result.3
            )
        }
    }
    
    public func getAllVectors(
        filter: LLMMetadataFilter?
    ) async throws -> [(id: String, vector: EmbeddingVector, metadata: [String: Any])] {
        if let filter = filter {
            return vectors.compactMap { (id, value) in
                let (vector, metadata) = value
                return filter.matches(metadata) ? (id, vector, metadata) : nil
            }
        } else {
            return vectors.map { (id, value) in
                let (vector, metadata) = value
                return (id, vector, metadata)
            }
        }
    }
    
    public func getVectors(ids: [String]) async throws -> [EmbeddingVector] {
        ids.compactMap { vectors[$0]?.0 }
    }
    
    public func delete(ids: [String]) async throws {
        for id in ids {
            vectors.removeValue(forKey: id)
        }
    }
    
    public func statistics() async -> VectorStoreStatistics {
        let dimensions = vectors.first?.value.0.dimensions ?? 0
        let memoryUsage = vectors.count * dimensions * MemoryLayout<Float>.size
        
        return VectorStoreStatistics(
            vectorCount: vectors.count,
            dimensions: dimensions,
            indexType: "in-memory",
            memoryUsage: memoryUsage
        )
    }
}
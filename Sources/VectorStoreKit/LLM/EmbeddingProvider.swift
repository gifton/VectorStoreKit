import Foundation

/// Dimensions for common embedding models
public enum EmbeddingDimension: Int, CaseIterable {
    case small = 384        // MiniLM, all-MiniLM-L6-v2
    case medium = 768       // BERT, all-mpnet-base-v2
    case large = 1536       // OpenAI ada-002
    case xlarge = 3072      // OpenAI text-embedding-3-large
    
    /// Model suggestions for each dimension
    public var suggestedModels: [String] {
        switch self {
        case .small:
            return ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2"]
        case .medium:
            return ["all-mpnet-base-v2", "bert-base-uncased"]
        case .large:
            return ["text-embedding-ada-002"]
        case .xlarge:
            return ["text-embedding-3-large"]
        }
    }
}

/// Configuration for embedding providers
public struct EmbeddingConfiguration {
    /// Model identifier
    public let model: String
    
    /// Expected output dimensions
    public let dimensions: Int
    
    /// Maximum tokens per request
    public let maxTokens: Int
    
    /// Whether to normalize embeddings (L2 normalization)
    public let normalize: Bool
    
    /// Batch size for processing
    public let batchSize: Int
    
    /// Timeout for requests
    public let timeout: TimeInterval
    
    public init(
        model: String,
        dimensions: Int,
        maxTokens: Int = 8192,
        normalize: Bool = true,
        batchSize: Int = 100,
        timeout: TimeInterval = 30.0
    ) {
        self.model = model
        self.dimensions = dimensions
        self.maxTokens = maxTokens
        self.normalize = normalize
        self.batchSize = batchSize
        self.timeout = timeout
    }
}

/// Protocol for embedding providers
public protocol EmbeddingProvider: Actor {
    /// Configuration for this provider
    var configuration: EmbeddingConfiguration { get }
    
    /// Generate embeddings for a batch of texts
    func embed(_ texts: [String]) async throws -> [EmbeddingVector]
    
    /// Generate embedding for a single text
    func embed(_ text: String) async throws -> EmbeddingVector
    
    /// Estimate token count for text (provider-specific)
    func estimateTokens(for text: String) -> Int
    
    /// Check if provider is available and configured
    func isAvailable() async -> Bool
}

/// Default implementations
public extension EmbeddingProvider {
    func embed(_ text: String) async throws -> EmbeddingVector {
        let results = try await embed([text])
        guard let result = results.first else {
            throw EmbeddingError.processingFailed("No embedding returned for text")
        }
        return result
    }
    
    func estimateTokens(for text: String) -> Int {
        // Simple estimation: ~4 characters per token
        return (text.count + 3) / 4
    }
}

/// Errors specific to embedding operations
public enum EmbeddingError: Error, LocalizedError {
    case providerNotConfigured(String)
    case dimensionMismatch(expected: Int, actual: Int)
    case textTooLong(tokens: Int, maxTokens: Int)
    case batchTooLarge(size: Int, maxSize: Int)
    case processingFailed(String)
    case networkError(Error)
    case rateLimitExceeded(retryAfter: TimeInterval?)
    
    public var errorDescription: String? {
        switch self {
        case .providerNotConfigured(let details):
            return "Embedding provider not configured: \(details)"
        case .dimensionMismatch(let expected, let actual):
            return "Dimension mismatch: expected \(expected), got \(actual)"
        case .textTooLong(let tokens, let maxTokens):
            return "Text too long: \(tokens) tokens exceeds limit of \(maxTokens)"
        case .batchTooLarge(let size, let maxSize):
            return "Batch too large: \(size) texts exceeds limit of \(maxSize)"
        case .processingFailed(let reason):
            return "Embedding processing failed: \(reason)"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .rateLimitExceeded(let retryAfter):
            if let retryAfter = retryAfter {
                return "Rate limit exceeded. Retry after \(retryAfter) seconds"
            }
            return "Rate limit exceeded"
        }
    }
}

/// Result of an embedding operation with metadata
public struct EmbeddingResult {
    /// The generated embedding vector
    public let vector: EmbeddingVector
    
    /// Original text that was embedded
    public let text: String
    
    /// Token count for the text
    public let tokenCount: Int
    
    /// Processing time in seconds
    public let processingTime: TimeInterval
    
    /// Provider-specific metadata
    public let metadata: [String: Any]
    
    public init(
        vector: EmbeddingVector,
        text: String,
        tokenCount: Int,
        processingTime: TimeInterval,
        metadata: [String: Any] = [:]
    ) {
        self.vector = vector
        self.text = text
        self.tokenCount = tokenCount
        self.processingTime = processingTime
        self.metadata = metadata
    }
}

/// Batch embedding result
public struct BatchEmbeddingResult {
    /// Successfully embedded texts
    public let results: [EmbeddingResult]
    
    /// Failed texts with errors
    public let failures: [(text: String, error: Error)]
    
    /// Total processing time
    public let totalTime: TimeInterval
    
    /// Average tokens per text
    public var averageTokens: Double {
        guard !results.isEmpty else { return 0 }
        let totalTokens = results.reduce(0) { $0 + $1.tokenCount }
        return Double(totalTokens) / Double(results.count)
    }
    
    /// Success rate
    public var successRate: Double {
        let total = results.count + failures.count
        guard total > 0 else { return 1.0 }
        return Double(results.count) / Double(total)
    }
}
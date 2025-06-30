import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// OpenAI embedding provider implementation
public actor OpenAIEmbeddingProvider: EmbeddingProvider {
    public let configuration: EmbeddingConfiguration
    private let apiKey: String
    private let baseURL: URL
    private let session: URLSession
    
    /// Rate limiting state
    private var lastRequestTime: Date = .distantPast
    private var requestsInWindow: Int = 0
    private let rateLimitWindow: TimeInterval = 60.0
    private let maxRequestsPerMinute: Int = 3000
    
    /// Initialize with API key and configuration
    public init(
        apiKey: String,
        model: String = "text-embedding-ada-002",
        dimensions: Int? = nil,
        baseURL: URL = URL(string: "https://api.openai.com/v1/embeddings")!
    ) {
        self.apiKey = apiKey
        self.baseURL = baseURL
        
        // Determine dimensions based on model if not specified
        let modelDimensions = dimensions ?? Self.dimensionsForModel(model)
        
        self.configuration = EmbeddingConfiguration(
            model: model,
            dimensions: modelDimensions,
            maxTokens: 8191,  // OpenAI limit
            normalize: true,
            batchSize: 100    // OpenAI supports up to 2048, but 100 is reasonable
        )
        
        // Configure session with custom timeout
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = configuration.timeout
        config.timeoutIntervalForResource = configuration.timeout * 2
        self.session = URLSession(configuration: config)
    }
    
    /// Get dimensions for known OpenAI models
    private static func dimensionsForModel(_ model: String) -> Int {
        switch model {
        case "text-embedding-ada-002":
            return 1536
        case "text-embedding-3-small":
            return 1536
        case "text-embedding-3-large":
            return 3072
        default:
            return 1536  // Default to ada-002 dimensions
        }
    }
    
    public func embed(_ texts: [String]) async throws -> [EmbeddingVector] {
        // Validate batch size
        guard texts.count <= configuration.batchSize else {
            throw EmbeddingError.batchTooLarge(
                size: texts.count,
                maxSize: configuration.batchSize
            )
        }
        
        // Check rate limits
        try await enforceRateLimit(requestCount: texts.count)
        
        // Validate text lengths
        for text in texts {
            let tokens = estimateTokens(for: text)
            if tokens > configuration.maxTokens {
                throw EmbeddingError.textTooLong(
                    tokens: tokens,
                    maxTokens: configuration.maxTokens
                )
            }
        }
        
        // Prepare request
        var request = URLRequest(url: baseURL)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody = OpenAIEmbeddingRequest(
            input: texts,
            model: configuration.model,
            dimensions: configuration.model.contains("3-") ? configuration.dimensions : nil
        )
        
        request.httpBody = try JSONEncoder().encode(requestBody)
        
        // Make request
        let startTime = Date()
        let (data, response) = try await session.data(for: request)
        let processingTime = Date().timeIntervalSince(startTime)
        
        // Check response
        guard let httpResponse = response as? HTTPURLResponse else {
            throw EmbeddingError.networkError(
                URLError(.badServerResponse)
            )
        }
        
        // Handle rate limiting
        if httpResponse.statusCode == 429 {
            let retryAfter = httpResponse.value(forHTTPHeaderField: "Retry-After")
                .flatMap { Double($0) }
            throw EmbeddingError.rateLimitExceeded(retryAfter: retryAfter)
        }
        
        guard httpResponse.statusCode == 200 else {
            let errorMessage = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw EmbeddingError.processingFailed(
                "HTTP \(httpResponse.statusCode): \(errorMessage)"
            )
        }
        
        // Parse response
        let embeddingResponse = try JSONDecoder().decode(
            OpenAIEmbeddingResponse.self,
            from: data
        )
        
        // Convert to vectors
        let vectors = try embeddingResponse.data.map { embedding in
            let floats = embedding.embedding.map { Float($0) }
            var vector = EmbeddingVector(dimensions: floats.count)
            vector.data = floats
            
            // Normalize if requested
            if configuration.normalize {
                vector.normalize()
            }
            
            // Validate dimensions
            guard vector.dimensions == configuration.dimensions else {
                throw EmbeddingError.dimensionMismatch(
                    expected: configuration.dimensions,
                    actual: vector.dimensions
                )
            }
            
            return vector
        }
        
        // Update rate limit tracking
        await updateRateLimitTracking(requestCount: texts.count)
        
        return vectors
    }
    
    public func estimateTokens(for text: String) -> Int {
        // OpenAI uses ~1 token per 4 characters for English text
        // This is a rough estimate; actual tokenization is more complex
        let baseEstimate = (text.count + 3) / 4
        
        // Add some buffer for special tokens
        return baseEstimate + 10
    }
    
    public func isAvailable() async -> Bool {
        // Check if API key is present
        guard !apiKey.isEmpty else { return false }
        
        // Could add a ping endpoint check here
        return true
    }
    
    /// Enforce rate limiting
    private func enforceRateLimit(requestCount: Int) async throws {
        let now = Date()
        
        // Reset window if needed
        if now.timeIntervalSince(lastRequestTime) > rateLimitWindow {
            requestsInWindow = 0
            lastRequestTime = now
        }
        
        // Check if we would exceed rate limit
        if requestsInWindow + requestCount > maxRequestsPerMinute {
            let waitTime = rateLimitWindow - now.timeIntervalSince(lastRequestTime)
            throw EmbeddingError.rateLimitExceeded(retryAfter: waitTime)
        }
    }
    
    /// Update rate limit tracking after successful request
    private func updateRateLimitTracking(requestCount: Int) async {
        requestsInWindow += requestCount
    }
}

// MARK: - OpenAI API Types

private struct OpenAIEmbeddingRequest: Codable {
    let input: [String]
    let model: String
    let dimensions: Int?
    
    enum CodingKeys: String, CodingKey {
        case input
        case model
        case dimensions
    }
}

private struct OpenAIEmbeddingResponse: Codable {
    let object: String
    let data: [EmbeddingData]
    let model: String
    let usage: Usage
    
    struct EmbeddingData: Codable {
        let object: String
        let index: Int
        let embedding: [Double]
    }
    
    struct Usage: Codable {
        let promptTokens: Int
        let totalTokens: Int
        
        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case totalTokens = "total_tokens"
        }
    }
}

// MARK: - Factory Method

public extension OpenAIEmbeddingProvider {
    /// Create provider from environment variables
    static func fromEnvironment(
        model: String = "text-embedding-ada-002"
    ) async -> OpenAIEmbeddingProvider? {
        guard let apiKey = ProcessInfo.processInfo.environment["OPENAI_API_KEY"] else {
            return nil
        }
        
        let baseURL = ProcessInfo.processInfo.environment["OPENAI_BASE_URL"]
            .flatMap { URL(string: $0) }
            ?? URL(string: "https://api.openai.com/v1/embeddings")!
        
        return OpenAIEmbeddingProvider(
            apiKey: apiKey,
            model: model,
            baseURL: baseURL
        )
    }
}
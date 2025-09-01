import XCTest
@testable import VectorStoreKit

final class EmbeddingProviderTests: XCTestCase {
    
    func testEmbeddingDimensions() {
        XCTAssertEqual(EmbeddingDimension.small.rawValue, 384)
        XCTAssertEqual(EmbeddingDimension.medium.rawValue, 768)
        XCTAssertEqual(EmbeddingDimension.large.rawValue, 1536)
        XCTAssertEqual(EmbeddingDimension.xlarge.rawValue, 3072)
    }
    
    func testEmbeddingConfiguration() {
        let config = EmbeddingConfiguration(
            model: "test-model",
            dimensions: 768,
            maxTokens: 512,
            normalize: true,
            batchSize: 32
        )
        
        XCTAssertEqual(config.model, "test-model")
        XCTAssertEqual(config.dimensions, 768)
        XCTAssertEqual(config.maxTokens, 512)
        XCTAssertTrue(config.normalize)
        XCTAssertEqual(config.batchSize, 32)
    }
    
    func testEmbeddingError() {
        let error1 = EmbeddingError.dimensionMismatch(expected: 768, actual: 384)
        XCTAssertTrue(error1.errorDescription?.contains("768") ?? false)
        XCTAssertTrue(error1.errorDescription?.contains("384") ?? false)
        
        let error2 = EmbeddingError.textTooLong(tokens: 1000, maxTokens: 512)
        XCTAssertTrue(error2.errorDescription?.contains("1000") ?? false)
        XCTAssertTrue(error2.errorDescription?.contains("512") ?? false)
    }
    
    func testOpenAIEmbeddingProvider() async throws {
        // Test initialization
        let provider = OpenAIEmbeddingProvider(
            apiKey: "test-key",
            model: "text-embedding-ada-002"
        )
        
        let config = await provider.configuration
        XCTAssertEqual(config.model, "text-embedding-ada-002")
        XCTAssertEqual(config.dimensions, 1536)
        XCTAssertEqual(config.maxTokens, 8191)
        
        // Test token estimation
        let tokens = await provider.estimateTokens(for: "Hello world test")
        XCTAssertGreaterThan(tokens, 0)
    }
    
    func testLocalEmbeddingProvider() async throws {
        // Create mock encoder
        let encoder = try await OptimizedVectorEncoder(
            config: OptimizedVectorEncoder.Config(
                inputDimensions: 768,
                outputDimensions: 384,
                hiddenLayers: [512],
                activation: .relu
            )
        )
        
        let provider = LocalEmbeddingProvider(
            encoder: encoder,
            dimensions: 384
        )
        
        let config = await provider.configuration
        XCTAssertEqual(config.dimensions, 384)
        XCTAssertEqual(config.model, "local-bert-base")
        
        // Test availability
        let isAvailable = await provider.isAvailable()
        XCTAssertTrue(isAvailable)
    }
    
    func testHybridEmbeddingProvider() async throws {
        // Create mock providers
        let primary = MockEmbeddingProvider(dimensions: 768, shouldFail: true)
        let fallback = MockEmbeddingProvider(dimensions: 768, shouldFail: false)
        
        let hybrid = await HybridEmbeddingProvider(
            primary: primary,
            fallback: fallback
        )
        
        // Test that it falls back when primary fails
        let embeddings = try await hybrid.embed(["test text"])
        XCTAssertEqual(embeddings.count, 1)
        XCTAssertEqual(embeddings[0].dimensions, 768)
        
        // Verify it used fallback
        let fallbackCallCount = await fallback.embedCallCount
        XCTAssertEqual(fallbackCallCount, 1)
    }
    
    func testEmbeddingCache() async throws {
        let provider = MockEmbeddingProvider(dimensions: 384)
        let cache = EmbeddingCache(provider: provider, maxSize: 2)
        
        // First call should compute
        let embedding1 = try await cache.embed("test1")
        XCTAssertEqual(embedding1.dimensions, 384)
        
        // Second call with same text should use cache
        let embedding2 = try await cache.embed("test1")
        XCTAssertEqual(embedding2.dimensions, 384)
        
        // Verify provider was only called once
        let callCount = await provider.embedCallCount
        XCTAssertEqual(callCount, 1)
        
        // Test cache eviction
        _ = try await cache.embed("test2")
        _ = try await cache.embed("test3") // Should evict "test1"
        
        // Now "test1" should require recomputation
        _ = try await cache.embed("test1")
        let finalCallCount = await provider.embedCallCount
        XCTAssertEqual(finalCallCount, 4) // Initial + test2 + test3 + test1 again
    }
}

// MARK: - Mock Embedding Provider

actor MockEmbeddingProvider: EmbeddingProvider {
    let configuration: EmbeddingConfiguration
    private let shouldFail: Bool
    private(set) var embedCallCount = 0
    
    init(dimensions: Int, shouldFail: Bool = false) {
        self.configuration = EmbeddingConfiguration(
            model: "mock",
            dimensions: dimensions
        )
        self.shouldFail = shouldFail
    }
    
    func embed(_ texts: [String]) async throws -> [EmbeddingVector] {
        embedCallCount += 1
        
        if shouldFail {
            throw EmbeddingError.processingFailed("Mock failure")
        }
        
        return texts.map { _ in
            var vector = EmbeddingVector(dimensions: configuration.dimensions)
            for i in 0..<configuration.dimensions {
                vector.data[i] = Float.random(in: -1...1)
            }
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
        !shouldFail
    }
}
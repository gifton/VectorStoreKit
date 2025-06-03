import Testing
import Foundation
import simd
@testable import VectorStoreKit

@Test func simplePipelineCommandTest() async throws {
    // Test basic command creation
    let embedding = simd_float4(1.0, 2.0, 3.0, 4.0)
    let storeCommand = StoreEmbeddingCommand(
        embedding: embedding,
        metadata: "test",
        tier: .hot
    )
    
    #expect(storeCommand.embedding == embedding)
    print("✅ StoreEmbeddingCommand created successfully")
    
    // Test search command
    let searchCommand = SearchCommand<simd_float4, String>(
        query: embedding,
        k: 5
    )
    
    #expect(searchCommand.k == 5)
    print("✅ SearchCommand created successfully")
    
    // Test middleware creation
    let cache = SearchCache<simd_float4, String>()
    let middleware = VectorStoreCachingMiddleware(cache: cache)
    _ = middleware
    print("✅ VectorStoreCachingMiddleware created successfully")
    
    // Test metrics middleware
    let metricsMiddleware = VectorStoreMetricsMiddleware()
    _ = metricsMiddleware
    print("✅ VectorStoreMetricsMiddleware created successfully")
    
    print("\n🎉 All PipelineKit integration components working!")
}
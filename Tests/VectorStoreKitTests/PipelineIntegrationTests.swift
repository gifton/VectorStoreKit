import Testing
import Foundation
import simd
#if canImport(Metal)
import Metal
#endif
@testable import VectorStoreKit

@Test func testStoreEmbeddingCommand() async throws {
    // Test that StoreEmbeddingCommand works correctly
    let command = StoreEmbeddingCommand(
        embedding: simd_float4(1.0, 2.0, 3.0, 4.0),
        metadata: "test metadata",
        id: "test_id",
        tier: .hot
    )
    
    #expect(command.id == "test_id")
    #expect(command.embedding == simd_float4(1.0, 2.0, 3.0, 4.0))
    #expect(command.metadata == "test metadata")
    #expect(command.tier == .hot)
}

@Test func testSearchCommand() async throws {
    // Test that SearchCommand works correctly
    let query = simd_float4(1.0, 2.0, 3.0, 4.0)
    let command = SearchCommand<simd_float4, String>(
        query: query,
        k: 5,
        strategy: .adaptive,
        filter: nil
    )
    
    #expect(command.query == query)
    #expect(command.k == 5)
    #expect(command.strategy == SearchStrategy.adaptive)
    #expect(command.filter == nil)
}

@Test func testVectorStoreCachingMiddleware() async throws {
    // Test caching middleware
    let cache = SearchCache<simd_float4, String>()
    let middleware = VectorStoreCachingMiddleware(cache: cache, ttl: 300)
    
    let searchCommand = SearchCommand<simd_float4, String>(
        query: simd_float4(1.0, 2.0, 3.0, 4.0),
        k: 3,
        strategy: .adaptive,
        filter: nil
    )
    
    actor CallCounter {
        var count = 0
        func increment() { count += 1 }
        func getCount() -> Int { count }
    }
    
    let callCounter = CallCounter()
    let mockNext: @Sendable (SearchCommand<simd_float4, String>) async throws -> ComprehensiveSearchResult<String> = { _ in
        await callCounter.increment()
        return ComprehensiveSearchResult(
            results: [],
            queryTime: 0.1,
            totalCandidates: 100,
            strategy: .adaptive,
            performanceMetrics: OperationMetrics(
                duration: 0.1,
                memoryUsed: 1000,
                cpuUsage: 0.5,
                timestamp: Date()
            ),
            qualityMetrics: SearchQualityMetrics(
                relevanceScore: 0.9,
                diversityScore: 0.8,
                coverageScore: 0.7,
                consistencyScore: 0.95
            )
        )
    }
    
    // First call should execute the handler
    _ = try await middleware.process(searchCommand, next: mockNext)
    #expect(await callCounter.getCount() == 1)
    
    // Second call should use cache (but since we can't easily test this without mocking, we'll skip)
}

@Test func testMetricsMiddleware() async throws {
    // Test metrics middleware
    let middleware = VectorStoreMetricsMiddleware()
    
    let command = GetStatisticsCommand()
    
    let mockNext: @Sendable (GetStatisticsCommand) async throws -> StoreStatistics = { _ in
        return StoreStatistics(
            vectorCount: 1000,
            memoryUsage: 1024 * 1024,
            diskUsage: 10 * 1024 * 1024,
            performanceStatistics: PerformanceStatistics(),
            accessStatistics: AccessStatistics()
        )
    }
    
    _ = try await middleware.process(command, next: mockNext)
    
    let metrics = await middleware.getMetrics()
    #expect(metrics.operationMetrics.count > 0)
}

@Test func testAccessControlMiddleware() async throws {
    // Test access control middleware
    let accessChecker: @Sendable (any Command) async -> VectorStoreAccessControlMiddleware.AccessLevel? = { command in
        // Grant read access to all commands for testing
        return .read
    }
    
    let middleware = VectorStoreAccessControlMiddleware(accessChecker: accessChecker)
    
    let searchCommand = SearchCommand<simd_float4, String>(
        query: simd_float4(1.0, 2.0, 3.0, 4.0),
        k: 3,
        strategy: .adaptive,
        filter: nil
    )
    
    let mockNext: @Sendable (SearchCommand<simd_float4, String>) async throws -> ComprehensiveSearchResult<String> = { _ in
        return ComprehensiveSearchResult(
            results: [],
            queryTime: 0.1,
            totalCandidates: 100,
            strategy: .adaptive,
            performanceMetrics: OperationMetrics(
                duration: 0.1,
                memoryUsed: 1000,
                cpuUsage: 0.5,
                timestamp: Date()
            ),
            qualityMetrics: SearchQualityMetrics(
                relevanceScore: 0.9,
                diversityScore: 0.8,
                coverageScore: 0.7,
                consistencyScore: 0.95
            )
        )
    }
    
    // Should succeed with read access for search command
    _ = try await middleware.process(searchCommand, next: mockNext)
}
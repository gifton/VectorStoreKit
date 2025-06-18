// BatchIndexingTests.swift
// VectorStoreKit Tests
//
// Tests for batch indexing operations with transaction support

import XCTest
@testable import VectorStoreKit
import simd

final class BatchIndexingTests: XCTestCase {
    
    var batchProcessor: BatchProcessor!
    var memoryPool: MemoryPoolManager!
    var metalCompute: MetalCompute!
    
    override func setUp() async throws {
        try await super.setUp()
        
        metalCompute = try await MetalCompute()
        memoryPool = MemoryPoolManager(
            configuration: .init(
                maxPoolSize: 100_000_000, // 100MB for tests
                allocationStrategy: .bestFit,
                defragmentationThreshold: 0.3
            )
        )
        
        batchProcessor = BatchProcessor(
            configuration: .init(
                optimalBatchSize: 100,
                maxConcurrentBatches: 2,
                useMetalAcceleration: true,
                enableProgressTracking: true,
                memoryLimit: 50_000_000 // 50MB
            ),
            metalCompute: metalCompute,
            memoryPool: memoryPool
        )
    }
    
    override func tearDown() async throws {
        batchProcessor = nil
        memoryPool = nil
        metalCompute = nil
        try await super.tearDown()
    }
    
    // MARK: - Basic Indexing Tests
    
    func testBasicBatchIndexing() async throws {
        // Create test index
        let index = MockVectorIndex<Vector512, TestMetadata>()
        
        // Generate test entries
        let entries = generateTestEntries(count: 1000)
        
        // Perform batch indexing
        let result = try await batchProcessor.indexVectors(
            entries: entries,
            into: index,
            options: .default
        )
        
        // Verify results
        XCTAssertEqual(result.totalEntries, 1000)
        XCTAssertEqual(result.successfulInserts, 1000)
        XCTAssertEqual(result.failedInserts, 0)
        XCTAssertGreaterThan(result.throughput, 0)
        XCTAssertFalse(result.transactionId.isEmpty)
        
        // Verify index state
        await XCTAssertEqualAsync(await index.count, 1000)
    }
    
    func testBatchIndexingWithValidation() async throws {
        let index = MockVectorIndex<Vector512, TestMetadata>()
        
        // Create entries with some invalid data
        var entries = generateTestEntries(count: 100)
        entries[50] = VectorEntry(
            id: "", // Invalid empty ID
            vector: Vector512(Array(repeating: 0, count: 512)),
            metadata: TestMetadata(value: "invalid"),
            tier: .hot
        )
        
        // Test with validation enabled
        let result = try await batchProcessor.indexVectors(
            entries: entries,
            into: index,
            options: .init(
                validateEntries: true,
                allowPartialSuccess: true,
                skipDuplicates: true
            )
        )
        
        XCTAssertEqual(result.successfulInserts, 99)
        XCTAssertEqual(result.failedInserts, 1)
        XCTAssertEqual(result.failures.count, 1)
        XCTAssertEqual(result.failures[0].id, "")
    }
    
    // MARK: - Transaction Tests
    
    func testTransactionCommit() async throws {
        let index = MockVectorIndex<Vector512, TestMetadata>()
        let entries = generateTestEntries(count: 500)
        
        let result = try await batchProcessor.indexVectors(
            entries: entries,
            into: index,
            options: .default
        )
        
        XCTAssertEqual(result.successfulInserts, 500)
        await XCTAssertEqualAsync(await index.count, 500)
        
        // Verify all entries were committed
        for entry in entries {
            await XCTAssertTrueAsync(await index.contains(id: entry.id))
        }
    }
    
    func testTransactionRollback() async throws {
        let index = MockVectorIndex<Vector512, TestMetadata>()
        var entries = generateTestEntries(count: 100)
        
        // Add invalid entry that will cause failure
        entries[50] = VectorEntry(
            id: "",
            vector: Vector512(Array(repeating: Float.nan, count: 512)), // NaN values
            metadata: TestMetadata(value: "invalid"),
            tier: .hot
        )
        
        // Attempt indexing without allowing partial success
        do {
            _ = try await batchProcessor.indexVectors(
                entries: entries,
                into: index,
                options: .init(
                    validateEntries: true,
                    allowPartialSuccess: false // Force rollback
                )
            )
            XCTFail("Should have thrown error")
        } catch {
            // Expected error
        }
        
        // Verify rollback - no entries should be in index
        await XCTAssertEqualAsync(await index.count, 0)
    }
    
    // MARK: - Concurrent Indexing Tests
    
    func testConcurrentMultiIndexBuilding() async throws {
        let index1 = MockVectorIndex<Vector512, TestMetadata>()
        let index2 = MockVectorIndex<Vector512, TestMetadata>()
        let entries = generateTestEntries(count: 500)
        
        let results = try await batchProcessor.buildIndexesConcurrently(
            entries: entries,
            indexes: [index1, index2],
            options: .default
        )
        
        XCTAssertEqual(results.count, 2)
        
        for (_, result) in results {
            XCTAssertEqual(result.successfulInserts, 500)
            XCTAssertEqual(result.failedInserts, 0)
        }
        
        await XCTAssertEqualAsync(await index1.count, 500)
        await XCTAssertEqualAsync(await index2.count, 500)
    }
    
    // MARK: - Performance Tests
    
    func testIndexingPerformance() async throws {
        let index = MockVectorIndex<Vector512, TestMetadata>()
        let entries = generateTestEntries(count: 10_000)
        
        await measureAsync(
            metrics: [XCTClockMetric(), XCTMemoryMetric()],
            options: XCTMeasureOptions.default
        ) {
            _ = try await batchProcessor.indexVectors(
                entries: entries,
                into: index,
                options: .fast
            )
        }
    }
    
    func testMemoryLimitRespected() async throws {
        let index = MockVectorIndex<Vector512, TestMetadata>()
        
        // Create batch processor with very small memory limit
        let limitedProcessor = BatchProcessor(
            configuration: .init(
                optimalBatchSize: 10,
                maxConcurrentBatches: 1,
                memoryLimit: 10_000 // 10KB - very small
            ),
            memoryPool: MemoryPoolManager(
                configuration: .init(
                    maxPoolSize: 10_000,
                    allocationStrategy: .bestFit
                )
            )
        )
        
        // Should still succeed but process in smaller batches
        let entries = generateTestEntries(count: 100)
        let result = try await limitedProcessor.indexVectors(
            entries: entries,
            into: index,
            options: .default
        )
        
        XCTAssertEqual(result.successfulInserts, 100)
        XCTAssertLessThanOrEqual(result.memoryPeakUsage, 10_000)
    }
    
    // MARK: - Edge Cases
    
    func testEmptyBatchIndexing() async throws {
        let index = MockVectorIndex<Vector512, TestMetadata>()
        let entries: [VectorEntry<Vector512, TestMetadata>] = []
        
        let result = try await batchProcessor.indexVectors(
            entries: entries,
            into: index,
            options: .default
        )
        
        XCTAssertEqual(result.totalEntries, 0)
        XCTAssertEqual(result.successfulInserts, 0)
        XCTAssertEqual(result.failedInserts, 0)
    }
    
    func testDuplicateHandling() async throws {
        let index = MockVectorIndex<Vector512, TestMetadata>()
        
        // Create entries with duplicates
        var entries = generateTestEntries(count: 100)
        entries.append(entries[0]) // Add duplicate
        
        // Test with skip duplicates
        let result = try await batchProcessor.indexVectors(
            entries: entries,
            into: index,
            options: .init(skipDuplicates: true)
        )
        
        XCTAssertEqual(result.successfulInserts, 100) // Original count
        await XCTAssertEqualAsync(await index.count, 100)
    }
    
    func testRetryLogic() async throws {
        let index = MockVectorIndex<Vector512, TestMetadata>()
        index.failureRate = 0.5 // 50% failure rate for first attempt
        
        let entries = generateTestEntries(count: 10)
        
        let result = try await batchProcessor.indexVectors(
            entries: entries,
            into: index,
            options: .init(maxRetries: 3)
        )
        
        // With retries, most should succeed
        XCTAssertGreaterThan(result.successfulInserts, 5)
    }
    
    // MARK: - Helper Methods
    
    private func generateTestEntries(count: Int) -> [VectorEntry<Vector512, TestMetadata>] {
        (0..<count).map { i in
            VectorEntry(
                id: "test_\(i)",
                vector: Vector512(Array(repeating: Float(i) / Float(count), count: 512)),
                metadata: TestMetadata(value: "test_\(i)"),
                tier: i % 10 == 0 ? .hot : .warm
            )
        }
    }
}

// MARK: - Mock Types

private struct TestMetadata: Codable, Sendable {
    let value: String
}

private actor MockVectorIndex<V: SIMD & Sendable, M: Codable & Sendable>: VectorIndex 
where V.Scalar: BinaryFloatingPoint {
    typealias Vector = V
    typealias Metadata = M
    typealias Configuration = MockIndexConfiguration
    typealias Statistics = MockIndexStatistics
    
    private var storage: [String: VectorEntry<V, M>] = [:]
    var failureRate: Double = 0.0
    private var attemptCounts: [String: Int] = [:]
    
    var count: Int { storage.count }
    var capacity: Int { 1_000_000 }
    var memoryUsage: Int { storage.count * MemoryLayout<V>.size }
    var configuration: Configuration { MockIndexConfiguration() }
    var isOptimized: Bool { false }
    
    func insert(_ entry: VectorEntry<V, M>) async throws -> InsertResult {
        let attemptCount = attemptCounts[entry.id, default: 0]
        attemptCounts[entry.id] = attemptCount + 1
        
        // Simulate transient failures for retry testing
        if attemptCount == 0 && Double.random(in: 0...1) < failureRate {
            throw MockError.transientFailure
        }
        
        let isNew = storage[entry.id] == nil
        storage[entry.id] = entry
        
        return InsertResult(
            success: true,
            insertTime: 0.001,
            memoryImpact: MemoryLayout<V>.size,
            indexReorganization: false
        )
    }
    
    func search(
        query: V,
        k: Int,
        strategy: SearchStrategy,
        filter: SearchFilter?
    ) async throws -> [SearchResult<M>] {
        []
    }
    
    func update(id: String, vector: V?, metadata: M?) async throws -> Bool {
        guard var entry = storage[id] else { return false }
        if let vector = vector { entry.vector = vector }
        if let metadata = metadata { entry.metadata = metadata }
        storage[id] = entry
        return true
    }
    
    func delete(id: String) async throws -> Bool {
        storage.removeValue(forKey: id) != nil
    }
    
    func contains(id: String) async -> Bool {
        storage[id] != nil
    }
    
    func optimize(strategy: OptimizationStrategy) async throws {}
    func compact() async throws {}
    func statistics() async -> Statistics { MockIndexStatistics() }
    func validateIntegrity() async throws -> IntegrityReport { IntegrityReport(isValid: true, issues: []) }
    func export(format: ExportFormat) async throws -> Data { Data() }
    func `import`(data: Data, format: ExportFormat) async throws {}
    func analyzeDistribution() async -> DistributionAnalysis { DistributionAnalysis() }
    func performanceProfile() async -> PerformanceProfile { PerformanceProfile() }
    func visualizationData() async -> VisualizationData { VisualizationData() }
}

private struct MockIndexConfiguration: IndexConfiguration {
    var identifier: String { "mock" }
    var memoryFootprint: Int { 0 }
    var supportsConcurrentReads: Bool { true }
    var supportsIncrementalIndexing: Bool { true }
}

private struct MockIndexStatistics: IndexStatistics {
    var averageSearchTime: TimeInterval { 0.001 }
    var indexEfficiency: Double { 0.9 }
    var customMetrics: [String: Any] { [:] }
}

private enum MockError: Error {
    case transientFailure
}

// MARK: - Test Helpers

extension XCTestCase {
    func XCTAssertEqualAsync<T: Equatable>(
        _ expression1: @autoclosure () async throws -> T,
        _ expression2: @autoclosure () async throws -> T,
        _ message: @autoclosure () -> String = "",
        file: StaticString = #filePath,
        line: UInt = #line
    ) async rethrows {
        let value1 = try await expression1()
        let value2 = try await expression2()
        XCTAssertEqual(value1, value2, message(), file: file, line: line)
    }
    
    func XCTAssertTrueAsync(
        _ expression: @autoclosure () async throws -> Bool,
        _ message: @autoclosure () -> String = "",
        file: StaticString = #filePath,
        line: UInt = #line
    ) async rethrows {
        let value = try await expression()
        XCTAssertTrue(value, message(), file: file, line: line)
    }
    
    func measureAsync(
        metrics: [XCTMetric] = [XCTClockMetric()],
        options: XCTMeasureOptions = XCTMeasureOptions.default,
        block: () async throws -> Void
    ) async rethrows {
        measure(metrics: metrics, options: options) {
            let expectation = expectation(description: "async measurement")
            Task {
                try await block()
                expectation.fulfill()
            }
            wait(for: [expectation], timeout: 60)
        }
    }
}
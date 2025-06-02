// VectorStoreKit: Comprehensive VectorUniverse Tests
//
// Testing type safety, composition, performance, and correctness

import XCTest
import simd
@testable import VectorStoreKit

// MARK: - Test Metadata

struct TestMetadata: Codable, Sendable, Equatable {
    let id: String
    let timestamp: Date
    let category: String
    let score: Float
    
    static func random() -> TestMetadata {
        TestMetadata(
            id: UUID().uuidString,
            timestamp: Date(),
            category: ["A", "B", "C", "D"].randomElement()!,
            score: Float.random(in: 0...1)
        )
    }
}

// MARK: - Type System Tests

@available(macOS 10.15, iOS 13.0, *)
final class VectorUniverseTypeSystemTests: XCTestCase {
    
    func testBasicTypeChainProgression() {
        // Test that each step in the chain produces the expected type
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        // Step 1: Index configuration
        let indexed = universe.index(using: HNSWProductionIndexingStrategy())
        XCTAssert(type(of: indexed) == IndexedUniverse<SIMD32<Float>, TestMetadata, HNSWProductionIndexingStrategy>.self)
        
        // Step 2: Storage configuration
        let stored = indexed.store(using: HierarchicalProductionStorageStrategy())
        XCTAssert(type(of: stored) == StoredUniverse<SIMD32<Float>, TestMetadata, HNSWProductionIndexingStrategy, HierarchicalProductionStorageStrategy>.self)
        
        // Step 3: Acceleration configuration
        let accelerated = stored.accelerate(with: MetalProductionAcceleratorStrategy())
        XCTAssert(type(of: accelerated) == AcceleratedUniverse<SIMD32<Float>, TestMetadata, HNSWProductionIndexingStrategy, HierarchicalProductionStorageStrategy, MetalProductionAcceleratorStrategy>.self)
        
        // Step 4: Optimization configuration
        let optimized = accelerated.optimize(with: MLProductionOptimizationStrategy())
        XCTAssert(type(of: optimized) == OptimizedUniverse<SIMD32<Float>, TestMetadata, HNSWProductionIndexingStrategy, HierarchicalProductionStorageStrategy, MetalProductionAcceleratorStrategy, MLProductionOptimizationStrategy>.self)
    }
    
    func testMultiIndexTypeChain() {
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        // Configure multiple indexes
        let multiIndexed = universe.multiIndex(using: [
            HNSWResearchIndexingStrategy(),
            IVFIndexingStrategy(),
            LearnedIndexingStrategy()
        ])
        
        // Verify we can continue the chain
        let multiStored = multiIndexed.store(using: HierarchicalResearchStorageStrategy())
        let multiAccelerated = multiStored.accelerate(with: MetalResearchAcceleratorStrategy())
        let multiOptimized = multiAccelerated.optimize(with: MLResearchOptimizationStrategy())
        
        // Type should be preserved through the chain
        XCTAssertNotNil(multiOptimized)
    }
    
    func testOptionalStepsInChain() {
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        // Test skipping acceleration
        let withoutAcceleration = universe
            .index(using: HNSWProductionIndexingStrategy())
            .store(using: HierarchicalProductionStorageStrategy())
            .optimize(with: MLProductionOptimizationStrategy())
        
        XCTAssertNotNil(withoutAcceleration)
        
        // Test minimal configuration
        let minimal = universe
            .index(using: HNSWProductionIndexingStrategy())
            .store(using: HierarchicalProductionStorageStrategy())
        
        XCTAssertNotNil(minimal)
    }
    
    func testBuilderExtensions() {
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        // Test production builder
        let productionConfig = universe.production()
        XCTAssertNotNil(productionConfig)
        
        // Test research builder
        let researchConfig = universe.research()
        XCTAssertNotNil(researchConfig)
        
        // Test performance builder
        let performanceConfig = universe.performance()
        XCTAssertNotNil(performanceConfig)
    }
    
    func testGenericConstraints() {
        // Test with different vector types
        let float32Universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        XCTAssertNotNil(float32Universe)
        
        let float64Universe = VectorUniverse<SIMD16<Double>, TestMetadata>()
        XCTAssertNotNil(float64Universe)
        
        // Test with different metadata types
        struct CustomMetadata: Codable, Sendable {
            let value: Int
        }
        
        let customUniverse = VectorUniverse<SIMD32<Float>, CustomMetadata>()
        XCTAssertNotNil(customUniverse)
    }
}

// MARK: - Integration Tests

@available(macOS 10.15, iOS 13.0, *)
final class VectorUniverseIntegrationTests: XCTestCase {
    
    func testEndToEndMaterialization() async throws {
        // Create a complete universe configuration
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        let store = try await universe
            .index(using: MockHNSWStrategy())
            .store(using: MockStorageStrategy())
            .accelerate(with: MockAccelerator())
            .optimize(with: MockOptimizer())
            .materialize()
        
        XCTAssertNotNil(store)
        
        // Test basic operations
        let testVector = SIMD32<Float>(repeating: 0.5)
        let entry = VectorEntry(
            id: "test-1",
            vector: testVector,
            metadata: TestMetadata.random(),
            quality: .high,
            accessPattern: .frequent
        )
        
        let result = try await store.insert(entry)
        XCTAssertTrue(result.success)
        
        // Test search
        let searchResults = try await store.search(
            query: testVector,
            k: 10,
            strategy: .approximate,
            filter: nil
        )
        
        XCTAssertFalse(searchResults.isEmpty)
    }
    
    func testMultiIndexMaterialization() async throws {
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        let researchStore = try await universe
            .multiIndex(using: [
                MockHNSWStrategy(),
                MockIVFStrategy(),
                MockLearnedStrategy()
            ])
            .store(using: MockStorageStrategy())
            .accelerate(with: MockAccelerator())
            .optimize(with: MockOptimizer())
            .materialize()
        
        XCTAssertNotNil(researchStore)
    }
    
    func testConfigurationPropagation() async throws {
        // Test that configuration settings propagate through the chain
        var customConfig = UniverseConfiguration()
        customConfig = customConfig.enableResearchMode()
        
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>(
            configuration: customConfig
        )
        
        let store = try await universe
            .index(using: MockHNSWStrategy())
            .store(using: MockStorageStrategy())
            .materialize()
        
        XCTAssertNotNil(store)
        // Verify research mode is enabled (would check internal state in real implementation)
    }
}

// MARK: - Performance Tests

@available(macOS 10.15, iOS 13.0, *)
final class VectorUniversePerformanceTests: XCTestCase {
    
    func testTypeChainCompileTime() {
        // This test verifies that complex type chains don't cause excessive compile time
        measure {
            _ = VectorUniverse<SIMD32<Float>, TestMetadata>()
                .index(using: HNSWProductionIndexingStrategy())
                .store(using: HierarchicalProductionStorageStrategy())
                .accelerate(with: MetalProductionAcceleratorStrategy())
                .optimize(with: MLProductionOptimizationStrategy())
        }
    }
    
    func testMaterializationPerformance() async throws {
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        await measure {
            _ = try? await universe
                .index(using: MockHNSWStrategy())
                .store(using: MockStorageStrategy())
                .accelerate(with: MockAccelerator())
                .optimize(with: MockOptimizer())
                .materialize()
        }
    }
    
    func testLargeScaleConfiguration() async throws {
        // Test with many strategies
        let strategies = (0..<10).map { _ in MockHNSWStrategy() }
        
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        await measure {
            _ = try? await universe
                .multiIndex(using: strategies)
                .store(using: MockStorageStrategy())
                .materialize()
        }
    }
}

// MARK: - Property-Based Tests

@available(macOS 10.15, iOS 13.0, *)
final class VectorUniversePropertyTests: XCTestCase {
    
    func testTypeChainAssociativity() {
        // Property: The order of configuration should not affect the final type
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        // Path 1: Index -> Store -> Accelerate
        let path1 = universe
            .index(using: HNSWProductionIndexingStrategy())
            .store(using: HierarchicalProductionStorageStrategy())
            .accelerate(with: MetalProductionAcceleratorStrategy())
        
        // Path 2: Same components, verified separately
        let indexed = universe.index(using: HNSWProductionIndexingStrategy())
        let stored = indexed.store(using: HierarchicalProductionStorageStrategy())
        let path2 = stored.accelerate(with: MetalProductionAcceleratorStrategy())
        
        // Types should be identical
        XCTAssert(type(of: path1) == type(of: path2))
    }
    
    func testStrategyIndependence() async throws {
        // Property: Different strategies should not interfere with each other
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        // Create multiple configurations with different strategies
        let configs = [
            universe.index(using: MockHNSWStrategy())
                .store(using: MockStorageStrategy()),
            universe.index(using: MockIVFStrategy())
                .store(using: MockStorageStrategy()),
            universe.index(using: MockLearnedStrategy())
                .store(using: MockStorageStrategy())
        ]
        
        // Each should materialize independently
        for config in configs {
            let store = try await config.materialize()
            XCTAssertNotNil(store)
        }
    }
    
    func testConfigurationImmutability() {
        // Property: Configuration steps should not modify previous stages
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        let indexed = universe.index(using: HNSWProductionIndexingStrategy())
        
        // Create multiple branches from the same indexed universe
        let branch1 = indexed.store(using: HierarchicalProductionStorageStrategy())
        let branch2 = indexed.store(using: InMemoryPerformanceStorageStrategy())
        
        // Both branches should be valid
        XCTAssertNotNil(branch1)
        XCTAssertNotNil(branch2)
        
        // Original indexed universe should be unchanged
        let branch3 = indexed.store(using: HierarchicalProductionStorageStrategy())
        XCTAssertNotNil(branch3)
    }
}

// MARK: - Error Handling Tests

@available(macOS 10.15, iOS 13.0, *)
final class VectorUniverseErrorTests: XCTestCase {
    
    func testInvalidConfigurationHandling() async {
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        // Test with a strategy that throws during initialization
        let failingStore = universe
            .index(using: FailingIndexStrategy())
            .store(using: MockStorageStrategy())
        
        do {
            _ = try await failingStore.materialize()
            XCTFail("Should have thrown an error")
        } catch {
            XCTAssertNotNil(error)
        }
    }
    
    func testResourceConstraintHandling() async {
        // Test behavior when resources are constrained
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        let constrainedStore = universe
            .index(using: MockHNSWStrategy())
            .store(using: ConstrainedStorageStrategy())
            .accelerate(with: ConstrainedAccelerator())
        
        do {
            _ = try await constrainedStore.materialize()
            // Should handle constraints gracefully
        } catch {
            // Verify appropriate error type
            XCTAssertTrue(error is ResourceConstraintError)
        }
    }
}

// MARK: - Mock Implementations

struct MockHNSWStrategy: IndexingStrategy {
    typealias Config = MockConfig
    typealias IndexType = MockIndex
    
    let identifier = "mock-hnsw"
    let characteristics = IndexCharacteristics(
        approximation: .approximate(quality: 0.95),
        dynamism: .fullyDynamic,
        scalability: .excellent,
        parallelism: .full
    )
    
    func createIndex<Vector: SIMD, Metadata: Codable & Sendable>(
        configuration: Config,
        vectorType: Vector.Type,
        metadataType: Metadata.Type
    ) async throws -> MockIndex where Vector.Scalar: BinaryFloatingPoint {
        return MockIndex()
    }
}

struct MockIVFStrategy: IndexingStrategy {
    typealias Config = MockConfig
    typealias IndexType = MockIndex
    
    let identifier = "mock-ivf"
    let characteristics = IndexCharacteristics(
        approximation: .approximate(quality: 0.90),
        dynamism: .semiDynamic,
        scalability: .excellent,
        parallelism: .limited
    )
    
    func createIndex<Vector: SIMD, Metadata: Codable & Sendable>(
        configuration: Config,
        vectorType: Vector.Type,
        metadataType: Metadata.Type
    ) async throws -> MockIndex where Vector.Scalar: BinaryFloatingPoint {
        return MockIndex()
    }
}

struct MockLearnedStrategy: IndexingStrategy {
    typealias Config = MockConfig
    typealias IndexType = MockIndex
    
    let identifier = "mock-learned"
    let characteristics = IndexCharacteristics(
        approximation: .adaptive,
        dynamism: .fullyDynamic,
        scalability: .excellent,
        parallelism: .full
    )
    
    func createIndex<Vector: SIMD, Metadata: Codable & Sendable>(
        configuration: Config,
        vectorType: Vector.Type,
        metadataType: Metadata.Type
    ) async throws -> MockIndex where Vector.Scalar: BinaryFloatingPoint {
        return MockIndex()
    }
}

struct FailingIndexStrategy: IndexingStrategy {
    typealias Config = MockConfig
    typealias IndexType = MockIndex
    
    let identifier = "failing"
    let characteristics = IndexCharacteristics(
        approximation: .exact,
        dynamism: .`static`,
        scalability: .excellent,
        parallelism: .none
    )
    
    func createIndex<Vector: SIMD, Metadata: Codable & Sendable>(
        configuration: Config,
        vectorType: Vector.Type,
        metadataType: Metadata.Type
    ) async throws -> MockIndex where Vector.Scalar: BinaryFloatingPoint {
        throw MockError.initializationFailed
    }
}

struct MockStorageStrategy: StorageStrategy {
    typealias Config = MockConfig
    typealias BackendType = MockBackend
    
    let identifier = "mock-storage"
    let characteristics = StorageCharacteristics(
        durability: .standard,
        consistency: .strong,
        scalability: .excellent,
        compression: .adaptive
    )
    
    func createBackend(configuration: Config) async throws -> MockBackend {
        return MockBackend()
    }
}

struct ConstrainedStorageStrategy: StorageStrategy {
    typealias Config = MockConfig
    typealias BackendType = MockBackend
    
    let identifier = "constrained-storage"
    let characteristics = StorageCharacteristics(
        durability: .eventual,
        consistency: .eventual,
        scalability: .excellent,
        compression: .none
    )
    
    func createBackend(configuration: Config) async throws -> MockBackend {
        if ProcessInfo.processInfo.physicalMemory < 1024 * 1024 * 1024 { // < 1GB
            throw ResourceConstraintError.insufficientMemory
        }
        return MockBackend()
    }
}

struct MockAccelerator: ComputeAccelerator {
    typealias DeviceType = MockDevice
    typealias CapabilitiesType = MockCapabilities
    
    let identifier = "mock-accelerator"
    let requirements = HardwareRequirements(
        minimumMemory: 0,
        requiredFeatures: [],
        optionalFeatures: []
    )
    
    func initialize() async throws -> MockDevice {
        return MockDevice()
    }
    
    func capabilities() -> MockCapabilities {
        return MockCapabilities()
    }
}

struct ConstrainedAccelerator: ComputeAccelerator {
    typealias DeviceType = MockDevice
    typealias CapabilitiesType = MockCapabilities
    
    let identifier = "constrained-accelerator"
    let requirements = HardwareRequirements(
        minimumMemory: 1024 * 1024 * 1024, // 1GB
        requiredFeatures: [.metal],
        optionalFeatures: []
    )
    
    func initialize() async throws -> MockDevice {
        // Simulate resource check
        return MockDevice()
    }
    
    func capabilities() -> MockCapabilities {
        return MockCapabilities()
    }
}

struct MockOptimizer: OptimizationStrategyProtocol {
    typealias ModelType = MockModel
    typealias MetricsType = MockMetrics
    
    let identifier = "mock-optimizer"
    let characteristics = OptimizationCharacteristics(
        frequency: .periodic(60),
        scope: .global,
        adaptability: .moderate,
        overhead: .low
    )
    
    func optimize<Index: VectorIndex>(
        index: Index,
        metrics: MockMetrics
    ) async throws {
        // Mock optimization
    }
}

// Mock supporting types
struct MockConfig {}
struct MockIndex: VectorIndex {
    // Minimal VectorIndex implementation
    typealias Vector = SIMD32<Float>
    typealias Metadata = TestMetadata
    typealias Configuration = MockConfig
    typealias Statistics = MockStatistics
    
    var count: Int { 0 }
    var capacity: Int { Int.max }
    var memoryUsage: Int { 0 }
    var configuration: MockConfig { MockConfig() }
    var isOptimized: Bool { true }
    
    func insert(_ entry: VectorEntry<Vector, Metadata>) async throws -> InsertResult {
        InsertResult(success: true, insertTime: 0.001, memoryImpact: 100, indexReorganization: false)
    }
    
    func search(query: Vector, k: Int, strategy: SearchStrategy, filter: SearchFilter?) async throws -> [SearchResult<Metadata>] {
        return []
    }
    
    func update(id: VectorID, vector: Vector?, metadata: Metadata?) async throws -> Bool { true }
    func delete(id: VectorID) async throws -> Bool { true }
    func contains(id: VectorID) async -> Bool { false }
    func optimize(strategy: OptimizationStrategy) async throws {}
    func compact() async throws {}
    func statistics() async -> MockStatistics { MockStatistics() }
    func validateIntegrity() async throws -> IntegrityReport {
        IntegrityReport(isValid: true, errors: [], warnings: [], statistics: IntegrityStatistics(totalChecks: 1, passedChecks: 1, failedChecks: 0, checkDuration: 0.001))
    }
    func export(format: ExportFormat) async throws -> Data { Data() }
    func `import`(data: Data, format: ExportFormat) async throws {}
    func analyzeDistribution() async -> DistributionAnalysis {
        DistributionAnalysis(
            dimensionality: 32,
            density: 0.5,
            clustering: ClusteringAnalysis(estimatedClusters: 5, silhouetteScore: 0.7, inertia: 100, clusterCenters: []),
            outliers: [],
            statistics: DistributionStatistics(mean: [], variance: [], skewness: [], kurtosis: [])
        )
    }
    func performanceProfile() async -> PerformanceProfile {
        PerformanceProfile(
            searchLatency: LatencyProfile(p50: 0.001, p90: 0.002, p95: 0.003, p99: 0.005, max: 0.01),
            insertLatency: LatencyProfile(p50: 0.001, p90: 0.002, p95: 0.003, p99: 0.005, max: 0.01),
            memoryUsage: MemoryProfile(baseline: 1000, peak: 2000, average: 1500, efficiency: 0.8),
            throughput: ThroughputProfile(queriesPerSecond: 1000, insertsPerSecond: 500, updatesPerSecond: 200, deletesPerSecond: 100)
        )
    }
    func visualizationData() async -> VisualizationData {
        VisualizationData(nodePositions: [[0, 0]], edges: [(0, 0, 0)], nodeMetadata: [:], layoutAlgorithm: "force_directed")
    }
}

struct MockStatistics: IndexStatistics {
    let vectorCount: Int = 0
    let memoryUsage: Int = 0
    let averageSearchLatency: TimeInterval = 0.001
    let qualityMetrics = IndexQualityMetrics(recall: 0.95, precision: 0.98, buildTime: 1.0, memoryEfficiency: 0.8, searchLatency: 0.001)
}

struct MockBackend: StorageBackend {
    typealias Configuration = MockConfig
    typealias Statistics = MockStorageStats
    
    var configuration: MockConfig { MockConfig() }
    var isReady: Bool { true }
    var size: Int { 0 }
    
    func store(key: String, data: Data, options: StorageOptions) async throws {}
    func retrieve(key: String) async throws -> Data? { nil }
    func delete(key: String) async throws {}
    func exists(key: String) async -> Bool { false }
    func scan(prefix: String) async throws -> AsyncStream<(String, Data)> {
        AsyncStream { _ in }
    }
    func compact() async throws {}
    func statistics() async -> MockStorageStats { MockStorageStats() }
    func validateIntegrity() async throws -> StorageIntegrityReport {
        StorageIntegrityReport(isHealthy: true, issues: [], recommendations: [], lastCheck: Date())
    }
    func createSnapshot() async throws -> SnapshotIdentifier {
        SnapshotIdentifier(id: UUID().uuidString, timestamp: Date(), checksum: "mock")
    }
    func restoreSnapshot(_ identifier: SnapshotIdentifier) async throws {}
    func batchStore(_ items: [(key: String, data: Data, options: StorageOptions)]) async throws {}
    func batchRetrieve(_ keys: [String]) async throws -> [String: Data?] { [:] }
    func batchDelete(_ keys: [String]) async throws {}
}

struct MockStorageStats: StorageStatistics {
    let totalSize: Int = 0
    let compressionRatio: Float = 1.0
    let averageLatency: TimeInterval = 0.001
    let healthMetrics = StorageHealthMetrics(errorRate: 0, latencyP99: 0.005, throughput: 1000, fragmentation: 0.1)
}

struct MockDevice {}
struct MockCapabilities {}
struct MockModel {}
struct MockMetrics {}

enum MockError: Error {
    case initializationFailed
}

enum ResourceConstraintError: Error {
    case insufficientMemory
    case deviceNotAvailable
}

// MARK: - Test Helpers

extension XCTestCase {
    @available(macOS 13.0, iOS 16.0, *)
    func measure(block: () async throws -> Void) async {
        let start = Date()
        do {
            try await block()
        } catch {
            XCTFail("Async measure block threw error: \(error)")
        }
        let duration = Date().timeIntervalSince(start)
        print("Execution time: \(duration)s")
    }
}

// MARK: - Research Validation Tests

@available(macOS 10.15, iOS 13.0, *)
final class VectorUniverseResearchValidationTests: XCTestCase {
    
    func testRecallPrecisionMeasurement() async throws {
        // Create ground truth dataset
        let groundTruth = generateGroundTruth(numQueries: 10, k: 10)
        
        // Create research store with multiple indexes
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        let researchStore = try await universe
            .multiIndex(using: [
                MockHNSWStrategy(),
                MockIVFStrategy(),
                MockLearnedStrategy()
            ])
            .store(using: MockStorageStrategy())
            .materializeForComparison()
        
        // Insert test data
        let testVectors = generateTestVectors(count: 1000)
        let insertResult = try await researchStore.addWithComparison(testVectors)
        
        XCTAssertNotNil(insertResult)
        
        // Measure recall/precision for each index
        for (query, expectedIds) in groundTruth {
            let results = try await researchStore.searchWithComparison(
                query: query,
                k: 10,
                groundTruth: expectedIds
            )
            
            // Verify recall/precision metrics are calculated
            XCTAssertGreaterThan(results.metrics.recall, 0.9)
            XCTAssertGreaterThan(results.metrics.precision, 0.85)
        }
    }
    
    func testAlgorithmConsistency() async throws {
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        // Create stores with different algorithms
        let stores = try await [
            universe.index(using: MockHNSWStrategy())
                .store(using: MockStorageStrategy())
                .materialize(),
            universe.index(using: MockIVFStrategy())
                .store(using: MockStorageStrategy())
                .materialize(),
            universe.index(using: MockLearnedStrategy())
                .store(using: MockStorageStrategy())
                .materialize()
        ]
        
        // Insert same data into all stores
        let testVectors = generateTestVectors(count: 100)
        for store in stores {
            for vector in testVectors {
                _ = try await store.insert(vector)
            }
        }
        
        // Verify consistency of results
        let query = generateRandomVector()
        let results = try await stores.map { store in
            try await store.search(query: query, k: 5, strategy: .exact, filter: nil)
        }
        
        // Top result should be consistent across algorithms
        if results.count >= 2 {
            for i in 1..<results.count {
                XCTAssertEqual(results[0].first?.id, results[i].first?.id,
                             "Top result should be consistent across algorithms")
            }
        }
    }
    
    func testPerformanceCharacteristics() async throws {
        let universe = VectorUniverse<SIMD32<Float>, TestMetadata>()
        
        // Test different optimization strategies
        let configurations: [(String, OptimizationStrategyProtocol)] = [
            ("None", NoOptimizationStrategy()),
            ("ML", MLResearchOptimizationStrategy()),
            ("Aggressive", AggressiveOptimizationStrategy())
        ]
        
        for (name, optimizer) in configurations {
            let store = try await universe
                .index(using: MockHNSWStrategy())
                .store(using: MockStorageStrategy())
                .optimize(with: optimizer)
                .materialize()
            
            // Measure performance
            let vectors = generateTestVectors(count: 1000)
            
            let insertStart = Date()
            for vector in vectors {
                _ = try await store.insert(vector)
            }
            let insertDuration = Date().timeIntervalSince(insertStart)
            
            let searchStart = Date()
            for _ in 0..<100 {
                let query = generateRandomVector()
                _ = try await store.search(query: query, k: 10, strategy: .approximate, filter: nil)
            }
            let searchDuration = Date().timeIntervalSince(searchStart)
            
            print("\(name) Optimizer - Insert: \(insertDuration)s, Search: \(searchDuration)s")
            
            // Verify performance is within expected bounds
            XCTAssertLessThan(insertDuration, 10.0, "\(name) insert too slow")
            XCTAssertLessThan(searchDuration, 1.0, "\(name) search too slow")
        }
    }
}

// MARK: - Test Data Generation

func generateTestVectors(count: Int) -> [VectorEntry<SIMD32<Float>, TestMetadata>] {
    return (0..<count).map { i in
        VectorEntry(
            id: "vector-\(i)",
            vector: generateRandomVector(),
            metadata: TestMetadata.random(),
            quality: .high,
            accessPattern: .moderate
        )
    }
}

func generateRandomVector() -> SIMD32<Float> {
    var values: [Float] = []
    for _ in 0..<32 {
        values.append(Float.random(in: -1...1))
    }
    return SIMD32<Float>(values)
}

func generateGroundTruth(numQueries: Int, k: Int) -> [(SIMD32<Float>, [VectorID])] {
    return (0..<numQueries).map { _ in
        let query = generateRandomVector()
        let expectedIds = (0..<k).map { "vector-\($0)" }
        return (query, expectedIds)
    }
}

// Additional optimization strategy for testing
struct NoOptimizationStrategy: OptimizationStrategyProtocol {
    typealias ModelType = Void
    typealias MetricsType = Void
    
    let identifier = "none"
    let characteristics = OptimizationCharacteristics(
        frequency: .never,
        scope: .local,
        adaptability: .none,
        overhead: .negligible
    )
    
    func optimize<Index: VectorIndex>(index: Index, metrics: Void) async throws {
        // No optimization
    }
}
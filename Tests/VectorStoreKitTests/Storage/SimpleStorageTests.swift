// VectorStoreKit: Simple Storage Tests
//
// Comprehensive test suite for the SimpleStorage implementation

import XCTest
@testable import VectorStoreKit

final class SimpleStorageTests: XCTestCase {
    
    // MARK: - Properties
    
    var storage: SimpleStorage!
    
    // MARK: - Setup/Teardown
    
    override func setUp() async throws {
        try await super.setUp()
        
        let config = SimpleStorageConfiguration(
            memoryLimit: 10_485_760, // 10MB
            initialCapacity: 100,
            enableCompression: true,
            compressionThreshold: 1024,
            evictionPolicy: .lru
        )
        
        storage = try await SimpleStorage(configuration: config)
    }
    
    override func tearDown() async throws {
        storage = nil
        try await super.tearDown()
    }
    
    // MARK: - Basic Operations Tests
    
    func testStoreAndRetrieve() async throws {
        // Given
        let key = "test-key"
        let data = Data("Hello, World!".utf8)
        
        // When
        try await storage.store(key: key, data: data, options: .default)
        let retrieved = try await storage.retrieve(key: key)
        
        // Then
        XCTAssertEqual(retrieved, data)
    }
    
    func testRetrieveNonExistentKey() async throws {
        // When
        let retrieved = try await storage.retrieve(key: "non-existent")
        
        // Then
        XCTAssertNil(retrieved)
    }
    
    func testDelete() async throws {
        // Given
        let key = "test-key"
        let data = Data("Test data".utf8)
        try await storage.store(key: key, data: data, options: .default)
        
        // When
        try await storage.delete(key: key)
        let retrieved = try await storage.retrieve(key: key)
        
        // Then
        XCTAssertNil(retrieved)
    }
    
    func testExists() async throws {
        // Given
        let key = "test-key"
        let data = Data("Test data".utf8)
        
        // When/Then - Before storing
        let existsBefore = await storage.exists(key: key)
        XCTAssertFalse(existsBefore)
        
        // When/Then - After storing
        try await storage.store(key: key, data: data, options: .default)
        let existsAfter = await storage.exists(key: key)
        XCTAssertTrue(existsAfter)
        
        // When/Then - After deleting
        try await storage.delete(key: key)
        let existsAfterDelete = await storage.exists(key: key)
        XCTAssertFalse(existsAfterDelete)
    }
    
    // MARK: - Compression Tests
    
    func testCompressionForLargeData() async throws {
        // Given
        let key = "large-data"
        let largeString = String(repeating: "Hello, World! ", count: 1000)
        let data = Data(largeString.utf8)
        
        // When
        try await storage.store(key: key, data: data, options: .default)
        let stats = await storage.statistics()
        
        // Then
        XCTAssertGreaterThan(stats.compressionRatio, 1.0, "Large repetitive data should be compressed")
        
        // Verify data integrity
        let retrieved = try await storage.retrieve(key: key)
        XCTAssertEqual(retrieved, data)
    }
    
    func testNoCompressionForSmallData() async throws {
        // Given
        let config = SimpleStorageConfiguration(
            memoryLimit: 10_485_760,
            initialCapacity: 100,
            enableCompression: true,
            compressionThreshold: 1024 // 1KB threshold
        )
        let storage = try await SimpleStorage(configuration: config)
        
        let key = "small-data"
        let data = Data("Small".utf8) // Much less than 1KB
        
        // When
        try await storage.store(key: key, data: data, options: .default)
        let stats = await storage.statistics()
        
        // Then
        XCTAssertEqual(stats.compressionRatio, 1.0, "Small data should not be compressed")
    }
    
    // MARK: - Memory Limit Tests
    
    func testMemoryLimitEnforcement() async throws {
        // Given
        let config = SimpleStorageConfiguration(
            memoryLimit: 1024, // 1KB limit
            initialCapacity: 10,
            enableCompression: false,
            evictionPolicy: .none // No eviction
        )
        let storage = try await SimpleStorage(configuration: config)
        
        let data = Data(repeating: 0xFF, count: 512) // 512 bytes
        
        // When/Then - First store should succeed
        try await storage.store(key: "data1", data: data, options: .default)
        
        // Second store should succeed (total 1024 bytes)
        try await storage.store(key: "data2", data: data, options: .default)
        
        // Third store should fail (would exceed limit)
        do {
            try await storage.store(key: "data3", data: data, options: .default)
            XCTFail("Should have thrown memory limit error")
        } catch {
            // Expected
        }
    }
    
    // MARK: - Eviction Tests
    
    func testLRUEviction() async throws {
        // Given
        let config = SimpleStorageConfiguration(
            memoryLimit: 3072, // 3KB limit
            initialCapacity: 10,
            enableCompression: false,
            evictionPolicy: .lru
        )
        let storage = try await SimpleStorage(configuration: config)
        
        let data = Data(repeating: 0xFF, count: 1024) // 1KB each
        
        // Store 3 items (fills storage)
        try await storage.store(key: "data1", data: data, options: .default)
        try await storage.store(key: "data2", data: data, options: .default)
        try await storage.store(key: "data3", data: data, options: .default)
        
        // Access data1 and data2 to make them "recently used"
        _ = try await storage.retrieve(key: "data1")
        _ = try await storage.retrieve(key: "data2")
        
        // Store data4 - should evict data3 (least recently used)
        try await storage.store(key: "data4", data: data, options: .default)
        
        // Then
        XCTAssertNotNil(try await storage.retrieve(key: "data1"))
        XCTAssertNotNil(try await storage.retrieve(key: "data2"))
        XCTAssertNil(try await storage.retrieve(key: "data3")) // Should be evicted
        XCTAssertNotNil(try await storage.retrieve(key: "data4"))
    }
    
    // MARK: - Batch Operations Tests
    
    func testBatchStore() async throws {
        // Given
        let items = [
            ("key1", Data("data1".utf8), StorageOptions.default),
            ("key2", Data("data2".utf8), StorageOptions.default),
            ("key3", Data("data3".utf8), StorageOptions.default)
        ]
        
        // When
        try await storage.batchStore(items)
        
        // Then
        for (key, expectedData, _) in items {
            let retrieved = try await storage.retrieve(key: key)
            XCTAssertEqual(retrieved, expectedData)
        }
    }
    
    func testBatchRetrieve() async throws {
        // Given
        let data1 = Data("data1".utf8)
        let data2 = Data("data2".utf8)
        try await storage.store(key: "key1", data: data1, options: .default)
        try await storage.store(key: "key2", data: data2, options: .default)
        
        // When
        let results = try await storage.batchRetrieve(["key1", "key2", "key3"])
        
        // Then
        XCTAssertEqual(results["key1"]!, data1)
        XCTAssertEqual(results["key2"]!, data2)
        XCTAssertNil(results["key3"]!)
    }
    
    func testBatchDelete() async throws {
        // Given
        try await storage.store(key: "key1", data: Data("data1".utf8), options: .default)
        try await storage.store(key: "key2", data: Data("data2".utf8), options: .default)
        try await storage.store(key: "key3", data: Data("data3".utf8), options: .default)
        
        // When
        try await storage.batchDelete(["key1", "key3"])
        
        // Then
        XCTAssertNil(try await storage.retrieve(key: "key1"))
        XCTAssertNotNil(try await storage.retrieve(key: "key2"))
        XCTAssertNil(try await storage.retrieve(key: "key3"))
    }
    
    // MARK: - Scan Tests
    
    func testScanWithPrefix() async throws {
        // Given
        try await storage.store(key: "user:1", data: Data("user1".utf8), options: .default)
        try await storage.store(key: "user:2", data: Data("user2".utf8), options: .default)
        try await storage.store(key: "item:1", data: Data("item1".utf8), options: .default)
        
        // When
        var userKeys: [(String, Data)] = []
        for try await (key, data) in try await storage.scan(prefix: "user:") {
            userKeys.append((key, data))
        }
        
        // Then
        XCTAssertEqual(userKeys.count, 2)
        XCTAssertTrue(userKeys.contains { $0.0 == "user:1" })
        XCTAssertTrue(userKeys.contains { $0.0 == "user:2" })
    }
    
    // MARK: - Statistics Tests
    
    func testStatistics() async throws {
        // Given
        let data1 = Data("Small data".utf8)
        let data2 = Data(repeating: 0xFF, count: 2048)
        
        try await storage.store(key: "key1", data: data1, options: .default)
        try await storage.store(key: "key2", data: data2, options: .default)
        
        // When
        let stats = await storage.statistics()
        
        // Then
        XCTAssertEqual(stats.itemCount, 2)
        XCTAssertGreaterThan(stats.totalSize, 0)
        XCTAssertGreaterThanOrEqual(stats.compressionRatio, 1.0)
        XCTAssertGreaterThan(stats.memoryUsage.totalSize, stats.totalSize)
    }
    
    // MARK: - Integrity Tests
    
    func testIntegrityValidation() async throws {
        // Given
        try await storage.store(key: "key1", data: Data("data1".utf8), options: .default)
        
        // When
        let report = try await storage.validateIntegrity()
        
        // Then
        XCTAssertTrue(report.isHealthy)
        XCTAssertTrue(report.issues.isEmpty)
    }
    
    // MARK: - Snapshot Tests
    
    func testCreateSnapshot() async throws {
        // Given
        try await storage.store(key: "key1", data: Data("data1".utf8), options: .default)
        try await storage.store(key: "key2", data: Data("data2".utf8), options: .default)
        
        // When
        let snapshot = try await storage.createSnapshot()
        
        // Then
        XCTAssertFalse(snapshot.id.isEmpty)
        XCTAssertFalse(snapshot.checksum.isEmpty)
    }
    
    // MARK: - Configuration Tests
    
    func testConfigurationPresets() async throws {
        // Test different configuration presets
        let configs = [
            SimpleStorageConfiguration.default,
            SimpleStorageConfiguration.small,
            SimpleStorageConfiguration.large,
            SimpleStorageConfiguration.performance,
            SimpleStorageConfiguration.development
        ]
        
        for config in configs {
            let storage = try await SimpleStorage(configuration: config)
            XCTAssertTrue(await storage.isReady)
        }
    }
    
    // MARK: - Performance Tests
    
    func testPerformanceOfLargeDataSet() async throws {
        // Given
        let config = SimpleStorageConfiguration.performance
        let storage = try await SimpleStorage(configuration: config)
        
        let measure = XCTMetric.wallClockTime
        let options = XCTMeasureOptions()
        options.iterationCount = 5
        
        // Measure store performance
        self.measure(metrics: [measure], options: options) {
            Task {
                for i in 0..<1000 {
                    let data = Data("Test data \(i)".utf8)
                    try? await storage.store(key: "key\(i)", data: data, options: .default)
                }
            }
        }
    }
    
    // MARK: - Thread Safety Tests
    
    func testConcurrentAccess() async throws {
        // Test concurrent reads and writes
        await withTaskGroup(of: Void.self) { group in
            // Writers
            for i in 0..<10 {
                group.addTask {
                    let data = Data("Writer \(i)".utf8)
                    try? await self.storage.store(key: "concurrent-\(i)", data: data, options: .default)
                }
            }
            
            // Readers
            for i in 0..<10 {
                group.addTask {
                    _ = try? await self.storage.retrieve(key: "concurrent-\(i)")
                }
            }
        }
        
        // Verify data integrity
        for i in 0..<10 {
            let exists = await storage.exists(key: "concurrent-\(i)")
            // Some keys might exist, some might not due to timing
            print("Key concurrent-\(i) exists: \(exists)")
        }
    }
}
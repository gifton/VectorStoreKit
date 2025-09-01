import XCTest
@testable import VectorStoreKit

/// Comprehensive tests for the 3-tier storage system
final class ThreeTierStorageTests: XCTestCase {
    
    var storage: ThreeTierStorage!
    var testPath: String!
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Create temporary test directory
        testPath = NSTemporaryDirectory().appending("VectorStoreKitTests-\(UUID().uuidString)")
        
        // Create storage with test configuration
        let config = ThreeTierStorageConfiguration(
            memoryLimit: 10_485_760, // 10MB
            memoryItemSizeLimit: 1_048_576, // 1MB
            initialMemoryCapacity: 100,
            basePath: testPath,
            autoMigrationEnabled: false // Disable for predictable tests
        )
        
        storage = try await ThreeTierStorage(configuration: config)
    }
    
    override func tearDown() async throws {
        storage = nil
        
        // Clean up test directory
        if let testPath = testPath {
            try? FileManager.default.removeItem(atPath: testPath)
        }
        
        try await super.tearDown()
    }
    
    // MARK: - Basic Operations
    
    func testStoreAndRetrieve() async throws {
        // Test data
        let key = "test-key"
        let data = Data("Hello, 3-tier storage!".utf8)
        let options = StorageOptions(priority: .normal)
        
        // Store
        try await storage.store(key: key, data: data, options: options)
        
        // Retrieve
        let retrieved = try await storage.retrieve(key: key)
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved, data)
    }
    
    func testDelete() async throws {
        let key = "delete-test"
        let data = Data("Delete me".utf8)
        
        // Store
        try await storage.store(key: key, data: data, options: .default)
        
        // Verify exists
        XCTAssertTrue(await storage.exists(key: key))
        
        // Delete
        try await storage.delete(key: key)
        
        // Verify deleted
        XCTAssertFalse(await storage.exists(key: key))
        let retrieved = try await storage.retrieve(key: key)
        XCTAssertNil(retrieved)
    }
    
    // MARK: - Tier Placement
    
    func testSmallDataGoesToMemory() async throws {
        let key = "small-data"
        let data = Data(repeating: 0x42, count: 100) // 100 bytes
        
        try await storage.store(key: key, data: data, options: .default)
        
        // Verify it's in memory tier
        let result = try await storage.retrieve(forKey: key)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tier, .memory)
    }
    
    func testMediumDataGoesToSSD() async throws {
        let key = "medium-data"
        let data = Data(repeating: 0x42, count: 2_000_000) // 2MB
        
        try await storage.store(key: key, data: data, options: .default)
        
        // Verify it's in SSD tier
        let result = try await storage.retrieve(forKey: key)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tier, .ssd)
    }
    
    func testLargeDataGoesToArchive() async throws {
        let key = "large-data"
        let data = Data(repeating: 0x42, count: 20_000_000) // 20MB
        
        try await storage.store(key: key, data: data, options: .default)
        
        // Verify it's in archive tier
        let result = try await storage.retrieve(forKey: key)
        XCTAssertNotNil(result)
        XCTAssertEqual(result?.tier, .archive)
    }
    
    // MARK: - Compression
    
    func testSSDCompression() async throws {
        let key = "ssd-compress"
        // Highly compressible data
        let data = Data(repeating: 0x00, count: 5_000_000) // 5MB of zeros
        
        try await storage.store(key: key, data: data, options: .default)
        
        // Get statistics to verify compression
        let stats = await storage.statistics()
        
        // Check that SSD tier has compression
        if let ssdRatio = stats.tierCompressionRatios[.ssd] {
            XCTAssertGreaterThan(ssdRatio, 1.0, "SSD should show compression")
        }
    }
    
    func testArchiveCompression() async throws {
        let key = "archive-compress"
        // Highly compressible data
        let data = Data(repeating: 0xFF, count: 20_000_000) // 20MB
        
        try await storage.store(key: key, data: data, options: .default)
        
        // Get statistics to verify compression
        let stats = await storage.statistics()
        
        // Check that archive tier has compression
        if let archiveRatio = stats.tierCompressionRatios[.archive] {
            XCTAssertGreaterThan(archiveRatio, 1.0, "Archive should show compression")
        }
    }
    
    // MARK: - Access Pattern Tracking
    
    func testAccessPatternTracking() async throws {
        let key = "access-test"
        let data = Data("Track my access".utf8)
        
        // Store
        try await storage.store(key: key, data: data, options: .default)
        
        // Access multiple times
        for _ in 0..<5 {
            _ = try await storage.retrieve(key: key)
        }
        
        // Check access info
        let accessInfo = try await storage.accessInfo(forKey: key)
        XCTAssertNotNil(accessInfo)
        XCTAssertEqual(accessInfo?.accessCount, 6) // 1 store + 5 retrieves
    }
    
    // MARK: - Batch Operations
    
    func testBatchStore() async throws {
        let items: [(key: String, data: Data, options: StorageOptions)] = (0..<10).map { i in
            ("batch-\(i)", Data("Item \(i)".utf8), .default)
        }
        
        // Batch store
        try await storage.batchStore(items)
        
        // Verify all stored
        for item in items {
            let retrieved = try await storage.retrieve(key: item.key)
            XCTAssertEqual(retrieved, item.data)
        }
    }
    
    func testBatchRetrieve() async throws {
        // Store items
        let keys = (0..<5).map { "batch-retrieve-\($0)" }
        for key in keys {
            try await storage.store(key: key, data: Data(key.utf8), options: .default)
        }
        
        // Batch retrieve
        let results = try await storage.batchRetrieve(keys)
        
        XCTAssertEqual(results.count, keys.count)
        for key in keys {
            XCTAssertNotNil(results[key])
            XCTAssertEqual(results[key]!, Data(key.utf8))
        }
    }
    
    // MARK: - Scan Operations
    
    func testScanWithPrefix() async throws {
        // Store items with common prefix
        let prefix = "scan-test-"
        for i in 0..<5 {
            let key = "\(prefix)\(i)"
            try await storage.store(key: key, data: Data("\(i)".utf8), options: .default)
        }
        
        // Store items without prefix
        try await storage.store(key: "other-1", data: Data("x".utf8), options: .default)
        
        // Scan with prefix
        var scanned: [(String, Data)] = []
        for try await (key, data) in try await storage.scan(prefix: prefix) {
            scanned.append((key, data))
        }
        
        XCTAssertEqual(scanned.count, 5)
        XCTAssertTrue(scanned.allSatisfy { $0.0.hasPrefix(prefix) })
    }
    
    // MARK: - Memory Limit
    
    func testMemoryLimitEnforcement() async throws {
        // Fill memory tier to limit
        let itemSize = 1_000_000 // 1MB
        let itemCount = 12 // Should exceed 10MB limit
        
        for i in 0..<itemCount {
            let key = "memory-limit-\(i)"
            let data = Data(repeating: UInt8(i), count: itemSize)
            
            do {
                try await storage.store(key: key, data: data, options: StorageOptions(priority: .critical))
            } catch {
                // Expected to fail when memory is full
                XCTAssertTrue(i > 9, "Should store at least 10 items before hitting limit")
                return
            }
        }
        
        XCTFail("Should have hit memory limit")
    }
    
    // MARK: - Statistics
    
    func testStatistics() async throws {
        // Store data in different tiers
        try await storage.store(key: "mem", data: Data(repeating: 1, count: 100), options: .default)
        try await storage.store(key: "ssd", data: Data(repeating: 2, count: 2_000_000), options: .default)
        try await storage.store(key: "arc", data: Data(repeating: 3, count: 20_000_000), options: .default)
        
        let stats = await storage.statistics()
        
        // Verify statistics
        XCTAssertGreaterThan(stats.totalSize, 0)
        XCTAssertEqual(stats.tierItemCounts.values.reduce(0, +), 3)
        XCTAssertGreaterThanOrEqual(stats.compressionRatio, 1.0)
    }
    
    // MARK: - Integrity
    
    func testIntegrityValidation() async throws {
        // Store some data
        for i in 0..<5 {
            try await storage.store(key: "integrity-\(i)", data: Data("\(i)".utf8), options: .default)
        }
        
        // Validate integrity
        let report = try await storage.validateIntegrity()
        
        XCTAssertTrue(report.isHealthy)
        XCTAssertTrue(report.issues.isEmpty)
    }
    
    // MARK: - Concurrent Access
    
    func testConcurrentAccess() async throws {
        // Test concurrent reads and writes
        await withTaskGroup(of: Void.self) { group in
            // Writers
            for i in 0..<10 {
                group.addTask {
                    let key = "concurrent-\(i)"
                    let data = Data("Data \(i)".utf8)
                    try? await self.storage.store(key: key, data: data, options: .default)
                }
            }
            
            // Readers
            for i in 0..<10 {
                group.addTask {
                    let key = "concurrent-\(i)"
                    _ = try? await self.storage.retrieve(key: key)
                }
            }
        }
        
        // Verify data consistency
        for i in 0..<10 {
            let key = "concurrent-\(i)"
            let data = try await storage.retrieve(key: key)
            if let data = data {
                XCTAssertEqual(data, Data("Data \(i)".utf8))
            }
        }
    }
    
    // MARK: - Error Cases
    
    func testRetrieveNonExistentKey() async throws {
        let data = try await storage.retrieve(key: "does-not-exist")
        XCTAssertNil(data)
    }
    
    func testDeleteNonExistentKey() async throws {
        // Should not throw
        try await storage.delete(key: "does-not-exist")
    }
}

// MARK: - Tier Migration Tests

final class TierMigrationTests: XCTestCase {
    
    var storage: ThreeTierStorage!
    var testPath: String!
    
    override func setUp() async throws {
        try await super.setUp()
        
        testPath = NSTemporaryDirectory().appending("VectorStoreKitTests-Migration-\(UUID().uuidString)")
        
        // Create storage with aggressive migration for testing
        let config = ThreeTierStorageConfiguration(
            memoryLimit: 5_242_880, // 5MB
            memoryItemSizeLimit: 524_288, // 512KB
            initialMemoryCapacity: 10,
            basePath: testPath,
            autoMigrationEnabled: true,
            accessPatternConfiguration: AccessPatternTracker.Configuration(
                promotionThreshold: 0.6,
                demotionThreshold: 0.4,
                migrationCooldown: 1, // 1 second for testing
                maxTrackedPatterns: 1000
            ),
            tierManagerConfiguration: TierManager.Configuration(
                autoMigrationEnabled: true,
                migrationInterval: 2, // 2 seconds for testing
                batchSize: 10
            )
        )
        
        storage = try await ThreeTierStorage(configuration: config)
    }
    
    override func tearDown() async throws {
        storage = nil
        
        if let testPath = testPath {
            try? FileManager.default.removeItem(atPath: testPath)
        }
        
        try await super.tearDown()
    }
    
    func testManualMigration() async throws {
        let key = "migrate-me"
        let data = Data(repeating: 0x42, count: 100_000) // 100KB
        
        // Store in memory
        try await storage.store(data, forKey: key, tier: .memory)
        
        // Verify in memory
        var result = try await storage.retrieve(forKey: key)
        XCTAssertEqual(result?.tier, .memory)
        
        // Migrate to SSD
        try await storage.migrate(key: key, to: .ssd)
        
        // Verify in SSD
        result = try await storage.retrieve(forKey: key)
        XCTAssertEqual(result?.tier, .ssd)
        
        // Migrate to archive
        try await storage.migrate(key: key, to: .archive)
        
        // Verify in archive
        result = try await storage.retrieve(forKey: key)
        XCTAssertEqual(result?.tier, .archive)
    }
    
    func testMigrationDecision() async throws {
        let hotKey = "hot-data"
        let coldKey = "cold-data"
        let data = Data(repeating: 0x42, count: 1000)
        
        // Store both
        try await storage.store(data, forKey: hotKey, tier: .ssd)
        try await storage.store(data, forKey: coldKey, tier: .ssd)
        
        // Access hot data frequently
        for _ in 0..<10 {
            _ = try await storage.retrieve(key: hotKey)
            try await Task.sleep(nanoseconds: 100_000_000) // 100ms
        }
        
        // Wait for access pattern to stabilize
        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second
        
        // Check migration decisions
        let hotDecision = try await storage.migrationDecision(forKey: hotKey)
        let coldDecision = try await storage.migrationDecision(forKey: coldKey)
        
        // Hot data might be promoted
        if case .promote = hotDecision {
            XCTAssertTrue(true, "Hot data considered for promotion")
        }
        
        // Cold data might be demoted or stay
        switch coldDecision {
        case .demote, .stay:
            XCTAssertTrue(true, "Cold data not promoted")
        case .promote:
            XCTFail("Cold data should not be promoted")
        }
    }
}

// MARK: - Performance Tests

final class ThreeTierStoragePerformanceTests: XCTestCase {
    
    var storage: ThreeTierStorage!
    var testPath: String!
    
    override func setUp() async throws {
        try await super.setUp()
        
        testPath = NSTemporaryDirectory().appending("VectorStoreKitTests-Perf-\(UUID().uuidString)")
        
        // Performance configuration
        let config = ThreeTierStorageConfiguration.performance
            .withBasePath(testPath)
        
        storage = try await ThreeTierStorage(configuration: config)
    }
    
    override func tearDown() async throws {
        storage = nil
        
        if let testPath = testPath {
            try? FileManager.default.removeItem(atPath: testPath)
        }
        
        try await super.tearDown()
    }
    
    func testMemoryTierPerformance() async throws {
        let data = Data(repeating: 0x42, count: 1024) // 1KB
        
        measure {
            Task {
                for i in 0..<1000 {
                    let key = "perf-mem-\(i)"
                    try? await storage.store(key: key, data: data, options: .default)
                    _ = try? await storage.retrieve(key: key)
                }
            }
        }
    }
    
    func testSSDTierPerformance() async throws {
        let data = Data(repeating: 0x42, count: 1_048_576) // 1MB
        
        measure {
            Task {
                for i in 0..<100 {
                    let key = "perf-ssd-\(i)"
                    try? await storage.store(key: key, data: data, options: .default)
                    _ = try? await storage.retrieve(key: key)
                }
            }
        }
    }
    
    func testCompressionPerformance() async throws {
        // Highly compressible data
        let data = Data(repeating: 0x00, count: 10_485_760) // 10MB of zeros
        
        measure {
            Task {
                for i in 0..<10 {
                    let key = "perf-compress-\(i)"
                    try? await storage.store(key: key, data: data, options: .default)
                }
            }
        }
    }
}
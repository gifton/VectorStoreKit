// VectorStoreKit: Partition Metadata Tests
//
// Comprehensive tests for partition metadata and management

import XCTest
@testable import VectorStoreKit

final class PartitionMetadataTests: XCTestCase {
    
    // MARK: - Metadata Tests
    
    func testPartitionMetadataCreation() {
        let metadata = PartitionMetadata.initial(
            partitionId: "test-partition",
            ownerNodeId: "node1"
        )
        
        XCTAssertEqual(metadata.partitionId, "test-partition")
        XCTAssertEqual(metadata.ownership.ownerNodeId, "node1")
        XCTAssertEqual(metadata.state, .creating)
        XCTAssertEqual(metadata.version.versionString, "1.0.0")
        XCTAssertFalse(metadata.ownership.isLeaseExpired)
    }
    
    func testVersionComparison() {
        let v1 = PartitionVersion(major: 1, minor: 0, patch: 0)
        let v2 = PartitionVersion(major: 1, minor: 1, patch: 0)
        let v3 = PartitionVersion(major: 2, minor: 0, patch: 0)
        
        XCTAssertTrue(v1 < v2)
        XCTAssertTrue(v2 < v3)
        XCTAssertFalse(v3 < v1)
    }
    
    func testLifecycleStateCapabilities() {
        // Active state
        XCTAssertTrue(PartitionLifecycleState.active.canServeReads)
        XCTAssertTrue(PartitionLifecycleState.active.canServeWrites)
        
        // Splitting state
        XCTAssertTrue(PartitionLifecycleState.splitting.canServeReads)
        XCTAssertFalse(PartitionLifecycleState.splitting.canServeWrites)
        
        // Error state
        XCTAssertFalse(PartitionLifecycleState.error.canServeReads)
        XCTAssertFalse(PartitionLifecycleState.error.canServeWrites)
    }
    
    func testStatisticsLoadScore() {
        let stats = PartitionStats(
            vectorCount: 1_000_000,
            sizeInBytes: 1_073_741_824, // 1 GB
            readCount: 5000,
            writeCount: 1000,
            avgQueryLatencyMs: 10,
            p99QueryLatencyMs: 50
        )
        
        let loadScore = stats.loadScore
        
        // Load score should consider size, access, and latency
        XCTAssertGreaterThan(loadScore, 0)
        
        // 1 GB = 1.0, 6000 ops = 6.0, 50ms p99 = 0.5
        let expectedScore = 1.0 + 6.0 + 0.5
        XCTAssertEqual(loadScore, expectedScore, accuracy: 0.1)
    }
    
    func testOwnershipLeaseExpiry() {
        let futureDate = Date().addingTimeInterval(300)
        let pastDate = Date().addingTimeInterval(-300)
        
        let activeOwnership = PartitionOwnership(
            ownerNodeId: "node1",
            leaseExpiry: futureDate
        )
        
        let expiredOwnership = PartitionOwnership(
            ownerNodeId: "node2",
            leaseExpiry: pastDate
        )
        
        XCTAssertFalse(activeOwnership.isLeaseExpired)
        XCTAssertTrue(expiredOwnership.isLeaseExpired)
        
        XCTAssertGreaterThan(activeOwnership.leaseTimeRemaining, 0)
        XCTAssertEqual(expiredOwnership.leaseTimeRemaining, 0)
    }
    
    func testHealthStatus() {
        let healthyCheck = HealthCheck(name: "disk", status: .healthy)
        let degradedCheck = HealthCheck(name: "memory", status: .degraded)
        let unhealthyCheck = HealthCheck(name: "network", status: .unhealthy)
        
        let overallHealthy = PartitionHealth(
            status: .healthy,
            checks: [healthyCheck]
        )
        
        let overallDegraded = PartitionHealth(
            status: .degraded,
            checks: [healthyCheck, degradedCheck]
        )
        
        let overallUnhealthy = PartitionHealth(
            status: .unhealthy,
            checks: [healthyCheck, degradedCheck, unhealthyCheck]
        )
        
        XCTAssertEqual(overallHealthy.status, .healthy)
        XCTAssertEqual(overallDegraded.status, .degraded)
        XCTAssertEqual(overallUnhealthy.status, .unhealthy)
    }
    
    // MARK: - Metadata Update Tests
    
    func testMetadataStateUpdate() {
        let metadata = PartitionMetadata.initial(
            partitionId: "test",
            ownerNodeId: "node1"
        )
        
        let updatedMetadata = metadata.applying(.setState(.active))
        
        XCTAssertEqual(updatedMetadata.state, .active)
        XCTAssertEqual(metadata.state, .creating) // Original unchanged
        XCTAssertGreaterThan(updatedMetadata.timestamps.lastModified, metadata.timestamps.lastModified)
    }
    
    func testMetadataStatisticsUpdate() {
        let metadata = PartitionMetadata.initial(
            partitionId: "test",
            ownerNodeId: "node1"
        )
        
        let newStats = PartitionStats(
            vectorCount: 1000,
            sizeInBytes: 1024 * 1024,
            readCount: 100,
            writeCount: 50
        )
        
        let updatedMetadata = metadata.applying(.updateStatistics(newStats))
        
        XCTAssertEqual(updatedMetadata.statistics.vectorCount, 1000)
        XCTAssertEqual(updatedMetadata.statistics.sizeInBytes, 1024 * 1024)
    }
    
    func testMetadataLeaseRenewal() {
        let metadata = PartitionMetadata.initial(
            partitionId: "test",
            ownerNodeId: "node1"
        )
        
        let newExpiry = Date().addingTimeInterval(600)
        let updatedMetadata = metadata.applying(.renewLease(
            newExpiry: newExpiry,
            version: 2
        ))
        
        XCTAssertEqual(updatedMetadata.ownership.leaseExpiry, newExpiry)
        XCTAssertEqual(updatedMetadata.ownership.leaseVersion, 2)
        XCTAssertEqual(updatedMetadata.ownership.ownerNodeId, "node1") // Same owner
    }
    
    func testMetadataOwnershipTransfer() {
        let metadata = PartitionMetadata.initial(
            partitionId: "test",
            ownerNodeId: "node1"
        )
        
        let newExpiry = Date().addingTimeInterval(300)
        let updatedMetadata = metadata.applying(.transferOwnership(
            newOwner: "node2",
            expiry: newExpiry
        ))
        
        XCTAssertEqual(updatedMetadata.ownership.ownerNodeId, "node2")
        XCTAssertEqual(updatedMetadata.ownership.leaseVersion, 2)
        XCTAssertEqual(updatedMetadata.ownership.previousOwners.count, 1)
        XCTAssertEqual(updatedMetadata.ownership.previousOwners.first?.nodeId, "node1")
    }
    
    func testMetadataCustomFields() {
        let metadata = PartitionMetadata.initial(
            partitionId: "test",
            ownerNodeId: "node1"
        )
        
        let withCustom = metadata
            .applying(.addCustomMetadata(key: "region", value: "us-west"))
            .applying(.addCustomMetadata(key: "tier", value: "hot"))
        
        XCTAssertEqual(withCustom.customMetadata["region"], "us-west")
        XCTAssertEqual(withCustom.customMetadata["tier"], "hot")
        
        let removed = withCustom.applying(.removeCustomMetadata(key: "region"))
        XCTAssertNil(removed.customMetadata["region"])
        XCTAssertEqual(removed.customMetadata["tier"], "hot")
    }
    
    func testMigrationRecord() {
        let metadata = PartitionMetadata.initial(
            partitionId: "test",
            ownerNodeId: "node1"
        )
        
        let migration = MigrationRecord(
            fromVersion: "1.0.0",
            toVersion: "2.0.0",
            timestamp: Date(),
            duration: 120.5,
            success: true,
            details: "Schema update"
        )
        
        let updated = metadata.applying(.recordMigration(migration))
        
        XCTAssertEqual(updated.version.migrationHistory.count, 1)
        XCTAssertEqual(updated.version.migrationHistory.first?.toVersion, "2.0.0")
    }
}

// MARK: - Partition Manager Tests

final class PartitionManagerTests: XCTestCase {
    
    var manager: PartitionManager!
    
    override func setUp() async throws {
        try await super.setUp()
        
        let config = PartitionManager.Configuration(
            leaseDuration: 5, // Short for testing
            leaseRenewalInterval: 1,
            healthCheckInterval: 1,
            autoRenewLeases: false, // Manual control for tests
            enableHealthMonitoring: false
        )
        
        manager = PartitionManager(configuration: config)
    }
    
    // MARK: - Partition Creation Tests
    
    func testCreatePartition() async throws {
        let metadata = try await manager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 1000)
        )
        
        XCTAssertEqual(metadata.partitionId, "p1")
        XCTAssertEqual(metadata.ownership.ownerNodeId, "node1")
        XCTAssertEqual(metadata.state, .creating)
        
        // Verify partition can be retrieved
        let retrieved = await manager.getPartition("p1")
        XCTAssertNotNil(retrieved)
        XCTAssertEqual(retrieved?.partitionId, "p1")
    }
    
    func testCreateDuplicatePartition() async throws {
        _ = try await manager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 1000)
        )
        
        // Try to create duplicate
        do {
            _ = try await manager.createPartition(
                id: "p1",
                ownerNode: "node2",
                bounds: .hash(min: 0, max: 1000)
            )
            XCTFail("Should have thrown error")
        } catch PartitionManagerError.partitionAlreadyExists {
            // Expected
        }
    }
    
    func testActivatePartition() async throws {
        let metadata = try await manager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 1000)
        )
        
        try await manager.activatePartition("p1")
        
        let activated = await manager.getPartition("p1")
        XCTAssertEqual(activated?.state, .active)
    }
    
    // MARK: - Lease Management Tests
    
    func testAcquireLease() async throws {
        _ = try await manager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 1000)
        )
        
        // Renew lease for same node
        let lease = try await manager.acquireLease(
            partitionId: "p1",
            nodeId: "node1"
        )
        
        XCTAssertEqual(lease.nodeId, "node1")
        XCTAssertGreaterThan(lease.version, 1)
    }
    
    func testLeaseTransfer() async throws {
        _ = try await manager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 1000)
        )
        
        // Wait for lease to expire
        try await Task.sleep(nanoseconds: 6_000_000_000) // 6 seconds
        
        // Node2 acquires lease
        let lease = try await manager.acquireLease(
            partitionId: "p1",
            nodeId: "node2"
        )
        
        XCTAssertEqual(lease.nodeId, "node2")
        
        let metadata = await manager.getPartition("p1")
        XCTAssertEqual(metadata?.ownership.ownerNodeId, "node2")
        XCTAssertEqual(metadata?.ownership.previousOwners.count, 1)
    }
    
    func testLeaseConflict() async throws {
        _ = try await manager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 1000)
        )
        
        // Try to acquire lease from different node while active
        do {
            _ = try await manager.acquireLease(
                partitionId: "p1",
                nodeId: "node2"
            )
            XCTFail("Should have thrown error")
        } catch PartitionManagerError.leaseHeldByOtherNode {
            // Expected
        }
    }
    
    // MARK: - Split Tests
    
    func testPartitionSplit() async throws {
        let metadata = try await manager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 1000)
        )
        
        try await manager.activatePartition("p1")
        
        // Start split
        try await manager.startPartitionSplit(
            partitionId: "p1",
            targetPartitions: ["p1_left", "p1_right"]
        )
        
        let splitting = await manager.getPartition("p1")
        XCTAssertEqual(splitting?.state, .splitting)
        
        // Complete split
        try await manager.completePartitionSplit(
            sourcePartitionId: "p1",
            resultPartitions: [
                (id: "p1_left", owner: "node1", bounds: .hash(min: 0, max: 500)),
                (id: "p1_right", owner: "node1", bounds: .hash(min: 501, max: 1000))
            ]
        )
        
        // Verify new partitions exist
        let left = await manager.getPartition("p1_left")
        let right = await manager.getPartition("p1_right")
        
        XCTAssertNotNil(left)
        XCTAssertNotNil(right)
        
        // Original partition should be marked for deletion
        let original = await manager.getPartition("p1")
        XCTAssertEqual(original?.state, .deleting)
    }
    
    // MARK: - Health Management Tests
    
    func testHealthUpdates() async throws {
        _ = try await manager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 1000)
        )
        
        // Add health checks
        try await manager.updateHealth(
            partitionId: "p1",
            healthCheck: HealthCheck(name: "disk", status: .healthy)
        )
        
        try await manager.updateHealth(
            partitionId: "p1",
            healthCheck: HealthCheck(name: "memory", status: .degraded)
        )
        
        let metadata = await manager.getPartition("p1")
        XCTAssertEqual(metadata?.health.status, .degraded)
        XCTAssertEqual(metadata?.health.checks.count, 2)
    }
    
    // MARK: - Statistics Tests
    
    func testStatisticsUpdate() async throws {
        _ = try await manager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 1000)
        )
        
        let stats = PartitionStats(
            vectorCount: 5000,
            sizeInBytes: 1024 * 1024 * 100,
            readCount: 1000,
            writeCount: 500
        )
        
        try await manager.updateStatistics(partitionId: "p1", stats: stats)
        
        let metadata = await manager.getPartition("p1")
        XCTAssertEqual(metadata?.statistics.vectorCount, 5000)
    }
    
    func testNodeStatistics() async throws {
        // Create multiple partitions on same node
        for i in 0..<3 {
            _ = try await manager.createPartition(
                id: "p\(i)",
                ownerNode: "node1",
                bounds: .hash(min: i * 1000, max: (i + 1) * 1000)
            )
            
            let stats = PartitionStats(
                vectorCount: 1000,
                sizeInBytes: 1024 * 1024,
                readCount: 100,
                writeCount: 50
            )
            
            try await manager.updateStatistics(partitionId: "p\(i)", stats: stats)
        }
        
        let nodeStats = await manager.getNodeStatistics(nodeId: "node1")
        
        XCTAssertEqual(nodeStats.partitionCount, 3)
        XCTAssertEqual(nodeStats.totalVectors, 3000)
        XCTAssertEqual(nodeStats.totalReads, 300)
    }
    
    // MARK: - Query Tests
    
    func testFindPartitionsByState() async throws {
        // Create partitions in different states
        let p1 = try await manager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 1000)
        )
        
        let p2 = try await manager.createPartition(
            id: "p2",
            ownerNode: "node1",
            bounds: .hash(min: 1001, max: 2000)
        )
        
        try await manager.activatePartition("p2")
        
        let creating = await manager.findPartitions(byState: .creating)
        let active = await manager.findPartitions(byState: .active)
        
        XCTAssertEqual(creating.count, 1)
        XCTAssertEqual(creating.first?.partitionId, "p1")
        
        XCTAssertEqual(active.count, 1)
        XCTAssertEqual(active.first?.partitionId, "p2")
    }
    
    // MARK: - Event Tests
    
    func testEventSubscription() async throws {
        let expectation = XCTestExpectation(description: "Event received")
        var receivedEvents: [PartitionEvent] = []
        
        let subscriptionId = await manager.subscribe { event in
            receivedEvents.append(event)
            expectation.fulfill()
        }
        
        _ = try await manager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 1000)
        )
        
        await fulfillment(of: [expectation], timeout: 5)
        
        XCTAssertEqual(receivedEvents.count, 1)
        if case .created(let partitionId, let nodeId) = receivedEvents.first {
            XCTAssertEqual(partitionId, "p1")
            XCTAssertEqual(nodeId, "node1")
        } else {
            XCTFail("Expected created event")
        }
        
        await manager.unsubscribe(subscriptionId)
    }
}
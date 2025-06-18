// VectorStoreKit: Partition Assignment Tests
//
// Comprehensive tests for partition assignment and rebalancing

import XCTest
@testable import VectorStoreKit

final class PartitionAssignmentTests: XCTestCase {
    
    // MARK: - Node Info Tests
    
    func testNodeInfoCapacity() {
        let node = NodeInfo(
            nodeId: "node1",
            capacity: NodeCapacity(maxPartitions: 10),
            currentLoad: NodeLoad(partitionCount: 7),
            health: NodeHealth(status: .healthy)
        )
        
        XCTAssertTrue(node.canAcceptPartitions)
        XCTAssertEqual(node.availableCapacityRatio, 0.3, accuracy: 0.01)
        
        // Full node
        let fullNode = NodeInfo(
            nodeId: "node2",
            capacity: NodeCapacity(maxPartitions: 10),
            currentLoad: NodeLoad(partitionCount: 10),
            health: NodeHealth(status: .healthy)
        )
        
        XCTAssertFalse(fullNode.canAcceptPartitions)
        XCTAssertEqual(fullNode.availableCapacityRatio, 0.0)
    }
    
    func testNodeHealthStatus() {
        let healthyNode = NodeInfo(
            nodeId: "node1",
            capacity: NodeCapacity(),
            currentLoad: NodeLoad(),
            health: NodeHealth(status: .healthy)
        )
        
        let unhealthyNode = NodeInfo(
            nodeId: "node2",
            capacity: NodeCapacity(),
            currentLoad: NodeLoad(),
            health: NodeHealth(status: .unhealthy)
        )
        
        XCTAssertTrue(healthyNode.canAcceptPartitions)
        XCTAssertFalse(unhealthyNode.canAcceptPartitions)
    }
    
    // MARK: - Assignment Strategy Tests
    
    func testLeastLoadedStrategy() async {
        let strategy = LeastLoadedStrategy()
        
        let nodes = [
            NodeInfo(
                nodeId: "node1",
                capacity: NodeCapacity(maxPartitions: 10),
                currentLoad: NodeLoad(partitionCount: 5),
                health: NodeHealth()
            ),
            NodeInfo(
                nodeId: "node2",
                capacity: NodeCapacity(maxPartitions: 10),
                currentLoad: NodeLoad(partitionCount: 2),
                health: NodeHealth()
            ),
            NodeInfo(
                nodeId: "node3",
                capacity: NodeCapacity(maxPartitions: 10),
                currentLoad: NodeLoad(partitionCount: 7),
                health: NodeHealth()
            )
        ]
        
        let assignments = [
            "node1": Set(["p1", "p2", "p3", "p4", "p5"]),
            "node2": Set(["p6", "p7"]),
            "node3": Set(["p8", "p9", "p10", "p11", "p12", "p13", "p14"])
        ]
        
        let partition = Partition(id: "p15", nodeId: "", bounds: .hash(min: 0, max: 100))
        
        let selectedNode = await strategy.assignPartition(
            partition,
            availableNodes: nodes,
            currentAssignments: assignments
        )
        
        // Should select node2 (least loaded)
        XCTAssertEqual(selectedNode, "node2")
    }
    
    func testLocationAwareStrategy() async {
        let preferredLocation = NodeLocation(region: "us-west", zone: "us-west-1a")
        let strategy = LocationAwareStrategy(preferredLocations: [preferredLocation])
        
        let nodes = [
            NodeInfo(
                nodeId: "node1",
                capacity: NodeCapacity(),
                currentLoad: NodeLoad(partitionCount: 5),
                health: NodeHealth(),
                location: NodeLocation(region: "us-east", zone: "us-east-1a")
            ),
            NodeInfo(
                nodeId: "node2",
                capacity: NodeCapacity(),
                currentLoad: NodeLoad(partitionCount: 5),
                health: NodeHealth(),
                location: preferredLocation
            ),
            NodeInfo(
                nodeId: "node3",
                capacity: NodeCapacity(),
                currentLoad: NodeLoad(partitionCount: 2),
                health: NodeHealth(),
                location: NodeLocation(region: "eu-west", zone: "eu-west-1a")
            )
        ]
        
        let partition = Partition(id: "p1", nodeId: "", bounds: .hash(min: 0, max: 100))
        
        let selectedNode = await strategy.assignPartition(
            partition,
            availableNodes: nodes,
            currentAssignments: [:]
        )
        
        // Should select node2 (preferred location) even though node3 has less load
        XCTAssertEqual(selectedNode, "node2")
    }
    
    func testRoundRobinStrategy() async {
        let strategy = RoundRobinStrategy(nodeOrder: ["node1", "node2", "node3"])
        
        let nodes = [
            NodeInfo(nodeId: "node1", capacity: NodeCapacity(), currentLoad: NodeLoad(), health: NodeHealth()),
            NodeInfo(nodeId: "node2", capacity: NodeCapacity(), currentLoad: NodeLoad(), health: NodeHealth()),
            NodeInfo(nodeId: "node3", capacity: NodeCapacity(), currentLoad: NodeLoad(), health: NodeHealth())
        ]
        
        let partitions = [
            Partition(id: "p1", nodeId: "", bounds: .hash(min: 0, max: 100)),
            Partition(id: "p2", nodeId: "", bounds: .hash(min: 101, max: 200)),
            Partition(id: "p3", nodeId: "", bounds: .hash(min: 201, max: 300)),
            Partition(id: "p4", nodeId: "", bounds: .hash(min: 301, max: 400))
        ]
        
        let assignments = await strategy.assignPartitions(
            partitions,
            availableNodes: nodes,
            currentAssignments: [:]
        )
        
        // Should distribute round-robin
        XCTAssertEqual(assignments["p1"], "node1")
        XCTAssertEqual(assignments["p2"], "node2")
        XCTAssertEqual(assignments["p3"], "node3")
        XCTAssertEqual(assignments["p4"], "node1")
    }
    
    // MARK: - Partition Assigner Tests
    
    func testPartitionAssigner() async throws {
        let strategy = LeastLoadedStrategy()
        let assigner = PartitionAssigner(strategy: strategy)
        
        // Register nodes
        let node1 = NodeInfo(
            nodeId: "node1",
            capacity: NodeCapacity(maxPartitions: 10),
            currentLoad: NodeLoad(),
            health: NodeHealth()
        )
        
        let node2 = NodeInfo(
            nodeId: "node2",
            capacity: NodeCapacity(maxPartitions: 10),
            currentLoad: NodeLoad(),
            health: NodeHealth()
        )
        
        await assigner.registerNode(node1)
        await assigner.registerNode(node2)
        
        // Assign partitions
        let partition1 = Partition(id: "p1", nodeId: "", bounds: .hash(min: 0, max: 100))
        let assignedNode1 = try await assigner.assignPartition(partition1)
        
        XCTAssertTrue(["node1", "node2"].contains(assignedNode1))
        
        // Verify assignment is recorded
        let assignment = await assigner.getAssignment(for: "p1")
        XCTAssertEqual(assignment, assignedNode1)
        
        // Verify node assignments
        let nodeAssignments = await assigner.getNodeAssignments(assignedNode1)
        XCTAssertTrue(nodeAssignments.contains("p1"))
    }
    
    func testMultiplePartitionAssignment() async throws {
        let strategy = LeastLoadedStrategy()
        let assigner = PartitionAssigner(strategy: strategy)
        
        // Register nodes
        for i in 1...3 {
            let node = NodeInfo(
                nodeId: "node\(i)",
                capacity: NodeCapacity(maxPartitions: 10),
                currentLoad: NodeLoad(),
                health: NodeHealth()
            )
            await assigner.registerNode(node)
        }
        
        // Create partitions
        var partitions: [Partition] = []
        for i in 1...9 {
            partitions.append(
                Partition(id: "p\(i)", nodeId: "", bounds: .hash(min: i * 100, max: (i + 1) * 100))
            )
        }
        
        // Assign all partitions
        let assignments = try await assigner.assignPartitions(partitions)
        
        XCTAssertEqual(assignments.count, 9)
        
        // Check balance
        let report = await assigner.getBalanceReport()
        XCTAssertEqual(report.totalPartitions, 9)
        XCTAssertEqual(report.avgPartitionsPerNode, 3.0)
        
        // Should be well balanced
        XCTAssertLessThanOrEqual(report.maxLoad - report.minLoad, 1)
    }
    
    func testNodeRemoval() async throws {
        let assigner = PartitionAssigner(strategy: LeastLoadedStrategy())
        
        // Register and remove node
        let node = NodeInfo(nodeId: "node1", capacity: NodeCapacity(), currentLoad: NodeLoad(), health: NodeHealth())
        await assigner.registerNode(node)
        await assigner.removeNode("node1")
        
        // Try to assign - should fail
        let partition = Partition(id: "p1", nodeId: "", bounds: .hash(min: 0, max: 100))
        
        do {
            _ = try await assigner.assignPartition(partition)
            XCTFail("Should have thrown error")
        } catch AssignmentError.noAvailableNodes {
            // Expected
        }
    }
    
    func testBalanceReport() async throws {
        let assigner = PartitionAssigner(strategy: RoundRobinStrategy())
        
        // Register nodes
        for i in 1...4 {
            let node = NodeInfo(
                nodeId: "node\(i)",
                capacity: NodeCapacity(maxPartitions: 20),
                currentLoad: NodeLoad(),
                health: NodeHealth()
            )
            await assigner.registerNode(node)
        }
        
        // Assign uneven number of partitions
        var partitions: [Partition] = []
        for i in 1...10 {
            partitions.append(
                Partition(id: "p\(i)", nodeId: "", bounds: .hash(min: i * 100, max: (i + 1) * 100))
            )
        }
        
        _ = try await assigner.assignPartitions(partitions)
        
        let report = await assigner.getBalanceReport()
        
        XCTAssertEqual(report.totalPartitions, 10)
        XCTAssertEqual(report.nodeCount, 4)
        XCTAssertEqual(report.avgPartitionsPerNode, 2.5)
        XCTAssertGreaterThan(report.standardDeviation, 0)
        
        // With round-robin, should be fairly balanced
        XCTAssertLessThanOrEqual(report.imbalanceRatio, 0.5)
    }
}

// MARK: - Rebalancing Coordinator Tests

final class RebalancingCoordinatorTests: XCTestCase {
    
    var partitionManager: PartitionManager!
    var partitionAssigner: PartitionAssigner!
    var coordinator: RebalancingCoordinator!
    
    override func setUp() async throws {
        try await super.setUp()
        
        partitionManager = PartitionManager(
            configuration: PartitionManager.Configuration(
                leaseDuration: 300,
                autoRenewLeases: false,
                enableHealthMonitoring: false
            )
        )
        
        partitionAssigner = PartitionAssigner(strategy: LeastLoadedStrategy())
        
        coordinator = RebalancingCoordinator(
            partitionManager: partitionManager,
            partitionAssigner: partitionAssigner,
            configuration: RebalancingCoordinator.Configuration(
                minImprovementThreshold: 0.1,
                cooldownPeriod: 1, // Short for testing
                dryRunMode: true // Use dry run for tests
            )
        )
    }
    
    func testRebalancingPlanGeneration() async throws {
        // Create imbalanced setup
        await setupImbalancedSystem()
        
        // Generate plan
        let plan = try await coordinator.analyzeAndPlan()
        
        XCTAssertNotNil(plan)
        XCTAssertFalse(plan!.moves.isEmpty)
        XCTAssertGreaterThan(plan!.estimatedImprovement, 0)
    }
    
    func testBalancedSystemNoPlan() async throws {
        // Create balanced setup
        await setupBalancedSystem()
        
        // Should not generate plan
        let plan = try await coordinator.analyzeAndPlan()
        XCTAssertNil(plan)
    }
    
    func testRebalancingExecution() async throws {
        // Create imbalanced setup
        await setupImbalancedSystem()
        
        // Generate and execute plan
        let plan = try await coordinator.analyzeAndPlan()!
        let operationId = try await coordinator.executeRebalancing(plan)
        
        XCTAssertFalse(operationId.isEmpty)
        
        // Wait for completion
        try await Task.sleep(nanoseconds: UInt64(plan.moves.count + 1) * 1_000_000_000)
        
        // Check status
        let status = await coordinator.getOperationStatus(operationId)
        XCTAssertNotNil(status)
        XCTAssertEqual(status?.state, "completed")
        XCTAssertEqual(status?.progress, 1.0)
        XCTAssertEqual(status?.completedMoves, plan.moves.count)
    }
    
    func testOperationCancellation() async throws {
        // Create large plan
        let moves = (0..<10).map { i in
            PartitionMove(
                partitionId: "p\(i)",
                fromNode: "node1",
                toNode: "node2",
                reason: "Test"
            )
        }
        
        let plan = RebalancingPlan(moves: moves, estimatedImprovement: 0.5)
        let operationId = try await coordinator.executeRebalancing(plan)
        
        // Cancel immediately
        try await coordinator.cancelOperation(operationId)
        
        // Check status
        let status = await coordinator.getOperationStatus(operationId)
        XCTAssertNotNil(status)
        XCTAssertTrue(["cancelling", "failed"].contains(status?.state))
    }
    
    func testCooldownPeriod() async throws {
        // Create and execute first operation
        await setupImbalancedSystem()
        
        let plan1 = try await coordinator.analyzeAndPlan()!
        _ = try await coordinator.executeRebalancing(plan1)
        
        // Try immediately again - should be in cooldown
        let plan2 = try await coordinator.analyzeAndPlan()
        XCTAssertNil(plan2)
        
        // Wait for cooldown
        try await Task.sleep(nanoseconds: 2_000_000_000)
        
        // Now should work
        let plan3 = try await coordinator.analyzeAndPlan()
        // Might be nil if system is now balanced
    }
    
    // MARK: - Helper Methods
    
    private func setupImbalancedSystem() async {
        // Register nodes
        for i in 1...3 {
            let node = NodeInfo(
                nodeId: "node\(i)",
                capacity: NodeCapacity(maxPartitions: 10),
                currentLoad: NodeLoad(),
                health: NodeHealth()
            )
            await partitionAssigner.registerNode(node)
        }
        
        // Create partitions with imbalanced assignment
        // Node1: 7 partitions, Node2: 2 partitions, Node3: 1 partition
        for i in 1...10 {
            let nodeId: String
            if i <= 7 {
                nodeId = "node1"
            } else if i <= 9 {
                nodeId = "node2"
            } else {
                nodeId = "node3"
            }
            
            _ = try? await partitionManager.createPartition(
                id: "p\(i)",
                ownerNode: nodeId,
                bounds: .hash(min: i * 100, max: (i + 1) * 100)
            )
            
            // Record in assigner
            _ = try? await partitionAssigner.assignPartition(
                Partition(id: "p\(i)", nodeId: nodeId, bounds: .hash(min: i * 100, max: (i + 1) * 100))
            )
        }
    }
    
    private func setupBalancedSystem() async {
        // Register nodes
        for i in 1...3 {
            let node = NodeInfo(
                nodeId: "node\(i)",
                capacity: NodeCapacity(maxPartitions: 10),
                currentLoad: NodeLoad(),
                health: NodeHealth()
            )
            await partitionAssigner.registerNode(node)
        }
        
        // Create partitions with balanced assignment
        // Each node gets 3-4 partitions
        for i in 1...10 {
            let nodeId = "node\((i - 1) % 3 + 1)"
            
            _ = try? await partitionManager.createPartition(
                id: "p\(i)",
                ownerNode: nodeId,
                bounds: .hash(min: i * 100, max: (i + 1) * 100)
            )
            
            // Record in assigner
            _ = try? await partitionAssigner.assignPartition(
                Partition(id: "p\(i)", nodeId: nodeId, bounds: .hash(min: i * 100, max: (i + 1) * 100))
            )
        }
    }
}

// MARK: - Partition Migrator Tests

final class PartitionMigratorTests: XCTestCase {
    
    var partitionManager: PartitionManager!
    var migrator: PartitionMigrator!
    
    override func setUp() async throws {
        try await super.setUp()
        
        partitionManager = PartitionManager()
        migrator = PartitionMigrator()
    }
    
    func testBasicMigration() async throws {
        // Create partition
        _ = try await partitionManager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 100)
        )
        
        try await partitionManager.activatePartition("p1")
        
        // Migrate to node2
        try await migrator.migratePartition(
            partitionId: "p1",
            fromNode: "node1",
            toNode: "node2",
            partitionManager: partitionManager
        )
        
        // Verify ownership changed
        let metadata = await partitionManager.getPartition("p1")
        XCTAssertEqual(metadata?.ownership.ownerNodeId, "node2")
    }
    
    func testMigrationWithWrongOwner() async throws {
        // Create partition owned by node1
        _ = try await partitionManager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 100)
        )
        
        // Try to migrate from wrong node
        do {
            try await migrator.migratePartition(
                partitionId: "p1",
                fromNode: "node2",
                toNode: "node3",
                partitionManager: partitionManager
            )
            XCTFail("Should have thrown error")
        } catch RebalancingError.incorrectOwner {
            // Expected
        }
    }
}
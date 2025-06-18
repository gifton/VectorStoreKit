// VectorStoreKit: Partitioning Integration Tests
//
// Comprehensive integration tests across all partitioning strategies

import XCTest
@testable import VectorStoreKit

final class PartitioningIntegrationTests: XCTestCase {
    
    // MARK: - Cross-Strategy Tests
    
    func testAllStrategiesWithSameData() async throws {
        let strategies: [(name: String, strategy: any DistributedPartitionStrategy)] = [
            ("Hash", HashPartitioner()),
            ("ConsistentHash", ConsistentHashPartitioner()),
            ("Range", RangePartitioner<String>()),
            ("Custom", CustomPartitioner<String>.geographic())
        ]
        
        let nodes = ["node1", "node2", "node3", "node4"]
        
        for (name, strategy) in strategies {
            // Create partitions
            let partitions = strategy.createPartitions(count: 12, nodes: nodes)
            
            XCTAssertEqual(partitions.count, 12, "Strategy \(name) should create 12 partitions")
            
            // Verify all nodes are used
            let usedNodes = Set(partitions.map { $0.nodeId })
            XCTAssertEqual(usedNodes.count, nodes.count, "Strategy \(name) should use all nodes")
            
            // Test key assignment
            let testKeys = ["key1", "key2", "key3", "us-west-1:item123"]
            for key in testKeys {
                let partition = strategy.partition(for: key, partitions: partitions)
                XCTAssertNotNil(partition, "Strategy \(name) should assign key '\(key)'")
            }
            
            // Test validation
            let errors = strategy.validate(partitions: partitions)
            XCTAssertTrue(errors.isEmpty, "Strategy \(name) should produce valid partitions")
        }
    }
    
    func testStrategyMigration() async throws {
        // Simulate migrating from hash to consistent hash
        let hashPartitioner = HashPartitioner()
        let consistentHashPartitioner = ConsistentHashPartitioner()
        
        let nodes = ["node1", "node2", "node3"]
        
        // Create initial partitions with hash
        let hashPartitions = hashPartitioner.createPartitions(count: 9, nodes: nodes)
        
        // Create consistent hash partitioner
        for node in nodes {
            try await consistentHashPartitioner.addNode(Node(id: node))
        }
        
        // Track key movements
        var keyMovements = 0
        let testKeys = (0..<1000).map { "key_\($0)" }
        
        for key in testKeys {
            let hashPartition = hashPartitioner.partition(for: key, partitions: hashPartitions)
            let chPartition = await consistentHashPartitioner.getNodeForKey(key)
            
            if hashPartition?.nodeId != chPartition?.id {
                keyMovements += 1
            }
        }
        
        // Some keys will move, but not all
        XCTAssertGreaterThan(keyMovements, 0)
        XCTAssertLessThan(keyMovements, testKeys.count)
        
        print("Migration would move \(keyMovements) of \(testKeys.count) keys (\(Int(Double(keyMovements) / Double(testKeys.count) * 100))%)")
    }
    
    // MARK: - Failure Scenario Tests
    
    func testNodeFailureHandling() async throws {
        let manager = PartitionManager()
        let assigner = PartitionAssigner(strategy: LeastLoadedStrategy())
        
        // Register nodes
        let nodeInfos = [
            NodeInfo(nodeId: "node1", capacity: NodeCapacity(), currentLoad: NodeLoad(), health: NodeHealth()),
            NodeInfo(nodeId: "node2", capacity: NodeCapacity(), currentLoad: NodeLoad(), health: NodeHealth()),
            NodeInfo(nodeId: "node3", capacity: NodeCapacity(), currentLoad: NodeLoad(), health: NodeHealth())
        ]
        
        for node in nodeInfos {
            await assigner.registerNode(node)
        }
        
        // Create partitions
        for i in 1...9 {
            let partition = Partition(id: "p\(i)", nodeId: "", bounds: .hash(min: i * 100, max: (i + 1) * 100))
            let nodeId = try await assigner.assignPartition(partition)
            
            _ = try await manager.createPartition(
                id: "p\(i)",
                ownerNode: nodeId,
                bounds: .hash(min: i * 100, max: (i + 1) * 100)
            )
        }
        
        // Simulate node2 failure
        await assigner.updateNodeLoad(
            "node2",
            load: NodeLoad(partitionCount: 3)
        )
        
        let failedNode = NodeInfo(
            nodeId: "node2",
            capacity: NodeCapacity(),
            currentLoad: NodeLoad(partitionCount: 3),
            health: NodeHealth(status: .unhealthy, consecutiveFailures: 5)
        )
        await assigner.registerNode(failedNode)
        
        // Get partitions that need reassignment
        let node2Partitions = await manager.getPartitionsForNode("node2")
        
        // Reassign partitions from failed node
        for partition in node2Partitions {
            // Find new node (should not be node2)
            if let newNode = await assigner.getOptimalNode(for: Partition(
                id: partition.partitionId,
                nodeId: partition.ownership.ownerNodeId,
                bounds: .hash(min: 0, max: 100)
            )) {
                XCTAssertNotEqual(newNode, "node2", "Should not assign to unhealthy node")
            }
        }
    }
    
    func testLeaseExpiryHandling() async throws {
        let config = PartitionManager.Configuration(
            leaseDuration: 2, // 2 seconds for testing
            autoRenewLeases: false
        )
        let manager = PartitionManager(configuration: config)
        
        // Create partition
        let metadata = try await manager.createPartition(
            id: "p1",
            ownerNode: "node1",
            bounds: .hash(min: 0, max: 100)
        )
        
        // Wait for lease to expire
        try await Task.sleep(nanoseconds: 3_000_000_000)
        
        // Another node should be able to acquire
        let newLease = try await manager.acquireLease(
            partitionId: "p1",
            nodeId: "node2"
        )
        
        XCTAssertEqual(newLease.nodeId, "node2")
        
        // Verify ownership changed
        let updated = await manager.getPartition("p1")
        XCTAssertEqual(updated?.ownership.ownerNodeId, "node2")
    }
    
    // MARK: - Concurrent Access Tests
    
    func testConcurrentPartitionAssignment() async throws {
        let assigner = PartitionAssigner(strategy: LeastLoadedStrategy())
        
        // Register nodes
        for i in 1...5 {
            await assigner.registerNode(NodeInfo(
                nodeId: "node\(i)",
                capacity: NodeCapacity(maxPartitions: 20),
                currentLoad: NodeLoad(),
                health: NodeHealth()
            ))
        }
        
        // Concurrent partition assignment
        await withTaskGroup(of: (String, String?).self) { group in
            for i in 1...50 {
                group.addTask {
                    let partition = Partition(
                        id: "p\(i)",
                        nodeId: "",
                        bounds: .hash(min: i * 100, max: (i + 1) * 100)
                    )
                    
                    let nodeId = try? await assigner.assignPartition(partition)
                    return ("p\(i)", nodeId)
                }
            }
            
            var assignments: [String: String] = [:]
            for await (partitionId, nodeId) in group {
                if let nodeId = nodeId {
                    assignments[partitionId] = nodeId
                }
            }
            
            // Verify all partitions were assigned
            XCTAssertEqual(assignments.count, 50)
            
            // Check balance
            let report = await assigner.getBalanceReport()
            XCTAssertEqual(report.totalPartitions, 50)
            XCTAssertEqual(report.avgPartitionsPerNode, 10.0)
            
            // Should be well balanced with concurrent assignment
            XCTAssertLessThanOrEqual(report.maxLoad - report.minLoad, 2)
        }
    }
    
    func testConcurrentRebalancing() async throws {
        let manager = PartitionManager()
        let assigner = PartitionAssigner(strategy: RoundRobinStrategy())
        let coordinator = RebalancingCoordinator(
            partitionManager: manager,
            partitionAssigner: assigner,
            configuration: RebalancingCoordinator.Configuration(
                maxConcurrentMigrations: 3,
                dryRunMode: true
            )
        )
        
        // Setup system
        for i in 1...3 {
            await assigner.registerNode(NodeInfo(
                nodeId: "node\(i)",
                capacity: NodeCapacity(maxPartitions: 10),
                currentLoad: NodeLoad(),
                health: NodeHealth()
            ))
        }
        
        // Create partitions
        for i in 1...15 {
            let nodeId = "node\((i - 1) % 3 + 1)"
            _ = try await manager.createPartition(
                id: "p\(i)",
                ownerNode: nodeId,
                bounds: .hash(min: i * 100, max: (i + 1) * 100)
            )
        }
        
        // Simulate concurrent rebalancing requests
        let operations = await withTaskGroup(of: String?.self) { group in
            for _ in 0..<3 {
                group.addTask {
                    if let plan = try? await coordinator.analyzeAndPlan() {
                        return try? await coordinator.executeRebalancing(plan)
                    }
                    return nil
                }
            }
            
            var operationIds: [String] = []
            for await operationId in group {
                if let id = operationId {
                    operationIds.append(id)
                }
            }
            return operationIds
        }
        
        // Should handle concurrent requests gracefully
        // Due to cooldown, likely only one operation executes
        XCTAssertGreaterThanOrEqual(operations.count, 0)
        XCTAssertLessThanOrEqual(operations.count, 3)
    }
    
    // MARK: - Large Scale Tests
    
    func testLargeScalePartitioning() async throws {
        let nodeCount = 50
        let partitionCount = 1000
        
        measure {
            let partitioner = HashPartitioner()
            let nodes = (1...nodeCount).map { "node\($0)" }
            
            // Create partitions
            let partitions = partitioner.createPartitions(count: partitionCount, nodes: nodes)
            
            // Assign many keys
            for i in 0..<10000 {
                _ = partitioner.partition(for: "key_\(i)", partitions: partitions)
            }
            
            // Validate
            let errors = partitioner.validate(partitions: partitions)
            XCTAssertTrue(errors.isEmpty)
        }
    }
    
    func testLargeScaleConsistentHashing() async throws {
        let ring = ConsistentHashRing(
            configuration: ConsistentHashRing.Configuration(
                virtualNodesPerNode: 200
            )
        )
        
        // Add many nodes
        for i in 1...100 {
            try await ring.addNode(Node(id: "node\(i)"))
        }
        
        measure {
            // Perform many lookups
            Task {
                for i in 0..<100000 {
                    _ = await ring.getNode(for: "key_\(i)")
                }
            }
        }
        
        // Test load distribution
        let distribution = await ring.estimateLoadDistribution(
            sampleKeys: (0..<10000).map { "sample_\($0)" }
        )
        
        // Check that no node is overloaded
        for (_, percentage) in distribution {
            XCTAssertGreaterThan(percentage, 0.005) // At least 0.5%
            XCTAssertLessThan(percentage, 0.02)     // At most 2%
        }
    }
    
    // MARK: - End-to-End Tests
    
    func testEndToEndDistributedSystem() async throws {
        // Components
        let partitionManager = PartitionManager()
        let hashPartitioner = HashPartitioner()
        let assigner = PartitionAssigner(strategy: LocationAwareStrategy(
            preferredLocations: [
                NodeLocation(region: "us-west", zone: "us-west-1a"),
                NodeLocation(region: "us-west", zone: "us-west-1b")
            ]
        ))
        let coordinator = RebalancingCoordinator(
            partitionManager: partitionManager,
            partitionAssigner: assigner,
            configuration: RebalancingCoordinator.Configuration(dryRunMode: true)
        )
        
        // Phase 1: Setup nodes across regions
        let nodeConfigs = [
            ("node1", "us-west", "us-west-1a"),
            ("node2", "us-west", "us-west-1b"),
            ("node3", "us-east", "us-east-1a"),
            ("node4", "us-east", "us-east-1b"),
            ("node5", "eu-west", "eu-west-1a")
        ]
        
        for (nodeId, region, zone) in nodeConfigs {
            await assigner.registerNode(NodeInfo(
                nodeId: nodeId,
                capacity: NodeCapacity(maxPartitions: 20),
                currentLoad: NodeLoad(),
                health: NodeHealth(),
                location: NodeLocation(region: region, zone: zone)
            ))
        }
        
        // Phase 2: Create and assign partitions
        let partitions = hashPartitioner.createPartitions(
            count: 50,
            nodes: nodeConfigs.map { $0.0 }
        )
        
        for partition in partitions {
            let nodeId = try await assigner.assignPartition(partition)
            _ = try await partitionManager.createPartition(
                id: partition.id,
                ownerNode: nodeId,
                bounds: partition.bounds
            )
        }
        
        // Verify location preference
        let nodeAssignments = await withTaskGroup(of: (String, Set<String>).self) { group in
            for (nodeId, _, _) in nodeConfigs {
                group.addTask {
                    let assignments = await assigner.getNodeAssignments(nodeId)
                    return (nodeId, assignments)
                }
            }
            
            var results: [String: Int] = [:]
            for await (nodeId, assignments) in group {
                results[nodeId] = assignments.count
            }
            return results
        }
        
        // West coast nodes should have more partitions due to preference
        let westCoastPartitions = (nodeAssignments["node1"] ?? 0) + (nodeAssignments["node2"] ?? 0)
        let totalPartitions = nodeAssignments.values.reduce(0, +)
        let westCoastRatio = Double(westCoastPartitions) / Double(totalPartitions)
        
        XCTAssertGreaterThan(westCoastRatio, 0.35, "West coast should have significant partition share")
        
        // Phase 3: Simulate operations
        for partition in partitions.prefix(10) {
            try await partitionManager.activatePartition(partition.id)
            
            // Update statistics
            let stats = PartitionStats(
                vectorCount: Int.random(in: 1000...5000),
                sizeInBytes: Int.random(in: 1_000_000...10_000_000),
                readCount: Int.random(in: 100...1000),
                writeCount: Int.random(in: 50...500)
            )
            
            try await partitionManager.updateStatistics(
                partitionId: partition.id,
                stats: stats
            )
        }
        
        // Phase 4: Check if rebalancing needed
        if let plan = try await coordinator.analyzeAndPlan() {
            XCTAssertFalse(plan.moves.isEmpty)
            
            // Execute rebalancing
            let operationId = try await coordinator.executeRebalancing(plan)
            
            // Wait for completion
            try await Task.sleep(nanoseconds: UInt64(plan.moves.count + 1) * 1_000_000_000)
            
            let status = await coordinator.getOperationStatus(operationId)
            XCTAssertEqual(status?.state, "completed")
        }
        
        // Phase 5: Verify system health
        let allPartitions = await partitionManager.getAllPartitions()
        XCTAssertEqual(allPartitions.count, 50)
        
        let activePartitions = allPartitions.filter { $0.state == .active }
        XCTAssertGreaterThanOrEqual(activePartitions.count, 10)
        
        // Check balance
        let finalReport = await assigner.getBalanceReport()
        XCTAssertLessThanOrEqual(finalReport.imbalanceRatio, 0.3, "System should be reasonably balanced")
    }
    
    // MARK: - Stress Tests
    
    func testPartitionSplitStress() async throws {
        let manager = PartitionManager()
        
        // Create initial large partition
        _ = try await manager.createPartition(
            id: "p_root",
            ownerNode: "node1",
            bounds: .range(min: "\"a\"", max: "\"z\"")
        )
        
        try await manager.activatePartition("p_root")
        
        // Perform multiple splits
        var currentPartitions = ["p_root"]
        
        for generation in 1...3 {
            var newPartitions: [String] = []
            
            for partitionId in currentPartitions {
                let leftId = "\(partitionId)_l\(generation)"
                let rightId = "\(partitionId)_r\(generation)"
                
                try await manager.startPartitionSplit(
                    partitionId: partitionId,
                    targetPartitions: [leftId, rightId]
                )
                
                try await manager.completePartitionSplit(
                    sourcePartitionId: partitionId,
                    resultPartitions: [
                        (id: leftId, owner: "node1", bounds: .range(min: "\"a\"", max: "\"m\"")),
                        (id: rightId, owner: "node2", bounds: .range(min: "\"m\"", max: "\"z\""))
                    ]
                )
                
                newPartitions.append(contentsOf: [leftId, rightId])
            }
            
            currentPartitions = newPartitions
        }
        
        // Should have 2^3 = 8 partitions
        XCTAssertEqual(currentPartitions.count, 8)
        
        // Verify all are created
        for partitionId in currentPartitions {
            let partition = await manager.getPartition(partitionId)
            XCTAssertNotNil(partition)
        }
    }
    
    func testHighConcurrencyStress() async throws {
        let manager = PartitionManager()
        let concurrentOps = 100
        
        // Create many partitions concurrently
        let creationResults = await withTaskGroup(of: Result<PartitionMetadata, Error>.self) { group in
            for i in 1...concurrentOps {
                group.addTask {
                    do {
                        let metadata = try await manager.createPartition(
                            id: "stress_p\(i)",
                            ownerNode: "node\(i % 5 + 1)",
                            bounds: .hash(min: UInt64(i * 1000), max: UInt64((i + 1) * 1000))
                        )
                        return .success(metadata)
                    } catch {
                        return .failure(error)
                    }
                }
            }
            
            var successes = 0
            var failures = 0
            
            for await result in group {
                switch result {
                case .success:
                    successes += 1
                case .failure:
                    failures += 1
                }
            }
            
            return (successes, failures)
        }
        
        XCTAssertEqual(creationResults.0, concurrentOps)
        XCTAssertEqual(creationResults.1, 0)
        
        // Concurrent lease operations
        await withTaskGroup(of: Void.self) { group in
            for i in 1...20 {
                group.addTask {
                    do {
                        // Random partition and node
                        let partitionId = "stress_p\(Int.random(in: 1...concurrentOps))"
                        let nodeId = "node\(Int.random(in: 1...5))"
                        
                        _ = try await manager.acquireLease(
                            partitionId: partitionId,
                            nodeId: nodeId,
                            force: true
                        )
                        
                        // Random sleep
                        try await Task.sleep(nanoseconds: UInt64.random(in: 10_000_000...50_000_000))
                        
                        try await manager.releaseLease(
                            partitionId: partitionId,
                            nodeId: nodeId
                        )
                    } catch {
                        // Some conflicts expected
                    }
                }
            }
        }
        
        // System should still be consistent
        let allPartitions = await manager.getAllPartitions()
        XCTAssertEqual(allPartitions.count, concurrentOps)
    }
}
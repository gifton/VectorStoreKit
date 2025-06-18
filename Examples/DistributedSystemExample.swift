// VectorStoreKit: Distributed System Example
//
// Demonstrates how to use the distributed partitioning system

import Foundation
import VectorStoreKit

@main
struct DistributedSystemExample {
    
    static func main() async throws {
        print("🚀 VectorStoreKit Distributed System Example")
        print("=" * 50)
        
        // Initialize components
        let partitionManager = PartitionManager(
            configuration: PartitionManager.Configuration(
                leaseDuration: 300,
                leaseRenewalInterval: 60,
                healthCheckInterval: 30,
                autoRenewLeases: true,
                enableHealthMonitoring: true
            )
        )
        
        // Example 1: Basic Partitioning
        print("\n📊 Example 1: Basic Hash Partitioning")
        try await demonstrateHashPartitioning()
        
        // Example 2: Consistent Hash Ring
        print("\n🔄 Example 2: Consistent Hash Ring")
        try await demonstrateConsistentHashing()
        
        // Example 3: Range-Based Partitioning
        print("\n📏 Example 3: Range-Based Partitioning")
        try await demonstrateRangePartitioning()
        
        // Example 4: Custom Partitioning
        print("\n🎯 Example 4: Custom Geographic Partitioning")
        try await demonstrateCustomPartitioning()
        
        // Example 5: Automatic Assignment
        print("\n🤖 Example 5: Automatic Partition Assignment")
        try await demonstrateAutomaticAssignment(partitionManager: partitionManager)
        
        // Example 6: Rebalancing
        print("\n⚖️ Example 6: System Rebalancing")
        try await demonstrateRebalancing(partitionManager: partitionManager)
        
        // Example 7: Full System Integration
        print("\n🌐 Example 7: Full Distributed System")
        try await demonstrateFullSystem()
        
        print("\n✅ Examples completed!")
    }
    
    // MARK: - Example 1: Hash Partitioning
    
    static func demonstrateHashPartitioning() async throws {
        let partitioner = HashPartitioner()
        let nodes = ["node1", "node2", "node3", "node4"]
        
        // Create partitions
        let partitions = partitioner.createPartitions(count: 16, nodes: nodes)
        
        print("Created \(partitions.count) partitions across \(nodes.count) nodes")
        
        // Show partition distribution
        let distribution = Dictionary(grouping: partitions, by: { $0.nodeId })
        for (node, parts) in distribution {
            print("  \(node): \(parts.count) partitions")
        }
        
        // Demonstrate key routing
        let testKeys = ["user:123", "product:456", "order:789"]
        print("\nKey routing:")
        for key in testKeys {
            if let partition = partitioner.partition(for: key, partitions: partitions) {
                print("  '\(key)' -> partition \(partition.id) on \(partition.nodeId)")
            }
        }
        
        // Validate partitions
        let errors = partitioner.validate(partitions: partitions)
        print("\nValidation: \(errors.isEmpty ? "✅ Passed" : "❌ Failed with \(errors.count) errors")")
    }
    
    // MARK: - Example 2: Consistent Hashing
    
    static func demonstrateConsistentHashing() async throws {
        let ring = ConsistentHashRing(
            configuration: ConsistentHashRing.Configuration(
                virtualNodesPerNode: 150,
                replicationFactor: 3
            )
        )
        
        // Add nodes with different weights
        let nodes = [
            Node(id: "small-node", weight: 0.5),
            Node(id: "medium-node", weight: 1.0),
            Node(id: "large-node", weight: 2.0)
        ]
        
        for node in nodes {
            try await ring.addNode(node)
        }
        
        print("Ring created with \(await ring.nodeCount) nodes and \(await ring.ringSize) virtual nodes")
        
        // Show distribution
        let distribution = await ring.getNodeDistribution()
        for (nodeId, vnodeCount) in distribution {
            print("  \(nodeId): \(vnodeCount) virtual nodes")
        }
        
        // Test key assignment with replication
        print("\nKey assignment with replication:")
        let testKey = "important-data"
        let replicas = await ring.getNodes(for: testKey)
        print("  '\(testKey)' replicated to:")
        for (i, node) in replicas.enumerated() {
            print("    Replica \(i + 1): \(node.id)")
        }
        
        // Demonstrate node addition impact
        print("\nAdding new node...")
        try await ring.addNode(Node(id: "new-node"))
        
        // Check how many keys would move
        let sampleKeys = (0..<1000).map { "key_\($0)" }
        var moved = 0
        for key in sampleKeys {
            let oldNode = replicas.first
            let newNode = await ring.getNode(for: key)
            if oldNode?.id != newNode?.id {
                moved += 1
            }
        }
        
        print("  \(moved) of \(sampleKeys.count) keys would move (\(Int(Double(moved) / Double(sampleKeys.count) * 100))%)")
    }
    
    // MARK: - Example 3: Range Partitioning
    
    static func demonstrateRangePartitioning() async throws {
        let partitioner = RangePartitioner<String>(
            configuration: RangePartitioner.Configuration(
                maxKeysPerPartition: 1000,
                autoSplit: true,
                autoMerge: true,
                splitThreshold: 0.8,
                mergeThreshold: 0.2
            )
        )
        
        let nodes = ["node1", "node2", "node3"]
        
        // Create alphabetical range partitions
        let partitions = [
            Partition(id: "p1", nodeId: "node1", bounds: .range(min: "\"a\"", max: "\"f\"")),
            Partition(id: "p2", nodeId: "node2", bounds: .range(min: "\"f\"", max: "\"n\"")),
            Partition(id: "p3", nodeId: "node3", bounds: .range(min: "\"n\"", max: "\"t\"")),
            Partition(id: "p4", nodeId: "node1", bounds: .range(min: "\"t\"", max: "\"z\""))
        ]
        
        print("Range partitions:")
        for partition in partitions {
            if case .range(let min, let max) = partition.bounds {
                print("  \(partition.id): [\(min) - \(max)] on \(partition.nodeId)")
            }
        }
        
        // Test key routing
        let testKeys = ["apple", "banana", "orange", "watermelon"]
        print("\nKey routing:")
        for key in testKeys {
            if let partition = partitioner.partition(for: key, partitions: partitions) {
                print("  '\(key)' -> partition \(partition.id)")
            }
        }
        
        // Test range query
        print("\nRange query from 'd' to 'p':")
        let rangePartitions = partitioner.partitionsForRangeQuery(
            from: "d",
            to: "p",
            partitions: partitions
        )
        print("  Spans \(rangePartitions.count) partitions: \(rangePartitions.map { $0.id }.joined(separator: ", "))")
        
        // Demonstrate split suggestion
        let stats = PartitionStatistics(
            vectorCounts: [
                "p1": 900,  // Near capacity
                "p2": 400,
                "p3": 300,
                "p4": 100
            ]
        )
        
        if let plan = partitioner.suggestRebalancing(partitions: partitions, stats: stats) {
            print("\nRebalancing suggestions:")
            for split in plan.splits {
                print("  Split partition \(split.partitionId): \(split.reason)")
            }
        }
    }
    
    // MARK: - Example 4: Custom Partitioning
    
    static func demonstrateCustomPartitioning() async throws {
        // Geographic partitioner
        let geoPartitioner = CustomPartitioner<String>.geographic()
        
        // Create region-based partitions
        let partitions = [
            Partition(id: "p-us-west", nodeId: "us-west-node", 
                     bounds: .custom(["type": "geo", "region": "us-west-1"])),
            Partition(id: "p-us-east", nodeId: "us-east-node", 
                     bounds: .custom(["type": "geo", "region": "us-east-1"])),
            Partition(id: "p-eu", nodeId: "eu-node", 
                     bounds: .custom(["type": "geo", "region": "eu-west-1"])),
            Partition(id: "p-asia", nodeId: "asia-node", 
                     bounds: .custom(["type": "geo", "region": "ap-southeast-1"]))
        ]
        
        print("Geographic partitions:")
        for partition in partitions {
            if case .custom(let metadata) = partition.bounds {
                print("  \(partition.id): region=\(metadata["region"] ?? "unknown") on \(partition.nodeId)")
            }
        }
        
        // Test geographic routing
        let testData = [
            "us-west-1:user:123",
            "us-east-1:order:456",
            "eu-west-1:product:789",
            "ap-southeast-1:session:012"
        ]
        
        print("\nGeographic routing:")
        for key in testData {
            if let partition = geoPartitioner.partition(for: key, partitions: partitions) {
                print("  '\(key)' -> \(partition.id)")
            }
        }
        
        // Time-based partitioner example
        print("\n\nTime-based partitioning:")
        let timePartitioner = CustomPartitioner<Date>.timeBased(hoursPerPartition: 6)
        
        let timePartitions = [
            Partition(id: "morning", nodeId: "node1", 
                     bounds: .custom(["type": "time", "startHour": "0", "endHour": "6"])),
            Partition(id: "day", nodeId: "node2", 
                     bounds: .custom(["type": "time", "startHour": "6", "endHour": "18"])),
            Partition(id: "evening", nodeId: "node3", 
                     bounds: .custom(["type": "time", "startHour": "18", "endHour": "24"]))
        ]
        
        let calendar = Calendar.current
        let now = Date()
        
        for hour in [3, 12, 21] {
            var components = calendar.dateComponents([.year, .month, .day], from: now)
            components.hour = hour
            
            if let date = calendar.date(from: components),
               let partition = timePartitioner.partition(for: date, partitions: timePartitions) {
                print("  \(hour):00 -> \(partition.id)")
            }
        }
    }
    
    // MARK: - Example 5: Automatic Assignment
    
    static func demonstrateAutomaticAssignment(partitionManager: PartitionManager) async throws {
        // Create assigner with least-loaded strategy
        let assigner = PartitionAssigner(strategy: LeastLoadedStrategy())
        
        // Register nodes with different capacities
        let nodes = [
            NodeInfo(
                nodeId: "high-capacity",
                capacity: NodeCapacity(maxPartitions: 50),
                currentLoad: NodeLoad(partitionCount: 10),
                health: NodeHealth()
            ),
            NodeInfo(
                nodeId: "medium-capacity",
                capacity: NodeCapacity(maxPartitions: 30),
                currentLoad: NodeLoad(partitionCount: 15),
                health: NodeHealth()
            ),
            NodeInfo(
                nodeId: "low-capacity",
                capacity: NodeCapacity(maxPartitions: 20),
                currentLoad: NodeLoad(partitionCount: 18),
                health: NodeHealth()
            )
        ]
        
        for node in nodes {
            await assigner.registerNode(node)
        }
        
        print("Registered nodes:")
        for node in nodes {
            print("  \(node.nodeId): \(node.currentLoad.partitionCount)/\(node.capacity.maxPartitions) partitions (%.0f%% full)" 
                  .replacingOccurrences(of: "%.0f%%", with: "\(Int(100 - node.availableCapacityRatio * 100))%"))
        }
        
        // Assign new partitions
        print("\nAssigning 5 new partitions:")
        for i in 1...5 {
            let partition = Partition(
                id: "new_p\(i)",
                nodeId: "",
                bounds: .hash(min: UInt64(i * 1000), max: UInt64((i + 1) * 1000))
            )
            
            let assignedNode = try await assigner.assignPartition(partition)
            print("  \(partition.id) -> \(assignedNode)")
            
            // Update load for demonstration
            if let node = nodes.first(where: { $0.nodeId == assignedNode }) {
                await assigner.updateNodeLoad(
                    assignedNode,
                    load: NodeLoad(partitionCount: node.currentLoad.partitionCount + 1)
                )
            }
        }
        
        // Show balance report
        let report = await assigner.getBalanceReport()
        print("\nBalance report:")
        print("  Total partitions: \(report.totalPartitions)")
        print("  Average per node: \(String(format: "%.1f", report.avgPartitionsPerNode))")
        print("  Standard deviation: \(String(format: "%.2f", report.standardDeviation))")
        print("  Imbalance ratio: \(String(format: "%.2f", report.imbalanceRatio))")
    }
    
    // MARK: - Example 6: Rebalancing
    
    static func demonstrateRebalancing(partitionManager: PartitionManager) async throws {
        let assigner = PartitionAssigner(strategy: LeastLoadedStrategy())
        let coordinator = RebalancingCoordinator(
            partitionManager: partitionManager,
            partitionAssigner: assigner,
            configuration: RebalancingCoordinator.Configuration(
                maxConcurrentMigrations: 3,
                minImprovementThreshold: 0.1,
                dryRunMode: true // Dry run for demo
            )
        )
        
        // Create imbalanced scenario
        print("Creating imbalanced system:")
        
        // Register nodes
        for i in 1...3 {
            await assigner.registerNode(NodeInfo(
                nodeId: "node\(i)",
                capacity: NodeCapacity(maxPartitions: 20),
                currentLoad: NodeLoad(),
                health: NodeHealth()
            ))
        }
        
        // Create partitions with poor distribution
        let distribution = [
            ("node1", 12),  // Overloaded
            ("node2", 3),   // Underloaded
            ("node3", 5)    // Normal
        ]
        
        var partitionCount = 0
        for (nodeId, count) in distribution {
            print("  \(nodeId): \(count) partitions")
            
            for _ in 0..<count {
                partitionCount += 1
                _ = try await partitionManager.createPartition(
                    id: "p\(partitionCount)",
                    ownerNode: nodeId,
                    bounds: .hash(min: UInt64(partitionCount * 1000), max: UInt64((partitionCount + 1) * 1000))
                )
            }
        }
        
        // Analyze and plan rebalancing
        print("\nAnalyzing system balance...")
        
        if let plan = try await coordinator.analyzeAndPlan() {
            print("Rebalancing plan generated:")
            print("  Moves to execute: \(plan.moves.count)")
            print("  Estimated improvement: \(Int(plan.estimatedImprovement * 100))%")
            
            for move in plan.moves {
                print("  - Move \(move.partitionId) from \(move.fromNode) to \(move.toNode)")
            }
            
            // Execute rebalancing
            print("\nExecuting rebalancing (dry run)...")
            let operationId = try await coordinator.executeRebalancing(plan)
            
            // Monitor progress
            for _ in 0..<3 {
                try await Task.sleep(nanoseconds: 1_000_000_000)
                
                if let status = await coordinator.getOperationStatus(operationId) {
                    print("  Progress: \(Int(status.progress * 100))% (\(status.completedMoves)/\(status.totalMoves) moves)")
                }
            }
            
            print("\n✅ Rebalancing completed!")
        } else {
            print("System is already balanced!")
        }
    }
    
    // MARK: - Example 7: Full System
    
    static func demonstrateFullSystem() async throws {
        print("Setting up distributed vector store system...")
        
        // Initialize all components
        let partitionManager = PartitionManager()
        let consistentHash = ConsistentHashRing()
        let assigner = PartitionAssigner(
            strategy: LocationAwareStrategy(
                preferredLocations: [
                    NodeLocation(region: "us-west", zone: "us-west-1a")
                ]
            )
        )
        
        // Setup nodes in different regions
        let nodeConfigs: [(String, String, String)] = [
            ("us-west-1a", "us-west", "us-west-1a"),
            ("us-west-1b", "us-west", "us-west-1b"),
            ("us-east-1a", "us-east", "us-east-1a"),
            ("eu-west-1a", "eu-west", "eu-west-1a")
        ]
        
        print("\nRegistering nodes:")
        for (nodeId, region, zone) in nodeConfigs {
            let nodeInfo = NodeInfo(
                nodeId: nodeId,
                capacity: NodeCapacity(maxPartitions: 25),
                currentLoad: NodeLoad(),
                health: NodeHealth(),
                location: NodeLocation(region: region, zone: zone)
            )
            
            await assigner.registerNode(nodeInfo)
            try await consistentHash.addNode(Node(id: nodeId))
            
            print("  ✅ \(nodeId) (region: \(region))")
        }
        
        // Create and assign partitions
        print("\nCreating partitions...")
        let partitioner = HashPartitioner()
        let partitions = partitioner.createPartitions(
            count: 20,
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
        
        // Show distribution
        print("\nPartition distribution:")
        for (nodeId, _, _) in nodeConfigs {
            let assignments = await assigner.getNodeAssignments(nodeId)
            print("  \(nodeId): \(assignments.count) partitions")
        }
        
        // Demonstrate key operations
        print("\nKey operations:")
        
        // Insert vectors
        let vectors = [
            ("vec1", "user profile embedding"),
            ("vec2", "product description"),
            ("vec3", "image feature vector")
        ]
        
        for (id, description) in vectors {
            if let node = await consistentHash.getNode(for: id) {
                print("  Store '\(id)' (\(description)) -> \(node.id)")
            }
        }
        
        // Query with replication
        print("\nQuery with replication:")
        let queryKey = "important-query"
        let replicas = await consistentHash.getNodes(for: queryKey)
        print("  Query '\(queryKey)' can be served by:")
        for (i, node) in replicas.enumerated() {
            print("    Replica \(i + 1): \(node.id)")
        }
        
        // Health monitoring
        print("\nHealth monitoring:")
        for partition in partitions.prefix(5) {
            let health = HealthCheck(
                name: "storage",
                status: .healthy,
                message: "Storage OK"
            )
            
            try await partitionManager.updateHealth(
                partitionId: partition.id,
                healthCheck: health
            )
            
            if let metadata = await partitionManager.getPartition(partition.id) {
                print("  \(partition.id): \(metadata.health.status)")
            }
        }
        
        print("\n🎉 Distributed system fully operational!")
    }
}

// Helper extension
extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
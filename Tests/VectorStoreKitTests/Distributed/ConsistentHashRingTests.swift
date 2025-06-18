// VectorStoreKit: Consistent Hash Ring Tests
//
// Comprehensive tests for consistent hash ring implementation

import XCTest
@testable import VectorStoreKit

final class ConsistentHashRingTests: XCTestCase {
    
    // MARK: - Basic Operations
    
    func testRingCreation() async {
        let ring = ConsistentHashRing()
        
        let nodeCount = await ring.nodeCount
        let ringSize = await ring.ringSize
        
        XCTAssertEqual(nodeCount, 0)
        XCTAssertEqual(ringSize, 0)
    }
    
    func testAddNode() async throws {
        let ring = ConsistentHashRing()
        
        let node = Node(id: "node1", weight: 1.0)
        try await ring.addNode(node)
        
        let nodeCount = await ring.nodeCount
        let ringSize = await ring.ringSize
        
        XCTAssertEqual(nodeCount, 1)
        XCTAssertEqual(ringSize, 150) // Default virtual nodes per node
        
        // Check that we can retrieve the node
        let retrievedNode = await ring.getNode(for: "test-key")
        XCTAssertNotNil(retrievedNode)
        XCTAssertEqual(retrievedNode?.id, "node1")
    }
    
    func testAddMultipleNodes() async throws {
        let ring = ConsistentHashRing()
        
        let nodes = [
            Node(id: "node1"),
            Node(id: "node2"),
            Node(id: "node3")
        ]
        
        for node in nodes {
            try await ring.addNode(node)
        }
        
        let nodeCount = await ring.nodeCount
        let ringSize = await ring.ringSize
        
        XCTAssertEqual(nodeCount, 3)
        XCTAssertEqual(ringSize, 450) // 3 nodes * 150 virtual nodes
        
        // Check distribution
        let distribution = await ring.getNodeDistribution()
        XCTAssertEqual(distribution.count, 3)
        
        for (_, count) in distribution {
            XCTAssertEqual(count, 150)
        }
    }
    
    func testRemoveNode() async throws {
        let ring = ConsistentHashRing()
        
        // Add nodes
        try await ring.addNode(Node(id: "node1"))
        try await ring.addNode(Node(id: "node2"))
        
        // Remove one
        try await ring.removeNode(nodeId: "node1")
        
        let nodeCount = await ring.nodeCount
        let ringSize = await ring.ringSize
        
        XCTAssertEqual(nodeCount, 1)
        XCTAssertEqual(ringSize, 150)
        
        // Verify node1 is gone
        let distribution = await ring.getNodeDistribution()
        XCTAssertNil(distribution["node1"])
        XCTAssertEqual(distribution["node2"], 150)
    }
    
    // MARK: - Key Distribution
    
    func testKeyDistribution() async throws {
        let ring = ConsistentHashRing()
        
        // Add nodes
        let nodes = ["node1", "node2", "node3", "node4"]
        for nodeId in nodes {
            try await ring.addNode(Node(id: nodeId))
        }
        
        // Test key distribution
        var distribution: [String: Int] = [:]
        
        for i in 0..<10000 {
            let key = "key_\(i)"
            if let node = await ring.getNode(for: key) {
                distribution[node.id, default: 0] += 1
            }
        }
        
        // Each node should get roughly 25% of keys (±5%)
        let expectedPerNode = 2500
        let tolerance = 500
        
        for nodeId in nodes {
            let count = distribution[nodeId] ?? 0
            XCTAssertGreaterThan(count, expectedPerNode - tolerance)
            XCTAssertLessThan(count, expectedPerNode + tolerance)
        }
    }
    
    func testConsistentMapping() async throws {
        let ring = ConsistentHashRing()
        
        // Add nodes
        try await ring.addNode(Node(id: "node1"))
        try await ring.addNode(Node(id: "node2"))
        
        // Test that same key always maps to same node
        let testKeys = ["user123", "product456", "order789"]
        var mappings: [String: String] = [:]
        
        for key in testKeys {
            if let node = await ring.getNode(for: key) {
                mappings[key] = node.id
            }
        }
        
        // Query again and verify same mapping
        for key in testKeys {
            if let node = await ring.getNode(for: key) {
                XCTAssertEqual(node.id, mappings[key])
            }
        }
    }
    
    // MARK: - Virtual Nodes and Weights
    
    func testWeightedNodes() async throws {
        let config = ConsistentHashRing.Configuration(virtualNodesPerNode: 100)
        let ring = ConsistentHashRing(configuration: config)
        
        // Add nodes with different weights
        try await ring.addNode(Node(id: "small", weight: 0.5))
        try await ring.addNode(Node(id: "medium", weight: 1.0))
        try await ring.addNode(Node(id: "large", weight: 2.0))
        
        let distribution = await ring.getNodeDistribution()
        
        // Verify virtual nodes are proportional to weights
        XCTAssertEqual(distribution["small"], 50)   // 0.5 * 100
        XCTAssertEqual(distribution["medium"], 100) // 1.0 * 100
        XCTAssertEqual(distribution["large"], 200)  // 2.0 * 100
    }
    
    func testCustomVirtualNodeCount() async throws {
        let config = ConsistentHashRing.Configuration(virtualNodesPerNode: 50)
        let ring = ConsistentHashRing(configuration: config)
        
        try await ring.addNode(Node(id: "node1"))
        
        let ringSize = await ring.ringSize
        XCTAssertEqual(ringSize, 50)
    }
    
    // MARK: - Replication
    
    func testReplicationNodes() async throws {
        let config = ConsistentHashRing.Configuration(replicationFactor: 3)
        let ring = ConsistentHashRing(configuration: config)
        
        // Add enough nodes for replication
        for i in 1...5 {
            try await ring.addNode(Node(id: "node\(i)"))
        }
        
        // Get replica nodes
        let replicas = await ring.getNodes(for: "test-key")
        
        XCTAssertEqual(replicas.count, 3)
        
        // Verify all replicas are unique
        let uniqueIds = Set(replicas.map { $0.id })
        XCTAssertEqual(uniqueIds.count, 3)
    }
    
    func testReplicationWithInsufficientNodes() async throws {
        let config = ConsistentHashRing.Configuration(replicationFactor: 3)
        let ring = ConsistentHashRing(configuration: config)
        
        // Add only 2 nodes
        try await ring.addNode(Node(id: "node1"))
        try await ring.addNode(Node(id: "node2"))
        
        // Should return all available nodes
        let replicas = await ring.getNodes(for: "test-key")
        
        XCTAssertEqual(replicas.count, 2)
    }
    
    // MARK: - Node States
    
    func testNodeStates() async throws {
        let ring = ConsistentHashRing()
        
        // Add active node
        let activeNode = Node(id: "active", state: .active)
        try await ring.addNode(activeNode)
        
        // Try to add inactive node (should fail)
        let inactiveNode = Node(id: "inactive", state: .inactive)
        
        do {
            try await ring.addNode(inactiveNode)
            XCTFail("Should not allow adding inactive node")
        } catch ConsistentHashError.invalidNodeState(_, _) {
            // Expected
        }
        
        // Update node state
        try await ring.updateNodeState(nodeId: "active", state: .leaving)
        
        // Verify state was updated
        let nodes = await ring.getAllNodes()
        XCTAssertEqual(nodes.first?.state, .leaving)
    }
    
    // MARK: - Load Distribution
    
    func testLoadDistributionEstimation() async throws {
        let ring = ConsistentHashRing()
        
        // Add nodes
        for i in 1...4 {
            try await ring.addNode(Node(id: "node\(i)"))
        }
        
        // Generate sample keys
        let sampleKeys = (0..<1000).map { "sample_key_\(i)" }
        
        let distribution = await ring.estimateLoadDistribution(sampleKeys: sampleKeys)
        
        // Each node should get roughly 25%
        for (_, percentage) in distribution {
            XCTAssertGreaterThan(percentage, 0.2)
            XCTAssertLessThan(percentage, 0.3)
        }
    }
    
    func testIsBalanced() async throws {
        let ring = ConsistentHashRing()
        
        // Add evenly weighted nodes
        for i in 1...4 {
            try await ring.addNode(Node(id: "node\(i)"))
        }
        
        let isBalanced = await ring.isBalanced()
        XCTAssertTrue(isBalanced)
        
        // Add a heavily weighted node
        try await ring.addNode(Node(id: "heavy", weight: 5.0))
        
        let isBalancedAfter = await ring.isBalanced(threshold: 0.1)
        XCTAssertFalse(isBalancedAfter) // Should be unbalanced now
    }
    
    // MARK: - Key Ranges
    
    func testKeyRanges() async throws {
        let ring = ConsistentHashRing()
        
        try await ring.addNode(Node(id: "node1"))
        
        let ranges = await ring.getKeyRanges(for: "node1")
        
        // Should have ranges equal to virtual nodes
        XCTAssertEqual(ranges.count, 150)
        
        // Ranges should not overlap
        let sortedRanges = ranges.sorted { $0.0 < $1.0 }
        for i in 0..<(sortedRanges.count - 1) {
            XCTAssertLessThan(sortedRanges[i].1, sortedRanges[i+1].0)
        }
    }
    
    // MARK: - Error Handling
    
    func testDuplicateNode() async throws {
        let ring = ConsistentHashRing()
        
        let node = Node(id: "duplicate")
        try await ring.addNode(node)
        
        // Try to add again
        do {
            try await ring.addNode(node)
            XCTFail("Should not allow duplicate nodes")
        } catch ConsistentHashError.nodeAlreadyExists(_) {
            // Expected
        }
    }
    
    func testRemoveNonexistentNode() async throws {
        let ring = ConsistentHashRing()
        
        do {
            try await ring.removeNode(nodeId: "nonexistent")
            XCTFail("Should fail to remove nonexistent node")
        } catch ConsistentHashError.nodeNotFound(_) {
            // Expected
        }
    }
    
    func testMinimumNodes() async throws {
        let config = ConsistentHashRing.Configuration(minimumNodes: 2)
        let ring = ConsistentHashRing(configuration: config)
        
        // Add 2 nodes
        try await ring.addNode(Node(id: "node1"))
        try await ring.addNode(Node(id: "node2"))
        
        // Remove one (should work)
        try await ring.removeNode(nodeId: "node1")
        
        // Try to remove the last one (should fail)
        do {
            try await ring.removeNode(nodeId: "node2")
            XCTFail("Should not allow going below minimum nodes")
        } catch ConsistentHashError.belowMinimumNodes(_, _) {
            // Expected
        }
    }
    
    // MARK: - Performance
    
    func testLookupPerformance() async throws {
        let ring = ConsistentHashRing()
        
        // Add many nodes
        for i in 1...20 {
            try await ring.addNode(Node(id: "node\(i)"))
        }
        
        // Measure lookup performance
        let startTime = Date()
        
        for i in 0..<100000 {
            _ = await ring.getNode(for: "perf_key_\(i)")
        }
        
        let elapsed = Date().timeIntervalSince(startTime)
        print("100k lookups took \(elapsed) seconds")
        
        // Should be fast
        XCTAssertLessThan(elapsed, 1.0) // Less than 1 second for 100k lookups
    }
    
    func testNodeStatistics() async throws {
        let ring = ConsistentHashRing()
        
        try await ring.addNode(Node(id: "node1"))
        
        // Perform some lookups
        for i in 0..<100 {
            _ = await ring.getNode(for: "key_\(i)")
        }
        
        let stats = await ring.getNodeStatistics(nodeId: "node1")
        XCTAssertNotNil(stats)
        XCTAssertEqual(stats?.keyLookups, 100)
    }
}

// MARK: - Consistent Hash Partitioner Tests

final class ConsistentHashPartitionerTests: XCTestCase {
    
    func testPartitionerIntegration() async throws {
        let partitioner = ConsistentHashPartitioner()
        
        // Add nodes
        let node1 = Node(id: "node1")
        let node2 = Node(id: "node2")
        
        try await partitioner.addNode(node1)
        try await partitioner.addNode(node2)
        
        // Get ring info
        let info = await partitioner.getRingInfo()
        XCTAssertEqual(info.nodeCount, 2)
        XCTAssertTrue(info.isBalanced)
        
        // Test key lookup
        let node = await partitioner.getNodeForKey("test-key")
        XCTAssertNotNil(node)
    }
    
    func testReplicaNodes() async throws {
        let config = ConsistentHashRing.Configuration(replicationFactor: 3)
        let partitioner = ConsistentHashPartitioner(configuration: config)
        
        // Add nodes
        for i in 1...5 {
            try await partitioner.addNode(Node(id: "node\(i)"))
        }
        
        // Get replicas
        let replicas = await partitioner.getReplicaNodes(for: "test-key")
        XCTAssertEqual(replicas.count, 3)
        
        // All should be unique
        let uniqueIds = Set(replicas.map { $0.id })
        XCTAssertEqual(uniqueIds.count, 3)
    }
}
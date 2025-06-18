// VectorStoreKit: Partition Strategy Tests
//
// Comprehensive tests for partitioning strategies

import XCTest
@testable import VectorStoreKit

final class PartitionStrategyTests: XCTestCase {
    
    // MARK: - Core Types Tests
    
    func testPartitionCreation() {
        let partition = Partition(
            id: "test-partition",
            nodeId: "node-1",
            bounds: .hash(min: 0, max: 1000),
            state: .active,
            metadata: PartitionMetadata(vectorCount: 100, sizeInBytes: 1024)
        )
        
        XCTAssertEqual(partition.id, "test-partition")
        XCTAssertEqual(partition.nodeId, "node-1")
        XCTAssertEqual(partition.state, .active)
        XCTAssertEqual(partition.metadata.vectorCount, 100)
        
        // Test bounds
        if case .hash(let min, let max) = partition.bounds {
            XCTAssertEqual(min, 0)
            XCTAssertEqual(max, 1000)
        } else {
            XCTFail("Expected hash bounds")
        }
    }
    
    func testPartitionStates() {
        let states: [PartitionState] = [.active, .migrating, .splitting, .merging, .readonly, .offline]
        
        for state in states {
            let partition = Partition(
                id: "test",
                nodeId: "node",
                bounds: .unbounded,
                state: state
            )
            XCTAssertEqual(partition.state, state)
        }
    }
    
    func testPartitionBoundsTypes() {
        // Hash bounds
        let hashBounds = PartitionBounds.hash(min: 100, max: 200)
        if case .hash(let min, let max) = hashBounds {
            XCTAssertEqual(min, 100)
            XCTAssertEqual(max, 200)
        } else {
            XCTFail("Expected hash bounds")
        }
        
        // Range bounds
        let rangeBounds = PartitionBounds.range(min: "a", max: "z")
        if case .range(let min, let max) = rangeBounds {
            XCTAssertEqual(min, "a")
            XCTAssertEqual(max, "z")
        } else {
            XCTFail("Expected range bounds")
        }
        
        // Custom bounds
        let customData = "custom".data(using: .utf8)!
        let customBounds = PartitionBounds.custom(customData)
        if case .custom(let data) = customBounds {
            XCTAssertEqual(data, customData)
        } else {
            XCTFail("Expected custom bounds")
        }
        
        // Unbounded
        let unbounded = PartitionBounds.unbounded
        if case .unbounded = unbounded {
            // Success
        } else {
            XCTFail("Expected unbounded")
        }
    }
    
    // MARK: - Hash Partitioner Tests
    
    func testHashPartitionerCreation() {
        let partitioner = HashPartitioner(hashFunction: .sha256, virtualNodes: 100)
        XCTAssertNotNil(partitioner)
    }
    
    func testHashPartitionerCreatePartitions() {
        let partitioner = HashPartitioner()
        let nodes = ["node-1", "node-2", "node-3"]
        let partitions = partitioner.createPartitions(count: 6, nodes: nodes)
        
        XCTAssertEqual(partitions.count, 6)
        
        // Check that all nodes are used
        let assignedNodes = Set(partitions.map { $0.nodeId })
        XCTAssertEqual(assignedNodes, Set(nodes))
        
        // Check that partitions cover the full hash space
        var coveredRanges: [(UInt64, UInt64)] = []
        for partition in partitions {
            if case .hash(let min, let max) = partition.bounds {
                coveredRanges.append((min, max))
            }
        }
        
        // Sort by min value
        coveredRanges.sort { $0.0 < $1.0 }
        
        // First partition should start at 0
        XCTAssertEqual(coveredRanges.first?.0, 0)
        
        // Last partition should end at UInt64.max
        XCTAssertEqual(coveredRanges.last?.1, UInt64.max)
        
        // Check for no gaps
        for i in 0..<(coveredRanges.count - 1) {
            let currentMax = coveredRanges[i].1
            let nextMin = coveredRanges[i + 1].0
            XCTAssertEqual(currentMax + 1, nextMin, "Gap found between partitions")
        }
    }
    
    func testHashPartitionerPartitionForKey() {
        let partitioner = HashPartitioner()
        let partitions = partitioner.createPartitions(count: 4, nodes: ["node-1"])
        
        // Test various keys
        let testKeys = ["user123", "product456", "order789", "test", ""]
        
        for key in testKeys {
            let partition = partitioner.partition(for: key, partitions: partitions)
            XCTAssertNotNil(partition, "Should find partition for key: \(key)")
            
            // Same key should always map to same partition
            let partition2 = partitioner.partition(for: key, partitions: partitions)
            XCTAssertEqual(partition?.id, partition2?.id)
        }
    }
    
    func testHashPartitionerDistribution() {
        let partitioner = HashPartitioner()
        let partitions = partitioner.createPartitions(count: 10, nodes: ["node-1", "node-2"])
        
        // Generate many keys and check distribution
        var distribution: [String: Int] = [:]
        
        for i in 0..<1000 {
            let key = "key_\(i)"
            if let partition = partitioner.partition(for: key, partitions: partitions) {
                distribution[partition.id, default: 0] += 1
            }
        }
        
        // Check that all partitions got some keys
        XCTAssertEqual(distribution.count, partitions.count)
        
        // Check for reasonable distribution (each partition should get 50-150 keys)
        for (_, count) in distribution {
            XCTAssertGreaterThan(count, 50)
            XCTAssertLessThan(count, 150)
        }
    }
    
    func testHashPartitionerValidation() {
        let partitioner = HashPartitioner()
        
        // Valid partitions
        let validPartitions = partitioner.createPartitions(count: 3, nodes: ["node-1"])
        let errors = partitioner.validate(partitions: validPartitions)
        XCTAssertTrue(errors.isEmpty, "Valid partitions should have no errors")
        
        // Partitions with gaps
        let gappedPartitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .hash(min: 0, max: 100)),
            Partition(id: "p2", nodeId: "n1", bounds: .hash(min: 200, max: 300)) // Gap from 101-199
        ]
        let gapErrors = partitioner.validate(partitions: gappedPartitions)
        XCTAssertFalse(gapErrors.isEmpty)
        XCTAssertTrue(gapErrors.contains { error in
            if case .gapInPartitions = error { return true }
            return false
        })
        
        // Overlapping partitions
        let overlappingPartitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .hash(min: 0, max: 200)),
            Partition(id: "p2", nodeId: "n1", bounds: .hash(min: 100, max: 300)) // Overlap 100-200
        ]
        let overlapErrors = partitioner.validate(partitions: overlappingPartitions)
        XCTAssertFalse(overlapErrors.isEmpty)
        XCTAssertTrue(overlapErrors.contains { error in
            if case .overlappingPartitions = error { return true }
            return false
        })
    }
    
    func testHashPartitionerRebalancing() {
        let partitioner = HashPartitioner()
        let partitions = partitioner.createPartitions(count: 4, nodes: ["node-1", "node-2"])
        
        // Create unbalanced statistics
        let stats = PartitionStatistics(
            vectorCounts: [
                "partition_0": 10000,  // Overloaded
                "partition_1": 1000,   // Underloaded
                "partition_2": 5000,   // Normal
                "partition_3": 5000    // Normal
            ],
            sizes: [:],
            loadMetrics: [
                "partition_0": 1000.0,  // High load
                "partition_1": 100.0,
                "partition_2": 500.0,
                "partition_3": 400.0
            ],
            nodeAssignments: [
                "node-1": ["partition_0", "partition_1"],
                "node-2": ["partition_2", "partition_3"]
            ]
        )
        
        let plan = partitioner.suggestRebalancing(partitions: partitions, stats: stats)
        XCTAssertNotNil(plan)
        
        // Should suggest splitting the hot partition
        XCTAssertFalse(plan?.splits.isEmpty ?? true)
        if let split = plan?.splits.first {
            XCTAssertEqual(split.partitionId, "partition_0")
            XCTAssertTrue(split.reason.contains("High load"))
        }
    }
    
    func testHashPartitionerEdgeCases() {
        let partitioner = HashPartitioner()
        
        // Empty partitions
        let emptyResult = partitioner.partition(for: "test", partitions: [])
        XCTAssertNil(emptyResult)
        
        // Single partition
        let singlePartition = [
            Partition(id: "single", nodeId: "node", bounds: .hash(min: 0, max: UInt64.max))
        ]
        let singleResult = partitioner.partition(for: "anything", partitions: singlePartition)
        XCTAssertEqual(singleResult?.id, "single")
        
        // Wraparound hash range
        let wraparoundPartitions = [
            Partition(id: "wrap", nodeId: "node", bounds: .hash(min: UInt64.max - 100, max: 100))
        ]
        
        // Test keys that should fall in wraparound range
        let wrapResult = partitioner.partition(for: "test", partitions: wraparoundPartitions)
        XCTAssertNotNil(wrapResult)
    }
    
    // MARK: - Base Strategy Tests
    
    func testBasePartitionStrategy() {
        let strategy = BasePartitionStrategy<String>(
            partitioner: { key, partitions in
                partitions.first { _ in key.hasPrefix("a") }
            },
            queryPartitioner: { _, partitions in
                partitions
            },
            layoutCreator: { count, nodes in
                (0..<count).map { i in
                    Partition(
                        id: "base_\(i)",
                        nodeId: nodes[i % nodes.count],
                        bounds: .unbounded
                    )
                }
            }
        )
        
        let partitions = strategy.createPartitions(count: 2, nodes: ["node1", "node2"])
        XCTAssertEqual(partitions.count, 2)
        
        let result = strategy.partition(for: "apple", partitions: partitions)
        XCTAssertNotNil(result)
    }
    
    // MARK: - Performance Tests
    
    func testHashPartitionerPerformance() {
        let partitioner = HashPartitioner()
        let partitions = partitioner.createPartitions(count: 100, nodes: ["n1", "n2", "n3"])
        
        measure {
            for i in 0..<10000 {
                _ = partitioner.partition(for: "key_\(i)", partitions: partitions)
            }
        }
    }
}

// MARK: - Test Helpers

extension PartitionStrategyTests {
    
    func assertPartitionsCoverFullRange(_ partitions: [Partition]) {
        var ranges: [(UInt64, UInt64)] = []
        
        for partition in partitions {
            if case .hash(let min, let max) = partition.bounds {
                ranges.append((min, max))
            }
        }
        
        ranges.sort { $0.0 < $1.0 }
        
        // Check start and end
        XCTAssertEqual(ranges.first?.0, 0, "First partition should start at 0")
        XCTAssertEqual(ranges.last?.1, UInt64.max, "Last partition should end at UInt64.max")
        
        // Check continuity
        for i in 0..<(ranges.count - 1) {
            XCTAssertEqual(
                ranges[i].1 + 1,
                ranges[i + 1].0,
                "Partitions should be continuous"
            )
        }
    }
}
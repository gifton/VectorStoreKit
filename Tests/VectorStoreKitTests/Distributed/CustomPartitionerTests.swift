// VectorStoreKit: Custom Partitioner Tests
//
// Comprehensive tests for custom partitioning strategies

import XCTest
@testable import VectorStoreKit

final class CustomPartitionerTests: XCTestCase {
    
    // MARK: - Basic Custom Partitioner Tests
    
    func testBasicCustomPartitioner() {
        // Create a simple even/odd partitioner
        let partitionFunction = PartitionFunction<Int>(
            apply: { key, partitions in
                let isEven = key % 2 == 0
                return partitions.first { partition in
                    guard case .custom(let metadata) = partition.bounds,
                          let type = metadata["type"] else {
                        return false
                    }
                    return type == (isEven ? "even" : "odd")
                }
            },
            applyQuery: { key, partitions in
                let isEven = key % 2 == 0
                return partitions.filter { partition in
                    guard case .custom(let metadata) = partition.bounds,
                          let type = metadata["type"] else {
                        return false
                    }
                    return type == (isEven ? "even" : "odd")
                }
            }
        )
        
        let config = CustomPartitioner<Int>.Configuration(
            name: "even_odd",
            partitionFunction: partitionFunction
        )
        
        let partitioner = CustomPartitioner(configuration: config)
        
        // Create test partitions
        let partitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .custom(["type": "even"])),
            Partition(id: "p2", nodeId: "n2", bounds: .custom(["type": "odd"]))
        ]
        
        // Test even numbers go to p1
        XCTAssertEqual(partitioner.partition(for: 2, partitions: partitions)?.id, "p1")
        XCTAssertEqual(partitioner.partition(for: 4, partitions: partitions)?.id, "p1")
        
        // Test odd numbers go to p2
        XCTAssertEqual(partitioner.partition(for: 1, partitions: partitions)?.id, "p2")
        XCTAssertEqual(partitioner.partition(for: 3, partitions: partitions)?.id, "p2")
    }
    
    func testCustomRebalancingStrategy() {
        // Create custom rebalancing that always suggests a specific move
        let rebalancingStrategy = RebalancingStrategy { partitions, stats in
            guard partitions.count >= 2 else { return nil }
            
            return RebalancingPlan(
                moves: [
                    PartitionMove(
                        partitionId: partitions[0].id,
                        fromNode: partitions[0].nodeId,
                        toNode: partitions[1].nodeId,
                        reason: "Custom rebalancing test"
                    )
                ],
                estimatedImprovement: 0.5
            )
        }
        
        let config = CustomPartitioner<String>.Configuration(
            name: "test",
            partitionFunction: PartitionFunction(
                apply: { _, partitions in partitions.first },
                applyQuery: { _, partitions in partitions }
            ),
            rebalancingStrategy: rebalancingStrategy
        )
        
        let partitioner = CustomPartitioner(configuration: config)
        
        let partitions = [
            Partition(id: "p1", nodeId: "n1"),
            Partition(id: "p2", nodeId: "n2")
        ]
        
        let plan = partitioner.suggestRebalancing(
            partitions: partitions,
            stats: PartitionStatistics()
        )
        
        XCTAssertNotNil(plan)
        XCTAssertEqual(plan?.moves.count, 1)
        XCTAssertEqual(plan?.moves.first?.partitionId, "p1")
        XCTAssertEqual(plan?.estimatedImprovement, 0.5)
    }
    
    func testCustomValidationRules() {
        // Create validation that checks for specific metadata
        let validationRules = ValidationRules { partitions in
            var errors: [PartitionError] = []
            
            for partition in partitions {
                guard case .custom(let metadata) = partition.bounds else {
                    errors.append(.invalidBounds("Missing custom bounds"))
                    continue
                }
                
                if metadata["required_field"] == nil {
                    errors.append(.invalidBounds("Missing required_field in partition \(partition.id)"))
                }
            }
            
            return errors
        }
        
        let config = CustomPartitioner<String>.Configuration(
            name: "validated",
            partitionFunction: PartitionFunction(
                apply: { _, partitions in partitions.first },
                applyQuery: { _, partitions in partitions }
            ),
            validationRules: validationRules
        )
        
        let partitioner = CustomPartitioner(configuration: config)
        
        let validPartitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .custom(["required_field": "value"]))
        ]
        
        let invalidPartitions = [
            Partition(id: "p2", nodeId: "n2", bounds: .custom(["other_field": "value"]))
        ]
        
        let validErrors = partitioner.validate(partitions: validPartitions)
        XCTAssertTrue(validErrors.isEmpty)
        
        let invalidErrors = partitioner.validate(partitions: invalidPartitions)
        XCTAssertFalse(invalidErrors.isEmpty)
        XCTAssertEqual(invalidErrors.count, 1)
    }
    
    // MARK: - Geographic Partitioner Tests
    
    func testGeographicPartitioner() {
        let partitioner = CustomPartitioner<String>.geographic()
        
        // Create region-based partitions
        let partitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .custom(["type": "geo", "region": "us-west-1"])),
            Partition(id: "p2", nodeId: "n2", bounds: .custom(["type": "geo", "region": "us-east-1"])),
            Partition(id: "p3", nodeId: "n3", bounds: .custom(["type": "geo", "region": "eu-west-1"]))
        ]
        
        // Test key routing based on region prefix
        XCTAssertEqual(
            partitioner.partition(for: "us-west-1:user123", partitions: partitions)?.id,
            "p1"
        )
        
        XCTAssertEqual(
            partitioner.partition(for: "us-east-1:order456", partitions: partitions)?.id,
            "p2"
        )
        
        XCTAssertEqual(
            partitioner.partition(for: "eu-west-1:product789", partitions: partitions)?.id,
            "p3"
        )
        
        // Test query routing
        let queryResults = partitioner.partitionsForQuery("us-west-1:query", partitions: partitions)
        XCTAssertEqual(queryResults.count, 1)
        XCTAssertEqual(queryResults.first?.id, "p1")
    }
    
    func testGeographicPartitionerRebalancing() {
        let partitioner = CustomPartitioner<String>.geographic()
        
        let partitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .custom(["type": "geo", "region": "us-west-1"])),
            Partition(id: "p2", nodeId: "n2", bounds: .custom(["type": "geo", "region": "us-east-1"]))
        ]
        
        let stats = PartitionStatistics(
            vectorCounts: ["p1": 1000, "p2": 100]
        )
        
        // Geographic partitioner shouldn't move data between regions
        let plan = partitioner.suggestRebalancing(partitions: partitions, stats: stats)
        XCTAssertNil(plan)
    }
    
    // MARK: - Time-Based Partitioner Tests
    
    func testTimeBasedPartitioner() {
        let partitioner = CustomPartitioner<Date>.timeBased(hoursPerPartition: 6)
        
        // Create hour-based partitions
        let partitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .custom(["type": "time", "startHour": "0", "endHour": "6"])),
            Partition(id: "p2", nodeId: "n1", bounds: .custom(["type": "time", "startHour": "6", "endHour": "12"])),
            Partition(id: "p3", nodeId: "n2", bounds: .custom(["type": "time", "startHour": "12", "endHour": "18"])),
            Partition(id: "p4", nodeId: "n2", bounds: .custom(["type": "time", "startHour": "18", "endHour": "24"]))
        ]
        
        // Test dates routing to correct partitions
        let calendar = Calendar.current
        let baseDate = Date()
        
        // 3 AM should go to p1
        var components = calendar.dateComponents([.year, .month, .day], from: baseDate)
        components.hour = 3
        if let date3am = calendar.date(from: components) {
            XCTAssertEqual(
                partitioner.partition(for: date3am, partitions: partitions)?.id,
                "p1"
            )
        }
        
        // 9 AM should go to p2
        components.hour = 9
        if let date9am = calendar.date(from: components) {
            XCTAssertEqual(
                partitioner.partition(for: date9am, partitions: partitions)?.id,
                "p2"
            )
        }
        
        // 3 PM should go to p3
        components.hour = 15
        if let date3pm = calendar.date(from: components) {
            XCTAssertEqual(
                partitioner.partition(for: date3pm, partitions: partitions)?.id,
                "p3"
            )
        }
        
        // 9 PM should go to p4
        components.hour = 21
        if let date9pm = calendar.date(from: components) {
            XCTAssertEqual(
                partitioner.partition(for: date9pm, partitions: partitions)?.id,
                "p4"
            )
        }
    }
    
    func testTimeBasedPartitionerValidation() {
        let partitioner = CustomPartitioner<Date>.timeBased(hoursPerPartition: 8)
        
        // Valid partitions covering 24 hours
        let validPartitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .custom(["type": "time", "startHour": "0", "endHour": "8"])),
            Partition(id: "p2", nodeId: "n1", bounds: .custom(["type": "time", "startHour": "8", "endHour": "16"])),
            Partition(id: "p3", nodeId: "n2", bounds: .custom(["type": "time", "startHour": "16", "endHour": "24"]))
        ]
        
        let validErrors = partitioner.validate(partitions: validPartitions)
        XCTAssertTrue(validErrors.isEmpty)
        
        // Invalid partitions with gap
        let gappedPartitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .custom(["type": "time", "startHour": "0", "endHour": "8"])),
            Partition(id: "p2", nodeId: "n1", bounds: .custom(["type": "time", "startHour": "10", "endHour": "24"]))
        ]
        
        let gapErrors = partitioner.validate(partitions: gappedPartitions)
        XCTAssertFalse(gapErrors.isEmpty)
        XCTAssertTrue(gapErrors.contains { error in
            if case .gapInPartitions = error { return true }
            return false
        })
        
        // Invalid partitions with overlap
        let overlappingPartitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .custom(["type": "time", "startHour": "0", "endHour": "12"])),
            Partition(id: "p2", nodeId: "n1", bounds: .custom(["type": "time", "startHour": "8", "endHour": "24"]))
        ]
        
        let overlapErrors = partitioner.validate(partitions: overlappingPartitions)
        XCTAssertFalse(overlapErrors.isEmpty)
        XCTAssertTrue(overlapErrors.contains { error in
            if case .overlappingPartitions = error { return true }
            return false
        })
    }
    
    // MARK: - Modulo Partitioner Tests
    
    func testModuloPartitioner() {
        let divisor = 4
        let partitioner = CustomPartitioner<Int>.modulo(divisor: divisor)
        
        // Create bucket-based partitions
        var partitions: [Partition] = []
        for i in 0..<divisor {
            partitions.append(
                Partition(
                    id: "p\(i)",
                    nodeId: "n\(i % 2)",
                    bounds: .custom(["bucket": "\(i)"])
                )
            )
        }
        
        // Test key distribution
        XCTAssertEqual(partitioner.partition(for: 0, partitions: partitions)?.id, "p0")
        XCTAssertEqual(partitioner.partition(for: 1, partitions: partitions)?.id, "p1")
        XCTAssertEqual(partitioner.partition(for: 2, partitions: partitions)?.id, "p2")
        XCTAssertEqual(partitioner.partition(for: 3, partitions: partitions)?.id, "p3")
        XCTAssertEqual(partitioner.partition(for: 4, partitions: partitions)?.id, "p0") // Wraps around
        XCTAssertEqual(partitioner.partition(for: 17, partitions: partitions)?.id, "p1") // 17 % 4 = 1
    }
    
    // MARK: - Default Partition Creation Tests
    
    func testDefaultPartitionCreation() {
        let partitioner = CustomPartitioner<String>.geographic()
        
        let nodes = ["node1", "node2", "node3"]
        let partitions = partitioner.createPartitions(count: 6, nodes: nodes)
        
        XCTAssertEqual(partitions.count, 6)
        
        // Check that partitions are distributed across nodes
        let nodeDistribution = Dictionary(grouping: partitions, by: { $0.nodeId })
        XCTAssertEqual(nodeDistribution["node1"]?.count, 2)
        XCTAssertEqual(nodeDistribution["node2"]?.count, 2)
        XCTAssertEqual(nodeDistribution["node3"]?.count, 2)
        
        // Check that all have custom bounds
        for partition in partitions {
            guard case .custom(let metadata) = partition.bounds else {
                XCTFail("Expected custom bounds")
                continue
            }
            XCTAssertNotNil(metadata["type"])
            XCTAssertEqual(metadata["type"], "geo")
        }
    }
    
    // MARK: - Default Rebalancing Tests
    
    func testDefaultRebalancing() {
        let config = CustomPartitioner<String>.Configuration(
            name: "test",
            partitionFunction: PartitionFunction(
                apply: { _, partitions in partitions.first },
                applyQuery: { _, partitions in partitions }
            )
        )
        
        let partitioner = CustomPartitioner(configuration: config)
        
        let partitions = [
            Partition(id: "p1", nodeId: "n1"),
            Partition(id: "p2", nodeId: "n1"),
            Partition(id: "p3", nodeId: "n2")
        ]
        
        let stats = PartitionStatistics(
            vectorCounts: [
                "p1": 1000,
                "p2": 1000,
                "p3": 100
            ]
        )
        
        let plan = partitioner.suggestRebalancing(partitions: partitions, stats: stats)
        XCTAssertNotNil(plan)
        
        // Should suggest moving from n1 to n2
        if let move = plan?.moves.first {
            XCTAssertEqual(move.fromNode, "n1")
            XCTAssertEqual(move.toNode, "n2")
        }
    }
    
    // MARK: - Performance Tests
    
    func testCustomPartitionerPerformance() {
        let partitioner = CustomPartitioner<Int>.modulo(divisor: 10)
        
        // Create partitions
        var partitions: [Partition] = []
        for i in 0..<10 {
            partitions.append(
                Partition(
                    id: "p\(i)",
                    nodeId: "n\(i % 3)",
                    bounds: .custom(["bucket": "\(i)"])
                )
            )
        }
        
        measure {
            // Perform many lookups
            for i in 0..<100000 {
                _ = partitioner.partition(for: i, partitions: partitions)
            }
        }
    }
    
    func testComplexCustomFunction() {
        // Test with a more complex partition function
        let partitionFunction = PartitionFunction<String>(
            apply: { key, partitions in
                // Hash-based with prefix extraction
                let components = key.split(separator: ":")
                guard components.count >= 2 else { return partitions.first }
                
                let prefix = String(components[0])
                let hash = prefix.hashValue
                let bucket = abs(hash) % partitions.count
                
                return partitions.first { partition in
                    guard case .custom(let metadata) = partition.bounds,
                          let bucketStr = metadata["bucket"],
                          let partitionBucket = Int(bucketStr) else {
                        return false
                    }
                    return partitionBucket == bucket
                }
            },
            applyQuery: { query, partitions in
                // For queries, might need to check multiple buckets
                return partitions
            }
        )
        
        let config = CustomPartitioner<String>.Configuration(
            name: "complex",
            partitionFunction: partitionFunction
        )
        
        let partitioner = CustomPartitioner(configuration: config)
        
        let partitions = [
            Partition(id: "p0", nodeId: "n1", bounds: .custom(["bucket": "0"])),
            Partition(id: "p1", nodeId: "n2", bounds: .custom(["bucket": "1"])),
            Partition(id: "p2", nodeId: "n3", bounds: .custom(["bucket": "2"]))
        ]
        
        // Test deterministic routing
        let key1 = "user:123"
        let key2 = "user:456"
        let key3 = "order:789"
        
        let p1 = partitioner.partition(for: key1, partitions: partitions)
        let p1Again = partitioner.partition(for: key1, partitions: partitions)
        XCTAssertEqual(p1?.id, p1Again?.id) // Same key always goes to same partition
        
        // Different prefixes might go to different partitions
        let p2 = partitioner.partition(for: key2, partitions: partitions)
        let p3 = partitioner.partition(for: key3, partitions: partitions)
        XCTAssertNotNil(p2)
        XCTAssertNotNil(p3)
    }
}
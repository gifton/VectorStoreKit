// VectorStoreKit: Range Partitioner Tests
//
// Comprehensive tests for range-based partitioning

import XCTest
@testable import VectorStoreKit

final class RangePartitionerTests: XCTestCase {
    
    // MARK: - RangeKey Tests
    
    func testStringRangeKey() {
        // Test midpoint
        let mid1 = String.midpoint(between: "a", and: "z")
        XCTAssertNotNil(mid1)
        XCTAssertGreaterThan(mid1!, "a")
        XCTAssertLessThan(mid1!, "z")
        
        let mid2 = String.midpoint(between: "apple", and: "banana")
        XCTAssertNotNil(mid2)
        
        // Test divisibility
        XCTAssertTrue(String.isDivisible(from: "a", to: "z"))
        XCTAssertFalse(String.isDivisible(from: "z", to: "a")) // Wrong order
    }
    
    func testIntRangeKey() {
        // Test midpoint
        let mid1 = Int.midpoint(between: 0, and: 100)
        XCTAssertEqual(mid1, 50)
        
        let mid2 = Int.midpoint(between: 10, and: 11)
        XCTAssertEqual(mid2, 10) // Integer division
        
        // Test divisibility
        XCTAssertTrue(Int.isDivisible(from: 0, to: 100))
        XCTAssertFalse(Int.isDivisible(from: 10, to: 11)) // Only 1 apart
    }
    
    func testDateRangeKey() {
        let date1 = Date(timeIntervalSince1970: 1000)
        let date2 = Date(timeIntervalSince1970: 2000)
        
        // Test midpoint
        let mid = Date.midpoint(between: date1, and: date2)
        XCTAssertNotNil(mid)
        XCTAssertEqual(mid?.timeIntervalSince1970, 1500)
        
        // Test divisibility
        XCTAssertTrue(Date.isDivisible(from: date1, to: date2))
        
        let closeDate1 = Date(timeIntervalSince1970: 1000)
        let closeDate2 = Date(timeIntervalSince1970: 1000.5)
        XCTAssertFalse(Date.isDivisible(from: closeDate1, to: closeDate2))
    }
    
    // MARK: - Basic Partitioner Tests
    
    func testStringPartitioner() {
        let partitioner = RangePartitioner<String>()
        
        // Create partitions
        let partitions = partitioner.createPartitions(count: 4, nodes: ["node1", "node2"])
        XCTAssertEqual(partitions.count, 4)
        
        // Check that partitions cover the alphabet
        let firstPartition = partitions.first!
        if case .range(let min, _) = firstPartition.bounds {
            XCTAssertEqual(min, "a")
        }
        
        let lastPartition = partitions.last!
        if case .range(_, let max) = lastPartition.bounds {
            XCTAssertTrue(max.contains("z"))
        }
    }
    
    func testPartitionForKey() {
        let partitioner = RangePartitioner<String>()
        
        // Create manual partitions
        let partitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .range(min: "\"a\"", max: "\"f\"")),
            Partition(id: "p2", nodeId: "n1", bounds: .range(min: "\"f\"", max: "\"m\"")),
            Partition(id: "p3", nodeId: "n2", bounds: .range(min: "\"m\"", max: "\"t\"")),
            Partition(id: "p4", nodeId: "n2", bounds: .range(min: "\"t\"", max: "\"z\""))
        ]
        
        // Test key assignment
        let p1 = partitioner.partition(for: "apple", partitions: partitions)
        XCTAssertEqual(p1?.id, "p1")
        
        let p2 = partitioner.partition(for: "hello", partitions: partitions)
        XCTAssertEqual(p2?.id, "p2")
        
        let p3 = partitioner.partition(for: "orange", partitions: partitions)
        XCTAssertEqual(p3?.id, "p3")
        
        let p4 = partitioner.partition(for: "world", partitions: partitions)
        XCTAssertEqual(p4?.id, "p4")
    }
    
    // MARK: - Range Query Tests
    
    func testRangeQuery() {
        let partitioner = RangePartitioner<String>()
        
        let partitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .range(min: "\"a\"", max: "\"f\"")),
            Partition(id: "p2", nodeId: "n1", bounds: .range(min: "\"f\"", max: "\"m\"")),
            Partition(id: "p3", nodeId: "n2", bounds: .range(min: "\"m\"", max: "\"t\"")),
            Partition(id: "p4", nodeId: "n2", bounds: .range(min: "\"t\"", max: "\"z\""))
        ]
        
        // Query spanning multiple partitions
        let results = partitioner.partitionsForRangeQuery(
            from: "d",
            to: "p",
            partitions: partitions
        )
        
        XCTAssertEqual(results.count, 3) // p1, p2, p3
        XCTAssertEqual(results.map { $0.id }, ["p1", "p2", "p3"])
        
        // Query within single partition
        let singleResult = partitioner.partitionsForRangeQuery(
            from: "b",
            to: "c",
            partitions: partitions
        )
        
        XCTAssertEqual(singleResult.count, 1)
        XCTAssertEqual(singleResult.first?.id, "p1")
    }
    
    // MARK: - Split and Merge Tests
    
    func testPartitionSplit() {
        let partitioner = RangePartitioner<Int>()
        
        let partition = Partition(
            id: "original",
            nodeId: "node1",
            bounds: .range(min: "0", max: "100")
        )
        
        let split = partitioner.splitPartition(partition)
        XCTAssertNotNil(split)
        
        if let (left, right) = split {
            XCTAssertEqual(left.id, "original_left")
            XCTAssertEqual(right.id, "original_right")
            
            // Check bounds
            if case .range(let minL, let maxL) = left.bounds,
               case .range(let minR, let maxR) = right.bounds {
                XCTAssertEqual(minL, "0")
                XCTAssertEqual(maxL, "50") // Midpoint
                XCTAssertEqual(minR, "50")
                XCTAssertEqual(maxR, "100")
            }
        }
    }
    
    func testPartitionMerge() {
        let partitioner = RangePartitioner<Int>()
        
        let p1 = Partition(
            id: "p1",
            nodeId: "node1",
            bounds: .range(min: "0", max: "50")
        )
        
        let p2 = Partition(
            id: "p2",
            nodeId: "node1",
            bounds: .range(min: "50", max: "100")
        )
        
        let merged = partitioner.mergePartitions(p1, p2)
        XCTAssertNotNil(merged)
        
        if let merged = merged {
            XCTAssertEqual(merged.id, "p1_p2_merged")
            
            if case .range(let min, let max) = merged.bounds {
                XCTAssertEqual(min, "0")
                XCTAssertEqual(max, "100")
            }
        }
    }
    
    // MARK: - Rebalancing Tests
    
    func testAutoSplitSuggestion() {
        let config = RangePartitioner<String>.Configuration(
            maxKeysPerPartition: 1000,
            autoSplit: true,
            splitThreshold: 0.8
        )
        let partitioner = RangePartitioner<String>(configuration: config)
        
        let partitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .range(min: "\"a\"", max: "\"m\"")),
            Partition(id: "p2", nodeId: "n1", bounds: .range(min: "\"m\"", max: "\"z\""))
        ]
        
        let stats = PartitionStatistics(
            vectorCounts: [
                "p1": 900, // 90% full, above threshold
                "p2": 500  // 50% full
            ]
        )
        
        let plan = partitioner.suggestRebalancing(partitions: partitions, stats: stats)
        XCTAssertNotNil(plan)
        
        // Should suggest splitting p1
        XCTAssertEqual(plan?.splits.count, 1)
        XCTAssertEqual(plan?.splits.first?.partitionId, "p1")
    }
    
    func testAutoMergeSuggestion() {
        let config = RangePartitioner<String>.Configuration(
            maxKeysPerPartition: 1000,
            autoMerge: true,
            mergeThreshold: 0.2
        )
        let partitioner = RangePartitioner<String>(configuration: config)
        
        let partitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .range(min: "\"a\"", max: "\"m\"")),
            Partition(id: "p2", nodeId: "n1", bounds: .range(min: "\"m\"", max: "\"z\""))
        ]
        
        let stats = PartitionStatistics(
            vectorCounts: [
                "p1": 100, // 10% full
                "p2": 150  // 15% full
            ]
        )
        
        let plan = partitioner.suggestRebalancing(partitions: partitions, stats: stats)
        XCTAssertNotNil(plan)
        
        // Should suggest merging p1 and p2
        XCTAssertEqual(plan?.merges.count, 1)
        XCTAssertEqual(Set(plan?.merges.first?.partitionIds ?? []), Set(["p1", "p2"]))
    }
    
    // MARK: - Validation Tests
    
    func testValidation() {
        let partitioner = RangePartitioner<String>()
        
        // Valid partitions
        let validPartitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .range(min: "\"a\"", max: "\"m\"")),
            Partition(id: "p2", nodeId: "n1", bounds: .range(min: "\"m\"", max: "\"z\""))
        ]
        
        let validErrors = partitioner.validate(partitions: validPartitions)
        XCTAssertTrue(validErrors.isEmpty)
        
        // Overlapping partitions
        let overlappingPartitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .range(min: "\"a\"", max: "\"n\"")),
            Partition(id: "p2", nodeId: "n1", bounds: .range(min: "\"m\"", max: "\"z\""))
        ]
        
        let overlapErrors = partitioner.validate(partitions: overlappingPartitions)
        XCTAssertFalse(overlapErrors.isEmpty)
        XCTAssertTrue(overlapErrors.contains { error in
            if case .overlappingPartitions = error { return true }
            return false
        })
        
        // Gap in partitions
        let gappedPartitions = [
            Partition(id: "p1", nodeId: "n1", bounds: .range(min: "\"a\"", max: "\"f\"")),
            Partition(id: "p2", nodeId: "n1", bounds: .range(min: "\"m\"", max: "\"z\""))
        ]
        
        let gapErrors = partitioner.validate(partitions: gappedPartitions)
        XCTAssertFalse(gapErrors.isEmpty)
        XCTAssertTrue(gapErrors.contains { error in
            if case .gapInPartitions = error { return true }
            return false
        })
    }
    
    // MARK: - Performance Tests
    
    func testRangeQueryPerformance() {
        let partitioner = RangePartitioner<Int>()
        
        // Create many partitions
        var partitions: [Partition] = []
        for i in 0..<100 {
            let min = i * 1000
            let max = (i + 1) * 1000
            let partition = Partition(
                id: "p\(i)",
                nodeId: "node\(i % 10)",
                bounds: .range(min: "\(min)", max: "\(max)")
            )
            partitions.append(partition)
        }
        
        measure {
            // Perform many range queries
            for i in 0..<1000 {
                let start = i * 10
                let end = start + 100
                _ = partitioner.partitionsForRangeQuery(
                    from: start,
                    to: end,
                    partitions: partitions
                )
            }
        }
    }
    
    func testKeyLookupPerformance() {
        let partitioner = RangePartitioner<String>()
        
        // Create partitions
        let partitions = partitioner.createPartitions(count: 26, nodes: ["n1", "n2", "n3"])
        
        measure {
            // Perform many key lookups
            for i in 0..<10000 {
                let key = "key_\(i)"
                _ = partitioner.partition(for: key, partitions: partitions)
            }
        }
    }
}
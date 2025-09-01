// SimplePartitionTest.swift
// Standalone test to verify partitioning implementation

import Foundation
@testable import VectorStoreKit

@main
struct SimplePartitionTest {
    static func main() async throws {
        print("🧪 Testing Distributed Partition Strategy Implementation...")
        
        // Test 1: Create partitions
        print("\n1️⃣ Testing partition creation...")
        let partitioner = HashPartitioner(hashFunction: .sha256)
        let nodes = ["node-1", "node-2", "node-3"]
        let partitions = partitioner.createPartitions(count: 6, nodes: nodes)
        
        print("   Created \(partitions.count) partitions")
        for (i, partition) in partitions.enumerated() {
            if case .hash(let min, let max) = partition.bounds {
                print("   Partition \(i): node=\(partition.nodeId), range=\(min)...\(max)")
            }
        }
        
        // Test 2: Partition key assignment
        print("\n2️⃣ Testing key partitioning...")
        let testKeys = ["user123", "product456", "order789", "session42", "item99"]
        var distribution: [String: Int] = [:]
        
        for key in testKeys {
            if let partition = partitioner.partition(for: key, partitions: partitions) {
                distribution[partition.id, default: 0] += 1
                print("   Key '\(key)' -> \(partition.id)")
            }
        }
        
        // Test 3: Distribution test
        print("\n3️⃣ Testing distribution with 1000 keys...")
        distribution.removeAll()
        
        for i in 0..<1000 {
            let key = "key_\(i)"
            if let partition = partitioner.partition(for: key, partitions: partitions) {
                distribution[partition.id, default: 0] += 1
            }
        }
        
        print("   Distribution across partitions:")
        for (partitionId, count) in distribution.sorted(by: { $0.key < $1.key }) {
            let percentage = Double(count) / 10.0
            print("   \(partitionId): \(count) keys (\(String(format: "%.1f", percentage))%)")
        }
        
        // Test 4: Validation
        print("\n4️⃣ Testing partition validation...")
        let errors = partitioner.validate(partitions: partitions)
        if errors.isEmpty {
            print("   ✅ No validation errors")
        } else {
            print("   ❌ Validation errors found:")
            for error in errors {
                print("      - \(error)")
            }
        }
        
        // Test 5: Rebalancing suggestions
        print("\n5️⃣ Testing rebalancing suggestions...")
        let stats = PartitionStatistics(
            vectorCounts: [
                "partition_0": 5000,
                "partition_1": 1000,
                "partition_2": 3000,
                "partition_3": 3000,
                "partition_4": 2000,
                "partition_5": 6000
            ],
            loadMetrics: [
                "partition_0": 500.0,
                "partition_1": 100.0,
                "partition_2": 300.0,
                "partition_3": 300.0,
                "partition_4": 200.0,
                "partition_5": 600.0
            ]
        )
        
        if let plan = partitioner.suggestRebalancing(partitions: partitions, stats: stats) {
            print("   Rebalancing suggestions:")
            print("   - Moves: \(plan.moves.count)")
            print("   - Splits: \(plan.splits.count)")
            print("   - Estimated improvement: \(String(format: "%.1f", plan.estimatedImprovement * 100))%")
            
            for move in plan.moves {
                print("   📦 Move \(move.partitionId) from \(move.fromNode) to \(move.toNode)")
            }
            
            for split in plan.splits {
                print("   ✂️ Split \(split.partitionId): \(split.reason)")
            }
        } else {
            print("   No rebalancing needed")
        }
        
        // Test 6: Edge cases
        print("\n6️⃣ Testing edge cases...")
        
        // Empty partitions
        let emptyResult = partitioner.partition(for: "test", partitions: [])
        print("   Empty partitions: \(emptyResult == nil ? "✅ nil" : "❌ not nil")")
        
        // Single partition
        let singlePartition = [
            Partition(id: "single", nodeId: "node", bounds: .hash(min: 0, max: UInt64.max))
        ]
        let singleResult = partitioner.partition(for: "test", partitions: singlePartition)
        print("   Single partition: \(singleResult?.id == "single" ? "✅" : "❌")")
        
        print("\n✅ All partition strategy tests completed!")
    }
}
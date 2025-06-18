// VectorStoreKit: Simple Migration Example
//
// Demonstrates basic migration from single-node to distributed system

import Foundation
import VectorStoreKit

@main
struct SimpleMigrationExample {
    
    static func main() async throws {
        print("🚀 VectorStoreKit Simple Migration Example")
        print("=" * 50)
        
        // Step 1: Create source (single-node) store
        print("\n📦 Step 1: Setting up source store...")
        let sourceStore = try await createSourceStore()
        
        // Step 2: Populate with sample data
        print("\n📝 Step 2: Populating source with sample data...")
        try await populateSourceStore(sourceStore)
        
        // Step 3: Create distributed target system
        print("\n🌐 Step 3: Setting up distributed system...")
        let targetSystem = try await createDistributedSystem()
        
        // Step 4: Configure migration
        print("\n⚙️ Step 4: Configuring migration...")
        let migrationConfig = MigrationConfiguration(
            batchSize: 1000,
            parallelWorkers: 2,
            enableZeroDowntime: false,  // Simple migration
            enableCheckpointing: true,
            validationConfig: ValidationConfiguration(
                validateBatches: true,
                sampleRate: 0.1,
                validateAfterMigration: true
            )
        )
        
        // Step 5: Create migrator
        let migrator = BatchMigrator(
            configuration: migrationConfig,
            sourceStore: SingleNodeStoreAdapter(store: sourceStore),
            targetSystem: DistributedSystemImplementation(system: targetSystem)
        )
        
        // Step 6: Run migration with progress monitoring
        print("\n🔄 Step 5: Starting migration...")
        print("Progress:")
        
        var lastPercentage = 0.0
        for try await progress in migrator.migrate() {
            // Print progress bar
            if progress.percentage - lastPercentage >= 1.0 {
                printProgressBar(
                    current: progress.migratedCount,
                    total: progress.totalCount,
                    phase: progress.phase
                )
                lastPercentage = progress.percentage
            }
        }
        
        print("\n\n✅ Migration completed!")
        
        // Step 7: Verify migration
        print("\n🔍 Step 6: Verifying migration...")
        try await verifyMigration(source: sourceStore, target: targetSystem)
        
        // Step 8: Show statistics
        print("\n📊 Step 7: Migration Statistics")
        await showStatistics(source: sourceStore, target: targetSystem)
        
        print("\n🎉 Migration example completed successfully!")
    }
    
    // MARK: - Store Creation
    
    static func createSourceStore() async throws -> VectorStore<SIMD32<Float>, SampleMetadata, HNSWIndex<SIMD32<Float>, SampleMetadata>, HierarchicalStorage, BasicLRUVectorCache<SIMD32<Float>>> {
        // Create single-node store using VectorUniverse
        let store = try await VectorUniverse<SIMD32<Float>, SampleMetadata>()
            .indexing(HNSWIndexingStrategy<SIMD32<Float>, SampleMetadata>(
                maxConnections: 16,
                efConstruction: 200
            ))
            .storage(HierarchicalResearchStorageStrategy())
            .caching(LRUCachingStrategy(maxMemory: 50_000_000))
            .materialize()
        
        print("✓ Created single-node store with HNSW index")
        return store
    }
    
    static func createDistributedSystem() async throws -> MockDistributedSystem {
        // In a real scenario, this would create an actual distributed system
        let system = MockDistributedSystem(
            nodes: ["node1", "node2", "node3"],
            partitionStrategy: .consistentHash(virtualNodes: 150),
            replicationFactor: 2
        )
        
        try await system.initialize()
        
        print("✓ Created distributed system with 3 nodes")
        return system
    }
    
    // MARK: - Data Population
    
    static func populateSourceStore(_ store: VectorStore<SIMD32<Float>, SampleMetadata, HNSWIndex<SIMD32<Float>, SampleMetadata>, HierarchicalStorage, BasicLRUVectorCache<SIMD32<Float>>>) async throws {
        let vectorCount = 10_000
        let dimension = 128
        
        print("Generating \(vectorCount) vectors of dimension \(dimension)...")
        
        // Generate sample vectors in batches
        let batchSize = 1000
        for batchStart in stride(from: 0, to: vectorCount, by: batchSize) {
            var entries: [VectorEntry<SIMD32<Float>, SampleMetadata>] = []
            
            for i in batchStart..<min(batchStart + batchSize, vectorCount) {
                // Create sample vector
                var values = [Float](repeating: 0, count: dimension)
                for j in 0..<dimension {
                    values[j] = Float.random(in: -1...1)
                }
                
                // Normalize vector
                let norm = sqrt(values.reduce(0) { $0 + $1 * $1 })
                if norm > 0 {
                    values = values.map { $0 / norm }
                }
                
                let vector = SIMD32<Float>(values[0..<32])  // Using SIMD32 for example
                
                let entry = VectorEntry(
                    id: "vec_\(i)",
                    vector: vector,
                    metadata: SampleMetadata(
                        category: ["A", "B", "C", "D"][i % 4],
                        timestamp: Date(),
                        score: Float.random(in: 0...1)
                    )
                )
                
                entries.append(entry)
            }
            
            _ = try await store.add(entries)
            
            print("  Added batch \(batchStart / batchSize + 1)/\(vectorCount / batchSize)")
        }
        
        print("✓ Successfully populated source store with \(vectorCount) vectors")
    }
    
    // MARK: - Migration Verification
    
    static func verifyMigration(
        source: VectorStore<SIMD32<Float>, SampleMetadata, HNSWIndex<SIMD32<Float>, SampleMetadata>, HierarchicalStorage, BasicLRUVectorCache<SIMD32<Float>>>,
        target: MockDistributedSystem
    ) async throws {
        print("Running migration verification...")
        
        // Create validator
        let validator = MigrationValidator<SIMD32<Float>, SampleMetadata>(
            configuration: MigrationValidator.Configuration(
                sampleRate: 0.05,  // Check 5% of data
                validateCounts: true,
                validateSimilarity: true,
                validateSearchQuality: true,
                testQueryCount: 10
            )
        )
        
        // Run validation
        let result = try await validator.validate(
            source: SingleNodeStoreAdapter(store: source),
            target: DistributedSystemAdapter(system: target)
        ) { progress in
            print("  \(progress.phase): \(Int(progress.progress * 100))%")
        }
        
        if result.isSuccessful {
            print("✓ Migration validation passed!")
        } else {
            print("❌ Migration validation failed:")
            for error in result.errors {
                print("  - \(error)")
            }
        }
        
        if !result.warnings.isEmpty {
            print("⚠️ Warnings:")
            for warning in result.warnings {
                print("  - \(warning)")
            }
        }
    }
    
    // MARK: - Statistics
    
    static func showStatistics(
        source: VectorStore<SIMD32<Float>, SampleMetadata, HNSWIndex<SIMD32<Float>, SampleMetadata>, HierarchicalStorage, BasicLRUVectorCache<SIMD32<Float>>>,
        target: MockDistributedSystem
    ) async {
        let sourceStats = await source.statistics()
        let targetStats = await target.globalStatistics()
        
        print("Source Store:")
        print("  Vectors: \(sourceStats.vectorCount)")
        print("  Memory: \(formatBytes(sourceStats.memoryUsage))")
        print("  Disk: \(formatBytes(sourceStats.diskUsage))")
        
        print("\nDistributed System:")
        print("  Total Vectors: \(targetStats.totalVectors)")
        print("  Nodes: \(targetStats.nodeCount)")
        print("  Partitions: \(targetStats.partitionCount)")
        print("  Memory: \(formatBytes(targetStats.totalMemoryUsage))")
        print("  Disk: \(formatBytes(targetStats.totalDiskUsage))")
        
        print("\nPartition Distribution:")
        for (node, count) in targetStats.vectorsPerNode {
            print("  \(node): \(count) vectors")
        }
    }
    
    // MARK: - Helper Functions
    
    static func printProgressBar(current: Int, total: Int, phase: String) {
        let percentage = Double(current) / Double(total)
        let filledLength = Int(percentage * 40)
        let bar = String(repeating: "█", count: filledLength) + 
                 String(repeating: "░", count: 40 - filledLength)
        
        print("\r[\(bar)] \(Int(percentage * 100))% - \(phase) (\(current)/\(total))", terminator: "")
        fflush(stdout)
    }
    
    static func formatBytes(_ bytes: Int) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var size = Double(bytes)
        var unitIndex = 0
        
        while size >= 1024 && unitIndex < units.count - 1 {
            size /= 1024
            unitIndex += 1
        }
        
        return String(format: "%.2f %@", size, units[unitIndex])
    }
}

// MARK: - Sample Types

struct SampleMetadata: Codable, Sendable {
    let category: String
    let timestamp: Date
    let score: Float
}

// MARK: - Mock Distributed System

/// Mock distributed system for demonstration
class MockDistributedSystem {
    let nodes: [String]
    let partitionStrategy: PartitioningStrategy
    let replicationFactor: Int
    private var data: [String: VectorEntry<SIMD32<Float>, SampleMetadata>] = [:]
    private var partitionManager: PartitionManager
    
    enum PartitioningStrategy {
        case consistentHash(virtualNodes: Int)
        case range
        case custom
    }
    
    init(nodes: [String], partitionStrategy: PartitioningStrategy, replicationFactor: Int) {
        self.nodes = nodes
        self.partitionStrategy = partitionStrategy
        self.replicationFactor = replicationFactor
        self.partitionManager = PartitionManager()
    }
    
    func initialize() async throws {
        // Initialize partitions
        for (index, node) in nodes.enumerated() {
            _ = try await partitionManager.createPartition(
                id: "partition_\(index)",
                ownerNode: node,
                bounds: .hash(
                    min: UInt64(index) * (UInt64.max / UInt64(nodes.count)),
                    max: UInt64(index + 1) * (UInt64.max / UInt64(nodes.count))
                )
            )
        }
    }
    
    func globalStatistics() async -> GlobalSystemStatistics {
        GlobalSystemStatistics(
            totalVectors: data.count,
            nodeCount: nodes.count,
            partitionCount: nodes.count,
            totalMemoryUsage: data.count * 512,  // Rough estimate
            totalDiskUsage: data.count * 1024,   // Rough estimate
            vectorsPerNode: Dictionary(
                uniqueKeysWithValues: nodes.map { node in
                    (node, data.count / nodes.count)  // Simplified distribution
                }
            )
        )
    }
}

struct GlobalSystemStatistics {
    let totalVectors: Int
    let nodeCount: Int
    let partitionCount: Int
    let totalMemoryUsage: Int
    let totalDiskUsage: Int
    let vectorsPerNode: [String: Int]
}

// MARK: - Adapters

/// Adapter to make single-node store compatible with migration interfaces
struct SingleNodeStoreAdapter: SourceStore {
    typealias Vector = SIMD32<Float>
    typealias Metadata = SampleMetadata
    
    let store: VectorStore<Vector, Metadata, HNSWIndex<Vector, Metadata>, HierarchicalStorage, BasicLRUVectorCache<Vector>>
    
    func statistics() async -> StoreStatistics {
        await store.statistics()
    }
    
    func getAllIds() async throws -> [String] {
        // In real implementation, would enumerate all IDs
        // For demo, returning sample IDs
        return (0..<10_000).map { "vec_\($0)" }
    }
    
    func getEntry(id: String) async throws -> VectorEntry<Vector, Metadata>? {
        // In real implementation, would fetch from store
        // For demo, returning mock entry
        return VectorEntry(
            id: id,
            vector: SIMD32<Float>(repeating: 0.1),
            metadata: SampleMetadata(
                category: "A",
                timestamp: Date(),
                score: 0.5
            )
        )
    }
    
    func enableChangeTracking(since: Date) async throws -> ChangeTracker {
        MockChangeTracker()
    }
    
    func disableChangeTracking() async throws {
        // No-op for demo
    }
    
    func withReadLock<T>(_ operation: () async throws -> T) async throws -> T {
        try await operation()
    }
}

/// Adapter for distributed system
struct DistributedSystemImplementation: TargetDistributedSystem {
    typealias Vector = SIMD32<Float>
    typealias Metadata = SampleMetadata
    
    let system: MockDistributedSystem
    
    func prepareForMigration(expectedVectors: Int, expectedSize: Int) async throws {
        print("  Preparing distributed system for \(expectedVectors) vectors...")
    }
    
    func batchInsert(entries: [VectorEntry<Vector, Metadata>], options: DistributedInsertOptions) async throws -> (successCount: Int, failureCount: Int) {
        // Simulate batch insert
        return (successCount: entries.count, failureCount: 0)
    }
    
    func upsert(_ entry: VectorEntry<Vector, Metadata>) async throws -> Bool {
        true
    }
    
    func delete(id: String) async throws -> Bool {
        true
    }
    
    func getEntry(id: String) async throws -> VectorEntry<Vector, Metadata>? {
        // Return mock entry for demo
        return VectorEntry(
            id: id,
            vector: SIMD32<Float>(repeating: 0.1),
            metadata: SampleMetadata(
                category: "A",
                timestamp: Date(),
                score: 0.5
            )
        )
    }
    
    func optimizeAfterMigration() async throws {
        print("  Optimizing distributed system...")
    }
}

/// Mock change tracker
struct MockChangeTracker: ChangeTracker {
    func getChanges() async throws -> [ChangeRecord] {
        []  // No changes for simple migration
    }
    
    func markProcessed(_ changes: [ChangeRecord]) async throws {
        // No-op for demo
    }
}

// Helper extension
extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
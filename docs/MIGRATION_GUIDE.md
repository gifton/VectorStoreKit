# VectorStoreKit Migration Guide

## From Single-Node to Distributed Architecture

This guide helps you migrate from VectorStoreKit's original single-node APIs to the new distributed system architecture. The distributed system provides horizontal scalability, fault tolerance, and high availability while maintaining the performance characteristics of the original system.

## Table of Contents

1. [Architectural Overview](#architectural-overview)
2. [Key Changes](#key-changes)
3. [API Comparison](#api-comparison)
4. [Migration Steps](#migration-steps)
5. [Code Examples](#code-examples)
6. [Common Patterns](#common-patterns)
7. [Performance Considerations](#performance-considerations)
8. [Backwards Compatibility](#backwards-compatibility)
9. [Troubleshooting](#troubleshooting)

## Architectural Overview

### Old Architecture (Single-Node)

```
┌─────────────────────────────────┐
│     VectorStore                 │
│  ┌─────────────────────────┐    │
│  │  Index (HNSW/IVF/etc)  │    │
│  └─────────────────────────┘    │
│  ┌─────────────────────────┐    │
│  │  Storage Backend       │    │
│  └─────────────────────────┘    │
│  ┌─────────────────────────┐    │
│  │  Cache Layer          │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
```

### New Architecture (Distributed)

```
┌─────────────────────────────────────────────────────┐
│                Distributed System                    │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │   Node 1    │  │   Node 2    │  │   Node 3    │ │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │ │
│  │ │Partition│ │  │ │Partition│ │  │ │Partition│ │ │
│  │ │    1    │ │  │ │    2    │ │  │ │    3    │ │ │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                     │
│  ┌────────────────────────────────────────────────┐ │
│  │         Partition Manager & Coordinator        │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

## Key Changes

### 1. Data Partitioning
- **Old**: All data stored in a single index
- **New**: Data distributed across multiple partitions using configurable strategies

### 2. Scalability
- **Old**: Vertical scaling only (bigger machine)
- **New**: Horizontal scaling (add more nodes)

### 3. Fault Tolerance
- **Old**: Single point of failure
- **New**: Replication and automatic failover

### 4. Consistency
- **Old**: Always consistent (single node)
- **New**: Configurable consistency levels (eventual, strong)

### 5. Metal Acceleration
- **Old**: Direct Metal compute access
- **New**: Distributed Metal compute with intelligent routing

## API Comparison

### Basic Operations

| Operation | Old API | New API |
|-----------|---------|---------|
| **Initialize Store** | `VectorStore(index:storage:cache:)` | `DistributedVectorSystem.builder()` |
| **Add Vectors** | `store.add(entries)` | `system.add(entries, partition: .auto)` |
| **Search** | `store.search(query:k:)` | `system.search(query:k:consistency:)` |
| **Update** | `store.update(id:vector:)` | `system.update(id:vector:partition:)` |
| **Delete** | `store.delete(id:)` | `system.delete(id:partition:)` |

### Advanced Operations

| Operation | Old API | New API |
|-----------|---------|---------|
| **Batch Operations** | `store.add(entries)` | `system.batch { batch in ... }` |
| **Optimization** | `store.optimize()` | `system.rebalance()` |
| **Statistics** | `store.statistics()` | `system.globalStatistics()` |
| **Export** | `store.export()` | `system.exportPartition()` |

## Migration Steps

### Step 1: Assess Your Current Implementation

```swift
// Identify your current setup
let currentConfig = VectorStoreKit.detectConfiguration()
print("Current setup:")
print("- Index type: \(currentConfig.indexType)")
print("- Vector count: \(currentConfig.vectorCount)")
print("- Storage size: \(currentConfig.storageSize)")
```

### Step 2: Plan Your Distribution Strategy

```swift
// Choose partitioning strategy based on your data
let strategy = PartitioningStrategy.recommendation(
    dataSize: currentConfig.storageSize,
    queryPatterns: .analyzed,
    hardware: .available
)
```

### Step 3: Create Migration Configuration

```swift
let migrationConfig = MigrationConfiguration(
    sourceStore: existingStore,
    targetNodes: ["node1", "node2", "node3"],
    partitionStrategy: .consistentHash(virtualNodes: 150),
    replicationFactor: 3,
    migrationBatchSize: 10_000,
    enableZeroDowntime: true
)
```

### Step 4: Execute Migration

```swift
let migrator = BatchMigrator(configuration: migrationConfig)

// Monitor progress
for await progress in migrator.migrate() {
    print("Migration progress: \(progress.percentage)%")
    print("- Migrated: \(progress.migratedCount)")
    print("- Remaining: \(progress.remainingCount)")
    print("- ETA: \(progress.estimatedTimeRemaining)")
}
```

## Code Examples

### Before (Single-Node)

```swift
// Old initialization
let store = try await VectorUniverse<SIMD32<Float>, MyMetadata>()
    .indexing(HNSWIndexingStrategy(maxConnections: 16))
    .storage(HierarchicalResearchStorageStrategy())
    .caching(LRUCachingStrategy(maxMemory: 100_000_000))
    .materialize()

// Old operations
try await store.add(vectors)
let results = try await store.search(query: queryVector, k: 10)
```

### After (Distributed)

```swift
// New initialization
let system = try await DistributedVectorSystem.builder()
    .withNodes(["node1", "node2", "node3"])
    .withPartitioning(.consistentHash(virtualNodes: 150))
    .withReplication(factor: 3)
    .withIndex(.hnsw(maxConnections: 16))
    .withStorage(.hierarchical)
    .withCache(.distributed(maxMemory: 300_000_000))
    .build()

// New operations with consistency control
try await system.add(vectors, consistency: .quorum)
let results = try await system.search(
    query: queryVector, 
    k: 10,
    consistency: .one,
    timeout: .seconds(5)
)
```

### Using the Legacy Adapter

For gradual migration, use the `LegacyAdapter` to maintain existing API compatibility:

```swift
// Wrap distributed system with legacy API
let legacyStore = LegacyAdapter(distributedSystem: system)

// Use existing code unchanged
try await legacyStore.add(vectors)  // Automatically distributed
let results = try await legacyStore.search(query: queryVector, k: 10)
```

## Common Patterns

### Pattern 1: Gradual Migration

```swift
// Start with read traffic on distributed system
let migrationStrategy = GradualMigration()
    .percentageToNewSystem(read: 10, write: 0)
    .withFallback(.automatic)
    .withValidation(.enabled)

// Gradually increase traffic
for percentage in stride(from: 10, through: 100, by: 10) {
    migrationStrategy.updatePercentage(read: percentage)
    try await Task.sleep(for: .hours(24))
    
    // Monitor and validate
    let metrics = await migrationStrategy.compareMetrics()
    guard metrics.isHealthy else {
        // Rollback if issues detected
        migrationStrategy.rollback()
        break
    }
}
```

### Pattern 2: Blue-Green Deployment

```swift
// Set up parallel systems
let blueSystem = existingStore  // Current production
let greenSystem = try await createDistributedSystem()  // New system

// Sync data
let syncer = DataSynchronizer(source: blueSystem, target: greenSystem)
try await syncer.performInitialSync()
try await syncer.enableContinuousSync()

// Switch traffic atomically
let loadBalancer = TrafficManager()
try await loadBalancer.switchToGreen(
    validation: .comprehensive,
    rollbackTimeout: .minutes(30)
)
```

### Pattern 3: Partition-by-Partition Migration

```swift
// Migrate one partition at a time
let partitions = try await analyzedPartitions(for: existingData)

for partition in partitions {
    // Create partition in distributed system
    let distributedPartition = try await system.createPartition(
        id: partition.id,
        bounds: partition.bounds,
        node: partition.optimalNode
    )
    
    // Migrate data
    let migrator = PartitionMigrator(batchSize: 5000)
    try await migrator.migrate(
        from: existingStore,
        to: distributedPartition,
        filter: partition.dataFilter
    )
    
    // Validate
    let validator = MigrationValidator()
    let validation = try await validator.validate(
        source: existingStore,
        target: distributedPartition,
        sampleRate: 0.1
    )
    
    guard validation.isSuccessful else {
        throw MigrationError.validationFailed(partition.id, validation.errors)
    }
}
```

## Performance Considerations

### 1. Network Overhead

The distributed system introduces network communication:

```swift
// Optimize batch sizes for network efficiency
let optimalBatchSize = NetworkOptimizer.calculateBatchSize(
    vectorDimensions: 512,
    networkBandwidth: .gigabit,
    latency: .milliseconds(1)
)

// Use compression for large transfers
system.configure(.compression(.zstd(level: 3)))
```

### 2. Consistency vs Performance Trade-offs

| Consistency Level | Write Performance | Read Performance | Use Case |
|------------------|------------------|-----------------|-----------|
| `.one` | Fastest | Fastest | Cache, logs |
| `.quorum` | Moderate | Moderate | General use |
| `.all` | Slowest | Consistent | Critical data |

### 3. Metal Compute Distribution

```swift
// Configure Metal compute affinity
system.configureMetalCompute(
    .preferLocal(threshold: 1000),  // Use local GPU for <1000 vectors
    .distributedAbove(1000)         // Distribute larger computations
)
```

### 4. Caching Strategy

```swift
// Distributed cache configuration
let cacheConfig = DistributedCacheConfig(
    localCacheSize: 100_000_000,      // 100MB per node
    coordinationStrategy: .consistent, // Consistent hashing
    evictionPolicy: .lru,
    ttl: .minutes(30)
)
```

## Backwards Compatibility

### Using Compatibility Mode

```swift
// Enable compatibility mode for existing code
let system = try await DistributedVectorSystem.builder()
    .withCompatibilityMode(.v1)
    .build()

// Old API calls are automatically translated
let store = system.v1CompatibleStore()
```

### Feature Parity Matrix

| Feature | Single-Node | Distributed | Notes |
|---------|------------|-------------|-------|
| HNSW Index | ✅ | ✅ | Partitioned HNSW |
| IVF Index | ✅ | ✅ | Natural fit for distribution |
| Learned Index | ✅ | ⚠️ | Requires coordination |
| Product Quantization | ✅ | ✅ | Per-partition codebooks |
| Neural Clustering | ✅ | ✅ | Distributed training |
| Metal Acceleration | ✅ | ✅ | Node-local + distributed |
| Streaming Operations | ✅ | ✅ | Enhanced with partitioning |

## Troubleshooting

### Common Issues

#### 1. Uneven Data Distribution

```swift
// Check partition balance
let report = await system.getBalanceReport()
if report.imbalanceRatio > 0.2 {
    // Rebalance if >20% imbalance
    try await system.rebalance(
        strategy: .minimizeDataMovement,
        maxConcurrentMoves: 5
    )
}
```

#### 2. Consistency Violations

```swift
// Enable consistency checking
let monitor = ConsistencyMonitor(system: system)
monitor.onViolation { violation in
    logger.error("Consistency violation: \(violation)")
    // Take corrective action
}
```

#### 3. Performance Degradation

```swift
// Compare distributed vs single-node performance
let benchmark = MigrationBenchmark(
    singleNode: legacyStore,
    distributed: system
)

let comparison = try await benchmark.run(
    operations: [.search, .insert, .update],
    duration: .minutes(10)
)

print("Performance comparison:")
print("- Search: \(comparison.searchSpeedup)x")
print("- Insert: \(comparison.insertSpeedup)x")
print("- Update: \(comparison.updateSpeedup)x")
```

### Migration Rollback

If issues arise, use the rollback functionality:

```swift
// Create rollback point before migration
let rollbackPoint = try await MigrationRollback.createCheckpoint(
    system: existingStore,
    metadata: ["version": "pre-migration", "date": Date()]
)

// If migration fails
do {
    try await performMigration()
} catch {
    logger.error("Migration failed: \(error)")
    try await rollbackPoint.restore()
}
```

## Next Steps

1. **Test in Development**: Start with a development environment
2. **Benchmark**: Compare performance characteristics
3. **Plan Capacity**: Determine optimal node count and resources
4. **Monitor**: Set up comprehensive monitoring before production migration
5. **Iterate**: Use gradual migration for production systems

For additional support, see:
- [Distributed System Architecture](./DISTRIBUTED_ARCHITECTURE.md)
- [Performance Tuning Guide](./PERFORMANCE_TUNING.md)
- [Operations Guide](./OPERATIONS_GUIDE.md)
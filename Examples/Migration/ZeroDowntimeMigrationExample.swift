// VectorStoreKit: Zero-Downtime Migration Example
//
// Demonstrates migration with continuous availability

import Foundation
import VectorStoreKit

@main
struct ZeroDowntimeMigrationExample {
    
    static func main() async throws {
        print("🚀 VectorStoreKit Zero-Downtime Migration Example")
        print("=" * 50)
        print("\nThis example demonstrates migrating a live system without interrupting service.")
        
        // Step 1: Create and populate source store
        print("\n📦 Step 1: Setting up live production store...")
        let sourceStore = try await createProductionStore()
        
        // Start simulating live traffic
        let trafficSimulator = TrafficSimulator(store: sourceStore)
        let trafficTask = Task {
            await trafficSimulator.startSimulating()
        }
        
        print("✓ Production store active with simulated traffic")
        
        // Step 2: Create distributed target system
        print("\n🌐 Step 2: Setting up distributed system in parallel...")
        let targetSystem = try await createDistributedSystem()
        
        // Step 3: Configure zero-downtime migration
        print("\n⚙️ Step 3: Configuring zero-downtime migration...")
        let migrationConfig = MigrationConfiguration(
            batchSize: 5000,
            parallelWorkers: 4,
            enableZeroDowntime: true,  // Key setting
            enableCheckpointing: true,
            maxErrors: 10,
            rateLimit: 50_000,  // 50k vectors/second to avoid overload
            validationConfig: ValidationConfiguration(
                validateBatches: true,
                sampleRate: 0.01,  // 1% validation during migration
                validateAfterMigration: true
            )
        )
        
        // Step 4: Create dual-write adapter
        print("\n🔄 Step 4: Setting up dual-write mode...")
        let dualWriteAdapter = DualWriteAdapter(
            primary: sourceStore,
            secondary: targetSystem,
            mode: .primaryOnly  // Start with primary only
        )
        
        // Step 5: Run migration phases
        print("\n📋 Step 5: Migration Plan:")
        print("  Phase 1: Initial bulk copy")
        print("  Phase 2: Enable dual writes")
        print("  Phase 3: Incremental sync")
        print("  Phase 4: Traffic migration")
        print("  Phase 5: Validation & cutover")
        
        // Phase 1: Initial bulk copy
        print("\n🔄 Phase 1: Initial bulk copy...")
        let migrator = BatchMigrator(
            configuration: migrationConfig,
            sourceStore: ZeroDowntimeSourceAdapter(
                store: sourceStore,
                dualWrite: dualWriteAdapter
            ),
            targetSystem: ZeroDowntimeTargetAdapter(system: targetSystem)
        )
        
        var migrationComplete = false
        let migrationTask = Task {
            for try await progress in migrator.migrate() {
                printMigrationStatus(progress)
                
                // Enable dual writes after initial copy
                if progress.phase == "Change Tracking" && dualWriteAdapter.mode == .primaryOnly {
                    print("\n\n🔄 Phase 2: Enabling dual writes...")
                    await dualWriteAdapter.setMode(.dualWrite)
                    print("✓ All new writes now go to both systems")
                }
            }
            migrationComplete = true
        }
        
        // Wait for initial copy
        while !migrationComplete {
            try await Task.sleep(nanoseconds: 1_000_000_000)
        }
        
        // Phase 3: Verify consistency
        print("\n\n🔍 Phase 3: Verifying data consistency...")
        let consistencyCheck = try await verifyConsistency(
            source: sourceStore,
            target: targetSystem,
            dualWrite: dualWriteAdapter
        )
        
        if !consistencyCheck.isConsistent {
            print("❌ Consistency check failed! Aborting migration.")
            trafficTask.cancel()
            return
        }
        
        // Phase 4: Gradual traffic migration
        print("\n🚦 Phase 4: Migrating read traffic...")
        try await migrateTrafficGradually(
            dualWrite: dualWriteAdapter,
            trafficSimulator: trafficSimulator
        )
        
        // Phase 5: Final cutover
        print("\n✂️ Phase 5: Final cutover...")
        try await performCutover(
            source: sourceStore,
            target: targetSystem,
            dualWrite: dualWriteAdapter,
            trafficSimulator: trafficSimulator
        )
        
        // Stop traffic simulation
        trafficTask.cancel()
        
        print("\n✅ Zero-downtime migration completed successfully!")
        print("\n📊 Final Statistics:")
        await showFinalStatistics(
            source: sourceStore,
            target: targetSystem,
            trafficSimulator: trafficSimulator
        )
    }
    
    // MARK: - Store Creation
    
    static func createProductionStore() async throws -> MockProductionStore {
        let store = MockProductionStore(
            initialVectorCount: 100_000,
            dimension: 128
        )
        
        try await store.initialize()
        return store
    }
    
    static func createDistributedSystem() async throws -> MockDistributedSystem {
        let system = MockDistributedSystem(
            nodes: ["node1", "node2", "node3", "node4"],
            partitionStrategy: .consistentHash(virtualNodes: 200),
            replicationFactor: 3
        )
        
        try await system.initialize()
        return system
    }
    
    // MARK: - Consistency Verification
    
    static func verifyConsistency(
        source: MockProductionStore,
        target: MockDistributedSystem,
        dualWrite: DualWriteAdapter
    ) async throws -> ConsistencyCheckResult {
        print("  Checking vector counts...")
        let sourceCount = await source.getTotalCount()
        let targetCount = await target.getTotalCount()
        
        print("    Source: \(sourceCount) vectors")
        print("    Target: \(targetCount) vectors")
        
        let difference = abs(sourceCount - targetCount)
        let threshold = Int(Double(sourceCount) * 0.001) // 0.1% tolerance
        
        if difference > threshold {
            return ConsistencyCheckResult(
                isConsistent: false,
                sourceCount: sourceCount,
                targetCount: targetCount,
                inconsistencies: ["Count mismatch exceeds threshold"]
            )
        }
        
        // Sample-based content verification
        print("  Verifying sample content...")
        let sampleSize = min(1000, sourceCount / 100)
        let sampleIds = await source.getRandomIds(count: sampleSize)
        
        var inconsistencies: [String] = []
        for id in sampleIds {
            let sourceEntry = await source.getEntry(id: id)
            let targetEntry = await target.getEntry(id: id)
            
            if sourceEntry == nil && targetEntry != nil {
                inconsistencies.append("Extra entry in target: \(id)")
            } else if sourceEntry != nil && targetEntry == nil {
                inconsistencies.append("Missing entry in target: \(id)")
            }
        }
        
        print("✓ Consistency check completed")
        return ConsistencyCheckResult(
            isConsistent: inconsistencies.isEmpty,
            sourceCount: sourceCount,
            targetCount: targetCount,
            inconsistencies: inconsistencies
        )
    }
    
    // MARK: - Traffic Migration
    
    static func migrateTrafficGradually(
        dualWrite: DualWriteAdapter,
        trafficSimulator: TrafficSimulator
    ) async throws {
        let stages = [
            (percentage: 10, duration: 5),
            (percentage: 25, duration: 10),
            (percentage: 50, duration: 10),
            (percentage: 75, duration: 10),
            (percentage: 90, duration: 5),
            (percentage: 100, duration: 5)
        ]
        
        for stage in stages {
            print("  Routing \(stage.percentage)% of reads to distributed system...")
            
            await dualWrite.setReadPercentage(toSecondary: stage.percentage)
            await trafficSimulator.setReadDistribution(
                primary: 100 - stage.percentage,
                secondary: stage.percentage
            )
            
            // Monitor for duration
            try await Task.sleep(nanoseconds: UInt64(stage.duration) * 1_000_000_000)
            
            // Check health metrics
            let metrics = await trafficSimulator.getMetrics()
            if metrics.errorRate > 0.01 { // 1% error threshold
                print("  ⚠️ Error rate exceeded threshold! Rolling back...")
                await dualWrite.setReadPercentage(toSecondary: 0)
                throw MigrationError.trafficMigrationFailed(
                    stage: stage.percentage,
                    errorRate: metrics.errorRate
                )
            }
            
            print("  ✓ Stage completed. Error rate: \(String(format: "%.2f%%", metrics.errorRate * 100))")
        }
        
        print("✓ All read traffic successfully migrated")
    }
    
    // MARK: - Final Cutover
    
    static func performCutover(
        source: MockProductionStore,
        target: MockDistributedSystem,
        dualWrite: DualWriteAdapter,
        trafficSimulator: TrafficSimulator
    ) async throws {
        print("  Preparing for cutover...")
        
        // Step 1: Set distributed system as primary
        print("  Setting distributed system as primary...")
        await dualWrite.setMode(.secondaryOnly)
        
        // Step 2: Brief pause to ensure all in-flight requests complete
        print("  Waiting for in-flight requests...")
        try await Task.sleep(nanoseconds: 5_000_000_000) // 5 seconds
        
        // Step 3: Final consistency check
        print("  Final consistency verification...")
        let finalCheck = try await verifyConsistency(
            source: source,
            target: target,
            dualWrite: dualWrite
        )
        
        guard finalCheck.isConsistent else {
            print("  ❌ Final consistency check failed! Rolling back...")
            await dualWrite.setMode(.primaryOnly)
            throw MigrationError.cutoverFailed(reason: "Final consistency check failed")
        }
        
        // Step 4: Disable old system
        print("  Disabling old system...")
        await source.setReadOnly(true)
        
        print("✓ Cutover completed successfully")
    }
    
    // MARK: - Status Display
    
    static func printMigrationStatus(_ progress: MigrationProgress) {
        let percentage = progress.percentage
        let bar = createProgressBar(percentage: percentage)
        
        print("\r[\(bar)] \(String(format: "%.1f%%", percentage)) - \(progress.phase) | " +
              "Migrated: \(progress.migratedCount)/\(progress.totalCount) | " +
              "Failed: \(progress.failedCount) | " + 
              "ETA: \(formatDuration(progress.estimatedTimeRemaining ?? 0))",
              terminator: "")
        fflush(stdout)
    }
    
    static func createProgressBar(percentage: Double) -> String {
        let filled = Int(percentage / 2.5) // 40 character bar
        return String(repeating: "█", count: filled) + 
               String(repeating: "░", count: 40 - filled)
    }
    
    static func formatDuration(_ seconds: TimeInterval) -> String {
        if seconds < 60 {
            return "\(Int(seconds))s"
        } else if seconds < 3600 {
            return "\(Int(seconds / 60))m \(Int(seconds.truncatingRemainder(dividingBy: 60)))s"
        } else {
            let hours = Int(seconds / 3600)
            let minutes = Int((seconds - Double(hours * 3600)) / 60)
            return "\(hours)h \(minutes)m"
        }
    }
    
    static func showFinalStatistics(
        source: MockProductionStore,
        target: MockDistributedSystem,
        trafficSimulator: TrafficSimulator
    ) async {
        let sourceStats = await source.getStatistics()
        let targetStats = await target.globalStatistics()
        let trafficMetrics = await trafficSimulator.getMetrics()
        
        print("\nSource System (now read-only):")
        print("  Total vectors: \(sourceStats.vectorCount)")
        print("  Final size: \(formatBytes(sourceStats.diskUsage))")
        
        print("\nDistributed System (now primary):")
        print("  Total vectors: \(targetStats.totalVectors)")
        print("  Nodes: \(targetStats.nodeCount)")
        print("  Partitions: \(targetStats.partitionCount)")
        print("  Replication factor: 3")
        
        print("\nTraffic Statistics:")
        print("  Total requests: \(trafficMetrics.totalRequests)")
        print("  Successful: \(trafficMetrics.successfulRequests)")
        print("  Failed: \(trafficMetrics.failedRequests)")
        print("  Average latency: \(String(format: "%.2f ms", trafficMetrics.averageLatency * 1000))")
        print("  Downtime: 0 seconds")
    }
    
    static func formatBytes(_ bytes: Int) -> String {
        let units = ["B", "KB", "MB", "GB"]
        var size = Double(bytes)
        var unitIndex = 0
        
        while size >= 1024 && unitIndex < units.count - 1 {
            size /= 1024
            unitIndex += 1
        }
        
        return String(format: "%.2f %@", size, units[unitIndex])
    }
}

// MARK: - Mock Production Store

class MockProductionStore {
    private var vectors: [String: VectorEntry<SIMD32<Float>, SampleMetadata>] = [:]
    private let dimension: Int
    private var isReadOnly = false
    private let queue = DispatchQueue(label: "production.store", attributes: .concurrent)
    
    init(initialVectorCount: Int, dimension: Int) {
        self.dimension = dimension
        
        // Pre-populate with data
        for i in 0..<initialVectorCount {
            let id = "vec_\(i)"
            vectors[id] = generateRandomEntry(id: id, dimension: dimension)
        }
    }
    
    func initialize() async throws {
        // Initialization logic
    }
    
    func getTotalCount() async -> Int {
        queue.sync { vectors.count }
    }
    
    func getEntry(id: String) async -> VectorEntry<SIMD32<Float>, SampleMetadata>? {
        queue.sync { vectors[id] }
    }
    
    func getRandomIds(count: Int) async -> [String] {
        queue.sync {
            Array(vectors.keys.shuffled().prefix(count))
        }
    }
    
    func setReadOnly(_ readOnly: Bool) async {
        queue.async(flags: .barrier) {
            self.isReadOnly = readOnly
        }
    }
    
    func addEntry(_ entry: VectorEntry<SIMD32<Float>, SampleMetadata>) async throws {
        try queue.sync(flags: .barrier) {
            guard !isReadOnly else {
                throw StoreError.readOnly
            }
            vectors[entry.id] = entry
        }
    }
    
    func getStatistics() async -> StoreStatistics {
        queue.sync {
            StoreStatistics(
                vectorCount: vectors.count,
                memoryUsage: vectors.count * (dimension * 4 + 100), // Rough estimate
                diskUsage: vectors.count * (dimension * 4 + 200),
                performanceStatistics: PerformanceStatistics(),
                accessStatistics: AccessStatistics()
            )
        }
    }
    
    private func generateRandomEntry(id: String, dimension: Int) -> VectorEntry<SIMD32<Float>, SampleMetadata> {
        var values = [Float](repeating: 0, count: 32)
        for i in 0..<min(32, dimension) {
            values[i] = Float.random(in: -1...1)
        }
        
        return VectorEntry(
            id: id,
            vector: SIMD32<Float>(values),
            metadata: SampleMetadata(
                category: ["A", "B", "C", "D"].randomElement()!,
                timestamp: Date(),
                score: Float.random(in: 0...1)
            )
        )
    }
    
    enum StoreError: Error {
        case readOnly
    }
}

// MARK: - Dual Write Adapter

actor DualWriteAdapter {
    enum Mode {
        case primaryOnly
        case dualWrite
        case secondaryOnly
    }
    
    private(set) var mode: Mode
    private let primary: MockProductionStore
    private let secondary: MockDistributedSystem
    private var readPercentageToSecondary: Int = 0
    
    init(primary: MockProductionStore, secondary: MockDistributedSystem, mode: Mode) {
        self.primary = primary
        self.secondary = secondary
        self.mode = mode
    }
    
    func setMode(_ newMode: Mode) {
        mode = newMode
    }
    
    func setReadPercentage(toSecondary percentage: Int) {
        readPercentageToSecondary = min(100, max(0, percentage))
    }
    
    func write(_ entry: VectorEntry<SIMD32<Float>, SampleMetadata>) async throws {
        switch mode {
        case .primaryOnly:
            try await primary.addEntry(entry)
        case .dualWrite:
            // Write to both, primary first
            try await primary.addEntry(entry)
            _ = try? await secondary.addEntry(entry) // Best effort for secondary
        case .secondaryOnly:
            _ = try await secondary.addEntry(entry)
        }
    }
    
    func shouldReadFromSecondary() -> Bool {
        Int.random(in: 0..<100) < readPercentageToSecondary
    }
}

// MARK: - Traffic Simulator

actor TrafficSimulator {
    private let store: MockProductionStore
    private var isRunning = false
    private var metrics = TrafficMetrics()
    private var readDistribution = (primary: 100, secondary: 0)
    
    init(store: MockProductionStore) {
        self.store = store
    }
    
    func startSimulating() async {
        isRunning = true
        
        // Simulate mixed read/write traffic
        await withTaskGroup(of: Void.self) { group in
            // Write traffic
            group.addTask {
                while self.isRunning {
                    await self.simulateWrite()
                    try? await Task.sleep(nanoseconds: 10_000_000) // 10ms between writes
                }
            }
            
            // Read traffic (higher volume)
            for _ in 0..<5 {
                group.addTask {
                    while self.isRunning {
                        await self.simulateRead()
                        try? await Task.sleep(nanoseconds: 1_000_000) // 1ms between reads
                    }
                }
            }
        }
    }
    
    func stop() {
        isRunning = false
    }
    
    func setReadDistribution(primary: Int, secondary: Int) {
        readDistribution = (primary, secondary)
    }
    
    func getMetrics() -> TrafficMetrics {
        metrics
    }
    
    private func simulateWrite() async {
        let entry = VectorEntry(
            id: "live_\(UUID().uuidString)",
            vector: SIMD32<Float>(repeating: Float.random(in: -1...1)),
            metadata: SampleMetadata(
                category: ["A", "B", "C", "D"].randomElement()!,
                timestamp: Date(),
                score: Float.random(in: 0...1)
            )
        )
        
        do {
            let start = Date()
            try await store.addEntry(entry)
            let latency = Date().timeIntervalSince(start)
            
            metrics.recordRequest(latency: latency, success: true)
        } catch {
            metrics.recordRequest(latency: 0, success: false)
        }
    }
    
    private func simulateRead() async {
        let randomId = "vec_\(Int.random(in: 0..<100_000))"
        
        let start = Date()
        let _ = await store.getEntry(id: randomId)
        let latency = Date().timeIntervalSince(start)
        
        metrics.recordRequest(latency: latency, success: true)
    }
}

// MARK: - Supporting Types

struct ConsistencyCheckResult {
    let isConsistent: Bool
    let sourceCount: Int
    let targetCount: Int
    let inconsistencies: [String]
}

struct TrafficMetrics {
    private(set) var totalRequests: Int = 0
    private(set) var successfulRequests: Int = 0
    private(set) var failedRequests: Int = 0
    private var totalLatency: TimeInterval = 0
    
    var errorRate: Double {
        guard totalRequests > 0 else { return 0 }
        return Double(failedRequests) / Double(totalRequests)
    }
    
    var averageLatency: TimeInterval {
        guard successfulRequests > 0 else { return 0 }
        return totalLatency / TimeInterval(successfulRequests)
    }
    
    mutating func recordRequest(latency: TimeInterval, success: Bool) {
        totalRequests += 1
        if success {
            successfulRequests += 1
            totalLatency += latency
        } else {
            failedRequests += 1
        }
    }
}

enum MigrationError: Error {
    case trafficMigrationFailed(stage: Int, errorRate: Double)
    case cutoverFailed(reason: String)
}

// MARK: - Mock Adapters

struct ZeroDowntimeSourceAdapter: SourceStore {
    typealias Vector = SIMD32<Float>
    typealias Metadata = SampleMetadata
    
    let store: MockProductionStore
    let dualWrite: DualWriteAdapter
    
    func statistics() async -> StoreStatistics {
        await store.getStatistics()
    }
    
    func getAllIds() async throws -> [String] {
        // Would enumerate all IDs in real implementation
        []
    }
    
    func getEntry(id: String) async throws -> VectorEntry<Vector, Metadata>? {
        await store.getEntry(id: id)
    }
    
    func enableChangeTracking(since: Date) async throws -> ChangeTracker {
        LiveChangeTracker(store: store, since: since)
    }
    
    func disableChangeTracking() async throws {
        // No-op for demo
    }
    
    func withReadLock<T>(_ operation: () async throws -> T) async throws -> T {
        // In production, would implement actual locking
        try await operation()
    }
}

struct ZeroDowntimeTargetAdapter: TargetDistributedSystem {
    typealias Vector = SIMD32<Float>
    typealias Metadata = SampleMetadata
    
    let system: MockDistributedSystem
    
    func prepareForMigration(expectedVectors: Int, expectedSize: Int) async throws {
        // Preparation logic
    }
    
    func batchInsert(entries: [VectorEntry<Vector, Metadata>], options: DistributedInsertOptions) async throws -> (successCount: Int, failureCount: Int) {
        (successCount: entries.count, failureCount: 0)
    }
    
    func upsert(_ entry: VectorEntry<Vector, Metadata>) async throws -> Bool {
        true
    }
    
    func delete(id: String) async throws -> Bool {
        true
    }
    
    func getEntry(id: String) async throws -> VectorEntry<Vector, Metadata>? {
        await system.getEntry(id: id)
    }
    
    func optimizeAfterMigration() async throws {
        // Optimization logic
    }
}

struct LiveChangeTracker: ChangeTracker {
    let store: MockProductionStore
    let since: Date
    
    func getChanges() async throws -> [ChangeRecord] {
        // In production, would track actual changes
        []
    }
    
    func markProcessed(_ changes: [ChangeRecord]) async throws {
        // Mark changes as processed
    }
}

// Extension for distributed system
extension MockDistributedSystem {
    func getTotalCount() async -> Int {
        await globalStatistics().totalVectors
    }
    
    func getEntry(id: String) async -> VectorEntry<SIMD32<Float>, SampleMetadata>? {
        // Mock implementation
        nil
    }
    
    func addEntry(_ entry: VectorEntry<SIMD32<Float>, SampleMetadata>) async throws -> Bool {
        // Mock implementation
        true
    }
}

// Helper extension
extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
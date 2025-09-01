// VectorStoreKit: Production Deployment Example
//
// Demonstrates a complete production setup with distributed system, monitoring, and best practices

import Foundation
import VectorStoreKit
import os

@main
struct ProductionDeploymentExample {
    
    static func main() async throws {
        print("🚀 VectorStoreKit Production Deployment Example")
        print("=" * 50)
        
        // 1. Initialize distributed system
        let system = try await setupDistributedSystem()
        
        // 2. Configure monitoring
        let monitoring = try await setupMonitoring()
        
        // 3. Create production vector stores
        let stores = try await createProductionStores(system: system, monitoring: monitoring)
        
        // 4. Demonstrate production operations
        print("\n🔧 Production Operations:")
        
        // Data ingestion pipeline
        await demonstrateDataIngestion(stores: stores, monitoring: monitoring)
        
        // High-throughput search
        await demonstrateHighThroughputSearch(stores: stores, monitoring: monitoring)
        
        // Automated maintenance
        await demonstrateAutomatedMaintenance(system: system, monitoring: monitoring)
        
        // Failure recovery
        await demonstrateFailureRecovery(system: system, monitoring: monitoring)
        
        // Performance optimization
        await demonstratePerformanceOptimization(stores: stores, monitoring: monitoring)
        
        // 5. Export metrics and reports
        await exportProductionMetrics(monitoring: monitoring)
        
        print("\n✅ Production deployment example completed!")
    }
    
    // MARK: - System Setup
    
    static func setupDistributedSystem() async throws -> DistributedVectorSystem {
        print("\n🌐 Setting up distributed system...")
        
        // Initialize components
        let partitionManager = PartitionManager(
            configuration: PartitionManager.Configuration(
                leaseDuration: 300,
                leaseRenewalInterval: 60,
                healthCheckInterval: 30,
                maxConsecutiveFailures: 3,
                autoRenewLeases: true,
                enableHealthMonitoring: true
            )
        )
        
        let consistentHash = ConsistentHashRing(
            configuration: ConsistentHashRing.Configuration(
                virtualNodesPerNode: 200,
                replicationFactor: 3,
                minimumNodes: 3
            )
        )
        
        let assigner = PartitionAssigner(
            strategy: LocationAwareStrategy(
                preferredLocations: [
                    NodeLocation(region: "us-west", zone: "us-west-2a"),
                    NodeLocation(region: "us-west", zone: "us-west-2b")
                ],
                fallbackStrategy: LeastLoadedStrategy()
            )
        )
        
        let coordinator = RebalancingCoordinator(
            partitionManager: partitionManager,
            partitionAssigner: assigner,
            configuration: RebalancingCoordinator.Configuration(
                maxConcurrentMigrations: 5,
                maxPartitionsPerOperation: 20,
                minImprovementThreshold: 0.15,
                cooldownPeriod: 600,
                dryRunMode: false
            )
        )
        
        // Register production nodes
        let nodes = [
            ("prod-node-1", "us-west", "us-west-2a", 100),
            ("prod-node-2", "us-west", "us-west-2b", 100),
            ("prod-node-3", "us-west", "us-west-2c", 100),
            ("prod-node-4", "us-east", "us-east-1a", 50),
            ("prod-node-5", "eu-west", "eu-west-1a", 50)
        ]
        
        for (nodeId, region, zone, capacity) in nodes {
            let nodeInfo = NodeInfo(
                nodeId: nodeId,
                capacity: NodeCapacity(
                    maxPartitions: capacity,
                    maxStorageBytes: 1_099_511_627_776, // 1TB
                    maxMemoryBytes: 68_719_476_736, // 64GB
                    maxConnections: 10000
                ),
                currentLoad: NodeLoad(),
                health: NodeHealth(status: .healthy),
                location: NodeLocation(region: region, zone: zone),
                tags: Set(["production", region])
            )
            
            await assigner.registerNode(nodeInfo)
            try await consistentHash.addNode(Node(id: nodeId))
            
            print("  ✅ Registered node: \(nodeId) in \(zone)")
        }
        
        // Create partitions
        let partitioner = HashPartitioner(configuration: HashPartitioner.Configuration(hashFunction: .xxhash))
        let partitions = partitioner.createPartitions(
            count: 100,
            nodes: nodes.map { $0.0 }
        )
        
        for partition in partitions {
            let assignedNode = try await assigner.assignPartition(partition)
            _ = try await partitionManager.createPartition(
                id: partition.id,
                ownerNode: assignedNode,
                bounds: partition.bounds,
                replicas: []
            )
        }
        
        print("  ✅ Created \(partitions.count) partitions")
        
        return DistributedVectorSystem(
            partitionManager: partitionManager,
            consistentHash: consistentHash,
            assigner: assigner,
            coordinator: coordinator,
            partitions: partitions
        )
    }
    
    // MARK: - Monitoring Setup
    
    static func setupMonitoring() async throws -> MonitoringIntegration {
        print("\n📊 Setting up monitoring and observability...")
        
        let monitoring = MonitoringIntegration(
            metricsConfig: MetricsCollectorConfiguration.production(),
            tracingConfig: DistributedTracerConfiguration.production(),
            alertingConfig: AlertingSystemConfiguration.production(),
            dashboardConfig: ObservabilityDashboardConfiguration.production()
        )
        
        // Configure alert handlers
        await monitoring.alerting.addHandler(EmailAlertHandler())
        await monitoring.alerting.addHandler(SlackAlertHandler())
        await monitoring.alerting.addHandler(PagerDutyHandler())
        
        // Configure metric exporters
        await monitoring.dashboard.addExporter(
            format: .prometheus,
            handler: PrometheusExporter(endpoint: "http://prometheus:9090/metrics")
        )
        
        await monitoring.dashboard.addExporter(
            format: .json,
            handler: CloudWatchExporter(region: "us-west-2")
        )
        
        print("  ✅ Monitoring configured with alerts and exporters")
        
        return monitoring
    }
    
    // MARK: - Production Stores
    
    static func createProductionStores(
        system: DistributedVectorSystem,
        monitoring: MonitoringIntegration
    ) async throws -> ProductionStores {
        print("\n🏪 Creating production vector stores...")
        
        // High-performance store for real-time queries
        let realtimeStore = try await VectorStore<SIMD128<Float>, ProductMetadata>(
            configuration: StoreConfiguration(
                indexType: .hybrid(
                    HybridIndexConfiguration(
                        dimensions: 128,
                        hnswConfig: HNSWConfiguration(
                            dimensions: 128,
                            maxConnections: 64,
                            efConstruction: 500,
                            similarity: .cosine
                        ),
                        ivfConfig: IVFConfiguration(
                            dimensions: 128,
                            numberOfCentroids: 1024,
                            searchConfiguration: IVFSearchConfiguration(nProbe: 20)
                        ),
                        routingStrategy: .adaptive,
                        adaptiveThreshold: 0.9
                    )
                ),
                cacheType: .adaptive(
                    AdaptiveCacheConfiguration<SIMD128<Float>>(
                        hotCacheSize: 10000,
                        warmCacheSize: 50000,
                        coldCacheSize: 200000,
                        adaptationInterval: 300
                    )
                ),
                storageConfiguration: HierarchicalStorageConfiguration(
                    hotTier: HotTierConfiguration(maxSizeBytes: 10_737_418_240), // 10GB
                    warmTier: WarmTierConfiguration(maxSizeBytes: 107_374_182_400), // 100GB
                    coldTier: ColdTierConfiguration(compressionLevel: 6),
                    archiveTier: ArchiveTierConfiguration(compressionLevel: 9)
                ),
                metalConfiguration: MetalComputeConfiguration(
                    device: nil,
                    useSharedMemory: true,
                    enableProfiling: false
                ),
                persistenceURL: URL(fileURLWithPath: "/var/vectorstore/realtime")
            )
        )
        
        // Analytics store for batch processing
        let analyticsStore = try await VectorStore<SIMD256<Float>, AnalyticsMetadata>(
            configuration: StoreConfiguration(
                indexType: .ivf(
                    IVFConfiguration(
                        dimensions: 256,
                        numberOfCentroids: 4096,
                        trainingConfiguration: IVFTrainingConfiguration(
                            sampleSize: 100000,
                            maxIterations: 50,
                            convergenceThreshold: 0.0001
                        ),
                        searchConfiguration: IVFSearchConfiguration(
                            nProbe: 50,
                            useHeap: true,
                            rerank: true
                        ),
                        quantization: .product(
                            ProductQuantizationConfig(
                                subvectorCount: 32,
                                bitsPerSubvector: 8,
                                trainingIterations: 100
                            )
                        )
                    )
                ),
                cacheType: .lfu(
                    LFUCacheConfiguration<SIMD256<Float>>(
                        maxCapacity: 50000,
                        minFrequency: 2
                    )
                ),
                persistenceURL: URL(fileURLWithPath: "/var/vectorstore/analytics")
            )
        )
        
        print("  ✅ Created real-time and analytics stores")
        
        return ProductionStores(
            realtime: realtimeStore,
            analytics: analyticsStore,
            monitoring: monitoring
        )
    }
    
    // MARK: - Production Operations
    
    static func demonstrateDataIngestion(
        stores: ProductionStores,
        monitoring: MonitoringIntegration
    ) async {
        print("\n📥 Data Ingestion Pipeline")
        print("-" * 30)
        
        // Start monitoring
        let span = await monitoring.tracer.startSpan(
            name: "data_ingestion",
            attributes: ["batch_size": "10000"]
        )
        
        let startTime = Date()
        
        // Simulate streaming data ingestion
        let batchSize = 1000
        let totalBatches = 10
        
        for batchNum in 1...totalBatches {
            // Record batch metrics
            await monitoring.metrics.recordOperation(
                .custom(name: "ingestion_batch", value: Double(batchNum)),
                duration: 0,
                metadata: ["batch_size": "\(batchSize)"]
            )
            
            // Generate batch data
            var entries: [VectorEntry<SIMD128<Float>, ProductMetadata>] = []
            
            for i in 1...batchSize {
                let productId = "prod_\(batchNum)_\(i)"
                let vector = generateProductEmbedding(productId: productId)
                let metadata = ProductMetadata(
                    name: "Product \(productId)",
                    category: ["Electronics", "Clothing", "Books", "Home"].randomElement()!,
                    price: Double.random(in: 10...1000),
                    rating: Float.random(in: 3.0...5.0),
                    inStock: Bool.random()
                )
                
                entries.append(VectorEntry(
                    id: productId,
                    vector: vector,
                    metadata: metadata
                ))
            }
            
            // Ingest batch with monitoring
            do {
                let insertStart = Date()
                try await stores.realtime.addBatch(
                    entries,
                    options: InsertOptions(
                        deduplication: .update,
                        background: true
                    )
                )
                
                let insertDuration = Date().timeIntervalSince(insertStart)
                await monitoring.metrics.recordOperation(
                    .insert,
                    duration: insertDuration,
                    metadata: ["batch": "\(batchNum)", "size": "\(batchSize)"]
                )
                
                print("  Batch \(batchNum)/\(totalBatches) ingested (\(batchSize) items in \(String(format: "%.2f", insertDuration))s)")
                
            } catch {
                await monitoring.alerting.checkCondition(
                    value: 1,
                    rule: monitoring.alerting.ingestFailureRule
                )
                print("  ❌ Batch \(batchNum) failed: \(error)")
            }
            
            // Simulate real-time delay
            try? await Task.sleep(nanoseconds: 100_000_000) // 0.1s
        }
        
        let totalDuration = Date().timeIntervalSince(startTime)
        await span.end(attributes: [
            "total_items": "\(batchSize * totalBatches)",
            "duration_seconds": "\(totalDuration)"
        ])
        
        print("  ✅ Ingested \(batchSize * totalBatches) items in \(String(format: "%.2f", totalDuration))s")
        print("  Throughput: \(Int(Double(batchSize * totalBatches) / totalDuration)) items/second")
    }
    
    static func demonstrateHighThroughputSearch(
        stores: ProductionStores,
        monitoring: MonitoringIntegration
    ) async {
        print("\n🔍 High-Throughput Search")
        print("-" * 30)
        
        // Simulate concurrent search load
        let concurrentSearches = 100
        let searchesPerClient = 10
        
        let overallStart = Date()
        var totalSearches = 0
        var totalLatency: Double = 0
        
        await withTaskGroup(of: (Int, Double).self) { group in
            for clientId in 1...concurrentSearches {
                group.addTask {
                    var clientSearches = 0
                    var clientLatency: Double = 0
                    
                    for _ in 1...searchesPerClient {
                        let queryVector = generateProductEmbedding(productId: "query_\(clientId)")
                        
                        let searchStart = Date()
                        
                        do {
                            let results = try await stores.realtime.search(
                                query: queryVector,
                                k: 10,
                                strategy: .approximate(probes: 5), // Fast approximate search
                                filter: MetadataFilter<ProductMetadata> { metadata in
                                    metadata.inStock && metadata.rating >= 4.0
                                }
                            )
                            
                            let searchDuration = Date().timeIntervalSince(searchStart)
                            clientLatency += searchDuration
                            clientSearches += 1
                            
                            // Record metrics
                            await monitoring.metrics.recordQuery(
                                latency: searchDuration,
                                resultCount: results.count,
                                metadata: ["client": "\(clientId)", "strategy": "approximate"]
                            )
                            
                        } catch {
                            await monitoring.metrics.recordOperation(
                                .custom(name: "search_error", value: 1),
                                duration: 0,
                                metadata: ["error": "\(error)"]
                            )
                        }
                    }
                    
                    return (clientSearches, clientLatency)
                }
            }
            
            // Collect results
            for await (searches, latency) in group {
                totalSearches += searches
                totalLatency += latency
            }
        }
        
        let totalDuration = Date().timeIntervalSince(overallStart)
        let avgLatency = totalLatency / Double(totalSearches)
        let throughput = Double(totalSearches) / totalDuration
        
        print("  Total searches: \(totalSearches)")
        print("  Total duration: \(String(format: "%.2f", totalDuration))s")
        print("  Throughput: \(Int(throughput)) searches/second")
        print("  Average latency: \(String(format: "%.3f", avgLatency * 1000))ms")
        
        // Check SLA
        if avgLatency > 0.05 { // 50ms SLA
            await monitoring.alerting.checkCondition(
                value: avgLatency * 1000,
                rule: monitoring.alerting.highLatencyRule
            )
        }
    }
    
    static func demonstrateAutomatedMaintenance(
        system: DistributedVectorSystem,
        monitoring: MonitoringIntegration
    ) async {
        print("\n🔧 Automated Maintenance")
        print("-" * 30)
        
        // 1. Check system balance
        let balanceReport = await system.assigner.getBalanceReport()
        print("  System balance: \(String(format: "%.2f", 1.0 - balanceReport.imbalanceRatio)) (imbalance: \(String(format: "%.2f", balanceReport.imbalanceRatio)))")
        
        // 2. Trigger rebalancing if needed
        if let plan = try? await system.coordinator.analyzeAndPlan() {
            print("  🔄 Rebalancing needed: \(plan.moves.count) moves")
            
            let operationId = try? await system.coordinator.executeRebalancing(plan)
            if let id = operationId {
                print("  Started rebalancing operation: \(id)")
                
                // Monitor progress
                for _ in 1...3 {
                    try? await Task.sleep(nanoseconds: 2_000_000_000)
                    
                    if let status = await system.coordinator.getOperationStatus(id) {
                        print("  Progress: \(Int(status.progress * 100))%")
                    }
                }
            }
        } else {
            print("  ✅ System is balanced")
        }
        
        // 3. Health checks
        print("\n  Running health checks...")
        let partitions = await system.partitionManager.getAllPartitions()
        var healthyCount = 0
        var degradedCount = 0
        var unhealthyCount = 0
        
        for partition in partitions.prefix(10) { // Check first 10 for demo
            let health = partition.health.status
            switch health {
            case .healthy: healthyCount += 1
            case .degraded: degradedCount += 1
            case .unhealthy: unhealthyCount += 1
            case .unknown: break
            }
        }
        
        print("  Health summary: ✅ \(healthyCount) healthy, ⚠️ \(degradedCount) degraded, ❌ \(unhealthyCount) unhealthy")
        
        // 4. Lease renewal
        let needingRenewal = await system.partitionManager.getPartitionsNeedingLeaseRenewal(threshold: 120)
        if !needingRenewal.isEmpty {
            print("  🔄 Renewing \(needingRenewal.count) partition leases...")
            
            for partition in needingRenewal {
                _ = try? await system.partitionManager.acquireLease(
                    partitionId: partition.partitionId,
                    nodeId: partition.ownership.ownerNodeId
                )
            }
        }
    }
    
    static func demonstrateFailureRecovery(
        system: DistributedVectorSystem,
        monitoring: MonitoringIntegration
    ) async {
        print("\n🚨 Failure Recovery Simulation")
        print("-" * 30)
        
        // Simulate node failure
        let failedNode = "prod-node-3"
        print("  Simulating failure of node: \(failedNode)")
        
        // Update node health
        await system.assigner.registerNode(
            NodeInfo(
                nodeId: failedNode,
                capacity: NodeCapacity(maxPartitions: 100),
                currentLoad: NodeLoad(partitionCount: 20),
                health: NodeHealth(
                    status: .unhealthy,
                    lastHeartbeat: Date().addingTimeInterval(-120),
                    consecutiveFailures: 5
                ),
                location: NodeLocation(region: "us-west", zone: "us-west-2c")
            )
        )
        
        // Trigger alert
        await monitoring.alerting.checkCondition(
            value: 1,
            rule: monitoring.alerting.nodeFailureRule
        )
        
        // Find affected partitions
        let affectedPartitions = await system.partitionManager.getPartitionsForNode(failedNode)
        print("  Affected partitions: \(affectedPartitions.count)")
        
        // Reassign partitions
        print("  Reassigning partitions to healthy nodes...")
        
        for partition in affectedPartitions {
            // Find new node
            if let newNode = await system.assigner.getOptimalNode(
                for: Partition(
                    id: partition.partitionId,
                    nodeId: partition.ownership.ownerNodeId,
                    bounds: .hash(min: 0, max: 1000)
                )
            ) {
                // Transfer ownership
                _ = try? await system.partitionManager.acquireLease(
                    partitionId: partition.partitionId,
                    nodeId: newNode,
                    force: true
                )
                
                print("    Partition \(partition.partitionId) -> \(newNode)")
            }
        }
        
        print("  ✅ Recovery completed")
    }
    
    static func demonstratePerformanceOptimization(
        stores: ProductionStores,
        monitoring: MonitoringIntegration
    ) async {
        print("\n⚡ Performance Optimization")
        print("-" * 30)
        
        // 1. Analyze query patterns
        let queryStats = await monitoring.metrics.getQueryStatistics(
            timeRange: TimeRange(
                start: Date().addingTimeInterval(-3600),
                end: Date()
            )
        )
        
        print("  Query statistics (last hour):")
        print("    Total queries: \(queryStats.totalQueries)")
        print("    Avg latency: \(String(format: "%.2f", queryStats.averageLatency * 1000))ms")
        print("    P99 latency: \(String(format: "%.2f", queryStats.p99Latency * 1000))ms")
        
        // 2. Optimize based on patterns
        if queryStats.averageLatency > 0.02 { // 20ms threshold
            print("\n  🔧 Applying optimizations...")
            
            // Increase cache size
            print("    - Adjusting cache parameters")
            
            // Optimize index
            print("    - Rebuilding index for better performance")
            try? await stores.realtime.optimize(.full)
            
            // Pre-warm cache with popular items
            print("    - Pre-warming cache with frequently accessed vectors")
        }
        
        // 3. GPU utilization
        let gpuStats = await monitoring.metrics.getResourceStatistics().gpuUsage
        print("\n  GPU utilization: \(Int(gpuStats.utilization))%")
        print("  GPU memory: \(gpuStats.memoryUsed / 1_048_576)MB / \(gpuStats.memoryTotal / 1_048_576)MB")
        
        if gpuStats.utilization < 50 {
            print("  💡 GPU underutilized - consider increasing batch sizes")
        }
    }
    
    // MARK: - Metrics Export
    
    static func exportProductionMetrics(monitoring: MonitoringIntegration) async {
        print("\n📈 Exporting Production Metrics")
        print("-" * 30)
        
        // Export to different formats
        let timeRange = TimeRange(
            start: Date().addingTimeInterval(-3600),
            end: Date()
        )
        
        // Prometheus format
        if let prometheusData = try? await monitoring.dashboard.export(
            format: .prometheus,
            timeRange: timeRange
        ) {
            print("  ✅ Exported Prometheus metrics (\(prometheusData.count) bytes)")
        }
        
        // JSON format for analysis
        if let jsonData = try? await monitoring.dashboard.export(
            format: .json,
            timeRange: timeRange
        ) {
            print("  ✅ Exported JSON metrics (\(jsonData.count) bytes)")
            
            // Parse and show summary
            if let metrics = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
               let summary = metrics["summary"] as? [String: Any] {
                print("\n  Summary:")
                print("    Total operations: \(summary["total_operations"] ?? 0)")
                print("    Success rate: \(summary["success_rate"] ?? "N/A")")
                print("    Avg response time: \(summary["avg_response_time"] ?? "N/A")")
            }
        }
        
        // Generate health report
        let healthScore = await monitoring.dashboard.calculateHealthScore()
        print("\n  System Health Score: \(Int(healthScore * 100))%")
        
        if healthScore < 0.9 {
            print("  ⚠️ Health score below threshold - investigation recommended")
        }
    }
    
    // MARK: - Helper Functions
    
    static func generateProductEmbedding(productId: String) -> SIMD128<Float> {
        var embedding = SIMD128<Float>(repeating: 0)
        let hash = productId.hashValue
        
        for i in 0..<128 {
            embedding[i] = Float((hash &+ i) % 100) / 100.0
        }
        
        // Normalize
        let magnitude = sqrt((0..<128).reduce(Float(0)) { $0 + embedding[$1] * embedding[$1] })
        if magnitude > 0 {
            embedding /= magnitude
        }
        
        return embedding
    }
}

// MARK: - Supporting Types

struct DistributedVectorSystem {
    let partitionManager: PartitionManager
    let consistentHash: ConsistentHashRing
    let assigner: PartitionAssigner
    let coordinator: RebalancingCoordinator
    let partitions: [Partition]
}

struct ProductionStores {
    let realtime: VectorStore<SIMD128<Float>, ProductMetadata>
    let analytics: VectorStore<SIMD256<Float>, AnalyticsMetadata>
    let monitoring: MonitoringIntegration
}

struct ProductMetadata: VectorMetadata {
    let name: String
    let category: String
    let price: Double
    let rating: Float
    let inStock: Bool
}

struct AnalyticsMetadata: VectorMetadata {
    let sessionId: String
    let userId: String
    let timestamp: Date
    let eventType: String
    let properties: [String: String]
}

// Mock alert handlers
struct EmailAlertHandler: AlertHandler {
    func handle(_ alert: Alert) async {
        print("📧 Email alert: \(alert.rule.name) - \(alert.message)")
    }
}

struct SlackAlertHandler: AlertHandler {
    func handle(_ alert: Alert) async {
        print("💬 Slack alert: \(alert.rule.name) - \(alert.message)")
    }
}

struct PagerDutyHandler: AlertHandler {
    func handle(_ alert: Alert) async {
        if alert.severity == .critical {
            print("🚨 PagerDuty alert: \(alert.rule.name) - \(alert.message)")
        }
    }
}

// Mock exporters
struct PrometheusExporter: ExportHandler {
    let endpoint: String
    
    func export(_ data: Data, metadata: ExportMetadata) async throws {
        print("📊 Exporting to Prometheus: \(endpoint)")
    }
}

struct CloudWatchExporter: ExportHandler {
    let region: String
    
    func export(_ data: Data, metadata: ExportMetadata) async throws {
        print("☁️ Exporting to CloudWatch: \(region)")
    }
}

// Helper extensions
extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}

// Mock monitoring integration extensions
extension MonitoringIntegration {
    var alerting: AlertingSystem {
        get async { alertingSystem }
    }
    
    var ingestFailureRule: AlertRule {
        AlertRule(
            id: "ingest_failure",
            name: "Ingestion Failure",
            condition: .threshold(metric: "ingest_errors", operator: .greaterThan, value: 0),
            severity: .high,
            cooldown: 300
        )
    }
    
    var highLatencyRule: AlertRule {
        AlertRule(
            id: "high_latency",
            name: "High Query Latency",
            condition: .threshold(metric: "query_latency_ms", operator: .greaterThan, value: 50),
            severity: .warning,
            cooldown: 60
        )
    }
    
    var nodeFailureRule: AlertRule {
        AlertRule(
            id: "node_failure",
            name: "Node Failure Detected",
            condition: .threshold(metric: "node_failures", operator: .greaterThan, value: 0),
            severity: .critical,
            cooldown: 0
        )
    }
}
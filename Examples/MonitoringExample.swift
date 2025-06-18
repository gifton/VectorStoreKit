// VectorStoreKit: Monitoring Example
//
// Demonstrates monitoring and observability features

import Foundation
import VectorStoreKit

@main
struct MonitoringExample {
    static func main() async throws {
        print("VectorStoreKit Monitoring Example")
        print("=================================\n")
        
        // Initialize monitoring system
        let monitoring = await MonitoringSystem(configuration: .development)
        
        // Set up console export handlers for visibility
        let consoleHandler = ConsoleExportHandler()
        _ = await monitoring.metricsCollector.addExportHandler(consoleHandler)
        _ = await monitoring.distributedTracer.addExportHandler(consoleHandler)
        _ = await monitoring.alertingSystem.addDeliveryHandler(consoleHandler)
        
        // Example 1: Monitor partition operations
        print("1. Monitoring Partition Operations")
        print("----------------------------------")
        
        try await monitoring.recordPartitionOperation(
            type: .create,
            partitionId: "partition-001",
            nodeId: "node-001"
        ) {
            // Simulate partition creation
            try await Task.sleep(nanoseconds: 100_000_000) // 100ms
            print("Created partition-001")
        }
        
        try await monitoring.recordPartitionOperation(
            type: .split,
            partitionId: "partition-001",
            nodeId: "node-001"
        ) {
            // Simulate partition split
            try await Task.sleep(nanoseconds: 200_000_000) // 200ms
            print("Split partition-001")
        }
        
        // Example 2: Monitor queries with tracing
        print("\n2. Monitoring Query Performance")
        print("--------------------------------")
        
        let results = try await monitoring.recordQuery(
            partitionId: "partition-001",
            queryType: .knn,
            vectorCount: 10000
        ) {
            // Simulate k-NN query
            try await Task.sleep(nanoseconds: 50_000_000) // 50ms
            return ["result1", "result2", "result3"]
        }
        print("Query returned \(results.count) results")
        
        // Example 3: Update node health
        print("\n3. Node Health Monitoring")
        print("-------------------------")
        
        await monitoring.updateNodeHealth(
            nodeId: "node-001",
            status: .healthy,
            partitionCounts: [
                .active: 10,
                .creating: 2,
                .migrating: 1
            ],
            activeConnections: 25
        )
        
        // Example 4: Resource utilization monitoring
        print("\n4. Resource Utilization")
        print("-----------------------")
        
        await monitoring.metricsCollector.recordResourceUtilization(
            ResourceMetrics(
                nodeId: "node-001",
                cpuUsage: 65.5,
                memoryUsed: 8_589_934_592, // 8GB
                memoryAvailable: 8_589_934_592, // 8GB
                diskReadBytes: 1_073_741_824, // 1GB
                diskWriteBytes: 536_870_912, // 512MB
                gpuMetrics: GPUMetrics(
                    utilization: 80.0,
                    memoryUsed: 4_294_967_296, // 4GB
                    memoryTotal: 8_589_934_592, // 8GB
                    temperature: 75.0,
                    powerUsage: 250.0
                )
            )
        )
        
        // Example 5: Custom alerts
        print("\n5. Custom Alert Rules")
        print("---------------------")
        
        let customRule = AlertRule(
            name: "High Vector Count",
            description: "Alert when partition has too many vectors",
            condition: .threshold(
                metric: "partition.vector.count",
                operator: .greaterThan,
                value: 1_000_000
            ),
            severity: .warning
        )
        
        let ruleId = await monitoring.addAlertRule(customRule)
        print("Added custom alert rule: \(customRule.name)")
        
        // Example 6: Distributed tracing
        print("\n6. Distributed Tracing")
        print("----------------------")
        
        if let context = await monitoring.createDistributedOperationContext(
            operationName: "distributed.search",
            attributes: ["search.type": "multi-partition"]
        ) {
            // Simulate cross-partition operation
            if let childSpan = await context.createChildSpan(
                operationName: "partition.search",
                attributes: ["partition.id": "partition-002"]
            ) {
                try await Task.sleep(nanoseconds: 30_000_000) // 30ms
                
                await monitoring.distributedTracer.endSpan(
                    context: childSpan,
                    status: .ok,
                    attributes: ["results.count": "5"]
                )
            }
            
            // Get propagation headers for cross-node call
            let headers = await context.getPropagationHeaders()
            print("Propagation headers: \(headers)")
        }
        
        // Example 7: Get dashboard view
        print("\n7. Dashboard View")
        print("-----------------")
        
        let dashboard = await monitoring.getDashboard()
        print("System Health Score: \(dashboard.healthScore)")
        print("CPU Usage: \(dashboard.systemMetrics.cpuUsage)%")
        print("Memory Usage: \(dashboard.systemMetrics.memoryUsage)%")
        print("Active Partitions: \(dashboard.partitionMetrics.activePartitions)")
        print("Query P99 Latency: \(dashboard.queryMetrics.p99Latency)ms")
        print("Active Alerts: \(dashboard.alertMetrics.activeCount)")
        
        // Example 8: Export metrics
        print("\n8. Metrics Export")
        print("-----------------")
        
        let jsonData = try await monitoring.exportMetrics(format: .json)
        print("Exported \(jsonData.count) bytes of metrics in JSON format")
        
        let prometheusData = try await monitoring.exportMetrics(format: .prometheus)
        if let prometheusString = String(data: prometheusData, encoding: .utf8) {
            print("\nPrometheus format sample:")
            print(prometheusString.prefix(200) + "...")
        }
        
        // Example 9: Historical queries
        print("\n9. Historical Data Query")
        print("------------------------")
        
        let historicalResult = await monitoring.observabilityDashboard.queryHistorical(
            metrics: ["resource.cpu.usage", "query.latency.p99"],
            startTime: Date().addingTimeInterval(-3600), // 1 hour ago
            endTime: Date(),
            aggregation: .average
        )
        print("Queried \(historicalResult.metrics.count) historical metrics")
        
        // Example 10: Alert management
        print("\n10. Alert Management")
        print("--------------------")
        
        let activeAlerts = await monitoring.getActiveAlerts()
        print("Active alerts: \(activeAlerts.count)")
        
        for alert in activeAlerts {
            print("- [\(alert.severity.rawValue)] \(alert.title)")
            
            // Acknowledge critical alerts
            if alert.severity == .critical {
                await monitoring.acknowledgeAlert(alert.id, by: "admin")
                print("  Acknowledged by admin")
            }
        }
        
        // Wait a bit to see some monitoring output
        print("\nMonitoring active... (press Ctrl+C to stop)")
        try await Task.sleep(nanoseconds: 5_000_000_000) // 5 seconds
    }
}

// Example custom metric exporter
struct CustomMetricsExporter: MetricExportHandler {
    func export(_ snapshot: MetricsSnapshot) async {
        // Custom export logic
        print("Custom export: \(snapshot.timeSeries.count) time series")
    }
}

// Example custom trace processor
struct CustomTraceProcessor: TraceExportHandler {
    func export(_ trace: Trace) async {
        // Process completed traces
        let duration = trace.duration ?? 0
        print("Trace completed: \(trace.traceId.value) - Duration: \(duration)s")
    }
}

// Example alert webhook handler
struct WebhookAlertHandler: AlertDeliveryHandler {
    let webhookURL: URL
    
    init(webhookURL: URL) {
        self.webhookURL = webhookURL
    }
    
    func deliver(_ alert: Alert) async {
        // Send alert to webhook
        print("Sending alert to webhook: \(alert.title)")
    }
    
    func deliverResolution(_ alert: Alert) async {
        // Send resolution to webhook
        print("Sending resolution to webhook: \(alert.title)")
    }
}
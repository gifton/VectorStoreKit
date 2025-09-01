// VectorStoreKit: Monitoring Tests
//
// Tests for monitoring and observability components

import XCTest
@testable import VectorStoreKit

final class MonitoringTests: XCTestCase {
    
    // MARK: - Metrics Collector Tests
    
    func testMetricsCollector() async throws {
        let collector = MetricsCollector(configuration: .init(
            retentionPeriod: 3600,
            aggregationInterval: 10,
            enableGPUMetrics: false
        ))
        
        // Record partition operation
        await collector.recordPartitionOperation(
            PartitionOperation(
                type: .create,
                partitionId: "test-partition",
                duration: 100.5,
                success: true
            )
        )
        
        // Record query metrics
        await collector.recordQueryPerformance(
            QueryMetrics(
                partitionId: "test-partition",
                type: .knn,
                latency: 25.3,
                vectorsScanned: 1000,
                resultsReturned: 10
            )
        )
        
        // Get snapshot
        let snapshot = await collector.getMetricsSnapshot()
        XCTAssertFalse(snapshot.timeSeries.isEmpty)
        XCTAssertNotNil(snapshot.aggregates["counters"])
    }
    
    // MARK: - Distributed Tracer Tests
    
    func testDistributedTracer() async throws {
        let tracer = DistributedTracer(configuration: .init(
            maxActiveTraces: 100,
            samplingRate: 1.0
        ))
        
        // Start trace
        let context = await tracer.startTrace(
            operationName: "test.operation",
            attributes: ["test": "true"]
        )
        
        XCTAssertNotNil(context)
        
        if let context = context {
            // Create child span
            let childContext = await tracer.startSpan(
                operationName: "test.child",
                parentContext: context
            )
            
            XCTAssertNotNil(childContext)
            
            // End child span
            if let childContext = childContext {
                await tracer.endSpan(context: childContext, status: .ok)
            }
            
            // End root span
            await tracer.endSpan(
                context: SpanContext(traceContext: context, spanId: context.spanId),
                status: .ok
            )
            
            // Verify trace
            let trace = await tracer.getActiveTrace(context.traceId)
            XCTAssertNil(trace) // Should be completed
            
            let completedTraces = await tracer.getCompletedTraces(limit: 10)
            XCTAssertFalse(completedTraces.isEmpty)
        }
    }
    
    // MARK: - Alerting System Tests
    
    func testAlertingSystem() async throws {
        let alerting = AlertingSystem(configuration: .init(
            evaluationInterval: 1,
            enableSuppression: false
        ))
        
        // Add alert rule
        let rule = AlertRule(
            name: "Test Alert",
            description: "Test alert rule",
            condition: .threshold(
                metric: "test.metric",
                operator: .greaterThan,
                value: 100
            ),
            severity: .warning
        )
        
        let ruleId = await alerting.addRule(rule)
        
        // Verify rule was added
        let rules = await alerting.getRules()
        XCTAssertEqual(rules.count, 1)
        XCTAssertEqual(rules.first?.name, "Test Alert")
        
        // Test alert delivery handler
        let testHandler = TestAlertHandler()
        _ = await alerting.addDeliveryHandler(testHandler)
        
        // Get active alerts (should be empty initially)
        let activeAlerts = await alerting.getActiveAlerts()
        XCTAssertTrue(activeAlerts.isEmpty)
    }
    
    // MARK: - Observability Dashboard Tests
    
    func testObservabilityDashboard() async throws {
        let metricsCollector = MetricsCollector()
        let dashboard = ObservabilityDashboard(
            configuration: .init(
                aggregationWindow: 10,
                historicalRetention: 3600
            ),
            metricsCollector: metricsCollector
        )
        
        // Record some metrics
        await metricsCollector.recordResourceUtilization(
            ResourceMetrics(
                nodeId: "test-node",
                cpuUsage: 50.0,
                memoryUsed: 1_073_741_824,
                memoryAvailable: 1_073_741_824,
                diskReadBytes: 0,
                diskWriteBytes: 0
            )
        )
        
        // Get dashboard view
        let view = await dashboard.getRealTimeDashboard()
        XCTAssertEqual(view.systemMetrics.cpuUsage, 50.0)
        XCTAssertGreaterThan(view.healthScore, 0)
        
        // Test export
        let jsonData = try await dashboard.exportMetrics(format: .json)
        XCTAssertGreaterThan(jsonData.count, 0)
        
        // Verify JSON is valid
        let decoded = try JSONDecoder().decode(DashboardView.self, from: jsonData)
        XCTAssertEqual(decoded.systemMetrics.cpuUsage, view.systemMetrics.cpuUsage)
    }
    
    // MARK: - Integration Test
    
    func testMonitoringSystemIntegration() async throws {
        let monitoring = await MonitoringSystem(configuration: .development)
        
        // Test partition operation monitoring
        try await monitoring.recordPartitionOperation(
            type: .create,
            partitionId: "integration-test",
            nodeId: "node-001"
        ) {
            // Simulate work
            try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        }
        
        // Test query monitoring
        let results = try await monitoring.recordQuery(
            partitionId: "integration-test",
            queryType: .knn,
            vectorCount: 100
        ) {
            // Simulate query
            try await Task.sleep(nanoseconds: 5_000_000) // 5ms
            return ["result1", "result2"]
        }
        
        XCTAssertEqual(results.count, 2)
        
        // Get dashboard
        let dashboard = await monitoring.getDashboard()
        XCTAssertGreaterThan(dashboard.healthScore, 0)
        
        // Export metrics
        let prometheusData = try await monitoring.exportMetrics(format: .prometheus)
        XCTAssertGreaterThan(prometheusData.count, 0)
    }
    
    // MARK: - Performance Test
    
    func testMetricsCollectorPerformance() async throws {
        let collector = MetricsCollector(configuration: .init(
            enableGPUMetrics: false,
            enableDetailedProfiling: false
        ))
        
        measure {
            let expectation = self.expectation(description: "Metrics recording")
            
            Task {
                // Record 1000 metrics
                for i in 0..<1000 {
                    await collector.recordQueryPerformance(
                        QueryMetrics(
                            partitionId: "partition-\(i % 10)",
                            type: .knn,
                            latency: Double.random(in: 10...100),
                            vectorsScanned: Int.random(in: 100...10000),
                            resultsReturned: Int.random(in: 1...100)
                        )
                    )
                }
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 5.0)
        }
    }
}

// MARK: - Test Helpers

class TestAlertHandler: AlertDeliveryHandler {
    var deliveredAlerts: [Alert] = []
    var resolvedAlerts: [Alert] = []
    
    func deliver(_ alert: Alert) async {
        deliveredAlerts.append(alert)
    }
    
    func deliverResolution(_ alert: Alert) async {
        resolvedAlerts.append(alert)
    }
}
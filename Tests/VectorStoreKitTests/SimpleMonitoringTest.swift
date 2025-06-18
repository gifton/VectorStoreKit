// Simple test to verify monitoring components compile and work

import XCTest
@testable import VectorStoreKit

final class SimpleMonitoringTest: XCTestCase {
    
    func testBasicMonitoring() async throws {
        // Test MetricsCollector
        let collector = MetricsCollector()
        await collector.recordPartitionOperation(
            PartitionOperation(
                type: .create,
                partitionId: "test",
                duration: 100,
                success: true
            )
        )
        
        // Test DistributedTracer
        let tracer = DistributedTracer()
        let context = await tracer.startTrace(operationName: "test")
        XCTAssertNotNil(context)
        
        // Test AlertingSystem
        let alerting = AlertingSystem()
        let rule = AlertRule(
            name: "Test",
            description: "Test rule",
            condition: .threshold(metric: "test", operator: .greaterThan, value: 100)
        )
        _ = await alerting.addRule(rule)
        
        // Test ObservabilityDashboard
        let dashboard = ObservabilityDashboard(metricsCollector: collector)
        let view = await dashboard.getRealTimeDashboard()
        XCTAssertGreaterThanOrEqual(view.healthScore, 0)
        
        print("✅ All monitoring components initialized successfully")
    }
}
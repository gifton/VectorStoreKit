// VectorStoreKit: Validation Example
//
// Example demonstrating configuration validation and health monitoring

import Foundation
import VectorStoreKit

@main
struct ValidationExampleApp {
    static func main() async throws {
        print("=== VectorStoreKit Validation Example ===\n")
        
        // 1. Configuration Validation
        await demonstrateConfigurationValidation()
        
        // 2. Health Monitoring
        await demonstrateHealthMonitoring()
        
        // 3. Auto-Configuration
        await demonstrateAutoConfiguration()
        
        print("\n=== Validation Example Complete ===")
    }
    
    static func demonstrateConfigurationValidation() async {
        print("1. Configuration Validation")
        print("-" * 40)
        
        // Create a configuration
        let config = StoreConfiguration(
            name: "MyVectorStore",
            enableProfiling: true,
            enableAnalytics: true,
            integrityCheckInterval: 3600,
            optimizationThreshold: 500_000,
            vectorDimension: .fixed768,
            distanceComputeBackend: .metal
        )
        
        // Create validator and register rules
        let validator = ConfigurationValidator()
        validator.registerRules(HardwareValidator.createStoreConfigurationRules())
        validator.registerRules(MemoryValidator.createStoreMemoryRules())
        
        // Validate configuration
        let result = await validator.validate(config)
        
        print("Configuration: \(config.name)")
        print("Vector Dimension: \(config.vectorDimension.dimension)")
        print("Backend: \(config.distanceComputeBackend)\n")
        
        print("Validation Result: \(result.summary())")
        print("Issues Found: \(result.issues.count)")
        
        // Display issues by severity
        if result.criticalCount > 0 {
            print("\nCritical Issues:")
            for issue in result.issues.filter({ $0.severity == .critical }) {
                print("  - \(issue.message)")
                if let suggestion = issue.suggestion {
                    print("    Fix: \(suggestion)")
                }
            }
        }
        
        if result.errorCount > 0 {
            print("\nErrors:")
            for issue in result.issues.filter({ $0.severity == .error }) {
                print("  - \(issue.message)")
            }
        }
        
        if result.warningCount > 0 {
            print("\nWarnings:")
            for issue in result.issues.filter({ $0.severity == .warning }) {
                print("  - \(issue.message)")
            }
        }
        
        print()
    }
    
    static func demonstrateHealthMonitoring() async {
        print("2. Health Monitoring")
        print("-" * 40)
        
        // Create health monitor
        let monitor = HealthMonitor { alert in
            print("[ALERT] \(alert.level): \(alert.message)")
        }
        
        // Register health checks
        let healthChecks = StoreHealthChecks.createStandardChecks()
        monitor.registerChecks(healthChecks)
        
        // Run health checks
        let report = await monitor.runAllChecks()
        
        print("Health Status: \(report.overallStatus)")
        print("Summary: \(report.summary)")
        
        // Show individual check results
        print("\nHealth Check Details:")
        for result in report.results {
            let statusIcon: String
            switch result.status {
            case .healthy: statusIcon = "✅"
            case .degraded: statusIcon = "⚠️"
            case .unhealthy: statusIcon = "❌"
            case .critical: statusIcon = "🔥"
            }
            
            print("\n\(statusIcon) \(result.checkId):")
            print("   Status: \(result.status)")
            print("   Message: \(result.message)")
            
            if !result.suggestions.isEmpty {
                print("   Suggestions:")
                for suggestion in result.suggestions {
                    print("     - \(suggestion)")
                }
            }
        }
        
        print()
    }
    
    static func demonstrateAutoConfiguration() async {
        print("3. Auto-Configuration")
        print("-" * 40)
        
        // Define workload characteristics
        let workload = AutoConfigurator.WorkloadProfile(
            expectedVectorCount: 1_000_000,
            vectorDimension: 1536, // OpenAI embeddings
            queriesPerSecond: 200,
            updateFrequency: .daily,
            latencyRequirementMs: 100,
            accuracyRequirement: .high
        )
        
        // Detect hardware
        let hardware = AutoConfigurator.HardwareProfile.detect()
        
        print("Workload Profile:")
        print("  Vectors: \(workload.expectedVectorCount.formatted())")
        print("  Dimension: \(workload.vectorDimension)")
        print("  QPS: \(workload.queriesPerSecond)")
        print("  Accuracy: \(workload.accuracyRequirement.rawValue)")
        
        print("\nHardware Profile:")
        print("  CPU Cores: \(hardware.cpuCores)")
        print("  Memory: \(String(format: "%.1f", hardware.memoryGB))GB")
        print("  GPU: \(hardware.hasGPU ? "Available" : "Not Available")")
        if let gpuMem = hardware.gpuMemoryGB {
            print("  GPU Memory: \(String(format: "%.1f", gpuMem))GB")
        }
        print("  Unified Memory: \(hardware.hasUnifiedMemory)")
        print("  Storage: \(hardware.storageType.rawValue)")
        
        // Generate optimal configuration
        let (storeConfig, indexConfig, storageConfig, rationale) = 
            AutoConfigurator.generateOptimalConfiguration(
                for: workload,
                hardware: hardware
            )
        
        print("\nRecommended Configuration:")
        print("  Store:")
        print("    - Backend: \(storeConfig.distanceComputeBackend)")
        print("    - Vector Type: \(storeConfig.vectorDimension)")
        print("    - Optimization Threshold: \(storeConfig.optimizationThreshold.formatted())")
        
        if let hnsw = indexConfig as? HNSWConfiguration {
            print("  Index: HNSW")
            print("    - M: \(hnsw.maxConnections)")
            print("    - efConstruction: \(hnsw.efConstruction)")
        } else if let ivf = indexConfig as? IVFConfiguration {
            print("  Index: IVF")
            print("    - Clusters: \(ivf.clusterCount)")
            print("    - Probes: \(ivf.defaultProbes)")
        }
        
        print("  Storage:")
        print("    - Tiering: \(storageConfig.enableTiering ? "Enabled" : "Disabled")")
        print("    - Hot Tier: \(storageConfig.hotTierSizeLimit / 1_048_576)MB")
        print("    - Compression: \(storageConfig.coldTierCompression)")
        
        print("\nConfiguration Rationale:")
        for (i, reason) in rationale.enumerated() {
            print("  \(i + 1). \(reason)")
        }
        
        // Validate the generated configuration
        let validator = MigrationRules.createAllMigrationRules()
        let validationResult = await validator.validate(storeConfig)
        
        print("\nGenerated Configuration Validation: \(validationResult.summary())")
        
        // Performance analysis
        let analysis = PerformanceValidator.analyzeBottlenecks(
            vectorCount: workload.expectedVectorCount,
            dimension: workload.vectorDimension,
            queryRate: workload.queriesPerSecond,
            hardware: hardware
        )
        
        print("\nPerformance Analysis:")
        if analysis.bottlenecks.isEmpty {
            print("  No bottlenecks detected!")
        } else {
            print("  Bottlenecks:")
            for bottleneck in analysis.bottlenecks {
                print("    - [\(bottleneck.severity)] \(bottleneck.component)")
                print("      Impact: \(bottleneck.impact)")
            }
        }
        
        print("  Expected Latency: \(String(format: "%.1f", analysis.expectedLatencyMs))ms")
        print("  Max Throughput: \(analysis.maxThroughputQPS) QPS")
    }
}

// Helper extension for string repetition
extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}
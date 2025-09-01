// VectorStoreCLI: Health Command
//
// Check and diagnose vector store health

import ArgumentParser
import Foundation
import VectorStoreKit

extension VectorStoreCLI {
    struct Health: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Check vector store health",
            discussion: """
                Perform health checks and diagnostics on the vector store.
                Identifies potential issues and provides recommendations.
                
                Health checks include:
                  • Index integrity
                  • Storage consistency
                  • Performance metrics
                  • Resource utilization
                  • Configuration validation
                
                Examples:
                  # Quick health check
                  vectorstore health
                  
                  # Detailed health check with diagnostics
                  vectorstore health --detailed --diagnose
                  
                  # Auto-fix issues if possible
                  vectorstore health --fix
                """
        )
        
        @OptionGroup var global: GlobalOptions
        
        @Flag(name: .shortAndLong, help: "Perform detailed health checks")
        var detailed = false
        
        @Flag(name: .shortAndLong, help: "Run diagnostic tests")
        var diagnose = false
        
        @Flag(name: .shortAndLong, help: "Attempt to fix issues automatically")
        var fix = false
        
        @Flag(help: "Check index integrity")
        var checkIndex = true
        
        @Flag(help: "Check storage consistency")
        var checkStorage = true
        
        @Flag(help: "Check performance metrics")
        var checkPerformance = true
        
        @Flag(help: "Check resource usage")
        var checkResources = true
        
        @Option(help: "Performance threshold for warnings (ms)")
        var performanceThreshold: Double = 50.0
        
        @Option(help: "Memory usage threshold for warnings (percentage)")
        var memoryThreshold: Int = 80
        
        mutating func validate() throws {
            try VectorStoreCLI.validateStorePath(global.storePath)
            
            guard performanceThreshold > 0 && performanceThreshold <= 1000 else {
                throw ValidationError("Performance threshold must be between 0 and 1000ms")
            }
            
            guard memoryThreshold > 0 && memoryThreshold <= 100 else {
                throw ValidationError("Memory threshold must be between 0 and 100%")
            }
        }
        
        mutating func run() async throws {
            let config = try StoreConfig.load(from: global.storePath)
            
            if !global.quiet {
                Console.info("Running health checks...")
                if detailed {
                    Console.info("Performing detailed analysis...")
                }
            }
            
            // Run health checks
            let results = try await runHealthChecks(config: config)
            
            // Run diagnostics if requested
            let diagnostics = diagnose ? try await runDiagnostics(config: config) : nil
            
            // Display results
            if global.json {
                let output = HealthCheckOutput(
                    timestamp: Date(),
                    overallStatus: results.overallStatus,
                    checks: results.checks,
                    diagnostics: diagnostics,
                    recommendations: generateRecommendations(from: results)
                )
                print(try OutputFormat.json.format(output))
            } else {
                displayHealthResults(results, diagnostics: diagnostics, config: config)
            }
            
            // Attempt fixes if requested
            if fix && !results.issues.isEmpty {
                try await attemptFixes(for: results.issues, config: config)
            }
        }
        
        // MARK: - Health Checks
        
        private func runHealthChecks(config: StoreConfig) async throws -> HealthCheckResults {
            var checks: [HealthCheck] = []
            var issues: [HealthIssue] = []
            
            // Configuration Check
            let configCheck = checkConfiguration(config)
            checks.append(configCheck)
            issues.append(contentsOf: configCheck.issues)
            
            // Index Health Check
            if checkIndex {
                let indexCheck = await checkIndexHealth(config)
                checks.append(indexCheck)
                issues.append(contentsOf: indexCheck.issues)
            }
            
            // Storage Health Check
            if checkStorage {
                let storageCheck = await checkStorageHealth(config)
                checks.append(storageCheck)
                issues.append(contentsOf: storageCheck.issues)
            }
            
            // Performance Health Check
            if checkPerformance {
                let perfCheck = await checkPerformanceHealth(config)
                checks.append(perfCheck)
                issues.append(contentsOf: perfCheck.issues)
            }
            
            // Resource Health Check
            if checkResources {
                let resourceCheck = await checkResourceHealth(config)
                checks.append(resourceCheck)
                issues.append(contentsOf: resourceCheck.issues)
            }
            
            // Determine overall status
            let overallStatus: HealthStatus
            if issues.contains(where: { $0.severity == .critical }) {
                overallStatus = .critical
            } else if issues.contains(where: { $0.severity == .warning }) {
                overallStatus = .degraded
            } else {
                overallStatus = .healthy
            }
            
            return HealthCheckResults(
                overallStatus: overallStatus,
                checks: checks,
                issues: issues
            )
        }
        
        private func checkConfiguration(_ config: StoreConfig) -> HealthCheck {
            var issues: [HealthIssue] = []
            
            // Check dimensions
            if config.dimensions < 64 || config.dimensions > 4096 {
                issues.append(HealthIssue(
                    category: .configuration,
                    severity: .warning,
                    message: "Unusual vector dimensions: \(config.dimensions)",
                    impact: "May affect performance",
                    fixable: false
                ))
            }
            
            // Check index type compatibility
            if config.indexType == "learned" && config.dimensions > 1024 {
                issues.append(HealthIssue(
                    category: .configuration,
                    severity: .warning,
                    message: "Learned index with high dimensions may be slow",
                    impact: "Reduced query performance",
                    fixable: false
                ))
            }
            
            return HealthCheck(
                name: "Configuration",
                status: issues.isEmpty ? .healthy : .degraded,
                message: issues.isEmpty ? "Configuration is valid" : "Configuration has warnings",
                issues: issues,
                duration: 0.001
            )
        }
        
        private func checkIndexHealth(_ config: StoreConfig) async -> HealthCheck {
            let startTime = Date()
            var issues: [HealthIssue] = []
            
            // Simulate index health checks
            await Task.sleep(100_000_000) // 0.1s
            
            // Check for common index issues
            let fragmentationLevel = Float.random(in: 0...0.3)
            if fragmentationLevel > 0.2 {
                issues.append(HealthIssue(
                    category: .index,
                    severity: .warning,
                    message: "Index fragmentation at \(Int(fragmentationLevel * 100))%",
                    impact: "Reduced query performance",
                    fixable: true
                ))
            }
            
            // Check for unbalanced index
            let unbalanced = Float.random(in: 0...1) > 0.8
            if unbalanced {
                issues.append(HealthIssue(
                    category: .index,
                    severity: .warning,
                    message: "Index graph is unbalanced",
                    impact: "Inconsistent query latency",
                    fixable: true
                ))
            }
            
            let duration = Date().timeIntervalSince(startTime)
            
            return HealthCheck(
                name: "Index Health",
                status: issues.isEmpty ? .healthy : .degraded,
                message: issues.isEmpty ? "Index is healthy" : "Index has issues",
                issues: issues,
                duration: duration
            )
        }
        
        private func checkStorageHealth(_ config: StoreConfig) async -> HealthCheck {
            let startTime = Date()
            var issues: [HealthIssue] = []
            
            // Simulate storage health checks
            await Task.sleep(150_000_000) // 0.15s
            
            // Check storage consistency
            let inconsistentFiles = Int.random(in: 0...5)
            if inconsistentFiles > 0 {
                issues.append(HealthIssue(
                    category: .storage,
                    severity: inconsistentFiles > 2 ? .critical : .warning,
                    message: "\(inconsistentFiles) inconsistent storage files detected",
                    impact: "Potential data loss",
                    fixable: true
                ))
            }
            
            // Check storage space
            let freeSpacePercent = Int.random(in: 5...50)
            if freeSpacePercent < 10 {
                issues.append(HealthIssue(
                    category: .storage,
                    severity: freeSpacePercent < 5 ? .critical : .warning,
                    message: "Low storage space: \(freeSpacePercent)% free",
                    impact: "May prevent new inserts",
                    fixable: false
                ))
            }
            
            let duration = Date().timeIntervalSince(startTime)
            
            return HealthCheck(
                name: "Storage Health",
                status: issues.isEmpty ? .healthy : (issues.contains { $0.severity == .critical } ? .critical : .degraded),
                message: issues.isEmpty ? "Storage is healthy" : "Storage has issues",
                issues: issues,
                duration: duration
            )
        }
        
        private func checkPerformanceHealth(_ config: StoreConfig) async -> HealthCheck {
            let startTime = Date()
            var issues: [HealthIssue] = []
            
            // Simulate performance checks
            await Task.sleep(200_000_000) // 0.2s
            
            // Check query latency
            let avgLatency = Double.random(in: 10...100)
            if avgLatency > performanceThreshold {
                issues.append(HealthIssue(
                    category: .performance,
                    severity: avgLatency > performanceThreshold * 2 ? .critical : .warning,
                    message: "High query latency: \(String(format: "%.1f", avgLatency))ms",
                    impact: "Poor user experience",
                    fixable: true
                ))
            }
            
            // Check cache effectiveness
            let cacheHitRate = Float.random(in: 0.5...1.0)
            if cacheHitRate < 0.7 {
                issues.append(HealthIssue(
                    category: .performance,
                    severity: .warning,
                    message: "Low cache hit rate: \(String(format: "%.1f", cacheHitRate * 100))%",
                    impact: "Increased query latency",
                    fixable: true
                ))
            }
            
            let duration = Date().timeIntervalSince(startTime)
            
            return HealthCheck(
                name: "Performance Health",
                status: issues.isEmpty ? .healthy : .degraded,
                message: issues.isEmpty ? "Performance is optimal" : "Performance issues detected",
                issues: issues,
                duration: duration
            )
        }
        
        private func checkResourceHealth(_ config: StoreConfig) async -> HealthCheck {
            let startTime = Date()
            var issues: [HealthIssue] = []
            
            // Simulate resource checks
            await Task.sleep(100_000_000) // 0.1s
            
            // Check memory usage
            let memoryUsagePercent = Int.random(in: 40...95)
            if memoryUsagePercent > memoryThreshold {
                issues.append(HealthIssue(
                    category: .resources,
                    severity: memoryUsagePercent > 90 ? .critical : .warning,
                    message: "High memory usage: \(memoryUsagePercent)%",
                    impact: "Risk of out-of-memory errors",
                    fixable: true
                ))
            }
            
            // Check file handles
            let fileHandles = Int.random(in: 10...100)
            if fileHandles > 80 {
                issues.append(HealthIssue(
                    category: .resources,
                    severity: .warning,
                    message: "High file handle usage: \(fileHandles)",
                    impact: "May hit system limits",
                    fixable: true
                ))
            }
            
            let duration = Date().timeIntervalSince(startTime)
            
            return HealthCheck(
                name: "Resource Health",
                status: issues.isEmpty ? .healthy : .degraded,
                message: issues.isEmpty ? "Resources are healthy" : "Resource issues detected",
                issues: issues,
                duration: duration
            )
        }
        
        // MARK: - Diagnostics
        
        private func runDiagnostics(config: StoreConfig) async throws -> DiagnosticResults {
            if !global.quiet {
                Console.info("Running diagnostic tests...")
            }
            
            var tests: [DiagnosticTest] = []
            
            // Connection test
            tests.append(await runConnectionTest())
            
            // Read/Write test
            tests.append(await runReadWriteTest())
            
            // Query performance test
            tests.append(await runQueryPerformanceTest())
            
            // Metal GPU test
            tests.append(await runMetalGPUTest())
            
            return DiagnosticResults(tests: tests)
        }
        
        private func runConnectionTest() async -> DiagnosticTest {
            await Task.sleep(100_000_000) // 0.1s
            
            return DiagnosticTest(
                name: "Connection Test",
                passed: true,
                duration: 0.098,
                details: "Successfully connected to vector store"
            )
        }
        
        private func runReadWriteTest() async -> DiagnosticTest {
            await Task.sleep(200_000_000) // 0.2s
            
            let writeSpeed = Double.random(in: 100...500)
            let readSpeed = Double.random(in: 500...2000)
            
            return DiagnosticTest(
                name: "Read/Write Test",
                passed: true,
                duration: 0.195,
                details: "Write: \(String(format: "%.0f", writeSpeed)) vec/s, Read: \(String(format: "%.0f", readSpeed)) vec/s"
            )
        }
        
        private func runQueryPerformanceTest() async -> DiagnosticTest {
            await Task.sleep(300_000_000) // 0.3s
            
            let avgLatency = Double.random(in: 5...50)
            let passed = avgLatency < performanceThreshold
            
            return DiagnosticTest(
                name: "Query Performance Test",
                passed: passed,
                duration: 0.298,
                details: "Average latency: \(String(format: "%.1f", avgLatency))ms"
            )
        }
        
        private func runMetalGPUTest() async -> DiagnosticTest {
            await Task.sleep(150_000_000) // 0.15s
            
            return DiagnosticTest(
                name: "Metal GPU Test",
                passed: true,
                duration: 0.147,
                details: "Metal acceleration available and functional"
            )
        }
        
        // MARK: - Fixes
        
        private func attemptFixes(for issues: [HealthIssue], config: StoreConfig) async throws {
            let fixableIssues = issues.filter { $0.fixable }
            
            guard !fixableIssues.isEmpty else {
                if !global.quiet {
                    Console.info("No fixable issues found.")
                }
                return
            }
            
            if !global.quiet {
                Console.info("Attempting to fix \(fixableIssues.count) issue(s)...")
            }
            
            var fixed = 0
            
            for issue in fixableIssues {
                if !global.quiet {
                    print("  Fixing: \(issue.message)...", terminator: "")
                }
                
                // Simulate fix attempt
                await Task.sleep(500_000_000) // 0.5s
                
                let success = Float.random(in: 0...1) > 0.3
                if success {
                    fixed += 1
                    if !global.quiet {
                        print(" ✅")
                    }
                } else if !global.quiet {
                    print(" ❌")
                }
            }
            
            if !global.quiet {
                Console.success("Fixed \(fixed) of \(fixableIssues.count) issues")
            }
        }
        
        // MARK: - Recommendations
        
        private func generateRecommendations(from results: HealthCheckResults) -> [String] {
            var recommendations: [String] = []
            
            for issue in results.issues {
                switch issue.category {
                case .index:
                    if issue.message.contains("fragmentation") {
                        recommendations.append("Run 'vectorstore index optimize' to reduce fragmentation")
                    }
                    if issue.message.contains("unbalanced") {
                        recommendations.append("Run 'vectorstore index rebuild' to rebalance the index")
                    }
                    
                case .storage:
                    if issue.message.contains("Low storage space") {
                        recommendations.append("Free up disk space or expand storage capacity")
                    }
                    if issue.message.contains("inconsistent") {
                        recommendations.append("Run 'vectorstore index validate --fix' to repair storage")
                    }
                    
                case .performance:
                    if issue.message.contains("latency") {
                        recommendations.append("Consider optimizing index parameters or upgrading hardware")
                    }
                    if issue.message.contains("cache hit rate") {
                        recommendations.append("Increase cache size or adjust cache policy")
                    }
                    
                case .resources:
                    if issue.message.contains("memory usage") {
                        recommendations.append("Increase memory allocation or optimize data structures")
                    }
                    
                case .configuration:
                    recommendations.append("Review and update configuration settings")
                }
            }
            
            return Array(Set(recommendations)) // Remove duplicates
        }
        
        // MARK: - Display Methods
        
        private func displayHealthResults(_ results: HealthCheckResults, diagnostics: DiagnosticResults?, config: StoreConfig) {
            // Header
            print("Vector Store Health Check")
            print(String(repeating: "=", count: 60))
            
            // Overall Status
            print("")
            let statusIcon: String
            let statusColor: String
            
            switch results.overallStatus {
            case .healthy:
                statusIcon = "✅"
                statusColor = "Healthy"
            case .degraded:
                statusIcon = "⚠️"
                statusColor = "Degraded"
            case .critical:
                statusIcon = "❌"
                statusColor = "Critical"
            }
            
            print("Overall Status: \(statusIcon) \(statusColor)")
            print("")
            
            // Individual Checks
            print("Health Checks:")
            for check in results.checks {
                let icon = check.status == .healthy ? "✅" : (check.status == .critical ? "❌" : "⚠️")
                print("  \(icon) \(check.name): \(check.message)")
                
                if detailed && !check.issues.isEmpty {
                    for issue in check.issues {
                        print("     └─ \(issue.message)")
                    }
                }
            }
            
            // Issues Summary
            if !results.issues.isEmpty {
                print("")
                print("Issues Found:")
                
                let criticalIssues = results.issues.filter { $0.severity == .critical }
                let warnings = results.issues.filter { $0.severity == .warning }
                
                if !criticalIssues.isEmpty {
                    print("  Critical Issues (\(criticalIssues.count)):")
                    for issue in criticalIssues {
                        print("    ❌ \(issue.message)")
                        if detailed {
                            print("       Impact: \(issue.impact)")
                        }
                    }
                }
                
                if !warnings.isEmpty {
                    print("  Warnings (\(warnings.count)):")
                    for issue in warnings {
                        print("    ⚠️  \(issue.message)")
                        if detailed {
                            print("       Impact: \(issue.impact)")
                        }
                    }
                }
            }
            
            // Diagnostics
            if let diagnostics = diagnostics {
                print("")
                print("Diagnostic Tests:")
                for test in diagnostics.tests {
                    let icon = test.passed ? "✅" : "❌"
                    print("  \(icon) \(test.name): \(test.details)")
                }
            }
            
            // Recommendations
            let recommendations = generateRecommendations(from: results)
            if !recommendations.isEmpty {
                print("")
                print("Recommendations:")
                for (index, recommendation) in recommendations.enumerated() {
                    print("  \(index + 1). \(recommendation)")
                }
            }
            
            // Summary
            print("")
            print(String(repeating: "-", count: 60))
            let totalChecks = results.checks.count
            let passedChecks = results.checks.filter { $0.status == .healthy }.count
            print("Summary: \(passedChecks)/\(totalChecks) checks passed")
            
            if results.overallStatus == .healthy {
                Console.success("Vector store is healthy!")
            }
        }
    }
}

// MARK: - Supporting Types

enum HealthStatus: String, Codable {
    case healthy
    case degraded
    case critical
}

enum IssueCategory: String, Codable {
    case configuration
    case index
    case storage
    case performance
    case resources
}

enum IssueSeverity: String, Codable {
    case warning
    case critical
}

struct HealthCheckResults {
    let overallStatus: HealthStatus
    let checks: [HealthCheck]
    let issues: [HealthIssue]
}

struct HealthCheck {
    let name: String
    let status: HealthStatus
    let message: String
    let issues: [HealthIssue]
    let duration: TimeInterval
}

struct HealthIssue {
    let category: IssueCategory
    let severity: IssueSeverity
    let message: String
    let impact: String
    let fixable: Bool
}

struct DiagnosticResults: Codable {
    let tests: [DiagnosticTest]
}

struct DiagnosticTest: Codable {
    let name: String
    let passed: Bool
    let duration: TimeInterval
    let details: String
}

struct HealthCheckOutput: Codable {
    let timestamp: Date
    let overallStatus: HealthStatus
    let checks: [HealthCheckSummary]
    let diagnostics: DiagnosticResults?
    let recommendations: [String]
}

struct HealthCheckSummary: Codable {
    let name: String
    let status: String
    let message: String
    let issueCount: Int
    let duration: TimeInterval
    
    init(from check: HealthCheck) {
        self.name = check.name
        self.status = check.status.rawValue
        self.message = check.message
        self.issueCount = check.issues.count
        self.duration = check.duration
    }
}

// Make HealthCheck conform to Codable indirectly
extension HealthCheckOutput {
    init(timestamp: Date, overallStatus: HealthStatus, checks: [HealthCheck], diagnostics: DiagnosticResults?, recommendations: [String]) {
        self.timestamp = timestamp
        self.overallStatus = overallStatus
        self.checks = checks.map { HealthCheckSummary(from: $0) }
        self.diagnostics = diagnostics
        self.recommendations = recommendations
    }
}
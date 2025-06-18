// VectorStoreKit: Benchmark Runner
//
// Command-line tool for running comprehensive benchmarks

import Foundation
import ArgumentParser
import VectorStoreKit

@main
struct VectorStoreKitBenchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vsk-benchmark",
        abstract: "Run comprehensive benchmarks for VectorStoreKit",
        version: "1.0.0",
        subcommands: [
            Run.self,
            Compare.self,
            List.self,
            Export.self,
            CI.self
        ],
        defaultSubcommand: Run.self
    )
}

// MARK: - Run Command

struct Run: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run benchmarks with specified configuration"
    )
    
    @Option(name: .shortAndLong, help: "Configuration file path")
    var config: String?
    
    @Option(name: .shortAndLong, help: "Preset configuration: quick, standard, comprehensive")
    var preset: String = "standard"
    
    @Option(name: .shortAndLong, help: "Output directory for results")
    var output: String = "./benchmark-results"
    
    @Option(name: .shortAndLong, help: "Number of iterations")
    var iterations: Int?
    
    @Option(name: .shortAndLong, help: "Vector counts (comma-separated)")
    var vectorCounts: String?
    
    @Option(name: .shortAndLong, help: "Dimensions (comma-separated)")
    var dimensions: String?
    
    @Flag(name: .shortAndLong, help: "Run only vector operation benchmarks")
    var vectorOps = false
    
    @Flag(name: .shortAndLong, help: "Run only index benchmarks")
    var indexes = false
    
    @Flag(name: .shortAndLong, help: "Run only Metal acceleration benchmarks")
    var metal = false
    
    @Flag(name: .shortAndLong, help: "Run only cache benchmarks")
    var cache = false
    
    @Flag(name: .shortAndLong, help: "Run only ML benchmarks")
    var ml = false
    
    @Flag(name: .shortAndLong, help: "Run only scalability benchmarks")
    var scalability = false
    
    @Flag(name: .shortAndLong, help: "Run only concurrency benchmarks")
    var concurrency = false
    
    @Flag(name: .shortAndLong, help: "Run only memory benchmarks")
    var memory = false
    
    @Flag(name: .shortAndLong, help: "Run only end-to-end benchmarks")
    var endToEnd = false
    
    @Flag(name: .shortAndLong, help: "Compare with baseline")
    var compareBaseline = false
    
    @Option(name: .long, help: "Baseline results path")
    var baseline: String?
    
    @Flag(name: .shortAndLong, help: "Enable verbose output")
    var verbose = false
    
    func run() async throws {
        print("🚀 VectorStoreKit Benchmark Runner")
        print("==================================\n")
        
        // Load or create configuration
        let configuration = try loadConfiguration()
        
        // Create output directory
        try createOutputDirectory()
        
        // Initialize framework and metrics
        let framework = BenchmarkFramework(
            configuration: BenchmarkFramework.Configuration(
                iterations: configuration.execution.iterations,
                warmupIterations: configuration.execution.warmupIterations,
                timeout: configuration.general.timeout,
                collectMemoryMetrics: configuration.execution.profileMemory,
                collectCPUMetrics: configuration.execution.profileCPU
            )
        )
        
        let metrics = PerformanceMetrics()
        
        // Select reporter based on output format
        let reporter = createReporter()
        
        // Start benchmarking session
        reporter.startSession(configuration.general.name)
        
        if verbose {
            print("Configuration:")
            print("  • Iterations: \(configuration.execution.iterations)")
            print("  • Warmup: \(configuration.execution.warmupIterations)")
            print("  • Vector counts: \(configuration.data.vectorCounts)")
            print("  • Dimensions: \(configuration.data.dimensions)")
            print("")
        }
        
        // Collect all results
        var allResults: [String: BenchmarkFramework.Statistics] = [:]
        let startTime = Date()
        
        // Run selected benchmark suites
        if shouldRunVectorOps(configuration) {
            print("📊 Running Vector Operations Benchmarks...")
            let vectorBenchmarks = VectorOperationsBenchmarks(
                framework: framework,
                metrics: metrics
            )
            let results = try await vectorBenchmarks.runAll()
            allResults.merge(results) { _, new in new }
            print("✅ Vector Operations complete\n")
        }
        
        if shouldRunIndexes(configuration) {
            print("📊 Running Index Benchmarks...")
            let indexBenchmarks = IndexBenchmarkRunner(
                configuration: createIndexBenchmarkConfig(configuration)
            )
            let results = try await indexBenchmarks.runAllBenchmarks()
            // Convert results to framework statistics
            for result in results {
                let key = "\(result.indexType)_\(result.dimensions)d_\(result.datasetSize)"
                allResults[key] = createStatistics(from: result)
            }
            print("✅ Index Benchmarks complete\n")
        }
        
        if shouldRunMetal(configuration) {
            print("📊 Running Metal Acceleration Benchmarks...")
            if let metalBenchmarks = try? MetalAccelerationBenchmarks(
                framework: framework,
                metrics: metrics
            ) {
                let results = try await metalBenchmarks.runAll()
                allResults.merge(results) { _, new in new }
                print("✅ Metal Acceleration complete\n")
            } else {
                print("⚠️  Metal not available on this system\n")
            }
        }
        
        if shouldRunCache(configuration) {
            print("📊 Running Cache Benchmarks...")
            let cacheBenchmarks = CacheBenchmarks(
                framework: framework,
                metrics: metrics
            )
            let results = try await cacheBenchmarks.runAll()
            allResults.merge(results) { _, new in new }
            print("✅ Cache Benchmarks complete\n")
        }
        
        if shouldRunML(configuration) {
            print("📊 Running ML Benchmarks...")
            if let mlBenchmarks = try? MLBenchmarks(
                framework: framework,
                metrics: metrics
            ) {
                let results = try await mlBenchmarks.runAll()
                allResults.merge(results) { _, new in new }
                print("✅ ML Benchmarks complete\n")
            } else {
                print("⚠️  ML components not available\n")
            }
        }
        
        if shouldRunScalability(configuration) {
            print("📊 Running Scalability Benchmarks...")
            let scalabilityBenchmarks = ScalabilityBenchmarks(
                framework: framework,
                metrics: metrics
            )
            let results = try await scalabilityBenchmarks.runAll()
            allResults.merge(results) { _, new in new }
            print("✅ Scalability Benchmarks complete\n")
        }
        
        if shouldRunConcurrency(configuration) {
            print("📊 Running Concurrency Benchmarks...")
            let concurrencyBenchmarks = ConcurrencyBenchmarks(
                framework: framework,
                metrics: metrics
            )
            let results = try await concurrencyBenchmarks.runAll()
            allResults.merge(results) { _, new in new }
            print("✅ Concurrency Benchmarks complete\n")
        }
        
        if shouldRunMemory(configuration) {
            print("📊 Running Memory Benchmarks...")
            let memoryBenchmarks = MemoryBenchmarks(
                framework: framework,
                metrics: metrics
            )
            let results = try await memoryBenchmarks.runAll()
            allResults.merge(results) { _, new in new }
            print("✅ Memory Benchmarks complete\n")
        }
        
        if shouldRunEndToEnd(configuration) {
            print("📊 Running End-to-End Benchmarks...")
            let endToEndBenchmarks = EndToEndBenchmarks(
                framework: framework,
                metrics: metrics
            )
            let results = try await endToEndBenchmarks.runAll()
            allResults.merge(results) { _, new in new }
            print("✅ End-to-End Benchmarks complete\n")
        }
        
        let totalTime = Date().timeIntervalSince(startTime)
        
        // Generate report
        reporter.endSession()
        let report = reporter.generateReport(
            results: Array(allResults.values),
            profile: await metrics.stopCollection()
        )
        
        // Save results
        if configuration.general.saveResults {
            try saveResults(
                configuration: configuration,
                results: allResults,
                report: report,
                totalTime: totalTime
            )
        }
        
        // Compare with baseline if requested
        if compareBaseline, let baselinePath = baseline ?? configuration.general.baselinePath {
            print("\n📊 Comparing with baseline...")
            try await compareWithBaseline(
                current: allResults,
                baselinePath: baselinePath,
                configuration: configuration
            )
        }
        
        // Print summary
        print("\n✅ Benchmarks completed in \(String(format: "%.2f", totalTime)) seconds")
        print("📁 Results saved to: \(output)")
        
        if verbose {
            print("\nSummary:")
            print("  • Total benchmarks: \(allResults.count)")
            print("  • Average time per benchmark: \(String(format: "%.3f", totalTime / Double(allResults.count)))s")
        }
    }
    
    private func loadConfiguration() throws -> BenchmarkConfiguration {
        if let configPath = config {
            return try BenchmarkConfiguration.load(from: configPath)
        }
        
        var baseConfig: BenchmarkConfiguration
        switch preset.lowercased() {
        case "quick":
            baseConfig = .quick
        case "comprehensive":
            baseConfig = .comprehensive
        case "ci":
            baseConfig = .ci
        default:
            baseConfig = .standard
        }
        
        // Apply command-line overrides
        let builder = BenchmarkConfigurationBuilder()
            .withName(baseConfig.general.name)
        
        if let iterations = iterations {
            builder.withIterations(iterations)
        }
        
        if let vectorCounts = parseIntArray(vectorCounts) {
            builder.withVectorCounts(vectorCounts)
        }
        
        if let dimensions = parseIntArray(dimensions) {
            builder.withDimensions(dimensions)
        }
        
        // Apply suite filters
        let suites = BenchmarkConfiguration.BenchmarkSuites(
            runVectorOperations: vectorOps || (!hasAnyFilter() && baseConfig.suites.runVectorOperations),
            runIndexBenchmarks: indexes || (!hasAnyFilter() && baseConfig.suites.runIndexBenchmarks),
            runMetalAcceleration: metal || (!hasAnyFilter() && baseConfig.suites.runMetalAcceleration),
            runDistributed: false, // Not implemented in CLI yet
            runCache: cache || (!hasAnyFilter() && baseConfig.suites.runCache),
            runML: ml || (!hasAnyFilter() && baseConfig.suites.runML),
            runScalability: scalability || (!hasAnyFilter() && baseConfig.suites.runScalability),
            runConcurrency: concurrency || (!hasAnyFilter() && baseConfig.suites.runConcurrency),
            runMemory: memory || (!hasAnyFilter() && baseConfig.suites.runMemory),
            runEndToEnd: endToEnd || (!hasAnyFilter() && baseConfig.suites.runEndToEnd)
        )
        
        builder.withSuites(suites)
        
        return builder.build()
    }
    
    private func createOutputDirectory() throws {
        let url = URL(fileURLWithPath: output)
        try FileManager.default.createDirectory(
            at: url,
            withIntermediateDirectories: true
        )
    }
    
    private func createReporter() -> BenchmarkReporter {
        if verbose {
            return TextReporter()
        } else {
            return BenchmarkReporter()
        }
    }
    
    private func hasAnyFilter() -> Bool {
        return vectorOps || indexes || metal || cache || ml ||
               scalability || concurrency || memory || endToEnd
    }
    
    private func shouldRunVectorOps(_ config: BenchmarkConfiguration) -> Bool {
        return config.suites.runVectorOperations
    }
    
    private func shouldRunIndexes(_ config: BenchmarkConfiguration) -> Bool {
        return config.suites.runIndexBenchmarks
    }
    
    private func shouldRunMetal(_ config: BenchmarkConfiguration) -> Bool {
        return config.suites.runMetalAcceleration
    }
    
    private func shouldRunCache(_ config: BenchmarkConfiguration) -> Bool {
        return config.suites.runCache
    }
    
    private func shouldRunML(_ config: BenchmarkConfiguration) -> Bool {
        return config.suites.runML
    }
    
    private func shouldRunScalability(_ config: BenchmarkConfiguration) -> Bool {
        return config.suites.runScalability
    }
    
    private func shouldRunConcurrency(_ config: BenchmarkConfiguration) -> Bool {
        return config.suites.runConcurrency
    }
    
    private func shouldRunMemory(_ config: BenchmarkConfiguration) -> Bool {
        return config.suites.runMemory
    }
    
    private func shouldRunEndToEnd(_ config: BenchmarkConfiguration) -> Bool {
        return config.suites.runEndToEnd
    }
    
    private func parseIntArray(_ string: String?) -> [Int]? {
        guard let string = string else { return nil }
        return string.split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    }
    
    private func createIndexBenchmarkConfig(
        _ config: BenchmarkConfiguration
    ) -> IndexBenchmarkRunner.BenchmarkConfiguration {
        return IndexBenchmarkRunner.BenchmarkConfiguration(
            dimensions: config.data.dimensions.first ?? 128,
            vectorCounts: config.data.vectorCounts,
            queryCount: config.data.queryCounts.first ?? 1000,
            k: config.data.kValues.first ?? 10,
            iterations: config.execution.iterations,
            warmupIterations: config.execution.warmupIterations
        )
    }
    
    private func createStatistics(
        from result: IndexBenchmarkRunner.BenchmarkRunResult
    ) -> BenchmarkFramework.Statistics {
        return BenchmarkFramework.Statistics(
            measurements: result.searchTimes.map { time in
                BenchmarkFramework.Measurement(
                    value: time,
                    unit: .seconds
                )
            }
        )
    }
    
    private func saveResults(
        configuration: BenchmarkConfiguration,
        results: [String: BenchmarkFramework.Statistics],
        report: String,
        totalTime: TimeInterval
    ) throws {
        let timestamp = ISO8601DateFormatter().string(from: Date())
        let resultsDir = URL(fileURLWithPath: output)
            .appendingPathComponent(timestamp)
        
        try FileManager.default.createDirectory(
            at: resultsDir,
            withIntermediateDirectories: true
        )
        
        // Save configuration
        try configuration.save(
            to: resultsDir.appendingPathComponent("config.json").path
        )
        
        // Save raw results
        let resultsData = try JSONSerialization.data(
            withJSONObject: results.mapValues { stats in
                [
                    "mean": stats.mean,
                    "median": stats.median,
                    "stdDev": stats.standardDeviation,
                    "min": stats.minimum,
                    "max": stats.maximum,
                    "p95": stats.percentile95,
                    "p99": stats.percentile99,
                    "count": stats.count
                ]
            }
        )
        
        try resultsData.write(
            to: resultsDir.appendingPathComponent("results.json")
        )
        
        // Save report
        try report.write(
            to: resultsDir.appendingPathComponent("report.md"),
            atomically: true,
            encoding: .utf8
        )
        
        // Save summary
        let summary = """
        Benchmark Run Summary
        ====================
        
        Date: \(Date())
        Total Time: \(String(format: "%.2f", totalTime)) seconds
        Configuration: \(configuration.general.name)
        Total Benchmarks: \(results.count)
        
        Top Results:
        -----------
        \(formatTopResults(results))
        """
        
        try summary.write(
            to: resultsDir.appendingPathComponent("summary.txt"),
            atomically: true,
            encoding: .utf8
        )
    }
    
    private func formatTopResults(_ results: [String: BenchmarkFramework.Statistics]) -> String {
        let sorted = results.sorted { $0.value.mean < $1.value.mean }
        let top5 = sorted.prefix(5)
        
        return top5.map { name, stats in
            "• \(name): \(formatTime(stats.mean)) (±\(formatTime(stats.standardDeviation)))"
        }.joined(separator: "\n")
    }
    
    private func formatTime(_ seconds: Double) -> String {
        if seconds < 0.001 {
            return String(format: "%.2f μs", seconds * 1_000_000)
        } else if seconds < 1.0 {
            return String(format: "%.2f ms", seconds * 1_000)
        } else {
            return String(format: "%.2f s", seconds)
        }
    }
    
    private func compareWithBaseline(
        current: [String: BenchmarkFramework.Statistics],
        baselinePath: String,
        configuration: BenchmarkConfiguration
    ) async throws {
        // Load baseline results
        let baselineData = try Data(contentsOf: URL(fileURLWithPath: baselinePath))
        let baseline = try JSONDecoder().decode(
            [String: [String: Double]].self,
            from: baselineData
        )
        
        // Convert to comparison format
        let baselineResults = BenchmarkResults(
            id: "baseline",
            timestamp: Date(),
            configuration: configuration,
            results: baseline
        )
        
        let currentResults = BenchmarkResults(
            id: "current",
            timestamp: Date(),
            configuration: configuration,
            results: current
        )
        
        // Compare
        let comparison = ComparisonTools.compare(
            baseline: baselineResults,
            current: currentResults
        )
        
        // Print comparison report
        let report = ComparisonTools.generateReport(
            comparison: comparison,
            format: .markdown
        )
        
        print(report)
        
        // Save comparison
        let resultsDir = URL(fileURLWithPath: output)
        try report.write(
            to: resultsDir.appendingPathComponent("comparison.md"),
            atomically: true,
            encoding: .utf8
        )
    }
}

// MARK: - Compare Command

struct Compare: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Compare benchmark results"
    )
    
    @Argument(help: "Baseline results path")
    var baseline: String
    
    @Argument(help: "Current results path")
    var current: String
    
    @Option(name: .shortAndLong, help: "Output format: markdown, html, json, csv")
    var format: String = "markdown"
    
    @Option(name: .shortAndLong, help: "Output file path")
    var output: String?
    
    func run() async throws {
        print("📊 Comparing benchmark results...")
        
        // Load results
        let baselineData = try Data(contentsOf: URL(fileURLWithPath: baseline))
        let currentData = try Data(contentsOf: URL(fileURLWithPath: current))
        
        // Parse results
        let baselineResults = try JSONDecoder().decode(
            BenchmarkResults.self,
            from: baselineData
        )
        
        let currentResults = try JSONDecoder().decode(
            BenchmarkResults.self,
            from: currentData
        )
        
        // Compare
        let comparison = ComparisonTools.compare(
            baseline: baselineResults,
            current: currentResults
        )
        
        // Generate report
        let reportFormat: ComparisonTools.ReportFormat
        switch format.lowercased() {
        case "html":
            reportFormat = .html
        case "json":
            reportFormat = .json
        case "csv":
            reportFormat = .csv
        default:
            reportFormat = .markdown
        }
        
        let report = ComparisonTools.generateReport(
            comparison: comparison,
            format: reportFormat
        )
        
        // Output
        if let outputPath = output {
            try report.write(
                to: URL(fileURLWithPath: outputPath),
                atomically: true,
                encoding: .utf8
            )
            print("✅ Comparison saved to: \(outputPath)")
        } else {
            print(report)
        }
    }
}

// MARK: - List Command

struct List: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "List available benchmark results"
    )
    
    @Option(name: .shortAndLong, help: "Results directory")
    var directory: String = "./benchmark-results"
    
    func run() async throws {
        let url = URL(fileURLWithPath: directory)
        let contents = try FileManager.default.contentsOfDirectory(
            at: url,
            includingPropertiesForKeys: [.creationDateKey]
        )
        
        print("📁 Available benchmark results:\n")
        
        let results = try contents
            .filter { $0.hasDirectoryPath }
            .compactMap { dir -> (url: URL, date: Date)? in
                let date = try dir.resourceValues(forKeys: [.creationDateKey]).creationDate
                return date.map { (dir, $0) }
            }
            .sorted { $0.date > $1.date }
        
        for (url, date) in results {
            let name = url.lastPathComponent
            let formatter = DateFormatter()
            formatter.dateStyle = .medium
            formatter.timeStyle = .short
            
            print("  • \(name) - \(formatter.string(from: date))")
            
            // Check for summary
            let summaryPath = url.appendingPathComponent("summary.txt")
            if FileManager.default.fileExists(atPath: summaryPath.path),
               let summary = try? String(contentsOf: summaryPath) {
                if let firstLine = summary.split(separator: "\n").dropFirst(4).first {
                    print("    \(firstLine)")
                }
            }
        }
        
        print("\nTotal: \(results.count) results")
    }
}

// MARK: - Export Command

struct Export: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Export benchmark results in various formats"
    )
    
    @Argument(help: "Results directory to export")
    var results: String
    
    @Option(name: .shortAndLong, help: "Export format: json, csv, html")
    var format: String = "json"
    
    @Option(name: .shortAndLong, help: "Output file path")
    var output: String
    
    func run() async throws {
        print("📤 Exporting benchmark results...")
        
        let resultsUrl = URL(fileURLWithPath: results)
        let resultsPath = resultsUrl.appendingPathComponent("results.json")
        
        guard FileManager.default.fileExists(atPath: resultsPath.path) else {
            throw ExportError.resultsNotFound(results)
        }
        
        let data = try Data(contentsOf: resultsPath)
        
        switch format.lowercased() {
        case "csv":
            let csv = try convertToCSV(data: data)
            try csv.write(
                to: URL(fileURLWithPath: output),
                atomically: true,
                encoding: .utf8
            )
            
        case "html":
            let html = try convertToHTML(data: data)
            try html.write(
                to: URL(fileURLWithPath: output),
                atomically: true,
                encoding: .utf8
            )
            
        default:
            // JSON - just copy
            try data.write(to: URL(fileURLWithPath: output))
        }
        
        print("✅ Results exported to: \(output)")
    }
    
    private func convertToCSV(data: Data) throws -> String {
        let results = try JSONDecoder().decode(
            [String: [String: Double]].self,
            from: data
        )
        
        var csv = "Benchmark,Metric,Value\n"
        
        for (benchmark, metrics) in results.sorted(by: { $0.key < $1.key }) {
            for (metric, value) in metrics.sorted(by: { $0.key < $1.key }) {
                csv += "\"\(benchmark)\",\"\(metric)\",\(value)\n"
            }
        }
        
        return csv
    }
    
    private func convertToHTML(data: Data) throws -> String {
        let results = try JSONDecoder().decode(
            [String: [String: Double]].self,
            from: data
        )
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Results</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <h1>VectorStoreKit Benchmark Results</h1>
            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
                \(generateTableRows(results))
            </table>
        </body>
        </html>
        """
    }
    
    private func generateTableRows(_ results: [String: [String: Double]]) -> String {
        return results.sorted(by: { $0.key < $1.key }).map { benchmark, metrics in
            """
            <tr>
                <td>\(benchmark)</td>
                <td>\(formatValue(metrics["mean"]))</td>
                <td>\(formatValue(metrics["median"]))</td>
                <td>\(formatValue(metrics["stdDev"]))</td>
                <td>\(formatValue(metrics["min"]))</td>
                <td>\(formatValue(metrics["max"]))</td>
            </tr>
            """
        }.joined(separator: "\n")
    }
    
    private func formatValue(_ value: Double?) -> String {
        guard let value = value else { return "-" }
        
        if value < 0.001 {
            return String(format: "%.3e", value)
        } else if value < 1.0 {
            return String(format: "%.3f", value)
        } else {
            return String(format: "%.2f", value)
        }
    }
    
    enum ExportError: LocalizedError {
        case resultsNotFound(String)
        
        var errorDescription: String? {
            switch self {
            case .resultsNotFound(let path):
                return "Results not found at: \(path)"
            }
        }
    }
}

// MARK: - CI Command

struct CI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run benchmarks optimized for CI environments"
    )
    
    @Option(name: .shortAndLong, help: "Baseline results path for regression detection")
    var baseline: String?
    
    @Option(name: .shortAndLong, help: "Maximum allowed regression percentage")
    var regressionThreshold: Double = 5.0
    
    @Flag(name: .long, help: "Fail on any regression")
    var strictMode = false
    
    func run() async throws {
        print("🚀 Running CI benchmarks...")
        
        // Use CI preset configuration
        let config = BenchmarkConfiguration.ci
        
        // Run benchmarks
        let runCommand = Run()
        runCommand.preset = "ci"
        runCommand.output = "./ci-benchmark-results"
        
        try await runCommand.run()
        
        // Compare with baseline if provided
        if let baselinePath = baseline {
            print("\n📊 Checking for regressions...")
            
            let resultsPath = "./ci-benchmark-results/latest/results.json"
            let compareCommand = Compare()
            compareCommand.baseline = baselinePath
            compareCommand.current = resultsPath
            compareCommand.format = "json"
            
            let comparisonOutput = "./ci-benchmark-results/comparison.json"
            compareCommand.output = comparisonOutput
            
            try await compareCommand.run()
            
            // Check for regressions
            let comparisonData = try Data(contentsOf: URL(fileURLWithPath: comparisonOutput))
            let comparison = try JSONDecoder().decode(
                ComparisonTools.ComparisonResult.self,
                from: comparisonData
            )
            
            var hasRegression = false
            var regressionMessages: [String] = []
            
            for metric in comparison.comparisons {
                if strictMode && metric.improvement == .regression {
                    hasRegression = true
                    regressionMessages.append(
                        "❌ Regression in \(metric.name): \(formatPercent(metric.percentChange))"
                    )
                } else if metric.percentChange < -(regressionThreshold / 100) {
                    hasRegression = true
                    regressionMessages.append(
                        "❌ Significant regression in \(metric.name): \(formatPercent(metric.percentChange))"
                    )
                }
            }
            
            if hasRegression {
                print("\n⚠️  Performance regressions detected:")
                for message in regressionMessages {
                    print(message)
                }
                Darwin.exit(1)
            } else {
                print("\n✅ No performance regressions detected")
            }
        }
        
        print("\n✅ CI benchmarks completed successfully")
    }
    
    private func formatPercent(_ value: Double) -> String {
        return String(format: "%+.1f%%", value * 100)
    }
}
// VectorStoreKit: Benchmark Reporter
//
// Various output formats for benchmark results

import Foundation
import simd

/// Base benchmark reporter
open class BenchmarkReporter {
    
    // MARK: - Types
    
    public enum OutputFormat {
        case text
        case json
        case csv
        case markdown
        case html
    }
    
    public struct ReportOptions {
        public let includeSystemInfo: Bool
        public let includeDetailedStats: Bool
        public let compareBaseline: Bool
        public let highlightThresholds: HighlightThresholds?
        
        public struct HighlightThresholds {
            public let excellentSpeedup: Double
            public let goodSpeedup: Double
            public let warningSlowdown: Double
            public let criticalSlowdown: Double
            
            public init(
                excellentSpeedup: Double = 2.0,
                goodSpeedup: Double = 1.2,
                warningSlowdown: Double = 0.8,
                criticalSlowdown: Double = 0.5
            ) {
                self.excellentSpeedup = excellentSpeedup
                self.goodSpeedup = goodSpeedup
                self.warningSlowdown = warningSlowdown
                self.criticalSlowdown = criticalSlowdown
            }
        }
        
        public init(
            includeSystemInfo: Bool = true,
            includeDetailedStats: Bool = true,
            compareBaseline: Bool = false,
            highlightThresholds: HighlightThresholds? = HighlightThresholds()
        ) {
            self.includeSystemInfo = includeSystemInfo
            self.includeDetailedStats = includeDetailedStats
            self.compareBaseline = compareBaseline
            self.highlightThresholds = highlightThresholds
        }
    }
    
    // MARK: - Properties
    
    private let options: ReportOptions
    private var sessionName: String = ""
    private var currentSection: String?
    private var logs: [(timestamp: Date, level: LogLevel, message: String)] = []
    
    public enum LogLevel {
        case info, warning, error, success
    }
    
    // MARK: - Initialization
    
    public init(options: ReportOptions = ReportOptions()) {
        self.options = options
    }
    
    // MARK: - Session Management
    
    open func startSession(_ name: String) {
        sessionName = name
        log("Started benchmark session: \(name)")
    }
    
    open func endSession() {
        log("Completed benchmark session: \(sessionName)")
    }
    
    open func startSection(_ name: String) {
        currentSection = name
        log("Starting section: \(name)")
    }
    
    open func endSection() {
        if let section = currentSection {
            log("Completed section: \(section)")
        }
        currentSection = nil
    }
    
    // MARK: - Logging
    
    open func log(_ message: String, level: LogLevel = .info) {
        logs.append((Date(), level, message))
        print("[\(levelSymbol(level))] \(message)")
    }
    
    open func progress(_ message: String) {
        print("  → \(message)")
    }
    
    private func levelSymbol(_ level: LogLevel) -> String {
        switch level {
        case .info: return "ℹ️"
        case .warning: return "⚠️"
        case .error: return "❌"
        case .success: return "✅"
        }
    }
    
    // MARK: - Report Generation
    
    open func generateReport(
        results: [BenchmarkFramework.Statistics],
        profile: PerformanceMetrics.Profile? = nil
    ) -> String {
        var report = ""
        
        if options.includeSystemInfo {
            report += generateSystemInfo() + "\n\n"
        }
        
        report += generateSummary(results) + "\n\n"
        
        if options.includeDetailedStats {
            report += generateDetailedStats(results) + "\n\n"
        }
        
        if let profile = profile {
            report += generatePerformanceProfile(profile) + "\n\n"
        }
        
        return report
    }
    
    // MARK: - Protected Methods for Subclasses
    
    internal func generateSystemInfo() -> String {
        var info = "# System Information\n"
        info += "- Platform: \(getPlatformInfo())\n"
        info += "- CPU: \(getCPUInfo())\n"
        info += "- Memory: \(getMemoryInfo())\n"
        info += "- Date: \(DateFormatter.localizedString(from: Date(), dateStyle: .full, timeStyle: .full))\n"
        return info
    }
    
    internal func generateSummary(_ results: [BenchmarkFramework.Statistics]) -> String {
        return "# Summary\nOverride in subclass"
    }
    
    internal func generateDetailedStats(_ results: [BenchmarkFramework.Statistics]) -> String {
        return "# Detailed Statistics\nOverride in subclass"
    }
    
    internal func generatePerformanceProfile(_ profile: PerformanceMetrics.Profile) -> String {
        return "# Performance Profile\nOverride in subclass"
    }
    
    // MARK: - System Info Helpers
    
    private func getPlatformInfo() -> String {
        #if os(macOS)
        return "macOS \(ProcessInfo.processInfo.operatingSystemVersionString)"
        #elseif os(iOS)
        return "iOS"
        #elseif os(tvOS)
        return "tvOS"
        #elseif os(watchOS)
        return "watchOS"
        #else
        return "Unknown"
        #endif
    }
    
    private func getCPUInfo() -> String {
        #if arch(arm64)
        return "Apple Silicon (ARM64)"
        #elseif arch(x86_64)
        return "Intel x86_64"
        #else
        return "Unknown"
        #endif
    }
    
    private func getMemoryInfo() -> String {
        let totalMemory = ProcessInfo.processInfo.physicalMemory
        return formatBytes(Int(totalMemory))
    }
    
    // MARK: - Formatting Helpers
    
    internal func formatTime(_ seconds: Double) -> String {
        if seconds < 0.001 {
            return String(format: "%.2f μs", seconds * 1_000_000)
        } else if seconds < 1.0 {
            return String(format: "%.2f ms", seconds * 1_000)
        } else {
            return String(format: "%.2f s", seconds)
        }
    }
    
    internal func formatBytes(_ bytes: Int) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var size = Double(bytes)
        var unitIndex = 0
        
        while size >= 1024 && unitIndex < units.count - 1 {
            size /= 1024
            unitIndex += 1
        }
        
        return String(format: "%.2f %@", size, units[unitIndex])
    }
    
    internal func formatNumber(_ number: Double) -> String {
        if number >= 1_000_000 {
            return String(format: "%.2fM", number / 1_000_000)
        } else if number >= 1_000 {
            return String(format: "%.2fK", number / 1_000)
        } else {
            return String(format: "%.2f", number)
        }
    }
}

// MARK: - Text Reporter

public class TextReporter: BenchmarkReporter {
    
    override public func generateSummary(_ results: [BenchmarkFramework.Statistics]) -> String {
        var summary = "BENCHMARK SUMMARY\n"
        summary += "================\n\n"
        
        for (index, stats) in results.enumerated() {
            summary += "Benchmark #\(index + 1)\n"
            summary += "  Mean: \(formatTime(stats.mean))\n"
            summary += "  Median: \(formatTime(stats.median))\n"
            summary += "  Std Dev: \(formatTime(stats.standardDeviation))\n"
            summary += "  Min: \(formatTime(stats.minimum))\n"
            summary += "  Max: \(formatTime(stats.maximum))\n"
            summary += "  95th %ile: \(formatTime(stats.percentile95))\n"
            summary += "  99th %ile: \(formatTime(stats.percentile99))\n\n"
        }
        
        return summary
    }
}

// MARK: - JSON Reporter

public class JSONReporter: BenchmarkReporter {
    
    override public func generateReport(
        results: [BenchmarkFramework.Statistics],
        profile: PerformanceMetrics.Profile? = nil
    ) -> String {
        let report = JSONReport(
            session: sessionName,
            timestamp: Date(),
            systemInfo: getSystemInfoDict(),
            results: results,
            profile: profile,
            logs: logs.map { JSONReport.LogEntry(timestamp: $0.0, level: $0.1.rawValue, message: $0.2) }
        )
        
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        
        if let data = try? encoder.encode(report) {
            return String(data: data, encoding: .utf8) ?? "{}"
        }
        
        return "{}"
    }
    
    private func getSystemInfoDict() -> [String: String] {
        return [
            "platform": getPlatformInfo(),
            "cpu": getCPUInfo(),
            "memory": getMemoryInfo()
        ]
    }
    
    struct JSONReport: Codable {
        let session: String
        let timestamp: Date
        let systemInfo: [String: String]
        let results: [BenchmarkFramework.Statistics]
        let profile: PerformanceMetrics.Profile?
        let logs: [LogEntry]
        
        struct LogEntry: Codable {
            let timestamp: Date
            let level: Int
            let message: String
        }
    }
}

// MARK: - CSV Reporter

public class CSVReporter: BenchmarkReporter {
    
    override public func generateReport(
        results: [BenchmarkFramework.Statistics],
        profile: PerformanceMetrics.Profile? = nil
    ) -> String {
        var csv = "Benchmark,Mean,Median,StdDev,Min,Max,P95,P99,Count\n"
        
        for (index, stats) in results.enumerated() {
            csv += "\(index),\(stats.mean),\(stats.median),\(stats.standardDeviation),"
            csv += "\(stats.minimum),\(stats.maximum),\(stats.percentile95),\(stats.percentile99),"
            csv += "\(stats.count)\n"
        }
        
        return csv
    }
}

// MARK: - Markdown Reporter

public class MarkdownReporter: BenchmarkReporter {
    
    override public func generateSummary(_ results: [BenchmarkFramework.Statistics]) -> String {
        var markdown = "## Benchmark Summary\n\n"
        
        markdown += "| Metric | "
        for i in 0..<results.count {
            markdown += "Benchmark \(i + 1) | "
        }
        markdown += "\n"
        
        markdown += "|--------|"
        for _ in 0..<results.count {
            markdown += "------------|"
        }
        markdown += "\n"
        
        // Add rows for each metric
        let metrics = [
            ("Mean", { (s: BenchmarkFramework.Statistics) in self.formatTime(s.mean) }),
            ("Median", { (s: BenchmarkFramework.Statistics) in self.formatTime(s.median) }),
            ("Std Dev", { (s: BenchmarkFramework.Statistics) in self.formatTime(s.standardDeviation) }),
            ("Min", { (s: BenchmarkFramework.Statistics) in self.formatTime(s.minimum) }),
            ("Max", { (s: BenchmarkFramework.Statistics) in self.formatTime(s.maximum) }),
            ("95th %ile", { (s: BenchmarkFramework.Statistics) in self.formatTime(s.percentile95) }),
            ("99th %ile", { (s: BenchmarkFramework.Statistics) in self.formatTime(s.percentile99) })
        ]
        
        for (name, extractor) in metrics {
            markdown += "| \(name) | "
            for stats in results {
                markdown += "\(extractor(stats)) | "
            }
            markdown += "\n"
        }
        
        return markdown
    }
    
    override public func generatePerformanceProfile(_ profile: PerformanceMetrics.Profile) -> String {
        var markdown = "## Performance Profile: \(profile.name)\n\n"
        markdown += "- **Duration**: \(formatTime(profile.duration))\n"
        markdown += "- **Start Time**: \(ISO8601DateFormatter().string(from: profile.startTime))\n"
        markdown += "- **End Time**: \(ISO8601DateFormatter().string(from: profile.endTime))\n\n"
        
        if !profile.metrics.isEmpty {
            markdown += "### Metrics\n\n"
            markdown += "| Metric | Samples | Mean | P50 | P90 | P95 | P99 |\n"
            markdown += "|--------|---------|------|-----|-----|-----|-----|\n"
            
            for (name, timeSeries) in profile.metrics.sorted(by: { $0.key < $1.key }) {
                let stats = timeSeries.statistics
                markdown += "| \(name) | \(timeSeries.samples.count) | "
                markdown += "\(formatValue(stats.mean, unit: timeSeries.samples.first?.unit ?? "")) | "
                markdown += "\(formatValue(stats.p50, unit: timeSeries.samples.first?.unit ?? "")) | "
                markdown += "\(formatValue(stats.p90, unit: timeSeries.samples.first?.unit ?? "")) | "
                markdown += "\(formatValue(stats.p95, unit: timeSeries.samples.first?.unit ?? "")) | "
                markdown += "\(formatValue(stats.p99, unit: timeSeries.samples.first?.unit ?? "")) |\n"
            }
        }
        
        return markdown
    }
    
    private func formatValue(_ value: Double, unit: String) -> String {
        switch unit {
        case "seconds", "s":
            return formatTime(value)
        case "bytes", "B":
            return formatBytes(Int(value))
        case "ops/sec", "ops/s":
            return formatNumber(value)
        case "ratio", "%":
            return String(format: "%.2f%%", value * 100)
        default:
            return String(format: "%.2f", value)
        }
    }
}

// MARK: - HTML Reporter

public class HTMLReporter: BenchmarkReporter {
    
    override public func generateReport(
        results: [BenchmarkFramework.Statistics],
        profile: PerformanceMetrics.Profile? = nil
    ) -> String {
        var html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>VectorStoreKit Benchmark Report</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; font-weight: 600; }
                .good { color: #28a745; }
                .warning { color: #ffc107; }
                .bad { color: #dc3545; }
                .chart { margin: 20px 0; }
                pre { background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <h1>VectorStoreKit Benchmark Report</h1>
        """
        
        if options.includeSystemInfo {
            html += "<h2>System Information</h2>"
            html += "<pre>\(generateSystemInfo())</pre>"
        }
        
        html += "<h2>Results Summary</h2>"
        html += generateHTMLTable(results)
        
        if let profile = profile {
            html += "<h2>Performance Profile</h2>"
            html += generateHTMLProfile(profile)
        }
        
        html += """
        </body>
        </html>
        """
        
        return html
    }
    
    private func generateHTMLTable(_ results: [BenchmarkFramework.Statistics]) -> String {
        var table = "<table>"
        table += "<tr><th>Benchmark</th><th>Mean</th><th>Median</th><th>Std Dev</th><th>Min</th><th>Max</th><th>P95</th><th>P99</th></tr>"
        
        for (index, stats) in results.enumerated() {
            table += "<tr>"
            table += "<td>Benchmark \(index + 1)</td>"
            table += "<td>\(formatTime(stats.mean))</td>"
            table += "<td>\(formatTime(stats.median))</td>"
            table += "<td>\(formatTime(stats.standardDeviation))</td>"
            table += "<td>\(formatTime(stats.minimum))</td>"
            table += "<td>\(formatTime(stats.maximum))</td>"
            table += "<td>\(formatTime(stats.percentile95))</td>"
            table += "<td>\(formatTime(stats.percentile99))</td>"
            table += "</tr>"
        }
        
        table += "</table>"
        return table
    }
    
    private func generateHTMLProfile(_ profile: PerformanceMetrics.Profile) -> String {
        var html = "<div class='profile'>"
        html += "<p><strong>Duration:</strong> \(formatTime(profile.duration))</p>"
        html += "<p><strong>Metrics:</strong> \(profile.metrics.count)</p>"
        html += "</div>"
        return html
    }
}

// MARK: - Extensions for LogLevel

extension BenchmarkReporter.LogLevel {
    var rawValue: Int {
        switch self {
        case .info: return 0
        case .warning: return 1
        case .error: return 2
        case .success: return 3
        }
    }
}
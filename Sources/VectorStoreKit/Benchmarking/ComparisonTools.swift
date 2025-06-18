// VectorStoreKit: Comparison Tools
//
// Compare results across benchmark runs

import Foundation

/// Tools for comparing benchmark results
public struct ComparisonTools {
    
    // MARK: - Types
    
    public struct ComparisonResult: Codable {
        public let baseline: BenchmarkRun
        public let current: BenchmarkRun
        public let comparisons: [MetricComparison]
        public let summary: Summary
        
        public struct BenchmarkRun: Codable {
            public let id: String
            public let timestamp: Date
            public let configuration: String
            public let results: [String: Double]
        }
        
        public struct MetricComparison: Codable {
            public let name: String
            public let baselineValue: Double
            public let currentValue: Double
            public let difference: Double
            public let percentChange: Double
            public let improvement: ImprovementLevel
            public let statisticallySignificant: Bool
            public let pValue: Double?
        }
        
        public enum ImprovementLevel: String, Codable {
            case regression = "regression"
            case noChange = "no_change"
            case minorImprovement = "minor_improvement"
            case improvement = "improvement"
            case majorImprovement = "major_improvement"
        }
        
        public struct Summary: Codable {
            public let totalMetrics: Int
            public let improvements: Int
            public let regressions: Int
            public let unchanged: Int
            public let overallImprovement: Double
            public let significantChanges: Int
        }
    }
    
    public struct ThresholdConfig: Sendable {
        public let regressionThreshold: Double
        public let improvementThreshold: Double
        public let majorImprovementThreshold: Double
        public let significanceLevel: Double
        
        public init(
            regressionThreshold: Double = -0.05, // 5% slower
            improvementThreshold: Double = 0.05, // 5% faster
            majorImprovementThreshold: Double = 0.20, // 20% faster
            significanceLevel: Double = 0.05 // p < 0.05
        ) {
            self.regressionThreshold = regressionThreshold
            self.improvementThreshold = improvementThreshold
            self.majorImprovementThreshold = majorImprovementThreshold
            self.significanceLevel = significanceLevel
        }
        
        public static let strict = ThresholdConfig(
            regressionThreshold: -0.01,
            improvementThreshold: 0.01,
            majorImprovementThreshold: 0.10
        )
        
        public static let standard = ThresholdConfig()
        
        public static let lenient = ThresholdConfig(
            regressionThreshold: -0.10,
            improvementThreshold: 0.10,
            majorImprovementThreshold: 0.30
        )
    }
    
    // MARK: - Comparison Methods
    
    /// Compare two benchmark runs
    public static func compare(
        baseline: BenchmarkResults,
        current: BenchmarkResults,
        thresholds: ThresholdConfig = .standard
    ) -> ComparisonResult {
        let baselineRun = ComparisonResult.BenchmarkRun(
            id: baseline.id,
            timestamp: baseline.timestamp,
            configuration: baseline.configuration.general.name,
            results: flattenResults(baseline.results)
        )
        
        let currentRun = ComparisonResult.BenchmarkRun(
            id: current.id,
            timestamp: current.timestamp,
            configuration: current.configuration.general.name,
            results: flattenResults(current.results)
        )
        
        let comparisons = compareMetrics(
            baseline: baselineRun.results,
            current: currentRun.results,
            thresholds: thresholds
        )
        
        let summary = summarizeComparisons(comparisons)
        
        return ComparisonResult(
            baseline: baselineRun,
            current: currentRun,
            comparisons: comparisons,
            summary: summary
        )
    }
    
    /// Compare multiple benchmark runs
    public static func compareMultiple(
        runs: [BenchmarkResults],
        baselineIndex: Int = 0
    ) -> [ComparisonResult] {
        guard runs.count > 1, baselineIndex < runs.count else { return [] }
        
        let baseline = runs[baselineIndex]
        var comparisons: [ComparisonResult] = []
        
        for (index, run) in runs.enumerated() {
            if index != baselineIndex {
                comparisons.append(compare(baseline: baseline, current: run))
            }
        }
        
        return comparisons
    }
    
    /// Find best performing configuration
    public static func findBest(
        runs: [BenchmarkResults],
        metric: String
    ) -> BenchmarkResults? {
        return runs.max { run1, run2 in
            let value1 = flattenResults(run1.results)[metric] ?? .infinity
            let value2 = flattenResults(run2.results)[metric] ?? .infinity
            return value1 > value2 // Lower is better for most metrics
        }
    }
    
    /// Generate performance trends
    public static func analyzeTrends(
        runs: [BenchmarkResults],
        metric: String
    ) -> BenchmarkTrendAnalysis? {
        guard runs.count >= 3 else { return nil }
        
        let sortedRuns = runs.sorted { $0.timestamp < $1.timestamp }
        let values = sortedRuns.compactMap { run in
            flattenResults(run.results)[metric]
        }
        
        guard values.count == sortedRuns.count else { return nil }
        
        // Calculate linear regression
        let n = Double(values.count)
        let x = Array(0..<values.count).map { Double($0) }
        let y = values
        
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumX2 = x.map { $0 * $0 }.reduce(0, +)
        
        let slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
        let intercept = (sumY - slope * sumX) / n
        
        // Calculate R-squared
        let yMean = sumY / n
        let ssTotal = y.map { pow($0 - yMean, 2) }.reduce(0, +)
        let ssResidual = zip(x, y).map { x, y in
            let predicted = slope * x + intercept
            return pow(y - predicted, 2)
        }.reduce(0, +)
        let rSquared = 1 - (ssResidual / ssTotal)
        
        return BenchmarkTrendAnalysis(
            metric: metric,
            dataPoints: zip(sortedRuns.map { $0.timestamp }, values).map { (timestamp: $0, value: $1) },
            slope: slope,
            intercept: intercept,
            rSquared: rSquared,
            trend: categorizeTrend(slope: slope, values: values)
        )
    }
    
    // MARK: - Reporting
    
    /// Generate comparison report
    public static func generateReport(
        comparison: ComparisonResult,
        format: ReportFormat = .markdown
    ) -> String {
        switch format {
        case .markdown:
            return generateMarkdownReport(comparison)
        case .html:
            return generateHTMLReport(comparison)
        case .json:
            return generateJSONReport(comparison)
        case .csv:
            return generateCSVReport(comparison)
        }
    }
    
    public enum ReportFormat {
        case markdown
        case html
        case json
        case csv
    }
    
    // MARK: - Statistical Analysis
    
    /// Perform statistical significance test
    public static func testSignificance(
        baseline: [Double],
        current: [Double],
        alpha: Double = 0.05
    ) -> (significant: Bool, pValue: Double) {
        guard baseline.count >= 2 && current.count >= 2 else {
            return (false, 1.0)
        }
        
        // Welch's t-test
        let n1 = Double(baseline.count)
        let n2 = Double(current.count)
        
        let mean1 = baseline.reduce(0, +) / n1
        let mean2 = current.reduce(0, +) / n2
        
        let var1 = baseline.map { pow($0 - mean1, 2) }.reduce(0, +) / (n1 - 1)
        let var2 = current.map { pow($0 - mean2, 2) }.reduce(0, +) / (n2 - 1)
        
        let se = sqrt(var1 / n1 + var2 / n2)
        let t = (mean1 - mean2) / se
        
        // Approximate degrees of freedom (Welch-Satterthwaite equation)
        let df = pow(var1 / n1 + var2 / n2, 2) /
            (pow(var1 / n1, 2) / (n1 - 1) + pow(var2 / n2, 2) / (n2 - 1))
        
        // Approximate p-value
        let pValue = approximatePValue(t: abs(t), df: df)
        
        return (pValue < alpha, pValue)
    }
    
    // MARK: - Private Helpers
    
    private static func flattenResults(_ results: [String: Any]) -> [String: Double] {
        var flattened: [String: Double] = [:]
        
        for (key, value) in results {
            if let doubleValue = value as? Double {
                flattened[key] = doubleValue
            } else if let stats = value as? BenchmarkFramework.Statistics {
                flattened["\(key).mean"] = stats.mean
                flattened["\(key).median"] = stats.median
                flattened["\(key).p95"] = stats.percentile95
                flattened["\(key).p99"] = stats.percentile99
            }
        }
        
        return flattened
    }
    
    private static func compareMetrics(
        baseline: [String: Double],
        current: [String: Double],
        thresholds: ThresholdConfig
    ) -> [ComparisonResult.MetricComparison] {
        var comparisons: [ComparisonResult.MetricComparison] = []
        
        let allMetrics = Set(baseline.keys).union(Set(current.keys))
        
        for metric in allMetrics.sorted() {
            let baselineValue = baseline[metric] ?? 0
            let currentValue = current[metric] ?? 0
            
            guard baselineValue > 0 else { continue }
            
            let difference = currentValue - baselineValue
            let percentChange = difference / baselineValue
            
            let improvement: ComparisonResult.ImprovementLevel
            if percentChange <= thresholds.regressionThreshold {
                improvement = .regression
            } else if percentChange >= thresholds.majorImprovementThreshold {
                improvement = .majorImprovement
            } else if percentChange >= thresholds.improvementThreshold {
                improvement = .improvement
            } else if abs(percentChange) < 0.01 {
                improvement = .noChange
            } else {
                improvement = .minorImprovement
            }
            
            // For metrics where lower is better (time, memory)
            let actualImprovement = metric.contains("time") || metric.contains("memory") ?
                invertImprovement(improvement) : improvement
            
            comparisons.append(ComparisonResult.MetricComparison(
                name: metric,
                baselineValue: baselineValue,
                currentValue: currentValue,
                difference: difference,
                percentChange: percentChange,
                improvement: actualImprovement,
                statisticallySignificant: false, // Would need multiple samples
                pValue: nil
            ))
        }
        
        return comparisons
    }
    
    private static func summarizeComparisons(
        _ comparisons: [ComparisonResult.MetricComparison]
    ) -> ComparisonResult.Summary {
        let improvements = comparisons.filter {
            $0.improvement == .improvement || $0.improvement == .majorImprovement
        }.count
        
        let regressions = comparisons.filter {
            $0.improvement == .regression
        }.count
        
        let unchanged = comparisons.filter {
            $0.improvement == .noChange
        }.count
        
        let significantChanges = comparisons.filter {
            $0.statisticallySignificant
        }.count
        
        let overallImprovement = comparisons
            .map { $0.percentChange }
            .reduce(0, +) / Double(comparisons.count)
        
        return ComparisonResult.Summary(
            totalMetrics: comparisons.count,
            improvements: improvements,
            regressions: regressions,
            unchanged: unchanged,
            overallImprovement: overallImprovement,
            significantChanges: significantChanges
        )
    }
    
    private static func invertImprovement(
        _ improvement: ComparisonResult.ImprovementLevel
    ) -> ComparisonResult.ImprovementLevel {
        switch improvement {
        case .regression:
            return .majorImprovement
        case .noChange:
            return .noChange
        case .minorImprovement:
            return .minorImprovement
        case .improvement:
            return .regression
        case .majorImprovement:
            return .regression
        }
    }
    
    private static func categorizeTrend(slope: Double, values: [Double]) -> TrendType {
        let avgValue = values.reduce(0, +) / Double(values.count)
        let normalizedSlope = slope / avgValue
        
        if abs(normalizedSlope) < 0.01 {
            return .stable
        } else if normalizedSlope < -0.05 {
            return .improving // Lower is better
        } else if normalizedSlope > 0.05 {
            return .degrading
        } else {
            return normalizedSlope < 0 ? .slightlyImproving : .slightlyDegrading
        }
    }
    
    private static func approximatePValue(t: Double, df: Double) -> Double {
        // Simplified p-value approximation
        if t < 1.0 {
            return 1.0
        } else if t < 2.0 {
            return 0.1
        } else if t < 2.5 {
            return 0.05
        } else if t < 3.0 {
            return 0.01
        } else {
            return 0.001
        }
    }
    
    // MARK: - Report Generation
    
    private static func generateMarkdownReport(_ comparison: ComparisonResult) -> String {
        var report = """
        # Benchmark Comparison Report
        
        ## Summary
        
        - **Baseline**: \(comparison.baseline.id) (\(formatDate(comparison.baseline.timestamp)))
        - **Current**: \(comparison.current.id) (\(formatDate(comparison.current.timestamp)))
        - **Total Metrics**: \(comparison.summary.totalMetrics)
        - **Improvements**: \(comparison.summary.improvements) (\(formatPercent(Double(comparison.summary.improvements) / Double(comparison.summary.totalMetrics))))
        - **Regressions**: \(comparison.summary.regressions) (\(formatPercent(Double(comparison.summary.regressions) / Double(comparison.summary.totalMetrics))))
        - **Overall Change**: \(formatPercent(comparison.summary.overallImprovement))
        
        ## Detailed Comparison
        
        | Metric | Baseline | Current | Change | Status |
        |--------|----------|---------|--------|--------|
        """
        
        for comp in comparison.comparisons.sorted(by: { abs($0.percentChange) > abs($1.percentChange) }) {
            let status = formatStatus(comp.improvement)
            report += "| \(comp.name) | \(formatValue(comp.baselineValue)) | \(formatValue(comp.currentValue)) | \(formatPercent(comp.percentChange)) | \(status) |\n"
        }
        
        return report
    }
    
    private static func generateHTMLReport(_ comparison: ComparisonResult) -> String {
        // Simplified HTML report
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Comparison</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
                .improvement { color: green; }
                .regression { color: red; }
                .unchanged { color: gray; }
            </style>
        </head>
        <body>
            <h1>Benchmark Comparison Report</h1>
            <!-- Report content would go here -->
        </body>
        </html>
        """
    }
    
    private static func generateJSONReport(_ comparison: ComparisonResult) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        
        if let data = try? encoder.encode(comparison) {
            return String(data: data, encoding: .utf8) ?? "{}"
        }
        return "{}"
    }
    
    private static func generateCSVReport(_ comparison: ComparisonResult) -> String {
        var csv = "Metric,Baseline,Current,Difference,Percent Change,Status\n"
        
        for comp in comparison.comparisons {
            csv += "\(comp.name),\(comp.baselineValue),\(comp.currentValue),\(comp.difference),\(comp.percentChange),\(comp.improvement.rawValue)\n"
        }
        
        return csv
    }
    
    // MARK: - Formatting Helpers
    
    private static func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
    
    private static func formatPercent(_ value: Double) -> String {
        return String(format: "%+.1f%%", value * 100)
    }
    
    private static func formatValue(_ value: Double) -> String {
        if value < 0.001 {
            return String(format: "%.3e", value)
        } else if value < 1.0 {
            return String(format: "%.3f", value)
        } else if value < 1000 {
            return String(format: "%.2f", value)
        } else {
            return String(format: "%.1e", value)
        }
    }
    
    private static func formatStatus(_ improvement: ComparisonResult.ImprovementLevel) -> String {
        switch improvement {
        case .regression:
            return "⚠️ Regression"
        case .noChange:
            return "➖ No Change"
        case .minorImprovement:
            return "➕ Minor Improvement"
        case .improvement:
            return "✅ Improvement"
        case .majorImprovement:
            return "🚀 Major Improvement"
        }
    }
}

// MARK: - Supporting Types

public struct BenchmarkResults: Codable {
    public let id: String
    public let timestamp: Date
    public let configuration: BenchmarkConfiguration
    public let results: [String: Any]
    
    public init(
        id: String = UUID().uuidString,
        timestamp: Date = Date(),
        configuration: BenchmarkConfiguration,
        results: [String: Any]
    ) {
        self.id = id
        self.timestamp = timestamp
        self.configuration = configuration
        self.results = results
    }
    
    // Custom Codable implementation
    private enum CodingKeys: String, CodingKey {
        case id, timestamp, configuration, results
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(configuration, forKey: .configuration)
        
        // Convert Any to JSON-serializable values
        let resultsData = try JSONSerialization.data(withJSONObject: results)
        let resultsString = String(data: resultsData, encoding: .utf8)!
        try container.encode(resultsString, forKey: .results)
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        timestamp = try container.decode(Date.self, forKey: .timestamp)
        configuration = try container.decode(BenchmarkConfiguration.self, forKey: .configuration)
        
        // Decode results from JSON string
        let resultsString = try container.decode(String.self, forKey: .results)
        let resultsData = resultsString.data(using: .utf8)!
        results = try JSONSerialization.jsonObject(with: resultsData) as! [String: Any]
    }
}

public struct BenchmarkTrendAnalysis {
    public let metric: String
    public let dataPoints: [(timestamp: Date, value: Double)]
    public let slope: Double
    public let intercept: Double
    public let rSquared: Double
    public let trend: TrendType
}

public enum TrendType {
    case improving
    case slightlyImproving
    case stable
    case slightlyDegrading
    case degrading
}
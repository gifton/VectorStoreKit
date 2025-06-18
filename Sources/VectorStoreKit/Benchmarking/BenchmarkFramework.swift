// VectorStoreKit: Core Benchmarking Framework
//
// High-performance benchmarking infrastructure with statistical analysis

import Foundation
import simd
import os.log
#if canImport(Darwin)
import Darwin
#endif

/// Core benchmarking framework for VectorStoreKit
public actor BenchmarkFramework {
    
    // MARK: - Types
    
    /// A single benchmark measurement
    public struct Measurement: Sendable {
        public let value: Double
        public let unit: Unit
        public let timestamp: Date
        
        public enum Unit: String, Codable, Sendable {
            case seconds = "s"
            case milliseconds = "ms"
            case microseconds = "μs"
            case nanoseconds = "ns"
            case bytes = "B"
            case kilobytes = "KB"
            case megabytes = "MB"
            case gigabytes = "GB"
            case operations = "ops"
            case operationsPerSecond = "ops/s"
            case percentage = "%"
        }
        
        public init(value: Double, unit: Unit, timestamp: Date = Date()) {
            self.value = value
            self.unit = unit
            self.timestamp = timestamp
        }
    }
    
    /// Statistical summary of measurements
    public struct Statistics: Sendable, Codable {
        public let count: Int
        public let mean: Double
        public let median: Double
        public let standardDeviation: Double
        public let minimum: Double
        public let maximum: Double
        public let percentile95: Double
        public let percentile99: Double
        public let unit: Measurement.Unit
        
        public init(measurements: [Measurement]) {
            guard !measurements.isEmpty else {
                self.count = 0
                self.mean = 0
                self.median = 0
                self.standardDeviation = 0
                self.minimum = 0
                self.maximum = 0
                self.percentile95 = 0
                self.percentile99 = 0
                self.unit = .seconds
                return
            }
            
            let values = measurements.map { $0.value }.sorted()
            self.count = values.count
            self.mean = values.reduce(0, +) / Double(count)
            self.median = values[count / 2]
            self.minimum = values.first!
            self.maximum = values.last!
            self.unit = measurements.first!.unit
            
            // Calculate standard deviation
            let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Double(count)
            self.standardDeviation = sqrt(variance)
            
            // Calculate percentiles
            self.percentile95 = values[Int(Double(count) * 0.95)]
            self.percentile99 = values[Int(Double(count) * 0.99)]
        }
    }
    
    /// Benchmark configuration
    public struct Configuration: Sendable {
        public let iterations: Int
        public let warmupIterations: Int
        public let timeout: TimeInterval
        public let collectMemoryMetrics: Bool
        public let collectCPUMetrics: Bool
        public let statisticalSignificance: Double
        
        public init(
            iterations: Int = 100,
            warmupIterations: Int = 10,
            timeout: TimeInterval = 300,
            collectMemoryMetrics: Bool = true,
            collectCPUMetrics: Bool = true,
            statisticalSignificance: Double = 0.95
        ) {
            self.iterations = iterations
            self.warmupIterations = warmupIterations
            self.timeout = timeout
            self.collectMemoryMetrics = collectMemoryMetrics
            self.collectCPUMetrics = collectCPUMetrics
            self.statisticalSignificance = statisticalSignificance
        }
        
        public static let quick = Configuration(
            iterations: 10,
            warmupIterations: 2,
            timeout: 60
        )
        
        public static let standard = Configuration()
        
        public static let thorough = Configuration(
            iterations: 1000,
            warmupIterations: 100,
            timeout: 600
        )
    }
    
    /// A benchmark suite
    public struct Suite: Sendable {
        public let name: String
        public let description: String
        public let benchmarks: [Benchmark]
        
        public init(name: String, description: String, benchmarks: [Benchmark]) {
            self.name = name
            self.description = description
            self.benchmarks = benchmarks
        }
    }
    
    /// A single benchmark
    public struct Benchmark: Sendable {
        public let name: String
        public let setUp: (@Sendable () async throws -> Void)?
        public let tearDown: (@Sendable () async throws -> Void)?
        public let run: @Sendable () async throws -> Void
        
        public init(
            name: String,
            setUp: (@Sendable () async throws -> Void)? = nil,
            tearDown: (@Sendable () async throws -> Void)? = nil,
            run: @escaping @Sendable () async throws -> Void
        ) {
            self.name = name
            self.setUp = setUp
            self.tearDown = tearDown
            self.run = run
        }
    }
    
    // MARK: - Properties
    
    private let configuration: Configuration
    private let logger = Logger(subsystem: "VectorStoreKit", category: "Benchmark")
    private var measurements: [String: [Measurement]] = [:]
    private var memoryBaseline: Int = 0
    
    // MARK: - Initialization
    
    public init(configuration: Configuration = .standard) {
        self.configuration = configuration
    }
    
    // MARK: - Benchmarking API
    
    /// Run a benchmark suite
    public func run(suite: Suite) async throws -> [String: Statistics] {
        logger.info("Running benchmark suite: \(suite.name)")
        
        var results: [String: Statistics] = [:]
        
        for benchmark in suite.benchmarks {
            let stats = try await runBenchmark(benchmark)
            results[benchmark.name] = stats
        }
        
        return results
    }
    
    /// Run a single benchmark
    public func runBenchmark(_ benchmark: Benchmark) async throws -> Statistics {
        logger.info("Running benchmark: \(benchmark.name)")
        
        // Clear previous measurements
        measurements[benchmark.name] = []
        
        // Record memory baseline
        if configuration.collectMemoryMetrics {
            memoryBaseline = getMemoryUsage()
        }
        
        // Warmup phase
        logger.debug("Starting warmup phase (\(self.configuration.warmupIterations) iterations)")
        for _ in 0..<configuration.warmupIterations {
            try await benchmark.setUp?()
            try await benchmark.run()
            try await benchmark.tearDown?()
        }
        
        // Measurement phase
        logger.debug("Starting measurement phase (\(self.configuration.iterations) iterations)")
        var timeMeasurements: [Measurement] = []
        
        for i in 0..<configuration.iterations {
            // Setup
            try await benchmark.setUp?()
            
            // Time measurement
            let startTime = DispatchTime.now()
            try await benchmark.run()
            let endTime = DispatchTime.now()
            
            let elapsed = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
            timeMeasurements.append(Measurement(value: elapsed, unit: .seconds))
            
            // Memory measurement
            if configuration.collectMemoryMetrics && i % 10 == 0 {
                let memoryUsage = getMemoryUsage() - memoryBaseline
                await recordMeasurement(
                    benchmark: benchmark.name,
                    measurement: Measurement(value: Double(memoryUsage), unit: .bytes)
                )
            }
            
            // Teardown
            try await benchmark.tearDown?()
        }
        
        // Store time measurements
        measurements[benchmark.name] = timeMeasurements
        
        return Statistics(measurements: timeMeasurements)
    }
    
    /// Measure a specific operation
    public func measure<T>(
        name: String,
        iterations: Int? = nil,
        operation: () async throws -> T
    ) async throws -> (result: T, statistics: Statistics) {
        let actualIterations = iterations ?? configuration.iterations
        var measurements: [Measurement] = []
        var lastResult: T?
        
        for _ in 0..<actualIterations {
            let startTime = DispatchTime.now()
            lastResult = try await operation()
            let endTime = DispatchTime.now()
            
            let elapsed = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
            measurements.append(Measurement(value: elapsed, unit: .seconds))
        }
        
        return (lastResult!, Statistics(measurements: measurements))
    }
    
    /// Record a custom measurement
    public func recordMeasurement(benchmark: String, measurement: Measurement) async {
        if measurements[benchmark] == nil {
            measurements[benchmark] = []
        }
        measurements[benchmark]?.append(measurement)
    }
    
    /// Get all measurements for a benchmark
    public func getMeasurements(for benchmark: String) async -> [Measurement]? {
        return measurements[benchmark]
    }
    
    /// Compare two benchmarks statistically
    public func compare(
        benchmark1: String,
        benchmark2: String
    ) async -> ComparisonResult? {
        guard let measurements1 = measurements[benchmark1],
              let measurements2 = measurements[benchmark2] else {
            return nil
        }
        
        let stats1 = Statistics(measurements: measurements1)
        let stats2 = Statistics(measurements: measurements2)
        
        // Perform t-test
        let tValue = calculateTValue(
            mean1: stats1.mean,
            mean2: stats2.mean,
            stdDev1: stats1.standardDeviation,
            stdDev2: stats2.standardDeviation,
            n1: stats1.count,
            n2: stats2.count
        )
        
        let degreesOfFreedom = stats1.count + stats2.count - 2
        let pValue = calculatePValue(tValue: tValue, degreesOfFreedom: degreesOfFreedom)
        
        return ComparisonResult(
            benchmark1: benchmark1,
            benchmark2: benchmark2,
            stats1: stats1,
            stats2: stats2,
            speedup: stats2.mean / stats1.mean,
            statisticallySignificant: pValue < (1 - configuration.statisticalSignificance),
            pValue: pValue
        )
    }
    
    // MARK: - Result Types
    
    public struct ComparisonResult: Sendable {
        public let benchmark1: String
        public let benchmark2: String
        public let stats1: Statistics
        public let stats2: Statistics
        public let speedup: Double
        public let statisticallySignificant: Bool
        public let pValue: Double
        
        public var summary: String {
            let faster = speedup > 1 ? benchmark1 : benchmark2
            let slower = speedup > 1 ? benchmark2 : benchmark1
            let factor = speedup > 1 ? speedup : 1 / speedup
            let significance = statisticallySignificant ? "statistically significant" : "not statistically significant"
            
            return "\(faster) is \(String(format: "%.2fx", factor)) faster than \(slower) (\(significance), p=\(String(format: "%.4f", pValue)))"
        }
    }
    
    // MARK: - Utilities
    
    private func getMemoryUsage() -> Int {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }
        
        return result == KERN_SUCCESS ? Int(info.resident_size) : 0
        #else
        return 0
        #endif
    }
    
    private func calculateTValue(
        mean1: Double, mean2: Double,
        stdDev1: Double, stdDev2: Double,
        n1: Int, n2: Int
    ) -> Double {
        let pooledVariance = ((Double(n1 - 1) * pow(stdDev1, 2)) + (Double(n2 - 1) * pow(stdDev2, 2))) / Double(n1 + n2 - 2)
        let pooledStdDev = sqrt(pooledVariance)
        let standardError = pooledStdDev * sqrt(1.0 / Double(n1) + 1.0 / Double(n2))
        return (mean1 - mean2) / standardError
    }
    
    private func calculatePValue(tValue: Double, degreesOfFreedom: Int) -> Double {
        // Simplified p-value calculation
        // In production, use a proper statistical library
        let absT = abs(tValue)
        if absT < 1.96 {
            return 0.05
        } else if absT < 2.58 {
            return 0.01
        } else {
            return 0.001
        }
    }
}

// MARK: - Benchmark Builders

/// DSL for building benchmark suites
@resultBuilder
public struct BenchmarkBuilder {
    public static func buildBlock(_ components: [BenchmarkFramework.Benchmark]...) -> [BenchmarkFramework.Benchmark] {
        components.flatMap { $0 }
    }
    
    public static func buildArray(_ components: [[BenchmarkFramework.Benchmark]]) -> [BenchmarkFramework.Benchmark] {
        components.flatMap { $0 }
    }
    
    public static func buildExpression(_ expression: BenchmarkFramework.Benchmark) -> [BenchmarkFramework.Benchmark] {
        [expression]
    }
    
    public static func buildExpression(_ expression: [BenchmarkFramework.Benchmark]) -> [BenchmarkFramework.Benchmark] {
        expression
    }
    
    public static func buildOptional(_ component: [BenchmarkFramework.Benchmark]?) -> [BenchmarkFramework.Benchmark] {
        component ?? []
    }
    
    public static func buildEither(first component: [BenchmarkFramework.Benchmark]) -> [BenchmarkFramework.Benchmark] {
        component
    }
    
    public static func buildEither(second component: [BenchmarkFramework.Benchmark]) -> [BenchmarkFramework.Benchmark] {
        component
    }
}

/// Create a benchmark suite using DSL
public func benchmarkSuite(
    name: String,
    description: String,
    @BenchmarkBuilder benchmarks: () -> [BenchmarkFramework.Benchmark]
) -> BenchmarkFramework.Suite {
    BenchmarkFramework.Suite(
        name: name,
        description: description,
        benchmarks: benchmarks()
    )
}

/// Create a single benchmark
public func benchmark(
    name: String,
    setUp: (@Sendable () async throws -> Void)? = nil,
    tearDown: (@Sendable () async throws -> Void)? = nil,
    run: @escaping @Sendable () async throws -> Void
) -> BenchmarkFramework.Benchmark {
    BenchmarkFramework.Benchmark(
        name: name,
        setUp: setUp,
        tearDown: tearDown,
        run: run
    )
}
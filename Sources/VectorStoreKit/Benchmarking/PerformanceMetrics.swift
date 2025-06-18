// VectorStoreKit: Performance Metrics Collection
//
// Comprehensive metrics collection and analysis for benchmarking

import Foundation
import simd
import os.log
#if canImport(Darwin)
import Darwin
import QuartzCore
#endif

/// Performance metrics collector
public actor PerformanceMetrics {
    
    // MARK: - Types
    
    /// Metric types that can be collected
    public enum MetricType: String, CaseIterable, Codable, Sendable {
        case latency = "latency"
        case throughput = "throughput"
        case memory = "memory"
        case cpu = "cpu"
        case gpu = "gpu"
        case diskIO = "disk_io"
        case networkIO = "network_io"
        case cacheHitRate = "cache_hit_rate"
        case accuracy = "accuracy"
        case custom = "custom"
    }
    
    /// A performance metric sample
    public struct Sample: Sendable, Codable {
        public let timestamp: Date
        public let type: MetricType
        public let name: String
        public let value: Double
        public let unit: String
        public let metadata: [String: String]
        
        public init(
            type: MetricType,
            name: String,
            value: Double,
            unit: String,
            metadata: [String: String] = [:],
            timestamp: Date = Date()
        ) {
            self.timestamp = timestamp
            self.type = type
            self.name = name
            self.value = value
            self.unit = unit
            self.metadata = metadata
        }
    }
    
    /// Time series data for a metric
    public struct TimeSeries: Sendable, Codable {
        public let metric: String
        public let samples: [Sample]
        public let statistics: Statistics
        
        public struct Statistics: Sendable, Codable {
            public let mean: Double
            public let median: Double
            public let stdDev: Double
            public let min: Double
            public let max: Double
            public let p50: Double
            public let p90: Double
            public let p95: Double
            public let p99: Double
            public let p999: Double
        }
    }
    
    /// Resource usage snapshot
    public struct ResourceUsage: Sendable, Codable {
        public let timestamp: Date
        public let cpuUsage: Double
        public let memoryUsage: MemoryInfo
        public let diskIO: DiskIOInfo?
        public let gpuUsage: GPUInfo?
        
        public struct MemoryInfo: Sendable, Codable {
            public let used: Int
            public let peak: Int
            public let available: Int
            public let pressure: Double
        }
        
        public struct DiskIOInfo: Sendable, Codable {
            public let readBytes: Int
            public let writeBytes: Int
            public let readOps: Int
            public let writeOps: Int
        }
        
        public struct GPUInfo: Sendable, Codable {
            public let utilization: Double
            public let memoryUsed: Int
            public let temperature: Double?
        }
    }
    
    /// Performance profile
    public struct Profile: Sendable, Codable {
        public let name: String
        public let startTime: Date
        public let endTime: Date
        public let metrics: [String: TimeSeries]
        public let resourceUsage: [ResourceUsage]
        public let metadata: [String: String]
        
        public var duration: TimeInterval {
            endTime.timeIntervalSince(startTime)
        }
    }
    
    // MARK: - Properties
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "Metrics")
    private var samples: [String: [Sample]] = [:]
    private var resourceSnapshots: [ResourceUsage] = []
    private var startTime: Date?
    private var metadata: [String: String] = [:]
    private let samplingInterval: TimeInterval
    private var samplingTask: Task<Void, Never>?
    
    // MARK: - Initialization
    
    public init(samplingInterval: TimeInterval = 0.1) {
        self.samplingInterval = samplingInterval
    }
    
    deinit {
        samplingTask?.cancel()
    }
    
    // MARK: - Collection Control
    
    /// Start collecting metrics
    public func startCollection(name: String, metadata: [String: String] = [:]) async {
        logger.info("Starting metrics collection: \(name)")
        
        self.startTime = Date()
        self.metadata = metadata
        self.metadata["name"] = name
        
        // Clear previous data
        samples.removeAll()
        resourceSnapshots.removeAll()
        
        // Start resource monitoring
        samplingTask = Task {
            while !Task.isCancelled {
                await collectResourceSnapshot()
                try? await Task.sleep(for: .seconds(samplingInterval))
            }
        }
    }
    
    /// Stop collecting metrics and return profile
    public func stopCollection() async -> Profile? {
        logger.info("Stopping metrics collection")
        
        samplingTask?.cancel()
        samplingTask = nil
        
        guard let startTime = startTime else { return nil }
        
        // Build time series for each metric
        var timeSeries: [String: TimeSeries] = [:]
        for (metric, metricSamples) in samples {
            if let series = buildTimeSeries(metric: metric, samples: metricSamples) {
                timeSeries[metric] = series
            }
        }
        
        return Profile(
            name: metadata["name"] ?? "Unknown",
            startTime: startTime,
            endTime: Date(),
            metrics: timeSeries,
            resourceUsage: resourceSnapshots,
            metadata: metadata
        )
    }
    
    // MARK: - Metric Recording
    
    /// Record a latency measurement
    public func recordLatency(
        name: String,
        value: TimeInterval,
        metadata: [String: String] = [:]
    ) async {
        await recordSample(Sample(
            type: .latency,
            name: name,
            value: value,
            unit: "seconds",
            metadata: metadata
        ))
    }
    
    /// Record throughput
    public func recordThroughput(
        name: String,
        operations: Int,
        duration: TimeInterval,
        metadata: [String: String] = [:]
    ) async {
        let throughput = Double(operations) / duration
        await recordSample(Sample(
            type: .throughput,
            name: name,
            value: throughput,
            unit: "ops/sec",
            metadata: metadata
        ))
    }
    
    /// Record memory usage
    public func recordMemory(
        name: String,
        bytes: Int,
        metadata: [String: String] = [:]
    ) async {
        await recordSample(Sample(
            type: .memory,
            name: name,
            value: Double(bytes),
            unit: "bytes",
            metadata: metadata
        ))
    }
    
    /// Record cache hit rate
    public func recordCacheHitRate(
        name: String,
        hits: Int,
        total: Int,
        metadata: [String: String] = [:]
    ) async {
        let hitRate = total > 0 ? Double(hits) / Double(total) : 0
        await recordSample(Sample(
            type: .cacheHitRate,
            name: name,
            value: hitRate,
            unit: "ratio",
            metadata: metadata
        ))
    }
    
    /// Record accuracy metric
    public func recordAccuracy(
        name: String,
        value: Double,
        metadata: [String: String] = [:]
    ) async {
        await recordSample(Sample(
            type: .accuracy,
            name: name,
            value: value,
            unit: "ratio",
            metadata: metadata
        ))
    }
    
    /// Record custom metric
    public func recordCustom(
        name: String,
        value: Double,
        unit: String,
        metadata: [String: String] = [:]
    ) async {
        await recordSample(Sample(
            type: .custom,
            name: name,
            value: value,
            unit: unit,
            metadata: metadata
        ))
    }
    
    // MARK: - Analysis
    
    /// Analyze performance between two time points
    public func analyze(
        from: Date? = nil,
        to: Date? = nil
    ) async -> MetricsPerformanceAnalysis {
        let filteredSamples = samples.mapValues { samples in
            samples.filter { sample in
                if let from = from, sample.timestamp < from { return false }
                if let to = to, sample.timestamp > to { return false }
                return true
            }
        }
        
        var analysis = MetricsPerformanceAnalysis()
        
        // Analyze each metric type
        for (metric, metricSamples) in filteredSamples {
            if let summary = analyzeMetric(metric: metric, samples: metricSamples) {
                analysis.metrics[metric] = summary
            }
        }
        
        // Analyze resource usage
        let filteredResources = resourceSnapshots.filter { snapshot in
            if let from = from, snapshot.timestamp < from { return false }
            if let to = to, snapshot.timestamp > to { return false }
            return true
        }
        
        if !filteredResources.isEmpty {
            analysis.resourceSummary = analyzeResources(filteredResources)
        }
        
        return analysis
    }
    
    // MARK: - Private Methods
    
    private func recordSample(_ sample: Sample) async {
        let key = "\(sample.type.rawValue).\(sample.name)"
        if samples[key] == nil {
            samples[key] = []
        }
        samples[key]?.append(sample)
    }
    
    private func collectResourceSnapshot() async {
        let snapshot = ResourceUsage(
            timestamp: Date(),
            cpuUsage: getCPUUsage(),
            memoryUsage: getMemoryInfo(),
            diskIO: nil, // Would require platform-specific implementation
            gpuUsage: getGPUInfo()
        )
        resourceSnapshots.append(snapshot)
    }
    
    private func buildTimeSeries(metric: String, samples: [Sample]) -> TimeSeries? {
        guard !samples.isEmpty else { return nil }
        
        let values = samples.map { $0.value }.sorted()
        let stats = TimeSeries.Statistics(
            mean: values.reduce(0, +) / Double(values.count),
            median: values[values.count / 2],
            stdDev: calculateStandardDeviation(values),
            min: values.first!,
            max: values.last!,
            p50: percentile(values, 0.50),
            p90: percentile(values, 0.90),
            p95: percentile(values, 0.95),
            p99: percentile(values, 0.99),
            p999: percentile(values, 0.999)
        )
        
        return TimeSeries(
            metric: metric,
            samples: samples,
            statistics: stats
        )
    }
    
    private func analyzeMetric(metric: String, samples: [Sample]) -> MetricSummary? {
        guard !samples.isEmpty else { return nil }
        
        let values = samples.map { $0.value }
        let first = samples.first!
        
        return MetricSummary(
            name: metric,
            type: first.type,
            unit: first.unit,
            sampleCount: samples.count,
            mean: values.reduce(0, +) / Double(values.count),
            trend: calculateTrend(samples)
        )
    }
    
    private func analyzeResources(_ snapshots: [ResourceUsage]) -> ResourceSummary {
        let cpuValues = snapshots.map { $0.cpuUsage }
        let memoryValues = snapshots.map { Double($0.memoryUsage.used) }
        
        return ResourceSummary(
            avgCPU: cpuValues.reduce(0, +) / Double(cpuValues.count),
            peakCPU: cpuValues.max() ?? 0,
            avgMemory: Int(memoryValues.reduce(0, +) / Double(memoryValues.count)),
            peakMemory: snapshots.map { $0.memoryUsage.peak }.max() ?? 0
        )
    }
    
    // MARK: - System Metrics
    
    private func getCPUUsage() -> Double {
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
        
        if result == KERN_SUCCESS {
            return Double(info.user_time.seconds) + Double(info.system_time.seconds)
        }
        #endif
        return 0
    }
    
    private func getMemoryInfo() -> ResourceUsage.MemoryInfo {
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
        
        if result == KERN_SUCCESS {
            return ResourceUsage.MemoryInfo(
                used: Int(info.resident_size),
                peak: Int(info.resident_size_max),
                available: 0, // Would require additional system calls
                pressure: 0
            )
        }
        #endif
        
        return ResourceUsage.MemoryInfo(
            used: 0,
            peak: 0,
            available: 0,
            pressure: 0
        )
    }
    
    private func getGPUInfo() -> ResourceUsage.GPUInfo? {
        // Metal performance shaders statistics would go here
        return nil
    }
    
    // MARK: - Utilities
    
    private func calculateStandardDeviation(_ values: [Double]) -> Double {
        guard values.count > 1 else { return 0 }
        let mean = values.reduce(0, +) / Double(values.count)
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Double(values.count - 1)
        return sqrt(variance)
    }
    
    private func percentile(_ sortedValues: [Double], _ p: Double) -> Double {
        let index = Int(Double(sortedValues.count - 1) * p)
        return sortedValues[index]
    }
    
    private func calculateTrend(_ samples: [Sample]) -> Double {
        guard samples.count > 1 else { return 0 }
        
        // Simple linear regression
        let n = Double(samples.count)
        let x = Array(0..<samples.count).map { Double($0) }
        let y = samples.map { $0.value }
        
        let sumX = x.reduce(0, +)
        let sumY = y.reduce(0, +)
        let sumXY = zip(x, y).map(*).reduce(0, +)
        let sumX2 = x.map { $0 * $0 }.reduce(0, +)
        
        let slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
        return slope
    }
}

// MARK: - Analysis Results

public struct MetricsPerformanceAnalysis: Sendable {
    public var metrics: [String: MetricSummary] = [:]
    public var resourceSummary: ResourceSummary?
}

public struct MetricSummary: Sendable {
    public let name: String
    public let type: PerformanceMetrics.MetricType
    public let unit: String
    public let sampleCount: Int
    public let mean: Double
    public let trend: Double // Positive = increasing, negative = decreasing
}

public struct ResourceSummary: Sendable {
    public let avgCPU: Double
    public let peakCPU: Double
    public let avgMemory: Int
    public let peakMemory: Int
}

// MARK: - Measurement Helpers

/// Measure and record latency of an async operation
public func measureLatency<T>(
    _ name: String,
    metrics: PerformanceMetrics,
    metadata: [String: String] = [:],
    operation: () async throws -> T
) async throws -> T {
    let start = CACurrentMediaTime()
    let result = try await operation()
    let elapsed = CACurrentMediaTime() - start
    
    await metrics.recordLatency(
        name: name,
        value: elapsed,
        metadata: metadata
    )
    
    return result
}

/// Measure throughput of batch operations
public func measureThroughput(
    _ name: String,
    metrics: PerformanceMetrics,
    operations: Int,
    metadata: [String: String] = [:],
    operation: () async throws -> Void
) async throws {
    let start = CACurrentMediaTime()
    try await operation()
    let elapsed = CACurrentMediaTime() - start
    
    await metrics.recordThroughput(
        name: name,
        operations: operations,
        duration: elapsed,
        metadata: metadata
    )
}
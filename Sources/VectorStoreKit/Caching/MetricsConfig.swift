// VectorStoreKit: Metrics and Logging Configuration
//
// Configurable metrics collection and logging for cache performance

import Foundation
import os.log

/// Configuration for cache metrics collection and logging
public struct CacheMetricsConfig: Sendable {
    /// Sample rate for metrics (0.0 to 1.0)
    public let sampleRate: Float
    
    /// Log level flags
    public let logLevel: LogLevel
    
    /// Whether to collect detailed access patterns
    public let collectAccessPatterns: Bool
    
    /// Whether to collect latency percentiles
    public let collectLatencyPercentiles: Bool
    
    /// Interval for batching log messages (seconds)
    public let logBatchInterval: TimeInterval
    
    /// Maximum log messages per batch
    public let maxLogBatchSize: Int
    
    public enum LogLevel: Int, Sendable {
        case none = 0
        case error = 1
        case warning = 2
        case info = 3
        case debug = 4
        
        var osLogType: OSLogType {
            switch self {
            case .none: return .default
            case .error: return .error
            case .warning: return .default
            case .info: return .info
            case .debug: return .debug
            }
        }
    }
    
    public init(
        sampleRate: Float = 0.01, // 1% sampling by default
        logLevel: LogLevel = .warning,
        collectAccessPatterns: Bool = false,
        collectLatencyPercentiles: Bool = false,
        logBatchInterval: TimeInterval = 60, // 1 minute
        maxLogBatchSize: Int = 100
    ) {
        self.sampleRate = max(0, min(1, sampleRate))
        self.logLevel = logLevel
        self.collectAccessPatterns = collectAccessPatterns
        self.collectLatencyPercentiles = collectLatencyPercentiles
        self.logBatchInterval = logBatchInterval
        self.maxLogBatchSize = maxLogBatchSize
    }
    
    /// Default production configuration
    public static let production = CacheMetricsConfig(
        sampleRate: 0.001, // 0.1% sampling
        logLevel: .error,
        collectAccessPatterns: false,
        collectLatencyPercentiles: false
    )
    
    /// Development configuration with more verbose logging
    public static let development = CacheMetricsConfig(
        sampleRate: 1.0, // 100% sampling
        logLevel: .debug,
        collectAccessPatterns: true,
        collectLatencyPercentiles: true,
        logBatchInterval: 10 // More frequent in dev
    )
    
    /// Performance testing configuration
    public static let performanceTest = CacheMetricsConfig(
        sampleRate: 1.0,
        logLevel: .info,
        collectAccessPatterns: true,
        collectLatencyPercentiles: true,
        logBatchInterval: 5
    )
}

/// Metrics collector with configurable sampling and batching
public actor CacheMetricsCollector {
    private let config: CacheMetricsConfig
    private let logger: Logger
    private var logBuffer: [LogEntry] = []
    private var lastFlush = ContinuousClock.now
    
    // Metrics storage
    private var accessCount: Int = 0
    private var hitCount: Int = 0
    private var missCount: Int = 0
    private var evictionCount: Int = 0
    private var latencies: [TimeInterval] = []
    private let maxLatencySamples = 10000
    
    private struct LogEntry {
        let timestamp: ContinuousClock.Instant
        let level: CacheMetricsConfig.LogLevel
        let message: String
        let metadata: [String: Any]
    }
    
    public init(config: CacheMetricsConfig = .production, subsystem: String = "VectorStoreKit", category: String = "Cache") {
        self.config = config
        self.logger = Logger(subsystem: subsystem, category: category)
        
        // Start flush timer
        Task {
            await startFlushTimer()
        }
    }
    
    /// Record a cache access (sampled based on config)
    public func recordAccess(hit: Bool) async {
        accessCount += 1
        if hit {
            hitCount += 1
        } else {
            missCount += 1
        }
        
        // Sample for detailed logging
        if shouldSample() {
            await log(level: .debug, "Cache access", metadata: ["hit": hit, "total_accesses": accessCount])
        }
    }
    
    /// Record access latency
    public func recordLatency(_ latency: TimeInterval) async {
        guard config.collectLatencyPercentiles else { return }
        
        latencies.append(latency)
        if latencies.count > maxLatencySamples {
            latencies.removeFirst(latencies.count - maxLatencySamples)
        }
        
        // Log high latency events
        if latency > 0.001 { // 1ms threshold
            await log(level: .warning, "High latency detected", metadata: ["latency_ms": latency * 1000])
        }
    }
    
    /// Record eviction event
    public func recordEviction(count: Int = 1) async {
        evictionCount += count
        
        if shouldSample() {
            await log(level: .debug, "Cache eviction", metadata: ["count": count, "total_evictions": evictionCount])
        }
    }
    
    /// Log a message with batching
    public func log(level: CacheMetricsConfig.LogLevel, _ message: String, metadata: [String: Any] = [:]) async {
        guard level.rawValue <= config.logLevel.rawValue else { return }
        
        let entry = LogEntry(
            timestamp: ContinuousClock.now,
            level: level,
            message: message,
            metadata: metadata
        )
        
        logBuffer.append(entry)
        
        // Check if we should flush
        if logBuffer.count >= config.maxLogBatchSize {
            await flush()
        }
    }
    
    /// Get current metrics summary
    public func getMetrics() async -> CacheMetricsSummary {
        let hitRate = accessCount > 0 ? Float(hitCount) / Float(accessCount) : 0
        
        var percentiles: LatencyPercentiles?
        if config.collectLatencyPercentiles && !latencies.isEmpty {
            let sorted = latencies.sorted()
            percentiles = LatencyPercentiles(
                p50: sorted[sorted.count / 2],
                p95: sorted[min(Int(Double(sorted.count) * 0.95), sorted.count - 1)],
                p99: sorted[min(Int(Double(sorted.count) * 0.99), sorted.count - 1)],
                max: sorted.last ?? 0
            )
        }
        
        return CacheMetricsSummary(
            accessCount: accessCount,
            hitCount: hitCount,
            missCount: missCount,
            hitRate: hitRate,
            evictionCount: evictionCount,
            latencyPercentiles: percentiles
        )
    }
    
    // MARK: - Private Methods
    
    private func shouldSample() -> Bool {
        Float.random(in: 0...1) <= config.sampleRate
    }
    
    private func startFlushTimer() async {
        while !Task.isCancelled {
            do {
                try await Task.sleep(for: .seconds(config.logBatchInterval))
                await flush()
            } catch {
                break
            }
        }
    }
    
    private func flush() async {
        guard !logBuffer.isEmpty else { return }
        
        // Group by log level
        let grouped = Dictionary(grouping: logBuffer) { $0.level }
        
        for (level, entries) in grouped {
            let summary = "[\(entries.count) events] " + entries.prefix(3).map { $0.message }.joined(separator: "; ")
            
            logger.log(level: level.osLogType, "\(summary, privacy: .public)")
            
            // Log sample metadata for debug level
            if level == .debug, let firstEntry = entries.first {
                logger.debug("Sample metadata: \(String(describing: firstEntry.metadata), privacy: .public)")
            }
        }
        
        logBuffer.removeAll()
        lastFlush = ContinuousClock.now
    }
}

/// Summary of cache metrics
public struct CacheMetricsSummary: Sendable {
    public let accessCount: Int
    public let hitCount: Int
    public let missCount: Int
    public let hitRate: Float
    public let evictionCount: Int
    public let latencyPercentiles: LatencyPercentiles?
}

public struct LatencyPercentiles: Sendable {
    public let p50: TimeInterval
    public let p95: TimeInterval
    public let p99: TimeInterval
    public let max: TimeInterval
}
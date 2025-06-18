// VectorStoreKit: Storage Performance Monitor
//
// Monitors and tracks storage system performance metrics

import Foundation
import os.log

/// Monitors storage system performance metrics
actor StoragePerformanceMonitor {
    // MARK: - Types
    
    /// Storage operation types
    enum StorageOperation {
        case store
        case retrieve
        case delete
        case compact
        case migrate
    }
    
    /// Operation metrics
    private struct OperationMetrics {
        var count: Int = 0
        var totalDuration: TimeInterval = 0
        var totalDataSize: Int = 0
        var errors: Int = 0
        var latencies: [TimeInterval] = []
        
        var averageLatency: TimeInterval {
            guard count > 0 else { return 0 }
            return totalDuration / TimeInterval(count)
        }
        
        var latencyP99: TimeInterval {
            guard !latencies.isEmpty else { return 0 }
            let sorted = latencies.sorted()
            let index = Int(Double(sorted.count) * 0.99)
            return sorted[min(index, sorted.count - 1)]
        }
    }
    
    // MARK: - Properties
    
    private let enabled: Bool
    private let logger = Logger(subsystem: "VectorStoreKit", category: "StoragePerformanceMonitor")
    
    // Metrics storage
    private var operationMetrics: [StorageTier: [StorageOperation: OperationMetrics]] = [:]
    private var startTime: Date?
    private var totalOperations: Int = 0
    
    // Performance thresholds
    private let latencyThresholds: [StorageTier: TimeInterval] = [
        .hot: 0.0001,      // 100μs
        .warm: 0.001,      // 1ms
        .cold: 0.01,       // 10ms
        .frozen: 0.1       // 100ms
    ]
    
    // Sampling configuration
    private let maxLatencySamples = 1000
    private let samplingRate: Double = 0.1 // Sample 10% of operations for detailed latency
    
    // MARK: - Initialization
    
    init(enabled: Bool) {
        self.enabled = enabled
        
        // Initialize metrics structure
        for tier in StorageTier.allCases {
            operationMetrics[tier] = [:]
            for operation in [StorageOperation.store, .retrieve, .delete] {
                operationMetrics[tier]?[operation] = OperationMetrics()
            }
        }
    }
    
    // MARK: - Public Methods
    
    /// Start monitoring
    func start() async {
        guard enabled else { return }
        
        startTime = Date()
        logger.info("Performance monitoring started")
    }
    
    /// Record an operation
    func recordOperation(
        _ operation: StorageOperation,
        duration: TimeInterval,
        dataSize: Int,
        tier: StorageTier,
        error: Error? = nil
    ) async {
        guard enabled else { return }
        
        totalOperations += 1
        
        // Update metrics
        if operationMetrics[tier] == nil {
            operationMetrics[tier] = [:]
        }
        
        if operationMetrics[tier]?[operation] == nil {
            operationMetrics[tier]?[operation] = OperationMetrics()
        }
        
        operationMetrics[tier]?[operation]?.count += 1
        operationMetrics[tier]?[operation]?.totalDuration += duration
        operationMetrics[tier]?[operation]?.totalDataSize += dataSize
        
        if error != nil {
            operationMetrics[tier]?[operation]?.errors += 1
        }
        
        // Sample latency based on sampling rate
        if Double.random(in: 0...1) < samplingRate {
            operationMetrics[tier]?[operation]?.latencies.append(duration)
            
            // Keep only recent samples
            if let count = operationMetrics[tier]?[operation]?.latencies.count,
               count > maxLatencySamples {
                operationMetrics[tier]?[operation]?.latencies = Array(
                    operationMetrics[tier]![operation]!.latencies.suffix(maxLatencySamples / 2)
                )
            }
        }
        
        // Check for performance anomalies
        if let threshold = latencyThresholds[tier], duration > threshold * 10 {
            // Note: logger.warning requires iOS 14+, using info for compatibility
            logger.info("High latency detected: \(String(describing: operation)) on \(String(describing: tier)) took \(duration)s")
        }
    }
    
    /// Get average latency across all operations
    var averageLatency: TimeInterval {
        get async {
            var totalDuration: TimeInterval = 0
            var totalCount = 0
            
            for (_, tierMetrics) in operationMetrics {
                for (_, metrics) in tierMetrics {
                    totalDuration += metrics.totalDuration
                    totalCount += metrics.count
                }
            }
            
            return totalCount > 0 ? totalDuration / TimeInterval(totalCount) : 0
        }
    }
    
    /// Get comprehensive statistics
    func getStatistics() async -> (errorRate: Float, latencyP99: TimeInterval, throughput: Float) {
        var totalErrors = 0
        var totalCount = 0
        var allLatencies: [TimeInterval] = []
        
        // Collect all metrics
        for (_, tierMetrics) in operationMetrics {
            for (_, metrics) in tierMetrics {
                totalErrors += metrics.errors
                totalCount += metrics.count
                allLatencies.append(contentsOf: metrics.latencies)
            }
        }
        
        // Calculate error rate
        let errorRate = totalCount > 0 ? Float(totalErrors) / Float(totalCount) : 0
        
        // Calculate P99 latency
        let latencyP99: TimeInterval
        if !allLatencies.isEmpty {
            let sorted = allLatencies.sorted()
            let index = Int(Double(sorted.count) * 0.99)
            latencyP99 = sorted[min(index, sorted.count - 1)]
        } else {
            latencyP99 = 0
        }
        
        // Calculate throughput (operations per second)
        let throughput: Float
        if let start = startTime {
            let elapsed = Date().timeIntervalSince(start)
            throughput = elapsed > 0 ? Float(totalOperations) / Float(elapsed) : 0
        } else {
            throughput = 0
        }
        
        return (errorRate, latencyP99, throughput)
    }
    
    /// Get detailed performance report
    func getDetailedReport() async -> StoragePerformanceReport {
        var tierReports: [StorageTier: TierPerformanceReport] = [:]
        
        for (tier, tierMetrics) in operationMetrics {
            var operationReports: [StorageOperation: StorageOperationReport] = [:]
            
            for (operation, metrics) in tierMetrics {
                let report = StorageOperationReport(
                    operation: operation,
                    count: metrics.count,
                    averageLatency: metrics.averageLatency,
                    latencyP99: metrics.latencyP99,
                    errorRate: metrics.count > 0 ? Float(metrics.errors) / Float(metrics.count) : 0,
                    averageDataSize: metrics.count > 0 ? metrics.totalDataSize / metrics.count : 0,
                    throughput: calculateThroughput(metrics: metrics)
                )
                operationReports[operation] = report
            }
            
            tierReports[tier] = TierPerformanceReport(
                tier: tier,
                operations: operationReports,
                totalOperations: tierMetrics.values.reduce(0) { $0 + $1.count },
                totalErrors: tierMetrics.values.reduce(0) { $0 + $1.errors }
            )
        }
        
        let stats = await getStatistics()
        
        return StoragePerformanceReport(
            startTime: startTime ?? Date(),
            endTime: Date(),
            totalOperations: totalOperations,
            overallErrorRate: stats.errorRate,
            overallLatencyP99: stats.latencyP99,
            overallThroughput: stats.throughput,
            tierReports: tierReports
        )
    }
    
    /// Reset all metrics
    func reset() async {
        for tier in operationMetrics.keys {
            if let tierOps = operationMetrics[tier] {
                for operation in tierOps.keys {
                    operationMetrics[tier]?[operation] = OperationMetrics()
                }
            }
        }
        
        totalOperations = 0
        startTime = Date()
        
        logger.info("Performance metrics reset")
    }
    
    // MARK: - Private Methods
    
    private func calculateThroughput(metrics: OperationMetrics) -> Float {
        guard let start = startTime, metrics.totalDuration > 0 else { return 0 }
        
        let elapsed = Date().timeIntervalSince(start)
        return elapsed > 0 ? Float(metrics.count) / Float(elapsed) : 0
    }
}

// MARK: - Performance Report Types

struct StoragePerformanceReport {
    let startTime: Date
    let endTime: Date
    let totalOperations: Int
    let overallErrorRate: Float
    let overallLatencyP99: TimeInterval
    let overallThroughput: Float
    let tierReports: [StorageTier: TierPerformanceReport]
}

struct TierPerformanceReport {
    let tier: StorageTier
    let operations: [StoragePerformanceMonitor.StorageOperation: StorageOperationReport]
    let totalOperations: Int
    let totalErrors: Int
}

struct StorageOperationReport {
    let operation: StoragePerformanceMonitor.StorageOperation
    let count: Int
    let averageLatency: TimeInterval
    let latencyP99: TimeInterval
    let errorRate: Float
    let averageDataSize: Int
    let throughput: Float
}

// MARK: - StorageTier Extension
// StorageTier already conforms to CaseIterable in VectorTypes.swift
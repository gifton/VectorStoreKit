// ValidationResults.swift
// VectorStoreKit
//
// Result types for comprehensive validation and testing framework
// Provides detailed feedback for integration, performance, and stress testing

import Foundation

// MARK: - Integration Validation Results

/// Result of comprehensive integration validation
public struct IntegrationValidationResult: Sendable {
    public let overallSuccess: Bool
    public let testResults: [String: Bool]
    public let errors: [Error]
    public let timestamp: Date
    
    /// Success rate (percentage of tests that passed)
    public var successRate: Float {
        let successCount = testResults.values.filter { $0 }.count
        return Float(successCount) / Float(testResults.count)
    }
    
    /// Failed tests for debugging
    public var failedTests: [String] {
        return testResults.compactMap { key, value in
            value ? nil : key
        }
    }
    
    /// Format results for logging
    public func formattedSummary() -> String {
        let successPercentage = Int(successRate * 100)
        return """
        Integration Validation Results:
        - Overall Success: \(overallSuccess ? "✅ PASS" : "❌ FAIL")
        - Success Rate: \(successPercentage)% (\(testResults.values.filter { $0 }.count)/\(testResults.count))
        - Failed Tests: \(failedTests.isEmpty ? "None" : failedTests.joined(separator: ", "))
        - Errors: \(errors.count)
        - Timestamp: \(timestamp)
        """
    }
}

// MARK: - Performance Validation Results

/// Result of comprehensive performance validation
public struct PerformanceValidationResult: Sendable {
    public let allocationTime: TimeInterval
    public let reuseTime: TimeInterval
    public let batchTime: TimeInterval
    public let pressureHandlingTime: TimeInterval
    public let totalTime: TimeInterval
    public let allocationThroughput: Float
    public let reuseThroughput: Float
    public let batchThroughput: Float
    public let bufferReuseRate: Float
    public let averageAllocationSize: Float
    public let memoryEfficiency: Float
    public let performanceScore: Float
    public let gpuUtilization: Float
    
    /// Overall performance assessment
    public var performanceGrade: PerformanceGrade {
        switch performanceScore {
        case 90...:
            return .excellent
        case 75..<90:
            return .good
        case 60..<75:
            return .acceptable
        case 40..<60:
            return .poor
        default:
            return .failing
        }
    }
    
    /// Reuse improvement factor (how much faster reuse is vs initial allocation)
    public var reuseImprovementFactor: Float {
        guard allocationTime > 0 else { return 1.0 }
        return Float(allocationTime / reuseTime)
    }
    
    /// Performance insights and recommendations
    public var insights: [String] {
        var insights: [String] = []
        
        if bufferReuseRate >= 0.8 {
            insights.append("Excellent buffer reuse rate (\(String(format: "%.1f%%", bufferReuseRate * 100)))")
        } else if bufferReuseRate < 0.6 {
            insights.append("Low buffer reuse rate (\(String(format: "%.1f%%", bufferReuseRate * 100))) - consider pool size adjustments")
        }
        
        if reuseImprovementFactor >= 2.0 {
            insights.append("Buffer reuse provides \(String(format: "%.1fx", reuseImprovementFactor)) performance improvement")
        }
        
        if memoryEfficiency >= 0.8 {
            insights.append("High memory efficiency (\(String(format: "%.1f%%", memoryEfficiency * 100)))")
        } else if memoryEfficiency < 0.6 {
            insights.append("Memory efficiency could be improved (\(String(format: "%.1f%%", memoryEfficiency * 100)))")
        }
        
        if allocationThroughput >= 5000 {
            insights.append("High allocation throughput (\(String(format: "%.0f", allocationThroughput)) ops/sec)")
        } else if allocationThroughput < 2000 {
            insights.append("Allocation throughput is below target (\(String(format: "%.0f", allocationThroughput)) ops/sec)")
        }
        
        return insights
    }
    
    /// Format results for reporting
    public func formattedSummary() -> String {
        return """
        Performance Validation Results:
        - Performance Score: \(String(format: "%.1f", performanceScore))/100 (\(performanceGrade.description))
        - Allocation Throughput: \(String(format: "%.0f", allocationThroughput)) ops/sec
        - Reuse Throughput: \(String(format: "%.0f", reuseThroughput)) ops/sec
        - Batch Throughput: \(String(format: "%.0f", batchThroughput)) ops/sec
        - Buffer Reuse Rate: \(String(format: "%.1f%%", bufferReuseRate * 100))
        - Memory Efficiency: \(String(format: "%.1f%%", memoryEfficiency * 100))
        - Reuse Improvement: \(String(format: "%.1fx", reuseImprovementFactor)) faster
        - GPU Utilization: \(String(format: "%.1f%%", gpuUtilization * 100))
        
        Key Insights:
        \(insights.map { "• \($0)" }.joined(separator: "\n"))
        """
    }
}

public enum PerformanceGrade: String, Sendable {
    case excellent = "Excellent"
    case good = "Good"
    case acceptable = "Acceptable"
    case poor = "Poor"
    case failing = "Failing"
    
    public var description: String {
        return self.rawValue
    }
}

// MARK: - Shader Validation Results

/// Result of Metal shader integration validation
public struct ShaderValidationResult: Sendable {
    public let compilationSuccess: Bool
    public let pipelineCreationSuccess: Bool
    public let bufferIntegrationSuccess: Bool
    public let computationCorrectness: Bool
    public let error: Error?
    
    /// Overall shader integration success
    public var overallSuccess: Bool {
        return compilationSuccess && pipelineCreationSuccess && bufferIntegrationSuccess && computationCorrectness
    }
    
    /// Validation score (0-100)
    public var validationScore: Float {
        var score: Float = 0
        if compilationSuccess { score += 25 }
        if pipelineCreationSuccess { score += 25 }
        if bufferIntegrationSuccess { score += 25 }
        if computationCorrectness { score += 25 }
        return score
    }
    
    /// Format results for reporting
    public func formattedSummary() -> String {
        return """
        Shader Validation Results:
        - Overall Success: \(overallSuccess ? "✅ PASS" : "❌ FAIL")
        - Validation Score: \(String(format: "%.0f", validationScore))/100
        - Compilation: \(compilationSuccess ? "✅" : "❌")
        - Pipeline Creation: \(pipelineCreationSuccess ? "✅" : "❌")
        - Buffer Integration: \(bufferIntegrationSuccess ? "✅" : "❌")
        - Computation Correctness: \(computationCorrectness ? "✅" : "❌")
        \(error != nil ? "- Error: \(error!.localizedDescription)" : "")
        """
    }
}

// MARK: - Stress Test Results

/// Result of comprehensive stress testing
public struct StressTestResult: Sendable {
    public let duration: TimeInterval
    public let totalOperations: Int
    public let successfulAllocations: Int
    public let failedAllocations: Int
    public let memoryPressureEvents: Int
    public let concurrentOperations: Int
    public let finalMemoryUsage: Int
    public let peakMemoryUsage: Int
    public let finalPerformanceScore: Float
    public let errors: [Error]
    
    /// Success rate for allocations
    public var allocationSuccessRate: Float {
        let totalAttempts = successfulAllocations + failedAllocations
        guard totalAttempts > 0 else { return 1.0 }
        return Float(successfulAllocations) / Float(totalAttempts)
    }
    
    /// Operations per second during stress test
    public var operationsPerSecond: Float {
        guard duration > 0 else { return 0 }
        return Float(totalOperations) / Float(duration)
    }
    
    /// Memory utilization efficiency
    public var memoryUtilization: Float {
        guard peakMemoryUsage > 0 else { return 0 }
        return Float(finalMemoryUsage) / Float(peakMemoryUsage)
    }
    
    /// Stress test assessment
    public var stressTestGrade: StressTestGrade {
        if allocationSuccessRate >= 0.95 && finalPerformanceScore >= 70 && errors.count <= 5 {
            return .excellent
        } else if allocationSuccessRate >= 0.9 && finalPerformanceScore >= 60 && errors.count <= 20 {
            return .good
        } else if allocationSuccessRate >= 0.8 && finalPerformanceScore >= 50 && errors.count <= 50 {
            return .acceptable
        } else if allocationSuccessRate >= 0.6 && errors.count <= 100 {
            return .poor
        } else {
            return .failing
        }
    }
    
    /// Format results for reporting
    public func formattedSummary() -> String {
        return """
        Stress Test Results:
        - Grade: \(stressTestGrade.description)
        - Duration: \(String(format: "%.2f", duration)) seconds
        - Total Operations: \(totalOperations)
        - Operations/Second: \(String(format: "%.0f", operationsPerSecond))
        - Allocation Success Rate: \(String(format: "%.1f%%", allocationSuccessRate * 100))
        - Memory Pressure Events: \(memoryPressureEvents)
        - Concurrent Operations: \(concurrentOperations)
        - Final Performance Score: \(String(format: "%.1f", finalPerformanceScore))
        - Memory Usage: \(String(format: "%.1f MB", Double(finalMemoryUsage) / 1_048_576))
        - Peak Memory: \(String(format: "%.1f MB", Double(peakMemoryUsage) / 1_048_576))
        - Memory Utilization: \(String(format: "%.1f%%", memoryUtilization * 100))
        - Error Count: \(errors.count)
        """
    }
}

public enum StressTestGrade: String, Sendable {
    case excellent = "Excellent"
    case good = "Good"
    case acceptable = "Acceptable"
    case poor = "Poor"
    case failing = "Failing"
    
    public var description: String {
        return self.rawValue
    }
}

// MARK: - Benchmark Results

/// Individual benchmark result
public struct BenchmarkResult: Sendable {
    public let operations: Int
    public let duration: TimeInterval
    public let throughput: Float
    public let memoryUsage: Int
    public let score: Float
    
    /// Format for reporting
    public func formattedSummary() -> String {
        return """
        Operations: \(operations), Duration: \(String(format: "%.3f", duration))s, 
        Throughput: \(String(format: "%.0f", throughput)) ops/sec, 
        Memory: \(String(format: "%.1f MB", Double(memoryUsage) / 1_048_576)), 
        Score: \(String(format: "%.1f", score))/100
        """
    }
}

/// Comprehensive validation benchmark results
public struct ValidationBenchmarkResults: Sendable {
    public let results: [String: BenchmarkResult]
    public let overallScore: Float
    public let timestamp: Date
    
    /// Best performing benchmark
    public var bestBenchmark: (name: String, result: BenchmarkResult)? {
        return results.max { $0.value.score < $1.value.score }.map { (name: $0.key, result: $0.value) }
    }
    
    /// Worst performing benchmark
    public var worstBenchmark: (name: String, result: BenchmarkResult)? {
        return results.min { $0.value.score < $1.value.score }.map { (name: $0.key, result: $0.value) }
    }
    
    /// Average throughput across all benchmarks
    public var averageThroughput: Float {
        let throughputs = results.values.map { $0.throughput }
        return throughputs.reduce(0, +) / Float(throughputs.count)
    }
    
    /// Total memory usage across all benchmarks
    public var totalMemoryUsage: Int {
        return results.values.reduce(0) { $0 + $1.memoryUsage }
    }
    
    /// Format results for comprehensive reporting
    public func formattedSummary() -> String {
        let sortedResults = results.sorted { $0.value.score > $1.value.score }
        let resultSummaries = sortedResults.map { "\($0.key): \($0.value.formattedSummary())" }
        
        return """
        Benchmark Results Summary:
        - Overall Score: \(String(format: "%.1f", overallScore))/100
        - Average Throughput: \(String(format: "%.0f", averageThroughput)) ops/sec
        - Total Memory Usage: \(String(format: "%.1f MB", Double(totalMemoryUsage) / 1_048_576))
        - Best Benchmark: \(bestBenchmark?.name ?? "None") (\(String(format: "%.1f", bestBenchmark?.result.score ?? 0)))
        - Worst Benchmark: \(worstBenchmark?.name ?? "None") (\(String(format: "%.1f", worstBenchmark?.result.score ?? 0)))
        - Timestamp: \(timestamp)
        
        Individual Results:
        \(resultSummaries.map { "• \($0)" }.joined(separator: "\n"))
        """
    }
}
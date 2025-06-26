// UnifiedBufferStatistics.swift
// VectorStoreKit
//
// Comprehensive statistics for unified buffer management
// Provides detailed metrics for performance monitoring and optimization

import Foundation

/// Comprehensive statistics for unified buffer management
public struct UnifiedBufferStatistics: Sendable {
    public let totalAllocations: UInt64
    public let totalBytesAllocated: UInt64
    public let bufferReuseCount: UInt64
    public let memoryPressureEvents: UInt64
    public let vectorPool: BufferPoolStatistics
    public let quantizedPool: BufferPoolStatistics
    public let generalPool: BufferPoolStatistics
    public let commandPool: BufferPoolStatistics
    public let currentMemoryUsage: Int
    public let peakMemoryUsage: Int
    
    // MARK: - Computed Properties
    
    /// Buffer reuse rate (0.0 to 1.0)
    public var reuseRate: Float {
        guard totalAllocations > 0 else { return 0 }
        return Float(bufferReuseCount) / Float(totalAllocations)
    }
    
    /// Average allocation size in bytes
    public var averageAllocationSize: Float {
        guard totalAllocations > 0 else { return 0 }
        return Float(totalBytesAllocated) / Float(totalAllocations)
    }
    
    /// Memory efficiency ratio (current/peak)
    public var memoryEfficiency: Float {
        guard peakMemoryUsage > 0 else { return 1.0 }
        return Float(currentMemoryUsage) / Float(peakMemoryUsage)
    }
    
    /// Current memory usage in MB
    public var currentMemoryUsageMB: Double {
        Double(currentMemoryUsage) / 1_048_576
    }
    
    /// Peak memory usage in MB
    public var peakMemoryUsageMB: Double {
        Double(peakMemoryUsage) / 1_048_576
    }
    
    /// Average bytes per allocation
    public var averageBytesPerAllocation: Double {
        guard totalAllocations > 0 else { return 0 }
        return Double(totalBytesAllocated) / Double(totalAllocations)
    }
    
    /// Memory pressure events per 1000 allocations
    public var memoryPressureRate: Float {
        guard totalAllocations > 0 else { return 0 }
        return Float(memoryPressureEvents) * 1000.0 / Float(totalAllocations)
    }
    
    // MARK: - Pool-Specific Analysis
    
    /// Get statistics for a specific pool type
    public func statistics(for poolType: BufferPoolType) -> BufferPoolStatistics {
        switch poolType {
        case .vector:
            return vectorPool
        case .quantized:
            return quantizedPool
        case .general:
            return generalPool
        case .command:
            return commandPool
        case .matrix:
            return generalPool // Matrix operations use general pool
        }
    }
    
    /// Total buffers across all pools
    public var totalBuffersInPools: Int {
        return vectorPool.totalBuffers + quantizedPool.totalBuffers + 
               generalPool.totalBuffers + commandPool.totalBuffers
    }
    
    /// Total memory in pools
    public var totalPoolMemory: Int {
        return vectorPool.memoryUsage + quantizedPool.memoryUsage + 
               generalPool.memoryUsage + commandPool.memoryUsage
    }
    
    // MARK: - Performance Metrics
    
    /// Performance score (0-100) based on efficiency metrics
    public var performanceScore: Float {
        let reuseScore = reuseRate * 25.0          // 25% weight for reuse rate
        let efficiencyScore = memoryEfficiency * 25.0  // 25% weight for memory efficiency
        let pressureScore = max(0, 25.0 - memoryPressureRate) // 25% weight (lower pressure = higher score)
        let utilizationScore = 25.0 // 25% base utilization score
        
        return min(100.0, reuseScore + efficiencyScore + pressureScore + Float(utilizationScore))
    }
    
    /// Get recommendations based on current statistics
    public var recommendations: [String] {
        var recommendations: [String] = []
        
        if reuseRate < 0.6 {
            recommendations.append("Buffer reuse rate is low (\(String(format: "%.1f%%", reuseRate * 100))). Consider adjusting pool sizes or allocation patterns.")
        }
        
        if memoryEfficiency < 0.7 {
            recommendations.append("Memory efficiency is low (\(String(format: "%.1f%%", memoryEfficiency * 100))). Consider more aggressive cleanup during memory pressure.")
        }
        
        if memoryPressureRate > 10.0 {
            recommendations.append("High memory pressure events (\(String(format: "%.1f", memoryPressureRate)) per 1000 allocations). Consider reducing maximum memory usage limits.")
        }
        
        if currentMemoryUsageMB > 1000 {
            recommendations.append("High current memory usage (\(String(format: "%.1f MB", currentMemoryUsageMB))). Consider implementing more aggressive buffer eviction.")
        }
        
        // Pool-specific recommendations
        if vectorPool.totalBuffers > generalPool.totalBuffers * 2 {
            recommendations.append("Vector pool has significantly more buffers than general pool. Consider balancing pool allocations.")
        }
        
        return recommendations
    }
    
    // MARK: - Detailed Analysis
    
    /// Get detailed breakdown by pool type
    public var poolBreakdown: PoolBreakdown {
        return PoolBreakdown(
            vector: PoolAnalysis(statistics: vectorPool, poolType: .vector),
            quantized: PoolAnalysis(statistics: quantizedPool, poolType: .quantized),
            general: PoolAnalysis(statistics: generalPool, poolType: .general),
            command: PoolAnalysis(statistics: commandPool, poolType: .command)
        )
    }
    
    /// Format statistics for logging or debugging
    public func formattedSummary() -> String {
        return """
        Unified Buffer Manager Statistics:
        - Total Allocations: \(totalAllocations)
        - Buffer Reuse Rate: \(String(format: "%.1f%%", reuseRate * 100))
        - Current Memory: \(String(format: "%.1f MB", currentMemoryUsageMB))
        - Peak Memory: \(String(format: "%.1f MB", peakMemoryUsageMB))
        - Memory Efficiency: \(String(format: "%.1f%%", memoryEfficiency * 100))
        - Performance Score: \(String(format: "%.1f", performanceScore))
        - Pool Breakdown:
          * Vector: \(vectorPool.totalBuffers) buffers, \(String(format: "%.1f MB", Double(vectorPool.memoryUsage) / 1_048_576))
          * Quantized: \(quantizedPool.totalBuffers) buffers, \(String(format: "%.1f MB", Double(quantizedPool.memoryUsage) / 1_048_576))
          * General: \(generalPool.totalBuffers) buffers, \(String(format: "%.1f MB", Double(generalPool.memoryUsage) / 1_048_576))
          * Command: \(commandPool.totalBuffers) buffers, \(String(format: "%.1f MB", Double(commandPool.memoryUsage) / 1_048_576))
        """
    }
}

// MARK: - Supporting Types

/// Analysis of individual pool performance
public struct PoolAnalysis: Sendable {
    public let statistics: BufferPoolStatistics
    public let poolType: BufferPoolType
    
    /// Pool utilization efficiency
    public var utilization: Float {
        let expectedSizes = poolType.typicalSizes
        guard !expectedSizes.isEmpty else { return 1.0 }
        
        let actualSizes = statistics.sizeDistribution.keys
        let matchingCount = actualSizes.filter { actualSize in
            expectedSizes.contains { expectedSize in
                abs(actualSize - expectedSize) < expectedSize / 10 // Within 10%
            }
        }.count
        
        return Float(matchingCount) / Float(actualSizes.count)
    }
    
    /// Memory usage per buffer
    public var averageBufferSize: Int {
        guard statistics.totalBuffers > 0 else { return 0 }
        return statistics.memoryUsage / statistics.totalBuffers
    }
    
    /// Size distribution analysis
    public var isWellDistributed: Bool {
        let sizeVariance = calculateSizeVariance()
        return sizeVariance < 0.5 // Low variance indicates good distribution
    }
    
    private func calculateSizeVariance() -> Float {
        let sizes = statistics.sizeDistribution.map { Float($0.key) }
        guard sizes.count > 1 else { return 0 }
        
        let mean = sizes.reduce(0, +) / Float(sizes.count)
        let variance = sizes.map { pow($0 - mean, 2) }.reduce(0, +) / Float(sizes.count)
        
        return variance / (mean * mean) // Coefficient of variation
    }
}

/// Breakdown of all pool analyses
public struct PoolBreakdown: Sendable {
    public let vector: PoolAnalysis
    public let quantized: PoolAnalysis
    public let general: PoolAnalysis
    public let command: PoolAnalysis
    
    /// Most efficient pool
    public var mostEfficientPool: BufferPoolType {
        let pools = [
            (vector.poolType, vector.utilization),
            (quantized.poolType, quantized.utilization),
            (general.poolType, general.utilization),
            (command.poolType, command.utilization)
        ]
        
        return pools.max { $0.1 < $1.1 }?.0 ?? .general
    }
    
    /// Least efficient pool
    public var leastEfficientPool: BufferPoolType {
        let pools = [
            (vector.poolType, vector.utilization),
            (quantized.poolType, quantized.utilization),
            (general.poolType, general.utilization),
            (command.poolType, command.utilization)
        ]
        
        return pools.min { $0.1 < $1.1 }?.0 ?? .general
    }
    
    /// Overall pool efficiency
    public var overallEfficiency: Float {
        return (vector.utilization + quantized.utilization + general.utilization + command.utilization) / 4.0
    }
}
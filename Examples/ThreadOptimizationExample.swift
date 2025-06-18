// VectorStoreKit: Thread Optimization Example
//
// Demonstrates thread configuration optimization and performance profiling
//

import Foundation
import VectorStoreKit

@main
struct ThreadOptimizationExample {
    static func main() async throws {
        print("🚀 VectorStoreKit Thread Optimization Demo")
        print("=========================================\n")
        
        // Initialize Metal components
        let device = try MetalDevice()
        let bufferPool = MetalBufferPool(device: device)
        let pipelineManager = MetalPipelineManager(device: device)
        
        // Initialize profilers
        let performanceProfiler = try await MetalPerformanceProfiler(device: device)
        let threadOptimizer = await MetalThreadConfigurationOptimizer(
            device: device,
            profiler: nil
        )
        
        // Initialize optimized distance compute
        let distanceCompute = await MetalDistanceComputeOptimized(
            device: device,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager,
            threadOptimizer: threadOptimizer,
            performanceProfiler: performanceProfiler
        )
        
        // Generate test data
        let dimensions = [128, 256, 512, 1024]
        let candidateCounts = [1000, 10000, 100000]
        
        print("📊 Testing thread optimization across different configurations...")
        print("================================================================\n")
        
        for dim in dimensions {
            for count in candidateCounts {
                print("Testing dimension=\(dim), candidates=\(count)")
                
                // Generate random vectors
                let query = generateRandomVector(dimension: dim)
                let candidates = (0..<count).map { _ in
                    generateRandomVector(dimension: dim)
                }
                
                // Test with optimization
                let startOptimized = CFAbsoluteTimeGetCurrent()
                let _ = try await distanceCompute.computeDistances(
                    query: query,
                    candidates: candidates,
                    metric: .euclidean
                )
                let timeOptimized = CFAbsoluteTimeGetCurrent() - startOptimized
                
                // Get configuration details
                let config = await threadOptimizer.getOptimalConfiguration(
                    for: .distanceComputation,
                    workSize: count,
                    vectorDimension: dim
                )
                
                print("""
                  ✅ Optimized: \(String(format: "%.2f", timeOptimized * 1000))ms
                     Thread config: \(config.threadsPerThreadgroup.width) threads/group
                     Occupancy: \(String(format: "%.1f%%", config.estimatedOccupancy * 100))
                     Throughput: \(String(format: "%.0f", Double(count) / timeOptimized)) vectors/sec
                
                """)
            }
        }
        
        // Test 512-dimensional optimization
        print("\n🎯 Special 512-dimensional vector optimization")
        print("=============================================\n")
        
        let query512 = Vector512.random()
        let candidates512 = (0..<10000).map { _ in Vector512.random() }
        
        // Test normalized vs non-normalized
        for normalized in [false, true] {
            let startTime = CFAbsoluteTimeGetCurrent()
            let _ = try await distanceCompute.computeDistances512(
                query: query512,
                candidates: candidates512,
                metric: .cosine,
                normalized: normalized
            )
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            
            print("""
                Cosine distance (normalized=\(normalized)): \
                \(String(format: "%.2f", duration * 1000))ms
                """)
        }
        
        // Test batch operations with 2D optimization
        print("\n🔄 Batch operation with 2D thread optimization")
        print("==============================================\n")
        
        let queries = (0..<100).map { _ in generateRandomVector(dimension: 256) }
        let batchCandidates = (0..<1000).map { _ in generateRandomVector(dimension: 256) }
        
        let batchStart = CFAbsoluteTimeGetCurrent()
        let batchResults = try await distanceCompute.batchComputeDistances(
            queries: queries,
            candidates: batchCandidates,
            metric: .euclidean
        )
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStart
        
        let totalDistances = queries.count * batchCandidates.count
        print("""
            Batch computation: \(queries.count)x\(batchCandidates.count) = \(totalDistances) distances
            Time: \(String(format: "%.2f", batchTime * 1000))ms
            Throughput: \(String(format: "%.0f", Double(totalDistances) / batchTime)) distances/sec
            """)
        
        // Get and display performance summary
        print("\n📈 Performance Profile Summary")
        print("==============================\n")
        
        let summary = await performanceProfiler.getPerformanceSummary()
        
        print("Total operations: \(summary.totalOperations)")
        print("\nOperation breakdown:")
        for (category, categorySummary) in summary.categorySummaries {
            print("""
                  \(category.rawValue): \
                  \(categorySummary.totalOperations) ops, \
                  avg \(String(format: "%.2f", categorySummary.averageTime * 1000))ms
                  """)
        }
        
        print("\nTop kernels:")
        for kernel in summary.kernelSummaries.prefix(5) {
            print("""
                  \(kernel.name): \
                  \(kernel.executionCount) executions, \
                  avg \(kernel.averageThreadsPerExecution) threads
                  """)
        }
        
        print("\nMemory usage:")
        print("  Current: \(formatBytes(summary.currentMemoryUsage))")
        print("  Peak: \(formatBytes(summary.peakMemoryUsage))")
        
        // Benchmark thread configurations
        print("\n🏃 Benchmarking thread configurations")
        print("=====================================\n")
        
        if let function = await device.makeFunction(name: "euclideanDistance"),
           let pipeline = try? await device.makeComputePipelineState(function: function) {
            
            let benchmarkResults = try await threadOptimizer.benchmarkConfigurations(
                for: pipeline,
                workSize: 50000,
                testIterations: 5
            )
            
            print("Configuration benchmark results:")
            for (index, config) in benchmarkResults.configurations.enumerated() {
                let isBest = config.configuration.threadsPerThreadgroup.width == 
                            benchmarkResults.bestConfiguration.threadsPerThreadgroup.width
                let marker = isBest ? "⭐️" : "  "
                print("""
                    \(marker) \(config.configuration.threadsPerThreadgroup.width) threads: \
                    \(String(format: "%.3f", config.averageTime * 1000))ms
                    """)
            }
            print("\nBest configuration speedup: \(String(format: "%.2f", benchmarkResults.speedupFactor))x")
        }
        
        // Export profiling report
        print("\n💾 Exporting profiling report...")
        let report = await performanceProfiler.exportProfilingData()
        
        // Save to file
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        encoder.dateEncodingStrategy = .iso8601
        
        if let jsonData = try? encoder.encode(report) {
            let filename = "profiling_report_\(Date().timeIntervalSince1970).json"
            let url = URL(fileURLWithPath: filename)
            try jsonData.write(to: url)
            print("Report saved to: \(filename)")
        }
        
        print("\n✅ Thread optimization demo complete!")
    }
    
    // Helper to generate random vectors
    static func generateRandomVector(dimension: Int) -> [Float] {
        return (0..<dimension).map { _ in Float.random(in: -1...1) }
    }
    
    // Helper to format bytes
    static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}

// Extension to create random Vector512
extension Vector512 {
    static func random() -> Vector512 {
        var values = [Float](repeating: 0, count: 512)
        for i in 0..<512 {
            values[i] = Float.random(in: -1...1)
        }
        return Vector512(values)
    }
}
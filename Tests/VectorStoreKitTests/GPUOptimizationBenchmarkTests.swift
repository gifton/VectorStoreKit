// GPUOptimizationBenchmarkTests.swift
// VectorStoreKit
//
// Comprehensive benchmarks for GPU optimization improvements

import XCTest
@testable import VectorStoreKit
import Metal

final class GPUOptimizationBenchmarkTests: XCTestCase {
    
    var metalDevice: MetalDevice!
    var bufferPool: MetalBufferPool!
    var pipelineManager: MetalPipelineManager!
    var profiler: MetalProfiler!
    var commandBufferPool: MetalCommandBufferPool!
    var batchOptimizer: MetalBatchOptimizer!
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Initialize Metal components
        metalDevice = try await MetalDevice()
        
        let poolConfig = MetalBufferPoolConfiguration(
            initialPoolSize: 32,
            maxPoolSize: 128,
            bufferSizeClasses: [1024, 4096, 16384, 65536, 262144, 1048576]
        )
        bufferPool = await MetalBufferPool(device: metalDevice, configuration: poolConfig)
        
        pipelineManager = MetalPipelineManager(device: metalDevice)
        profiler = MetalProfiler(enabled: true)
        commandBufferPool = MetalCommandBufferPool(device: metalDevice, profiler: profiler)
        batchOptimizer = await MetalBatchOptimizer(device: metalDevice, profiler: profiler)
        
        // Pre-warm pools
        try await commandBufferPool.prewarm(count: 8)
        await bufferPool.prewarmPool()
    }
    
    override func tearDown() async throws {
        // Get final statistics
        let stats = await profiler.getStatistics()
        print("\n=== GPU Optimization Benchmark Results ===")
        print("Total operations: \(stats.totalOperations)")
        print("Average GPU utilization: \(String(format: "%.1f%%", stats.averageGPUUtilization * 100))")
        print("Peak GPU utilization: \(String(format: "%.1f%%", stats.peakGPUUtilization * 100))")
        print("Average latency: \(String(format: "%.3f ms", stats.averageLatency * 1000))")
        print("Command buffer pool hit rate: \(String(format: "%.1f%%", stats.gpuMetrics.commandBufferPoolHitRate * 100))")
        print("Peak GPU memory: \(ByteCountFormatter.string(fromByteCount: Int64(stats.gpuMetrics.peakGPUMemoryUsage), countStyle: .binary))")
        
        await profiler.reset()
        await commandBufferPool.clear()
        await bufferPool.clear()
        
        try await super.tearDown()
    }
    
    // MARK: - Distance Computation Benchmarks
    
    func testOptimizedDistanceComputation() async throws {
        let distanceCompute = MetalDistanceCompute(
            device: metalDevice,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager,
            profiler: profiler,
            commandBufferPool: commandBufferPool
        )
        
        // Test different vector dimensions
        let dimensions = [128, 256, 512, 1024, 2048]
        let candidateCounts = [100, 1000, 10000, 50000]
        
        print("\n--- Distance Computation Benchmarks ---")
        
        for dim in dimensions {
            for count in candidateCounts {
                // Generate test data
                let query = (0..<dim).map { _ in Float.random(in: -1...1) }
                let candidates = (0..<count).map { _ in
                    (0..<dim).map { _ in Float.random(in: -1...1) }
                }
                
                // Benchmark
                let startTime = CFAbsoluteTimeGetCurrent()
                
                let distances = try await distanceCompute.computeDistances(
                    query: query,
                    candidates: candidates,
                    metric: .euclidean
                )
                
                let duration = CFAbsoluteTimeGetCurrent() - startTime
                let throughput = Double(count) / duration
                
                print("Dim: \(dim), Candidates: \(count) - Time: \(String(format: "%.3f", duration))s, Throughput: \(String(format: "%.0f", throughput)) vectors/sec")
                
                XCTAssertEqual(distances.count, count)
            }
        }
    }
    
    func testOptimized512DimensionalVectors() async throws {
        let distanceCompute = MetalDistanceCompute(
            device: metalDevice,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager,
            profiler: profiler,
            commandBufferPool: commandBufferPool
        )
        
        print("\n--- 512-Dimensional Vector Optimization ---")
        
        // Create Vector512 instances
        let query = Vector512.random()
        let candidateCounts = [1000, 10000, 50000, 100000]
        
        for count in candidateCounts {
            let candidates = (0..<count).map { _ in Vector512.random() }
            
            // Benchmark optimized 512-dim computation
            let startTime = CFAbsoluteTimeGetCurrent()
            
            let distances = try await distanceCompute.computeDistances512(
                query: query,
                candidates: candidates,
                metric: .euclidean,
                normalized: false
            )
            
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            let throughput = Double(count) / duration
            
            print("512-dim vectors: \(count) - Time: \(String(format: "%.3f", duration))s, Throughput: \(String(format: "%.0f", throughput)) vectors/sec")
            
            XCTAssertEqual(distances.count, count)
        }
    }
    
    // MARK: - Batch Processing Benchmarks
    
    func testBatchProcessingOptimization() async throws {
        let distanceCompute = MetalDistanceCompute(
            device: metalDevice,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager,
            profiler: profiler,
            commandBufferPool: commandBufferPool
        )
        
        print("\n--- Batch Processing Optimization ---")
        
        let dimension = 512
        let numQueries = 100
        let numCandidates = 10000
        
        // Generate test data
        let queries = (0..<numQueries).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
        let candidates = (0..<numCandidates).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
        
        // Benchmark batch computation
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let batchDistances = try await distanceCompute.batchComputeDistances(
            queries: queries,
            candidates: candidates,
            metric: .euclidean
        )
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        let totalComputations = numQueries * numCandidates
        let throughput = Double(totalComputations) / duration
        
        print("Batch: \(numQueries)x\(numCandidates) - Time: \(String(format: "%.3f", duration))s, Throughput: \(String(format: "%.0f", throughput)) computations/sec")
        
        XCTAssertEqual(batchDistances.count, numQueries)
        XCTAssertEqual(batchDistances[0].count, numCandidates)
    }
    
    // MARK: - Matrix Operation Benchmarks
    
    func testOptimizedMatrixOperations() async throws {
        let matrixCompute = await MetalMatrixCompute(
            device: metalDevice,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager,
            profiler: profiler,
            commandBufferPool: commandBufferPool,
            batchOptimizer: batchOptimizer
        )
        
        print("\n--- Matrix Operation Benchmarks ---")
        
        let matrixSizes = [64, 128, 256, 512]
        
        for size in matrixSizes {
            // Generate test matrices
            let matrixA = (0..<size).map { _ in
                (0..<size).map { _ in Float.random(in: -1...1) }
            }
            let matrixB = (0..<size).map { _ in
                (0..<size).map { _ in Float.random(in: -1...1) }
            }
            
            // Benchmark single matrix multiplication
            let startTime = CFAbsoluteTimeGetCurrent()
            
            let result = try await matrixCompute.matrixMultiply(
                matrixA: matrixA,
                matrixB: matrixB
            )
            
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            let gflops = (2.0 * Double(size * size * size)) / (duration * 1e9)
            
            print("Matrix multiply \(size)x\(size) - Time: \(String(format: "%.3f", duration))s, GFLOPS: \(String(format: "%.1f", gflops))")
            
            XCTAssertEqual(result.count, size)
            XCTAssertEqual(result[0].count, size)
        }
    }
    
    func testBatchMatrixMultiplication() async throws {
        let matrixCompute = await MetalMatrixCompute(
            device: metalDevice,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager,
            profiler: profiler,
            commandBufferPool: commandBufferPool,
            batchOptimizer: batchOptimizer
        )
        
        print("\n--- Batch Matrix Multiplication ---")
        
        let batchSizes = [10, 50, 100, 200]
        let matrixSize = 128
        
        for batchSize in batchSizes {
            // Generate batch of matrix pairs
            let pairs = (0..<batchSize).map { _ in
                let matrixA = (0..<matrixSize).map { _ in
                    (0..<matrixSize).map { _ in Float.random(in: -1...1) }
                }
                let matrixB = (0..<matrixSize).map { _ in
                    (0..<matrixSize).map { _ in Float.random(in: -1...1) }
                }
                return (matrixA: matrixA, matrixB: matrixB)
            }
            
            // Benchmark batch multiplication
            let startTime = CFAbsoluteTimeGetCurrent()
            
            let results = try await matrixCompute.batchMatrixMultiply(pairs: pairs)
            
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            let throughput = Double(batchSize) / duration
            let totalGflops = (2.0 * Double(batchSize * matrixSize * matrixSize * matrixSize)) / (duration * 1e9)
            
            print("Batch size: \(batchSize) - Time: \(String(format: "%.3f", duration))s, Throughput: \(String(format: "%.1f", throughput)) ops/sec, GFLOPS: \(String(format: "%.1f", totalGflops))")
            
            XCTAssertEqual(results.count, batchSize)
        }
    }
    
    // MARK: - Adaptive Batch Sizing
    
    func testAdaptiveBatchSizing() async throws {
        print("\n--- Adaptive Batch Sizing ---")
        
        // Test different operations
        let operations: [BatchOperation] = [
            .distanceComputation,
            .matrixMultiplication,
            .quantization,
            .normalization
        ]
        
        let totalElements = 100000
        let vectorDimension = 512
        
        for operation in operations {
            let config = await batchOptimizer.getOptimalBatchSize(
                operation: operation,
                dataType: .float32,
                vectorDimension: vectorDimension,
                totalElements: totalElements
            )
            
            print("\(operation): Batch size: \(config.batchSize), Batches: \(config.numBatches), Double buffering: \(config.useDoubleBUffering)")
        }
        
        // Simulate GPU load and test adaptation
        await batchOptimizer.updateGPUState(utilization: 0.9, memoryPressure: 0.7)
        
        let highLoadConfig = await batchOptimizer.getOptimalBatchSize(
            operation: .distanceComputation,
            dataType: .float32,
            vectorDimension: vectorDimension,
            totalElements: totalElements
        )
        
        print("High GPU load - Adapted batch size: \(highLoadConfig.batchSize)")
    }
    
    // MARK: - Command Buffer Pool Performance
    
    func testCommandBufferPoolPerformance() async throws {
        print("\n--- Command Buffer Pool Performance ---")
        
        let iterations = 1000
        var pooledTime: TimeInterval = 0
        var directTime: TimeInterval = 0
        
        // Test with pooling
        let poolStartTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            let buffer = try await commandBufferPool.getCommandBuffer()
            buffer.commit()
        }
        pooledTime = CFAbsoluteTimeGetCurrent() - poolStartTime
        
        // Test without pooling
        let directStartTime = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            guard let buffer = await metalDevice.makeCommandBuffer() else {
                throw MetalComputeError.commandBufferCreationFailed
            }
            buffer.commit()
        }
        directTime = CFAbsoluteTimeGetCurrent() - directStartTime
        
        let improvement = ((directTime - pooledTime) / directTime) * 100
        
        print("Command buffer creation - Pooled: \(String(format: "%.3f", pooledTime))s, Direct: \(String(format: "%.3f", directTime))s")
        print("Improvement: \(String(format: "%.1f%%", improvement))")
        
        let poolStats = await commandBufferPool.statistics
        print("Pool hit rate: \(String(format: "%.1f%%", poolStats.hitRate * 100))")
    }
    
    // MARK: - Memory Access Pattern Optimization
    
    func testOptimizedMemoryAccessPatterns() async throws {
        print("\n--- Memory Access Pattern Optimization ---")
        
        // Compare standard vs optimized shaders
        let distanceCompute = MetalDistanceCompute(
            device: metalDevice,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager,
            profiler: profiler,
            commandBufferPool: commandBufferPool
        )
        
        let dimension = 1024
        let candidateCount = 50000
        
        // Generate aligned data for optimal memory access
        let query = (0..<dimension).map { _ in Float.random(in: -1...1) }
        let candidates = (0..<candidateCount).map { _ in
            (0..<dimension).map { _ in Float.random(in: -1...1) }
        }
        
        // Pre-warm GPU
        _ = try await distanceCompute.computeDistances(
            query: Array(query.prefix(128)),
            candidates: Array(candidates.prefix(100)),
            metric: .euclidean
        )
        
        // Benchmark optimized computation
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let distances = try await distanceCompute.computeDistances(
            query: query,
            candidates: candidates,
            metric: .euclidean
        )
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        let bandwidth = Double(candidateCount * dimension * MemoryLayout<Float>.size * 2) / duration / 1e9
        
        print("Memory bandwidth achieved: \(String(format: "%.1f", bandwidth)) GB/s")
        print("Theoretical peak (M1 Pro/Max): ~400 GB/s")
        print("Efficiency: \(String(format: "%.1f%%", (bandwidth / 400) * 100))")
        
        XCTAssertEqual(distances.count, candidateCount)
    }
}

// MARK: - Helper Extensions

extension Vector512 {
    static func random() -> Vector512 {
        let values = (0..<512).map { _ in Float.random(in: -1...1) }
        return values.withUnsafeBufferPointer { buffer in
            Vector512(buffer)
        }
    }
}
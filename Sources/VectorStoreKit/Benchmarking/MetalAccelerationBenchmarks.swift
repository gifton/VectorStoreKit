// VectorStoreKit: Metal Acceleration Benchmarks
//
// GPU vs CPU performance benchmarks

import Foundation
import simd
import Metal
import MetalPerformanceShaders

/// Benchmarks for Metal GPU acceleration
public struct MetalAccelerationBenchmarks {
    
    private let framework: BenchmarkFramework
    private let metrics: PerformanceMetrics
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    
    public init(
        framework: BenchmarkFramework = BenchmarkFramework(),
        metrics: PerformanceMetrics = PerformanceMetrics()
    ) throws {
        self.framework = framework
        self.metrics = metrics
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw BenchmarkError.metalNotAvailable
        }
        self.device = device
        
        guard let queue = device.makeCommandQueue() else {
            throw BenchmarkError.metalInitializationFailed
        }
        self.commandQueue = queue
    }
    
    enum BenchmarkError: Error {
        case metalNotAvailable
        case metalInitializationFailed
        case pipelineCreationFailed
    }
    
    // MARK: - Main Benchmark Suites
    
    /// Run all Metal acceleration benchmarks
    public func runAll() async throws -> [String: BenchmarkFramework.Statistics] {
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        // Distance computation benchmarks
        results.merge(try await runDistanceComputeBenchmarks()) { _, new in new }
        
        // Matrix operation benchmarks
        results.merge(try await runMatrixOperationBenchmarks()) { _, new in new }
        
        // Memory transfer benchmarks
        results.merge(try await runMemoryTransferBenchmarks()) { _, new in new }
        
        // Kernel optimization benchmarks
        results.merge(try await runKernelOptimizationBenchmarks()) { _, new in new }
        
        return results
    }
    
    // MARK: - Distance Compute Benchmarks
    
    private func runDistanceComputeBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        // Create Metal compute dependencies
        let metalDevice = try MetalDevice()
        let bufferPool = MetalBufferPool(
            device: await metalDevice.device,
            configuration: MetalBufferPool.Configuration()
        )
        let pipelineManager = MetalPipelineManager(device: metalDevice)
        let metalCompute = MetalDistanceCompute(
            device: metalDevice,
            bufferPool: bufferPool,
            pipelineManager: pipelineManager
        )
        let vectorCounts = [1_000, 10_000, 100_000, 1_000_000]
        let dimensions = [128, 256, 512]
        
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        for dim in dimensions {
            for count in vectorCounts {
                let suite = benchmarkSuite(
                    name: "Metal Distance Compute (\(dim)D, \(count) vectors)",
                    description: "GPU vs CPU distance computations"
                ) {
                    // Generate test data
                    let queries = generateVectorBatch(count: 100, dimensions: dim)
                    let candidates = generateVectorBatch(count: count, dimensions: dim)
                    
                    // CPU baseline
                    benchmark(name: "cpu_euclidean_\(dim)d_\(count)") {
                        let distances = computeDistancesCPU(
                            queries: queries,
                            candidates: candidates,
                            metric: .euclidean
                        )
                        blackHole(distances)
                    }
                    
                    // GPU accelerated
                    benchmark(name: "gpu_euclidean_\(dim)d_\(count)") {
                        let distances = try await metalCompute.computeDistances(
                            queries: queries,
                            candidates: candidates,
                            metric: .euclidean
                        )
                        blackHole(distances)
                    }
                    
                    // CPU cosine
                    benchmark(name: "cpu_cosine_\(dim)d_\(count)") {
                        let distances = computeDistancesCPU(
                            queries: queries,
                            candidates: candidates,
                            metric: .cosine
                        )
                        blackHole(distances)
                    }
                    
                    // GPU cosine
                    benchmark(name: "gpu_cosine_\(dim)d_\(count)") {
                        let distances = try await metalCompute.computeDistances(
                            queries: queries,
                            candidates: candidates,
                            metric: .cosine
                        )
                        blackHole(distances)
                    }
                }
                
                let suiteResults = try await framework.run(suite: suite)
                results.merge(suiteResults) { _, new in new }
            }
        }
        
        return results
    }
    
    // MARK: - Matrix Operation Benchmarks
    
    private func runMatrixOperationBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let sizes = [256, 512, 1024, 2048]
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        for size in sizes {
            let suite = benchmarkSuite(
                name: "Matrix Operations (\(size)x\(size))",
                description: "GPU-accelerated matrix operations"
            ) {
                // Matrix multiplication
                benchmark(name: "gpu_matmul_\(size)") {
                    try await benchmarkMatrixMultiply(size: size)
                }
                
                // Matrix transpose
                benchmark(name: "gpu_transpose_\(size)") {
                    try await benchmarkMatrixTranspose(size: size)
                }
                
                // Element-wise operations
                benchmark(name: "gpu_elementwise_\(size)") {
                    try await benchmarkElementwiseOps(size: size)
                }
                
                // Reduction operations
                benchmark(name: "gpu_reduction_\(size)") {
                    try await benchmarkReduction(size: size)
                }
            }
            
            let suiteResults = try await framework.run(suite: suite)
            results.merge(suiteResults) { _, new in new }
        }
        
        return results
    }
    
    // MARK: - Memory Transfer Benchmarks
    
    private func runMemoryTransferBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Memory Transfers",
            description: "CPU-GPU memory transfer performance"
        ) {
            let sizes = [1_000_000, 10_000_000, 100_000_000] // Elements
            
            for size in sizes {
                let data = [Float](repeating: 1.0, count: size)
                let byteSize = size * MemoryLayout<Float>.stride
                
                // Shared memory (zero-copy)
                benchmark(name: "shared_memory_\(formatBytes(byteSize))") {
                    let buffer = device.makeBuffer(
                        bytes: data,
                        length: byteSize,
                        options: .storageModeShared
                    )
                    blackHole(buffer)
                }
                
                // Private memory (GPU-only)
                benchmark(name: "private_memory_\(formatBytes(byteSize))") {
                    let buffer = device.makeBuffer(
                        length: byteSize,
                        options: .storageModePrivate
                    )
                    
                    // Need blit encoder to copy data
                    let commandBuffer = commandQueue.makeCommandBuffer()!
                    let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
                    
                    let stagingBuffer = device.makeBuffer(
                        bytes: data,
                        length: byteSize,
                        options: .storageModeShared
                    )!
                    
                    blitEncoder.copy(
                        from: stagingBuffer,
                        sourceOffset: 0,
                        to: buffer!,
                        destinationOffset: 0,
                        size: byteSize
                    )
                    
                    blitEncoder.endEncoding()
                    commandBuffer.commit()
                    commandBuffer.waitUntilCompleted()
                    
                    blackHole(buffer)
                }
                
                // Managed memory (automatic sync)
                benchmark(name: "managed_memory_\(formatBytes(byteSize))") {
                    let buffer = device.makeBuffer(
                        bytes: data,
                        length: byteSize,
                        options: .storageModeManaged
                    )
                    blackHole(buffer)
                }
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Kernel Optimization Benchmarks
    
    private func runKernelOptimizationBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Kernel Optimizations",
            description: "Different kernel optimization strategies"
        ) {
            let vectorCount = 100_000
            let dimensions = 512
            
            // Naive kernel
            benchmark(name: "naive_distance_kernel") {
                try await benchmarkNaiveKernel(
                    vectorCount: vectorCount,
                    dimensions: dimensions
                )
            }
            
            // Optimized with shared memory
            benchmark(name: "shared_memory_kernel") {
                try await benchmarkSharedMemoryKernel(
                    vectorCount: vectorCount,
                    dimensions: dimensions
                )
            }
            
            // Vectorized kernel
            benchmark(name: "vectorized_kernel") {
                try await benchmarkVectorizedKernel(
                    vectorCount: vectorCount,
                    dimensions: dimensions
                )
            }
            
            // Tiled kernel
            benchmark(name: "tiled_kernel") {
                try await benchmarkTiledKernel(
                    vectorCount: vectorCount,
                    dimensions: dimensions
                )
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Helper Functions
    
    private func generateVectorBatch(count: Int, dimensions: Int) -> [[Float]] {
        return (0..<count).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }
    }
    
    private func computeDistancesCPU(
        queries: [[Float]],
        candidates: [[Float]],
        metric: DistanceMetric
    ) -> [[Float]] {
        var results = [[Float]](repeating: [Float](repeating: 0, count: candidates.count), count: queries.count)
        
        for (i, query) in queries.enumerated() {
            for (j, candidate) in candidates.enumerated() {
                results[i][j] = computeDistance(query, candidate, metric: metric)
            }
        }
        
        return results
    }
    
    private func computeDistance(_ a: [Float], _ b: [Float], metric: DistanceMetric) -> Float {
        switch metric {
        case .euclidean:
            var sum: Float = 0
            for i in 0..<a.count {
                let diff = a[i] - b[i]
                sum += diff * diff
            }
            return sqrt(sum)
            
        case .cosine:
            var dot: Float = 0
            var normA: Float = 0
            var normB: Float = 0
            for i in 0..<a.count {
                dot += a[i] * b[i]
                normA += a[i] * a[i]
                normB += b[i] * b[i]
            }
            return 1.0 - (dot / (sqrt(normA) * sqrt(normB)))
            
        case .manhattan:
            var sum: Float = 0
            for i in 0..<a.count {
                sum += abs(a[i] - b[i])
            }
            return sum
        }
    }
    
    private func benchmarkMatrixMultiply(size: Int) async throws {
        let matrixA = device.makeBuffer(
            length: size * size * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        
        let matrixB = device.makeBuffer(
            length: size * size * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        
        let matrixC = device.makeBuffer(
            length: size * size * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )!
        
        // Use Metal Performance Shaders for matrix multiplication
        let matmul = MPSMatrixMultiplication(
            device: device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: size,
            resultColumns: size,
            interiorColumns: size,
            alpha: 1.0,
            beta: 0.0
        )
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        let matrixDescriptor = MPSMatrixDescriptor(
            rows: size,
            columns: size,
            rowBytes: size * MemoryLayout<Float>.stride,
            dataType: .float32
        )
        
        let mpsA = MPSMatrix(buffer: matrixA, descriptor: matrixDescriptor)
        let mpsB = MPSMatrix(buffer: matrixB, descriptor: matrixDescriptor)
        let mpsC = MPSMatrix(buffer: matrixC, descriptor: matrixDescriptor)
        
        matmul.encode(
            commandBuffer: commandBuffer,
            leftMatrix: mpsA,
            rightMatrix: mpsB,
            resultMatrix: mpsC
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        blackHole(matrixC)
    }
    
    private func benchmarkMatrixTranspose(size: Int) async throws {
        // Implementation would use custom Metal kernel
        blackHole(size)
    }
    
    private func benchmarkElementwiseOps(size: Int) async throws {
        // Implementation would use custom Metal kernel
        blackHole(size)
    }
    
    private func benchmarkReduction(size: Int) async throws {
        // Implementation would use custom Metal kernel or MPS
        blackHole(size)
    }
    
    private func benchmarkNaiveKernel(vectorCount: Int, dimensions: Int) async throws {
        // Implementation would use basic Metal kernel
        blackHole((vectorCount, dimensions))
    }
    
    private func benchmarkSharedMemoryKernel(vectorCount: Int, dimensions: Int) async throws {
        // Implementation would use Metal kernel with threadgroup memory
        blackHole((vectorCount, dimensions))
    }
    
    private func benchmarkVectorizedKernel(vectorCount: Int, dimensions: Int) async throws {
        // Implementation would use Metal kernel with vector types
        blackHole((vectorCount, dimensions))
    }
    
    private func benchmarkTiledKernel(vectorCount: Int, dimensions: Int) async throws {
        // Implementation would use Metal kernel with tiling
        blackHole((vectorCount, dimensions))
    }
    
    private func formatBytes(_ bytes: Int) -> String {
        let units = ["B", "KB", "MB", "GB"]
        var size = Double(bytes)
        var unitIndex = 0
        
        while size >= 1024 && unitIndex < units.count - 1 {
            size /= 1024
            unitIndex += 1
        }
        
        return String(format: "%.0f%@", size, units[unitIndex])
    }
}

// MARK: - GPU vs CPU Comparison Report

public extension MetalAccelerationBenchmarks {
    
    struct ComparisonReport {
        public let operation: String
        public let dimensions: Int
        public let vectorCount: Int
        public let cpuTime: TimeInterval
        public let gpuTime: TimeInterval
        public let speedup: Double
        public let efficiency: Double // GPU efficiency percentage
        
        public var summary: String {
            return String(format: """
                %@:
                  Dimensions: %d
                  Vectors: %d
                  CPU Time: %.3f ms
                  GPU Time: %.3f ms
                  Speedup: %.2fx
                  Efficiency: %.1f%%
                """,
                operation,
                dimensions,
                vectorCount,
                cpuTime * 1000,
                gpuTime * 1000,
                speedup,
                efficiency * 100
            )
        }
    }
    
    func generateComparisonReport(
        from results: [String: BenchmarkFramework.Statistics]
    ) -> [ComparisonReport] {
        var reports: [ComparisonReport] = []
        
        // Parse results and create comparison reports
        // Implementation would match CPU and GPU results
        
        return reports
    }
}
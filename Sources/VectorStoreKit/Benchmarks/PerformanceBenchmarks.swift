// VectorStoreKit: Performance Benchmarks
//
// Specialized benchmarks for performance optimization and edge cases

import Foundation
import simd
import Accelerate

/// Performance-focused benchmarks for VectorStoreKit
public class PerformanceBenchmarks {
    
    // MARK: - Distance Computation Benchmarks
    
    /// Benchmark different distance computation methods
    public static func benchmarkDistanceComputations() async throws {
        print("\n📏 Distance Computation Benchmarks")
        print("=" * 50)
        
        let dimensions = [64, 128, 256, 512, 1024]
        let vectorCount = 10_000
        
        for dim in dimensions {
            print("\nDimension: \(dim)")
            
            // Generate test vectors
            let vectors = (0..<vectorCount).map { _ in
                (0..<dim).map { _ in Float.random(in: -1...1) }
            }
            let query = (0..<dim).map { _ in Float.random(in: -1...1) }
            
            // Benchmark different methods
            let methods: [(String, ([Float], [Float]) -> Float)] = [
                ("Naive", naiveEuclideanDistance),
                ("SIMD", simdEuclideanDistance),
                ("Accelerate", accelerateEuclideanDistance),
                ("Metal", { a, b in 0.0 }) // Placeholder for Metal implementation
            ]
            
            for (name, method) in methods {
                if name == "Metal" { continue } // Skip Metal for now
                
                let start = DispatchTime.now()
                var sum: Float = 0
                
                for vector in vectors {
                    sum += method(query, vector)
                }
                
                let elapsed = elapsedTime(since: start)
                let throughput = Double(vectorCount) / elapsed
                
                print("  \(name): \(String(format: "%.3f", elapsed))s (\(String(format: "%.0f", throughput)) ops/s)")
            }
        }
    }
    
    // MARK: - Batch Processing Benchmarks
    
    /// Benchmark batch processing strategies
    public static func benchmarkBatchProcessing() async throws {
        print("\n📦 Batch Processing Benchmarks")
        print("=" * 50)
        
        let dimensions = 128
        let totalVectors = 100_000
        let batchSizes = [1, 10, 100, 1000, 10000]
        
        // Create test index
        let config = IVFConfiguration(
            dimensions: dimensions,
            numberOfCentroids: 256,
            numberOfProbes: 16
        )
        let index = try await IVFIndex<SIMD128<Float>, TestMetadata>(configuration: config)
        
        // Train index
        let trainingData = (0..<10_000).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }
        try await index.train(on: trainingData)
        
        // Test different batch sizes
        for batchSize in batchSizes {
            print("\nBatch size: \(batchSize)")
            
            // Prepare batches
            let vectors = (0..<totalVectors).map { i in
                VectorEntry(
                    id: "batch_\(i)",
                    vector: SIMD128<Float>(repeating: Float(i) / Float(totalVectors)),
                    metadata: TestMetadata(label: "batch")
                )
            }
            
            // Single insert benchmark
            if batchSize == 1 {
                let start = DispatchTime.now()
                
                for vector in vectors.prefix(1000) {
                    _ = try await index.insert(vector)
                }
                
                let elapsed = elapsedTime(since: start)
                print("  Single inserts: \(elapsed)s (\(1000.0/elapsed) inserts/s)")
            }
            
            // Batch insert benchmark
            let start = DispatchTime.now()
            
            for batchStart in stride(from: 0, to: min(10000, vectors.count), by: batchSize) {
                let batch = Array(vectors[batchStart..<min(batchStart + batchSize, vectors.count)])
                
                // Simulate batch insert (would need actual batch API)
                for entry in batch {
                    _ = try await index.insert(entry)
                }
            }
            
            let elapsed = elapsedTime(since: start)
            let vectorsProcessed = min(10000, vectors.count)
            print("  Batch inserts: \(elapsed)s (\(Double(vectorsProcessed)/elapsed) inserts/s)")
        }
    }
    
    // MARK: - Memory Allocation Benchmarks
    
    /// Benchmark memory allocation patterns
    public static func benchmarkMemoryAllocation() async throws {
        print("\n💾 Memory Allocation Benchmarks")
        print("=" * 50)
        
        let dimensions = 128
        let configurations: [(String, Int)] = [
            ("Small buckets", 10),
            ("Medium buckets", 100),
            ("Large buckets", 1000),
            ("Huge buckets", 10000)
        ]
        
        for (name, bucketSize) in configurations {
            print("\n\(name) (size: \(bucketSize)):")
            
            let config = LearnedIndexConfiguration(
                dimensions: dimensions,
                modelArchitecture: .linear,
                bucketSize: bucketSize
            )
            
            let index = try await LearnedIndex<SIMD128<Float>, TestMetadata>(configuration: config)
            
            // Measure allocation patterns
            var memorySnapshots: [Int] = []
            
            for i in 0..<10_000 {
                let entry = VectorEntry(
                    id: "mem_\(i)",
                    vector: SIMD128<Float>(repeating: Float(i % 100)),
                    metadata: TestMetadata(label: "memory")
                )
                _ = try await index.insert(entry)
                
                if i % 1000 == 0 {
                    let memory = await index.memoryUsage
                    memorySnapshots.append(memory)
                }
            }
            
            // Analyze memory growth
            let avgGrowth = memorySnapshots.enumerated().dropFirst().map { i, mem in
                mem - memorySnapshots[i - 1]
            }.reduce(0, +) / max(1, memorySnapshots.count - 1)
            
            print("  Initial memory: \(formatBytes(memorySnapshots.first ?? 0))")
            print("  Final memory: \(formatBytes(memorySnapshots.last ?? 0))")
            print("  Average growth: \(formatBytes(avgGrowth))/1k vectors")
        }
    }
    
    // MARK: - Query Optimization Benchmarks
    
    /// Benchmark query optimization strategies
    public static func benchmarkQueryOptimization() async throws {
        print("\n🔍 Query Optimization Benchmarks")
        print("=" * 50)
        
        let dimensions = 128
        let vectorCount = 50_000
        
        // Create and populate hybrid index
        let config = HybridIndexConfiguration(
            dimensions: dimensions,
            routingStrategy: .adaptive
        )
        let index = try await HybridIndex<SIMD128<Float>, TestMetadata>(configuration: config)
        
        // Generate and insert data
        let vectors = (0..<vectorCount).map { i in
            let pattern = i % 3
            return (0..<dimensions).map { d in
                switch pattern {
                case 0: return Float(d) / Float(dimensions) // Linear
                case 1: return sin(Float(d) * 0.1) // Periodic
                default: return Float.random(in: -1...1) // Random
                }
            }
        }
        
        try await index.train(on: Array(vectors.prefix(10_000)))
        
        for (i, vector) in vectors.enumerated() {
            let entry = VectorEntry(
                id: "opt_\(i)",
                vector: SIMD128<Float>(vector),
                metadata: TestMetadata(label: "pattern_\(i % 3)")
            )
            _ = try await index.insert(entry)
        }
        
        // Test different query strategies
        let strategies: [(String, SearchStrategy)] = [
            ("Exact", .exact),
            ("Approximate (0.9)", .approximate(quality: 0.9)),
            ("Approximate (0.5)", .approximate(quality: 0.5)),
            ("Adaptive", .adaptive),
            ("Hybrid", .hybrid),
            ("Learned", .learned)
        ]
        
        for (name, strategy) in strategies {
            print("\nStrategy: \(name)")
            
            let queries = (0..<100).map { _ in
                SIMD128<Float>((0..<dimensions).map { _ in Float.random(in: -1...1) })
            }
            
            let start = DispatchTime.now()
            var totalResults = 0
            
            for query in queries {
                let results = try await index.search(
                    query: query,
                    k: 10,
                    strategy: strategy
                )
                totalResults += results.count
            }
            
            let elapsed = elapsedTime(since: start)
            let qps = Double(queries.count) / elapsed
            
            print("  Time: \(String(format: "%.3f", elapsed))s")
            print("  QPS: \(String(format: "%.0f", qps))")
            print("  Avg results: \(totalResults / queries.count)")
        }
    }
    
    // MARK: - Concurrent Access Benchmarks
    
    /// Benchmark concurrent read/write patterns
    public static func benchmarkConcurrentAccess() async throws {
        print("\n🔄 Concurrent Access Benchmarks")
        print("=" * 50)
        
        let dimensions = 64
        let config = HybridIndexConfiguration(dimensions: dimensions)
        let index = try await HybridIndex<SIMD64<Float>, TestMetadata>(configuration: config)
        
        // Initialize with some data
        let initialVectors = (0..<1000).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }
        try await index.train(on: initialVectors)
        
        // Test different read/write ratios
        let ratios: [(String, Double)] = [
            ("Read-heavy (90/10)", 0.1),
            ("Balanced (50/50)", 0.5),
            ("Write-heavy (10/90)", 0.9)
        ]
        
        for (name, writeRatio) in ratios {
            print("\n\(name):")
            
            let operationCount = 10_000
            let start = DispatchTime.now()
            
            await withTaskGroup(of: Void.self) { group in
                // Spawn concurrent tasks
                for taskId in 0..<8 {
                    group.addTask {
                        for op in 0..<(operationCount / 8) {
                            let isWrite = Double.random(in: 0...1) < writeRatio
                            
                            if isWrite {
                                // Write operation
                                let vector = (0..<dimensions).map { _ in Float.random(in: -1...1) }
                                let entry = VectorEntry(
                                    id: "concurrent_\(taskId)_\(op)",
                                    vector: SIMD64<Float>(vector),
                                    metadata: TestMetadata(label: "concurrent")
                                )
                                _ = try? await index.insert(entry)
                            } else {
                                // Read operation
                                let query = SIMD64<Float>(repeating: Float.random(in: -1...1))
                                _ = try? await index.search(query: query, k: 5)
                            }
                        }
                    }
                }
            }
            
            let elapsed = elapsedTime(since: start)
            let opsPerSec = Double(operationCount) / elapsed
            
            print("  Total time: \(String(format: "%.3f", elapsed))s")
            print("  Operations/sec: \(String(format: "%.0f", opsPerSec))")
        }
    }
    
    // MARK: - Helper Functions
    
    private static func naiveEuclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<min(a.count, b.count) {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    private static func simdEuclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        let count = min(a.count, b.count)
        var sum: Float = 0
        
        // Process in SIMD chunks
        let simdCount = count & ~3 // Round down to multiple of 4
        
        for i in stride(from: 0, to: simdCount, by: 4) {
            let va = SIMD4<Float>(a[i], a[i+1], a[i+2], a[i+3])
            let vb = SIMD4<Float>(b[i], b[i+1], b[i+2], b[i+3])
            let diff = va - vb
            sum += (diff * diff).sum()
        }
        
        // Handle remaining elements
        for i in simdCount..<count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        
        return sqrt(sum)
    }
    
    private static func accelerateEuclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        let count = min(a.count, b.count)
        var diff = [Float](repeating: 0, count: count)
        
        // Compute difference
        vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(count))
        
        // Compute squared distance
        var squaredDistance: Float = 0
        vDSP_dotpr(diff, 1, diff, 1, &squaredDistance, vDSP_Length(count))
        
        return sqrt(squaredDistance)
    }
    
    private static func elapsedTime(since start: DispatchTime) -> TimeInterval {
        let end = DispatchTime.now()
        return Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    }
    
    private static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}

// MARK: - Specialized Index Benchmarks

extension PerformanceBenchmarks {
    
    /// Benchmark IVF-specific optimizations
    public static func benchmarkIVFOptimizations() async throws {
        print("\n🎯 IVF Optimization Benchmarks")
        print("=" * 50)
        
        let dimensions = 128
        let vectorCount = 100_000
        
        // Test different centroid counts
        let centroidCounts = [64, 256, 1024, 4096]
        
        for centroids in centroidCounts {
            print("\nCentroids: \(centroids)")
            
            let config = IVFConfiguration(
                dimensions: dimensions,
                numberOfCentroids: centroids,
                numberOfProbes: max(1, centroids / 10),
                trainingSampleSize: min(100_000, vectorCount)
            )
            
            let index = try await IVFIndex<SIMD128<Float>, TestMetadata>(configuration: config)
            
            // Generate clustered data
            let vectors = generateClusteredVectors(
                count: vectorCount,
                dimensions: dimensions,
                clusters: centroids / 4
            )
            
            // Training time
            let trainStart = DispatchTime.now()
            try await index.train(on: Array(vectors.prefix(config.trainingSampleSize)))
            let trainTime = elapsedTime(since: trainStart)
            
            // Insert subset
            for (i, vector) in vectors.prefix(10_000).enumerated() {
                let entry = VectorEntry(
                    id: "ivf_\(i)",
                    vector: SIMD128<Float>(vector),
                    metadata: TestMetadata(label: "ivf")
                )
                _ = try await index.insert(entry)
            }
            
            // Search performance with different probe counts
            let probeCounts = [1, centroids / 20, centroids / 10, centroids / 5]
            
            for probes in probeCounts {
                let searchConfig = IVFSearchConfiguration(
                    multiProbeConfig: MultiProbeConfiguration(
                        baseProbes: probes,
                        maxProbes: probes * 2
                    )
                )
                
                let engine = IVFSearchEngine(configuration: searchConfig)
                
                let queries = vectors.suffix(100).map { SIMD128<Float>($0) }
                let searchStart = DispatchTime.now()
                
                for query in queries {
                    _ = try await engine.search(
                        query: query,
                        k: 10,
                        in: index
                    )
                }
                
                let searchTime = elapsedTime(since: searchStart)
                let qps = Double(queries.count) / searchTime
                
                print("  Probes: \(probes), QPS: \(String(format: "%.0f", qps))")
            }
            
            print("  Training time: \(String(format: "%.3f", trainTime))s")
            print("  Memory usage: \(formatBytes(await index.memoryUsage))")
        }
    }
    
    /// Benchmark Learned index architectures
    public static func benchmarkLearnedArchitectures() async throws {
        print("\n🧠 Learned Architecture Benchmarks")
        print("=" * 50)
        
        let dimensions = 64
        let vectorCount = 10_000
        
        // Test different architectures
        let architectures: [(String, LearnedIndexConfiguration.ModelArchitecture)] = [
            ("Linear", .linear),
            ("Small MLP", .mlp(hiddenSizes: [128])),
            ("Medium MLP", .mlp(hiddenSizes: [256, 128])),
            ("Large MLP", .mlp(hiddenSizes: [512, 256, 128])),
            ("Residual-2", .residual(layers: 2, hiddenSize: 256)),
            ("Residual-4", .residual(layers: 4, hiddenSize: 256))
        ]
        
        // Generate learnable patterns
        let vectors = (0..<vectorCount).map { i in
            let position = Float(i) / Float(vectorCount)
            return (0..<dimensions).map { d in
                // Mix of learnable patterns
                let linear = Float(d) * position / Float(dimensions)
                let periodic = sin(Float(d) * position * .pi)
                let noise = Float.random(in: -0.05...0.05)
                return linear * 0.5 + periodic * 0.5 + noise
            }
        }
        
        for (name, architecture) in architectures {
            print("\n\(name):")
            
            let config = LearnedIndexConfiguration(
                dimensions: dimensions,
                modelArchitecture: architecture,
                bucketSize: 100,
                trainingConfig: .init(
                    epochs: 20,
                    batchSize: 32,
                    learningRate: 0.001
                )
            )
            
            let index = try await LearnedIndex<SIMD64<Float>, TestMetadata>(configuration: config)
            
            // Insert and train
            for (i, vector) in vectors.enumerated() {
                let entry = VectorEntry(
                    id: "learned_\(i)",
                    vector: SIMD64<Float>(vector),
                    metadata: TestMetadata(label: "learned")
                )
                _ = try await index.insert(entry)
            }
            
            let trainStart = DispatchTime.now()
            try await index.train()
            let trainTime = elapsedTime(since: trainStart)
            
            // Test prediction accuracy
            var correctPredictions = 0
            let testQueries = vectors.suffix(100)
            
            for (i, query) in testQueries.enumerated() {
                let results = try await index.search(
                    query: SIMD64<Float>(query),
                    k: 1
                )
                
                if results.first?.id == "learned_\(vectorCount - 100 + i)" {
                    correctPredictions += 1
                }
            }
            
            let accuracy = Float(correctPredictions) / Float(testQueries.count)
            
            print("  Training time: \(String(format: "%.3f", trainTime))s")
            print("  Memory usage: \(formatBytes(await index.memoryUsage))")
            print("  Accuracy: \(String(format: "%.1f", accuracy * 100))%")
        }
    }
    
    private static func generateClusteredVectors(
        count: Int,
        dimensions: Int,
        clusters: Int
    ) -> [[Float]] {
        var vectors: [[Float]] = []
        let vectorsPerCluster = count / clusters
        
        for cluster in 0..<clusters {
            // Generate cluster center
            let center = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            
            // Generate vectors around center
            for _ in 0..<vectorsPerCluster {
                let vector = center.map { $0 + Float.random(in: -0.2...0.2) }
                vectors.append(vector)
            }
        }
        
        // Fill remaining with random vectors
        while vectors.count < count {
            vectors.append((0..<dimensions).map { _ in Float.random(in: -1...1) })
        }
        
        return vectors.shuffled()
    }
}

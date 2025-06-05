// VectorStoreKit: Benchmark Example
//
// Example demonstrating comprehensive benchmarking of vector indexes

import Foundation
import VectorStoreKit

/// Example demonstrating how to run comprehensive benchmarks
@main
struct BenchmarkExample {
    
    static func main() async throws {
        print("🚀 VectorStoreKit Comprehensive Benchmark Example")
        print("=" * 60)
        print()
        
        // Run different benchmark scenarios
        try await runBasicBenchmarks()
        try await runPerformanceComparison()
        try await runScalabilityTest()
        try await runSpecializedBenchmarks()
        try await runRealWorldScenario()
        
        print("\n✅ All benchmarks completed!")
    }
    
    // MARK: - Basic Benchmarks
    
    static func runBasicBenchmarks() async throws {
        print("\n📊 Running Basic Benchmarks")
        print("-" * 40)
        
        // Create benchmark configuration
        let config = ComprehensiveBenchmarks.BenchmarkConfiguration(
            dimensions: 128,
            vectorCounts: [1_000, 10_000],
            queryCount: 100,
            k: 10,
            iterations: 3,
            warmupIterations: 1
        )
        
        // Run benchmarks
        let benchmarks = ComprehensiveBenchmarks(configuration: config)
        let report = try await benchmarks.runAllBenchmarks()
        
        // Display results
        print("\nBenchmark Results:")
        print("Best Insert Performance: \(report.bestInsertPerformance)")
        print("Best Search Performance: \(report.bestSearchPerformance)")
        print("Best Memory Efficiency: \(report.bestMemoryEfficiency)")
        print("Best Recall: \(report.bestRecall)")
        
        // Detailed results
        for result in report.results {
            print("\n\(result.indexType) (\(result.vectorCount) vectors):")
            print("  Insert: \(String(format: "%.2f", result.insertTime))s")
            print("  Search: \(String(format: "%.2f", result.searchTime))s")
            print("  Memory: \(formatBytes(result.memoryUsage))")
            print("  Recall: \(String(format: "%.2f%%", result.recall * 100))")
        }
    }
    
    // MARK: - Performance Comparison
    
    static func runPerformanceComparison() async throws {
        print("\n🏃 Running Performance Comparison")
        print("-" * 40)
        
        let dimensions = 128
        let vectorCount = 50_000
        
        // Generate test data
        print("Generating test data...")
        let vectors = generateMixedDataset(count: vectorCount, dimensions: dimensions)
        let queries = generateQueries(from: vectors, count: 100)
        
        // Benchmark each index type
        let results = try await benchmarkAllIndexTypes(
            vectors: vectors,
            queries: queries,
            dimensions: dimensions
        )
        
        // Compare results
        print("\nPerformance Comparison:")
        print("Index Type | Insert/s | Query/s | Memory | Recall")
        print("-" * 50)
        
        for (indexType, metrics) in results {
            print(String(format: "%-10s | %8.0f | %7.0f | %6s | %.2f%%",
                indexType,
                metrics.insertsPerSecond,
                metrics.queriesPerSecond,
                formatBytes(metrics.memoryUsage),
                metrics.recall * 100
            ))
        }
    }
    
    // MARK: - Scalability Test
    
    static func runScalabilityTest() async throws {
        print("\n📈 Running Scalability Test")
        print("-" * 40)
        
        let dimensions = 64
        let vectorCounts = [1_000, 10_000, 50_000, 100_000]
        
        // Test hybrid index scalability
        let config = HybridIndexConfiguration(
            dimensions: dimensions,
            routingStrategy: .adaptive
        )
        
        print("Testing Hybrid Index scalability...")
        
        for count in vectorCounts {
            let index = try await HybridIndex<SIMD64<Float>, TestMetadata>(
                configuration: config
            )
            
            // Generate data
            let vectors = generateScalableDataset(count: count, dimensions: dimensions)
            
            // Train if needed
            if count >= 1000 {
                let trainingStart = DispatchTime.now()
                try await index.train(on: Array(vectors.prefix(min(10_000, count))))
                let trainTime = elapsedTime(since: trainingStart)
                print("\n\(count) vectors - Training: \(String(format: "%.2f", trainTime))s")
            }
            
            // Measure insert performance
            let insertStart = DispatchTime.now()
            for (i, vector) in vectors.enumerated() {
                let entry = VectorEntry(
                    id: "scale_\(i)",
                    vector: SIMD64<Float>(vector),
                    metadata: TestMetadata(label: "scale")
                )
                _ = try await index.insert(entry)
                
                if i % 10_000 == 0 && i > 0 {
                    print("  Inserted \(i) vectors...")
                }
            }
            let insertTime = elapsedTime(since: insertStart)
            
            // Measure search performance
            let queries = generateQueries(from: vectors, count: 100)
            let searchStart = DispatchTime.now()
            
            for query in queries {
                _ = try await index.search(
                    query: SIMD64<Float>(query),
                    k: 10
                )
            }
            let searchTime = elapsedTime(since: searchStart)
            
            // Report metrics
            let memoryUsage = await index.memoryUsage
            print("  Insert: \(String(format: "%.2f", insertTime))s (\(String(format: "%.0f", Double(count)/insertTime)) vec/s)")
            print("  Search: \(String(format: "%.3f", searchTime))s (\(String(format: "%.0f", 100.0/searchTime)) qps)")
            print("  Memory: \(formatBytes(memoryUsage))")
        }
    }
    
    // MARK: - Specialized Benchmarks
    
    static func runSpecializedBenchmarks() async throws {
        print("\n🔬 Running Specialized Benchmarks")
        print("-" * 40)
        
        // Distance computation benchmarks
        print("\n1. Distance Computation Methods:")
        try await PerformanceBenchmarks.benchmarkDistanceComputations()
        
        // Batch processing benchmarks
        print("\n2. Batch Processing:")
        try await PerformanceBenchmarks.benchmarkBatchProcessing()
        
        // Memory allocation patterns
        print("\n3. Memory Allocation Patterns:")
        try await PerformanceBenchmarks.benchmarkMemoryAllocation()
        
        // Query optimization strategies
        print("\n4. Query Optimization:")
        try await PerformanceBenchmarks.benchmarkQueryOptimization()
        
        // Concurrent access patterns
        print("\n5. Concurrent Access:")
        try await PerformanceBenchmarks.benchmarkConcurrentAccess()
    }
    
    // MARK: - Real-World Scenario
    
    static func runRealWorldScenario() async throws {
        print("\n🌍 Running Real-World Scenario")
        print("-" * 40)
        
        // Simulate a recommendation system scenario
        print("\nScenario: Product Recommendation System")
        print("- 100,000 products with 256-dimensional embeddings")
        print("- Mixed query patterns (popular items + long tail)")
        print("- Continuous updates (new products)")
        
        let dimensions = 256
        let productCount = 100_000
        
        // Create hybrid index with adaptive routing
        let config = HybridIndexConfiguration(
            dimensions: dimensions,
            routingStrategy: .adaptive,
            adaptiveThreshold: 0.7,
            ivfConfig: IVFConfiguration(
                dimensions: dimensions,
                numberOfCentroids: 1024,
                numberOfProbes: 32
            ),
            learnedConfig: LearnedIndexConfiguration(
                dimensions: dimensions,
                modelArchitecture: .mlp(hiddenSizes: [512, 256]),
                bucketSize: 100
            )
        )
        
        let index = try await HybridIndex<SIMD256<Float>, ProductMetadata>(
            configuration: config
        )
        
        // Generate product embeddings
        print("\nGenerating product embeddings...")
        let products = generateProductEmbeddings(count: productCount, dimensions: dimensions)
        
        // Train index
        print("Training index...")
        let trainStart = DispatchTime.now()
        try await index.train(on: Array(products.prefix(20_000)))
        let trainTime = elapsedTime(since: trainStart)
        print("Training completed in \(String(format: "%.2f", trainTime))s")
        
        // Initial bulk load
        print("\nBulk loading products...")
        let loadStart = DispatchTime.now()
        
        for (i, embedding) in products.enumerated() {
            let entry = VectorEntry(
                id: "product_\(i)",
                vector: SIMD256<Float>(embedding),
                metadata: ProductMetadata(
                    name: "Product \(i)",
                    category: ["Electronics", "Clothing", "Books", "Home"][i % 4],
                    price: Float.random(in: 10...1000),
                    popularity: Float.random(in: 0...1)
                )
            )
            _ = try await index.insert(entry)
            
            if i % 10_000 == 0 && i > 0 {
                print("  Loaded \(i) products...")
            }
        }
        
        let loadTime = elapsedTime(since: loadStart)
        print("Bulk load completed in \(String(format: "%.2f", loadTime))s")
        
        // Simulate mixed query workload
        print("\nSimulating query workload...")
        
        // Popular items (frequent queries)
        let popularQueries = products.prefix(100).map { embedding in
            embedding.map { $0 + Float.random(in: -0.05...0.05) }
        }
        
        // Long tail queries
        let longTailQueries = products.suffix(1000).map { embedding in
            embedding.map { $0 + Float.random(in: -0.1...0.1) }
        }
        
        // New/unseen queries
        let newQueries = (0..<100).map { _ in
            generateProductEmbeddings(count: 1, dimensions: dimensions)[0]
        }
        
        // Run mixed workload
        let workloadStart = DispatchTime.now()
        var totalQueries = 0
        var totalResults = 0
        
        // Simulate time-based query pattern
        for minute in 0..<5 {
            print("\nMinute \(minute + 1):")
            
            // 70% popular, 20% long tail, 10% new
            let queries = (0..<100).map { i in
                if i < 70 {
                    return SIMD256<Float>(popularQueries[i % popularQueries.count])
                } else if i < 90 {
                    return SIMD256<Float>(longTailQueries[i % longTailQueries.count])
                } else {
                    return SIMD256<Float>(newQueries[i % newQueries.count])
                }
            }
            
            let minuteStart = DispatchTime.now()
            
            for query in queries {
                let results = try await index.search(
                    query: query,
                    k: 20,
                    strategy: .adaptive
                )
                totalResults += results.count
                totalQueries += 1
            }
            
            let minuteTime = elapsedTime(since: minuteStart)
            let qps = Double(queries.count) / minuteTime
            
            print("  Queries: \(queries.count), QPS: \(String(format: "%.0f", qps))")
            
            // Add new products (simulating catalog updates)
            if minute % 2 == 0 {
                let newProducts = generateProductEmbeddings(count: 100, dimensions: dimensions)
                for (i, embedding) in newProducts.enumerated() {
                    let entry = VectorEntry(
                        id: "new_product_\(minute)_\(i)",
                        vector: SIMD256<Float>(embedding),
                        metadata: ProductMetadata(
                            name: "New Product \(minute)-\(i)",
                            category: "New Arrivals",
                            price: Float.random(in: 50...500),
                            popularity: 0.0
                        )
                    )
                    _ = try await index.insert(entry)
                }
                print("  Added 100 new products")
            }
            
            // Check adaptive routing
            let stats = await index.statistics()
            print("  Routing - IVF: \(String(format: "%.1f%%", stats.routingStatistics.ivfRatio * 100)), Learned: \(String(format: "%.1f%%", stats.routingStatistics.learnedRatio * 100))")
        }
        
        let workloadTime = elapsedTime(since: workloadStart)
        
        // Final statistics
        print("\n📊 Final Statistics:")
        print("Total queries: \(totalQueries)")
        print("Average results per query: \(totalResults / totalQueries)")
        print("Total time: \(String(format: "%.2f", workloadTime))s")
        print("Overall QPS: \(String(format: "%.0f", Double(totalQueries) / workloadTime))")
        
        let finalStats = await index.statistics()
        print("\nIndex Statistics:")
        print("Total vectors: \(await index.count)")
        print("Memory usage: \(formatBytes(await index.memoryUsage))")
        print("Routing distribution:")
        print("  IVF queries: \(finalStats.routingStatistics.ivfQueries)")
        print("  Learned queries: \(finalStats.routingStatistics.learnedQueries)")
        print("  Hybrid queries: \(finalStats.routingStatistics.hybridQueries)")
    }
    
    // MARK: - Helper Functions
    
    static func generateMixedDataset(count: Int, dimensions: Int) -> [[Float]] {
        var vectors: [[Float]] = []
        
        // 1/3 clustered
        let clusterCount = 10
        let clusteredCount = count / 3
        for i in 0..<clusteredCount {
            let cluster = i % clusterCount
            let center = Float(cluster) / Float(clusterCount)
            let vector = (0..<dimensions).map { d in
                center + sin(Float(d) * 0.1) * 0.5 + Float.random(in: -0.1...0.1)
            }
            vectors.append(vector)
        }
        
        // 1/3 sequential
        let sequentialCount = count / 3
        for i in 0..<sequentialCount {
            let position = Float(i) / Float(sequentialCount)
            let vector = (0..<dimensions).map { d in
                Float(d) * position / Float(dimensions) + Float.random(in: -0.05...0.05)
            }
            vectors.append(vector)
        }
        
        // 1/3 random
        let randomCount = count - vectors.count
        for _ in 0..<randomCount {
            let vector = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            vectors.append(vector)
        }
        
        return vectors.shuffled()
    }
    
    static func generateScalableDataset(count: Int, dimensions: Int) -> [[Float]] {
        // Generate data with controllable properties for scalability testing
        (0..<count).map { i in
            let pattern = i % 5
            return (0..<dimensions).map { d in
                switch pattern {
                case 0: return Float(i % 100) / 100.0 // Cyclic
                case 1: return sin(Float(d + i) * 0.01) // Wave
                case 2: return Float(d) / Float(dimensions) // Linear
                case 3: return Float.random(in: 0...1) // Random
                default: return cos(Float(i) * 0.001) * sin(Float(d) * 0.1) // Complex
                }
            }
        }
    }
    
    static func generateProductEmbeddings(count: Int, dimensions: Int) -> [[Float]] {
        // Simulate product embeddings with semantic structure
        (0..<count).map { i in
            let category = i % 4
            let baseVector = (0..<dimensions).map { d in
                // Category-specific patterns
                switch category {
                case 0: return sin(Float(d) * 0.05) // Electronics
                case 1: return cos(Float(d) * 0.07) // Clothing
                case 2: return Float(d % 50) / 50.0 // Books
                default: return 1.0 - Float(d) / Float(dimensions) // Home
                }
            }
            
            // Add variation
            return baseVector.map { $0 * 0.8 + Float.random(in: -0.2...0.2) }
        }
    }
    
    static func generateQueries(from vectors: [[Float]], count: Int) -> [[Float]] {
        // Mix of exact and approximate queries
        (0..<count).map { i in
            if i < count / 2 {
                // Near-exact matches
                let base = vectors[i % vectors.count]
                return base.map { $0 + Float.random(in: -0.05...0.05) }
            } else {
                // Random queries
                return (0..<vectors[0].count).map { _ in Float.random(in: -1...1) }
            }
        }
    }
    
    static func benchmarkAllIndexTypes(
        vectors: [[Float]],
        queries: [[Float]],
        dimensions: Int
    ) async throws -> [String: PerformanceMetrics] {
        var results: [String: PerformanceMetrics] = [:]
        
        // IVF Index
        let ivfConfig = IVFConfiguration(
            dimensions: dimensions,
            numberOfCentroids: 256,
            numberOfProbes: 16
        )
        let ivfMetrics = try await benchmarkIndex(
            IVFIndex<SIMD128<Float>, TestMetadata>(configuration: ivfConfig),
            vectors: vectors,
            queries: queries,
            requiresTraining: true
        )
        results["IVF"] = ivfMetrics
        
        // Learned Index
        let learnedConfig = LearnedIndexConfiguration(
            dimensions: dimensions,
            modelArchitecture: .mlp(hiddenSizes: [256, 128]),
            bucketSize: 100
        )
        let learnedMetrics = try await benchmarkIndex(
            LearnedIndex<SIMD128<Float>, TestMetadata>(configuration: learnedConfig),
            vectors: vectors,
            queries: queries,
            requiresTraining: true,
            trainAfterInsert: true
        )
        results["Learned"] = learnedMetrics
        
        // Hybrid Index
        let hybridConfig = HybridIndexConfiguration(
            dimensions: dimensions,
            routingStrategy: .adaptive
        )
        let hybridMetrics = try await benchmarkIndex(
            HybridIndex<SIMD128<Float>, TestMetadata>(configuration: hybridConfig),
            vectors: vectors,
            queries: queries,
            requiresTraining: true
        )
        results["Hybrid"] = hybridMetrics
        
        return results
    }
    
    static func benchmarkIndex<Index: VectorIndex>(
        _ index: Index,
        vectors: [[Float]],
        queries: [[Float]],
        requiresTraining: Bool = false,
        trainAfterInsert: Bool = false
    ) async throws -> PerformanceMetrics where Index.Vector == SIMD128<Float>, Index.Metadata == TestMetadata {
        // Training phase
        if requiresTraining && !trainAfterInsert {
            if let trainableIndex = index as? any TrainableIndex {
                try await trainableIndex.train(on: Array(vectors.prefix(10_000)))
            }
        }
        
        // Insert phase
        let insertStart = DispatchTime.now()
        for (i, vector) in vectors.enumerated() {
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: SIMD128<Float>(vector),
                metadata: TestMetadata(label: "bench")
            )
            _ = try await index.insert(entry)
        }
        let insertTime = elapsedTime(since: insertStart)
        
        // Train after insert if needed
        if trainAfterInsert {
            if let trainableIndex = index as? any TrainableIndex {
                try await trainableIndex.train()
            }
        }
        
        // Search phase
        let searchStart = DispatchTime.now()
        var totalRecall: Float = 0
        
        for query in queries {
            let results = try await index.search(
                query: SIMD128<Float>(query),
                k: 10
            )
            // Simple recall estimation
            totalRecall += Float(results.count) / 10.0
        }
        
        let searchTime = elapsedTime(since: searchStart)
        let recall = totalRecall / Float(queries.count)
        
        return PerformanceMetrics(
            insertsPerSecond: Double(vectors.count) / insertTime,
            queriesPerSecond: Double(queries.count) / searchTime,
            memoryUsage: await index.memoryUsage,
            recall: recall
        )
    }
    
    static func elapsedTime(since start: DispatchTime) -> TimeInterval {
        let end = DispatchTime.now()
        return Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    }
    
    static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}

// MARK: - Supporting Types

struct PerformanceMetrics {
    let insertsPerSecond: Double
    let queriesPerSecond: Double
    let memoryUsage: Int
    let recall: Float
}

struct ProductMetadata: Codable, Sendable {
    let name: String
    let category: String
    let price: Float
    let popularity: Float
}

// Protocol for trainable indexes
protocol TrainableIndex {
    func train(on samples: [[Float]]) async throws
    func train() async throws
}

extension IVFIndex: TrainableIndex {
    func train() async throws {
        // IVF doesn't support training without samples
        throw VectorStoreError.operationNotSupported("IVF requires training samples")
    }
}

extension LearnedIndex: TrainableIndex {
    func train(on samples: [[Float]]) async throws {
        try await self.train()
    }
}

extension HybridIndex: TrainableIndex {}

// MARK: - SIMD Extensions

extension SIMD256<Float> {
    init(_ array: [Float]) {
        self.init()
        for i in 0..<Swift.min(256, array.count) {
            self[i] = array[i]
        }
    }
}

// String multiplication helper
extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}

// Placeholder error type
enum VectorStoreError: Error {
    case operationNotSupported(String)
}
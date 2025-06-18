// VectorStoreKit: Comprehensive Benchmarking Suite
//
// Benchmarks for IVF, Learned, and Hybrid indexes

import Foundation
import simd
import Dispatch

/// Comprehensive benchmarking suite for VectorStoreKit indexes
public class ComprehensiveBenchmarks {
    
    // MARK: - Benchmark Configuration
    
    public struct ComprehensiveBenchmarkConfig {
        public let dimensions: Int
        public let vectorCounts: [Int]
        public let queryCount: Int
        public let k: Int
        public let iterations: Int
        public let warmupIterations: Int
        
        public init(
            dimensions: Int = 128,
            vectorCounts: [Int] = [1_000, 10_000, 100_000],
            queryCount: Int = 1_000,
            k: Int = 10,
            iterations: Int = 5,
            warmupIterations: Int = 2
        ) {
            self.dimensions = dimensions
            self.vectorCounts = vectorCounts
            self.queryCount = queryCount
            self.k = k
            self.iterations = iterations
            self.warmupIterations = warmupIterations
        }
        
        public static let small = ComprehensiveBenchmarkConfig(
            dimensions: 64,
            vectorCounts: [100, 1_000, 10_000],
            queryCount: 100,
            iterations: 3
        )
        
        public static let standard = ComprehensiveBenchmarkConfig()
        
        public static let large = ComprehensiveBenchmarkConfig(
            dimensions: 256,
            vectorCounts: [10_000, 100_000, 1_000_000],
            queryCount: 10_000,
            iterations: 10
        )
    }
    
    // MARK: - Benchmark Results
    
    public struct BenchmarkRunResult {
        public let indexType: String
        public let vectorCount: Int
        public let insertTime: TimeInterval
        public let searchTime: TimeInterval
        public let memoryUsage: Int
        public let recall: Float
        public let precision: Float
        public let throughput: ThroughputMetrics
        
        public struct ThroughputMetrics {
            public let insertsPerSecond: Double
            public let queriesPerSecond: Double
        }
    }
    
    public struct ComparisonReport {
        public let results: [BenchmarkRunResult]
        public let bestInsertPerformance: String
        public let bestSearchPerformance: String
        public let bestMemoryEfficiency: String
        public let bestRecall: String
        public let summary: String
    }
    
    // MARK: - Initialization
    
    private let configuration: ComprehensiveBenchmarkConfig
    private let reporter: BenchmarkReporter
    private var groundTruth: [String: [String]] = [:] // Query ID -> Ground truth result IDs
    
    public init(
        configuration: ComprehensiveBenchmarkConfig = .standard,
        reporter: BenchmarkReporter = BenchmarkReporter()
    ) {
        self.configuration = configuration
        self.reporter = reporter
    }
    
    // MARK: - Main Benchmark Entry Point
    
    public func runAllBenchmarks() async throws -> ComparisonReport {
        reporter.startSession("VectorStoreKit Comprehensive Benchmarks")
        
        var allResults: [BenchmarkRunResult] = []
        
        for vectorCount in configuration.vectorCounts {
            reporter.startSection("Benchmarking with \(vectorCount) vectors")
            
            // Generate test data
            let (vectors, queries) = generateTestData(
                vectorCount: vectorCount,
                queryCount: configuration.queryCount,
                dimensions: configuration.dimensions
            )
            
            // Compute ground truth for recall calculation
            await computeGroundTruth(vectors: vectors, queries: queries)
            
            // Benchmark each index type
            let ivfResult = try await benchmarkIVFIndex(vectors: vectors, queries: queries)
            allResults.append(ivfResult)
            
            let learnedResult = try await benchmarkLearnedIndex(vectors: vectors, queries: queries)
            allResults.append(learnedResult)
            
            let hybridResult = try await benchmarkHybridIndex(vectors: vectors, queries: queries)
            allResults.append(hybridResult)
            
            reporter.endSection()
        }
        
        let report = generateComparisonReport(results: allResults)
        reporter.printReport(report)
        
        return report
    }
    
    // MARK: - IVF Index Benchmarks
    
    private func benchmarkIVFIndex(
        vectors: [[Float]],
        queries: [[Float]]
    ) async throws -> BenchmarkRunResult {
        reporter.startBenchmark("IVF Index")
        
        // Create index with different configurations
        let centroids = min(Int(sqrt(Double(vectors.count))), 1024)
        let config = IVFConfiguration(
            dimensions: configuration.dimensions,
            numberOfCentroids: centroids,
            numberOfProbes: max(1, centroids / 10)
        )
        
        let index = try await IVFIndex<SIMD128<Float>, TestMetadata>(configuration: config)
        
        // Training phase
        let trainStart = DispatchTime.now()
        let trainingSample = Array(vectors.shuffled().prefix(min(vectors.count, 100_000)))
        try await index.train(on: trainingSample)
        let trainTime = elapsedTime(since: trainStart)
        reporter.log("IVF training completed in \(trainTime)s")
        
        // Insert phase
        let (insertTime, insertThroughput) = try await measureInsertPerformance(
            index: index,
            vectors: vectors
        )
        
        // Search phase
        let (searchTime, searchThroughput, recall) = try await measureSearchPerformance(
            index: index,
            queries: queries
        )
        
        // Memory usage
        let memoryUsage = await index.memoryUsage
        
        reporter.endBenchmark()
        
        return BenchmarkRunResult(
            indexType: "IVF",
            vectorCount: vectors.count,
            insertTime: insertTime,
            searchTime: searchTime,
            memoryUsage: memoryUsage,
            recall: recall,
            precision: recall, // Approximate
            throughput: BenchmarkRunResult.ThroughputMetrics(
                insertsPerSecond: insertThroughput,
                queriesPerSecond: searchThroughput
            )
        )
    }
    
    // MARK: - Learned Index Benchmarks
    
    private func benchmarkLearnedIndex(
        vectors: [[Float]],
        queries: [[Float]]
    ) async throws -> BenchmarkRunResult {
        reporter.startBenchmark("Learned Index")
        
        // Create index with MLP architecture
        let config = LearnedIndexConfiguration(
            dimensions: configuration.dimensions,
            modelArchitecture: .mlp(hiddenSizes: [256, 128]),
            bucketSize: max(10, vectors.count / 1000),
            trainingConfig: .init(epochs: min(50, max(10, vectors.count / 1000)))
        )
        
        let index = try await LearnedIndex<SIMD128<Float>, TestMetadata>(configuration: config)
        
        // Insert phase (includes implicit training data collection)
        let (insertTime, insertThroughput) = try await measureInsertPerformance(
            index: index,
            vectors: vectors
        )
        
        // Training phase
        let trainStart = DispatchTime.now()
        try await index.train()
        let trainTime = elapsedTime(since: trainStart)
        reporter.log("Learned model training completed in \(trainTime)s")
        
        // Search phase
        let (searchTime, searchThroughput, recall) = try await measureSearchPerformance(
            index: index,
            queries: queries
        )
        
        // Memory usage
        let memoryUsage = await index.memoryUsage
        
        reporter.endBenchmark()
        
        return BenchmarkRunResult(
            indexType: "Learned",
            vectorCount: vectors.count,
            insertTime: insertTime + trainTime,
            searchTime: searchTime,
            memoryUsage: memoryUsage,
            recall: recall,
            precision: recall,
            throughput: BenchmarkRunResult.ThroughputMetrics(
                insertsPerSecond: insertThroughput,
                queriesPerSecond: searchThroughput
            )
        )
    }
    
    // MARK: - Hybrid Index Benchmarks
    
    private func benchmarkHybridIndex(
        vectors: [[Float]],
        queries: [[Float]]
    ) async throws -> BenchmarkRunResult {
        reporter.startBenchmark("Hybrid Index")
        
        // Create index with adaptive routing
        let config = HybridIndexConfiguration(
            dimensions: configuration.dimensions,
            routingStrategy: .adaptive,
            adaptiveThreshold: 0.7
        )
        
        let index = try await HybridIndex<SIMD128<Float>, TestMetadata>(configuration: config)
        
        // Training phase
        let trainStart = DispatchTime.now()
        let trainingSample = Array(vectors.shuffled().prefix(min(vectors.count, 100_000)))
        try await index.train(on: trainingSample)
        let trainTime = elapsedTime(since: trainStart)
        reporter.log("Hybrid training completed in \(trainTime)s")
        
        // Insert phase
        let (insertTime, insertThroughput) = try await measureInsertPerformance(
            index: index,
            vectors: vectors
        )
        
        // Search phase with adaptive learning
        let (searchTime, searchThroughput, recall) = try await measureAdaptiveSearchPerformance(
            index: index,
            queries: queries
        )
        
        // Memory usage
        let memoryUsage = await index.memoryUsage
        
        // Get routing statistics
        let stats = await index.statistics()
        reporter.log("Routing stats - IVF: \(stats.routingStatistics.ivfRatio), Learned: \(stats.routingStatistics.learnedRatio)")
        
        reporter.endBenchmark()
        
        return BenchmarkRunResult(
            indexType: "Hybrid",
            vectorCount: vectors.count,
            insertTime: insertTime,
            searchTime: searchTime,
            memoryUsage: memoryUsage,
            recall: recall,
            precision: recall,
            throughput: BenchmarkRunResult.ThroughputMetrics(
                insertsPerSecond: insertThroughput,
                queriesPerSecond: searchThroughput
            )
        )
    }
    
    // MARK: - Performance Measurement Utilities
    
    private func measureInsertPerformance<Vector: SIMD, Metadata: Codable & Sendable>(
        index: any VectorIndex,
        vectors: [[Float]]
    ) async throws -> (time: TimeInterval, throughput: Double) where Vector.Scalar: BinaryFloatingPoint {
        // Warmup
        for i in 0..<min(100, vectors.count) {
            let entry = VectorEntry(
                id: "warmup_\(i)",
                vector: SIMD128<Float>(vectors[i]),
                metadata: TestMetadata(label: "warmup")
            )
            _ = try await index.insert(entry)
        }
        
        // Clear warmup data if possible
        if vectors.count > 100 {
            for i in 0..<100 {
                _ = try await index.delete(id: "warmup_\(i)")
            }
        }
        
        // Actual measurement
        let start = DispatchTime.now()
        
        for (i, vector) in vectors.enumerated() {
            let entry = VectorEntry(
                id: "vec_\(i)",
                vector: SIMD128<Float>(vector),
                metadata: TestMetadata(label: "bench_\(i)")
            )
            _ = try await index.insert(entry)
            
            if i % 1000 == 0 {
                reporter.progress("Inserted \(i)/\(vectors.count) vectors")
            }
        }
        
        let elapsed = elapsedTime(since: start)
        let throughput = Double(vectors.count) / elapsed
        
        reporter.log("Insert performance: \(throughput) vectors/second")
        
        return (elapsed, throughput)
    }
    
    private func measureSearchPerformance<Vector: SIMD, Metadata: Codable & Sendable>(
        index: any VectorIndex,
        queries: [[Float]]
    ) async throws -> (time: TimeInterval, throughput: Double, recall: Float) where Vector.Scalar: BinaryFloatingPoint {
        var totalTime: TimeInterval = 0
        var totalRecall: Float = 0
        
        // Perform multiple iterations
        for iteration in 0..<configuration.iterations {
            var iterationRecall: Float = 0
            
            let start = DispatchTime.now()
            
            for (i, query) in queries.enumerated() {
                let results = try await index.search(
                    query: SIMD128<Float>(query),
                    k: configuration.k
                )
                
                // Calculate recall
                if let groundTruth = groundTruth["query_\(i)"] {
                    let resultIds = Set(results.map { $0.id })
                    let groundTruthIds = Set(groundTruth.prefix(configuration.k))
                    let intersection = resultIds.intersection(groundTruthIds).count
                    iterationRecall += Float(intersection) / Float(configuration.k)
                }
            }
            
            let elapsed = elapsedTime(since: start)
            totalTime += elapsed
            totalRecall += iterationRecall / Float(queries.count)
            
            if iteration >= configuration.warmupIterations {
                reporter.log("Iteration \(iteration): \(elapsed)s, recall: \(iterationRecall / Float(queries.count))")
            }
        }
        
        // Calculate averages (excluding warmup)
        let validIterations = configuration.iterations - configuration.warmupIterations
        let avgTime = totalTime / Double(validIterations)
        let avgRecall = totalRecall / Float(validIterations)
        let throughput = Double(queries.count) / avgTime
        
        reporter.log("Search performance: \(throughput) queries/second, recall: \(avgRecall)")
        
        return (avgTime, throughput, avgRecall)
    }
    
    private func measureAdaptiveSearchPerformance<Vector: SIMD, Metadata: Codable & Sendable>(
        index: HybridIndex<Vector, Metadata>,
        queries: [[Float]]
    ) async throws -> (time: TimeInterval, throughput: Double, recall: Float) where Vector.Scalar: BinaryFloatingPoint {
        // Initial search to trigger adaptation
        let adaptationQueries = queries.prefix(100)
        for query in adaptationQueries {
            _ = try await index.search(
                query: SIMD128<Float>(query),
                k: configuration.k
            )
        }
        
        // Measure adapted performance
        return try await measureSearchPerformance<Vector, Metadata>(index: index, queries: queries)
    }
    
    // MARK: - Test Data Generation
    
    private func generateTestData(
        vectorCount: Int,
        queryCount: Int,
        dimensions: Int
    ) -> (vectors: [[Float]], queries: [[Float]]) {
        reporter.log("Generating \(vectorCount) vectors and \(queryCount) queries of dimension \(dimensions)")
        
        // Generate vectors with mixed characteristics
        var vectors: [[Float]] = []
        
        // 1/3 clustered data
        let clusteredCount = vectorCount / 3
        let clusterCount = Int(sqrt(Double(clusteredCount)))
        for cluster in 0..<clusterCount {
            let center = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            let vectorsPerCluster = clusteredCount / clusterCount
            
            for _ in 0..<vectorsPerCluster {
                let vector = center.map { $0 + Float.random(in: -0.1...0.1) }
                vectors.append(vector)
            }
        }
        
        // 1/3 sequential/patterned data
        let sequentialCount = vectorCount / 3
        for i in 0..<sequentialCount {
            let position = Float(i) / Float(sequentialCount)
            let vector = (0..<dimensions).map { d in
                sin(Float(d) * position * .pi) + cos(Float(i % 10) * .pi / 5)
            }
            vectors.append(vector)
        }
        
        // 1/3 random data
        let randomCount = vectorCount - vectors.count
        for _ in 0..<randomCount {
            let vector = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            vectors.append(vector)
        }
        
        // Generate queries (mix of exact matches and nearby vectors)
        var queries: [[Float]] = []
        
        // 50% exact or near matches
        for i in 0..<queryCount/2 {
            let baseIdx = i % vectors.count
            let query = vectors[baseIdx].map { $0 + Float.random(in: -0.05...0.05) }
            queries.append(query)
        }
        
        // 50% random queries
        for _ in queryCount/2..<queryCount {
            let query = (0..<dimensions).map { _ in Float.random(in: -1...1) }
            queries.append(query)
        }
        
        return (vectors.shuffled(), queries.shuffled())
    }
    
    // MARK: - Ground Truth Computation
    
    private func computeGroundTruth(vectors: [[Float]], queries: [[Float]]) async {
        reporter.log("Computing ground truth for recall calculation...")
        
        await withTaskGroup(of: (String, [String]).self) { group in
            for (qIdx, query) in queries.enumerated() {
                group.addTask {
                    var distances: [(String, Float)] = []
                    
                    for (vIdx, vector) in vectors.enumerated() {
                        let distance = self.euclideanDistance(query, vector)
                        distances.append(("vec_\(vIdx)", distance))
                    }
                    
                    distances.sort { $0.1 < $1.1 }
                    let topK = distances.prefix(self.configuration.k).map { $0.0 }
                    
                    return ("query_\(qIdx)", topK)
                }
            }
            
            for await (queryId, results) in group {
                groundTruth[queryId] = results
            }
        }
    }
    
    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<min(a.count, b.count) {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    private func elapsedTime(since start: DispatchTime) -> TimeInterval {
        let end = DispatchTime.now()
        return Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    }
    
    // MARK: - Report Generation
    
    private func generateComparisonReport(results: [BenchmarkRunResult]) -> ComparisonReport {
        // Group by vector count
        var groupedResults: [Int: [BenchmarkRunResult]] = [:]
        for result in results {
            groupedResults[result.vectorCount, default: []].append(result)
        }
        
        // Find best performers
        let bestInsert = results.min { $0.insertTime < $1.insertTime }?.indexType ?? "Unknown"
        let bestSearch = results.max { $0.throughput.queriesPerSecond > $1.throughput.queriesPerSecond }?.indexType ?? "Unknown"
        let bestMemory = results.min { $0.memoryUsage < $1.memoryUsage }?.indexType ?? "Unknown"
        let bestRecall = results.max { $0.recall > $1.recall }?.indexType ?? "Unknown"
        
        // Generate summary
        var summary = "VectorStoreKit Benchmark Summary\n"
        summary += "================================\n\n"
        
        for (vectorCount, groupResults) in groupedResults.sorted(by: { $0.key < $1.key }) {
            summary += "Vector Count: \(vectorCount)\n"
            summary += "-----------------\n"
            
            for result in groupResults.sorted(by: { $0.indexType < $1.indexType }) {
                summary += "\n\(result.indexType) Index:\n"
                summary += "  Insert: \(String(format: "%.2f", result.insertTime))s (\(String(format: "%.0f", result.throughput.insertsPerSecond)) vectors/s)\n"
                summary += "  Search: \(String(format: "%.2f", result.searchTime))s (\(String(format: "%.0f", result.throughput.queriesPerSecond)) queries/s)\n"
                summary += "  Memory: \(formatBytes(result.memoryUsage))\n"
                summary += "  Recall: \(String(format: "%.2f%%", result.recall * 100))\n"
            }
            summary += "\n"
        }
        
        summary += "Best Performers:\n"
        summary += "  Fastest Insert: \(bestInsert)\n"
        summary += "  Fastest Search: \(bestSearch)\n"
        summary += "  Lowest Memory: \(bestMemory)\n"
        summary += "  Highest Recall: \(bestRecall)\n"
        
        return ComparisonReport(
            results: results,
            bestInsertPerformance: bestInsert,
            bestSearchPerformance: bestSearch,
            bestMemoryEfficiency: bestMemory,
            bestRecall: bestRecall,
            summary: summary
        )
    }
    
    private func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}

// MARK: - Benchmark Reporter

public class BenchmarkReporter {
    private var currentSection: String?
    private var currentBenchmark: String?
    private let startTime = DispatchTime.now()
    
    public init() {}
    
    public func startSession(_ name: String) {
        print("\n🚀 Starting Benchmark Session: \(name)")
        print("=" * 60)
    }
    
    public func startSection(_ name: String) {
        currentSection = name
        print("\n📊 \(name)")
        print("-" * 40)
    }
    
    public func endSection() {
        if let section = currentSection {
            print("✅ Completed: \(section)\n")
        }
        currentSection = nil
    }
    
    public func startBenchmark(_ name: String) {
        currentBenchmark = name
        print("\n⏱️  Benchmarking \(name)...")
    }
    
    public func endBenchmark() {
        if let benchmark = currentBenchmark {
            print("✓ \(benchmark) complete")
        }
        currentBenchmark = nil
    }
    
    public func log(_ message: String) {
        print("   • \(message)")
    }
    
    public func progress(_ message: String) {
        print("   ⟳ \(message)", terminator: "\r")
        fflush(stdout)
    }
    
    public func printReport(_ report: ComprehensiveBenchmarks.ComparisonReport) {
        print("\n" + "=" * 60)
        print(report.summary)
        print("=" * 60)
        
        let totalTime = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
        print("\n⏰ Total benchmark time: \(String(format: "%.2f", totalTime))s")
    }
}

// MARK: - String Extension

private extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}

// MARK: - Test Metadata

struct TestMetadata: Codable, Sendable {
    let label: String
    let timestamp: Date
    
    init(label: String, timestamp: Date = Date()) {
        self.label = label
        self.timestamp = timestamp
    }
}
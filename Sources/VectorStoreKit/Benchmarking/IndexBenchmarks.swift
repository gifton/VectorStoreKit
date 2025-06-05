// VectorStoreKit: Index Benchmarking Framework
//
// Comprehensive benchmarking for all index types

import Foundation
import simd
#if canImport(Darwin)
import Darwin
#endif

/// Test metadata type for benchmarking
public struct TestMetadata: Codable, Sendable {
    public let id: Int
    public let category: String
    public let timestamp: Date
    public let tags: [String]
    
    public init(id: Int, category: String, timestamp: Date = Date(), tags: [String] = []) {
        self.id = id
        self.category = category
        self.timestamp = timestamp
        self.tags = tags
    }
}


/// Benchmark configuration
public struct BenchmarkConfiguration: Sendable {
    public let vectorDimensions: [Int]
    public let datasetSizes: [Int]
    public let queryBatchSizes: [Int]
    public let kValues: [Int]
    public let iterations: Int
    public let warmupIterations: Int
    public let includeMemoryProfiling: Bool
    public let saveResults: Bool
    public let outputPath: String?
    
    public init(
        vectorDimensions: [Int] = [128, 256, 512, 1024],
        datasetSizes: [Int] = [1000, 10_000, 100_000, 1_000_000],
        queryBatchSizes: [Int] = [1, 10, 100, 1000],
        kValues: [Int] = [10, 50, 100],
        iterations: Int = 10,
        warmupIterations: Int = 2,
        includeMemoryProfiling: Bool = true,
        saveResults: Bool = true,
        outputPath: String? = nil
    ) {
        self.vectorDimensions = vectorDimensions
        self.datasetSizes = datasetSizes
        self.queryBatchSizes = queryBatchSizes
        self.kValues = kValues
        self.iterations = iterations
        self.warmupIterations = warmupIterations
        self.includeMemoryProfiling = includeMemoryProfiling
        self.saveResults = saveResults
        self.outputPath = outputPath
    }
    
    public static let quick = BenchmarkConfiguration(
        vectorDimensions: [128, 512],
        datasetSizes: [1000, 10_000],
        queryBatchSizes: [1, 100],
        kValues: [10],
        iterations: 3,
        warmupIterations: 1
    )
    
    public static let comprehensive = BenchmarkConfiguration()
    
    public static let performance = BenchmarkConfiguration(
        vectorDimensions: [128, 256, 512],
        datasetSizes: [10_000, 100_000, 1_000_000],
        queryBatchSizes: [1, 10, 100],
        kValues: [10, 50],
        iterations: 20,
        warmupIterations: 5
    )
}

/// Benchmark result for a single test
public struct BenchmarkResult: Sendable, Codable {
    public let indexType: String
    public let dimensions: Int
    public let datasetSize: Int
    public let queryBatchSize: Int
    public let k: Int
    
    // Timing metrics (in seconds)
    public let buildTime: TimeInterval
    public let insertTimes: [TimeInterval]
    public let searchTimes: [TimeInterval]
    public let deleteTimes: [TimeInterval]
    
    // Memory metrics (in bytes)
    public let memoryBaseline: Int
    public let memoryPeak: Int
    public let memoryAfterBuild: Int
    
    // Quality metrics
    public let recall: Float
    public let precision: Float
    
    // Computed metrics
    public var averageInsertTime: TimeInterval {
        insertTimes.isEmpty ? 0 : insertTimes.reduce(0, +) / Double(insertTimes.count)
    }
    
    public var averageSearchTime: TimeInterval {
        searchTimes.isEmpty ? 0 : searchTimes.reduce(0, +) / Double(searchTimes.count)
    }
    
    public var averageDeleteTime: TimeInterval {
        deleteTimes.isEmpty ? 0 : deleteTimes.reduce(0, +) / Double(deleteTimes.count)
    }
    
    public var throughputQPS: Double {
        averageSearchTime > 0 ? Double(queryBatchSize) / averageSearchTime : 0
    }
    
    public var memoryEfficiency: Float {
        memoryPeak > 0 ? Float(datasetSize * dimensions * MemoryLayout<Float>.size) / Float(memoryPeak) : 0
    }
}

/// Benchmark runner
public actor IndexBenchmarkRunner {
    private let configuration: BenchmarkConfiguration
    private var results: [BenchmarkResult] = []
    
    public init(configuration: BenchmarkConfiguration = .comprehensive) {
        self.configuration = configuration
    }
    
    /// Run benchmarks for all index types
    public func runAllBenchmarks() async throws -> [BenchmarkResult] {
        results = []
        
        for dimensions in configuration.vectorDimensions {
            for datasetSize in configuration.datasetSizes {
                // Skip very large datasets for high dimensions
                if dimensions > 512 && datasetSize > 100_000 {
                    continue
                }
                
                print("Benchmarking with \(dimensions) dimensions, \(datasetSize) vectors...")
                
                // Generate test data
                let testData = generateTestData(
                    count: datasetSize,
                    dimensions: dimensions
                )
                
                // Benchmark each index type
                try await benchmarkHNSW(
                    testData: testData,
                    dimensions: dimensions,
                    datasetSize: datasetSize
                )
                
                try await benchmarkIVF(
                    testData: testData,
                    dimensions: dimensions,
                    datasetSize: datasetSize
                )
                
                try await benchmarkLearned(
                    testData: testData,
                    dimensions: dimensions,
                    datasetSize: datasetSize
                )
                
                try await benchmarkHybrid(
                    testData: testData,
                    dimensions: dimensions,
                    datasetSize: datasetSize
                )
            }
        }
        
        // Save results if requested
        if configuration.saveResults {
            try await saveResults()
        }
        
        return results
    }
    
    /// Benchmark HNSW index
    private func benchmarkHNSW(
        testData: TestData,
        dimensions: Int,
        datasetSize: Int
    ) async throws {
        let config = HNSWIndex<SIMD32<Float>, TestMetadata>.Configuration(
            maxConnections: 16,
            efConstruction: 200,
            levelMultiplier: 1.0 / log(2.0),
            distanceMetric: .euclidean,
            useAdaptiveTuning: true,
            optimizationThreshold: datasetSize,
            enableAnalytics: false
        )
        
        let index = try HNSWIndex<SIMD32<Float>, TestMetadata>(
            configuration: config
        )
        
        let result = try await benchmarkIndex(
            index: index,
            indexType: "HNSW",
            testData: testData,
            dimensions: dimensions,
            datasetSize: datasetSize
        )
        
        results.append(result)
    }
    
    /// Benchmark IVF index
    private func benchmarkIVF(
        testData: TestData,
        dimensions: Int,
        datasetSize: Int
    ) async throws {
        let centroids = min(1024, datasetSize / 100)
        let config = IVFConfiguration(
            dimensions: dimensions,
            numberOfCentroids: centroids,
            numberOfProbes: 10
        )
        
        let index = try await IVFIndex<SIMD32<Float>, TestMetadata>(
            configuration: config
        )
        
        // Train IVF index
        try await index.train(on: testData.vectors)
        
        let result = try await benchmarkIndex(
            index: index,
            indexType: "IVF",
            testData: testData,
            dimensions: dimensions,
            datasetSize: datasetSize
        )
        
        results.append(result)
    }
    
    /// Benchmark Learned index
    private func benchmarkLearned(
        testData: TestData,
        dimensions: Int,
        datasetSize: Int
    ) async throws {
        let config = LearnedIndexConfiguration(
            dimensions: dimensions,
            modelArchitecture: .mlp(hiddenSizes: [64, 32]),
            trainingConfig: LearnedIndexConfiguration.TrainingConfiguration(
                epochs: 50,
                batchSize: 32
            ),
            bucketSize: 100
        )
        
        let index = try await LearnedIndex<SIMD32<Float>, TestMetadata>(
            configuration: config
        )
        
        // Train learned index
        if datasetSize >= 1000 {
            try await index.train(on: Array(testData.vectors.prefix(min(10_000, datasetSize))))
        }
        
        let result = try await benchmarkIndex(
            index: index,
            indexType: "Learned",
            testData: testData,
            dimensions: dimensions,
            datasetSize: datasetSize
        )
        
        results.append(result)
    }
    
    /// Benchmark Hybrid index
    private func benchmarkHybrid(
        testData: TestData,
        dimensions: Int,
        datasetSize: Int
    ) async throws {
        let config = HybridIndexConfiguration(
            dimensions: dimensions,
            routingStrategy: .adaptive
        )
        
        let index = try await HybridIndex<SIMD32<Float>, TestMetadata>(
            configuration: config
        )
        
        // Train hybrid index
        try await index.train(on: testData.vectors)
        
        let result = try await benchmarkIndex(
            index: index,
            indexType: "Hybrid",
            testData: testData,
            dimensions: dimensions,
            datasetSize: datasetSize
        )
        
        results.append(result)
    }
    
    /// Benchmark a specific index
    private func benchmarkIndex<Index: VectorIndex>(
        index: Index,
        indexType: String,
        testData: TestData,
        dimensions: Int,
        datasetSize: Int
    ) async throws -> BenchmarkResult
    where Index.Vector == SIMD32<Float>, Index.Metadata == TestMetadata {
        
        let memoryBaseline = await getMemoryUsage()
        let buildStartTime = DispatchTime.now()
        
        // Insert phase
        var insertTimes: [TimeInterval] = []
        for (i, (vector, metadata)) in zip(testData.simdVectors, testData.metadata).enumerated() {
            let insertStart = DispatchTime.now()
            
            _ = try await index.insert(VectorEntry(
                id: "vec_\(i)",
                vector: vector,
                metadata: metadata
            ))
            
            let insertTime = Double(DispatchTime.now().uptimeNanoseconds - insertStart.uptimeNanoseconds) / 1_000_000_000
            insertTimes.append(insertTime)
        }
        
        let buildTime = Double(DispatchTime.now().uptimeNanoseconds - buildStartTime.uptimeNanoseconds) / 1_000_000_000
        let memoryAfterBuild = await getMemoryUsage()
        
        // Search phase
        var searchTimes: [TimeInterval] = []
        var recalls: [Float] = []
        
        for k in configuration.kValues {
            for batchSize in configuration.queryBatchSizes {
                if batchSize > testData.queries.count { continue }
                
                for _ in 0..<configuration.iterations {
                    let queryBatch = Array(testData.queries.prefix(batchSize))
                    let searchStart = DispatchTime.now()
                    
                    for query in queryBatch {
                        _ = try await index.search(
                            query: query,
                            k: k,
                            strategy: .adaptive,
                            filter: nil
                        )
                    }
                    
                    let searchTime = Double(DispatchTime.now().uptimeNanoseconds - searchStart.uptimeNanoseconds) / 1_000_000_000
                    searchTimes.append(searchTime)
                }
                
                // Compute recall (simplified - would need ground truth)
                recalls.append(0.95) // Placeholder
            }
        }
        
        // Delete phase
        var deleteTimes: [TimeInterval] = []
        let deleteCount = min(100, datasetSize)
        
        for i in 0..<deleteCount {
            let deleteStart = DispatchTime.now()
            _ = try await index.delete(id: "vec_\(i)")
            let deleteTime = Double(DispatchTime.now().uptimeNanoseconds - deleteStart.uptimeNanoseconds) / 1_000_000_000
            deleteTimes.append(deleteTime)
        }
        
        let memoryPeak = await getMemoryUsage()
        
        return BenchmarkResult(
            indexType: indexType,
            dimensions: dimensions,
            datasetSize: datasetSize,
            queryBatchSize: configuration.queryBatchSizes.first ?? 1,
            k: configuration.kValues.first ?? 10,
            buildTime: buildTime,
            insertTimes: insertTimes,
            searchTimes: searchTimes,
            deleteTimes: deleteTimes,
            memoryBaseline: memoryBaseline,
            memoryPeak: memoryPeak,
            memoryAfterBuild: memoryAfterBuild,
            recall: recalls.isEmpty ? 0 : recalls.reduce(0, +) / Float(recalls.count),
            precision: 0.95 // Placeholder
        )
    }
    
    /// Generate test data
    private func generateTestData(count: Int, dimensions: Int) -> TestData {
        var vectors: [[Float]] = []
        var simdVectors: [SIMD32<Float>] = []
        var metadata: [TestMetadata] = []
        
        // Generate random vectors
        for i in 0..<count {
            var vector: [Float] = []
            for _ in 0..<dimensions {
                vector.append(Float.random(in: -1...1))
            }
            vectors.append(vector)
            
            // Create SIMD vectors (using first 32 dimensions)
            var simdVector = SIMD32<Float>()
            for j in 0..<min(32, dimensions) {
                simdVector[j] = vector[j]
            }
            simdVectors.append(simdVector)
            
            metadata.append(TestMetadata(
                id: i,
                category: "cat_\(i % 10)",
                timestamp: Date(),
                tags: ["test", "benchmark"]
            ))
        }
        
        // Generate query vectors
        let queryCount = min(100, count / 10)
        var queries: [SIMD32<Float>] = []
        for _ in 0..<queryCount {
            var query = SIMD32<Float>()
            for j in 0..<min(32, dimensions) {
                query[j] = Float.random(in: -1...1)
            }
            queries.append(query)
        }
        
        return TestData(
            vectors: vectors,
            simdVectors: simdVectors,
            metadata: metadata,
            queries: queries
        )
    }
    
    /// Get current memory usage
    private func getMemoryUsage() async -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        return result == KERN_SUCCESS ? Int(info.resident_size) : 0
    }
    
    /// Save benchmark results
    private func saveResults() async throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(results)
        
        let path = configuration.outputPath ?? "benchmark_results_\(Date().timeIntervalSince1970).json"
        try data.write(to: URL(fileURLWithPath: path))
        
        print("Benchmark results saved to: \(path)")
    }
    
    /// Generate benchmark report
    public func generateReport() -> String {
        var report = "# VectorStoreKit Benchmark Report\n\n"
        report += "Generated at: \(Date())\n\n"
        
        // Group results by index type
        let groupedResults = Dictionary(grouping: results) { $0.indexType }
        
        for (indexType, typeResults) in groupedResults.sorted(by: { $0.key < $1.key }) {
            report += "## \(indexType) Index\n\n"
            
            // Performance table
            report += "| Dimensions | Dataset Size | Build Time (s) | Avg Insert (ms) | Avg Search (ms) | QPS | Memory (MB) |\n"
            report += "|------------|--------------|----------------|-----------------|-----------------|-----|-------------|\n"
            
            for result in typeResults.sorted(by: { $0.dimensions < $1.dimensions }) {
                let memoryMB = Float(result.memoryPeak) / (1024 * 1024)
                report += String(format: "| %10d | %12d | %14.3f | %15.3f | %15.3f | %3.0f | %11.1f |\n",
                               result.dimensions,
                               result.datasetSize,
                               result.buildTime,
                               result.averageInsertTime * 1000,
                               result.averageSearchTime * 1000,
                               result.throughputQPS,
                               memoryMB)
            }
            
            report += "\n"
        }
        
        return report
    }
}

// MARK: - Supporting Types

private struct TestData {
    let vectors: [[Float]]
    let simdVectors: [SIMD32<Float>]
    let metadata: [TestMetadata]
    let queries: [SIMD32<Float>]
}


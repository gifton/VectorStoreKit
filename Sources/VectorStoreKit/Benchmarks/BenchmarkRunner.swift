// VectorStoreKit: Benchmark Runner
//
// Command-line interface for running comprehensive benchmarks

import Foundation
import ArgumentParser

@main
struct VectorStoreKitBenchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vsk-bench",
        abstract: "Run comprehensive benchmarks for VectorStoreKit indexes",
        discussion: """
        This tool benchmarks the performance of different vector index types:
        - IVF (Inverted File) Index
        - Learned Index with neural models
        - Hybrid Index combining both approaches
        
        The benchmarks measure:
        - Insert performance and throughput
        - Search latency and throughput
        - Memory usage
        - Recall accuracy
        - Adaptive behavior (for hybrid indexes)
        """,
        version: "1.0.0"
    )
    
    @Option(name: .shortAndLong, help: "Benchmark size: small, standard, or large")
    var size: BenchmarkSize = .standard
    
    @Option(name: .shortAndLong, help: "Vector dimensions")
    var dimensions: Int?
    
    @Option(name: .shortAndLong, help: "Number of vectors to test (comma-separated)")
    var vectorCounts: String?
    
    @Option(name: .shortAndLong, help: "Number of queries to run")
    var queries: Int?
    
    @Option(name: .shortAndLong, help: "Number of nearest neighbors to retrieve")
    var k: Int?
    
    @Option(name: .shortAndLong, help: "Output format: text, json, or csv")
    var output: OutputFormat = .text
    
    @Option(name: .shortAndLong, help: "Save results to file")
    var file: String?
    
    @Flag(name: .shortAndLong, help: "Run quick benchmarks (fewer iterations)")
    var quick = false
    
    @Flag(name: .shortAndLong, help: "Show detailed progress")
    var verbose = false
    
    enum BenchmarkSize: String, ExpressibleByArgument {
        case small, standard, large
    }
    
    enum OutputFormat: String, ExpressibleByArgument {
        case text, json, csv
    }
    
    func run() async throws {
        // Build configuration
        let config = buildConfiguration()
        
        // Create reporter based on output format
        let reporter = createReporter()
        
        // Run benchmarks
        let benchmarks = ComprehensiveBenchmarks(
            configuration: config,
            reporter: reporter
        )
        
        print("🚀 Starting VectorStoreKit benchmarks...")
        print("Configuration:")
        print("  • Dimensions: \(config.dimensions)")
        print("  • Vector counts: \(config.vectorCounts)")
        print("  • Queries: \(config.queryCount)")
        print("  • k: \(config.k)")
        print("  • Iterations: \(config.iterations)")
        print()
        
        let startTime = Date()
        let report = try await benchmarks.runAllBenchmarks()
        let duration = Date().timeIntervalSince(startTime)
        
        // Save results if requested
        if let filename = file {
            try saveResults(report, to: filename)
            print("\n💾 Results saved to: \(filename)")
        }
        
        print("\n✅ Benchmarks completed in \(String(format: "%.2f", duration)) seconds")
    }
    
    private func buildConfiguration() -> ComprehensiveBenchmarks.BenchmarkConfiguration {
        var baseConfig: ComprehensiveBenchmarks.BenchmarkConfiguration
        
        switch size {
        case .small:
            baseConfig = .small
        case .standard:
            baseConfig = .standard
        case .large:
            baseConfig = .large
        }
        
        // Override with custom values if provided
        let vectorCounts = parseVectorCounts() ?? baseConfig.vectorCounts
        let iterations = quick ? 2 : baseConfig.iterations
        let warmup = quick ? 1 : baseConfig.warmupIterations
        
        return ComprehensiveBenchmarks.BenchmarkConfiguration(
            dimensions: dimensions ?? baseConfig.dimensions,
            vectorCounts: vectorCounts,
            queryCount: queries ?? baseConfig.queryCount,
            k: k ?? baseConfig.k,
            iterations: iterations,
            warmupIterations: warmup
        )
    }
    
    private func parseVectorCounts() -> [Int]? {
        guard let countsString = vectorCounts else { return nil }
        
        let counts = countsString
            .split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        
        return counts.isEmpty ? nil : counts
    }
    
    private func createReporter() -> BenchmarkReporter {
        switch output {
        case .text:
            return BenchmarkReporter()
        case .json:
            return JSONBenchmarkReporter()
        case .csv:
            return CSVBenchmarkReporter()
        }
    }
    
    private func saveResults(_ report: ComprehensiveBenchmarks.ComparisonReport, to filename: String) throws {
        let data: Data
        
        switch output {
        case .text:
            data = report.summary.data(using: .utf8)!
        case .json:
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            data = try encoder.encode(report.results)
        case .csv:
            data = generateCSV(from: report).data(using: .utf8)!
        }
        
        try data.write(to: URL(fileURLWithPath: filename))
    }
    
    private func generateCSV(from report: ComprehensiveBenchmarks.ComparisonReport) -> String {
        var csv = "Index Type,Vector Count,Insert Time (s),Search Time (s),Memory (bytes),Recall,Inserts/sec,Queries/sec\n"
        
        for result in report.results {
            csv += "\(result.indexType),"
            csv += "\(result.vectorCount),"
            csv += "\(result.insertTime),"
            csv += "\(result.searchTime),"
            csv += "\(result.memoryUsage),"
            csv += "\(result.recall),"
            csv += "\(result.throughput.insertsPerSecond),"
            csv += "\(result.throughput.queriesPerSecond)\n"
        }
        
        return csv
    }
}

// MARK: - JSON Reporter

class JSONBenchmarkReporter: BenchmarkReporter {
    private var events: [BenchmarkEvent] = []
    
    struct BenchmarkEvent: Codable {
        let timestamp: Date
        let type: EventType
        let message: String
        
        enum EventType: String, Codable {
            case session, section, benchmark, log, progress
        }
    }
    
    override func startSession(_ name: String) {
        events.append(BenchmarkEvent(
            timestamp: Date(),
            type: .session,
            message: name
        ))
    }
    
    override func log(_ message: String) {
        events.append(BenchmarkEvent(
            timestamp: Date(),
            type: .log,
            message: message
        ))
    }
    
    override func printReport(_ report: ComprehensiveBenchmarks.ComparisonReport) {
        // JSON output is handled by the main command
    }
}

// MARK: - CSV Reporter

class CSVBenchmarkReporter: BenchmarkReporter {
    override func printReport(_ report: ComprehensiveBenchmarks.ComparisonReport) {
        // CSV output is handled by the main command
    }
}

// MARK: - Additional Benchmark Scenarios

extension ComprehensiveBenchmarks {
    
    /// Run specialized benchmark scenarios
    public func runSpecializedBenchmarks() async throws {
        // Streaming benchmark
        try await benchmarkStreamingPerformance()
        
        // Concurrent operations benchmark
        try await benchmarkConcurrentOperations()
        
        // Memory pressure benchmark
        try await benchmarkMemoryPressure()
        
        // Adaptive learning benchmark
        try await benchmarkAdaptiveLearning()
    }
    
    private func benchmarkStreamingPerformance() async throws {
        reporter.startSection("Streaming Performance")
        
        // Test incremental inserts and searches
        let batchSize = 1000
        let totalBatches = 10
        
        // Create indexes
        let indexes: [(String, any VectorIndex)] = [
            ("IVF", try await IVFIndex<SIMD128<Float>, TestMetadata>(
                configuration: IVFConfiguration(dimensions: configuration.dimensions)
            )),
            ("Learned", try await LearnedIndex<SIMD128<Float>, TestMetadata>(
                configuration: LearnedIndexConfiguration(dimensions: configuration.dimensions)
            )),
            ("Hybrid", try await HybridIndex<SIMD128<Float>, TestMetadata>(
                configuration: HybridIndexConfiguration(dimensions: configuration.dimensions)
            ))
        ]
        
        for (name, index) in indexes {
            reporter.log("Testing \(name) streaming performance...")
            
            var totalInsertTime: TimeInterval = 0
            var totalSearchTime: TimeInterval = 0
            
            for batch in 0..<totalBatches {
                // Generate batch data
                let vectors = (0..<batchSize).map { _ in
                    (0..<configuration.dimensions).map { _ in Float.random(in: -1...1) }
                }
                
                // Insert batch
                let insertStart = DispatchTime.now()
                for (i, vector) in vectors.enumerated() {
                    let entry = VectorEntry(
                        id: "stream_\(batch)_\(i)",
                        vector: SIMD128<Float>(vector),
                        metadata: TestMetadata(label: "stream")
                    )
                    _ = try await index.insert(entry)
                }
                totalInsertTime += elapsedTime(since: insertStart)
                
                // Search on growing index
                let searchStart = DispatchTime.now()
                for _ in 0..<100 {
                    let query = (0..<configuration.dimensions).map { _ in Float.random(in: -1...1) }
                    _ = try await index.search(
                        query: SIMD128<Float>(query),
                        k: configuration.k
                    )
                }
                totalSearchTime += elapsedTime(since: searchStart)
                
                reporter.progress("Processed batch \(batch + 1)/\(totalBatches)")
            }
            
            reporter.log("\(name) streaming results:")
            reporter.log("  Total insert time: \(totalInsertTime)s")
            reporter.log("  Total search time: \(totalSearchTime)s")
        }
        
        reporter.endSection()
    }
    
    private func benchmarkConcurrentOperations() async throws {
        reporter.startSection("Concurrent Operations")
        
        // Test concurrent reads and writes
        let concurrencyLevels = [1, 2, 4, 8, 16]
        
        for level in concurrencyLevels {
            reporter.log("Testing with \(level) concurrent operations...")
            
            let index = try await HybridIndex<SIMD128<Float>, TestMetadata>(
                configuration: HybridIndexConfiguration(dimensions: configuration.dimensions)
            )
            
            // Prepare index
            let vectors = (0..<1000).map { _ in
                (0..<configuration.dimensions).map { _ in Float.random(in: -1...1) }
            }
            try await index.train(on: vectors)
            
            // Concurrent operations
            let start = DispatchTime.now()
            
            await withTaskGroup(of: Void.self) { group in
                // Writers
                for i in 0..<level/2 {
                    group.addTask {
                        for j in 0..<100 {
                            let vector = (0..<self.configuration.dimensions).map { _ in Float.random(in: -1...1) }
                            let entry = VectorEntry(
                                id: "concurrent_\(i)_\(j)",
                                vector: SIMD128<Float>(vector),
                                metadata: TestMetadata(label: "concurrent")
                            )
                            _ = try? await index.insert(entry)
                        }
                    }
                }
                
                // Readers
                for i in 0..<level/2 {
                    group.addTask {
                        for _ in 0..<200 {
                            let query = (0..<self.configuration.dimensions).map { _ in Float.random(in: -1...1) }
                            _ = try? await index.search(
                                query: SIMD128<Float>(query),
                                k: self.configuration.k
                            )
                        }
                    }
                }
            }
            
            let elapsed = elapsedTime(since: start)
            reporter.log("  Concurrency level \(level): \(elapsed)s")
        }
        
        reporter.endSection()
    }
    
    private func benchmarkMemoryPressure() async throws {
        reporter.startSection("Memory Pressure Test")
        
        // Test behavior under memory constraints
        let index = try await LearnedIndex<SIMD128<Float>, TestMetadata>(
            configuration: LearnedIndexConfiguration(
                dimensions: configuration.dimensions,
                bucketSize: 1000
            )
        )
        
        var peakMemory = 0
        let vectors = (0..<10000).map { _ in
            (0..<configuration.dimensions).map { _ in Float.random(in: -1...1) }
        }
        
        // Insert with memory monitoring
        for (i, vector) in vectors.enumerated() {
            let entry = VectorEntry(
                id: "mem_\(i)",
                vector: SIMD128<Float>(vector),
                metadata: TestMetadata(label: "memory")
            )
            _ = try await index.insert(entry)
            
            if i % 1000 == 0 {
                let currentMemory = await index.memoryUsage
                peakMemory = max(peakMemory, currentMemory)
                reporter.log("Vectors: \(i), Memory: \(formatBytes(currentMemory))")
                
                // Trigger optimization to test memory cleanup
                if i % 5000 == 0 {
                    try await index.optimize(strategy: .light)
                }
            }
        }
        
        reporter.log("Peak memory usage: \(formatBytes(peakMemory))")
        
        reporter.endSection()
    }
    
    private func benchmarkAdaptiveLearning() async throws {
        reporter.startSection("Adaptive Learning")
        
        let index = try await HybridIndex<SIMD128<Float>, TestMetadata>(
            configuration: HybridIndexConfiguration(
                dimensions: configuration.dimensions,
                routingStrategy: .adaptive,
                adaptiveThreshold: 0.7
            )
        )
        
        // Generate training data with distinct patterns
        let clusteredVectors = (0..<1000).map { i in
            let cluster = i / 100
            let center = Float(cluster) / 10
            return (0..<configuration.dimensions).map { d in
                center + sin(Float(d) * 0.1) + Float.random(in: -0.1...0.1)
            }
        }
        
        let sequentialVectors = (0..<1000).map { i in
            let position = Float(i) / 1000
            return (0..<configuration.dimensions).map { d in
                Float(d) * position / Float(configuration.dimensions)
            }
        }
        
        // Train and insert
        try await index.train(on: clusteredVectors + sequentialVectors)
        
        for (i, vector) in (clusteredVectors + sequentialVectors).enumerated() {
            let entry = VectorEntry(
                id: "adaptive_\(i)",
                vector: SIMD128<Float>(vector),
                metadata: TestMetadata(label: i < 1000 ? "clustered" : "sequential")
            )
            _ = try await index.insert(entry)
        }
        
        // Test adaptive routing
        reporter.log("Testing adaptive routing behavior...")
        
        // Queries that should prefer IVF
        let clusteredQueries = clusteredVectors.prefix(100).map { vector in
            vector.map { $0 + Float.random(in: -0.05...0.05) }
        }
        
        // Queries that should prefer Learned
        let sequentialQueries = sequentialVectors.prefix(100).map { vector in
            vector.map { $0 + Float.random(in: -0.01...0.01) }
        }
        
        // Run queries multiple times to observe adaptation
        for iteration in 0..<5 {
            reporter.log("Adaptation iteration \(iteration + 1)...")
            
            // Mixed queries
            let queries = (clusteredQueries + sequentialQueries).shuffled()
            
            for query in queries {
                _ = try await index.search(
                    query: SIMD128<Float>(query),
                    k: configuration.k
                )
            }
            
            // Check routing statistics
            let stats = await index.statistics()
            reporter.log("  IVF usage: \(stats.routingStatistics.ivfRatio)")
            reporter.log("  Learned usage: \(stats.routingStatistics.learnedRatio)")
            reporter.log("  Hybrid usage: \(stats.routingStatistics.hybridRatio)")
        }
        
        reporter.endSection()
    }
}
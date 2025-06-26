// SIMDOptimizationBenchmarks.swift
// VectorStoreKit
//
// Comprehensive benchmarks for SIMD-optimized distance computations

import Foundation
import simd
import Accelerate
import os.log

/// Comprehensive SIMD optimization benchmarks
public struct SIMDOptimizationBenchmarks {
    
    private let logger = os.Logger(subsystem: "VectorStoreKit", category: "SIMDBenchmarks")
    
    /// Benchmark configuration
    public struct BenchmarkConfig {
        let vectorCount: Int
        let warmupIterations: Int
        let benchmarkIterations: Int
        let reportProgress: Bool
        
        public init(
            vectorCount: Int = 10000,
            warmupIterations: Int = 100,
            benchmarkIterations: Int = 1000,
            reportProgress: Bool = true
        ) {
            self.vectorCount = vectorCount
            self.warmupIterations = warmupIterations
            self.benchmarkIterations = benchmarkIterations
            self.reportProgress = reportProgress
        }
    }
    
    /// Benchmark results for a single metric
    public struct BenchmarkResult {
        let metricName: String
        let originalTime: TimeInterval
        let optimizedTime: TimeInterval
        let speedup: Double
        let throughput: Double  // operations per second
        let accuracy: Float     // accuracy compared to reference
        
        public var description: String {
            """
            \(metricName):
              Original: \(String(format: "%.3f", originalTime * 1000))ms
              Optimized: \(String(format: "%.3f", optimizedTime * 1000))ms
              Speedup: \(String(format: "%.2f", speedup))x
              Throughput: \(String(format: "%.0f", throughput)) ops/sec
              Accuracy: \(String(format: "%.6f", accuracy))
            """
        }
    }
    
    /// Complete benchmark suite results
    public struct SuiteBenchmarkResults {
        let results: [BenchmarkResult]
        let totalTime: TimeInterval
        let avgSpeedup: Double
        let config: BenchmarkConfig
        
        public var summary: String {
            let avgSpeedup = results.map(\.speedup).reduce(0, +) / Double(results.count)
            let minAccuracy = results.map(\.accuracy).min() ?? 0.0
            
            return """
            === SIMD Optimization Benchmark Summary ===
            Vectors tested: \(config.vectorCount)
            Iterations: \(config.benchmarkIterations)
            Total time: \(String(format: "%.2f", totalTime))s
            Average speedup: \(String(format: "%.2f", avgSpeedup))x
            Minimum accuracy: \(String(format: "%.6f", minAccuracy))
            
            Individual Results:
            \(results.map(\.description).joined(separator: "\n\n"))
            """
        }
    }
    
    public init() {}
    
    /// Run comprehensive SIMD benchmarks
    public func runComprehensiveBenchmarks(config: BenchmarkConfig = BenchmarkConfig()) async throws -> SuiteBenchmarkResults {
        logger.info("Starting comprehensive SIMD optimization benchmarks")
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Generate test data
        let testVectors = generateTestVectors(count: config.vectorCount)
        let query = testVectors[0]
        let candidates = Array(testVectors[1...])
        
        var results: [BenchmarkResult] = []
        
        // Benchmark Euclidean Distance
        if config.reportProgress {
            print("Benchmarking Euclidean Distance...")
        }
        let euclideanResult = try await benchmarkEuclideanDistance(
            query: query,
            candidates: candidates,
            config: config
        )
        results.append(euclideanResult)
        
        // Benchmark Manhattan Distance
        if config.reportProgress {
            print("Benchmarking Manhattan Distance...")
        }
        let manhattanResult = try await benchmarkManhattanDistance(
            query: query,
            candidates: candidates,
            config: config
        )
        results.append(manhattanResult)
        
        // Benchmark Cosine Distance
        if config.reportProgress {
            print("Benchmarking Cosine Distance...")
        }
        let cosineResult = try await benchmarkCosineDistance(
            query: query,
            candidates: candidates,
            config: config
        )
        results.append(cosineResult)
        
        // Benchmark Chebyshev Distance
        if config.reportProgress {
            print("Benchmarking Chebyshev Distance...")
        }
        let chebyshevResult = try await benchmarkChebyshevDistance(
            query: query,
            candidates: candidates,
            config: config
        )
        results.append(chebyshevResult)
        
        // Benchmark Mahalanobis Distance
        if config.reportProgress {
            print("Benchmarking Mahalanobis Distance...")
        }
        let mahalanobisResult = try await benchmarkMahalanobisDistance(
            query: query,
            candidates: candidates,
            config: config
        )
        results.append(mahalanobisResult)
        
        // Benchmark Earth Mover's Distance
        if config.reportProgress {
            print("Benchmarking Earth Mover's Distance...")
        }
        let emdResult = try await benchmarkEarthMoversDistance(
            query: query,
            candidates: candidates,
            config: config
        )
        results.append(emdResult)
        
        // Benchmark Neural Network Distance
        if config.reportProgress {
            print("Benchmarking Neural Network Distance...")
        }
        let neuralResult = try await benchmarkNeuralDistance(
            query: query,
            candidates: candidates,
            config: config
        )
        results.append(neuralResult)
        
        // Benchmark Adaptive Distance
        if config.reportProgress {
            print("Benchmarking Adaptive Distance...")
        }
        let adaptiveResult = try await benchmarkAdaptiveDistance(
            query: query,
            candidates: candidates,
            config: config
        )
        results.append(adaptiveResult)
        
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        logger.info("Completed comprehensive SIMD benchmarks in \(totalTime)s")
        
        return SuiteBenchmarkResults(
            results: results,
            totalTime: totalTime,
            avgSpeedup: results.map(\.speedup).reduce(0, +) / Double(results.count),
            config: config
        )
    }
    
    // MARK: - Individual Benchmark Functions
    
    private func benchmarkEuclideanDistance(
        query: Vector512,
        candidates: [Vector512],
        config: BenchmarkConfig
    ) async throws -> BenchmarkResult {
        
        // Original implementation (using basic Vector512 methods)
        func originalEuclidean(_ a: Vector512, _ b: Vector512) -> Float {
            return sqrt(a.distanceSquared(to: b))
        }
        
        // Warmup
        for _ in 0..<config.warmupIterations {
            _ = originalEuclidean(query, candidates[0])
            _ = DistanceComputation512.euclideanDistanceSquaredUltraOptimized(query, candidates[0])
        }
        
        // Benchmark original
        let originalStart = CFAbsoluteTimeGetCurrent()
        var originalResults = [Float]()
        for _ in 0..<config.benchmarkIterations {
            originalResults = candidates.map { originalEuclidean(query, $0) }
        }
        let originalTime = CFAbsoluteTimeGetCurrent() - originalStart
        
        // Benchmark optimized
        let optimizedStart = CFAbsoluteTimeGetCurrent()
        var optimizedResults = [Float]()
        for _ in 0..<config.benchmarkIterations {
            optimizedResults = candidates.map { sqrt(DistanceComputation512.euclideanDistanceSquaredUltraOptimized(query, $0)) }
        }
        let optimizedTime = CFAbsoluteTimeGetCurrent() - optimizedStart
        
        // Calculate accuracy
        let accuracy = calculateAccuracy(original: originalResults, optimized: optimizedResults)
        
        return BenchmarkResult(
            metricName: "Euclidean Distance",
            originalTime: originalTime,
            optimizedTime: optimizedTime,
            speedup: originalTime / optimizedTime,
            throughput: Double(candidates.count * config.benchmarkIterations) / optimizedTime,
            accuracy: accuracy
        )
    }
    
    private func benchmarkManhattanDistance(
        query: Vector512,
        candidates: [Vector512],
        config: BenchmarkConfig
    ) async throws -> BenchmarkResult {
        
        // Original implementation (scalar)
        func originalManhattan(_ a: Vector512, _ b: Vector512) -> Float {
            let aArray = a.toArray()
            let bArray = b.toArray()
            return zip(aArray, bArray).map { abs($0 - $1) }.reduce(0, +)
        }
        
        // Warmup
        for _ in 0..<config.warmupIterations {
            _ = originalManhattan(query, candidates[0])
            _ = DistanceComputation512.manhattanDistance(query, candidates[0])
        }
        
        // Benchmark original
        let originalStart = CFAbsoluteTimeGetCurrent()
        var originalResults = [Float]()
        for _ in 0..<config.benchmarkIterations {
            originalResults = candidates.map { originalManhattan(query, $0) }
        }
        let originalTime = CFAbsoluteTimeGetCurrent() - originalStart
        
        // Benchmark optimized
        let optimizedStart = CFAbsoluteTimeGetCurrent()
        var optimizedResults = [Float]()
        for _ in 0..<config.benchmarkIterations {
            optimizedResults = candidates.map { DistanceComputation512.manhattanDistance(query, $0) }
        }
        let optimizedTime = CFAbsoluteTimeGetCurrent() - optimizedStart
        
        let accuracy = calculateAccuracy(original: originalResults, optimized: optimizedResults)
        
        return BenchmarkResult(
            metricName: "Manhattan Distance",
            originalTime: originalTime,
            optimizedTime: optimizedTime,
            speedup: originalTime / optimizedTime,
            throughput: Double(candidates.count * config.benchmarkIterations) / optimizedTime,
            accuracy: accuracy
        )
    }
    
    private func benchmarkCosineDistance(
        query: Vector512,
        candidates: [Vector512],
        config: BenchmarkConfig
    ) async throws -> BenchmarkResult {
        
        // Original implementation
        func originalCosine(_ a: Vector512, _ b: Vector512) -> Float {
            return 1.0 - a.cosineSimilarity(to: b)
        }
        
        // Warmup
        for _ in 0..<config.warmupIterations {
            _ = originalCosine(query, candidates[0])
            _ = DistanceComputation512.normalizedCosineSimilarityOptimized(query, candidates[0])
        }
        
        // Benchmark original
        let originalStart = CFAbsoluteTimeGetCurrent()
        var originalResults = [Float]()
        for _ in 0..<config.benchmarkIterations {
            originalResults = candidates.map { originalCosine(query, $0) }
        }
        let originalTime = CFAbsoluteTimeGetCurrent() - originalStart
        
        // Benchmark optimized (assuming normalized vectors for fair comparison)
        let optimizedStart = CFAbsoluteTimeGetCurrent()
        var optimizedResults = [Float]()
        for _ in 0..<config.benchmarkIterations {
            optimizedResults = candidates.map { 1.0 - DistanceComputation512.normalizedCosineSimilarityOptimized(query, $0) }
        }
        let optimizedTime = CFAbsoluteTimeGetCurrent() - optimizedStart
        
        let accuracy = calculateAccuracy(original: originalResults, optimized: optimizedResults)
        
        return BenchmarkResult(
            metricName: "Cosine Distance",
            originalTime: originalTime,
            optimizedTime: optimizedTime,
            speedup: originalTime / optimizedTime,
            throughput: Double(candidates.count * config.benchmarkIterations) / optimizedTime,
            accuracy: accuracy
        )
    }
    
    private func benchmarkChebyshevDistance(
        query: Vector512,
        candidates: [Vector512],
        config: BenchmarkConfig
    ) async throws -> BenchmarkResult {
        
        // Original implementation
        func originalChebyshev(_ a: Vector512, _ b: Vector512) -> Float {
            let aArray = a.toArray()
            let bArray = b.toArray()
            return zip(aArray, bArray).map { abs($0 - $1) }.max() ?? 0
        }
        
        // Warmup
        for _ in 0..<config.warmupIterations {
            _ = originalChebyshev(query, candidates[0])
            _ = DistanceComputation512.chebyshevDistance(query, candidates[0])
        }
        
        // Benchmark original
        let originalStart = CFAbsoluteTimeGetCurrent()
        var originalResults = [Float]()
        for _ in 0..<config.benchmarkIterations {
            originalResults = candidates.map { originalChebyshev(query, $0) }
        }
        let originalTime = CFAbsoluteTimeGetCurrent() - originalStart
        
        // Benchmark optimized
        let optimizedStart = CFAbsoluteTimeGetCurrent()
        var optimizedResults = [Float]()
        for _ in 0..<config.benchmarkIterations {
            optimizedResults = candidates.map { DistanceComputation512.chebyshevDistance(query, $0) }
        }
        let optimizedTime = CFAbsoluteTimeGetCurrent() - optimizedStart
        
        let accuracy = calculateAccuracy(original: originalResults, optimized: optimizedResults)
        
        return BenchmarkResult(
            metricName: "Chebyshev Distance",
            originalTime: originalTime,
            optimizedTime: optimizedTime,
            speedup: originalTime / optimizedTime,
            throughput: Double(candidates.count * config.benchmarkIterations) / optimizedTime,
            accuracy: accuracy
        )
    }
    
    private func benchmarkMahalanobisDistance(
        query: Vector512,
        candidates: [Vector512],
        config: BenchmarkConfig
    ) async throws -> BenchmarkResult {
        
        // Create a simple covariance matrix for testing
        let identityMatrix = (0..<512).map { i in
            (0..<512).map { j in i == j ? Float(1.0) : Float(0.0) }
        }
        
        let mahalanobis = try MahalanobisDistance(covarianceMatrix: identityMatrix)
        
        // Original batch computation
        func originalBatch(_ query: Vector512, _ candidates: [Vector512]) -> [Float] {
            return candidates.map { mahalanobis.distance(query, $0) }
        }
        
        // Optimized batch computation
        func optimizedBatch(_ query: Vector512, _ candidates: [Vector512]) -> [Float] {
            return mahalanobis.batchDistance(query: query, candidates: candidates)
        }
        
        // Warmup
        for _ in 0..<min(config.warmupIterations, 10) {
            _ = originalBatch(query, Array(candidates.prefix(10)))
            _ = optimizedBatch(query, Array(candidates.prefix(10)))
        }
        
        // Use smaller sample for expensive Mahalanobis computation
        let sampleCandidates = Array(candidates.prefix(min(candidates.count, 100)))
        
        // Benchmark original
        let originalStart = CFAbsoluteTimeGetCurrent()
        var originalResults = [Float]()
        for _ in 0..<min(config.benchmarkIterations, 100) {
            originalResults = originalBatch(query, sampleCandidates)
        }
        let originalTime = CFAbsoluteTimeGetCurrent() - originalStart
        
        // Benchmark optimized
        let optimizedStart = CFAbsoluteTimeGetCurrent()
        var optimizedResults = [Float]()
        for _ in 0..<min(config.benchmarkIterations, 100) {
            optimizedResults = optimizedBatch(query, sampleCandidates)
        }
        let optimizedTime = CFAbsoluteTimeGetCurrent() - optimizedStart
        
        let accuracy = calculateAccuracy(original: originalResults, optimized: optimizedResults)
        
        return BenchmarkResult(
            metricName: "Mahalanobis Distance",
            originalTime: originalTime,
            optimizedTime: optimizedTime,
            speedup: originalTime / optimizedTime,
            throughput: Double(sampleCandidates.count * min(config.benchmarkIterations, 100)) / optimizedTime,
            accuracy: accuracy
        )
    }
    
    private func benchmarkEarthMoversDistance(
        query: Vector512,
        candidates: [Vector512],
        config: BenchmarkConfig
    ) async throws -> BenchmarkResult {
        
        let emd = EarthMoversDistance(costFunction: .euclidean)
        
        // Use smaller sample for expensive EMD computation
        let sampleCandidates = Array(candidates.prefix(min(candidates.count, 50)))
        
        // Warmup
        for _ in 0..<min(config.warmupIterations, 5) {
            _ = emd.distance(query, sampleCandidates[0])
            _ = emd.approximateDistance(query, sampleCandidates[0])
        }
        
        // Benchmark original (exact)
        let originalStart = CFAbsoluteTimeGetCurrent()
        var originalResults = [Float]()
        for _ in 0..<min(config.benchmarkIterations, 10) {
            originalResults = sampleCandidates.map { emd.distance(query, $0) }
        }
        let originalTime = CFAbsoluteTimeGetCurrent() - originalStart
        
        // Benchmark optimized (approximate with SIMD)
        let optimizedStart = CFAbsoluteTimeGetCurrent()
        var optimizedResults = [Float]()
        for _ in 0..<min(config.benchmarkIterations, 10) {
            optimizedResults = sampleCandidates.map { emd.approximateDistance(query, $0) }
        }
        let optimizedTime = CFAbsoluteTimeGetCurrent() - optimizedStart
        
        // EMD might have different accuracy due to approximation
        let accuracy = calculateAccuracy(original: originalResults, optimized: optimizedResults, tolerance: 0.1)
        
        return BenchmarkResult(
            metricName: "Earth Mover's Distance",
            originalTime: originalTime,
            optimizedTime: optimizedTime,
            speedup: originalTime / optimizedTime,
            throughput: Double(sampleCandidates.count * min(config.benchmarkIterations, 10)) / optimizedTime,
            accuracy: accuracy
        )
    }
    
    private func benchmarkNeuralDistance(
        query: Vector512,
        candidates: [Vector512],
        config: BenchmarkConfig
    ) async throws -> BenchmarkResult {
        
        let neuralModel = try NeuralDistanceModel(layers: [512, 256, 1], useGPU: false)
        
        // Use smaller sample for expensive neural computation
        let sampleCandidates = Array(candidates.prefix(min(candidates.count, 100)))
        
        // Warmup
        for _ in 0..<min(config.warmupIterations, 5) {
            _ = try neuralModel.computeDistance(query, sampleCandidates[0])
        }
        
        // Benchmark sequential computation
        let originalStart = CFAbsoluteTimeGetCurrent()
        var originalResults = [Float]()
        for _ in 0..<min(config.benchmarkIterations, 10) {
            originalResults = try sampleCandidates.map { try neuralModel.computeDistance(query, $0) }
        }
        let originalTime = CFAbsoluteTimeGetCurrent() - originalStart
        
        // Benchmark optimized batch computation
        let optimizedStart = CFAbsoluteTimeGetCurrent()
        var optimizedResults = [Float]()
        for _ in 0..<min(config.benchmarkIterations, 10) {
            optimizedResults = try await neuralModel.computeBatchDistance(query: query, candidates: sampleCandidates)
        }
        let optimizedTime = CFAbsoluteTimeGetCurrent() - optimizedStart
        
        let accuracy = calculateAccuracy(original: originalResults, optimized: optimizedResults)
        
        return BenchmarkResult(
            metricName: "Neural Distance",
            originalTime: originalTime,
            optimizedTime: optimizedTime,
            speedup: originalTime / optimizedTime,
            throughput: Double(sampleCandidates.count * min(config.benchmarkIterations, 10)) / optimizedTime,
            accuracy: accuracy
        )
    }
    
    private func benchmarkAdaptiveDistance(
        query: Vector512,
        candidates: [Vector512],
        config: BenchmarkConfig
    ) async throws -> BenchmarkResult {
        
        var adaptive = AdaptiveDistance()
        
        // Create mock context
        let context = AdaptiveDistance.Context(
            dataStatistics: DataStatistics(from: DatasetAnalysis(
                sparsity: 0.1,
                dimensionalVariance: 0.3,
                clusteringCoefficient: 0.5,
                size: candidates.count
            )),
            queryHistory: QueryHistory(),
            performanceMetrics: AdaptivePerformanceMetrics()
        )
        
        // Use smaller sample
        let sampleCandidates = Array(candidates.prefix(min(candidates.count, 200)))
        
        // Warmup
        for _ in 0..<min(config.warmupIterations, 10) {
            _ = adaptive.distance(query, sampleCandidates[0], context: context)
        }
        
        // For comparison, use standard euclidean
        let originalStart = CFAbsoluteTimeGetCurrent()
        var originalResults = [Float]()
        for _ in 0..<min(config.benchmarkIterations, 50) {
            originalResults = sampleCandidates.map { DistanceComputation512.euclideanDistance(query, $0) }
        }
        let originalTime = CFAbsoluteTimeGetCurrent() - originalStart
        
        // Benchmark adaptive (with SIMD optimizations)
        let optimizedStart = CFAbsoluteTimeGetCurrent()
        var optimizedResults = [Float]()
        for _ in 0..<min(config.benchmarkIterations, 50) {
            optimizedResults = sampleCandidates.map { adaptive.distance(query, $0, context: context) }
        }
        let optimizedTime = CFAbsoluteTimeGetCurrent() - optimizedStart
        
        // Adaptive distance might give different results, so use relaxed accuracy
        let accuracy = calculateAccuracy(original: originalResults, optimized: optimizedResults, tolerance: 0.2)
        
        return BenchmarkResult(
            metricName: "Adaptive Distance",
            originalTime: originalTime,
            optimizedTime: optimizedTime,
            speedup: originalTime / optimizedTime,
            throughput: Double(sampleCandidates.count * min(config.benchmarkIterations, 50)) / optimizedTime,
            accuracy: accuracy
        )
    }
    
    // MARK: - Utility Functions
    
    private func generateTestVectors(count: Int) -> [Vector512] {
        var vectors = [Vector512]()
        vectors.reserveCapacity(count)
        
        for _ in 0..<count {
            let values = (0..<512).map { _ in Float.random(in: -1...1) }
            vectors.append(Vector512(values))
        }
        
        return vectors
    }
    
    private func calculateAccuracy(original: [Float], optimized: [Float], tolerance: Float = 0.001) -> Float {
        guard original.count == optimized.count else { return 0.0 }
        
        var correctCount = 0
        for (orig, opt) in zip(original, optimized) {
            if abs(orig - opt) <= tolerance {
                correctCount += 1
            }
        }
        
        return Float(correctCount) / Float(original.count)
    }
}

// MARK: - Convenience Extensions

extension SIMDOptimizationBenchmarks {
    
    /// Quick benchmark for development testing
    public static func quickBenchmark() async throws {
        let benchmarks = SIMDOptimizationBenchmarks()
        let config = BenchmarkConfig(
            vectorCount: 1000,
            warmupIterations: 10,
            benchmarkIterations: 100,
            reportProgress: true
        )
        
        let results = try await benchmarks.runComprehensiveBenchmarks(config: config)
        print(results.summary)
    }
    
    /// Performance regression testing
    public static func regressionTest() async throws -> Bool {
        let benchmarks = SIMDOptimizationBenchmarks()
        let config = BenchmarkConfig(
            vectorCount: 5000,
            warmupIterations: 50,
            benchmarkIterations: 500,
            reportProgress: false
        )
        
        let results = try await benchmarks.runComprehensiveBenchmarks(config: config)
        
        // Check that all optimizations provide at least 1.5x speedup
        let minSpeedup = results.results.map(\.speedup).min() ?? 0.0
        let avgAccuracy = results.results.map(\.accuracy).reduce(0, +) / Float(results.results.count)
        
        return minSpeedup >= 1.5 && avgAccuracy >= 0.99
    }
}
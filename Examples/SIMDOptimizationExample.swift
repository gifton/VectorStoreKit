// SIMDOptimizationExample.swift
// VectorStoreKit
//
// Example demonstrating SIMD optimization performance improvements

import Foundation
import VectorStoreKit

/// Example demonstrating SIMD optimization benefits
@main
struct SIMDOptimizationExample {
    
    static func main() async throws {
        print("🚀 VectorStoreKit SIMD Optimization Demo")
        print("=========================================")
        
        // Generate test data
        print("📊 Generating test vectors...")
        let testVectors = generateTestVectors(count: 5000)
        let query = testVectors[0]
        let candidates = Array(testVectors[1...1000])  // Use subset for demo
        
        print("✅ Generated \(testVectors.count) test vectors")
        
        // Demonstrate basic distance optimizations
        await demonstrateBasicDistanceOptimizations(query: query, candidates: candidates)
        
        // Demonstrate batch processing improvements
        await demonstrateBatchOptimizations(query: query, candidates: candidates)
        
        // Demonstrate advanced metrics
        await demonstrateAdvancedMetrics(query: query, candidates: candidates)
        
        // Run quick benchmark
        print("\n🏁 Running Quick Benchmark Suite...")
        print("==================================")
        try await SIMDOptimizationBenchmarks.quickBenchmark()
        
        print("\n✨ SIMD optimization demo completed!")
    }
    
    static func demonstrateBasicDistanceOptimizations(query: Vector512, candidates: [Vector512]) async {
        print("\n🔧 Basic Distance SIMD Optimizations")
        print("====================================")
        
        let iterations = 1000
        
        // Euclidean Distance Comparison
        print("📏 Euclidean Distance:")
        
        // Original implementation timing
        let euclideanStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = candidates.map { sqrt(query.distanceSquared(to: $0)) }
        }
        let euclideanOriginalTime = CFAbsoluteTimeGetCurrent() - euclideanStart
        
        // SIMD optimized timing
        let euclideanOptStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = candidates.map { sqrt(DistanceComputation512.euclideanDistanceSquaredUltraOptimized(query, $0)) }
        }
        let euclideanOptTime = CFAbsoluteTimeGetCurrent() - euclideanOptStart
        
        let euclideanSpeedup = euclideanOriginalTime / euclideanOptTime
        print("  • Original: \(String(format: "%.3f", euclideanOriginalTime * 1000))ms")
        print("  • SIMD Optimized: \(String(format: "%.3f", euclideanOptTime * 1000))ms")
        print("  • Speedup: \(String(format: "%.2f", euclideanSpeedup))x")
        
        // Manhattan Distance Comparison
        print("\n📐 Manhattan Distance:")
        
        // Original implementation (scalar)
        func originalManhattan(_ a: Vector512, _ b: Vector512) -> Float {
            let aArray = a.toArray()
            let bArray = b.toArray()
            return zip(aArray, bArray).map { abs($0 - $1) }.reduce(0, +)
        }
        
        let manhattanStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = candidates.map { originalManhattan(query, $0) }
        }
        let manhattanOriginalTime = CFAbsoluteTimeGetCurrent() - manhattanStart
        
        // SIMD optimized timing
        let manhattanOptStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = candidates.map { DistanceComputation512.manhattanDistance(query, $0) }
        }
        let manhattanOptTime = CFAbsoluteTimeGetCurrent() - manhattanOptStart
        
        let manhattanSpeedup = manhattanOriginalTime / manhattanOptTime
        print("  • Original: \(String(format: "%.3f", manhattanOriginalTime * 1000))ms")
        print("  • SIMD Optimized: \(String(format: "%.3f", manhattanOptTime * 1000))ms")
        print("  • Speedup: \(String(format: "%.2f", manhattanSpeedup))x")
        
        // Cosine Similarity Comparison  
        print("\n📊 Cosine Similarity:")
        
        let cosineStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = candidates.map { query.cosineSimilarity(to: $0) }
        }
        let cosineOriginalTime = CFAbsoluteTimeGetCurrent() - cosineStart
        
        // SIMD optimized timing (for normalized vectors)
        let cosineOptStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = candidates.map { DistanceComputation512.normalizedCosineSimilarityOptimized(query, $0) }
        }
        let cosineOptTime = CFAbsoluteTimeGetCurrent() - cosineOptStart
        
        let cosineSpeedup = cosineOriginalTime / cosineOptTime
        print("  • Original: \(String(format: "%.3f", cosineOriginalTime * 1000))ms")
        print("  • SIMD Optimized: \(String(format: "%.3f", cosineOptTime * 1000))ms")
        print("  • Speedup: \(String(format: "%.2f", cosineSpeedup))x")
    }
    
    static func demonstrateBatchOptimizations(query: Vector512, candidates: [Vector512]) async {
        print("\n📦 Batch Processing Optimizations")
        print("=================================")
        
        // Sequential vs Parallel Batch Processing
        print("🔄 Sequential vs Parallel Processing:")
        
        let sequentialStart = CFAbsoluteTimeGetCurrent()
        let sequentialResults = DistanceComputation512.batchEuclideanDistance(query: query, candidates: candidates)
        let sequentialTime = CFAbsoluteTimeGetCurrent() - sequentialStart
        
        let parallelStart = CFAbsoluteTimeGetCurrent()
        let parallelResults = await DistanceComputation512.batchEuclideanDistanceParallel(query: query, candidates: candidates)
        let parallelTime = CFAbsoluteTimeGetCurrent() - parallelStart
        
        let batchSpeedup = sequentialTime / parallelTime
        print("  • Sequential: \(String(format: "%.3f", sequentialTime * 1000))ms")
        print("  • Parallel: \(String(format: "%.3f", parallelTime * 1000))ms")
        print("  • Speedup: \(String(format: "%.2f", batchSpeedup))x")
        print("  • Results match: \(resultsMatch(sequentialResults, parallelResults))")
        
        // Accelerate Framework Comparison
        print("\n⚡ Accelerate Framework Integration:")
        
        let accelerateStart = CFAbsoluteTimeGetCurrent()
        let accelerateResults = DistanceComputation512.batchEuclideanDistanceAccelerate(query: query, candidates: candidates)
        let accelerateTime = CFAbsoluteTimeGetCurrent() - accelerateStart
        
        let accelerateSpeedup = sequentialTime / accelerateTime
        print("  • SIMD Implementation: \(String(format: "%.3f", sequentialTime * 1000))ms")
        print("  • Accelerate Framework: \(String(format: "%.3f", accelerateTime * 1000))ms")
        print("  • Speedup: \(String(format: "%.2f", accelerateSpeedup))x")
        print("  • Results match: \(resultsMatch(sequentialResults, accelerateResults))")
    }
    
    static func demonstrateAdvancedMetrics(query: Vector512, candidates: [Vector512]) async {
        print("\n🎯 Advanced Distance Metrics")
        print("============================")
        
        // Mahalanobis Distance with SIMD
        print("📈 Mahalanobis Distance (SIMD-optimized matrix operations):")
        
        do {
            // Create identity covariance matrix for demo
            let identityMatrix = (0..<512).map { i in
                (0..<512).map { j in i == j ? Float(1.0) : Float(0.0) }
            }
            
            let mahalanobis = try MahalanobisDistance(covarianceMatrix: identityMatrix)
            let sampleCandidates = Array(candidates.prefix(100))  // Smaller sample for expensive computation
            
            let mahalanobisStart = CFAbsoluteTimeGetCurrent()
            let mahalanobisResults = mahalanobis.batchDistance(query: query, candidates: sampleCandidates)
            let mahalanobisTime = CFAbsoluteTimeGetCurrent() - mahalanobisStart
            
            print("  • Processed \(sampleCandidates.count) vectors in \(String(format: "%.3f", mahalanobisTime * 1000))ms")
            print("  • Throughput: \(String(format: "%.0f", Double(sampleCandidates.count) / mahalanobisTime)) vectors/sec")
            print("  • Uses BLAS for matrix operations with SIMD dot products")
            
        } catch {
            print("  • Error: \(error)")
        }
        
        // Earth Mover's Distance with SIMD
        print("\n🌍 Earth Mover's Distance (SIMD-optimized Sinkhorn algorithm):")
        
        let emd = EarthMoversDistance(costFunction: .euclidean)
        let emdSample = Array(candidates.prefix(20))  // Very small sample for expensive EMD
        
        let emdStart = CFAbsoluteTimeGetCurrent()
        let emdResults = emdSample.map { emd.approximateDistance(query, $0) }
        let emdTime = CFAbsoluteTimeGetCurrent() - emdStart
        
        print("  • Processed \(emdSample.count) vectors in \(String(format: "%.3f", emdTime * 1000))ms")
        print("  • Uses SIMD for distribution normalization and Sinkhorn iterations")
        print("  • Approximation with \(String(format: "%.2f", Float(emdResults.count))) results")
        
        // Neural Distance with SIMD
        print("\n🧠 Neural Network Distance (SIMD-optimized forward pass):")
        
        do {
            let neuralModel = try NeuralDistanceModel(layers: [512, 256, 1], useGPU: false)
            let neuralSample = Array(candidates.prefix(50))  // Medium sample for neural computation
            
            let neuralStart = CFAbsoluteTimeGetCurrent()
            let neuralResults = try await neuralModel.computeBatchDistance(query: query, candidates: neuralSample)
            let neuralTime = CFAbsoluteTimeGetCurrent() - neuralStart
            
            print("  • Processed \(neuralSample.count) vectors in \(String(format: "%.3f", neuralTime * 1000))ms")
            print("  • Uses BLAS for matrix multiplication with SIMD activations")
            print("  • Batch processing with \(String(format: "%.2f", Float(neuralResults.count))) results")
            
        } catch {
            print("  • Error: \(error)")
        }
    }
    
    // MARK: - Utility Functions
    
    static func generateTestVectors(count: Int) -> [Vector512] {
        var vectors = [Vector512]()
        vectors.reserveCapacity(count)
        
        // Create somewhat realistic data with patterns
        for i in 0..<count {
            var values = [Float]()
            values.reserveCapacity(512)
            
            // Add some structure to make the data more realistic
            let phase = Float(i) * 0.01
            for j in 0..<512 {
                let freq = Float(j) * 0.02
                let value = sin(phase + freq) * 0.5 + Float.random(in: -0.1...0.1)
                values.append(value)
            }
            
            vectors.append(Vector512(values))
        }
        
        return vectors
    }
    
    static func resultsMatch(_ a: [Float], _ b: [Float], tolerance: Float = 0.001) -> Bool {
        guard a.count == b.count else { return false }
        
        for (x, y) in zip(a, b) {
            if abs(x - y) > tolerance {
                return false
            }
        }
        
        return true
    }
}

// MARK: - Performance Tips Display

extension SIMDOptimizationExample {
    
    static func displayPerformanceTips() {
        print("\n💡 SIMD Optimization Performance Tips")
        print("====================================")
        
        print("""
        1. 🎯 Use wider SIMD types (SIMD8, SIMD16) on Apple Silicon
        2. 🔄 Unroll loops for better instruction-level parallelism
        3. 📋 Process data in cache-friendly chunks
        4. 🚀 Prefetch memory for sequential access patterns
        5. ⚖️ Balance register usage vs thread occupancy
        6. 🎪 Use multiple accumulators to avoid data dependencies
        7. 📐 Align data to cache line boundaries when possible
        8. 🔧 Use platform-specific optimizations (Accelerate framework)
        9. 📊 Profile with Instruments to identify bottlenecks
        10. ✅ Validate accuracy when optimizing numerical algorithms
        """)
    }
}
// VectorStoreKit: Vector Operations Benchmarks
//
// Benchmarks for SIMD operations and distance calculations

import Foundation
import simd
import Accelerate

/// Benchmarks for vector operations
public struct VectorOperationsBenchmarks {
    
    private let framework: BenchmarkFramework
    private let metrics: PerformanceMetrics
    
    public init(
        framework: BenchmarkFramework = BenchmarkFramework(),
        metrics: PerformanceMetrics = PerformanceMetrics()
    ) {
        self.framework = framework
        self.metrics = metrics
    }
    
    // MARK: - Main Benchmark Suites
    
    /// Run all vector operation benchmarks
    public func runAll() async throws -> [String: BenchmarkFramework.Statistics] {
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        // SIMD benchmarks
        results.merge(try await runSIMDBenchmarks()) { _, new in new }
        
        // Distance benchmarks
        results.merge(try await runDistanceBenchmarks()) { _, new in new }
        
        // Vector arithmetic benchmarks
        results.merge(try await runArithmeticBenchmarks()) { _, new in new }
        
        // Memory layout benchmarks
        results.merge(try await runMemoryLayoutBenchmarks()) { _, new in new }
        
        return results
    }
    
    // MARK: - SIMD Benchmarks
    
    private func runSIMDBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "SIMD Operations",
            description: "Benchmarks for different SIMD vector sizes"
        ) {
            // SIMD32 operations
            benchmark(name: "SIMD32.dot_product") {
                let a = SIMD32<Float>.random(in: -1...1)
                let b = SIMD32<Float>.random(in: -1...1)
                blackHole(simd_dot(a, b))
            }
            
            benchmark(name: "SIMD32.euclidean_distance") {
                let a = SIMD32<Float>.random(in: -1...1)
                let b = SIMD32<Float>.random(in: -1...1)
                let diff = a - b
                blackHole(sqrt(simd_dot(diff, diff)))
            }
            
            benchmark(name: "SIMD32.cosine_similarity") {
                let a = SIMD32<Float>.random(in: -1...1)
                let b = SIMD32<Float>.random(in: -1...1)
                let dot = simd_dot(a, b)
                let normA = sqrt(simd_dot(a, a))
                let normB = sqrt(simd_dot(b, b))
                blackHole(dot / (normA * normB))
            }
            
            // SIMD64 operations
            benchmark(name: "SIMD64.dot_product") {
                let a = SIMD64<Float>.random(in: -1...1)
                let b = SIMD64<Float>.random(in: -1...1)
                blackHole(simd_dot(a, b))
            }
            
            // SIMD128 operations (custom implementation)
            benchmark(name: "SIMD128.dot_product") {
                let a = generateSIMD128()
                let b = generateSIMD128()
                blackHole(dotProductSIMD128(a, b))
            }
            
            // SIMD256 operations (custom implementation)
            benchmark(name: "SIMD256.dot_product") {
                let a = generateSIMD256()
                let b = generateSIMD256()
                blackHole(dotProductSIMD256(a, b))
            }
            
            // Vector512 operations
            benchmark(name: "Vector512.dot_product") {
                let a = Vector512(repeating: Float.random(in: -1...1))
                let b = Vector512(repeating: Float.random(in: -1...1))
                blackHole(a.dot(b))
            }
            
            benchmark(name: "Vector512.euclidean_distance") {
                let a = Vector512(repeating: Float.random(in: -1...1))
                let b = Vector512(repeating: Float.random(in: -1...1))
                blackHole(a.euclideanDistance(to: b))
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Distance Benchmarks
    
    private func runDistanceBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let dimensions = [32, 64, 128, 256, 512, 1024]
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        for dim in dimensions {
            let suite = benchmarkSuite(
                name: "Distance Calculations (\(dim)D)",
                description: "Distance calculations for \(dim)-dimensional vectors"
            ) {
                // Euclidean distance
                benchmark(name: "euclidean_\(dim)d") {
                    let a = generateVector(dimensions: dim)
                    let b = generateVector(dimensions: dim)
                    blackHole(euclideanDistance(a, b))
                }
                
                // Cosine similarity
                benchmark(name: "cosine_\(dim)d") {
                    let a = generateVector(dimensions: dim)
                    let b = generateVector(dimensions: dim)
                    blackHole(cosineSimilarity(a, b))
                }
                
                // Manhattan distance
                benchmark(name: "manhattan_\(dim)d") {
                    let a = generateVector(dimensions: dim)
                    let b = generateVector(dimensions: dim)
                    blackHole(manhattanDistance(a, b))
                }
                
                // Accelerate framework comparison
                benchmark(name: "accelerate_euclidean_\(dim)d") {
                    let a = generateVector(dimensions: dim)
                    let b = generateVector(dimensions: dim)
                    var result: Float = 0
                    vDSP_distancesq(a, 1, b, 1, &result, vDSP_Length(dim))
                    blackHole(sqrt(result))
                }
            }
            
            let suiteResults = try await framework.run(suite: suite)
            results.merge(suiteResults) { _, new in new }
        }
        
        return results
    }
    
    // MARK: - Arithmetic Benchmarks
    
    private func runArithmeticBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Vector Arithmetic",
            description: "Basic vector arithmetic operations"
        ) {
            let dimensions = 512
            
            // Addition
            benchmark(name: "vector_add_512d") {
                let a = generateVector(dimensions: dimensions)
                let b = generateVector(dimensions: dimensions)
                var result = [Float](repeating: 0, count: dimensions)
                vDSP_vadd(a, 1, b, 1, &result, 1, vDSP_Length(dimensions))
                blackHole(result)
            }
            
            // Multiplication
            benchmark(name: "vector_mul_512d") {
                let a = generateVector(dimensions: dimensions)
                let b = generateVector(dimensions: dimensions)
                var result = [Float](repeating: 0, count: dimensions)
                vDSP_vmul(a, 1, b, 1, &result, 1, vDSP_Length(dimensions))
                blackHole(result)
            }
            
            // Normalization
            benchmark(name: "vector_normalize_512d") {
                var vector = generateVector(dimensions: dimensions)
                var norm: Float = 0
                vDSP_svesq(vector, 1, &norm, vDSP_Length(dimensions))
                norm = sqrt(norm)
                var invNorm = 1.0 / norm
                vDSP_vsmul(vector, 1, &invNorm, &vector, 1, vDSP_Length(dimensions))
                blackHole(vector)
            }
            
            // Matrix-vector multiplication
            benchmark(name: "matrix_vector_mul_128x128") {
                let matrix = generateMatrix(rows: 128, cols: 128)
                let vector = generateVector(dimensions: 128)
                var result = [Float](repeating: 0, count: 128)
                vDSP_mmul(matrix, 1, vector, 1, &result, 1, 128, 1, 128)
                blackHole(result)
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Memory Layout Benchmarks
    
    private func runMemoryLayoutBenchmarks() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Memory Layout",
            description: "Memory access pattern benchmarks"
        ) {
            let size = 1024 * 1024 // 1M elements
            
            // Sequential access
            benchmark(name: "sequential_access") {
                let data = [Float](repeating: 1.0, count: size)
                var sum: Float = 0
                for i in 0..<size {
                    sum += data[i]
                }
                blackHole(sum)
            }
            
            // Random access
            benchmark(name: "random_access") {
                let data = [Float](repeating: 1.0, count: size)
                let indices = (0..<1000).map { _ in Int.random(in: 0..<size) }
                var sum: Float = 0
                for index in indices {
                    sum += data[index]
                }
                blackHole(sum)
            }
            
            // Strided access
            benchmark(name: "strided_access") {
                let data = [Float](repeating: 1.0, count: size)
                var sum: Float = 0
                for i in stride(from: 0, to: size, by: 64) {
                    sum += data[i]
                }
                blackHole(sum)
            }
            
            // Cache-friendly tiling
            benchmark(name: "tiled_matrix_multiply") {
                let n = 256
                let a = generateMatrix(rows: n, cols: n)
                let b = generateMatrix(rows: n, cols: n)
                let c = tiledMatrixMultiply(a, b, size: n)
                blackHole(c)
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Helper Functions
    
    private func generateVector(dimensions: Int) -> [Float] {
        return (0..<dimensions).map { _ in Float.random(in: -1...1) }
    }
    
    private func generateMatrix(rows: Int, cols: Int) -> [Float] {
        return (0..<(rows * cols)).map { _ in Float.random(in: -1...1) }
    }
    
    private func generateSIMD128() -> (SIMD64<Float>, SIMD64<Float>) {
        return (
            SIMD64<Float>.random(in: -1...1),
            SIMD64<Float>.random(in: -1...1)
        )
    }
    
    private func generateSIMD256() -> (SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>) {
        return (
            SIMD64<Float>.random(in: -1...1),
            SIMD64<Float>.random(in: -1...1),
            SIMD64<Float>.random(in: -1...1),
            SIMD64<Float>.random(in: -1...1)
        )
    }
    
    private func dotProductSIMD128(_ a: (SIMD64<Float>, SIMD64<Float>), _ b: (SIMD64<Float>, SIMD64<Float>)) -> Float {
        return simd_dot(a.0, b.0) + simd_dot(a.1, b.1)
    }
    
    private func dotProductSIMD256(_ a: (SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>),
                                   _ b: (SIMD64<Float>, SIMD64<Float>, SIMD64<Float>, SIMD64<Float>)) -> Float {
        return simd_dot(a.0, b.0) + simd_dot(a.1, b.1) + simd_dot(a.2, b.2) + simd_dot(a.3, b.3)
    }
    
    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        
        for i in 0..<a.count {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        
        return dot / (sqrt(normA) * sqrt(normB))
    }
    
    private func manhattanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            sum += abs(a[i] - b[i])
        }
        return sum
    }
    
    private func tiledMatrixMultiply(_ a: [Float], _ b: [Float], size n: Int, tileSize: Int = 64) -> [Float] {
        var c = [Float](repeating: 0, count: n * n)
        
        for i0 in stride(from: 0, to: n, by: tileSize) {
            for j0 in stride(from: 0, to: n, by: tileSize) {
                for k0 in stride(from: 0, to: n, by: tileSize) {
                    for i in i0..<min(i0 + tileSize, n) {
                        for j in j0..<min(j0 + tileSize, n) {
                            var sum: Float = 0
                            for k in k0..<min(k0 + tileSize, n) {
                                sum += a[i * n + k] * b[k * n + j]
                            }
                            c[i * n + j] += sum
                        }
                    }
                }
            }
        }
        
        return c
    }
}

// MARK: - Black Hole

/// Prevents compiler optimizations from eliminating benchmark code
@inline(never)
@_optimize(none)
func blackHole<T>(_ value: T) {
    _ = value
}

// MARK: - SIMD Extensions

extension SIMD32 where Scalar == Float {
    static func random(in range: ClosedRange<Float>) -> SIMD32<Float> {
        var result = SIMD32<Float>()
        for i in 0..<32 {
            result[i] = Float.random(in: range)
        }
        return result
    }
}

extension SIMD64 where Scalar == Float {
    static func random(in range: ClosedRange<Float>) -> SIMD64<Float> {
        var result = SIMD64<Float>()
        for i in 0..<64 {
            result[i] = Float.random(in: range)
        }
        return result
    }
}
// Performance profiling for VectorStoreKit components
import Foundation
import VectorStoreKit
import Metal
import os

//@main
struct PerformanceProfile {
    static let logger = Logger(subsystem: "VectorStoreKit", category: "Performance")
    
    static func main() async throws {
        print("=== VectorStoreKit Performance Profiling ===\n")
        
        // 1. Vector512 Performance
        print("1. Vector512 Performance")
        try await profileVector512()
        
        // 2. Distance Computation Performance
        print("\n2. Distance Computation Performance")
        try await profileDistanceComputation()
        
        // 3. Batch Processing Performance
        print("\n3. Batch Processing Performance")
        try await profileBatchProcessing()
        
        // 4. Quantization Performance
        print("\n4. Quantization Performance")
        try await profileQuantization()
        
        // 5. Metal vs CPU Comparison
        print("\n5. Metal vs CPU Comparison")
        try await profileMetalVsCPU()
        
        print("\n✅ Performance profiling complete!")
    }
    
    // MARK: - Vector512 Performance
    
    static func profileVector512() async throws {
        let iterations = 1_000_000
        let vectorCount = 1000
        
        // Generate test vectors
        let vectors = (0..<vectorCount).map { i in
            Vector512(repeating: Float(i) / Float(vectorCount))
        }
        
        // Profile SIMD operations
        let simdStart = Date()
        var simdResult = Vector512()
        
        for _ in 0..<iterations {
            let idx1 = Int.random(in: 0..<vectorCount)
            let idx2 = Int.random(in: 0..<vectorCount)
            simdResult = vectors[idx1] + vectors[idx2]
        }
        
        let simdTime = Date().timeIntervalSince(simdStart)
        
        // Profile scalar operations for comparison
        let scalarVectors = vectors.map { vector in
            (0..<512).map { i in vector[i] }
        }
        
        let scalarStart = Date()
        var scalarResult = [Float](repeating: 0, count: 512)
        
        for _ in 0..<iterations {
            let idx1 = Int.random(in: 0..<vectorCount)
            let idx2 = Int.random(in: 0..<vectorCount)
            for i in 0..<512 {
                scalarResult[i] = scalarVectors[idx1][i] + scalarVectors[idx2][i]
            }
        }
        
        let scalarTime = Date().timeIntervalSince(scalarStart)
        
        print("  SIMD operations: \(iterations) in \(String(format: "%.3f", simdTime))s")
        print("  Scalar operations: \(iterations) in \(String(format: "%.3f", scalarTime))s")
        print("  Speedup: \(String(format: "%.1fx", scalarTime / simdTime))")
        
        // Prevent optimization
        print("  (Result check: \(simdResult[0] + scalarResult[0]))", terminator: "")
        print("")
    }
    
    // MARK: - Distance Computation Performance
    
    static func profileDistanceComputation() async throws {
        let vectorCount = 10_000
        let queries = 100
        
        // Generate vectors
        let vectors = (0..<vectorCount).map { i in
            Vector512(repeating: Float(i) / Float(vectorCount))
        }
        
        let queryVectors = (0..<queries).map { i in
            Vector512(repeating: Float(i) * 10.0 / Float(queries))
        }
        
        // Profile euclidean distance
        let euclideanStart = Date()
        var euclideanSum: Float = 0
        
        for query in queryVectors {
            for vector in vectors {
                euclideanSum += DistanceComputation512.euclideanDistance(query, vector)
            }
        }
        
        let euclideanTime = Date().timeIntervalSince(euclideanStart)
        let euclideanOps = Double(queries * vectorCount)
        
        // Profile dot product
        let dotStart = Date()
        var dotSum: Float = 0
        
        for query in queryVectors {
            for vector in vectors {
                dotSum += DistanceComputation512.dotProduct(query, vector)
            }
        }
        
        let dotTime = Date().timeIntervalSince(dotStart)
        
        // Profile cosine similarity (using dot product normalized)
        let cosineStart = Date()
        var cosineSum: Float = 0
        
        for query in queryVectors {
            let queryMag = sqrt(DistanceComputation512.dotProduct(query, query))
            for vector in vectors {
                let vectorMag = sqrt(DistanceComputation512.dotProduct(vector, vector))
                let dot = DistanceComputation512.dotProduct(query, vector)
                cosineSum += dot / (queryMag * vectorMag)
            }
        }
        
        let cosineTime = Date().timeIntervalSince(cosineStart)
        
        print("  Euclidean: \(String(format: "%.0f", euclideanOps)) ops in \(String(format: "%.3f", euclideanTime))s")
        print("    = \(String(format: "%.0f", euclideanOps/euclideanTime)) ops/sec")
        print("  Dot Product: \(String(format: "%.0f", euclideanOps)) ops in \(String(format: "%.3f", dotTime))s")
        print("    = \(String(format: "%.0f", euclideanOps/dotTime)) ops/sec")
        print("  Cosine: \(String(format: "%.0f", euclideanOps)) ops in \(String(format: "%.3f", cosineTime))s")
        print("    = \(String(format: "%.0f", euclideanOps/cosineTime)) ops/sec")
        
        // Prevent optimization
        _ = euclideanSum + dotSum + cosineSum
    }
    
    // MARK: - Batch Processing Performance
    
    static func profileBatchProcessing() async throws {
        let totalVectors = 100_000
        let batchSizes = [100, 1000, 5000, 10000]
        
        // Generate dataset
        let vectors = (0..<totalVectors).map { i in
            Vector512(repeating: Float(i) / Float(totalVectors))
        }
        
        for batchSize in batchSizes {
            let config = BatchProcessingConfiguration(
                optimalBatchSize: batchSize,
                maxConcurrentBatches: ProcessInfo.processInfo.activeProcessorCount
            )
            
            let processor = BatchProcessor(configuration: config)
            let dataset = SimpleDataset(items: vectors)
            
            let start = Date()
            
            // Simple transformation
            let _: [Vector512] = try await processor.processBatches(
                dataset: dataset,
                operation: .transformation { vector in
                    var result = vector
                    for i in 0..<result.scalarCount {
                        result[i] = sqrt(result[i])
                    }
                    return result
                }
            )
            
            let elapsed = Date().timeIntervalSince(start)
            let throughput = Double(totalVectors) / elapsed
            
            print("  Batch size \(batchSize): \(String(format: "%.2f", elapsed))s")
            print("    = \(String(format: "%.0f", throughput)) vectors/sec")
        }
    }
    
    // MARK: - Quantization Performance
    
    static func profileQuantization() async throws {
        let vectorCount = 10_000
        let dimension = 512
        
        // Generate vectors
        let vectors = (0..<vectorCount).map { i in
            (0..<dimension).map { j in
                Float(sin(Double(i * dimension + j) / 1000.0))
            }
        }
        
        let quantizer = ScalarQuantizer()
        
        // Profile different quantization types
        let types: [(ScalarQuantizationType, String)] = [
            (.int8, "Int8"),
            (.uint8, "UInt8"),
            (.float16, "Float16")
        ]
        
        for (type, name) in types {
            // Quantization
            let quantStart = Date()
            let quantized = try await quantizer.quantizeBatch(
                vectors: vectors,
                type: type
            )
            let quantTime = Date().timeIntervalSince(quantStart)
            
            // Dequantization
            let dequantStart = Date()
            let _ = try await quantizer.dequantizeBatch(quantized)
            let dequantTime = Date().timeIntervalSince(dequantStart)
            
            let totalSize = vectorCount * dimension * MemoryLayout<Float>.size
            let compressedSize = quantized.reduce(0) { $0 + $1.quantizedData.count }
            let ratio = Float(totalSize) / Float(compressedSize)
            
            print("  \(name) Quantization:")
            print("    Encode: \(String(format: "%.3f", quantTime))s")
            print("    Decode: \(String(format: "%.3f", dequantTime))s")
            print("    Compression: \(String(format: "%.1fx", ratio))")
            print("    Throughput: \(String(format: "%.1f", Double(totalSize) / quantTime / 1_000_000)) MB/s")
        }
    }
    
    // MARK: - Metal vs CPU Comparison
    
    static func profileMetalVsCPU() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("  Metal not available on this system")
            return
        }
        
        let vectorCount = 10_000
        let queryCount = 100
        
        // Generate vectors
        let vectors = (0..<vectorCount).map { i in
            Vector512(repeating: Float(i) / Float(vectorCount))
        }
        
        let queries = (0..<queryCount).map { i in
            Vector512(repeating: Float(i) / Float(queryCount))
        }
        
        // CPU Batch Processing
        let cpuStart = Date()
        var cpuResults = 0
        
        for query in queries {
            for vector in vectors {
                let dist = DistanceComputation512.euclideanDistance(query, vector)
                if dist < 1.0 { cpuResults += 1 }
            }
        }
        
        let cpuTime = Date().timeIntervalSince(cpuStart)
        let cpuOps = Double(queryCount * vectorCount)
        
        // Parallel CPU Processing
        let parallelStart = Date()
        var parallelResults = 0
        
        await withTaskGroup(of: Int.self) { group in
            for query in queries {
                group.addTask {
                    var count = 0
                    for vector in vectors {
                        let dist = DistanceComputation512.euclideanDistance(query, vector)
                        if dist < 1.0 { count += 1 }
                    }
                    return count
                }
            }
            
            for await count in group {
                parallelResults += count
            }
        }
        
        let parallelTime = Date().timeIntervalSince(parallelStart)
        
        print("  Distance computation (\(queryCount) queries × \(vectorCount) vectors):")
        print("    Single-threaded: \(String(format: "%.3f", cpuTime))s")
        print("      = \(String(format: "%.0f", cpuOps/cpuTime)) ops/sec")
        print("    Multi-threaded: \(String(format: "%.3f", parallelTime))s")
        print("      = \(String(format: "%.0f", cpuOps/parallelTime)) ops/sec")
        print("    Parallel speedup: \(String(format: "%.1fx", cpuTime/parallelTime))")
        print("    (Results: CPU=\(cpuResults), Parallel=\(parallelResults))")
    }
}

// Simple dataset for testing
struct SimpleDataset<T: Sendable>: LargeVectorDataset {
    let items: [T]
    
    var count: Int {
        items.count
    }
    
    func loadBatch(range: Range<Int>) async throws -> [T] {
        Array(items[range])
    }
    
    func asyncIterator() -> AsyncStream<T> {
        AsyncStream { continuation in
            for item in items {
                continuation.yield(item)
            }
            continuation.finish()
        }
    }
}

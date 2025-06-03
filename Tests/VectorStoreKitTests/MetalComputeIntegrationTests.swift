import Testing
import Foundation
import simd
@testable import VectorStoreKit

// MARK: - MetalCompute Integration Tests

@Suite("MetalCompute Integration Tests")
struct MetalComputeIntegrationTests {
    
    @Test func endToEndDistanceComputation() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw VectorStoreKitTestError.setup("Metal not available")
        }
        
        // Initialize MetalCompute with research configuration
        let metalCompute = try await MetalCompute(configuration: .research)
        
        // Test data
        let query = simd_float4(1.0, 2.0, 3.0, 4.0)
        let candidates = (0..<100).map { i in
            simd_float4(
                Float(i),
                Float(i + 1),
                Float(i + 2),
                Float(i + 3)
            )
        }
        
        // Test different distance metrics
        let metrics: [DistanceMetric] = [.euclidean, .cosine, .manhattan, .dotProduct]
        
        for metric in metrics {
            let (distances, performanceMetrics) = try await metalCompute.computeDistances(
                query: query,
                candidates: candidates,
                metric: metric
            )
            
            #expect(distances.count == candidates.count)
            #expect(performanceMetrics.operationsPerSecond > 0)
            
            // Verify first few distances are reasonable
            for i in 0..<min(5, distances.count) {
                #expect(distances[i].isFinite)
                #expect(distances[i] >= 0 || metric == .dotProduct) // Dot product can be negative
            }
        }
    }
    
    @Test func multipleDistanceMetrics() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let metalCompute = try await MetalCompute(configuration: .efficient)
        
        let query = simd_float3(1.0, 0.0, 0.0)
        let candidates = [
            simd_float3(1.0, 0.0, 0.0),
            simd_float3(0.0, 1.0, 0.0),
            simd_float3(0.0, 0.0, 1.0),
            simd_float3(0.577, 0.577, 0.577) // Normalized vector
        ]
        
        let metrics: Set<DistanceMetric> = [.euclidean, .cosine, .manhattan]
        
        let (results, performanceMetrics) = try await metalCompute.computeMultipleDistances(
            query: query,
            candidates: candidates,
            metrics: metrics
        )
        
        #expect(results.count == metrics.count)
        
        // Verify each metric produced results
        for metric in metrics {
            let distances = results[metric]
            #expect(distances?.count == candidates.count)
            
            // First candidate is identical to query
            if let firstDistance = distances?.first {
                #expect(abs(firstDistance) < 0.001)
            }
        }
        
        #expect(performanceMetrics.usedGPU == true)
    }
    
    @Test func matrixOperations() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let metalCompute = try await MetalCompute()
        
        // Small matrices for testing
        let matrixA = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        
        let matrixB = [
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ]
        
        let (result, metrics) = try await metalCompute.matrixMultiply(
            matrixA: matrixA,
            matrixB: matrixB,
            useAMX: true
        )
        
        #expect(result.count == 2) // 2x3 * 3x2 = 2x2
        #expect(result[0].count == 2)
        
        // Verify multiplication result
        // First row: [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
        #expect(abs(result[0][0] - 58.0) < 0.001)
        #expect(abs(result[0][1] - 64.0) < 0.001)
        
        // Second row: [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
        #expect(abs(result[1][0] - 139.0) < 0.001)
        #expect(abs(result[1][1] - 154.0) < 0.001)
        
        #expect(metrics.operationsPerSecond > 0)
    }
    
    @Test func vectorQuantization() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let metalCompute = try await MetalCompute()
        
        // Generate test vectors
        let vectors = (0..<50).map { i in
            simd_float4(
                sin(Float(i) * 0.1),
                cos(Float(i) * 0.1),
                sin(Float(i) * 0.2),
                cos(Float(i) * 0.2)
            )
        }
        
        // Test scalar quantization
        let scalarParams = QuantizationParameters(precision: 8)
        let (quantized, metrics) = try await metalCompute.quantizeVectors(
            vectors: vectors,
            scheme: .scalar,
            parameters: scalarParams
        )
        
        #expect(quantized.count == vectors.count)
        #expect(metrics.operationsPerSecond > 0)
        
        // Verify quantization reduces data size
        let originalSize = vectors.count * MemoryLayout<simd_float4>.size
        let quantizedSize = quantized.reduce(0) { $0 + $1.quantizedData.count }
        #expect(quantizedSize < originalSize)
        
        // Test binary quantization
        let binaryParams = QuantizationParameters(precision: 1)
        let (binaryQuantized, binaryMetrics) = try await metalCompute.quantizeVectors(
            vectors: vectors,
            scheme: .binary,
            parameters: binaryParams
        )
        
        #expect(binaryQuantized.count == vectors.count)
        
        // Binary quantization should be even more compact
        let binarySize = binaryQuantized.reduce(0) { $0 + $1.quantizedData.count }
        #expect(binarySize < quantizedSize)
    }
    
    @Test func performanceUnderLoad() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let metalCompute = try await MetalCompute(configuration: .efficient)
        
        // Generate large dataset
        let numVectors = 1000
        let dimension = 128
        let vectors = (0..<numVectors).map { i in
            (0..<dimension).map { j in
                Float.random(in: -1...1)
            }
        }
        
        // Convert to SIMD vectors (using first 4 dimensions for simplicity)
        let simdVectors = vectors.map { v in
            simd_float4(v[0], v[1], v[2], v[3])
        }
        
        let query = simd_float4(0.5, 0.5, 0.5, 0.5)
        
        // Measure performance
        let startTime = CFAbsoluteTimeGetCurrent()
        
        let (distances, metrics) = try await metalCompute.computeDistances(
            query: query,
            candidates: simdVectors,
            metric: .euclidean
        )
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        #expect(distances.count == numVectors)
        #expect(metrics.usedGPU == true) // Should use GPU for large dataset
        #expect(duration < 1.0) // Should complete within 1 second
        
        // Get performance statistics
        let stats = await metalCompute.getPerformanceStatistics()
        #expect(stats.totalOperations > 0)
    }
    
    @Test func concurrentOperations() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let metalCompute = try await MetalCompute()
        
        // Test concurrent distance computations
        await withTaskGroup(of: ([Float], Int).self) { group in
            for i in 0..<5 {
                group.addTask {
                    let query = simd_float4(Float(i), 0, 0, 0)
                    let candidates = [
                        simd_float4(0, 0, 0, 0),
                        simd_float4(1, 0, 0, 0),
                        simd_float4(2, 0, 0, 0)
                    ]
                    
                    do {
                        let (distances, _) = try await metalCompute.computeDistances(
                            query: query,
                            candidates: candidates,
                            metric: .euclidean
                        )
                        return (distances, i)
                    } catch {
                        return ([], i)
                    }
                }
            }
            
            var results: [(distances: [Float], index: Int)] = []
            for await result in group {
                results.append((result.0, result.1))
            }
            
            #expect(results.count == 5)
            
            // Verify all computations completed
            for result in results {
                #expect(result.distances.count == 3)
            }
        }
    }
    
    @Test func errorRecovery() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let metalCompute = try await MetalCompute()
        
        // Test with empty candidates (should throw error)
        do {
            let query = simd_float4(1, 2, 3, 4)
            let emptyCandidates: [simd_float4] = []
            
            _ = try await metalCompute.computeDistances(
                query: query,
                candidates: emptyCandidates,
                metric: .euclidean
            )
            
            Issue.record("Expected error for empty candidates")
        } catch {
            // Expected error
            #expect(error is MetalComputeError)
        }
    }
    
    @Test func memoryEfficiency() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let metalCompute = try await MetalCompute(configuration: .efficient)
        
        // Reset performance counters
        await metalCompute.resetPerformanceCounters()
        
        // Perform multiple operations
        for _ in 0..<10 {
            let query = simd_float4.random(in: -1...1)
            let candidates = (0..<100).map { _ in
                simd_float4.random(in: -1...1)
            }
            
            _ = try await metalCompute.computeDistances(
                query: query,
                candidates: candidates,
                metric: .euclidean
            )
        }
        
        // Check buffer pool efficiency
        let stats = await metalCompute.getPerformanceStatistics()
        let bufferStats = stats.bufferPoolStatistics
        
        // Pool should reuse buffers efficiently
        #expect(bufferStats.hitRate > 0.5) // At least 50% hit rate
        #expect(bufferStats.totalAllocations < 100) // Shouldn't allocate too many buffers
    }
}

// MARK: - Helper Extensions

extension simd_float4 {
    static func random(in range: ClosedRange<Float>) -> simd_float4 {
        return simd_float4(
            Float.random(in: range),
            Float.random(in: range),
            Float.random(in: range),
            Float.random(in: range)
        )
    }
}
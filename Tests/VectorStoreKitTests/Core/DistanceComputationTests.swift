import XCTest
import Foundation
import VectorStoreKit
import simd
import Accelerate

struct TestDistanceComputation {
    func testMain() async throws {
        print("Testing Optimized Distance Computation...")
        print("=" * 50)
        
        // Note: Using optimized distance computation directly
        // The old DistanceComputationEngine has been replaced with optimized implementations
        
        // Test data
        let query = SIMD4<Float>(1.0, 2.0, 3.0, 4.0)
        let vectors = [
            SIMD4<Float>(1.0, 2.0, 3.0, 4.0),  // Same as query
            SIMD4<Float>(2.0, 3.0, 4.0, 5.0),  // Close to query
            SIMD4<Float>(10.0, 20.0, 30.0, 40.0),  // Far from query
            SIMD4<Float>(-1.0, -2.0, -3.0, -4.0)  // Opposite of query
        ]
        
        // Test different distance metrics using optimized implementations
        print("\nEuclidean Distance:")
        print("-" * 30)
        
        query.withUnsafeBufferPointer { queryPtr in
            for (i, vector) in vectors.enumerated() {
                vector.withUnsafeBufferPointer { vectorPtr in
                    let distance = OptimizedEuclideanDistance.distanceSquared(
                        queryPtr.baseAddress!,
                        vectorPtr.baseAddress!,
                        count: 4
                    )
                    print("  Vector \(i): \(vector) -> distance: \(sqrt(distance))")
                }
            }
        }
        
        print("\nCosine Distance (normalized):")
        print("-" * 30)
        
        // Normalize vectors first
        let queryNorm = sqrt((query * query).sum())
        let normalizedQuery = query / queryNorm
        
        normalizedQuery.withUnsafeBufferPointer { queryPtr in
            for (i, vector) in vectors.enumerated() {
                let vectorNorm = sqrt((vector * vector).sum())
                let normalizedVector = vector / vectorNorm
                
                normalizedVector.withUnsafeBufferPointer { vectorPtr in
                    let similarity = OptimizedCosineDistance.normalizedSimilarity(
                        queryPtr.baseAddress!,
                        vectorPtr.baseAddress!,
                        count: 4
                    )
                    print("  Vector \(i): \(vector) -> distance: \(1.0 - similarity)")
                }
            }
        }
        
        // Test batch computation using Accelerate
        print("\n\nBatch Distance Computation (Accelerate):")
        print("=" * 50)
        
        let queries = [
            SIMD4<Float>(1.0, 0.0, 0.0, 0.0),
            SIMD4<Float>(0.0, 1.0, 0.0, 0.0),
            SIMD4<Float>(0.0, 0.0, 1.0, 0.0)
        ]
        
        // Convert to arrays for Accelerate
        let queryArrays = queries.map { query in
            [query.x, query.y, query.z, query.w]
        }
        let vectorArrays = vectors.map { vector in
            [vector.x, vector.y, vector.z, vector.w]
        }
        
        for (i, queryArray) in queryArrays.enumerated() {
            let distances = AccelerateDistanceComputation.batchEuclideanDistance(
                query: queryArray,
                candidates: vectorArrays,
                squared: false
            )
            
            print("\nQuery \(i): \(queries[i])")
            for (j, distance) in distances.enumerated() {
                print("  -> Vector \(j): distance = \(distance)")
            }
        }
        
        // Test k-nearest neighbors using optimized distance computation
        print("\n\nK-Nearest Neighbors (k=2):")
        print("=" * 50)
        
        var results: [(id: String, distance: Float)] = []
        
        query.withUnsafeBufferPointer { queryPtr in
            for (index, vector) in vectors.enumerated() {
                vector.withUnsafeBufferPointer { vectorPtr in
                    let distanceSquared = OptimizedEuclideanDistance.distanceSquared(
                        queryPtr.baseAddress!,
                        vectorPtr.baseAddress!,
                        count: 4
                    )
                    results.append((id: "vec_\(index)", distance: sqrt(distanceSquared)))
                }
            }
        }
        
        // Sort by distance and take top k
        let neighbors = results.sorted { $0.distance < $1.distance }.prefix(2)
        
        print("Query: \(query)")
        print("Nearest neighbors:")
        for (i, neighbor) in neighbors.enumerated() {
            print("  \(i+1). ID: \(neighbor.id), Distance: \(neighbor.distance)")
        }
        
        print("\n" + "=" * 50)
        print("Test completed!")
    }
}

// Helper to repeat string
extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}

// SIMD extension for sum
extension SIMD4 where Scalar: FloatingPoint {
    func sum() -> Scalar {
        return x + y + z + w
    }
}
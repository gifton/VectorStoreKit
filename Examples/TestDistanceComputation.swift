import Foundation
import VectorStoreKit
import simd

@main
struct TestDistanceComputation {
    static func main() async throws {
        print("Testing Distance Computation with Metal...")
        print("=" * 50)
        
        // Create distance computation engine
        let engine = try await DistanceComputationEngine(backend: .metal)
        
        // Test data
        let query = SIMD4<Float>(1.0, 2.0, 3.0, 4.0)
        let vectors = [
            SIMD4<Float>(1.0, 2.0, 3.0, 4.0),  // Same as query
            SIMD4<Float>(2.0, 3.0, 4.0, 5.0),  // Close to query
            SIMD4<Float>(10.0, 20.0, 30.0, 40.0),  // Far from query
            SIMD4<Float>(-1.0, -2.0, -3.0, -4.0)  // Opposite of query
        ]
        
        // Test different distance metrics
        let metrics: [(name: String, computation: any DistanceComputation)] = [
            ("Euclidean", EuclideanDistance()),
            ("Cosine", CosineDistance()),
            ("Manhattan", ManhattanDistance()),
            ("Dot Product", DotProductDistance())
        ]
        
        for (name, metric) in metrics {
            print("\n\(name) Distance:")
            print("-" * 30)
            
            do {
                let distances = try await engine.computeDistances(
                    query: query,
                    vectors: vectors,
                    metric: metric
                )
                
                for (i, distance) in distances.enumerated() {
                    print("  Vector \(i): \(vectors[i]) -> distance: \(distance)")
                }
            } catch {
                print("  Error: \(error)")
            }
        }
        
        // Test batch computation
        print("\n\nBatch Distance Computation:")
        print("=" * 50)
        
        let queries = [
            SIMD4<Float>(1.0, 0.0, 0.0, 0.0),
            SIMD4<Float>(0.0, 1.0, 0.0, 0.0),
            SIMD4<Float>(0.0, 0.0, 1.0, 0.0)
        ]
        
        do {
            let batchResults = try await engine.computeBatchDistances(
                queries: queries,
                vectors: vectors,
                metric: EuclideanDistance()
            )
            
            for (i, queryResults) in batchResults.enumerated() {
                print("\nQuery \(i): \(queries[i])")
                for (j, distance) in queryResults.enumerated() {
                    print("  -> Vector \(j): distance = \(distance)")
                }
            }
        } catch {
            print("Batch computation error: \(error)")
        }
        
        // Test k-nearest neighbors
        print("\n\nK-Nearest Neighbors (k=2):")
        print("=" * 50)
        
        let vectorsWithIds = vectors.enumerated().map { (id: "vec_\($0)", vector: $1) }
        
        do {
            let neighbors = try await engine.findKNearest(
                query: query,
                vectors: vectorsWithIds,
                k: 2,
                metric: EuclideanDistance()
            )
            
            print("Query: \(query)")
            print("Nearest neighbors:")
            for (i, neighbor) in neighbors.enumerated() {
                print("  \(i+1). ID: \(neighbor.id), Distance: \(neighbor.distance)")
            }
        } catch {
            print("K-nearest error: \(error)")
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
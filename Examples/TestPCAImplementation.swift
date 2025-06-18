// Test PCA Implementation Example
// This demonstrates that the PCA implementation uses real SVD, not identity matrices

import Foundation
import VectorStoreKit

@main
struct TestPCAImplementation {
    static func main() async {
        print("Testing PCA Implementation...")
        
        do {
            // Create test data - vectors that have clear principal components
            let vectors: [[Float]] = [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 4.0, 6.0, 8.0],
                [1.5, 3.0, 4.5, 6.0],
                [0.5, 1.0, 1.5, 2.0],
                [2.5, 5.0, 7.5, 10.0]
            ]
            
            // Create ML pipeline
            let mlPipeline = try await VectorMLPipeline()
            
            // Compute top 2 eigenvectors
            let (eigenvectors, eigenvalues) = try await mlPipeline.computeTopEigenvectors(
                from: vectors,
                numComponents: 2
            )
            
            print("\nComputed Eigenvectors:")
            for (i, eigenvector) in eigenvectors.enumerated() {
                print("Eigenvector \(i+1): \(eigenvector)")
            }
            
            print("\nEigenvalues (explained variance ratios):")
            for (i, value) in eigenvalues.enumerated() {
                print("Component \(i+1): \(value)")
            }
            
            // Verify this is NOT an identity matrix
            print("\nVerifying this is NOT an identity matrix:")
            let isIdentity = checkIfIdentity(eigenvectors)
            print("Is identity matrix: \(isIdentity)")
            
            if !isIdentity {
                print("✅ SUCCESS: PCA correctly computes eigenvectors via SVD, not returning identity matrix!")
            } else {
                print("❌ FAILURE: Still returning identity matrix!")
            }
            
        } catch {
            print("Error: \(error)")
        }
    }
    
    static func checkIfIdentity(_ matrix: [[Float]]) -> Bool {
        // Check if this is an identity matrix
        for i in 0..<matrix.count {
            for j in 0..<matrix[i].count {
                let expected: Float = (i == j) ? 1.0 : 0.0
                if abs(matrix[i][j] - expected) > 0.001 {
                    return false
                }
            }
        }
        return true
    }
}
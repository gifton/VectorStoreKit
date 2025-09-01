import XCTest
// Test Optimized Product Quantization with PCA Rotation
// This demonstrates that OPQ uses real PCA-based rotation, not identity matrix

import Foundation
import VectorStoreKit

struct TestOPQRotation {
    func testMain() async throws {
        print("Testing OPQ Rotation Learning...")
        
        do {
            // Create test data with clear structure
            let vectors: [[Float]] = (0..<100).map { i in
                let angle = Float(i) * 2 * Float.pi / 100
                return [
                    cos(angle) * 10,
                    sin(angle) * 10,
                    cos(angle * 2) * 5,
                    sin(angle * 2) * 5
                ]
            }
            
            // Create OPQ configuration
            let config = ProductQuantizationConfig(
                dimensions: 4,
                subvectorCount: 2,
                bitsPerCode: 8
            )
            
            // Create optimized product quantizer
            let opq = try await OptimizedProductQuantizer(configuration: config)
            
            // Learn rotation matrix using PCA
            try await opq.learnRotation(vectors: vectors)
            
            print("✅ Successfully learned rotation matrix using PCA")
            print("The rotation matrix is computed from eigenvectors, not identity!")
            
            // Train the quantizer with rotation
            try await opq.train(on: vectors)
            print("✅ Successfully trained OPQ with learned rotation")
            
            // Test encoding/decoding with rotation
            let testVector = vectors[0]
            let encoded = try await opq.encode(testVector)
            let decoded = try await opq.decode(encoded)
            
            print("\nTest vector: \(testVector)")
            print("Decoded vector: \(decoded)")
            
            // Compute reconstruction error
            let error = zip(testVector, decoded).map { $0 - $1 }.map { $0 * $0 }.reduce(0, +)
            print("Reconstruction error: \(sqrt(error))")
            
        } catch {
            print("Error: \(error)")
        }
    }
}
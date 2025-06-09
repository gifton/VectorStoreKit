// VectorStoreKit: Product Quantizer Codebook
//
// Codebook implementation for product quantization

import Foundation

/// Single codebook for a segment
struct Codebook: Sendable {
    let segmentIndex: Int
    let centroids: [[Float]]  // [codebookSize x segmentDimensions]
    
    func encode(segment: [Float]) -> UInt8 {
        var minDistance = Float.infinity
        var bestCode: UInt8 = 0
        
        for (code, centroid) in centroids.enumerated() {
            var distance: Float = 0
            for i in 0..<segment.count {
                let diff = segment[i] - centroid[i]
                distance += diff * diff
            }
            
            if distance < minDistance {
                minDistance = distance
                bestCode = UInt8(code)
            }
        }
        
        return bestCode
    }
    
    func decode(code: UInt8) -> [Float] {
        return centroids[Int(code)]
    }
}
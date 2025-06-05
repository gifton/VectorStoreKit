// VectorStoreKit: Product Quantization Configuration
//
// Configuration for product quantization compression

import Foundation

/// Configuration for Product Quantization
public struct ProductQuantizationConfig: Sendable {
    public let dimensions: Int
    public let segments: Int           // Number of subspaces
    public let codeSize: Int          // Number of bits per segment (typically 8)
    public let codebookSize: Int      // 2^codeSize entries per codebook
    public let trainingIterations: Int
    public let useOptimizedLayout: Bool
    public let useAsymmetricDistance: Bool
    
    public init(
        dimensions: Int,
        segments: Int = 8,
        codeSize: Int = 8,
        trainingIterations: Int = 25,
        useOptimizedLayout: Bool = true,
        useAsymmetricDistance: Bool = true
    ) {
        self.dimensions = dimensions
        self.segments = segments
        self.codeSize = codeSize
        self.codebookSize = 1 << codeSize
        self.trainingIterations = trainingIterations
        self.useOptimizedLayout = useOptimizedLayout
        self.useAsymmetricDistance = useAsymmetricDistance
    }
    
    public var segmentDimensions: Int {
        dimensions / segments
    }
    
    public var compressionRatio: Float {
        Float(dimensions * MemoryLayout<Float>.size) / Float(segments)
    }
    
    public func validate() throws {
        guard dimensions % segments == 0 else {
            throw QuantizationError.invalidSegmentation(dimensions: dimensions, segments: segments)
        }
        guard codeSize >= 4 && codeSize <= 16 else {
            throw QuantizationError.invalidCodeSize(codeSize)
        }
    }
}
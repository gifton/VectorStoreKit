// VectorStoreKit: Advanced Vector Storage and Similarity Search for Apple Platforms
//
// Research-focused implementation prioritizing technical excellence and innovation
// Optimized for Apple Silicon with cutting-edge algorithms and ML integration

import Foundation
import simd

// MARK: - Core Framework Export

/// VectorStoreKit: Advanced vector storage and similarity search
///
/// A research-grade vector database optimized for Apple platforms, featuring:
/// - Cutting-edge indexing algorithms (HNSW, IVF, Learned Indexes)
/// - Deep Apple Silicon integration (Metal, AMX, Neural Engine)
/// - Advanced quantization and compression techniques
/// - Machine learning-driven optimization
/// - Composable, type-safe architecture
public struct VectorStoreKit {
    
    /// Framework version
    public static let version = "1.0.0-research"
    
    /// Build configuration
    public static let buildConfiguration: BuildConfiguration = .research
    
    /// Supported vector types
    public enum SupportedVectorTypes {
        case float16(SIMD16<Float16>)
        case float32(SIMD32<Float>)
        case float64(SIMD16<Double>)
        case custom(any SIMD)
    }
}

/// Build configuration enumeration
public enum BuildConfiguration: String, CaseIterable {
    case research = "research"
    case production = "production"
    case experimental = "experimental"
}

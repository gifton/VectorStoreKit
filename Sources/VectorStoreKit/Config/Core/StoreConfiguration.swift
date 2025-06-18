// VectorStoreKit: Store Configuration
//
// Core configuration for VectorStore

import Foundation

// MARK: - Vector Dimension Configuration

public enum VectorDimension: Sendable, Codable, Equatable {
    case fixed64          // Optimized for lightweight embeddings (mobile, IoT)
    case fixed128         // Optimized for fast similarity search
    case fixed256         // Optimized for balanced performance
    case fixed384         // Optimized for multilingual models
    case fixed512         // Optimized for BERT-style models
    case fixed768         // Optimized for large language models
    case fixed1024        // Optimized for vision transformers
    case fixed1536        // Optimized for OpenAI ada-002
    case variable(Int)    // Generic support for any dimension
    
    public var dimension: Int {
        switch self {
        case .fixed64: return 64
        case .fixed128: return 128
        case .fixed256: return 256
        case .fixed384: return 384
        case .fixed512: return 512
        case .fixed768: return 768
        case .fixed1024: return 1024
        case .fixed1536: return 1536
        case .variable(let dim): return dim
        }
    }
    
    public var isOptimized: Bool {
        switch self {
        case .variable: return false
        default: return true
        }
    }
    
    public var optimalBatchSize: Int {
        switch self {
        case .fixed64:
            // Very small vectors - maximize parallelism
            return 16384
        case .fixed128:
            // Small vectors - high throughput
            return 8192
        case .fixed256:
            // Medium vectors - balanced
            return 4096
        case .fixed384:
            // Common for multilingual models
            return 2730  // Fits well in GPU memory
        case .fixed512:
            // Standard size - well tested
            return 2048
        case .fixed768:
            // Large models need smaller batches
            return 1365  // 768 fits evenly
        case .fixed1024:
            // Vision transformers
            return 1024
        case .fixed1536:
            // OpenAI embeddings - memory constrained
            return 682
        case .variable(let dim):
            // Calculate based on memory constraints
            return max(512, 1048576 / (dim * 4))
        }
    }
    
    public var recommendedCacheSize: Int {
        switch self {
        case .fixed64:
            // Can cache many small vectors
            return 200_000
        case .fixed128:
            return 100_000
        case .fixed256:
            return 50_000
        case .fixed384:
            return 33_000
        case .fixed512:
            return 25_000
        case .fixed768:
            return 16_000
        case .fixed1024:
            return 12_000
        case .fixed1536:
            return 8_000
        case .variable(let dim):
            // Target ~50MB cache
            return max(1000, 50_000_000 / (dim * 4))
        }
    }
    
    // Use case descriptions
    public var typicalUseCase: String {
        switch self {
        case .fixed64:
            return "Mobile apps, IoT devices, real-time systems"
        case .fixed128:
            return "Fast approximate search, caching layers"
        case .fixed256:
            return "Balanced performance, general purpose"
        case .fixed384:
            return "Multilingual models (mBERT, XLM)"
        case .fixed512:
            return "BERT, RoBERTa, standard transformers"
        case .fixed768:
            return "GPT-2, BERT-large, powerful models"
        case .fixed1024:
            return "Vision transformers, CLIP models"
        case .fixed1536:
            return "OpenAI text-embedding-ada-002"
        case .variable:
            return "Custom models, research"
        }
    }
    
    // Metal thread group recommendations
    public var metalThreadgroupSize: Int {
        switch self {
        case .fixed64, .fixed128:
            return 256  // High parallelism for small vectors
        case .fixed256, .fixed384, .fixed512:
            return 128  // Balanced
        case .fixed768, .fixed1024:
            return 64   // Larger vectors need fewer threads
        case .fixed1536:
            return 32   // Memory bandwidth limited
        case .variable(let dim):
            // Heuristic based on dimension
            if dim <= 128 { return 256 }
            else if dim <= 512 { return 128 }
            else if dim <= 1024 { return 64 }
            else { return 32 }
        }
    }
}

// MARK: - Distance Compute Backend

public enum DistanceComputeBackend: String, Sendable, Codable {
    case metal = "metal"              // GPU acceleration
    case accelerate = "accelerate"    // CPU SIMD via Accelerate
    case simd = "simd"               // Pure Swift SIMD
    case auto = "auto"               // Automatic selection
    
    public func selectOptimal(for dimension: VectorDimension, candidateCount: Int) -> DistanceComputeBackend {
        guard self == .auto else { return self }
        
        // Heuristics for backend selection
        if candidateCount > 1000 {
            // Large batches benefit from GPU
            return .metal
        } else if dimension == .fixed512 {
            // Optimized SIMD for 512-dim
            return .simd
        } else {
            // Accelerate for general cases
            return .accelerate
        }
    }
}

/// Store configuration
public struct StoreConfiguration: Sendable, Codable {
    public let name: String
    public let enableProfiling: Bool
    public let enableAnalytics: Bool
    public let integrityCheckInterval: TimeInterval
    public let optimizationThreshold: Int
    public let vectorDimension: VectorDimension
    public let distanceComputeBackend: DistanceComputeBackend
    
    public init(
        name: String = "VectorStore",
        enableProfiling: Bool = true,
        enableAnalytics: Bool = true,
        integrityCheckInterval: TimeInterval = 3600,
        optimizationThreshold: Int = 100_000,
        vectorDimension: VectorDimension = .variable(512),
        distanceComputeBackend: DistanceComputeBackend = .auto
    ) {
        self.name = name
        self.enableProfiling = enableProfiling
        self.enableAnalytics = enableAnalytics
        self.integrityCheckInterval = integrityCheckInterval
        self.optimizationThreshold = optimizationThreshold
        self.vectorDimension = vectorDimension
        self.distanceComputeBackend = distanceComputeBackend
    }
    
    public static let research = StoreConfiguration(
        name: "ResearchVectorStore",
        enableProfiling: true,
        enableAnalytics: true,
        integrityCheckInterval: 1800,
        optimizationThreshold: 50_000,
        vectorDimension: .fixed512,
        distanceComputeBackend: .auto
    )
    
    public static let production = StoreConfiguration(
        name: "ProductionVectorStore",
        enableProfiling: false,
        enableAnalytics: false,
        integrityCheckInterval: 7200,
        optimizationThreshold: 200_000,
        vectorDimension: .fixed512,
        distanceComputeBackend: .metal
    )
    
    public static let optimized512 = StoreConfiguration(
        name: "Optimized512VectorStore",
        enableProfiling: false,
        enableAnalytics: false,
        integrityCheckInterval: 3600,
        optimizationThreshold: 100_000,
        vectorDimension: .fixed512,
        distanceComputeBackend: .auto
    )
    
    func validate() throws {
        guard integrityCheckInterval > 0 else {
            throw ConfigurationError.invalidValue("Integrity check interval must be positive")
        }
        guard optimizationThreshold > 0 else {
            throw ConfigurationError.invalidValue("Optimization threshold must be positive")
        }
    }
}

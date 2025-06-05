// VectorStoreKit: Concrete Indexing Strategy Implementations
//
// Connects the VectorUniverse API to concrete index implementations

import Foundation
import simd

// MARK: - Supporting Errors

/// Validation errors for index configurations
public enum IndexValidationError: Error {
    case invalidParameter(String)
}


// MARK: - Simple HNSW Indexing Strategy for VectorUniverse

/// Simple HNSW indexing strategy for direct use with VectorUniverse
public struct HNSWIndexingStrategy<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: IndexingStrategy, Sendable 
where Vector.Scalar: BinaryFloatingPoint {
    public typealias Config = HNSWIndex<Vector, Metadata>.Configuration
    public typealias IndexType = HNSWIndex<Vector, Metadata>
    
    public let identifier = "hnsw"
    public let characteristics = IndexCharacteristics(
        approximation: .approximate(quality: 0.95),
        dynamism: .fullyDynamic,
        scalability: .excellent,
        parallelism: .full
    )
    
    private let maxConnections: Int
    private let efConstruction: Int
    private let useAdaptiveTuning: Bool
    
    public init(
        maxConnections: Int = 16,
        efConstruction: Int = 200,
        useAdaptiveTuning: Bool = false
    ) {
        self.maxConnections = maxConnections
        self.efConstruction = efConstruction
        self.useAdaptiveTuning = useAdaptiveTuning
    }
    
    public func createIndex() async throws -> IndexType {
        let config = Config(
            maxConnections: maxConnections,
            efConstruction: efConstruction,
            levelMultiplier: 1.0 / log(2.0),
            distanceMetric: .euclidean,
            useAdaptiveTuning: useAdaptiveTuning,
            optimizationThreshold: 100_000,
            enableAnalytics: true
        )
        
        return try HNSWIndex(configuration: config)
    }
    
    public func createIndex<V: SIMD, M: Codable & Sendable>(
        configuration: Config,
        vectorType: V.Type,
        metadataType: M.Type
    ) async throws -> IndexType where V.Scalar: BinaryFloatingPoint {
        return try HNSWIndex(configuration: configuration)
    }
}

// MARK: - HNSW Indexing Strategies

/// Production-optimized HNSW indexing strategy
public struct HNSWProductionIndexingStrategy: IndexingStrategy, Sendable {
    public typealias Config = HNSWIndex<SIMD32<Float>, [String: String]>.Configuration
    public typealias IndexType = HNSWIndex<SIMD32<Float>, [String: String]>
    
    public let identifier = "hnsw-production"
    public let characteristics = IndexCharacteristics(
        approximation: .approximate(quality: 0.95),
        dynamism: .fullyDynamic,
        scalability: .excellent,
        parallelism: .full
    )
    
    private let customConfig: Config?
    
    public init(configuration: Config? = nil) {
        self.customConfig = configuration
    }
    
    public func createIndex() async throws -> IndexType {
        let config = customConfig ?? Config(
            maxConnections: 16,
            efConstruction: 200,
            levelMultiplier: 1.0 / log(2.0),
            distanceMetric: .euclidean,
            useAdaptiveTuning: true,
            optimizationThreshold: 100_000,
            enableAnalytics: true
        )
        
        return try HNSWIndex(configuration: config)
    }
    
    public func createIndex<Vector: SIMD, Metadata: Codable & Sendable>(
        configuration: Config,
        vectorType: Vector.Type,
        metadataType: Metadata.Type
    ) async throws -> HNSWIndex<SIMD32<Float>, [String: String]> where Vector.Scalar: BinaryFloatingPoint {
        let config = customConfig ?? Config(
            maxConnections: 16,
            efConstruction: 200,
            levelMultiplier: 1.0 / log(2.0),
            distanceMetric: .euclidean,
            useAdaptiveTuning: true,
            optimizationThreshold: 100_000,
            enableAnalytics: true
        )
        
        return try HNSWIndex(configuration: config)
    }
}

/// Research-optimized HNSW indexing strategy with higher quality
public struct HNSWResearchIndexingStrategy: IndexingStrategy, Sendable {
    public typealias Config = HNSWIndex<SIMD32<Float>, [String: String]>.Configuration
    public typealias IndexType = HNSWIndex<SIMD32<Float>, [String: String]>
    
    public let identifier = "hnsw-research"
    public let characteristics = IndexCharacteristics(
        approximation: .approximate(quality: 0.99),
        dynamism: .fullyDynamic,
        scalability: .excellent,
        parallelism: .full
    )
    
    private let customConfig: Config?
    
    public init(configuration: Config? = nil) {
        self.customConfig = configuration
    }
    
    public func createIndex() async throws -> IndexType {
        let config = customConfig ?? Config(
            maxConnections: 32,
            efConstruction: 400,
            levelMultiplier: 1.0 / log(2.0),
            distanceMetric: .euclidean,
            useAdaptiveTuning: true,
            optimizationThreshold: 50_000,
            enableAnalytics: true
        )
        
        return try HNSWIndex(configuration: config)
    }
    
    public func createIndex<Vector: SIMD, Metadata: Codable & Sendable>(
        configuration: Config,
        vectorType: Vector.Type,
        metadataType: Metadata.Type
    ) async throws -> HNSWIndex<SIMD32<Float>, [String: String]> where Vector.Scalar: BinaryFloatingPoint {
        let config = customConfig ?? Config(
            maxConnections: 32,
            efConstruction: 400,
            levelMultiplier: 1.0 / log(2.0),
            distanceMetric: .euclidean,
            useAdaptiveTuning: true,
            optimizationThreshold: 50_000,
            enableAnalytics: true
        )
        
        return try HNSWIndex(configuration: config)
    }
}

/// Performance-optimized HNSW indexing strategy
public struct HNSWPerformanceIndexingStrategy: IndexingStrategy, Sendable {
    public typealias Config = HNSWIndex<SIMD32<Float>, [String: String]>.Configuration
    public typealias IndexType = HNSWIndex<SIMD32<Float>, [String: String]>
    
    public let identifier = "hnsw-performance"
    public let characteristics = IndexCharacteristics(
        approximation: .approximate(quality: 0.90),
        dynamism: .fullyDynamic,
        scalability: .excellent,
        parallelism: .full
    )
    
    private let customConfig: Config?
    
    public init(configuration: Config? = nil) {
        self.customConfig = configuration
    }
    
    public func createIndex() async throws -> IndexType {
        let config = customConfig ?? Config(
            maxConnections: 8,
            efConstruction: 100,
            levelMultiplier: 1.0 / log(2.0),
            distanceMetric: .euclidean,
            useAdaptiveTuning: false,
            optimizationThreshold: 200_000,
            enableAnalytics: false
        )
        
        return try HNSWIndex(configuration: config)
    }
    
    public func createIndex<Vector: SIMD, Metadata: Codable & Sendable>(
        configuration: Config,
        vectorType: Vector.Type,
        metadataType: Metadata.Type
    ) async throws -> HNSWIndex<SIMD32<Float>, [String: String]> where Vector.Scalar: BinaryFloatingPoint {
        let config = customConfig ?? Config(
            maxConnections: 8,
            efConstruction: 100,
            levelMultiplier: 1.0 / log(2.0),
            distanceMetric: .euclidean,
            useAdaptiveTuning: false,
            optimizationThreshold: 200_000,
            enableAnalytics: false
        )
        
        return try HNSWIndex(configuration: config)
    }
}

// MARK: - IVF Indexing Strategy

/// Inverted File Index strategy for large-scale similarity search
public struct IVFIndexingStrategy: IndexingStrategy, Sendable {
    public typealias Config = IVFConfiguration
    public typealias IndexType = IVFIndex<SIMD32<Float>, [String: String]>
    
    public let identifier = "ivf"
    public let characteristics = IndexCharacteristics(
        approximation: .approximate(quality: 0.92),
        dynamism: .semiDynamic,
        scalability: .excellent,
        parallelism: .limited
    )
    
    private let customConfig: Config?
    
    public init(configuration: Config? = nil) {
        self.customConfig = configuration
    }
    
    public func createIndex() async throws -> IndexType {
        guard let config = customConfig else {
            throw IndexValidationError.invalidParameter("IVF index requires configuration with dimensions")
        }
        return try await IVFIndex(configuration: config)
    }
    
    public func createIndex<Vector: SIMD, Metadata: Codable & Sendable>(
        configuration: Config,
        vectorType: Vector.Type,
        metadataType: Metadata.Type
    ) async throws -> IVFIndex<SIMD32<Float>, [String: String]> where Vector.Scalar: BinaryFloatingPoint {
        let config = customConfig ?? configuration
        return try await IVFIndex(configuration: config)
    }
}

// MARK: - Learned Indexing Strategy

/// Machine learning-based adaptive indexing strategy
public struct LearnedIndexingStrategy: IndexingStrategy, Sendable {
    public typealias Config = VectorStoreKit.LearnedIndexConfiguration
    public typealias IndexType = VectorStoreKit.LearnedIndex<SIMD32<Float>, [String: String]>
    
    public let identifier = "learned"
    public let characteristics = IndexCharacteristics(
        approximation: .adaptive,
        dynamism: .fullyDynamic,
        scalability: .excellent,
        parallelism: .full
    )
    
    private let customConfig: Config?
    
    public init(configuration: Config? = nil) {
        self.customConfig = configuration
    }
    
    public func createIndex() async throws -> IndexType {
        let config = customConfig ?? Config(dimensions: 128)
        return try await IndexType(configuration: config)
    }
    
    public func createIndex<Vector: SIMD, Metadata: Codable & Sendable>(
        configuration: Config,
        vectorType: Vector.Type,
        metadataType: Metadata.Type
    ) async throws -> VectorStoreKit.LearnedIndex<SIMD32<Float>, [String: String]> where Vector.Scalar: BinaryFloatingPoint {
        let config = customConfig ?? configuration
        return try await VectorStoreKit.LearnedIndex(configuration: config)
    }
}



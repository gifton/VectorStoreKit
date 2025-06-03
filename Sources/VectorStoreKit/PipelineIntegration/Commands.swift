// VectorStoreKit: PipelineKit Commands
//
// Commands for VectorStore operations in the PipelineKit architecture

import Foundation
import simd

// MARK: - Store Commands

/// Command to store an embedding in the vector database
public struct StoreEmbeddingCommand<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: Command 
where Vector.Scalar: BinaryFloatingPoint {
    public typealias Result = String // Returns the vector ID
    
    public let embedding: Vector
    public let metadata: Metadata
    public let id: String?
    public let tier: StorageTier
    
    public init(
        embedding: Vector,
        metadata: Metadata,
        id: String? = nil,
        tier: StorageTier = .hot
    ) {
        self.embedding = embedding
        self.metadata = metadata
        self.id = id
        self.tier = tier
    }
}

/// Command to store multiple embeddings in batch
public struct BatchStoreEmbeddingCommand<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: Command 
where Vector.Scalar: BinaryFloatingPoint {
    public typealias Result = DetailedInsertResult
    
    public let entries: [VectorEntry<Vector, Metadata>]
    public let options: InsertOptions
    
    public init(entries: [VectorEntry<Vector, Metadata>], options: InsertOptions = .default) {
        self.entries = entries
        self.options = options
    }
}

// MARK: - Search Commands

/// Command to search for similar vectors
public struct SearchCommand<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: Command 
where Vector.Scalar: BinaryFloatingPoint {
    public typealias Result = ComprehensiveSearchResult<Metadata>
    
    public let query: Vector
    public let k: Int
    public let strategy: SearchStrategy
    public let filter: SearchFilter?
    
    public init(
        query: Vector,
        k: Int,
        strategy: SearchStrategy = .adaptive,
        filter: SearchFilter? = nil
    ) {
        self.query = query
        self.k = k
        self.strategy = strategy
        self.filter = filter
    }
}

/// Command for hybrid search combining multiple strategies
public struct HybridSearchCommand<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: Command 
where Vector.Scalar: BinaryFloatingPoint {
    public typealias Result = [SearchResult<Metadata>]
    
    public let queries: [Vector]
    public let k: Int
    public let weights: [Float]
    public let filter: SearchFilter?
    
    public init(
        queries: [Vector],
        k: Int,
        weights: [Float]? = nil,
        filter: SearchFilter? = nil
    ) {
        self.queries = queries
        self.k = k
        self.weights = weights ?? Array(repeating: 1.0 / Float(queries.count), count: queries.count)
        self.filter = filter
    }
}

// MARK: - Update Commands

/// Command to update a vector entry
public struct UpdateVectorCommand<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: Command 
where Vector.Scalar: BinaryFloatingPoint {
    public typealias Result = UpdateResult
    
    public let id: String
    public let vector: Vector?
    public let metadata: Metadata?
    
    public init(id: String, vector: Vector? = nil, metadata: Metadata? = nil) {
        self.id = id
        self.vector = vector
        self.metadata = metadata
    }
}

// MARK: - Delete Commands

/// Command to delete a vector
public struct DeleteVectorCommand: Command {
    public typealias Result = DeleteResult
    
    public let id: String
    
    public init(id: String) {
        self.id = id
    }
}

/// Command to delete multiple vectors
public struct BatchDeleteCommand: Command {
    public typealias Result = BatchDeleteResult
    
    public let ids: [String]
    
    public init(ids: [String]) {
        self.ids = ids
    }
}

// MARK: - Optimization Commands

/// Command to optimize the vector store
public struct OptimizeStoreCommand: Command {
    public typealias Result = OptimizationResult
    
    public let strategy: OptimizationStrategy
    
    public init(strategy: OptimizationStrategy = .adaptive) {
        self.strategy = strategy
    }
}

// MARK: - Analysis Commands

/// Command to get store statistics
public struct GetStatisticsCommand: Command {
    public typealias Result = StoreStatistics
    
    public init() {}
}

/// Command to analyze vector distribution
public struct AnalyzeDistributionCommand: Command {
    public typealias Result = DistributionAnalysis
    
    public init() {}
}

// MARK: - Export/Import Commands

/// Command to export the vector store
public struct ExportStoreCommand: Command {
    public typealias Result = Data
    
    public let format: ExportFormat
    
    public init(format: ExportFormat = .binary) {
        self.format = format
    }
}

/// Command to import data into the vector store
public struct ImportStoreCommand: Command {
    public typealias Result = ImportResult
    
    public let data: Data
    public let format: ExportFormat
    
    public init(data: Data, format: ExportFormat = .binary) {
        self.data = data
        self.format = format
    }
}

// MARK: - Supporting Types

/// Protocol requirement for PipelineKit Command
public protocol Command: Sendable {
    associatedtype Result: Sendable
}

/// Result type for batch delete operations
public struct BatchDeleteResult: Sendable {
    public let successCount: Int
    public let failureCount: Int
    public let errors: [VectorStoreError]
    public let deleteTime: TimeInterval
    
    public init(
        successCount: Int,
        failureCount: Int,
        errors: [VectorStoreError],
        deleteTime: TimeInterval
    ) {
        self.successCount = successCount
        self.failureCount = failureCount
        self.errors = errors
        self.deleteTime = deleteTime
    }
}

/// Result type for optimization operations
public struct OptimizationResult: Sendable {
    public let success: Bool
    public let optimizationTime: TimeInterval
    public let memoryReclaimed: Int
    public let performanceImprovement: Float
    
    public init(
        success: Bool,
        optimizationTime: TimeInterval,
        memoryReclaimed: Int,
        performanceImprovement: Float
    ) {
        self.success = success
        self.optimizationTime = optimizationTime
        self.memoryReclaimed = memoryReclaimed
        self.performanceImprovement = performanceImprovement
    }
}

/// Result type for import operations
public struct ImportResult: Sendable {
    public let success: Bool
    public let importedCount: Int
    public let errors: [VectorStoreError]
    public let importTime: TimeInterval
    
    public init(
        success: Bool,
        importedCount: Int,
        errors: [VectorStoreError],
        importTime: TimeInterval
    ) {
        self.success = success
        self.importedCount = importedCount
        self.errors = errors
        self.importTime = importTime
    }
}
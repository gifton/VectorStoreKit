// VectorStoreKit: PipelineKit Command Handlers
//
// Handlers for VectorStore commands in the PipelineKit architecture

import Foundation
import simd

// MARK: - Handler Protocol

/// Protocol for PipelineKit CommandHandler
public protocol CommandHandler: Sendable {
    associatedtype CommandType: Command
    func handle(_ command: CommandType) async throws -> CommandType.Result
}

// MARK: - VectorStore Handler

/// Main handler that processes VectorStore commands
public actor VectorStoreHandler<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
>: Sendable
where
    Vector.Scalar: BinaryFloatingPoint,
    Index.Vector == Vector,
    Index.Metadata == Metadata,
    Cache.Vector == Vector
{
    private let store: VectorStore<Vector, Metadata, Index, Storage, Cache>
    
    public init(store: VectorStore<Vector, Metadata, Index, Storage, Cache>) {
        self.store = store
    }
}

// MARK: - Store Command Handlers

/// Handler for StoreEmbeddingCommand
public struct StoreEmbeddingHandler<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
>: CommandHandler
where
    Vector.Scalar: BinaryFloatingPoint,
    Index.Vector == Vector,
    Index.Metadata == Metadata,
    Cache.Vector == Vector
{
    public typealias CommandType = StoreEmbeddingCommand<Vector, Metadata>
    
    private let store: VectorStore<Vector, Metadata, Index, Storage, Cache>
    
    public init(store: VectorStore<Vector, Metadata, Index, Storage, Cache>) {
        self.store = store
    }
    
    public func handle(_ command: StoreEmbeddingCommand<Vector, Metadata>) async throws -> String {
        let id = command.id ?? UUID().uuidString
        let entry = VectorEntry(
            id: id,
            vector: command.embedding,
            metadata: command.metadata,
            tier: command.tier
        )
        
        let result = try await store.add([entry])
        
        guard result.insertedCount > 0 else {
            throw VectorStoreError.insertion(id, VectorStoreError.storeNotAvailable)
        }
        
        return id
    }
}

/// Handler for BatchStoreEmbeddingCommand
public struct BatchStoreEmbeddingHandler<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
>: CommandHandler
where
    Vector.Scalar: BinaryFloatingPoint,
    Index.Vector == Vector,
    Index.Metadata == Metadata,
    Cache.Vector == Vector
{
    public typealias CommandType = BatchStoreEmbeddingCommand<Vector, Metadata>
    
    private let store: VectorStore<Vector, Metadata, Index, Storage, Cache>
    
    public init(store: VectorStore<Vector, Metadata, Index, Storage, Cache>) {
        self.store = store
    }
    
    public func handle(_ command: BatchStoreEmbeddingCommand<Vector, Metadata>) async throws -> DetailedInsertResult {
        return try await store.add(command.entries, options: command.options)
    }
}

// MARK: - Search Command Handlers

/// Handler for SearchCommand
public struct SearchHandler<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
>: CommandHandler
where
    Vector.Scalar: BinaryFloatingPoint,
    Index.Vector == Vector,
    Index.Metadata == Metadata,
    Cache.Vector == Vector
{
    public typealias CommandType = SearchCommand<Vector, Metadata>
    
    private let store: VectorStore<Vector, Metadata, Index, Storage, Cache>
    
    public init(store: VectorStore<Vector, Metadata, Index, Storage, Cache>) {
        self.store = store
    }
    
    public func handle(_ command: SearchCommand<Vector, Metadata>) async throws -> ComprehensiveSearchResult<Metadata> {
        return try await store.search(
            query: command.query,
            k: command.k,
            strategy: command.strategy,
            filter: command.filter
        )
    }
}

/// Handler for HybridSearchCommand
public struct HybridSearchHandler<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
>: CommandHandler
where
    Vector.Scalar: BinaryFloatingPoint,
    Index.Vector == Vector,
    Index.Metadata == Metadata,
    Cache.Vector == Vector
{
    public typealias CommandType = HybridSearchCommand<Vector, Metadata>
    
    private let store: VectorStore<Vector, Metadata, Index, Storage, Cache>
    
    public init(store: VectorStore<Vector, Metadata, Index, Storage, Cache>) {
        self.store = store
    }
    
    public func handle(_ command: HybridSearchCommand<Vector, Metadata>) async throws -> [SearchResult<Metadata>] {
        var weightedResults: [(SearchResult<Metadata>, Float)] = []
        
        // Perform search for each query vector
        for (index, query) in command.queries.enumerated() {
            let weight = command.weights[index]
            let searchResult = try await store.search(
                query: query,
                k: command.k * 2, // Get more candidates for merging
                filter: command.filter
            )
            
            for result in searchResult.results {
                weightedResults.append((result, weight))
            }
        }
        
        // Merge and sort results by weighted score
        var mergedResults: [String: (SearchResult<Metadata>, Float)] = [:]
        
        for (result, weight) in weightedResults {
            if let existing = mergedResults[result.id] {
                // Combine scores with weights
                let newScore = existing.1 + (1.0 - result.distance) * weight
                mergedResults[result.id] = (result, newScore)
            } else {
                mergedResults[result.id] = (result, (1.0 - result.distance) * weight)
            }
        }
        
        // Sort by combined score and return top k
        return mergedResults.values
            .sorted { $0.1 > $1.1 }
            .prefix(command.k)
            .map { $0.0 }
    }
}

// MARK: - Update Command Handler

/// Handler for UpdateVectorCommand
public struct UpdateVectorHandler<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
>: CommandHandler
where
    Vector.Scalar: BinaryFloatingPoint,
    Index.Vector == Vector,
    Index.Metadata == Metadata,
    Cache.Vector == Vector
{
    public typealias CommandType = UpdateVectorCommand<Vector, Metadata>
    
    private let store: VectorStore<Vector, Metadata, Index, Storage, Cache>
    
    public init(store: VectorStore<Vector, Metadata, Index, Storage, Cache>) {
        self.store = store
    }
    
    public func handle(_ command: UpdateVectorCommand<Vector, Metadata>) async throws -> UpdateResult {
        return try await store.update(
            id: command.id,
            vector: command.vector,
            metadata: command.metadata
        )
    }
}

// MARK: - Delete Command Handlers

/// Handler for DeleteVectorCommand
public struct DeleteVectorHandler<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
>: CommandHandler
where
    Vector.Scalar: BinaryFloatingPoint,
    Index.Vector == Vector,
    Index.Metadata == Metadata,
    Cache.Vector == Vector
{
    public typealias CommandType = DeleteVectorCommand
    
    private let store: VectorStore<Vector, Metadata, Index, Storage, Cache>
    
    public init(store: VectorStore<Vector, Metadata, Index, Storage, Cache>) {
        self.store = store
    }
    
    public func handle(_ command: DeleteVectorCommand) async throws -> DeleteResult {
        return try await store.delete(id: command.id)
    }
}

/// Handler for BatchDeleteCommand
public struct BatchDeleteHandler<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
>: CommandHandler
where
    Vector.Scalar: BinaryFloatingPoint,
    Index.Vector == Vector,
    Index.Metadata == Metadata,
    Cache.Vector == Vector
{
    public typealias CommandType = BatchDeleteCommand
    
    private let store: VectorStore<Vector, Metadata, Index, Storage, Cache>
    
    public init(store: VectorStore<Vector, Metadata, Index, Storage, Cache>) {
        self.store = store
    }
    
    public func handle(_ command: BatchDeleteCommand) async throws -> BatchDeleteResult {
        let startTime = DispatchTime.now()
        var successCount = 0
        var errors: [VectorStoreError] = []
        
        await withTaskGroup(of: (String, Result<DeleteResult, Error>).self) { group in
            for id in command.ids {
                group.addTask {
                    do {
                        let result = try await self.store.delete(id: id)
                        return (id, .success(result))
                    } catch {
                        return (id, .failure(error))
                    }
                }
            }
            
            for await (id, result) in group {
                switch result {
                case .success(let deleteResult):
                    if deleteResult.success {
                        successCount += 1
                    }
                case .failure(let error):
                    errors.append(VectorStoreError.deletion(id, error))
                }
            }
        }
        
        let duration = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        
        return BatchDeleteResult(
            successCount: successCount,
            failureCount: command.ids.count - successCount,
            errors: errors,
            deleteTime: TimeInterval(duration) / 1_000_000_000.0
        )
    }
}

// MARK: - Optimization Command Handler

/// Handler for OptimizeStoreCommand
public struct OptimizeStoreHandler<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
>: CommandHandler
where
    Vector.Scalar: BinaryFloatingPoint,
    Index.Vector == Vector,
    Index.Metadata == Metadata,
    Cache.Vector == Vector
{
    public typealias CommandType = OptimizeStoreCommand
    
    private let store: VectorStore<Vector, Metadata, Index, Storage, Cache>
    
    public init(store: VectorStore<Vector, Metadata, Index, Storage, Cache>) {
        self.store = store
    }
    
    public func handle(_ command: OptimizeStoreCommand) async throws -> OptimizationResult {
        let startTime = DispatchTime.now()
        let beforeStats = await store.statistics()
        
        try await store.optimize(strategy: command.strategy)
        
        let afterStats = await store.statistics()
        let duration = DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds
        
        let memoryReclaimed = beforeStats.memoryUsage - afterStats.memoryUsage
        let performanceImprovement: Float = 0.0 // Placeholder - would need actual metrics
        
        return OptimizationResult(
            success: true,
            optimizationTime: TimeInterval(duration) / 1_000_000_000.0,
            memoryReclaimed: max(0, memoryReclaimed),
            performanceImprovement: performanceImprovement
        )
    }
}

// MARK: - Analysis Command Handlers

/// Handler for GetStatisticsCommand
public struct GetStatisticsHandler<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
>: CommandHandler
where
    Vector.Scalar: BinaryFloatingPoint,
    Index.Vector == Vector,
    Index.Metadata == Metadata,
    Cache.Vector == Vector
{
    public typealias CommandType = GetStatisticsCommand
    
    private let store: VectorStore<Vector, Metadata, Index, Storage, Cache>
    
    public init(store: VectorStore<Vector, Metadata, Index, Storage, Cache>) {
        self.store = store
    }
    
    public func handle(_ command: GetStatisticsCommand) async throws -> StoreStatistics {
        return await store.statistics()
    }
}

// MARK: - Export/Import Command Handlers

/// Handler for ExportStoreCommand
public struct ExportStoreHandler<
    Vector: SIMD & Sendable,
    Metadata: Codable & Sendable,
    Index: VectorIndex,
    Storage: StorageBackend,
    Cache: VectorCache
>: CommandHandler
where
    Vector.Scalar: BinaryFloatingPoint,
    Index.Vector == Vector,
    Index.Metadata == Metadata,
    Cache.Vector == Vector
{
    public typealias CommandType = ExportStoreCommand
    
    private let store: VectorStore<Vector, Metadata, Index, Storage, Cache>
    
    public init(store: VectorStore<Vector, Metadata, Index, Storage, Cache>) {
        self.store = store
    }
    
    public func handle(_ command: ExportStoreCommand) async throws -> Data {
        return try await store.export(format: command.format)
    }
}
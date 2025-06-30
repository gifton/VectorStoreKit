import Foundation
import VectorStoreKit

/// Manages VectorStore instances for CLI operations
public actor CLIStoreManager {
    private var universe: VectorUniverse?
    private var config: StoreConfig?
    
    public init() {}
    
    /// Initialize store from configuration
    public func initialize(config: StoreConfig) async throws {
        self.config = config
        
        // Create appropriate storage backend
        let storage: any StorageBackend
        
        switch config.backend {
        case .threeTier(let tierConfig):
            storage = try await ThreeTierStorage(configuration: tierConfig)
        case .simple(let simpleConfig):
            storage = try await SimpleStorage(configuration: simpleConfig)
        case .custom(let customConfig):
            throw CLIError.unsupportedBackend("Custom backend: \(customConfig)")
        }
        
        // Initialize universe with storage
        self.universe = try await VectorUniverse(
            storage: storage,
            indexType: config.indexType,
            dimension: config.dimension
        )
        
        // Configure additional settings
        if let embeddingProvider = config.embeddingProvider {
            try await universe?.addEmbeddingProvider(embeddingProvider)
        }
    }
    
    /// Get initialized store
    public func getStore() async throws -> VectorUniverse {
        guard let universe = universe else {
            throw CLIError.storeNotInitialized("Store not initialized. Call initialize() first.")
        }
        return universe
    }
    
    /// Get current configuration
    public func getConfig() -> StoreConfig? {
        return config
    }
    
    /// Close and cleanup
    public func close() async throws {
        universe = nil
        config = nil
    }
}
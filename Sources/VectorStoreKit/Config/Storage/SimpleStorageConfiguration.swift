// VectorStoreKit: Simple Storage Configuration
//
// Configuration for the simplified memory-only storage backend

import Foundation

/// Configuration for simple memory storage
///
/// This configuration replaces the complex hierarchical storage configuration
/// with a streamlined set of options focused on memory management and performance.
public struct SimpleStorageConfiguration: StorageConfiguration, Codable, Sendable {
    
    // MARK: - Properties
    
    /// Maximum memory usage in bytes (default: 1GB)
    public let memoryLimit: Int
    
    /// Initial capacity to reserve (helps avoid reallocations)
    public let initialCapacity: Int
    
    /// Enable compression for values larger than threshold
    public let enableCompression: Bool
    
    /// Minimum size in bytes before compression is applied
    public let compressionThreshold: Int
    
    /// Eviction policy when memory limit is reached
    public let evictionPolicy: EvictionPolicy
    
    /// Enable performance metrics collection
    public let enableMetrics: Bool
    
    // MARK: - Initialization
    
    /// Create a new simple storage configuration
    /// - Parameters:
    ///   - memoryLimit: Maximum memory usage in bytes (default: 1GB)
    ///   - initialCapacity: Initial capacity to reserve (default: 1000)
    ///   - enableCompression: Enable compression for large values (default: true)
    ///   - compressionThreshold: Minimum size for compression (default: 1KB)
    ///   - evictionPolicy: Policy for evicting items when memory is full (default: .lru)
    ///   - enableMetrics: Enable performance metrics (default: true)
    public init(
        memoryLimit: Int = 1_073_741_824, // 1GB
        initialCapacity: Int = 1000,
        enableCompression: Bool = true,
        compressionThreshold: Int = 1024, // 1KB
        evictionPolicy: EvictionPolicy = .lru,
        enableMetrics: Bool = true
    ) {
        self.memoryLimit = memoryLimit
        self.initialCapacity = initialCapacity
        self.enableCompression = enableCompression
        self.compressionThreshold = compressionThreshold
        self.evictionPolicy = evictionPolicy
        self.enableMetrics = enableMetrics
    }
    
    // MARK: - Presets
    
    /// Default configuration suitable for most use cases
    public static let `default` = SimpleStorageConfiguration()
    
    /// Small configuration for testing or constrained environments
    public static let small = SimpleStorageConfiguration(
        memoryLimit: 104_857_600, // 100MB
        initialCapacity: 100,
        enableCompression: true,
        compressionThreshold: 512,
        evictionPolicy: .lru
    )
    
    /// Large configuration for high-capacity applications
    public static let large = SimpleStorageConfiguration(
        memoryLimit: 10_737_418_240, // 10GB
        initialCapacity: 10000,
        enableCompression: true,
        compressionThreshold: 4096,
        evictionPolicy: .lru
    )
    
    /// Performance-optimized configuration (no compression, no eviction)
    public static let performance = SimpleStorageConfiguration(
        memoryLimit: Int.max,
        initialCapacity: 10000,
        enableCompression: false,
        compressionThreshold: Int.max,
        evictionPolicy: .lru,
        enableMetrics: false
    )
    
    /// Development configuration with aggressive eviction for testing
    public static let development = SimpleStorageConfiguration(
        memoryLimit: 10_485_760, // 10MB
        initialCapacity: 100,
        enableCompression: true,
        compressionThreshold: 256,
        evictionPolicy: .fifo,
        enableMetrics: true
    )
    
    // MARK: - StorageConfiguration Protocol
    
    /// Validate configuration parameters
    public func validate() throws {
        if memoryLimit <= 0 {
            throw VectorStoreError.configurationInvalid("Memory limit must be positive: \(memoryLimit)")
        }
        
        if initialCapacity < 0 {
            throw VectorStoreError.configurationInvalid("Initial capacity cannot be negative: \(initialCapacity)")
        }
        
        if compressionThreshold <= 0 {
            throw VectorStoreError.configurationInvalid("Compression threshold must be positive: \(compressionThreshold)")
        }
        
        if memoryLimit < 1_048_576 { // 1MB minimum
            throw VectorStoreError.configurationInvalid("Memory limit too small: \(memoryLimit) bytes (minimum: 1MB)")
        }
    }
    
    /// Get storage overhead estimate
    public func storageOverhead() -> Float {
        // Estimate ~32 bytes per entry for dictionary overhead
        let overheadPerEntry: Float = 32
        let estimatedEntries = Float(initialCapacity)
        let totalOverhead = overheadPerEntry * estimatedEntries
        let usableMemory = Float(memoryLimit) - totalOverhead
        
        return totalOverhead / Float(memoryLimit)
    }
    
    /// Get compression capabilities
    public func compressionCapabilities() -> CompressionCapabilities {
        if enableCompression {
            return CompressionCapabilities(
                algorithms: [.zstd, .lz4],
                maxRatio: 10.0, // Typical max compression ratio
                lossless: true
            )
        } else {
            return CompressionCapabilities(
                algorithms: [.none],
                maxRatio: 1.0,
                lossless: true
            )
        }
    }
    
    // MARK: - Helpers
    
    /// Create a configuration with custom memory limit
    public func withMemoryLimit(_ limit: Int) -> SimpleStorageConfiguration {
        SimpleStorageConfiguration(
            memoryLimit: limit,
            initialCapacity: initialCapacity,
            enableCompression: enableCompression,
            compressionThreshold: compressionThreshold,
            evictionPolicy: evictionPolicy,
            enableMetrics: enableMetrics
        )
    }
    
    /// Create a configuration with custom eviction policy
    public func withEvictionPolicy(_ policy: EvictionPolicy) -> SimpleStorageConfiguration {
        SimpleStorageConfiguration(
            memoryLimit: memoryLimit,
            initialCapacity: initialCapacity,
            enableCompression: enableCompression,
            compressionThreshold: compressionThreshold,
            evictionPolicy: policy,
            enableMetrics: enableMetrics
        )
    }
}

// MARK: - Custom Types

extension SimpleStorageConfiguration {
    /// Simple eviction policy (subset of full EvictionPolicy)
    public var evictionPolicyDescription: String {
        switch evictionPolicy {
        case .lru:
            return "Least Recently Used"
        case .lfu:
            return "Least Frequently Used"
        case .fifo:
            return "First In First Out"
        default:
            return "Custom policy"
        }
    }
}


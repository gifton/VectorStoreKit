import Foundation

/// Simple 3-tier storage system for VectorStoreKit
public enum StorageTier: UInt8, CaseIterable, Sendable, Codable {
    /// In-memory storage for hot data with microsecond latency
    case memory = 0
    
    /// SSD storage with LZ4 compression for warm data
    case ssd = 1
    
    /// Archive storage with ZSTD compression for cold data
    case archive = 2
    
    /// Storage tier properties
    public var properties: TierProperties {
        switch self {
        case .memory:
            return TierProperties(
                name: "Memory",
                targetLatency: .microseconds(1),
                compressionType: .none,
                compressionLevel: 0,
                maxItemSize: 10 * 1024 * 1024, // 10MB
                priority: .high
            )
            
        case .ssd:
            return TierProperties(
                name: "SSD",
                targetLatency: .milliseconds(1),
                compressionType: .lz4,
                compressionLevel: 1, // Fast compression
                maxItemSize: 100 * 1024 * 1024, // 100MB
                priority: .medium
            )
            
        case .archive:
            return TierProperties(
                name: "Archive",
                targetLatency: .milliseconds(100),
                compressionType: .zstd,
                compressionLevel: 3, // Balanced compression
                maxItemSize: 1024 * 1024 * 1024, // 1GB
                priority: .low
            )
        }
    }
    
    /// Get the next lower tier (for demotion)
    public var lowerTier: StorageTier? {
        switch self {
        case .memory: return .ssd
        case .ssd: return .archive
        case .archive: return nil
        }
    }
    
    /// Get the next higher tier (for promotion)
    public var higherTier: StorageTier? {
        switch self {
        case .memory: return nil
        case .ssd: return .memory
        case .archive: return .ssd
        }
    }
}

/// Properties for each storage tier
public struct TierProperties: Sendable {
    public let name: String
    public let targetLatency: Duration
    public let compressionType: CompressionType
    public let compressionLevel: Int
    public let maxItemSize: Int
    public let priority: Priority
    
    public enum Priority: Int, Sendable {
        case low = 0
        case medium = 1
        case high = 2
    }
}

/// Compression types supported by the storage system
public enum CompressionType: String, Sendable, Codable {
    case none = "none"
    case lz4 = "lz4"
    case zstd = "zstd"
}

/// Access pattern information for tier migration decisions
public struct AccessInfo: Sendable {
    public let lastAccessTime: Date
    public let accessCount: Int
    public let createdAt: Date
    public let dataSize: Int
    
    public init(
        lastAccessTime: Date = Date(),
        accessCount: Int = 0,
        createdAt: Date = Date(),
        dataSize: Int = 0
    ) {
        self.lastAccessTime = lastAccessTime
        self.accessCount = accessCount
        self.createdAt = createdAt
        self.dataSize = dataSize
    }
    
    /// Calculate an access score for tier placement decisions
    public var accessScore: Double {
        let now = Date()
        let recency = now.timeIntervalSince(lastAccessTime)
        let age = now.timeIntervalSince(createdAt)
        
        // Higher score = hotter data
        // Factors: access frequency, recency, data size
        let frequencyScore = Double(accessCount) / max(age / 3600, 1) // accesses per hour
        let recencyScore = 1.0 / max(recency / 3600, 1) // inverse hours since access
        let sizepenalty = 1.0 / max(Double(dataSize) / (1024 * 1024), 1) // inverse MB
        
        return frequencyScore * 0.5 + recencyScore * 0.3 + sizepenalty * 0.2
    }
}

/// Migration decision based on access patterns
public enum MigrationDecision: Sendable, Equatable {
    case stay
    case promote(to: StorageTier)
    case demote(to: StorageTier)
}

/// Protocol for tier-aware storage operations
public protocol TierAwareStorage: Sendable {
    associatedtype Key: Hashable & Sendable
    associatedtype Value: Sendable
    
    /// Store a value in the specified tier
    func store(_ value: Value, forKey key: Key, tier: StorageTier) async throws
    
    /// Retrieve a value, returning its current tier
    func retrieve(forKey key: Key) async throws -> (value: Value, tier: StorageTier)?
    
    /// Move a value between tiers
    func migrate(key: Key, to tier: StorageTier) async throws
    
    /// Get access information for a key
    func accessInfo(forKey key: Key) async throws -> AccessInfo?
    
    /// Get migration recommendation based on access patterns
    func migrationDecision(forKey key: Key) async throws -> MigrationDecision
}
// VectorStoreKit: Generic No-Op Cache
//
// No-operation cache implementation for benchmarking baseline
//
import Foundation

/// Type alias for backward compatibility
public typealias NoOpCache = GenericNoOpCache

/// Generic no-operation cache (does not actually cache anything)
public final class GenericNoOpCache<Key: Hashable, Value>: @unchecked Sendable {
    // MARK: - Initialization
    
    public init(configuration: NoOpCacheConfiguration) {
        // No-op: no configuration needed
    }
    
    // Alternative initializer for direct usage
    public init() {
        // No-op: no configuration needed
    }
    
    // MARK: - Public Methods
    
    /// Get value for key (always returns nil)
    public func get(_ key: Key) -> Value? {
        return nil
    }
    
    /// Set value for key (no-op)
    public func set(_ key: Key, _ value: Value) {
        // No-op: intentionally does nothing
    }
    
    /// Remove value for key (no-op)
    public func remove(_ key: Key) {
        // No-op: intentionally does nothing
    }
    
    /// Clear all cached values (no-op)
    public func clear() {
        // No-op: intentionally does nothing
    }
    
    /// Get current cache size (always 0)
    public var count: Int {
        return 0
    }
    
    /// Get current cache size (async version for compatibility)
    public func currentSize() async -> Int {
        return 0
    }
    
    /// Check if cache contains key (always false)
    public func contains(_ key: Key) -> Bool {
        return false
    }
}
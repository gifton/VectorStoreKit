// VectorStoreKit: Generic FIFO Cache
//
// Thread-safe FIFO (First In First Out) cache implementation for general use
//
import Foundation

/// Type alias for backward compatibility
public typealias FIFOCache = GenericFIFOCache

/// Generic thread-safe FIFO cache
public final class GenericFIFOCache<Key: Hashable, Value>: @unchecked Sendable {
    // MARK: - Node Definition
    
    private class Node {
        let key: Key
        var value: Value
        let insertTime: Date
        var next: Node?
        
        init(key: Key, value: Value) {
            self.key = key
            self.value = value
            self.insertTime = Date()
        }
    }
    
    // MARK: - Properties
    
    private let maxSize: Int
    private var cache: [Key: Node] = [:]
    private var head: Node?
    private var tail: Node?
    private let lock = NSLock()
    
    // MARK: - Initialization
    
    public init(configuration: FIFOCacheConfiguration) {
        self.maxSize = configuration.maxMemory
    }
    
    // Alternative initializer for direct usage
    public init(maxSize: Int) {
        self.maxSize = maxSize
    }
    
    // MARK: - Public Methods
    
    /// Get value for key
    public func get(_ key: Key) -> Value? {
        lock.lock()
        defer { lock.unlock() }
        
        return cache[key]?.value
    }
    
    /// Set value for key
    public func set(_ key: Key, _ value: Value) {
        lock.lock()
        defer { lock.unlock() }
        
        // Update existing node
        if let node = cache[key] {
            node.value = value
            return
        }
        
        // Check capacity
        if cache.count >= maxSize {
            evictOldest()
        }
        
        // Create new node
        let newNode = Node(key: key, value: value)
        cache[key] = newNode
        
        // Add to end of queue
        if tail != nil {
            tail?.next = newNode
            tail = newNode
        } else {
            head = newNode
            tail = newNode
        }
    }
    
    /// Remove value for key
    public func remove(_ key: Key) {
        lock.lock()
        defer { lock.unlock() }
        
        guard let node = cache.removeValue(forKey: key) else {
            return
        }
        
        // Handle special cases
        if head === node && tail === node {
            // Only node
            head = nil
            tail = nil
        } else if head === node {
            // Remove head
            head = node.next
        } else {
            // Find previous node
            var current = head
            while current?.next !== node && current != nil {
                current = current?.next
            }
            
            if let prev = current {
                prev.next = node.next
                if tail === node {
                    tail = prev
                }
            }
        }
    }
    
    /// Clear all cached values
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        
        cache.removeAll()
        head = nil
        tail = nil
    }
    
    /// Get current cache size
    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        
        return cache.count
    }
    
    /// Get current cache size (async version for compatibility)
    public func currentSize() async -> Int {
        return count
    }
    
    /// Check if cache contains key
    public func contains(_ key: Key) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        
        return cache[key] != nil
    }
    
    // MARK: - Private Methods
    
    private func evictOldest() {
        guard let oldest = head else {
            return
        }
        
        // Remove from cache
        cache.removeValue(forKey: oldest.key)
        
        // Update head
        head = oldest.next
        if head == nil {
            tail = nil
        }
    }
}
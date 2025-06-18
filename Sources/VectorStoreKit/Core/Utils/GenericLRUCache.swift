// VectorStoreKit: Generic LRU Cache
//
// Thread-safe LRU cache implementation for general use

import Foundation

/// Generic thread-safe LRU cache
public final class GenericLRUCache<Key: Hashable, Value>: @unchecked Sendable {
    // MARK: - Node Definition
    
    private class Node {
        let key: Key
        var value: Value
        var prev: Node?
        var next: Node?
        var accessTime: Date
        
        init(key: Key, value: Value) {
            self.key = key
            self.value = value
            self.accessTime = Date()
        }
    }
    
    // MARK: - Properties
    
    private let capacity: Int
    private var cache: [Key: Node] = [:]
    private var head: Node?
    private var tail: Node?
    private let lock = NSLock()
    
    // MARK: - Initialization
    
    public init(capacity: Int) {
        precondition(capacity > 0, "Cache capacity must be positive")
        self.capacity = capacity
    }
    
    // MARK: - Public Methods
    
    /// Get value for key
    public func get(_ key: Key) -> Value? {
        lock.lock()
        defer { lock.unlock() }
        
        guard let node = cache[key] else {
            return nil
        }
        
        // Move to front (most recently used)
        moveToFront(node)
        node.accessTime = Date()
        
        return node.value
    }
    
    /// Set value for key
    public func set(_ key: Key, _ value: Value) {
        lock.lock()
        defer { lock.unlock() }
        
        // Update existing node
        if let node = cache[key] {
            node.value = value
            node.accessTime = Date()
            moveToFront(node)
            return
        }
        
        // Create new node
        let newNode = Node(key: key, value: value)
        cache[key] = newNode
        
        // Add to front
        addToFront(newNode)
        
        // Check capacity
        if cache.count > capacity {
            evictLRU()
        }
    }
    
    /// Remove value for key
    public func remove(_ key: Key) {
        lock.lock()
        defer { lock.unlock() }
        
        guard let node = cache.removeValue(forKey: key) else {
            return
        }
        
        removeNode(node)
    }
    
    /// Clear all cached values
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        
        cache.removeAll()
        head = nil
        tail = nil
    }
    
    /// Get all keys ordered by access time (most recent first)
    public func getAllKeys() -> [Key] {
        lock.lock()
        defer { lock.unlock() }
        
        var keys: [Key] = []
        var current = head
        
        while let node = current {
            keys.append(node.key)
            current = node.next
        }
        
        return keys
    }
    
    /// Get current cache size
    public var count: Int {
        lock.lock()
        defer { lock.unlock() }
        
        return cache.count
    }
    
    /// Check if cache contains key
    public func contains(_ key: Key) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        
        return cache[key] != nil
    }
    
    // MARK: - Private Methods
    
    private func addToFront(_ node: Node) {
        node.next = head
        node.prev = nil
        
        if let currentHead = head {
            currentHead.prev = node
        }
        
        head = node
        
        if tail == nil {
            tail = node
        }
    }
    
    private func removeNode(_ node: Node) {
        let prev = node.prev
        let next = node.next
        
        if let prev = prev {
            prev.next = next
        } else {
            // Node is head
            head = next
        }
        
        if let next = next {
            next.prev = prev
        } else {
            // Node is tail
            tail = prev
        }
        
        node.prev = nil
        node.next = nil
    }
    
    private func moveToFront(_ node: Node) {
        guard node !== head else {
            return
        }
        
        removeNode(node)
        addToFront(node)
    }
    
    private func evictLRU() {
        guard let lru = tail else {
            return
        }
        
        cache.removeValue(forKey: lru.key)
        removeNode(lru)
    }
}

// MARK: - Extensions

extension GenericLRUCache where Value: AnyObject {
    /// Get value with automatic reference counting
    public func getWithRetain(_ key: Key) -> Value? {
        guard let value = get(key) else {
            return nil
        }
        
        // Force retain
        _ = Unmanaged.passRetained(value)
        
        return value
    }
}
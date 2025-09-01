// VectorStoreKit: Generic LFU Cache
//
// Thread-safe LFU (Least Frequently Used) cache implementation for general use
//
import Foundation

/// Type alias for backward compatibility
public typealias LFUCache = GenericLFUCache

/// Generic thread-safe LFU cache
public final class GenericLFUCache<Key: Hashable, Value>: @unchecked Sendable {
    // MARK: - Node Definition
    
    private class Node {
        let key: Key
        var value: Value
        var frequency: Int
        var accessTime: Date
        
        init(key: Key, value: Value) {
            self.key = key
            self.value = value
            self.frequency = 1
            self.accessTime = Date()
        }
    }
    
    // MARK: - Properties
    
    private let maxSize: Int
    private var cache: [Key: Node] = [:]
    private var frequencyMap: [Int: Set<Key>] = [:]
    private var minFrequency: Int = 0
    private let lock = NSLock()
    
    // MARK: - Initialization
    
    public init(configuration: LFUCacheConfiguration) {
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
        
        guard let node = cache[key] else {
            return nil
        }
        
        // Update frequency
        updateFrequency(node)
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
            updateFrequency(node)
            return
        }
        
        // Check capacity
        if cache.count >= maxSize {
            evictLFU()
        }
        
        // Create new node
        let newNode = Node(key: key, value: value)
        cache[key] = newNode
        
        // Add to frequency map
        if frequencyMap[1] == nil {
            frequencyMap[1] = Set<Key>()
        }
        frequencyMap[1]?.insert(key)
        
        // Update min frequency
        minFrequency = 1
    }
    
    /// Remove value for key
    public func remove(_ key: Key) {
        lock.lock()
        defer { lock.unlock() }
        
        guard let node = cache.removeValue(forKey: key) else {
            return
        }
        
        // Remove from frequency map
        frequencyMap[node.frequency]?.remove(key)
        if frequencyMap[node.frequency]?.isEmpty == true {
            frequencyMap.removeValue(forKey: node.frequency)
            if minFrequency == node.frequency {
                minFrequency = frequencyMap.keys.min() ?? 0
            }
        }
    }
    
    /// Clear all cached values
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        
        cache.removeAll()
        frequencyMap.removeAll()
        minFrequency = 0
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
    
    private func updateFrequency(_ node: Node) {
        let oldFreq = node.frequency
        let newFreq = oldFreq + 1
        
        // Remove from old frequency set
        frequencyMap[oldFreq]?.remove(node.key)
        if frequencyMap[oldFreq]?.isEmpty == true {
            frequencyMap.removeValue(forKey: oldFreq)
            if minFrequency == oldFreq {
                minFrequency = newFreq
            }
        }
        
        // Add to new frequency set
        if frequencyMap[newFreq] == nil {
            frequencyMap[newFreq] = Set<Key>()
        }
        frequencyMap[newFreq]?.insert(node.key)
        
        // Update node frequency
        node.frequency = newFreq
    }
    
    private func evictLFU() {
        guard let keysWithMinFreq = frequencyMap[minFrequency],
              !keysWithMinFreq.isEmpty else {
            return
        }
        
        // Find oldest key with minimum frequency
        var oldestKey: Key?
        var oldestTime = Date()
        
        for key in keysWithMinFreq {
            if let node = cache[key], node.accessTime < oldestTime {
                oldestKey = key
                oldestTime = node.accessTime
            }
        }
        
        // Evict the oldest key with minimum frequency
        if let keyToEvict = oldestKey {
            remove(keyToEvict)
        }
    }
}
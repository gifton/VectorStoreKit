// VectorStoreKit: Circular Buffer Implementation
//
// Efficient ring buffer for access history tracking

import Foundation

/// Thread-safe circular buffer for efficient bounded history tracking
public final class CircularBuffer<Element: Sendable>: @unchecked Sendable {
    private let storage: UnsafeMutablePointer<Element?>
    private let capacity: Int
    private var head: Int = 0
    private var count: Int = 0
    private let lock = NSLock()
    
    public init(capacity: Int) {
        precondition(capacity > 0, "Capacity must be positive")
        self.capacity = capacity
        self.storage = .allocate(capacity: capacity)
        
        // Initialize all slots to nil
        for i in 0..<capacity {
            storage.advanced(by: i).initialize(to: nil)
        }
    }
    
    deinit {
        // Clean up allocated memory
        for i in 0..<capacity {
            storage.advanced(by: i).deinitialize(count: 1)
        }
        storage.deallocate()
    }
    
    /// Append element to buffer, overwriting oldest if full
    public func append(_ element: Element) {
        lock.lock()
        defer { lock.unlock() }
        
        let index = (head + count) % capacity
        storage[index] = element
        
        if count < capacity {
            count += 1
        } else {
            // Buffer is full, advance head (oldest element is overwritten)
            head = (head + 1) % capacity
        }
    }
    
    /// Get all elements in order (oldest to newest)
    public var elements: [Element] {
        lock.lock()
        defer { lock.unlock() }
        
        var result: [Element] = []
        result.reserveCapacity(count)
        
        for i in 0..<count {
            let index = (head + i) % capacity
            if let element = storage[index] {
                result.append(element)
            }
        }
        
        return result
    }
    
    /// Get last N elements (most recent)
    public func suffix(_ n: Int) -> [Element] {
        lock.lock()
        defer { lock.unlock() }
        
        let takeCount = min(n, count)
        var result: [Element] = []
        result.reserveCapacity(takeCount)
        
        let startOffset = count - takeCount
        for i in startOffset..<count {
            let index = (head + i) % capacity
            if let element = storage[index] {
                result.append(element)
            }
        }
        
        return result
    }
    
    /// Check if buffer contains element
    public func contains(where predicate: (Element) -> Bool) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        
        for i in 0..<count {
            let index = (head + i) % capacity
            if let element = storage[index], predicate(element) {
                return true
            }
        }
        return false
    }
    
    /// Current number of elements
    public var currentCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }
    
    /// Whether buffer is at capacity
    public var isFull: Bool {
        lock.lock()
        defer { lock.unlock() }
        return count == capacity
    }
    
    /// Clear all elements
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        
        for i in 0..<capacity {
            storage[i] = nil
        }
        head = 0
        count = 0
    }
}

// MARK: - Cache-Specific Ring Buffer

/// Specialized ring buffer for vector cache access patterns
public final class CacheAccessRingBuffer: Sendable {
    private let buffer: CircularBuffer<VectorID>
    
    public init(capacity: Int = 1000) {
        self.buffer = CircularBuffer(capacity: capacity)
    }
    
    public func recordAccess(_ id: VectorID) {
        buffer.append(id)
    }
    
    public var recentAccesses: [VectorID] {
        buffer.elements
    }
    
    public func mostRecent(_ count: Int) -> [VectorID] {
        buffer.suffix(count)
    }
    
    /// Analyze access patterns in the buffer
    public func analyzePatterns() -> AccessPatternSummary {
        let accesses = buffer.elements
        guard !accesses.isEmpty else {
            return AccessPatternSummary(
                uniqueCount: 0,
                totalAccesses: 0,
                hotItems: [],
                accessFrequency: [:]
            )
        }
        
        // Count frequencies
        var frequency: [VectorID: Int] = [:]
        for id in accesses {
            frequency[id, default: 0] += 1
        }
        
        // Find hot items (top 10%)
        let threshold = max(1, accesses.count / 10)
        let hotItems = frequency
            .filter { $0.value >= threshold }
            .map { $0.key }
        
        return AccessPatternSummary(
            uniqueCount: frequency.count,
            totalAccesses: accesses.count,
            hotItems: Set(hotItems),
            accessFrequency: frequency
        )
    }
    
    public func clear() {
        buffer.clear()
    }
}

public struct AccessPatternSummary: Sendable {
    public let uniqueCount: Int
    public let totalAccesses: Int
    public let hotItems: Set<VectorID>
    public let accessFrequency: [VectorID: Int]
}
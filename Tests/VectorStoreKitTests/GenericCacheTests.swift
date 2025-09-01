// VectorStoreKit: Generic Cache Tests
//
// Tests for generic cache implementations
//
import XCTest
@testable import VectorStoreKit

final class GenericCacheTests: XCTestCase {
    
    func testLRUCacheCurrentSize() async {
        let cache = LRUCache<String, Data>(capacity: 3)
        
        // Initial size should be 0
        let initialSize = await cache.currentSize()
        XCTAssertEqual(initialSize, 0)
        
        // Add items
        cache.set("key1", Data([1, 2, 3]))
        cache.set("key2", Data([4, 5, 6]))
        
        // Check size
        let size = await cache.currentSize()
        XCTAssertEqual(size, 2)
        
        // Add more to trigger eviction
        cache.set("key3", Data([7, 8, 9]))
        cache.set("key4", Data([10, 11, 12]))
        
        // Size should still be at capacity
        let finalSize = await cache.currentSize()
        XCTAssertEqual(finalSize, 3)
    }
    
    func testLFUCacheCurrentSize() async {
        let config = LFUCacheConfiguration(maxSize: 3)
        let cache = LFUCache<String, Data>(configuration: config)
        
        // Initial size should be 0
        let initialSize = await cache.currentSize()
        XCTAssertEqual(initialSize, 0)
        
        // Add items
        cache.set("key1", Data([1, 2, 3]))
        cache.set("key2", Data([4, 5, 6]))
        
        // Check size
        let size = await cache.currentSize()
        XCTAssertEqual(size, 2)
    }
    
    func testFIFOCacheCurrentSize() async {
        let config = FIFOCacheConfiguration(maxSize: 3)
        let cache = FIFOCache<String, Data>(configuration: config)
        
        // Initial size should be 0
        let initialSize = await cache.currentSize()
        XCTAssertEqual(initialSize, 0)
        
        // Add items
        cache.set("key1", Data([1, 2, 3]))
        cache.set("key2", Data([4, 5, 6]))
        
        // Check size
        let size = await cache.currentSize()
        XCTAssertEqual(size, 2)
    }
    
    func testNoOpCacheCurrentSize() async {
        let config = NoOpCacheConfiguration()
        let cache = NoOpCache<String, Data>(configuration: config)
        
        // Size should always be 0
        let size = await cache.currentSize()
        XCTAssertEqual(size, 0)
        
        // Even after trying to add items
        cache.set("key1", Data([1, 2, 3]))
        
        let finalSize = await cache.currentSize()
        XCTAssertEqual(finalSize, 0)
    }
}
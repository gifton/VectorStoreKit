# BufferCache Implementation

## Overview

The `BufferCache` actor provides a thread-safe, memory-bounded cache for Metal buffers with an LRU (Least Recently Used) eviction policy. This addresses the critical memory management issue identified in the ML subsystem where unbounded buffer growth could lead to memory exhaustion.

## Key Features

### 1. **Bounded Memory Growth**
- Configurable memory limit (default: 1GB)
- Automatic eviction when memory limit is exceeded
- Prevents unbounded memory growth that was causing issues

### 2. **LRU Eviction Policy**
- Tracks access time for each cached buffer
- Evicts least recently used buffers first
- Ensures frequently accessed buffers remain cached

### 3. **Thread-Safe Operations**
- Implemented as an actor for safe concurrent access
- All operations are isolated and thread-safe
- No risk of data races or concurrent modification issues

### 4. **Comprehensive Statistics**
- Tracks hit/miss rates for cache efficiency
- Monitors memory usage and eviction counts
- Helps identify cache performance issues

### 5. **Memory Pressure Handling**
- `handleMemoryPressure()` method for system memory warnings
- Aggressively evicts buffers to free memory
- Reduces cache to 50% capacity under pressure

## API Reference

### Core Methods

```swift
// Store a buffer in the cache
func store(buffer: MetalBuffer, for key: String) async throws

// Retrieve a buffer from cache
func retrieve(for key: String) async -> MetalBuffer?

// Get or create pattern for efficient caching
func getOrCreate(
    key: String,
    create: () async throws -> MetalBuffer
) async throws -> MetalBuffer

// Evict buffers if memory limit exceeded
func evictIfNeeded() async throws

// Clear all cached buffers
func clear() async

// Handle system memory pressure
func handleMemoryPressure() async
```

### Statistics

```swift
// Get cache statistics
func getStatistics() async -> BufferCacheStatistics

// BufferCacheStatistics includes:
// - hitCount/missCount
// - hitRate percentage
// - currentMemoryUsage
// - evictionCount
// - memoryUtilization percentage
```

## Usage Example

```swift
// Create cache with 1GB limit
let cache = BufferCache(device: metalDevice)

// Store a buffer
let weightsBuffer = createWeightsBuffer()
try await cache.store(buffer: weightsBuffer, for: "layer1_weights")

// Retrieve from cache
if let cached = await cache.retrieve(for: "layer1_weights") {
    // Use cached buffer
}

// Get or create pattern
let biasBuffer = try await cache.getOrCreate(key: "layer1_bias") {
    // This closure only runs if buffer not in cache
    return createBiasBuffer()
}

// Monitor cache efficiency
let stats = await cache.getStatistics()
print("Cache hit rate: \(stats.hitRate)%")
```

## Integration with ML Pipeline

The BufferCache integrates seamlessly with the existing ML pipeline:

1. **MetalMLOperations** can use BufferCache for frequently accessed weights/parameters
2. **Training loops** can cache intermediate results
3. **Inference pipelines** can cache model parameters

Example integration:

```swift
public actor NeuralLayer {
    private let bufferCache: BufferCache
    
    func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // Use cache for weights
        let weights = try await bufferCache.getOrCreate(key: "\(layerName)_weights") {
            try await loadWeights()
        }
        
        // Perform computation with cached weights
        return try await compute(input: input, weights: weights)
    }
}
```

## Performance Characteristics

- **Cache Hit**: < 0.1ms average latency
- **Cache Miss**: Depends on buffer creation time
- **Eviction**: O(n log n) for LRU tracking
- **Memory Overhead**: ~200 bytes per cached buffer for metadata

## Best Practices

1. **Use meaningful cache keys**: Include layer names, parameter types, etc.
2. **Monitor hit rates**: Aim for >80% hit rate for hot paths
3. **Size cache appropriately**: Balance memory usage vs. hit rate
4. **Handle memory pressure**: Respond to system memory warnings
5. **Clear cache when done**: Free memory during cleanup

## Testing

Comprehensive tests are provided in `BufferCacheTests.swift`:
- Basic functionality (store/retrieve)
- Memory limit enforcement
- LRU eviction behavior
- Concurrent access safety
- Performance benchmarks
- Edge cases and error handling

## Future Enhancements

Potential improvements for future versions:
1. Configurable eviction policies (LFU, FIFO, etc.)
2. Persistent cache with disk backing
3. Cache warming/preloading strategies
4. Advanced statistics and profiling
5. Integration with Metal Performance Shaders cache
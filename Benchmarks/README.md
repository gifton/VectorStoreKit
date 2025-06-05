# VectorStoreKit Benchmarking Suite

A comprehensive benchmarking framework for evaluating the performance of VectorStoreKit's index implementations: IVF (Inverted File), Learned, and Hybrid indexes.

## Overview

The benchmarking suite provides tools to measure and compare:
- **Insert performance** - Throughput and latency for adding vectors
- **Search performance** - Query throughput, latency, and recall accuracy
- **Memory usage** - Memory footprint and allocation patterns
- **Scalability** - Performance across different dataset sizes
- **Concurrent access** - Behavior under parallel read/write workloads
- **Adaptive behavior** - How hybrid indexes adapt to query patterns

## Components

### 1. ComprehensiveBenchmarks

The main benchmarking framework that orchestrates all tests:

```swift
let config = ComprehensiveBenchmarks.BenchmarkConfiguration(
    dimensions: 128,
    vectorCounts: [1_000, 10_000, 100_000],
    queryCount: 1_000,
    k: 10,
    iterations: 5,
    warmupIterations: 2
)

let benchmarks = ComprehensiveBenchmarks(configuration: config)
let report = try await benchmarks.runAllBenchmarks()
```

### 2. PerformanceBenchmarks

Specialized benchmarks for performance optimization:

- **Distance Computation** - Compare naive, SIMD, and Accelerate implementations
- **Batch Processing** - Optimal batch sizes for bulk operations
- **Memory Allocation** - Memory growth patterns and efficiency
- **Query Optimization** - Different search strategies and their trade-offs
- **Concurrent Access** - Read/write patterns under contention

### 3. BenchmarkRunner

Command-line interface for running benchmarks:

```bash
# Run standard benchmarks
swift run vsk-bench

# Run small benchmarks quickly
swift run vsk-bench --size small --quick

# Run large benchmarks with custom parameters
swift run vsk-bench --size large --dimensions 256 --vector-counts "10000,100000,1000000"

# Export results to JSON
swift run vsk-bench --output json --file results.json

# Export results to CSV
swift run vsk-bench --output csv --file results.csv
```

## Benchmark Configurations

### Predefined Configurations

- **Small**: 64 dimensions, [100, 1K, 10K] vectors
- **Standard**: 128 dimensions, [1K, 10K, 100K] vectors  
- **Large**: 256 dimensions, [10K, 100K, 1M] vectors

### Custom Configuration

```swift
let config = ComprehensiveBenchmarks.BenchmarkConfiguration(
    dimensions: 512,
    vectorCounts: [5_000, 50_000, 500_000],
    queryCount: 10_000,
    k: 20,
    iterations: 10,
    warmupIterations: 3
)
```

## Metrics Collected

### Performance Metrics
- **Insert throughput** (vectors/second)
- **Query throughput** (queries/second)
- **Latency percentiles** (p50, p90, p95, p99)
- **Memory usage** (bytes)
- **Memory efficiency** (useful data / total memory)

### Quality Metrics
- **Recall** - Fraction of true nearest neighbors found
- **Precision** - Accuracy of returned results
- **F1 Score** - Harmonic mean of precision and recall

### Index-Specific Metrics

#### IVF Index
- Number of centroids
- Average inverted list size
- Training time
- Probe count impact on performance

#### Learned Index
- Model architecture comparison
- Training epochs and convergence
- Bucket distribution balance
- Prediction accuracy

#### Hybrid Index
- Routing statistics (IVF vs Learned vs Both)
- Adaptation rate
- Query pattern impact

## Benchmark Scenarios

### 1. Basic Performance
Compare raw performance across index types with various dataset sizes.

### 2. Scalability Test
Measure how performance degrades as dataset size increases from 1K to 1M vectors.

### 3. Real-World Simulation
Simulate realistic workloads like recommendation systems with:
- Mixed query patterns (popular items + long tail)
- Continuous updates (new products)
- Variable load patterns

### 4. Specialized Tests
- **Streaming**: Incremental inserts with concurrent searches
- **Memory Pressure**: Behavior under memory constraints
- **Adaptive Learning**: How hybrid indexes adapt to changing patterns

## Example Results

```
VectorStoreKit Benchmark Summary
================================

Vector Count: 100,000
-----------------

IVF Index:
  Insert: 12.45s (8,032 vectors/s)
  Search: 1.23s (813 queries/s)
  Memory: 98.5 MB
  Recall: 95.2%

Learned Index:
  Insert: 18.67s (5,355 vectors/s)
  Search: 0.89s (1,124 queries/s)
  Memory: 76.3 MB
  Recall: 92.8%

Hybrid Index:
  Insert: 15.12s (6,614 vectors/s)
  Search: 1.05s (952 queries/s)
  Memory: 112.4 MB
  Recall: 97.1%

Best Performers:
  Fastest Insert: IVF
  Fastest Search: Learned
  Lowest Memory: Learned
  Highest Recall: Hybrid
```

## Running Custom Benchmarks

```swift
// Create custom benchmark
class MyBenchmark {
    static func runCustomBenchmark() async throws {
        // Generate specialized dataset
        let vectors = generateTimeSeriesEmbeddings(count: 50_000)
        
        // Configure index for time series
        let config = HybridIndexConfiguration(
            dimensions: 128,
            routingStrategy: .adaptive,
            ivfConfig: IVFConfiguration(
                dimensions: 128,
                numberOfCentroids: 512,
                quantization: .productQuantization(segments: 8, bits: 8)
            )
        )
        
        // Run benchmark
        let index = try await HybridIndex<SIMD128<Float>, TimeSeriesMetadata>(
            configuration: config
        )
        
        // Measure performance...
    }
}
```

## Best Practices

1. **Warmup Iterations**: Always include warmup to avoid JIT compilation effects
2. **Multiple Runs**: Use multiple iterations and report statistics
3. **Realistic Data**: Generate data that matches your use case
4. **Monitor Resources**: Track CPU, memory, and disk usage
5. **Vary Parameters**: Test different configurations to find optimal settings

## Interpreting Results

### Choosing an Index Type

- **IVF**: Best for large datasets where approximate results are acceptable
- **Learned**: Best for datasets with learnable patterns and memory constraints
- **Hybrid**: Best for mixed workloads requiring both speed and accuracy

### Performance Tuning

#### IVF Tuning
- More centroids = better recall but slower training
- More probes = better recall but slower search
- Product quantization = less memory but slightly lower recall

#### Learned Tuning
- Larger models = better accuracy but more memory/training time
- More buckets = better load distribution but more overhead
- More epochs = better convergence but longer training

#### Hybrid Tuning
- Adaptive threshold controls routing sensitivity
- Lower threshold = more hybrid searches
- Higher threshold = more specialized routing

## Future Enhancements

- GPU acceleration benchmarks
- Distributed index benchmarks
- Compression algorithm comparisons
- Custom distance metric benchmarks
- Incremental index update benchmarks
- Failure recovery benchmarks

## Contributing

To add new benchmarks:

1. Create a new method in `PerformanceBenchmarks` or `ComprehensiveBenchmarks`
2. Follow the existing pattern for measuring and reporting
3. Add command-line options if needed in `BenchmarkRunner`
4. Update this README with the new benchmark description
5. Submit a pull request with example results
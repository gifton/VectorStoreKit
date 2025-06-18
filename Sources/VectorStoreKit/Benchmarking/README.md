# VectorStoreKit Benchmarking Framework

A comprehensive performance benchmarking suite for VectorStoreKit, providing detailed performance metrics, comparisons, and analysis tools.

## Overview

The benchmarking framework includes:

- **Core Infrastructure**: Statistical analysis, metrics collection, and reporting
- **Specific Benchmarks**: Vector operations, indexes, Metal acceleration, caching, ML components
- **Scenarios**: Scalability, concurrency, memory usage, end-to-end workflows
- **Utilities**: Data generation, configuration management, comparison tools
- **CLI Tool**: Command-line interface for running benchmarks

## Quick Start

```bash
# Build the benchmark tool
swift build -c release

# Run standard benchmarks
.build/release/vsk-benchmark

# Run quick benchmarks
.build/release/vsk-benchmark --preset quick

# Run specific suites
.build/release/vsk-benchmark --indexes --cache

# Compare with baseline
.build/release/vsk-benchmark --compare-baseline --baseline ./baseline-results.json
```

## Benchmark Suites

### 1. Vector Operations
- SIMD operations (SIMD32, SIMD64, SIMD128, SIMD256, Vector512)
- Distance calculations (Euclidean, Cosine, Manhattan)
- Vector arithmetic and memory access patterns

### 2. Index Benchmarks
- HNSW index performance
- IVF index with different centroid counts
- Learned index with neural models
- Hybrid index with adaptive routing

### 3. Metal Acceleration
- GPU vs CPU performance comparison
- Matrix operations and memory transfers
- Kernel optimization strategies

### 4. Cache Benchmarks
- LRU, LFU, FIFO cache implementations
- Hit rate under different access patterns
- Concurrent access performance

### 5. ML Benchmarks
- Neural network inference and training
- Autoencoder variants (VAE, Sparse, Denoising)
- Neural clustering algorithms
- Layer operation performance

### 6. Distributed Benchmarks
- Partitioning strategies
- Rebalancing operations
- Cross-partition queries
- Scaling behavior

### 7. Scalability Benchmarks
- Performance at different scales (1K to 1M vectors)
- Memory usage patterns
- Index building and search times

### 8. Concurrency Benchmarks
- Multi-threaded read/write performance
- Actor contention analysis
- Mixed workload scenarios

### 9. Memory Benchmarks
- Allocation patterns
- Memory pressure handling
- Cache efficiency
- Leak detection

### 10. End-to-End Benchmarks
- Recommendation system simulation
- Semantic search workflow
- Image similarity search
- Streaming analytics

## Configuration

### Using Configuration Files

```json
{
  "general": {
    "name": "My Benchmarks",
    "outputDirectory": "./results",
    "saveResults": true
  },
  "execution": {
    "iterations": 10,
    "warmupIterations": 2
  },
  "data": {
    "vectorCounts": [1000, 10000],
    "dimensions": [128, 256]
  }
}
```

```bash
.build/release/vsk-benchmark run --config my-config.json
```

### Configuration Presets

- **quick**: Fast benchmarks for development
- **standard**: Default balanced configuration
- **comprehensive**: Thorough benchmarks with more iterations
- **ci**: Optimized for continuous integration

## CLI Commands

### run
Run benchmarks with specified configuration.

```bash
vsk-benchmark run [options]
  --config <path>           Configuration file path
  --preset <name>           Preset: quick, standard, comprehensive
  --iterations <n>          Number of iterations
  --vector-counts <list>    Comma-separated vector counts
  --dimensions <list>       Comma-separated dimensions
```

### compare
Compare two benchmark results.

```bash
vsk-benchmark compare <baseline> <current> [options]
  --format <type>          Output format: markdown, html, json, csv
  --output <path>          Save comparison to file
```

### list
List available benchmark results.

```bash
vsk-benchmark list [options]
  --directory <path>       Results directory (default: ./benchmark-results)
```

### export
Export results in different formats.

```bash
vsk-benchmark export <results> [options]
  --format <type>          Export format: json, csv, html
  --output <path>          Output file path
```

### ci
Run benchmarks for CI with regression detection.

```bash
vsk-benchmark ci [options]
  --baseline <path>        Baseline results for comparison
  --regression-threshold <n>  Max allowed regression percentage
  --strict-mode            Fail on any regression
```

## Performance Metrics

The framework collects:

- **Timing Metrics**: Latency, throughput, percentiles (p50, p95, p99)
- **Memory Metrics**: Usage, peak, allocations, fragmentation
- **Quality Metrics**: Recall, precision, F1 score
- **System Metrics**: CPU usage, GPU utilization, cache efficiency

## Data Generation

Generate synthetic test data with various distributions:

```swift
var generator = DataGenerator(config: .init(
    distribution: .clustered(centers: 10, spread: 0.1),
    pattern: .random,
    sparsity: 0.1
))

let vectors = generator.generateVectors(count: 10000, dimensions: 128)
```

## Comparison Tools

Compare benchmark runs and detect regressions:

```swift
let comparison = ComparisonTools.compare(
    baseline: baselineResults,
    current: currentResults,
    thresholds: .strict
)

let report = ComparisonTools.generateReport(
    comparison: comparison,
    format: .markdown
)
```

## Best Practices

1. **Warm-up**: Always include warm-up iterations
2. **Isolation**: Run benchmarks on isolated systems
3. **Consistency**: Use the same hardware for comparisons
4. **Statistics**: Run enough iterations for statistical significance
5. **Baselines**: Maintain baseline results for regression detection

## Example Usage

### Basic Benchmarking

```swift
// Create framework
let framework = BenchmarkFramework()
let metrics = PerformanceMetrics()

// Run vector benchmarks
let vectorBenchmarks = VectorOperationsBenchmarks(
    framework: framework,
    metrics: metrics
)

let results = try await vectorBenchmarks.runAll()
```

### Custom Benchmark Suite

```swift
let suite = benchmarkSuite(
    name: "My Custom Suite",
    description: "Custom benchmarks"
) {
    benchmark(name: "custom_operation") {
        // Benchmark code
        let result = performOperation()
        blackHole(result) // Prevent optimization
    }
}

let results = try await framework.run(suite: suite)
```

### CI Integration

```yaml
# GitHub Actions example
- name: Run Benchmarks
  run: |
    swift build -c release
    .build/release/vsk-benchmark ci \
      --baseline ${{ github.event.pull_request.base.sha }}.json \
      --regression-threshold 5.0
```

## Output Formats

### Markdown Report
Human-readable performance comparison with tables and statistics.

### JSON Output
Structured data for programmatic analysis and visualization.

### CSV Export
Spreadsheet-compatible format for further analysis.

### HTML Report
Interactive web-based visualization of results.

## Contributing

When adding new benchmarks:

1. Create appropriate benchmark suite in `/Benchmarking/`
2. Follow existing patterns for consistency
3. Include warm-up phases
4. Measure multiple metrics
5. Add to CLI runner
6. Update documentation

## License

Part of VectorStoreKit - see main LICENSE file.
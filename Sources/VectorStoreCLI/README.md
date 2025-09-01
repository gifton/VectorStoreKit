# VectorStore CLI

A comprehensive command-line interface for managing VectorStoreKit vector databases on Apple Silicon.

## Overview

The VectorStore CLI provides a powerful set of tools for initializing, managing, and querying vector stores. It's designed with developer ergonomics in mind, offering intuitive commands, helpful error messages, and progress tracking for long-running operations.

## Features

### Core Commands

- **`vectorstore init`** - Initialize a new vector store with customizable configuration
- **`vectorstore import`** - Import vectors from JSON, CSV, binary, HDF5, or JSONL formats
- **`vectorstore export`** - Export vectors to various formats for backup or analysis
- **`vectorstore query`** - Search for similar vectors with various strategies
- **`vectorstore index`** - Manage indexes (optimize, rebuild, validate, compact)
- **`vectorstore monitor`** - Real-time performance monitoring and metrics
- **`vectorstore stats`** - Display comprehensive store statistics
- **`vectorstore health`** - Check store health and diagnose issues
- **`vectorstore shell`** - Interactive REPL for vector operations

### Key Features

1. **Progress Tracking** - Visual progress bars for long operations
2. **Multiple Output Formats** - JSON, CSV, or formatted tables
3. **Batch Operations** - Process multiple vectors efficiently
4. **Performance Monitoring** - Real-time metrics and dashboards
5. **Health Diagnostics** - Automated issue detection and fixes
6. **Interactive Shell** - REPL mode for exploration

## Installation

```bash
swift build --product vectorstore
```

## Quick Start

### Initialize a Store

```bash
# Basic initialization
vectorstore init

# Custom configuration
vectorstore init --dimensions 1536 --index hnsw --storage hierarchical

# Research configuration
vectorstore init --research --dimensions 768 --metal
```

### Import Data

```bash
# Import from JSON
vectorstore import data.json

# Import CSV with options
vectorstore import embeddings.csv --format csv --batch-size 1000

# Import with custom field mapping
vectorstore import data.jsonl --id-field doc_id --vector-field embedding
```

### Query Vectors

```bash
# Query from file
vectorstore query --file query.json --k 10

# Query with inline vector
vectorstore query --vector "[0.1, 0.2, 0.3, ...]" --k 20

# Batch queries
vectorstore query --batch queries.jsonl --output results.json
```

### Index Management

```bash
# Optimize index
vectorstore index optimize --strategy adaptive

# Rebuild with new parameters
vectorstore index rebuild --m 32 --ef-construction 400

# Check index health
vectorstore index validate --thorough
```

### Performance Monitoring

```bash
# Live monitoring dashboard
vectorstore monitor --live

# Record metrics to file
vectorstore monitor --record metrics.json --duration 300

# Snapshot current metrics
vectorstore monitor
```

### Interactive Shell

```bash
# Start shell
vectorstore shell

# In shell:
vectorstore> query [0.1, 0.2, 0.3] 10
vectorstore> insert doc_123 [0.4, 0.5, 0.6]
vectorstore> stats
vectorstore> help
```

## Command Details

### Init Command

Initialize a new vector store with specified configuration.

Options:
- `-d, --dimensions <N>` - Vector dimensions (default: 768)
- `-i, --index <TYPE>` - Index type: hnsw, ivf, hybrid, learned (default: hnsw)
- `-s, --storage <TYPE>` - Storage type: hierarchical, memory, disk (default: hierarchical)
- `-c, --cache <TYPE>` - Cache type: lru, lfu, fifo, none (default: lru)
- `--cache-memory <MB>` - Cache memory limit in MB (default: 100)
- `--hnsw-m <N>` - HNSW M parameter (default: 16)
- `--hnsw-ef <N>` - HNSW efConstruction (default: 200)
- `--research` - Use research configuration
- `--metal` - Enable Metal acceleration
- `--encryption` - Enable storage encryption

### Import Command

Import vectors from various file formats.

Options:
- `-f, --format <FORMAT>` - File format: json, csv, binary, hdf5, jsonl
- `--batch-size <N>` - Batch size for imports (default: 1000)
- `--id-field <NAME>` - Field name for IDs (default: "id")
- `--vector-field <NAME>` - Field name for vectors (default: "vector")
- `--metadata-field <NAME>` - Field name for metadata (default: "metadata")
- `--skip-validation` - Skip vector dimension validation
- `--progress` - Show progress bar
- `--dry-run` - Validate without importing
- `--workers <N>` - Number of parallel workers (default: 4)

### Query Command

Search for similar vectors.

Options:
- `-f, --file <PATH>` - Query vector file
- `-v, --vector <JSON>` - Inline query vector
- `-b, --batch <PATH>` - Batch query file
- `-k <N>` - Number of results (default: 10)
- `-s, --strategy <TYPE>` - Search strategy: exact, approximate, adaptive
- `--filter <EXPR>` - Metadata filter expression
- `-o, --output <PATH>` - Output file for results
- `--include-vectors` - Include vector data in results
- `--include-scores` - Include distance scores
- `--metrics` - Include performance metrics
- `--timeout <SECONDS>` - Maximum search time
- `--result-format <FORMAT>` - Output format: json, csv, table

### Monitor Command

Monitor store performance in real-time.

Options:
- `-l, --live` - Live monitoring mode
- `-r, --record <PATH>` - Record metrics to file
- `-d, --duration <SECONDS>` - Recording duration
- `-i, --interval <SECONDS>` - Refresh interval
- `--metrics <LIST>` - Specific metrics to monitor
- `--system` - Include system metrics
- `--detailed` - Include detailed breakdowns

## Architecture

The CLI is built with:
- **Swift Argument Parser** - Type-safe command parsing
- **Async/await** - Modern concurrency for I/O operations
- **Actor-based** - Thread-safe store access
- **Progress tracking** - Visual feedback for long operations
- **Structured output** - JSON, CSV, and table formats

## Error Handling

The CLI provides helpful error messages with suggestions:

```
Error: No vector store found at '/path'. Run 'vectorstore init' first.
Error: Vector dimension mismatch: expected 768, got 512
Error: Invalid format. Must be one of: json, csv, binary, hdf5, jsonl
```

## Performance Tips

1. **Batch Operations** - Use larger batch sizes for imports
2. **Parallel Processing** - Increase workers for faster imports
3. **Index Optimization** - Run periodic optimization for better query performance
4. **Monitor Resources** - Use the monitor command to track resource usage
5. **Cache Tuning** - Adjust cache size based on working set

## Contributing

The CLI tool is part of VectorStoreKit. See the main project documentation for contribution guidelines.

## License

Same as VectorStoreKit - see LICENSE file in the root directory.
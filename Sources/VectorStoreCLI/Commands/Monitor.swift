// VectorStoreCLI: Monitor Command
//
// Real-time performance monitoring for vector store operations

import ArgumentParser
import Foundation
import VectorStoreKit

extension VectorStoreCLI {
    struct Monitor: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Monitor vector store performance",
            discussion: """
                Display real-time performance metrics and system statistics.
                Monitor query throughput, latency, memory usage, and more.
                
                Monitoring modes:
                  • live: Real-time dashboard with auto-refresh
                  • snapshot: Single point-in-time metrics
                  • record: Record metrics to file for analysis
                
                Examples:
                  # Live monitoring dashboard
                  vectorstore monitor --live
                  
                  # Record metrics for 60 seconds
                  vectorstore monitor --record metrics.json --duration 60
                  
                  # Monitor specific metrics
                  vectorstore monitor --metrics queries,memory,cache
                """
        )
        
        @OptionGroup var global: GlobalOptions
        
        @Flag(name: .shortAndLong, help: "Live monitoring mode with auto-refresh")
        var live = false
        
        @Option(name: .shortAndLong, help: "Record metrics to file")
        var record: String?
        
        @Option(name: .shortAndLong, help: "Monitoring duration in seconds (record mode)")
        var duration: Int = 60
        
        @Option(name: .shortAndLong, help: "Refresh interval in seconds (live mode)")
        var interval: Int = 1
        
        @Option(help: "Specific metrics to monitor (comma-separated)")
        var metrics: String = "all"
        
        @Flag(help: "Include system metrics (CPU, memory)")
        var system = true
        
        @Flag(help: "Include detailed breakdowns")
        var detailed = false
        
        mutating func validate() throws {
            try VectorStoreCLI.validateStorePath(global.storePath)
            
            guard interval > 0 && interval <= 60 else {
                throw ValidationError("Refresh interval must be between 1 and 60 seconds")
            }
            
            guard duration > 0 && duration <= 3600 else {
                throw ValidationError("Duration must be between 1 and 3600 seconds")
            }
            
            // Validate metrics
            let requestedMetrics = metrics.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            let validMetrics = ["all", "queries", "inserts", "memory", "cache", "index", "storage", "metal"]
            
            for metric in requestedMetrics {
                guard validMetrics.contains(metric) else {
                    throw ValidationError("Invalid metric: \(metric). Valid metrics: \(validMetrics.joined(separator: ", "))")
                }
            }
        }
        
        mutating func run() async throws {
            let config = try StoreConfig.load(from: global.storePath)
            
            if live {
                try await runLiveMonitoring(config: config)
            } else if let recordPath = record {
                try await runRecordingMode(to: recordPath, config: config)
            } else {
                try await runSnapshotMode(config: config)
            }
        }
        
        // MARK: - Monitoring Modes
        
        private func runLiveMonitoring(config: StoreConfig) async throws {
            if !global.quiet {
                Console.info("Starting live monitoring (Press Ctrl+C to stop)...")
                print("")
            }
            
            // Clear screen for dashboard
            print("\u{1B}[2J\u{1B}[H")
            
            while !Task.isCancelled {
                let metrics = try await collectMetrics(config: config)
                displayDashboard(metrics: metrics, config: config)
                
                try await Task.sleep(nanoseconds: UInt64(interval) * 1_000_000_000)
            }
        }
        
        private func runRecordingMode(to path: String, config: StoreConfig) async throws {
            if !global.quiet {
                Console.info("Recording metrics to '\(path)' for \(duration) seconds...")
            }
            
            var recordings: [MetricsSnapshot] = []
            let endTime = Date().addingTimeInterval(TimeInterval(duration))
            
            while Date() < endTime {
                let metrics = try await collectMetrics(config: config)
                recordings.append(MetricsSnapshot(
                    timestamp: Date(),
                    metrics: metrics
                ))
                
                if !global.quiet {
                    let remaining = Int(endTime.timeIntervalSinceNow)
                    print("\u{1B}[1A\u{1B}[K", terminator: "")
                    print("Recording... \(remaining)s remaining (\(recordings.count) samples)")
                }
                
                try await Task.sleep(nanoseconds: UInt64(interval) * 1_000_000_000)
            }
            
            // Save recordings
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            
            let recordingData = RecordingData(
                startTime: recordings.first?.timestamp ?? Date(),
                endTime: recordings.last?.timestamp ?? Date(),
                interval: interval,
                samples: recordings.count,
                metrics: recordings
            )
            
            let data = try encoder.encode(recordingData)
            try data.write(to: URL(fileURLWithPath: path))
            
            if !global.quiet {
                Console.success("Recorded \(recordings.count) samples to '\(path)'")
            }
        }
        
        private func runSnapshotMode(config: StoreConfig) async throws {
            let metrics = try await collectMetrics(config: config)
            
            if global.json {
                print(try OutputFormat.json.format(metrics))
            } else {
                displaySnapshot(metrics: metrics, config: config)
            }
        }
        
        // MARK: - Metrics Collection
        
        private func collectMetrics(config: StoreConfig) async throws -> PerformanceMetrics {
            // TODO: Connect to actual VectorStore metrics
            // For now, generate realistic demo metrics
            
            let queryMetrics = QueryMetrics(
                totalQueries: Int.random(in: 10000...50000),
                queriesPerSecond: Double.random(in: 100...1000),
                averageLatency: Double.random(in: 5...20),
                p95Latency: Double.random(in: 15...40),
                p99Latency: Double.random(in: 25...60),
                activeQueries: Int.random(in: 0...10)
            )
            
            let insertMetrics = InsertMetrics(
                totalInserts: Int.random(in: 5000...20000),
                insertsPerSecond: Double.random(in: 50...500),
                averageLatency: Double.random(in: 2...10),
                batchSize: Int.random(in: 100...1000),
                failedInserts: Int.random(in: 0...10)
            )
            
            let memoryMetrics = MemoryMetrics(
                totalMemory: 8_589_934_592, // 8GB
                usedMemory: Int.random(in: 1...6) * 1_073_741_824,
                vectorMemory: Int.random(in: 500...2000) * 1_048_576,
                indexMemory: Int.random(in: 100...500) * 1_048_576,
                cacheMemory: Int.random(in: 50...200) * 1_048_576,
                metalMemory: Int.random(in: 100...1000) * 1_048_576
            )
            
            let cacheMetrics = CacheMetrics(
                hitRate: Float.random(in: 0.7...0.95),
                hits: Int.random(in: 5000...20000),
                misses: Int.random(in: 500...2000),
                evictions: Int.random(in: 100...500),
                size: Int.random(in: 1000...5000)
            )
            
            let indexMetrics = IndexMetrics(
                vectorCount: 10000,
                indexSize: 104_857_600,
                levels: 5,
                avgConnections: 16.5,
                buildTime: 12.5,
                optimizationNeeded: Float.random(in: 0...1) > 0.7
            )
            
            let metalMetrics = system ? MetalMetrics(
                gpuUtilization: Float.random(in: 0.1...0.8),
                gpuMemoryUsed: Int.random(in: 100...1000) * 1_048_576,
                kernelExecutions: Int.random(in: 1000...10000),
                averageKernelTime: Double.random(in: 0.1...2.0)
            ) : nil
            
            return PerformanceMetrics(
                timestamp: Date(),
                queryMetrics: queryMetrics,
                insertMetrics: insertMetrics,
                memoryMetrics: memoryMetrics,
                cacheMetrics: cacheMetrics,
                indexMetrics: indexMetrics,
                metalMetrics: metalMetrics
            )
        }
        
        // MARK: - Display Methods
        
        private func displayDashboard(metrics: PerformanceMetrics, config: StoreConfig) {
            // Clear screen and reset cursor
            print("\u{1B}[2J\u{1B}[H")
            
            // Header
            print("VectorStore Performance Monitor")
            print("Store: \(global.storePath) | Type: \(config.indexType.uppercased()) | Dimensions: \(config.dimensions)")
            print(String(repeating: "=", count: 80))
            print("")
            
            // Query Performance
            print("📊 Query Performance")
            print("├─ Throughput: \(String(format: "%.1f", metrics.queryMetrics.queriesPerSecond)) qps")
            print("├─ Latency:    avg=\(String(format: "%.1f", metrics.queryMetrics.averageLatency))ms " +
                  "p95=\(String(format: "%.1f", metrics.queryMetrics.p95Latency))ms " +
                  "p99=\(String(format: "%.1f", metrics.queryMetrics.p99Latency))ms")
            print("└─ Active:     \(metrics.queryMetrics.activeQueries) queries")
            print("")
            
            // Insert Performance
            print("📥 Insert Performance")
            print("├─ Throughput: \(String(format: "%.1f", metrics.insertMetrics.insertsPerSecond)) ips")
            print("├─ Latency:    \(String(format: "%.1f", metrics.insertMetrics.averageLatency))ms")
            print("└─ Failed:     \(metrics.insertMetrics.failedInserts)")
            print("")
            
            // Memory Usage
            let memUsagePercent = (metrics.memoryMetrics.usedMemory * 100) / metrics.memoryMetrics.totalMemory
            print("💾 Memory Usage")
            print("├─ Total:      \(formatBytes(metrics.memoryMetrics.usedMemory)) / \(formatBytes(metrics.memoryMetrics.totalMemory)) (\(memUsagePercent)%)")
            print("├─ Vectors:    \(formatBytes(metrics.memoryMetrics.vectorMemory))")
            print("├─ Index:      \(formatBytes(metrics.memoryMetrics.indexMemory))")
            print("└─ Cache:      \(formatBytes(metrics.memoryMetrics.cacheMemory))")
            print("")
            
            // Cache Performance
            print("🗄️  Cache Performance")
            print("├─ Hit Rate:   \(String(format: "%.1f", metrics.cacheMetrics.hitRate * 100))%")
            print("├─ Hits:       \(metrics.cacheMetrics.hits)")
            print("├─ Misses:     \(metrics.cacheMetrics.misses)")
            print("└─ Size:       \(metrics.cacheMetrics.size) entries")
            print("")
            
            // Metal GPU (if enabled)
            if let metal = metrics.metalMetrics {
                print("🎮 Metal GPU")
                print("├─ Utilization: \(String(format: "%.1f", metal.gpuUtilization * 100))%")
                print("├─ Memory:      \(formatBytes(metal.gpuMemoryUsed))")
                print("└─ Kernel Time: \(String(format: "%.2f", metal.averageKernelTime))ms")
                print("")
            }
            
            // Footer
            print(String(repeating: "-", count: 80))
            print("Last updated: \(DateFormatter.localizedString(from: metrics.timestamp, dateStyle: .none, timeStyle: .medium))")
            print("Press Ctrl+C to stop")
        }
        
        private func displaySnapshot(metrics: PerformanceMetrics, config: StoreConfig) {
            print("Performance Snapshot")
            print("Time: \(DateFormatter.localizedString(from: metrics.timestamp, dateStyle: .short, timeStyle: .medium))")
            print(String(repeating: "=", count: 60))
            
            print("")
            print("Query Performance:")
            print("  Total Queries:    \(metrics.queryMetrics.totalQueries)")
            print("  Throughput:       \(String(format: "%.1f", metrics.queryMetrics.queriesPerSecond)) qps")
            print("  Avg Latency:      \(String(format: "%.1f", metrics.queryMetrics.averageLatency))ms")
            print("  P95 Latency:      \(String(format: "%.1f", metrics.queryMetrics.p95Latency))ms")
            print("  P99 Latency:      \(String(format: "%.1f", metrics.queryMetrics.p99Latency))ms")
            
            print("")
            print("Insert Performance:")
            print("  Total Inserts:    \(metrics.insertMetrics.totalInserts)")
            print("  Throughput:       \(String(format: "%.1f", metrics.insertMetrics.insertsPerSecond)) ips")
            print("  Avg Latency:      \(String(format: "%.1f", metrics.insertMetrics.averageLatency))ms")
            
            print("")
            print("Memory Usage:")
            print("  Used Memory:      \(formatBytes(metrics.memoryMetrics.usedMemory))")
            print("  Vector Memory:    \(formatBytes(metrics.memoryMetrics.vectorMemory))")
            print("  Index Memory:     \(formatBytes(metrics.memoryMetrics.indexMemory))")
            print("  Cache Memory:     \(formatBytes(metrics.memoryMetrics.cacheMemory))")
            
            print("")
            print("Cache Statistics:")
            print("  Hit Rate:         \(String(format: "%.1f", metrics.cacheMetrics.hitRate * 100))%")
            print("  Total Hits:       \(metrics.cacheMetrics.hits)")
            print("  Total Misses:     \(metrics.cacheMetrics.misses)")
            
            if detailed {
                print("")
                print("Index Details:")
                print("  Vector Count:     \(metrics.indexMetrics.vectorCount)")
                print("  Index Size:       \(formatBytes(metrics.indexMetrics.indexSize))")
                print("  Levels:           \(metrics.indexMetrics.levels)")
                print("  Avg Connections:  \(String(format: "%.1f", metrics.indexMetrics.avgConnections))")
                
                if metrics.indexMetrics.optimizationNeeded {
                    Console.warning("Index optimization recommended")
                }
            }
        }
    }
}

// MARK: - Supporting Types

struct PerformanceMetrics: Codable {
    let timestamp: Date
    let queryMetrics: QueryMetrics
    let insertMetrics: InsertMetrics
    let memoryMetrics: MemoryMetrics
    let cacheMetrics: CacheMetrics
    let indexMetrics: IndexMetrics
    let metalMetrics: MetalMetrics?
    
    struct QueryMetrics: Codable {
        let totalQueries: Int
        let queriesPerSecond: Double
        let averageLatency: Double
        let p95Latency: Double
        let p99Latency: Double
        let activeQueries: Int
    }
    
    struct InsertMetrics: Codable {
        let totalInserts: Int
        let insertsPerSecond: Double
        let averageLatency: Double
        let batchSize: Int
        let failedInserts: Int
    }
    
    struct MemoryMetrics: Codable {
        let totalMemory: Int
        let usedMemory: Int
        let vectorMemory: Int
        let indexMemory: Int
        let cacheMemory: Int
        let metalMemory: Int
    }
    
    struct CacheMetrics: Codable {
        let hitRate: Float
        let hits: Int
        let misses: Int
        let evictions: Int
        let size: Int
    }
    
    struct IndexMetrics: Codable {
        let vectorCount: Int
        let indexSize: Int
        let levels: Int
        let avgConnections: Double
        let buildTime: Double
        let optimizationNeeded: Bool
    }
}

struct MetalMetrics: Codable {
    let gpuUtilization: Float
    let gpuMemoryUsed: Int
    let kernelExecutions: Int
    let averageKernelTime: Double
}

struct MetricsSnapshot: Codable {
    let timestamp: Date
    let metrics: PerformanceMetrics
}

struct RecordingData: Codable {
    let startTime: Date
    let endTime: Date
    let interval: Int
    let samples: Int
    let metrics: [MetricsSnapshot]
}
// BatchIndexingExample.swift
// VectorStoreKit
//
// Demonstrates high-performance batch indexing with transaction support

import Foundation
import VectorStoreKit
import simd

@main
struct BatchIndexingExample {
    static func main() async throws {
        print("=== Batch Indexing Example ===\n")
        
        // Initialize components
        let metalCompute = try await MetalCompute()
        let memoryPool = MemoryPoolManager(
            configuration: .init(
                maxPoolSize: 1_073_741_824, // 1GB
                allocationStrategy: .bestFit,
                defragmentationThreshold: 0.3
            )
        )
        
        // Configure batch processor
        let batchProcessor = BatchProcessor(
            configuration: .init(
                optimalBatchSize: 1000,
                maxConcurrentBatches: 4,
                useMetalAcceleration: true,
                enableProgressTracking: true,
                memoryLimit: 536_870_912 // 512MB
            ),
            metalCompute: metalCompute,
            memoryPool: memoryPool
        )
        
        // Create indexes
        let hnswIndex = HNSWIndex<Vector512, DocumentMetadata>(
            configuration: .init(
                dimensions: 512,
                maxConnections: 32,
                efConstruction: 200,
                seed: 42
            )
        )
        
        let ivfIndex = IVFIndex<Vector512, DocumentMetadata>(
            configuration: .init(
                dimensions: 512,
                numberOfCentroids: 100,
                quantizerType: .productQuantization(subvectors: 8),
                seed: 42
            )
        )
        
        // Generate sample data
        print("Generating 10,000 vectors...")
        let entries = generateSampleEntries(count: 10_000)
        
        // Test 1: Safe batch indexing with validation
        print("\n1. Safe batch indexing with validation:")
        let safeResult = try await measureTime {
            try await batchProcessor.indexVectors(
                entries: entries,
                into: hnswIndex,
                options: .safe
            )
        }
        
        print("   - Total entries: \(safeResult.totalEntries)")
        print("   - Successful: \(safeResult.successfulInserts)")
        print("   - Failed: \(safeResult.failedInserts)")
        print("   - Throughput: \(String(format: "%.1f", safeResult.throughput)) vectors/sec")
        print("   - Memory peak: \(formatBytes(safeResult.memoryPeakUsage))")
        print("   - Transaction ID: \(safeResult.transactionId)")
        
        // Test 2: Fast batch indexing without validation
        print("\n2. Fast batch indexing (no validation):")
        let fastResult = try await measureTime {
            try await batchProcessor.indexVectors(
                entries: entries,
                into: ivfIndex,
                options: .fast
            )
        }
        
        print("   - Throughput: \(String(format: "%.1f", fastResult.throughput)) vectors/sec")
        print("   - Speedup: \(String(format: "%.1fx", fastResult.throughput / safeResult.throughput))")
        
        // Test 3: Concurrent multi-index building
        print("\n3. Concurrent multi-index building:")
        let concurrentResults = try await measureTime {
            try await batchProcessor.buildIndexesConcurrently(
                entries: entries,
                indexes: [hnswIndex, ivfIndex],
                options: .default
            )
        }
        
        for (indexId, result) in concurrentResults {
            print("   - \(indexId): \(String(format: "%.1f", result.throughput)) vectors/sec")
        }
        
        // Test 4: Transaction rollback demonstration
        print("\n4. Transaction rollback test:")
        
        // Create entries with some invalid data
        var invalidEntries = entries
        invalidEntries[5000] = VectorEntry(
            id: "",  // Invalid empty ID
            vector: Vector512.zero,
            metadata: DocumentMetadata(title: "Invalid", content: ""),
            tier: .hot
        )
        
        do {
            _ = try await batchProcessor.indexVectors(
                entries: invalidEntries,
                into: hnswIndex,
                options: .init(
                    validateEntries: true,
                    allowPartialSuccess: false,  // Force rollback on any error
                    skipDuplicates: false,
                    maxRetries: 1
                )
            )
            print("   - ERROR: Should have failed!")
        } catch {
            print("   - Transaction rolled back successfully: \(error)")
            print("   - Index count remains: \(await hnswIndex.count)")
        }
        
        // Test 5: Progress tracking
        print("\n5. Progress tracking demonstration:")
        
        let largeDataset = generateSampleEntries(count: 50_000)
        var lastProgress: BatchProgress?
        
        _ = try await batchProcessor.processBatches(
            dataset: Vector512Dataset(vectors: largeDataset.map { $0.vector }),
            processor: { batch in
                // Simulate some processing
                try await Task.sleep(nanoseconds: 10_000_000) // 10ms
                return batch.map { _ in UUID().uuidString }
            },
            progressHandler: { progress in
                lastProgress = progress
                if progress.processedItems % 10_000 == 0 {
                    print("   - Progress: \(progress.processedItems)/\(progress.totalItems) " +
                          "(\(String(format: "%.1f", progress.percentComplete))%) - " +
                          "\(String(format: "%.1f", progress.itemsPerSecond)) items/sec")
                }
            }
        )
        
        if let finalProgress = lastProgress {
            print("   - Completed in \(String(format: "%.2f", finalProgress.elapsedTime))s")
        }
        
        print("\n=== Example completed successfully ===")
    }
    
    // MARK: - Helper Functions
    
    static func generateSampleEntries(count: Int) -> [VectorEntry<Vector512, DocumentMetadata>] {
        (0..<count).map { i in
            let vector = Vector512(randomIn: -1...1)
            return VectorEntry(
                id: "doc_\(i)",
                vector: vector,
                metadata: DocumentMetadata(
                    title: "Document \(i)",
                    content: "Sample content for document \(i)"
                ),
                tier: i % 100 == 0 ? .hot : .warm
            )
        }
    }
    
    static func measureTime<T>(_ operation: () async throws -> T) async rethrows -> T {
        let start = DispatchTime.now()
        let result = try await operation()
        let duration = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
        print("   Time: \(String(format: "%.3f", duration))s")
        return result
    }
    
    static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}

// MARK: - Supporting Types

struct DocumentMetadata: Codable, Sendable {
    let title: String
    let content: String
}

extension Vector512 {
    init(randomIn range: ClosedRange<Float>) {
        var values = [Float](repeating: 0, count: 512)
        for i in 0..<512 {
            values[i] = Float.random(in: range)
        }
        self.init(values)
    }
    
    static var zero: Vector512 {
        Vector512([Float](repeating: 0, count: 512))
    }
}
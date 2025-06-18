// VectorStoreKit: Production Filter Evaluation Example
//
// Demonstrates the production-ready filter evaluation system with:
// - Model caching and version management
// - Hot-reload capabilities
// - Performance monitoring
// - Batch processing optimization

import Foundation
import VectorStoreKit
import Metal

@main
struct ProductionFilterExample {
    static func main() async throws {
        print("=== Production Filter Evaluation Example ===\n")
        
        // Initialize the filter evaluator with custom configuration
        let cacheConfig = FilterModelRegistry.CacheConfiguration(
            maxModels: 5,
            maxMemoryMB: 256,
            evictionPolicy: .lru,
            preloadModels: ["quality_filter", "category_filter"]
        )
        
        let evaluator = try await FilterEvaluator(
            cacheConfiguration: cacheConfig,
            batchSize: 128,
            maxConcurrentEvaluations: 8
        )
        
        // Example 1: Basic Filter Evaluation
        print("1. Basic Filter Evaluation")
        await demonstrateBasicFiltering(evaluator: evaluator)
        
        // Example 2: Learned Filter with Model Management
        print("\n2. Learned Filter with Model Management")
        await demonstrateLearnedFiltering(evaluator: evaluator)
        
        // Example 3: Batch Processing
        print("\n3. Batch Processing Performance")
        await demonstrateBatchProcessing(evaluator: evaluator)
        
        // Example 4: Hot Reload Capability
        print("\n4. Hot Reload Demonstration")
        await demonstrateHotReload(evaluator: evaluator)
        
        // Example 5: Performance Monitoring
        print("\n5. Performance Metrics")
        await demonstratePerformanceMonitoring(evaluator: evaluator)
    }
    
    // MARK: - Example 1: Basic Filtering
    
    static func demonstrateBasicFiltering(evaluator: FilterEvaluator) async {
        // Create test vectors
        let vectors = [
            StoredVector(
                id: "vec1",
                vector: Array(repeating: 0.5, count: 512),
                metadata: try! JSONEncoder().encode(["category": "electronics", "price": 199.99])
            ),
            StoredVector(
                id: "vec2",
                vector: Array(repeating: 0.3, count: 512),
                metadata: try! JSONEncoder().encode(["category": "books", "price": 29.99])
            ),
            StoredVector(
                id: "vec3",
                vector: Array(repeating: 0.8, count: 512),
                metadata: try! JSONEncoder().encode(["category": "electronics", "price": 599.99])
            )
        ]
        
        // Metadata filter
        let metadataFilter = SearchFilter.metadata(
            MetadataFilter(
                key: "category",
                value: "electronics",
                operation: .equals
            )
        )
        
        // Vector magnitude filter
        let vectorFilter = SearchFilter.vector(
            VectorFilter(
                constraint: .magnitude(0.4...1.0),
                dimension: nil,
                range: nil
            )
        )
        
        // Composite filter
        let compositeFilter = SearchFilter.composite(
            CompositeFilter(
                filters: [metadataFilter, vectorFilter],
                operation: .and
            )
        )
        
        // Test filters
        for vector in vectors {
            let passesMetadata = try! await evaluator.evaluateFilter(metadataFilter, vector: vector)
            let passesVector = try! await evaluator.evaluateFilter(vectorFilter, vector: vector)
            let passesComposite = try! await evaluator.evaluateFilter(compositeFilter, vector: vector)
            
            print("  Vector \(vector.id):")
            print("    Metadata filter: \(passesMetadata)")
            print("    Vector filter: \(passesVector)")
            print("    Composite filter: \(passesComposite)")
        }
    }
    
    // MARK: - Example 2: Learned Filtering
    
    static func demonstrateLearnedFiltering(evaluator: FilterEvaluator) async {
        // Preload models for better performance
        do {
            try await evaluator.preloadModel("quality_filter")
            try await evaluator.preloadModel("category_filter", version: .init(major: 1, minor: 2, patch: 0))
            print("  Models preloaded successfully")
        } catch {
            print("  Model preloading error: \(error)")
        }
        
        // Create test vectors
        let testVectors = generateTestVectors(count: 10, dimension: 512)
        
        // Test different learned filters
        let filters = [
            LearnedFilter(
                modelIdentifier: "quality_filter",
                confidence: 0.8,
                parameters: ["threshold": "0.7"]
            ),
            LearnedFilter(
                modelIdentifier: "category_filter:1.2.0",
                confidence: 0.6,
                parameters: ["category": "premium"]
            )
        ]
        
        for (idx, filter) in filters.enumerated() {
            print("\n  Testing learned filter \(idx + 1): \(filter.modelIdentifier)")
            
            var passCount = 0
            let startTime = Date()
            
            for vector in testVectors {
                let passes = await evaluator.evaluateLearnedFilter(filter, vector: vector)
                if passes { passCount += 1 }
            }
            
            let duration = Date().timeIntervalSince(startTime)
            print("    Passed: \(passCount)/\(testVectors.count)")
            print("    Time: \(String(format: "%.3f", duration))s")
            print("    Avg per vector: \(String(format: "%.1f", duration * 1000 / Double(testVectors.count)))ms")
        }
    }
    
    // MARK: - Example 3: Batch Processing
    
    static func demonstrateBatchProcessing(evaluator: FilterEvaluator) async {
        let batchSizes = [100, 500, 1000]
        
        for batchSize in batchSizes {
            print("\n  Testing batch size: \(batchSize)")
            
            // Generate test data
            let vectors = generateTestVectors(count: batchSize, dimension: 512)
            
            // Create a complex filter
            let filter = SearchFilter.composite(
                CompositeFilter(
                    filters: [
                        .metadata(MetadataFilter(
                            key: "score",
                            value: "0.5",
                            operation: .greaterThanOrEqual
                        )),
                        .vector(VectorFilter(
                            constraint: .magnitude(0.3...0.9),
                            dimension: nil,
                            range: nil
                        )),
                        .learned(LearnedFilter(
                            modelIdentifier: "quality_filter",
                            confidence: 0.7,
                            parameters: [:]
                        ))
                    ],
                    operation: .and
                )
            )
            
            // Measure batch processing time
            let startTime = Date()
            let filtered = try! await evaluator.filterVectors(vectors, filter: filter)
            let duration = Date().timeIntervalSince(startTime)
            
            print("    Processed: \(vectors.count) vectors")
            print("    Passed: \(filtered.count) vectors (\(String(format: "%.1f", Double(filtered.count) / Double(vectors.count) * 100))%)")
            print("    Total time: \(String(format: "%.3f", duration))s")
            print("    Throughput: \(String(format: "%.0f", Double(vectors.count) / duration)) vectors/sec")
        }
    }
    
    // MARK: - Example 4: Hot Reload
    
    static func demonstrateHotReload(evaluator: FilterEvaluator) async {
        print("  Setting up hot reload for model updates...")
        
        // Simulate model path (in real use, this would be an actual model directory)
        let modelPath = URL(fileURLWithPath: "/tmp/models/quality_filter/1.0.0")
        
        // Enable hot reload with callback
        await evaluator.enableModelHotReload(
            modelId: "quality_filter",
            path: modelPath
        ) { modelId, version in
            print("  Model updated: \(modelId) -> v\(version.string)")
        }
        
        print("  Hot reload enabled. Model will automatically update when files change.")
        
        // In production, you could test this by:
        // 1. Modifying the model files at modelPath
        // 2. Observing automatic reload and version update
        // 3. Seeing improved predictions without restart
    }
    
    // MARK: - Example 5: Performance Monitoring
    
    static func demonstratePerformanceMonitoring(evaluator: FilterEvaluator) async {
        // Get comprehensive metrics
        let metrics = await evaluator.getMetrics()
        
        print("\n  Filter Evaluation Metrics:")
        print("    Learned filter evaluations: \(metrics.learnedFilterEvaluations)")
        print("    Learned filter errors: \(metrics.learnedFilterErrors)")
        print("    Average learned filter time: \(String(format: "%.2f", metrics.averageLearnedFilterTimeMs))ms")
        
        print("\n  Batch Processing Metrics:")
        print("    Batch operations: \(metrics.batchFilterOperations)")
        print("    Vectors processed: \(metrics.vectorsProcessed)")
        print("    Vectors passed: \(metrics.vectorsPassed)")
        print("    Filter pass rate: \(String(format: "%.1f", metrics.filterPassRate * 100))%")
        print("    Average batch time: \(String(format: "%.2f", metrics.averageBatchFilterTimeMs))ms")
        
        print("\n  Model Cache Metrics:")
        print("    Cache hits: \(metrics.modelCacheHits)")
        print("    Cache misses: \(metrics.modelCacheMisses)")
        print("    Cache hit rate: \(String(format: "%.1f", Double(metrics.modelCacheHits) / Double(metrics.modelCacheHits + metrics.modelCacheMisses) * 100))%")
        print("    Models loaded: \(metrics.modelsLoaded)")
        print("    Average model load time: \(String(format: "%.2f", metrics.averageModelLoadTimeMs))ms")
        
        // Reset metrics for next measurement period
        await evaluator.resetMetrics()
        print("\n  Metrics reset for next measurement period")
    }
    
    // MARK: - Helper Functions
    
    static func generateTestVectors(count: Int, dimension: Int) -> [StoredVector] {
        (0..<count).map { i in
            // Generate random vector
            let vector = (0..<dimension).map { _ in Float.random(in: -1...1) }
            
            // Normalize to create realistic magnitude
            let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
            let normalized = vector.map { $0 / magnitude }
            
            // Generate metadata
            let metadata = try! JSONEncoder().encode([
                "id": i,
                "score": Float.random(in: 0...1),
                "category": ["A", "B", "C", "D"].randomElement()!,
                "timestamp": Date().timeIntervalSince1970
            ])
            
            return StoredVector(
                id: "vec_\(i)",
                vector: normalized,
                metadata: metadata
            )
        }
    }
}

// MARK: - Performance Testing Extension

extension ProductionFilterExample {
    /// Advanced performance testing with different scenarios
    static func runPerformanceTests(evaluator: FilterEvaluator) async {
        print("\n=== Performance Testing Suite ===")
        
        // Test 1: Model loading performance
        await testModelLoadingPerformance(evaluator: evaluator)
        
        // Test 2: Concurrent evaluation performance
        await testConcurrentEvaluation(evaluator: evaluator)
        
        // Test 3: Memory pressure handling
        await testMemoryPressure(evaluator: evaluator)
    }
    
    static func testModelLoadingPerformance(evaluator: FilterEvaluator) async {
        print("\n1. Model Loading Performance")
        
        let modelIds = ["model_1", "model_2", "model_3", "model_4", "model_5"]
        
        // Cold start (models not in cache)
        var coldStartTimes: [TimeInterval] = []
        for modelId in modelIds {
            let start = Date()
            do {
                _ = try await evaluator.preloadModel(modelId)
                coldStartTimes.append(Date().timeIntervalSince(start))
            } catch {
                print("  Failed to load \(modelId): \(error)")
            }
        }
        
        // Warm start (models in cache)
        var warmStartTimes: [TimeInterval] = []
        for modelId in modelIds {
            let start = Date()
            do {
                _ = try await evaluator.preloadModel(modelId)
                warmStartTimes.append(Date().timeIntervalSince(start))
            } catch {
                print("  Failed to load \(modelId): \(error)")
            }
        }
        
        print("  Cold start avg: \(String(format: "%.3f", coldStartTimes.reduce(0, +) / Double(coldStartTimes.count)))s")
        print("  Warm start avg: \(String(format: "%.3f", warmStartTimes.reduce(0, +) / Double(warmStartTimes.count)))s")
        print("  Speedup: \(String(format: "%.1fx", coldStartTimes.reduce(0, +) / warmStartTimes.reduce(0, +)))")
    }
    
    static func testConcurrentEvaluation(evaluator: FilterEvaluator) async {
        print("\n2. Concurrent Evaluation Performance")
        
        let vectorCount = 1000
        let vectors = generateTestVectors(count: vectorCount, dimension: 512)
        let filter = LearnedFilter(
            modelIdentifier: "quality_filter",
            confidence: 0.7,
            parameters: [:]
        )
        
        // Sequential evaluation
        let seqStart = Date()
        var seqResults = 0
        for vector in vectors {
            if await evaluator.evaluateLearnedFilter(filter, vector: vector) {
                seqResults += 1
            }
        }
        let seqDuration = Date().timeIntervalSince(seqStart)
        
        // Concurrent evaluation
        let concStart = Date()
        let concResults = await withTaskGroup(of: Bool.self) { group in
            for vector in vectors {
                group.addTask {
                    await evaluator.evaluateLearnedFilter(filter, vector: vector)
                }
            }
            
            var count = 0
            for await result in group {
                if result { count += 1 }
            }
            return count
        }
        let concDuration = Date().timeIntervalSince(concStart)
        
        print("  Sequential: \(String(format: "%.3f", seqDuration))s (\(seqResults) passed)")
        print("  Concurrent: \(String(format: "%.3f", concDuration))s (\(concResults) passed)")
        print("  Speedup: \(String(format: "%.1fx", seqDuration / concDuration))")
    }
    
    static func testMemoryPressure(evaluator: FilterEvaluator) async {
        print("\n3. Memory Pressure Handling")
        
        // Simulate loading many models to trigger eviction
        let modelCount = 20
        var loadTimes: [TimeInterval] = []
        
        for i in 0..<modelCount {
            let modelId = "stress_test_model_\(i)"
            let start = Date()
            
            do {
                _ = try await evaluator.preloadModel(modelId)
                loadTimes.append(Date().timeIntervalSince(start))
            } catch {
                print("  Model \(i) failed: \(error)")
            }
            
            if i % 5 == 4 {
                let metrics = await evaluator.getMetrics()
                print("  After \(i + 1) models: \(metrics.modelsLoaded) loaded, memory: \(metrics.memoryUsageMB)MB")
            }
        }
        
        let avgLoadTime = loadTimes.reduce(0, +) / Double(loadTimes.count)
        print("  Average load time under pressure: \(String(format: "%.3f", avgLoadTime))s")
    }
}
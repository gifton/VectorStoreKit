import Foundation
import VectorStoreKit

/// Example demonstrating VectorStoreKit debugging utilities
@main
struct DebuggingExample {
    static func main() async throws {
        print("VectorStoreKit Debugging Example")
        print("================================\n")
        
        // Initialize debugger
        let debugger = VectorStoreDebugger(debugLevel: .debug, isEnabled: true)
        
        // Create a vector store with specific configuration
        let config = StoreConfiguration(
            dimensions: 256,
            distanceMetric: .cosine,
            indexingStrategy: HNSWIndexingStrategy(m: 16, efConstruction: 200, efSearch: 50),
            enableMetalAcceleration: true,
            enableCaching: true,
            cacheConfiguration: LRUCacheConfiguration(maxCapacity: 1000)
        )
        
        let store = VectorStore(configuration: config)
        
        // MARK: - Configuration Validation
        
        print("1. Validating Configuration")
        print("---------------------------")
        
        let validator = ConfigurationValidator()
        let validationResult = await validator.validate(config)
        
        print("Configuration is valid: \(validationResult.isValid)")
        
        if !validationResult.errors.isEmpty {
            print("\nErrors:")
            for error in validationResult.errors {
                print("  - [\(error.field)] \(error.message)")
            }
        }
        
        if !validationResult.warnings.isEmpty {
            print("\nWarnings:")
            for warning in validationResult.warnings {
                print("  - [\(warning.field)] \(warning.message) (Impact: \(warning.impact))")
            }
        }
        
        if !validationResult.suggestions.isEmpty {
            print("\nSuggestions:")
            for suggestion in validationResult.suggestions {
                print("  - [\(suggestion.field)] \(suggestion.suggestion)")
                print("    Benefit: \(suggestion.benefit)")
            }
        }
        
        print("\nPerformance Estimates:")
        print("  Memory per vector: \(formatBytes(validationResult.performanceEstimate.memoryPerVector))")
        print("  Search complexity: \(validationResult.performanceEstimate.searchComplexity)")
        print("  Build complexity: \(validationResult.performanceEstimate.buildComplexity)")
        print("  Estimated throughput: \(validationResult.performanceEstimate.estimatedThroughput) ops/sec")
        
        // MARK: - Adding Vectors with Performance Tracking
        
        print("\n2. Adding Vectors with Performance Tracking")
        print("-------------------------------------------")
        
        let vectorCount = 1000
        let addHandle = await debugger.startTracing(operation: "add_vectors")
        
        for i in 0..<vectorCount {
            // Track memory allocation
            await debugger.trackMemoryAllocation(
                component: "VectorData",
                bytes: config.dimensions * MemoryLayout<Float>.size,
                operation: "add_vector_\(i)"
            )
            
            // Create and add vector
            let vector = generateRandomVector(dimensions: config.dimensions)
            try await store.add(vector, id: "vec_\(i)")
            
            if i % 100 == 0 {
                print("  Added \(i + 1) vectors...")
            }
        }
        
        await addHandle.complete()
        print("  Completed adding \(vectorCount) vectors")
        
        // MARK: - Store Inspection
        
        print("\n3. Inspecting Store State")
        print("-------------------------")
        
        let storeInfo = try await debugger.inspectStore(store)
        print("Store Information:")
        print("  Component: \(storeInfo.componentName)")
        print("  Vector count: \(storeInfo.metrics["vectorCount"] ?? 0)")
        print("  Index count: \(storeInfo.metrics["indexCount"] ?? 0)")
        
        // MARK: - Memory Analysis
        
        print("\n4. Memory Usage Analysis")
        print("------------------------")
        
        let memoryReport = await debugger.analyzeMemoryUsage()
        print("Memory Usage:")
        print("  Total: \(formatBytes(memoryReport.totalUsage.totalBytes))")
        print("  Peak: \(formatBytes(memoryReport.totalUsage.peakBytes))")
        print("  Allocations: \(memoryReport.totalUsage.allocations)")
        
        print("\nTop Memory Consumers:")
        for component in memoryReport.components.prefix(5) {
            print("  - \(component.name): \(formatBytes(component.usage.totalBytes)) (\(String(format: "%.1f%%", component.percentage)))")
        }
        
        // MARK: - Query Performance Analysis
        
        print("\n5. Query Performance Analysis")
        print("-----------------------------")
        
        let queryVector = generateRandomVector(dimensions: config.dimensions)
        let k = 10
        
        // Explain query plan
        let queryExplainer = QueryExplainer()
        let queryPlan = try await queryExplainer.explain(
            query: queryVector,
            k: k,
            store: store
        )
        
        print("Query Execution Plan:")
        print(await queryExplainer.formatPlan(queryPlan, format: .tree))
        
        // Execute query with tracing
        let queryHandle = await debugger.startTracing(operation: "search_query")
        let results = try await store.search(query: queryVector, k: k)
        await queryHandle.complete()
        
        print("\nQuery returned \(results.count) results")
        
        // MARK: - Performance Report
        
        print("\n6. Performance Report")
        print("--------------------")
        
        let perfReport = await debugger.getPerformanceReport()
        
        print("Operation Summary:")
        for op in perfReport.operations.sorted(by: { $0.totalTime > $1.totalTime }).prefix(5) {
            print("  - \(op.name):")
            print("    Count: \(op.count)")
            print("    Total time: \(String(format: "%.2fms", op.totalTime * 1000))")
            print("    Average time: \(String(format: "%.2fms", op.averageTime * 1000))")
        }
        
        // MARK: - Consistency Check
        
        print("\n7. Consistency Validation")
        print("-------------------------")
        
        let consistencyChecker = ConsistencyChecker()
        let validationResult2 = try await consistencyChecker.validateStore(store)
        
        print("Store is consistent: \(validationResult2.isValid)")
        
        if !validationResult2.issues.isEmpty {
            print("\nIssues found:")
            for issue in validationResult2.issues.prefix(5) {
                print("  [\(issue.severity)] \(issue.component): \(issue.message)")
                if let suggestion = issue.suggestion {
                    print("    → \(suggestion)")
                }
            }
        }
        
        // MARK: - Metal GPU Analysis
        
        if config.enableMetalAcceleration,
           let device = MTLCreateSystemDefaultDevice() {
            print("\n8. Metal GPU Performance")
            print("------------------------")
            
            let metalDebugger = try MetalDebugger(device: device)
            let metalReport = await metalDebugger.generatePerformanceReport()
            
            print("GPU Memory Usage:")
            print("  Total: \(formatBytes(metalReport.memoryUsage.totalAllocated))")
            print("  GPU-only: \(formatBytes(metalReport.memoryUsage.gpuMemory))")
            print("  Shared: \(formatBytes(metalReport.memoryUsage.sharedMemory))")
            
            if !metalReport.recommendations.isEmpty {
                print("\nGPU Optimization Recommendations:")
                for rec in metalReport.recommendations {
                    print("  - \(rec)")
                }
            }
        }
        
        // MARK: - Visualization
        
        print("\n9. Vector Space Visualization")
        print("-----------------------------")
        
        let viz = VisualizationUtilities()
        
        // Sample some vectors for visualization
        let sampleSize = min(100, vectorCount)
        var sampleVectors: [Vector] = []
        for i in 0..<sampleSize {
            if let result = try await store.get(id: "vec_\(i * (vectorCount / sampleSize))") {
                sampleVectors.append(result.vector)
            }
        }
        
        print("Reducing \(sampleVectors.count) vectors to 2D using PCA...")
        let reduced = try await viz.reduceVectorDimensions(
            vectors: sampleVectors,
            targetDimensions: 2,
            method: .pca
        )
        
        print("Visualization data generated:")
        print("  Points: \(reduced.count)")
        if let first = reduced.first, let last = reduced.last {
            print("  First point: \(first.id) at [\(first.coordinates.map { String(format: "%.3f", $0) }.joined(separator: ", "))]")
            print("  Last point: \(last.id) at [\(last.coordinates.map { String(format: "%.3f", $0) }.joined(separator: ", "))]")
        }
        
        // MARK: - Diagnostic Dump
        
        print("\n10. Generating Diagnostic Dump")
        print("------------------------------")
        
        let dump = try await debugger.generateDiagnosticDump(for: store, includeVectors: false)
        
        print("Diagnostic dump created:")
        print("  Timestamp: \(dump.timestamp)")
        print("  Store info captured: ✓")
        print("  Performance data captured: ✓")
        print("  Memory analysis captured: ✓")
        print("  Index reports: \(dump.indexReports.count)")
        
        // Export logs
        let logger = DebugLogger(subsystem: "debugging-example", level: .debug)
        logger.info("Debugging session completed")
        
        if let logData = logger.exportLogs(format: .json),
           let logFile = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?.appendingPathComponent("vectorstore-debug.json") {
            try logData.write(to: logFile)
            print("\nDebug logs exported to: \(logFile.path)")
        }
        
        print("\nDebugging example completed successfully!")
    }
    
    // MARK: - Helper Functions
    
    static func generateRandomVector(dimensions: Int) -> Vector {
        let data = (0..<dimensions).map { _ in Float.random(in: -1...1) }
        // Normalize for cosine distance
        let norm = sqrt(data.map { $0 * $0 }.reduce(0, +))
        return Vector(data: data.map { $0 / norm })
    }
    
    static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}
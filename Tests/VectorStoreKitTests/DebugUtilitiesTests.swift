import XCTest
@testable import VectorStoreKit

final class DebugUtilitiesTests: XCTestCase {
    
    // MARK: - VectorStoreDebugger Tests
    
    func testVectorStoreDebuggerInitialization() async throws {
        let debugger = VectorStoreDebugger()
        await debugger.setEnabled(true)
        await debugger.setDebugLevel(.debug)
        
        // Create a test store
        let config = StoreConfiguration(
            dimensions: 128,
            distanceMetric: .euclidean,
            indexingStrategy: HNSWIndexingStrategy(m: 16, efConstruction: 200)
        )
        let store = VectorStore(configuration: config)
        
        // Test inspection
        let debugInfo = try await debugger.inspectStore(store)
        XCTAssertEqual(debugInfo.componentName, "VectorStore")
        XCTAssertNotNil(debugInfo.state["configuration"])
    }
    
    func testPerformanceTracing() async throws {
        let debugger = VectorStoreDebugger()
        
        // Start tracing
        let handle = await debugger.startTracing(operation: "test_operation")
        
        // Simulate some work
        try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        
        // Complete tracing
        await handle.complete()
        
        // Get performance report
        let report = await debugger.getPerformanceReport()
        XCTAssertFalse(report.operations.isEmpty)
        
        let testOp = report.operations.first { $0.name == "test_operation" }
        XCTAssertNotNil(testOp)
        XCTAssertGreaterThan(testOp!.totalTime, 0)
    }
    
    // MARK: - PerformanceTracer Tests
    
    func testPerformanceTracerBottleneckDetection() async throws {
        let tracer = PerformanceTracer()
        
        // Simulate operations with different durations
        for i in 0..<10 {
            let handle = await tracer.startOperation("fast_op")
            try await Task.sleep(nanoseconds: 1_000_000) // 1ms
            await handle.complete()
        }
        
        for i in 0..<5 {
            let handle = await tracer.startOperation("slow_op")
            try await Task.sleep(nanoseconds: 100_000_000) // 100ms
            await handle.complete()
        }
        
        // Find bottlenecks
        let bottlenecks = await tracer.findBottlenecks(threshold: 0.05) // 50ms
        XCTAssertFalse(bottlenecks.isEmpty)
        
        let slowOpBottleneck = bottlenecks.first { $0.operation == "slow_op" }
        XCTAssertNotNil(slowOpBottleneck)
        XCTAssertGreaterThan(slowOpBottleneck!.impact, 0)
    }
    
    func testBatchTracingHandle() async throws {
        let tracer = PerformanceTracer()
        
        let handle = await tracer.traceBatchOperation("batch_process", batchSize: 1000)
        
        // Simulate batch processing
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms
        
        await handle.complete(processedItems: 950) // Some items filtered out
        
        let report = await tracer.generateReport()
        let batchOp = report.operations.first { $0.name.contains("batch_process") }
        XCTAssertNotNil(batchOp)
    }
    
    // MARK: - MemoryAnalyzer Tests
    
    func testMemoryAnalyzerAllocationTracking() async throws {
        let analyzer = MemoryAnalyzer()
        
        // Track allocations
        await analyzer.trackAllocation(
            component: "TestComponent",
            bytes: 1_048_576, // 1MB
            operation: "initialize"
        )
        
        await analyzer.trackAllocation(
            component: "TestComponent",
            bytes: 524_288, // 512KB
            operation: "expand"
        )
        
        // Track deallocation
        await analyzer.trackDeallocation(
            component: "TestComponent",
            bytes: 524_288,
            operation: "shrink"
        )
        
        // Analyze memory
        let report = await analyzer.analyze()
        XCTAssertFalse(report.components.isEmpty)
        
        let testComponent = report.components.first { $0.name == "TestComponent" }
        XCTAssertNotNil(testComponent)
        XCTAssertEqual(testComponent!.usage.totalBytes, 1_048_576) // 1MB remaining
    }
    
    func testMemoryPressureDetection() async throws {
        let analyzer = MemoryAnalyzer()
        
        let pressure = await analyzer.detectMemoryPressure()
        XCTAssertNotNil(pressure)
        // Can't assert specific value as it depends on system state
    }
    
    func testMemoryTrendAnalysis() async throws {
        let analyzer = MemoryAnalyzer()
        
        // Track allocations over time
        for i in 0..<5 {
            await analyzer.trackAllocation(
                component: "TrendTest",
                bytes: 100_000 * (i + 1),
                operation: "allocate_\(i)"
            )
            try await Task.sleep(nanoseconds: 10_000_000) // 10ms
        }
        
        let trend = await analyzer.getMemoryTrend(duration: 1.0) // Last second
        XCTAssertNotNil(trend)
    }
    
    // MARK: - IndexInspector Tests
    
    func testIndexInspectorHNSW() async throws {
        let inspector = IndexInspector()
        
        // Create test HNSW index
        let config = HNSWConfiguration(m: 16, efConstruction: 200)
        let index = HNSWIndex(configuration: config)
        
        // Add some test vectors
        for i in 0..<100 {
            let vector = Vector(data: Array(repeating: Float(i), count: 128))
            try await index.add(vector, id: "vec_\(i)")
        }
        
        // Inspect the index
        let result = await inspector.inspectHNSW(index)
        XCTAssertFalse(result.layerStatistics.isEmpty)
        XCTAssertNotNil(result.connectivityStatistics)
        XCTAssertNotNil(result.searchStatistics)
    }
    
    // MARK: - QueryExplainer Tests
    
    func testQueryExplainerGeneratePlan() async throws {
        let explainer = QueryExplainer()
        
        let config = StoreConfiguration(
            dimensions: 128,
            distanceMetric: .euclidean,
            indexingStrategy: HNSWIndexingStrategy(m: 16, efConstruction: 200),
            enableMetalAcceleration: true
        )
        let store = VectorStore(configuration: config)
        
        let query = Vector(data: Array(repeating: 0.5, count: 128))
        
        let plan = try await explainer.explain(query: query, k: 10, store: store)
        XCTAssertNotNil(plan)
        XCTAssertGreaterThan(plan.estimatedTotalCost, 0)
        XCTAssertEqual(plan.root.operation, "HNSW Search")
    }
    
    func testQueryPlanFormatting() async throws {
        let explainer = QueryExplainer()
        
        let plan = QueryPlan(root: QueryPlanNode(
            operation: "Test Operation",
            estimatedCost: 100,
            estimatedRows: 1000,
            children: [
                QueryPlanNode(
                    operation: "Child Operation",
                    estimatedCost: 50,
                    estimatedRows: 500
                )
            ]
        ))
        
        // Test tree format
        let treeFormat = await explainer.formatPlan(plan, format: .tree)
        XCTAssertTrue(treeFormat.contains("Test Operation"))
        XCTAssertTrue(treeFormat.contains("Child Operation"))
        
        // Test JSON format
        let jsonFormat = await explainer.formatPlan(plan, format: .json)
        XCTAssertTrue(jsonFormat.contains("\"operation\""))
        
        // Test Graphviz format
        let dotFormat = await explainer.formatPlan(plan, format: .graphviz)
        XCTAssertTrue(dotFormat.contains("digraph"))
    }
    
    // MARK: - DebugLogger Tests
    
    func testDebugLoggerLevels() {
        let logger = DebugLogger(subsystem: "test", level: .warning)
        
        // These should not be logged
        logger.trace("trace message")
        logger.debug("debug message")
        logger.info("info message")
        
        // These should be logged
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")
        
        let logs = logger.getRecentLogs(count: 10)
        XCTAssertEqual(logs.count, 3) // Only warning, error, critical
    }
    
    func testDebugLoggerSearch() {
        let logger = DebugLogger(subsystem: "test", level: .debug)
        
        logger.debug("Test message one")
        logger.info("Test message two")
        logger.warning("Different content")
        
        let results = logger.searchLogs(containing: "Test message")
        XCTAssertEqual(results.count, 2)
    }
    
    func testDebugLoggerExport() throws {
        let logger = DebugLogger(subsystem: "test", level: .debug)
        
        logger.debug("Export test 1")
        logger.info("Export test 2")
        
        // Test JSON export
        let jsonData = logger.exportLogs(format: .json)
        XCTAssertNotNil(jsonData)
        
        // Test CSV export
        let csvData = logger.exportLogs(format: .csv)
        XCTAssertNotNil(csvData)
        
        if let csvString = String(data: csvData!, encoding: .utf8) {
            XCTAssertTrue(csvString.contains("Export test"))
        }
    }
    
    // MARK: - ConsistencyChecker Tests
    
    func testConsistencyCheckerValidation() async throws {
        let checker = ConsistencyChecker()
        
        let config = StoreConfiguration(
            dimensions: 128,
            distanceMetric: .euclidean,
            indexingStrategy: HNSWIndexingStrategy(m: 2, efConstruction: 10) // Too small
        )
        let store = VectorStore(configuration: config)
        
        let result = try await checker.validateStore(store)
        XCTAssertFalse(result.issues.isEmpty)
        
        // Should find issue with M parameter being too small
        let mParamIssue = result.issues.first { $0.message.contains("M parameter") }
        XCTAssertNotNil(mParamIssue)
        XCTAssertEqual(mParamIssue?.severity, .error)
    }
    
    // MARK: - MetalDebugger Tests
    
    func testMetalDebuggerInitialization() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        
        let debugger = try MetalDebugger(device: device)
        
        // Track a buffer
        let buffer = device.makeBuffer(length: 1024, options: .storageModeShared)!
        await debugger.trackBufferAllocation(
            label: "test_buffer",
            buffer: buffer,
            purpose: "testing"
        )
        
        // Generate performance report
        let report = await debugger.generatePerformanceReport()
        XCTAssertNotNil(report)
        XCTAssertGreaterThan(report.memoryUsage.totalAllocated, 0)
    }
    
    func testMetalShaderValidation() async throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        
        let debugger = try MetalDebugger(device: device)
        
        let buffer = device.makeBuffer(length: 1000, options: [])! // Not 16-byte aligned
        
        let issues = await debugger.validateShaderInputs(
            kernel: "test_kernel",
            buffers: [(buffer, 0)],
            threadgroupSize: MTLSize(width: 31, height: 1, depth: 1), // Not multiple of 32
            gridSize: MTLSize(width: 1024, height: 1, depth: 1)
        )
        
        XCTAssertFalse(issues.isEmpty)
        
        // Should find alignment issue
        let alignmentIssue = issues.first { $0.message.contains("aligned") }
        XCTAssertNotNil(alignmentIssue)
        
        // Should find threadgroup size issue
        let threadgroupIssue = issues.first { $0.message.contains("multiple of 32") }
        XCTAssertNotNil(threadgroupIssue)
    }
    
    // MARK: - ConfigurationValidator Tests
    
    func testConfigurationValidatorBasic() async throws {
        let validator = ConfigurationValidator()
        
        let goodConfig = StoreConfiguration(
            dimensions: 256,
            distanceMetric: .euclidean,
            indexingStrategy: HNSWIndexingStrategy(m: 16, efConstruction: 200)
        )
        
        let result = await validator.validate(goodConfig)
        XCTAssertTrue(result.isValid)
        XCTAssertTrue(result.errors.isEmpty)
    }
    
    func testConfigurationValidatorErrors() async throws {
        let validator = ConfigurationValidator()
        
        let badConfig = StoreConfiguration(
            dimensions: -1, // Invalid
            distanceMetric: .euclidean,
            indexingStrategy: HNSWIndexingStrategy(m: 2, efConstruction: 10) // Too small
        )
        
        let result = await validator.validate(badConfig)
        XCTAssertFalse(result.isValid)
        XCTAssertFalse(result.errors.isEmpty)
        
        // Should find dimension error
        let dimError = result.errors.first { $0.field == "dimensions" }
        XCTAssertNotNil(dimError)
    }
    
    func testConfigurationOptimization() async throws {
        let validator = ConfigurationValidator()
        
        let baseConfig = StoreConfiguration(
            dimensions: 512,
            distanceMetric: .euclidean,
            indexingStrategy: HNSWIndexingStrategy(m: 16, efConstruction: 200)
        )
        
        // Test optimization for different use cases
        let accuracyOptimized = await validator.optimizeConfiguration(baseConfig, for: .highAccuracy)
        XCTAssertNil(accuracyOptimized.quantizationConfiguration) // No quantization for accuracy
        
        let latencyOptimized = await validator.optimizeConfiguration(baseConfig, for: .lowLatency)
        XCTAssertTrue(latencyOptimized.enableMetalAcceleration)
        XCTAssertTrue(latencyOptimized.enableCaching)
        
        let memoryOptimized = await validator.optimizeConfiguration(baseConfig, for: .lowMemory)
        XCTAssertNotNil(memoryOptimized.quantizationConfiguration)
    }
    
    // MARK: - VisualizationUtilities Tests
    
    func testVisualizationPCA() async throws {
        let viz = VisualizationUtilities()
        
        // Create test vectors
        var vectors: [Vector] = []
        for i in 0..<20 {
            let data = (0..<10).map { j in Float(i) + Float(j) * 0.1 }
            vectors.append(Vector(data: data))
        }
        
        let reduced = try await viz.reduceVectorDimensions(
            vectors: vectors,
            targetDimensions: 2,
            method: .pca
        )
        
        XCTAssertEqual(reduced.count, vectors.count)
        XCTAssertEqual(reduced[0].coordinates.count, 2)
    }
    
    func testVisualizationDistanceMatrix() async throws {
        let viz = VisualizationUtilities()
        
        // Create test vectors
        let vectors = [
            Vector(data: [1.0, 0.0, 0.0]),
            Vector(data: [0.0, 1.0, 0.0]),
            Vector(data: [0.0, 0.0, 1.0]),
            Vector(data: [1.0, 1.0, 0.0])
        ]
        
        let matrixData = await viz.computeDistanceMatrix(
            vectors: vectors,
            metric: .euclidean
        )
        
        XCTAssertEqual(matrixData.matrix.count, 4)
        XCTAssertEqual(matrixData.matrix[0].count, 4)
        
        // Check diagonal is zero
        for i in 0..<4 {
            XCTAssertEqual(matrixData.matrix[i][i], 0)
        }
        
        // Check symmetry
        for i in 0..<4 {
            for j in 0..<4 {
                XCTAssertEqual(matrixData.matrix[i][j], matrixData.matrix[j][i])
            }
        }
    }
    
    func testVisualizationExport() async throws {
        let viz = VisualizationUtilities()
        
        let testData = ClusterVisualizationData(
            points: [
                ("vec_0", [1.0, 2.0]),
                ("vec_1", [3.0, 4.0]),
                ("vec_2", [5.0, 6.0])
            ],
            clusterAssignments: [0, 0, 1],
            clusterCenters: [(0, [2.0, 3.0]), (1, [5.0, 6.0])],
            method: .pca
        )
        
        // Test JSON export
        let jsonData = try await viz.exportVisualizationData(testData, format: .json)
        XCTAssertGreaterThan(jsonData.count, 0)
        
        // Test CSV export
        let csvData = try await viz.exportVisualizationData(testData, format: .csv)
        let csvString = String(data: csvData, encoding: .utf8)!
        XCTAssertTrue(csvString.contains("id,x,y,cluster"))
        XCTAssertTrue(csvString.contains("vec_0"))
    }
    
    // MARK: - Integration Tests
    
    func testDebuggerIntegration() async throws {
        // Test that all components work together
        let debugger = VectorStoreDebugger()
        
        let config = StoreConfiguration(
            dimensions: 64,
            distanceMetric: .euclidean,
            indexingStrategy: HNSWIndexingStrategy(m: 8, efConstruction: 100),
            enableMetalAcceleration: true,
            enableCaching: true
        )
        let store = VectorStore(configuration: config)
        
        // Add some vectors
        for i in 0..<50 {
            let vector = Vector(data: Array(repeating: Float(i) / 50.0, count: 64))
            try await store.add(vector, id: "vec_\(i)")
        }
        
        // Generate comprehensive diagnostic dump
        let dump = try await debugger.generateDiagnosticDump(for: store, includeVectors: false)
        
        XCTAssertNotNil(dump)
        XCTAssertNotNil(dump.storeInfo)
        XCTAssertNotNil(dump.performanceReport)
        XCTAssertNotNil(dump.memoryReport)
    }
}
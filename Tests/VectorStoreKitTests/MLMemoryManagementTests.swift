// VectorStoreKit: ML Memory Management Tests
//
// Tests for memory management in ML operations

import XCTest
@testable import VectorStoreKit
import Metal

final class MLMemoryManagementTests: XCTestCase {
    var device: MTLDevice!
    var memoryManager: MLMemoryManager!
    var pipeline: MetalMLPipeline!
    var bufferPool: PressureAwareBufferPool!
    
    override func setUp() async throws {
        try await super.setUp()
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        
        self.device = device
        self.memoryManager = MLMemoryManager(device: device, maxMemoryGB: 0.5) // 500MB for testing
        self.pipeline = try MetalMLPipeline(device: device)
        self.bufferPool = PressureAwareBufferPool(
            device: device,
            memoryManager: memoryManager,
            maxPoolSize: 50
        )
    }
    
    override func tearDown() async throws {
        // Ensure cleanup
        await bufferPool.clearAll()
        try await super.tearDown()
    }
    
    // MARK: - Memory Manager Tests
    
    func testMemoryTracking() async throws {
        // Get initial stats
        let initialStats = await memoryManager.getStatistics()
        XCTAssertEqual(initialStats.currentUsage, 0)
        XCTAssertEqual(initialStats.bufferCount, 0)
        
        // Allocate buffer
        let size = 1024 * 1024 // 1M floats = 4MB
        let buffer = try await pipeline.allocateBuffer(size: size)
        await memoryManager.trackBuffer(buffer)
        
        // Check tracking
        let stats = await memoryManager.getStatistics()
        XCTAssertEqual(stats.bufferCount, 1)
        XCTAssertEqual(stats.currentUsage, size * MemoryLayout<Float>.stride)
        XCTAssertGreaterThanOrEqual(stats.peakUsage, stats.currentUsage)
        
        // Untrack buffer
        await memoryManager.untrackBuffer(buffer)
        
        // Verify cleanup
        let finalStats = await memoryManager.getStatistics()
        XCTAssertEqual(finalStats.currentUsage, 0)
        XCTAssertEqual(finalStats.bufferCount, 0)
    }
    
    func testMemoryPressureDetection() async throws {
        // Allocate buffers until we hit pressure
        var buffers: [MetalBuffer] = []
        let bufferSize = 10 * 1024 * 1024 // 10M floats = 40MB each
        
        var pressureDetected = false
        await memoryManager.registerCleanupCallback { level in
            if level >= .warning {
                pressureDetected = true
            }
        }
        
        // Allocate until pressure
        for _ in 0..<20 {
            let (currentBytes, percentage) = await memoryManager.getCurrentUsage()
            
            if percentage > 0.7 {
                break
            }
            
            if await memoryManager.canAllocate(size: bufferSize * MemoryLayout<Float>.stride) {
                let buffer = try await pipeline.allocateBuffer(size: bufferSize)
                await memoryManager.trackBuffer(buffer)
                buffers.append(buffer)
            }
        }
        
        XCTAssertTrue(pressureDetected, "Memory pressure should have been detected")
        
        // Cleanup
        for buffer in buffers {
            await memoryManager.untrackBuffer(buffer)
        }
    }
    
    // MARK: - Buffer Pool Tests
    
    func testBufferPoolReuse() async throws {
        let size = 1024
        
        // First allocation - miss
        let buffer1 = try await bufferPool.acquire(size: size)
        let stats1 = await bufferPool.getStatistics()
        XCTAssertEqual(stats1.hitRate, 0.0) // First allocation is always a miss
        
        // Release back to pool
        // Buffer1 will be automatically released when out of scope
        
        // Second allocation - should hit
        let buffer2 = try await bufferPool.acquire(size: size)
        let stats2 = await bufferPool.getStatistics()
        XCTAssertGreaterThan(stats2.hitRate, 0.0) // Should have a hit
        
        _ = buffer2 // Silence unused warning
    }
    
    func testBufferPoolMemoryPressure() async throws {
        // Fill pool with buffers
        var managedBuffers: [ManagedMetalBuffer] = []
        let bufferSize = 1024 * 1024 // 1M floats
        
        for _ in 0..<10 {
            let buffer = try await bufferPool.acquire(size: bufferSize)
            managedBuffers.append(buffer)
        }
        
        // Clear managed buffers to release them
        managedBuffers.removeAll()
        
        // Pool should now have cached buffers
        let statsBeforePressure = await bufferPool.getStatistics()
        XCTAssertGreaterThan(statsBeforePressure.totalAllocated, 0)
        
        // Simulate memory pressure
        await memoryManager.forceCleanup()
        
        // Give time for cleanup
        try await Task.sleep(nanoseconds: 100_000_000) // 100ms
        
        // Pool should have fewer buffers after pressure
        let statsAfterPressure = await bufferPool.getStatistics()
        XCTAssertLessThanOrEqual(
            statsAfterPressure.totalAllocated,
            statsBeforePressure.totalAllocated
        )
    }
    
    // MARK: - Gradient Checkpointing Tests
    
    func testGradientCheckpointing() async throws {
        let checkpointer = GradientCheckpointer(metalPipeline: pipeline)
        
        // Create test buffer
        let size = 1024
        let buffer = try await pipeline.allocateBuffer(size: size)
        
        // Initialize with test data
        let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: size)
        for i in 0..<size {
            ptr[i] = Float(i)
        }
        
        // Checkpoint
        try await checkpointer.checkpoint(buffer, key: "test_activation")
        
        // Verify checkpoint exists
        let retrieved = await checkpointer.retrieve(key: "test_activation")
        XCTAssertNotNil(retrieved)
        
        // Verify data integrity
        if let retrieved = retrieved {
            let retrievedPtr = retrieved.buffer.contents().bindMemory(to: Float.self, capacity: size)
            for i in 0..<size {
                XCTAssertEqual(retrievedPtr[i], Float(i), accuracy: 0.001)
            }
        }
        
        // Check memory usage
        let memoryUsage = await checkpointer.getMemoryUsage()
        XCTAssertEqual(memoryUsage, size * MemoryLayout<Float>.stride)
        
        // Clear checkpoint
        await checkpointer.clear(key: "test_activation")
        
        // Verify cleared
        let afterClear = await checkpointer.retrieve(key: "test_activation")
        XCTAssertNil(afterClear)
        
        let finalMemoryUsage = await checkpointer.getMemoryUsage()
        XCTAssertEqual(finalMemoryUsage, 0)
    }
    
    // MARK: - Memory Profiling Tests
    
    func testMemoryProfiling() async throws {
        let profiler = MLMemoryProfiler()
        
        // Record allocations
        await profiler.recordAllocation(size: 1024 * 4, source: "test1")
        await profiler.recordAllocation(size: 2048 * 4, source: "test2")
        await profiler.recordAllocation(size: 512 * 4, source: "test1")
        
        // Record deallocations
        await profiler.recordDeallocation(size: 1024 * 4)
        
        // Generate report
        let report = await profiler.generateReport()
        
        XCTAssertEqual(report.totalAllocations, 3)
        XCTAssertEqual(report.largestAllocation, 2048 * 4)
        XCTAssertGreaterThan(report.averageAllocationSize, 0)
        
        // Check allocations by source
        XCTAssertEqual(report.allocationsBySource["test1"], 2)
        XCTAssertEqual(report.allocationsBySource["test2"], 1)
        
        // Check timeline
        XCTAssertFalse(report.timeline.isEmpty)
    }
    
    // MARK: - Integration Tests
    
    func testMemoryEfficientTraining() async throws {
        // Create a small network
        let network = try await NeuralNetwork(metalPipeline: pipeline)
        
        let inputSize = 128
        let hiddenSize = 64
        let outputSize = 10
        
        // Add layers
        let layer1 = try await DenseLayer(
            inputSize: inputSize,
            outputSize: hiddenSize,
            activation: .relu,
            metalPipeline: pipeline
        )
        await network.addLayer(layer1)
        
        let layer2 = try await DenseLayer(
            inputSize: hiddenSize,
            outputSize: outputSize,
            activation: .softmax,
            metalPipeline: pipeline
        )
        await network.addLayer(layer2)
        
        // Enable gradient checkpointing
        await network.enableGradientCheckpointing()
        
        // Create small training data
        let batchSize = 16
        var trainingData: [(input: MetalBuffer, target: MetalBuffer)] = []
        
        for _ in 0..<5 {
            let input = try await bufferPool.acquire(size: batchSize * inputSize).metalBuffer
            let target = try await bufferPool.acquire(size: batchSize * outputSize).metalBuffer
            
            // Initialize with random data
            initializeRandomBuffer(input)
            initializeOneHotBuffer(target, classes: outputSize)
            
            trainingData.append((input, target))
        }
        
        // Track memory before training
        let statsBefore = await memoryManager.getStatistics()
        
        // Train for one epoch
        let config = NetworkTrainingConfig(
            epochs: 1,
            batchSize: batchSize,
            learningRate: 0.01
        )
        
        try await network.train(data: trainingData, config: config)
        
        // Check memory after training
        let statsAfter = await memoryManager.getStatistics()
        
        // Memory usage should be reasonable
        XCTAssertLessThan(
            statsAfter.currentUsage,
            100 * 1024 * 1024 // Should use less than 100MB
        )
        
        print("Memory usage: \(statsAfter.currentUsageMB)MB")
        print("Peak usage: \(statsAfter.peakUsageMB)MB")
    }
    
    func testMemoryLeakDetection() async throws {
        // Initial state
        let initialStats = await memoryManager.getStatistics()
        
        // Run allocation/deallocation cycle
        for _ in 0..<10 {
            autoreleasepool {
                Task {
                    let buffer = try? await bufferPool.acquire(size: 1024)
                    _ = buffer // Use buffer
                    // Should be auto-released
                }
            }
        }
        
        // Wait for cleanup
        try await Task.sleep(nanoseconds: 500_000_000) // 500ms
        
        // Check for leaks
        let finalStats = await memoryManager.getStatistics()
        
        // Should return to initial state (allowing for some pool caching)
        let allowedDifference = 10 * 1024 * 1024 // Allow 10MB for pool caching
        XCTAssertLessThan(
            abs(finalStats.currentUsage - initialStats.currentUsage),
            allowedDifference,
            "Potential memory leak detected"
        )
    }
    
    // MARK: - Performance Tests
    
    func testBufferPoolPerformance() async throws {
        let size = 1024 * 1024 // 1M floats
        let iterations = 100
        
        // Measure pooled allocation
        let pooledStart = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<iterations {
            let buffer = try await bufferPool.acquire(size: size)
            _ = buffer.metalBuffer // Use buffer
            // Auto-released
        }
        
        let pooledTime = CFAbsoluteTimeGetCurrent() - pooledStart
        
        // Measure direct allocation
        let directStart = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<iterations {
            guard let buffer = device.makeBuffer(
                length: size * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else {
                XCTFail("Allocation failed")
                return
            }
            _ = buffer // Use buffer
        }
        
        let directTime = CFAbsoluteTimeGetCurrent() - directStart
        
        print("Pooled allocation time: \(pooledTime)s")
        print("Direct allocation time: \(directTime)s")
        print("Speedup: \(directTime / pooledTime)x")
        
        // Pooled should be faster
        XCTAssertLessThan(pooledTime, directTime)
    }
    
    // MARK: - Helper Functions
    
    private func initializeRandomBuffer(_ buffer: MetalBuffer) {
        let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
        for i in 0..<buffer.count {
            ptr[i] = Float.random(in: -1...1)
        }
    }
    
    private func initializeOneHotBuffer(_ buffer: MetalBuffer, classes: Int) {
        let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: buffer.count)
        let samples = buffer.count / classes
        
        // Zero initialize
        for i in 0..<buffer.count {
            ptr[i] = 0
        }
        
        // Set random class for each sample
        for sample in 0..<samples {
            let classIndex = Int.random(in: 0..<classes)
            ptr[sample * classes + classIndex] = 1.0
        }
    }
}
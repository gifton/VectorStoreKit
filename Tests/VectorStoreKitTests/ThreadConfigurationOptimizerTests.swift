// VectorStoreKit: Thread Configuration Optimizer Tests
//
// Tests for Metal thread configuration optimization
//

import XCTest
@testable import VectorStoreKit

final class ThreadConfigurationOptimizerTests: XCTestCase {
    
    var device: MetalDevice!
    var optimizer: MetalThreadConfigurationOptimizer!
    
    override func setUp() async throws {
        device = try MetalDevice()
        optimizer = await MetalThreadConfigurationOptimizer(device: device)
    }
    
    override func tearDown() {
        device = nil
        optimizer = nil
    }
    
    func testBasicConfiguration() async throws {
        // Test basic thread configuration for distance computation
        let config = await optimizer.getOptimalConfiguration(
            for: .distanceComputation,
            workSize: 1000,
            vectorDimension: 128
        )
        
        // Verify configuration is valid
        XCTAssertGreaterThan(config.threadsPerThreadgroup.width, 0)
        XCTAssertGreaterThan(config.threadgroupsPerGrid.width, 0)
        
        // Verify total work coverage
        let totalThreads = config.threadsPerThreadgroup.width * config.threadgroupsPerGrid.width
        XCTAssertGreaterThanOrEqual(totalThreads, 1000)
        
        // Verify occupancy estimate
        XCTAssertGreaterThan(config.estimatedOccupancy, 0)
        XCTAssertLessThanOrEqual(config.estimatedOccupancy, 1.0)
    }
    
    func testConfigurationCaching() async throws {
        // First call should calculate configuration
        let start1 = CFAbsoluteTimeGetCurrent()
        let config1 = await optimizer.getOptimalConfiguration(
            for: .distanceComputation,
            workSize: 1000,
            vectorDimension: 128
        )
        let time1 = CFAbsoluteTimeGetCurrent() - start1
        
        // Second call should use cache
        let start2 = CFAbsoluteTimeGetCurrent()
        let config2 = await optimizer.getOptimalConfiguration(
            for: .distanceComputation,
            workSize: 1000,
            vectorDimension: 128
        )
        let time2 = CFAbsoluteTimeGetCurrent() - start2
        
        // Verify configs are identical
        XCTAssertEqual(config1.threadsPerThreadgroup.width, config2.threadsPerThreadgroup.width)
        XCTAssertEqual(config1.threadgroupsPerGrid.width, config2.threadgroupsPerGrid.width)
        
        // Cached call should be faster
        XCTAssertLessThan(time2, time1 * 0.5)
    }
    
    func test512DimensionalOptimization() async throws {
        // Test special optimization for 512-dimensional vectors
        let config = await optimizer.getOptimalConfiguration(
            for: .distanceComputation,
            workSize: 10000,
            vectorDimension: 512
        )
        
        // Should use 64 threads for 512-dim vectors
        XCTAssertEqual(config.threadsPerThreadgroup.width, 64)
    }
    
    func test2DConfiguration() async throws {
        // Test 2D configuration for matrix operations
        let config = await optimizer.getOptimal2DConfiguration(
            for: .matrixMultiplication,
            rows: 1024,
            columns: 1024,
            tileSize: 16
        )
        
        // Verify 2D configuration
        XCTAssertGreaterThan(config.threadsPerThreadgroup.width, 0)
        XCTAssertGreaterThan(config.threadsPerThreadgroup.height, 0)
        XCTAssertGreaterThan(config.threadgroupsPerGrid.width, 0)
        XCTAssertGreaterThan(config.threadgroupsPerGrid.height, 0)
        
        // Verify tile size is aligned to wavefront
        XCTAssertEqual(config.tileSize % 32, 0) // Apple Silicon uses 32-thread warps
        
        // Verify memory calculation
        XCTAssertGreaterThan(config.sharedMemoryRequired, 0)
    }
    
    func testVariousWorkSizes() async throws {
        let workSizes = [1, 10, 100, 1000, 10000, 100000, 1000000]
        
        for workSize in workSizes {
            let config = await optimizer.getOptimalConfiguration(
                for: .distanceComputation,
                workSize: workSize,
                vectorDimension: 128
            )
            
            // Verify configuration covers all work
            let totalThreads = config.threadsPerThreadgroup.width * config.threadgroupsPerGrid.width
            XCTAssertGreaterThanOrEqual(totalThreads, workSize)
            
            // Verify threads per group doesn't exceed work size
            XCTAssertLessThanOrEqual(config.threadsPerThreadgroup.width, max(workSize, 32))
        }
    }
    
    func testSharedMemoryConstraints() async throws {
        // Test configuration with shared memory requirements
        let config = await optimizer.getOptimalConfiguration(
            for: .clustering,
            workSize: 1000,
            vectorDimension: 128,
            sharedMemoryPerThread: 1024 // 1KB per thread
        )
        
        // Verify shared memory doesn't exceed limits
        let capabilities = await device.capabilities
        XCTAssertLessThanOrEqual(
            config.sharedMemoryRequired,
            capabilities.maxThreadgroupMemoryLength
        )
    }
    
    func testDifferentOperationTypes() async throws {
        let operations: [OperationType] = [
            .distanceComputation,
            .matrixMultiplication,
            .quantization,
            .vectorNormalization,
            .clustering
        ]
        
        for operation in operations {
            let config = await optimizer.getOptimalConfiguration(
                for: operation,
                workSize: 10000
            )
            
            // Each operation should have valid configuration
            XCTAssertGreaterThan(config.threadsPerThreadgroup.width, 0)
            XCTAssertGreaterThan(config.threadgroupsPerGrid.width, 0)
            
            // Different operations should have different preferred sizes
            if operation == .matrixMultiplication {
                // Matrix multiplication should prefer larger threadgroups
                XCTAssertGreaterThanOrEqual(config.threadsPerThreadgroup.width, 128)
            }
        }
    }
    
    func testPerformanceComparison() async throws {
        // Skip if not in performance test mode
        guard ProcessInfo.processInfo.environment["PERFORMANCE_TESTS"] != nil else {
            throw XCTSkip("Performance tests disabled")
        }
        
        // Create a simple distance computation pipeline
        let functionName = "euclideanDistance"
        guard let function = await device.makeFunction(name: functionName) else {
            throw XCTSkip("Metal function not available")
        }
        
        let pipeline = try await device.makeComputePipelineState(function: function)
        
        // Benchmark different configurations
        let results = try await optimizer.benchmarkConfigurations(
            for: pipeline,
            workSize: 100000,
            testIterations: 10
        )
        
        // Verify we found improvements
        XCTAssertGreaterThan(results.speedupFactor, 1.0)
        XCTAssertFalse(results.configurations.isEmpty)
        
        // Log results
        print("Best configuration: \(results.bestConfiguration.threadsPerThreadgroup.width) threads")
        print("Speedup factor: \(results.speedupFactor)x")
    }
}
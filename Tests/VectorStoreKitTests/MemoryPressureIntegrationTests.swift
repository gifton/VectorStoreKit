// MemoryPressureIntegrationTests.swift
// VectorStoreKit
//
// Integration tests for comprehensive memory pressure handling across all components

import XCTest
@testable import VectorStoreKit
import Foundation

final class MemoryPressureIntegrationTests: XCTestCase {
    
    var memoryManager: MemoryManager!
    var batchProcessor: BatchProcessor!
    var cachingStrategy: MultiLevelCachingStrategy<SIMD16<Float>, String>!
    var memoryPool: EnhancedBufferPool!
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        
        // Initialize the memory manager
        memoryManager = MemoryManager()
        
        // Create and register components
        Task {
            await setupComponents()
        }
    }
    
    override func tearDownWithError() throws {
        memoryManager = nil
        batchProcessor = nil
        cachingStrategy = nil
        memoryPool = nil
        try super.tearDownWithError()
    }
    
    private func setupComponents() async {
        // 1. Create and register batch processor
        batchProcessor = BatchProcessor(
            configuration: BatchProcessingConfiguration(
                optimalBatchSize: 1000,
                maxConcurrentBatches: 4,
                memoryLimit: 100_000_000 // 100MB limit
            )
        )
        await memoryManager.registerBatchSizeManager(batchProcessor)
        await memoryManager.registerMemoryPressureAwareComponent(batchProcessor)
        
        // 2. Create and register caching strategy
        let cacheConfigs = [
            CacheLevelConfiguration(level: .l1, maxMemory: 10_000_000), // 10MB
            CacheLevelConfiguration(level: .l2, maxMemory: 50_000_000), // 50MB
            CacheLevelConfiguration(level: .l3, maxMemory: 100_000_000) // 100MB
        ]
        
        do {
            cachingStrategy = try await MultiLevelCachingStrategy(configurations: cacheConfigs)
            await memoryManager.registerMemoryPressureAwareComponent(cachingStrategy)
        } catch {
            XCTFail("Failed to create caching strategy: \(error)")
        }
        
        // 3. Create and register enhanced buffer pool
        memoryPool = EnhancedBufferPool(
            maxPoolSize: 200,
            defaultAlignment: 64,
            memoryManager: memoryManager
        )
        await memoryManager.registerBufferPool(memoryPool)
    }
    
    func testBasicMemoryPressureFlow() async throws {
        await setupComponents()
        
        // Initial state - should be normal
        let initialStats = await memoryManager.getMemoryStatistics()
        XCTAssertEqual(initialStats.currentPressureLevel, .normal)
        
        // Get initial batch size
        let initialBatchSize = await batchProcessor.getCurrentBatchSize()
        XCTAssertEqual(initialBatchSize, 1000)
        
        // Simulate warning level memory pressure
        await memoryManager.handleMemoryPressure(level: .warning)
        
        // Verify batch size was reduced
        let warningBatchSize = await batchProcessor.getCurrentBatchSize()
        XCTAssertLessThan(warningBatchSize, initialBatchSize)
        XCTAssertEqual(warningBatchSize, 700) // Should be reduced by 30%
        
        // Simulate critical memory pressure
        await memoryManager.handleMemoryPressure(level: .critical)
        
        // Verify batch size was further reduced
        let criticalBatchSize = await batchProcessor.getCurrentBatchSize()
        XCTAssertLessThan(criticalBatchSize, warningBatchSize)
        
        // Simulate recovery to normal
        await memoryManager.handleMemoryPressure(level: .normal)
        
        // Verify batch size was restored
        let recoveredBatchSize = await batchProcessor.getCurrentBatchSize()
        XCTAssertEqual(recoveredBatchSize, initialBatchSize)
    }
    
    func testMemoryPressureStatistics() async throws {
        await setupComponents()
        
        // Trigger memory pressure events
        await memoryManager.handleMemoryPressure(level: .warning)
        await memoryManager.handleMemoryPressure(level: .critical)
        await memoryManager.handleMemoryPressure(level: .normal)
        
        // Check overall statistics
        let managerStats = await memoryManager.getMemoryStatistics()
        XCTAssertEqual(managerStats.pressureEventCount, 3)
        XCTAssertGreaterThan(managerStats.totalHandlingTime, 0)
        
        // Check component statistics
        let componentStats = await memoryManager.getComponentStatistics()
        XCTAssertGreaterThan(componentStats.totalComponents, 0)
        XCTAssertGreaterThan(componentStats.totalPressureEvents, 0)
    }
    
    func testCoordinatedMemoryPressureResponse() async throws {
        await setupComponents()
        
        // Add some data to test with
        let testVectors = generateTestVectors(count: 100)
        
        // Process some data to establish memory usage
        let dataset = Vector512Dataset(vectors: testVectors)
        let processor: @Sendable ([Vector512]) async throws -> [String] = { batch in
            return batch.map { _ in UUID().uuidString }
        }
        
        _ = try await batchProcessor.processBatches(
            dataset: dataset,
            processor: processor
        )
        
        // Get initial memory usage
        let initialUsage = await batchProcessor.getCurrentMemoryUsage()
        XCTAssertGreaterThan(initialUsage, 0)
        
        // Trigger critical memory pressure
        await memoryManager.handleMemoryPressure(level: .critical)
        
        // Verify memory usage was reduced
        let reducedUsage = await batchProcessor.getCurrentMemoryUsage()
        XCTAssertLessThan(reducedUsage, initialUsage)
        
        // Verify all components responded
        let batchProcessorStats = await batchProcessor.getMemoryStatistics()
        XCTAssertGreaterThan(batchProcessorStats.pressureEventCount, 0)
        
        let cachingStats = await cachingStrategy.getMemoryStatistics()
        XCTAssertEqual(cachingStats.componentName, "MultiLevelCachingStrategy")
    }
    
    func testMemoryPoolIntegration() async throws {
        await setupComponents()
        
        // Allocate some buffers
        let buffer1 = try await memoryPool.acquireBuffer(size: 4096)
        let buffer2 = try await memoryPool.acquireBuffer(size: 8192)
        let buffer3 = try await memoryPool.acquireBuffer(size: 16384)
        
        // Get initial statistics
        let initialStats = await memoryPool.getStatistics()
        let initialMemoryUsage = initialStats.currentlyInUse
        XCTAssertGreaterThan(initialMemoryUsage, 0)
        
        // Trigger critical memory pressure
        await memoryManager.handleMemoryPressure(level: .critical)
        
        // Pool should have been cleared
        let postPressureStats = await memoryPool.getStatistics()
        // Note: allocated buffers are still in use, but pool should be cleared
        
        // Clean up
        await memoryPool.releaseBuffer(buffer1, size: 4096)
        await memoryPool.releaseBuffer(buffer2, size: 8192)
        await memoryPool.releaseBuffer(buffer3, size: 16384)
    }
    
    func testConcurrentMemoryPressureHandling() async throws {
        await setupComponents()
        
        // Create multiple concurrent tasks that might trigger memory pressure
        await withTaskGroup(of: Void.self) { group in
            // Task 1: Process data
            group.addTask {
                let testVectors = self.generateTestVectors(count: 50)
                let dataset = Vector512Dataset(vectors: testVectors)
                let processor: @Sendable ([Vector512]) async throws -> [String] = { batch in
                    return batch.map { _ in UUID().uuidString }
                }
                
                do {
                    _ = try await self.batchProcessor.processBatches(
                        dataset: dataset,
                        processor: processor
                    )
                } catch {
                    // Expected to potentially fail under memory pressure
                }
            }
            
            // Task 2: Trigger memory pressure
            group.addTask {
                try? await Task.sleep(for: .milliseconds(10))
                await self.memoryManager.handleMemoryPressure(level: .warning)
                
                try? await Task.sleep(for: .milliseconds(20))
                await self.memoryManager.handleMemoryPressure(level: .critical)
                
                try? await Task.sleep(for: .milliseconds(30))
                await self.memoryManager.handleMemoryPressure(level: .normal)
            }
            
            // Task 3: Monitor statistics
            group.addTask {
                for _ in 0..<5 {
                    try? await Task.sleep(for: .milliseconds(15))
                    let stats = await self.memoryManager.getMemoryStatistics()
                    // Just ensure we can get stats during pressure events
                    XCTAssertNotNil(stats)
                }
            }
        }
        
        // Verify system handled concurrent pressure correctly
        let finalStats = await memoryManager.getMemoryStatistics()
        XCTAssertGreaterThan(finalStats.pressureEventCount, 0)
    }
    
    func testMemoryPressureRecovery() async throws {
        await setupComponents()
        
        // Establish baseline
        let initialBatchSize = await batchProcessor.getCurrentBatchSize()
        
        // Apply severe memory pressure
        await memoryManager.handleMemoryPressure(level: .critical)
        let criticalBatchSize = await batchProcessor.getCurrentBatchSize()
        XCTAssertLessThan(criticalBatchSize, initialBatchSize)
        
        // Recovery should restore to normal
        await memoryManager.handleMemoryPressure(level: .normal)
        let recoveredBatchSize = await batchProcessor.getCurrentBatchSize()
        XCTAssertEqual(recoveredBatchSize, initialBatchSize)
        
        // Test gradual recovery through warning
        await memoryManager.handleMemoryPressure(level: .critical)
        await memoryManager.handleMemoryPressure(level: .warning)
        await memoryManager.handleMemoryPressure(level: .normal)
        
        let finalBatchSize = await batchProcessor.getCurrentBatchSize()
        XCTAssertEqual(finalBatchSize, initialBatchSize)
    }
    
    // MARK: - Helper Methods
    
    private func generateTestVectors(count: Int) -> [Vector512] {
        return (0..<count).map { _ in
            Vector512(repeating: Float.random(in: -1...1))
        }
    }
}
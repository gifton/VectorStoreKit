import XCTest
// MemoryPressureExample.swift
// VectorStoreKit
//
// Demonstrates comprehensive memory pressure handling across all VectorStoreKit components

import Foundation
import VectorStoreKit
import os.log

final class PressureHandlingTests: XCTestCase {
    func testMain() async throws {
        let logger = Logger(subsystem: "VectorStoreKit.Examples", category: "MemoryPressure")
        
        do {
            // Initialize the centralized memory manager
            let memoryManager = MemoryManager()
            logger.info("🧠 Initialized centralized memory manager")
            
            // Create and register batch processor
            let batchProcessor = BatchProcessor(
                configuration: BatchProcessingConfiguration(
                    optimalBatchSize: 2000,
                    maxConcurrentBatches: 6,
                    memoryLimit: 200_000_000 // 200MB limit
                )
            )
            await memoryManager.registerBatchSizeManager(batchProcessor)
            await memoryManager.registerMemoryPressureAwareComponent(batchProcessor)
            logger.info("📦 Registered batch processor with memory management")
            
            // Create and register multi-level caching strategy
            let cacheConfigs = [
                CacheLevelConfiguration(level: .l1, maxMemory: 20_000_000),  // 20MB
                CacheLevelConfiguration(level: .l2, maxMemory: 100_000_000), // 100MB
                CacheLevelConfiguration(level: .l3, maxMemory: 200_000_000)  // 200MB
            ]
            
            let cachingStrategy = try await MultiLevelCachingStrategy<SIMD16<Float>, String>(
                configurations: cacheConfigs
            )
            await memoryManager.registerMemoryPressureAwareComponent(cachingStrategy)
            logger.info("💾 Registered multi-level caching strategy")
            
            // Create and register enhanced buffer pool
            let bufferPool = EnhancedBufferPool(
                maxPoolSize: 500,
                defaultAlignment: 64,
                memoryManager: memoryManager
            )
            await memoryManager.registerBufferPool(bufferPool)
            logger.info("🔧 Registered enhanced buffer pool")
            
            // Demonstrate normal operation
            logger.info("🟢 Starting normal operation demonstration...")
            await demonstrateNormalOperation(
                memoryManager: memoryManager,
                batchProcessor: batchProcessor,
                cachingStrategy: cachingStrategy,
                bufferPool: bufferPool
            )
            
            // Demonstrate memory pressure scenarios
            logger.info("🟡 Demonstrating warning-level memory pressure...")
            await demonstrateMemoryPressure(
                level: .warning,
                memoryManager: memoryManager,
                batchProcessor: batchProcessor
            )
            
            logger.info("🔴 Demonstrating critical-level memory pressure...")
            await demonstrateMemoryPressure(
                level: .critical,
                memoryManager: memoryManager,
                batchProcessor: batchProcessor
            )
            
            // Demonstrate recovery
            logger.info("🟢 Demonstrating memory pressure recovery...")
            await demonstrateMemoryRecovery(
                memoryManager: memoryManager,
                batchProcessor: batchProcessor
            )
            
            // Show comprehensive statistics
            await showComprehensiveStatistics(memoryManager: memoryManager)
            
            logger.info("✅ Memory pressure integration example completed successfully")
            
        } catch {
            logger.error("❌ Example failed: \(error)")
        }
    }
    
    // MARK: - Demonstration Functions
    
    func testDemonstrateNormalOperation(
        memoryManager: MemoryManager,
        batchProcessor: BatchProcessor,
        cachingStrategy: MultiLevelCachingStrategy<SIMD16<Float>, String>,
        bufferPool: EnhancedBufferPool
    ) async {
        let logger = Logger(subsystem: "VectorStoreKit.Examples", category: "NormalOperation")
        
        // Show initial statistics
        let initialStats = await memoryManager.getMemoryStatistics()
        logger.info("Initial memory pressure level: \(initialStats.currentPressureLevel.rawValue)")
        logger.info("Initial memory usage: \(initialStats.currentMemoryUsage) bytes")
        
        // Demonstrate batch processing
        let testVectors = generateTestVectors(count: 5000)
        let dataset = Vector512Dataset(vectors: testVectors)
        
        let processor: @Sendable ([Vector512]) async throws -> [String] = { batch in
            logger.debug("Processing batch of \(batch.count) vectors")
            // Simulate some processing time
            try await Task.sleep(for: .milliseconds(10))
            return batch.map { _ in UUID().uuidString }
        }
        
        do {
            let results = try await batchProcessor.processBatches(
                dataset: dataset,
                processor: processor
            ) { progress in
                if progress.currentBatch % 10 == 0 {
                    logger.info("Progress: \(String(format: "%.1f", progress.percentComplete))% complete")
                }
            }
            logger.info("Processed \(results.count) vectors successfully")
        } catch {
            logger.error("Batch processing failed: \(error)")
        }
        
        // Demonstrate caching
        for i in 0..<100 {
            let vector = SIMD16<Float>(repeating: Float(i))
            await cachingStrategy.set(id: "vector_\(i)", vector: vector, metadata: "metadata_\(i)")
        }
        logger.info("Cached 100 vectors in multi-level cache")
        
        // Demonstrate buffer allocation
        var buffers: [UnsafeMutableRawPointer] = []
        for _ in 0..<20 {
            if let buffer = try? await bufferPool.acquireBuffer(size: 8192) {
                buffers.append(buffer)
            }
        }
        logger.info("Allocated \(buffers.count) buffers from pool")
        
        // Clean up buffers
        for buffer in buffers {
            await bufferPool.releaseBuffer(buffer, size: 8192)
        }
        logger.info("Released all allocated buffers")
    }
    
    func testDemonstrateMemoryPressure(
        level: SystemMemoryPressure,
        memoryManager: MemoryManager,
        batchProcessor: BatchProcessor
    ) async {
        let logger = Logger(subsystem: "VectorStoreKit.Examples", category: "MemoryPressure")
        
        // Get baseline metrics
        let beforeBatchSize = await batchProcessor.getCurrentBatchSize()
        let beforeMemoryUsage = await batchProcessor.getCurrentMemoryUsage()
        
        logger.info("Before pressure - Batch size: \(beforeBatchSize), Memory usage: \(beforeMemoryUsage)")
        
        // Trigger memory pressure
        await memoryManager.handleMemoryPressure(level: level)
        
        // Show the effects
        let afterBatchSize = await batchProcessor.getCurrentBatchSize()
        let afterMemoryUsage = await batchProcessor.getCurrentMemoryUsage()
        
        logger.info("After \(level.rawValue) pressure - Batch size: \(afterBatchSize), Memory usage: \(afterMemoryUsage)")
        
        let reductionPercentage = Int(100 * (1.0 - Float(afterBatchSize) / Float(beforeBatchSize)))
        logger.info("Batch size reduced by \(reductionPercentage)%")
        
        // Get updated statistics
        let stats = await memoryManager.getMemoryStatistics()
        logger.info("Pressure events handled: \(stats.pressureEventCount)")
        logger.info("Total handling time: \(String(format: "%.3f", stats.totalHandlingTime))s")
    }
    
    func testDemonstrateMemoryRecovery(
        memoryManager: MemoryManager,
        batchProcessor: BatchProcessor
    ) async {
        let logger = Logger(subsystem: "VectorStoreKit.Examples", category: "MemoryRecovery")
        
        // Trigger recovery
        await memoryManager.handleMemoryPressure(level: .normal)
        
        // Show recovery effects
        let recoveredBatchSize = await batchProcessor.getCurrentBatchSize()
        let recoveredMemoryUsage = await batchProcessor.getCurrentMemoryUsage()
        
        logger.info("Recovered - Batch size: \(recoveredBatchSize), Memory usage: \(recoveredMemoryUsage)")
        
        // Verify components are back to normal operation
        let testVectors = generateTestVectors(count: 1000)
        let dataset = Vector512Dataset(vectors: testVectors)
        
        let processor: @Sendable ([Vector512]) async throws -> [String] = { batch in
            return batch.map { _ in "recovered" }
        }
        
        do {
            let results = try await batchProcessor.processBatches(
                dataset: dataset,
                processor: processor
            )
            logger.info("Successfully processed \(results.count) vectors after recovery")
        } catch {
            logger.error("Post-recovery processing failed: \(error)")
        }
    }
    
    func testShowComprehensiveStatistics(memoryManager: MemoryManager) async {
        let logger = Logger(subsystem: "VectorStoreKit.Examples", category: "Statistics")
        
        // Get overall memory manager statistics
        let managerStats = await memoryManager.getMemoryStatistics()
        
        logger.info("=== Memory Manager Statistics ===")
        logger.info("Total pressure events: \(managerStats.pressureEventCount)")
        logger.info("Total handling time: \(String(format: "%.3f", managerStats.totalHandlingTime))s")
        logger.info("Average handling time: \(String(format: "%.3f", managerStats.averageHandlingTime))s")
        logger.info("Current memory usage: \(managerStats.currentMemoryUsage) bytes")
        logger.info("Memory utilization: \(String(format: "%.1f", managerStats.memoryUtilization))%")
        logger.info("Buffers released: \(managerStats.buffersReleased)")
        logger.info("Cache entries evicted: \(managerStats.cacheEntriesEvicted)")
        logger.info("Batch size reductions: \(managerStats.batchSizeReductions)")
        
        // Get component-specific statistics
        let componentStats = await memoryManager.getComponentStatistics()
        
        logger.info("=== Component Statistics ===")
        logger.info("Total registered components: \(componentStats.totalComponents)")
        logger.info("Total memory usage: \(componentStats.totalMemoryUsage) bytes")
        logger.info("Average memory per component: \(componentStats.averageMemoryPerComponent) bytes")
        logger.info("Total pressure events across components: \(componentStats.totalPressureEvents)")
        logger.info("Average pressure events per component: \(String(format: "%.1f", componentStats.averagePressureEventsPerComponent))")
        
        // Show individual component details
        for componentStat in componentStats.componentStats {
            logger.info("Component '\(componentStat.componentName)':")
            logger.info("  Memory usage: \(componentStat.currentMemoryUsage) bytes")
            logger.info("  Peak usage: \(componentStat.peakMemoryUsage) bytes")
            logger.info("  Pressure events: \(componentStat.pressureEventCount)")
            if let lastHandled = componentStat.lastPressureHandled {
                logger.info("  Last pressure handled: \(lastHandled)")
            }
        }
    }
    
    // MARK: - Helper Functions
    
    func testGenerateTestVectors(count: Int) -> [Vector512] {
        return (0..<count).map { i in
            let values = (0..<512).map { j in
                Float(sin(Double(i + j) * 0.01))
            }
            return Vector512(values)
        }
    }
}
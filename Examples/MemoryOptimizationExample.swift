// VectorStoreKit: Memory Optimization Example
//
// Demonstrates memory-efficient ML operations with proper cleanup

import Foundation
import VectorStoreKit
@preconcurrency import Metal

@main
struct MemoryOptimizationExample {
    static func main() async throws {
        print("=== VectorStoreKit Memory Optimization Example ===\n")
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        // Create memory manager
        let memoryManager = MLMemoryManager(device: device, maxMemoryGB: 1.0)
        
        // Create ML pipeline with memory management
        let pipeline = try MetalMLPipeline(device: device)
        
        // Create pressure-aware buffer pool
        let bufferPool = PressureAwareBufferPool(
            device: device,
            memoryManager: memoryManager,
            maxPoolSize: 50
        )
        
        print("1. Demonstrating memory-efficient training")
        try await demonstrateEfficientTraining(
            pipeline: pipeline,
            memoryManager: memoryManager,
            bufferPool: bufferPool
        )
        
        print("\n2. Demonstrating gradient checkpointing")
        try await demonstrateGradientCheckpointing(pipeline: pipeline)
        
        print("\n3. Demonstrating memory pressure handling")
        try await demonstrateMemoryPressureHandling(
            memoryManager: memoryManager,
            bufferPool: bufferPool
        )
        
        print("\n4. Demonstrating memory profiling")
        try await demonstrateMemoryProfiling(pipeline: pipeline)
        
        print("\n5. Demonstrating batch processing with memory limits")
        try await demonstrateBatchProcessing(
            pipeline: pipeline,
            memoryManager: memoryManager
        )
        
        // Print final statistics
        await printMemoryStatistics(memoryManager: memoryManager, bufferPool: bufferPool)
    }
    
    // MARK: - Memory-Efficient Training
    
    static func demonstrateEfficientTraining(
        pipeline: MetalMLPipeline,
        memoryManager: MLMemoryManager,
        bufferPool: PressureAwareBufferPool
    ) async throws {
        print("Setting up neural network with memory management...")
        
        // Create a simple network
        let network = try await NeuralNetwork(metalPipeline: pipeline)
        
        // Add layers with explicit memory management
        let inputSize = 512
        let hiddenSize = 256
        let outputSize = 10
        
        // Dense layer with managed buffers
        let denseLayer = try await DenseLayer(
            inputSize: inputSize,
            outputSize: hiddenSize,
            activation: .relu,
            name: "dense1",
            metalPipeline: pipeline
        )
        await network.addLayer(denseLayer)
        
        // LSTM layer with gradient checkpointing
        let lstmConfig = LSTMConfig(
            hiddenSize: hiddenSize,
            returnSequences: false,
            dropout: 0.1
        )
        let lstmLayer = try await LSTMLayer(
            inputSize: hiddenSize,
            config: lstmConfig,
            name: "lstm1",
            metalPipeline: pipeline
        )
        await network.addLayer(lstmLayer)
        
        // Output layer
        let outputLayer = try await DenseLayer(
            inputSize: hiddenSize,
            outputSize: outputSize,
            activation: .softmax,
            name: "output",
            metalPipeline: pipeline
        )
        await network.addLayer(outputLayer)
        
        // Enable gradient checkpointing
        await network.enableGradientCheckpointing()
        
        // Training with memory-aware batching
        let batchSize = 32
        let sequenceLength = 20
        
        print("Training with memory-efficient batching...")
        
        for epoch in 1...3 {
            print("Epoch \(epoch):")
            
            // Check memory before batch
            let (usedBytes, percentage) = await memoryManager.getCurrentUsage()
            print("  Memory usage: \(formatBytes(usedBytes)) (\(String(format: "%.1f%%", percentage * 100)))")
            
            // Process mini-batches
            for batch in 0..<5 {
                // Allocate input buffer from pool
                let inputBuffer = try await bufferPool.acquire(
                    size: batchSize * sequenceLength * inputSize
                ).metalBuffer
                
                // Initialize with random data
                let inputPtr = inputBuffer.buffer.contents().bindMemory(
                    to: Float.self,
                    capacity: inputBuffer.count
                )
                for i in 0..<inputBuffer.count {
                    inputPtr[i] = Float.random(in: -1...1)
                }
                
                // Allocate target buffer
                let targetBuffer = try await bufferPool.acquire(
                    size: batchSize * outputSize
                ).metalBuffer
                
                // Initialize targets (one-hot encoded)
                let targetPtr = targetBuffer.buffer.contents().bindMemory(
                    to: Float.self,
                    capacity: targetBuffer.count
                )
                for i in 0..<targetBuffer.count {
                    targetPtr[i] = 0
                }
                // Set random class labels
                for i in 0..<batchSize {
                    let classIndex = Int.random(in: 0..<outputSize)
                    targetPtr[i * outputSize + classIndex] = 1.0
                }
                
                // Forward pass
                let output = try await network.forward(inputBuffer)
                
                // Compute loss (simplified)
                let loss = computeSimpleLoss(
                    predictions: output,
                    targets: targetBuffer
                )
                
                print("    Batch \(batch + 1): Loss = \(loss)")
                
                // Buffers will be automatically released when ManagedMetalBuffer goes out of scope
            }
            
            // Force cleanup between epochs
            if epoch < 3 {
                await memoryManager.forceCleanup()
            }
        }
        
        print("Training completed with efficient memory usage")
    }
    
    // MARK: - Gradient Checkpointing
    
    static func demonstrateGradientCheckpointing(pipeline: MetalMLPipeline) async throws {
        print("Demonstrating gradient checkpointing...")
        
        let checkpointer = GradientCheckpointer(metalPipeline: pipeline)
        
        // Simulate forward pass with checkpointing
        let layerSizes = [1024, 512, 256, 128]
        var activations: [MetalBuffer] = []
        
        print("Forward pass with checkpointing:")
        
        // Initial input
        var currentActivation = try await pipeline.allocateBuffer(size: layerSizes[0])
        
        for (i, size) in layerSizes.enumerated() {
            // Process layer
            let nextSize = i < layerSizes.count - 1 ? layerSizes[i + 1] : size
            let nextActivation = try await pipeline.allocateBuffer(size: nextSize)
            
            // Checkpoint every other layer to save memory
            if i % 2 == 0 {
                try await checkpointer.checkpoint(currentActivation, key: "layer_\(i)")
                print("  Checkpointed layer \(i) activation")
                
                // Release original activation
                await pipeline.releaseBuffer(currentActivation)
            } else {
                // Keep activation for backward pass
                activations.append(currentActivation)
            }
            
            currentActivation = nextActivation
        }
        
        print("\nBackward pass with recomputation:")
        
        // Simulate backward pass
        for i in (0..<layerSizes.count).reversed() {
            if i % 2 == 0 {
                // Retrieve checkpointed activation
                if let checkpoint = await checkpointer.retrieve(key: "layer_\(i)") {
                    print("  Retrieved checkpoint for layer \(i)")
                    // Use checkpoint for gradient computation
                    await pipeline.releaseBuffer(checkpoint)
                }
            } else {
                // Use stored activation
                if !activations.isEmpty {
                    let activation = activations.removeLast()
                    print("  Using stored activation for layer \(i)")
                    await pipeline.releaseBuffer(activation)
                }
            }
        }
        
        // Clear remaining checkpoints
        await checkpointer.clearAll()
        
        let memoryUsage = await checkpointer.getMemoryUsage()
        print("Checkpoint memory usage: \(formatBytes(memoryUsage))")
    }
    
    // MARK: - Memory Pressure Handling
    
    static func demonstrateMemoryPressureHandling(
        memoryManager: MLMemoryManager,
        bufferPool: PressureAwareBufferPool
    ) async throws {
        print("Simulating memory pressure scenarios...")
        
        // Register cleanup callback
        await memoryManager.registerCleanupCallback { level in
            print("  Memory pressure callback triggered: \(level)")
        }
        
        // Allocate buffers until we hit pressure
        var managedBuffers: [ManagedMetalBuffer] = []
        let bufferSize = 1024 * 1024 // 1M floats = 4MB
        
        print("Allocating buffers to trigger memory pressure...")
        
        for i in 0..<100 {
            do {
                let buffer = try await bufferPool.acquire(size: bufferSize)
                managedBuffers.append(buffer)
                
                if i % 10 == 0 {
                    let stats = await memoryManager.getStatistics()
                    print("  Allocated \(i + 1) buffers, usage: \(formatBytes(stats.currentUsage))")
                }
            } catch {
                print("  Allocation failed at buffer \(i + 1): \(error)")
                break
            }
        }
        
        // Check pool statistics
        let poolStats = await bufferPool.getStatistics()
        print("\nPool statistics:")
        print("  Hit rate: \(String(format: "%.1f%%", poolStats.hitRate * 100))")
        print("  Total buffers: \(poolStats.totalAllocated)")
        
        // Clear buffers
        managedBuffers.removeAll()
        print("\nReleased all buffers")
    }
    
    // MARK: - Memory Profiling
    
    static func demonstrateMemoryProfiling(pipeline: MetalMLPipeline) async throws {
        print("Profiling memory usage...")
        
        let profiler = MLMemoryProfiler()
        
        // Profile various operations
        let operations = [
            ("Small allocation", 1024),
            ("Medium allocation", 1024 * 256),
            ("Large allocation", 1024 * 1024),
            ("Very large allocation", 1024 * 1024 * 4)
        ]
        
        for (name, size) in operations {
            await profiler.recordAllocation(size: size * MemoryLayout<Float>.stride, source: name)
            
            // Simulate some work
            let buffer = try await pipeline.allocateBuffer(size: size)
            
            // Initialize buffer
            let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: size)
            for i in 0..<min(100, size) {
                ptr[i] = Float(i)
            }
            
            // Release
            await pipeline.releaseBuffer(buffer)
            await profiler.recordDeallocation(size: size * MemoryLayout<Float>.stride)
        }
        
        // Generate report
        let report = await profiler.generateReport()
        
        print("\nMemory Profile Report:")
        print("  Total allocations: \(report.totalAllocations)")
        print("  Peak memory usage: \(formatBytes(report.peakMemoryUsage))")
        print("  Average allocation size: \(formatBytes(Int(report.averageAllocationSize)))")
        print("  Largest allocation: \(formatBytes(report.largestAllocation))")
        
        print("\nAllocations by source:")
        for (source, count) in report.allocationsBySource.sorted(by: { $0.value > $1.value }) {
            print("  \(source): \(count)")
        }
    }
    
    // MARK: - Batch Processing with Memory Limits
    
    static func demonstrateBatchProcessing(
        pipeline: MetalMLPipeline,
        memoryManager: MLMemoryManager
    ) async throws {
        print("Processing large dataset with memory constraints...")
        
        let totalSamples = 10000
        let featureSize = 512
        let maxBatchSize = 128
        
        // Calculate optimal batch size based on available memory
        let (_, memoryPercentage) = await memoryManager.getCurrentUsage()
        let availablePercentage = 1.0 - memoryPercentage
        let optimalBatchSize = min(
            maxBatchSize,
            Int(Double(maxBatchSize) * availablePercentage)
        )
        
        print("Optimal batch size: \(optimalBatchSize) (available memory: \(String(format: "%.1f%%", availablePercentage * 100)))")
        
        // Process in batches
        let operations = await pipeline.getOperations()
        var processedSamples = 0
        
        while processedSamples < totalSamples {
            let batchSize = min(optimalBatchSize, totalSamples - processedSamples)
            
            // Check if we can allocate
            let requiredMemory = batchSize * featureSize * MemoryLayout<Float>.stride
            if await memoryManager.canAllocate(size: requiredMemory) {
                // Process batch
                let batchBuffer = try await pipeline.allocateBuffer(size: batchSize * featureSize)
                
                // Simulate processing
                try await operations.scaleBuffer(
                    batchBuffer,
                    scale: 0.5,
                    output: batchBuffer
                )
                
                processedSamples += batchSize
                
                if processedSamples % 1000 == 0 {
                    print("  Processed \(processedSamples)/\(totalSamples) samples")
                }
                
                // Release buffer
                await pipeline.releaseBuffer(batchBuffer)
            } else {
                print("  Memory limit reached, forcing cleanup...")
                await memoryManager.forceCleanup()
            }
        }
        
        print("Completed processing \(totalSamples) samples")
    }
    
    // MARK: - Helper Functions
    
    static func computeSimpleLoss(predictions: MetalBuffer, targets: MetalBuffer) -> Float {
        // Simple MSE loss calculation
        let predPtr = predictions.buffer.contents().bindMemory(
            to: Float.self,
            capacity: predictions.count
        )
        let targetPtr = targets.buffer.contents().bindMemory(
            to: Float.self,
            capacity: targets.count
        )
        
        var loss: Float = 0
        for i in 0..<min(predictions.count, targets.count) {
            let diff = predPtr[i] - targetPtr[i]
            loss += diff * diff
        }
        
        return loss / Float(predictions.count)
    }
    
    static func printMemoryStatistics(
        memoryManager: MLMemoryManager,
        bufferPool: PressureAwareBufferPool
    ) async {
        print("\n=== Final Memory Statistics ===")
        
        let stats = await memoryManager.getStatistics()
        print("Memory Manager:")
        print("  Current usage: \(formatBytes(stats.currentUsage))")
        print("  Peak usage: \(formatBytes(stats.peakUsage))")
        print("  Buffer count: \(stats.bufferCount)")
        print("  Allocations: \(stats.allocationCount)")
        print("  Deallocations: \(stats.deallocationCount)")
        print("  Pressure events: \(stats.pressureEventCount)")
        print("  Current pressure level: \(stats.pressureLevel)")
        
        let poolStats = await bufferPool.getStatistics()
        print("\nBuffer Pool:")
        print("  Hit rate: \(String(format: "%.1f%%", poolStats.hitRate * 100))")
        print("  Pooled buffers: \(poolStats.totalAllocated)")
        
        // System memory info
        MemoryDebugger.printMemoryUsage()
    }
    
    static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}
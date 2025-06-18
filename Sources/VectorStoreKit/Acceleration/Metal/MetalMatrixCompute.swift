// VectorStoreKit: Metal Matrix Computation
//
// Hardware-accelerated matrix operations

import Foundation
@preconcurrency import Metal
@preconcurrency import MetalPerformanceShaders
import simd
import os.log

/// Hardware-accelerated matrix operations engine
public actor MetalMatrixCompute {
    
    // MARK: - Properties
    
    private let device: MetalDevice
    private let bufferPool: MetalBufferPool
    private let pipelineManager: MetalPipelineManager
    private let profiler: MetalProfiler?
    private let commandBufferPool: MetalCommandBufferPool
    private let batchOptimizer: MetalBatchOptimizer
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalMatrixCompute")
    
    // MARK: - Initialization
    
    public init(
        device: MetalDevice,
        bufferPool: MetalBufferPool,
        pipelineManager: MetalPipelineManager,
        profiler: MetalProfiler? = nil,
        commandBufferPool: MetalCommandBufferPool? = nil,
        batchOptimizer: MetalBatchOptimizer? = nil
    ) async {
        self.device = device
        self.bufferPool = bufferPool
        self.pipelineManager = pipelineManager
        self.profiler = profiler
        
        if let commandBufferPool = commandBufferPool {
            self.commandBufferPool = commandBufferPool
        } else {
            self.commandBufferPool = MetalCommandBufferPool(device: device, profiler: profiler)
        }
        
        if let batchOptimizer = batchOptimizer {
            self.batchOptimizer = batchOptimizer
        } else {
            self.batchOptimizer = await MetalBatchOptimizer(device: device, profiler: profiler)
        }
    }
    
    // MARK: - Matrix Multiplication
    
    /// Perform matrix multiplication using Metal Performance Shaders or custom kernels
    public func matrixMultiply(
        matrixA: [[Float]],
        matrixB: [[Float]],
        useAMX: Bool = true
    ) async throws -> [[Float]] {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Validate dimensions
        guard !matrixA.isEmpty && !matrixB.isEmpty else {
            throw MetalMatrixError.emptyMatrix
        }
        
        let rowsA = matrixA.count
        let colsA = matrixA[0].count
        let rowsB = matrixB.count
        let colsB = matrixB[0].count
        
        guard colsA == rowsB else {
            throw MetalMatrixError.dimensionMismatch(
                a: (rowsA, colsA),
                b: (rowsB, colsB)
            )
        }
        
        logger.debug("Matrix multiply: (\(rowsA)×\(colsA)) × (\(rowsB)×\(colsB))")
        
        // Choose optimal execution path
        let result: [[Float]]
        
        if await device.supports(feature: .mps) {
            result = try await matrixMultiplyMPS(
                matrixA: matrixA,
                matrixB: matrixB
            )
        } else {
            result = try await matrixMultiplyCustom(
                matrixA: matrixA,
                matrixB: matrixB
            )
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        await profiler?.recordOperation(.matrixMultiplication, duration: duration, dataSize: rowsA * colsB)
        
        return result
    }
    
    /// Batch matrix multiplication for multiple matrix pairs with optimized GPU utilization
    public func batchMatrixMultiply(
        pairs: [(matrixA: [[Float]], matrixB: [[Float]])]
    ) async throws -> [[[Float]]] {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Get optimal batch configuration
        let avgMatrixSize = pairs.map { $0.matrixA.count * $0.matrixA[0].count }.reduce(0, +) / pairs.count
        let batchConfig = await batchOptimizer.getOptimalBatchSize(
            operation: .matrixMultiplication,
            dataType: .float32,
            vectorDimension: avgMatrixSize,
            totalElements: pairs.count
        )
        
        logger.debug("Using batch size: \(batchConfig.batchSize) for \(pairs.count) matrix operations")
        
        // Group similar-sized operations for GPU efficiency
        let groupedPairs = pairs.enumerated().sorted { lhs, rhs in
            let lhsSize = lhs.element.matrixA.count * lhs.element.matrixA[0].count
            let rhsSize = rhs.element.matrixA.count * rhs.element.matrixA[0].count
            return lhsSize < rhsSize
        }
        
        var results = Array(repeating: [[Float]](), count: pairs.count)
        
        // Use optimized batch processing with double buffering if available
        if batchConfig.useDoubleBUffering && pairs.count > batchConfig.batchSize {
            results = try await batchMatrixMultiplyWithDoubleBuffering(
                groupedPairs: groupedPairs,
                batchConfig: batchConfig
            )
        } else {
            // Standard batch processing
            for batchStart in stride(from: 0, to: groupedPairs.count, by: batchConfig.batchSize) {
                let batchEnd = min(batchStart + batchConfig.batchSize, groupedPairs.count)
                let batch = groupedPairs[batchStart..<batchEnd]
                
                // Process batch using fused operations
                try await processBatchFused(batch: Array(batch), results: &results)
            }
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        let throughput = Double(pairs.count) / duration
        
        // Record performance metrics
        await batchOptimizer.recordBatchPerformance(
            operation: .matrixMultiplication,
            dataType: .float32,
            vectorDimension: avgMatrixSize,
            batchSize: batchConfig.batchSize,
            executionTime: duration,
            throughput: throughput
        )
        
        await profiler?.recordOperation(
            .matrixMultiplication,
            duration: duration,
            dataSize: pairs.count
        )
        
        logger.debug("Batch matrix multiply completed: \(pairs.count) operations in \(duration)s (\(throughput) ops/sec)")
        
        return results
    }
    
    // MARK: - Matrix Operations
    
    /// Matrix addition
    public func add(
        _ matrixA: [[Float]],
        _ matrixB: [[Float]]
    ) async throws -> [[Float]] {
        return try await elementWise(.add, matrixA: matrixA, matrixB: matrixB)
    }
    
    /// Matrix subtraction
    public func subtract(
        _ matrixA: [[Float]],
        _ matrixB: [[Float]]
    ) async throws -> [[Float]] {
        return try await elementWise(.subtract, matrixA: matrixA, matrixB: matrixB)
    }
    
    /// Matrix scalar multiplication
    public func scalarMultiply(
        _ matrix: [[Float]],
        scalar: Float
    ) async throws -> [[Float]] {
        return try await elementWise(.multiply, matrixA: matrix, scalar: scalar)
    }
    
    /// Matrix row sum
    public func rowSum(_ matrix: [[Float]]) async throws -> [Float] {
        let rows = matrix.count
        let cols = matrix[0].count
        
        // For small matrices, use CPU
        if rows * cols < 1000 {
            return matrix.map { row in row.reduce(0, +) }
        }
        
        // Flatten matrix
        let flat = matrix.flatMap { $0 }
        let inputBuffer = try await bufferPool.getBuffer(for: flat)
        let outputBuffer = try await bufferPool.getBuffer(size: rows * MemoryLayout<Float>.size)
        
        defer {
            Task {
                await bufferPool.returnBuffer(inputBuffer)
                await bufferPool.returnBuffer(outputBuffer)
            }
        }
        
        // Get pipeline
        let pipeline = try await pipelineManager.getPipeline(functionName: "matrixRowSum")
        
        // Get command buffer from pool
        let commandBuffer = try await commandBufferPool.getCommandBuffer(label: "MatrixRowSum")
        
        guard let encoder = commandBuffer.buffer.makeComputeCommandEncoder() else {
            throw MetalMatrixError.commandBufferCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        
        var colsVal = UInt32(cols)
        encoder.setBytes(&colsVal, length: MemoryLayout<UInt32>.size, index: 2)
        
        // Dispatch
        let threadsPerGroup = min(256, rows)
        let threadGroups = (rows + threadsPerGroup - 1) / threadsPerGroup
        
        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
        
        // Execute
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        // Extract result
        let resultPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: rows)
        return Array(UnsafeBufferPointer(start: resultPointer, count: rows))
    }
    
    /// Transpose a matrix
    public func transpose(_ matrix: [[Float]]) async throws -> [[Float]] {
        guard !matrix.isEmpty else {
            throw MetalMatrixError.emptyMatrix
        }
        
        let rows = matrix.count
        let cols = matrix[0].count
        
        // For small matrices, CPU is efficient
        if rows * cols < 1000 {
            return transposeCPU(matrix)
        }
        
        // Use Metal for larger matrices
        let flattened = matrix.flatMap { $0 }
        let inputBuffer = try await bufferPool.getBuffer(for: flattened)
        let outputBuffer = try await bufferPool.getBuffer(size: flattened.count * MemoryLayout<Float>.size)
        
        defer {
            Task {
                await bufferPool.returnBuffer(inputBuffer)
                await bufferPool.returnBuffer(outputBuffer)
            }
        }
        
        // Execute transpose kernel
        try await executeTranspose(
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            rows: rows,
            cols: cols
        )
        
        // Extract result
        let resultPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: rows * cols)
        let resultArray = Array(UnsafeBufferPointer(start: resultPointer, count: rows * cols))
        
        // Reshape to 2D
        return (0..<cols).map { col in
            (0..<rows).map { row in
                resultArray[col * rows + row]
            }
        }
    }
    
    /// Element-wise matrix operations
    public func elementWise(
        _ operation: ElementWiseOperation,
        matrixA: [[Float]],
        matrixB: [[Float]]? = nil,
        scalar: Float? = nil
    ) async throws -> [[Float]] {
        
        guard !matrixA.isEmpty else {
            throw MetalMatrixError.emptyMatrix
        }
        
        let rows = matrixA.count
        let cols = matrixA[0].count
        
        // Validate dimensions if matrix B is provided
        if let matrixB = matrixB {
            guard matrixB.count == rows && matrixB[0].count == cols else {
                throw MetalMatrixError.dimensionMismatch(
                    a: (rows, cols),
                    b: (matrixB.count, matrixB[0].count)
                )
            }
        }
        
        // Execute operation
        return try await executeElementWise(
            operation: operation,
            matrixA: matrixA,
            matrixB: matrixB,
            scalar: scalar
        )
    }
    
    // MARK: - Private Methods
    
    private func matrixMultiplyMPS(
        matrixA: [[Float]],
        matrixB: [[Float]]
    ) async throws -> [[Float]] {
        
        let rowsA = matrixA.count
        let colsA = matrixA[0].count
        let colsB = matrixB[0].count
        
        // Flatten matrices
        let flatA = matrixA.flatMap { $0 }
        let flatB = matrixB.flatMap { $0 }
        
        // Create buffers
        let bufferA = try await bufferPool.getBuffer(for: flatA)
        let bufferB = try await bufferPool.getBuffer(for: flatB)
        let bufferC = try await bufferPool.getBuffer(size: rowsA * colsB * MemoryLayout<Float>.size)
        
        defer {
            Task {
                await bufferPool.returnBuffer(bufferA)
                await bufferPool.returnBuffer(bufferB)
                await bufferPool.returnBuffer(bufferC)
            }
        }
        
        // Create matrix descriptors
        let descA = MPSMatrixDescriptor(
            rows: rowsA,
            columns: colsA,
            rowBytes: colsA * MemoryLayout<Float>.size,
            dataType: .float32
        )
        
        let descB = MPSMatrixDescriptor(
            rows: colsA,
            columns: colsB,
            rowBytes: colsB * MemoryLayout<Float>.size,
            dataType: .float32
        )
        
        let descC = MPSMatrixDescriptor(
            rows: rowsA,
            columns: colsB,
            rowBytes: colsB * MemoryLayout<Float>.size,
            dataType: .float32
        )
        
        // Create matrices
        let matA = MPSMatrix(buffer: bufferA, descriptor: descA)
        let matB = MPSMatrix(buffer: bufferB, descriptor: descB)
        let matC = MPSMatrix(buffer: bufferC, descriptor: descC)
        
        // Create and execute matrix multiplication
        guard let commandBuffer = await device.makeCommandBuffer() else {
            throw MetalMatrixError.commandBufferCreationFailed
        }
        
        let multiplication = MPSMatrixMultiplication(
            device: await device.device,
            transposeLeft: false,
            transposeRight: false,
            resultRows: rowsA,
            resultColumns: colsB,
            interiorColumns: colsA,
            alpha: 1.0,
            beta: 0.0
        )
        
        multiplication.encode(
            commandBuffer: commandBuffer,
            leftMatrix: matA,
            rightMatrix: matB,
            resultMatrix: matC
        )
        
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        // Extract result
        let resultPointer = bufferC.contents().bindMemory(to: Float.self, capacity: rowsA * colsB)
        let resultArray = Array(UnsafeBufferPointer(start: resultPointer, count: rowsA * colsB))
        
        // Reshape to 2D
        return (0..<rowsA).map { row in
            (0..<colsB).map { col in
                resultArray[row * colsB + col]
            }
        }
    }
    
    private func matrixMultiplyCustom(
        matrixA: [[Float]],
        matrixB: [[Float]]
    ) async throws -> [[Float]] {
        
        let rowsA = matrixA.count
        let colsA = matrixA[0].count
        let colsB = matrixB[0].count
        
        // Flatten matrices
        let flatA = matrixA.flatMap { $0 }
        let flatB = matrixB.flatMap { $0 }
        
        // Create buffers
        let bufferA = try await bufferPool.getBuffer(for: flatA)
        let bufferB = try await bufferPool.getBuffer(for: flatB)
        let bufferC = try await bufferPool.getBuffer(size: rowsA * colsB * MemoryLayout<Float>.size)
        
        defer {
            Task {
                await bufferPool.returnBuffer(bufferA)
                await bufferPool.returnBuffer(bufferB)
                await bufferPool.returnBuffer(bufferC)
            }
        }
        
        // Get pipeline state
        let pipeline = try await pipelineManager.getPipeline(functionName: "tiledMatrixMultiply")
        
        // Get command buffer from pool
        let commandBuffer = try await commandBufferPool.getCommandBuffer(label: "MatrixRowSum")
        
        guard let encoder = commandBuffer.buffer.makeComputeCommandEncoder() else {
            throw MetalMatrixError.commandBufferCreationFailed
        }
        
        // Set pipeline state
        encoder.setComputePipelineState(pipeline)
        
        // Set buffers
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)
        
        // Set dimensions
        var rowsAVal = UInt32(rowsA)
        var colsAVal = UInt32(colsA)
        var colsBVal = UInt32(colsB)
        encoder.setBytes(&rowsAVal, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&colsAVal, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&colsBVal, length: MemoryLayout<UInt32>.size, index: 5)
        
        // Calculate thread groups
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (colsB + 15) / 16,
            height: (rowsA + 15) / 16,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        // Execute
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        // Extract result
        let resultPointer = bufferC.contents().bindMemory(to: Float.self, capacity: rowsA * colsB)
        let resultArray = Array(UnsafeBufferPointer(start: resultPointer, count: rowsA * colsB))
        
        // Reshape to 2D
        return (0..<rowsA).map { row in
            (0..<colsB).map { col in
                resultArray[row * colsB + col]
            }
        }
    }
    
    private func executeTranspose(
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        rows: Int,
        cols: Int
    ) async throws {
        
        // Get pipeline state
        let pipeline = try await pipelineManager.getPipeline(functionName: "matrixTransposeCoalesced")
        
        // Get command buffer from pool
        let commandBuffer = try await commandBufferPool.getCommandBuffer(label: "MatrixRowSum")
        
        guard let encoder = commandBuffer.buffer.makeComputeCommandEncoder() else {
            throw MetalMatrixError.commandBufferCreationFailed
        }
        
        // Set pipeline state
        encoder.setComputePipelineState(pipeline)
        
        // Set buffers
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        
        // Set dimensions
        var rowsVal = UInt32(rows)
        var colsVal = UInt32(cols)
        encoder.setBytes(&rowsVal, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&colsVal, length: MemoryLayout<UInt32>.size, index: 3)
        
        // Calculate thread groups
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (cols + 15) / 16,
            height: (rows + 15) / 16,
            depth: 1
        )
        
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        // Execute
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
    }
    
    private func executeElementWise(
        operation: ElementWiseOperation,
        matrixA: [[Float]],
        matrixB: [[Float]]?,
        scalar: Float?
    ) async throws -> [[Float]] {
        
        // For small matrices or simple operations, use CPU
        let elementCount = matrixA.count * matrixA[0].count
        if elementCount < 1000 {
            return elementWiseCPU(
                operation: operation,
                matrixA: matrixA,
                matrixB: matrixB,
                scalar: scalar
            )
        }
        
        let rows = matrixA.count
        let cols = matrixA[0].count
        
        // Flatten matrix A
        let flatA = matrixA.flatMap { $0 }
        let bufferA = try await bufferPool.getBuffer(for: flatA)
        let bufferResult = try await bufferPool.getBuffer(size: elementCount * MemoryLayout<Float>.size)
        var bufferB: MTLBuffer?
        
        defer {
            Task {
                await bufferPool.returnBuffer(bufferA)
                await bufferPool.returnBuffer(bufferResult)
                if let bufferB = bufferB {
                    await bufferPool.returnBuffer(bufferB)
                }
            }
        }
        
        // Get appropriate pipeline
        let pipelineName: String
        switch operation {
        case .add:
            pipelineName = "matrixAdd"
        case .subtract:
            pipelineName = "matrixSubtract"
        case .multiply:
            if matrixB != nil {
                pipelineName = "matrixMultiplyElementwise"
            } else {
                pipelineName = "matrixScalarMultiply"
            }
        case .divide:
            pipelineName = "matrixDivideElementwise"
        case .maximum:
            pipelineName = "matrixMaxElementwise"
        case .minimum:
            pipelineName = "matrixMinElementwise"
        }
        
        let pipeline = try await pipelineManager.getPipeline(functionName: pipelineName)
        
        // Get command buffer from pool
        let commandBuffer = try await commandBufferPool.getCommandBuffer(label: "MatrixRowSum")
        
        guard let encoder = commandBuffer.buffer.makeComputeCommandEncoder() else {
            throw MetalMatrixError.commandBufferCreationFailed
        }
        
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        
        // Handle second operand
        if let matrixB = matrixB {
            let flatB = matrixB.flatMap { $0 }
            bufferB = try await bufferPool.getBuffer(for: flatB)
            encoder.setBuffer(bufferB!, offset: 0, index: 1)
        } else if let scalar = scalar, operation == .multiply {
            var scalarVal = scalar
            encoder.setBytes(&scalarVal, length: MemoryLayout<Float>.size, index: 1)
        }
        
        encoder.setBuffer(bufferResult, offset: 0, index: 2)
        
        // Dispatch
        let threadsPerGroup = 256
        let threadGroups = (elementCount + threadsPerGroup - 1) / threadsPerGroup
        
        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        )
        encoder.endEncoding()
        
        // Execute
        await withCheckedContinuation { continuation in
            commandBuffer.addCompletedHandler { _ in
                continuation.resume()
            }
            commandBuffer.commit()
        }
        
        // Extract result
        let resultPointer = bufferResult.contents().bindMemory(to: Float.self, capacity: elementCount)
        let resultArray = Array(UnsafeBufferPointer(start: resultPointer, count: elementCount))
        
        // Reshape to 2D
        return (0..<rows).map { row in
            (0..<cols).map { col in
                resultArray[row * cols + col]
            }
        }
    }
    
    // MARK: - Optimized Batch Processing
    
    private func batchMatrixMultiplyWithDoubleBuffering(
        groupedPairs: [(offset: Int, element: (matrixA: [[Float]], matrixB: [[Float]]))],
        batchConfig: BatchConfiguration
    ) async throws -> [[[Float]]] {
        
        var results = Array(repeating: [[Float]](), count: groupedPairs.count)
        let batchSize = batchConfig.batchSize
        
        // Create double buffer setup
        var currentBatch: [(Int, (matrixA: [[Float]], matrixB: [[Float]]))] = []
        var nextBatch: [(Int, (matrixA: [[Float]], matrixB: [[Float]]))] = []
        
        for batchStart in stride(from: 0, to: groupedPairs.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, groupedPairs.count)
            
            // Prepare next batch while processing current
            if batchStart > 0 {
                // Process current batch
                try await processBatchFused(batch: currentBatch, results: &results)
            }
            
            // Swap buffers
            swap(&currentBatch, &nextBatch)
            
            // Load next batch
            currentBatch = Array(groupedPairs[batchStart..<batchEnd])
        }
        
        // Process final batch
        if !currentBatch.isEmpty {
            try await processBatchFused(batch: currentBatch, results: &results)
        }
        
        return results
    }
    
    private func processBatchFused(
        batch: [(offset: Int, element: (matrixA: [[Float]], matrixB: [[Float]]))],
        results: inout [[[Float]]]
    ) async throws {
        
        // Process batch with minimal GPU synchronization
        try await withThrowingTaskGroup(of: (Int, [[Float]]).self) { group in
            // Use a single command buffer for the entire batch if possible
            let shouldFuse = batch.allSatisfy { pair in
                pair.element.matrixA.count == batch[0].element.matrixA.count &&
                pair.element.matrixA[0].count == batch[0].element.matrixA[0].count &&
                pair.element.matrixB[0].count == batch[0].element.matrixB[0].count
            }
            
            if shouldFuse && batch.count > 1 {
                // Fused batch processing
                let fusedResults = try await matrixMultiplyFused(batch: batch)
                for (idx, (originalIndex, _)) in batch.enumerated() {
                    results[originalIndex] = fusedResults[idx]
                }
            } else {
                // Individual processing
                for (originalIndex, pair) in batch {
                    group.addTask {
                        let result = try await self.matrixMultiply(
                            matrixA: pair.matrixA,
                            matrixB: pair.matrixB
                        )
                        return (originalIndex, result)
                    }
                }
                
                for try await (index, result) in group {
                    results[index] = result
                }
            }
        }
    }
    
    private func matrixMultiplyFused(
        batch: [(offset: Int, element: (matrixA: [[Float]], matrixB: [[Float]]))]
    ) async throws -> [[[Float]]] {
        
        // This would implement a fused matrix multiplication kernel
        // that processes multiple matrix pairs in a single GPU dispatch
        // For now, fall back to individual processing
        
        var results: [[[Float]]] = []
        for (_, pair) in batch {
            let result = try await matrixMultiply(
                matrixA: pair.matrixA,
                matrixB: pair.matrixB
            )
            results.append(result)
        }
        return results
    }
    
    // MARK: - CPU Fallbacks
    
    private func transposeCPU(_ matrix: [[Float]]) -> [[Float]] {
        let rows = matrix.count
        let cols = matrix[0].count
        
        return (0..<cols).map { col in
            (0..<rows).map { row in
                matrix[row][col]
            }
        }
    }
    
    private func matrixMultiplyCPU(matrixA: [[Float]], matrixB: [[Float]]) -> [[Float]] {
        let rowsA = matrixA.count
        let colsA = matrixA[0].count
        let colsB = matrixB[0].count
        
        var result = Array(repeating: Array(repeating: Float(0), count: colsB), count: rowsA)
        
        for i in 0..<rowsA {
            for j in 0..<colsB {
                for k in 0..<colsA {
                    result[i][j] += matrixA[i][k] * matrixB[k][j]
                }
            }
        }
        
        return result
    }
    
    private func elementWiseCPU(
        operation: ElementWiseOperation,
        matrixA: [[Float]],
        matrixB: [[Float]]?,
        scalar: Float?
    ) -> [[Float]] {
        
        let rows = matrixA.count
        let cols = matrixA[0].count
        
        return (0..<rows).map { row in
            (0..<cols).map { col in
                let a = matrixA[row][col]
                let b = matrixB?[row][col] ?? scalar ?? 0
                
                switch operation {
                case .add:
                    return a + b
                case .subtract:
                    return a - b
                case .multiply:
                    return a * b
                case .divide:
                    return b != 0 ? a / b : 0
                case .maximum:
                    return max(a, b)
                case .minimum:
                    return min(a, b)
                }
            }
        }
    }
}

// MARK: - Supporting Types

/// Matrix operation errors
public enum MetalMatrixError: Error, LocalizedError {
    case emptyMatrix
    case dimensionMismatch(a: (Int, Int), b: (Int, Int))
    case commandBufferCreationFailed
    case invalidOperation
    
    public var errorDescription: String? {
        switch self {
        case .emptyMatrix:
            return "Empty matrix provided"
        case .dimensionMismatch(let a, let b):
            return "Matrix dimensions incompatible: \(a) × \(b)"
        case .commandBufferCreationFailed:
            return "Failed to create command buffer"
        case .invalidOperation:
            return "Invalid matrix operation"
        }
    }
}

/// Element-wise operations
public enum ElementWiseOperation {
    case add
    case subtract
    case multiply
    case divide
    case maximum
    case minimum
}

/// Matrix computation statistics
public struct MatrixComputeStatistics: Sendable {
    public let totalOperations: UInt64
    public let gpuOperations: UInt64
    public let cpuFallbacks: UInt64
    public let averageOperationTime: TimeInterval
    public let peakMemoryUsage: Int
}
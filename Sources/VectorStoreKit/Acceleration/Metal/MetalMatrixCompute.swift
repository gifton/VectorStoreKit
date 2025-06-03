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
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalMatrixCompute")
    
    // MARK: - Initialization
    
    public init(
        device: MetalDevice,
        bufferPool: MetalBufferPool,
        pipelineManager: MetalPipelineManager,
        profiler: MetalProfiler? = nil
    ) {
        self.device = device
        self.bufferPool = bufferPool
        self.pipelineManager = pipelineManager
        self.profiler = profiler
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
    
    /// Batch matrix multiplication for multiple matrix pairs
    public func batchMatrixMultiply(
        pairs: [(matrixA: [[Float]], matrixB: [[Float]])]
    ) async throws -> [[[Float]]] {
        
        return try await withThrowingTaskGroup(of: (Int, [[Float]]).self) { group in
            for (index, pair) in pairs.enumerated() {
                group.addTask {
                    let result = try await self.matrixMultiply(
                        matrixA: pair.matrixA,
                        matrixB: pair.matrixB
                    )
                    return (index, result)
                }
            }
            
            var results = Array(repeating: [[Float]](), count: pairs.count)
            for try await (index, result) in group {
                results[index] = result
            }
            
            return results
        }
    }
    
    // MARK: - Matrix Operations
    
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
        
        // Use custom kernel for matrix multiplication
        let _ = try await pipelineManager.getPipeline(for: .matrixMultiply)
        
        // Implementation would follow similar pattern to MPS version
        // For now, fallback to CPU
        return matrixMultiplyCPU(matrixA: matrixA, matrixB: matrixB)
    }
    
    private func executeTranspose(
        inputBuffer: MTLBuffer,
        outputBuffer: MTLBuffer,
        rows: Int,
        cols: Int
    ) async throws {
        
        // In a real implementation, this would use a transpose kernel
        // For now, we'll use CPU fallback
        let inputPointer = inputBuffer.contents().bindMemory(to: Float.self, capacity: rows * cols)
        let outputPointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: rows * cols)
        
        for row in 0..<rows {
            for col in 0..<cols {
                outputPointer[col * rows + row] = inputPointer[row * cols + col]
            }
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
        
        // Use Metal for larger operations
        let _ = try await pipelineManager.getPipeline(for: .elementwiseOperations)
        
        // Implementation would encode operation type and execute
        // For now, fallback to CPU
        return elementWiseCPU(
            operation: operation,
            matrixA: matrixA,
            matrixB: matrixB,
            scalar: scalar
        )
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
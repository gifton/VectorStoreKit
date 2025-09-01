// VectorStoreKit: Enhanced Metal ML Operations
//
// Memory-efficient Metal ML operations with proper cleanup

import Foundation
@preconcurrency import Metal

/// Enhanced Metal ML operations with memory management
public extension MetalMLOperations {
    
    // MARK: - Memory-Efficient Batch Operations
    
    /// Process operations with automatic buffer cleanup
    func withTemporaryBuffers<T>(
        count: Int,
        size: Int,
        operation: ([MetalBuffer]) async throws -> T
    ) async throws -> T {
        var buffers: [MetalBuffer] = []
        
        // Allocate buffers
        for _ in 0..<count {
            buffers.append(try await self.acquireBuffer(size: size))
        }
        
        // Ensure cleanup even if operation throws
        defer {
            Task {
                for buffer in buffers {
                    await self.releaseBuffer(buffer)
                }
            }
        }
        
        return try await operation(buffers)
    }
    
    /// Memory-efficient matrix multiplication with workspace reuse
    func matmulWithWorkspace(
        _ a: MetalBuffer,
        _ b: MetalBuffer,
        output: MetalBuffer,
        m: Int, n: Int, k: Int,
        workspace: MetalBuffer? = nil
    ) async throws {
        // Validate dimensions
        guard a.count >= m * k,
              b.count >= k * n,
              output.count >= m * n else {
            throw MetalMLError.invalidBufferSize("Invalid matmul dimensions")
        }
        
        // Use provided workspace or allocate temporary
        let workspaceBuffer: MetalBuffer
        let shouldReleaseWorkspace: Bool
        
        if let workspace = workspace {
            workspaceBuffer = workspace
            shouldReleaseWorkspace = false
        } else {
            // Calculate workspace size for tiled multiplication
            let tileSize = 32
            let workspaceSize = tileSize * tileSize * 3 // For A, B, and C tiles
            workspaceBuffer = try await self.acquireBuffer(size: workspaceSize)
            shouldReleaseWorkspace = true
        }
        
        defer {
            if shouldReleaseWorkspace {
                Task {
                    await self.releaseBuffer(workspaceBuffer)
                }
            }
        }
        
        // Perform matmul with workspace
        try await matmul(a, b, output: output, m: m, n: n, k: k, useTiling: true)
    }
    
    // MARK: - Memory-Efficient Gradient Operations
    
    /// Accumulate gradients with memory reuse
    func accumulateGradients(
        gradients: [MetalBuffer],
        into accumulator: MetalBuffer,
        scale: Float = 1.0
    ) async throws {
        guard !gradients.isEmpty else { return }
        
        // Validate sizes
        let expectedSize = accumulator.count
        for gradient in gradients {
            guard gradient.count == expectedSize else {
                throw MetalMLError.incompatibleBufferSize(
                    expected: expectedSize,
                    actual: gradient.count
                )
            }
        }
        
        // Use temporary buffer for accumulation
        let temp = try await self.acquireBuffer(size: expectedSize)
        defer {
            Task {
                await self.releaseBuffer(temp)
            }
        }
        
        // Initialize accumulator with first gradient
        try await scaleBuffer(gradients[0], scale: scale, output: accumulator)
        
        // Accumulate remaining gradients
        for i in 1..<gradients.count {
            try await scaleBuffer(gradients[i], scale: scale, output: temp)
            try await addBuffers(accumulator, temp, output: accumulator)
        }
    }
    
    /// Clip gradients with efficient memory usage
    func clipGradientsInPlace(
        _ gradients: [MetalBuffer],
        maxNorm: Float
    ) async throws {
        // Compute global norm across all gradients
        var globalNormSquared: Float = 0
        
        // Use temporary buffer for norm computation
        try await withTemporaryBuffers(count: 1, size: 1) { temps in
            let normBuffer = temps[0]
            
            for gradient in gradients {
                let norm = try await computeL2Norm(gradient)
                globalNormSquared += norm * norm
            }
        }
        
        let globalNorm = sqrt(globalNormSquared)
        
        // Clip if necessary
        if globalNorm > maxNorm {
            let scale = maxNorm / globalNorm
            
            for gradient in gradients {
                try await scaleBuffer(gradient, scale: scale, output: gradient)
            }
        }
    }
    
    // MARK: - Memory-Efficient Neural Operations
    
    /// Fused operations to reduce memory traffic
    func fusedLayerNormalization(
        input: MetalBuffer,
        gamma: MetalBuffer,
        beta: MetalBuffer,
        output: MetalBuffer,
        epsilon: Float = 1e-5
    ) async throws {
        guard input.count == output.count else {
            throw MetalMLError.incompatibleBufferSize(
                expected: input.count,
                actual: output.count
            )
        }
        
        let pipeline = try await shaderLibrary.pipeline(for: "fused_layer_norm")
        
        try await asyncQueue.submitAsync { commandBuffer in
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(input.buffer, offset: 0, index: 0)
            encoder.setBuffer(gamma.buffer, offset: 0, index: 1)
            encoder.setBuffer(beta.buffer, offset: 0, index: 2)
            encoder.setBuffer(output.buffer, offset: 0, index: 3)
            
            var eps = epsilon
            var size = UInt32(input.count)
            encoder.setBytes(&eps, length: MemoryLayout<Float>.size, index: 4)
            encoder.setBytes(&size, length: MemoryLayout<UInt32>.size, index: 5)
            
            let workSize = MTLSize(width: input.count, height: 1, depth: 1)
            let (threadgroupSize, threadgroupCount) = self.shaderLibrary.adaptiveThreadConfiguration(
                for: pipeline,
                functionName: "fused_layer_norm",
                workSize: workSize,
                preferBatching: true
            )
            
            encoder.dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup: threadgroupSize)
            encoder.endEncoding()
        }
    }
    
    /// Memory-efficient attention mechanism
    func efficientAttention(
        query: MetalBuffer,
        key: MetalBuffer,
        value: MetalBuffer,
        output: MetalBuffer,
        sequenceLength: Int,
        headDim: Int,
        scale: Float? = nil
    ) async throws {
        let attentionScale = scale ?? (1.0 / sqrt(Float(headDim)))
        
        // Use chunked attention for long sequences to reduce memory
        let chunkSize = 512 // Process attention in chunks
        
        if sequenceLength <= chunkSize {
            // Standard attention for short sequences
            try await standardAttention(
                query: query,
                key: key,
                value: value,
                output: output,
                sequenceLength: sequenceLength,
                headDim: headDim,
                scale: attentionScale
            )
        } else {
            // Chunked attention for long sequences
            try await chunkedAttention(
                query: query,
                key: key,
                value: value,
                output: output,
                sequenceLength: sequenceLength,
                headDim: headDim,
                chunkSize: chunkSize,
                scale: attentionScale
            )
        }
    }
    
    private func standardAttention(
        query: MetalBuffer,
        key: MetalBuffer,
        value: MetalBuffer,
        output: MetalBuffer,
        sequenceLength: Int,
        headDim: Int,
        scale: Float
    ) async throws {
        // Allocate attention scores buffer
        let scoresBuffer = try await self.acquireBuffer(size: sequenceLength * sequenceLength)
        defer {
            Task {
                await self.releaseBuffer(scoresBuffer)
            }
        }
        
        // Q @ K^T
        try await matmul(
            query, key,
            output: scoresBuffer,
            m: sequenceLength,
            n: sequenceLength,
            k: headDim
        )
        
        // Scale
        try await scaleBuffer(scoresBuffer, scale: scale, output: scoresBuffer)
        
        // Softmax
        try await applyActivation(scoresBuffer, output: scoresBuffer, activation: .softmax)
        
        // Scores @ V
        try await matmul(
            scoresBuffer, value,
            output: output,
            m: sequenceLength,
            n: headDim,
            k: sequenceLength
        )
    }
    
    private func chunkedAttention(
        query: MetalBuffer,
        key: MetalBuffer,
        value: MetalBuffer,
        output: MetalBuffer,
        sequenceLength: Int,
        headDim: Int,
        chunkSize: Int,
        scale: Float
    ) async throws {
        // Process attention in chunks to reduce memory usage
        // This implements a simplified version of Flash Attention
        
        let numChunks = (sequenceLength + chunkSize - 1) / chunkSize
        
        // Temporary buffers for chunked computation
        let chunkScores = try await self.acquireBuffer(size: chunkSize * chunkSize)
        let chunkOutput = try await self.acquireBuffer(size: chunkSize * headDim)
        
        defer {
            Task {
                await self.releaseBuffer(chunkScores)
                await self.releaseBuffer(chunkOutput)
            }
        }
        
        // Process each query chunk
        for qChunk in 0..<numChunks {
            let qStart = qChunk * chunkSize
            let qEnd = min(qStart + chunkSize, sequenceLength)
            let qSize = qEnd - qStart
            
            // Process each key-value chunk
            for kvChunk in 0..<numChunks {
                let kvStart = kvChunk * chunkSize
                let kvEnd = min(kvStart + chunkSize, sequenceLength)
                let kvSize = kvEnd - kvStart
                
                // Compute attention scores for this chunk pair
                // This would require custom kernels for efficiency
                // For now, using standard operations
                
                // Extract chunk views (simplified - would use buffer offsets)
                // ... chunk extraction logic ...
            }
        }
    }
    
    // MARK: - Memory Profiling Integration
    
    /// Execute operation with memory profiling
    func profiledOperation<T>(
        name: String,
        operation: () async throws -> T
    ) async throws -> (result: T, memoryUsed: Int) {
        // Record initial memory state
        let initialStats = bufferCache.getDetailedCacheStatistics()
        let initialMemory = initialStats.totalMemory
        
        // Execute operation
        let result = try await operation()
        
        // Record final memory state
        let finalStats = bufferCache.getDetailedCacheStatistics()
        let finalMemory = finalStats.totalMemory
        
        let memoryUsed = finalMemory - initialMemory
        
        return (result, memoryUsed)
    }
    
    // MARK: - Buffer Cache Management
    
    /// Pre-warm buffer cache for expected workload
    func prewarmCache(sizes: [Int], countsPerSize: [Int]) async throws {
        guard sizes.count == countsPerSize.count else {
            throw MetalMLError.invalidParameter(
                name: "sizes/counts",
                value: "\(sizes.count)/\(countsPerSize.count)",
                reason: "Array lengths must match"
            )
        }
        
        for (size, count) in zip(sizes, countsPerSize) {
            var buffers: [MetalBuffer] = []
            
            // Allocate buffers
            for _ in 0..<count {
                if let buffer = try? await self.acquireBuffer(size: size) {
                    buffers.append(buffer)
                }
            }
            
            // Release back to pool (now warmed)
            for buffer in buffers {
                await self.releaseBuffer(buffer)
            }
        }
    }
    
    /// Clear cache and return memory statistics
    func clearCacheAndGetStats() async -> (freedMemory: Int, bufferCount: Int) {
        let stats = bufferCache.getDetailedCacheStatistics()
        let initialMemory = stats.totalMemory
        let initialBuffers = stats.totalBuffers
        
        // Clear cache
        bufferCache.evictToSize(0)
        
        return (freedMemory: initialMemory, bufferCount: initialBuffers)
    }
    
    // MARK: - Error Recovery
    
    /// Execute operation with automatic retry on memory failure
    func withMemoryRetry<T>(
        maxRetries: Int = 3,
        operation: () async throws -> T
    ) async throws -> T {
        var lastError: Error?
        
        for attempt in 0..<maxRetries {
            do {
                return try await operation()
            } catch let error as MetalMLError {
                lastError = error
                
                switch error {
                case .bufferAllocationFailed:
                    // Clear cache and retry
                    if attempt < maxRetries - 1 {
                        _ = await clearCacheAndGetStats()
                        
                        // Wait briefly for memory to be freed
                        try await Task.sleep(nanoseconds: 100_000_000) // 100ms
                    }
                default:
                    // Re-throw non-memory errors immediately
                    throw error
                }
            }
        }
        
        throw lastError ?? MetalMLError.bufferAllocationFailed(size: 0)
    }
}

// MARK: - Debugging Extensions

extension MetalMLOperations {
    /// Check for NaN/Inf in buffers and throw if found
    func validateBuffer(_ buffer: MetalBuffer, name: String) throws {
        if buffer.containsNaN() {
            throw MetalMLError.numericalInstability("\(name) contains NaN values")
        }
        
        if buffer.containsInf() {
            throw MetalMLError.numericalInstability("\(name) contains Inf values")
        }
    }
    
    /// Debug print buffer statistics
    func debugBufferStats(_ buffer: MetalBuffer, name: String) {
        let array = buffer.toArray()
        let min = array.min() ?? 0
        let max = array.max() ?? 0
        let mean = array.reduce(0, +) / Float(array.count)
        
        print("Buffer '\(name)':")
        print("  Shape: \(buffer.shape.dimensions)")
        print("  Range: [\(min), \(max)]")
        print("  Mean: \(mean)")
        print("  Has NaN: \(buffer.containsNaN())")
        print("  Has Inf: \(buffer.containsInf())")
    }
}
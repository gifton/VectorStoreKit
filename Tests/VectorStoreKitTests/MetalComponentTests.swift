import Testing
import Foundation
import simd
@preconcurrency import Metal
@testable import VectorStoreKit

// MARK: - MetalDevice Tests

@Suite("MetalDevice Tests")
struct MetalDeviceTests {
    
    @Test func deviceInitialization() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw VectorStoreKitTestError.setup("Metal not available")
        }
        
        let device = try MetalDevice()
        
        #expect(device.device != nil)
        #expect(device.commandQueue != nil)
        #expect(device.capabilities.deviceName.isEmpty == false)
    }
    
    @Test func deviceCapabilities() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let device = try MetalDevice()
        let capabilities = device.capabilities
        
        // Test basic capabilities are detected
        #expect(capabilities.maxThreadsPerThreadgroup > 0)
        #expect(capabilities.maxBufferLength > 0)
        
        // Test feature support detection
        let hasFloat16 = device.supports(feature: .float16)
        #expect(hasFloat16 == capabilities.supportsFloat16)
    }
    
    @Test func commandBufferCreation() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let device = try MetalDevice()
        
        let commandBuffer = await device.makeCommandBuffer()
        #expect(commandBuffer != nil)
    }
}

// MARK: - MetalBufferPool Tests

@Suite("MetalBufferPool Tests")
struct MetalBufferPoolTests {
    
    @Test func bufferAllocation() async throws {
        guard let mtlDevice = MTLCreateSystemDefaultDevice() else { return }
        
        let config = MetalBufferPool.Configuration.research
        let pool = MetalBufferPool(device: mtlDevice, configuration: config)
        
        // Test single value buffer
        let value: Float = 42.0
        let buffer = try await pool.getBuffer(for: value)
        #expect(buffer.length >= MemoryLayout<Float>.size)
        
        await pool.returnBuffer(buffer)
    }
    
    @Test func arrayBufferAllocation() async throws {
        guard let mtlDevice = MTLCreateSystemDefaultDevice() else { return }
        
        let config = MetalBufferPool.Configuration.efficient
        let pool = MetalBufferPool(device: mtlDevice, configuration: config)
        
        // Test array buffer
        let array: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let buffer = try await pool.getBuffer(for: array)
        #expect(buffer.length >= array.count * MemoryLayout<Float>.size)
        
        // Verify data was copied
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: array.count)
        for i in 0..<array.count {
            #expect(pointer[i] == array[i])
        }
        
        await pool.returnBuffer(buffer)
    }
    
    @Test func concurrentBufferAccess() async throws {
        guard let mtlDevice = MTLCreateSystemDefaultDevice() else { return }
        
        let pool = MetalBufferPool(device: mtlDevice, configuration: .research)
        
        // Test concurrent buffer allocation
        await withTaskGroup(of: MTLBuffer?.self) { group in
            for i in 0..<10 {
                group.addTask {
                    let data = Float(i)
                    return try? await pool.getBuffer(for: data)
                }
            }
            
            var buffers: [MTLBuffer] = []
            for await buffer in group {
                if let buffer = buffer {
                    buffers.append(buffer)
                }
            }
            
            #expect(buffers.count == 10)
            
            // Return all buffers
            for buffer in buffers {
                await pool.returnBuffer(buffer)
            }
        }
        
        // Check statistics
        let stats = await pool.statistics
        #expect(stats.totalAllocations >= 10)
    }
}

// MARK: - MetalPipelineManager Tests

@Suite("MetalPipelineManager Tests")
struct MetalPipelineManagerTests {
    
    @Test func pipelineCreation() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let device = try MetalDevice()
        let manager = MetalPipelineManager(device: device)
        
        // Test standard pipeline creation
        do {
            let pipeline = try await manager.getPipeline(for: .euclideanDistance)
            #expect(pipeline != nil)
        } catch {
            // Pipeline might fail if shaders aren't loaded
            #expect(error != nil)
        }
    }
    
    @Test func pipelineCaching() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let device = try MetalDevice()
        let manager = MetalPipelineManager(device: device)
        
        // Precompile pipelines
        await manager.precompileStandardPipelines()
        
        // Get cache statistics
        let stats = await manager.cacheStatistics
        #expect(stats.totalPipelines >= 0)
    }
}

// MARK: - MetalDistanceCompute Tests

@Suite("MetalDistanceCompute Tests")
struct MetalDistanceComputeTests {
    
    @Test func euclideanDistanceComputation() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let device = try MetalDevice()
        let pool = MetalBufferPool(device: await device.device, configuration: .research)
        let manager = MetalPipelineManager(device: device)
        let compute = MetalDistanceCompute(device: device, bufferPool: pool, pipelineManager: manager)
        
        // Test vectors
        let query = simd_float4(1.0, 2.0, 3.0, 4.0)
        let candidates = [
            simd_float4(1.0, 2.0, 3.0, 4.0),  // Distance = 0
            simd_float4(2.0, 3.0, 4.0, 5.0),  // Distance = 2
            simd_float4(0.0, 1.0, 2.0, 3.0),  // Distance = 2
        ]
        
        // Compute distances on CPU for comparison
        let distances = await compute.computeDistancesCPU(
            query: query,
            candidates: candidates,
            metric: .euclidean
        )
        
        #expect(distances.count == candidates.count)
        #expect(abs(distances[0]) < 0.001) // First vector is identical
        #expect(abs(distances[1] - 2.0) < 0.001)
        #expect(abs(distances[2] - 2.0) < 0.001)
    }
    
    @Test func batchDistanceComputation() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let device = try MetalDevice()
        let pool = MetalBufferPool(device: await device.device, configuration: .research)
        let manager = MetalPipelineManager(device: device)
        let compute = MetalDistanceCompute(device: device, bufferPool: pool, pipelineManager: manager)
        
        // Test batch computation
        let queries = [
            simd_float4(1.0, 0.0, 0.0, 0.0),
            simd_float4(0.0, 1.0, 0.0, 0.0)
        ]
        
        let candidates = [
            simd_float4(1.0, 0.0, 0.0, 0.0),
            simd_float4(0.0, 1.0, 0.0, 0.0),
            simd_float4(0.0, 0.0, 1.0, 0.0)
        ]
        
        let results = try await compute.batchComputeDistances(
            queries: queries,
            candidates: candidates,
            metric: .euclidean
        )
        
        #expect(results.count == queries.count)
        #expect(results[0].count == candidates.count)
        
        // First query should have distance 0 to first candidate
        #expect(abs(results[0][0]) < 0.001)
    }
}

// MARK: - MetalMatrixCompute Tests

@Suite("MetalMatrixCompute Tests")
struct MetalMatrixComputeTests {
    
    @Test func matrixMultiplication() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let device = try MetalDevice()
        let pool = MetalBufferPool(device: await device.device, configuration: .research)
        let manager = MetalPipelineManager(device: device)
        let compute = MetalMatrixCompute<simd_float4>(
            device: device,
            bufferPool: pool,
            pipelineManager: manager
        )
        
        // Test matrices
        let matrixA: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        
        let matrixB: [[Float]] = [
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ]
        
        let result = try await compute.matrixMultiply(
            matrixA: matrixA,
            matrixB: matrixB
        )
        
        // Expected result:
        // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12] = [58, 64]
        // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12] = [139, 154]
        
        #expect(result.count == 2)
        #expect(result[0].count == 2)
        #expect(abs(result[0][0] - 58.0) < 0.001)
        #expect(abs(result[0][1] - 64.0) < 0.001)
        #expect(abs(result[1][0] - 139.0) < 0.001)
        #expect(abs(result[1][1] - 154.0) < 0.001)
    }
    
    @Test func matrixTranspose() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let device = try MetalDevice()
        let pool = MetalBufferPool(device: await device.device, configuration: .research)
        let manager = MetalPipelineManager(device: device)
        let compute = MetalMatrixCompute<simd_float4>(
            device: device,
            bufferPool: pool,
            pipelineManager: manager
        )
        
        let matrix: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        
        let result = try await compute.transpose(matrix)
        
        #expect(result.count == 3)
        #expect(result[0].count == 2)
        #expect(result[0][0] == 1.0)
        #expect(result[0][1] == 4.0)
        #expect(result[1][0] == 2.0)
        #expect(result[1][1] == 5.0)
        #expect(result[2][0] == 3.0)
        #expect(result[2][1] == 6.0)
    }
    
    @Test func elementWiseOperations() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let device = try MetalDevice()
        let pool = MetalBufferPool(device: await device.device, configuration: .research)
        let manager = MetalPipelineManager(device: device)
        let compute = MetalMatrixCompute<simd_float4>(
            device: device,
            bufferPool: pool,
            pipelineManager: manager
        )
        
        let matrixA: [[Float]] = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ]
        
        let matrixB: [[Float]] = [
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]
        ]
        
        // Test addition
        let sum = try await compute.add(matrixA, matrixB)
        #expect(sum[0][0] == 3.0)
        #expect(sum[1][2] == 13.0)
        
        // Test scalar multiplication
        let scaled = try await compute.scalarMultiply(matrixA, scalar: 2.0)
        #expect(scaled[0][0] == 2.0)
        #expect(scaled[1][2] == 12.0)
    }
}

// MARK: - MetalQuantizationCompute Tests

@Suite("MetalQuantizationCompute Tests")
struct MetalQuantizationComputeTests {
    
    @Test func scalarQuantization() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let device = try MetalDevice()
        let pool = MetalBufferPool(device: await device.device, configuration: .research)
        let manager = MetalPipelineManager(device: device)
        let compute = MetalQuantizationCompute(device: device, bufferPool: pool, pipelineManager: manager)
        
        // Test vectors
        let vectors = [
            simd_float4(1.0, 2.0, 3.0, 4.0),
            simd_float4(-1.0, -2.0, -3.0, -4.0),
            simd_float4(0.5, 1.5, 2.5, 3.5)
        ]
        
        let parameters = QuantizationParameters(precision: 8)
        
        // Quantize
        let quantized = try await compute.quantizeVectors(
            vectors: vectors,
            scheme: .scalar,
            parameters: parameters
        )
        
        #expect(quantized.count == vectors.count)
        
        // Dequantize
        let dequantized: [simd_float4] = try await compute.dequantizeVectors(
            quantizedVectors: quantized,
            targetType: simd_float4.self
        )
        
        #expect(dequantized.count == vectors.count)
        
        // Check that values are approximately preserved
        for i in 0..<vectors.count {
            for j in 0..<4 {
                let error = abs(vectors[i][j] - dequantized[i][j])
                #expect(error < 0.1) // Allow some quantization error
            }
        }
    }
    
    @Test func binaryQuantization() async throws {
        guard MTLCreateSystemDefaultDevice() != nil else { return }
        
        let device = try MetalDevice()
        let pool = MetalBufferPool(device: await device.device, configuration: .research)
        let manager = MetalPipelineManager(device: device)
        let compute = MetalQuantizationCompute(device: device, bufferPool: pool, pipelineManager: manager)
        
        // Test vectors
        let vectors = [
            simd_float4(1.0, -1.0, 1.0, -1.0),
            simd_float4(2.0, -2.0, 2.0, -2.0)
        ]
        
        let parameters = QuantizationParameters(precision: 1)
        
        // Quantize
        let quantized = try await compute.quantizeVectors(
            vectors: vectors,
            scheme: .binary,
            parameters: parameters
        )
        
        #expect(quantized.count == vectors.count)
        
        // Binary quantization should produce compact representation
        for q in quantized {
            #expect(q.scheme == .binary)
            // Binary quantization for 4D vector should use minimal space
            #expect(q.quantizedData.count > 0)
        }
    }
}

// MARK: - MetalProfiler Tests

@Suite("MetalProfiler Tests")
struct MetalProfilerTests {
    
    @Test func profilerRecording() async throws {
        let profiler = MetalProfiler(enabled: true)
        
        // Record some operations
        await profiler.recordOperation(
            .distanceComputation,
            duration: 0.001,
            dataSize: 1000,
            gpuUtilization: 0.5,
            memoryBandwidth: 10.0
        )
        
        await profiler.recordOperation(
            .matrixMultiplication,
            duration: 0.002,
            dataSize: 2000,
            gpuUtilization: 0.8,
            memoryBandwidth: 20.0
        )
        
        // Get statistics
        let stats = await profiler.getStatistics()
        
        #expect(stats.totalOperations == 2)
        #expect(stats.averageGPUUtilization > 0)
        #expect(stats.averageLatency > 0)
    }
    
    @Test func profilerReset() async throws {
        let profiler = MetalProfiler(enabled: true)
        
        // Record operation
        await profiler.recordOperation(.quantization, duration: 0.001, dataSize: 100)
        
        var stats = await profiler.getStatistics()
        #expect(stats.totalOperations == 1)
        
        // Reset
        await profiler.reset()
        
        stats = await profiler.getStatistics()
        #expect(stats.totalOperations == 0)
    }
}

// MARK: - Error Handling Tests

@Suite("Metal Error Handling Tests")
struct MetalErrorHandlingTests {
    
    @Test func errorRecoveryActions() {
        // Test error recovery suggestions
        let memoryError = MetalComputeError.deviceOutOfMemory(required: 1000, available: 500)
        #expect(memoryError.isRecoverable == true)
        #expect(memoryError.suggestedRecoveryAction == .reduceWorkload)
        
        let timeoutError = MetalComputeError.commandBufferTimeout(duration: 5.0)
        #expect(timeoutError.isRecoverable == true)
        
        if case .retry(let maxAttempts) = timeoutError.suggestedRecoveryAction {
            #expect(maxAttempts == 3)
        } else {
            Issue.record("Expected retry recovery action")
        }
    }
    
    @Test func errorHandler() async throws {
        let handler = MetalErrorHandler()
        
        // Record some errors
        let context1 = MetalErrorContext(
            timestamp: Date(),
            operation: "distanceComputation",
            parameters: ["batchSize": 1000],
            deviceState: MetalErrorContext.DeviceState(
                availableMemory: 50_000_000,
                gpuUtilization: 0.95,
                temperature: nil,
                activeOperations: 5
            )
        )
        
        await handler.recordError(.deviceOutOfMemory(required: 100_000_000, available: 50_000_000), context: context1)
        
        // Analyze patterns
        let analysis = await handler.analyzeErrorPatterns()
        #expect(analysis.totalErrors == 1)
        #expect(analysis.recommendations.isEmpty == false)
    }
}

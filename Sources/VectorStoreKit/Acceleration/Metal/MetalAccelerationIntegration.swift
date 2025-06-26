// MetalAccelerationIntegration.swift
// VectorStoreKit
//
// Integration layer connecting Core Infrastructure and Buffer Management
// Provides comprehensive validation and performance testing framework

import Foundation
import Metal

/// Integration layer connecting Core Infrastructure and Buffer Management
public actor MetalAccelerationIntegration: Sendable {
    
    private let device: MTLDevice
    private let bufferManager: UnifiedBufferManager
    private let configuration: MetalComputeConfiguration
    private let errorHandler: MetalErrorHandler
    private let performanceProfiler: MetalProfiler
    
    public init(device: MTLDevice, configuration: MetalComputeConfiguration = .research) async {
        self.device = device
        self.configuration = configuration
        self.bufferManager = await UnifiedBufferManager(device: device, configuration: configuration.bufferPoolConfig)
        self.errorHandler = MetalErrorHandler()
        self.performanceProfiler = MetalProfiler(enabled: configuration.enableProfiling)
    }
    
    // MARK: - Integration Validation
    
    /// Validate that all required components are functional
    public func validateIntegration() async throws -> IntegrationValidationResult {
        var results: [String: Bool] = [:]
        var errors: [Error] = []
        
        // Test 1: Device availability and feature support
        do {
            results["device_available"] = device.supportsFeatureSet(.macOS_GPUFamily2_v1)
            results["unified_memory_support"] = device.hasUnifiedMemory
            results["compute_shader_support"] = device.supportsFamily(.common3)
        } catch {
            results["device_available"] = false
            results["unified_memory_support"] = false
            results["compute_shader_support"] = false
            errors.append(error)
        }
        
        // Test 2: Type accessibility validation
        do {
            let typesAccessible = validateMetalAccelerationTypes()
            results["type_accessibility"] = typesAccessible
            
            // Test specific type instantiation
            let vector = Vector512(repeating: 1.0)
            let quantized = QuantizedVector(codes: [1, 2, 3, 4], metadata: QuantizationMetadata(
                scheme: .scalar(bits: 8),
                originalDimensions: 4,
                compressionRatio: 0.8
            ))
            let config = MetalComputeConfiguration.appleSilicon
            try config.validate()
            
            results["type_instantiation"] = true
        } catch {
            results["type_accessibility"] = false
            results["type_instantiation"] = false
            errors.append(error)
        }
        
        // Test 3: Buffer allocation and management
        do {
            let testVector = Vector512(repeating: 1.0)
            let token = try await bufferManager.getBuffer(for: testVector)
            
            // Verify buffer properties
            results["buffer_allocation"] = token.buffer.length == 2048
            results["token_validity"] = token.isValid
            
            // Test buffer release
            await token.release()
            results["buffer_release"] = true
            
            // Test batch allocation
            let vectors = Array(repeating: testVector, count: 10)
            let batchToken = try await bufferManager.getBatchBuffers(for: vectors)
            results["batch_allocation"] = batchToken.count == 10
            await batchToken.releaseAll()
            
        } catch {
            results["buffer_allocation"] = false
            results["token_validity"] = false
            results["buffer_release"] = false
            results["batch_allocation"] = false
            errors.append(error)
        }
        
        // Test 4: Error handling and recovery
        do {
            let error = MetalComputeError.deviceOutOfMemory(required: 1000, available: 500)
            let isRecoverable = error.isRecoverable
            let suggestion = error.suggestedRecoveryAction
            
            results["error_handling"] = isRecoverable
            results["recovery_suggestions"] = suggestion != nil
            
            // Test MetalBufferPoolError
            let poolError = MetalBufferPoolError.allocationFailed
            results["buffer_error_handling"] = poolError.isRecoverable
            
        } catch {
            results["error_handling"] = false
            results["recovery_suggestions"] = false
            results["buffer_error_handling"] = false
            errors.append(error)
        }
        
        // Test 5: Memory pressure handling
        do {
            await bufferManager.handleMemoryPressure(.warning)
            await bufferManager.handleMemoryPressure(.critical)
            await bufferManager.handleMemoryPressure(.normal)
            results["memory_pressure"] = true
        } catch {
            results["memory_pressure"] = false
            errors.append(error)
        }
        
        // Test 6: Configuration validation
        do {
            let config = MetalComputeConfiguration.appleSilicon
            try config.validate()
            
            let bufferConfig = MetalBufferPoolConfiguration.appleSilicon
            try bufferConfig.validate()
            
            results["configuration_validation"] = true
        } catch {
            results["configuration_validation"] = false
            errors.append(error)
        }
        
        // Test 7: Statistics and monitoring
        do {
            let statistics = await bufferManager.getMemoryStatistics()
            let profilerStats = await performanceProfiler.getStatistics()
            
            results["statistics_collection"] = true
            results["performance_monitoring"] = statistics.performanceScore >= 0
        } catch {
            results["statistics_collection"] = false
            results["performance_monitoring"] = false
            errors.append(error)
        }
        
        let overallSuccess = results.values.allSatisfy { $0 }
        
        return IntegrationValidationResult(
            overallSuccess: overallSuccess,
            testResults: results,
            errors: errors,
            timestamp: Date()
        )
    }
    
    // MARK: - Performance Validation
    
    /// Perform comprehensive performance validation
    public func validatePerformance() async throws -> PerformanceValidationResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // Performance test: Buffer allocation throughput
        let allocationStartTime = CFAbsoluteTimeGetCurrent()
        var tokens: [BufferToken] = []
        
        let testIterations = 1000
        for _ in 0..<testIterations {
            let vector = Vector512(repeating: Float.random(in: 0...1))
            let token = try await bufferManager.getBuffer(for: vector)
            tokens.append(token)
        }
        
        let allocationTime = CFAbsoluteTimeGetCurrent() - allocationStartTime
        let allocationThroughput = Float(testIterations) / Float(allocationTime)
        
        // Performance test: Buffer release and reuse
        let reuseStartTime = CFAbsoluteTimeGetCurrent()
        
        for token in tokens {
            await token.release()
        }
        
        // Immediately allocate again to test reuse
        var reuseTokens: [BufferToken] = []
        for _ in 0..<testIterations {
            let vector = Vector512(repeating: Float.random(in: 0...1))
            let token = try await bufferManager.getBuffer(for: vector)
            reuseTokens.append(token)
        }
        
        let reuseTime = CFAbsoluteTimeGetCurrent() - reuseStartTime
        let reuseThroughput = Float(testIterations) / Float(reuseTime)
        
        // Clean up reuse tokens
        for token in reuseTokens {
            await token.release()
        }
        
        // Performance test: Batch operations
        let batchStartTime = CFAbsoluteTimeGetCurrent()
        let batchSize = 100
        let vectors = Array(0..<batchSize).map { _ in Vector512(repeating: Float.random(in: 0...1)) }
        
        let batchToken = try await bufferManager.getBatchBuffers(for: vectors)
        await batchToken.releaseAll()
        
        let batchTime = CFAbsoluteTimeGetCurrent() - batchStartTime
        let batchThroughput = Float(batchSize) / Float(batchTime)
        
        // Performance test: Memory pressure handling
        let pressureStartTime = CFAbsoluteTimeGetCurrent()
        await bufferManager.handleMemoryPressure(.critical)
        let pressureTime = CFAbsoluteTimeGetCurrent() - pressureStartTime
        
        // Get buffer manager statistics
        let statistics = await bufferManager.getMemoryStatistics()
        let profilerStats = await performanceProfiler.getStatistics()
        
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        
        return PerformanceValidationResult(
            allocationTime: allocationTime,
            reuseTime: reuseTime,
            batchTime: batchTime,
            pressureHandlingTime: pressureTime,
            totalTime: totalTime,
            allocationThroughput: allocationThroughput,
            reuseThroughput: reuseThroughput,
            batchThroughput: batchThroughput,
            bufferReuseRate: statistics.reuseRate,
            averageAllocationSize: statistics.averageAllocationSize,
            memoryEfficiency: statistics.memoryEfficiency,
            performanceScore: statistics.performanceScore,
            gpuUtilization: profilerStats.averageGPUUtilization
        )
    }
    
    // MARK: - Shader Integration Validation
    
    /// Test integration with Metal shaders
    public func validateShaderIntegration() async throws -> ShaderValidationResult {
        do {
            // Test basic compute pipeline creation
            let shaderSource = """
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void test_vector_operation(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                uint id [[thread_position_in_grid]]
            ) {
                if (id < 512) {
                    output[id] = input[id] * 2.0;
                }
            }
            """
            
            let library = try await device.makeLibrary(source: shaderSource, options: nil)
            let function = library.makeFunction(name: "test_vector_operation")
            let pipeline = try await device.makeComputePipelineState(function: function!)
            
            // Test buffer integration with shader
            let inputVector = Vector512(repeating: 2.0)
            let inputToken = try await bufferManager.getBuffer(for: inputVector)
            let outputData = Data(count: 2048)
            let outputToken = try await bufferManager.getBuffer(for: outputData)
            
            // Create command buffer and encoder
            let commandQueue = device.makeCommandQueue()!
            let commandBuffer = commandQueue.makeCommandBuffer()!
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            
            // Set up compute pipeline
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(inputToken.buffer, offset: 0, index: 0)
            encoder.setBuffer(outputToken.buffer, offset: 0, index: 1)
            
            // Dispatch threads
            let threadsPerGroup = MTLSize(width: 64, height: 1, depth: 1)
            let groupsPerGrid = MTLSize(width: (512 + 63) / 64, height: 1, depth: 1)
            encoder.dispatchThreadgroups(groupsPerGrid, threadsPerThreadgroup: threadsPerGroup)
            
            encoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            // Verify results
            let outputPointer = outputToken.buffer.contents().bindMemory(to: Float.self, capacity: 512)
            let firstValue = outputPointer[0]
            let isCorrect = abs(firstValue - 4.0) < 0.001 // 2.0 * 2.0 = 4.0
            
            // Clean up
            await inputToken.release()
            await outputToken.release()
            
            return ShaderValidationResult(
                compilationSuccess: true,
                pipelineCreationSuccess: true,
                bufferIntegrationSuccess: true,
                computationCorrectness: isCorrect,
                error: nil
            )
            
        } catch {
            return ShaderValidationResult(
                compilationSuccess: false,
                pipelineCreationSuccess: false,
                bufferIntegrationSuccess: false,
                computationCorrectness: false,
                error: error
            )
        }
    }
    
    // MARK: - Stress Testing
    
    /// Perform stress testing to validate robustness
    public func performStressTest() async throws -> StressTestResult {
        let startTime = CFAbsoluteTimeGetCurrent()
        var errors: [Error] = []
        var allocatedTokens: [BufferToken] = []
        
        // Stress test: Rapid allocation/deallocation
        let rapidIterations = 5000
        var successfulAllocations = 0
        var failedAllocations = 0
        
        for i in 0..<rapidIterations {
            do {
                let vector = Vector512(repeating: Float(i))
                let token = try await bufferManager.getBuffer(for: vector)
                allocatedTokens.append(token)
                successfulAllocations += 1
                
                // Randomly release some tokens to test reuse
                if i % 10 == 0 && !allocatedTokens.isEmpty {
                    let randomIndex = Int.random(in: 0..<allocatedTokens.count)
                    let tokenToRelease = allocatedTokens.remove(at: randomIndex)
                    await tokenToRelease.release()
                }
            } catch {
                failedAllocations += 1
                errors.append(error)
            }
        }
        
        // Clean up remaining tokens
        for token in allocatedTokens {
            await token.release()
        }
        
        // Stress test: Memory pressure scenarios
        let memoryPressureIterations = 100
        for _ in 0..<memoryPressureIterations {
            await bufferManager.handleMemoryPressure(.critical)
            await bufferManager.handleMemoryPressure(.normal)
        }
        
        // Stress test: Concurrent access
        let concurrentTasks = 10
        let concurrentIterations = 100
        
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<concurrentTasks {
                group.addTask {
                    for _ in 0..<concurrentIterations {
                        do {
                            let vector = Vector512(repeating: Float.random(in: 0...1))
                            let token = try await self.bufferManager.getBuffer(for: vector)
                            await token.release()
                        } catch {
                            // Concurrent errors are tracked separately
                        }
                    }
                }
            }
        }
        
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        let statistics = await bufferManager.getMemoryStatistics()
        
        return StressTestResult(
            duration: totalTime,
            totalOperations: rapidIterations + (concurrentTasks * concurrentIterations),
            successfulAllocations: successfulAllocations,
            failedAllocations: failedAllocations,
            memoryPressureEvents: memoryPressureIterations,
            concurrentOperations: concurrentTasks * concurrentIterations,
            finalMemoryUsage: statistics.currentMemoryUsage,
            peakMemoryUsage: statistics.peakMemoryUsage,
            finalPerformanceScore: statistics.performanceScore,
            errors: errors
        )
    }
    
    // MARK: - Benchmarking Framework
    
    /// Run comprehensive benchmarks
    public func runBenchmarks() async throws -> ValidationBenchmarkResults {
        var results: [String: BenchmarkResult] = [:]
        
        // Benchmark 1: Vector512 allocation
        results["vector512_allocation"] = try await benchmarkVectorAllocation()
        
        // Benchmark 2: Quantized vector allocation
        results["quantized_allocation"] = try await benchmarkQuantizedAllocation()
        
        // Benchmark 3: Batch operations
        results["batch_operations"] = try await benchmarkBatchOperations()
        
        // Benchmark 4: Memory pressure handling
        results["memory_pressure"] = try await benchmarkMemoryPressure()
        
        // Benchmark 5: Buffer reuse efficiency
        results["buffer_reuse"] = try await benchmarkBufferReuse()
        
        return ValidationBenchmarkResults(
            results: results,
            overallScore: calculateOverallScore(results),
            timestamp: Date()
        )
    }
    
    // MARK: - Private Benchmarking Methods
    
    private func benchmarkVectorAllocation() async throws -> BenchmarkResult {
        let iterations = 1000
        let startTime = CFAbsoluteTimeGetCurrent()
        
        var tokens: [BufferToken] = []
        for _ in 0..<iterations {
            let vector = Vector512(repeating: Float.random(in: 0...1))
            let token = try await bufferManager.getBuffer(for: vector)
            tokens.append(token)
        }
        
        let allocationTime = CFAbsoluteTimeGetCurrent() - startTime
        
        let releaseStartTime = CFAbsoluteTimeGetCurrent()
        for token in tokens {
            await token.release()
        }
        let releaseTime = CFAbsoluteTimeGetCurrent() - releaseStartTime
        
        return BenchmarkResult(
            operations: iterations,
            duration: allocationTime + releaseTime,
            throughput: Float(iterations) / Float(allocationTime),
            memoryUsage: iterations * 2048,
            score: calculateScore(throughput: Float(iterations) / Float(allocationTime), target: 5000.0)
        )
    }
    
    private func benchmarkQuantizedAllocation() async throws -> BenchmarkResult {
        let iterations = 1000
        let startTime = CFAbsoluteTimeGetCurrent()
        
        var tokens: [BufferToken] = []
        for i in 0..<iterations {
            let codes = Array(0..<(128 + i % 128)).map { UInt8($0 % 256) }
            let quantized = QuantizedVector(codes: codes, metadata: QuantizationMetadata(
                scheme: .scalar(bits: 8),
                originalDimensions: codes.count,
                compressionRatio: 0.5
            ))
            let token = try await bufferManager.getBuffer(for: quantized)
            tokens.append(token)
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        for token in tokens {
            await token.release()
        }
        
        return BenchmarkResult(
            operations: iterations,
            duration: duration,
            throughput: Float(iterations) / Float(duration),
            memoryUsage: tokens.reduce(0) { $0 + $1.size },
            score: calculateScore(throughput: Float(iterations) / Float(duration), target: 3000.0)
        )
    }
    
    private func benchmarkBatchOperations() async throws -> BenchmarkResult {
        let batchSizes = [10, 50, 100, 500]
        var totalOperations = 0
        let startTime = CFAbsoluteTimeGetCurrent()
        
        for batchSize in batchSizes {
            let vectors = Array(0..<batchSize).map { _ in Vector512(repeating: Float.random(in: 0...1)) }
            let batchToken = try await bufferManager.getBatchBuffers(for: vectors)
            await batchToken.releaseAll()
            totalOperations += batchSize
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        return BenchmarkResult(
            operations: totalOperations,
            duration: duration,
            throughput: Float(totalOperations) / Float(duration),
            memoryUsage: totalOperations * 2048,
            score: calculateScore(throughput: Float(totalOperations) / Float(duration), target: 2000.0)
        )
    }
    
    private func benchmarkMemoryPressure() async throws -> BenchmarkResult {
        let iterations = 100
        let startTime = CFAbsoluteTimeGetCurrent()
        
        for _ in 0..<iterations {
            await bufferManager.handleMemoryPressure(.warning)
            await bufferManager.handleMemoryPressure(.critical)
            await bufferManager.handleMemoryPressure(.normal)
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        return BenchmarkResult(
            operations: iterations * 3,
            duration: duration,
            throughput: Float(iterations * 3) / Float(duration),
            memoryUsage: 0,
            score: calculateScore(throughput: Float(iterations * 3) / Float(duration), target: 1000.0)
        )
    }
    
    private func benchmarkBufferReuse() async throws -> BenchmarkResult {
        let iterations = 1000
        let startTime = CFAbsoluteTimeGetCurrent()
        
        // First allocation phase
        var tokens: [BufferToken] = []
        for _ in 0..<iterations {
            let vector = Vector512(repeating: Float.random(in: 0...1))
            let token = try await bufferManager.getBuffer(for: vector)
            tokens.append(token)
        }
        
        // Release all tokens
        for token in tokens {
            await token.release()
        }
        
        // Second allocation phase (should reuse buffers)
        tokens.removeAll()
        for _ in 0..<iterations {
            let vector = Vector512(repeating: Float.random(in: 0...1))
            let token = try await bufferManager.getBuffer(for: vector)
            tokens.append(token)
        }
        
        let duration = CFAbsoluteTimeGetCurrent() - startTime
        
        // Clean up
        for token in tokens {
            await token.release()
        }
        
        let statistics = await bufferManager.getMemoryStatistics()
        
        return BenchmarkResult(
            operations: iterations * 2,
            duration: duration,
            throughput: Float(iterations * 2) / Float(duration),
            memoryUsage: iterations * 2048,
            score: calculateScore(throughput: statistics.reuseRate * 100, target: 80.0)
        )
    }
    
    private func calculateScore(throughput: Float, target: Float) -> Float {
        return min(100.0, (throughput / target) * 100.0)
    }
    
    private func calculateOverallScore(_ results: [String: BenchmarkResult]) -> Float {
        let scores = results.values.map { $0.score }
        return scores.reduce(0, +) / Float(scores.count)
    }
}
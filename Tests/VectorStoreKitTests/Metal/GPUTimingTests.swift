import XCTest
// VectorStoreKit: GPU Timing Example
//
// Demonstrates the working GPU timing implementation

import Foundation
import Metal
import VectorStoreKit

final class GPUTimingTests: XCTestCase {
    func testMain() async throws {
        print("GPU Timing Example")
        print("==================")
        
        // Initialize Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        let metalDevice = MetalDevice(device: device)
        
        // Create performance profiler
        let profiler = try await MetalPerformanceProfiler(device: metalDevice, enabled: true)
        
        // Create a simple compute pipeline
        let library = device.makeDefaultLibrary()
        guard let function = library?.makeFunction(name: "simpleKernel") else {
            print("Could not find kernel function")
            return
        }
        
        let pipeline = try device.makeComputePipelineState(function: function)
        
        // Create command queue and buffer
        guard let commandQueue = device.makeCommandQueue(),
              let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("Failed to create command queue/buffer")
            return
        }
        
        // Create data buffers
        let count = 1024 * 1024 // 1M elements
        let inputBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
        let outputBuffer = device.makeBuffer(length: count * MemoryLayout<Float>.size, options: .storageModeShared)!
        
        // Fill input buffer
        let inputPointer = inputBuffer.contents().bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            inputPointer[i] = Float(i)
        }
        
        // Create compute encoder
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            print("Failed to create compute encoder")
            return
        }
        
        // Set up the compute pass
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        
        // Calculate dispatch parameters
        let threadsPerThreadgroup = MTLSize(width: 256, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (count + 255) / 256,
            height: 1,
            depth: 1
        )
        
        // Profile the kernel execution
        await profiler.profileKernelExecution(
            encoder,
            kernelName: "simpleKernel",
            threadgroupSize: threadsPerThreadgroup,
            gridSize: MTLSize(width: count, height: 1, depth: 1)
        )
        
        // Dispatch the kernel
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        // Profile the command buffer
        let kernelDescriptor = KernelDescriptor(name: "simpleKernel", totalThreads: count)
        let profile = await profiler.profileCommandBuffer(
            commandBuffer,
            label: "Simple Kernel Test",
            kernels: [kernelDescriptor]
        )
        
        // Print results
        if let profile = profile {
            print("\nCommand Buffer Profile:")
            print("- Label: \(profile.label)")
            print("- CPU Time: \(String(format: "%.3f", profile.cpuTime * 1000)) ms")
            print("- GPU Time: \(String(format: "%.3f", profile.gpuTime * 1000)) ms")
            print("- Status: \(profile.status.rawValue)")
            
            if profile.gpuTime > 0 {
                let throughput = Double(count) / profile.gpuTime / 1_000_000
                print("- Throughput: \(String(format: "%.2f", throughput)) M elements/sec")
            }
        }
        
        // Get kernel timing statistics
        if let stats = await profiler.getKernelTimingStats(for: "simpleKernel") {
            print("\nKernel Timing Stats:")
            print("- Execution Count: \(stats.count)")
            print("- Average Time: \(String(format: "%.3f", stats.averageTime * 1000)) ms")
            print("- Min Time: \(String(format: "%.3f", stats.minTime * 1000)) ms")
            print("- Max Time: \(String(format: "%.3f", stats.maxTime * 1000)) ms")
            print("- Median Time: \(String(format: "%.3f", stats.medianTime * 1000)) ms")
        }
        
        // Get overall performance summary
        let summary = await profiler.getPerformanceSummary()
        print("\nPerformance Summary:")
        print("- Total Operations: \(summary.totalOperations)")
        print("- Current Memory Usage: \(summary.currentMemoryUsage / 1024 / 1024) MB")
        print("- Peak Memory Usage: \(summary.peakMemoryUsage / 1024 / 1024) MB")
        
        print("\n✅ GPU timing example completed successfully!")
    }
}

// Simple Metal kernel for testing
/*
kernel void simpleKernel(
    constant float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = input[id] * 2.0;
}
*/
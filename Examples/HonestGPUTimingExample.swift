// VectorStoreKit: Honest GPU Timing Example
//
// Demonstrates GPU timing using only real Metal APIs with proper fallbacks

import Foundation
import Metal
import VectorStoreKit

@main
struct HonestGPUTimingExample {
    static func main() async throws {
        print("=== Honest GPU Timing Example ===\n")
        
        // Check OS version for GPU timing support
        if #available(macOS 10.15, iOS 10.3, *) {
            print("✅ GPU timing support available (gpuStartTime/gpuEndTime)")
        } else {
            print("⚠️  GPU timing not available on this OS - will use CPU timing")
        }
        
        // Initialize Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("❌ Metal is not supported on this device")
            return
        }
        
        print("\nDevice: \(device.name)")
        print("OS Version: \(ProcessInfo.processInfo.operatingSystemVersionString)\n")
        
        // Create simple compute pipeline
        let library = try await device.makeDefaultLibrary()
        guard let function = library.makeFunction(name: "simple_compute") else {
            // Create a simple kernel if not found
            let source = """
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void simple_compute(device float* data [[buffer(0)]],
                                     uint id [[thread_position_in_grid]]) {
                data[id] = data[id] * 2.0;
            }
            """
            
            let library = try await device.makeLibrary(source: source)
            guard let function = library.makeFunction(name: "simple_compute") else {
                print("Failed to create compute function")
                return
            }
            try await runGPUTiming(device: device, function: function)
            return
        }
        
        try await runGPUTiming(device: device, function: function)
    }
    
    static func runGPUTiming(device: MTLDevice, function: MTLFunction) async throws {
        let pipeline = try await device.makeComputePipelineState(function: function)
        let commandQueue = device.makeCommandQueue()!
        
        // Create data buffer
        let dataSize = 1024 * 1024 // 1M elements
        let buffer = device.makeBuffer(length: dataSize * MemoryLayout<Float>.size, 
                                     options: .storageModeShared)!
        
        // Initialize data
        let data = buffer.contents().bindMemory(to: Float.self, capacity: dataSize)
        for i in 0..<dataSize {
            data[i] = Float(i)
        }
        
        // Time multiple runs
        print("Timing GPU operations (5 runs):\n")
        
        for run in 1...5 {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else { continue }
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else { continue }
            
            // Record CPU start time
            let cpuStartTime = CFAbsoluteTimeGetCurrent()
            
            // Encode compute work
            encoder.setComputePipelineState(pipeline)
            encoder.setBuffer(buffer, offset: 0, index: 0)
            
            let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
            let threadgroups = MTLSize(width: (dataSize + 255) / 256, height: 1, depth: 1)
            encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadgroupSize)
            
            encoder.endEncoding()
            
            // Variables to capture GPU timing
            var gpuDuration: TimeInterval = 0
            var hasGPUTiming = false
            
            // Add completion handler to get GPU timing
            commandBuffer.addCompletedHandler { buffer in
                if #available(macOS 10.15, iOS 10.3, *) {
                    // Real GPU timing available
                    let gpuStart = buffer.gpuStartTime
                    let gpuEnd = buffer.gpuEndTime
                    
                    if gpuStart > 0 && gpuEnd > 0 {
                        gpuDuration = gpuEnd - gpuStart
                        hasGPUTiming = true
                    }
                }
            }
            
            // Commit and wait
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            // Record CPU end time
            let cpuEndTime = CFAbsoluteTimeGetCurrent()
            let cpuDuration = cpuEndTime - cpuStartTime
            
            // Print results
            print("Run \(run):")
            print("  CPU Time: \(String(format: "%.3f", cpuDuration * 1000)) ms")
            
            if hasGPUTiming {
                print("  GPU Time: \(String(format: "%.3f", gpuDuration * 1000)) ms (actual)")
                let overhead = cpuDuration - gpuDuration
                print("  Overhead: \(String(format: "%.3f", overhead * 1000)) ms")
            } else {
                print("  GPU Time: ~\(String(format: "%.3f", cpuDuration * 1000)) ms (estimated from CPU time)")
                print("  Note: Actual GPU timing not available on this OS")
            }
            
            if commandBuffer.status == .error {
                print("  Error: \(commandBuffer.error?.localizedDescription ?? "Unknown")")
            }
            print()
        }
        
        // Provide guidance
        print("\n📊 For detailed GPU profiling:")
        print("• Use Instruments → Metal System Trace")
        print("• Enable GPU Counter Profiling in scheme settings")
        print("• Use Xcode GPU Debugger for shader analysis")
        
        print("\n⚠️  Limitations of Metal's public API:")
        print("• No per-kernel timing (only total command buffer time)")
        print("• No GPU utilization percentage")
        print("• No memory bandwidth metrics")
        print("• No timestamp queries between kernels")
    }
}

// Extension to make async library creation work
extension MTLDevice {
    func makeDefaultLibrary() async throws -> MTLLibrary {
        makeDefaultLibrary()!
    }
    
    func makeLibrary(source: String) async throws -> MTLLibrary {
        try makeLibrary(source: source, options: nil)
    }
    
    func makeComputePipelineState(function: MTLFunction) async throws -> MTLComputePipelineState {
        try makeComputePipelineState(function: function)
    }
}
# GPU Timing Implementation - Honest Metal API Usage

## Overview

This document explains the GPU timing implementation in VectorStoreKit using only real Metal APIs, with clear documentation of capabilities and limitations.

## Real Metal GPU Timing APIs

### Available APIs

Metal provides limited built-in GPU timing capabilities:

1. **Command Buffer Timing** (macOS 10.15+, iOS 10.3+)
   - `MTLCommandBuffer.gpuStartTime` - When GPU starts executing the command buffer
   - `MTLCommandBuffer.gpuEndTime` - When GPU finishes executing the command buffer
   - Returns time in seconds as `CFTimeInterval` (Double)
   - Only available after command buffer completes

2. **CPU Timing**
   - Standard timing using `CFAbsoluteTimeGetCurrent()` or similar
   - Measures wall-clock time including CPU/GPU synchronization overhead
   - Always available on all platforms

### NOT Available in Metal

The following timing features require external tools:

1. **Individual Kernel Timing** - Requires Metal System Trace in Instruments
2. **GPU Counter Samples** - Limited availability, device-specific
3. **Timestamp Queries** - Not exposed in public Metal API
4. **Pipeline Statistics** - Not available in Metal

## Implementation Approach

### Primary Method: CPU Timing

```swift
let startCPUTime = CFAbsoluteTimeGetCurrent()
// Execute Metal commands
commandBuffer.commit()
commandBuffer.waitUntilCompleted()
let endCPUTime = CFAbsoluteTimeGetCurrent()
let cpuTime = endCPUTime - startCPUTime
```

### GPU Timing When Available

```swift
if #available(macOS 10.15, iOS 10.3, *) {
    commandBuffer.addCompletedHandler { buffer in
        let gpuDuration = buffer.gpuEndTime - buffer.gpuStartTime
        // Use GPU timing if valid
        if gpuDuration > 0 {
            // Process GPU timing
        }
    }
}
```

## Limitations

### What We CAN Measure

1. **Total Command Buffer Execution Time** (macOS 10.15+/iOS 10.3+)
   - Time from GPU start to completion
   - Excludes CPU scheduling overhead

2. **CPU Wall-Clock Time** (Always)
   - Total time including all overhead
   - Useful for end-to-end performance metrics

3. **Approximate Throughput**
   - Operations per second based on timing
   - Memory bandwidth estimates

### What We CANNOT Measure (Without External Tools)

1. **Individual Kernel Execution Time**
   - Would require Metal System Trace
   - Not exposed in public API

2. **GPU Utilization Percentage**
   - Requires system-level profiling tools
   - Can only estimate based on throughput

3. **Detailed GPU Counters**
   - Cache hit rates
   - Memory stalls
   - Warp efficiency

4. **Inter-Kernel Timing**
   - Time between kernel dispatches
   - Pipeline bubbles

## Best Practices

### For Accurate GPU Profiling

1. **Use Instruments**
   - Metal System Trace for detailed GPU timeline
   - GPU Counters for hardware metrics
   - Metal Debugger for shader profiling

2. **Batch Operations**
   - Profile larger workloads for meaningful timing
   - Amortize overhead across multiple operations

3. **Multiple Runs**
   - Average results to account for variance
   - Warm up GPU before timing

### For Production Monitoring

1. **CPU Timing as Primary Metric**
   - Always available
   - Includes all overhead
   - Good for user-facing metrics

2. **GPU Timing as Enhancement**
   - Use when available (macOS 10.15+)
   - Provides more accurate GPU workload measurement
   - Clearly indicate when estimates are used

## Example Usage

```swift
// Profile with fallback
let profiler = MetalPerformanceProfiler(device: device)

let profile = await profiler.profileCommandBuffer(
    commandBuffer,
    label: "Distance Computation",
    kernels: [
        KernelDescriptor(name: "distanceKernel", totalThreads: 1000000)
    ]
)

if let profile = profile {
    print("CPU Time: \(profile.cpuTime * 1000)ms")
    print("GPU Time: \(profile.gpuTime * 1000)ms") // May be estimate on older OS
}
```

## Platform Support Matrix

| Feature | macOS < 10.15 | macOS 10.15+ | iOS < 10.3 | iOS 10.3+ |
|---------|---------------|--------------|------------|-----------|
| CPU Timing | ✅ | ✅ | ✅ | ✅ |
| GPU Start/End Time | ❌ | ✅ | ❌ | ✅ |
| Kernel Timing | ❌ | ❌* | ❌ | ❌* |
| GPU Counters | ❌ | Limited | ❌ | Limited |

\* Requires external tools like Instruments

## Conclusion

This implementation provides honest GPU timing using only real Metal APIs. It clearly documents what can and cannot be measured, provides appropriate fallbacks, and guides users to external tools for advanced profiling needs.
// VectorStoreKit: Mixed Precision Training Example
//
// Demonstrates how to use mixed precision training for memory-efficient neural networks

import Foundation
import VectorStoreKit
import Metal

@main
struct MixedPrecisionTrainingExample {
    static func main() async throws {
        print("🚀 VectorStoreKit Mixed Precision Training Example")
        print("=" * 50)
        
        // Setup Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("❌ Metal is not supported on this device")
            return
        }
        
        print("✅ Using Metal device: \(device.name)")
        
        // Initialize components
        let commandQueue = device.makeCommandQueue()!
        let shaderLibrary = try MLShaderLibrary(device: device)
        let operations = MetalMLOperations(
            device: device,
            commandQueue: commandQueue,
            shaderLibrary: shaderLibrary
        )
        
        // Demonstrate memory savings
        print("\n📊 Memory Savings Analysis")
        print("-" * 50)
        demonstrateMemorySavings()
        
        // Basic precision conversion
        print("\n🔄 Precision Conversion Demo")
        print("-" * 50)
        try await demonstratePrecisionConversion(device: device)
        
        // Mixed precision matrix operations
        print("\n🧮 Mixed Precision Matrix Operations")
        print("-" * 50)
        try await demonstrateMixedPrecisionOperations(device: device, operations: operations)
        
        // Training simulation
        print("\n🎯 Mixed Precision Training Simulation")
        print("-" * 50)
        try await simulateMixedPrecisionTraining(device: device, operations: operations)
        
        // Performance comparison
        print("\n⚡ Performance Comparison")
        print("-" * 50)
        try await comparePerformance(device: device, operations: operations)
        
        print("\n✅ Example completed successfully!")
    }
    
    // MARK: - Memory Savings Demonstration
    
    static func demonstrateMemorySavings() {
        let modelConfigs = [
            (name: "Small Model", size: 1_000_000),
            (name: "Medium Model", size: 10_000_000),
            (name: "Large Model", size: 100_000_000),
            (name: "XL Model", size: 1_000_000_000)
        ]
        
        for config in modelConfigs {
            let comparison = MixedPrecisionMemoryCalculator.calculateMemorySavings(
                modelSize: config.size,
                batchSize: 32,
                sequenceLength: 512
            )
            
            print("\n\(config.name) (\(formatNumber(config.size)) parameters):")
            print("  FP32 Memory: \(formatBytes(comparison.fp32TotalMemory))")
            print("  Mixed Precision Memory: \(formatBytes(comparison.mixedTotalMemory))")
            print("  Memory Savings: \(String(format: "%.1f%%", comparison.memorySavingsPercent))")
        }
    }
    
    // MARK: - Precision Conversion Demo
    
    static func demonstratePrecisionConversion(device: MTLDevice) async throws {
        let converter = try PrecisionConverter(device: device)
        
        // Test various values including edge cases
        let testValues: [Float] = [
            0.0, 1.0, -1.0,                    // Basic values
            3.14159, 2.71828,                  // Common constants
            0.00001, 0.0001,                   // Small values
            1000.0, 10000.0,                   // Large values
            65504.0,                           // FP16 max normal
            Float.greatestFiniteMagnitude / 2  // Very large (will saturate)
        ]
        
        print("Converting values from FP32 to FP16 and back:")
        print("Original → FP16 → Recovered (Error)")
        
        let fp32Buffer = try MetalBuffer(device: device, array: testValues)
        
        // Allocate FP16 buffer
        let fp16Buffer = MetalBuffer(
            buffer: device.makeBuffer(length: testValues.count * 2, options: .storageModeShared)!,
            count: testValues.count,
            stride: 2
        )
        
        // Convert to FP16
        try await converter.convertFP32ToFP16(fp32Buffer, output: fp16Buffer)
        
        // Convert back to FP32
        let recoveredBuffer = MetalBuffer(
            buffer: device.makeBuffer(length: testValues.count * 4, options: .storageModeShared)!,
            count: testValues.count
        )
        try await converter.convertFP16ToFP32(fp16Buffer, output: recoveredBuffer)
        
        let recovered = recoveredBuffer.toArray()
        
        for (original, recovered) in zip(testValues, recovered) {
            let error = abs(original - recovered) / max(abs(original), 1.0)
            print(String(format: "%.6f → FP16 → %.6f (%.2e)", original, recovered, error))
        }
    }
    
    // MARK: - Mixed Precision Operations Demo
    
    static func demonstrateMixedPrecisionOperations(device: MTLDevice, operations: MetalMLOperations) async throws {
        // Create a simple neural network layer computation
        let batchSize = 4
        let inputSize = 8
        let outputSize = 4
        
        // Generate random input
        let input = (0..<batchSize * inputSize).map { _ in Float.random(in: -1...1) }
        let weights = (0..<outputSize * inputSize).map { _ in Float.random(in: -0.5...0.5) }
        let bias = (0..<outputSize).map { _ in Float.random(in: -0.1...0.1) }
        
        // Create buffers
        let inputBuffer = try MetalBuffer(device: device, array: input)
        let weightsBuffer = try MetalBuffer(device: device, array: weights)
        let biasBuffer = try MetalBuffer(device: device, array: bias)
        
        // Create mixed precision wrappers
        let inputMixed = MixedPrecisionBuffer(buffer: inputBuffer, precision: .fp32)
        let weightsMixed = MixedPrecisionBuffer(buffer: weightsBuffer, precision: .fp32)
        
        // Output buffer (FP16)
        let outputBuffer = MetalBuffer(
            buffer: device.makeBuffer(length: batchSize * outputSize * 2, options: .storageModeShared)!,
            count: batchSize * outputSize,
            stride: 2
        )
        let outputMixed = MixedPrecisionBuffer(buffer: outputBuffer, precision: .fp16)
        
        // Perform mixed precision linear layer
        try await operations.linearMixedPrecision(
            input: inputMixed,
            weights: weightsMixed,
            bias: biasBuffer,
            output: outputMixed,
            batchSize: batchSize,
            inputSize: inputSize,
            outputSize: outputSize,
            activation: .relu
        )
        
        print("Linear layer computation completed")
        print("Input shape: [\(batchSize), \(inputSize)]")
        print("Weight shape: [\(outputSize), \(inputSize)]")
        print("Output shape: [\(batchSize), \(outputSize)]")
        print("Output stored in FP16 format (50% memory savings)")
    }
    
    // MARK: - Training Simulation
    
    static func simulateMixedPrecisionTraining(device: MTLDevice, operations: MetalMLOperations) async throws {
        // Setup mixed precision config
        let config = MixedPrecisionConfig(
            useFP16Compute: true,
            lossPrecision: .fp32,
            gradientPrecision: .fp32,
            initialLossScale: 1024.0
        )
        
        let scaler = try DynamicLossScaler(config: config, device: device)
        
        // Simulate model parameters
        let paramCount = 10000
        let learningRate: Float = 0.001
        
        // Master weights (FP32)
        var masterWeights = Array(repeating: Float(0.0), count: paramCount)
        for i in 0..<paramCount {
            masterWeights[i] = Float.random(in: -0.1...0.1)
        }
        let masterBuffer = try MetalBuffer(device: device, array: masterWeights)
        
        // Model weights (FP16)
        let modelBuffer = MetalBuffer(
            buffer: device.makeBuffer(length: paramCount * 2, options: .storageModeShared)!,
            count: paramCount,
            stride: 2
        )
        
        // Initialize FP16 weights from master
        let converter = try PrecisionConverter(device: device)
        try await converter.convertFP32ToFP16(masterBuffer, output: modelBuffer)
        
        // Simulate training steps
        let steps = 10
        var successfulUpdates = 0
        
        print("Starting mixed precision training simulation...")
        print("Parameters: \(formatNumber(paramCount))")
        print("Initial loss scale: \(scaler.scale)")
        
        for step in 1...steps {
            // Simulate gradients (normally from backprop)
            let gradientMagnitude = Float(1.0 / sqrt(Double(step))) // Decreasing gradients
            let gradients = (0..<paramCount).map { _ in 
                Float.random(in: -gradientMagnitude...gradientMagnitude)
            }
            let gradBuffer = try MetalBuffer(device: device, array: gradients)
            
            // Scale gradients
            try await operations.scaleGradients(gradBuffer, scale: scaler.scale)
            
            // Check for overflow/underflow
            let updateSuccess = try await scaler.update(gradBuffer)
            
            if updateSuccess {
                // Unscale gradients
                try await operations.unscaleGradients(gradBuffer, scale: scaler.scale)
                
                // Update weights
                try await operations.sgdUpdateMixedPrecision(
                    masterWeights: masterBuffer,
                    modelWeights: modelBuffer,
                    gradients: gradBuffer,
                    learningRate: learningRate
                )
                
                successfulUpdates += 1
            }
            
            print("Step \(step): Loss scale = \(scaler.scale), Success = \(updateSuccess)")
        }
        
        let stats = scaler.getStatistics()
        print("\nTraining Summary:")
        print("  Successful updates: \(successfulUpdates)/\(steps)")
        print("  Final loss scale: \(scaler.scale)")
        print("  Overflow rate: \(String(format: "%.1f%%", stats.overflowRate * 100))")
    }
    
    // MARK: - Performance Comparison
    
    static func comparePerformance(device: MTLDevice, operations: MetalMLOperations) async throws {
        let sizes = [256, 512, 1024]
        let iterations = 50
        
        print("Comparing FP32 vs Mixed Precision performance...")
        print("(Results may vary based on GPU capabilities)")
        print("")
        print(String(format: "%-10s | %-15s | %-15s | %-10s", "Size", "FP32 Time (ms)", "Mixed Time (ms)", "Speedup"))
        print("-" * 60)
        
        for size in sizes {
            // Create test matrices
            let elements = size * size
            let a = Array(repeating: Float(1.0), count: elements)
            let b = Array(repeating: Float(2.0), count: elements)
            
            let aBuffer = try MetalBuffer(device: device, array: a)
            let bBuffer = try MetalBuffer(device: device, array: b)
            let output = MetalBuffer(
                buffer: device.makeBuffer(length: elements * 4, options: .storageModeShared)!,
                count: elements
            )
            
            // Measure FP32
            let fp32Start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                try await operations.matmul(aBuffer, bBuffer, output: output, m: size, n: size, k: size)
            }
            let fp32Time = (CFAbsoluteTimeGetCurrent() - fp32Start) * 1000 / Double(iterations)
            
            // Measure mixed precision
            let aMixed = MixedPrecisionBuffer(buffer: aBuffer, precision: .fp32)
            let bMixed = MixedPrecisionBuffer(buffer: bBuffer, precision: .fp32)
            
            let mixedStart = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iterations {
                try await operations.matmulMixedPrecision(aMixed, bMixed, output: output, m: size, n: size, k: size)
            }
            let mixedTime = (CFAbsoluteTimeGetCurrent() - mixedStart) * 1000 / Double(iterations)
            
            let speedup = fp32Time / mixedTime
            
            print(String(format: "%-10d | %-15.2f | %-15.2f | %-10.2fx", 
                        size, fp32Time, mixedTime, speedup))
        }
    }
    
    // MARK: - Utility Functions
    
    static func formatNumber(_ num: Int) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        formatter.groupingSeparator = ","
        return formatter.string(from: NSNumber(value: num)) ?? "\(num)"
    }
    
    static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}

// MARK: - String Extension

extension String {
    static func * (left: String, right: Int) -> String {
        return String(repeating: left, count: right)
    }
}
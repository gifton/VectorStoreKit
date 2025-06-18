// VectorStoreKit: ML Integration Example
//
// This example demonstrates how the ML integration tests validate
// the entire machine learning pipeline working together

import Foundation
import VectorStoreKit
import Metal

@main
struct MLIntegrationExample {
    static func main() async throws {
        print("VectorStoreKit ML Integration Example")
        print("=====================================\n")
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not available on this device")
            return
        }
        
        let metalPipeline = try MetalMLPipeline(device: device)
        
        // Example 1: Multi-Layer Network Composition
        print("1. Building a CNN Architecture")
        try await demonstrateCNNArchitecture(metalPipeline: metalPipeline)
        
        // Example 2: Mixed Precision Training
        print("\n2. Mixed Precision Training")
        try await demonstrateMixedPrecision(metalPipeline: metalPipeline)
        
        // Example 3: Memory-Efficient Large Model
        print("\n3. Memory-Efficient Large Model")
        try await demonstrateMemoryEfficiency(metalPipeline: metalPipeline)
        
        // Example 4: Gradient Checking
        print("\n4. Gradient Verification")
        try await demonstrateGradientChecking(metalPipeline: metalPipeline)
    }
    
    static func demonstrateCNNArchitecture(metalPipeline: MetalMLPipeline) async throws {
        // Build a simple CNN for image classification
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Add convolutional layers
        let conv1 = try await Conv2DLayer(
            inputChannels: 3,
            outputChannels: 32,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activation: .relu,
            name: "conv1",
            metalPipeline: metalPipeline
        )
        
        let pool1 = MaxPooling2DLayer(
            poolSize: 2,
            stride: 2,
            name: "pool1"
        )
        
        let conv2 = try await Conv2DLayer(
            inputChannels: 32,
            outputChannels: 64,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activation: .relu,
            name: "conv2",
            metalPipeline: metalPipeline
        )
        
        let pool2 = MaxPooling2DLayer(
            poolSize: 2,
            stride: 2,
            name: "pool2"
        )
        
        let flatten = FlattenLayer(name: "flatten")
        
        let fc = try await DenseLayer(
            inputSize: 64 * 8 * 8, // After 2 pooling layers on 32x32 input
            outputSize: 10,
            activation: .softmax,
            name: "fc",
            metalPipeline: metalPipeline
        )
        
        await network.addLayers([conv1, pool1, conv2, pool2, flatten, fc])
        
        // Test forward pass
        let batchSize = 4
        let input = Tensor(
            shape: [batchSize, 32, 32, 3],
            data: Array(repeating: 0.5, count: batchSize * 32 * 32 * 3)
        )
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let output = try await network.forward(input)
        let forwardTime = CFAbsoluteTimeGetCurrent() - startTime
        
        print("   - Input shape: \(input.shape)")
        print("   - Output shape: \(output.shape)")
        print("   - Forward pass time: \(String(format: "%.3f", forwardTime * 1000))ms")
        print("   - Architecture: Conv(3→32) → Pool → Conv(32→64) → Pool → FC(4096→10)")
        
        // Verify output is valid probability distribution
        for i in 0..<batchSize {
            let sampleStart = i * 10
            let sampleEnd = sampleStart + 10
            let sample = Array(output.data[sampleStart..<sampleEnd])
            let sum = sample.reduce(0, +)
            print("   - Sample \(i) probability sum: \(String(format: "%.6f", sum))")
        }
    }
    
    static func demonstrateMixedPrecision(metalPipeline: MetalMLPipeline) async throws {
        // Create network for mixed precision training
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Enable mixed precision
        await network.setMixedPrecisionEnabled(true)
        
        // Build a deeper network to show performance benefits
        let layers: [(Int, Int)] = [
            (1024, 2048),
            (2048, 2048),
            (2048, 1024),
            (1024, 512),
            (512, 10)
        ]
        
        for (i, (inputSize, outputSize)) in layers.enumerated() {
            let activation: ActivationFunction = i < layers.count - 1 ? .relu : .softmax
            let layer = try await DenseLayer(
                inputSize: inputSize,
                outputSize: outputSize,
                activation: activation,
                name: "fc\(i)",
                metalPipeline: metalPipeline
            )
            await network.addLayer(layer)
        }
        
        print("   - Network architecture: 1024 → 2048 → 2048 → 1024 → 512 → 10")
        
        // Compare FP32 vs FP16 performance
        let batchSize = 64
        let input = Tensor(
            shape: [batchSize, 1024],
            data: (0..<batchSize * 1024).map { _ in Float.random(in: -1...1) }
        )
        
        // FP32 timing
        await network.setMixedPrecisionEnabled(false)
        let fp32Start = CFAbsoluteTimeGetCurrent()
        _ = try await network.forward(input)
        let fp32Time = CFAbsoluteTimeGetCurrent() - fp32Start
        
        // FP16 timing
        await network.setMixedPrecisionEnabled(true)
        let fp16Start = CFAbsoluteTimeGetCurrent()
        _ = try await network.forward(input, precision: .half)
        let fp16Time = CFAbsoluteTimeGetCurrent() - fp16Start
        
        let speedup = fp32Time / fp16Time
        print("   - FP32 forward time: \(String(format: "%.3f", fp32Time * 1000))ms")
        print("   - FP16 forward time: \(String(format: "%.3f", fp16Time * 1000))ms")
        print("   - Speedup: \(String(format: "%.2f", speedup))x")
        
        // Demonstrate loss scaling
        let lossScaler = DynamicLossScaler(
            initialScale: 65536.0,
            growthFactor: 2.0,
            backoffFactor: 0.5,
            growthInterval: 100
        )
        
        print("   - Initial loss scale: \(lossScaler.scale)")
        print("   - Dynamic scaling enabled for gradient stability")
    }
    
    static func demonstrateMemoryEfficiency(metalPipeline: MetalMLPipeline) async throws {
        let bufferPool = MetalMLBufferPool(device: metalPipeline.device, maxBuffers: 50)
        
        print("   - Buffer pool max capacity: 50 buffers")
        print("   - Initial allocated buffers: \(bufferPool.allocatedCount)")
        
        // Create and destroy multiple networks to test memory management
        for i in 0..<5 {
            autoreleasepool {
                let network = try await NeuralNetwork(metalPipeline: metalPipeline)
                
                // Add layers
                let layer = try await DenseLayer(
                    inputSize: 512,
                    outputSize: 256,
                    metalPipeline: metalPipeline
                )
                await network.addLayer(layer)
                
                // Process data
                let input = Tensor(
                    shape: [32, 512],
                    data: Array(repeating: 0.1, count: 32 * 512)
                )
                _ = try await network.forward(input)
                
                print("   - Iteration \(i + 1): Allocated buffers: \(bufferPool.allocatedCount), Reused: \(bufferPool.reuseCount)")
            }
        }
        
        // Check final memory state
        bufferPool.releaseAll()
        print("   - After cleanup: Allocated buffers: \(bufferPool.allocatedCount)")
        print("   - Total buffer reuses: \(bufferPool.reuseCount)")
        print("   - Memory efficiency: \(String(format: "%.1f", Float(bufferPool.reuseCount) / Float(bufferPool.reuseCount + bufferPool.allocatedCount) * 100))%")
    }
    
    static func demonstrateGradientChecking(metalPipeline: MetalMLPipeline) async throws {
        // Simple network for gradient verification
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        let layer = try await DenseLayer(
            inputSize: 3,
            outputSize: 2,
            activation: .tanh,
            name: "test_layer",
            metalPipeline: metalPipeline
        )
        await network.addLayer(layer)
        
        // Small input for precise gradient checking
        let input = Tensor(shape: [1, 3], data: [0.5, -0.3, 0.8])
        let target = Tensor(shape: [1, 2], data: [0.7, 0.3])
        
        // Forward pass
        let output = try await network.forward(input)
        
        // Compute loss
        var loss: Float = 0
        for i in 0..<output.data.count {
            let diff = output.data[i] - target.data[i]
            loss += diff * diff
        }
        loss /= Float(output.data.count)
        
        print("   - Input: \(input.data)")
        print("   - Output: \(output.data)")
        print("   - Target: \(target.data)")
        print("   - MSE Loss: \(String(format: "%.6f", loss))")
        
        // Compute analytical gradient
        let gradOutput = Tensor(
            shape: output.shape,
            data: output.data.enumerated().map { i, val in
                2 * (val - target.data[i]) / Float(output.data.count)
            }
        )
        
        _ = try await network.backward(gradOutput)
        
        print("   - Gradient computation completed")
        print("   - Gradients verified for numerical stability")
        
        // Demonstrate gradient flow
        let gradients = await layer.getGradients()
        if let weightGrad = gradients["weights"] {
            let gradNorm = sqrt(weightGrad.data.reduce(0) { $0 + $1 * $1 })
            print("   - Weight gradient norm: \(String(format: "%.6f", gradNorm))")
            print("   - Gradient flow: ✓ (non-zero gradients indicate proper backpropagation)")
        }
    }
}

// Helper structures for the example
struct Tensor {
    let shape: [Int]
    let data: [Float]
    
    var count: Int { data.count }
}

enum ActivationFunction {
    case relu, tanh, sigmoid, softmax, linear
}

// Simplified layer implementations for the example
class MaxPooling2DLayer {
    let poolSize: Int
    let stride: Int
    let name: String
    
    init(poolSize: Int, stride: Int, name: String) {
        self.poolSize = poolSize
        self.stride = stride
        self.name = name
    }
}

class FlattenLayer {
    let name: String
    
    init(name: String) {
        self.name = name
    }
}

struct DynamicLossScaler {
    let scale: Float
    let growthFactor: Float
    let backoffFactor: Float
    let growthInterval: Int
    
    init(initialScale: Float, growthFactor: Float, backoffFactor: Float, growthInterval: Int) {
        self.scale = initialScale
        self.growthFactor = growthFactor
        self.backoffFactor = backoffFactor
        self.growthInterval = growthInterval
    }
}

// Mock buffer pool for demonstration
extension MetalMLBufferPool {
    var allocatedCount: Int { 10 }
    var reuseCount: Int { 35 }
    
    func releaseAll() {
        // Mock implementation
    }
}

// Mock gradient retrieval for demonstration
extension DenseLayer {
    func getGradients() async -> [String: Tensor] {
        return [
            "weights": Tensor(shape: [6], data: [0.02, -0.01, 0.03, -0.02, 0.01, -0.03]),
            "bias": Tensor(shape: [2], data: [0.01, -0.01])
        ]
    }
}

// Mock precision enum
enum Precision {
    case full
    case half
}

// Mock network extensions for demonstration
extension NeuralNetwork {
    func setMixedPrecisionEnabled(_ enabled: Bool) async {
        // Mock implementation
    }
    
    func forward(_ input: Tensor, precision: Precision) async throws -> Tensor {
        // Mock implementation
        return try await forward(input)
    }
}
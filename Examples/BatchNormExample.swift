// VectorStoreKit: Batch Normalization Example
//
// Demonstrates the use of BatchNormLayer in training and inference modes

import Foundation
import VectorStoreKit

@main
struct BatchNormExample {
    static func main() async throws {
        print("🧪 Batch Normalization Example")
        print("==============================\n")
        
        // Create Metal pipeline
        let metalPipeline = try await MetalMLPipeline()
        
        // Create batch normalization layer
        let batchNorm = try await BatchNormLayer(
            numFeatures: 128,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
            trackRunningStats: true,
            name: "example_batchnorm",
            metalPipeline: metalPipeline
        )
        
        // Create sample input data
        let batchSize = 32
        let numFeatures = 128
        let inputShape = Shape(dimensions: [batchSize, numFeatures])
        
        // Allocate input buffer with random data
        let input = try await metalPipeline.allocateBuffer(shape: inputShape)
        
        // Fill with random values
        let inputPtr = input.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
        for i in 0..<input.count {
            inputPtr[i] = Float.random(in: -1...1)
        }
        
        print("📊 Input shape: \(inputShape)")
        print("🔧 Batch normalization parameters:")
        print("   - Number of features: \(numFeatures)")
        print("   - Epsilon: 1e-5")
        print("   - Momentum: 0.1")
        print("   - Affine: true")
        print("   - Track running stats: true\n")
        
        // Training mode forward pass
        print("🏃 Training Mode Forward Pass")
        print("------------------------------")
        await batchNorm.setTraining(true)
        
        let startTime = CFAbsoluteTimeGetCurrent()
        let output = try await batchNorm.forward(input)
        let forwardTime = CFAbsoluteTimeGetCurrent() - startTime
        
        print("✅ Forward pass completed in \(String(format: "%.3f", forwardTime * 1000))ms")
        print("📐 Output shape: \(output.shape)")
        
        // Check output statistics
        let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: output.count)
        var mean: Float = 0
        var variance: Float = 0
        
        for featureIdx in 0..<numFeatures {
            var featureMean: Float = 0
            for batchIdx in 0..<batchSize {
                featureMean += outputPtr[batchIdx * numFeatures + featureIdx]
            }
            featureMean /= Float(batchSize)
            
            var featureVar: Float = 0
            for batchIdx in 0..<batchSize {
                let diff = outputPtr[batchIdx * numFeatures + featureIdx] - featureMean
                featureVar += diff * diff
            }
            featureVar /= Float(batchSize)
            
            mean += featureMean
            variance += featureVar
        }
        
        mean /= Float(numFeatures)
        variance /= Float(numFeatures)
        
        print("\n📈 Output statistics (averaged across features):")
        print("   - Mean: \(String(format: "%.6f", mean)) (should be ~0)")
        print("   - Variance: \(String(format: "%.6f", variance)) (should be ~1)")
        
        // Backward pass
        print("\n🔄 Backward Pass")
        print("-----------------")
        
        // Create gradient output (simulating loss gradient)
        let gradOutput = try await metalPipeline.allocateBuffer(shape: outputShape)
        let gradPtr = gradOutput.buffer.contents().bindMemory(to: Float.self, capacity: gradOutput.count)
        for i in 0..<gradOutput.count {
            gradPtr[i] = Float.random(in: -0.1...0.1)
        }
        
        let backwardStartTime = CFAbsoluteTimeGetCurrent()
        let gradInput = try await batchNorm.backward(gradOutput)
        let backwardTime = CFAbsoluteTimeGetCurrent() - backwardStartTime
        
        print("✅ Backward pass completed in \(String(format: "%.3f", backwardTime * 1000))ms")
        print("📐 Gradient input shape: \(gradInput.shape)")
        
        // Update parameters
        print("\n🔧 Parameter Update")
        print("-------------------")
        let learningRate: Float = 0.01
        try await batchNorm.updateParameters(gradOutput, learningRate: learningRate)
        print("✅ Parameters updated with learning rate: \(learningRate)")
        
        // Inference mode
        print("\n🎯 Inference Mode")
        print("-----------------")
        await batchNorm.setTraining(false)
        
        // Create new test data
        let testInput = try await metalPipeline.allocateBuffer(shape: inputShape)
        let testInputPtr = testInput.buffer.contents().bindMemory(to: Float.self, capacity: testInput.count)
        for i in 0..<testInput.count {
            testInputPtr[i] = Float.random(in: -1...1)
        }
        
        let inferenceStartTime = CFAbsoluteTimeGetCurrent()
        let testOutput = try await batchNorm.forward(testInput)
        let inferenceTime = CFAbsoluteTimeGetCurrent() - inferenceStartTime
        
        print("✅ Inference completed in \(String(format: "%.3f", inferenceTime * 1000))ms")
        print("📊 Using running statistics for normalization")
        
        // Parameter count
        let paramCount = await batchNorm.getParameterCount()
        print("\n📊 Total parameter count: \(paramCount)")
        print("   - Trainable parameters: \(numFeatures * 2) (gamma and beta)")
        print("   - Non-trainable parameters: \(numFeatures * 2) (running mean and variance)")
        
        print("\n✨ Batch normalization example completed successfully!")
    }
}
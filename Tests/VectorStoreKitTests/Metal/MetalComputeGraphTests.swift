import XCTest
// VectorStoreKit: Compute Graph Example
//
// Demonstrates operation optimization through graph analysis and fusion
//

import Foundation
import VectorStoreKit
import Metal

final class ComputeGraphTests: XCTestCase {
    func testMain() async throws {
        print("=== VectorStoreKit Compute Graph Example ===\n")
        
        // Create Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Error: Metal is not available on this system")
            return
        }
        
        print("Using Metal device: \(device.name)")
        
        // Create compute graph
        let graph = try ComputeGraph(device: device)
        
        // Example 1: Simple linear computation
        print("\n1. Building a simple linear computation graph...")
        await demonstrateSimpleGraph(graph: graph, device: device)
        
        // Example 2: Neural network layer with fusion
        print("\n2. Building a neural network layer with operation fusion...")
        await demonstrateNeuralNetworkLayer(graph: graph, device: device)
        
        // Example 3: Complex graph with optimization
        print("\n3. Building a complex graph with multiple optimizations...")
        await demonstrateComplexGraph(graph: graph, device: device)
    }
    
    func testDemonstrateSimpleGraph(graph: ComputeGraph, device: MTLDevice) async {
        // Clear any existing graph
        await graph.clear()
        
        // Create input nodes
        let x = await graph.addOperation(.add, inputs: [], name: "x")
        let y = await graph.addOperation(.add, inputs: [], name: "y")
        
        // Create computation: z = x + y
        let z = await graph.addOperation(.add, inputs: [x, y], name: "z")
        
        // Create computation: w = z * 2
        let two = await graph.addOperation(.add, inputs: [], name: "two")
        let w = await graph.addOperation(.multiply, inputs: [z, two], name: "w")
        
        print("Created graph: w = (x + y) * 2")
        
        // Optimize the graph
        do {
            try await graph.optimize()
            let (total, fused, reduction) = await graph.getOptimizationStats()
            print("Optimization results: \(total) operations, \(fused) fused, \(String(format: "%.1f", reduction * 100))% reduction")
        } catch {
            print("Optimization error: \(error)")
        }
        
        // Show execution order
        let executionOrder = await graph.getExecutionOrder()
        print("Execution order: \(executionOrder.count) nodes")
    }
    
    func testDemonstrateNeuralNetworkLayer(graph: ComputeGraph, device: MTLDevice) async {
        // Clear any existing graph
        await graph.clear()
        
        // Create nodes for a typical neural network layer
        let input = await graph.addOperation(.add, inputs: [], name: "input")
        let weights = await graph.addOperation(.add, inputs: [], name: "weights")
        let bias = await graph.addOperation(.add, inputs: [], name: "bias")
        
        // Build linear layer with activation
        let output = await graph.buildLinearLayer(
            input: input,
            weights: weights,
            bias: bias,
            activation: .relu,
            name: "layer1"
        )
        
        print("Created neural network layer: output = relu(input @ weights + bias)")
        
        // Optimize the graph
        do {
            try await graph.optimize()
            let (total, fused, reduction) = await graph.getOptimizationStats()
            print("Optimization results: \(total) operations, \(fused) fused, \(String(format: "%.1f", reduction * 100))% reduction")
            
            // Check if operations were fused
            if fused > 0 {
                print("✓ Operations successfully fused for better performance!")
            }
        } catch {
            print("Optimization error: \(error)")
        }
    }
    
    func testDemonstrateComplexGraph(graph: ComputeGraph, device: MTLDevice) async {
        // Clear any existing graph
        await graph.clear()
        
        // Create a more complex graph with multiple paths
        print("Building multi-layer perceptron...")
        
        // Input layer
        let input = await graph.addOperation(.add, inputs: [], name: "input")
        
        // First hidden layer (128 units)
        let w1 = await graph.addOperation(.add, inputs: [], name: "weights1")
        let b1 = await graph.addOperation(.add, inputs: [], name: "bias1")
        let h1 = await graph.buildLinearLayer(
            input: input,
            weights: w1,
            bias: b1,
            activation: .relu,
            name: "hidden1"
        )
        
        // Add dropout
        let h1_dropout = await graph.addOperation(.dropout(rate: 0.5), inputs: [h1], name: "dropout1")
        
        // Second hidden layer (64 units)
        let w2 = await graph.addOperation(.add, inputs: [], name: "weights2")
        let b2 = await graph.addOperation(.add, inputs: [], name: "bias2")
        let h2 = await graph.buildLinearLayer(
            input: h1_dropout,
            weights: w2,
            bias: b2,
            activation: .relu,
            name: "hidden2"
        )
        
        // Output layer (10 units)
        let w3 = await graph.addOperation(.add, inputs: [], name: "weights3")
        let b3 = await graph.addOperation(.add, inputs: [], name: "bias3")
        let output = await graph.buildLinearLayer(
            input: h2,
            weights: w3,
            bias: b3,
            activation: nil, // No activation on output
            name: "output"
        )
        
        // Add softmax
        let predictions = await graph.addOperation(.activation(.softmax), inputs: [output], name: "predictions")
        
        print("Created 3-layer MLP with dropout")
        
        // Optimize the graph
        let startTime = Date()
        do {
            try await graph.optimize()
            let optimizationTime = Date().timeIntervalSince(startTime)
            
            let (total, fused, reduction) = await graph.getOptimizationStats()
            print("\nOptimization completed in \(String(format: "%.3f", optimizationTime))s")
            print("Results: \(total) operations, \(fused) fused, \(String(format: "%.1f", reduction * 100))% reduction")
            
            // Show execution order
            let executionOrder = await graph.getExecutionOrder()
            print("Final execution order: \(executionOrder.count) nodes")
            
            // Demonstrate what operations were fused
            var fusedOperations: [String] = []
            for nodeId in executionOrder {
                if let node = await graph.getNode(nodeId) {
                    if !node.fusedOperations.isEmpty {
                        fusedOperations.append("\(nodeId): \(node.fusedOperations.count) operations fused")
                    }
                }
            }
            
            if !fusedOperations.isEmpty {
                print("\nFused operations:")
                for op in fusedOperations {
                    print("  - \(op)")
                }
            }
            
        } catch {
            print("Optimization error: \(error)")
        }
        
        // Demonstrate execution (with dummy data)
        print("\nSimulating execution...")
        
        // Create dummy input buffers
        var inputBuffers: [NodeID: MetalBuffer] = [:]
        
        // Add all input nodes
        let inputNodes = [input, w1, b1, w2, b2, w3, b3]
        for node in inputNodes {
            if let buffer = device.makeBuffer(length: 1024 * MemoryLayout<Float>.stride, options: .storageModeShared) {
                inputBuffers[node] = MetalBuffer(buffer: buffer, count: 1024)
            }
        }
        
        do {
            let result = try await graph.execute(inputs: inputBuffers)
            print("Execution completed:")
            print("  - Time: \(String(format: "%.3f", result.executionTime))s")
            print("  - Operations executed: \(result.operationsExecuted)")
            print("  - Operations fused: \(result.operationsFused)")
            print("  - Output buffers: \(result.outputs.count)")
        } catch {
            print("Execution error: \(error)")
        }
    }
}

// Extension to make NodeID printable
extension NodeID: CustomStringConvertible {
    public var description: String {
        return "Node(\(id.prefix(8)))"
    }
}
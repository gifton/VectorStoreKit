// VectorStoreKit: Compute Graph Tests
//
// Tests for operation optimization through graph analysis and fusion
//

import XCTest
@testable import VectorStoreKit
import Metal

@available(macOS 13.0, iOS 16.0, *)
final class ComputeGraphTests: XCTestCase {
    var device: MTLDevice!
    var graph: ComputeGraph!
    
    override func setUp() async throws {
        try await super.setUp()
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        self.device = device
        self.graph = try ComputeGraph(device: device)
    }
    
    override func tearDown() async throws {
        await graph.clear()
        graph = nil
        device = nil
        try await super.tearDown()
    }
    
    // MARK: - Basic Tests
    
    func testNodeCreation() async throws {
        // Create input nodes
        let input1 = await graph.addOperation(.add, inputs: [], name: "input1")
        let input2 = await graph.addOperation(.add, inputs: [], name: "input2")
        
        // Create operation node
        let output = await graph.addOperation(.multiply, inputs: [input1, input2], name: "output")
        
        // Verify nodes exist
        let hasInput1 = await graph.hasNode(input1)
        let hasInput2 = await graph.hasNode(input2)
        let hasOutput = await graph.hasNode(output)
        
        XCTAssertTrue(hasInput1)
        XCTAssertTrue(hasInput2)
        XCTAssertTrue(hasOutput)
        
        // Verify node properties
        if let outputNode = await graph.getNode(output) {
            XCTAssertEqual(outputNode.inputs.count, 2)
            XCTAssertTrue(outputNode.inputs.contains(input1))
            XCTAssertTrue(outputNode.inputs.contains(input2))
        } else {
            XCTFail("Output node not found")
        }
    }
    
    func testTopologicalSort() async throws {
        // Create a simple DAG: input1 -> add -> multiply -> output
        //                     input2 ----^        ^
        //                     input3 -------------'
        let input1 = await graph.addOperation(.add, inputs: [], name: "input1")
        let input2 = await graph.addOperation(.add, inputs: [], name: "input2")
        let input3 = await graph.addOperation(.add, inputs: [], name: "input3")
        
        let add = await graph.addOperation(.add, inputs: [input1, input2], name: "add")
        let multiply = await graph.addOperation(.multiply, inputs: [add, input3], name: "multiply")
        
        // Optimize to get execution order
        try await graph.optimize()
        
        let executionOrder = await graph.getExecutionOrder()
        
        // Verify topological order
        XCTAssertEqual(executionOrder.count, 5)
        
        // Inputs should come before operations that use them
        let addIndex = executionOrder.firstIndex(of: add)!
        let multiplyIndex = executionOrder.firstIndex(of: multiply)!
        let input1Index = executionOrder.firstIndex(of: input1)!
        let input2Index = executionOrder.firstIndex(of: input2)!
        let input3Index = executionOrder.firstIndex(of: input3)!
        
        XCTAssertLessThan(input1Index, addIndex)
        XCTAssertLessThan(input2Index, addIndex)
        XCTAssertLessThan(addIndex, multiplyIndex)
        XCTAssertLessThan(input3Index, multiplyIndex)
    }
    
    // MARK: - Optimization Tests
    
    func testOperationFusion() async throws {
        // Create a fusable pattern: matmul -> bias -> activation
        let input = await graph.addOperation(.add, inputs: [], name: "input")
        let weights = await graph.addOperation(.add, inputs: [], name: "weights")
        let bias = await graph.addOperation(.add, inputs: [], name: "bias")
        
        let matmul = await graph.addOperation(
            .matmul(m: 10, n: 10, k: 10),
            inputs: [input, weights],
            name: "matmul"
        )
        let biased = await graph.addOperation(
            .bias(size: 10),
            inputs: [matmul, bias],
            name: "biased"
        )
        let activated = await graph.addOperation(
            .activation(.relu),
            inputs: [biased],
            name: "activated"
        )
        
        // Get stats before optimization
        let (totalBefore, _, _) = await graph.getOptimizationStats()
        XCTAssertEqual(totalBefore, 0) // Not optimized yet
        
        // Optimize
        try await graph.optimize()
        
        // Get stats after optimization
        let (total, fused, reduction) = await graph.getOptimizationStats()
        
        XCTAssertEqual(total, 6) // All nodes
        XCTAssertGreaterThan(fused, 0) // Some operations should be fused
        XCTAssertGreaterThan(reduction, 0.0) // Should have some reduction
        
        // Check that matmul node is marked as fused
        if let matmulNode = await graph.getNode(matmul) {
            XCTAssertTrue(matmulNode.isFused)
        }
    }
    
    func testCommonSubexpressionElimination() async throws {
        // Create duplicate operations
        let input1 = await graph.addOperation(.add, inputs: [], name: "input1")
        let input2 = await graph.addOperation(.add, inputs: [], name: "input2")
        
        // Two identical add operations
        let add1 = await graph.addOperation(.add, inputs: [input1, input2])
        let add2 = await graph.addOperation(.add, inputs: [input1, input2])
        
        // Use both results
        let _ = await graph.addOperation(.multiply, inputs: [add1, add2])
        
        // Optimize
        try await graph.optimize()
        
        // One of the adds should be eliminated
        let (total, fused, _) = await graph.getOptimizationStats()
        XCTAssertGreaterThan(fused, 0)
    }
    
    // MARK: - Execution Tests
    
    func testSimpleExecution() async throws {
        // Create simple graph: input -> activation -> output
        let input = await graph.addOperation(.add, inputs: [], name: "input")
        let activated = await graph.addOperation(.activation(.relu), inputs: [input])
        
        // Create input buffer
        let inputBuffer = MetalBuffer(
            buffer: device.makeBuffer(length: 100 * MemoryLayout<Float>.stride, options: .storageModeShared)!,
            count: 100
        )
        
        // Execute
        let result = try await graph.execute(inputs: [input: inputBuffer])
        
        // Verify execution
        XCTAssertEqual(result.operationsExecuted, 1) // Only activation executed (input is provided)
        XCTAssertNotNil(result.outputs[activated])
    }
    
    func testLinearLayerBuilder() async throws {
        // Create inputs
        let input = await graph.addOperation(.add, inputs: [], name: "input")
        let weights = await graph.addOperation(.add, inputs: [], name: "weights")
        let bias = await graph.addOperation(.add, inputs: [], name: "bias")
        
        // Build linear layer
        let output = await graph.buildLinearLayer(
            input: input,
            weights: weights,
            bias: bias,
            activation: .relu,
            name: "linear"
        )
        
        // Verify structure
        XCTAssertTrue(await graph.hasNode(output))
        
        // Should have created matmul, bias, and activation nodes
        let executionOrder = await graph.getExecutionOrder()
        XCTAssertGreaterThanOrEqual(executionOrder.count, 6) // 3 inputs + 3 operations
    }
    
    // MARK: - Error Handling Tests
    
    func testCycleDetection() async throws {
        // Try to create a cycle
        let node1 = await graph.addOperation(.add, inputs: [], name: "node1")
        let node2 = await graph.addOperation(.add, inputs: [node1], name: "node2")
        
        // This would create a cycle if we could modify inputs after creation
        // Since we can't create cycles with our current API, this test just verifies
        // that optimization doesn't fail on valid DAGs
        do {
            try await graph.optimize()
        } catch {
            XCTFail("Optimization should not fail on valid DAG")
        }
    }
    
    // MARK: - Performance Tests
    
    func testLargeGraphOptimization() async throws {
        // Create a large graph
        var previousNodes: [NodeID] = []
        
        // Create initial layer
        for i in 0..<10 {
            let node = await graph.addOperation(.add, inputs: [], name: "input\(i)")
            previousNodes.append(node)
        }
        
        // Create multiple layers
        for layer in 0..<5 {
            var newNodes: [NodeID] = []
            
            for i in 0..<8 {
                // Connect to multiple previous nodes
                let inputs = Array(previousNodes.shuffled().prefix(3))
                let node = await graph.addOperation(
                    layer % 2 == 0 ? .add : .multiply,
                    inputs: inputs,
                    name: "layer\(layer)_node\(i)"
                )
                newNodes.append(node)
            }
            
            previousNodes = newNodes
        }
        
        // Measure optimization time
        let start = Date()
        try await graph.optimize()
        let optimizationTime = Date().timeIntervalSince(start)
        
        // Should optimize reasonably quickly
        XCTAssertLessThan(optimizationTime, 1.0) // Less than 1 second
        
        let (total, fused, reduction) = await graph.getOptimizationStats()
        print("Large graph optimization: \(total) operations, \(fused) fused, \(reduction * 100)% reduction")
    }
}
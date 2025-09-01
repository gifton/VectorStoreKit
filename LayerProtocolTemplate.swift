// LayerProtocolTemplate.swift
// Template for fixing layer protocol conformance issues in VectorStoreKit
// This is a temporary file to guide layer fixes

import Foundation
import Metal

/// Template implementation of NeuralLayer protocol
/// Copy this template and fill in the TODO sections for each layer type
actor LayerTemplate: NeuralLayer {
    // MARK: - Properties
    
    // TODO: Add layer-specific properties
    private let metalPipeline: MetalMLPipeline
    private var isTraining: Bool = true
    
    // TODO: Add parameter names for this layer
    // Example:
    // private let weightsName: String
    // private let biasName: String
    
    // Parameter store
    private let parameterStore: ParameterStore
    
    // TODO: Add cached buffers for backward pass if needed
    // Example:
    // private var lastInput: MetalBuffer?
    // private var lastOutput: MetalBuffer?
    
    // MARK: - Initialization
    
    init(
        // TODO: Add layer-specific parameters
        metalPipeline: MetalMLPipeline
    ) async throws {
        self.metalPipeline = metalPipeline
        self.parameterStore = await ParameterStore(device: metalPipeline.device)
        
        // TODO: Initialize layer-specific properties
        
        // TODO: Initialize parameters if this layer has any
        // Example:
        // try await initializeParameters()
    }
    
    // MARK: - NeuralLayer Protocol Implementation
    
    /// Forward pass through the layer
    func forward(_ input: MetalBuffer) async throws -> MetalBuffer {
        // TODO: Implement forward pass logic
        // Example structure:
        // 1. Cache input if needed for backward pass
        // 2. Allocate output buffer
        // 3. Perform computation (using Metal kernels or CPU)
        // 4. Cache output if needed
        // 5. Return output
        
        fatalError("TODO: Implement forward pass")
    }
    
    /// Backward pass through the layer
    func backward(_ gradOutput: MetalBuffer) async throws -> MetalBuffer {
        // TODO: Implement backward pass logic
        // Example structure:
        // 1. Retrieve cached values from forward pass
        // 2. Compute gradients w.r.t. parameters (if any)
        // 3. Store parameter gradients
        // 4. Compute gradients w.r.t. input
        // 5. Return input gradients
        
        fatalError("TODO: Implement backward pass")
    }
    
    /// Update layer parameters using gradients
    func updateParameters(_ gradients: MetalBuffer, learningRate: Float) async throws {
        // TODO: Implement parameter updates
        // If layer has no parameters, just return
        // Example for layers with parameters:
        // 1. Get parameter buffers
        // 2. Apply gradient descent update
        // 3. Handle any constraints (e.g., weight clipping)
        
        // For layers without parameters:
        // return
        
        fatalError("TODO: Implement parameter updates or return if no parameters")
    }
    
    /// Get current parameters as Metal buffer
    func getParameters() async -> MetalBuffer? {
        // TODO: Return parameters if layer has any
        // For layers without parameters:
        // return nil
        
        // For layers with parameters:
        // 1. Concatenate all parameters into single buffer
        // 2. Return the buffer
        
        fatalError("TODO: Return parameters or nil")
    }
    
    /// Number of trainable parameters
    func getParameterCount() async -> Int {
        // TODO: Return total parameter count
        // For layers without parameters:
        // return 0
        
        // For layers with parameters:
        // return weightsCount + biasCount + ...
        
        fatalError("TODO: Return parameter count")
    }
    
    /// Set training mode
    /// IMPORTANT: This must be nonisolated to match protocol requirement!
    nonisolated func setTraining(_ training: Bool) {
        // We need to use a task to update the actor's state
        Task { @MainActor in
            await self.updateTrainingMode(training)
        }
    }
    
    /// Internal async method to update training mode
    private func updateTrainingMode(_ training: Bool) async {
        self.isTraining = training
        // TODO: Add any layer-specific training mode changes
    }
    
    /// Zero out accumulated gradients
    func zeroGradients() async {
        // TODO: Zero out gradient buffers if layer accumulates gradients
        // Most layers don't need to override this
        // The default implementation in the protocol extension is usually sufficient
    }
    
    /// Scale gradients by a factor
    func scaleGradients(_ scale: Float) async {
        // TODO: Scale gradient buffers if layer has gradients
        // Most layers don't need to override this
        // The default implementation in the protocol extension is usually sufficient
    }
    
    /// Update parameters using an optimizer
    func updateParametersWithOptimizer(_ optimizer: any Optimizer) async throws {
        // TODO: Use optimizer to update parameters
        // The default implementation often suffices, but can be overridden for custom behavior
        // Default implementation:
        if let params = await getParameters() {
            try await updateParameters(params, learningRate: await optimizer.getCurrentLearningRate())
        }
    }
    
    // MARK: - Helper Methods
    
    // TODO: Add any layer-specific helper methods
    // Example:
    // private func initializeParameters() async throws {
    //     // Initialize weights and biases
    // }
}

// MARK: - Notes for Implementation

/*
 Key Points When Implementing Layers:
 
 1. The `setTraining` method MUST be `nonisolated` to match the protocol.
    Use Task to update actor state asynchronously.
 
 2. All other protocol methods should be regular actor methods (isolated).
 
 3. For layers without parameters (activation layers, dropout, etc.):
    - updateParameters should do nothing
    - getParameters should return nil
    - getParameterCount should return 0
 
 4. For layers with parameters (dense, conv, normalization):
    - Track parameter buffers in ParameterStore
    - Implement proper initialization (He, Xavier, etc.)
    - Cache necessary values during forward pass for backward
 
 5. Memory management:
    - Use metalPipeline.allocateBuffer() for temporary buffers
    - Call metalPipeline.releaseBuffer() when done
    - Be careful about buffer lifetimes across forward/backward
 
 6. Error handling:
    - Check for nil cached values in backward pass
    - Validate buffer sizes match expectations
    - Handle Metal kernel failures gracefully
 
 7. Performance considerations:
    - Minimize CPU-GPU synchronization
    - Use Metal kernels for heavy computations
    - Batch operations when possible
*/
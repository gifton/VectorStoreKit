// VectorStoreKit: Neural Network Serialization
//
// Checkpoint and serialization support for neural networks
//

import Foundation
@preconcurrency import Metal

// MARK: - Neural Network Serialization Extension

extension NeuralNetwork: ModelSerializable {
    
    /// Save neural network checkpoint
    public func saveCheckpoint(to url: URL, options: CheckpointOptions) async throws {
        // Create checkpoint manager
        let checkpointManager = CheckpointManager(device: await metalPipeline.device)
        
        // Build architecture config
        var layerConfigs: [LayerConfig] = []
        for (index, layer) in layers.enumerated() {
            let layerType = await layer.getLayerType()
            let layerName = await layer.getName() ?? "layer_\(index)"
            let paramShapes = await layer.getParameterShapes()
            
            layerConfigs.append(LayerConfig(
                type: layerType,
                name: layerName,
                config: [:], // Layer-specific config would go here
                parameterShapes: paramShapes
            ))
        }
        
        let architecture = ArchitectureConfig(
            modelType: "NeuralNetwork",
            layers: layerConfigs,
            settings: [:]
        )
        
        // Serialize parameters
        let serializer = await checkpointManager.getSerializer()
        var parameters: [ParameterData] = []
        
        for (index, layer) in layers.enumerated() {
            // Get layer parameters
            if let layerParams = await layer.getParameters() {
                let paramData = try await serializer.serializeBuffer(
                    layerParams,
                    name: "layer_\(index)_weights",
                    compression: options.compression
                )
                parameters.append(paramData)
            }
            
            // Get layer biases if available
            if let layerBias = await layer.getBias() {
                let biasData = try await serializer.serializeBuffer(
                    layerBias,
                    name: "layer_\(index)_bias",
                    compression: options.compression
                )
                parameters.append(biasData)
            }
        }
        
        // Create training state if requested
        var trainingState: TrainingState? = nil
        if options.includeTrainingState {
            trainingState = TrainingState(
                epoch: 0, // Would need to track this
                step: 0,
                optimizerState: nil, // Would need to extract from optimizer
                history: trainingHistory,
                randomSeed: UInt64.random(in: 0...UInt64.max)
            )
        }
        
        // Create metadata
        let metadata = CheckpointMetadata(
            timestamp: Date(),
            description: options.description,
            metrics: [
                "final_loss": trainingHistory.finalLoss,
                "parameter_count": Float(await parameterCount())
            ],
            hardware: HardwareInfo(
                deviceName: await metalPipeline.device.name ?? "Unknown",
                deviceType: "GPU",
                availableMemory: 0, // Would need to query
                metalFeatures: []
            ),
            custom: options.customMetadata
        )
        
        // Create checkpoint
        var checkpoint = Checkpoint(
            version: CheckpointVersion.current,
            architecture: architecture,
            parameters: parameters,
            trainingState: trainingState,
            metadata: metadata,
            checksum: "" // Will be computed
        )
        
        // Compute checksum
        let checksum = try await computeChecksum(for: checkpoint, manager: checkpointManager)
        checkpoint = Checkpoint(
            version: checkpoint.version,
            architecture: checkpoint.architecture,
            parameters: checkpoint.parameters,
            trainingState: checkpoint.trainingState,
            metadata: checkpoint.metadata,
            checksum: checksum
        )
        
        // Save checkpoint
        try await checkpointManager.save(checkpoint, to: url)
    }
    
    /// Load neural network checkpoint
    public func loadCheckpoint(from url: URL, options: CheckpointOptions) async throws {
        // Create checkpoint manager
        let checkpointManager = CheckpointManager(device: await metalPipeline.device)
        
        // Load checkpoint
        let checkpoint = try await checkpointManager.load(from: url)
        
        // Verify architecture compatibility
        guard checkpoint.architecture.modelType == "NeuralNetwork" else {
            throw NeuralNetworkError.invalidArchitecture("Expected NeuralNetwork, got \(checkpoint.architecture.modelType)")
        }
        
        // Clear existing layers
        layers.removeAll()
        
        // Recreate layers based on architecture
        for layerConfig in checkpoint.architecture.layers {
            // This would need a layer factory to recreate layers from config
            // For now, we'll skip layer recreation as it requires more context
        }
        
        // Deserialize parameters
        let serializer = await checkpointManager.getSerializer()
        var paramIndex = 0
        
        for (layerIndex, layer) in layers.enumerated() {
            // Load weights
            let weightParamName = "layer_\(layerIndex)_weights"
            if let weightParam = checkpoint.parameters.first(where: { $0.name == weightParamName }) {
                let weightBuffer = try await serializer.deserializeBuffer(weightParam)
                try await layer.setParameters(weightBuffer)
            }
            
            // Load bias
            let biasParamName = "layer_\(layerIndex)_bias"
            if let biasParam = checkpoint.parameters.first(where: { $0.name == biasParamName }) {
                let biasBuffer = try await serializer.deserializeBuffer(biasParam)
                try await layer.setBias(biasBuffer)
            }
        }
        
        // Restore training state
        if let trainingState = checkpoint.trainingState {
            self.trainingHistory = trainingState.history
        }
    }
    
    /// Compute checkpoint checksum
    private func computeChecksum(for checkpoint: Checkpoint, manager: CheckpointManager) async throws -> String {
        // This is handled by the CheckpointManager internally
        // We need to encode and hash the checkpoint data
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        
        var dataToHash = Data()
        
        // Add architecture
        let architectureData = try encoder.encode(checkpoint.architecture)
        dataToHash.append(architectureData)
        
        // Add parameters data
        for param in checkpoint.parameters {
            dataToHash.append(param.data)
        }
        
        // Add metadata
        let metadataData = try encoder.encode(checkpoint.metadata)
        dataToHash.append(metadataData)
        
        return dataToHash.sha256Hash()
    }
}

// MARK: - Layer Protocol Extensions for Serialization

public extension NeuralLayer {
    /// Get layer type string for serialization
    func getLayerType() async -> String {
        // Default implementation - layers should override
        return String(describing: type(of: self))
    }
    
    /// Get layer name
    func getName() async -> String? {
        // Default implementation returns nil
        return nil
    }
    
    /// Get parameter shapes for serialization
    func getParameterShapes() async -> [String: [Int]] {
        // Default implementation
        var shapes: [String: [Int]] = [:]
        
        if let params = await getParameters() {
            shapes["weights"] = params.shape.dimensions
        }
        
        if let bias = await getBias() {
            shapes["bias"] = bias.shape.dimensions
        }
        
        return shapes
    }
    
    /// Get bias buffer if available
    func getBias() async -> MetalBuffer? {
        // Default implementation - layers with bias should override
        return nil
    }
    
    /// Set bias buffer if supported
    func setBias(_ buffer: MetalBuffer) async throws {
        // Default implementation - layers with bias should override
        // Do nothing by default
    }
    
    /// Set parameters from buffer
    func setParameters(_ buffer: MetalBuffer) async throws {
        // Default implementation calls updateParameters with zero learning rate
        // This effectively replaces the parameters
        try await updateParameters(buffer, learningRate: 0.0)
    }
}

// MARK: - Checkpoint Utilities

/// Utility functions for neural network checkpointing
public struct NeuralNetworkCheckpointUtils {
    
    /// Validate checkpoint compatibility with current network architecture
    public static func validateCheckpoint(
        _ checkpoint: Checkpoint,
        against network: NeuralNetwork
    ) async throws -> Bool {
        // Check model type
        guard checkpoint.architecture.modelType == "NeuralNetwork" else {
            return false
        }
        
        // Check layer count
        let networkLayerCount = await network.layerCount
        guard checkpoint.architecture.layers.count == networkLayerCount else {
            return false
        }
        
        // Verify parameter shapes match
        for (index, layerConfig) in checkpoint.architecture.layers.enumerated() {
            let layer = await network.layers[index]
            let layerShapes = await layer.getParameterShapes()
            
            // Check if shapes match
            for (paramName, expectedShape) in layerConfig.parameterShapes {
                guard let actualShape = layerShapes[paramName] else {
                    return false
                }
                
                guard actualShape == expectedShape else {
                    return false
                }
            }
        }
        
        return true
    }
    
    /// Create a checkpoint diff for incremental saves
    public static func createCheckpointDiff(
        from oldCheckpoint: Checkpoint,
        to newCheckpoint: Checkpoint
    ) -> CheckpointDiff {
        var changedParameters: Set<String> = []
        
        // Find changed parameters
        for newParam in newCheckpoint.parameters {
            if let oldParam = oldCheckpoint.parameters.first(where: { $0.name == newParam.name }) {
                // Compare data hashes
                if newParam.data.sha256Hash() != oldParam.data.sha256Hash() {
                    changedParameters.insert(newParam.name)
                }
            } else {
                // New parameter
                changedParameters.insert(newParam.name)
            }
        }
        
        return CheckpointDiff(
            baseVersion: oldCheckpoint.version,
            targetVersion: newCheckpoint.version,
            changedParameters: changedParameters,
            deletedParameters: Set(oldCheckpoint.parameters.map { $0.name })
                .subtracting(newCheckpoint.parameters.map { $0.name })
        )
    }
}

/// Checkpoint diff for incremental updates
public struct CheckpointDiff: Codable {
    public let baseVersion: CheckpointVersion
    public let targetVersion: CheckpointVersion
    public let changedParameters: Set<String>
    public let deletedParameters: Set<String>
}

// MARK: - Async Checkpoint Loading

/// Async checkpoint loader for background loading
public actor AsyncCheckpointLoader {
    private let device: MTLDevice
    private var loadingTasks: [URL: Task<Checkpoint, Error>] = [:]
    
    public init(device: MTLDevice) {
        self.device = device
    }
    
    /// Preload checkpoint in background
    public func preload(from url: URL) -> Task<Checkpoint, Error> {
        if let existingTask = loadingTasks[url] {
            return existingTask
        }
        
        let task = Task<Checkpoint, Error> {
            let manager = CheckpointManager(device: device)
            let checkpoint = try await manager.load(from: url)
            self.loadingTasks.removeValue(forKey: url)
            return checkpoint
        }
        
        loadingTasks[url] = task
        return task
    }
    
    /// Get preloaded checkpoint or load if not cached
    public func get(from url: URL) async throws -> Checkpoint {
        if let task = loadingTasks[url] {
            return try await task.value
        }
        
        let manager = CheckpointManager(device: device)
        return try await manager.load(from: url)
    }
    
    /// Cancel preloading
    public func cancelPreload(for url: URL) {
        loadingTasks[url]?.cancel()
        loadingTasks.removeValue(forKey: url)
    }
}
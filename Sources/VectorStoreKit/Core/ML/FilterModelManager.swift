// VectorStoreKit: Filter Model Manager
//
// Production-ready model management for learned filter evaluation

import Foundation
@preconcurrency import Metal
import os.log

/// Model registry for filter evaluation models
public actor FilterModelRegistry {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let logger = Logger(subsystem: "VectorStoreKit", category: "FilterModelRegistry")
    
    // Model storage
    private var models: [String: FilterModel] = [:]
    private var modelVersions: [String: [Version: FilterModel]] = [:]
    private var activeVersions: [String: Version] = [:]
    
    // Caching configuration
    private let cacheConfiguration: CacheConfiguration
    private var accessHistory: GenericLRUCache<String, Date>
    private var memoryUsage: Int = 0
    
    // Hot-reload support
    private var modelWatchers: [String: ModelWatcher] = [:]
    private var reloadCallbacks: [(String, Version) -> Void] = []
    
    // Performance metrics
    private var metrics = FilterModelMetrics()
    
    // MARK: - Types
    
    public struct Version: Comparable, Hashable, Sendable {
        let major: Int
        let minor: Int
        let patch: Int
        
        public static func < (lhs: Version, rhs: Version) -> Bool {
            if lhs.major != rhs.major { return lhs.major < rhs.major }
            if lhs.minor != rhs.minor { return lhs.minor < rhs.minor }
            return lhs.patch < rhs.patch
        }
        
        var string: String { "\(major).\(minor).\(patch)" }
    }
    
    public struct CacheConfiguration: Sendable {
        let maxModels: Int
        let maxMemoryMB: Int
        let evictionPolicy: EvictionPolicy
        let preloadModels: [String]
        
        public enum EvictionPolicy: Sendable {
            case lru
            case lfu
            case fifo
            case adaptive
        }
        
        public static let `default` = CacheConfiguration(
            maxModels: 10,
            maxMemoryMB: 512,
            evictionPolicy: .lru,
            preloadModels: []
        )
    }
    
    public struct FilterModel: Sendable {
        let id: String
        let version: Version
        let network: NeuralNetwork
        let metadata: ModelMetadata
        let inputDimension: Int
        let loadedAt: Date
        let estimatedMemoryMB: Int
    }
    
    public struct ModelMetadata: Sendable, Codable {
        let trainingDate: Date
        let accuracy: Float
        let f1Score: Float
        let trainingDatasetSize: Int
        let architecture: String
        let hyperparameters: [String: String]
        let checksum: String
    }
    
    // MARK: - Initialization
    
    public init(
        metalPipeline: MetalMLPipeline,
        cacheConfiguration: CacheConfiguration = .default
    ) async throws {
        self.metalPipeline = metalPipeline
        self.cacheConfiguration = cacheConfiguration
        self.accessHistory = GenericLRUCache(capacity: cacheConfiguration.maxModels)
        
        // Preload models if specified
        for modelId in cacheConfiguration.preloadModels {
            do {
                _ = try await loadModel(modelId)
                logger.info("Preloaded model: \(modelId)")
            } catch {
                logger.error("Failed to preload model \(modelId): \(error)")
            }
        }
    }
    
    // MARK: - Model Loading
    
    /// Load a model with automatic version management
    public func loadModel(_ modelId: String, version: Version? = nil) async throws -> FilterModel {
        // Check if model is already loaded
        let cacheKey = modelCacheKey(modelId: modelId, version: version)
        if let existingModel = getCachedModel(modelId: modelId, version: version) {
            await updateAccessHistory(cacheKey)
            metrics.cacheHits += 1
            return existingModel
        }
        
        metrics.cacheMisses += 1
        
        // Check memory constraints before loading
        let modelPath = getModelPath(modelId: modelId, version: version)
        let estimatedSize = try await estimateModelSize(at: modelPath)
        
        if needsEviction(forSize: estimatedSize) {
            try await evictModelsForSpace(estimatedSize)
        }
        
        // Load model from disk
        let startTime = Date()
        let model = try await loadModelFromDisk(
            modelId: modelId,
            version: version ?? getLatestVersion(for: modelId),
            path: modelPath
        )
        
        // Cache the model
        cacheModel(model)
        await updateAccessHistory(cacheKey)
        
        let loadTime = Date().timeIntervalSince(startTime)
        metrics.totalLoadTime += loadTime
        metrics.modelsLoaded += 1
        
        logger.info("Loaded model \(modelId) v\(model.version.string) in \(String(format: "%.2f", loadTime))s")
        
        return model
    }
    
    /// Load model from disk with validation
    private func loadModelFromDisk(
        modelId: String,
        version: Version,
        path: URL
    ) async throws -> FilterModel {
        // Load metadata
        let metadataPath = path.appendingPathComponent("metadata.json")
        let metadataData = try Data(contentsOf: metadataPath)
        let metadata = try JSONDecoder().decode(ModelMetadata.self, from: metadataData)
        
        // Verify checksum
        let modelData = try Data(contentsOf: path.appendingPathComponent("model.vsm"))
        let checksum = computeChecksum(for: modelData)
        guard checksum == metadata.checksum else {
            throw VectorStoreError(
                category: .serialization,
                code: .dataCorrupted,
                message: "Model checksum mismatch for \(modelId)"
            )
        }
        
        // Create neural network
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Load network weights
        try await loadNetworkWeights(
            network: network,
            from: path.appendingPathComponent("weights"),
            architecture: metadata.architecture
        )
        
        // Determine input dimension from architecture
        let inputDimension = try parseInputDimension(from: metadata.architecture)
        
        // Estimate memory usage
        let parameterCount = await network.getTotalParameters()
        let estimatedMemoryMB = (parameterCount * MemoryLayout<Float>.size) / (1024 * 1024)
        
        return FilterModel(
            id: modelId,
            version: version,
            network: network,
            metadata: metadata,
            inputDimension: inputDimension,
            loadedAt: Date(),
            estimatedMemoryMB: estimatedMemoryMB
        )
    }
    
    /// Load network weights based on architecture
    private func loadNetworkWeights(
        network: NeuralNetwork,
        from weightsPath: URL,
        architecture: String
    ) async throws {
        // Parse architecture string (e.g., "linear_512_1" or "mlp_512_256_128_1")
        let components = architecture.split(separator: "_")
        guard components.count >= 2 else {
            throw VectorStoreError.invalidInput(
                value: architecture,
                reason: "Invalid architecture format"
            )
        }
        
        let layerType = String(components[0])
        let dimensions = components.dropFirst().compactMap { Int($0) }
        
        switch layerType {
        case "linear":
            guard dimensions.count == 2 else {
                throw VectorStoreError.invalidInput(
                    value: architecture,
                    reason: "Linear architecture requires 2 dimensions"
                )
            }
            
            let layer = try await DenseLayer(
                inputSize: dimensions[0],
                outputSize: dimensions[1],
                activation: .sigmoid,
                metalPipeline: metalPipeline
            )
            
            // Load weights for this layer
            try await loadLayerWeights(
                layer: layer,
                from: weightsPath.appendingPathComponent("layer_0")
            )
            
            await network.addLayer(layer)
            
        case "mlp":
            guard dimensions.count >= 2 else {
                throw VectorStoreError.invalidInput(
                    value: architecture,
                    reason: "MLP architecture requires at least 2 dimensions"
                )
            }
            
            // Create MLP layers
            for i in 0..<(dimensions.count - 1) {
                let activation: ActivationType = (i == dimensions.count - 2) ? .sigmoid : .relu
                
                let layer = try await DenseLayer(
                    inputSize: dimensions[i],
                    outputSize: dimensions[i + 1],
                    activation: activation,
                    metalPipeline: metalPipeline
                )
                
                // Load weights for this layer
                try await loadLayerWeights(
                    layer: layer,
                    from: weightsPath.appendingPathComponent("layer_\(i)")
                )
                
                await network.addLayer(layer)
            }
            
        case "attention":
            // Support for attention-based filter models
            guard dimensions.count >= 3 else {
                throw VectorStoreError.invalidInput(
                    value: architecture,
                    reason: "Attention architecture requires at least 3 dimensions"
                )
            }
            
            // Implementation would create attention layers
            throw VectorStoreError(
                category: .configuration,
                code: .formatNotSupported,
                message: "Attention architecture not yet implemented"
            )
            
        default:
            throw VectorStoreError.invalidInput(
                value: layerType,
                reason: "Unsupported architecture type"
            )
        }
    }
    
    /// Load weights for a specific layer
    private func loadLayerWeights(layer: DenseLayer, from path: URL) async throws {
        let weightsData = try Data(contentsOf: path.appendingPathComponent("weights.bin"))
        let biasData = try Data(contentsOf: path.appendingPathComponent("bias.bin"))
        
        // Convert data to Float arrays
        let weights = weightsData.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self))
        }
        
        let bias = biasData.withUnsafeBytes { bytes in
            Array(bytes.bindMemory(to: Float.self))
        }
        
        // Load into layer
        try await layer.loadWeights(weights: weights, bias: bias)
    }
    
    // MARK: - Model Management
    
    /// Update model version
    public func updateModel(
        _ modelId: String,
        to version: Version,
        notify: Bool = true
    ) async throws {
        let model = try await loadModel(modelId, version: version)
        
        // Update active version
        activeVersions[modelId] = version
        
        // Store in version history
        if modelVersions[modelId] == nil {
            modelVersions[modelId] = [:]
        }
        modelVersions[modelId]?[version] = model
        
        // Notify callbacks if requested
        if notify {
            for callback in reloadCallbacks {
                callback(modelId, version)
            }
        }
        
        metrics.modelUpdates += 1
        logger.info("Updated model \(modelId) to version \(version.string)")
    }
    
    /// Get model for evaluation
    public func getModel(_ modelId: String, version: Version? = nil) async throws -> FilterModel {
        return try await loadModel(modelId, version: version)
    }
    
    /// Evaluate a vector using a model
    public func evaluate(
        modelId: String,
        vector: [Float],
        confidence: Float = 0.5
    ) async throws -> Bool {
        let startTime = Date()
        
        let model = try await getModel(modelId)
        
        // Validate input dimension
        guard vector.count == model.inputDimension else {
            throw VectorStoreError.dimensionMismatch(
                expected: model.inputDimension,
                actual: vector.count
            )
        }
        
        // Create input buffer
        let inputBuffer = try await metalPipeline.allocateBuffer(shape: TensorShape(vector.count))
        defer {
            Task {
                await metalPipeline.releaseBuffer(inputBuffer)
            }
        }
        
        // Copy input data
        let inputPtr = inputBuffer.buffer.contents().bindMemory(
            to: Float.self,
            capacity: vector.count
        )
        for i in 0..<vector.count {
            inputPtr[i] = vector[i]
        }
        
        // Forward pass
        let outputBuffer = try await model.network.forward(inputBuffer)
        defer {
            Task {
                await metalPipeline.releaseBuffer(outputBuffer)
            }
        }
        
        // Extract prediction
        let outputPtr = outputBuffer.buffer.contents().bindMemory(to: Float.self, capacity: 1)
        let prediction = outputPtr[0]
        
        // Update metrics
        let evaluationTime = Date().timeIntervalSince(startTime)
        metrics.totalEvaluationTime += evaluationTime
        metrics.evaluations += 1
        
        return prediction >= confidence
    }
    
    // MARK: - Cache Management
    
    private func getCachedModel(modelId: String, version: Version?) -> FilterModel? {
        if let version = version {
            return modelVersions[modelId]?[version]
        } else if let activeVersion = activeVersions[modelId] {
            return modelVersions[modelId]?[activeVersion]
        } else {
            return models[modelId]
        }
    }
    
    private func cacheModel(_ model: FilterModel) {
        models[model.id] = model
        
        if modelVersions[model.id] == nil {
            modelVersions[model.id] = [:]
        }
        modelVersions[model.id]?[model.version] = model
        
        memoryUsage += model.estimatedMemoryMB
    }
    
    private func needsEviction(forSize size: Int) -> Bool {
        let totalSize = memoryUsage + size
        return totalSize > cacheConfiguration.maxMemoryMB ||
               models.count >= cacheConfiguration.maxModels
    }
    
    private func evictModelsForSpace(_ requiredSize: Int) async throws {
        var freedSpace = 0
        var modelsToEvict: [String] = []
        
        // Get least recently used models
        let sortedModels = accessHistory.getAllKeys().reversed()
        
        for modelKey in sortedModels {
            guard freedSpace < requiredSize else { break }
            
            if let model = models[modelKey] {
                modelsToEvict.append(modelKey)
                freedSpace += model.estimatedMemoryMB
            }
        }
        
        // Evict models
        for modelId in modelsToEvict {
            if let model = models.removeValue(forKey: modelId) {
                memoryUsage -= model.estimatedMemoryMB
                accessHistory.remove(modelId)
                metrics.evictions += 1
                logger.info("Evicted model \(modelId) to free \(model.estimatedMemoryMB)MB")
            }
        }
    }
    
    private func updateAccessHistory(_ key: String) async {
        accessHistory.set(key, Date())
    }
    
    // MARK: - Hot Reload
    
    /// Enable hot reload for a model
    public func enableHotReload(
        for modelId: String,
        path: URL,
        callback: @escaping (String, Version) -> Void
    ) async {
        let watcher = ModelWatcher(
            modelId: modelId,
            path: path,
            callback: { [weak self] modelId, version in
                Task {
                    guard let self = self else { return }
                    do {
                        try await self.updateModel(modelId, to: version)
                        callback(modelId, version)
                    } catch {
                        await self.logger.error("Hot reload failed for \(modelId): \(error)")
                    }
                }
            }
        )
        
        modelWatchers[modelId] = watcher
        await watcher.start()
        
        logger.info("Enabled hot reload for model \(modelId)")
    }
    
    /// Disable hot reload for a model
    public func disableHotReload(for modelId: String) async {
        if let watcher = modelWatchers.removeValue(forKey: modelId) {
            await watcher.stop()
            logger.info("Disabled hot reload for model \(modelId)")
        }
    }
    
    /// Register callback for model updates
    public func onModelUpdate(_ callback: @escaping (String, Version) -> Void) {
        reloadCallbacks.append(callback)
    }
    
    // MARK: - Metrics
    
    /// Get performance metrics
    public func getMetrics() -> FilterModelMetrics {
        var metrics = self.metrics
        metrics.currentCacheSize = models.count
        metrics.memoryUsageMB = memoryUsage
        
        if metrics.modelsLoaded > 0 {
            metrics.averageLoadTimeMs = (metrics.totalLoadTime / Double(metrics.modelsLoaded)) * 1000
        }
        
        if metrics.evaluations > 0 {
            metrics.averageEvaluationTimeMs = (metrics.totalEvaluationTime / Double(metrics.evaluations)) * 1000
        }
        
        let totalRequests = metrics.cacheHits + metrics.cacheMisses
        if totalRequests > 0 {
            metrics.cacheHitRate = Float(metrics.cacheHits) / Float(totalRequests)
        }
        
        return metrics
    }
    
    /// Reset metrics
    public func resetMetrics() {
        metrics = FilterModelMetrics()
    }
    
    // MARK: - Helper Methods
    
    private func modelCacheKey(modelId: String, version: Version?) -> String {
        if let version = version {
            return "\(modelId)_v\(version.string)"
        } else {
            return modelId
        }
    }
    
    private func getModelPath(modelId: String, version: Version?) -> URL {
        let documentsPath = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first!
        
        let modelsDirectory = documentsPath.appendingPathComponent("VectorStoreKit/Models")
        
        if let version = version {
            return modelsDirectory
                .appendingPathComponent(modelId)
                .appendingPathComponent(version.string)
        } else {
            // Get latest version
            return modelsDirectory
                .appendingPathComponent(modelId)
                .appendingPathComponent("latest")
        }
    }
    
    private func getLatestVersion(for modelId: String) -> Version {
        // In production, this would scan the model directory
        // For now, return a default version
        return Version(major: 1, minor: 0, patch: 0)
    }
    
    private func estimateModelSize(at path: URL) async throws -> Int {
        let fileManager = FileManager.default
        
        guard fileManager.fileExists(atPath: path.path) else {
            throw VectorStoreError(
                category: .storage,
                code: .storageUnavailable,
                message: "Model not found at path: \(path.path)"
            )
        }
        
        // Calculate total size of model files
        var totalSize: Int64 = 0
        let enumerator = fileManager.enumerator(at: path, includingPropertiesForKeys: [.fileSizeKey])
        
        while let fileURL = enumerator?.nextObject() as? URL {
            if let fileSize = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                totalSize += Int64(fileSize)
            }
        }
        
        return Int(totalSize / (1024 * 1024)) // Convert to MB
    }
    
    private func computeChecksum(for data: Data) -> String {
        // Simple checksum using CRC32
        var crc: UInt32 = 0xFFFFFFFF
        
        for byte in data {
            crc ^= UInt32(byte)
            for _ in 0..<8 {
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320 : crc >> 1
            }
        }
        
        return String(format: "%08X", ~crc)
    }
    
    private func parseInputDimension(from architecture: String) throws -> Int {
        let components = architecture.split(separator: "_")
        guard components.count >= 2,
              let dimension = Int(components[1]) else {
            throw VectorStoreError.invalidInput(
                value: architecture,
                reason: "Cannot parse input dimension from architecture"
            )
        }
        return dimension
    }
}

// MARK: - Supporting Types

/// Performance metrics for filter models
public struct FilterModelMetrics: Sendable {
    public var cacheHits: Int = 0
    public var cacheMisses: Int = 0
    public var cacheHitRate: Float = 0
    public var modelsLoaded: Int = 0
    public var modelUpdates: Int = 0
    public var evictions: Int = 0
    public var evaluations: Int = 0
    public var totalLoadTime: TimeInterval = 0
    public var totalEvaluationTime: TimeInterval = 0
    public var averageLoadTimeMs: Double = 0
    public var averageEvaluationTimeMs: Double = 0
    public var currentCacheSize: Int = 0
    public var memoryUsageMB: Int = 0
}

/// Model file watcher for hot reload
actor ModelWatcher {
    private let modelId: String
    private let path: URL
    private let callback: (String, FilterModelRegistry.Version) -> Void
    private var fileMonitor: DispatchSourceFileSystemObject?
    private let queue = DispatchQueue(label: "vectorstore.modelwatcher", qos: .background)
    
    init(
        modelId: String,
        path: URL,
        callback: @escaping (String, FilterModelRegistry.Version) -> Void
    ) {
        self.modelId = modelId
        self.path = path
        self.callback = callback
    }
    
    func start() {
        let descriptor = open(path.path, O_EVTONLY)
        guard descriptor >= 0 else { return }
        
        fileMonitor = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: descriptor,
            eventMask: [.write, .rename],
            queue: queue
        )
        
        fileMonitor?.setEventHandler { [weak self] in
            guard let self = self else { return }
            Task {
                await self.handleFileChange()
            }
        }
        
        fileMonitor?.setCancelHandler {
            close(descriptor)
        }
        
        fileMonitor?.resume()
    }
    
    func stop() {
        fileMonitor?.cancel()
        fileMonitor = nil
    }
    
    private func handleFileChange() {
        // Parse version from directory name or metadata
        let version = parseVersion(from: path)
        callback(modelId, version)
    }
    
    private func parseVersion(from path: URL) -> FilterModelRegistry.Version {
        // Try to parse version from path
        let versionString = path.lastPathComponent
        let components = versionString.split(separator: ".")
        
        if components.count == 3,
           let major = Int(components[0]),
           let minor = Int(components[1]),
           let patch = Int(components[2]) {
            return FilterModelRegistry.Version(
                major: major,
                minor: minor,
                patch: patch
            )
        }
        
        // Default to 1.0.0
        return FilterModelRegistry.Version(major: 1, minor: 0, patch: 0)
    }
}

// MARK: - Errors

enum FilterModelError: Error {
    case modelNotFound(String)
    case invalidArchitecture(String)
    case checksumMismatch(expected: String, actual: String)
    case dimensionMismatch(expected: Int, actual: Int)
}
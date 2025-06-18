// VectorStoreKit: Neural Clustering Configuration
//
// Configuration for neural network-based clustering

import Foundation

/// Configuration for neural clustering in IVF index
public struct NeuralClusteringConfig: Sendable, Codable {
    // MARK: - Network Architecture
    
    /// Hidden layer sizes for cluster assignment network
    public let clusterHiddenSizes: [Int]
    
    /// Hidden layer sizes for probe prediction network
    public let probeHiddenSizes: [Int]
    
    /// Activation function for hidden layers
    public let activation: Activation
    
    /// Dropout rate for regularization
    public let dropoutRate: Float
    
    /// Whether to use batch normalization
    public let useBatchNorm: Bool
    
    // MARK: - Training Parameters
    
    /// Number of training epochs
    public let epochs: Int
    
    /// Batch size for training
    public let batchSize: Int
    
    /// Learning rate
    public let learningRate: Float
    
    /// Learning rate decay
    public let learningRateDecay: Float
    
    /// Weight decay for L2 regularization
    public let weightDecay: Float
    
    /// Gradient clipping value
    public let gradientClipValue: Float?
    
    /// Early stopping patience
    public let earlyStoppingPatience: Int
    
    // MARK: - Clustering Parameters
    
    /// Method for initializing centroids
    public let initMethod: CentroidInitMethod
    
    /// Whether to use soft clustering (probabilistic assignments)
    public let softClustering: Bool
    
    /// Temperature parameter for soft clustering
    public let temperature: Float
    
    /// Minimum cluster size (rebalance if below)
    public let minClusterSize: Int
    
    // MARK: - Adaptive Features
    
    /// Enable adaptive probe count prediction
    public let adaptiveProbing: Bool
    
    /// Enable online adaptation to query patterns
    public let onlineAdaptation: Bool
    
    /// Size of query history buffer
    public let queryHistorySize: Int
    
    /// Number of queries before adaptation
    public let adaptationInterval: Int
    
    /// Learning rate for online updates
    public let onlineLearningRate: Float
    
    // MARK: - Performance
    
    /// Use Metal acceleration
    public let useMetalAcceleration: Bool
    
    /// Cache cluster assignments
    public let cacheAssignments: Bool
    
    /// Cache size for assignments
    public let assignmentCacheSize: Int
    
    // MARK: - Initialization
    
    public init(
        clusterHiddenSizes: [Int] = [256, 128],
        probeHiddenSizes: [Int] = [64, 32],
        activation: ActivationType = .relu,
        dropoutRate: Float = 0.2,
        useBatchNorm: Bool = true,
        epochs: Int = 100,
        batchSize: Int = 32,
        learningRate: Float = 0.001,
        learningRateDecay: Float = 0.95,
        weightDecay: Float = 0.0001,
        gradientClipValue: Float? = 1.0,
        earlyStoppingPatience: Int = 10,
        initMethod: CentroidInitMethod = .kMeansPlusPlus,
        softClustering: Bool = true,
        temperature: Float = 1.0,
        minClusterSize: Int = 10,
        adaptiveProbing: Bool = true,
        onlineAdaptation: Bool = true,
        queryHistorySize: Int = 10000,
        adaptationInterval: Int = 1000,
        onlineLearningRate: Float = 0.0001,
        useMetalAcceleration: Bool = true,
        cacheAssignments: Bool = true,
        assignmentCacheSize: Int = 10000
    ) {
        self.clusterHiddenSizes = clusterHiddenSizes
        self.probeHiddenSizes = probeHiddenSizes
        self.activation = activation
        self.dropoutRate = dropoutRate
        self.useBatchNorm = useBatchNorm
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.learningRateDecay = learningRateDecay
        self.weightDecay = weightDecay
        self.gradientClipValue = gradientClipValue
        self.earlyStoppingPatience = earlyStoppingPatience
        self.initMethod = initMethod
        self.softClustering = softClustering
        self.temperature = temperature
        self.minClusterSize = minClusterSize
        self.adaptiveProbing = adaptiveProbing
        self.onlineAdaptation = onlineAdaptation
        self.queryHistorySize = queryHistorySize
        self.adaptationInterval = adaptationInterval
        self.onlineLearningRate = onlineLearningRate
        self.useMetalAcceleration = useMetalAcceleration
        self.cacheAssignments = cacheAssignments
        self.assignmentCacheSize = assignmentCacheSize
    }
    
    // MARK: - Presets
    
    /// Fast configuration with minimal features
    public static let fast = NeuralClusteringConfig(
        clusterHiddenSizes: [128],
        probeHiddenSizes: [32],
        useBatchNorm: false,
        epochs: 50,
        batchSize: 64,
        learningRate: 0.01,
        adaptiveProbing: false,
        onlineAdaptation: false
    )
    
    /// Balanced configuration
    public static let balanced = NeuralClusteringConfig()
    
    /// High-quality configuration with all features
    public static let highQuality = NeuralClusteringConfig(
        clusterHiddenSizes: [512, 256, 128],
        probeHiddenSizes: [128, 64, 32],
        epochs: 200,
        batchSize: 16,
        learningRate: 0.0005,
        learningRateDecay: 0.99,
        weightDecay: 0.001,
        earlyStoppingPatience: 20,
        temperature: 0.5,
        queryHistorySize: 50000,
        adaptationInterval: 500
    )
    
    // MARK: - Validation
    
    public func validate() throws {
        guard !clusterHiddenSizes.isEmpty else {
            throw ConfigurationError.invalidValue("Cluster hidden sizes cannot be empty")
        }
        
        guard clusterHiddenSizes.allSatisfy({ $0 > 0 }) else {
            throw ConfigurationError.invalidValue("All hidden sizes must be positive")
        }
        
        guard epochs > 0 else {
            throw ConfigurationError.invalidValue("Epochs must be positive")
        }
        
        guard batchSize > 0 else {
            throw ConfigurationError.invalidValue("Batch size must be positive")
        }
        
        guard learningRate > 0 && learningRate <= 1 else {
            throw ConfigurationError.invalidValue("Learning rate must be in (0, 1]")
        }
        
        guard temperature > 0 else {
            throw ConfigurationError.invalidValue("Temperature must be positive")
        }
        
        guard dropoutRate >= 0 && dropoutRate < 1 else {
            throw ConfigurationError.invalidValue("Dropout rate must be in [0, 1)")
        }
    }
}

// MARK: - Supporting Types

/// Activation function types
// ActivationType is defined in Core/ML/Activations.swift

/// Centroid initialization methods
public enum CentroidInitMethod: String, Codable, Sendable {
    case random = "random"
    case kMeansPlusPlus = "kmeans++"
    case neuralInit = "neural"
    case hierarchical = "hierarchical"
}

// MARK: - Extensions

extension NeuralClusteringConfig {
    /// Estimated memory usage for the neural networks
    public func estimatedMemoryUsage(dimensions: Int, clusters: Int) -> Int {
        // Cluster network parameters
        var clusterParams = 0
        var prevSize = dimensions
        
        for hiddenSize in clusterHiddenSizes {
            clusterParams += prevSize * hiddenSize + hiddenSize // Weights + bias
            prevSize = hiddenSize
        }
        clusterParams += prevSize * clusters + clusters // Output layer
        
        // Probe network parameters (if enabled)
        var probeParams = 0
        if adaptiveProbing {
            prevSize = dimensions + 1 // +1 for target recall
            for hiddenSize in probeHiddenSizes {
                probeParams += prevSize * hiddenSize + hiddenSize
                prevSize = hiddenSize
            }
            probeParams += prevSize + 1 // Single output
        }
        
        // Total memory (4 bytes per float parameter)
        let totalParams = clusterParams + probeParams
        var memory = totalParams * MemoryLayout<Float>.size
        
        // Add overhead for gradients during training
        memory *= 2
        
        // Add cache memory if enabled
        if cacheAssignments {
            memory += assignmentCacheSize * clusters * MemoryLayout<Float>.size
        }
        
        return memory
    }
    
    /// Create a configuration suitable for the given data characteristics
    public static func autoConfiguration(
        dimensions: Int,
        dataSize: Int,
        clusters: Int
    ) -> NeuralClusteringConfig {
        // Adjust hidden sizes based on dimensions
        let clusterHiddenSizes: [Int]
        if dimensions < 128 {
            clusterHiddenSizes = [dimensions * 2, dimensions]
        } else if dimensions < 512 {
            clusterHiddenSizes = [256, 128]
        } else {
            clusterHiddenSizes = [512, 256, 128]
        }
        
        // Adjust batch size based on data size
        let batchSize = min(max(32, dataSize / 1000), 256)
        
        // Adjust epochs based on data complexity
        let epochs = min(max(50, dataSize / 10000), 200)
        
        return NeuralClusteringConfig(
            clusterHiddenSizes: clusterHiddenSizes,
            epochs: epochs,
            batchSize: batchSize,
            adaptiveProbing: dataSize > 10000,
            onlineAdaptation: dataSize > 50000,
            queryHistorySize: min(dataSize / 10, 50000)
        )
    }
}
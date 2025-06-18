// VectorStoreKit: Vector ML Pipeline
//
// High-level ML pipeline for vector database operations
//

import Foundation
import simd
@preconcurrency import Metal

/// High-level ML pipeline for vector operations
public actor VectorMLPipeline {
    // MARK: - Properties
    
    private let metalPipeline: MetalMLPipeline
    private let device: MTLDevice
    
    // Component actors
    private let encoder: VectorEncoder
    private let dimensionReducer: DimensionReduction
    private let similarityLearner: SimilarityLearning
    private var clusterOptimizer: NeuralClustering?
    
    // Performance tracking
    private var performanceMetrics = VectorMLMetrics()
    
    // MARK: - Initialization
    
    public init(device: MTLDevice? = nil) async throws {
        // Initialize Metal device
        guard let metalDevice = device ?? MTLCreateSystemDefaultDevice() else {
            throw VectorMLError.metalNotAvailable
        }
        self.device = metalDevice
        
        // Initialize Metal ML pipeline
        self.metalPipeline = try MetalMLPipeline(device: metalDevice)
        
        // Initialize components
        self.encoder = try await VectorEncoder(metalPipeline: metalPipeline)
        self.dimensionReducer = try await DimensionReduction(metalPipeline: metalPipeline)
        self.similarityLearner = try await SimilarityLearning(metalPipeline: metalPipeline)
    }
    
    // MARK: - Vector Encoding
    
    /// Encode vectors using a trained autoencoder
    public func encodeVectors(_ vectors: [Vector]) async throws -> [Vector] {
        let result = try await encoder.encode(vectors)
        performanceMetrics.vectorsEncoded += vectors.count
        return result
    }
    
    /// Train an autoencoder for vector compression
    public func trainEncoder(
        on vectors: [Vector],
        targetDimensions: Int,
        config: VectorEncoderConfig = VectorEncoderConfig()
    ) async throws {
        var modifiedConfig = config
        if modifiedConfig.encodedDimension == nil {
            modifiedConfig = VectorEncoderConfig(
                encoderHiddenSizes: config.encoderHiddenSizes,
                encodedDimension: targetDimensions,
                epochs: config.epochs,
                batchSize: config.batchSize,
                learningRate: config.learningRate,
                useDropout: config.useDropout,
                dropoutRate: config.dropoutRate,
                useBatchNorm: config.useBatchNorm,
                bottleneckActivation: config.bottleneckActivation,
                convergenceThreshold: config.convergenceThreshold
            )
        }
        
        try await encoder.train(on: vectors, config: modifiedConfig)
        performanceMetrics.encoderTrainingTime = Date().timeIntervalSince1970
    }
    
    // MARK: - Similarity Learning
    
    /// Learn a similarity function from positive and negative pairs
    public func learnSimilarity(
        positives: [(Vector, Vector)],
        negatives: [(Vector, Vector)],
        config: SimilarityLearningConfig = SimilarityLearningConfig()
    ) async throws -> SimilarityModel {
        let contrastiveConfig = ContrastiveLearningConfig(
            hiddenSize: config.hiddenSize,
            embeddingDimension: config.hiddenSize / 2,
            epochs: config.epochs,
            batchSize: config.batchSize,
            learningRate: config.learningRate,
            margin: config.margin
        )
        
        let model = try await similarityLearner.learnFromPairs(
            positives: positives,
            negatives: negatives,
            config: contrastiveConfig
        )
        
        performanceMetrics.similarityComputations += positives.count + negatives.count
        return model
    }
    
    /// Compute similarity using a learned model
    public func computeSimilarity(
        _ vector1: Vector,
        _ vector2: Vector,
        using model: SimilarityModel
    ) async throws -> Float {
        let similarity = try await similarityLearner.computeSimilarity(
            vector1,
            vector2,
            using: model
        )
        performanceMetrics.similarityComputations += 1
        return similarity
    }
    
    // MARK: - Clustering Optimization
    
    /// Optimize cluster assignments using neural approach
    public func optimizeClusters(
        _ vectors: [Vector],
        clusterCount: Int,
        config: ClusterOptimizationConfig = ClusterOptimizationConfig()
    ) async throws -> ClusterAssignments {
        let dimensions = vectors.first?.count ?? 0
        guard dimensions > 0 else {
            throw VectorMLError.invalidInput("Empty vectors provided")
        }
        
        // Initialize neural clustering
        let clusteringConfig = NeuralClusteringConfiguration(
            dimensions: dimensions,
            numberOfClusters: clusterCount,
            trainingEpochs: config.epochs,
            batchSize: config.batchSize,
            learningRate: config.learningRate,
            adaptiveProbing: config.adaptiveProbing
        )
        
        self.clusterOptimizer = try await NeuralClustering(configuration: clusteringConfig)
        
        guard let optimizer = clusterOptimizer else {
            throw VectorMLError.clusterOptimizerNotInitialized
        }
        
        // Convert vectors to Float arrays for NeuralClustering
        let floatVectors = vectors.map { Array($0) }
        
        // Train clustering
        let result = try await optimizer.train(vectors: floatVectors)
        
        // Create assignments
        var assignments = ClusterAssignments(clusterCount: clusterCount)
        
        // Assign vectors to clusters
        for (idx, vector) in floatVectors.enumerated() {
            let probabilities = try await optimizer.getClusterProbabilities(for: vector)
            let clusterIdx = probabilities.enumerated().max(by: { $0.1 < $1.1 })?.0 ?? 0
            assignments.assign(vectorIndex: idx, to: clusterIdx)
        }
        
        // Store centroids
        assignments.centroids = result.centroids.map { Vector($0) }
        
        performanceMetrics.clustersOptimized += 1
        return assignments
    }
    
    /// Predict optimal cluster probes for a query
    public func predictOptimalProbes(
        for query: Vector,
        targetRecall: Float = 0.95
    ) async throws -> Int {
        guard let optimizer = clusterOptimizer else {
            throw VectorMLError.clusterOptimizerNotInitialized
        }
        
        let floatQuery = Array(query)
        return try await optimizer.predictProbeCount(for: floatQuery, targetRecall: targetRecall)
    }
    
    // MARK: - Dimension Reduction
    
    /// Reduce dimensions while preserving similarity structure
    public func reduceDimensions(
        _ vectors: [Vector],
        targetDim: Int,
        method: DimensionReductionMethod = .pca
    ) async throws -> [Vector] {
        guard !vectors.isEmpty else {
            throw VectorMLError.invalidInput("Empty vector array")
        }
        
        let originalDim = vectors.first?.count ?? 0
        guard targetDim < originalDim else {
            throw VectorMLError.invalidInput("Target dimension must be less than original")
        }
        
        let result: [Vector]
        
        switch method {
        case .pca:
            let pcaResult = try await dimensionReducer.pca(
                vectors,
                targetDim: targetDim,
                whiten: false
            )
            result = pcaResult.transformedVectors
            
        case .autoencoder:
            // Train autoencoder if not already trained
            let config = VectorEncoderConfig(
                encoderHiddenSizes: [originalDim / 2],
                encodedDimension: targetDim,
                epochs: 50
            )
            try await encoder.train(on: vectors, config: config)
            result = try await encoder.encode(vectors)
            
        case .neuralPCA:
            // Use autoencoder with linear activation for neural PCA
            try await encoder.createDenoisingEncoder(
                inputDim: originalDim,
                encodedDim: targetDim,
                noiseLevel: 0.0 // No noise for neural PCA
            )
            let config = VectorEncoderConfig(
                encoderHiddenSizes: [],
                encodedDimension: targetDim,
                epochs: 100,
                bottleneckActivation: .linear
            )
            try await encoder.train(on: vectors, config: config)
            result = try await encoder.encode(vectors)
            
        case .randomProjection:
            let projectionResult = try await dimensionReducer.randomProjection(
                vectors,
                targetDim: targetDim
            )
            result = projectionResult.transformedVectors
        }
        
        performanceMetrics.dimensionReductions += 1
        return result
    }
    
    // MARK: - Advanced Features
    
    /// Create specialized encoder variants
    public func createSparseEncoder(
        inputDim: Int,
        encodedDim: Int,
        sparsityWeight: Float = 0.01
    ) async throws {
        try await encoder.createSparseEncoder(
            inputDim: inputDim,
            encodedDim: encodedDim,
            sparsityWeight: sparsityWeight
        )
    }
    
    public func createDenoisingEncoder(
        inputDim: Int,
        encodedDim: Int,
        noiseLevel: Float = 0.1
    ) async throws {
        try await encoder.createDenoisingEncoder(
            inputDim: inputDim,
            encodedDim: encodedDim,
            noiseLevel: noiseLevel
        )
    }
    
    /// Learn from triplets for better similarity
    public func learnFromTriplets(
        triplets: [(anchor: Vector, positive: Vector, negative: Vector)],
        config: TripletLearningConfig = TripletLearningConfig()
    ) async throws -> SimilarityModel {
        let model = try await similarityLearner.learnFromTriplets(
            triplets: triplets,
            config: config
        )
        return model
    }
    
    // MARK: - Performance Metrics
    
    /// Get current performance metrics
    public func getMetrics() async -> VectorMLMetrics {
        var metrics = performanceMetrics
        
        // Add component metrics
        let encoderMetrics = await encoder.getMetrics()
        metrics.vectorsEncoded = encoderMetrics.vectorsEncoded
        
        return metrics
    }
    
    /// Reset performance metrics
    public func resetMetrics() async {
        performanceMetrics = VectorMLMetrics()
        await encoder.resetMetrics()
    }
    
    // MARK: - PCA Implementation
    
    /// Compute top eigenvectors using Metal-accelerated SVD
    /// This replaces any placeholder identity matrix implementations
    public func computeTopEigenvectors(
        from vectors: [Vector],
        numComponents: Int
    ) async throws -> (eigenvectors: [[Float]], eigenvalues: [Float]) {
        guard !vectors.isEmpty else {
            throw VectorMLError.invalidInput("Empty vector array")
        }
        
        let dimensions = vectors.first?.count ?? 0
        guard numComponents <= dimensions else {
            throw VectorMLError.invalidInput("Number of components must be less than or equal to dimensions")
        }
        
        // Use the dimension reducer's PCA implementation which properly computes eigenvectors via SVD
        let pcaResult = try await dimensionReducer.pca(
            vectors,
            targetDim: numComponents,
            whiten: false
        )
        
        // Convert the flattened components matrix to 2D array
        let componentsFlat = pcaResult.components
        var eigenvectors: [[Float]] = []
        
        // Each eigenvector has 'dimensions' elements
        for i in 0..<numComponents {
            let startIdx = i * dimensions
            let endIdx = (i + 1) * dimensions
            let eigenvector = Array(componentsFlat[startIdx..<endIdx])
            eigenvectors.append(eigenvector)
        }
        
        // Extract eigenvalues from explained variance (approximate)
        // Note: The actual eigenvalues would need to be exposed from DimensionReduction
        // For now, we'll use the explained variance ratios as a proxy
        let eigenvalues = pcaResult.explainedVarianceRatio
        
        return (eigenvectors: eigenvectors, eigenvalues: eigenvalues)
    }
}

// MARK: - Supporting Types

/// Vector type alias for clarity
public typealias Vector = [Float]

/// Configuration for similarity learning
public struct SimilarityLearningConfig: Sendable {
    public let hiddenSize: Int
    public let epochs: Int
    public let batchSize: Int
    public let learningRate: Float
    public let margin: Float
    
    public init(
        hiddenSize: Int = 256,
        epochs: Int = 50,
        batchSize: Int = 32,
        learningRate: Float = 0.001,
        margin: Float = 1.0
    ) {
        self.hiddenSize = hiddenSize
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.margin = margin
    }
}

/// Configuration for cluster optimization
public struct ClusterOptimizationConfig: Sendable {
    public let epochs: Int
    public let batchSize: Int
    public let learningRate: Float
    public let adaptiveProbing: Bool
    
    public init(
        epochs: Int = 100,
        batchSize: Int = 32,
        learningRate: Float = 0.001,
        adaptiveProbing: Bool = true
    ) {
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.adaptiveProbing = adaptiveProbing
    }
}

/// Dimension reduction methods
public enum DimensionReductionMethod: Sendable {
    case pca
    case autoencoder
    case neuralPCA
    case randomProjection
}

/// Cluster assignments result
public struct ClusterAssignments: Sendable {
    public let clusterCount: Int
    public var assignments: [Int: Int] = [:] // vectorIndex -> clusterIndex
    public var centroids: [Vector] = []
    
    public init(clusterCount: Int) {
        self.clusterCount = clusterCount
    }
    
    public mutating func assign(vectorIndex: Int, to cluster: Int) {
        assignments[vectorIndex] = cluster
    }
    
    public func getCluster(for vectorIndex: Int) -> Int? {
        assignments[vectorIndex]
    }
}

/// Performance metrics for ML operations
public struct VectorMLMetrics: Sendable {
    public var vectorsEncoded: Int = 0
    public var encoderTrainingTime: TimeInterval = 0
    public var similarityComputations: Int = 0
    public var clustersOptimized: Int = 0
    public var dimensionReductions: Int = 0
    
    public var averageEncodingThroughput: Double {
        guard encoderTrainingTime > 0 else { return 0 }
        return Double(vectorsEncoded) / encoderTrainingTime
    }
}

/// Vector ML pipeline errors
public enum VectorMLError: LocalizedError {
    case metalNotAvailable
    case encoderNotInitialized
    case similarityNetworkNotInitialized
    case clusterOptimizerNotInitialized
    case invalidInput(String)
    case trainingFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .metalNotAvailable:
            return "Metal is not available on this device"
        case .encoderNotInitialized:
            return "Encoder network not initialized"
        case .similarityNetworkNotInitialized:
            return "Similarity network not initialized"
        case .clusterOptimizerNotInitialized:
            return "Cluster optimizer not initialized"
        case .invalidInput(let message):
            return "Invalid input: \(message)"
        case .trainingFailed(let message):
            return "Training failed: \(message)"
        }
    }
}
// VectorStoreKit: Similarity Learning
//
// Learn custom similarity functions for better vector matching
//

import Foundation
@preconcurrency import Metal

/// Learn similarity metrics from data
public actor SimilarityLearning {
    private let metalPipeline: MetalMLPipeline
    private let device: MTLDevice
    private var similarityNetwork: NeuralNetwork?
    private var tripletNetwork: NeuralNetwork?
    
    // Cached models
    private var trainedModels: [String: SimilarityModel] = [:]
    
    public init(metalPipeline: MetalMLPipeline) async throws {
        self.metalPipeline = metalPipeline
        self.device = await metalPipeline.device
    }
    
    // MARK: - Contrastive Learning
    
    /// Learn similarity from positive and negative pairs
    public func learnFromPairs(
        positives: [(Vector, Vector)],
        negatives: [(Vector, Vector)],
        config: ContrastiveLearningConfig = ContrastiveLearningConfig()
    ) async throws -> SimilarityModel {
        guard !positives.isEmpty && !negatives.isEmpty else {
            throw SimilarityLearningError.emptyTrainingData
        }
        
        let inputDim = positives.first?.0.count ?? 0
        
        // Initialize network
        try await initializeContrastiveNetwork(inputDim: inputDim, config: config)
        
        guard let network = similarityNetwork else {
            throw SimilarityLearningError.networkNotInitialized
        }
        
        // Prepare training data
        let trainingData = try await prepareContrastiveData(
            positives: positives,
            negatives: negatives
        )
        
        // Train with contrastive loss
        let startTime = Date()
        var bestLoss: Float = Float.greatestFiniteMagnitude
        
        for epoch in 0..<config.epochs {
            var epochLoss: Float = 0
            let shuffled = trainingData.shuffled()
            
            for batchStart in stride(from: 0, to: shuffled.count, by: config.batchSize) {
                let batchEnd = min(batchStart + config.batchSize, shuffled.count)
                let batch = Array(shuffled[batchStart..<batchEnd])
                
                let loss = try await trainContrastiveBatch(
                    batch,
                    network: network,
                    config: config
                )
                epochLoss += loss
            }
            
            epochLoss /= Float(shuffled.count / config.batchSize)
            
            // Track best model
            if epochLoss < bestLoss {
                bestLoss = epochLoss
            }
            
            // Log progress
            if epoch % 10 == 0 {
                print("Contrastive Learning - Epoch \(epoch): Loss = \(epochLoss)")
            }
        }
        
        let model = SimilarityModel(
            network: network,
            inputDimensions: inputDim,
            modelType: .contrastive,
            trainingTime: Date().timeIntervalSince(startTime),
            finalLoss: bestLoss
        )
        
        // Cache model
        trainedModels["contrastive_\(inputDim)"] = model
        
        return model
    }
    
    // MARK: - Triplet Learning
    
    /// Learn similarity from triplets (anchor, positive, negative)
    public func learnFromTriplets(
        triplets: [(anchor: Vector, positive: Vector, negative: Vector)],
        config: TripletLearningConfig = TripletLearningConfig()
    ) async throws -> SimilarityModel {
        guard !triplets.isEmpty else {
            throw SimilarityLearningError.emptyTrainingData
        }
        
        let inputDim = triplets.first?.anchor.count ?? 0
        
        // Initialize triplet network
        try await initializeTripletNetwork(inputDim: inputDim, config: config)
        
        guard let network = tripletNetwork else {
            throw SimilarityLearningError.networkNotInitialized
        }
        
        // Prepare training data
        let trainingData = try await prepareTripletData(triplets)
        
        // Train with triplet loss
        let startTime = Date()
        var bestLoss: Float = Float.greatestFiniteMagnitude
        
        for epoch in 0..<config.epochs {
            var epochLoss: Float = 0
            let shuffled = trainingData.shuffled()
            
            for batchStart in stride(from: 0, to: shuffled.count, by: config.batchSize) {
                let batchEnd = min(batchStart + config.batchSize, shuffled.count)
                let batch = Array(shuffled[batchStart..<batchEnd])
                
                let loss = try await trainTripletBatch(
                    batch,
                    network: network,
                    config: config
                )
                epochLoss += loss
            }
            
            epochLoss /= Float(shuffled.count / config.batchSize)
            
            if epochLoss < bestLoss {
                bestLoss = epochLoss
            }
            
            if epoch % 10 == 0 {
                print("Triplet Learning - Epoch \(epoch): Loss = \(epochLoss)")
            }
        }
        
        let model = SimilarityModel(
            network: network,
            inputDimensions: inputDim,
            modelType: .triplet,
            trainingTime: Date().timeIntervalSince(startTime),
            finalLoss: bestLoss
        )
        
        trainedModels["triplet_\(inputDim)"] = model
        
        return model
    }
    
    // MARK: - Similarity Computation
    
    /// Compute learned similarity between two vectors
    public func computeSimilarity(
        _ vector1: Vector,
        _ vector2: Vector,
        using model: SimilarityModel
    ) async throws -> Float {
        switch model.modelType {
        case .contrastive:
            return try await computeContrastiveSimilarity(
                vector1,
                vector2,
                model: model
            )
        case .triplet:
            return try await computeTripletSimilarity(
                vector1,
                vector2,
                model: model
            )
        case .cosine:
            return computeCosineSimilarity(vector1, vector2)
        }
    }
    
    /// Batch similarity computation
    public func computeBatchSimilarity(
        queries: [Vector],
        candidates: [Vector],
        using model: SimilarityModel
    ) async throws -> [[Float]] {
        var similarities: [[Float]] = []
        
        for query in queries {
            var querySimilarities: [Float] = []
            
            // Process candidates in batches for efficiency
            let batchSize = 1000
            for batchStart in stride(from: 0, to: candidates.count, by: batchSize) {
                let batchEnd = min(batchStart + batchSize, candidates.count)
                let batch = Array(candidates[batchStart..<batchEnd])
                
                let batchSims = try await computeBatchSimilarityForQuery(
                    query: query,
                    candidates: batch,
                    model: model
                )
                querySimilarities.append(contentsOf: batchSims)
            }
            
            similarities.append(querySimilarities)
        }
        
        return similarities
    }
    
    // MARK: - Model Management
    
    /// Get a cached model
    public func getCachedModel(key: String) -> SimilarityModel? {
        trainedModels[key]
    }
    
    /// Clear cached models
    public func clearCache() {
        trainedModels.removeAll()
    }
    
    // MARK: - Private Methods
    
    private func initializeContrastiveNetwork(
        inputDim: Int,
        config: ContrastiveLearningConfig
    ) async throws {
        self.similarityNetwork = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        guard let network = similarityNetwork else { return }
        
        // Siamese architecture - shared weights for both inputs
        let embeddingDim = config.embeddingDimension
        
        // Embedding network
        await network.addLayer(try await DenseLayer(
            inputSize: inputDim * 2, // Concatenated input
            outputSize: config.hiddenSize,
            activation: .relu,
            metalPipeline: metalPipeline
        ))
        
        await network.addLayer(try await BatchNormLayer(
            numFeatures: config.hiddenSize,
            metalPipeline: metalPipeline
        ))
        
        await network.addLayer(try await DenseLayer(
            inputSize: config.hiddenSize,
            outputSize: embeddingDim,
            activation: .relu,
            metalPipeline: metalPipeline
        ))
        
        // Similarity head
        await network.addLayer(try await DenseLayer(
            inputSize: embeddingDim,
            outputSize: embeddingDim / 2,
            activation: .relu,
            metalPipeline: metalPipeline
        ))
        
        await network.addLayer(try await DenseLayer(
            inputSize: embeddingDim / 2,
            outputSize: 1,
            activation: .sigmoid,
            metalPipeline: metalPipeline
        ))
    }
    
    private func initializeTripletNetwork(
        inputDim: Int,
        config: TripletLearningConfig
    ) async throws {
        self.tripletNetwork = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        guard let network = tripletNetwork else { return }
        
        // Embedding network for triplet loss
        await network.addLayer(try await DenseLayer(
            inputSize: inputDim,
            outputSize: config.hiddenSize,
            activation: .relu,
            metalPipeline: metalPipeline
        ))
        
        await network.addLayer(try await BatchNormLayer(
            numFeatures: config.hiddenSize,
            metalPipeline: metalPipeline
        ))
        
        await network.addLayer(try await DropoutLayer(
            rate: 0.2,
            metalPipeline: metalPipeline
        ))
        
        await network.addLayer(try await DenseLayer(
            inputSize: config.hiddenSize,
            outputSize: config.embeddingDimension,
            activation: .linear, // L2 normalized later
            metalPipeline: metalPipeline
        ))
    }
    
    private func prepareContrastiveData(
        positives: [(Vector, Vector)],
        negatives: [(Vector, Vector)]
    ) async throws -> [(input: MetalBuffer, label: Float)] {
        var trainingData: [(input: MetalBuffer, label: Float)] = []
        
        // Positive pairs (similar = 1)
        for (vec1, vec2) in positives {
            let combined = vec1 + vec2
            let buffer = try await vectorToMetalBuffer(combined)
            trainingData.append((input: buffer, label: 1.0))
        }
        
        // Negative pairs (dissimilar = 0)
        for (vec1, vec2) in negatives {
            let combined = vec1 + vec2
            let buffer = try await vectorToMetalBuffer(combined)
            trainingData.append((input: buffer, label: 0.0))
        }
        
        return trainingData
    }
    
    private func prepareTripletData(
        _ triplets: [(anchor: Vector, positive: Vector, negative: Vector)]
    ) async throws -> [(anchor: MetalBuffer, positive: MetalBuffer, negative: MetalBuffer)] {
        var trainingData: [(anchor: MetalBuffer, positive: MetalBuffer, negative: MetalBuffer)] = []
        
        for triplet in triplets {
            let anchorBuffer = try await vectorToMetalBuffer(triplet.anchor)
            let positiveBuffer = try await vectorToMetalBuffer(triplet.positive)
            let negativeBuffer = try await vectorToMetalBuffer(triplet.negative)
            
            trainingData.append((
                anchor: anchorBuffer,
                positive: positiveBuffer,
                negative: negativeBuffer
            ))
        }
        
        return trainingData
    }
    
    private func trainContrastiveBatch(
        _ batch: [(input: MetalBuffer, label: Float)],
        network: NeuralNetwork,
        config: ContrastiveLearningConfig
    ) async throws -> Float {
        var totalLoss: Float = 0
        
        for (input, label) in batch {
            // Forward pass
            let output = try await network.forward(input)
            
            // Compute contrastive loss
            let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: 1)
            let prediction = outputPtr[0]
            
            let loss: Float
            if label > 0.5 {
                // Positive pair - minimize distance
                loss = pow(1 - prediction, 2)
            } else {
                // Negative pair - maximize distance up to margin
                loss = pow(max(0, config.margin - prediction), 2)
            }
            
            totalLoss += loss
            
            // Compute gradient
            let gradient = try await metalPipeline.allocateBuffer(size: 1)
            let gradPtr = gradient.buffer.contents().bindMemory(to: Float.self, capacity: 1)
            
            if label > 0.5 {
                gradPtr[0] = -2 * (1 - prediction)
            } else {
                gradPtr[0] = prediction < config.margin ? 2 * (prediction - config.margin) : 0
            }
            
            // Backward pass
            let gradArray = Array(UnsafeBufferPointer(
                start: gradient.buffer.contents().bindMemory(to: Float.self, capacity: 1),
                count: 1
            ))
            _ = await network.backward(gradArray, learningRate: config.learningRate)
            
            // TODO: Update parameters with optimizer
            
            await metalPipeline.releaseBuffer(output)
            await metalPipeline.releaseBuffer(gradient)
        }
        
        return totalLoss / Float(batch.count)
    }
    
    private func trainTripletBatch(
        _ batch: [(anchor: MetalBuffer, positive: MetalBuffer, negative: MetalBuffer)],
        network: NeuralNetwork,
        config: TripletLearningConfig
    ) async throws -> Float {
        var totalLoss: Float = 0
        
        for (anchor, positive, negative) in batch {
            // Forward pass for all three
            let anchorEmbed = try await network.forward(anchor)
            let positiveEmbed = try await network.forward(positive)
            let negativeEmbed = try await network.forward(negative)
            
            // Compute distances
            let posDistance = try await computeEuclideanDistance(anchorEmbed, positiveEmbed)
            let negDistance = try await computeEuclideanDistance(anchorEmbed, negativeEmbed)
            
            // Triplet loss with margin
            let loss = max(0, posDistance - negDistance + config.margin)
            totalLoss += loss
            
            if loss > 0 {
                // Compute gradients
                // TODO: Implement proper triplet loss gradients
            }
            
            await metalPipeline.releaseBuffer(anchorEmbed)
            await metalPipeline.releaseBuffer(positiveEmbed)
            await metalPipeline.releaseBuffer(negativeEmbed)
        }
        
        return totalLoss / Float(batch.count)
    }
    
    private func computeContrastiveSimilarity(
        _ vector1: Vector,
        _ vector2: Vector,
        model: SimilarityModel
    ) async throws -> Float {
        let combined = vector1 + vector2
        let input = try await vectorToMetalBuffer(combined)
        
        let output = try await model.network.predict(input)
        let outputPtr = output.buffer.contents().bindMemory(to: Float.self, capacity: 1)
        let similarity = outputPtr[0]
        
        await metalPipeline.releaseBuffer(input)
        await metalPipeline.releaseBuffer(output)
        
        return similarity
    }
    
    private func computeTripletSimilarity(
        _ vector1: Vector,
        _ vector2: Vector,
        model: SimilarityModel
    ) async throws -> Float {
        // Get embeddings
        let embed1 = try await getEmbedding(vector1, model: model)
        let embed2 = try await getEmbedding(vector2, model: model)
        
        // Compute cosine similarity of embeddings
        let similarity = computeCosineSimilarity(embed1, embed2)
        
        return similarity
    }
    
    private func getEmbedding(_ vector: Vector, model: SimilarityModel) async throws -> Vector {
        let input = try await vectorToMetalBuffer(vector)
        let embedding = try await model.network.forward(input)
        
        let embeddingPtr = embedding.buffer.contents().bindMemory(
            to: Float.self,
            capacity: embedding.count
        )
        let result = Vector(Array(UnsafeBufferPointer(
            start: embeddingPtr,
            count: embedding.count
        )))
        
        await metalPipeline.releaseBuffer(input)
        await metalPipeline.releaseBuffer(embedding)
        
        return result
    }
    
    private func computeBatchSimilarityForQuery(
        query: Vector,
        candidates: [Vector],
        model: SimilarityModel
    ) async throws -> [Float] {
        var similarities: [Float] = []
        
        for candidate in candidates {
            let sim = try await computeSimilarity(query, candidate, using: model)
            similarities.append(sim)
        }
        
        return similarities
    }
    
    private func computeEuclideanDistance(
        _ buffer1: MetalBuffer,
        _ buffer2: MetalBuffer
    ) async throws -> Float {
        guard buffer1.count == buffer2.count else {
            throw SimilarityLearningError.dimensionMismatch
        }
        
        let ptr1 = buffer1.buffer.contents().bindMemory(to: Float.self, capacity: buffer1.count)
        let ptr2 = buffer2.buffer.contents().bindMemory(to: Float.self, capacity: buffer2.count)
        
        var distance: Float = 0
        for i in 0..<buffer1.count {
            let diff = ptr1[i] - ptr2[i]
            distance += diff * diff
        }
        
        return sqrt(distance)
    }
    
    private func computeCosineSimilarity(_ vector1: Vector, _ vector2: Vector) -> Float {
        guard vector1.count == vector2.count else { return 0 }
        
        var dotProduct: Float = 0
        var norm1: Float = 0
        var norm2: Float = 0
        
        for i in 0..<vector1.count {
            dotProduct += vector1[i] * vector2[i]
            norm1 += vector1[i] * vector1[i]
            norm2 += vector2[i] * vector2[i]
        }
        
        let denominator = sqrt(norm1) * sqrt(norm2)
        return denominator > 0 ? dotProduct / denominator : 0
    }
    
    private func vectorToMetalBuffer(_ vector: Vector) async throws -> MetalBuffer {
        let buffer = try await metalPipeline.allocateBuffer(size: vector.count)
        buffer.buffer.contents().copyMemory(
            from: vector,
            byteCount: vector.count * MemoryLayout<Float>.stride
        )
        return buffer
    }
}

// MARK: - Supporting Types

/// Similarity model wrapper
public struct SimilarityModel: Sendable {
    public let network: NeuralNetwork
    public let inputDimensions: Int
    public let modelType: SimilarityModelType
    public let trainingTime: TimeInterval
    public let finalLoss: Float
}

/// Types of similarity models
public enum SimilarityModelType: Sendable {
    case contrastive
    case triplet
    case cosine
}

/// Configuration for contrastive learning
public struct ContrastiveLearningConfig: Sendable {
    public let hiddenSize: Int
    public let embeddingDimension: Int
    public let epochs: Int
    public let batchSize: Int
    public let learningRate: Float
    public let margin: Float
    
    public init(
        hiddenSize: Int = 256,
        embeddingDimension: Int = 128,
        epochs: Int = 50,
        batchSize: Int = 32,
        learningRate: Float = 0.001,
        margin: Float = 1.0
    ) {
        self.hiddenSize = hiddenSize
        self.embeddingDimension = embeddingDimension
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.margin = margin
    }
}

/// Configuration for triplet learning
public struct TripletLearningConfig: Sendable {
    public let hiddenSize: Int
    public let embeddingDimension: Int
    public let epochs: Int
    public let batchSize: Int
    public let learningRate: Float
    public let margin: Float
    
    public init(
        hiddenSize: Int = 256,
        embeddingDimension: Int = 128,
        epochs: Int = 100,
        batchSize: Int = 32,
        learningRate: Float = 0.001,
        margin: Float = 0.5
    ) {
        self.hiddenSize = hiddenSize
        self.embeddingDimension = embeddingDimension
        self.epochs = epochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.margin = margin
    }
}

// MARK: - Errors

public enum SimilarityLearningError: LocalizedError {
    case emptyTrainingData
    case networkNotInitialized
    case dimensionMismatch
    case invalidModelType
    
    public var errorDescription: String? {
        switch self {
        case .emptyTrainingData:
            return "Training data is empty"
        case .networkNotInitialized:
            return "Neural network not initialized"
        case .dimensionMismatch:
            return "Vector dimensions do not match"
        case .invalidModelType:
            return "Invalid similarity model type"
        }
    }
}
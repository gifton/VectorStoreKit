// VectorStoreKit: Learned Index Configuration
//
// Configuration for neural network-based learned index

import Foundation

/// Configuration for Learned Index
public struct LearnedIndexConfiguration: IndexConfiguration {
    public let dimensions: Int
    public let modelArchitecture: ModelArchitecture
    public let trainingConfig: TrainingConfiguration
    public let bucketSize: Int
    public let useMetalAcceleration: Bool
    
    public enum ModelArchitecture: Sendable, Codable {
        case linear
        case mlp(hiddenSizes: [Int])
        case residual(layers: Int, hiddenSize: Int)
        
        var layerCount: Int {
            switch self {
            case .linear:
                return 1
            case .mlp(let hiddenSizes):
                return hiddenSizes.count + 1
            case .residual(let layers, _):
                return layers
            }
        }
    }
    
    public struct TrainingConfiguration: Sendable, Codable {
        public let epochs: Int
        public let batchSize: Int
        public let learningRate: Float
        public let regularization: Float
        public let validationSplit: Float
        
        public init(
            epochs: Int = 100,
            batchSize: Int = 32,
            learningRate: Float = 0.001,
            regularization: Float = 0.0001,
            validationSplit: Float = 0.1
        ) {
            self.epochs = epochs
            self.batchSize = batchSize
            self.learningRate = learningRate
            self.regularization = regularization
            self.validationSplit = validationSplit
        }
    }
    
    public init(
        dimensions: Int,
        modelArchitecture: ModelArchitecture = .mlp(hiddenSizes: [128, 64]),
        trainingConfig: TrainingConfiguration = TrainingConfiguration(),
        bucketSize: Int = 100,
        useMetalAcceleration: Bool = true
    ) {
        self.dimensions = dimensions
        self.modelArchitecture = modelArchitecture
        self.trainingConfig = trainingConfig
        self.bucketSize = bucketSize
        self.useMetalAcceleration = useMetalAcceleration
    }
    
    public func validate() throws {
        guard dimensions > 0 else {
            throw LearnedIndexError.invalidDimensions(dimensions)
        }
        guard bucketSize > 0 else {
            throw LearnedIndexError.invalidParameter("bucketSize", bucketSize)
        }
    }
    
    public func estimatedMemoryUsage(for vectorCount: Int) -> Int {
        let modelSize = estimateModelSize()
        let bucketCount = (vectorCount + bucketSize - 1) / bucketSize
        let bucketMemory = bucketCount * bucketSize * dimensions * MemoryLayout<Float>.size
        let indexMemory = vectorCount * MemoryLayout<String>.size
        return modelSize + bucketMemory + indexMemory
    }
    
    private func estimateModelSize() -> Int {
        switch modelArchitecture {
        case .linear:
            return dimensions * MemoryLayout<Float>.size * 2
        case .mlp(let hiddenSizes):
            var size = 0
            var prevSize = dimensions
            for hiddenSize in hiddenSizes {
                size += prevSize * hiddenSize * MemoryLayout<Float>.size
                prevSize = hiddenSize
            }
            return size
        case .residual(let layers, let hiddenSize):
            return layers * hiddenSize * hiddenSize * MemoryLayout<Float>.size
        }
    }
    
    public func computationalComplexity() -> ComputationalComplexity {
        return .logarithmic // O(log n) with learned model
    }
}
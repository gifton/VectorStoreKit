// VectorStoreKit: Autoencoder Types
//
// Common types and protocols for autoencoder implementations

import Foundation

/// Protocol for all autoencoder implementations
public protocol Autoencoder: Actor {
    associatedtype Config: AutoencoderConfiguration
    
    /// Encode vectors to lower-dimensional representations
    func encode(_ vectors: [[Float]]) async throws -> [[Float]]
    
    /// Decode embeddings back to original space
    func decode(_ embeddings: [[Float]]) async throws -> [[Float]]
    
    /// Encode and decode for reconstruction
    func reconstruct(_ vectors: [[Float]]) async throws -> [[Float]]
    
    /// Get reconstruction error
    func reconstructionError(_ vectors: [[Float]]) async throws -> Float
    
    /// Train the autoencoder
    func train(on data: [[Float]], validationData: [[Float]]?) async throws
    
    /// Check if the model is trained
    var isTrained: Bool { get }
}

/// Base protocol for autoencoder configurations
public protocol AutoencoderConfiguration: Sendable, Codable {
    var inputDimensions: Int { get }
    var encodedDimensions: Int { get }
    var training: AutoencoderTrainingConfiguration { get }
    var regularization: RegularizationConfig { get }
}

/// Training configuration shared across all autoencoders
public struct AutoencoderTrainingConfiguration: Sendable, Codable {
    public let batchSize: Int
    public let epochs: Int
    public let learningRate: Float
    public let optimizerType: OptimizerType
    public let lrSchedule: LearningRateSchedule?
    
    public enum OptimizerType: String, Sendable, Codable {
        case sgd = "sgd"
        case adam = "adam"
        case rmsprop = "rmsprop"
        case adagrad = "adagrad"
    }
    
    public enum LearningRateSchedule: Sendable, Codable {
        case exponentialDecay(rate: Float)
        case stepDecay(stepSize: Int, gamma: Float)
        case cosineAnnealing(tMax: Int)
        case warmupCosine(warmupSteps: Int)
    }
    
    public init(
        batchSize: Int = 128,
        epochs: Int = 100,
        learningRate: Float = 0.001,
        optimizerType: OptimizerType = .adam,
        lrSchedule: LearningRateSchedule? = .exponentialDecay(rate: 0.95)
    ) {
        self.batchSize = batchSize
        self.epochs = epochs
        self.learningRate = learningRate
        self.optimizerType = optimizerType
        self.lrSchedule = lrSchedule
    }
}

/// Regularization configuration
public struct RegularizationConfig: Sendable, Codable {
    public let l1Weight: Float
    public let l2Weight: Float
    public let dropout: Float
    public let gradientClipping: Float?
    
    public init(
        l1Weight: Float = 0,
        l2Weight: Float = 0.0001,
        dropout: Float = 0.1,
        gradientClipping: Float? = 5.0
    ) {
        self.l1Weight = l1Weight
        self.l2Weight = l2Weight
        self.dropout = dropout
        self.gradientClipping = gradientClipping
    }
}

/// Training history for tracking progress
public struct TrainingHistory: Codable, Sendable {
    public var epochs: [EpochHistory] = []
    public let patience: Int
    public let minDelta: Float
    
    public struct EpochHistory: Codable, Sendable {
        public let epoch: Int
        public let trainLoss: Float
        public let validationLoss: Float
        public let timestamp: Date
    }
    
    public init(patience: Int = 10, minDelta: Float = 0.0001) {
        self.patience = patience
        self.minDelta = minDelta
    }
    
    public mutating func addEpoch(epoch: Int, trainLoss: Float, validationLoss: Float) {
        epochs.append(EpochHistory(
            epoch: epoch,
            trainLoss: trainLoss,
            validationLoss: validationLoss,
            timestamp: Date()
        ))
    }
    
    public func shouldStop() -> Bool {
        guard epochs.count > patience else { return false }
        
        let recentEpochs = Array(epochs.suffix(patience))
        let bestLoss = recentEpochs.map { $0.validationLoss }.min() ?? Float.infinity
        let currentLoss = epochs.last?.validationLoss ?? Float.infinity
        
        return currentLoss > bestLoss - minDelta
    }
}

/// Autoencoder errors
public enum AutoencoderError: LocalizedError {
    case notTrained
    case insufficientTrainingData
    case dimensionMismatch
    case invalidConfiguration
    
    public var errorDescription: String? {
        switch self {
        case .notTrained:
            return "Autoencoder must be trained before use"
        case .insufficientTrainingData:
            return "Insufficient training data provided"
        case .dimensionMismatch:
            return "Input dimensions do not match configuration"
        case .invalidConfiguration:
            return "Invalid autoencoder configuration"
        }
    }
}
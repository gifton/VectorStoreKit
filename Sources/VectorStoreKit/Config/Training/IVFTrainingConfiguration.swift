// VectorStoreKit: IVF Training Configuration
//
// Configuration for IVF index training

import Foundation

/// Configuration for IVF training
public struct IVFTrainingConfiguration: Sendable {
    public let samplingStrategy: SamplingStrategy
    public let incrementalConfig: IncrementalTrainingConfig?
    public let convergenceThreshold: Float
    public let maxIterations: Int
    public let validationSplit: Float
    public let useMetalAcceleration: Bool
    
    public init(
        samplingStrategy: SamplingStrategy = .stratified(ratio: 0.1),
        incrementalConfig: IncrementalTrainingConfig? = nil,
        convergenceThreshold: Float = 0.001,
        maxIterations: Int = 25,
        validationSplit: Float = 0.1,
        useMetalAcceleration: Bool = true
    ) {
        self.samplingStrategy = samplingStrategy
        self.incrementalConfig = incrementalConfig
        self.convergenceThreshold = convergenceThreshold
        self.maxIterations = maxIterations
        self.validationSplit = validationSplit
        self.useMetalAcceleration = useMetalAcceleration
    }
}

/// Sampling strategies for training data
public enum SamplingStrategy: Sendable {
    case random(ratio: Float)
    case stratified(ratio: Float)
    case reservoir(size: Int)
    case coreset(size: Int, diversity: Float)
    case adaptive(targetSize: Int)
    case importance(weights: [Float])
}
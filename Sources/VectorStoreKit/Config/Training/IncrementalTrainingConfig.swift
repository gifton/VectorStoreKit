// VectorStoreKit: Incremental Training Configuration
//
// Configuration for incremental and online learning

import Foundation

/// Configuration for incremental training
public struct IncrementalTrainingConfig: Sendable {
    public let batchSize: Int
    public let learningRate: Float
    public let momentumFactor: Float
    public let memoryWindow: Int
    public let adaptiveThreshold: Float
    
    public init(
        batchSize: Int = 1000,
        learningRate: Float = 0.01,
        momentumFactor: Float = 0.9,
        memoryWindow: Int = 10000,
        adaptiveThreshold: Float = 0.05
    ) {
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.momentumFactor = momentumFactor
        self.memoryWindow = memoryWindow
        self.adaptiveThreshold = adaptiveThreshold
    }
}
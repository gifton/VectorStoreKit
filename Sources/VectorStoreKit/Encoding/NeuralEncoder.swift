// VectorStoreKit: Neural Encoder/Decoder
//
// Facade for autoencoder implementations that delegates to specific types

import Foundation
import Accelerate
import simd

/// Configuration for neural encoder/decoder
public struct NeuralEncoderConfiguration: Sendable, Codable {
    public let inputDimensions: Int
    public let encodedDimensions: Int
    public let architecture: EncoderArchitecture
    public let training: TrainingConfiguration
    public let regularization: RegularizationConfig
    
    public enum EncoderArchitecture: Sendable, Codable {
        case autoencoder(encoder: [Int], decoder: [Int])
        case variational(encoder: [Int], latentDim: Int, decoder: [Int])
        case sparse(sparsityTarget: Float, hiddenLayers: [Int])
        case denoising(noiseLevel: Float, hiddenLayers: [Int])
        case contractive(contractiveWeight: Float, hiddenLayers: [Int])
    }
    
    // Re-export types from AutoencoderTypes
//    public typealias TrainingConfiguration = TrainingConfiguration
    public typealias RegularizationConfig = VectorStoreKit.RegularizationConfig
    
    public init(
        inputDimensions: Int,
        encodedDimensions: Int,
        architecture: EncoderArchitecture = .autoencoder(
            encoder: [512, 256],
            decoder: [256, 512]
        ),
        training: TrainingConfiguration = TrainingConfiguration(
            batchSize: 128,
            epochs: 100,
            learningRate: 0.001,
            optimizerType: .adam,
            lrSchedule: .exponentialDecay(rate: 0.95)
        ),
        regularization: RegularizationConfig = RegularizationConfig(
            l1Weight: 0,
            l2Weight: 0.0001,
            dropout: 0.1,
            gradientClipping: 5.0
        )
    ) {
        self.inputDimensions = inputDimensions
        self.encodedDimensions = encodedDimensions
        self.architecture = architecture
        self.training = training
        self.regularization = regularization
    }
}

/// Neural encoder/decoder for vector embeddings
public actor NeuralEncoder {
    
    // MARK: - Properties
    
    private let configuration: NeuralEncoderConfiguration
    private let implementation: any Autoencoder
    private let metalCompute: MetalCompute?
    
    // MARK: - Initialization
    
    public init(
        configuration: NeuralEncoderConfiguration,
        metalCompute: MetalCompute? = nil
    ) async {
        self.configuration = configuration
        self.metalCompute = metalCompute
        
        // Create the appropriate autoencoder implementation
        switch configuration.architecture {
        case .autoencoder(let encoderLayers, let decoderLayers):
            let config = StandardAutoencoderConfiguration(
                inputDimensions: configuration.inputDimensions,
                encodedDimensions: configuration.encodedDimensions,
                encoderLayers: encoderLayers,
                decoderLayers: decoderLayers,
                training: configuration.training,
                regularization: configuration.regularization
            )
            self.implementation = await AutoencoderBase(
                configuration: config,
                metalCompute: metalCompute
            )
            
        case .variational(let encoderLayers, _, let decoderLayers):
            let config = VAEConfiguration(
                inputDimensions: configuration.inputDimensions,
                encodedDimensions: configuration.encodedDimensions,
                encoderLayers: encoderLayers,
                decoderLayers: decoderLayers,
                training: configuration.training,
                regularization: configuration.regularization
            )
            self.implementation = await VariationalAutoencoder(
                configuration: config,
                metalCompute: metalCompute
            )
            
        case .sparse(let sparsityTarget, let hiddenLayers):
            let config = SparseAutoencoderConfiguration(
                inputDimensions: configuration.inputDimensions,
                encodedDimensions: configuration.encodedDimensions,
                encoderLayers: hiddenLayers,
                decoderLayers: Array(hiddenLayers.reversed()),
                sparsityTarget: sparsityTarget,
                training: configuration.training,
                regularization: configuration.regularization
            )
            self.implementation = await SparseAutoencoder(
                configuration: config,
                metalCompute: metalCompute
            )
            
        case .denoising(let noiseLevel, let hiddenLayers):
            let config = DenoisingAutoencoderConfiguration(
                inputDimensions: configuration.inputDimensions,
                encodedDimensions: configuration.encodedDimensions,
                encoderLayers: hiddenLayers,
                decoderLayers: Array(hiddenLayers.reversed()),
                noiseLevel: noiseLevel,
                training: configuration.training,
                regularization: configuration.regularization
            )
            self.implementation = await DenoisingAutoencoder(
                configuration: config,
                metalCompute: metalCompute
            )
            
        case .contractive(let contractiveWeight, let hiddenLayers):
            let config = ContractiveAutoencoderConfiguration(
                inputDimensions: configuration.inputDimensions,
                encodedDimensions: configuration.encodedDimensions,
                encoderLayers: hiddenLayers,
                decoderLayers: Array(hiddenLayers.reversed()),
                contractiveWeight: contractiveWeight,
                training: configuration.training,
                regularization: configuration.regularization
            )
            self.implementation = await ContractiveAutoencoder(
                configuration: config,
                metalCompute: metalCompute
            )
        }
    }
    
    // MARK: - Encoding/Decoding
    
    /// Encode vectors to lower-dimensional representations
    public func encode(_ vectors: [[Float]]) async throws -> [[Float]] {
        try await implementation.encode(vectors)
    }
    
    /// Decode embeddings back to original space
    public func decode(_ embeddings: [[Float]]) async throws -> [[Float]] {
        try await implementation.decode(embeddings)
    }
    
    /// Encode and decode for reconstruction
    public func reconstruct(_ vectors: [[Float]]) async throws -> [[Float]] {
        try await implementation.reconstruct(vectors)
    }
    
    /// Get reconstruction error
    public func reconstructionError(_ vectors: [[Float]]) async throws -> Float {
        try await implementation.reconstructionError(vectors)
    }
    
    /// Check if the model is trained
    public var isTrained: Bool {
        get async {
            await implementation.isTrained
        }
    }
    
    // MARK: - Training
    
    /// Train the encoder/decoder on data
    public func train(on data: [[Float]], validationData: [[Float]]? = nil) async throws {
        try await implementation.train(on: data, validationData: validationData)
    }
    
    /// Fine-tune on new data
    public func finetune(on data: [[Float]], epochs: Int = 10) async throws {
        guard await isTrained else {
            throw NeuralEncoderError.notTrained
        }
        
        // Create a new configuration with lower learning rate for fine-tuning
        var finetuneConfig = configuration
        finetuneConfig.training = TrainingConfiguration(
            batchSize: configuration.training.batchSize,
            epochs: epochs,
            learningRate: configuration.training.learningRate * 0.1,
            optimizerType: configuration.training.optimizerType,
            lrSchedule: configuration.training.lrSchedule
        )
        
        // Create a new encoder with the fine-tune configuration and train it
        let finetuneEncoder = await NeuralEncoder(
            configuration: finetuneConfig,
            metalCompute: metalCompute
        )
        
        try await finetuneEncoder.train(on: data)
    }
    
    // MARK: - Model Persistence
    
    public func save(to url: URL) async throws {
        // Save configuration and implementation state
        let modelData = NeuralEncoderModel(
            configuration: configuration,
            trained: await isTrained
        )
        
        let encoder = JSONEncoder()
        let data = try encoder.encode(modelData)
        try data.write(to: url)
    }
    
    public func load(from url: URL) async throws {
        // Load configuration and recreate implementation
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let modelData = try decoder.decode(NeuralEncoderModel.self, from: data)
        
        // Note: This creates a new untrained model with the saved configuration
        // To truly restore a trained model, we'd need to save/load the weights
    }
}

// MARK: - Supporting Types

/// Model data for persistence
private struct NeuralEncoderModel: Codable {
    let configuration: NeuralEncoderConfiguration
    let trained: Bool
}

/// Neural encoder errors
public enum NeuralEncoderError: LocalizedError {
    case notTrained
    case insufficientTrainingData
    case dimensionMismatch
    case invalidConfiguration
    
    public var errorDescription: String? {
        switch self {
        case .notTrained:
            return "Neural encoder must be trained before use"
        case .insufficientTrainingData:
            return "Insufficient training data provided"
        case .dimensionMismatch:
            return "Input dimensions do not match configuration"
        case .invalidConfiguration:
            return "Invalid encoder configuration"
        }
    }
}

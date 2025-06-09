// VectorStoreKit: IVF Index Configuration
//
// Configuration for Inverted File (IVF) index

import Foundation

/// Configuration for IVF index
public struct IVFConfiguration: IndexConfiguration {
    public let dimensions: Int
    public let numberOfCentroids: Int
    public let numberOfProbes: Int
    public let trainingSampleSize: Int
    public let quantization: QuantizationType?
    public let clusteringConfig: KMeansClusteringConfiguration
    public let neuralClusteringConfig: NeuralClusteringConfig?
    public let clusteringMethod: ClusteringMethod
    
    public enum QuantizationType: Sendable, Codable {
        case productQuantization(segments: Int, bits: Int)
        case scalarQuantization(bits: Int)
    }
    
    public enum ClusteringMethod: String, Sendable, Codable {
        case kmeans = "kmeans"
        case neural = "neural"
        case hybrid = "hybrid" // Start with k-means, then refine with neural
    }
    
    public init(
        dimensions: Int,
        numberOfCentroids: Int = 1024,
        numberOfProbes: Int = 10,
        trainingSampleSize: Int = 100_000,
        quantization: QuantizationType? = nil,
        clusteringConfig: KMeansClusteringConfiguration = .default,
        neuralClusteringConfig: NeuralClusteringConfig? = nil,
        clusteringMethod: ClusteringMethod = .kmeans
    ) {
        self.dimensions = dimensions
        self.numberOfCentroids = numberOfCentroids
        self.numberOfProbes = numberOfProbes
        self.trainingSampleSize = trainingSampleSize
        self.quantization = quantization
        self.clusteringConfig = clusteringConfig
        self.neuralClusteringConfig = neuralClusteringConfig
        self.clusteringMethod = clusteringMethod
    }
    
    public func validate() throws {
        guard dimensions > 0 else {
            throw IVFError.invalidDimensions(dimensions)
        }
        guard numberOfCentroids > 0 else {
            throw IVFError.invalidParameter("numberOfCentroids", numberOfCentroids)
        }
        guard numberOfProbes > 0 && numberOfProbes <= numberOfCentroids else {
            throw IVFError.invalidParameter("numberOfProbes", numberOfProbes)
        }
    }
    
    public func estimatedMemoryUsage(for vectorCount: Int) -> Int {
        let centroidMemory = numberOfCentroids * dimensions * MemoryLayout<Float>.size
        let listMemory = vectorCount * (MemoryLayout<String>.size + 16) // ID + overhead
        let vectorMemory: Int
        
        switch quantization {
        case .productQuantization(let segments, let bits):
            vectorMemory = vectorCount * segments * (bits / 8)
        case .scalarQuantization(let bits):
            vectorMemory = vectorCount * dimensions * (bits / 8)
        case nil:
            vectorMemory = vectorCount * dimensions * MemoryLayout<Float>.size
        }
        
        var totalMemory = centroidMemory + listMemory + vectorMemory
        
        // Add neural clustering memory if enabled
        if clusteringMethod == .neural || clusteringMethod == .hybrid,
           let neuralConfig = neuralClusteringConfig {
            totalMemory += neuralConfig.estimatedMemoryUsage(
                dimensions: dimensions,
                clusters: numberOfCentroids
            )
        }
        
        return totalMemory
    }
    
    public func computationalComplexity() -> ComputationalComplexity {
        return .linearithmic // O(n log k) for search
    }
}
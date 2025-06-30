// LearnedDistance.swift
// VectorStoreKit
//
// Machine learning-based distance metric with model loading and caching

import Foundation
import simd
import Metal
import CoreML
import Accelerate

/// Learned distance metric using neural networks or other ML models
public struct LearnedDistance {
    
    /// Type of learned model
    public enum ModelType {
        case neural(layers: [Int])
        case siamese
        case metric(embedding: Int)
        case custom(model: LearnedDistanceModel)
    }
    
    private let model: LearnedDistanceModel
    private let useGPU: Bool
    private let batchSize: Int
    
    /// Initialize with model type
    public init(modelType: ModelType, useGPU: Bool = true, batchSize: Int = 32) throws {
        self.useGPU = useGPU
        self.batchSize = batchSize
        
        switch modelType {
        case .neural(let layers):
            self.model = try NeuralDistanceModel(layers: layers, useGPU: useGPU)
        case .siamese:
            self.model = try SiameseDistanceModel(useGPU: useGPU)
        case .metric(let embeddingDim):
            self.model = try MetricLearningModel(embeddingDim: embeddingDim, useGPU: useGPU)
        case .custom(let customModel):
            self.model = customModel
        }
    }
    
    /// Initialize from saved model
    public init(modelPath: URL, useGPU: Bool = true) throws {
        self.useGPU = useGPU
        self.batchSize = 32
        
        // Load model from disk
        let data = try Data(contentsOf: modelPath)
        self.model = try JSONDecoder().decode(AnyLearnedDistanceModel.self, from: data).model
    }
    
    /// Compute learned distance between two vectors
    public func distance(_ a: Vector512, _ b: Vector512) throws -> Float {
        return try model.computeDistance(a, b)
    }
    
    /// Batch computation for efficiency
    public func batchDistance(query: Vector512, candidates: [Vector512]) async throws -> [Float] {
        var results = [Float](repeating: 0, count: candidates.count)
        
        // Process in batches for GPU efficiency
        for batchStart in stride(from: 0, to: candidates.count, by: batchSize) {
            let batchEnd = min(batchStart + batchSize, candidates.count)
            let batch = Array(candidates[batchStart..<batchEnd])
            
            let batchResults = try await model.computeBatchDistance(query: query, candidates: batch)
            
            for (i, result) in batchResults.enumerated() {
                results[batchStart + i] = result
            }
        }
        
        return results
    }
    
    /// Update model with new training examples
    public func update(positives: [(Vector512, Vector512)], negatives: [(Vector512, Vector512)]) async throws {
        try await model.update(positives: positives, negatives: negatives)
    }
}

/// Protocol for learned distance models
public protocol LearnedDistanceModel: Codable {
    func computeDistance(_ a: Vector512, _ b: Vector512) throws -> Float
    func computeBatchDistance(query: Vector512, candidates: [Vector512]) async throws -> [Float]
    func update(positives: [(Vector512, Vector512)], negatives: [(Vector512, Vector512)]) async throws
}

/// Neural network-based distance model
public class NeuralDistanceModel: LearnedDistanceModel {
    private let layers: [Int]
    private var weights: [[Float]]
    private var biases: [[Float]]
    private let useGPU: Bool
    private var metalDevice: MTLDevice?
    private var commandQueue: MTLCommandQueue?
    
    public init(layers: [Int], useGPU: Bool) throws {
        self.layers = layers
        self.useGPU = useGPU
        
        // Initialize weights and biases
        weights = []
        biases = []
        
        // Input is concatenated vectors (1024 dimensions)
        var prevSize = 1024
        for layerSize in layers {
            // Xavier initialization
            let scale = sqrt(2.0 / Float(prevSize))
            let weight = (0..<(prevSize * layerSize)).map { _ in Float.random(in: -scale...scale) }
            let bias = [Float](repeating: 0, count: layerSize)
            
            weights.append(weight)
            biases.append(bias)
            prevSize = layerSize
        }
        
        // Output layer (single distance value)
        let outputScale = sqrt(2.0 / Float(prevSize))
        weights.append((0..<prevSize).map { _ in Float.random(in: -outputScale...outputScale) })
        biases.append([0])
        
        // Setup Metal if requested
        if useGPU {
            metalDevice = MTLCreateSystemDefaultDevice()
            commandQueue = metalDevice?.makeCommandQueue()
        }
    }
    
    /// SIMD-optimized neural network forward pass
    public func computeDistance(_ a: Vector512, _ b: Vector512) throws -> Float {
        // Concatenate vectors using SIMD
        var input = [Float](repeating: 0, count: 1024)
        
        let aArray = a.toArray()
        let bArray = b.toArray()
        
        // Use SIMD for efficient concatenation
        input.withUnsafeMutableBufferPointer { inputPtr in
            aArray.withUnsafeBufferPointer { aPtr in
                bArray.withUnsafeBufferPointer { bPtr in
                    let inputSIMD = inputPtr.baseAddress!.withMemoryRebound(to: SIMD8<Float>.self, capacity: 64) { $0 }
                    let aSIMD = aPtr.baseAddress!.withMemoryRebound(to: SIMD8<Float>.self, capacity: 64) { $0 }
                    let bSIMD = bPtr.baseAddress!.withMemoryRebound(to: SIMD8<Float>.self, capacity: 64) { $0 }
                    
                    // Copy first 512 elements (vector a)
                    for i in 0..<64 {
                        inputSIMD[i] = aSIMD[i]
                    }
                    
                    // Copy next 512 elements (vector b)
                    for i in 0..<64 {
                        inputSIMD[64 + i] = bSIMD[i]
                    }
                }
            }
        }
        
        // SIMD-optimized forward pass through network
        var activations = input
        
        for i in 0..<weights.count {
            let weight = weights[i]
            let bias = biases[i]
            let inputSize = activations.count
            let outputSize = bias.count
            
            var output = [Float](repeating: 0, count: outputSize)
            
            // Use BLAS for optimized matrix-vector multiplication
            cblas_sgemv(
                CblasRowMajor,
                CblasNoTrans,
                Int32(outputSize), Int32(inputSize),
                1.0,
                weight, Int32(inputSize),
                activations, 1,
                0.0,
                &output, 1
            )
            
            // Add bias and apply activation using SIMD
            output.withUnsafeMutableBufferPointer { outputPtr in
                bias.withUnsafeBufferPointer { biasPtr in
                    let outputChunks = (outputSize + 7) / 8
                    let outputSIMD = outputPtr.baseAddress!.withMemoryRebound(to: SIMD8<Float>.self, capacity: outputChunks) { $0 }
                    let biasSIMD = biasPtr.baseAddress!.withMemoryRebound(to: SIMD8<Float>.self, capacity: outputChunks) { $0 }
                    
                    let isLastLayer = (i == weights.count - 1)
                    
                    for j in 0..<outputChunks {
                        let startIdx = j * 8
                        let endIdx = min(startIdx + 8, outputSize)
                        let actualSize = endIdx - startIdx
                        
                        if actualSize == 8 {
                            // Full SIMD operation
                            let values = outputSIMD[j] + biasSIMD[j]
                            if isLastLayer {
                                outputSIMD[j] = values
                            } else {
                                // ReLU activation: max(0, x)
                                var result = SIMD8<Float>()
                                for k in 0..<8 {
                                    result[k] = values[k] > 0 ? values[k] : 0
                                }
                                outputSIMD[j] = result
                            }
                        } else {
                            // Handle remaining elements scalar
                            for k in startIdx..<endIdx {
                                let value = output[k] + bias[k]
                                output[k] = isLastLayer ? value : max(0, value)
                            }
                        }
                    }
                }
            }
            
            activations = output
        }
        
        // Apply sigmoid to get distance in [0, 1]
        return 1.0 / (1.0 + exp(-activations[0]))
    }
    
    /// SIMD-optimized batch neural network inference
    public func computeBatchDistance(query: Vector512, candidates: [Vector512]) async throws -> [Float] {
        guard !candidates.isEmpty else { return [] }
        
        let batchSize = candidates.count
        let inputSize = 1024  // 512 * 2
        
        // Prepare batch input matrix
        var batchInput = [Float](repeating: 0, count: batchSize * inputSize)
        let queryArray = query.toArray()
        
        // Fill batch input using SIMD
        batchInput.withUnsafeMutableBufferPointer { batchPtr in
            queryArray.withUnsafeBufferPointer { queryPtr in
                for (batchIdx, candidate) in candidates.enumerated() {
                    let candidateArray = candidate.toArray()
                    let rowOffset = batchIdx * inputSize
                    
                    // Copy query vector (first 512 elements)
                    let queryDest = batchPtr.baseAddress!.advanced(by: rowOffset)
                    queryDest.initialize(from: queryPtr.baseAddress!, count: 512)
                    
                    // Copy candidate vector (next 512 elements)
                    candidateArray.withUnsafeBufferPointer { candidatePtr in
                        let candidateDest = batchPtr.baseAddress!.advanced(by: rowOffset + 512)
                        candidateDest.initialize(from: candidatePtr.baseAddress!, count: 512)
                    }
                }
            }
        }
        
        // Batch forward pass through network
        var batchActivations = batchInput
        var currentBatchSize = batchSize
        var currentInputSize = inputSize
        
        for layerIdx in 0..<weights.count {
            let weight = weights[layerIdx]
            let bias = biases[layerIdx]
            let outputSize = bias.count
            
            var batchOutput = [Float](repeating: 0, count: currentBatchSize * outputSize)
            
            // Batch matrix multiplication using BLAS
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasTrans,
                Int32(currentBatchSize), Int32(outputSize), Int32(currentInputSize),
                1.0,
                batchActivations, Int32(currentInputSize),
                weight, Int32(currentInputSize),
                0.0,
                &batchOutput, Int32(outputSize)
            )
            
            // Add bias and apply activation for entire batch
            let isLastLayer = (layerIdx == weights.count - 1)
            
            for batchIdx in 0..<currentBatchSize {
                let rowOffset = batchIdx * outputSize
                
                for i in stride(from: 0, to: outputSize, by: 8) {
                    let endIdx = min(i + 8, outputSize)
                    let actualSize = endIdx - i
                    
                    if actualSize == 8 {
                        let outputPtr = batchOutput.withUnsafeMutableBufferPointer { $0.baseAddress!.advanced(by: rowOffset + i) }
                        let outputSIMD = outputPtr.withMemoryRebound(to: SIMD8<Float>.self, capacity: 1) { $0 }
                        
                        let biasPtr = bias.withUnsafeBufferPointer { $0.baseAddress!.advanced(by: i) }
                        let biasSIMD = biasPtr.withMemoryRebound(to: SIMD8<Float>.self, capacity: 1) { $0 }
                        
                        let values = outputSIMD.pointee + biasSIMD.pointee
                        if isLastLayer {
                            outputSIMD.pointee = values
                        } else {
                            // ReLU activation: max(0, x)
                            var result = SIMD8<Float>()
                            for k in 0..<8 {
                                result[k] = values[k] > 0 ? values[k] : 0
                            }
                            outputSIMD.pointee = result
                        }
                    } else {
                        // Handle remaining elements
                        for j in i..<endIdx {
                            let idx = rowOffset + j
                            let value = batchOutput[idx] + bias[j]
                            batchOutput[idx] = isLastLayer ? value : max(0, value)
                        }
                    }
                }
            }
            
            batchActivations = batchOutput
            currentInputSize = outputSize
        }
        
        // Extract results and apply sigmoid
        var results = [Float](repeating: 0, count: batchSize)
        for i in 0..<batchSize {
            let logit = batchActivations[i]  // Only one output per sample
            results[i] = 1.0 / (1.0 + exp(-logit))
        }
        
        return results
    }
    
    public func update(positives: [(Vector512, Vector512)], negatives: [(Vector512, Vector512)]) async throws {
        // Simplified gradient descent update
        let learningRate: Float = 0.001
        
        for _ in 0..<10 { // Mini epochs
            // Compute gradients for positive pairs (should have low distance)
            for (a, b) in positives {
                let predicted = try computeDistance(a, b)
                let error = predicted // Want this to be close to 0
                
                // Backpropagation would be implemented here
                // For now, simple weight adjustment
                for i in 0..<weights.count {
                    for j in 0..<weights[i].count {
                        weights[i][j] -= learningRate * error * 0.01
                    }
                }
            }
            
            // Compute gradients for negative pairs (should have high distance)
            for (a, b) in negatives {
                let predicted = try computeDistance(a, b)
                let error = 1.0 - predicted // Want this to be close to 1
                
                // Backpropagation would be implemented here
                for i in 0..<weights.count {
                    for j in 0..<weights[i].count {
                        weights[i][j] += learningRate * error * 0.01
                    }
                }
            }
        }
    }
    
    // Codable implementation
    enum CodingKeys: String, CodingKey {
        case layers, weights, biases, useGPU
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(layers, forKey: .layers)
        try container.encode(weights, forKey: .weights)
        try container.encode(biases, forKey: .biases)
        try container.encode(useGPU, forKey: .useGPU)
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        layers = try container.decode([Int].self, forKey: .layers)
        weights = try container.decode([[Float]].self, forKey: .weights)
        biases = try container.decode([[Float]].self, forKey: .biases)
        useGPU = try container.decode(Bool.self, forKey: .useGPU)
        
        if useGPU {
            metalDevice = MTLCreateSystemDefaultDevice()
            commandQueue = metalDevice?.makeCommandQueue()
        }
    }
}

/// Siamese network distance model
public class SiameseDistanceModel: LearnedDistanceModel {
    private let embeddingDim: Int = 128
    private var encoder: NeuralDistanceModel
    
    public init(useGPU: Bool) throws {
        // Siamese network with shared encoder
        encoder = try NeuralDistanceModel(layers: [256, 128], useGPU: useGPU)
    }
    
    public func computeDistance(_ a: Vector512, _ b: Vector512) throws -> Float {
        // Encode both vectors
        let embeddingA = try encode(a)
        let embeddingB = try encode(b)
        
        // Compute euclidean distance in embedding space
        var sum: Float = 0
        for i in 0..<embeddingDim {
            let diff = embeddingA[i] - embeddingB[i]
            sum += diff * diff
        }
        
        return sqrt(sum)
    }
    
    private func encode(_ vector: Vector512) throws -> [Float] {
        // Use first part of neural network as encoder
        // Simplified implementation
        return Array(vector.toArray().prefix(embeddingDim))
    }
    
    public func computeBatchDistance(query: Vector512, candidates: [Vector512]) async throws -> [Float] {
        return try candidates.map { try computeDistance(query, $0) }
    }
    
    public func update(positives: [(Vector512, Vector512)], negatives: [(Vector512, Vector512)]) async throws {
        try await encoder.update(positives: positives, negatives: negatives)
    }
    
    // Codable implementation
    enum CodingKeys: String, CodingKey {
        case embeddingDim, encoder
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(embeddingDim, forKey: .embeddingDim)
        try container.encode(self.encoder, forKey: .encoder)
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        embeddingDim = try container.decode(Int.self, forKey: .embeddingDim)
        encoder = try container.decode(NeuralDistanceModel.self, forKey: .encoder)
    }
}

/// Metric learning model
public class MetricLearningModel: LearnedDistanceModel {
    private let embeddingDim: Int
    private var transformMatrix: [[Float]]
    
    public init(embeddingDim: Int, useGPU: Bool) throws {
        self.embeddingDim = embeddingDim
        
        // Initialize with identity-like transformation
        transformMatrix = []
        for i in 0..<embeddingDim {
            var row = [Float](repeating: 0, count: 512)
            for j in 0..<512 {
                if j < embeddingDim {
                    row[j] = (i == j) ? 1.0 : 0.0
                }
            }
            transformMatrix.append(row)
        }
    }
    
    public func computeDistance(_ a: Vector512, _ b: Vector512) throws -> Float {
        // Transform vectors to learned metric space
        let transformedA = transform(a)
        let transformedB = transform(b)
        
        // Compute euclidean distance in transformed space
        var sum: Float = 0
        for i in 0..<embeddingDim {
            let diff = transformedA[i] - transformedB[i]
            sum += diff * diff
        }
        
        return sqrt(sum)
    }
    
    private func transform(_ vector: Vector512) -> [Float] {
        let input = vector.toArray()
        var output = [Float](repeating: 0, count: embeddingDim)
        
        for i in 0..<embeddingDim {
            var sum: Float = 0
            for j in 0..<512 {
                sum += transformMatrix[i][j] * input[j]
            }
            output[i] = sum
        }
        
        return output
    }
    
    public func computeBatchDistance(query: Vector512, candidates: [Vector512]) async throws -> [Float] {
        return try candidates.map { try computeDistance(query, $0) }
    }
    
    public func update(positives: [(Vector512, Vector512)], negatives: [(Vector512, Vector512)]) async throws {
        // Large Margin Nearest Neighbor (LMNN) style update
        let learningRate: Float = 0.0001
        
        for _ in 0..<5 { // Mini epochs
            // Pull positive pairs together
            for (a, b) in positives {
                let distance = try computeDistance(a, b)
                if distance > 0.1 { // Margin
                    // Update transformation matrix
                    updateTransform(a: a, b: b, pull: true, learningRate: learningRate)
                }
            }
            
            // Push negative pairs apart
            for (a, b) in negatives {
                let distance = try computeDistance(a, b)
                if distance < 1.0 { // Margin
                    updateTransform(a: a, b: b, pull: false, learningRate: learningRate)
                }
            }
        }
    }
    
    private func updateTransform(a: Vector512, b: Vector512, pull: Bool, learningRate: Float) {
        let direction: Float = pull ? -1.0 : 1.0
        let aArray = a.toArray()
        let bArray = b.toArray()
        
        for i in 0..<embeddingDim {
            for j in 0..<512 {
                let gradient = direction * (aArray[j] - bArray[j]) * 0.01
                transformMatrix[i][j] += learningRate * gradient
            }
        }
    }
    
    // Codable implementation
    enum CodingKeys: String, CodingKey {
        case embeddingDim, transformMatrix
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(embeddingDim, forKey: .embeddingDim)
        try container.encode(transformMatrix, forKey: .transformMatrix)
    }
    
    public required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        embeddingDim = try container.decode(Int.self, forKey: .embeddingDim)
        transformMatrix = try container.decode([[Float]].self, forKey: .transformMatrix)
    }
}

/// Type-erased wrapper for Codable
struct AnyLearnedDistanceModel: Codable {
    let model: LearnedDistanceModel
    
    enum ModelType: String, Codable {
        case neural, siamese, metric
    }
    
    enum CodingKeys: String, CodingKey {
        case type, model
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        if model is NeuralDistanceModel {
            try container.encode(ModelType.neural, forKey: .type)
            try container.encode(model as! NeuralDistanceModel, forKey: .model)
        } else if model is SiameseDistanceModel {
            try container.encode(ModelType.siamese, forKey: .type)
            try container.encode(model as! SiameseDistanceModel, forKey: .model)
        } else if model is MetricLearningModel {
            try container.encode(ModelType.metric, forKey: .type)
            try container.encode(model as! MetricLearningModel, forKey: .model)
        }
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(ModelType.self, forKey: .type)
        
        switch type {
        case .neural:
            model = try container.decode(NeuralDistanceModel.self, forKey: .model)
        case .siamese:
            model = try container.decode(SiameseDistanceModel.self, forKey: .model)
        case .metric:
            model = try container.decode(MetricLearningModel.self, forKey: .model)
        }
    }
}

/// Global cache for learned models
public actor LearnedDistanceCache {
    private var cache: [String: LearnedDistance] = [:]
    private var accessOrder: [String] = [] // Track access order for LRU
    private let maxCacheSize = 5
    
    public static let shared = LearnedDistanceCache()
    
    public func getModel(id: String) -> LearnedDistance? {
        if let model = cache[id] {
            // Move to end for LRU
            if let index = accessOrder.firstIndex(of: id) {
                accessOrder.remove(at: index)
            }
            accessOrder.append(id)
            return model
        }
        return nil
    }
    
    public func cacheModel(id: String, model: LearnedDistance) {
        // Remove existing entry if present
        if cache[id] != nil, let index = accessOrder.firstIndex(of: id) {
            accessOrder.remove(at: index)
        }
        
        // Evict least recently used if at capacity
        if cache.count >= maxCacheSize && cache[id] == nil {
            if let lruKey = accessOrder.first {
                cache.removeValue(forKey: lruKey)
                accessOrder.removeFirst()
            }
        }
        
        cache[id] = model
        accessOrder.append(id)
    }
    
    public func loadModel(id: String, from url: URL) async throws -> LearnedDistance {
        if let cached = cache[id] {
            return cached
        }
        
        let model = try LearnedDistance(modelPath: url)
        cacheModel(id: id, model: model)
        return model
    }
    
    public func clear() {
        cache.removeAll()
    }
}
// VectorStoreKit: Neural Clustering
//
// Neural network-based clustering for improved IVF index performance

import Foundation
import simd
@preconcurrency import Metal

/// Neural clustering for IVF index
public actor NeuralClustering {
    // MARK: - Properties
    
    private let configuration: NeuralClusteringConfiguration
    private var clusterNetwork: NeuralNetwork?
    private var probeNetwork: NeuralNetwork?
    private var centroids: [[Float]] = []
    private var queryHistory: RingBuffer<[Float]>
    private var performanceMetrics: NeuralClusteringMetrics
    private let metalPipeline: MetalMLPipeline
    
    // MARK: - Initialization
    
    public init(configuration: NeuralClusteringConfiguration) async throws {
        self.configuration = configuration
        self.queryHistory = RingBuffer(capacity: configuration.historySize)
        self.performanceMetrics = NeuralClusteringMetrics()
        
        // Initialize Metal pipeline
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw NeuralClusteringError.metalNotAvailable
        }
        self.metalPipeline = try MetalMLPipeline(device: device)
        
        // Initialize neural networks
        try await initializeNetworks()
    }
    
    // MARK: - Public Methods
    
    /// Train neural clustering on sample vectors
    public func train(
        vectors: [[Float]],
        initialCentroids: [[Float]]? = nil
    ) async throws -> NeuralClusteringResult {
        // Store initial centroids
        if let initial = initialCentroids {
            self.centroids = initial
        } else {
            self.centroids = try await initializeCentroids(from: vectors)
        }
        
        // Prepare training data
        let (inputs, targets) = prepareTrainingData(vectors: vectors)
        
        // Train cluster assignment network
        let clusterConfig = NetworkTrainingConfig(
            epochs: configuration.trainingEpochs,
            batchSize: configuration.batchSize,
            lossFunction: .crossEntropy,
            earlyStoppingPatience: 10
        )
        
        // Convert inputs and targets to MetalBuffer tuples
        var trainingData: [(input: MetalBuffer, target: MetalBuffer)] = []
        for (input, target) in zip(inputs, targets) {
            let inputBuffer = try await metalPipeline.allocateBuffer(size: input.count)
            let targetBuffer = try await metalPipeline.allocateBuffer(size: target.count)
            
            // Copy data to buffers
            let inputPtr = inputBuffer.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
            let targetPtr = targetBuffer.buffer.contents().bindMemory(to: Float.self, capacity: target.count)
            
            for i in 0..<input.count {
                inputPtr[i] = input[i]
            }
            for i in 0..<target.count {
                targetPtr[i] = target[i]
            }
            
            trainingData.append((input: inputBuffer, target: targetBuffer))
        }
        
        try await clusterNetwork?.train(
            data: trainingData,
            config: clusterConfig
        )
        
        // Update centroids based on learned assignments
        let newCentroids = try await updateCentroidsWithNetwork(vectors: vectors)
        self.centroids = newCentroids
        
        // Train probe prediction network if enabled
        if configuration.adaptiveProbing {
            try await trainProbeNetwork(vectors: vectors)
        }
        
        return NeuralClusteringResult(
            centroids: centroids,
            clusterNetwork: clusterNetwork,
            probeNetwork: probeNetwork,
            metrics: performanceMetrics
        )
    }
    
    /// Find nearest centroids using neural network
    public func findNearestCentroids(
        for vector: [Float],
        count: Int
    ) async throws -> [Int] {
        guard let network = clusterNetwork else {
            throw NeuralClusteringError.notTrained
        }
        
        // Get cluster probabilities from network
        let probabilities = await network.forward(vector)
        
        // Find top-k clusters
        let indexed = probabilities.enumerated().map { ($0.offset, $0.element) }
        let sorted = indexed.sorted { $0.1 > $1.1 }
        
        return Array(sorted.prefix(count).map { $0.0 })
    }
    
    /// Predict optimal probe count for a query
    public func predictProbeCount(
        for query: [Float],
        targetRecall: Float = 0.95
    ) async throws -> Int {
        guard let network = probeNetwork else {
            // Fallback to configured default
            return configuration.defaultProbes
        }
        
        // Prepare input: query features + target recall
        var input = query
        input.append(targetRecall)
        
        // Predict probe count
        let prediction = await network.forward(input)
        let probeCount = Int(round(prediction[0]))
        
        // Clamp to valid range
        return max(1, min(probeCount, centroids.count))
    }
    
    /// Update clustering based on query patterns
    public func adaptToQueries<Metadata: Codable & Sendable>(
        queries: [[Float]],
        results: [[SearchResult<Metadata>]]
    ) async throws {
        // Add to query history
        for query in queries {
            queryHistory.append(query)
        }
        
        // Update performance metrics
        await updateMetrics(queries: queries, results: results)
        
        // Retrain periodically based on history
        if queryHistory.currentCount >= configuration.adaptationThreshold {
            try await adaptClustering()
        }
    }
    
    /// Get current centroids
    public func getCentroids() async -> [[Float]] {
        return centroids
    }
    
    /// Get cluster assignment probabilities
    public func getClusterProbabilities(for vector: [Float]) async throws -> [Float] {
        guard let network = clusterNetwork else {
            throw NeuralClusteringError.notTrained
        }
        
        return await network.forward(vector)
    }
    
    // MARK: - Private Methods
    
    private func initializeNetworks() async throws {
        let dimensions = configuration.dimensions
        let numClusters = configuration.numberOfClusters
        
        // Initialize cluster assignment network
        self.clusterNetwork = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Add layers to cluster network
        await clusterNetwork?.addLayer(try await DenseLayer(
            inputSize: dimensions,
            outputSize: dimensions * 2,
            activation: .relu,
            metalPipeline: metalPipeline
        ))
        
        await clusterNetwork?.addLayer(try await DropoutLayer(
            rate: 0.2,
            metalPipeline: metalPipeline
        ))
        
        await clusterNetwork?.addLayer(try await DenseLayer(
            inputSize: dimensions * 2,
            outputSize: dimensions,
            activation: .relu,
            metalPipeline: metalPipeline
        ))
        
        await clusterNetwork?.addLayer(try await DenseLayer(
            inputSize: dimensions,
            outputSize: numClusters,
            activation: .softmax,
            metalPipeline: metalPipeline
        ))
        
        // Probe prediction network (if enabled)
        if configuration.adaptiveProbing {
            self.probeNetwork = try await NeuralNetwork(metalPipeline: metalPipeline)
            
            await probeNetwork?.addLayer(try await DenseLayer(
                inputSize: dimensions + 1, // +1 for target recall
                outputSize: 64,
                activation: .relu,
                metalPipeline: metalPipeline
            ))
            
            await probeNetwork?.addLayer(try await DropoutLayer(
                rate: 0.1,
                metalPipeline: metalPipeline
            ))
            
            await probeNetwork?.addLayer(try await DenseLayer(
                inputSize: 64,
                outputSize: 32,
                activation: .relu,
                metalPipeline: metalPipeline
            ))
            
            await probeNetwork?.addLayer(try await DenseLayer(
                inputSize: 32,
                outputSize: 1,
                activation: .linear, // Regression output
                metalPipeline: metalPipeline
            ))
        }
    }
    
    private func initializeCentroids(from vectors: [[Float]]) async throws -> [[Float]] {
        // K-means++ initialization
        var centroids: [[Float]] = []
        let k = configuration.numberOfClusters
        
        // First centroid: random selection
        let firstIdx = Int.random(in: 0..<vectors.count)
        centroids.append(vectors[firstIdx])
        
        // Remaining centroids: probability proportional to squared distance
        for _ in 1..<k {
            var minDistances: [Float] = []
            
            for vector in vectors {
                var minDist: Float = .infinity
                for centroid in centroids {
                    let dist = euclideanDistance(vector, centroid)
                    minDist = min(minDist, dist)
                }
                minDistances.append(minDist * minDist)
            }
            
            // Sample next centroid
            let totalDist = minDistances.reduce(0, +)
            let probabilities = minDistances.map { $0 / totalDist }
            let nextIdx = weightedSample(probabilities: probabilities)
            centroids.append(vectors[nextIdx])
        }
        
        return centroids
    }
    
    private func prepareTrainingData(
        vectors: [[Float]]
    ) -> (inputs: [[Float]], targets: [[Float]]) {
        var inputs: [[Float]] = []
        var targets: [[Float]] = []
        
        for vector in vectors {
            // Find nearest centroid
            var minDist: Float = .infinity
            var nearestIdx = 0
            
            for (idx, centroid) in centroids.enumerated() {
                let dist = euclideanDistance(vector, centroid)
                if dist < minDist {
                    minDist = dist
                    nearestIdx = idx
                }
            }
            
            // Create one-hot target
            var target = Array(repeating: Float(0), count: centroids.count)
            target[nearestIdx] = 1.0
            
            inputs.append(vector)
            targets.append(target)
        }
        
        return (inputs, targets)
    }
    
    private func updateCentroidsWithNetwork(vectors: [[Float]]) async throws -> [[Float]] {
        guard let network = clusterNetwork else {
            return centroids
        }
        
        var clusterSums = Array(repeating: Array(repeating: Float(0), count: configuration.dimensions), 
                               count: configuration.numberOfClusters)
        var clusterCounts = Array(repeating: Float(0), count: configuration.numberOfClusters)
        
        // Soft assignment using network predictions
        for vector in vectors {
            let probabilities = await network.forward(vector)
            
            for (clusterIdx, prob) in probabilities.enumerated() {
                // Weighted contribution to centroid
                for (dim, value) in vector.enumerated() {
                    clusterSums[clusterIdx][dim] += value * prob
                }
                clusterCounts[clusterIdx] += prob
            }
        }
        
        // Compute new centroids
        var newCentroids: [[Float]] = []
        for (clusterIdx, count) in clusterCounts.enumerated() {
            if count > 0 {
                let centroid = clusterSums[clusterIdx].map { $0 / count }
                newCentroids.append(centroid)
            } else {
                // Keep old centroid if no assignments
                newCentroids.append(centroids[clusterIdx])
            }
        }
        
        return newCentroids
    }
    
    private func trainProbeNetwork(vectors: [[Float]]) async throws {
        // Generate synthetic training data for probe prediction
        var inputs: [[Float]] = []
        var targets: [[Float]] = []
        
        // Sample queries and measure performance with different probe counts
        let sampleSize = min(1000, vectors.count)
        let sampleIndices = (0..<vectors.count).shuffled().prefix(sampleSize)
        
        for idx in sampleIndices {
            let query = vectors[idx]
            
            // Test different probe counts
            for probes in stride(from: 1, through: min(20, centroids.count), by: 2) {
                // Simulate search and measure recall
                let recall = await simulateSearchRecall(
                    query: query,
                    probes: probes,
                    vectors: vectors
                )
                
                // Create training example
                var input = query
                input.append(recall) // Target recall
                inputs.append(input)
                targets.append([Float(probes)])
            }
        }
        
        // Train probe network
        let probeConfig = NetworkTrainingConfig(
            epochs: 50,
            batchSize: 32,
            lossFunction: .mse,
            earlyStoppingPatience: 5
        )
        
        // Convert inputs and targets to MetalBuffer tuples
        var probeTrainingData: [(input: MetalBuffer, target: MetalBuffer)] = []
        for (input, target) in zip(inputs, targets) {
            let inputBuffer = try await metalPipeline.allocateBuffer(size: input.count)
            let targetBuffer = try await metalPipeline.allocateBuffer(size: target.count)
            
            // Copy data to buffers
            let inputPtr = inputBuffer.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
            let targetPtr = targetBuffer.buffer.contents().bindMemory(to: Float.self, capacity: target.count)
            
            for i in 0..<input.count {
                inputPtr[i] = input[i]
            }
            for i in 0..<target.count {
                targetPtr[i] = target[i]
            }
            
            probeTrainingData.append((input: inputBuffer, target: targetBuffer))
        }
        
        try await probeNetwork?.train(
            data: probeTrainingData,
            config: probeConfig
        )
    }
    
    private func simulateSearchRecall(
        query: [Float],
        probes: Int,
        vectors: [[Float]]
    ) async -> Float {
        // Simulate search recall by finding nearest vectors
        guard let network = clusterNetwork else {
            // Fallback to simple estimation
            return min(1.0, Float(probes) / Float(centroids.count) * 2.0)
        }
        
        // Get cluster probabilities for the query
        let probabilities = await network.forward(query)
        
        // Find top clusters to probe
        let topClusters = probabilities.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(probes)
            .map { $0.offset }
        
        // Estimate recall based on probability mass in selected clusters
        let selectedProbMass = topClusters.reduce(Float(0)) { sum, idx in
            sum + probabilities[idx]
        }
        
        // Apply a more realistic recall curve
        // Recall increases with probe count but with diminishing returns
        let adjustedRecall = 1.0 - exp(-3.0 * selectedProbMass)
        return min(1.0, adjustedRecall)
    }
    
    private func adaptClustering() async throws {
        // Re-cluster based on query patterns
        let queryVectors = Array(queryHistory)
        
        // Fine-tune cluster network on recent queries
        let (inputs, targets) = prepareTrainingData(vectors: queryVectors)
        
        let adaptConfig = NetworkTrainingConfig(
            epochs: 10, // Quick adaptation
            batchSize: 16,
            lossFunction: .crossEntropy
        )
        
        // Convert inputs and targets to MetalBuffer tuples
        var adaptTrainingData: [(input: MetalBuffer, target: MetalBuffer)] = []
        for (input, target) in zip(inputs, targets) {
            let inputBuffer = try await metalPipeline.allocateBuffer(size: input.count)
            let targetBuffer = try await metalPipeline.allocateBuffer(size: target.count)
            
            // Copy data to buffers
            let inputPtr = inputBuffer.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
            let targetPtr = targetBuffer.buffer.contents().bindMemory(to: Float.self, capacity: target.count)
            
            for i in 0..<input.count {
                inputPtr[i] = input[i]
            }
            for i in 0..<target.count {
                targetPtr[i] = target[i]
            }
            
            adaptTrainingData.append((input: inputBuffer, target: targetBuffer))
        }
        
        try await clusterNetwork?.train(
            data: adaptTrainingData,
            config: adaptConfig
        )
        
        // Update centroids
        centroids = try await updateCentroidsWithNetwork(vectors: queryVectors)
        
        // Track adaptation
        performanceMetrics.adaptationCount += 1
    }
    
    private func updateMetrics<Metadata: Codable & Sendable>(
        queries: [[Float]],
        results: [[SearchResult<Metadata>]]
    ) async {
        // Update performance tracking
        performanceMetrics.totalQueries += queries.count
        
        // Track cluster utilization and analyze result patterns
        for (query, queryResults) in zip(queries, results) {
            // Track cluster probabilities
            let probTask = Task { [weak self] in
                guard let self else { return nil }
                return try await self.getClusterProbabilities(for: query)
            }
            if let probabilities = try? await probTask.value,
               let probs = probabilities {
                performanceMetrics.updateClusterUtilization(probs)
            }
            
            // Analyze search results to improve clustering
            // Track which clusters the actual results came from
            if !queryResults.isEmpty {
                // Count of results per query helps understand search effectiveness
                performanceMetrics.averageProbes = (performanceMetrics.averageProbes * Float(performanceMetrics.totalQueries - queries.count) + Float(queryResults.count)) / Float(performanceMetrics.totalQueries)
            }
        }
    }
    
    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    private func weightedSample(probabilities: [Float]) -> Int {
        let random = Float.random(in: 0..<1)
        var cumulative: Float = 0
        
        for (idx, prob) in probabilities.enumerated() {
            cumulative += prob
            if random < cumulative {
                return idx
            }
        }
        
        return probabilities.count - 1
    }
}

// MARK: - Supporting Types

/// Configuration for neural clustering
public struct NeuralClusteringConfiguration: Sendable {
    public let dimensions: Int
    public let numberOfClusters: Int
    public let trainingEpochs: Int
    public let batchSize: Int
    public let learningRate: Float
    public let adaptiveProbing: Bool
    public let defaultProbes: Int
    public let historySize: Int
    public let adaptationThreshold: Int
    
    public init(
        dimensions: Int,
        numberOfClusters: Int,
        trainingEpochs: Int = 100,
        batchSize: Int = 32,
        learningRate: Float = 0.001,
        adaptiveProbing: Bool = true,
        defaultProbes: Int = 10,
        historySize: Int = 10000,
        adaptationThreshold: Int = 1000
    ) {
        self.dimensions = dimensions
        self.numberOfClusters = numberOfClusters
        self.trainingEpochs = trainingEpochs
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.adaptiveProbing = adaptiveProbing
        self.defaultProbes = defaultProbes
        self.historySize = historySize
        self.adaptationThreshold = adaptationThreshold
    }
}

/// Result of neural clustering
public struct NeuralClusteringResult: Sendable {
    public let centroids: [[Float]]
    public let clusterNetwork: NeuralNetwork?
    public let probeNetwork: NeuralNetwork?
    public let metrics: NeuralClusteringMetrics
}

/// Performance metrics for neural clustering
public struct NeuralClusteringMetrics: Sendable {
    var totalQueries: Int = 0
    var clusterHitRates: [Float] = []
    var averageProbes: Float = 0
    var adaptationCount: Int = 0
    
    mutating func updateClusterUtilization(_ probabilities: [Float]) {
        // Track which clusters are being used
        let maxCluster = probabilities.enumerated().max(by: { $0.1 < $1.1 })?.0 ?? 0
        
        if clusterHitRates.count <= maxCluster {
            clusterHitRates.append(contentsOf: Array(repeating: 0, count: maxCluster - clusterHitRates.count + 1))
        }
        clusterHitRates[maxCluster] += 1
    }
}

/// Ring buffer for query history
private struct RingBuffer<T> {
    private var buffer: [T?]
    private var writeIndex = 0
    private var count = 0
    private let capacity: Int
    
    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = Array(repeating: nil, count: capacity)
    }
    
    mutating func append(_ element: T) {
        buffer[writeIndex] = element
        writeIndex = (writeIndex + 1) % capacity
        count = Swift.min(count + 1, capacity)
    }
    
    var currentCount: Int { count }
    
    var elements: [T] {
        if count < capacity {
            return buffer.prefix(count).compactMap { $0 }
        } else {
            let firstPart = buffer[writeIndex...].compactMap { $0 }
            let secondPart = buffer[..<writeIndex].compactMap { $0 }
            return firstPart + secondPart
        }
    }
}

/// Neural clustering errors
public enum NeuralClusteringError: LocalizedError {
    case notTrained
    case invalidConfiguration(String)
    case trainingFailed(String)
    case metalNotAvailable
    
    public var errorDescription: String? {
        switch self {
        case .notTrained:
            return "Neural clustering model not trained"
        case .invalidConfiguration(let message):
            return "Invalid configuration: \(message)"
        case .trainingFailed(let message):
            return "Training failed: \(message)"
        case .metalNotAvailable:
            return "Metal is not available on this device"
        }
    }
}

// Extension to make RingBuffer Sequence
extension RingBuffer: Sequence {
    func makeIterator() -> AnyIterator<T> {
        return AnyIterator(elements.makeIterator())
    }
}

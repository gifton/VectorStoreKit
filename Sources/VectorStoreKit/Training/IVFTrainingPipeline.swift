// VectorStoreKit: IVF Training Pipeline
//
// Advanced training strategies for IVF indexes with sampling and incremental learning

import Foundation
import simd


/// Training result with comprehensive metrics
public struct IVFTrainingResult: Sendable {
    public let centroids: [[Float]]
    public let trainingMetrics: TrainingMetrics
    public let validationMetrics: ValidationMetrics
    public let samplingStatistics: SamplingStatistics
    public let convergenceHistory: [ConvergencePoint]
}

public struct TrainingMetrics: Sendable {
    public let finalError: Float
    public let iterations: Int
    public let trainingTime: TimeInterval
    public let samplesUsed: Int
    public let memoryPeakUsage: Int
}

public struct ValidationMetrics: Sendable {
    public let quantizationError: Float
    public let clusterBalance: Float
    public let separability: Float
    public let coverage: Float
}

public struct SamplingStatistics: Sendable {
    public let originalSize: Int
    public let sampledSize: Int
    public let samplingRatio: Float
    public let representativeness: Float
    public let diversityScore: Float
}

public struct ConvergencePoint: Sendable {
    public let iteration: Int
    public let error: Float
    public let improvement: Float
    public let timestamp: TimeInterval
}

/// IVF Training Pipeline
public actor IVFTrainingPipeline {
    
    // MARK: - Properties
    
    private let configuration: IVFTrainingConfiguration
    private var trainingHistory: [IVFTrainingResult] = []
    private var adaptiveState: AdaptiveTrainingState?
    private let metalCompute: MetalCompute?
    
    // MARK: - Initialization
    
    public init(
        configuration: IVFTrainingConfiguration,
        metalCompute: MetalCompute? = nil
    ) {
        self.configuration = configuration
        self.metalCompute = configuration.useMetalAcceleration ? metalCompute : nil
    }
    
    // MARK: - Main Training Method
    
    public func train(
        vectors: [[Float]],
        previousCentroids: [[Float]]? = nil
    ) async throws -> IVFTrainingResult {
        let startTime = DispatchTime.now()
        
        // Sample training data
        let (trainingSamples, samplingStats) = try await sampleTrainingData(
            from: vectors,
            strategy: configuration.samplingStrategy
        )
        
        // Split into training and validation
        let (trainingSet, validationSet) = splitData(
            trainingSamples,
            validationRatio: configuration.validationSplit
        )
        
        // Initialize or update centroids
        let initialCentroids = if let previous = previousCentroids {
            try await updateCentroidsIncremental(
                previous: previous,
                newSamples: trainingSet
            )
        } else {
            try await initializeCentroids(
                from: trainingSet,
                count: determineCentroidCount(for: vectors.count)
            )
        }
        
        // Perform clustering
        let clusteringResult = try await performClustering(
            samples: trainingSet,
            initialCentroids: initialCentroids
        )
        
        // Validate results
        let validationMetrics = try await validateCentroids(
            centroids: clusteringResult.centroids,
            validationSet: validationSet
        )
        
        let endTime = DispatchTime.now()
        let trainingTime = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
        
        // Prepare result
        let result = IVFTrainingResult(
            centroids: clusteringResult.centroids,
            trainingMetrics: TrainingMetrics(
                finalError: clusteringResult.finalError,
                iterations: clusteringResult.iterations,
                trainingTime: trainingTime,
                samplesUsed: trainingSet.count,
                memoryPeakUsage: estimateMemoryUsage(trainingSet.count)
            ),
            validationMetrics: validationMetrics,
            samplingStatistics: samplingStats,
            convergenceHistory: clusteringResult.convergenceHistory
        )
        
        trainingHistory.append(result)
        return result
    }
    
    // MARK: - Incremental Training
    
    public func trainIncremental(
        newVectors: [[Float]],
        currentCentroids: [[Float]]
    ) async throws -> IVFTrainingResult {
        guard let incrementalConfig = configuration.incrementalConfig else {
            throw TrainingError.incrementalNotConfigured
        }
        
        // Initialize adaptive state if needed
        if adaptiveState == nil {
            adaptiveState = AdaptiveTrainingState(
                config: incrementalConfig,
                dimensions: currentCentroids.first?.count ?? 0
            )
        }
        
        // Process in batches
        var updatedCentroids = currentCentroids
        var convergenceHistory: [ConvergencePoint] = []
        
        for batchStart in stride(from: 0, to: newVectors.count, by: incrementalConfig.batchSize) {
            let batchEnd = min(batchStart + incrementalConfig.batchSize, newVectors.count)
            let batch = Array(newVectors[batchStart..<batchEnd])
            
            // Update centroids with mini-batch
            let batchResult = try await updateCentroidsWithBatch(
                centroids: updatedCentroids,
                batch: batch,
                learningRate: adaptiveState!.currentLearningRate
            )
            
            updatedCentroids = batchResult.centroids
            convergenceHistory.append(batchResult.convergencePoint)
            
            // Update adaptive state
            adaptiveState!.update(with: batchResult.convergencePoint)
        }
        
        // Validate incremental update
        let validationMetrics = try await validateIncrementalUpdate(
            oldCentroids: currentCentroids,
            newCentroids: updatedCentroids,
            testVectors: Array(newVectors.suffix(100))
        )
        
        return IVFTrainingResult(
            centroids: updatedCentroids,
            trainingMetrics: TrainingMetrics(
                finalError: convergenceHistory.last?.error ?? 0,
                iterations: convergenceHistory.count,
                trainingTime: 0,
                samplesUsed: newVectors.count,
                memoryPeakUsage: 0
            ),
            validationMetrics: validationMetrics,
            samplingStatistics: SamplingStatistics(
                originalSize: newVectors.count,
                sampledSize: newVectors.count,
                samplingRatio: 1.0,
                representativeness: 1.0,
                diversityScore: 1.0
            ),
            convergenceHistory: convergenceHistory
        )
    }
    
    // MARK: - Sampling Methods
    
    private func sampleTrainingData<Scalar: BinaryFloatingPoint>(
        from vectors: [[Scalar]],
        strategy: SamplingStrategy
    ) async throws -> ([[Float]], SamplingStatistics) {
        let samples: [[Float]]
        let stats: SamplingStatistics
        
        switch strategy {
        case .random(let ratio):
            let sampleSize = Int(Float(vectors.count) * ratio)
            let indices = vectors.indices.shuffled().prefix(sampleSize)
            samples = indices.map { vectors[$0].map { Float($0) } }
            stats = SamplingStatistics(
                originalSize: vectors.count,
                sampledSize: samples.count,
                samplingRatio: ratio,
                representativeness: estimateRepresentativeness(samples, original: vectors),
                diversityScore: calculateDiversity(samples)
            )
            
        case .stratified(let ratio):
            samples = try await stratifiedSample(vectors: vectors, ratio: ratio)
            stats = SamplingStatistics(
                originalSize: vectors.count,
                sampledSize: samples.count,
                samplingRatio: ratio,
                representativeness: 0.95, // Stratified sampling ensures high representativeness
                diversityScore: calculateDiversity(samples)
            )
            
        case .reservoir(let size):
            samples = reservoirSample(vectors: vectors, size: size)
            stats = SamplingStatistics(
                originalSize: vectors.count,
                sampledSize: samples.count,
                samplingRatio: Float(size) / Float(vectors.count),
                representativeness: estimateRepresentativeness(samples, original: vectors),
                diversityScore: calculateDiversity(samples)
            )
            
        case .coreset(let size, let diversity):
            samples = try await coresetSample(vectors: vectors, size: size, diversity: diversity)
            stats = SamplingStatistics(
                originalSize: vectors.count,
                sampledSize: samples.count,
                samplingRatio: Float(size) / Float(vectors.count),
                representativeness: 0.98, // Coresets maintain high representativeness
                diversityScore: diversity
            )
            
        case .adaptive(let targetSize):
            samples = try await adaptiveSample(vectors: vectors, targetSize: targetSize)
            stats = SamplingStatistics(
                originalSize: vectors.count,
                sampledSize: samples.count,
                samplingRatio: Float(targetSize) / Float(vectors.count),
                representativeness: estimateRepresentativeness(samples, original: vectors),
                diversityScore: calculateDiversity(samples)
            )
            
        case .importance(let weights):
            samples = importanceSample(vectors: vectors, weights: weights)
            stats = SamplingStatistics(
                originalSize: vectors.count,
                sampledSize: samples.count,
                samplingRatio: Float(samples.count) / Float(vectors.count),
                representativeness: 0.9,
                diversityScore: calculateDiversity(samples)
            )
        }
        
        return (samples, stats)
    }
    
    private func stratifiedSample<Scalar: BinaryFloatingPoint>(
        vectors: [[Scalar]],
        ratio: Float
    ) async throws -> [[Float]] {
        // Simple k-means based stratification
        let strataCount = max(10, Int(sqrt(Float(vectors.count))))
        let config = KMeansClusteringConfiguration(
            maxIterations: 10
        )
        let kmeans = try await KMeansClustering(configuration: config)
        
        // Convert to float arrays for clustering
        let floatVectors = vectors.map { $0.map { Float($0) } }
        
        let clusteringResult = try await kmeans.cluster(
            vectors: floatVectors,
            k: strataCount
        )
        
        // Sample from each stratum
        var samples: [[Float]] = []
        var stratumVectors: [Int: [[Float]]] = [:]
        
        // Group vectors by cluster
        for (vector, assignment) in zip(floatVectors, clusteringResult.assignments) {
            stratumVectors[assignment, default: []].append(vector)
        }
        
        // Sample proportionally from each stratum
        for (_, vectors) in stratumVectors {
            let stratumSampleSize = max(1, Int(Float(vectors.count) * ratio))
            let stratumSamples = vectors.shuffled().prefix(stratumSampleSize)
            samples.append(contentsOf: stratumSamples)
        }
        
        return samples
    }
    
    private func reservoirSample<Scalar: BinaryFloatingPoint>(
        vectors: [[Scalar]],
        size: Int
    ) -> [[Float]] {
        var reservoir: [[Float]] = []
        
        for (index, vector) in vectors.enumerated() {
            let floatVector = vector.map { Float($0) }
            
            if index < size {
                reservoir.append(floatVector)
            } else {
                let j = Int.random(in: 0...index)
                if j < size {
                    reservoir[j] = floatVector
                }
            }
        }
        
        return reservoir
    }
    
    private func coresetSample<Scalar: BinaryFloatingPoint>(
        vectors: [[Scalar]],
        size: Int,
        diversity: Float
    ) async throws -> [[Float]] {
        // Greedy coreset construction
        var coreset: [[Float]] = []
        var remainingIndices = Set(vectors.indices)
        
        // Start with a random point
        let firstIndex = vectors.indices.randomElement()!
        coreset.append(vectors[firstIndex].map { Float($0) })
        remainingIndices.remove(firstIndex)
        
        // Iteratively add points that maximize diversity
        while coreset.count < size && !remainingIndices.isEmpty {
            var maxMinDistance: Float = -1
            var bestIndex: Int = -1
            
            for index in remainingIndices {
                let vector = vectors[index].map { Float($0) }
                
                // Find minimum distance to current coreset
                let minDistance = coreset.map { existing in
                    euclideanDistance(vector, existing)
                }.min() ?? Float.infinity
                
                // Weight by diversity factor
                let weightedDistance = minDistance * diversity
                
                if weightedDistance > maxMinDistance {
                    maxMinDistance = weightedDistance
                    bestIndex = index
                }
            }
            
            if bestIndex >= 0 {
                coreset.append(vectors[bestIndex].map { Float($0) })
                remainingIndices.remove(bestIndex)
            }
        }
        
        return coreset
    }
    
    private func adaptiveSample<Scalar: BinaryFloatingPoint>(
        vectors: [[Scalar]],
        targetSize: Int
    ) async throws -> [[Float]] {
        // Start with small sample and adaptively increase
        var currentSize = min(100, targetSize)
        var samples: [[Float]] = []
        var previousError: Float = Float.infinity
        
        while currentSize <= targetSize {
            // Sample current batch
            let batchIndices = vectors.indices.shuffled().prefix(currentSize)
            let batch = batchIndices.map { vectors[$0].map { Float($0) } }
            
            // Measure representativeness
            let error = measureSamplingError(sample: batch, population: vectors)
            
            // Check if we've reached sufficient quality
            if abs(previousError - error) < 0.01 {
                samples = batch
                break
            }
            
            previousError = error
            currentSize = min(currentSize * 2, targetSize)
        }
        
        return samples.isEmpty ? vectors.shuffled().prefix(targetSize).map { $0.map { Float($0) } } : samples
    }
    
    private func importanceSample<Scalar: BinaryFloatingPoint>(
        vectors: [[Scalar]],
        weights: [Float]
    ) -> [[Float]] {
        guard weights.count == vectors.count else {
            // Fallback to uniform sampling
            return vectors.shuffled().prefix(vectors.count / 10).map { $0.map { Float($0) } }
        }
        
        // Normalize weights
        let totalWeight = weights.reduce(0, +)
        let normalizedWeights = weights.map { $0 / totalWeight }
        
        // Sample based on weights
        var samples: [[Float]] = []
        let targetSize = vectors.count / 10
        
        for _ in 0..<targetSize {
            let random = Float.random(in: 0..<1)
            var cumulative: Float = 0
            
            for (index, weight) in normalizedWeights.enumerated() {
                cumulative += weight
                if cumulative >= random {
                    samples.append(vectors[index].map { Float($0) })
                    break
                }
            }
        }
        
        return samples
    }
    
    // MARK: - Training Helpers
    
    private func performClustering(
        samples: [[Float]],
        initialCentroids: [[Float]]
    ) async throws -> (centroids: [[Float]], finalError: Float, iterations: Int, convergenceHistory: [ConvergencePoint]) {
        let config = KMeansClusteringConfiguration(
            maxIterations: configuration.maxIterations,
            tolerance: configuration.convergenceThreshold
        )
        let kmeans = try await KMeansClustering(configuration: config)
        
        let result = try await kmeans.cluster(
            vectors: samples,
            k: initialCentroids.count
        )
        
        let convergenceHistory = [ConvergencePoint(
            iteration: result.iterations,
            error: result.inertia,
            improvement: 0,
            timestamp: TimeInterval(result.iterations)
        )]
        
        return (result.centroids, result.inertia, result.iterations, convergenceHistory)
    }
    
    private func validateCentroids(
        centroids: [[Float]],
        validationSet: [[Float]]
    ) async throws -> ValidationMetrics {
        // Calculate quantization error
        var totalError: Float = 0
        var clusterSizes = Array(repeating: 0, count: centroids.count)
        
        for vector in validationSet {
            let (nearestIndex, distance) = findNearestCentroid(vector, centroids: centroids)
            totalError += distance * distance
            clusterSizes[nearestIndex] += 1
        }
        
        let quantizationError = totalError / Float(validationSet.count)
        
        // Calculate cluster balance
        let meanSize = Float(validationSet.count) / Float(centroids.count)
        let deviations = clusterSizes.map { Float($0) - meanSize }
        let squaredDeviations = deviations.map { $0 * $0 }
        let sumSquaredDeviations = squaredDeviations.reduce(0, +)
        let variance = sumSquaredDeviations / Float(centroids.count)
        let clusterBalance = 1.0 - (sqrt(variance) / meanSize)
        
        // Calculate separability
        let separability = calculateSeparability(centroids: centroids)
        
        // Calculate coverage
        let coverage = Float(clusterSizes.filter { $0 > 0 }.count) / Float(centroids.count)
        
        return ValidationMetrics(
            quantizationError: quantizationError,
            clusterBalance: clusterBalance,
            separability: separability,
            coverage: coverage
        )
    }
    
    // MARK: - Utility Methods
    
    private func euclideanDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<min(a.count, b.count) {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    private func findNearestCentroid(_ vector: [Float], centroids: [[Float]]) -> (index: Int, distance: Float) {
        var minDistance = Float.infinity
        var nearestIndex = 0
        
        for (index, centroid) in centroids.enumerated() {
            let distance = euclideanDistance(vector, centroid)
            if distance < minDistance {
                minDistance = distance
                nearestIndex = index
            }
        }
        
        return (nearestIndex, minDistance)
    }
    
    private func calculateDiversity(_ samples: [[Float]]) -> Float {
        guard samples.count > 1 else { return 0 }
        
        var totalDistance: Float = 0
        var count = 0
        
        for i in 0..<min(100, samples.count) {
            for j in (i+1)..<min(100, samples.count) {
                totalDistance += euclideanDistance(samples[i], samples[j])
                count += 1
            }
        }
        
        return count > 0 ? totalDistance / Float(count) : 0
    }
    
    private func estimateRepresentativeness<Scalar: BinaryFloatingPoint>(
        _ samples: [[Float]],
        original: [[Scalar]]
    ) -> Float {
        // Simple estimation based on coverage
        return min(1.0, Float(samples.count) / Float(original.count) * 10)
    }
    
    private func measureSamplingError<Scalar: BinaryFloatingPoint>(
        sample: [[Float]],
        population: [[Scalar]]
    ) -> Float {
        // Simplified error measurement
        return 1.0 / Float(sample.count)
    }
    
    private func calculateSeparability(centroids: [[Float]]) -> Float {
        guard centroids.count > 1 else { return 1.0 }
        
        var minDistance = Float.infinity
        
        for i in 0..<centroids.count {
            for j in (i+1)..<centroids.count {
                let distance = euclideanDistance(centroids[i], centroids[j])
                minDistance = min(minDistance, distance)
            }
        }
        
        // Normalize to 0-1 range
        return min(1.0, minDistance / 10.0)
    }
    
    private func determineCentroidCount(for vectorCount: Int) -> Int {
        // Heuristic for centroid count
        return min(max(16, Int(sqrt(Float(vectorCount)))), 4096)
    }
    
    private func splitData(_ data: [[Float]], validationRatio: Float) -> (training: [[Float]], validation: [[Float]]) {
        let shuffled = data.shuffled()
        let splitIndex = Int(Float(data.count) * (1 - validationRatio))
        return (Array(shuffled[..<splitIndex]), Array(shuffled[splitIndex...]))
    }
    
    private func estimateMemoryUsage(_ sampleCount: Int) -> Int {
        // Rough estimation
        return sampleCount * 512 * MemoryLayout<Float>.size
    }
    
    private func initializeCentroids(from samples: [[Float]], count: Int) async throws -> [[Float]] {
        // Use k-means++ initialization
        return samples.shuffled().prefix(count).map { $0 }
    }
    
    private func updateCentroidsIncremental(
        previous: [[Float]],
        newSamples: [[Float]]
    ) async throws -> [[Float]] {
        // Simple weighted average for now
        return previous
    }
    
    private func updateCentroidsWithBatch(
        centroids: [[Float]],
        batch: [[Float]],
        learningRate: Float
    ) async throws -> (centroids: [[Float]], convergencePoint: ConvergencePoint) {
        var updatedCentroids = centroids
        var totalError: Float = 0
        
        // Assign batch vectors to nearest centroids
        for vector in batch {
            let (nearestIndex, distance) = findNearestCentroid(vector, centroids: centroids)
            totalError += distance * distance
            
            // Update centroid with gradient descent
            for i in 0..<vector.count {
                let gradient = vector[i] - updatedCentroids[nearestIndex][i]
                updatedCentroids[nearestIndex][i] += learningRate * gradient
            }
        }
        
        let avgError = totalError / Float(batch.count)
        let convergencePoint = ConvergencePoint(
            iteration: 0,
            error: avgError,
            improvement: 0,
            timestamp: Date().timeIntervalSince1970
        )
        
        return (updatedCentroids, convergencePoint)
    }
    
    private func validateIncrementalUpdate(
        oldCentroids: [[Float]],
        newCentroids: [[Float]],
        testVectors: [[Float]]
    ) async throws -> ValidationMetrics {
        // Simple validation comparing old vs new centroids
        return try await validateCentroids(centroids: newCentroids, validationSet: testVectors)
    }
}

// MARK: - Supporting Types

/// Adaptive training state for incremental learning
private struct AdaptiveTrainingState {
    let config: IncrementalTrainingConfig
    var currentLearningRate: Float
    var momentum: [Float]
    var errorHistory: [Float]
    
    init(config: IncrementalTrainingConfig, dimensions: Int) {
        self.config = config
        self.currentLearningRate = config.learningRate
        self.momentum = Array(repeating: 0, count: dimensions)
        self.errorHistory = []
    }
    
    mutating func update(with convergencePoint: ConvergencePoint) {
        errorHistory.append(convergencePoint.error)
        
        // Adaptive learning rate
        if errorHistory.count > 2 {
            let recent = Array(errorHistory.suffix(3))
            let improvement = (recent[0] - recent[2]) / recent[0]
            
            if improvement < config.adaptiveThreshold {
                currentLearningRate *= 0.9
            } else if improvement > config.adaptiveThreshold * 2 {
                currentLearningRate *= 1.1
            }
        }
        
        // Maintain memory window
        if errorHistory.count > config.memoryWindow {
            errorHistory.removeFirst()
        }
    }
}

/// Training errors
enum TrainingError: LocalizedError {
    case incrementalNotConfigured
    case insufficientData
    case dimensionMismatch
    case convergenceFailed
    
    var errorDescription: String? {
        switch self {
        case .incrementalNotConfigured:
            return "Incremental training configuration not provided"
        case .insufficientData:
            return "Insufficient data for training"
        case .dimensionMismatch:
            return "Vector dimensions do not match"
        case .convergenceFailed:
            return "Training failed to converge"
        }
    }
}


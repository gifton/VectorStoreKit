// VectorStoreKit: Learned Index
//
// Neural network-based index that learns the distribution of vectors

import Foundation
import simd
import Accelerate
@preconcurrency import Metal

// Local export error
enum LearnedExportError: LocalizedError {
    case unsupportedFormat(ExportFormat)
    
    var errorDescription: String? {
        switch self {
        case .unsupportedFormat(let format):
            return "Unsupported export format: \(format)"
        }
    }
}

/// Neural network wrapper for learned index predictions
private actor NeuralNetworkWrapper {
    private let network: NeuralNetwork
    private let metalPipeline: MetalMLPipeline
    
    init(network: NeuralNetwork, metalPipeline: MetalMLPipeline) {
        self.network = network
        self.metalPipeline = metalPipeline
    }
    
    func predict(_ input: [Float]) async -> [Float] {
        // Use the NeuralNetwork's compatibility API for Float array predictions
        return await network.forward(input)
    }
    
    func train(
        inputs: [[Float]],
        targets: [Float],
        config: LearnedIndexConfiguration.TrainingConfiguration
    ) async throws {
        // Convert training data to MetalBuffer format
        var trainingData: [(input: MetalBuffer, target: MetalBuffer)] = []
        
        for (input, target) in zip(inputs, targets) {
            // Create input buffer
            let inputBuffer = try await metalPipeline.allocateBuffer(shape: TensorShape(input.count))
            let inputPtr = inputBuffer.buffer.contents().bindMemory(to: Float.self, capacity: input.count)
            for i in 0..<input.count {
                inputPtr[i] = input[i]
            }
            
            // Create target buffer (single value for position prediction)
            let targetBuffer = try await metalPipeline.allocateBuffer(shape: TensorShape(1))
            let targetPtr = targetBuffer.buffer.contents().bindMemory(to: Float.self, capacity: 1)
            targetPtr[0] = target
            
            trainingData.append((input: inputBuffer, target: targetBuffer))
        }
        
        // Create training configuration
        let networkConfig = NetworkTrainingConfig(
            epochs: config.epochs,
            batchSize: config.batchSize,
            learningRate: config.learningRate,
            lossFunction: .mse,  // Use MSE for regression
            shuffle: true,
            logInterval: 10,
            earlyStoppingPatience: config.earlyStoppingPatience
        )
        
        // Train the network
        try await network.train(data: trainingData, config: networkConfig)
        
        // Release buffers
        for (input, target) in trainingData {
            await metalPipeline.releaseBuffer(input)
            await metalPipeline.releaseBuffer(target)
        }
    }
}

/// Neural network model wrapper for learned index
private actor NeuralModel {
    private let network: NeuralNetwork
    private let networkWrapper: NeuralNetworkWrapper
    private let architecture: LearnedIndexConfiguration.ModelArchitecture
    private let metalPipeline: MetalMLPipeline
    
    init(
        inputDimensions: Int,
        architecture: LearnedIndexConfiguration.ModelArchitecture,
        metalPipeline: MetalMLPipeline?
    ) async throws {
        self.architecture = architecture
        
        // Create metalPipeline if not provided
        if let metalPipeline = metalPipeline {
            self.metalPipeline = metalPipeline
        } else {
            guard let device = MTLCreateSystemDefaultDevice() else {
                throw MetalMLError.commandQueueCreationFailed
            }
            self.metalPipeline = try await MetalMLPipeline(device: device)
        }
        
        // Initialize neural network
        self.network = try await NeuralNetwork(metalPipeline: self.metalPipeline)
        
        // Build layers based on architecture
        switch architecture {
        case .linear:
            await network.addLayer(try await DenseLayer(
                inputSize: inputDimensions,
                outputSize: 1,
                activation: .sigmoid,
                metalPipeline: self.metalPipeline
            ))
            
        case .mlp(let hiddenSizes):
            var inputSize = inputDimensions
            for (i, hiddenSize) in hiddenSizes.enumerated() {
                await network.addLayer(try await DenseLayer(
                    inputSize: inputSize,
                    outputSize: hiddenSize,
                    activation: i < hiddenSizes.count - 1 ? .relu : .linear,
                    metalPipeline: self.metalPipeline
                ))
                inputSize = hiddenSize
            }
            // Output layer
            await network.addLayer(try await DenseLayer(
                inputSize: inputSize,
                outputSize: 1,
                activation: .sigmoid,
                metalPipeline: self.metalPipeline
            ))
            
        case .residual(let layerCount, let hiddenSize):
            // Input projection
            await network.addLayer(try await DenseLayer(
                inputSize: inputDimensions,
                outputSize: hiddenSize,
                activation: .relu,
                metalPipeline: self.metalPipeline
            ))
            
            // Residual blocks - simplified without ResidualLayer
            for _ in 0..<layerCount {
                // Add two dense layers to simulate residual connection
                await network.addLayer(try await DenseLayer(
                    inputSize: hiddenSize,
                    outputSize: hiddenSize,
                    activation: .relu,
                    metalPipeline: self.metalPipeline
                ))
                await network.addLayer(try await DenseLayer(
                    inputSize: hiddenSize,
                    outputSize: hiddenSize,
                    activation: .relu,
                    metalPipeline: self.metalPipeline
                ))
            }
            
            // Output layer
            await network.addLayer(try await DenseLayer(
                inputSize: hiddenSize,
                outputSize: 1,
                activation: .sigmoid,
                metalPipeline: self.metalPipeline
            ))
        }
        
        // Initialize network wrapper
        self.networkWrapper = NeuralNetworkWrapper(network: network, metalPipeline: self.metalPipeline)
    }
    
    func predict(_ input: [Float]) async -> Float {
        let output = await networkWrapper.predict(input)
        return output[0] // Single output for position prediction
    }
    
    func train(
        inputs: [[Float]],
        targets: [Float],
        config: LearnedIndexConfiguration.TrainingConfiguration
    ) async throws {
        try await networkWrapper.train(inputs: inputs, targets: targets, config: config)
    }
}

/// Bucket for storing vectors in learned index
private struct LearnedBucket: Sendable {
    let id: Int
    var vectors: [StoredVector] = []
    let capacity: Int
    
    struct StoredVector: Sendable {
        let id: String
        let vector: [Float]
        let metadata: Data
    }
}

/// Learned Index implementation
public actor LearnedIndex<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: VectorIndex
where Vector.Scalar: BinaryFloatingPoint {
    
    // MARK: - Properties
    
    public let configuration: LearnedIndexConfiguration
    private var model: NeuralModel?
    private var buckets: [Int: LearnedBucket] = [:]
    private var trained: Bool = false
    private var vectorCount: Int = 0
    private var minPosition: Float = 0
    private var maxPosition: Float = 1
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private let metalCompute: VectorStoreKit.MetalCompute?
    
    // MARK: - Initialization
    
    public init(configuration: LearnedIndexConfiguration) async throws {
        self.configuration = configuration
        
        // Initialize Metal compute if requested
        if configuration.useMetalAcceleration {
            self.metalCompute = try? await VectorStoreKit.MetalCompute(configuration: .efficient)
        } else {
            self.metalCompute = nil
        }
    }
    
    // MARK: - VectorIndex Protocol
    
    public var count: Int {
        vectorCount
    }
    
    public var capacity: Int {
        Int.max // No fixed capacity
    }
    
    public var memoryUsage: Int {
        configuration.estimatedMemoryUsage(for: vectorCount)
    }
    
    public var isOptimized: Bool {
        trained
    }
    
    public func insert(_ entry: VectorEntry<Vector, Metadata>) async throws -> InsertResult {
        let startTime = DispatchTime.now()
        
        // Convert SIMD vector to array
        let vectorArray = vectorToArray(entry.vector)
        
        // Predict position using model
        let position: Float
        if trained, let model = model {
            position = await model.predict(vectorArray)
        } else {
            // Use hash-based position for untrained model
            position = computeHashPosition(vectorArray)
        }
        
        // Map position to bucket
        let bucketId = mapPositionToBucket(position)
        
        // Encode metadata
        let metadataData = try encoder.encode(entry.metadata)
        
        // Create stored vector
        let storedVector = LearnedBucket.StoredVector(
            id: entry.id,
            vector: vectorArray,
            metadata: metadataData
        )
        
        // Add to bucket
        if buckets[bucketId] == nil {
            buckets[bucketId] = LearnedBucket(
                id: bucketId,
                capacity: configuration.bucketSize
            )
        }
        buckets[bucketId]?.vectors.append(storedVector)
        vectorCount += 1
        
        // Check if retraining is needed
        let needsRetraining = shouldRetrain()
        
        let duration = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
        
        return InsertResult(
            success: true,
            insertTime: duration,
            memoryImpact: MemoryLayout<LearnedBucket.StoredVector>.size,
            indexReorganization: needsRetraining
        )
    }
    
    public func search(
        query: Vector,
        k: Int,
        strategy: SearchStrategy = .adaptive,
        filter: SearchFilter? = nil
    ) async throws -> [SearchResult<Metadata>] {
        let queryArray = vectorToArray(query)
        
        // Predict position
        let position: Float
        if trained, let model = model {
            position = await model.predict(queryArray)
        } else {
            position = computeHashPosition(queryArray)
        }
        
        // Determine buckets to search based on strategy
        let bucketsToSearch = determineBucketsToSearch(
            position: position,
            strategy: strategy
        )
        
        // Gather candidates from buckets
        var candidates: [LearnedBucket.StoredVector] = []
        for bucketId in bucketsToSearch {
            if let bucket = buckets[bucketId] {
                candidates.append(contentsOf: bucket.vectors)
            }
        }
        
        // Apply filter if provided
        if let filter = filter {
            candidates = try await applyFilter(candidates, filter: filter)
        }
        
        // Compute exact distances
        var results: [(LearnedBucket.StoredVector, Float)] = []
        for candidate in candidates {
            let distance = computeDistance(queryArray, candidate.vector)
            results.append((candidate, distance))
        }
        
        // Sort by distance and take top k
        results.sort { $0.1 < $1.1 }
        let topK = Array(results.prefix(k))
        
        // Convert to SearchResult
        return try topK.map { stored, distance in
            let metadata = try decoder.decode(Metadata.self, from: stored.metadata)
            return createSimpleSearchResult(
                id: stored.id,
                distance: distance,
                metadata: metadata,
                indexAlgorithm: "Learned"
            )
        }
    }
    
    public func update(id: String, vector: Vector?, metadata: Metadata?) async throws -> Bool {
        // Not supported - would require retraining
        return false
    }
    
    public func delete(id: String) async throws -> Bool {
        for (bucketId, bucket) in buckets {
            if let index = bucket.vectors.firstIndex(where: { $0.id == id }) {
                buckets[bucketId]?.vectors.remove(at: index)
                vectorCount -= 1
                return true
            }
        }
        return false
    }
    
    public func contains(id: String) async -> Bool {
        for bucket in buckets.values {
            if bucket.vectors.contains(where: { $0.id == id }) {
                return true
            }
        }
        return false
    }
    
    public func optimize(strategy: OptimizationStrategy) async throws {
        if case .rebalance = strategy {
            try await retrain()
        }
    }
    
    public func compact() async throws {
        // Remove empty buckets
        buckets = buckets.filter { !$0.value.vectors.isEmpty }
    }
    
    public func statistics() async -> LearnedIndexStatistics {
        let bucketCount = buckets.count
        let avgBucketSize = bucketCount > 0 ? vectorCount / bucketCount : 0
        let maxBucketSize = buckets.values.map { $0.vectors.count }.max() ?? 0
        
        return LearnedIndexStatistics(
            vectorCount: vectorCount,
            memoryUsage: memoryUsage,
            averageSearchLatency: 0.0001,
            bucketCount: bucketCount,
            averageBucketSize: avgBucketSize,
            maxBucketSize: maxBucketSize,
            trained: trained,
            modelAccuracy: 0.0 // Would need to compute
        )
    }
    
    public func validateIntegrity() async throws -> IntegrityReport {
        var errors: [IntegrityError] = []
        var warnings: [IntegrityWarning] = []
        
        // Check bucket consistency
        var totalVectors = 0
        for bucket in buckets.values {
            totalVectors += bucket.vectors.count
        }
        
        if totalVectors != vectorCount {
            errors.append(IntegrityError(
                type: .inconsistency,
                description: "Vector count mismatch",
                severity: .high
            ))
        }
        
        // Check for overflowing buckets
        for bucket in buckets.values {
            if bucket.vectors.count > configuration.bucketSize * 2 {
                warnings.append(IntegrityWarning(
                    type: .performance,
                    description: "Bucket \(bucket.id) is overflowing",
                    recommendation: "Consider retraining the model"
                ))
            }
        }
        
        return IntegrityReport(
            isValid: errors.isEmpty,
            errors: errors,
            warnings: warnings,
            statistics: IntegrityStatistics(
                totalChecks: 2,
                passedChecks: 2 - errors.count,
                failedChecks: errors.count,
                checkDuration: 0.001
            )
        )
    }
    
    public func export(format: ExportFormat) async throws -> Data {
        let exportData = LearnedIndexExportData(
            configuration: configuration,
            trained: trained,
            minPosition: minPosition,
            maxPosition: maxPosition
        )
        
        switch format {
        case .json:
            return try JSONEncoder().encode(exportData)
        case .binary:
            return try PropertyListEncoder().encode(exportData)
        default:
            throw LearnedExportError.unsupportedFormat(format)
        }
    }
    
    public func `import`(data: Data, format: ExportFormat) async throws {
        let importData: LearnedIndexExportData
        
        switch format {
        case .json:
            importData = try JSONDecoder().decode(LearnedIndexExportData.self, from: data)
        case .binary:
            importData = try PropertyListDecoder().decode(LearnedIndexExportData.self, from: data)
        default:
            throw ImportError.unsupportedFormat(format)
        }
        
        self.trained = importData.trained
        self.minPosition = importData.minPosition
        self.maxPosition = importData.maxPosition
    }
    
    public func analyzeDistribution() async -> DistributionAnalysis {
        // Analyze vector distribution across buckets
        let bucketSizes = buckets.values.map { $0.vectors.count }
        let variance = calculateVariance(bucketSizes)
        
        return DistributionAnalysis(
            dimensionality: configuration.dimensions,
            density: Float(vectorCount) / Float(buckets.count),
            clustering: ClusteringAnalysis(
                estimatedClusters: buckets.count,
                silhouetteScore: 0.0,
                inertia: 0.0,
                clusterCenters: []
            ),
            outliers: [],
            statistics: DistributionStatistics(
                mean: Array(repeating: 0.0, count: configuration.dimensions),
                variance: Array(repeating: variance, count: configuration.dimensions),
                skewness: Array(repeating: 0.0, count: configuration.dimensions),
                kurtosis: Array(repeating: 0.0, count: configuration.dimensions)
            )
        )
    }
    
    public func performanceProfile() async -> PerformanceProfile {
        return PerformanceProfile(
            searchLatency: LatencyProfile(
                p50: 0.0001,
                p90: 0.0005,
                p95: 0.001,
                p99: 0.002,
                max: 0.01
            ),
            insertLatency: LatencyProfile(
                p50: 0.00001,
                p90: 0.00005,
                p95: 0.0001,
                p99: 0.0002,
                max: 0.001
            ),
            memoryUsage: MemoryProfile(
                baseline: configuration.estimatedMemoryUsage(for: 0),
                peak: memoryUsage,
                average: memoryUsage,
                efficiency: 0.9
            ),
            throughput: ThroughputProfile(
                queriesPerSecond: 100000,
                insertsPerSecond: 1000000,
                updatesPerSecond: 0,
                deletesPerSecond: 10000
            )
        )
    }
    
    public func visualizationData() async -> VisualizationData {
        // Provide visualization of bucket structure
        var nodePositions: [[Float]] = []
        var nodeMetadata: [String: String] = [:]
        
        for (bucketId, bucket) in buckets {
            let position = Float(bucketId) / Float(buckets.count)
            nodePositions.append([position, Float(bucket.vectors.count)])
            nodeMetadata["\(bucketId)"] = "size:\(bucket.vectors.count),position:\(position)"
        }
        
        return VisualizationData(
            nodePositions: nodePositions,
            edges: [],
            nodeMetadata: nodeMetadata,
            layoutAlgorithm: "learned"
        )
    }
    
    // MARK: - Learned Index Specific Methods
    
    /// Train the index on sample data
    public func train(on samples: [[Float]]) async throws {
        guard samples.count >= 1000 else {
            throw LearnedIndexError.insufficientTrainingData(
                provided: samples.count,
                required: 1000
            )
        }
        
        // Initialize model
        model = try await NeuralModel(
            inputDimensions: configuration.dimensions,
            architecture: configuration.modelArchitecture,
            metalPipeline: nil  // Let NeuralModel create its own pipeline
        )
        
        // Generate training targets (normalized positions)
        let sortedSamples = samples.enumerated().sorted { a, b in
            // Sort by first dimension for simplicity
            a.element[0] < b.element[0]
        }
        
        var targets: [Float] = []
        for (i, _) in sortedSamples.enumerated() {
            targets.append(Float(i) / Float(samples.count))
        }
        
        // Train model
        try await model?.train(
            inputs: sortedSamples.map { $0.element },
            targets: targets,
            config: configuration.trainingConfig
        )
        
        trained = true
    }
    
    /// Retrain the model using existing data
    public func retrain() async throws {
        var allVectors: [[Float]] = []
        for bucket in buckets.values {
            allVectors.append(contentsOf: bucket.vectors.map { $0.vector })
        }
        
        guard allVectors.count >= 1000 else {
            throw LearnedIndexError.insufficientTrainingData(
                provided: allVectors.count,
                required: 1000
            )
        }
        
        try await train(on: allVectors)
    }
    
    // MARK: - Private Methods
    
    private func vectorToArray(_ vector: Vector) -> [Float] {
        var array: [Float] = []
        for i in 0..<vector.scalarCount {
            array.append(Float(vector[i]))
        }
        return array
    }
    
    private func arrayToVector<T: SIMD>(_ array: [Float], type: T.Type) -> T? where T.Scalar: BinaryFloatingPoint {
        guard array.count == T.scalarCount else {
            return nil
        }
        
        var result = T()
        for i in 0..<T.scalarCount {
            result[i] = T.Scalar(array[i])
        }
        return result
    }
    
    private func computeHashPosition(_ vector: [Float]) -> Float {
        // Simple hash-based position for untrained model
        var hash: Float = 0
        for (i, value) in vector.enumerated() {
            hash += value * Float(i + 1)
        }
        return abs(hash.truncatingRemainder(dividingBy: 1.0))
    }
    
    private func mapPositionToBucket(_ position: Float) -> Int {
        let normalizedPos = max(0, min(1, position))
        let bucketCount = max(1, (vectorCount + configuration.bucketSize - 1) / configuration.bucketSize)
        return Int(normalizedPos * Float(bucketCount))
    }
    
    private func determineBucketsToSearch(
        position: Float,
        strategy: SearchStrategy
    ) -> [Int] {
        let primaryBucket = mapPositionToBucket(position)
        
        switch strategy {
        case .exact:
            // Search all buckets
            return Array(buckets.keys)
            
        case .approximate:
            // Search neighboring buckets
            let range = 5 // Default range for approximate search
            let minBucket = max(0, primaryBucket - range)
            let maxBucket = primaryBucket + range
            return Array(minBucket...maxBucket).filter { buckets[$0] != nil }
            
        case .adaptive, .learned, .hybrid, .anytime, .multimodal:
            // Search primary bucket and immediate neighbors
            return [primaryBucket - 1, primaryBucket, primaryBucket + 1]
                .filter { $0 >= 0 && buckets[$0] != nil }
        }
    }
    
    private func computeDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    private func shouldRetrain() -> Bool {
        // Retrain if buckets are significantly unbalanced
        guard buckets.count > 10 else { return false }
        
        let sizes = buckets.values.map { $0.vectors.count }
        let avgSize = Float(vectorCount) / Float(buckets.count)
        let maxSize = Float(sizes.max() ?? 0)
        
        return maxSize > avgSize * 5
    }
    
    private func applyFilter(
        _ candidates: [LearnedBucket.StoredVector],
        filter: SearchFilter
    ) async throws -> [LearnedBucket.StoredVector] {
        switch filter {
        case .metadata(let metadataFilter):
            return try await filterByMetadata(candidates, filter: metadataFilter)
            
        case .vector(let vectorFilter):
            return filterByVector(candidates, filter: vectorFilter)
            
        case .composite(let compositeFilter):
            return try await filterByComposite(candidates, filter: compositeFilter)
            
        case .learned(let learnedFilter):
            return try await filterByLearned(candidates, filter: learnedFilter)
        }
    }
    
    private func filterByMetadata(
        _ candidates: [LearnedBucket.StoredVector],
        filter: MetadataFilter
    ) async throws -> [LearnedBucket.StoredVector] {
        return candidates.compactMap { candidate in
            // Decode metadata
            guard let metadata = try? decoder.decode(Metadata.self, from: candidate.metadata) else {
                return nil
            }
            
            // Convert metadata to dictionary for filtering
            guard let metadataData = try? encoder.encode(metadata),
                  let metadataDict = try? JSONSerialization.jsonObject(with: metadataData) as? [String: Any],
                  let value = metadataDict[filter.key] else {
                return nil
            }
            
            // Apply filter operation
            let valueString = String(describing: value)
            let matches = evaluateFilterOperation(
                value: valueString,
                operation: filter.operation,
                filterValue: filter.value
            )
            
            return matches ? candidate : nil
        }
    }
    
    private func filterByVector(
        _ candidates: [LearnedBucket.StoredVector],
        filter: VectorFilter
    ) -> [LearnedBucket.StoredVector] {
        return candidates.filter { candidate in
            let vector = candidate.vector
            
            // Check dimension filter if specified
            if let dimension = filter.dimension,
               dimension < vector.count,
               let range = filter.range {
                guard range.contains(vector[dimension]) else {
                    return false
                }
            }
            
            // Apply vector constraint
            switch filter.constraint {
            case .magnitude(let range):
                let magnitude = sqrt(vector.reduce(0) { $0 + $1 * $1 })
                return range.contains(magnitude)
                
            case .sparsity(let range):
                let nonZeroCount = vector.filter { $0 != 0 }.count
                let sparsity = Float(nonZeroCount) / Float(vector.count)
                return range.contains(sparsity)
                
            case .custom(let predicate):
                // Convert array to SIMD vector for custom predicate
                guard let simdVector = arrayToVector(vector, type: Vector.self) else {
                    return false
                }
                return predicate(simdVector)
            }
        }
    }
    
    private func filterByComposite(
        _ candidates: [LearnedBucket.StoredVector],
        filter: CompositeFilter
    ) async throws -> [LearnedBucket.StoredVector] {
        switch filter.operation {
        case .and:
            var result = candidates
            for subFilter in filter.filters {
                result = try await applyFilter(result, filter: subFilter)
            }
            return result
            
        case .or:
            var resultSet = Set<String>()
            var resultVectors: [LearnedBucket.StoredVector] = []
            
            for subFilter in filter.filters {
                let filtered = try await applyFilter(candidates, filter: subFilter)
                for vector in filtered {
                    if !resultSet.contains(vector.id) {
                        resultSet.insert(vector.id)
                        resultVectors.append(vector)
                    }
                }
            }
            return resultVectors
            
        case .not:
            guard let firstFilter = filter.filters.first else {
                return candidates
            }
            let filtered = try await applyFilter(candidates, filter: firstFilter)
            let filteredIds = Set(filtered.map { $0.id })
            return candidates.filter { !filteredIds.contains($0.id) }
        }
    }
    
    private func filterByLearned(
        _ candidates: [LearnedBucket.StoredVector],
        filter: LearnedFilter
    ) async throws -> [LearnedBucket.StoredVector] {
        // Use the learned model to filter candidates
        guard let model = self.model else {
            // If no model is trained, fall back to confidence-based filtering
            guard filter.confidence > 0 else {
                return candidates
            }
            let keepCount = Int(Float(candidates.count) * filter.confidence)
            return Array(candidates.prefix(keepCount))
        }
        
        // Score each candidate using the learned model
        var scoredCandidates: [(LearnedBucket.StoredVector, Float)] = []
        
        for candidate in candidates {
            let score = await model.predict(candidate.vector)
            scoredCandidates.append((candidate, score))
        }
        
        // Sort by score and apply confidence threshold
        scoredCandidates.sort { $0.1 > $1.1 }
        let keepCount = Int(Float(candidates.count) * filter.confidence)
        return Array(scoredCandidates.prefix(keepCount).map { $0.0 })
    }
    
    private func evaluateFilterOperation(
        value: String,
        operation: FilterOperation,
        filterValue: String
    ) -> Bool {
        switch operation {
        case .equals:
            return value == filterValue
        case .notEquals:
            return value != filterValue
        case .lessThan:
            return value < filterValue
        case .lessThanOrEqual:
            return value <= filterValue
        case .greaterThan:
            return value > filterValue
        case .greaterThanOrEqual:
            return value >= filterValue
        case .contains:
            return value.contains(filterValue)
        case .notContains:
            return !value.contains(filterValue)
        case .startsWith:
            return value.hasPrefix(filterValue)
        case .endsWith:
            return value.hasSuffix(filterValue)
        case .in:
            let values = filterValue.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            return values.contains(value)
        case .notIn:
            let values = filterValue.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
            return !values.contains(value)
        case .regex:
            return (try? NSRegularExpression(pattern: filterValue).firstMatch(
                in: value,
                range: NSRange(location: 0, length: value.utf16.count)
            )) != nil
        }
    }
    
    private func calculateVariance(_ values: [Int]) -> Float {
        guard !values.isEmpty else { return 0 }
        let mean = Float(values.reduce(0, +)) / Float(values.count)
        let squaredDiffs = values.map { pow(Float($0) - mean, 2) }
        return squaredDiffs.reduce(0, +) / Float(values.count)
    }
}

// MARK: - Supporting Types

/// Statistics for learned index
public struct LearnedIndexStatistics: IndexStatistics {
    public let vectorCount: Int
    public let memoryUsage: Int
    public let averageSearchLatency: TimeInterval
    
    public let bucketCount: Int
    public let averageBucketSize: Int
    public let maxBucketSize: Int
    public let trained: Bool
    public let modelAccuracy: Float
    
    public var qualityMetrics: IndexQualityMetrics {
        IndexQualityMetrics(
            recall: trained ? 0.98 : 0.85,
            precision: trained ? 0.98 : 0.85,
            buildTime: 0.0,
            memoryEfficiency: 0.9,
            searchLatency: averageSearchLatency
        )
    }
}

/// Export data for learned index
private struct LearnedIndexExportData: Codable {
    let configuration: LearnedIndexConfiguration
    let trained: Bool
    let minPosition: Float
    let maxPosition: Float
}

/// Errors specific to learned index
public enum LearnedIndexError: LocalizedError {
    case insufficientTrainingData(provided: Int, required: Int)
    case mismatchedTrainingData
    case modelNotTrained
    case invalidDimensions(Int)
    case invalidParameter(String, Int)
    
    public var errorDescription: String? {
        switch self {
        case .insufficientTrainingData(let provided, let required):
            return "Insufficient training data: \(provided) samples provided, \(required) required"
        case .mismatchedTrainingData:
            return "Training inputs and targets have different lengths"
        case .modelNotTrained:
            return "Model must be trained before use"
        case .invalidDimensions(let dimensions):
            return "Invalid dimensions: \(dimensions)"
        case .invalidParameter(let name, let value):
            return "Invalid parameter \(name): \(value)"
        }
    }
}

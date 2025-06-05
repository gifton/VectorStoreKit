// VectorStoreKit: IVF (Inverted File) Index
//
// Efficient approximate nearest neighbor search using clustering

import Foundation
import simd

/// Stored vector in inverted list
public struct StoredVector: Sendable {
    let id: String
    let vector: [Float]
    let metadata: Data
}

/// IVF Index implementation
public actor IVFIndex<Vector: SIMD & Sendable, Metadata: Codable & Sendable>: VectorIndex
where Vector.Scalar: BinaryFloatingPoint {
    
    // MARK: - Properties
    
    public let configuration: IVFConfiguration
    private var centroids: [[Float]] = []
    private var invertedLists: [Int: [StoredVector]] = [:]
    private var trained: Bool = false
    private var vectorCount: Int = 0
    private let clustering: KMeansClustering
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    
    // MARK: - Initialization
    
    public init(configuration: IVFConfiguration) async throws {
        self.configuration = configuration
        self.clustering = try await KMeansClustering(
            configuration: configuration.clusteringConfig
        )
        
        // Initialize empty inverted lists
        for i in 0..<configuration.numberOfCentroids {
            invertedLists[i] = []
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
        
        // Ensure index is trained
        if !trained {
            throw IVFError.notTrained
        }
        
        // Convert SIMD vector to array
        let vectorArray = vectorToArray(entry.vector)
        
        // Find nearest centroid
        let centroidIndex = try await findNearestCentroid(vectorArray)
        
        // Encode metadata
        let metadataData = try encoder.encode(entry.metadata)
        
        // Create stored vector
        let storedVector = StoredVector(
            id: entry.id,
            vector: vectorArray,
            metadata: metadataData
        )
        
        // Add to inverted list
        invertedLists[centroidIndex]?.append(storedVector)
        vectorCount += 1
        
        let duration = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
        
        return InsertResult(
            success: true,
            insertTime: duration,
            memoryImpact: MemoryLayout<StoredVector>.size,
            indexReorganization: false
        )
    }
    
    public func search(
        query: Vector,
        k: Int,
        strategy: SearchStrategy = .adaptive,
        filter: SearchFilter? = nil
    ) async throws -> [SearchResult<Metadata>] {
        guard trained else {
            throw IVFError.notTrained
        }
        
        let queryArray = vectorToArray(query)
        
        // Determine number of probes based on strategy
        let probes = determineProbeCount(strategy: strategy)
        
        // Find nearest centroids to probe
        let nearestCentroids = try await findNearestCentroids(
            queryArray,
            count: probes
        )
        
        // Gather candidates from inverted lists
        var candidates: [StoredVector] = []
        for centroidIndex in nearestCentroids {
            if let list = invertedLists[centroidIndex] {
                candidates.append(contentsOf: list)
            }
        }
        
        // Apply filter if provided
        if let filter = filter {
            candidates = try await applyFilter(candidates, filter: filter)
        }
        
        // Compute exact distances and sort
        var results: [(StoredVector, Float)] = []
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
                indexAlgorithm: "IVF"
            )
        }
    }
    
    public func update(id: String, vector: Vector?, metadata: Metadata?) async throws -> Bool {
        // IVF doesn't support efficient updates
        // Would need to remove and re-insert
        return false
    }
    
    public func delete(id: String) async throws -> Bool {
        // Search through all inverted lists
        for (centroidIndex, list) in invertedLists {
            if let index = list.firstIndex(where: { $0.id == id }) {
                invertedLists[centroidIndex]?.remove(at: index)
                vectorCount -= 1
                return true
            }
        }
        return false
    }
    
    public func contains(id: String) async -> Bool {
        for list in invertedLists.values {
            if list.contains(where: { $0.id == id }) {
                return true
            }
        }
        return false
    }
    
    public func optimize(strategy: OptimizationStrategy) async throws {
        // Re-cluster if needed
        if case .rebalance = strategy {
            try await retrain()
        }
    }
    
    public func compact() async throws {
        // Remove empty lists
        invertedLists = invertedLists.filter { !$0.value.isEmpty }
    }
    
    public func statistics() async -> IVFStatistics {
        let listSizes = invertedLists.values.map { $0.count }
        let avgListSize = listSizes.isEmpty ? 0 : listSizes.reduce(0, +) / listSizes.count
        let maxListSize = listSizes.max() ?? 0
        let minListSize = listSizes.min() ?? 0
        
        return IVFStatistics(
            vectorCount: vectorCount,
            memoryUsage: memoryUsage,
            averageSearchLatency: 0.001,
            numberOfCentroids: configuration.numberOfCentroids,
            averageListSize: avgListSize,
            maxListSize: maxListSize,
            minListSize: minListSize,
            trained: trained
        )
    }
    
    public func validateIntegrity() async throws -> IntegrityReport {
        var errors: [IntegrityError] = []
        var warnings: [IntegrityWarning] = []
        
        // Check if trained
        if !trained {
            errors.append(IntegrityError(
                type: .invalid,
                description: "Index not trained",
                severity: .high
            ))
        }
        
        // Check list balance
        let stats = await statistics()
        if stats.maxListSize > stats.averageListSize * 10 {
            warnings.append(IntegrityWarning(
                type: .performance,
                description: "Unbalanced list sizes detected",
                recommendation: "Consider retraining with more centroids"
            ))
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
        let exportData = IVFExportData(
            configuration: configuration,
            centroids: centroids,
            trained: trained
        )
        
        switch format {
        case .json:
            return try JSONEncoder().encode(exportData)
        case .binary:
            return try PropertyListEncoder().encode(exportData)
        default:
            throw ExportError.unsupportedFormat(format)
        }
    }
    
    public func `import`(data: Data, format: ExportFormat) async throws {
        let importData: IVFExportData
        
        switch format {
        case .json:
            importData = try JSONDecoder().decode(IVFExportData.self, from: data)
        case .binary:
            importData = try PropertyListDecoder().decode(IVFExportData.self, from: data)
        default:
            throw ImportError.unsupportedFormat(format)
        }
        
        self.centroids = importData.centroids
        self.trained = importData.trained
    }
    
    public func analyzeDistribution() async -> DistributionAnalysis {
        let listSizes = invertedLists.values.map { $0.count }
        let variance = calculateVariance(listSizes)
        
        return DistributionAnalysis(
            dimensionality: configuration.dimensions,
            density: Float(vectorCount) / Float(configuration.numberOfCentroids),
            clustering: ClusteringAnalysis(
                estimatedClusters: configuration.numberOfCentroids,
                silhouetteScore: 0.0, // Would need to compute
                inertia: 0.0, // Would need to compute
                clusterCenters: centroids
            ),
            outliers: [], // Would need to identify
            statistics: DistributionStatistics(
                mean: Array(repeating: 0.0, count: configuration.dimensions),
                variance: Array(repeating: variance, count: configuration.dimensions),
                skewness: Array(repeating: 0.0, count: configuration.dimensions),
                kurtosis: Array(repeating: 0.0, count: configuration.dimensions)
            )
        )
    }
    
    public func performanceProfile() async -> PerformanceProfile {
        // Return estimated performance characteristics
        return PerformanceProfile(
            searchLatency: LatencyProfile(
                p50: 0.001,
                p90: 0.005,
                p95: 0.01,
                p99: 0.02,
                max: 0.1
            ),
            insertLatency: LatencyProfile(
                p50: 0.0001,
                p90: 0.0005,
                p95: 0.001,
                p99: 0.002,
                max: 0.01
            ),
            memoryUsage: MemoryProfile(
                baseline: configuration.estimatedMemoryUsage(for: 0),
                peak: memoryUsage,
                average: memoryUsage,
                efficiency: 0.8
            ),
            throughput: ThroughputProfile(
                queriesPerSecond: 10000,
                insertsPerSecond: 100000,
                updatesPerSecond: 0,
                deletesPerSecond: 1000
            )
        )
    }
    
    public func visualizationData() async -> VisualizationData {
        return VisualizationData(
            nodePositions: centroids,
            edges: [],
            nodeMetadata: [:],
            layoutAlgorithm: "k-means"
        )
    }
    
    // MARK: - IVF-specific Methods
    
    /// Train the index on a sample of vectors
    public func train(on samples: [[Float]]) async throws {
        guard samples.count >= configuration.numberOfCentroids else {
            throw IVFError.insufficientTrainingData(
                provided: samples.count,
                required: configuration.numberOfCentroids
            )
        }
        
        // Perform K-means clustering
        let result = try await clustering.cluster(
            vectors: samples,
            k: configuration.numberOfCentroids
        )
        
        self.centroids = result.centroids
        self.trained = true
        
        // Clear any existing data
        for i in 0..<configuration.numberOfCentroids {
            invertedLists[i] = []
        }
        vectorCount = 0
    }
    
    /// Retrain the index using existing vectors
    public func retrain() async throws {
        // Gather all vectors
        var allVectors: [[Float]] = []
        for list in invertedLists.values {
            allVectors.append(contentsOf: list.map { $0.vector })
        }
        
        guard allVectors.count >= configuration.numberOfCentroids else {
            throw IVFError.insufficientTrainingData(
                provided: allVectors.count,
                required: configuration.numberOfCentroids
            )
        }
        
        // Sample if too many vectors
        let samples = allVectors.count > configuration.trainingSampleSize
            ? Array(allVectors.shuffled().prefix(configuration.trainingSampleSize))
            : allVectors
        
        // Retrain
        try await train(on: samples)
        
        // Re-assign all vectors
        for list in invertedLists.values {
            for stored in list {
                let centroidIndex = try await findNearestCentroid(stored.vector)
                invertedLists[centroidIndex]?.append(stored)
            }
        }
    }
    
    // MARK: - Public Methods for Search Engines
    
    /// Get centroids for search operations
    public func getCentroids() async -> [[Float]] {
        return centroids
    }
    
    /// Get vectors from a specific inverted list
    internal func getVectorsFromList(_ centroidIndex: Int) async -> [StoredVector] {
        return invertedLists[centroidIndex] ?? []
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
    
    private func findNearestCentroid(_ vector: [Float]) async throws -> Int {
        var minDistance: Float = .infinity
        var nearestIndex = 0
        
        for (index, centroid) in centroids.enumerated() {
            let distance = computeDistance(vector, centroid)
            if distance < minDistance {
                minDistance = distance
                nearestIndex = index
            }
        }
        
        return nearestIndex
    }
    
    private func findNearestCentroids(_ vector: [Float], count: Int) async throws -> [Int] {
        var distances: [(Int, Float)] = []
        
        for (index, centroid) in centroids.enumerated() {
            let distance = computeDistance(vector, centroid)
            distances.append((index, distance))
        }
        
        // Sort by distance and take top count
        distances.sort { $0.1 < $1.1 }
        return Array(distances.prefix(count).map { $0.0 })
    }
    
    private func computeDistance(_ a: [Float], _ b: [Float]) -> Float {
        var sum: Float = 0
        for i in 0..<a.count {
            let diff = a[i] - b[i]
            sum += diff * diff
        }
        return sqrt(sum)
    }
    
    private func determineProbeCount(strategy: SearchStrategy) -> Int {
        switch strategy {
        case .exact:
            return configuration.numberOfCentroids
        case .approximate:
            return configuration.numberOfProbes
        case .adaptive, .learned, .hybrid, .anytime, .multimodal:
            return configuration.numberOfProbes
        }
    }
    
    private func applyFilter(
        _ candidates: [StoredVector],
        filter: SearchFilter
    ) async throws -> [StoredVector] {
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
        _ candidates: [StoredVector],
        filter: MetadataFilter
    ) async throws -> [StoredVector] {
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
        _ candidates: [StoredVector],
        filter: VectorFilter
    ) -> [StoredVector] {
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
        _ candidates: [StoredVector],
        filter: CompositeFilter
    ) async throws -> [StoredVector] {
        switch filter.operation {
        case .and:
            var result = candidates
            for subFilter in filter.filters {
                result = try await applyFilter(result, filter: subFilter)
            }
            return result
            
        case .or:
            var resultSet = Set<String>()
            var resultVectors: [StoredVector] = []
            
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
        _ candidates: [StoredVector],
        filter: LearnedFilter
    ) async throws -> [StoredVector] {
        // For now, implement a simple confidence-based filtering
        // In a real implementation, this would use the learned model
        guard filter.confidence > 0 else {
            return candidates
        }
        
        // Apply confidence threshold - keep top percentage based on confidence
        let keepCount = Int(Float(candidates.count) * filter.confidence)
        return Array(candidates.prefix(keepCount))
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

/// IVF-specific statistics
public struct IVFStatistics: IndexStatistics {
    public let vectorCount: Int
    public let memoryUsage: Int
    public let averageSearchLatency: TimeInterval
    
    public let numberOfCentroids: Int
    public let averageListSize: Int
    public let maxListSize: Int
    public let minListSize: Int
    public let trained: Bool
    
    public var qualityMetrics: IndexQualityMetrics {
        IndexQualityMetrics(
            recall: 0.95, // Estimated
            precision: 0.95,
            buildTime: 0.0,
            memoryEfficiency: 0.8,
            searchLatency: averageSearchLatency
        )
    }
}

/// Export data for IVF index
private struct IVFExportData: Codable {
    let configuration: IVFConfiguration
    let centroids: [[Float]]
    let trained: Bool
}

/// IVF-specific errors
public enum IVFError: LocalizedError {
    case notTrained
    case insufficientTrainingData(provided: Int, required: Int)
    case invalidDimensions(Int)
    case invalidParameter(String, Int)
    
    public var errorDescription: String? {
        switch self {
        case .notTrained:
            return "IVF index must be trained before use"
        case .insufficientTrainingData(let provided, let required):
            return "Insufficient training data: \(provided) vectors provided, \(required) required"
        case .invalidDimensions(let dimensions):
            return "Invalid dimensions: \(dimensions)"
        case .invalidParameter(let name, let value):
            return "Invalid parameter \(name): \(value)"
        }
    }
}

/// Errors for import/export
enum ImportError: LocalizedError {
    case decodingFailed
    case unsupportedFormat(ExportFormat)
}

enum ExportError: LocalizedError {
    case unsupportedFormat(ExportFormat)
}


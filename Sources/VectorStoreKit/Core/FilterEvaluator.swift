// VectorStoreKit: Production Filter Evaluation System
//
// High-performance filter evaluation with model caching and hot-reload

import Foundation
import simd
@preconcurrency import Metal
import os.log

/// Production-ready filter evaluator with model management
public actor FilterEvaluator {
    // MARK: - Properties
    
    private let modelRegistry: FilterModelRegistry
    private let metalPipeline: MetalMLPipeline
    private let logger = Logger(subsystem: "VectorStoreKit", category: "FilterEvaluator")
    
    // Performance monitoring
    private var metrics = FilterEvaluationMetrics()
    
    // Batch processing configuration
    private let batchSize: Int
    private let maxConcurrentEvaluations: Int
    
    // MARK: - Initialization
    
    public init(
        metalPipeline: MetalMLPipeline? = nil,
        cacheConfiguration: FilterModelRegistry.CacheConfiguration = .default,
        batchSize: Int = 64,
        maxConcurrentEvaluations: Int = 4
    ) async throws {
        // Initialize Metal pipeline if not provided
        let pipeline: MetalMLPipeline
        if let provided = metalPipeline {
            pipeline = provided
        } else {
            guard let device = MTLCreateSystemDefaultDevice() else {
                throw VectorStoreError.metalDeviceUnavailable()
            }
            pipeline = try MetalMLPipeline(device: device)
        }
        
        self.metalPipeline = pipeline
        self.modelRegistry = try await FilterModelRegistry(
            metalPipeline: pipeline,
            cacheConfiguration: cacheConfiguration
        )
        self.batchSize = batchSize
        self.maxConcurrentEvaluations = maxConcurrentEvaluations
        
        logger.info("FilterEvaluator initialized with batch size: \(batchSize)")
    }
    
    // MARK: - Static Instance
    
    /// Shared instance for backward compatibility
    public static let shared: FilterEvaluator = {
        let semaphore = DispatchSemaphore(value: 0)
        var instance: FilterEvaluator?
        var error: Error?
        
        Task {
            do {
                instance = try await FilterEvaluator()
            } catch let e {
                error = e
            }
            semaphore.signal()
        }
        
        semaphore.wait()
        
        if let instance = instance {
            return instance
        } else if let error = error {
            fatalError("Failed to initialize shared FilterEvaluator: \(error)")
        } else {
            fatalError("Failed to initialize shared FilterEvaluator: Unknown error")
        }
    }()
    
    // MARK: - Main Filter Evaluation
    
    /// Evaluates a search filter against a stored vector
    public func evaluateFilter(
        _ filter: SearchFilter,
        vector: StoredVector,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) async throws -> Bool {
        switch filter {
        case .metadata(let metadataFilter):
            return await evaluateMetadataFilter(metadataFilter, vector: vector, decoder: decoder, encoder: encoder)
            
        case .vector(let vectorFilter):
            return await evaluateVectorFilter(vectorFilter, vector: vector.vector)
            
        case .composite(let compositeFilter):
            return try await evaluateCompositeFilter(compositeFilter, vector: vector, decoder: decoder, encoder: encoder)
            
        case .learned(let learnedFilter):
            return await evaluateLearnedFilter(learnedFilter, vector: vector)
        }
    }
    
    // MARK: - Metadata Filtering
    
    /// Evaluates a metadata filter against a stored vector
    public func evaluateMetadataFilter(
        _ filter: MetadataFilter,
        vector: StoredVector,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) -> Bool {
        // Try to decode as dictionary
        guard let metadataDict = try? JSONSerialization.jsonObject(with: vector.metadata) as? [String: Any],
              let value = metadataDict[filter.key] else {
            return false
        }
        
        // Convert value to string for comparison
        let valueString = String(describing: value)
        
        return evaluateFilterOperation(
            value: valueString,
            operation: filter.operation,
            filterValue: filter.value
        )
    }
    
    // MARK: - Vector Filtering
    
    /// Lookup table for SIMD type evaluators
    private let simdEvaluatorLookup: [Int: (([Float], @Sendable (any SIMD) -> Bool) -> Bool)] = [
        2: { vector, predicate in evaluateCustomPredicate(vector, predicate: predicate, type: SIMD2<Float>.self) },
        3: { vector, predicate in evaluateCustomPredicate(vector, predicate: predicate, type: SIMD3<Float>.self) },
        4: { vector, predicate in evaluateCustomPredicate(vector, predicate: predicate, type: SIMD4<Float>.self) },
        8: { vector, predicate in evaluateCustomPredicate(vector, predicate: predicate, type: SIMD8<Float>.self) },
        16: { vector, predicate in evaluateCustomPredicate(vector, predicate: predicate, type: SIMD16<Float>.self) },
        32: { vector, predicate in evaluateCustomPredicate(vector, predicate: predicate, type: SIMD32<Float>.self) },
        64: { vector, predicate in evaluateCustomPredicate(vector, predicate: predicate, type: SIMD64<Float>.self) }
    ]
    
    /// Evaluates a vector filter against a vector array
    public func evaluateVectorFilter(
        _ filter: VectorFilter,
        vector: [Float]
    ) -> Bool {
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
            // Use lookup table for O(1) access to the appropriate evaluator
            if let evaluator = simdEvaluatorLookup[vector.count] {
                return evaluator(vector, predicate)
            } else {
                // For non-standard SIMD sizes, we can't evaluate the predicate
                // In production, you might want to handle this differently
                return false
            }
        }
    }
    
    // MARK: - SIMD Conversion Helpers
    
    /// Evaluates a custom predicate with a specific SIMD type
    private func evaluateCustomPredicate<T: SIMD>(
        _ vector: [Float],
        predicate: @Sendable (any SIMD) -> Bool,
        type: T.Type
    ) -> Bool where T.Scalar == Float {
        guard let simdVector = arrayToSIMD(vector, type: type) else {
            return false
        }
        return predicate(simdVector)
    }
    
    /// Converts a Float array to a specific SIMD type
    private func arrayToSIMD<T: SIMD>(_ array: [Float], type: T.Type) -> T? where T.Scalar == Float {
        guard array.count == T.scalarCount else {
            return nil
        }
        
        var result = T()
        for i in 0..<T.scalarCount {
            result[i] = array[i]
        }
        return result
    }
    
    // MARK: - Composite Filtering
    
    /// Evaluates a composite filter against a stored vector
    public func evaluateCompositeFilter(
        _ filter: CompositeFilter,
        vector: StoredVector,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) async throws -> Bool {
        switch filter.operation {
        case .and:
            // All filters must match
            for subFilter in filter.filters {
                if !(try await evaluateFilter(subFilter, vector: vector, decoder: decoder, encoder: encoder)) {
                    return false
                }
            }
            return true
            
        case .or:
            // At least one filter must match
            for subFilter in filter.filters {
                if try await evaluateFilter(subFilter, vector: vector, decoder: decoder, encoder: encoder) {
                    return true
                }
            }
            return false
            
        case .not:
            // First filter must not match
            guard let firstFilter = filter.filters.first else {
                return true
            }
            return !(try await evaluateFilter(firstFilter, vector: vector, decoder: decoder, encoder: encoder))
        }
    }
    
    // MARK: - Learned Filtering
    
    /// Evaluates a learned filter using production-ready model management
    public func evaluateLearnedFilter(_ filter: LearnedFilter, vector: StoredVector) async -> Bool {
        let startTime = Date()
        
        do {
            // Parse version from model identifier if present (format: "modelId:version")
            let components = filter.modelIdentifier.split(separator: ":")
            let modelId = String(components[0])
            let version: FilterModelRegistry.Version?
            
            if components.count > 1 {
                let versionComponents = components[1].split(separator: ".")
                if versionComponents.count == 3,
                   let major = Int(versionComponents[0]),
                   let minor = Int(versionComponents[1]),
                   let patch = Int(versionComponents[2]) {
                    version = FilterModelRegistry.Version(
                        major: major,
                        minor: minor,
                        patch: patch
                    )
                } else {
                    version = nil
                }
            } else {
                version = nil
            }
            
            // Evaluate using model registry
            let result = try await modelRegistry.evaluate(
                modelId: modelId,
                vector: vector.vector,
                confidence: filter.confidence
            )
            
            // Update metrics
            let evaluationTime = Date().timeIntervalSince(startTime)
            metrics.learnedFilterEvaluations += 1
            metrics.totalLearnedFilterTime += evaluationTime
            
            return result
            
        } catch {
            logger.error("Learned filter evaluation failed: \(error)")
            metrics.learnedFilterErrors += 1
            
            // Fallback to heuristic evaluation if model fails
            return await evaluateHeuristicFilter(filter, vector: vector)
        }
    }
    
    /// Heuristic fallback for learned filter evaluation
    private func evaluateHeuristicFilter(_ filter: LearnedFilter, vector: StoredVector) async -> Bool {
        // Extract parameters for simple heuristic evaluation
        let threshold = Float(filter.parameters["threshold"] ?? "0.5") ?? 0.5
        let dimension = Int(filter.parameters["dimension"] ?? "0") ?? 0
        let operation = filter.parameters["operation"] ?? "magnitude"
        
        switch operation {
        case "magnitude":
            // Check if vector magnitude meets threshold
            let magnitude = sqrt(vector.vector.reduce(0) { $0 + $1 * $1 })
            return magnitude >= threshold * filter.confidence
            
        case "dimension":
            // Check specific dimension against threshold
            guard dimension < vector.vector.count else { return false }
            return vector.vector[dimension] >= threshold * filter.confidence
            
        case "sparsity":
            // Check vector sparsity
            let nonZeroCount = vector.vector.filter { abs($0) > 1e-6 }.count
            let sparsity = Float(nonZeroCount) / Float(vector.vector.count)
            return sparsity >= threshold * filter.confidence
            
        default:
            // Default: use confidence-based threshold
            return filter.confidence > 0.5
        }
    }
    
    // MARK: - Filter Operation Evaluation
    
    /// Evaluates a filter operation between two string values
    public func evaluateFilterOperation(
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
    
    // MARK: - Batch Filtering
    
    /// Filters vectors with optimized batch processing
    public func filterVectors(
        _ vectors: [StoredVector],
        filter: SearchFilter,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) async throws -> [StoredVector] {
        let startTime = Date()
        
        // Process in batches for better performance
        var filtered: [StoredVector] = []
        filtered.reserveCapacity(vectors.count / 2) // Assume 50% pass rate
        
        // Create task groups for concurrent evaluation
        await withTaskGroup(of: (StoredVector, Bool)?.self) { group in
            // Limit concurrent tasks
            var activeTaskCount = 0
            var vectorIndex = 0
            
            while vectorIndex < vectors.count || activeTaskCount > 0 {
                // Add new tasks up to limit
                while activeTaskCount < maxConcurrentEvaluations && vectorIndex < vectors.count {
                    let vector = vectors[vectorIndex]
                    group.addTask { [weak self] in
                        guard let self = self else { return nil }
                        do {
                            let result = try await self.evaluateFilter(
                                filter,
                                vector: vector,
                                decoder: decoder,
                                encoder: encoder
                            )
                            return (vector, result)
                        } catch {
                            // Log error but continue processing
                            await self.logger.error("Filter evaluation failed: \(error)")
                            return nil
                        }
                    }
                    activeTaskCount += 1
                    vectorIndex += 1
                }
                
                // Collect results
                if let result = await group.next() {
                    activeTaskCount -= 1
                    if let (vector, passed) = result, passed {
                        filtered.append(vector)
                    }
                }
            }
        }
        
        // Update metrics
        let filterTime = Date().timeIntervalSince(startTime)
        metrics.batchFilterOperations += 1
        metrics.totalBatchFilterTime += filterTime
        metrics.vectorsProcessed += vectors.count
        metrics.vectorsPassed += filtered.count
        
        logger.info("Filtered \(vectors.count) vectors in \(String(format: "%.2f", filterTime))s, \(filtered.count) passed")
        
        return filtered
    }
    
    // MARK: - Model Management
    
    /// Preload a model for faster evaluation
    public func preloadModel(_ modelId: String, version: FilterModelRegistry.Version? = nil) async throws {
        _ = try await modelRegistry.loadModel(modelId, version: version)
        logger.info("Preloaded model: \(modelId)")
    }
    
    /// Enable hot reload for a model
    public func enableModelHotReload(
        modelId: String,
        path: URL,
        callback: @escaping (String, FilterModelRegistry.Version) -> Void = { _, _ in }
    ) async {
        await modelRegistry.enableHotReload(for: modelId, path: path, callback: callback)
    }
    
    /// Update model to a new version
    public func updateModel(
        _ modelId: String,
        to version: FilterModelRegistry.Version
    ) async throws {
        try await modelRegistry.updateModel(modelId, to: version)
    }
    
    // MARK: - Metrics
    
    /// Get evaluation metrics
    public func getMetrics() async -> FilterEvaluationMetrics {
        var metrics = self.metrics
        
        // Add model registry metrics
        let modelMetrics = await modelRegistry.getMetrics()
        metrics.modelCacheHits = modelMetrics.cacheHits
        metrics.modelCacheMisses = modelMetrics.cacheMisses
        metrics.modelsLoaded = modelMetrics.modelsLoaded
        metrics.averageModelLoadTimeMs = modelMetrics.averageLoadTimeMs
        
        // Calculate derived metrics
        if metrics.learnedFilterEvaluations > 0 {
            metrics.averageLearnedFilterTimeMs = (metrics.totalLearnedFilterTime / Double(metrics.learnedFilterEvaluations)) * 1000
        }
        
        if metrics.batchFilterOperations > 0 {
            metrics.averageBatchFilterTimeMs = (metrics.totalBatchFilterTime / Double(metrics.batchFilterOperations)) * 1000
        }
        
        if metrics.vectorsProcessed > 0 {
            metrics.filterPassRate = Float(metrics.vectorsPassed) / Float(metrics.vectorsProcessed)
        }
        
        return metrics
    }
    
    /// Reset metrics
    public func resetMetrics() async {
        metrics = FilterEvaluationMetrics()
        await modelRegistry.resetMetrics()
    }
}

// MARK: - Supporting Types

/// Performance metrics for filter evaluation
public struct FilterEvaluationMetrics: Sendable {
    public var learnedFilterEvaluations: Int = 0
    public var learnedFilterErrors: Int = 0
    public var totalLearnedFilterTime: TimeInterval = 0
    public var averageLearnedFilterTimeMs: Double = 0
    
    public var batchFilterOperations: Int = 0
    public var totalBatchFilterTime: TimeInterval = 0
    public var averageBatchFilterTimeMs: Double = 0
    
    public var vectorsProcessed: Int = 0
    public var vectorsPassed: Int = 0
    public var filterPassRate: Float = 0
    
    public var modelCacheHits: Int = 0
    public var modelCacheMisses: Int = 0
    public var modelsLoaded: Int = 0
    public var averageModelLoadTimeMs: Double = 0
}

// MARK: - Static Convenience Methods

/// Static methods for backward compatibility
public extension FilterEvaluator {
    /// Evaluates a search filter against a stored vector (static convenience)
    static func evaluateFilter(
        _ filter: SearchFilter,
        vector: StoredVector,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) async throws -> Bool {
        return try await shared.evaluateFilter(
            filter,
            vector: vector,
            decoder: decoder,
            encoder: encoder
        )
    }
    
    /// Evaluates a metadata filter (static convenience)
    static func evaluateMetadataFilter(
        _ filter: MetadataFilter,
        vector: StoredVector,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) async -> Bool {
        return await shared.evaluateMetadataFilter(
            filter,
            vector: vector,
            decoder: decoder,
            encoder: encoder
        )
    }
    
    /// Evaluates a vector filter (static convenience)
    static func evaluateVectorFilter(
        _ filter: VectorFilter,
        vector: [Float]
    ) async -> Bool {
        return await shared.evaluateVectorFilter(filter, vector: vector)
    }
    
    /// Evaluates a composite filter (static convenience)
    static func evaluateCompositeFilter(
        _ filter: CompositeFilter,
        vector: StoredVector,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) async throws -> Bool {
        return try await shared.evaluateCompositeFilter(
            filter,
            vector: vector,
            decoder: decoder,
            encoder: encoder
        )
    }
    
    /// Evaluates a learned filter (static convenience)
    static func evaluateLearnedFilter(
        _ filter: LearnedFilter,
        vector: StoredVector
    ) async -> Bool {
        return await shared.evaluateLearnedFilter(filter, vector: vector)
    }
    
    /// Filters vectors (static convenience)
    static func filterVectors(
        _ vectors: [StoredVector],
        filter: SearchFilter,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) async throws -> [StoredVector] {
        return try await shared.filterVectors(
            vectors,
            filter: filter,
            decoder: decoder,
            encoder: encoder
        )
    }
}

// MARK: - StoredVector Extension

extension StoredVector {
    /// Convenience method to check if this vector matches a filter
    public func matchesFilter(
        _ filter: SearchFilter,
        decoder: JSONDecoder = JSONDecoder(),
        encoder: JSONEncoder = JSONEncoder()
    ) async throws -> Bool {
        return try await FilterEvaluator.evaluateFilter(filter, vector: self, decoder: decoder, encoder: encoder)
    }
}

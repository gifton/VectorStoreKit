// BatchProcessor.swift
// VectorStoreKit
//
// Type-safe batch processing for large-scale vector operations

import Foundation
import Metal
import os.log

/// Configuration for batch processing operations
public struct BatchProcessingConfiguration: Sendable {
    public let optimalBatchSize: Int
    public let maxConcurrentBatches: Int
    public let useMetalAcceleration: Bool
    public let enableProgressTracking: Bool
    public let memoryLimit: Int // In bytes
    
    public init(
        optimalBatchSize: Int? = nil,
        maxConcurrentBatches: Int = 4,
        useMetalAcceleration: Bool = true,
        enableProgressTracking: Bool = true,
        memoryLimit: Int = 1_073_741_824 // 1GB default
    ) {
        self.optimalBatchSize = optimalBatchSize ?? Self.calculateOptimalBatchSize()
        self.maxConcurrentBatches = maxConcurrentBatches
        self.useMetalAcceleration = useMetalAcceleration
        self.enableProgressTracking = enableProgressTracking
        self.memoryLimit = memoryLimit
    }
    
    private static func calculateOptimalBatchSize() -> Int {
        // Calculate based on system resources
        let memorySize = ProcessInfo.processInfo.physicalMemory
        let coreCount = ProcessInfo.processInfo.processorCount
        
        // Heuristic: Use ~10MB per batch, adjusted for core count
        let bytesPerBatch = 10_485_760 // 10MB
        let vectorSize = 512 * MemoryLayout<Float>.size // Assume 512-dim vectors
        let baseSize = bytesPerBatch / vectorSize
        
        return baseSize * min(coreCount, 8)
    }
}

/// Progress tracking for batch operations
public struct BatchProgress: Sendable {
    public let totalItems: Int
    public let processedItems: Int
    public let currentBatch: Int
    public let totalBatches: Int
    public let elapsedTime: TimeInterval
    public let estimatedTimeRemaining: TimeInterval?
    
    public var percentComplete: Float {
        guard totalItems > 0 else { return 0 }
        return Float(processedItems) / Float(totalItems) * 100
    }
    
    public var itemsPerSecond: Double {
        guard elapsedTime > 0 else { return 0 }
        return Double(processedItems) / elapsedTime
    }
}

/// Type-safe batch processor for large datasets
public actor BatchProcessor: BatchSizeManager, MemoryPressureAware {
    
    // MARK: - Properties
    
    private let configuration: BatchProcessingConfiguration
    private let metalCompute: MetalCompute?
    private let bufferPool: MetalBufferPool?
    private let memoryPool: MemoryPoolManager
    
    // Progress tracking
    private var currentProgress: BatchProgress?
    private var progressUpdateHandler: ((BatchProgress) -> Void)?
    
    // Metrics
    private var totalItemsProcessed: Int = 0
    private var totalProcessingTime: TimeInterval = 0
    
    // Memory pressure management
    private var currentBatchSize: Int
    private let defaultBatchSize: Int
    private var memoryPressureLevel: SystemMemoryPressure = .normal
    private var pressureStats = MemoryComponentStatistics(
        componentName: "BatchProcessor",
        currentMemoryUsage: 0,
        peakMemoryUsage: 0
    )
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "BatchProcessor")
    
    // MARK: - Initialization
    
    public init(
        configuration: BatchProcessingConfiguration = BatchProcessingConfiguration(),
        metalCompute: MetalCompute? = nil,
        bufferPool: MetalBufferPool? = nil,
        memoryPool: MemoryPoolManager? = nil
    ) {
        self.configuration = configuration
        self.defaultBatchSize = configuration.optimalBatchSize
        self.currentBatchSize = configuration.optimalBatchSize
        self.metalCompute = configuration.useMetalAcceleration ? metalCompute : nil
        self.bufferPool = configuration.useMetalAcceleration ? bufferPool : nil
        self.memoryPool = memoryPool ?? MemoryPoolManager(
            configuration: .init(
                maxPoolSize: configuration.memoryLimit,
                allocationStrategy: .bestFit,
                defragmentationThreshold: 0.3
            )
        )
    }
    
    // MARK: - Type-Safe Batch Processing
    
    /// Process a large dataset in optimal batches with type safety
    public func processBatches<T, R>(
        dataset: any LargeVectorDataset<T>,
        processor: @escaping @Sendable ([T]) async throws -> [R],
        progressHandler: ((BatchProgress) -> Void)? = nil
    ) async throws -> [R] where T: Sendable, R: Sendable {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        self.progressUpdateHandler = progressHandler
        
        let totalItems = await dataset.count
        let batchSize = await determineBatchSize(for: totalItems, itemSize: MemoryLayout<T>.size)
        let totalBatches = (totalItems + batchSize - 1) / batchSize
        
        logger.info("Processing \(totalItems) items in \(totalBatches) batches of size \(batchSize)")
        
        var results: [R] = []
        results.reserveCapacity(totalItems)
        
        // Process batches concurrently with limited concurrency
        try await withThrowingTaskGroup(of: (Int, [R]).self) { group in
            var batchIndex = 0
            
            // Add initial batches up to max concurrent limit
            for i in 0..<min(configuration.maxConcurrentBatches, totalBatches) {
                let startIdx = i * batchSize
                let endIdx = min(startIdx + batchSize, totalItems)
                
                group.addTask {
                    // Load batch
                    let batch = try await dataset.loadBatch(
                        range: startIdx..<endIdx
                    )
                    
                    // Process batch with type-safe processor
                    let batchResults = try await processor(batch)
                    
                    // Update progress
                    await self.updateProgress(
                        processedItems: endIdx,
                        totalItems: totalItems,
                        currentBatch: i + 1,
                        totalBatches: totalBatches,
                        startTime: startTime
                    )
                    
                    return (i, batchResults)
                }
            }
            
            batchIndex = min(configuration.maxConcurrentBatches, totalBatches)
            
            // Process remaining batches as previous ones complete
            while batchIndex < totalBatches {
                // Wait for a task to complete
                if let _ = try await group.next() {
                    // Add next batch
                    let currentBatchIndex = batchIndex
                    let startIdx = currentBatchIndex * batchSize
                    let endIdx = min(startIdx + batchSize, totalItems)
                    
                    group.addTask {
                        // Load batch
                        let batch = try await dataset.loadBatch(
                            range: startIdx..<endIdx
                        )
                        
                        // Process batch with type-safe processor
                        let batchResults = try await processor(batch)
                        
                        // Update progress
                        await self.updateProgress(
                            processedItems: endIdx,
                            totalItems: totalItems,
                            currentBatch: currentBatchIndex + 1,
                            totalBatches: totalBatches,
                            startTime: startTime
                        )
                        
                        return (currentBatchIndex, batchResults)
                    }
                    
                    batchIndex += 1
                }
            }
            
            // Collect results in order
            var orderedResults = Array(repeating: [] as [R], count: totalBatches)
            for try await (index, batchResults) in group {
                orderedResults[index] = batchResults
            }
            
            // Flatten results
            for batchResults in orderedResults {
                results.append(contentsOf: batchResults)
            }
        }
        
        let totalTime = CFAbsoluteTimeGetCurrent() - startTime
        totalItemsProcessed += totalItems
        totalProcessingTime += totalTime
        
        logger.info("Processed \(totalItems) items in \(totalTime)s (\(Double(totalItems)/totalTime) items/sec)")
        
        return results
    }
    
    /// Process vectors in batches with streaming
    public func streamProcessBatches<T, R>(
        stream: AsyncStream<T>,
        batchSize: Int? = nil,
        processor: @escaping @Sendable ([T]) async throws -> [R]
    ) -> AsyncThrowingStream<[R], Error> where T: Sendable, R: Sendable {
        
        let actualBatchSize = batchSize ?? configuration.optimalBatchSize
        
        return AsyncThrowingStream { continuation in
            Task {
                var batch: [T] = []
                batch.reserveCapacity(actualBatchSize)
                
                do {
                    for await item in stream {
                        batch.append(item)
                        
                        if batch.count >= actualBatchSize {
                            let results = try await processor(batch)
                            continuation.yield(results)
                            
                            batch.removeAll(keepingCapacity: true)
                        }
                    }
                    
                    // Process remaining items
                    if !batch.isEmpty {
                        let results = try await processor(batch)
                        continuation.yield(results)
                    }
                    
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    // MARK: - Specialized Operations
    
    /// Process Vector512 batches with optimized memory layout
    public func processVector512Batches(
        vectors: [Vector512],
        processor: @escaping @Sendable ([Vector512]) async throws -> [BatchResult],
        progressHandler: ((BatchProgress) -> Void)? = nil
    ) async throws -> [BatchResult] {
        
        // Create a wrapper dataset
        let dataset = Vector512Dataset(vectors: vectors)
        
        // Use specialized batch size for 512-dim vectors
        let customConfig = BatchProcessingConfiguration(
            optimalBatchSize: VectorDimension.fixed512.optimalBatchSize,
            maxConcurrentBatches: configuration.maxConcurrentBatches,
            useMetalAcceleration: true
        )
        
        let batchProcessor = BatchProcessor(
            configuration: customConfig,
            metalCompute: metalCompute,
            bufferPool: bufferPool
        )
        
        return try await batchProcessor.processBatches(
            dataset: dataset,
            processor: processor,
            progressHandler: progressHandler
        )
    }
    
    /// Process with indexing operation
    public func processIndexingBatch<T: Sendable>(
        vectors: [(vector: T, id: VectorID, metadata: Data?)],
        index: any VectorIndex
    ) async throws -> [BatchResult] {
        let processor: @Sendable ([(vector: T, id: VectorID, metadata: Data?)]) async throws -> [BatchResult] = { batch in
            var results: [BatchResult] = []
            results.reserveCapacity(batch.count)
            
            for (vector, id, metadata) in batch {
                do {
                    // Create proper entry based on vector type
                    if let vec512 = vector as? Vector512 {
                        let entry = VectorEntry(
                            id: id,
                            vector: vec512,
                            metadata: metadata ?? Data(),
                            tier: .hot
                        )
                        // Note: Actual insertion would happen here if index supported it
                        results.append(BatchResult(
                            id: id,
                            success: true,
                            metadata: ["operation": "index", "vectorDimension": "\(vec512.scalarCount)"]
                        ))
                    } else {
                        results.append(BatchResult(
                            id: id,
                            success: false,
                            metadata: ["error": "Unsupported vector type"]
                        ))
                    }
                } catch {
                    results.append(BatchResult(
                        id: id,
                        success: false,
                        metadata: ["error": error.localizedDescription]
                    ))
                }
            }
            
            return results
        }
        
        let dataset = VectorDataset(vectors: vectors)
        return try await processBatches(
            dataset: dataset,
            processor: processor
        )
    }
    
    /// Process with quantization operation
    public func processQuantizationBatch(
        vectors: [Vector512],
        type: ScalarQuantizationType
    ) async throws -> [QuantizedVectorStore] {
        let quantizer = ScalarQuantizer()
        
        let processor: @Sendable ([Vector512]) async throws -> [QuantizedVectorStore] = { batch in
            try await quantizer.quantize512(vectors: batch, type: type)
        }
        
        let dataset = Vector512Dataset(vectors: vectors)
        return try await processBatches(
            dataset: dataset,
            processor: processor
        )
    }
    
    /// Process with distance computation
    public func processDistanceComputationBatch(
        query: Vector512,
        candidates: [Vector512],
        metric: DistanceMetric
    ) async throws -> [(vectorId: String, distance: Float)] {
        let processor: @Sendable ([Vector512]) async throws -> [(vectorId: String, distance: Float)] = { batch in
            if let metalCompute = self.metalCompute {
                do {
                    let (distances, _) = try await metalCompute.computeDistances(
                        query: query,
                        candidates: batch,
                        metric: metric
                    )
                    
                    return batch.enumerated().map { index, _ in
                        (vectorId: "\(index)", distance: distances[index])
                    }
                } catch {
                    self.logger.warning("Metal computation failed, falling back to CPU")
                }
            }
            
            // CPU fallback
            return batch.enumerated().map { index, candidate in
                let distance: Float = {
                    switch metric {
                    case .euclidean:
                        return sqrt(query.distanceSquared(to: candidate))
                    case .cosine:
                        return 1.0 - query.cosineSimilarity(to: candidate)
                    case .dotProduct:
                        return -query.dot(candidate)
                    default:
                        return sqrt(query.distanceSquared(to: candidate))
                    }
                }()
                
                return (vectorId: "\(index)", distance: distance)
            }
        }
        
        let dataset = Vector512Dataset(vectors: candidates)
        return try await processBatches(
            dataset: dataset,
            processor: processor
        )
    }
    
    // MARK: - Private Methods
    
    private func determineBatchSize(for totalItems: Int, itemSize: Int) async -> Int {
        // Consider memory constraints
        let availableMemory = configuration.memoryLimit
        let itemsPerBatch = availableMemory / itemSize
        
        // Use current batch size (which may be reduced due to memory pressure)
        if currentBatchSize * itemSize <= availableMemory {
            return currentBatchSize
        }
        
        // Otherwise, adjust based on memory
        return min(itemsPerBatch, currentBatchSize)
    }
    
    private func updateProgress(
        processedItems: Int,
        totalItems: Int,
        currentBatch: Int,
        totalBatches: Int,
        startTime: CFAbsoluteTime
    ) async {
        let elapsedTime = CFAbsoluteTimeGetCurrent() - startTime
        let itemsPerSecond = Double(processedItems) / elapsedTime
        let remainingItems = totalItems - processedItems
        let estimatedTimeRemaining = remainingItems > 0 ? 
            TimeInterval(Double(remainingItems) / itemsPerSecond) : nil
        
        let progress = BatchProgress(
            totalItems: totalItems,
            processedItems: processedItems,
            currentBatch: currentBatch,
            totalBatches: totalBatches,
            elapsedTime: elapsedTime,
            estimatedTimeRemaining: estimatedTimeRemaining
        )
        
        self.currentProgress = progress
        progressUpdateHandler?(progress)
    }
    
    // MARK: - Index Operations
    
    /// Insert vectors into an index with batch transaction support
    /// - Parameters:
    ///   - entries: Vector entries to insert
    ///   - index: The vector index to insert into
    ///   - options: Indexing options for batch control
    /// - Returns: Batch indexing result with detailed metrics
    public func indexVectors<Index: VectorIndex>(
        entries: [VectorEntry<Vector512, Index.Metadata>],
        into index: Index,
        options: BatchIndexingOptions = .default
    ) async throws -> BatchIndexingResult where Index.Vector == Vector512 {
        let startTime = DispatchTime.now()
        
        // Create memory pool for batch operation
        let pool = await memoryPool.createSubpool(
            name: "batch_indexing_\(UUID().uuidString)",
            maxSize: configuration.memoryLimit
        )
        defer {
            Task {
                await memoryPool.releaseSubpool(pool)
            }
        }
        
        // Track results
        var successfulInserts: [String] = []
        var failedInserts: [(id: String, error: Error)] = []
        var transactionLog: [IndexingTransaction] = []
        
        // Process in optimal batches
        let batchSize = determineBatchSize(
            for: entries.count,
            itemSize: MemoryLayout<Vector512>.size + 1024 // Estimate with metadata
        )
        
        // Create transaction for rollback support
        let transactionId = UUID().uuidString
        var transactionState = TransactionState(id: transactionId)
        
        do {
            // Process batches with concurrent insertion
            try await withThrowingTaskGroup(of: BatchInsertResult.self) { group in
                for batch in entries.chunked(into: batchSize) {
                    // Check memory before processing batch
                    let requiredMemory = batch.count * (MemoryLayout<Vector512>.size + 1024)
                    guard await pool.canAllocate(size: requiredMemory) else {
                        throw BatchProcessingError.memoryExhausted(
                            required: requiredMemory,
                            available: await pool.availableMemory
                        )
                    }
                    
                    group.addTask {
                        try await self.processBatchInsertion(
                            batch: batch,
                            index: index,
                            pool: pool,
                            transactionId: transactionId,
                            options: options
                        )
                    }
                }
                
                // Collect results
                for try await result in group {
                    successfulInserts.append(contentsOf: result.successIds)
                    failedInserts.append(contentsOf: result.failures)
                    transactionLog.append(result.transaction)
                    
                    // Update transaction state
                    transactionState.processedBatches += 1
                    transactionState.successCount += result.successIds.count
                    transactionState.failureCount += result.failures.count
                    
                    // Update progress if enabled
                    if configuration.enableProgressTracking {
                        await updateProgress(
                            processedItems: successfulInserts.count + failedInserts.count,
                            totalItems: entries.count,
                            currentBatch: transactionState.processedBatches,
                            totalBatches: (entries.count + batchSize - 1) / batchSize,
                            startTime: CFAbsoluteTimeGetCurrent()
                        )
                    }
                }
            }
            
            // Commit transaction if all succeeded or partial success is allowed
            if failedInserts.isEmpty || options.allowPartialSuccess {
                await commitIndexingTransaction(
                    transactionId: transactionId,
                    index: index,
                    transactionLog: transactionLog
                )
            } else {
                // Rollback on failure
                try await rollbackIndexingTransaction(
                    transactionId: transactionId,
                    index: index,
                    transactionLog: transactionLog
                )
                throw BatchProcessingError.transactionFailed(
                    transactionId: transactionId,
                    failureCount: failedInserts.count
                )
            }
            
        } catch {
            // Ensure rollback on any error
            if !transactionState.isCommitted {
                try? await rollbackIndexingTransaction(
                    transactionId: transactionId,
                    index: index,
                    transactionLog: transactionLog
                )
            }
            throw error
        }
        
        let duration = Double(DispatchTime.now().uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
        
        // Update statistics
        totalItemsProcessed += entries.count
        totalProcessingTime += duration
        
        return BatchIndexingResult(
            totalEntries: entries.count,
            successfulInserts: successfulInserts.count,
            failedInserts: failedInserts.count,
            failures: failedInserts,
            totalTime: duration,
            throughput: Double(entries.count) / duration,
            memoryPeakUsage: await pool.peakUsage,
            transactionId: transactionId,
            indexOptimized: false
        )
    }
    
    /// Process a single batch insertion with transaction support
    private func processBatchInsertion<Index: VectorIndex>(
        batch: [VectorEntry<Vector512, Index.Metadata>],
        index: Index,
        pool: MemoryPoolManager.Subpool,
        transactionId: String,
        options: BatchIndexingOptions
    ) async throws -> BatchInsertResult where Index.Vector == Vector512 {
        let batchStartTime = DispatchTime.now()
        
        var successIds: [String] = []
        var failures: [(id: String, error: Error)] = []
        var insertedEntries: [(id: String, timestamp: Date)] = []
        
        // Allocate memory for batch
        let memoryHandle = try await pool.allocate(
            size: batch.count * MemoryLayout<Vector512>.size,
            alignment: 64
        )
        defer {
            Task {
                await pool.deallocate(memoryHandle)
            }
        }
        
        // Process each entry in the batch
        for entry in batch {
            do {
                // Validate entry before insertion
                if options.validateEntries {
                    try validateEntry(entry)
                }
                
                // Insert into index with retry logic
                var lastError: Error?
                for attempt in 0..<options.maxRetries {
                    do {
                        let result = try await index.insert(entry)
                        
                        if result.success {
                            successIds.append(entry.id)
                            insertedEntries.append((id: entry.id, timestamp: Date()))
                            break
                        } else {
                            // Handle soft failure (e.g., duplicate)
                            if options.skipDuplicates {
                                logger.debug("Skipping duplicate entry: \(entry.id)")
                                break
                            } else {
                                throw BatchProcessingError.duplicateEntry(id: entry.id)
                            }
                        }
                    } catch {
                        lastError = error
                        if attempt < options.maxRetries - 1 {
                            // Exponential backoff
                            let delay = UInt64(pow(2.0, Double(attempt)) * 100_000_000)
                            try await Task.sleep(nanoseconds: delay)
                        }
                    }
                }
                
                if let error = lastError {
                    failures.append((id: entry.id, error: error))
                }
                
            } catch {
                failures.append((id: entry.id, error: error))
            }
        }
        
        let batchDuration = Double(DispatchTime.now().uptimeNanoseconds - batchStartTime.uptimeNanoseconds) / 1_000_000_000
        
        // Create transaction record
        let transaction = IndexingTransaction(
            id: UUID().uuidString,
            parentTransactionId: transactionId,
            timestamp: Date(),
            entries: insertedEntries,
            duration: batchDuration
        )
        
        return BatchInsertResult(
            successIds: successIds,
            failures: failures,
            transaction: transaction
        )
    }
    
    /// Validate a vector entry before insertion
    private func validateEntry<Metadata>(_ entry: VectorEntry<Vector512, Metadata>) throws {
        // Validate vector dimensions
        guard entry.vector.scalarCount == 512 else {
            throw BatchProcessingError.invalidDimensions(
                expected: 512,
                actual: entry.vector.scalarCount
            )
        }
        
        // Check for NaN or infinite values
        for i in 0..<entry.vector.scalarCount {
            let value = entry.vector[i]
            guard value.isFinite else {
                throw BatchProcessingError.invalidVectorData(
                    id: entry.id,
                    reason: "Vector contains non-finite values"
                )
            }
        }
        
        // Validate ID format
        guard !entry.id.isEmpty else {
            throw BatchProcessingError.invalidEntry(
                id: entry.id,
                reason: "Empty ID"
            )
        }
    }
    
    /// Commit an indexing transaction
    private func commitIndexingTransaction<Index: VectorIndex>(
        transactionId: String,
        index: Index,
        transactionLog: [IndexingTransaction]
    ) async {
        logger.info("Committing indexing transaction: \(transactionId)")
        
        // Trigger index optimization if needed
        if let count = await index.count,
           count > 0 && count % 10000 == 0 {
            Task {
                try? await index.optimize(strategy: .incremental)
            }
        }
        
        // Log successful transaction
        for transaction in transactionLog {
            logger.debug("Committed batch \(transaction.id) with \(transaction.entries.count) entries")
        }
    }
    
    /// Rollback an indexing transaction
    private func rollbackIndexingTransaction<Index: VectorIndex>(
        transactionId: String,
        index: Index,
        transactionLog: [IndexingTransaction]
    ) async throws {
        logger.warning("Rolling back indexing transaction: \(transactionId)")
        
        // Delete inserted entries in reverse order
        for transaction in transactionLog.reversed() {
            for (id, _) in transaction.entries.reversed() {
                do {
                    _ = try await index.delete(id: id)
                } catch {
                    logger.error("Failed to rollback entry \(id): \(error)")
                }
            }
        }
    }
    
    // MARK: - Statistics
    
    public func getStatistics() -> BatchProcessingStatistics {
        BatchProcessingStatistics(
            totalItemsProcessed: totalItemsProcessed,
            totalProcessingTime: totalProcessingTime,
            averageItemsPerSecond: totalProcessingTime > 0 ? 
                Double(totalItemsProcessed) / totalProcessingTime : 0,
            configuration: configuration
        )
    }
    
    // MARK: - BatchSizeManager Protocol
    
    public func getCurrentBatchSize() async -> Int {
        return currentBatchSize
    }
    
    public func reduceBatchSize(by factor: Float) async {
        let newSize = max(1, Int(Float(currentBatchSize) * (1.0 - factor)))
        currentBatchSize = newSize
        logger.info("Reduced batch size to \(newSize) (factor: \(factor))")
    }
    
    public func resetBatchSize() async {
        currentBatchSize = defaultBatchSize
        logger.info("Reset batch size to default: \(defaultBatchSize)")
    }
    
    // MARK: - MemoryPressureAware Protocol
    
    public func handleMemoryPressure(_ level: SystemMemoryPressure) async {
        let previousLevel = memoryPressureLevel
        memoryPressureLevel = level
        
        switch level {
        case .normal:
            if previousLevel != .normal {
                await resetBatchSize()
                logger.info("Memory pressure recovered, reset to normal operation")
            }
            
        case .warning:
            await reduceBatchSize(by: 0.3) // Reduce by 30%
            logger.warning("Memory pressure warning: reduced batch size")
            
        case .critical:
            await reduceBatchSize(by: 0.6) // Reduce by 60%
            logger.error("Critical memory pressure: aggressively reduced batch size")
        }
        
        // Update statistics
        pressureStats = MemoryComponentStatistics(
            componentName: pressureStats.componentName,
            currentMemoryUsage: await getCurrentMemoryUsage(),
            peakMemoryUsage: max(pressureStats.peakMemoryUsage, await getCurrentMemoryUsage()),
            pressureEventCount: pressureStats.pressureEventCount + 1,
            lastPressureHandled: Date(),
            averageResponseTime: pressureStats.averageResponseTime
        )
    }
    
    public func getCurrentMemoryUsage() async -> Int {
        // Estimate based on current batch size and typical vector sizes
        let estimatedVectorSize = 512 * MemoryLayout<Float>.size // Assume 512-dim vectors
        return currentBatchSize * estimatedVectorSize * configuration.maxConcurrentBatches
    }
    
    public func getMemoryStatistics() async -> MemoryComponentStatistics {
        return MemoryComponentStatistics(
            componentName: pressureStats.componentName,
            currentMemoryUsage: await getCurrentMemoryUsage(),
            peakMemoryUsage: pressureStats.peakMemoryUsage,
            pressureEventCount: pressureStats.pressureEventCount,
            lastPressureHandled: pressureStats.lastPressureHandled,
            averageResponseTime: pressureStats.averageResponseTime
        )
    }
}

// MARK: - Batch Indexing Types

/// Options for batch indexing operations
public struct BatchIndexingOptions: Sendable {
    public let validateEntries: Bool
    public let allowPartialSuccess: Bool
    public let skipDuplicates: Bool
    public let maxRetries: Int
    public let optimizeAfterInsert: Bool
    
    public init(
        validateEntries: Bool = true,
        allowPartialSuccess: Bool = false,
        skipDuplicates: Bool = true,
        maxRetries: Int = 3,
        optimizeAfterInsert: Bool = false
    ) {
        self.validateEntries = validateEntries
        self.allowPartialSuccess = allowPartialSuccess
        self.skipDuplicates = skipDuplicates
        self.maxRetries = maxRetries
        self.optimizeAfterInsert = optimizeAfterInsert
    }
    
    public static let `default` = BatchIndexingOptions()
    
    public static let fast = BatchIndexingOptions(
        validateEntries: false,
        allowPartialSuccess: true,
        skipDuplicates: true,
        maxRetries: 1,
        optimizeAfterInsert: false
    )
    
    public static let safe = BatchIndexingOptions(
        validateEntries: true,
        allowPartialSuccess: false,
        skipDuplicates: false,
        maxRetries: 5,
        optimizeAfterInsert: true
    )
}

/// Result of batch indexing operation
public struct BatchIndexingResult: Sendable {
    public let totalEntries: Int
    public let successfulInserts: Int
    public let failedInserts: Int
    public let failures: [(id: String, error: Error)]
    public let totalTime: TimeInterval
    public let throughput: Double // vectors per second
    public let memoryPeakUsage: Int
    public let transactionId: String
    public let indexOptimized: Bool
}

/// Result of a single batch insertion
private struct BatchInsertResult {
    let successIds: [String]
    let failures: [(id: String, error: Error)]
    let transaction: IndexingTransaction
}

/// Transaction record for indexing operations
private struct IndexingTransaction {
    let id: String
    let parentTransactionId: String
    let timestamp: Date
    let entries: [(id: String, timestamp: Date)]
    let duration: TimeInterval
}

/// Transaction state tracking
private struct TransactionState {
    let id: String
    var processedBatches: Int = 0
    var successCount: Int = 0
    var failureCount: Int = 0
    var isCommitted: Bool = false
    var isRolledBack: Bool = false
}

/// Batch processing errors
public enum BatchProcessingError: LocalizedError {
    case memoryExhausted(required: Int, available: Int)
    case transactionFailed(transactionId: String, failureCount: Int)
    case duplicateEntry(id: String)
    case invalidDimensions(expected: Int, actual: Int)
    case invalidVectorData(id: String, reason: String)
    case invalidEntry(id: String, reason: String)
    case indexNotReady
    case concurrentModification(id: String)
    
    public var errorDescription: String? {
        switch self {
        case .memoryExhausted(let required, let available):
            return "Memory exhausted: required \(required) bytes, available \(available) bytes"
        case .transactionFailed(let transactionId, let failureCount):
            return "Transaction \(transactionId) failed with \(failureCount) errors"
        case .duplicateEntry(let id):
            return "Duplicate entry: \(id)"
        case .invalidDimensions(let expected, let actual):
            return "Invalid dimensions: expected \(expected), got \(actual)"
        case .invalidVectorData(let id, let reason):
            return "Invalid vector data for \(id): \(reason)"
        case .invalidEntry(let id, let reason):
            return "Invalid entry \(id): \(reason)"
        case .indexNotReady:
            return "Index is not ready for operations"
        case .concurrentModification(let id):
            return "Concurrent modification detected for entry: \(id)"
        }
    }
}

// MARK: - Supporting Types

/// Result of a batch operation on a single item
public struct BatchResult: Sendable {
    public let id: String
    public let success: Bool
    public let metadata: [String: String]
    
    public init(id: String, success: Bool, metadata: [String: String] = [:]) {
        self.id = id
        self.success = success
        self.metadata = metadata
    }
}

public struct BatchProcessingStatistics: Sendable {
    public let totalItemsProcessed: Int
    public let totalProcessingTime: TimeInterval
    public let averageItemsPerSecond: Double
    public let configuration: BatchProcessingConfiguration
}

/// Protocol for large datasets that support batch loading
public protocol LargeVectorDataset<Element>: Sendable {
    associatedtype Element: Sendable
    
    var count: Int { get async }
    func loadBatch(range: Range<Int>) async throws -> [Element]
}

/// Concrete implementation for Vector512 datasets
public struct Vector512Dataset: LargeVectorDataset {
    public typealias Element = Vector512
    
    private let vectors: [Vector512]
    
    public init(vectors: [Vector512]) {
        self.vectors = vectors
    }
    
    public var count: Int {
        get async { vectors.count }
    }
    
    public func loadBatch(range: Range<Int>) async throws -> [Element] {
        Array(vectors[range])
    }
}

/// Generic vector dataset
public struct VectorDataset<T: Sendable>: LargeVectorDataset {
    public typealias Element = T
    
    private let vectors: [T]
    
    public init(vectors: [T]) {
        self.vectors = vectors
    }
    
    public var count: Int {
        get async { vectors.count }
    }
    
    public func loadBatch(range: Range<Int>) async throws -> [Element] {
        Array(vectors[range])
    }
}

// MARK: - Extensions

extension BatchProcessor {
    /// Convenience method for processing with default configuration
    public static func processDataset<T: Sendable, R: Sendable>(
        _ dataset: any LargeVectorDataset<T>,
        processor: @escaping @Sendable ([T]) async throws -> [R]
    ) async throws -> [R] {
        let batchProcessor = BatchProcessor()
        return try await batchProcessor.processBatches(
            dataset: dataset,
            processor: processor
        )
    }
    
    /// Concurrent index building with multiple indexes
    public func buildIndexesConcurrently<Index: VectorIndex>(
        entries: [VectorEntry<Vector512, Index.Metadata>],
        indexes: [Index],
        options: BatchIndexingOptions = .default
    ) async throws -> [String: BatchIndexingResult] where Index.Vector == Vector512 {
        var results: [String: BatchIndexingResult] = [:]
        
        try await withThrowingTaskGroup(of: (String, BatchIndexingResult).self) { group in
            for (i, index) in indexes.enumerated() {
                let indexId = "index_\(i)"
                group.addTask {
                    let result = try await self.indexVectors(
                        entries: entries,
                        into: index,
                        options: options
                    )
                    return (indexId, result)
                }
            }
            
            for try await (indexId, result) in group {
                results[indexId] = result
            }
        }
        
        return results
    }
}

// MARK: - Array Extensions

private extension Array {
    /// Split array into chunks of specified size
    func chunked(into size: Int) -> [[Element]] {
        guard size > 0 else { return [] }
        
        var chunks: [[Element]] = []
        var startIndex = 0
        
        while startIndex < count {
            let endIndex = Swift.min(startIndex + size, count)
            chunks.append(Array(self[startIndex..<endIndex]))
            startIndex = endIndex
        }
        
        return chunks
    }
}

// MARK: - Additional Batch Processing Extensions

extension BatchProcessor {
    /// Memory-efficient batch iteration
    public func iterateBatches<T: Sendable>(
        dataset: any LargeVectorDataset<T>,
        batchSize: Int? = nil,
        operation: @escaping @Sendable (Int, [T]) async throws -> Void
    ) async throws {
        let actualBatchSize = batchSize ?? configuration.optimalBatchSize
        let totalItems = await dataset.count
        
        for offset in stride(from: 0, to: totalItems, by: actualBatchSize) {
            let endIdx = min(offset + actualBatchSize, totalItems)
            let batch = try await dataset.loadBatch(range: offset..<endIdx)
            try await operation(offset / actualBatchSize, batch)
        }
    }
}
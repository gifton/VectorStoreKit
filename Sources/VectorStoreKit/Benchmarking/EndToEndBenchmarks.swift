// VectorStoreKit: End-to-End Benchmarks
//
// Real-world usage pattern benchmarks

import Foundation
import simd

/// Benchmarks for end-to-end scenarios
public struct EndToEndBenchmarks {
    
    private let framework: BenchmarkFramework
    private let metrics: PerformanceMetrics
    
    public init(
        framework: BenchmarkFramework = BenchmarkFramework(),
        metrics: PerformanceMetrics = PerformanceMetrics()
    ) {
        self.framework = framework
        self.metrics = metrics
    }
    
    // MARK: - Types
    
    public struct ScenarioResult {
        public let scenario: String
        public let totalTime: TimeInterval
        public let stages: [StageResult]
        public let throughput: ThroughputMetrics
        public let accuracy: AccuracyMetrics
        
        public struct StageResult {
            public let name: String
            public let duration: TimeInterval
            public let operations: Int
            public let memoryDelta: Int
        }
        
        public struct ThroughputMetrics {
            public let indexingRate: Double // vectors/second
            public let queryRate: Double // queries/second
            public let updateRate: Double // updates/second
        }
        
        public struct AccuracyMetrics {
            public let recall: Double
            public let precision: Double
            public let f1Score: Double
        }
    }
    
    // MARK: - Main Benchmark Suites
    
    /// Run all end-to-end benchmarks
    public func runAll() async throws -> [String: BenchmarkFramework.Statistics] {
        var results: [String: BenchmarkFramework.Statistics] = [:]
        
        // Recommendation system scenario
        results.merge(try await runRecommendationScenario()) { _, new in new }
        
        // Semantic search scenario
        results.merge(try await runSemanticSearchScenario()) { _, new in new }
        
        // Image similarity scenario
        results.merge(try await runImageSimilarityScenario()) { _, new in new }
        
        // Streaming analytics scenario
        results.merge(try await runStreamingAnalyticsScenario()) { _, new in new }
        
        return results
    }
    
    // MARK: - Recommendation System Scenario
    
    private func runRecommendationScenario() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Recommendation System",
            description: "E-commerce recommendation engine simulation"
        ) {
            // Scenario parameters
            let userEmbeddingDim = 128
            let productEmbeddingDim = 256
            let numUsers = 100_000
            let numProducts = 1_000_000
            let numInteractions = 10_000
            
            benchmark(
                name: "recommendation_full_pipeline",
                setUp: {
                    await metrics.startCollection(name: "recommendation_system")
                },
                tearDown: {
                    _ = await metrics.stopCollection()
                }
            ) {
                // Stage 1: Build product index
                let productIndexStart = Date()
                let productIndex = try await buildProductIndex(
                    count: numProducts,
                    dimensions: productEmbeddingDim
                )
                let productIndexTime = Date().timeIntervalSince(productIndexStart)
                
                await metrics.recordCustom(
                    name: "product_index_build_time",
                    value: productIndexTime,
                    unit: "seconds"
                )
                
                // Stage 2: Build user index
                let userIndexStart = Date()
                let userIndex = try await buildUserIndex(
                    count: numUsers,
                    dimensions: userEmbeddingDim
                )
                let userIndexTime = Date().timeIntervalSince(userIndexStart)
                
                await metrics.recordCustom(
                    name: "user_index_build_time",
                    value: userIndexTime,
                    unit: "seconds"
                )
                
                // Stage 3: Process user interactions
                let interactionStart = Date()
                var recommendations: [[String]] = []
                
                for i in 0..<numInteractions {
                    // Simulate user browsing
                    let userId = "user_\(Int.random(in: 0..<numUsers))"
                    let userVector = try await userIndex.get(id: userId)?.vector
                    
                    if let userVector = userVector {
                        // Find similar users (collaborative filtering)
                        let similarUsers = try await userIndex.search(
                            query: userVector,
                            k: 50,
                            filter: UserFilter(excludeId: userId)
                        )
                        
                        // Get products liked by similar users
                        var productScores: [String: Float] = [:]
                        for user in similarUsers {
                            // Simulate getting user's liked products
                            let likedProducts = generateLikedProducts(
                                userId: user.id,
                                count: 10
                            )
                            
                            for productId in likedProducts {
                                productScores[productId, default: 0] += user.score
                            }
                        }
                        
                        // Get top recommendations
                        let topProducts = productScores
                            .sorted { $0.value > $1.value }
                            .prefix(10)
                            .map { $0.key }
                        
                        recommendations.append(topProducts)
                    }
                    
                    // Record metrics periodically
                    if i % 1000 == 0 {
                        await metrics.recordThroughput(
                            name: "recommendations",
                            operations: 1000,
                            duration: Date().timeIntervalSince(interactionStart)
                        )
                    }
                }
                
                let interactionTime = Date().timeIntervalSince(interactionStart)
                
                // Stage 4: Update embeddings (simulated training)
                let updateStart = Date()
                let updateCount = numUsers / 100 // Update 1% of users
                
                for i in 0..<updateCount {
                    let userId = "user_\(i)"
                    let newEmbedding = generateVector(dimensions: userEmbeddingDim)
                    _ = try await userIndex.update(
                        id: userId,
                        vector: newEmbedding
                    )
                }
                
                let updateTime = Date().timeIntervalSince(updateStart)
                
                // Calculate final metrics
                let totalTime = productIndexTime + userIndexTime + interactionTime + updateTime
                let recommendationRate = Double(numInteractions) / interactionTime
                
                await metrics.recordCustom(
                    name: "total_pipeline_time",
                    value: totalTime,
                    unit: "seconds"
                )
                
                await metrics.recordCustom(
                    name: "recommendation_rate",
                    value: recommendationRate,
                    unit: "recommendations/second"
                )
                
                blackHole((productIndex, userIndex, recommendations))
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Semantic Search Scenario
    
    private func runSemanticSearchScenario() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Semantic Search",
            description: "Document search engine simulation"
        ) {
            // Scenario parameters
            let embeddingDim = 768 // BERT-like embeddings
            let numDocuments = 100_000
            let numQueries = 1_000
            let indexTypes = ["hnsw", "ivf", "hybrid"]
            
            for indexType in indexTypes {
                benchmark(
                    name: "semantic_search_\(indexType)",
                    setUp: {
                        await metrics.startCollection(
                            name: "semantic_search_\(indexType)"
                        )
                    },
                    tearDown: {
                        _ = await metrics.stopCollection()
                    }
                ) {
                    // Stage 1: Generate document embeddings
                    let embedStart = Date()
                    let documents = generateDocuments(count: numDocuments)
                    let embeddings = documents.map { doc in
                        generateSemanticEmbedding(text: doc.content, dimensions: embeddingDim)
                    }
                    let embedTime = Date().timeIntervalSince(embedStart)
                    
                    // Stage 2: Build search index
                    let indexStart = Date()
                    let searchIndex = try await buildSearchIndex(
                        type: indexType,
                        embeddings: embeddings,
                        documents: documents,
                        dimensions: embeddingDim
                    )
                    let indexTime = Date().timeIntervalSince(indexStart)
                    
                    // Stage 3: Process search queries
                    let searchStart = Date()
                    var searchResults: [[SearchResult]] = []
                    let queries = generateSearchQueries(count: numQueries)
                    
                    for (i, query) in queries.enumerated() {
                        // Generate query embedding
                        let queryEmbedding = generateSemanticEmbedding(
                            text: query.text,
                            dimensions: embeddingDim
                        )
                        
                        // Search with filters
                        let results = try await searchIndex.search(
                            query: queryEmbedding,
                            k: 20,
                            filter: query.filter
                        )
                        
                        searchResults.append(results)
                        
                        // Simulate result reranking
                        let rerankedResults = rerankResults(
                            results: results,
                            query: query.text
                        )
                        
                        // Record search latency
                        if i % 100 == 0 {
                            let elapsed = Date().timeIntervalSince(searchStart)
                            await metrics.recordLatency(
                                name: "search_latency",
                                value: elapsed / Double(i + 1)
                            )
                        }
                    }
                    
                    let searchTime = Date().timeIntervalSince(searchStart)
                    
                    // Stage 4: Incremental updates
                    let updateStart = Date()
                    let newDocuments = generateDocuments(count: 1000)
                    
                    for doc in newDocuments {
                        let embedding = generateSemanticEmbedding(
                            text: doc.content,
                            dimensions: embeddingDim
                        )
                        
                        _ = try await searchIndex.insert(VectorEntry(
                            id: doc.id,
                            vector: embedding,
                            metadata: doc
                        ))
                    }
                    
                    let updateTime = Date().timeIntervalSince(updateStart)
                    
                    // Calculate metrics
                    let totalTime = embedTime + indexTime + searchTime + updateTime
                    let avgSearchLatency = searchTime / Double(numQueries)
                    
                    await metrics.recordCustom(
                        name: "avg_search_latency",
                        value: avgSearchLatency * 1000,
                        unit: "milliseconds"
                    )
                    
                    await metrics.recordCustom(
                        name: "index_\(indexType)_build_time",
                        value: indexTime,
                        unit: "seconds"
                    )
                    
                    blackHole((searchIndex, searchResults))
                }
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Image Similarity Scenario
    
    private func runImageSimilarityScenario() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Image Similarity",
            description: "Visual search engine simulation"
        ) {
            // Scenario parameters
            let featureDim = 2048 // ResNet-like features
            let numImages = 1_000_000
            let numQueries = 500
            let batchSize = 100
            
            benchmark(
                name: "image_similarity_pipeline",
                setUp: {
                    await metrics.startCollection(name: "image_similarity")
                },
                tearDown: {
                    _ = await metrics.stopCollection()
                }
            ) {
                // Stage 1: Extract image features (simulated)
                let featureStart = Date()
                var imageFeatures: [[Float]] = []
                
                for batch in 0..<(numImages / batchSize) {
                    let batchFeatures = (0..<batchSize).map { _ in
                        generateImageFeatures(dimensions: featureDim)
                    }
                    imageFeatures.append(contentsOf: batchFeatures)
                    
                    if batch % 100 == 0 {
                        await metrics.recordThroughput(
                            name: "feature_extraction",
                            operations: batchSize * 100,
                            duration: Date().timeIntervalSince(featureStart)
                        )
                    }
                }
                
                let featureTime = Date().timeIntervalSince(featureStart)
                
                // Stage 2: Build visual index with quantization
                let indexStart = Date()
                let visualIndex = try await buildVisualIndex(
                    features: imageFeatures,
                    dimensions: featureDim,
                    useQuantization: true
                )
                let indexTime = Date().timeIntervalSince(indexStart)
                
                // Stage 3: Process visual queries
                let queryStart = Date()
                var queryResults: [[VisualSearchResult]] = []
                
                for i in 0..<numQueries {
                    // Simulate different query types
                    let queryType = ["similar", "duplicate", "near-duplicate"].randomElement()!
                    let queryFeatures = generateQueryFeatures(
                        type: queryType,
                        dimensions: featureDim
                    )
                    
                    // Multi-stage search
                    let coarseResults = try await visualIndex.search(
                        query: queryFeatures,
                        k: 1000,
                        strategy: .fast
                    )
                    
                    // Rerank with full precision
                    let fineResults = rerankVisualResults(
                        results: coarseResults,
                        query: queryFeatures,
                        topK: 50
                    )
                    
                    queryResults.append(fineResults)
                    
                    if i % 50 == 0 {
                        let elapsed = Date().timeIntervalSince(queryStart)
                        await metrics.recordLatency(
                            name: "visual_search_latency",
                            value: elapsed / Double(i + 1)
                        )
                    }
                }
                
                let queryTime = Date().timeIntervalSince(queryStart)
                
                // Stage 4: Deduplication check
                let dedupStart = Date()
                var duplicates = 0
                
                for i in 0..<min(10_000, numImages) {
                    let anchor = imageFeatures[i]
                    let nearDuplicates = try await visualIndex.search(
                        query: SIMD32<Float>(anchor),
                        k: 5,
                        filter: ExcludeIdFilter(id: "img_\(i)")
                    )
                    
                    for result in nearDuplicates {
                        if result.score > 0.95 {
                            duplicates += 1
                        }
                    }
                }
                
                let dedupTime = Date().timeIntervalSince(dedupStart)
                
                // Calculate metrics
                let totalTime = featureTime + indexTime + queryTime + dedupTime
                let duplicateRate = Double(duplicates) / Double(10_000)
                
                await metrics.recordCustom(
                    name: "duplicate_rate",
                    value: duplicateRate,
                    unit: "ratio"
                )
                
                await metrics.recordCustom(
                    name: "visual_index_size",
                    value: Double(await visualIndex.memoryUsage),
                    unit: "bytes"
                )
                
                blackHole((visualIndex, queryResults))
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Streaming Analytics Scenario
    
    private func runStreamingAnalyticsScenario() async throws -> [String: BenchmarkFramework.Statistics] {
        let suite = benchmarkSuite(
            name: "Streaming Analytics",
            description: "Real-time anomaly detection simulation"
        ) {
            // Scenario parameters
            let featureDim = 64
            let windowSize = 10_000
            let streamDuration = 60.0 // seconds
            let eventsPerSecond = 1000
            
            benchmark(
                name: "streaming_anomaly_detection",
                setUp: {
                    await metrics.startCollection(name: "streaming_analytics")
                },
                tearDown: {
                    _ = await metrics.stopCollection()
                }
            ) {
                // Initialize sliding window index
                let slidingIndex = try await SlidingWindowIndex(
                    windowSize: windowSize,
                    dimensions: featureDim
                )
                
                let startTime = Date()
                var detectedAnomalies = 0
                var processedEvents = 0
                
                // Simulate event stream
                while Date().timeIntervalSince(startTime) < streamDuration {
                    let batchStart = Date()
                    
                    // Process batch of events
                    for _ in 0..<eventsPerSecond {
                        // Generate event (normal or anomaly)
                        let isAnomaly = Double.random(in: 0...1) < 0.01 // 1% anomalies
                        let eventFeatures = generateEventFeatures(
                            dimensions: featureDim,
                            isAnomaly: isAnomaly
                        )
                        
                        // Check if event is anomalous
                        let neighbors = try await slidingIndex.search(
                            query: eventFeatures,
                            k: 10
                        )
                        
                        let avgDistance = neighbors.isEmpty ? Float.infinity :
                            neighbors.map { $0.distance }.reduce(0, +) / Float(neighbors.count)
                        
                        if avgDistance > 2.0 { // Threshold for anomaly
                            detectedAnomalies += 1
                        }
                        
                        // Add to sliding window
                        let eventId = "event_\(processedEvents)"
                        _ = try await slidingIndex.insert(VectorEntry(
                            id: eventId,
                            vector: eventFeatures,
                            metadata: EventMetadata(
                                timestamp: Date(),
                                isAnomaly: isAnomaly
                            )
                        ))
                        
                        processedEvents += 1
                    }
                    
                    // Record metrics
                    let batchTime = Date().timeIntervalSince(batchStart)
                    await metrics.recordThroughput(
                        name: "event_processing",
                        operations: eventsPerSecond,
                        duration: batchTime
                    )
                    
                    // Sleep to maintain rate
                    let sleepTime = 1.0 - batchTime
                    if sleepTime > 0 {
                        try await Task.sleep(nanoseconds: UInt64(sleepTime * 1_000_000_000))
                    }
                }
                
                let totalTime = Date().timeIntervalSince(startTime)
                let processingRate = Double(processedEvents) / totalTime
                
                await metrics.recordCustom(
                    name: "event_processing_rate",
                    value: processingRate,
                    unit: "events/second"
                )
                
                await metrics.recordCustom(
                    name: "anomalies_detected",
                    value: Double(detectedAnomalies),
                    unit: "count"
                )
                
                blackHole((slidingIndex, detectedAnomalies))
            }
        }
        
        return try await framework.run(suite: suite)
    }
    
    // MARK: - Helper Functions
    
    private func buildProductIndex(
        count: Int,
        dimensions: Int
    ) async throws -> HNSWIndex<SIMD32<Float>, ProductMetadata> {
        let config = HNSWIndex<SIMD32<Float>, ProductMetadata>.Configuration(
            maxConnections: 32,
            efConstruction: 400
        )
        
        let index = try HNSWIndex<SIMD32<Float>, ProductMetadata>(
            configuration: config
        )
        
        for i in 0..<count {
            let embedding = generateVector(dimensions: dimensions)
            let product = ProductMetadata(
                id: "product_\(i)",
                category: "category_\(i % 100)",
                price: Float.random(in: 10...1000),
                rating: Float.random(in: 1...5)
            )
            
            _ = try await index.insert(VectorEntry(
                id: product.id,
                vector: embedding,
                metadata: product
            ))
        }
        
        return index
    }
    
    private func buildUserIndex(
        count: Int,
        dimensions: Int
    ) async throws -> HNSWIndex<SIMD32<Float>, UserMetadata> {
        let config = HNSWIndex<SIMD32<Float>, UserMetadata>.Configuration(
            maxConnections: 16,
            efConstruction: 200
        )
        
        let index = try HNSWIndex<SIMD32<Float>, UserMetadata>(
            configuration: config
        )
        
        for i in 0..<count {
            let embedding = generateVector(dimensions: dimensions)
            let user = UserMetadata(
                id: "user_\(i)",
                segment: "segment_\(i % 10)",
                activityLevel: Float.random(in: 0...1)
            )
            
            _ = try await index.insert(VectorEntry(
                id: user.id,
                vector: embedding,
                metadata: user
            ))
        }
        
        return index
    }
    
    private func generateVector(dimensions: Int) -> SIMD32<Float> {
        var vector = SIMD32<Float>()
        for i in 0..<min(32, dimensions) {
            vector[i] = Float.random(in: -1...1)
        }
        return vector
    }
    
    private func generateLikedProducts(userId: String, count: Int) -> [String] {
        // Deterministic based on user ID for consistency
        let hash = userId.hashValue
        return (0..<count).map { i in
            "product_\((hash + i) % 1_000_000)"
        }
    }
    
    private func generateDocuments(count: Int) -> [Document] {
        return (0..<count).map { i in
            Document(
                id: "doc_\(i)",
                content: "Sample document content \(i)",
                category: "category_\(i % 20)",
                timestamp: Date()
            )
        }
    }
    
    private func generateSemanticEmbedding(text: String, dimensions: Int) -> SIMD32<Float> {
        // Simulate semantic embedding generation
        var embedding = SIMD32<Float>()
        let hash = text.hashValue
        for i in 0..<min(32, dimensions) {
            embedding[i] = Float(truncatingIfNeeded: hash &+ i) / Float(Int32.max)
        }
        return embedding
    }
    
    private func generateSearchQueries(count: Int) -> [SearchQuery] {
        return (0..<count).map { i in
            SearchQuery(
                text: "search query \(i)",
                filter: i % 10 == 0 ? createDocumentCategoryFilter(category: "category_\(i % 20)") : nil
            )
        }
    }
    
    private func rerankResults(results: [EndToEndSearchResult], query: String) -> [EndToEndSearchResult] {
        // Simulate reranking logic
        return results.sorted { $0.score > $1.score }
    }
    
    private func generateImageFeatures(dimensions: Int) -> [Float] {
        return (0..<dimensions).map { _ in Float.random(in: 0...1) }
    }
    
    private func generateQueryFeatures(type: String, dimensions: Int) -> SIMD32<Float> {
        var features = SIMD32<Float>()
        let baseValue: Float = type == "duplicate" ? 0.9 : 0.5
        
        for i in 0..<min(32, dimensions) {
            features[i] = baseValue + Float.random(in: -0.1...0.1)
        }
        return features
    }
    
    private func rerankVisualResults(
        results: [EndToEndSearchResult],
        query: SIMD32<Float>,
        topK: Int
    ) -> [VisualSearchResult] {
        return results.prefix(topK).map { result in
            VisualSearchResult(
                id: result.id,
                distance: result.distance,
                confidence: 1.0 / (1.0 + result.distance)
            )
        }
    }
    
    private func generateEventFeatures(
        dimensions: Int,
        isAnomaly: Bool
    ) -> SIMD32<Float> {
        var features = SIMD32<Float>()
        let mean: Float = isAnomaly ? 3.0 : 0.0
        let stdDev: Float = isAnomaly ? 2.0 : 1.0
        
        for i in 0..<min(32, dimensions) {
            features[i] = generateNormal(mean: mean, stdDev: stdDev)
        }
        return features
    }
    
    private func generateNormal(mean: Float, stdDev: Float) -> Float {
        // Box-Muller transform for normal distribution
        let u1 = Float.random(in: 0..<1)
        let u2 = Float.random(in: 0..<1)
        let z0 = sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
        return mean + stdDev * z0
    }
    
    private func buildSearchIndex(
        type: String,
        embeddings: [SIMD32<Float>],
        documents: [Document],
        dimensions: Int
    ) async throws -> any VectorIndex {
        switch type {
        case "hnsw":
            return try await buildHNSWSearchIndex(
                embeddings: embeddings,
                documents: documents
            )
        case "ivf":
            return try await buildIVFSearchIndex(
                embeddings: embeddings,
                documents: documents,
                dimensions: dimensions
            )
        case "hybrid":
            return try await buildHybridSearchIndex(
                embeddings: embeddings,
                documents: documents,
                dimensions: dimensions
            )
        default:
            fatalError("Unknown index type: \(type)")
        }
    }
    
    private func buildHNSWSearchIndex(
        embeddings: [SIMD32<Float>],
        documents: [Document]
    ) async throws -> HNSWIndex<SIMD32<Float>, Document> {
        let config = HNSWIndex<SIMD32<Float>, Document>.Configuration(
            maxConnections: 16,
            efConstruction: 200
        )
        
        let index = try HNSWIndex<SIMD32<Float>, Document>(
            configuration: config
        )
        
        for (embedding, document) in zip(embeddings, documents) {
            _ = try await index.insert(VectorEntry(
                id: document.id,
                vector: embedding,
                metadata: document
            ))
        }
        
        return index
    }
    
    private func buildIVFSearchIndex(
        embeddings: [SIMD32<Float>],
        documents: [Document],
        dimensions: Int
    ) async throws -> IVFIndex<SIMD32<Float>, Document> {
        let config = IVFConfiguration(
            dimensions: dimensions,
            numberOfCentroids: min(1024, embeddings.count / 100)
        )
        
        let index = try await IVFIndex<SIMD32<Float>, Document>(
            configuration: config
        )
        
        // Train on subset
        let trainingData = embeddings.prefix(min(10_000, embeddings.count)).map { vector in
            (0..<dimensions).map { i in i < 32 ? vector[i] : 0 }
        }
        try await index.train(on: trainingData)
        
        // Insert all
        for (embedding, document) in zip(embeddings, documents) {
            _ = try await index.insert(VectorEntry(
                id: document.id,
                vector: embedding,
                metadata: document
            ))
        }
        
        return index
    }
    
    private func buildHybridSearchIndex(
        embeddings: [SIMD32<Float>],
        documents: [Document],
        dimensions: Int
    ) async throws -> HybridIndex<SIMD32<Float>, Document> {
        let config = HybridIndexConfiguration(
            dimensions: dimensions,
            routingStrategy: .adaptive
        )
        
        let index = try await HybridIndex<SIMD32<Float>, Document>(
            configuration: config
        )
        
        // Train on data
        let trainingData = embeddings.prefix(min(10_000, embeddings.count)).map { vector in
            (0..<dimensions).map { i in i < 32 ? vector[i] : 0 }
        }
        try await index.train(on: trainingData)
        
        // Insert all
        for (embedding, document) in zip(embeddings, documents) {
            _ = try await index.insert(VectorEntry(
                id: document.id,
                vector: embedding,
                metadata: document
            ))
        }
        
        return index
    }
    
    private func buildVisualIndex(
        features: [[Float]],
        dimensions: Int,
        useQuantization: Bool
    ) async throws -> HNSWIndex<SIMD32<Float>, ImageMetadata> {
        let config = HNSWIndex<SIMD32<Float>, ImageMetadata>.Configuration(
            maxConnections: 32,
            efConstruction: 400,
            enableQuantization: useQuantization
        )
        
        let index = try HNSWIndex<SIMD32<Float>, ImageMetadata>(
            configuration: config
        )
        
        for (i, feature) in features.enumerated() {
            var vector = SIMD32<Float>()
            for j in 0..<min(32, dimensions) {
                vector[j] = feature[j]
            }
            
            let metadata = ImageMetadata(
                id: "img_\(i)",
                format: ["jpg", "png", "webp"].randomElement()!,
                width: Int.random(in: 100...4000),
                height: Int.random(in: 100...4000)
            )
            
            _ = try await index.insert(VectorEntry(
                id: metadata.id,
                vector: vector,
                metadata: metadata
            ))
        }
        
        return index
    }
}

// MARK: - Helper Types

private struct ProductMetadata: Codable, Sendable {
    let id: String
    let category: String
    let price: Float
    let rating: Float
}

private struct UserMetadata: Codable, Sendable {
    let id: String
    let segment: String
    let activityLevel: Float
}

private struct Document: Codable, Sendable {
    let id: String
    let content: String
    let category: String
    let timestamp: Date
}

private struct ImageMetadata: Codable, Sendable {
    let id: String
    let format: String
    let width: Int
    let height: Int
}

private struct EventMetadata: Codable, Sendable {
    let timestamp: Date
    let isAnomaly: Bool
}

private struct SearchQuery {
    let text: String
    let filter: SearchFilter?
}

private struct VisualSearchResult {
    let id: String
    let distance: Float
    let confidence: Float
}

private struct EndToEndSearchResult {
    let id: String
    let distance: Float
    let score: Float
}

private func createUserExcludeFilter(excludeId: String) -> SearchFilter {
    return .metadata(MetadataFilter(
        key: "id",
        operation: .notEquals,
        value: excludeId
    ))
}

private func createDocumentCategoryFilter(category: String) -> SearchFilter {
    return .metadata(MetadataFilter(
        key: "category",
        operation: .equals,
        value: category
    ))
}

private func createImageExcludeFilter(id: String) -> SearchFilter {
    return .metadata(MetadataFilter(
        key: "id",
        operation: .notEquals,
        value: id
    ))
}

// Mock sliding window index
private actor SlidingWindowIndex {
    private let windowSize: Int
    private let dimensions: Int
    private var entries: [(id: String, vector: SIMD32<Float>, metadata: EventMetadata)] = []
    
    init(windowSize: Int, dimensions: Int) {
        self.windowSize = windowSize
        self.dimensions = dimensions
    }
    
    func insert(_ entry: VectorEntry<SIMD32<Float>, EventMetadata>) async throws -> String {
        entries.append((entry.id, entry.vector, entry.metadata))
        
        // Remove old entries
        if entries.count > windowSize {
            entries.removeFirst(entries.count - windowSize)
        }
        
        return entry.id
    }
    
    func search(query: SIMD32<Float>, k: Int) async throws -> [EndToEndSearchResult] {
        var results: [(id: String, distance: Float)] = []
        
        for entry in entries {
            let diff = query - entry.vector
            let distance = sqrt(simd_dot(diff, diff))
            results.append((entry.id, distance))
        }
        
        return results
            .sorted { $0.distance < $1.distance }
            .prefix(k)
            .map { EndToEndSearchResult(id: $0.id, distance: $0.distance, score: 1.0 / (1.0 + $0.distance)) }
    }
}
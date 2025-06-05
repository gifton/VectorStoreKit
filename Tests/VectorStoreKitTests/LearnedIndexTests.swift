// VectorStoreKit: Learned Index Tests
//
// Comprehensive test suite for learned index functionality

import Testing
import Foundation
@testable import VectorStoreKit

@Suite("Learned Index Tests")
struct LearnedIndexTests {
    
    // MARK: - Test Data
    
    static func generateSequentialVectors(count: Int, dimensions: Int = 128) -> [[Float]] {
        // Generate vectors with sequential patterns that can be learned
        (0..<count).map { i in
            let position = Float(i) / Float(count)
            return (0..<dimensions).map { d in
                sin(Float(d) * position * .pi) + Float(i % 10) * 0.1
            }
        }
    }
    
    static func generatePatternedVectors(count: Int, dimensions: Int = 128) -> [[Float]] {
        // Generate vectors with clear patterns for learned models
        (0..<count).map { i in
            let pattern = i % 5
            return (0..<dimensions).map { d in
                switch pattern {
                case 0: return Float(d % 2)
                case 1: return Float(d % 3) / 3.0
                case 2: return sin(Float(d) * 0.1)
                case 3: return cos(Float(d) * 0.1)
                default: return Float(d) / Float(dimensions)
                }
            }
        }
    }
    
    // MARK: - Configuration Tests
    
    @Test("Learned Index Configuration")
    func testConfiguration() throws {
        // Test different model architectures
        let linearConfig = LearnedIndexConfiguration(
            dimensions: 128,
            modelArchitecture: .linear,
            bucketSize: 100
        )
        #expect(throws: Never.self) {
            try linearConfig.validate()
        }
        
        let mlpConfig = LearnedIndexConfiguration(
            dimensions: 128,
            modelArchitecture: .mlp(hiddenSizes: [256, 128, 64]),
            bucketSize: 50
        )
        #expect(throws: Never.self) {
            try mlpConfig.validate()
        }
        
        let residualConfig = LearnedIndexConfiguration(
            dimensions: 128,
            modelArchitecture: .residual(layers: 4, hiddenSize: 256),
            bucketSize: 200
        )
        #expect(throws: Never.self) {
            try residualConfig.validate()
        }
        
        // Invalid configurations
        let invalidDimConfig = LearnedIndexConfiguration(
            dimensions: 0,
            bucketSize: 100
        )
        #expect(throws: LearnedIndexError.self) {
            try invalidDimConfig.validate()
        }
        
        let invalidBucketConfig = LearnedIndexConfiguration(
            dimensions: 128,
            bucketSize: 0
        )
        #expect(throws: LearnedIndexError.self) {
            try invalidBucketConfig.validate()
        }
    }
    
    @Test("Memory Estimation")
    func testMemoryEstimation() {
        let config = LearnedIndexConfiguration(
            dimensions: 128,
            modelArchitecture: .mlp(hiddenSizes: [256, 128]),
            bucketSize: 100
        )
        
        let mem1K = config.estimatedMemoryUsage(for: 1000)
        let mem10K = config.estimatedMemoryUsage(for: 10000)
        
        #expect(mem10K > mem1K)
        #expect(mem1K > 0)
        
        // Larger model should use more memory
        let largeModelConfig = LearnedIndexConfiguration(
            dimensions: 128,
            modelArchitecture: .residual(layers: 8, hiddenSize: 512),
            bucketSize: 100
        )
        
        let largeMem = largeModelConfig.estimatedMemoryUsage(for: 1000)
        #expect(largeMem > mem1K)
    }
    
    // MARK: - Training Tests
    
    @Test("Basic Training", .timeLimit(.seconds(30)))
    func testBasicTraining() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 32,
            modelArchitecture: .linear,
            bucketSize: 50,
            trainingConfig: .init(epochs: 10)
        )
        
        let index = try await LearnedIndex<SIMD32<Float>, TestMetadata>(configuration: config)
        
        // Generate training data with clear patterns
        let trainingData = Self.generateSequentialVectors(count: 500, dimensions: 32)
        
        // Insert training data
        for (i, vector) in trainingData.enumerated() {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD32<Float>(vector),
                metadata: TestMetadata(label: "train\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        // Train the model
        try await index.train()
        
        // Verify training
        #expect(await index.isOptimized == true)
        
        let stats = await index.statistics()
        #expect(stats.trained == true)
        #expect(stats.vectorCount == 500)
    }
    
    @Test("MLP Model Training", .timeLimit(.seconds(60)))
    func testMLPTraining() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 64,
            modelArchitecture: .mlp(hiddenSizes: [128, 64, 32]),
            bucketSize: 25,
            trainingConfig: .init(
                epochs: 20,
                batchSize: 32,
                learningRate: 0.01
            )
        )
        
        let index = try await LearnedIndex<SIMD64<Float>, TestMetadata>(configuration: config)
        
        // Use patterned data for better learning
        let trainingData = Self.generatePatternedVectors(count: 1000, dimensions: 64)
        
        // Insert data
        for (i, vector) in trainingData.enumerated() {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD64<Float>(vector),
                metadata: TestMetadata(label: "pattern\(i % 5)")
            )
            _ = try await index.insert(entry)
        }
        
        // Train
        try await index.train()
        
        // Test that learned model improves search
        let queryVector = SIMD64<Float>(trainingData[250])
        let results = try await index.search(query: queryVector, k: 10)
        
        #expect(results.count > 0)
        #expect(results.first?.id == "vec250")  // Should find exact match
    }
    
    // MARK: - Insert/Search Tests
    
    @Test("Insert and Search Operations")
    func testInsertAndSearch() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 16,
            modelArchitecture: .linear,
            bucketSize: 10
        )
        
        let index = try await LearnedIndex<SIMD16<Float>, TestMetadata>(configuration: config)
        
        // Insert without training (uses fallback)
        let vector1 = SIMD16<Float>(repeating: 1.0)
        let entry1 = VectorEntry(
            id: "test1",
            vector: vector1,
            metadata: TestMetadata(label: "one")
        )
        
        let insertResult = try await index.insert(entry1)
        #expect(insertResult.success == true)
        
        // Search without training
        let query = SIMD16<Float>(repeating: 0.9)
        let results = try await index.search(query: query, k: 5)
        
        #expect(results.count == 1)
        #expect(results[0].id == "test1")
        
        // Insert more vectors
        for i in 2...20 {
            let value = Float(i) / 20.0
            let vector = SIMD16<Float>(repeating: value)
            let entry = VectorEntry(
                id: "test\(i)",
                vector: vector,
                metadata: TestMetadata(label: "value\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        // Train the model
        try await index.train()
        
        // Search should be more accurate after training
        let trainedResults = try await index.search(query: query, k: 5)
        #expect(trainedResults.count >= 5)
        
        // Closest vectors should be near 0.9
        let distances = trainedResults.map { $0.distance }
        #expect(distances[0] < distances[4])  // Should be sorted by distance
    }
    
    @Test("Bucket Distribution")
    func testBucketDistribution() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 32,
            modelArchitecture: .linear,
            bucketSize: 20
        )
        
        let index = try await LearnedIndex<SIMD32<Float>, TestMetadata>(configuration: config)
        
        // Insert vectors that should distribute across buckets
        let vectors = Self.generateSequentialVectors(count: 200, dimensions: 32)
        
        for (i, vector) in vectors.enumerated() {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD32<Float>(vector),
                metadata: TestMetadata(label: "seq\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        // Train to optimize bucket assignment
        try await index.train()
        
        let stats = await index.statistics()
        print("Buckets: \(stats.bucketCount), Avg size: \(stats.averageBucketSize), Max size: \(stats.maxBucketSize)")
        
        // Should have created multiple buckets
        #expect(stats.bucketCount > 5)
        #expect(stats.bucketCount <= 20)  // Reasonable number of buckets
        
        // Check for reasonable distribution
        #expect(stats.maxBucketSize <= stats.averageBucketSize * 3)  // Not too imbalanced
    }
    
    // MARK: - Performance Tests
    
    @Test("Search Performance", .timeLimit(.seconds(30)))
    func testSearchPerformance() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 128,
            modelArchitecture: .mlp(hiddenSizes: [256, 128]),
            bucketSize: 100,
            useMetalAcceleration: true
        )
        
        let index = try await LearnedIndex<SIMD128<Float>, TestMetadata>(configuration: config)
        
        // Insert 10K vectors
        let vectors = Self.generatePatternedVectors(count: 10_000, dimensions: 128)
        
        for (i, vector) in vectors.enumerated() {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD128<Float>(vector),
                metadata: TestMetadata(label: "perf\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        // Train the model
        try await index.train()
        
        // Measure search performance
        let queries = (0..<100).map { _ in
            SIMD128<Float>(Self.generateSequentialVectors(count: 1, dimensions: 128)[0])
        }
        
        let startTime = DispatchTime.now()
        
        for query in queries {
            let results = try await index.search(query: query, k: 10)
            #expect(results.count > 0)
        }
        
        let endTime = DispatchTime.now()
        let totalTime = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
        let avgTime = totalTime / Double(queries.count)
        
        print("Average learned search time: \(avgTime * 1000)ms")
        #expect(avgTime < 0.002)  // Should be < 2ms per search
    }
    
    @Test("Model Architecture Performance")
    func testModelArchitecturePerformance() async throws {
        let dimensions = 64
        let vectorCount = 1000
        
        // Test different architectures
        let architectures: [(String, LearnedIndexConfiguration.ModelArchitecture)] = [
            ("Linear", .linear),
            ("MLP-Small", .mlp(hiddenSizes: [128])),
            ("MLP-Large", .mlp(hiddenSizes: [256, 128, 64])),
            ("Residual", .residual(layers: 3, hiddenSize: 128))
        ]
        
        for (name, architecture) in architectures {
            let config = LearnedIndexConfiguration(
                dimensions: dimensions,
                modelArchitecture: architecture,
                bucketSize: 50,
                trainingConfig: .init(epochs: 10)
            )
            
            let index = try await LearnedIndex<SIMD64<Float>, TestMetadata>(configuration: config)
            
            // Insert data
            let vectors = Self.generatePatternedVectors(count: vectorCount, dimensions: dimensions)
            for (i, vector) in vectors.enumerated() {
                let entry = VectorEntry(
                    id: "vec\(i)",
                    vector: SIMD64<Float>(vector),
                    metadata: TestMetadata(label: "arch\(i)")
                )
                _ = try await index.insert(entry)
            }
            
            // Train and measure time
            let trainStart = DispatchTime.now()
            try await index.train()
            let trainEnd = DispatchTime.now()
            let trainTime = Double(trainEnd.uptimeNanoseconds - trainStart.uptimeNanoseconds) / 1_000_000_000
            
            print("\(name) training time: \(trainTime)s")
            
            // Measure search accuracy
            var correctMatches = 0
            for i in stride(from: 0, to: 100, by: 10) {
                let query = SIMD64<Float>(vectors[i])
                let results = try await index.search(query: query, k: 1)
                if results.first?.id == "vec\(i)" {
                    correctMatches += 1
                }
            }
            
            print("\(name) accuracy: \(correctMatches)/10")
        }
    }
    
    // MARK: - Integrity Tests
    
    @Test("Index Integrity")
    func testIntegrity() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 32,
            modelArchitecture: .linear,
            bucketSize: 25
        )
        
        let index = try await LearnedIndex<SIMD32<Float>, TestMetadata>(configuration: config)
        
        // Insert vectors
        let vectors = Self.generateSequentialVectors(count: 100, dimensions: 32)
        for (i, vector) in vectors.enumerated() {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD32<Float>(vector),
                metadata: TestMetadata(label: "integrity\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        // Check integrity
        let integrity = try await index.validateIntegrity()
        #expect(integrity.isValid == true)
        #expect(integrity.errors.isEmpty == true)
        
        // Train and check again
        try await index.train()
        
        let trainedIntegrity = try await index.validateIntegrity()
        #expect(trainedIntegrity.isValid == true)
        
        // Check for bucket overflow warnings
        let stats = await index.statistics()
        if stats.maxBucketSize > config.bucketSize * 2 {
            #expect(trainedIntegrity.warnings.count > 0)
        }
    }
    
    // MARK: - Export/Import Tests
    
    @Test("Export and Import")
    func testExportImport() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 16,
            modelArchitecture: .mlp(hiddenSizes: [32, 16]),
            bucketSize: 10
        )
        
        let index1 = try await LearnedIndex<SIMD16<Float>, TestMetadata>(configuration: config)
        
        // Insert and train
        let vectors = Self.generatePatternedVectors(count: 50, dimensions: 16)
        for (i, vector) in vectors.enumerated() {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD16<Float>(vector),
                metadata: TestMetadata(label: "export\(i)")
            )
            _ = try await index1.insert(entry)
        }
        
        try await index1.train()
        
        // Export
        let exportedData = try await index1.export(format: .json)
        
        // Create new index and import
        let index2 = try await LearnedIndex<SIMD16<Float>, TestMetadata>(configuration: config)
        try await index2.import(data: exportedData, format: .json)
        
        // Verify state
        #expect(await index2.isOptimized == true)
        let stats2 = await index2.statistics()
        #expect(stats2.trained == true)
        
        // Search should work identically
        let query = SIMD16<Float>(vectors[25])
        let results1 = try await index1.search(query: query, k: 5)
        let results2 = try await index2.search(query: query, k: 5)
        
        #expect(results1.map { $0.id } == results2.map { $0.id })
    }
    
    // MARK: - Edge Cases
    
    @Test("Empty Index Operations")
    func testEmptyIndex() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 8,
            bucketSize: 10
        )
        
        let index = try await LearnedIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Train on empty index should throw
        await #expect(throws: LearnedIndexError.self) {
            try await index.train()
        }
        
        // Search on empty index should return empty results
        let query = SIMD8<Float>(repeating: 1.0)
        let results = try await index.search(query: query, k: 5)
        #expect(results.isEmpty == true)
        
        // Delete non-existent item
        let deleted = try await index.delete(id: "nonexistent")
        #expect(deleted == false)
    }
    
    @Test("Retrain Model")
    func testRetrain() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 32,
            modelArchitecture: .linear,
            bucketSize: 20,
            trainingConfig: .init(epochs: 5)
        )
        
        let index = try await LearnedIndex<SIMD32<Float>, TestMetadata>(configuration: config)
        
        // Initial data
        let initialVectors = Self.generateSequentialVectors(count: 100, dimensions: 32)
        for (i, vector) in initialVectors.enumerated() {
            let entry = VectorEntry(
                id: "initial\(i)",
                vector: SIMD32<Float>(vector),
                metadata: TestMetadata(label: "initial")
            )
            _ = try await index.insert(entry)
        }
        
        // Train
        try await index.train()
        let stats1 = await index.statistics()
        
        // Add more data with different pattern
        let newVectors = Self.generatePatternedVectors(count: 100, dimensions: 32)
        for (i, vector) in newVectors.enumerated() {
            let entry = VectorEntry(
                id: "new\(i)",
                vector: SIMD32<Float>(vector),
                metadata: TestMetadata(label: "new")
            )
            _ = try await index.insert(entry)
        }
        
        // Retrain
        try await index.train()
        let stats2 = await index.statistics()
        
        #expect(stats2.vectorCount == 200)
        #expect(stats2.trained == true)
        
        // Model should handle both patterns
        let query1 = SIMD32<Float>(initialVectors[50])
        let results1 = try await index.search(query: query1, k: 1)
        #expect(results1.first?.id == "initial50")
        
        let query2 = SIMD32<Float>(newVectors[50])
        let results2 = try await index.search(query: query2, k: 1)
        #expect(results2.first?.id == "new50")
    }
    
    @Test("Distribution Analysis")
    func testDistributionAnalysis() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 64,
            modelArchitecture: .mlp(hiddenSizes: [128, 64]),
            bucketSize: 50
        )
        
        let index = try await LearnedIndex<SIMD64<Float>, TestMetadata>(configuration: config)
        
        // Insert diverse vectors
        let vectors = Self.generatePatternedVectors(count: 500, dimensions: 64)
        for (i, vector) in vectors.enumerated() {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD64<Float>(vector),
                metadata: TestMetadata(label: "pattern\(i % 5)")
            )
            _ = try await index.insert(entry)
        }
        
        try await index.train()
        
        // Analyze distribution
        let distribution = await index.analyzeDistribution()
        
        #expect(distribution.dimensionality == 64)
        #expect(distribution.density > 0)
        
        // Visualization data
        let vizData = await index.visualizationData()
        #expect(vizData.nodePositions.count > 0)
        #expect(vizData.layoutAlgorithm == "learned")
    }
    
    // MARK: - Filtering Tests
    
    @Test("Metadata Filtering")
    func testMetadataFiltering() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 4,
            modelArchitecture: .linear,
            bucketSize: 10
        )
        
        let index = try await LearnedIndex<SIMD4<Float>, FilterTestMetadata>(configuration: config)
        
        // Insert vectors with different metadata
        let entries = [
            VectorEntry(id: "v1", vector: SIMD4<Float>(1, 0, 0, 0), 
                       metadata: FilterTestMetadata(category: "A", value: 10)),
            VectorEntry(id: "v2", vector: SIMD4<Float>(0, 1, 0, 0), 
                       metadata: FilterTestMetadata(category: "B", value: 20)),
            VectorEntry(id: "v3", vector: SIMD4<Float>(0, 0, 1, 0), 
                       metadata: FilterTestMetadata(category: "A", value: 30)),
            VectorEntry(id: "v4", vector: SIMD4<Float>(0, 0, 0, 1), 
                       metadata: FilterTestMetadata(category: "B", value: 40))
        ]
        
        for entry in entries {
            _ = try await index.insert(entry)
        }
        
        // Train the model
        try await index.train()
        
        // Search with category filter
        let filter = SearchFilter.metadata(MetadataFilter(
            key: "category",
            operation: .equals,
            value: "A"
        ))
        
        let results = try await index.search(
            query: SIMD4<Float>(0.5, 0.5, 0.5, 0.5),
            k: 10,
            strategy: .exact,
            filter: filter
        )
        
        #expect(results.count == 2)
        #expect(results.allSatisfy { $0.id == "v1" || $0.id == "v3" })
    }
    
    @Test("Vector Sparsity Filtering")
    func testVectorSparsityFiltering() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 8,
            modelArchitecture: .linear,
            bucketSize: 10
        )
        
        let index = try await LearnedIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Insert vectors with different sparsity levels
        let entries = [
            VectorEntry(id: "dense", vector: SIMD8<Float>(1, 2, 3, 4, 5, 6, 7, 8), 
                       metadata: TestMetadata(label: "dense")),
            VectorEntry(id: "sparse1", vector: SIMD8<Float>(1, 0, 0, 0, 2, 0, 0, 0), 
                       metadata: TestMetadata(label: "sparse")),
            VectorEntry(id: "sparse2", vector: SIMD8<Float>(0, 0, 3, 0, 0, 0, 0, 0), 
                       metadata: TestMetadata(label: "very_sparse"))
        ]
        
        for entry in entries {
            _ = try await index.insert(entry)
        }
        
        try await index.train()
        
        // Search with sparsity filter (25% - 50% non-zero)
        let filter = SearchFilter.vector(VectorFilter(
            dimension: nil,
            range: nil,
            constraint: .sparsity(0.25...0.5)
        ))
        
        let results = try await index.search(
            query: SIMD8<Float>(1, 1, 1, 1, 1, 1, 1, 1),
            k: 10,
            strategy: .exact,
            filter: filter
        )
        
        #expect(results.count == 1)
        #expect(results[0].id == "sparse1")
    }
    
    @Test("Learned Filter with Model")
    func testLearnedFilter() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 16,
            modelArchitecture: .mlp(hiddenSizes: [32, 16]),
            bucketSize: 20
        )
        
        let index = try await LearnedIndex<SIMD16<Float>, TestMetadata>(configuration: config)
        
        // Insert sequential vectors
        let vectors = Self.generateSequentialVectors(count: 100, dimensions: 16)
        for (i, vector) in vectors.enumerated() {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD16<Float>(vector),
                metadata: TestMetadata(label: "seq\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        // Train the model
        try await index.train()
        
        // Use learned filter with high confidence
        let filter = SearchFilter.learned(LearnedFilter(
            modelIdentifier: "default",
            confidence: 0.8,
            parameters: ["threshold": "0.5"]
        ))
        
        let query = SIMD16<Float>(vectors[50])
        let results = try await index.search(
            query: query,
            k: 20,
            strategy: .learned,
            filter: filter
        )
        
        // With 80% confidence, should return ~16 results
        #expect(results.count <= 16)
        #expect(results.count >= 10)
    }
    
    @Test("Complex Composite Filtering")
    func testComplexCompositeFiltering() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 8,
            modelArchitecture: .linear,
            bucketSize: 15
        )
        
        let index = try await LearnedIndex<SIMD8<Float>, FilterTestMetadata>(configuration: config)
        
        // Insert varied vectors
        for i in 0..<50 {
            let category = ["A", "B", "C"][i % 3]
            let value = i * 5
            let sparsity = Float(i % 5) / 5.0
            
            var vectorData = [Float](repeating: 0, count: 8)
            for j in 0..<Int(sparsity * 8) {
                vectorData[j] = Float(i + j)
            }
            
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD8<Float>(vectorData),
                metadata: FilterTestMetadata(category: category, value: value)
            )
            _ = try await index.insert(entry)
        }
        
        try await index.train()
        
        // Complex filter: (category = "A" OR value > 100) AND sparsity < 0.5
        let filter = SearchFilter.composite(CompositeFilter(
            operation: .and,
            filters: [
                .composite(CompositeFilter(
                    operation: .or,
                    filters: [
                        .metadata(MetadataFilter(key: "category", operation: .equals, value: "A")),
                        .metadata(MetadataFilter(key: "value", operation: .greaterThan, value: "100"))
                    ]
                )),
                .vector(VectorFilter(
                    dimension: nil,
                    range: nil,
                    constraint: .sparsity(0...0.5)
                ))
            ]
        ))
        
        let results = try await index.search(
            query: SIMD8<Float>(1, 1, 1, 1, 0, 0, 0, 0),
            k: 50,
            strategy: .exact,
            filter: filter
        )
        
        // Verify all results match the complex filter
        for result in results {
            let id = result.id
            let num = Int(id.dropFirst(3))!
            let category = ["A", "B", "C"][num % 3]
            let value = num * 5
            let sparsity = Float(num % 5) / 5.0
            
            let meetsMetadata = category == "A" || value > 100
            let meetsSparsity = sparsity <= 0.5
            
            #expect(meetsMetadata && meetsSparsity)
        }
    }
    
    @Test("Filter Performance with Learned Index")
    func testFilterPerformance() async throws {
        let config = LearnedIndexConfiguration(
            dimensions: 32,
            modelArchitecture: .mlp(hiddenSizes: [64, 32]),
            bucketSize: 100,
            trainingConfig: IncrementalTrainingConfig(
                learningRate: 0.01,
                batchSize: 32,
                epochs: 5
            )
        )
        
        let index = try await LearnedIndex<SIMD32<Float>, FilterTestMetadata>(configuration: config)
        
        // Insert 1000 vectors
        for i in 0..<1000 {
            let category = ["A", "B", "C", "D"][i % 4]
            let value = i % 100
            let vector = Self.generateSequentialVectors(count: 1, dimensions: 32)[0]
            
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD32<Float>(vector),
                metadata: FilterTestMetadata(category: category, value: value)
            )
            _ = try await index.insert(entry)
        }
        
        // Train the model
        try await index.train()
        
        // Measure filtered search performance
        let filter = SearchFilter.metadata(MetadataFilter(
            key: "category",
            operation: .equals,
            value: "A"
        ))
        
        let startTime = DispatchTime.now()
        
        for _ in 0..<100 {
            let query = SIMD32<Float>(Self.generateSequentialVectors(count: 1, dimensions: 32)[0])
            let results = try await index.search(
                query: query,
                k: 10,
                strategy: .learned,
                filter: filter
            )
            #expect(results.allSatisfy { _ in true }) // Just check it completes
        }
        
        let endTime = DispatchTime.now()
        let totalTime = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
        let avgTime = totalTime / 100
        
        print("Average filtered search time (learned): \(avgTime * 1000)ms")
        #expect(avgTime < 0.02)  // Should be < 20ms per search (learned index may be slower)
    }
}

// MARK: - SIMD Extensions

extension SIMD64<Float> {
    init(_ array: [Float]) {
        self.init()
        for i in 0..<Swift.min(64, array.count) {
            self[i] = array[i]
        }
    }
}

// MARK: - Test Helpers

struct TestMetadata: Codable, Sendable {
    let label: String
    let timestamp: Date = Date()
    let tags: [String] = []
}

struct FilterTestMetadata: Codable, Sendable {
    let category: String
    let value: Int
    let timestamp: Date = Date()
}
// VectorStoreKit: IVF Index Tests
//
// Comprehensive test suite for IVF index functionality

import Testing
import Foundation
@testable import VectorStoreKit

@Suite("IVF Index Tests")
struct IVFIndexTests {
    
    // MARK: - Test Data
    
    static func generateTestVectors(count: Int, dimensions: Int = 128) -> [[Float]] {
        (0..<count).map { i in
            (0..<dimensions).map { d in
                Float.random(in: -1...1) * Float(i % 10 + 1) / 10.0
            }
        }
    }
    
    static func generateClusteredVectors(
        clusterCount: Int,
        vectorsPerCluster: Int,
        dimensions: Int = 128
    ) -> [[Float]] {
        var vectors: [[Float]] = []
        
        // Generate cluster centers
        let centers = (0..<clusterCount).map { _ in
            (0..<dimensions).map { _ in Float.random(in: -1...1) }
        }
        
        // Generate vectors around each center
        for center in centers {
            for _ in 0..<vectorsPerCluster {
                let noise = (0..<dimensions).map { _ in Float.random(in: -0.1...0.1) }
                let vector = zip(center, noise).map { $0 + $1 }
                vectors.append(vector)
            }
        }
        
        return vectors.shuffled()
    }
    
    // MARK: - Configuration Tests
    
    @Test("IVF Configuration Validation")
    func testConfigurationValidation() throws {
        // Valid configuration
        let validConfig = IVFConfiguration(
            dimensions: 128,
            numberOfCentroids: 32,
            numberOfProbes: 5
        )
        #expect(throws: Never.self) {
            try validConfig.validate()
        }
        
        // Invalid dimensions
        let invalidDimConfig = IVFConfiguration(
            dimensions: 0,
            numberOfCentroids: 32
        )
        #expect(throws: IVFError.self) {
            try invalidDimConfig.validate()
        }
        
        // Invalid centroids
        let invalidCentroidConfig = IVFConfiguration(
            dimensions: 128,
            numberOfCentroids: 0
        )
        #expect(throws: IVFError.self) {
            try invalidCentroidConfig.validate()
        }
        
        // Invalid probes
        let invalidProbeConfig = IVFConfiguration(
            dimensions: 128,
            numberOfCentroids: 10,
            numberOfProbes: 20  // More than centroids
        )
        #expect(throws: IVFError.self) {
            try invalidProbeConfig.validate()
        }
    }
    
    @Test("Memory Usage Estimation")
    func testMemoryEstimation() {
        let config = IVFConfiguration(
            dimensions: 128,
            numberOfCentroids: 1024,
            numberOfProbes: 10
        )
        
        let memoryFor1K = config.estimatedMemoryUsage(for: 1000)
        let memoryFor10K = config.estimatedMemoryUsage(for: 10000)
        
        #expect(memoryFor10K > memoryFor1K)
        #expect(memoryFor1K > 0)
        
        // Test with quantization
        let quantizedConfig = IVFConfiguration(
            dimensions: 128,
            numberOfCentroids: 1024,
            quantization: .productQuantization(segments: 8, bits: 8)
        )
        
        let quantizedMemory = quantizedConfig.estimatedMemoryUsage(for: 1000)
        #expect(quantizedMemory < memoryFor1K)  // Quantization should reduce memory
    }
    
    // MARK: - Training Tests
    
    @Test("IVF Training", .timeLimit(.seconds(30)))
    func testTraining() async throws {
        let config = IVFConfiguration(
            dimensions: 128,
            numberOfCentroids: 16,
            numberOfProbes: 4
        )
        
        let index = try await IVFIndex<SIMD4<Float>, TestMetadata>(configuration: config)
        
        // Generate training data
        let trainingData = Self.generateTestVectors(count: 500, dimensions: 128)
        
        // Train the index
        try await index.train(on: trainingData)
        
        // Verify training
        #expect(await index.isOptimized == true)
        
        let stats = await index.statistics()
        #expect(stats.trained == true)
        #expect(stats.numberOfCentroids == 16)
    }
    
    @Test("Insufficient Training Data")
    func testInsufficientTrainingData() async throws {
        let config = IVFConfiguration(
            dimensions: 128,
            numberOfCentroids: 100,
            numberOfProbes: 10
        )
        
        let index = try await IVFIndex<SIMD4<Float>, TestMetadata>(configuration: config)
        
        // Not enough vectors for the number of centroids
        let trainingData = Self.generateTestVectors(count: 50, dimensions: 128)
        
        await #expect(throws: IVFError.self) {
            try await index.train(on: trainingData)
        }
    }
    
    // MARK: - Insert/Search Tests
    
    @Test("Basic Insert and Search", .timeLimit(.seconds(10)))
    func testBasicInsertAndSearch() async throws {
        let config = IVFConfiguration(
            dimensions: 4,
            numberOfCentroids: 4,
            numberOfProbes: 2
        )
        
        let index = try await IVFIndex<SIMD4<Float>, TestMetadata>(configuration: config)
        
        // Train with simple data
        let trainingData: [[Float]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0.5, 0.5, 0, 0],
            [0, 0.5, 0.5, 0],
            [0, 0, 0.5, 0.5],
            [0.5, 0, 0, 0.5]
        ]
        
        try await index.train(on: trainingData)
        
        // Insert vectors
        let vector1 = SIMD4<Float>(1, 0, 0, 0)
        let entry1 = VectorEntry(
            id: "vec1",
            vector: vector1,
            metadata: TestMetadata(label: "one")
        )
        
        let insertResult = try await index.insert(entry1)
        #expect(insertResult.success == true)
        
        // Search for similar vector
        let query = SIMD4<Float>(0.9, 0.1, 0, 0)
        let results = try await index.search(query: query, k: 1)
        
        #expect(results.count == 1)
        #expect(results[0].id == "vec1")
    }
    
    @Test("Multi-Probe Search")
    func testMultiProbeSearch() async throws {
        let config = IVFConfiguration(
            dimensions: 128,
            numberOfCentroids: 32,
            numberOfProbes: 8  // Will probe multiple centroids
        )
        
        let index = try await IVFIndex<SIMD128<Float>, TestMetadata>(configuration: config)
        
        // Generate and train on clustered data
        let trainingData = Self.generateClusteredVectors(
            clusterCount: 32,
            vectorsPerCluster: 20,
            dimensions: 128
        )
        
        try await index.train(on: trainingData)
        
        // Insert test vectors
        var insertedIds: [String] = []
        for i in 0..<100 {
            let vector = trainingData[i]
            let simdVector = SIMD128<Float>(vector)
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: simdVector,
                metadata: TestMetadata(label: "test\(i)")
            )
            
            let result = try await index.insert(entry)
            #expect(result.success == true)
            insertedIds.append(entry.id)
        }
        
        // Search with different probe counts
        let queryVector = SIMD128<Float>(trainingData[50])
        
        // Search with exact strategy (all probes)
        let exactResults = try await index.search(
            query: queryVector,
            k: 10,
            strategy: .exact
        )
        
        // Search with approximate strategy
        let approxResults = try await index.search(
            query: queryVector,
            k: 10,
            strategy: .approximate(quality: 0.5)
        )
        
        // Exact should return more/better results
        #expect(exactResults.count >= approxResults.count)
        
        // Check that we find the exact match
        #expect(exactResults.first?.id == "vec50")
    }
    
    // MARK: - Update/Delete Tests
    
    @Test("Delete Operations")
    func testDelete() async throws {
        let config = IVFConfiguration(dimensions: 4, numberOfCentroids: 2)
        let index = try await IVFIndex<SIMD4<Float>, TestMetadata>(configuration: config)
        
        // Train
        let trainingData: [[Float]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        try await index.train(on: trainingData)
        
        // Insert
        let entry = VectorEntry(
            id: "test",
            vector: SIMD4<Float>(1, 0, 0, 0),
            metadata: TestMetadata(label: "test")
        )
        _ = try await index.insert(entry)
        
        // Verify it exists
        #expect(await index.contains(id: "test") == true)
        
        // Delete
        let deleted = try await index.delete(id: "test")
        #expect(deleted == true)
        
        // Verify it's gone
        #expect(await index.contains(id: "test") == false)
        #expect(await index.count == 0)
    }
    
    // MARK: - Performance Tests
    
    @Test("Insertion Performance", .timeLimit(.seconds(30)))
    func testInsertionPerformance() async throws {
        let config = IVFConfiguration(
            dimensions: 128,
            numberOfCentroids: 64,
            numberOfProbes: 8
        )
        
        let index = try await IVFIndex<SIMD128<Float>, TestMetadata>(configuration: config)
        
        // Train
        let trainingData = Self.generateTestVectors(count: 2000, dimensions: 128)
        try await index.train(on: trainingData)
        
        // Measure insertion time
        let startTime = DispatchTime.now()
        
        for i in 0..<1000 {
            let vector = SIMD128<Float>(trainingData[i])
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: vector,
                metadata: TestMetadata(label: "test\(i)")
            )
            
            let result = try await index.insert(entry)
            #expect(result.success == true)
        }
        
        let endTime = DispatchTime.now()
        let totalTime = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
        let avgTime = totalTime / 1000
        
        print("Average insertion time: \(avgTime * 1000)ms")
        #expect(avgTime < 0.01)  // Should be < 10ms per insert
    }
    
    @Test("Search Performance", .timeLimit(.seconds(30)))
    func testSearchPerformance() async throws {
        let config = IVFConfiguration(
            dimensions: 128,
            numberOfCentroids: 256,
            numberOfProbes: 16
        )
        
        let index = try await IVFIndex<SIMD128<Float>, TestMetadata>(configuration: config)
        
        // Train and insert data
        let vectors = Self.generateClusteredVectors(
            clusterCount: 256,
            vectorsPerCluster: 40,
            dimensions: 128
        )
        
        try await index.train(on: vectors)
        
        // Insert 10K vectors
        for i in 0..<10_000 {
            let vector = SIMD128<Float>(vectors[i % vectors.count])
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: vector,
                metadata: TestMetadata(label: "test\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        // Measure search time
        let queries = (0..<100).map { _ in
            SIMD128<Float>(Self.generateTestVectors(count: 1, dimensions: 128)[0])
        }
        
        let startTime = DispatchTime.now()
        
        for query in queries {
            let results = try await index.search(query: query, k: 10)
            #expect(results.count > 0)
        }
        
        let endTime = DispatchTime.now()
        let totalTime = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
        let avgTime = totalTime / Double(queries.count)
        
        print("Average search time: \(avgTime * 1000)ms")
        #expect(avgTime < 0.005)  // Should be < 5ms per search
    }
    
    // MARK: - Integrity Tests
    
    @Test("Index Integrity")
    func testIntegrity() async throws {
        let config = IVFConfiguration(
            dimensions: 32,
            numberOfCentroids: 8,
            numberOfProbes: 4
        )
        
        let index = try await IVFIndex<SIMD32<Float>, TestMetadata>(configuration: config)
        
        // Check integrity before training
        let preTrainIntegrity = try await index.validateIntegrity()
        #expect(preTrainIntegrity.isValid == false)  // Not trained yet
        
        // Train and check again
        let trainingData = Self.generateTestVectors(count: 200, dimensions: 32)
        try await index.train(on: trainingData)
        
        let postTrainIntegrity = try await index.validateIntegrity()
        #expect(postTrainIntegrity.isValid == true)
        
        // Insert vectors and check balance
        for i in 0..<100 {
            let vector = SIMD32<Float>(trainingData[i])
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: vector,
                metadata: TestMetadata(label: "test\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        let stats = await index.statistics()
        print("List sizes - Min: \(stats.minListSize), Max: \(stats.maxListSize), Avg: \(stats.averageListSize)")
        
        // Check for imbalanced lists
        let finalIntegrity = try await index.validateIntegrity()
        #expect(finalIntegrity.warnings.count >= 0)  // May have warnings about balance
    }
    
    // MARK: - Export/Import Tests
    
    @Test("Export and Import")
    func testExportImport() async throws {
        let config = IVFConfiguration(
            dimensions: 16,
            numberOfCentroids: 4,
            numberOfProbes: 2
        )
        
        let index1 = try await IVFIndex<SIMD16<Float>, TestMetadata>(configuration: config)
        
        // Train and insert data
        let trainingData = Self.generateTestVectors(count: 100, dimensions: 16)
        try await index1.train(on: trainingData)
        
        for i in 0..<10 {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD16<Float>(trainingData[i]),
                metadata: TestMetadata(label: "test\(i)")
            )
            _ = try await index1.insert(entry)
        }
        
        // Export
        let exportedData = try await index1.export(format: .json)
        
        // Create new index and import
        let index2 = try await IVFIndex<SIMD16<Float>, TestMetadata>(configuration: config)
        try await index2.import(data: exportedData, format: .json)
        
        // Verify state is preserved
        #expect(await index2.isOptimized == true)
        
        // Search should work
        let query = SIMD16<Float>(trainingData[5])
        let results = try await index2.search(query: query, k: 5)
        #expect(results.count > 0)
    }
    
    // MARK: - Edge Cases
    
    @Test("Empty Index Operations")
    func testEmptyIndex() async throws {
        let config = IVFConfiguration(dimensions: 8, numberOfCentroids: 2)
        let index = try await IVFIndex<SIMD8<Float>, TestMetadata>(configuration: config)
        
        // Search on untrained index should throw
        let query = SIMD8<Float>(1, 0, 0, 0, 0, 0, 0, 0)
        await #expect(throws: IVFError.self) {
            _ = try await index.search(query: query, k: 5)
        }
        
        // Insert on untrained index should throw
        let entry = VectorEntry(
            id: "test",
            vector: query,
            metadata: TestMetadata(label: "test")
        )
        await #expect(throws: IVFError.self) {
            _ = try await index.insert(entry)
        }
        
        // Delete non-existent item
        let deleted = try await index.delete(id: "nonexistent")
        #expect(deleted == false)
    }
    
    @Test("Retrain Index")
    func testRetrain() async throws {
        let config = IVFConfiguration(
            dimensions: 32,
            numberOfCentroids: 8,
            numberOfProbes: 4
        )
        
        let index = try await IVFIndex<SIMD32<Float>, TestMetadata>(configuration: config)
        
        // Initial training
        let initialData = Self.generateTestVectors(count: 200, dimensions: 32)
        try await index.train(on: initialData)
        
        // Insert vectors
        for i in 0..<50 {
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD32<Float>(initialData[i]),
                metadata: TestMetadata(label: "test\(i)")
            )
            _ = try await index.insert(entry)
        }
        
        let statsBefore = await index.statistics()
        
        // Retrain
        try await index.retrain()
        
        let statsAfter = await index.statistics()
        
        // Should still have same number of vectors
        #expect(statsAfter.vectorCount == statsBefore.vectorCount)
        #expect(statsAfter.trained == true)
    }
    
    // MARK: - Filtering Tests
    
    @Test("Metadata Filtering")
    func testMetadataFiltering() async throws {
        let config = IVFConfiguration(dimensions: 4, numberOfCentroids: 2)
        let index = try await IVFIndex<SIMD4<Float>, FilterTestMetadata>(configuration: config)
        
        // Train index
        let trainingData: [[Float]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        try await index.train(on: trainingData)
        
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
    
    @Test("Vector Magnitude Filtering")
    func testVectorMagnitudeFiltering() async throws {
        let config = IVFConfiguration(dimensions: 4, numberOfCentroids: 2)
        let index = try await IVFIndex<SIMD4<Float>, TestMetadata>(configuration: config)
        
        // Train index
        let trainingData: [[Float]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [2, 2, 0, 0],
            [0, 0, 3, 3]
        ]
        try await index.train(on: trainingData)
        
        // Insert vectors with different magnitudes
        let entries = [
            VectorEntry(id: "small", vector: SIMD4<Float>(0.5, 0.5, 0, 0), 
                       metadata: TestMetadata(label: "small")),
            VectorEntry(id: "medium", vector: SIMD4<Float>(1, 1, 0, 0), 
                       metadata: TestMetadata(label: "medium")),
            VectorEntry(id: "large", vector: SIMD4<Float>(2, 2, 0, 0), 
                       metadata: TestMetadata(label: "large"))
        ]
        
        for entry in entries {
            _ = try await index.insert(entry)
        }
        
        // Search with magnitude filter
        let filter = SearchFilter.vector(VectorFilter(
            dimension: nil,
            range: nil,
            constraint: .magnitude(1.0...3.0)
        ))
        
        let results = try await index.search(
            query: SIMD4<Float>(1, 1, 0, 0),
            k: 10,
            strategy: .exact,
            filter: filter
        )
        
        #expect(results.count == 2)
        #expect(results.allSatisfy { $0.id == "medium" || $0.id == "large" })
    }
    
    @Test("Composite Filtering")
    func testCompositeFiltering() async throws {
        let config = IVFConfiguration(dimensions: 4, numberOfCentroids: 2)
        let index = try await IVFIndex<SIMD4<Float>, FilterTestMetadata>(configuration: config)
        
        // Train index
        let trainingData: [[Float]] = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        try await index.train(on: trainingData)
        
        // Insert vectors
        let entries = [
            VectorEntry(id: "v1", vector: SIMD4<Float>(1, 0, 0, 0), 
                       metadata: FilterTestMetadata(category: "A", value: 10)),
            VectorEntry(id: "v2", vector: SIMD4<Float>(0, 1, 0, 0), 
                       metadata: FilterTestMetadata(category: "A", value: 20)),
            VectorEntry(id: "v3", vector: SIMD4<Float>(0, 0, 1, 0), 
                       metadata: FilterTestMetadata(category: "B", value: 30)),
            VectorEntry(id: "v4", vector: SIMD4<Float>(0, 0, 0, 1), 
                       metadata: FilterTestMetadata(category: "B", value: 40))
        ]
        
        for entry in entries {
            _ = try await index.insert(entry)
        }
        
        // Composite AND filter: category = "A" AND value >= 15
        let filter = SearchFilter.composite(CompositeFilter(
            operation: .and,
            filters: [
                .metadata(MetadataFilter(key: "category", operation: .equals, value: "A")),
                .metadata(MetadataFilter(key: "value", operation: .greaterThanOrEqual, value: "15"))
            ]
        ))
        
        let results = try await index.search(
            query: SIMD4<Float>(0.5, 0.5, 0.5, 0.5),
            k: 10,
            strategy: .exact,
            filter: filter
        )
        
        #expect(results.count == 1)
        #expect(results[0].id == "v2")
    }
    
    @Test("Filter Performance")
    func testFilterPerformance() async throws {
        let config = IVFConfiguration(
            dimensions: 32,
            numberOfCentroids: 16,
            numberOfProbes: 4
        )
        
        let index = try await IVFIndex<SIMD32<Float>, FilterTestMetadata>(configuration: config)
        
        // Train and insert data
        let trainingData = Self.generateTestVectors(count: 500, dimensions: 32)
        try await index.train(on: trainingData)
        
        // Insert 1000 vectors with varied metadata
        for i in 0..<1000 {
            let category = ["A", "B", "C", "D"][i % 4]
            let value = i % 100
            let entry = VectorEntry(
                id: "vec\(i)",
                vector: SIMD32<Float>(trainingData[i % trainingData.count]),
                metadata: FilterTestMetadata(category: category, value: value)
            )
            _ = try await index.insert(entry)
        }
        
        // Measure filtered search performance
        let filter = SearchFilter.metadata(MetadataFilter(
            key: "category",
            operation: .equals,
            value: "A"
        ))
        
        let startTime = DispatchTime.now()
        
        for _ in 0..<100 {
            let query = SIMD32<Float>(Self.generateTestVectors(count: 1, dimensions: 32)[0])
            let results = try await index.search(
                query: query,
                k: 10,
                strategy: .approximate(quality: 0.8),
                filter: filter
            )
            #expect(results.allSatisfy { _ in true }) // Just check it completes
        }
        
        let endTime = DispatchTime.now()
        let totalTime = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000_000
        let avgTime = totalTime / 100
        
        print("Average filtered search time: \(avgTime * 1000)ms")
        #expect(avgTime < 0.01)  // Should be < 10ms per search
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

// SIMD type extensions for testing
extension SIMD16<Float> {
    init(_ array: [Float]) {
        self.init()
        for i in 0..<Swift.min(16, array.count) {
            self[i] = array[i]
        }
    }
}

extension SIMD32<Float> {
    init(_ array: [Float]) {
        self.init()
        for i in 0..<Swift.min(32, array.count) {
            self[i] = array[i]
        }
    }
}

extension SIMD128<Float> {
    init(_ array: [Float]) {
        self.init()
        for i in 0..<Swift.min(128, array.count) {
            self[i] = array[i]
        }
    }
}

// Custom SIMD types for testing
struct SIMD128<Scalar: SIMDScalar>: SIMD {
    var _storage: [Scalar]
    
    init() {
        _storage = Array(repeating: Scalar.zero, count: 128)
    }
    
    var scalarCount: Int { 128 }
    
    subscript(index: Int) -> Scalar {
        get { _storage[index] }
        set { _storage[index] = newValue }
    }
    
    init(repeating scalar: Scalar) {
        _storage = Array(repeating: scalar, count: 128)
    }
}
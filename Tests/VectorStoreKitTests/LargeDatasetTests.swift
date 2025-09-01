// LargeDatasetTests.swift
// VectorStoreKitTests
//
// Tests for large dataset handling (100K-1M vectors)

import XCTest
@testable import VectorStoreKit
import Metal

final class LargeDatasetTests: XCTestCase {
    
    var metalDevice: MetalDevice!
    var bufferPool: MetalBufferPool!
    var tempDirectory: URL!
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Setup Metal if available
        if let device = MTLCreateSystemDefaultDevice() {
            metalDevice = try await MetalDevice(device: device)
            let config = MetalBufferPoolConfiguration(
                initialSize: 100,
                maxSize: 1000
            )
            bufferPool = try await MetalBufferPool(device: metalDevice, configuration: config)
        }
        
        // Create temp directory for test files
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: tempDirectory,
            withIntermediateDirectories: true
        )
    }
    
    override func tearDown() async throws {
        // Cleanup
        if let pool = bufferPool {
            await pool.shutdown()
        }
        
        try? FileManager.default.removeItem(at: tempDirectory)
        
        try await super.tearDown()
    }
    
    // MARK: - StreamingBufferManager Tests
    
    func testStreamingBufferManagerInitialization() async throws {
        guard let device = metalDevice?.device else {
            throw XCTSkip("Metal not available")
        }
        
        let manager = try await StreamingBufferManager(
            targetSize: 100_000_000, // 100MB
            device: device
        )
        
        XCTAssertNotNil(manager)
        
        let stats = await manager.getStatistics()
        XCTAssertEqual(stats.activeBufferCount, 0)
        XCTAssertEqual(stats.totalBytesStreamed, 0)
    }
    
    func testStreamingBatch() async throws {
        guard let device = metalDevice?.device else {
            throw XCTSkip("Metal not available")
        }
        
        // Create test data file
        let vectorCount = 1000
        let dimension = 512
        let data = createTestVectorData(count: vectorCount, dimension: dimension)
        let fileURL = tempDirectory.appendingPathComponent("test_vectors.bin")
        try data.write(to: fileURL)
        
        let manager = try await StreamingBufferManager(
            targetSize: data.count,
            device: device
        )
        
        try await manager.initializeStorageTiers(
            hotTierURL: nil,
            warmTierURL: fileURL,
            coldTierURL: nil
        )
        
        // Stream a batch
        let buffer = try await manager.streamBatch(
            range: 0..<100,
            vectorDimension: dimension,
            tier: .warm
        )
        
        XCTAssertNotNil(buffer)
        XCTAssertEqual(buffer.length, 100 * dimension * MemoryLayout<Float>.size)
        
        let stats = await manager.getStatistics()
        XCTAssertEqual(stats.activeBufferCount, 1)
        XCTAssertGreaterThan(stats.totalBytesStreamed, 0)
    }
    
    func testStreamingMemoryPressure() async throws {
        guard let device = metalDevice?.device else {
            throw XCTSkip("Metal not available")
        }
        
        let manager = try await StreamingBufferManager(
            targetSize: 100_000_000,
            device: device
        )
        
        // Simulate memory pressure
        await manager.handleMemoryPressure()
        
        let stats = await manager.getStatistics()
        XCTAssertEqual(stats.activeBufferCount, 0)
    }
    
    // MARK: - HierarchicalIndex Tests
    
    func testHierarchicalIndexCreation() async throws {
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: HierarchicalConfiguration(
                topLevelClusters: 10,
                leafIndexSize: 100
            )
        )
        
        let stats = await index.getStatistics()
        XCTAssertEqual(stats.totalVectors, 0)
        XCTAssertEqual(stats.numClusters, 10)
    }
    
    func testHierarchicalIndexInsertion() async throws {
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: HierarchicalConfiguration(
                topLevelClusters: 10,
                leafIndexSize: 100
            )
        )
        
        // Insert test vectors
        let vectors = (0..<100).map { i in
            Vector512(repeating: Float(i))
        }
        
        let entries = vectors.enumerated().map { index, vector in
            (vector, "vector_\(index)", nil as Data?)
        }
        
        try await index.insertBatch(entries)
        
        let stats = await index.getStatistics()
        XCTAssertEqual(stats.totalVectors, 100)
        XCTAssertGreaterThan(stats.numActiveClusters, 0)
    }
    
    func testHierarchicalIndexSearch() async throws {
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: HierarchicalConfiguration(
                topLevelClusters: 10,
                leafIndexSize: 100,
                probesPerQuery: 2
            )
        )
        
        // Insert vectors
        let vectors = (0..<1000).map { i in
            var v = Vector512()
            v[i % 512] = Float(i)
            return v
        }
        
        let entries = vectors.enumerated().map { index, vector in
            (vector, "vector_\(index)", nil as Data?)
        }
        
        try await index.insertBatch(entries)
        
        // Search
        let query = vectors[50]
        let results = try await index.search(query: query, k: 10)
        
        XCTAssertEqual(results.count, 10)
        XCTAssertEqual(results[0].id, "vector_50") // Should find itself
        XCTAssertEqual(results[0].distance, 0, accuracy: 0.001)
    }
    
    func testHierarchicalIndexStreamInsertion() async throws {
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: HierarchicalConfiguration(
                topLevelClusters: 10,
                leafIndexSize: 50
            )
        )
        
        // Create async stream of vectors
        let stream = AsyncStream<(Vector512, VectorID, Data?)> { continuation in
            Task {
                for i in 0..<200 {
                    let vector = Vector512(repeating: Float(i))
                    continuation.yield((vector, "vector_\(i)", nil))
                }
                continuation.finish()
            }
        }
        
        try await index.insertStream(stream)
        
        let stats = await index.getStatistics()
        XCTAssertEqual(stats.totalVectors, 200)
    }
    
    // MARK: - ScalarQuantizer Tests
    
    func testScalarQuantizerInt8() async throws {
        let quantizer = ScalarQuantizer()
        
        // Create test vector
        let vector = (0..<512).map { Float($0) / 512.0 - 0.5 }
        
        let quantized = try await quantizer.quantize(
            vector: vector,
            type: .int8
        )
        
        XCTAssertEqual(quantized.quantizationType, .int8)
        XCTAssertEqual(quantized.compressionRatio, 4.0, accuracy: 0.1)
        
        // Test dequantization
        let dequantized = try await quantizer.dequantize(quantized)
        
        // Check accuracy
        for i in 0..<vector.count {
            XCTAssertEqual(vector[i], dequantized[i], accuracy: 0.01)
        }
    }
    
    func testScalarQuantizerFloat16() async throws {
        let quantizer = ScalarQuantizer()
        
        let vectors = (0..<100).map { i in
            (0..<512).map { j in Float(i * 512 + j) / 51200.0 }
        }
        
        let quantizedBatch = try await quantizer.quantizeBatch(
            vectors: vectors,
            type: .float16
        )
        
        XCTAssertEqual(quantizedBatch.count, 100)
        XCTAssertEqual(quantizedBatch[0].compressionRatio, 2.0, accuracy: 0.1)
        
        // Test batch dequantization
        let dequantizedBatch = try await quantizer.dequantizeBatch(quantizedBatch)
        
        XCTAssertEqual(dequantizedBatch.count, 100)
        
        // Check first and last vectors
        for i in 0..<512 {
            XCTAssertEqual(vectors[0][i], dequantizedBatch[0][i], accuracy: 0.001)
            XCTAssertEqual(vectors[99][i], dequantizedBatch[99][i], accuracy: 0.001)
        }
    }
    
    func testScalarQuantizerDynamic() async throws {
        let quantizer = ScalarQuantizer()
        
        // Vector with small range - should select int8
        let smallRangeVector = (0..<512).map { _ in Float.random(in: -1...1) }
        
        let quantized = try await quantizer.quantize(
            vector: smallRangeVector,
            type: .dynamic
        )
        
        // Should select int8 for small range
        XCTAssertTrue(
            quantized.quantizationType == .int8 || 
            quantized.quantizationType == .uint8
        )
    }
    
    func testQuantizerMemorySavings() async throws {
        let quantizer = ScalarQuantizer()
        
        let vectorCount = 10000
        let dimension = 512
        
        let savings = await quantizer.calculateMemorySavings(
            vectorCount: vectorCount,
            dimension: dimension,
            quantizationType: .int8
        )
        
        XCTAssertEqual(savings.originalSize, vectorCount * dimension * 4)
        XCTAssertEqual(savings.quantizedSize, vectorCount * dimension * 1)
        XCTAssertEqual(savings.compressionRatio, 4.0)
    }
    
    // MARK: - BatchProcessor Tests
    
    func testBatchProcessorConfiguration() {
        let config = BatchProcessingConfiguration()
        
        XCTAssertGreaterThan(config.optimalBatchSize, 0)
        XCTAssertEqual(config.maxConcurrentBatches, 4)
        XCTAssertTrue(config.useMetalAcceleration)
    }
    
    func testBatchProcessorSimple() async throws {
        let processor = BatchProcessor()
        
        // Create test dataset
        let vectors = (0..<1000).map { i in
            Vector512(repeating: Float(i))
        }
        
        let dataset = Vector512Dataset(vectors: vectors)
        
        // Process with transformation
        let results: [Float] = try await processor.processBatches(
            dataset: dataset,
            operation: .transformation { entry in
                let (vector, _, _) = entry as! (Vector512, VectorID, Data?)
                return vector[0] // Return first element
            }
        )
        
        XCTAssertEqual(results.count, 1000)
        XCTAssertEqual(results[0], 0)
        XCTAssertEqual(results[999], 999)
    }
    
    func testBatchProcessorWithProgress() async throws {
        let processor = BatchProcessor()
        
        let vectors = (0..<10000).map { i in
            Vector512(repeating: Float(i % 100))
        }
        
        let dataset = Vector512Dataset(vectors: vectors)
        
        var progressUpdates: [BatchProgress] = []
        
        let _: [Any] = try await processor.processBatches(
            dataset: dataset,
            operation: .filtering { entry in
                let (vector, _, _) = entry as! (Vector512, VectorID, Data?)
                return vector[0] < 50
            },
            progressHandler: { progress in
                progressUpdates.append(progress)
            }
        )
        
        XCTAssertGreaterThan(progressUpdates.count, 0)
        
        // Check final progress
        if let lastProgress = progressUpdates.last {
            XCTAssertEqual(lastProgress.totalItems, 10000)
            XCTAssertEqual(lastProgress.processedItems, 10000)
            XCTAssertEqual(lastProgress.percentComplete, 100, accuracy: 0.1)
        }
    }
    
    func testBatchProcessorStreaming() async throws {
        let processor = BatchProcessor()
        
        // Create streaming source
        let stream = AsyncStream<Vector512> { continuation in
            Task {
                for i in 0..<1000 {
                    continuation.yield(Vector512(repeating: Float(i)))
                }
                continuation.finish()
            }
        }
        
        var resultCount = 0
        
        let resultStream = processor.streamProcessBatches(
            stream: stream,
            batchSize: 100,
            operation: .transformation { vector in
                return (vector as! Vector512)[0]
            }
        )
        
        for try await batch in resultStream {
            resultCount += batch.count
        }
        
        XCTAssertEqual(resultCount, 1000)
    }
    
    // MARK: - Performance Tests
    
    func testLargeDatasetPerformance() async throws {
        let vectorCount = 100_000
        let dimension = 512
        
        measure {
            let expectation = self.expectation(description: "Large dataset processing")
            
            Task {
                let processor = BatchProcessor()
                let vectors = (0..<vectorCount).map { i in
                    Vector512(repeating: Float(i % 1000) / 1000.0)
                }
                
                let dataset = Vector512Dataset(vectors: vectors)
                
                let _: [Any] = try await processor.processBatches(
                    dataset: dataset,
                    operation: .transformation { entry in
                        let (vector, _, _) = entry as! (Vector512, VectorID, Data?)
                        return vector.normalized()
                    }
                )
                
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 60.0)
        }
    }
    
    // MARK: - Integration Tests
    
    func testEndToEndLargeDatasetProcessing() async throws {
        // This test simulates a complete workflow for large dataset
        
        // 1. Create hierarchical index
        let index = try await HierarchicalIndex<Vector512, Data>(
            dimension: 512,
            configuration: HierarchicalConfiguration.forDatasetSize(100_000)
        )
        
        // 2. Generate test data
        let vectors = (0..<10_000).map { i in
            var v = Vector512()
            // Create clustered data
            let cluster = i / 1000
            v[cluster] = Float(i)
            return v.normalized()
        }
        
        // 3. Quantize for storage efficiency
        let quantizer = ScalarQuantizer()
        let quantizedVectors = try await quantizer.quantize512(
            vectors: vectors,
            type: .float16
        )
        
        print("Compression ratio: \(quantizedVectors[0].compressionRatio)")
        
        // 4. Batch insert into index
        let processor = BatchProcessor()
        let dataset = Vector512Dataset(vectors: vectors)
        
        let _: [Any] = try await processor.processBatches(
            dataset: dataset,
            operation: .indexing(index: index)
        )
        
        // 5. Verify index statistics
        let stats = await index.getStatistics()
        XCTAssertEqual(stats.totalVectors, 10_000)
        
        // 6. Test search on quantized data
        let query = vectors[5000]
        let results = try await index.search(query: query, k: 10)
        
        XCTAssertEqual(results.count, 10)
        XCTAssertEqual(results[0].id, "vector_5000")
    }
    
    // MARK: - Helper Methods
    
    private func createTestVectorData(count: Int, dimension: Int) -> Data {
        var data = Data()
        data.reserveCapacity(count * dimension * MemoryLayout<Float>.size)
        
        for i in 0..<count {
            for j in 0..<dimension {
                var value = Float(i * dimension + j) / Float(count * dimension)
                data.append(Data(bytes: &value, count: MemoryLayout<Float>.size))
            }
        }
        
        return data
    }
}
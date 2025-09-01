// StreamingBufferManagerTests.swift
// VectorStoreKitTests
//
// Tests for streaming buffer management

import XCTest
@testable import VectorStoreKit
import Metal

final class StreamingBufferManagerTests: XCTestCase {
    
    var tempDirectory: URL!
    var bufferManager: StreamingBufferManager!
    var metalDevice: MTLDevice!
    
    override func setUp() async throws {
        try await super.setUp()
        
        // Create temp directory
        tempDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(
            at: tempDirectory,
            withIntermediateDirectories: true
        )
        
        // Get Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }
        metalDevice = device
        
        // Create buffer manager
        bufferManager = try await StreamingBufferManager(
            device: metalDevice,
            targetSize: 1_000_000, // 1MB target
            pageSize: 4096
        )
    }
    
    override func tearDown() async throws {
        // Clean up temp directory
        if let tempDirectory = tempDirectory {
            try? FileManager.default.removeItem(at: tempDirectory)
        }
        
        try await super.tearDown()
    }
    
    // MARK: - File Tests
    
    func testAddAndLoadFile() async throws {
        // Create test data
        let vectorCount = 100
        let dimension = 512
        let vectors = (0..<vectorCount).map { i in
            Vector512(repeating: Float(i))
        }
        
        // Write vectors to file
        let fileURL = tempDirectory.appendingPathComponent("test_vectors.bin")
        try await writeVectorsToFile(vectors, url: fileURL)
        
        // Add file to manager
        try await bufferManager.addFile(
            url: fileURL,
            tier: .hot,
            metadata: StreamingBufferManager.FileMetadata(
                vectorCount: vectorCount,
                dimension: dimension,
                vectorSize: MemoryLayout<Float>.size * dimension
            )
        )
        
        // Load some vectors
        let loadedData = try await bufferManager.loadVectors(
            tier: .hot,
            offset: 0,
            count: 10
        )
        
        XCTAssertNotNil(loadedData)
        XCTAssertEqual(loadedData.count, 10 * dimension * MemoryLayout<Float>.size)
    }
    
    func testMultipleTiers() async throws {
        // Create files for different tiers
        let tiers: [(StorageTier, String)] = [
            (.hot, "hot_vectors.bin"),
            (.warm, "warm_vectors.bin"),
            (.cold, "cold_vectors.bin")
        ]
        
        for (tier, filename) in tiers {
            let vectors = (0..<50).map { i in
                Vector512(repeating: Float(tier.rawValue) * 100 + Float(i))
            }
            
            let fileURL = tempDirectory.appendingPathComponent(filename)
            try await writeVectorsToFile(vectors, url: fileURL)
            
            try await bufferManager.addFile(
                url: fileURL,
                tier: tier,
                metadata: StreamingBufferManager.FileMetadata(
                    vectorCount: 50,
                    dimension: 512,
                    vectorSize: MemoryLayout<Float>.size * 512
                )
            )
        }
        
        // Verify we can load from each tier
        for (tier, _) in tiers {
            let data = try await bufferManager.loadVectors(
                tier: tier,
                offset: 0,
                count: 5
            )
            
            XCTAssertNotNil(data)
            
            // Verify first value matches tier
            data.withUnsafeBytes { bytes in
                let floatPointer = bytes.bindMemory(to: Float.self)
                let firstValue = floatPointer[0]
                XCTAssertEqual(
                    firstValue,
                    Float(tier.rawValue) * 100,
                    accuracy: 0.01
                )
            }
        }
    }
    
    // MARK: - Prefetching Tests
    
    func testPrefetching() async throws {
        // Create test file
        let vectors = (0..<1000).map { i in
            Vector512(repeating: Float(i))
        }
        
        let fileURL = tempDirectory.appendingPathComponent("prefetch_test.bin")
        try await writeVectorsToFile(vectors, url: fileURL)
        
        try await bufferManager.addFile(
            url: fileURL,
            tier: .warm,
            metadata: StreamingBufferManager.FileMetadata(
                vectorCount: 1000,
                dimension: 512,
                vectorSize: MemoryLayout<Float>.size * 512
            )
        )
        
        // Prefetch some data
        try await bufferManager.prefetchVectors(
            tier: .warm,
            ranges: [
                0..<100,
                500..<600,
                900..<1000
            ]
        )
        
        // Loading prefetched data should be fast
        let start = Date()
        let _ = try await bufferManager.loadVectors(
            tier: .warm,
            offset: 500,
            count: 100
        )
        let loadTime = Date().timeIntervalSince(start)
        
        XCTAssertLessThan(loadTime, 0.01) // Should be very fast
    }
    
    // MARK: - Metal Buffer Tests
    
    func testCreateMetalBuffer() async throws {
        // Create test data
        let vectors = (0..<10).map { i in
            Vector512(repeating: Float(i))
        }
        
        let data = Data(
            bytes: vectors.flatMap { vector in
                (0..<512).map { i in vector[i] }
            },
            count: vectors.count * 512 * MemoryLayout<Float>.size
        )
        
        // Create Metal buffer
        let buffer = try await bufferManager.createMetalBuffer(
            from: data,
            options: []
        )
        
        XCTAssertNotNil(buffer)
        XCTAssertEqual(buffer.length, data.count)
    }
    
    // MARK: - Streaming Tests
    
    func testStreamVectors() async throws {
        // Create large file
        let vectorCount = 10000
        let vectors = (0..<vectorCount).map { i in
            Vector512(repeating: Float(i) / Float(vectorCount))
        }
        
        let fileURL = tempDirectory.appendingPathComponent("stream_test.bin")
        try await writeVectorsToFile(vectors, url: fileURL)
        
        try await bufferManager.addFile(
            url: fileURL,
            tier: .warm,
            metadata: StreamingBufferManager.FileMetadata(
                vectorCount: vectorCount,
                dimension: 512,
                vectorSize: MemoryLayout<Float>.size * 512
            )
        )
        
        // Stream vectors
        let stream = await bufferManager.streamVectors(
            tier: .warm,
            batchSize: 100
        )
        
        var processedCount = 0
        
        for await batch in stream {
            XCTAssertGreaterThan(batch.count, 0)
            processedCount += batch.count / (512 * MemoryLayout<Float>.size)
        }
        
        XCTAssertEqual(processedCount, vectorCount)
    }
    
    // MARK: - Memory Management Tests
    
    func testMemoryPressure() async throws {
        // Create multiple files
        for i in 0..<5 {
            let vectors = (0..<100).map { j in
                Vector512(repeating: Float(i * 100 + j))
            }
            
            let fileURL = tempDirectory.appendingPathComponent("pressure_\(i).bin")
            try await writeVectorsToFile(vectors, url: fileURL)
            
            try await bufferManager.addFile(
                url: fileURL,
                tier: .hot,
                metadata: StreamingBufferManager.FileMetadata(
                    vectorCount: 100,
                    dimension: 512,
                    vectorSize: MemoryLayout<Float>.size * 512
                )
            )
        }
        
        // Load many vectors to trigger memory pressure
        for i in 0..<5 {
            let _ = try await bufferManager.loadVectors(
                tier: .hot,
                offset: 0,
                count: 50
            )
        }
        
        // Should handle memory pressure gracefully
        let stats = await bufferManager.getStatistics()
        XCTAssertGreaterThan(stats.totalBytesLoaded, 0)
    }
    
    // MARK: - Performance Tests
    
    func testLoadPerformance() async throws {
        // Create large file
        let vectorCount = 100000
        let batchSize = 10000
        
        let fileURL = tempDirectory.appendingPathComponent("perf_test.bin")
        
        // Write in batches to avoid memory issues
        let fileHandle = try FileHandle(forWritingTo: fileURL)
        defer { try? fileHandle.close() }
        
        for batch in 0..<(vectorCount / batchSize) {
            let vectors = (0..<batchSize).map { i in
                Vector512(repeating: Float(batch * batchSize + i))
            }
            
            let data = Data(
                bytes: vectors.flatMap { vector in
                    (0..<512).map { i in vector[i] }
                },
                count: batchSize * 512 * MemoryLayout<Float>.size
            )
            
            fileHandle.write(data)
        }
        
        try await bufferManager.addFile(
            url: fileURL,
            tier: .warm,
            metadata: StreamingBufferManager.FileMetadata(
                vectorCount: vectorCount,
                dimension: 512,
                vectorSize: MemoryLayout<Float>.size * 512
            )
        )
        
        // Measure load performance
        measure {
            let expectation = self.expectation(description: "Load complete")
            
            Task {
                // Load 10K vectors
                let _ = try await bufferManager.loadVectors(
                    tier: .warm,
                    offset: 0,
                    count: 10000
                )
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 5.0)
        }
    }
    
    // MARK: - Helper Methods
    
    private func writeVectorsToFile(_ vectors: [Vector512], url: URL) async throws {
        let data = Data(
            bytes: vectors.flatMap { vector in
                (0..<512).map { i in vector[i] }
            },
            count: vectors.count * 512 * MemoryLayout<Float>.size
        )
        
        try data.write(to: url)
    }
}
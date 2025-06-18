// VectorStoreKit Model Serialization Tests

import XCTest
@testable import VectorStoreKit
import Metal

final class ModelSerializationTests: XCTestCase {
    var device: MTLDevice!
    var metalPipeline: MetalMLPipeline!
    
    override func setUp() async throws {
        device = MTLCreateSystemDefaultDevice()
        XCTAssertNotNil(device, "Metal device not available")
        metalPipeline = try MetalMLPipeline(device: device!)
    }
    
    // MARK: - Basic Serialization Tests
    
    func testSaveAndLoadSimpleNetwork() async throws {
        // Create network
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        await network.addLayers([
            try await DenseLayer(
                inputSize: 10,
                outputSize: 5,
                activation: .relu,
                metalPipeline: metalPipeline
            )
        ])
        
        // Save checkpoint
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_checkpoint.json")
        
        try await network.saveCheckpoint(to: url)
        
        // Load into new network
        let loadedNetwork = try await NeuralNetwork(metalPipeline: metalPipeline)
        try await loadedNetwork.loadCheckpoint(from: url)
        
        // Verify
        let originalCount = await network.parameterCount()
        let loadedCount = await loadedNetwork.parameterCount()
        XCTAssertEqual(originalCount, loadedCount)
        
        // Cleanup
        try? FileManager.default.removeItem(at: url)
    }
    
    func testSaveAndLoadComplexNetwork() async throws {
        // Create network with multiple layer types
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        await network.addLayers([
            try await DenseLayer(
                inputSize: 100,
                outputSize: 50,
                activation: .relu,
                name: "dense1",
                metalPipeline: metalPipeline
            ),
            try await BatchNormLayer(
                numFeatures: 50,
                name: "bn1",
                metalPipeline: metalPipeline
            ),
            try await DropoutLayer(
                rate: 0.5,
                metalPipeline: metalPipeline
            ),
            try await DenseLayer(
                inputSize: 50,
                outputSize: 10,
                activation: .softmax,
                name: "output",
                metalPipeline: metalPipeline
            )
        ])
        
        // Train briefly to initialize batch norm statistics
        let trainingData = try await createTrainingData(inputSize: 100, outputSize: 10, samples: 10)
        let config = NetworkTrainingConfig(epochs: 1, batchSize: 1, learningRate: 0.01)
        try await network.train(data: trainingData, config: config)
        
        // Save with options
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("complex_checkpoint.json")
        
        let options = CheckpointOptions(
            compression: .zstd,
            includeTrainingState: true,
            description: "Test complex network"
        )
        
        try await network.saveCheckpoint(to: url, options: options)
        
        // Load and verify
        let loadedNetwork = try await NeuralNetwork(metalPipeline: metalPipeline)
        try await loadedNetwork.loadCheckpoint(from: url, options: options)
        
        // Test outputs match
        let testInput = try await createTestInput(size: 100)
        let originalOutput = try await network.predict(testInput)
        let loadedOutput = try await loadedNetwork.predict(testInput)
        
        XCTAssertTrue(buffersMatch(originalOutput, loadedOutput, tolerance: 1e-5))
        
        // Cleanup
        try? FileManager.default.removeItem(at: url)
    }
    
    // MARK: - Compression Tests
    
    func testCompressionAlgorithms() async throws {
        let network = try await createTestNetwork()
        
        var sizes: [CompressionAlgorithm: Int] = [:]
        
        for algorithm in [CompressionAlgorithm.none, .lz4, .zstd] {
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("compression_\(algorithm.rawValue).json")
            
            let options = CheckpointOptions(compression: algorithm)
            try await network.saveCheckpoint(to: url, options: options)
            
            let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
            sizes[algorithm] = attrs[.size] as? Int ?? 0
            
            // Verify can load
            let loaded = try await NeuralNetwork(metalPipeline: metalPipeline)
            try await loaded.loadCheckpoint(from: url, options: options)
            
            try? FileManager.default.removeItem(at: url)
        }
        
        // Verify compression actually compresses
        XCTAssertLessThan(sizes[.lz4]!, sizes[.none]!)
        XCTAssertLessThan(sizes[.zstd]!, sizes[.none]!)
    }
    
    // MARK: - Validation Tests
    
    func testChecksumValidation() async throws {
        let network = try await createTestNetwork()
        
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("checksum_test.json")
        
        try await network.saveCheckpoint(to: url)
        
        // Corrupt the file
        var data = try Data(contentsOf: url)
        data[100] = data[100] ^ 0xFF // Flip some bits
        try data.write(to: url)
        
        // Should fail to load
        let loadedNetwork = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        do {
            try await loadedNetwork.loadCheckpoint(from: url)
            XCTFail("Should have failed checksum validation")
        } catch CheckpointError.checksumMismatch {
            // Expected
        } catch {
            XCTFail("Wrong error type: \(error)")
        }
        
        // Cleanup
        try? FileManager.default.removeItem(at: url)
    }
    
    func testVersionCompatibility() async throws {
        // This test would check version migration in a real implementation
        // For now, just verify current version
        XCTAssertEqual(CheckpointVersion.current.major, 1)
        XCTAssertEqual(CheckpointVersion.current.minor, 0)
        XCTAssertEqual(CheckpointVersion.current.patch, 0)
    }
    
    // MARK: - Metadata Tests
    
    func testMetadataLoading() async throws {
        let network = try await createTestNetwork()
        
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("metadata_test.json")
        
        let options = CheckpointOptions(
            description: "Test checkpoint",
            customMetadata: ["test": "value", "number": "42"]
        )
        
        try await network.saveCheckpoint(to: url, options: options)
        
        // Load just metadata
        let metadata = try await NeuralNetwork.checkpointMetadata(from: url)
        
        XCTAssertEqual(metadata.description, "Test checkpoint")
        XCTAssertEqual(metadata.custom["test"], "value")
        XCTAssertEqual(metadata.custom["number"], "42")
        XCTAssertGreaterThan(metadata.metrics["parameterCount"] ?? 0, 0)
        
        // Cleanup
        try? FileManager.default.removeItem(at: url)
    }
    
    // MARK: - Edge Cases
    
    func testEmptyNetwork() async throws {
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("empty_network.json")
        
        try await network.saveCheckpoint(to: url)
        
        let loaded = try await NeuralNetwork(metalPipeline: metalPipeline)
        try await loaded.loadCheckpoint(from: url)
        
        XCTAssertEqual(await loaded.parameterCount(), 0)
        
        // Cleanup
        try? FileManager.default.removeItem(at: url)
    }
    
    func testLargeNetwork() async throws {
        // Test with a reasonably large network
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        // Create a network with ~1M parameters
        await network.addLayers([
            try await DenseLayer(
                inputSize: 1000,
                outputSize: 500,
                metalPipeline: metalPipeline
            ),
            try await DenseLayer(
                inputSize: 500,
                outputSize: 1000,
                metalPipeline: metalPipeline
            )
        ])
        
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("large_network.json")
        
        let startTime = Date()
        try await network.saveCheckpoint(to: url)
        let saveTime = Date().timeIntervalSince(startTime)
        
        let loadStart = Date()
        let loaded = try await NeuralNetwork(metalPipeline: metalPipeline)
        try await loaded.loadCheckpoint(from: url)
        let loadTime = Date().timeIntervalSince(loadStart)
        
        print("Save time: \(saveTime)s, Load time: \(loadTime)s")
        
        XCTAssertEqual(await network.parameterCount(), await loaded.parameterCount())
        
        // Cleanup
        try? FileManager.default.removeItem(at: url)
    }
    
    // MARK: - Helper Methods
    
    private func createTestNetwork() async throws -> NeuralNetwork {
        let network = try await NeuralNetwork(metalPipeline: metalPipeline)
        
        await network.addLayers([
            try await DenseLayer(
                inputSize: 20,
                outputSize: 10,
                activation: .relu,
                metalPipeline: metalPipeline
            ),
            try await DenseLayer(
                inputSize: 10,
                outputSize: 5,
                activation: .linear,
                metalPipeline: metalPipeline
            )
        ])
        
        return network
    }
    
    private func createTrainingData(
        inputSize: Int,
        outputSize: Int,
        samples: Int
    ) async throws -> [(input: MetalBuffer, target: MetalBuffer)] {
        var data: [(input: MetalBuffer, target: MetalBuffer)] = []
        
        for _ in 0..<samples {
            let input = try await createTestInput(size: inputSize)
            let target = try await createTestInput(size: outputSize)
            data.append((input, target))
        }
        
        return data
    }
    
    private func createTestInput(size: Int) async throws -> MetalBuffer {
        let buffer = try await metalPipeline.allocateBuffer(size: size)
        let ptr = buffer.buffer.contents().bindMemory(to: Float.self, capacity: size)
        
        for i in 0..<size {
            ptr[i] = Float.random(in: -1...1)
        }
        
        return buffer
    }
    
    private func buffersMatch(
        _ a: MetalBuffer,
        _ b: MetalBuffer,
        tolerance: Float
    ) -> Bool {
        guard a.count == b.count else { return false }
        
        let aPtr = a.buffer.contents().bindMemory(to: Float.self, capacity: a.count)
        let bPtr = b.buffer.contents().bindMemory(to: Float.self, capacity: b.count)
        
        for i in 0..<a.count {
            if abs(aPtr[i] - bPtr[i]) > tolerance {
                return false
            }
        }
        
        return true
    }
}

// MARK: - Performance Tests

extension ModelSerializationTests {
    func testSerializationPerformance() async throws {
        let network = try await createTestNetwork()
        
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("perf_test.json")
        
        measure {
            let expectation = self.expectation(description: "Save checkpoint")
            
            Task {
                try await network.saveCheckpoint(to: url)
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10)
        }
        
        // Cleanup
        try? FileManager.default.removeItem(at: url)
    }
    
    func testDeserializationPerformance() async throws {
        let network = try await createTestNetwork()
        
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("perf_test.json")
        
        try await network.saveCheckpoint(to: url)
        
        measure {
            let expectation = self.expectation(description: "Load checkpoint")
            
            Task {
                let loaded = try await NeuralNetwork(metalPipeline: self.metalPipeline)
                try await loaded.loadCheckpoint(from: url)
                expectation.fulfill()
            }
            
            wait(for: [expectation], timeout: 10)
        }
        
        // Cleanup
        try? FileManager.default.removeItem(at: url)
    }
}
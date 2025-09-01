import XCTest
@testable import VectorStoreKit

/// Tests for the compression engine
final class CompressionEngineTests: XCTestCase {
    
    var engine: CompressionEngine!
    
    override func setUp() async throws {
        try await super.setUp()
        engine = CompressionEngine()
    }
    
    // MARK: - LZ4 Tests
    
    func testLZ4Compression() async throws {
        let original = Data("Hello, this is a test string that should compress well because it has repeated patterns patterns patterns!".utf8)
        
        // Compress
        let compressed = try await engine.compress(original, using: .lz4)
        
        // Should be smaller
        XCTAssertLessThan(compressed.count, original.count)
        
        // Decompress
        let decompressed = try await engine.decompress(compressed, using: .lz4)
        
        // Should match original
        XCTAssertEqual(decompressed, original)
    }
    
    func testLZ4CompressionLargeData() async throws {
        // Create large compressible data
        let pattern = Data("ABCDEFGHIJKLMNOPQRSTUVWXYZ".utf8)
        var original = Data()
        for _ in 0..<1000 {
            original.append(pattern)
        }
        
        // Compress
        let compressed = try await engine.compress(original, using: .lz4)
        
        // Should achieve good compression
        let ratio = Double(original.count) / Double(compressed.count)
        XCTAssertGreaterThan(ratio, 2.0, "Expected at least 2x compression")
        
        // Decompress
        let decompressed = try await engine.decompress(compressed, using: .lz4)
        XCTAssertEqual(decompressed, original)
    }
    
    func testLZ4RandomData() async throws {
        // Random data doesn't compress well
        var original = Data(count: 10000)
        original.withUnsafeMutableBytes { bytes in
            arc4random_buf(bytes.baseAddress!, bytes.count)
        }
        
        // Compress
        let compressed = try await engine.compress(original, using: .lz4)
        
        // Should not compress much (or might even be larger)
        let ratio = Double(original.count) / Double(compressed.count)
        XCTAssertLessThan(ratio, 1.5, "Random data shouldn't compress well")
        
        // But should still decompress correctly
        let decompressed = try await engine.decompress(compressed, using: .lz4)
        XCTAssertEqual(decompressed, original)
    }
    
    // MARK: - ZSTD Tests
    
    func testZSTDCompression() async throws {
        let original = Data(repeating: 0x42, count: 10000)
        
        // Compress
        let compressed = try await engine.compress(original, using: .zstd, level: 3)
        
        // Should achieve excellent compression for repeated data
        let ratio = Double(original.count) / Double(compressed.count)
        XCTAssertGreaterThan(ratio, 10.0, "Expected high compression for repeated data")
        
        // Decompress
        let decompressed = try await engine.decompress(compressed, using: .zstd)
        XCTAssertEqual(decompressed, original)
    }
    
    func testZSTDCompressionLevels() async throws {
        let original = Data("The quick brown fox jumps over the lazy dog. " .utf8) * 100
        
        var sizes: [Int: Int] = [:]
        
        // Test different compression levels
        for level in [1, 3, 5] {
            let compressed = try await engine.compress(original, using: .zstd, level: level)
            sizes[level] = compressed.count
            
            // Verify decompression works
            let decompressed = try await engine.decompress(compressed, using: .zstd)
            XCTAssertEqual(decompressed, original)
        }
        
        // Higher levels should produce smaller output (generally)
        if let size1 = sizes[1], let size3 = sizes[3], let size5 = sizes[5] {
            XCTAssertGreaterThanOrEqual(size1, size3, "Level 3 should compress better than level 1")
            XCTAssertGreaterThanOrEqual(size3, size5, "Level 5 should compress better than level 3")
        }
    }
    
    // MARK: - No Compression Tests
    
    func testNoCompression() async throws {
        let original = Data("No compression".utf8)
        
        // "Compress" with none
        let compressed = try await engine.compress(original, using: .none)
        
        // Should be identical
        XCTAssertEqual(compressed, original)
        
        // "Decompress"
        let decompressed = try await engine.decompress(compressed, using: .none)
        XCTAssertEqual(decompressed, original)
    }
    
    // MARK: - Statistics Tests
    
    func testCompressionStatistics() async throws {
        let data = Data(repeating: 0x55, count: 1000)
        
        // Get initial stats
        let initialStats = await engine.getStatistics()
        let initialCompressed = initialStats.totalCompressed
        
        // Compress some data
        _ = try await engine.compress(data, using: .lz4)
        _ = try await engine.compress(data, using: .zstd)
        
        // Get updated stats
        let stats = await engine.getStatistics()
        
        XCTAssertEqual(stats.totalCompressed, initialCompressed + 2)
        XCTAssertGreaterThan(stats.totalBytesIn, 0)
        XCTAssertGreaterThan(stats.totalBytesOut, 0)
        XCTAssertGreaterThan(stats.avgCompressionRatio, 0)
    }
    
    // MARK: - Edge Cases
    
    func testEmptyData() async throws {
        let empty = Data()
        
        // LZ4
        let lz4Compressed = try await engine.compress(empty, using: .lz4)
        let lz4Decompressed = try await engine.decompress(lz4Compressed, using: .lz4)
        XCTAssertEqual(lz4Decompressed, empty)
        
        // ZSTD
        let zstdCompressed = try await engine.compress(empty, using: .zstd)
        let zstdDecompressed = try await engine.decompress(zstdCompressed, using: .zstd)
        XCTAssertEqual(zstdDecompressed, empty)
    }
    
    func testSingleByteData() async throws {
        let single = Data([0x42])
        
        // LZ4
        let lz4Compressed = try await engine.compress(single, using: .lz4)
        let lz4Decompressed = try await engine.decompress(lz4Compressed, using: .lz4)
        XCTAssertEqual(lz4Decompressed, single)
        
        // ZSTD
        let zstdCompressed = try await engine.compress(single, using: .zstd)
        let zstdDecompressed = try await engine.decompress(zstdCompressed, using: .zstd)
        XCTAssertEqual(zstdDecompressed, single)
    }
    
    // MARK: - Performance Tests
    
    func testLZ4Performance() {
        let data = Data(repeating: 0x42, count: 1_048_576) // 1MB
        
        measure {
            Task {
                _ = try? await engine.compress(data, using: .lz4)
            }
        }
    }
    
    func testZSTDPerformance() {
        let data = Data(repeating: 0x42, count: 1_048_576) // 1MB
        
        measure {
            Task {
                _ = try? await engine.compress(data, using: .zstd, level: 3)
            }
        }
    }
    
    // MARK: - Concurrent Access
    
    func testConcurrentCompression() async throws {
        let data = Data("Concurrent compression test".utf8) * 100
        
        // Run multiple compressions concurrently
        await withTaskGroup(of: Data?.self) { group in
            for i in 0..<10 {
                group.addTask {
                    let algorithm: CompressionType = i % 2 == 0 ? .lz4 : .zstd
                    return try? await self.engine.compress(data, using: algorithm)
                }
            }
            
            var results: [Data?] = []
            for await result in group {
                results.append(result)
            }
            
            // All should succeed
            XCTAssertEqual(results.compactMap { $0 }.count, 10)
        }
    }
}

// MARK: - Helpers

extension Data {
    static func * (lhs: Data, rhs: Int) -> Data {
        var result = Data()
        for _ in 0..<rhs {
            result.append(lhs)
        }
        return result
    }
}
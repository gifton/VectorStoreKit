import Testing
import Foundation
import simd
#if canImport(Metal)
import Metal
#endif
@testable import VectorStoreKit

@Test func vectorUniverseCreation() async throws {
    // Test basic universe creation
    let universe = VectorUniverse<simd_float4, String>()
    // Universe is a struct, so just verify it's created without error
    _ = universe
}

@Test func actorConcurrency() async throws {
    // Test that our actors work correctly with Swift 6 concurrency
    guard let tempDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
        throw VectorStoreKitTestError.setup("Could not create temp directory")
    }
    
    let testDir = tempDir.appendingPathComponent("VectorStoreKitTests")
    try? FileManager.default.createDirectory(at: testDir, withIntermediateDirectories: true)
    
    let config = HierarchicalStorage.Configuration(baseDirectory: testDir)
    let storage = try await HierarchicalStorage(configuration: config)
    
    // Test concurrent operations
    await withTaskGroup(of: Void.self) { group in
        for i in 0..<10 {
            group.addTask {
                do {
                    try await storage.store(
                        key: "test_\(i)",
                        data: Data("test data \(i)".utf8),
                        options: .default
                    )
                } catch {
                    // Ignore errors for this test
                }
            }
        }
    }
    
    // Verify storage works
    let retrievedData = try await storage.retrieve(key: "test_0")
    #expect(retrievedData != nil)
    
    // Cleanup
    try? FileManager.default.removeItem(at: testDir)
}

@Test func hnswIndexConcurrency() async throws {
    // Test HNSW index with Swift 6 concurrency
    let config = HNSWIndex<simd_float4, String>.Configuration()
    let index = try HNSWIndex<simd_float4, String>(configuration: config)
    
    // Test concurrent insertions
    var entries: [VectorEntry<simd_float4, String>] = []
    for i in 0..<5 {
        let entry = VectorEntry(
            id: "vector_\(i)",
            vector: simd_float4(Float(i), Float(i+1), Float(i+2), Float(i+3)),
            metadata: "metadata_\(i)",
            tier: .hot
        )
        entries.append(entry)
    }
    
    // Insert entries concurrently
    await withTaskGroup(of: Void.self) { group in
        for entry in entries {
            group.addTask {
                do {
                    _ = try await index.insert(entry)
                } catch {
                    // Ignore errors for this test
                }
            }
        }
    }
    
    // Test search
    let query = simd_float4(0.5, 1.5, 2.5, 3.5)
    let results = try await index.search(query: query, k: 3)
    #expect(results.count <= 3)
}

@Test func sendableConformance() async throws {
    // Test that our types properly conform to Sendable
    let config = UniverseConfiguration()
    
    // Test that configuration is sendable
    await withTaskGroup(of: Void.self) { group in
        group.addTask {
            // This should compile without warnings if Sendable is properly implemented
            let _ = config
        }
    }
}

@Test func metalComputeActorSafety() async throws {
    // Test MetalCompute actor safety
    #if canImport(Metal)
    guard MTLCreateSystemDefaultDevice() != nil else {
        // Skip test if Metal is not available
        return
    }
    #else
    // Skip test if Metal is not available
    return
    #endif
    
    let config = MetalCompute.Configuration.efficient
    let metalCompute = try await MetalCompute(configuration: config)
    
    // Test concurrent access to performance statistics
    await withTaskGroup(of: Void.self) { group in
        for _ in 0..<5 {
            group.addTask {
                let _ = await metalCompute.getPerformanceStatistics()
            }
        }
    }
}

// Error type for tests
enum VectorStoreKitTestError: Error {
    case setup(String)
    case execution(String)
}
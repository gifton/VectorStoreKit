import Foundation
import VectorStoreKit

@main
struct ThreeTierStorageExample {
    static func main() async throws {
        print("=== Three-Tier Storage Example ===")
        
        // Create configuration for 3-tier storage
        let config = ThreeTierStorageConfiguration(
            memoryLimit: 100_000_000,      // 100MB memory tier
            memoryItemSizeLimit: 10_000_000, // 10MB max item size
            autoMigrationEnabled: true       // Enable automatic tier migration
        )
        
        print("Configuration created:")
        print("- Memory limit: \(config.memoryLimit / 1_000_000)MB")
        print("- Auto-migration: \(config.autoMigrationEnabled)")
        print("- Storage paths:")
        print("  - SSD: \(config.ssdPath)")
        print("  - Archive: \(config.archivePath)")
        
        // Create the storage system
        let storage = ThreeTierStorage(configuration: config)
        
        print("\n=== Testing Vector Storage ===")
        
        // Create test vectors
        let vectors: [(String, [Float])] = [
            ("vec1", Array(repeating: 0.1, count: 128)),
            ("vec2", Array(repeating: 0.2, count: 128)),
            ("vec3", Array(repeating: 0.3, count: 128)),
            ("vec4", Array(repeating: 0.4, count: 128)),
            ("vec5", Array(repeating: 0.5, count: 128))
        ]
        
        // Store vectors
        for (id, data) in vectors {
            try await storage.store(id: id, data: data, metadata: [:])
            print("Stored vector \(id) in tier: \(await storage.currentTier(for: id) ?? "unknown")")
        }
        
        // Retrieve and verify
        print("\n=== Retrieving Vectors ===")
        for (id, originalData) in vectors {
            if let retrieved = try await storage.retrieve(id: id) {
                let match = retrieved.data == originalData
                print("Retrieved \(id): \(match ? "✓" : "✗") (tier: \(retrieved.tier))")
            } else {
                print("Failed to retrieve \(id)")
            }
        }
        
        // Test access patterns
        print("\n=== Testing Access Patterns ===")
        // Simulate frequent access to vec1
        for i in 1...5 {
            _ = try await storage.retrieve(id: "vec1")
            print("Access #\(i) to vec1")
        }
        
        // Check if vec1 is promoted
        let vec1Tier = await storage.currentTier(for: "vec1")
        print("vec1 is now in tier: \(vec1Tier ?? "unknown")")
        
        // Get statistics
        print("\n=== Storage Statistics ===")
        let stats = await storage.statistics()
        print("Total vectors: \(stats.totalCount)")
        print("Memory tier: \(stats.memoryTierCount) vectors")
        print("SSD tier: \(stats.ssdTierCount) vectors")
        print("Archive tier: \(stats.archiveTierCount) vectors")
        print("Total size: \(stats.totalSize / 1_000_000)MB")
        
        print("\n=== Example Complete ===")
    }
}

// Extension to check current tier (for demo purposes)
extension ThreeTierStorage {
    func currentTier(for id: String) async -> StorageTier? {
        // This would need to be implemented in the actual storage class
        // For now, we'll just check which tier has the data
        if let _ = try? await memoryTier.retrieve(id: id) {
            return .memory
        } else if let _ = try? await ssdTier.retrieve(id: id) {
            return .ssd
        } else if let _ = try? await archiveTier.retrieve(id: id) {
            return .archive
        }
        return nil
    }
}
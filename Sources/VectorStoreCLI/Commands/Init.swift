// VectorStoreCLI: Init Command
//
// Initialize a new vector store with specified configuration

import ArgumentParser
import Foundation
import VectorStoreKit

extension VectorStoreCLI {
    struct Init: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Initialize a new vector store",
            discussion: """
                Creates a new vector store with the specified configuration.
                This command sets up the necessary directory structure and configuration files.
                
                Examples:
                  # Initialize with default settings
                  vectorstore init
                  
                  # Initialize with specific dimensions
                  vectorstore init --dimensions 768
                  
                  # Initialize with custom configuration
                  vectorstore init --dimensions 1536 --index hnsw --storage hierarchical
                """
        )
        
        @OptionGroup var global: GlobalOptions
        
        @Option(name: .shortAndLong, help: "Vector dimensions")
        var dimensions: Int = 768
        
        @Option(name: .shortAndLong, help: "Index type (hnsw, ivf, hybrid, learned)")
        var index: String = "hnsw"
        
        @Option(name: .shortAndLong, help: "Storage type (hierarchical, memory, disk)")
        var storage: String = "hierarchical"
        
        @Option(name: .shortAndLong, help: "Cache type (lru, lfu, fifo, none)")
        var cache: String = "lru"
        
        @Option(help: "Cache memory limit in MB")
        var cacheMemory: Int = 100
        
        @Option(help: "HNSW max connections (M parameter)")
        var hnswM: Int = 16
        
        @Option(help: "HNSW construction parameter (efConstruction)")
        var hnswEf: Int = 200
        
        @Flag(help: "Use research configuration with advanced features")
        var research = false
        
        @Flag(help: "Enable Metal acceleration")
        var metal = true
        
        @Flag(help: "Enable encryption for storage")
        var encryption = false
        
        mutating func validate() throws {
            // Validate dimensions
            guard dimensions > 0 && dimensions <= 4096 else {
                throw ValidationError("Dimensions must be between 1 and 4096")
            }
            
            // Validate index type
            let validIndexTypes = ["hnsw", "ivf", "hybrid", "learned"]
            guard validIndexTypes.contains(index.lowercased()) else {
                throw ValidationError("Invalid index type. Must be one of: \(validIndexTypes.joined(separator: ", "))")
            }
            
            // Validate storage type
            let validStorageTypes = ["hierarchical", "memory", "disk"]
            guard validStorageTypes.contains(storage.lowercased()) else {
                throw ValidationError("Invalid storage type. Must be one of: \(validStorageTypes.joined(separator: ", "))")
            }
            
            // Validate cache type
            let validCacheTypes = ["lru", "lfu", "fifo", "none"]
            guard validCacheTypes.contains(cache.lowercased()) else {
                throw ValidationError("Invalid cache type. Must be one of: \(validCacheTypes.joined(separator: ", "))")
            }
            
            // Check if store already exists
            let configPath = URL(fileURLWithPath: global.storePath)
                .appendingPathComponent(".vectorstore")
                .appendingPathComponent("config.json")
            
            if FileManager.default.fileExists(atPath: configPath.path) && !global.force {
                throw ValidationError("Vector store already exists at '\(global.storePath)'. Use --force to overwrite.")
            }
        }
        
        mutating func run() async throws {
            let startTime = Date()
            
            if !global.quiet {
                Console.info("Initializing vector store at '\(global.storePath)'...")
            }
            
            // Create directory structure
            let baseURL = URL(fileURLWithPath: global.storePath)
            let vectorstoreURL = baseURL.appendingPathComponent(".vectorstore")
            
            let directories = [
                vectorstoreURL,
                vectorstoreURL.appendingPathComponent("data"),
                vectorstoreURL.appendingPathComponent("indexes"),
                vectorstoreURL.appendingPathComponent("logs"),
                vectorstoreURL.appendingPathComponent("snapshots")
            ]
            
            for directory in directories {
                try FileManager.default.createDirectory(
                    at: directory,
                    withIntermediateDirectories: true
                )
            }
            
            // Create configuration
            let config = StoreConfig(
                dimensions: dimensions,
                indexType: index.lowercased(),
                storageType: storage.lowercased(),
                cacheType: cache.lowercased(),
                createdAt: Date(),
                version: "1.0.0"
            )
            
            try config.save(to: global.storePath)
            
            // Create metadata file
            let metadata = InitMetadata(
                dimensions: dimensions,
                indexType: index,
                storageType: storage,
                cacheType: cache,
                cacheMemoryMB: cacheMemory,
                hnswM: hnswM,
                hnswEfConstruction: hnswEf,
                research: research,
                metalEnabled: metal,
                encryptionEnabled: encryption,
                createdAt: Date(),
                initDuration: Date().timeIntervalSince(startTime)
            )
            
            if global.json {
                let output = try OutputFormat.json.format(metadata)
                print(output)
            } else if !global.quiet {
                Console.success("Vector store initialized successfully!")
                print("")
                print("Configuration:")
                print("  Dimensions:     \(dimensions)")
                print("  Index Type:     \(index)")
                print("  Storage Type:   \(storage)")
                print("  Cache Type:     \(cache) (\(cacheMemory)MB)")
                if index == "hnsw" {
                    print("  HNSW Settings:  M=\(hnswM), efConstruction=\(hnswEf)")
                }
                print("  Metal:          \(metal ? "Enabled" : "Disabled")")
                print("  Encryption:     \(encryption ? "Enabled" : "Disabled")")
                print("")
                print("Next steps:")
                print("  • Import data: vectorstore import <file>")
                print("  • Query vectors: vectorstore query --file <query>")
                print("  • Monitor performance: vectorstore monitor")
            }
        }
    }
}

// MARK: - Supporting Types

struct InitMetadata: Codable {
    let dimensions: Int
    let indexType: String
    let storageType: String
    let cacheType: String
    let cacheMemoryMB: Int
    let hnswM: Int
    let hnswEfConstruction: Int
    let research: Bool
    let metalEnabled: Bool
    let encryptionEnabled: Bool
    let createdAt: Date
    let initDuration: TimeInterval
}
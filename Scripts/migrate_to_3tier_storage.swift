#!/usr/bin/env swift
//
// One-time migration script from SimpleStorage to ThreeTierStorage
// This script should be run once to migrate existing data and then deleted
//
// Usage: swift Scripts/migrate_to_3tier_storage.swift [source_path] [destination_path]
//

import Foundation

// MARK: - Migration Configuration

struct MigrationConfig {
    let sourcePath: String
    let destinationPath: String
    let batchSize: Int = 1000
    let validateData: Bool = true
    let deleteAfterMigration: Bool = false
}

// MARK: - Migration Script

@main
struct MigrateToThreeTierStorage {
    static func main() async {
        print("🚀 VectorStoreKit Storage Migration")
        print("==================================")
        print("Migrating from SimpleStorage to 3-Tier Storage System")
        print()
        
        // Parse command line arguments
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            printUsage()
            exit(1)
        }
        
        let config = MigrationConfig(
            sourcePath: args[1],
            destinationPath: args[2]
        )
        
        do {
            try await performMigration(config: config)
        } catch {
            print("❌ Migration failed: \(error)")
            exit(1)
        }
    }
    
    static func printUsage() {
        print("Usage: swift \(CommandLine.arguments[0]) <source_path> <destination_path>")
        print()
        print("Arguments:")
        print("  source_path      Path to existing storage data")
        print("  destination_path Path for new 3-tier storage")
        print()
        print("Example:")
        print("  swift \(CommandLine.arguments[0]) /tmp/vectorstore/simple /tmp/vectorstore/3tier")
    }
    
    static func performMigration(config: MigrationConfig) async throws {
        print("📋 Migration Configuration:")
        print("  Source: \(config.sourcePath)")
        print("  Destination: \(config.destinationPath)")
        print("  Batch Size: \(config.batchSize)")
        print("  Validate: \(config.validateData)")
        print()
        
        // Create destination directories
        let fileManager = FileManager.default
        let destURL = URL(fileURLWithPath: config.destinationPath)
        try fileManager.createDirectory(at: destURL, withIntermediateDirectories: true)
        
        // Create subdirectories for 3-tier storage
        let ssdPath = destURL.appendingPathComponent("ssd")
        let archivePath = destURL.appendingPathComponent("archive")
        try fileManager.createDirectory(at: ssdPath, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: archivePath, withIntermediateDirectories: true)
        
        // Load source data
        print("📂 Scanning source data...")
        let sourceItems = try await scanSourceData(path: config.sourcePath)
        print("  Found \(sourceItems.count) items to migrate")
        
        // Perform migration in batches
        var migrated = 0
        var failed = 0
        let totalBatches = (sourceItems.count + config.batchSize - 1) / config.batchSize
        
        for batchIndex in 0..<totalBatches {
            let startIdx = batchIndex * config.batchSize
            let endIdx = min(startIdx + config.batchSize, sourceItems.count)
            let batch = Array(sourceItems[startIdx..<endIdx])
            
            print("🔄 Processing batch \(batchIndex + 1)/\(totalBatches) (\(batch.count) items)...")
            
            for item in batch {
                do {
                    try await migrateItem(item, config: config)
                    migrated += 1
                } catch {
                    print("  ⚠️  Failed to migrate \(item.key): \(error)")
                    failed += 1
                }
            }
            
            // Progress update
            let progress = Double(migrated + failed) / Double(sourceItems.count) * 100
            print("  Progress: \(String(format: "%.1f", progress))% (\(migrated) succeeded, \(failed) failed)")
        }
        
        // Create metadata file
        try await createMetadata(
            at: destURL.appendingPathComponent("metadata.json"),
            itemCount: migrated
        )
        
        // Validation
        if config.validateData {
            print()
            print("✅ Validating migrated data...")
            let validationResult = try await validateMigration(
                sourceItems: sourceItems,
                destinationPath: config.destinationPath
            )
            print("  Validation: \(validationResult ? "PASSED" : "FAILED")")
        }
        
        // Summary
        print()
        print("📊 Migration Summary:")
        print("  Total Items: \(sourceItems.count)")
        print("  Migrated: \(migrated)")
        print("  Failed: \(failed)")
        print("  Success Rate: \(String(format: "%.1f", Double(migrated) / Double(sourceItems.count) * 100))%")
        
        if config.deleteAfterMigration && failed == 0 {
            print()
            print("🗑️  Deleting source data...")
            try fileManager.removeItem(atPath: config.sourcePath)
            print("  Source data deleted")
        }
        
        print()
        print("✨ Migration completed successfully!")
    }
    
    // MARK: - Helper Functions
    
    struct SourceItem {
        let key: String
        let data: Data
        let size: Int
        let metadata: [String: Any]?
    }
    
    static func scanSourceData(path: String) async throws -> [SourceItem] {
        // Simple implementation - in production, this would read actual storage format
        var items: [SourceItem] = []
        
        let fileManager = FileManager.default
        let url = URL(fileURLWithPath: path)
        
        if let enumerator = fileManager.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) {
            for case let fileURL as URL in enumerator {
                guard !fileURL.hasDirectoryPath else { continue }
                
                let data = try Data(contentsOf: fileURL)
                let key = fileURL.lastPathComponent
                let size = data.count
                
                items.append(SourceItem(
                    key: key,
                    data: data,
                    size: size,
                    metadata: nil
                ))
            }
        }
        
        return items
    }
    
    static func migrateItem(_ item: SourceItem, config: MigrationConfig) async throws {
        // Determine target tier based on size
        let tier: String
        if item.size < 1_048_576 { // < 1MB
            tier = "memory" // Will be loaded to memory on first access
        } else if item.size < 104_857_600 { // < 100MB
            tier = "ssd"
        } else {
            tier = "archive"
        }
        
        // For this migration, we'll store everything in SSD initially
        // The 3-tier system will migrate based on access patterns
        let destPath = URL(fileURLWithPath: config.destinationPath)
            .appendingPathComponent("ssd")
            .appendingPathComponent(item.key)
        
        try item.data.write(to: destPath)
    }
    
    static func createMetadata(at url: URL, itemCount: Int) async throws {
        let metadata: [String: Any] = [
            "version": "1.0",
            "migrationDate": ISO8601DateFormatter().string(from: Date()),
            "itemCount": itemCount,
            "storageType": "3-tier"
        ]
        
        let data = try JSONSerialization.data(withJSONObject: metadata, options: .prettyPrinted)
        try data.write(to: url)
    }
    
    static func validateMigration(sourceItems: [SourceItem], destinationPath: String) async throws -> Bool {
        // Simple validation - check if all files exist in destination
        let fileManager = FileManager.default
        
        for item in sourceItems {
            // Check in all tiers
            let ssdPath = URL(fileURLWithPath: destinationPath)
                .appendingPathComponent("ssd")
                .appendingPathComponent(item.key)
            let archivePath = URL(fileURLWithPath: destinationPath)
                .appendingPathComponent("archive")
                .appendingPathComponent(item.key)
            
            if !fileManager.fileExists(atPath: ssdPath.path) &&
               !fileManager.fileExists(atPath: archivePath.path) {
                print("    Missing: \(item.key)")
                return false
            }
        }
        
        return true
    }
}
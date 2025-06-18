// VectorStoreCLI: Export Command
//
// Export vectors and metadata to various file formats

import ArgumentParser
import Foundation
import VectorStoreKit

extension VectorStoreCLI {
    struct Export: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Export vectors to a file",
            discussion: """
                Export vector data to various file formats for backup, analysis, or migration.
                Supports filtering and selective export of vectors.
                
                Supported formats:
                  • JSON: Human-readable format with full metadata
                  • CSV: Simple tabular format
                  • Binary: Efficient binary format for large datasets
                  • HDF5: Scientific computing format
                  • JSONL: JSON Lines for streaming
                
                Examples:
                  # Export all vectors to JSON
                  vectorstore export output.json
                  
                  # Export with specific format and compression
                  vectorstore export vectors.bin --format binary --compress
                  
                  # Export filtered subset
                  vectorstore export subset.json --filter "category:research" --limit 1000
                """
        )
        
        @OptionGroup var global: GlobalOptions
        
        @Argument(help: "Output file path")
        var output: String
        
        @Option(name: .shortAndLong, help: "Export format (json, csv, binary, hdf5, jsonl)")
        var format: String?
        
        @Option(help: "Maximum number of vectors to export")
        var limit: Int?
        
        @Option(help: "Skip this many vectors")
        var offset: Int = 0
        
        @Option(help: "Filter expression for selective export")
        var filter: String?
        
        @Flag(help: "Include vector data in export")
        var includeVectors = true
        
        @Flag(help: "Include metadata in export")
        var includeMetadata = true
        
        @Flag(help: "Compress output file")
        var compress = false
        
        @Flag(help: "Pretty print JSON output")
        var pretty = true
        
        @Option(help: "Batch size for export operations")
        var batchSize: Int = 1000
        
        @Flag(help: "Show progress during export")
        var progress = true
        
        mutating func validate() throws {
            // Check if store exists
            try VectorStoreCLI.validateStorePath(global.storePath)
            
            // Check if output directory exists
            let outputURL = URL(fileURLWithPath: output)
            let outputDir = outputURL.deletingLastPathComponent()
            
            guard FileManager.default.fileExists(atPath: outputDir.path) else {
                throw CLIError.fileNotFound("Output directory does not exist: \(outputDir.path)")
            }
            
            // Check if output file already exists
            if FileManager.default.fileExists(atPath: output) && !global.force {
                throw ValidationError("Output file already exists: \(output). Use --force to overwrite.")
            }
            
            // Validate format if specified
            if let format = format {
                let validFormats = ["json", "csv", "binary", "hdf5", "jsonl"]
                guard validFormats.contains(format.lowercased()) else {
                    throw CLIError.invalidFormat("Invalid format. Must be one of: \(validFormats.joined(separator: ", "))")
                }
            }
            
            // Validate batch size
            guard batchSize > 0 && batchSize <= 10000 else {
                throw ValidationError("Batch size must be between 1 and 10000")
            }
            
            // Validate offset
            guard offset >= 0 else {
                throw ValidationError("Offset must be non-negative")
            }
        }
        
        mutating func run() async throws {
            let startTime = Date()
            
            // Load store configuration
            let config = try StoreConfig.load(from: global.storePath)
            
            // Detect format if not specified
            let exportFormat = format ?? detectFormat(from: output)
            
            if !global.quiet {
                Console.info("Exporting to '\(output)' (format: \(exportFormat))...")
                if let filter = filter {
                    Console.info("Filter: \(filter)")
                }
                if let limit = limit {
                    Console.info("Limit: \(limit) vectors")
                }
            }
            
            // Create progress tracker
            let progressTracker = progress && !global.quiet ? ProgressTracker() : nil
            
            // Perform export based on format
            let result: ExportResult
            switch exportFormat.lowercased() {
            case "json":
                result = try await exportJSON(
                    to: output,
                    config: config,
                    progressTracker: progressTracker
                )
            case "jsonl":
                result = try await exportJSONL(
                    to: output,
                    config: config,
                    progressTracker: progressTracker
                )
            case "csv":
                result = try await exportCSV(
                    to: output,
                    config: config,
                    progressTracker: progressTracker
                )
            case "binary":
                result = try await exportBinary(
                    to: output,
                    config: config,
                    progressTracker: progressTracker
                )
            case "hdf5":
                result = try await exportHDF5(
                    to: output,
                    config: config,
                    progressTracker: progressTracker
                )
            default:
                throw CLIError.invalidFormat("Unsupported format: \(exportFormat)")
            }
            
            // Complete progress
            progressTracker?.complete()
            
            // Compress if requested
            if compress {
                try await compressFile(output)
            }
            
            let duration = Date().timeIntervalSince(startTime)
            let fileSize = try FileManager.default.attributesOfItem(
                atPath: compress ? "\(output).gz" : output
            )[.size] as? Int ?? 0
            
            // Output results
            if global.json {
                let output = ExportOutput(
                    file: self.output,
                    format: exportFormat,
                    exported: result.exported,
                    skipped: result.skipped,
                    duration: duration,
                    fileSize: fileSize,
                    compressed: compress
                )
                print(try OutputFormat.json.format(output))
            } else if !global.quiet {
                Console.success("Export completed in \(String(format: "%.2f", duration))s")
                print("")
                print("Results:")
                print("  Exported: \(result.exported) vectors")
                if result.skipped > 0 {
                    print("  Skipped:  \(result.skipped) vectors")
                }
                print("  File:     \(compress ? "\(output).gz" : output)")
                print("  Size:     \(formatBytes(fileSize))")
                print("  Rate:     \(String(format: "%.0f", Double(result.exported) / duration)) vectors/s")
            }
        }
        
        // MARK: - Export Methods
        
        private func detectFormat(from path: String) -> String {
            let url = URL(fileURLWithPath: path)
            let ext = url.pathExtension.lowercased()
            
            switch ext {
            case "json":
                return "json"
            case "jsonl", "ndjson":
                return "jsonl"
            case "csv", "tsv":
                return "csv"
            case "bin", "dat":
                return "binary"
            case "h5", "hdf5":
                return "hdf5"
            default:
                return "json" // Default to JSON
            }
        }
        
        private func exportJSON(
            to path: String,
            config: StoreConfig,
            progressTracker: ProgressTracker?
        ) async throws -> ExportResult {
            var vectors: [[String: Any]] = []
            var exported = 0
            var skipped = 0
            
            // TODO: Replace with actual store access
            let totalVectors = 1000 // Placeholder
            progressTracker?.start(total: min(totalVectors, limit ?? totalVectors))
            
            // Simulate export
            for i in 0..<min(totalVectors, limit ?? totalVectors) {
                if i < offset {
                    skipped += 1
                    continue
                }
                
                var item: [String: Any] = ["id": "vec_\(i)"]
                
                if includeVectors {
                    item["vector"] = Array(repeating: 0.0, count: config.dimensions)
                }
                
                if includeMetadata {
                    item["metadata"] = [
                        "index": i,
                        "timestamp": Date().timeIntervalSince1970
                    ]
                }
                
                vectors.append(item)
                exported += 1
                
                progressTracker?.update(current: exported)
            }
            
            // Write JSON
            let jsonObject: [String: Any] = [
                "version": "1.0",
                "dimensions": config.dimensions,
                "count": vectors.count,
                "vectors": vectors
            ]
            
            let jsonData = try JSONSerialization.data(
                withJSONObject: jsonObject,
                options: pretty ? [.prettyPrinted, .sortedKeys] : []
            )
            
            try jsonData.write(to: URL(fileURLWithPath: path))
            
            return ExportResult(exported: exported, skipped: skipped)
        }
        
        private func exportJSONL(
            to path: String,
            config: StoreConfig,
            progressTracker: ProgressTracker?
        ) async throws -> ExportResult {
            let fileURL = URL(fileURLWithPath: path)
            FileManager.default.createFile(atPath: path, contents: nil)
            
            guard let fileHandle = FileHandle(forWritingAtPath: path) else {
                throw CLIError.exportError("Failed to create output file")
            }
            defer { fileHandle.closeFile() }
            
            var exported = 0
            var skipped = 0
            
            // TODO: Replace with actual store access
            let totalVectors = 1000 // Placeholder
            progressTracker?.start(total: min(totalVectors, limit ?? totalVectors))
            
            // Export line by line
            for i in 0..<min(totalVectors, limit ?? totalVectors) {
                if i < offset {
                    skipped += 1
                    continue
                }
                
                var item: [String: Any] = ["id": "vec_\(i)"]
                
                if includeVectors {
                    item["vector"] = Array(repeating: 0.0, count: config.dimensions)
                }
                
                if includeMetadata {
                    item["metadata"] = [
                        "index": i,
                        "timestamp": Date().timeIntervalSince1970
                    ]
                }
                
                let jsonData = try JSONSerialization.data(withJSONObject: item)
                fileHandle.write(jsonData)
                fileHandle.write("\n".data(using: .utf8)!)
                
                exported += 1
                progressTracker?.update(current: exported)
            }
            
            return ExportResult(exported: exported, skipped: skipped)
        }
        
        private func exportCSV(
            to path: String,
            config: StoreConfig,
            progressTracker: ProgressTracker?
        ) async throws -> ExportResult {
            // TODO: Implement CSV export
            throw CLIError.operationFailed("CSV export not yet implemented")
        }
        
        private func exportBinary(
            to path: String,
            config: StoreConfig,
            progressTracker: ProgressTracker?
        ) async throws -> ExportResult {
            // TODO: Implement binary export
            throw CLIError.operationFailed("Binary export not yet implemented")
        }
        
        private func exportHDF5(
            to path: String,
            config: StoreConfig,
            progressTracker: ProgressTracker?
        ) async throws -> ExportResult {
            // TODO: Implement HDF5 export
            throw CLIError.operationFailed("HDF5 export not yet implemented")
        }
        
        private func compressFile(_ path: String) async throws {
            let inputURL = URL(fileURLWithPath: path)
            let outputURL = URL(fileURLWithPath: "\(path).gz")
            
            // Use gzip compression
            let task = Process()
            task.executableURL = URL(fileURLWithPath: "/usr/bin/gzip")
            task.arguments = ["-9", path]
            
            try task.run()
            task.waitUntilExit()
            
            if task.terminationStatus != 0 {
                throw CLIError.exportError("Compression failed")
            }
        }
        
        private func formatBytes(_ bytes: Int) -> String {
            let formatter = ByteCountFormatter()
            formatter.countStyle = .binary
            return formatter.string(fromByteCount: Int64(bytes))
        }
    }
}

// MARK: - Supporting Types

struct ExportResult {
    let exported: Int
    let skipped: Int
}

struct ExportOutput: Codable {
    let file: String
    let format: String
    let exported: Int
    let skipped: Int
    let duration: TimeInterval
    let fileSize: Int
    let compressed: Bool
}
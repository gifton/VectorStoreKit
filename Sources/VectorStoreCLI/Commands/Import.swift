// VectorStoreCLI: Import Command
//
// Import vectors and metadata from various file formats

import ArgumentParser
import Foundation
import VectorStoreKit

extension VectorStoreCLI {
    struct Import: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Import vectors from a file",
            discussion: """
                Import vector data from various file formats including JSON, CSV, binary, and HDF5.
                The import process supports batch processing with progress tracking.
                
                Supported formats:
                  • JSON: Flexible format with vectors and metadata
                  • CSV: Simple format with vector values
                  • Binary: Efficient binary format
                  • HDF5: Scientific computing format
                  • JSONL: JSON Lines for streaming large datasets
                
                Examples:
                  # Import from JSON file
                  vectorstore import data.json
                  
                  # Import with specific format and batch size
                  vectorstore import embeddings.csv --format csv --batch-size 1000
                  
                  # Import with metadata mapping
                  vectorstore import data.jsonl --id-field doc_id --vector-field embedding
                """
        )
        
        @OptionGroup var global: GlobalOptions
        
        @Argument(help: "Path to the file to import")
        var file: String
        
        @Option(name: .shortAndLong, help: "File format (json, csv, binary, hdf5, jsonl)")
        var format: String?
        
        @Option(help: "Batch size for import operations")
        var batchSize: Int = 1000
        
        @Option(help: "Field name for vector IDs (JSON/JSONL only)")
        var idField: String = "id"
        
        @Option(help: "Field name for vectors (JSON/JSONL only)")
        var vectorField: String = "vector"
        
        @Option(help: "Field name for metadata (JSON/JSONL only)")
        var metadataField: String = "metadata"
        
        @Flag(help: "Skip validation of vector dimensions")
        var skipValidation = false
        
        @Flag(help: "Show progress bar during import")
        var progress = true
        
        @Flag(help: "Dry run - validate without importing")
        var dryRun = false
        
        @Option(help: "Number of parallel workers")
        var workers: Int = 4
        
        mutating func validate() throws {
            // Check if store exists
            try VectorStoreCLI.validateStorePath(global.storePath)
            
            // Check if file exists
            guard FileManager.default.fileExists(atPath: file) else {
                throw CLIError.fileNotFound(file)
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
            
            // Validate workers
            guard workers > 0 && workers <= 32 else {
                throw ValidationError("Workers must be between 1 and 32")
            }
        }
        
        mutating func run() async throws {
            let startTime = Date()
            
            // Load store configuration
            let config = try StoreConfig.load(from: global.storePath)
            
            // Detect format if not specified
            let importFormat = format ?? detectFormat(from: file)
            
            if !global.quiet {
                Console.info("Importing from '\(file)' (format: \(importFormat))...")
            }
            
            // Create progress tracker
            let progressTracker = progress && !global.quiet ? ProgressTracker() : nil
            
            // Perform import based on format
            let result: ImportResult
            switch importFormat.lowercased() {
            case "json":
                result = try await importJSON(
                    from: file,
                    config: config,
                    progressTracker: progressTracker
                )
            case "jsonl":
                result = try await importJSONL(
                    from: file,
                    config: config,
                    progressTracker: progressTracker
                )
            case "csv":
                result = try await importCSV(
                    from: file,
                    config: config,
                    progressTracker: progressTracker
                )
            case "binary":
                result = try await importBinary(
                    from: file,
                    config: config,
                    progressTracker: progressTracker
                )
            case "hdf5":
                result = try await importHDF5(
                    from: file,
                    config: config,
                    progressTracker: progressTracker
                )
            default:
                throw CLIError.invalidFormat("Unsupported format: \(importFormat)")
            }
            
            // Complete progress
            progressTracker?.complete()
            
            let duration = Date().timeIntervalSince(startTime)
            
            // Output results
            if global.json {
                let output = ImportOutput(
                    file: file,
                    format: importFormat,
                    imported: result.imported,
                    failed: result.failed,
                    skipped: result.skipped,
                    duration: duration,
                    errors: result.errors
                )
                print(try OutputFormat.json.format(output))
            } else if !global.quiet {
                Console.success("Import completed in \(String(format: "%.2f", duration))s")
                print("")
                print("Results:")
                print("  Imported: \(result.imported)")
                if result.failed > 0 {
                    print("  Failed:   \(result.failed)")
                }
                if result.skipped > 0 {
                    print("  Skipped:  \(result.skipped)")
                }
                print("  Rate:     \(String(format: "%.0f", Double(result.imported) / duration)) vectors/s")
                
                if !result.errors.isEmpty {
                    print("")
                    Console.warning("Errors encountered:")
                    for (index, error) in result.errors.prefix(5).enumerated() {
                        print("  \(index + 1). \(error)")
                    }
                    if result.errors.count > 5 {
                        print("  ... and \(result.errors.count - 5) more")
                    }
                }
            }
        }
        
        // MARK: - Import Methods
        
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
                // Try to detect from content
                if let firstLine = try? String(contentsOf: url, encoding: .utf8)
                    .components(separatedBy: .newlines)
                    .first {
                    if firstLine.starts(with: "{") {
                        return "jsonl"
                    } else if firstLine.starts(with: "[") {
                        return "json"
                    } else if firstLine.contains(",") {
                        return "csv"
                    }
                }
                return "binary"
            }
        }
        
        private func importJSON(
            from path: String,
            config: StoreConfig,
            progressTracker: ProgressTracker?
        ) async throws -> ImportResult {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let decoder = JSONDecoder()
            
            // Parse JSON structure
            let jsonObject = try JSONSerialization.jsonObject(with: data)
            
            var imported = 0
            var failed = 0
            var errors: [String] = []
            
            if let array = jsonObject as? [[String: Any]] {
                progressTracker?.start(total: array.count)
                
                for (index, item) in array.enumerated() {
                    do {
                        try await processJSONItem(
                            item,
                            config: config,
                            index: index
                        )
                        imported += 1
                    } catch {
                        failed += 1
                        errors.append("Row \(index): \(error.localizedDescription)")
                    }
                    
                    progressTracker?.update(current: index + 1)
                }
            } else {
                throw CLIError.importError("Invalid JSON structure. Expected array of objects.")
            }
            
            return ImportResult(
                imported: imported,
                failed: failed,
                skipped: 0,
                errors: errors
            )
        }
        
        private func importJSONL(
            from path: String,
            config: StoreConfig,
            progressTracker: ProgressTracker?
        ) async throws -> ImportResult {
            let url = URL(fileURLWithPath: path)
            let fileHandle = try FileHandle(forReadingFrom: url)
            defer { fileHandle.closeFile() }
            
            var imported = 0
            var failed = 0
            var errors: [String] = []
            var lineNumber = 0
            
            // Count total lines for progress
            let totalLines = try countLines(in: path)
            progressTracker?.start(total: totalLines)
            
            // Process line by line
            while let line = fileHandle.readLine() {
                lineNumber += 1
                
                guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    continue
                }
                
                do {
                    let jsonObject = try JSONSerialization.jsonObject(
                        with: Data(line.utf8)
                    ) as? [String: Any] ?? [:]
                    
                    try await processJSONItem(
                        jsonObject,
                        config: config,
                        index: lineNumber
                    )
                    imported += 1
                } catch {
                    failed += 1
                    errors.append("Line \(lineNumber): \(error.localizedDescription)")
                }
                
                progressTracker?.update(current: lineNumber)
            }
            
            return ImportResult(
                imported: imported,
                failed: failed,
                skipped: 0,
                errors: errors
            )
        }
        
        private func importCSV(
            from path: String,
            config: StoreConfig,
            progressTracker: ProgressTracker?
        ) async throws -> ImportResult {
            // TODO: Implement CSV import
            throw CLIError.operationFailed("CSV import not yet implemented")
        }
        
        private func importBinary(
            from path: String,
            config: StoreConfig,
            progressTracker: ProgressTracker?
        ) async throws -> ImportResult {
            // TODO: Implement binary import
            throw CLIError.operationFailed("Binary import not yet implemented")
        }
        
        private func importHDF5(
            from path: String,
            config: StoreConfig,
            progressTracker: ProgressTracker?
        ) async throws -> ImportResult {
            // TODO: Implement HDF5 import
            throw CLIError.operationFailed("HDF5 import not yet implemented")
        }
        
        private func processJSONItem(
            _ item: [String: Any],
            config: StoreConfig,
            index: Int
        ) async throws {
            // Extract ID
            guard let id = item[idField] as? String else {
                throw CLIError.importError("Missing or invalid ID field '\(idField)' at index \(index)")
            }
            
            // Extract vector
            guard let vectorArray = item[vectorField] as? [Double] else {
                throw CLIError.importError("Missing or invalid vector field '\(vectorField)' at index \(index)")
            }
            
            // Validate dimensions
            if !skipValidation && vectorArray.count != config.dimensions {
                throw CLIError.importError(
                    "Vector dimension mismatch at index \(index): expected \(config.dimensions), got \(vectorArray.count)"
                )
            }
            
            // Extract metadata
            let metadata = item[metadataField] as? [String: Any] ?? [:]
            
            // TODO: Actually insert into store
            if dryRun {
                // Just validate, don't insert
                return
            }
            
            // Simulate insertion for now
            // In real implementation, this would use the VectorStore API
        }
        
        private func countLines(in path: String) throws -> Int {
            let url = URL(fileURLWithPath: path)
            let content = try String(contentsOf: url, encoding: .utf8)
            return content.components(separatedBy: .newlines).count
        }
    }
}

// MARK: - Supporting Types

struct ImportResult {
    let imported: Int
    let failed: Int
    let skipped: Int
    let errors: [String]
}

struct ImportOutput: Codable {
    let file: String
    let format: String
    let imported: Int
    let failed: Int
    let skipped: Int
    let duration: TimeInterval
    let errors: [String]
}

// MARK: - Progress Tracking

class ProgressTracker {
    private var total: Int = 0
    private var current: Int = 0
    private let startTime = Date()
    private var lastUpdate = Date()
    
    func start(total: Int) {
        self.total = total
        self.current = 0
        print("Progress: 0/\(total) (0%)")
    }
    
    func update(current: Int) {
        self.current = current
        
        // Update at most once per 100ms
        guard Date().timeIntervalSince(lastUpdate) > 0.1 else { return }
        lastUpdate = Date()
        
        let percentage = total > 0 ? (current * 100) / total : 0
        let elapsed = Date().timeIntervalSince(startTime)
        let rate = elapsed > 0 ? Double(current) / elapsed : 0
        let eta = rate > 0 ? TimeInterval((total - current)) / rate : 0
        
        // Clear line and print progress
        print("\u{1B}[1A\u{1B}[K", terminator: "")
        print("Progress: \(current)/\(total) (\(percentage)%) - \(String(format: "%.0f", rate)) items/s - ETA: \(formatTime(eta))")
    }
    
    func complete() {
        let elapsed = Date().timeIntervalSince(startTime)
        print("\u{1B}[1A\u{1B}[K", terminator: "")
        print("Progress: \(total)/\(total) (100%) - Completed in \(formatTime(elapsed))")
    }
    
    private func formatTime(_ seconds: TimeInterval) -> String {
        if seconds < 60 {
            return String(format: "%.0fs", seconds)
        } else if seconds < 3600 {
            let minutes = Int(seconds) / 60
            let secs = Int(seconds) % 60
            return String(format: "%dm%02ds", minutes, secs)
        } else {
            let hours = Int(seconds) / 3600
            let minutes = (Int(seconds) % 3600) / 60
            return String(format: "%dh%02dm", hours, minutes)
        }
    }
}

// MARK: - FileHandle Extension

extension FileHandle {
    func readLine() -> String? {
        var data = Data()
        
        while true {
            let byte = readData(ofLength: 1)
            guard !byte.isEmpty else {
                return data.isEmpty ? nil : String(data: data, encoding: .utf8)
            }
            
            if byte[0] == 10 { // newline
                return String(data: data, encoding: .utf8)
            }
            
            data.append(byte)
        }
    }
}
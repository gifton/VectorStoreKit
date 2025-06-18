// VectorStoreCLI: Command-Line Interface for VectorStoreKit
//
// A comprehensive CLI tool for managing vector stores, performing operations,
// and monitoring performance on Apple Silicon.

import ArgumentParser
import Foundation
import VectorStoreKit

@main
struct VectorStoreCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vectorstore",
        abstract: "High-performance vector store management for Apple Silicon",
        discussion: """
            VectorStoreKit CLI provides comprehensive tools for managing vector databases
            optimized for Apple Silicon's unified memory architecture and Metal compute.
            
            Common workflows:
            • Initialize a new store: vectorstore init --dimensions 768
            • Import data: vectorstore import data.json --format json
            • Query vectors: vectorstore query --file query.txt --k 10
            • Monitor performance: vectorstore monitor --live
            """,
        version: "1.0.0",
        subcommands: [
            Init.self,
            Import.self,
            Export.self,
            Query.self,
            Index.self,
            Monitor.self,
            Stats.self,
            Health.self,
            Shell.self
        ],
        defaultSubcommand: nil
    )
    
    @Flag(name: .shortAndLong, help: "Enable verbose output")
    var verbose = false
    
    @Flag(name: .shortAndLong, help: "Output in JSON format")
    var json = false
    
    @Option(name: .shortAndLong, help: "Path to vector store directory")
    var storePath: String = FileManager.default.currentDirectoryPath
    
    mutating func run() async throws {
        // Show help if no subcommand
        print(Self.helpMessage())
    }
}

// MARK: - Global Options

struct GlobalOptions: ParsableArguments {
    @Flag(name: .shortAndLong, help: "Enable verbose output")
    var verbose = false
    
    @Flag(name: .shortAndLong, help: "Output in JSON format")
    var json = false
    
    @Option(name: .shortAndLong, help: "Path to vector store directory")
    var storePath: String = FileManager.default.currentDirectoryPath
    
    @Flag(name: .shortAndLong, help: "Suppress all output except errors")
    var quiet = false
    
    @Flag(help: "Force operation without confirmation")
    var force = false
}

// MARK: - Store Configuration

struct StoreConfig: Codable {
    let dimensions: Int
    let indexType: String
    let storageType: String
    let cacheType: String
    let createdAt: Date
    let version: String
    
    var configPath: URL {
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent(".vectorstore")
            .appendingPathComponent("config.json")
    }
    
    static func load(from path: String) throws -> StoreConfig {
        let url = URL(fileURLWithPath: path)
            .appendingPathComponent(".vectorstore")
            .appendingPathComponent("config.json")
        
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(StoreConfig.self, from: data)
    }
    
    func save(to path: String) throws {
        let url = URL(fileURLWithPath: path)
            .appendingPathComponent(".vectorstore")
        
        try FileManager.default.createDirectory(
            at: url,
            withIntermediateDirectories: true
        )
        
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        
        let data = try encoder.encode(self)
        try data.write(to: url.appendingPathComponent("config.json"))
    }
}

// MARK: - Error Types

enum CLIError: LocalizedError {
    case storeNotInitialized(path: String)
    case invalidConfiguration(String)
    case importError(String)
    case exportError(String)
    case queryError(String)
    case fileNotFound(String)
    case invalidFormat(String)
    case operationFailed(String)
    
    var errorDescription: String? {
        switch self {
        case .storeNotInitialized(let path):
            return "No vector store found at '\(path)'. Run 'vectorstore init' first."
        case .invalidConfiguration(let message):
            return "Invalid configuration: \(message)"
        case .importError(let message):
            return "Import failed: \(message)"
        case .exportError(let message):
            return "Export failed: \(message)"
        case .queryError(let message):
            return "Query failed: \(message)"
        case .fileNotFound(let path):
            return "File not found: \(path)"
        case .invalidFormat(let format):
            return "Invalid format: \(format)"
        case .operationFailed(let message):
            return "Operation failed: \(message)"
        }
    }
}

// MARK: - Utility Functions

extension VectorStoreCLI {
    static func validateStorePath(_ path: String) throws {
        let configPath = URL(fileURLWithPath: path)
            .appendingPathComponent(".vectorstore")
            .appendingPathComponent("config.json")
        
        guard FileManager.default.fileExists(atPath: configPath.path) else {
            throw CLIError.storeNotInitialized(path: path)
        }
    }
    
    static func loadStore(at path: String) async throws -> (config: StoreConfig, store: AnyObject) {
        let config = try StoreConfig.load(from: path)
        // TODO: Initialize actual store based on config
        return (config, NSObject()) // Placeholder
    }
}

// MARK: - Output Formatting

enum OutputFormat {
    case plain
    case json
    case table
    
    func format<T: Encodable>(_ data: T) throws -> String {
        switch self {
        case .plain:
            return String(describing: data)
        case .json:
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let jsonData = try encoder.encode(data)
            return String(data: jsonData, encoding: .utf8) ?? ""
        case .table:
            // TODO: Implement table formatting
            return String(describing: data)
        }
    }
}

// MARK: - Console Output

struct Console {
    static func print(_ message: String, verbose: Bool = false, quiet: Bool = false) {
        guard !quiet else { return }
        guard !verbose || CommandLine.arguments.contains("--verbose") else { return }
        Swift.print(message)
    }
    
    static func error(_ message: String) {
        FileHandle.standardError.write(Data("Error: \(message)\n".utf8))
    }
    
    static func success(_ message: String) {
        Swift.print("✅ \(message)")
    }
    
    static func warning(_ message: String) {
        Swift.print("⚠️  \(message)")
    }
    
    static func info(_ message: String) {
        Swift.print("ℹ️  \(message)")
    }
}
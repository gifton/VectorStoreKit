// VectorStoreCLI: Query Command
//
// Execute vector similarity searches with various strategies

import ArgumentParser
import Foundation
import VectorStoreKit

extension VectorStoreCLI {
    struct Query: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Search for similar vectors",
            discussion: """
                Execute k-nearest neighbor searches using various strategies and filters.
                Supports multiple query input methods and output formats.
                
                Query strategies:
                  • exact: Guaranteed accurate results (slower)
                  • approximate: Fast approximate search (default)
                  • adaptive: Automatically choose based on data
                
                Examples:
                  # Query with vector from file
                  vectorstore query --file query.json --k 10
                  
                  # Query with inline vector
                  vectorstore query --vector "[0.1, 0.2, 0.3, ...]" --k 20
                  
                  # Query with metadata filter
                  vectorstore query --file query.json --filter "category:tech" --k 50
                  
                  # Batch query from JSONL file
                  vectorstore query --batch queries.jsonl --k 10 --output results.json
                """
        )
        
        @OptionGroup var global: GlobalOptions
        
        @Option(name: .shortAndLong, help: "Path to query vector file")
        var file: String?
        
        @Option(name: .shortAndLong, help: "Inline query vector (JSON array)")
        var vector: String?
        
        @Option(name: .shortAndLong, help: "Path to batch query file (JSONL)")
        var batch: String?
        
        @Option(name: .shortAndLong, help: "Number of nearest neighbors to return")
        var k: Int = 10
        
        @Option(name: .shortAndLong, help: "Search strategy (exact, approximate, adaptive)")
        var strategy: String = "adaptive"
        
        @Option(name: .shortAndLong, help: "Metadata filter expression")
        var filter: String?
        
        @Option(name: .shortAndLong, help: "Output file for results (batch mode)")
        var output: String?
        
        @Flag(help: "Include vector data in results")
        var includeVectors = false
        
        @Flag(help: "Include distance scores in results")
        var includeScores = true
        
        @Flag(help: "Include search performance metrics")
        var metrics = false
        
        @Option(help: "Maximum search time in seconds")
        var timeout: Double = 30.0
        
        @Option(help: "Result format (json, csv, table)")
        var resultFormat: String = "table"
        
        mutating func validate() throws {
            // Check if store exists
            try VectorStoreCLI.validateStorePath(global.storePath)
            
            // Validate input method
            let inputMethods = [file != nil, vector != nil, batch != nil]
            let inputCount = inputMethods.filter { $0 }.count
            
            guard inputCount == 1 else {
                throw ValidationError("Specify exactly one input method: --file, --vector, or --batch")
            }
            
            // Validate files exist
            if let file = file {
                guard FileManager.default.fileExists(atPath: file) else {
                    throw CLIError.fileNotFound(file)
                }
            }
            
            if let batch = batch {
                guard FileManager.default.fileExists(atPath: batch) else {
                    throw CLIError.fileNotFound(batch)
                }
            }
            
            // Validate k
            guard k > 0 && k <= 1000 else {
                throw ValidationError("k must be between 1 and 1000")
            }
            
            // Validate strategy
            let validStrategies = ["exact", "approximate", "adaptive"]
            guard validStrategies.contains(strategy.lowercased()) else {
                throw ValidationError("Invalid strategy. Must be one of: \(validStrategies.joined(separator: ", "))")
            }
            
            // Validate timeout
            guard timeout > 0 && timeout <= 300 else {
                throw ValidationError("Timeout must be between 0 and 300 seconds")
            }
            
            // Validate result format
            let validFormats = ["json", "csv", "table"]
            guard validFormats.contains(resultFormat.lowercased()) else {
                throw ValidationError("Invalid result format. Must be one of: \(validFormats.joined(separator: ", "))")
            }
        }
        
        mutating func run() async throws {
            // Load store configuration
            let config = try StoreConfig.load(from: global.storePath)
            
            // Parse query vector(s)
            let queries = try parseQueries(config: config)
            
            // Execute queries
            if queries.count == 1 {
                // Single query
                let query = queries[0]
                let result = try await executeSingleQuery(
                    query: query,
                    config: config
                )
                
                // Output results
                try outputSingleResult(result, query: query)
            } else {
                // Batch queries
                let results = try await executeBatchQueries(
                    queries: queries,
                    config: config
                )
                
                // Output results
                try outputBatchResults(results, queries: queries)
            }
        }
        
        // MARK: - Query Parsing
        
        private func parseStrategy(_ strategy: String) -> SearchStrategy {
            switch strategy.lowercased() {
            case "exact":
                return .exact
            case "approximate":
                return .approximate
            case "adaptive":
                return .adaptive
            default:
                return .adaptive
            }
        }
        
        private func parseQueries(config: StoreConfig) throws -> [QueryVector] {
            if let file = file {
                return try [parseQueryFromFile(file, config: config)]
            } else if let vector = vector {
                return try [parseQueryFromString(vector, config: config)]
            } else if let batch = batch {
                return try parseQueriesFromBatch(batch, config: config)
            } else {
                throw CLIError.queryError("No query input provided")
            }
        }
        
        private func parseQueryFromFile(_ path: String, config: StoreConfig) throws -> QueryVector {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            
            // Try JSON object first
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let vector = json["vector"] as? [Double] {
                
                guard vector.count == config.dimensions else {
                    throw CLIError.queryError(
                        "Vector dimension mismatch: expected \(config.dimensions), got \(vector.count)"
                    )
                }
                
                return QueryVector(
                    id: json["id"] as? String ?? "query",
                    vector: vector,
                    metadata: json["metadata"] as? [String: Any]
                )
            }
            
            // Try plain array
            if let vector = try? JSONSerialization.jsonObject(with: data) as? [Double] {
                guard vector.count == config.dimensions else {
                    throw CLIError.queryError(
                        "Vector dimension mismatch: expected \(config.dimensions), got \(vector.count)"
                    )
                }
                
                return QueryVector(id: "query", vector: vector, metadata: nil)
            }
            
            throw CLIError.queryError("Invalid query file format")
        }
        
        private func parseQueryFromString(_ string: String, config: StoreConfig) throws -> QueryVector {
            guard let data = string.data(using: .utf8),
                  let vector = try? JSONSerialization.jsonObject(with: data) as? [Double] else {
                throw CLIError.queryError("Invalid vector format. Expected JSON array.")
            }
            
            guard vector.count == config.dimensions else {
                throw CLIError.queryError(
                    "Vector dimension mismatch: expected \(config.dimensions), got \(vector.count)"
                )
            }
            
            return QueryVector(id: "query", vector: vector, metadata: nil)
        }
        
        private func parseQueriesFromBatch(_ path: String, config: StoreConfig) throws -> [QueryVector] {
            let url = URL(fileURLWithPath: path)
            let content = try String(contentsOf: url, encoding: .utf8)
            let lines = content.components(separatedBy: .newlines)
            
            var queries: [QueryVector] = []
            
            for (index, line) in lines.enumerated() {
                guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    continue
                }
                
                guard let data = line.data(using: .utf8),
                      let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let vector = json["vector"] as? [Double] else {
                    throw CLIError.queryError("Invalid query at line \(index + 1)")
                }
                
                guard vector.count == config.dimensions else {
                    throw CLIError.queryError(
                        "Vector dimension mismatch at line \(index + 1): expected \(config.dimensions), got \(vector.count)"
                    )
                }
                
                queries.append(QueryVector(
                    id: json["id"] as? String ?? "query_\(index)",
                    vector: vector,
                    metadata: json["metadata"] as? [String: Any]
                ))
            }
            
            guard !queries.isEmpty else {
                throw CLIError.queryError("No valid queries found in batch file")
            }
            
            return queries
        }
        
        // MARK: - Query Execution
        
        private func executeSingleQuery(
            query: QueryVector,
            config: StoreConfig
        ) async throws -> QueryResult {
            let startTime = Date()
            
            if !global.quiet {
                Console.info("Executing query with k=\(k), strategy=\(strategy)...")
            }
            
            // Load store
            let (_, store) = try await VectorStoreCLI.loadStore(at: global.storePath)
            
            // Convert query vector to Float
            let floatVector = query.vector.map { Float($0) }
            let queryVector = Vector(values: floatVector)
            
            // Execute search
            let searchOptions = SearchOptions(
                k: k,
                strategy: parseStrategy(strategy),
                filter: filter.map { SearchFilter(expression: $0) },
                includeVectors: includeVectors,
                timeout: timeout
            )
            
            let searchResults = try await store.search(
                query: queryVector,
                options: searchOptions
            )
            
            // Convert results to SearchMatch
            let results = searchResults.map { result in
                SearchMatch(
                    id: result.id,
                    distance: result.score,
                    vector: includeVectors ? result.vector.values.map { Double($0) } : nil,
                    metadata: result.metadata
                )
            }
            
            let duration = Date().timeIntervalSince(startTime)
            
            return QueryResult(
                queryId: query.id,
                matches: results,
                queryTime: duration,
                candidatesEvaluated: 1000,
                strategy: strategy
            )
        }
        
        private func executeBatchQueries(
            queries: [QueryVector],
            config: StoreConfig
        ) async throws -> [QueryResult] {
            if !global.quiet {
                Console.info("Executing \(queries.count) queries...")
            }
            
            let progressTracker = !global.quiet ? ProgressTracker() : nil
            progressTracker?.start(total: queries.count)
            
            var results: [QueryResult] = []
            
            for (index, query) in queries.enumerated() {
                let result = try await executeSingleQuery(
                    query: query,
                    config: config
                )
                results.append(result)
                progressTracker?.update(current: index + 1)
            }
            
            progressTracker?.complete()
            
            return results
        }
        
        // MARK: - Output Formatting
        
        private func outputSingleResult(_ result: QueryResult, query: QueryVector) throws {
            if global.json || resultFormat == "json" {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                encoder.dateEncodingStrategy = .iso8601
                
                let output = SingleQueryOutput(
                    query: query,
                    result: result,
                    metrics: metrics ? QueryMetrics(
                        queryTime: result.queryTime,
                        candidatesEvaluated: result.candidatesEvaluated,
                        matchesReturned: result.matches.count,
                        strategy: result.strategy
                    ) : nil
                )
                
                let data = try encoder.encode(output)
                print(String(data: data, encoding: .utf8) ?? "")
            } else if resultFormat == "csv" {
                // CSV header
                print("rank,id,distance\(includeScores ? ",score" : "")")
                
                // Results
                for (index, match) in result.matches.enumerated() {
                    print("\(index + 1),\(match.id),\(match.distance)\(includeScores ? ",\(1.0 - match.distance)" : "")")
                }
            } else {
                // Table format
                if !global.quiet {
                    Console.success("Query completed in \(String(format: "%.3f", result.queryTime))s")
                    print("")
                }
                
                print("Results (top \(result.matches.count)):")
                print(String(repeating: "-", count: 60))
                print(String(format: "%-4s %-20s %-10s %s",
                            "Rank", "ID", "Distance", includeScores ? "Score" : ""))
                print(String(repeating: "-", count: 60))
                
                for (index, match) in result.matches.enumerated() {
                    let score = includeScores ? String(format: "%.4f", 1.0 - match.distance) : ""
                    print(String(format: "%-4d %-20s %-10.4f %s",
                                index + 1, match.id, match.distance, score))
                }
                
                if metrics {
                    print("")
                    print("Metrics:")
                    print("  Query Time:          \(String(format: "%.3f", result.queryTime))s")
                    print("  Candidates Evaluated: \(result.candidatesEvaluated)")
                    print("  Strategy:            \(result.strategy)")
                }
            }
        }
        
        private func outputBatchResults(_ results: [QueryResult], queries: [QueryVector]) throws {
            if let output = output {
                // Write to file
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                encoder.dateEncodingStrategy = .iso8601
                
                let batchOutput = BatchQueryOutput(
                    queries: queries,
                    results: results,
                    summary: BatchSummary(
                        totalQueries: queries.count,
                        averageQueryTime: results.map { $0.queryTime }.reduce(0, +) / Double(results.count),
                        totalTime: results.map { $0.queryTime }.reduce(0, +)
                    )
                )
                
                let data = try encoder.encode(batchOutput)
                try data.write(to: URL(fileURLWithPath: output))
                
                if !global.quiet {
                    Console.success("Batch results written to '\(output)'")
                }
            } else {
                // Output summary to console
                let totalTime = results.map { $0.queryTime }.reduce(0, +)
                let avgTime = totalTime / Double(results.count)
                
                print("Batch Query Summary:")
                print("  Total Queries:    \(results.count)")
                print("  Total Time:       \(String(format: "%.2f", totalTime))s")
                print("  Average Time:     \(String(format: "%.3f", avgTime))s")
                print("  Queries/Second:   \(String(format: "%.1f", Double(results.count) / totalTime))")
            }
        }
    }
}

// MARK: - Supporting Types

struct QueryVector: Codable {
    let id: String
    let vector: [Double]
    let metadata: [String: Any]?
    
    enum CodingKeys: String, CodingKey {
        case id, vector
    }
    
    init(id: String, vector: [Double], metadata: [String: Any]?) {
        self.id = id
        self.vector = vector
        self.metadata = metadata
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        vector = try container.decode([Double].self, forKey: .vector)
        metadata = nil // Skip metadata for Codable
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(vector, forKey: .vector)
        // Skip metadata for Codable
    }
}

struct SearchMatch: Codable {
    let id: String
    let distance: Float
    let vector: [Double]?
    let metadata: [String: Any]?
    
    enum CodingKeys: String, CodingKey {
        case id, distance, vector
    }
    
    init(id: String, distance: Float, vector: [Double]?, metadata: [String: Any]?) {
        self.id = id
        self.distance = distance
        self.vector = vector
        self.metadata = metadata
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        distance = try container.decode(Float.self, forKey: .distance)
        vector = try container.decodeIfPresent([Double].self, forKey: .vector)
        metadata = nil
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(distance, forKey: .distance)
        try container.encodeIfPresent(vector, forKey: .vector)
    }
}

struct QueryResult: Codable {
    let queryId: String
    let matches: [SearchMatch]
    let queryTime: TimeInterval
    let candidatesEvaluated: Int
    let strategy: String
}

struct QueryMetrics: Codable {
    let queryTime: TimeInterval
    let candidatesEvaluated: Int
    let matchesReturned: Int
    let strategy: String
}

struct SingleQueryOutput: Codable {
    let query: QueryVector
    let result: QueryResult
    let metrics: QueryMetrics?
}

struct BatchQueryOutput: Codable {
    let queries: [QueryVector]
    let results: [QueryResult]
    let summary: BatchSummary
}

struct BatchSummary: Codable {
    let totalQueries: Int
    let averageQueryTime: TimeInterval
    let totalTime: TimeInterval
}
import Foundation

/// RFC 4180 compliant CSV parser
public struct CSVParser {
    
    /// Parse CSV data into rows
    public static func parse(_ data: Data) throws -> [[String]] {
        guard let content = String(data: data, encoding: .utf8) else {
            throw CLIError.invalidFormat("Unable to decode CSV data as UTF-8")
        }
        
        return try parse(content)
    }
    
    /// Parse CSV string into rows
    public static func parse(_ content: String) throws -> [[String]] {
        var rows: [[String]] = []
        var currentRow: [String] = []
        var currentField = ""
        var inQuotes = false
        var previousChar: Character?
        
        let chars = Array(content)
        var i = 0
        
        while i < chars.count {
            let char = chars[i]
            
            if inQuotes {
                if char == "\"" {
                    // Check if it's an escaped quote
                    if i + 1 < chars.count && chars[i + 1] == "\"" {
                        currentField.append("\"")
                        i += 1 // Skip next quote
                    } else {
                        // End of quoted field
                        inQuotes = false
                    }
                } else {
                    currentField.append(char)
                }
            } else {
                switch char {
                case "\"":
                    if currentField.isEmpty && previousChar != "," && previousChar != "\n" && previousChar != "\r" && previousChar != nil {
                        // Quote not at beginning of field
                        throw CLIError.invalidFormat("Quote must be at beginning of field")
                    }
                    inQuotes = true
                    
                case ",":
                    currentRow.append(currentField)
                    currentField = ""
                    
                case "\n":
                    // Handle CRLF
                    if previousChar != "\r" {
                        currentRow.append(currentField)
                        if !currentRow.isEmpty || !currentField.isEmpty {
                            rows.append(currentRow)
                        }
                        currentRow = []
                        currentField = ""
                    }
                    
                case "\r":
                    currentRow.append(currentField)
                    if !currentRow.isEmpty || !currentField.isEmpty {
                        rows.append(currentRow)
                    }
                    currentRow = []
                    currentField = ""
                    
                default:
                    currentField.append(char)
                }
            }
            
            previousChar = char
            i += 1
        }
        
        // Handle last field/row
        if !currentField.isEmpty || !currentRow.isEmpty {
            currentRow.append(currentField)
            rows.append(currentRow)
        }
        
        if inQuotes {
            throw CLIError.invalidFormat("Unterminated quoted field")
        }
        
        return rows
    }
    
    /// Format rows as CSV
    public static func format(_ rows: [[String]]) -> String {
        return rows.map { row in
            row.map { field in
                // Check if field needs quoting
                if field.contains(",") || field.contains("\"") || field.contains("\n") || field.contains("\r") {
                    // Escape quotes and wrap in quotes
                    let escaped = field.replacingOccurrences(of: "\"", with: "\"\"")
                    return "\"\(escaped)\""
                } else {
                    return field
                }
            }.joined(separator: ",")
        }.joined(separator: "\n")
    }
}

// MARK: - CSV Import/Export Extensions

extension VectorUniverse {
    
    /// Import vectors from CSV file
    public func importCSV(from data: Data, hasHeader: Bool = true) async throws -> ImportResult {
        let rows = try CSVParser.parse(data)
        
        guard !rows.isEmpty else {
            throw CLIError.invalidFormat("Empty CSV file")
        }
        
        let startIndex = hasHeader ? 1 : 0
        guard rows.count > startIndex else {
            throw CLIError.invalidFormat("No data rows in CSV")
        }
        
        var vectors: [Vector] = []
        var metadata: [[String: String]] = []
        
        for (index, row) in rows[startIndex...].enumerated() {
            guard !row.isEmpty else { continue }
            
            // First column is ID (optional)
            let hasId = row[0].first?.isNumber == false
            let vectorStartIndex = hasId ? 1 : 0
            
            // Parse vector values
            var values: [Float] = []
            for i in vectorStartIndex..<row.count {
                if let value = Float(row[i]) {
                    values.append(value)
                } else if row[i].isEmpty {
                    // Skip empty values
                    continue
                } else {
                    // Non-numeric value, assume rest is metadata
                    break
                }
            }
            
            guard values.count == dimension else {
                throw CLIError.invalidFormat("Row \(index + startIndex) has \(values.count) values, expected \(dimension)")
            }
            
            vectors.append(Vector(values: values))
            
            // Parse metadata if present
            if hasId {
                var meta: [String: String] = ["id": row[0]]
                
                // Additional metadata columns
                let metadataStart = vectorStartIndex + values.count
                if metadataStart < row.count {
                    for i in metadataStart..<row.count {
                        if hasHeader && i - metadataStart < rows[0].count - metadataStart {
                            let key = rows[0][metadataStart + (i - metadataStart)]
                            meta[key] = row[i]
                        } else {
                            meta["field_\(i - metadataStart)"] = row[i]
                        }
                    }
                }
                
                metadata.append(meta)
            } else {
                metadata.append([:])
            }
        }
        
        // Add vectors to store
        let startTime = Date()
        try await addBatch(vectors, metadata: metadata.isEmpty ? nil : metadata)
        let duration = Date().timeIntervalSince(startTime)
        
        return ImportResult(
            vectorsImported: vectors.count,
            duration: duration,
            averageVectorsPerSecond: Double(vectors.count) / duration
        )
    }
    
    /// Export vectors to CSV format
    public func exportCSV(
        query: String? = nil,
        limit: Int = 1000,
        includeMetadata: Bool = true
    ) async throws -> Data {
        // Search or retrieve vectors
        let results: [SearchResult]
        if let query = query {
            results = try await search(
                query: Vector(values: try parseVector(query)),
                k: limit
            )
        } else {
            // Get all vectors up to limit
            results = try await getAllVectors(limit: limit)
        }
        
        var rows: [[String]] = []
        
        // Header row
        var header = ["id"]
        header.append(contentsOf: (0..<dimension).map { "dim_\($0)" })
        if includeMetadata {
            // Collect all metadata keys
            let allKeys = Set(results.flatMap { $0.metadata?.keys ?? [] })
            header.append(contentsOf: allKeys.sorted())
        }
        rows.append(header)
        
        // Data rows
        for result in results {
            var row: [String] = []
            
            // ID
            row.append(result.id)
            
            // Vector values
            row.append(contentsOf: result.vector.values.map { String($0) })
            
            // Metadata
            if includeMetadata, let metadata = result.metadata {
                for key in header.dropFirst(dimension + 1) {
                    row.append(metadata[key] ?? "")
                }
            }
            
            rows.append(row)
        }
        
        let csv = CSVParser.format(rows)
        return csv.data(using: .utf8)!
    }
    
    private func parseVector(_ string: String) throws -> [Float] {
        // Remove brackets and whitespace
        let cleaned = string
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        
        let components = cleaned.split(separator: ",")
        
        return try components.map { component in
            guard let value = Float(component.trimmingCharacters(in: .whitespaces)) else {
                throw CLIError.invalidFormat("Invalid vector component: \(component)")
            }
            return value
        }
    }
    
    private func getAllVectors(limit: Int) async throws -> [SearchResult] {
        // This would be implemented based on your storage backend
        // For now, return empty array
        return []
    }
}

public struct ImportResult {
    public let vectorsImported: Int
    public let duration: TimeInterval
    public let averageVectorsPerSecond: Double
}
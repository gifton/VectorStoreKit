import Foundation

/// Formats data as ASCII tables for CLI output
public struct TableFormatter {
    
    public struct Column {
        let header: String
        let width: Int
        let alignment: Alignment
        
        public enum Alignment {
            case left, center, right
        }
        
        public init(header: String, width: Int? = nil, alignment: Alignment = .left) {
            self.header = header
            self.width = width ?? max(header.count + 2, 10)
            self.alignment = alignment
        }
    }
    
    public static func format(
        columns: [Column],
        rows: [[String]],
        style: TableStyle = .simple
    ) -> String {
        var output = ""
        
        // Calculate actual column widths
        var actualWidths = columns.map { $0.width }
        
        // Adjust widths based on content
        for (colIndex, column) in columns.enumerated() {
            let maxContentWidth = rows.map { row in
                colIndex < row.count ? row[colIndex].count : 0
            }.max() ?? 0
            
            actualWidths[colIndex] = max(
                column.width,
                column.header.count + 2,
                maxContentWidth + 2
            )
        }
        
        // Build table
        switch style {
        case .simple:
            output += buildSimpleTable(columns: columns, rows: rows, widths: actualWidths)
        case .bordered:
            output += buildBorderedTable(columns: columns, rows: rows, widths: actualWidths)
        case .markdown:
            output += buildMarkdownTable(columns: columns, rows: rows, widths: actualWidths)
        }
        
        return output
    }
    
    private static func buildSimpleTable(
        columns: [Column],
        rows: [[String]],
        widths: [Int]
    ) -> String {
        var output = ""
        
        // Header
        for (index, column) in columns.enumerated() {
            output += pad(column.header, width: widths[index], alignment: column.alignment)
            if index < columns.count - 1 {
                output += " "
            }
        }
        output += "\n"
        
        // Separator
        for (index, width) in widths.enumerated() {
            output += String(repeating: "-", count: width)
            if index < widths.count - 1 {
                output += " "
            }
        }
        output += "\n"
        
        // Rows
        for row in rows {
            for (index, column) in columns.enumerated() {
                let value = index < row.count ? row[index] : ""
                output += pad(value, width: widths[index], alignment: column.alignment)
                if index < columns.count - 1 {
                    output += " "
                }
            }
            output += "\n"
        }
        
        return output
    }
    
    private static func buildBorderedTable(
        columns: [Column],
        rows: [[String]],
        widths: [Int]
    ) -> String {
        var output = ""
        
        // Top border
        output += "┌"
        for (index, width) in widths.enumerated() {
            output += String(repeating: "─", count: width)
            if index < widths.count - 1 {
                output += "┬"
            }
        }
        output += "┐\n"
        
        // Header
        output += "│"
        for (index, column) in columns.enumerated() {
            output += pad(column.header, width: widths[index], alignment: column.alignment)
            output += "│"
        }
        output += "\n"
        
        // Header separator
        output += "├"
        for (index, width) in widths.enumerated() {
            output += String(repeating: "─", count: width)
            if index < widths.count - 1 {
                output += "┼"
            }
        }
        output += "┤\n"
        
        // Rows
        for row in rows {
            output += "│"
            for (index, column) in columns.enumerated() {
                let value = index < row.count ? row[index] : ""
                output += pad(value, width: widths[index], alignment: column.alignment)
                output += "│"
            }
            output += "\n"
        }
        
        // Bottom border
        output += "└"
        for (index, width) in widths.enumerated() {
            output += String(repeating: "─", count: width)
            if index < widths.count - 1 {
                output += "┴"
            }
        }
        output += "┘\n"
        
        return output
    }
    
    private static func buildMarkdownTable(
        columns: [Column],
        rows: [[String]],
        widths: [Int]
    ) -> String {
        var output = ""
        
        // Header
        output += "|"
        for column in columns {
            output += " \(column.header) |"
        }
        output += "\n"
        
        // Separator
        output += "|"
        for column in columns {
            switch column.alignment {
            case .left:
                output += ":---|"
            case .center:
                output += ":---:|"
            case .right:
                output += "---:|"
            }
        }
        output += "\n"
        
        // Rows
        for row in rows {
            output += "|"
            for (index, _) in columns.enumerated() {
                let value = index < row.count ? row[index] : ""
                output += " \(value) |"
            }
            output += "\n"
        }
        
        return output
    }
    
    private static func pad(
        _ string: String,
        width: Int,
        alignment: Column.Alignment
    ) -> String {
        let padding = max(0, width - string.count)
        
        switch alignment {
        case .left:
            return string + String(repeating: " ", count: padding)
        case .center:
            let leftPad = padding / 2
            let rightPad = padding - leftPad
            return String(repeating: " ", count: leftPad) + string + String(repeating: " ", count: rightPad)
        case .right:
            return String(repeating: " ", count: padding) + string
        }
    }
    
    public enum TableStyle {
        case simple
        case bordered
        case markdown
    }
}

// MARK: - Search Result Formatting

extension Array where Element == SearchResult {
    
    public func formatAsTable(
        style: TableFormatter.TableStyle = .simple,
        includeVector: Bool = false,
        maxVectorDisplay: Int = 5
    ) -> String {
        var columns = [
            TableFormatter.Column(header: "Rank", width: 6, alignment: .right),
            TableFormatter.Column(header: "ID", width: 20),
            TableFormatter.Column(header: "Score", width: 12, alignment: .right)
        ]
        
        if includeVector {
            columns.append(TableFormatter.Column(header: "Vector", width: 40))
        }
        
        // Add metadata columns
        let allMetadataKeys = Set(self.flatMap { $0.metadata?.keys ?? [] })
        for key in allMetadataKeys.sorted() {
            columns.append(TableFormatter.Column(header: key, width: 15))
        }
        
        let rows = self.enumerated().map { index, result in
            var row = [
                String(index + 1),
                result.id,
                String(format: "%.4f", result.score)
            ]
            
            if includeVector {
                let vectorPreview = result.vector.values
                    .prefix(maxVectorDisplay)
                    .map { String(format: "%.2f", $0) }
                    .joined(separator: ", ")
                let suffix = result.vector.values.count > maxVectorDisplay ? "..." : ""
                row.append("[\(vectorPreview)\(suffix)]")
            }
            
            // Add metadata values
            for key in allMetadataKeys.sorted() {
                row.append(result.metadata?[key] ?? "")
            }
            
            return row
        }
        
        return TableFormatter.format(columns: columns, rows: rows, style: style)
    }
}

// MARK: - Store Info Formatting

extension StoreInfo {
    
    public func formatAsTable(style: TableFormatter.TableStyle = .simple) -> String {
        let columns = [
            TableFormatter.Column(header: "Property", width: 25),
            TableFormatter.Column(header: "Value", width: 40)
        ]
        
        let rows = [
            ["Name", name],
            ["Dimension", String(dimension)],
            ["Vector Count", String(vectorCount)],
            ["Index Type", indexType],
            ["Storage Backend", storageBackend],
            ["Created", createdAt.formatted()],
            ["Last Modified", lastModified.formatted()],
            ["Size on Disk", formatBytes(sizeOnDisk)]
        ]
        
        return TableFormatter.format(columns: columns, rows: rows, style: style)
    }
    
    private func formatBytes(_ bytes: Int64) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var size = Double(bytes)
        var unitIndex = 0
        
        while size >= 1024 && unitIndex < units.count - 1 {
            size /= 1024
            unitIndex += 1
        }
        
        if unitIndex == 0 {
            return "\(Int(size)) \(units[unitIndex])"
        } else {
            return String(format: "%.2f %@", size, units[unitIndex])
        }
    }
}

// MARK: - Performance Stats Formatting

public struct PerformanceStats {
    public let operation: String
    public let count: Int
    public let totalTime: TimeInterval
    public let avgTime: TimeInterval
    public let minTime: TimeInterval
    public let maxTime: TimeInterval
    public let throughput: Double
    
    public static func formatMultiple(
        _ stats: [PerformanceStats],
        style: TableFormatter.TableStyle = .simple
    ) -> String {
        let columns = [
            TableFormatter.Column(header: "Operation", width: 20),
            TableFormatter.Column(header: "Count", width: 10, alignment: .right),
            TableFormatter.Column(header: "Avg (ms)", width: 12, alignment: .right),
            TableFormatter.Column(header: "Min (ms)", width: 12, alignment: .right),
            TableFormatter.Column(header: "Max (ms)", width: 12, alignment: .right),
            TableFormatter.Column(header: "Total (s)", width: 12, alignment: .right),
            TableFormatter.Column(header: "Throughput", width: 15, alignment: .right)
        ]
        
        let rows = stats.map { stat in
            [
                stat.operation,
                String(stat.count),
                String(format: "%.2f", stat.avgTime * 1000),
                String(format: "%.2f", stat.minTime * 1000),
                String(format: "%.2f", stat.maxTime * 1000),
                String(format: "%.2f", stat.totalTime),
                String(format: "%.0f/s", stat.throughput)
            ]
        }
        
        return TableFormatter.format(columns: columns, rows: rows, style: style)
    }
}
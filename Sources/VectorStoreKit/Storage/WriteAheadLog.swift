// VectorStoreKit: Write-Ahead Log Implementation
//
// Provides crash-resistant transaction logging for HierarchicalStorage

import Foundation
import CryptoKit
import os.log

/// Write-ahead log for ensuring durability in storage operations
actor WriteAheadLog {
    // MARK: - Properties
    
    private let directory: URL
    private let configuration: WALConfiguration
    private let logger = Logger(subsystem: "VectorStoreKit", category: "WriteAheadLog")
    
    private var currentLogFile: URL?
    private var fileHandle: FileHandle?
    private var nextSequence: UInt64 = 0
    private var pendingEntries: [UInt64: WALEntry] = [:]
    private var completedSequences: Set<UInt64> = []
    
    // File naming
    private let logFilePrefix = "wal"
    private let logFileExtension = "log"
    private let checkpointFile = "checkpoint.dat"
    
    // MARK: - Initialization
    
    init(directory: URL, configuration: WALConfiguration) async throws {
        self.directory = directory
        self.configuration = configuration
        
        // Create directory if needed
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true,
            attributes: nil
        )
        
        // Recover from existing logs
        try await recover()
        
        // Open new log file
        try await openNewLogFile()
        
        logger.info("WAL initialized with sequence: \(self.nextSequence)")
    }
    
    // MARK: - Public Methods
    
    /// Append new operation to the log
    func appendNew(operation: WALOperation) async throws -> WALEntry {
        let entry = WALEntry(
            operation: operation,
            timestamp: Date(),
            sequenceNumber: nextSequence
        )
        nextSequence += 1
        
        try await append(entry)
        pendingEntries[entry.sequenceNumber] = entry
        
        return entry
    }
    
    /// Mark entry as completed
    func markCompleted(sequenceNumber: UInt64) async throws {
        guard pendingEntries[sequenceNumber] != nil else {
            throw WALError.invalidSequence(sequenceNumber)
        }
        
        pendingEntries.removeValue(forKey: sequenceNumber)
        completedSequences.insert(sequenceNumber)
        
        // Write completion marker
        let marker = CompletionMarker(sequenceNumber: sequenceNumber)
        try await writeMarker(marker)
        
        // Check if we should checkpoint
        if shouldCheckpoint() {
            try await checkpoint()
        }
    }
    
    /// Mark entry as failed
    func markFailed(sequenceNumber: UInt64, error: Error) async throws {
        guard pendingEntries[sequenceNumber] != nil else {
            throw WALError.invalidSequence(sequenceNumber)
        }
        
        pendingEntries.removeValue(forKey: sequenceNumber)
        
        // Write failure marker
        let marker = FailureMarker(sequenceNumber: sequenceNumber, error: error.localizedDescription)
        try await writeMarker(marker)
    }
    
    /// Get incomplete entries for recovery
    func getIncompleteEntries() async throws -> [WALEntry] {
        return Array(pendingEntries.values).sorted { $0.sequenceNumber < $1.sequenceNumber }
    }
    
    /// Compact the WAL by removing completed entries
    func compact() async throws {
        logger.info("Starting WAL compaction")
        
        // Close current file
        try? fileHandle?.close()
        fileHandle = nil
        
        // Create new compacted log
        let tempFile = directory.appendingPathComponent("wal_compact.tmp")
        
        // Write only pending entries to new file
        FileManager.default.createFile(atPath: tempFile.path, contents: nil)
        guard let tempHandle = FileHandle(forWritingAtPath: tempFile.path) else {
            throw WALError.fileCreationFailed
        }
        
        // Write header
        let header = WALHeader(version: 1, sequence: nextSequence)
        let headerData = try JSONEncoder().encode(header)
        try tempHandle.write(contentsOf: headerData)
        try tempHandle.write(contentsOf: "\n".data(using: .utf8)!)
        
        // Write pending entries
        for entry in pendingEntries.values.sorted(by: { $0.sequenceNumber < $1.sequenceNumber }) {
            try await writeEntry(entry, to: tempHandle)
        }
        
        try tempHandle.synchronize()
        tempHandle.closeFile()
        
        // Replace old log with compacted one
        if let currentLogFile = currentLogFile {
            try FileManager.default.removeItem(at: currentLogFile)
        }
        
        let newLogFile = generateLogFileName()
        try FileManager.default.moveItem(at: tempFile, to: newLogFile)
        
        currentLogFile = newLogFile
        fileHandle = FileHandle(forWritingAtPath: newLogFile.path)
        _ = try? fileHandle?.seekToEnd()
        
        // Update checkpoint
        try await writeCheckpoint()
        
        logger.info("WAL compaction completed")
    }
    
    /// Get statistics
    func getStatistics() async -> WALStatistics {
        let fileSize = (try? fileHandle?.seekToEnd()) ?? 0
        
        return WALStatistics(
            totalEntries: nextSequence,
            pendingEntries: pendingEntries.count,
            averageEntrySize: pendingEntries.isEmpty ? 0 : Int(fileSize) / pendingEntries.count,
            syncFrequency: configuration.syncInterval
        )
    }
    
    /// Validate integrity
    func validateIntegrity() async -> StorageIntegrityReport {
        var issues: [StorageIssue] = []
        
        // Check if log file exists and is readable
        if let currentLogFile = currentLogFile {
            if !FileManager.default.isReadableFile(atPath: currentLogFile.path) {
                issues.append(StorageIssue(
                    type: .corruption,
                    description: "WAL file is not readable",
                    impact: .critical
                ))
            }
        } else {
            issues.append(StorageIssue(
                type: .consistency,
                description: "No active WAL file",
                impact: .major
            ))
        }
        
        // Check pending entries
        if pendingEntries.count > configuration.maxPendingEntries {
            issues.append(StorageIssue(
                type: .performance,
                description: "Too many pending WAL entries",
                impact: .moderate
            ))
        }
        
        return StorageIntegrityReport(
            isHealthy: issues.filter { $0.impact == .critical || $0.impact == .major }.isEmpty,
            issues: issues,
            recommendations: issues.isEmpty ? [] : ["Consider running WAL compaction"],
            lastCheck: Date()
        )
    }
    
    // MARK: - Private Methods
    
    private func append(_ entry: WALEntry) async throws {
        guard let fileHandle = fileHandle else {
            throw WALError.noActiveLog
        }
        
        try await writeEntry(entry, to: fileHandle)
        
        // Sync based on configuration
        if configuration.syncMode == .synchronous ||
           (configuration.syncMode == .periodic && shouldSync()) {
            try fileHandle.synchronize()
        }
    }
    
    private func writeEntry(_ entry: WALEntry, to handle: FileHandle) async throws {
        // Encode entry
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let entryData = try encoder.encode(entry)
        
        // Calculate checksum
        let checksum = CRC32.checksum(data: entryData)
        
        // Write record: [length][checksum][data][newline]
        var record = Data()
        record.append(contentsOf: withUnsafeBytes(of: UInt32(entryData.count)) { Data($0) })
        record.append(contentsOf: withUnsafeBytes(of: checksum) { Data($0) })
        record.append(entryData)
        record.append("\n".data(using: .utf8)!)
        
        try handle.write(contentsOf: record)
    }
    
    private func writeMarker(_ marker: any WALMarker) async throws {
        guard let fileHandle = fileHandle else {
            throw WALError.noActiveLog
        }
        
        let markerData = try JSONEncoder().encode(AnyWALMarker(marker))
        try fileHandle.write(contentsOf: markerData)
        try fileHandle.write(contentsOf: "\n".data(using: .utf8)!)
        
        if configuration.syncMode == .synchronous {
            try fileHandle.synchronize()
        }
    }
    
    private func recover() async throws {
        logger.info("Starting WAL recovery")
        
        // Read checkpoint if exists
        let checkpointURL = directory.appendingPathComponent(checkpointFile)
        if FileManager.default.fileExists(atPath: checkpointURL.path),
           let checkpointData = try? Data(contentsOf: checkpointURL),
           let checkpoint = try? JSONDecoder().decode(WALCheckpoint.self, from: checkpointData) {
            nextSequence = checkpoint.nextSequence
            completedSequences = checkpoint.completedSequences
        }
        
        // Find all log files
        let logFiles = try FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ).filter { $0.pathExtension == logFileExtension }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        
        // Replay logs
        for logFile in logFiles {
            try await replayLog(at: logFile)
        }
        
        logger.info("WAL recovery completed, recovered \(self.pendingEntries.count) pending entries")
    }
    
    private func replayLog(at url: URL) async throws {
        guard let fileHandle = FileHandle(forReadingAtPath: url.path) else {
            logger.warning("Could not open log file for replay: \(url.lastPathComponent)")
            return
        }
        
        defer { fileHandle.closeFile() }
        
        // Read header
        guard let headerLine = readLine(from: fileHandle),
              let header = try? JSONDecoder().decode(WALHeader.self, from: headerLine) else {
            logger.warning("Invalid WAL header in file: \(url.lastPathComponent)")
            return
        }
        
        // Update sequence if needed
        if header.sequence > nextSequence {
            nextSequence = header.sequence
        }
        
        // Read entries
        while let recordData = readRecord(from: fileHandle) {
            if let entry = try? JSONDecoder().decode(WALEntry.self, from: recordData) {
                if !completedSequences.contains(entry.sequenceNumber) {
                    pendingEntries[entry.sequenceNumber] = entry
                }
            } else if let marker = try? JSONDecoder().decode(AnyWALMarker.self, from: recordData) {
                switch marker.type {
                case .completion:
                    if let seq = marker.sequenceNumber {
                        pendingEntries.removeValue(forKey: seq)
                        completedSequences.insert(seq)
                    }
                case .failure:
                    if let seq = marker.sequenceNumber {
                        pendingEntries.removeValue(forKey: seq)
                    }
                }
            }
        }
    }
    
    private func readLine(from handle: FileHandle) -> Data? {
        var lineData = Data()
        
        while true {
            guard let byte = try? handle.read(upToCount: 1), !byte.isEmpty else {
                return lineData.isEmpty ? nil : lineData
            }
            
            if byte[0] == 10 { // newline
                return lineData
            }
            
            lineData.append(byte[0])
        }
    }
    
    private func readRecord(from handle: FileHandle) -> Data? {
        // Read length
        guard let lengthData = try? handle.read(upToCount: 4),
              lengthData.count == 4 else {
            return nil
        }
        
        let length = lengthData.withUnsafeBytes { $0.load(as: UInt32.self) }
        
        // Read checksum
        guard let checksumData = try? handle.read(upToCount: 4),
              checksumData.count == 4 else {
            return nil
        }
        
        let checksum = checksumData.withUnsafeBytes { $0.load(as: UInt32.self) }
        
        // Read data
        guard let data = try? handle.read(upToCount: Int(length)),
              data.count == Int(length) else {
            return nil
        }
        
        // Verify checksum
        let calculatedChecksum = CRC32.checksum(data: data)
        guard calculatedChecksum == checksum else {
            logger.warning("Checksum mismatch in WAL record")
            return nil
        }
        
        // Skip newline
        _ = try? handle.read(upToCount: 1)
        
        return data
    }
    
    private func openNewLogFile() async throws {
        let logFile = generateLogFileName()
        
        FileManager.default.createFile(atPath: logFile.path, contents: nil)
        
        guard let handle = FileHandle(forWritingAtPath: logFile.path) else {
            throw WALError.fileCreationFailed
        }
        
        // Write header
        let header = WALHeader(version: 1, sequence: nextSequence)
        let headerData = try JSONEncoder().encode(header)
        try handle.write(contentsOf: headerData)
        try handle.write(contentsOf: "\n".data(using: .utf8)!)
        
        currentLogFile = logFile
        fileHandle = handle
    }
    
    private func generateLogFileName() -> URL {
        let timestamp = Int(Date().timeIntervalSince1970)
        return directory.appendingPathComponent("\(logFilePrefix)_\(timestamp).\(logFileExtension)")
    }
    
    private func shouldCheckpoint() -> Bool {
        return completedSequences.count > configuration.checkpointThreshold
    }
    
    private func shouldSync() -> Bool {
        // Simple time-based sync for now
        return true
    }
    
    private func checkpoint() async throws {
        logger.info("Creating checkpoint")
        
        // Write checkpoint file
        try await writeCheckpoint()
        
        // Clean up old completed sequences
        completedSequences.removeAll()
        
        // Compact if needed
        if pendingEntries.count < configuration.compactionThreshold {
            try await compact()
        }
    }
    
    private func writeCheckpoint() async throws {
        let checkpoint = WALCheckpoint(
            timestamp: Date(),
            nextSequence: nextSequence,
            completedSequences: completedSequences
        )
        
        let checkpointData = try JSONEncoder().encode(checkpoint)
        let checkpointURL = directory.appendingPathComponent(checkpointFile)
        
        // Write atomically
        try checkpointData.write(to: checkpointURL, options: .atomic)
    }
}

// MARK: - Supporting Types

// WALConfiguration is defined in Config/Storage/WALConfiguration.swift
// Adding extension for additional properties needed by WriteAheadLog
extension WALConfiguration {
    var syncInterval: TimeInterval { 1.0 }
    var maxPendingEntries: Int { 10000 }
    var checkpointThreshold: Int { 1000 }
    var compactionThreshold: Int { 100 }
}

/// WAL entry
struct WALEntry: Codable {
    let operation: WALOperation
    let timestamp: Date
    let sequenceNumber: UInt64
}

/// WAL operations
enum WALOperation: Codable {
    case store(key: String, data: Data, options: StorageOptions)
    case delete(key: String)
    
    // Custom Codable implementation since StorageOptions is not Codable
    enum CodingKeys: String, CodingKey {
        case type
        case key
        case data
        case compression
        case durability
        case priority
        case ttl
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        
        switch type {
        case "store":
            let key = try container.decode(String.self, forKey: .key)
            let data = try container.decode(Data.self, forKey: .data)
            let compressionRaw = try container.decode(String.self, forKey: .compression)
            let durabilityRaw = try container.decode(String.self, forKey: .durability)
            let priorityRaw = try container.decode(Int.self, forKey: .priority)
            let ttl = try container.decodeIfPresent(TimeInterval.self, forKey: .ttl)
            
            // Recreate StorageOptions
            let compression = CompressionLevel(rawValue: compressionRaw) ?? .adaptive
            let durability: DurabilityLevel = durabilityRaw == "none" ? .none :
                                              durabilityRaw == "eventual" ? .eventual :
                                              durabilityRaw == "strict" ? .strict :
                                              durabilityRaw == "extreme" ? .extreme : .standard
            let priority = StoragePriority(rawValue: priorityRaw) ?? .normal
            
            let options = StorageOptions(
                compression: compression,
                durability: durability,
                priority: priority,
                ttl: ttl
            )
            self = .store(key: key, data: data, options: options)
            
        case "delete":
            let key = try container.decode(String.self, forKey: .key)
            self = .delete(key: key)
            
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: "Unknown operation type")
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        switch self {
        case .store(let key, let data, let options):
            try container.encode("store", forKey: .type)
            try container.encode(key, forKey: .key)
            try container.encode(data, forKey: .data)
            try container.encode(options.compression.rawValue, forKey: .compression)
            
            // Encode durability as string
            let durabilityString: String
            switch options.durability {
            case .none: durabilityString = "none"
            case .eventual: durabilityString = "eventual"
            case .standard: durabilityString = "standard"
            case .strict: durabilityString = "strict"
            case .extreme: durabilityString = "extreme"
            }
            try container.encode(durabilityString, forKey: .durability)
            
            try container.encode(options.priority.rawValue, forKey: .priority)
            try container.encodeIfPresent(options.ttl, forKey: .ttl)
            
        case .delete(let key):
            try container.encode("delete", forKey: .type)
            try container.encode(key, forKey: .key)
        }
    }
}

/// WAL header
private struct WALHeader: Codable {
    let version: Int
    let sequence: UInt64
}

/// WAL checkpoint
private struct WALCheckpoint: Codable {
    let timestamp: Date
    let nextSequence: UInt64
    let completedSequences: Set<UInt64>
}

/// WAL markers
private protocol WALMarker: Codable {
    var type: MarkerType { get }
}

private enum MarkerType: String, Codable {
    case completion
    case failure
}

private struct CompletionMarker: WALMarker {
    let type: MarkerType
    let sequenceNumber: UInt64
    
    init(sequenceNumber: UInt64) {
        self.type = .completion
        self.sequenceNumber = sequenceNumber
    }
}

private struct FailureMarker: WALMarker {
    let type: MarkerType
    let sequenceNumber: UInt64
    let error: String
    
    init(sequenceNumber: UInt64, error: String) {
        self.type = .failure
        self.sequenceNumber = sequenceNumber
        self.error = error
    }
}

private struct AnyWALMarker: Codable {
    let type: MarkerType
    let sequenceNumber: UInt64?
    let error: String?
    
    init(_ marker: any WALMarker) {
        self.type = marker.type
        if let completion = marker as? CompletionMarker {
            self.sequenceNumber = completion.sequenceNumber
            self.error = nil
        } else if let failure = marker as? FailureMarker {
            self.sequenceNumber = failure.sequenceNumber
            self.error = failure.error
        } else {
            self.sequenceNumber = nil
            self.error = nil
        }
    }
}

/// WAL errors
private enum WALError: LocalizedError {
    case fileCreationFailed
    case noActiveLog
    case invalidSequence(UInt64)
    case corruptedEntry
    
    var errorDescription: String? {
        switch self {
        case .fileCreationFailed:
            return "Failed to create WAL file"
        case .noActiveLog:
            return "No active WAL file"
        case .invalidSequence(let seq):
            return "Invalid sequence number: \(seq)"
        case .corruptedEntry:
            return "Corrupted WAL entry"
        }
    }
}

// MARK: - CRC32 Implementation

private struct CRC32 {
    private static let polynomial: UInt32 = 0xEDB88320
    private static let table: [UInt32] = {
        var table = [UInt32](repeating: 0, count: 256)
        for i in 0..<256 {
            var c = UInt32(i)
            for _ in 0..<8 {
                c = (c & 1) != 0 ? (c >> 1) ^ polynomial : c >> 1
            }
            table[i] = c
        }
        return table
    }()
    
    static func checksum(data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFFFFFF
        
        for byte in data {
            let index = Int((crc ^ UInt32(byte)) & 0xFF)
            crc = (crc >> 8) ^ table[index]
        }
        
        return ~crc
    }
}
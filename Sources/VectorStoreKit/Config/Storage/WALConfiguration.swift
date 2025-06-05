// VectorStoreKit: Write-Ahead Log Configuration
//
// Configuration for WAL durability and performance

import Foundation

/// Write-ahead log configuration
public struct WALConfiguration: Sendable, Codable {
    public let enabled: Bool
    public let syncMode: WALSyncMode
    public let maxFileSize: Int
    public let compressionEnabled: Bool
    
    public init(
        enabled: Bool = true,
        syncMode: WALSyncMode = .periodic,
        maxFileSize: Int = 64 * 1024 * 1024, // 64 MB
        compressionEnabled: Bool = true
    ) {
        self.enabled = enabled
        self.syncMode = syncMode
        self.maxFileSize = maxFileSize
        self.compressionEnabled = compressionEnabled
    }
    
    public static let `default` = WALConfiguration(
        enabled: true,
        syncMode: .periodic,
        maxFileSize: 64 * 1024 * 1024, // 64 MB
        compressionEnabled: true
    )
    
    public static let highPerformance = WALConfiguration(
        enabled: true,
        syncMode: .async,
        maxFileSize: 256 * 1024 * 1024, // 256 MB
        compressionEnabled: false
    )
}

/// WAL synchronization modes
public enum WALSyncMode: Sendable, Codable {
    case synchronous  // Sync after each write
    case periodic     // Sync periodically
    case async        // Async sync
}
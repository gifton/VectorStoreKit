// VectorStoreKit: Storage Tier Supporting Types
//
// Shared types and utilities for storage tier implementations

import Foundation
import CryptoKit

// MARK: - Encryption Engine

/// Basic encryption engine implementation
public final class EncryptionEngine: Sendable {
    private let key: SymmetricKey
    
    public init(settings: EncryptionSettings) throws {
        // Generate or derive key based on settings
        self.key = SymmetricKey(size: .bits256)
    }
    
    public func encrypt(_ data: Data) throws -> Data {
        do {
            let sealedBox = try AES.GCM.seal(data, using: key)
            return sealedBox.combined ?? Data()
        } catch {
            throw StorageError.corruptedData("Failed to encrypt data: \(error.localizedDescription)")
        }
    }
    
    public func decrypt(_ data: Data) throws -> Data {
        do {
            let sealedBox = try AES.GCM.SealedBox(combined: data)
            return try AES.GCM.open(sealedBox, using: key)
        } catch {
            throw StorageError.corruptedData("Failed to decrypt data: \(error.localizedDescription)")
        }
    }
}

// MARK: - Storage Error Extensions

extension StorageError {
    static func corruptedData(_ message: String) -> StorageError {
        .retrieveFailed("Corrupted data: \(message)")
    }
}
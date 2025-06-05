// VectorStoreKit: Configuration Error
//
// Shared configuration validation error type

import Foundation

/// Shared configuration validation error
public enum ConfigurationError: LocalizedError {
    case invalidValue(String)
    case invalidConfiguration(String)
    case memoryLimitExceeded
    
    public var errorDescription: String? {
        switch self {
        case .invalidValue(let message):
            return message
        case .invalidConfiguration(let message):
            return message
        case .memoryLimitExceeded:
            return "Memory limit exceeded"
        }
    }
}
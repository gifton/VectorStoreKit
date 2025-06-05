// VectorStoreKit: Encryption Settings
//
// Configuration for data encryption at rest

import Foundation

/// Encryption settings for data at rest
public enum EncryptionSettings: Sendable, Codable, Equatable {
    case disabled
    case aes256
    case chacha20
    case custom(algorithm: String, keyDerivation: String)
    
    // Custom Codable implementation for associated values
    private enum CodingKeys: String, CodingKey {
        case type, algorithm, keyDerivation
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        
        switch type {
        case "disabled":
            self = .disabled
        case "aes256":
            self = .aes256
        case "chacha20":
            self = .chacha20
        case "custom":
            let algorithm = try container.decode(String.self, forKey: .algorithm)
            let keyDerivation = try container.decode(String.self, forKey: .keyDerivation)
            self = .custom(algorithm: algorithm, keyDerivation: keyDerivation)
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: "Unknown encryption type")
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        switch self {
        case .disabled:
            try container.encode("disabled", forKey: .type)
        case .aes256:
            try container.encode("aes256", forKey: .type)
        case .chacha20:
            try container.encode("chacha20", forKey: .type)
        case .custom(let algorithm, let keyDerivation):
            try container.encode("custom", forKey: .type)
            try container.encode(algorithm, forKey: .algorithm)
            try container.encode(keyDerivation, forKey: .keyDerivation)
        }
    }
}
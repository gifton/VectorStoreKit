// VectorStoreKit: Migration Settings
//
// Configuration for automatic tier management

import Foundation

/// Migration settings for automatic tier management
public enum MigrationSettings: Sendable, Codable {
    case disabled
    case automatic
    case intelligent
    case custom(rules: [MigrationRule])
    
    // Custom Codable implementation for associated values
    private enum CodingKeys: String, CodingKey {
        case type, rules
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        
        switch type {
        case "disabled":
            self = .disabled
        case "automatic":
            self = .automatic
        case "intelligent":
            self = .intelligent
        case "custom":
            let rules = try container.decode(Array<MigrationRule>.self, forKey: .rules)
            self = .custom(rules: rules)
        default:
            throw DecodingError.dataCorruptedError(forKey: .type, in: container, debugDescription: "Unknown migration type")
        }
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        
        switch self {
        case .disabled:
            try container.encode("disabled", forKey: .type)
        case .automatic:
            try container.encode("automatic", forKey: .type)
        case .intelligent:
            try container.encode("intelligent", forKey: .type)
        case .custom(let rules):
            try container.encode("custom", forKey: .type)
            try container.encode(rules, forKey: .rules)
        }
    }
}

/// Rule for custom migration settings
public struct MigrationRule: Sendable, Codable {
    public let condition: String
    public let action: String
    public let priority: Int
    
    public init(condition: String, action: String, priority: Int) {
        self.condition = condition
        self.action = action
        self.priority = priority
    }
}
// VectorStoreKit: Universe Configuration
//
// Configuration container for vector universe

import Foundation

/// Configuration container for vector universe
public struct UniverseConfiguration: Sendable {
    private var settings: [String: any Sendable] = [:]
    
    public init() {
        // Default settings
        settings["enableProfiling"] = true
        settings["enableAnalytics"] = true
    }
    
    /// Build store configuration from universe settings
    func buildStoreConfiguration() -> StoreConfiguration {
        StoreConfiguration(
            name: settings["name"] as? String ?? "VectorStore",
            enableProfiling: settings["enableProfiling"] as? Bool ?? true,
            enableAnalytics: settings["enableAnalytics"] as? Bool ?? true,
            integrityCheckInterval: settings["integrityCheckInterval"] as? TimeInterval ?? 3600,
            optimizationThreshold: settings["optimizationThreshold"] as? Int ?? 100_000
        )
    }
    
    /// Get a configuration value with a default
    public func get<T>(_ key: String, default defaultValue: T) -> T {
        return settings[key] as? T ?? defaultValue
    }
    
    /// Set a configuration value
    public mutating func set<T: Sendable>(_ key: String, value: T) {
        settings[key] = value
    }
}
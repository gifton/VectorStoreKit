// VectorStoreKit: Store Configuration
//
// Core configuration for VectorStore

import Foundation

/// Store configuration
public struct StoreConfiguration: Sendable, Codable {
    public let name: String
    public let enableProfiling: Bool
    public let enableAnalytics: Bool
    public let integrityCheckInterval: TimeInterval
    public let optimizationThreshold: Int
    
    public init(
        name: String = "VectorStore",
        enableProfiling: Bool = true,
        enableAnalytics: Bool = true,
        integrityCheckInterval: TimeInterval = 3600,
        optimizationThreshold: Int = 100_000
    ) {
        self.name = name
        self.enableProfiling = enableProfiling
        self.enableAnalytics = enableAnalytics
        self.integrityCheckInterval = integrityCheckInterval
        self.optimizationThreshold = optimizationThreshold
    }
    
    public static let research = StoreConfiguration(
        name: "ResearchVectorStore",
        enableProfiling: true,
        enableAnalytics: true,
        integrityCheckInterval: 1800,
        optimizationThreshold: 50_000
    )
    
    public static let production = StoreConfiguration(
        name: "ProductionVectorStore",
        enableProfiling: false,
        enableAnalytics: false,
        integrityCheckInterval: 7200,
        optimizationThreshold: 200_000
    )
    
    func validate() throws {
        guard integrityCheckInterval > 0 else {
            throw ConfigurationError.invalidValue("Integrity check interval must be positive")
        }
        guard optimizationThreshold > 0 else {
            throw ConfigurationError.invalidValue("Optimization threshold must be positive")
        }
    }
}

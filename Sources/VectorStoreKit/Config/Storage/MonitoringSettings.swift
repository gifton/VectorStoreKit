// VectorStoreKit: Monitoring Settings
//
// Configuration for performance monitoring

import Foundation

/// Monitoring settings
public struct MonitoringSettings: Sendable, Codable {
    public let enabled: Bool
    public let detailedMetrics: Bool
    public let performanceLogging: Bool
    
    public init(
        enabled: Bool = true,
        detailedMetrics: Bool = false,
        performanceLogging: Bool = false
    ) {
        self.enabled = enabled
        self.detailedMetrics = detailedMetrics
        self.performanceLogging = performanceLogging
    }
    
    public static let enabled = MonitoringSettings(
        enabled: true,
        detailedMetrics: false,
        performanceLogging: false
    )
    
    public static let comprehensive = MonitoringSettings(
        enabled: true,
        detailedMetrics: true,
        performanceLogging: true
    )
}
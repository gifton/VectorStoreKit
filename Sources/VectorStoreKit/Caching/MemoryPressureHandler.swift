// VectorStoreKit: Memory Pressure Handler
//
// System memory pressure monitoring and response

import Foundation
import os.log
#if canImport(Darwin)
import Darwin
#endif
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

// MARK: - Memory Pressure Level

/// Memory pressure levels
public enum MemoryPressureLevel: Int, Sendable, Comparable {
    case normal = 0
    case warning = 1
    case urgent = 2
    case critical = 3
    
    var description: String {
        switch self {
        case .normal: return "Normal"
        case .warning: return "Warning"
        case .urgent: return "Urgent"
        case .critical: return "Critical"
        }
    }
    
    public static func < (lhs: MemoryPressureLevel, rhs: MemoryPressureLevel) -> Bool {
        return lhs.rawValue < rhs.rawValue
    }
}

// MARK: - Memory Pressure Handler

/// Monitors system memory pressure and provides adaptive responses
public actor MemoryPressureHandler {
    
    // MARK: - Properties
    
    private var currentPressure: MemoryPressureLevel = .normal
    private var subscribers: [UUID: (MemoryPressureLevel) -> Void] = [:]
    private var isMonitoring = false
    
    // Memory tracking
    private var lastMemorySnapshot: MemorySnapshot?
    private var memoryHistory: [MemorySnapshot] = []
    private let maxHistorySize = 100
    
    // Thresholds
    private let warningThreshold: Float = 0.75   // 75% memory usage
    private let urgentThreshold: Float = 0.85    // 85% memory usage
    private let criticalThreshold: Float = 0.95  // 95% memory usage
    
    // Logger
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MemoryPressure")
    
    // Platform-specific monitoring
    #if os(macOS)
    private var memoryPressureSource: DispatchSourceMemoryPressure?
    #endif
    
    private var monitoringTask: Task<Void, Never>?
    
    // MARK: - Types
    
    private struct MemorySnapshot {
        let timestamp: Date
        let usedMemory: Int64
        let totalMemory: Int64
        let availableMemory: Int64
        let pressure: MemoryPressureLevel
        
        var usageRatio: Float {
            totalMemory > 0 ? Float(usedMemory) / Float(totalMemory) : 0
        }
    }
    
    // MARK: - Initialization
    
    public init() {
        setupPlatformMonitoring()
        startMonitoring()
    }
    
    deinit {
        monitoringTask?.cancel()
        #if os(macOS)
        memoryPressureSource?.cancel()
        #endif
    }
    
    // MARK: - Public Methods
    
    /// Get current memory pressure level
    public func currentPressureLevel() -> MemoryPressureLevel {
        currentPressure
    }
    
    /// Subscribe to pressure updates
    public func subscribe(
        handler: @escaping (MemoryPressureLevel) -> Void
    ) -> UUID {
        let id = UUID()
        subscribers[id] = handler
        
        // Send current state immediately
        handler(currentPressure)
        
        return id
    }
    
    /// Unsubscribe from pressure updates
    public func unsubscribe(id: UUID) {
        subscribers.removeValue(forKey: id)
    }
    
    /// Stream pressure updates
    public func pressureUpdates() -> AsyncStream<MemoryPressureLevel> {
        AsyncStream { continuation in
            let id = subscribe { pressure in
                continuation.yield(pressure)
            }
            
            continuation.onTermination = { [weak self] _ in
                Task {
                    await self?.unsubscribe(id: id)
                }
            }
        }
    }
    
    /// Get memory statistics
    public func memoryStatistics() -> MemoryStatistics {
        let snapshot = getCurrentMemorySnapshot()
        
        return MemoryStatistics(
            usedMemory: snapshot.usedMemory,
            totalMemory: snapshot.totalMemory,
            availableMemory: snapshot.availableMemory,
            usageRatio: snapshot.usageRatio,
            pressure: snapshot.pressure,
            trend: analyzeTrend()
        )
    }
    
    /// Force a manual memory pressure check
    public func checkMemoryPressure() {
        let snapshot = getCurrentMemorySnapshot()
        updatePressureLevel(from: snapshot)
    }
    
    // MARK: - Private Methods
    
    private func setupPlatformMonitoring() {
        #if os(macOS)
        setupMacOSMonitoring()
        #elseif os(iOS)
        setupiOSMonitoring()
        #endif
    }
    
    #if os(macOS)
    private func setupMacOSMonitoring() {
        // Create dispatch source for memory pressure events
        memoryPressureSource = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: .global(qos: .background)
        )
        
        memoryPressureSource?.setEventHandler { [weak self] in
            Task {
                await self?.handleSystemMemoryPressure()
            }
        }
        
        memoryPressureSource?.resume()
    }
    #endif
    
    #if os(iOS)
    private func setupiOSMonitoring() {
        // Register for memory warnings
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task {
                await self?.handleSystemMemoryPressure()
            }
        }
    }
    #endif
    
    private func startMonitoring() {
        guard !isMonitoring else { return }
        isMonitoring = true
        
        monitoringTask = Task {
            while !Task.isCancelled {
                checkMemoryPressure()
                
                // Check every 5 seconds
                try? await Task.sleep(for: .seconds(5))
            }
        }
    }
    
    private func getCurrentMemorySnapshot() -> MemorySnapshot {
        let memInfo = getMemoryInfo()
        
        return MemorySnapshot(
            timestamp: Date(),
            usedMemory: memInfo.used,
            totalMemory: memInfo.total,
            availableMemory: memInfo.available,
            pressure: calculatePressureLevel(usageRatio: Float(memInfo.used) / Float(memInfo.total))
        )
    }
    
    private func getMemoryInfo() -> (total: Int64, used: Int64, available: Int64) {
        #if os(macOS)
        return getMacOSMemoryInfo()
        #elseif os(iOS)
        return getiOSMemoryInfo()
        #else
        // Fallback for other platforms
        return (total: ProcessInfo.processInfo.physicalMemory,
                used: 0,
                available: ProcessInfo.processInfo.physicalMemory)
        #endif
    }
    
    #if os(macOS)
    private func getMacOSMemoryInfo() -> (total: Int64, used: Int64, available: Int64) {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let total = ProcessInfo.processInfo.physicalMemory
            let used = Int64(info.resident_size)
            let available = Int64(total) - used
            
            return (total: Int64(total), used: used, available: available)
        }
        #endif
        
        // Fallback
        let total = ProcessInfo.processInfo.physicalMemory
        return (total: Int64(total), used: 0, available: Int64(total))
    }
    #endif
    
    #if os(iOS)
    private func getiOSMemoryInfo() -> (total: Int64, used: Int64, available: Int64) {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_,
                         task_flavor_t(MACH_TASK_BASIC_INFO),
                         $0,
                         &count)
            }
        }
        
        if result == KERN_SUCCESS {
            let total = ProcessInfo.processInfo.physicalMemory
            let used = Int64(info.resident_size)
            let available = Int64(total) - used
            
            return (total: Int64(total), used: used, available: available)
        }
        #endif
        
        // Fallback
        let total = ProcessInfo.processInfo.physicalMemory
        return (total: Int64(total), used: 0, available: Int64(total))
    }
    #endif
    
    private func calculatePressureLevel(usageRatio: Float) -> MemoryPressureLevel {
        switch usageRatio {
        case ..<warningThreshold:
            return .normal
        case ..<urgentThreshold:
            return .warning
        case ..<criticalThreshold:
            return .urgent
        default:
            return .critical
        }
    }
    
    private func updatePressureLevel(from snapshot: MemorySnapshot) {
        let newPressure = snapshot.pressure
        
        // Store snapshot
        memoryHistory.append(snapshot)
        if memoryHistory.count > maxHistorySize {
            memoryHistory.removeFirst()
        }
        lastMemorySnapshot = snapshot
        
        // Check if pressure changed
        if newPressure != currentPressure {
            let oldPressure = currentPressure
            currentPressure = newPressure
            
            logger.info("Memory pressure changed: \(oldPressure.description) -> \(newPressure.description)")
            
            // Notify subscribers
            notifySubscribers(newPressure)
        }
    }
    
    private func handleSystemMemoryPressure() async {
        logger.warning("System memory pressure event received")
        
        // Force immediate check
        checkMemoryPressure()
        
        // If not already critical, escalate
        if currentPressure != .critical {
            currentPressure = .urgent
            notifySubscribers(currentPressure)
        }
    }
    
    private func notifySubscribers(_ pressure: MemoryPressureLevel) {
        for handler in subscribers.values {
            handler(pressure)
        }
    }
    
    private func analyzeTrend() -> MemoryTrend {
        guard memoryHistory.count >= 5 else { return .stable }
        
        let recentSnapshots = Array(memoryHistory.suffix(10))
        let usageRatios = recentSnapshots.map { $0.usageRatio }
        
        // Calculate trend using linear regression
        let n = Float(usageRatios.count)
        let indices = (0..<usageRatios.count).map { Float($0) }
        
        let sumX = indices.reduce(0, +)
        let sumY = usageRatios.reduce(0, +)
        let sumXY = zip(indices, usageRatios).map { $0 * $1 }.reduce(0, +)
        let sumXX = indices.map { $0 * $0 }.reduce(0, +)
        
        let slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
        
        if slope > 0.01 {
            return .increasing
        } else if slope < -0.01 {
            return .decreasing
        } else {
            return .stable
        }
    }
}

// MARK: - Supporting Types

/// Memory statistics
public struct MemoryStatistics: Sendable {
    public let usedMemory: Int64
    public let totalMemory: Int64
    public let availableMemory: Int64
    public let usageRatio: Float
    public let pressure: MemoryPressureLevel
    public let trend: MemoryTrend
    
    public var usedMemoryMB: Double {
        Double(usedMemory) / 1024 / 1024
    }
    
    public var totalMemoryMB: Double {
        Double(totalMemory) / 1024 / 1024
    }
    
    public var availableMemoryMB: Double {
        Double(availableMemory) / 1024 / 1024
    }
}

/// Memory usage trend
public enum MemoryTrend: Sendable {
    case increasing
    case stable
    case decreasing
}
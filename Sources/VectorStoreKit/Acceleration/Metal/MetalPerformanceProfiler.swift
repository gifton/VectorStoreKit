// VectorStoreKit: Metal Performance Profiler
//
// Advanced performance profiling with GPU counters and timing measurements
//

import Foundation
@preconcurrency import Metal
@preconcurrency import MetalPerformanceShaders
import os.log
import os.signpost

/// Advanced performance profiler with Metal GPU counters and detailed metrics
public actor MetalPerformanceProfiler {
    
    // MARK: - Properties
    
    private let device: MetalDevice
    private let enabled: Bool
    private let signpostLog: OSLog
    private var signpostIntervalState: OSSignpostIntervalState?
    
    // Profiling data storage
    private var operationProfiles: [OperationProfile] = []
    private var kernelProfiles: [String: KernelProfile] = [:]
    private var memoryProfiles: [MemoryProfileEntry] = []
    
    // GPU timing tracking
    private let supportsGPUTiming: Bool
    private var kernelTimings: [String: [TimeInterval]] = [:]
    private let maxTimingHistory = 1000
    
    // Performance counters
    private var counterSets: [MTLCounterSet]?
    private var currentCounterSample: MTLCounterSampleBuffer?
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "PerformanceProfiler")
    
    // MARK: - Initialization
    
    public init(device: MetalDevice, enabled: Bool = true) async throws {
        self.device = device
        self.enabled = enabled
        self.signpostLog = OSLog(subsystem: "VectorStoreKit", category: .pointsOfInterest)
        
        // Check GPU timing support based on OS version
        // GPU timing (gpuStartTime/gpuEndTime) is available on macOS 10.15+ and iOS 10.3+
        if #available(macOS 10.15, iOS 10.3, *) {
            self.supportsGPUTiming = true
        } else {
            self.supportsGPUTiming = false
            logger.info("GPU timing not available on this OS version - will use CPU timing")
        }
        
        if enabled {
            // Initialize performance counters if available
            await initializePerformanceCounters()
        }
        
        logger.info("Performance profiler initialized - GPU timing: \(self.supportsGPUTiming)")
    }
    
    // MARK: - Profiling Control
    
    /// Begin profiling an operation
    public func beginOperation(
        _ name: String,
        category: ProfilingCategory,
        metadata: [String: Any] = [:]
    ) -> OperationHandle {
        guard enabled else {
            return OperationHandle(id: UUID(), signpostID: .invalid)
        }
        
        let id = UUID()
        let signpostID = OSSignpostID(log: signpostLog, object: self)
        
        os_signpost(.begin, log: signpostLog, name: "Operation", signpostID: signpostID)
        
        let profile = OperationProfile(
            id: id,
            name: name,
            category: category,
            startTime: CFAbsoluteTimeGetCurrent()
        )
        
        operationProfiles.append(profile)
        
        return OperationHandle(id: id, signpostID: signpostID)
    }
    
    /// End profiling an operation
    public func endOperation(
        _ handle: OperationHandle,
        metrics: ProfilerOperationMetrics? = nil
    ) {
        guard enabled else { return }
        
        os_signpost(.end, log: signpostLog, name: "Operation", signpostID: handle.signpostID)
        
        if let index = operationProfiles.firstIndex(where: { $0.id == handle.id }) {
            operationProfiles[index].endTime = CFAbsoluteTimeGetCurrent()
            operationProfiles[index].metrics = metrics
        }
    }
    
    /// Profile a Metal command buffer execution
    /// 
    /// - Important: GPU timing is only available on macOS 10.15+ and iOS 10.3+.
    ///   On older systems, only CPU timing is available.
    /// - Note: For precise kernel-level timing, use Metal System Trace in Instruments.
    public func profileCommandBuffer(
        _ commandBuffer: MTLCommandBuffer,
        label: String,
        kernels: [KernelDescriptor]
    ) async -> CommandBufferProfile? {
        guard enabled else { return nil }
        
        let startCPUTime = CFAbsoluteTimeGetCurrent()
        var gpuStartTime: TimeInterval = 0
        var gpuEndTime: TimeInterval = 0
        var hasGPUTiming = false
        
        // Add GPU timing handlers if available
        if supportsGPUTiming {
            commandBuffer.addCompletedHandler { [weak self] buffer in
                // GPU timing properties are available on macOS 10.15+ and iOS 10.3+
                if #available(macOS 10.15, iOS 10.3, *) {
                    gpuStartTime = buffer.gpuStartTime
                    gpuEndTime = buffer.gpuEndTime
                    hasGPUTiming = gpuStartTime > 0 && gpuEndTime > 0
                    
                    if hasGPUTiming {
                        self?.logger.debug("GPU timing available: start=\(gpuStartTime), end=\(gpuEndTime)")
                    }
                } else {
                    self?.logger.debug("GPU timing not available on this OS version")
                }
            }
        }
        
        // Track kernel executions
        for kernel in kernels {
            await recordKernelExecution(kernel)
        }
        
        // Wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let endCPUTime = CFAbsoluteTimeGetCurrent()
        
        // Calculate timings
        let cpuTime = endCPUTime - startCPUTime
        let gpuTime: TimeInterval
        
        if hasGPUTiming && gpuEndTime > gpuStartTime {
            gpuTime = gpuEndTime - gpuStartTime
            
            // Record kernel timings if GPU timing is available
            if commandBuffer.status == .completed && kernels.count > 0 {
                let kernelTime = gpuTime / Double(kernels.count)
                for kernel in kernels {
                    recordKernelTiming(kernel.name, duration: kernelTime)
                }
            }
        } else {
            // Fallback: estimate GPU time as CPU time (this is just an approximation)
            gpuTime = cpuTime
            logger.debug("Using CPU time as GPU time estimate (no GPU timing available)")
        }
        
        let profile = CommandBufferProfile(
            label: label,
            cpuTime: cpuTime,
            gpuTime: gpuTime,
            kernelCount: kernels.count,
            status: commandBuffer.status,
            error: commandBuffer.error
        )
        
        logger.debug("""
            Command buffer '\(label)': \
            CPU=\(cpuTime * 1000)ms, \
            GPU=\(hasGPUTiming ? "\(gpuTime * 1000)" : "~\(gpuTime * 1000)")ms, \
            \(kernels.count) kernels
            """)
        
        return profile
    }
    
    /// Profile kernel execution with timing
    public func profileKernelExecution(
        _ encoder: MTLComputeCommandEncoder,
        kernelName: String,
        threadgroupSize: MTLSize,
        gridSize: MTLSize
    ) {
        guard enabled else { return }
        
        // Calculate total threads
        let totalThreads = gridSize.width * gridSize.height * gridSize.depth
        
        // Record kernel descriptor for later timing analysis
        let descriptor = KernelDescriptor(
            name: kernelName,
            totalThreads: totalThreads
        )
        
        // The actual timing will be captured via command buffer completion handler
        Task {
            await recordKernelExecution(descriptor)
        }
    }
    
    /// Get kernel timing statistics
    public func getKernelTimingStats(for kernelName: String) -> KernelTimingStats? {
        guard let timings = kernelTimings[kernelName], !timings.isEmpty else { return nil }
        
        let sorted = timings.sorted()
        let count = timings.count
        let total = timings.reduce(0, +)
        
        return KernelTimingStats(
            count: count,
            totalTime: total,
            averageTime: total / Double(count),
            minTime: sorted.first!,
            maxTime: sorted.last!,
            medianTime: sorted[count / 2]
        )
    }
    
    // MARK: - Memory Profiling
    
    /// Record memory allocation
    public func recordMemoryAllocation(
        size: Int,
        type: MemoryAllocationType,
        label: String? = nil
    ) {
        guard enabled else { return }
        
        let profile = MemoryProfileEntry(
            timestamp: Date(),
            allocationType: type,
            size: size,
            label: label,
            isAllocation: true
        )
        
        memoryProfiles.append(profile)
        
        // Maintain history limit
        if memoryProfiles.count > 10000 {
            memoryProfiles.removeFirst(5000)
        }
    }
    
    /// Record memory deallocation
    public func recordMemoryDeallocation(
        size: Int,
        type: MemoryAllocationType
    ) {
        guard enabled else { return }
        
        let profile = MemoryProfileEntry(
            timestamp: Date(),
            allocationType: type,
            size: size,
            label: nil,
            isAllocation: false
        )
        
        memoryProfiles.append(profile)
    }
    
    // MARK: - Analysis
    
    /// Get performance summary
    public func getPerformanceSummary() -> PerformanceSummary {
        let completedOperations = operationProfiles.filter { $0.endTime != nil }
        
        // Group by category
        let categorySummaries = Dictionary(
            grouping: completedOperations,
            by: { $0.category }
        ).mapValues { operations in
            CategorySummary(
                totalOperations: operations.count,
                totalTime: operations.compactMap { $0.duration }.reduce(0, +),
                averageTime: operations.compactMap { $0.duration }.average(),
                operations: operations.map { op in
                    OperationSummary(
                        name: op.name,
                        duration: op.duration ?? 0,
                        metrics: op.metrics
                    )
                }
            )
        }
        
        // Analyze kernel usage
        let kernelSummaries = kernelProfiles.map { name, profile in
            KernelSummary(
                name: name,
                executionCount: profile.executionCount,
                totalThreads: profile.totalThreads,
                averageThreadsPerExecution: profile.totalThreads / max(1, profile.executionCount)
            )
        }.sorted { $0.executionCount > $1.executionCount }
        
        // Memory analysis
        let currentMemoryUsage = calculateCurrentMemoryUsage()
        let peakMemoryUsage = calculatePeakMemoryUsage()
        
        return PerformanceSummary(
            categorySummaries: categorySummaries,
            kernelSummaries: kernelSummaries,
            currentMemoryUsage: currentMemoryUsage,
            peakMemoryUsage: peakMemoryUsage,
            totalOperations: operationProfiles.count
        )
    }
    
    /// Export profiling data
    public func exportProfilingData() -> ProfilingReport {
        let summary = getPerformanceSummary()
        
        return ProfilingReport(
            timestamp: Date(),
            deviceInfo: DeviceInfo(
                name: device.capabilities.deviceName,
                maxThreadsPerThreadgroup: device.capabilities.maxThreadsPerThreadgroup,
                memoryBandwidth: device.capabilities.memoryBandwidth
            ),
            summary: summary,
            operationProfiles: operationProfiles,
            kernelProfiles: kernelProfiles.map { KernelProfileData(name: $0.key, profile: $0.value) },
            memoryProfiles: memoryProfiles.suffix(1000) // Last 1000 entries
        )
    }
    
    /// Reset all profiling data
    public func reset() {
        operationProfiles.removeAll()
        kernelProfiles.removeAll()
        memoryProfiles.removeAll()
        kernelTimings.removeAll()
        currentCounterSample = nil
        
        logger.info("Profiling data reset")
    }
    
    // MARK: - Private Methods
    
    private func initializePerformanceCounters() async {
        // Get available counter sets
        let availableCounterSets = await device.device.counterSets ?? []
        
        // Look for GPU performance counters
        self.counterSets = availableCounterSets.filter { counterSet in
            // Filter for relevant performance counters
            counterSet.name.contains("GPU") || counterSet.name.contains("Performance")
        }
        
        if !self.counterSets!.isEmpty {
            logger.info("Found \(self.counterSets!.count) performance counter sets")
        }
    }
    
    private func recordKernelExecution(_ kernel: KernelDescriptor) async {
        if kernelProfiles[kernel.name] == nil {
            kernelProfiles[kernel.name] = KernelProfile()
        }
        
        kernelProfiles[kernel.name]?.executionCount += 1
        kernelProfiles[kernel.name]?.totalThreads += kernel.totalThreads
    }
    
    private func recordKernelTiming(_ kernelName: String, duration: TimeInterval) {
        if kernelTimings[kernelName] == nil {
            kernelTimings[kernelName] = []
        }
        
        kernelTimings[kernelName]?.append(duration)
        
        // Maintain history limit per kernel
        if let count = kernelTimings[kernelName]?.count, count > maxTimingHistory {
            kernelTimings[kernelName]?.removeFirst(count - maxTimingHistory)
        }
    }
    
    private func calculateCurrentMemoryUsage() -> Int {
        var currentUsage = 0
        for profile in memoryProfiles {
            if profile.isAllocation {
                currentUsage += profile.size
            } else {
                currentUsage -= profile.size
            }
        }
        return max(0, currentUsage)
    }
    
    private func calculatePeakMemoryUsage() -> Int {
        var currentUsage = 0
        var peakUsage = 0
        
        for profile in memoryProfiles {
            if profile.isAllocation {
                currentUsage += profile.size
            } else {
                currentUsage -= profile.size
            }
            peakUsage = max(peakUsage, currentUsage)
        }
        
        return peakUsage
    }
}

// MARK: - Supporting Types

/// Profiling category for operations
public enum ProfilingCategory: String, CaseIterable, Sendable {
    case distanceComputation = "Distance"
    case indexing = "Indexing"
    case quantization = "Quantization"
    case search = "Search"
    case training = "Training"
    case io = "I/O"
    case memory = "Memory"
}

/// Handle for tracking operations
public struct OperationHandle {
    let id: UUID
    let signpostID: OSSignpostID
}

/// Operation metrics
public struct OperationMetrics: Sendable {
    public let itemsProcessed: Int
    public let bytesProcessed: Int
    public let gpuUtilization: Float?
    
    public init(itemsProcessed: Int, bytesProcessed: Int, gpuUtilization: Float? = nil) {
        self.itemsProcessed = itemsProcessed
        self.bytesProcessed = bytesProcessed
        self.gpuUtilization = gpuUtilization
    }
}

/// Kernel descriptor
public struct KernelDescriptor {
    public let name: String
    public let totalThreads: Int
    
    public init(name: String, totalThreads: Int) {
        self.name = name
        self.totalThreads = totalThreads
    }
}

/// Command buffer profile
public struct CommandBufferProfile: Sendable {
    public let label: String
    public let cpuTime: TimeInterval
    public let gpuTime: TimeInterval
    public let kernelCount: Int
    public let status: MTLCommandBufferStatus
    public let error: Error?
}

/// Memory allocation type
public enum MemoryAllocationType: String, Sendable {
    case buffer = "Buffer"
    case texture = "Texture"
    case heap = "Heap"
    case cache = "Cache"
}

/// Kernel timing statistics
public struct KernelTimingStats: Sendable {
    public let count: Int
    public let totalTime: TimeInterval
    public let averageTime: TimeInterval
    public let minTime: TimeInterval
    public let maxTime: TimeInterval
    public let medianTime: TimeInterval
}

// MARK: - Internal Types

/// Operation profile data
public struct OperationProfile: Sendable, Codable {
    public let id: UUID
    public let name: String
    public let category: ProfilingCategory
    public let startTime: CFAbsoluteTime
    public var endTime: CFAbsoluteTime?
    // Metadata is not Codable, so we exclude it from encoding
    public var metrics: ProfilerOperationMetrics?
    
    public var duration: TimeInterval? {
        guard let endTime = endTime else { return nil }
        return endTime - startTime
    }
}

/// Kernel execution profile
public struct KernelProfile: Sendable, Codable {
    public var executionCount: Int = 0
    public var totalThreads: Int = 0
}

/// Memory profile entry
public struct MemoryProfileEntry: Sendable, Codable {
    public let timestamp: Date
    public let allocationType: MemoryAllocationType
    public let size: Int
    public let label: String?
    public let isAllocation: Bool
}

// MARK: - Report Types

/// Performance summary
public struct PerformanceSummary: Sendable {
    public let categorySummaries: [ProfilingCategory: CategorySummary]
    public let kernelSummaries: [KernelSummary]
    public let currentMemoryUsage: Int
    public let peakMemoryUsage: Int
    public let totalOperations: Int
}

/// Category summary
public struct CategorySummary: Sendable {
    public let totalOperations: Int
    public let totalTime: TimeInterval
    public let averageTime: TimeInterval
    public let operations: [OperationSummary]
}

/// Operation summary
public struct OperationSummary: Sendable {
    public let name: String
    public let duration: TimeInterval
    public let metrics: ProfilerOperationMetrics?
}

/// Kernel summary
public struct KernelSummary: Sendable {
    public let name: String
    public let executionCount: Int
    public let totalThreads: Int
    public let averageThreadsPerExecution: Int
}

/// Complete profiling report
public struct ProfilingReport: Sendable, Codable {
    public let timestamp: Date
    public let deviceInfo: DeviceInfo
    public let summary: PerformanceSummary
    public let operationProfiles: [OperationProfile]
    public let kernelProfiles: [KernelProfileData]
    public let memoryProfiles: [MemoryProfileEntry]
}

/// Device information
public struct DeviceInfo: Sendable, Codable {
    public let name: String
    public let maxThreadsPerThreadgroup: Int
    public let memoryBandwidth: Float
}

/// Kernel profile data for export
public struct KernelProfileData: Sendable, Codable {
    public let name: String
    public let profile: KernelProfile
}

// MARK: - Extensions

extension Array where Element == TimeInterval {
    func average() -> TimeInterval {
        guard !isEmpty else { return 0 }
        return reduce(0, +) / TimeInterval(count)
    }
}

// Make types Codable for export
extension PerformanceSummary: Codable {}
extension CategorySummary: Codable {}
extension OperationSummary: Codable {
    enum CodingKeys: String, CodingKey {
        case name, duration
        // metrics intentionally excluded as it's not Codable
    }
    
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try container.decode(String.self, forKey: .name)
        self.duration = try container.decode(TimeInterval.self, forKey: .duration)
        self.metrics = nil // Cannot decode non-Codable metrics
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(duration, forKey: .duration)
        // metrics intentionally not encoded
    }
}
extension KernelSummary: Codable {}
extension OperationProfile {
    enum CodingKeys: String, CodingKey {
        case id, name, category, startTime, endTime
    }
}
// MemoryProfileEntry Codable conformance is in struct definition
// ProfilerOperationMetrics intentionally not Codable due to PercentileData
extension ProfilingCategory: Codable {}
extension MemoryAllocationType: Codable {}
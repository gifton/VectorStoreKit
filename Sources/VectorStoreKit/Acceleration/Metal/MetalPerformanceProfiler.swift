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
    
    // GPU timestamp support
    private let supportsTimestamps: Bool
    private var timestampBuffer: MTLBuffer?
    private let maxTimestamps = 1000
    
    // Performance counters
    private var counterSets: [MTLCounterSet]?
    private var currentCounterSample: MTLCounterSampleBuffer?
    
    private let logger = Logger(subsystem: "VectorStoreKit", category: "PerformanceProfiler")
    
    // MARK: - Initialization
    
    public init(device: MetalDevice, enabled: Bool = true) async throws {
        self.device = device
        self.enabled = enabled
        self.signpostLog = OSLog(subsystem: "VectorStoreKit", category: .pointsOfInterest)
        
        // Check timestamp support
        self.supportsTimestamps = await device.device.supportsCounterSampling(.timestamp)
        
        if enabled && supportsTimestamps {
            // Allocate timestamp buffer
            let timestampSize = MemoryLayout<MTLTimestamp>.size * maxTimestamps * 2
            self.timestampBuffer = await device.device.makeBuffer(
                length: timestampSize,
                options: .storageModeShared
            )
            
            // Initialize performance counters if available
            await initializePerformanceCounters()
        }
        
        logger.info("Performance profiler initialized - timestamps: \(self.supportsTimestamps)")
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
    public func profileCommandBuffer(
        _ commandBuffer: MTLCommandBuffer,
        label: String,
        kernels: [KernelDescriptor]
    ) async -> CommandBufferProfile? {
        guard enabled else { return nil }
        
        let startCPUTime = CFAbsoluteTimeGetCurrent()
        var gpuStartTime: CFAbsoluteTime = 0
        var gpuEndTime: CFAbsoluteTime = 0
        
        // Add GPU timestamp handlers if supported
        if supportsTimestamps {
            commandBuffer.addScheduledHandler { buffer in
                gpuStartTime = buffer.gpuStartTime
            }
            
            commandBuffer.addCompletedHandler { buffer in
                gpuEndTime = buffer.gpuEndTime
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
        let gpuTime = supportsTimestamps ? (gpuEndTime - gpuStartTime) : cpuTime
        
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
            GPU=\(gpuTime * 1000)ms, \
            \(kernels.count) kernels
            """)
        
        return profile
    }
    
    /// Add GPU timestamp markers
    public func addTimestampMarker(
        to encoder: MTLComputeCommandEncoder,
        label: String,
        at location: TimestampLocation
    ) -> Int? {
        guard enabled && supportsTimestamps,
              let buffer = timestampBuffer else { return nil }
        
        let index = operationProfiles.count % maxTimestamps
        let offset = index * MemoryLayout<MTLTimestamp>.size * 2
        
        switch location {
        case .begin:
            encoder.writeTimestamp(to: buffer, atIndex: offset)
        case .end:
            encoder.writeTimestamp(to: buffer, atIndex: offset + MemoryLayout<MTLTimestamp>.size)
        }
        
        return index
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
        
        if !counterSets!.isEmpty {
            logger.info("Found \(counterSets!.count) performance counter sets")
        }
    }
    
    private func recordKernelExecution(_ kernel: KernelDescriptor) async {
        if kernelProfiles[kernel.name] == nil {
            kernelProfiles[kernel.name] = KernelProfile()
        }
        
        kernelProfiles[kernel.name]?.executionCount += 1
        kernelProfiles[kernel.name]?.totalThreads += kernel.totalThreads
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
public enum ProfilingCategory: String, CaseIterable {
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
public enum MemoryAllocationType: String {
    case buffer = "Buffer"
    case texture = "Texture"
    case heap = "Heap"
    case cache = "Cache"
}

/// Timestamp location
public enum TimestampLocation {
    case begin
    case end
}

// MARK: - Internal Types

/// Operation profile data
private struct OperationProfile {
    let id: UUID
    let name: String
    let category: ProfilingCategory
    let startTime: CFAbsoluteTime
    var endTime: CFAbsoluteTime?
    // Metadata is not Codable, so we exclude it from encoding
    var metrics: ProfilerOperationMetrics?
    
    var duration: TimeInterval? {
        guard let endTime = endTime else { return nil }
        return endTime - startTime
    }
}

/// Kernel execution profile
private struct KernelProfile: Codable {
    var executionCount: Int = 0
    var totalThreads: Int = 0
}

/// Memory profile entry
private struct MemoryProfileEntry {
    let timestamp: Date
    let allocationType: MemoryAllocationType
    let size: Int
    let label: String?
    let isAllocation: Bool
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
extension OperationSummary: Codable {}
extension KernelSummary: Codable {}
extension OperationProfile: Codable {
    enum CodingKeys: String, CodingKey {
        case id, name, category, startTime, endTime
    }
}
extension MemoryProfileEntry: Codable {}
extension ProfilerOperationMetrics: Codable {}
extension ProfilingCategory: Codable {}
extension MemoryAllocationType: Codable {}
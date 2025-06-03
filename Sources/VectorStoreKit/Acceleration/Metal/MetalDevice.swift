// VectorStoreKit: Metal Device Management
//
// Centralized Metal device and resource management

import Foundation
@preconcurrency import Metal
@preconcurrency import MetalPerformanceShaders
import os.log

/// Manages Metal device initialization and capabilities detection
public actor MetalDevice {
    
    // MARK: - Properties
    
    /// The Metal device
    public let device: MTLDevice
    
    /// Command queue for GPU operations
    public let commandQueue: MTLCommandQueue
    
    /// Hardware capabilities
    public let capabilities: HardwareCapabilities
    
    /// Shared library for compute functions
    private let library: MTLLibrary
    
    /// Logger
    private let logger = Logger(subsystem: "VectorStoreKit", category: "MetalDevice")
    
    // MARK: - Initialization
    
    /// Initialize Metal device manager
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalDeviceError.noDeviceAvailable
        }
        
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalDeviceError.commandQueueCreationFailed
        }
        
        self.device = device
        self.commandQueue = commandQueue
        self.capabilities = HardwareCapabilities(device: device)
        
        // Create or load shader library
        if let defaultLibrary = device.makeDefaultLibrary() {
            self.library = defaultLibrary
        } else {
            self.library = try Self.createLibraryFromSource(device: device)
        }
        
        logger.info("Initialized Metal device: \(device.name)")
        logger.info("Capabilities: \(self.capabilities.description)")
    }
    
    // MARK: - Public Methods
    
    /// Create a new command buffer
    public func makeCommandBuffer() -> MTLCommandBuffer? {
        return commandQueue.makeCommandBuffer()
    }
    
    /// Get a function from the library
    public func makeFunction(name: String) -> MTLFunction? {
        return library.makeFunction(name: name)
    }
    
    /// Create a compute pipeline state
    public func makeComputePipelineState(function: MTLFunction) async throws -> MTLComputePipelineState {
        return try await device.makeComputePipelineState(function: function)
    }
    
    /// Check if device supports a specific feature
    public func supports(feature: MetalDeviceFeature) -> Bool {
        switch feature {
        case .float16:
            return capabilities.supportsFloat16
        case .bfloat16:
            return capabilities.supportsBfloat16
        case .raytracing:
            return capabilities.supportsRaytracing
        case .amx:
            return capabilities.hasAMX
        case .mps:
            return capabilities.hasMPS
        }
    }
    
    // MARK: - Private Methods
    
    private static func createLibraryFromSource(device: MTLDevice) throws -> MTLLibrary {
        // Try to compile shaders from .metal files
        let shaderSources = try loadShaderSources()
        
        do {
            return try device.makeLibrary(source: shaderSources, options: nil)
        } catch {
            throw MetalDeviceError.libraryCreationFailed(error.localizedDescription)
        }
    }
    
    private static func loadShaderSources() throws -> String {
        // Get the path to the Shaders directory
        let shadersPath = #filePath
            .replacingOccurrences(of: "MetalDevice.swift", with: "Shaders/")
        
        let shaderFiles = [
            "DistanceShaders.metal",
            "MatrixShaders.metal",
            "QuantizationShaders.metal"
        ]
        
        var combinedSource = ""
        
        // Try to load shader files
        for fileName in shaderFiles {
            let filePath = shadersPath + fileName
            if let source = try? String(contentsOfFile: filePath, encoding: .utf8) {
                combinedSource += source + "\n\n"
            }
        }
        
        // Fallback to basic embedded shaders if files not found
        if combinedSource.isEmpty {
            return """
            #include <metal_stdlib>
            using namespace metal;
            
            kernel void euclideanDistance(
                constant float* queryVector [[buffer(0)]],
                constant float* candidateVectors [[buffer(1)]],
                device float* distances [[buffer(2)]],
                constant uint& vectorDimension [[buffer(3)]],
                uint id [[thread_position_in_grid]]
            ) {
                float sum = 0.0;
                uint candidateOffset = id * vectorDimension;
                
                for (uint i = 0; i < vectorDimension; ++i) {
                    float diff = queryVector[i] - candidateVectors[candidateOffset + i];
                    sum += diff * diff;
                }
                
                distances[id] = sqrt(sum);
            }
            """
        }
        
        return combinedSource
    }
}

// MARK: - Supporting Types

/// Metal device errors
public enum MetalDeviceError: Error, LocalizedError {
    case noDeviceAvailable
    case commandQueueCreationFailed
    case libraryCreationFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .noDeviceAvailable:
            return "No Metal device available"
        case .commandQueueCreationFailed:
            return "Failed to create command queue"
        case .libraryCreationFailed(let reason):
            return "Failed to create shader library: \(reason)"
        }
    }
}

/// Metal device features that can be queried
public enum MetalDeviceFeature {
    case float16
    case bfloat16
    case raytracing
    case amx
    case mps
}

/// Hardware capabilities for Metal compute
public struct HardwareCapabilities: Sendable, CustomStringConvertible {
    public let deviceName: String
    public let hasAMX: Bool
    public let hasMPS: Bool
    public let supportsFloat16: Bool
    public let supportsBfloat16: Bool
    public let supportsRaytracing: Bool
    public let maxThreadsPerThreadgroup: Int
    public let maxThreadgroupMemoryLength: Int
    public let memoryBandwidth: Float // GB/s
    public let supportsParallelMetrics: Bool
    
    public init(device: MTLDevice) {
        self.deviceName = device.name
        self.hasAMX = device.name.contains("Apple") // Simplified detection
        self.hasMPS = true // Available on all modern devices
        self.supportsFloat16 = device.supportsFamily(.apple4)
        self.supportsBfloat16 = device.supports32BitMSAA
        self.supportsRaytracing = device.supportsRaytracing
        self.maxThreadsPerThreadgroup = device.maxThreadsPerThreadgroup.width
        self.maxThreadgroupMemoryLength = device.maxThreadgroupMemoryLength
        self.memoryBandwidth = Self.estimateMemoryBandwidth(for: device)
        self.supportsParallelMetrics = true
    }
    
    public func supportsMetric(_ metric: DistanceMetric) -> Bool {
        // All metrics are supported on Metal
        return true
    }
    
    public func supportsQuantization(_ scheme: QuantizationScheme) -> Bool {
        return true // Simplified
    }
    
    public var description: String {
        """
        Device: \(deviceName)
        Features: [float16: \(supportsFloat16), bfloat16: \(supportsBfloat16), \
        raytracing: \(supportsRaytracing), AMX: \(hasAMX)]
        Performance: \(memoryBandwidth) GB/s bandwidth, \
        max \(maxThreadsPerThreadgroup) threads/group
        """
    }
    
    private static func estimateMemoryBandwidth(for device: MTLDevice) -> Float {
        // Simplified bandwidth estimation based on device family
        if device.name.contains("Ultra") { return 800.0 }
        if device.name.contains("Max") { return 400.0 }
        if device.name.contains("Pro") { return 200.0 }
        return 100.0 // Base estimate
    }
}

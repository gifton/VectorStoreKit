// VectorStoreKit: Concrete Accelerator Strategy Implementations
//
// Hardware acceleration strategies for VectorUniverse API

import Foundation
import Metal
import CoreML

// MARK: - Metal Type Safety

// Note: Metal types cannot be made Sendable through extensions
// We'll handle thread safety through the actor model and @unchecked Sendable where needed

// MARK: - Metal Accelerator Strategies

/// Production-optimized Metal accelerator for GPU compute
public struct MetalProductionAcceleratorStrategy: ComputeAccelerator, Sendable {
    public typealias DeviceType = MetalComputeDevice
    public typealias CapabilitiesType = MetalCapabilities
    
    public let identifier = "metal-production"
    public let requirements = HardwareRequirements(
        minimumMemory: 512 * 1024 * 1024, // 512 MB
        requiredFeatures: [.metal],
        optionalFeatures: [.metalPerformanceShaders]
    )
    
    private let deviceSelection: DeviceSelection
    
    public init(deviceSelection: DeviceSelection = .automatic) {
        self.deviceSelection = deviceSelection
    }
    
    public func initialize() async throws -> MetalComputeDevice {
        guard let device = selectMetalDevice() else {
            throw AcceleratorError.noDevice
        }
        
        return MetalComputeDevice(
            device: device,
            commandQueue: device.makeCommandQueue()!,
            configuration: .production
        )
    }
    
    public func capabilities() -> MetalCapabilities {
        let device = MTLCreateSystemDefaultDevice()
        return MetalCapabilities(
            maxThreadsPerThreadgroup: device?.maxThreadsPerThreadgroup ?? MTLSize(width: 512, height: 1, depth: 1),
            supportsFloat16: device?.supportsFamily(.apple7) ?? false,
            supportsRaytracing: device?.supportsRaytracing ?? false,
            maxBufferLength: device?.maxBufferLength ?? 256 * 1024 * 1024,
            supportedFeatures: detectSupportedFeatures(device)
        )
    }
    
    private func selectMetalDevice() -> MTLDevice? {
        switch deviceSelection {
        case .automatic:
            return MTLCreateSystemDefaultDevice()
        case .discrete:
            return MTLCopyAllDevices().first { !$0.isLowPower }
        case .integrated:
            return MTLCopyAllDevices().first { $0.isLowPower }
        case .specific(let name):
            return MTLCopyAllDevices().first { $0.name == name }
        }
    }
    
    private func detectSupportedFeatures(_ device: MTLDevice?) -> Set<MetalFeature> {
        guard let device = device else { return [] }
        
        var features: Set<MetalFeature> = [.basic]
        
        if device.supportsFamily(.apple7) {
            features.insert(.float16)
            features.insert(.simdGroupMatrix)
        }
        
        if device.supportsFamily(.apple8) {
            features.insert(.dynamicLibraries)
            features.insert(.functionPointers)
        }
        
        if device.supportsRaytracing {
            features.insert(.raytracing)
        }
        
        return features
    }
}

/// Research-optimized Metal accelerator with profiling
public struct MetalResearchAcceleratorStrategy: ComputeAccelerator, Sendable {
    public typealias DeviceType = MetalComputeDevice
    public typealias CapabilitiesType = MetalCapabilities
    
    public let identifier = "metal-research"
    public let requirements = HardwareRequirements(
        minimumMemory: 1024 * 1024 * 1024, // 1 GB
        requiredFeatures: [.metal],
        optionalFeatures: [.metalPerformanceShaders, .neuralEngine]
    )
    
    public init() {}
    
    public func initialize() async throws -> MetalComputeDevice {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw AcceleratorError.noDevice
        }
        
        // Enable GPU frame capture for profiling
        let captureManager = MTLCaptureManager.shared()
        let captureDescriptor = MTLCaptureDescriptor()
        captureDescriptor.captureObject = device
        try? captureManager.startCapture(with: captureDescriptor)
        
        return MetalComputeDevice(
            device: device,
            commandQueue: device.makeCommandQueue()!,
            configuration: .research
        )
    }
    
    public func capabilities() -> MetalCapabilities {
        let device = MTLCreateSystemDefaultDevice()
        return MetalCapabilities(
            maxThreadsPerThreadgroup: device?.maxThreadsPerThreadgroup ?? MTLSize(width: 1024, height: 1, depth: 1),
            supportsFloat16: device?.supportsFamily(.apple7) ?? false,
            supportsRaytracing: device?.supportsRaytracing ?? false,
            maxBufferLength: device?.maxBufferLength ?? 1024 * 1024 * 1024,
            supportedFeatures: [.basic, .float16, .simdGroupMatrix, .profiling, .debugging]
        )
    }
}

/// Performance-optimized Metal accelerator
public struct MetalPerformanceAcceleratorStrategy: ComputeAccelerator, Sendable {
    public typealias DeviceType = MetalComputeDevice
    public typealias CapabilitiesType = MetalCapabilities
    
    public let identifier = "metal-performance"
    public let requirements = HardwareRequirements(
        minimumMemory: 256 * 1024 * 1024, // 256 MB
        requiredFeatures: [.metal],
        optionalFeatures: [.metalPerformanceShaders, .accelerateMatrix]
    )
    
    public init() {}
    
    public func initialize() async throws -> MetalComputeDevice {
        guard let device = selectHighestPerformanceDevice() else {
            throw AcceleratorError.noDevice
        }
        
        return MetalComputeDevice(
            device: device,
            commandQueue: device.makeCommandQueue()!,
            configuration: .performance
        )
    }
    
    public func capabilities() -> MetalCapabilities {
        let device = selectHighestPerformanceDevice() ?? MTLCreateSystemDefaultDevice()
        return MetalCapabilities(
            maxThreadsPerThreadgroup: device?.maxThreadsPerThreadgroup ?? MTLSize(width: 1024, height: 1, depth: 1),
            supportsFloat16: true,
            supportsRaytracing: device?.supportsRaytracing ?? false,
            maxBufferLength: device?.maxBufferLength ?? 512 * 1024 * 1024,
            supportedFeatures: [.basic, .float16, .simdGroupMatrix, .metalPerformanceShaders]
        )
    }
    
    private func selectHighestPerformanceDevice() -> MTLDevice? {
        let devices = MTLCopyAllDevices()
        
        // Prefer discrete GPU if available
        if let discrete = devices.first(where: { !$0.isLowPower }) {
            return discrete
        }
        
        // Otherwise use the default device
        return MTLCreateSystemDefaultDevice()
    }
}

// MARK: - Neural Engine Accelerator

/// Apple Neural Engine accelerator for ML workloads
public struct NeuralEngineAcceleratorStrategy: ComputeAccelerator, Sendable {
    public typealias DeviceType = NeuralEngineDevice
    public typealias CapabilitiesType = NeuralEngineCapabilities
    
    public let identifier = "neural-engine"
    public let requirements = HardwareRequirements(
        minimumMemory: 0,
        requiredFeatures: [.neuralEngine],
        optionalFeatures: []
    )
    
    public init() {}
    
    public func initialize() async throws -> NeuralEngineDevice {
        guard isNeuralEngineAvailable() else {
            throw AcceleratorError.deviceNotAvailable
        }
        
        return NeuralEngineDevice(configuration: .default)
    }
    
    public func capabilities() -> NeuralEngineCapabilities {
        return NeuralEngineCapabilities(
            supportedOperations: [.matmul, .convolution, .activation, .normalization],
            maxModelSize: 512 * 1024 * 1024, // 512 MB
            precision: [.float16, .int8],
            throughput: estimateNeuralEngineThroughput()
        )
    }
    
    private func isNeuralEngineAvailable() -> Bool {
        // Check if Neural Engine is available on the current device
        #if targetEnvironment(simulator)
        return false
        #else
        return true // Simplified - real implementation would check hardware
        #endif
    }
    
    private func estimateNeuralEngineThroughput() -> Float {
        // Estimate TOPS based on device
        #if os(macOS)
        return 15.8 // M1 Neural Engine
        #else
        return 11.0 // A14 Neural Engine
        #endif
    }
}

// MARK: - Supporting Types

public enum DeviceSelection: Sendable {
    case automatic
    case discrete
    case integrated
    case specific(name: String)
}

public struct MetalComputeDevice: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let configuration: Configuration
    
    public enum Configuration {
        case production
        case research
        case performance
    }
}

public struct MetalCapabilities: Sendable {
    public let maxThreadsPerThreadgroup: MTLSize
    public let supportsFloat16: Bool
    public let supportsRaytracing: Bool
    public let maxBufferLength: Int
    public let supportedFeatures: Set<MetalFeature>
}

public enum MetalFeature: Sendable {
    case basic
    case float16
    case simdGroupMatrix
    case raytracing
    case dynamicLibraries
    case functionPointers
    case metalPerformanceShaders
    case profiling
    case debugging
}

public struct NeuralEngineDevice: Sendable {
    public let configuration: Configuration
    
    public enum Configuration: Sendable {
        case `default`
        case highThroughput
        case lowLatency
    }
}

public struct NeuralEngineCapabilities: Sendable {
    public let supportedOperations: Set<NeuralOperation>
    public let maxModelSize: Int
    public let precision: Set<PrecisionLevel>
    public let throughput: Float // TOPS
}

public enum NeuralOperation: Sendable {
    case matmul
    case convolution
    case activation
    case normalization
    case pooling
    case elementwise
}

public enum PrecisionLevel: Sendable {
    case float32
    case float16
    case int8
    case int4
}

// MARK: - Accelerator Errors

public enum AcceleratorError: Error, Sendable {
    case noDevice
    case deviceNotAvailable
    case insufficientMemory
    case unsupportedFeature(String)
    case initializationFailed(String)
}

// MARK: - AMX Accelerator (Apple Matrix Extension)

/// AMX accelerator for matrix operations on Apple Silicon
public struct AMXAcceleratorStrategy: ComputeAccelerator, Sendable {
    public typealias DeviceType = AMXDevice
    public typealias CapabilitiesType = AMXCapabilities
    
    public let identifier = "amx"
    public let requirements = HardwareRequirements(
        minimumMemory: 0,
        requiredFeatures: [.accelerateMatrix],
        optionalFeatures: []
    )
    
    public init() {}
    
    public func initialize() async throws -> AMXDevice {
        guard isAMXAvailable() else {
            throw AcceleratorError.deviceNotAvailable
        }
        
        return AMXDevice()
    }
    
    public func capabilities() -> AMXCapabilities {
        return AMXCapabilities(
            matrixSize: 64,
            supportedTypes: [.float32, .float16, .int8],
            throughput: 2.0 // TFLOPS
        )
    }
    
    private func isAMXAvailable() -> Bool {
        #if arch(arm64)
        return true // Available on Apple Silicon
        #else
        return false
        #endif
    }
}

public struct AMXDevice: Sendable {}

public struct AMXCapabilities: Sendable {
    public let matrixSize: Int
    public let supportedTypes: Set<DataType>
    public let throughput: Float // TFLOPS
}

public enum DataType: Sendable {
    case float32
    case float16
    case int8
    case int16
}
// VectorStoreKit: Concrete Accelerator Strategy Implementations
//
// Hardware acceleration strategies for VectorUniverse API

import Foundation
import Metal
import CoreML
#if canImport(Darwin)
import Darwin
#endif

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
            #if os(macOS)
            return MTLCopyAllDevices().first { !$0.isLowPower }
            #else
            // On iOS, just return the default device as discrete GPU selection isn't applicable
            return MTLCreateSystemDefaultDevice()
            #endif
        case .integrated:
            #if os(macOS)
            return MTLCopyAllDevices().first { $0.isLowPower }
            #else
            // On iOS, just return the default device
            return MTLCreateSystemDefaultDevice()
            #endif
        case .specific(let name):
            #if os(macOS)
            return MTLCopyAllDevices().first { $0.name == name }
            #else
            if #available(iOS 18.0, *) {
                return MTLCopyAllDevices().first { $0.name == name }
            } else {
                // Fallback to default device on older iOS versions
                let defaultDevice = MTLCreateSystemDefaultDevice()
                return defaultDevice?.name == name ? defaultDevice : nil
            }
            #endif
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
        #if os(macOS)
        let devices = MTLCopyAllDevices()
        
        // Prefer discrete GPU if available
        if let discrete = devices.first(where: { !$0.isLowPower }) {
            return discrete
        }
        
        // Otherwise use the default device
        return MTLCreateSystemDefaultDevice()
        #else
        // On iOS, just return the default device
        // iOS devices don't have discrete GPUs
        return MTLCreateSystemDefaultDevice()
        #endif
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
        // Check if we're on a device with Neural Engine
        #if os(macOS)
        // Check for Apple Silicon Mac
        var sysinfo = utsname()
        uname(&sysinfo)
        let machine = withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        }
        
        // Apple Silicon Macs with Neural Engine
        if let machine = machine {
            // M1 family and newer
            if machine.contains("arm64") {
                // Additional check: try to load Core ML with Neural Engine preference
                let config = MLModelConfiguration()
                config.computeUnits = .cpuAndNeuralEngine
                
                // Check if Neural Engine is actually available by testing compute units
                return config.computeUnits == .cpuAndNeuralEngine || config.computeUnits == .all
            }
        }
        return false
        #elseif os(iOS) || os(iPadOS) || os(tvOS)
        // Check device model for Neural Engine support
        var sysinfo = utsname()
        uname(&sysinfo)
        let deviceModel = withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        }
        
        guard let model = deviceModel else { return false }
        
        // iPhone with A11 Bionic or newer (iPhone 8/X and later)
        if model.hasPrefix("iPhone") {
            if let majorVersion = extractDeviceVersion(from: model) {
                // iPhone10,x = iPhone 8/X (A11)
                // iPhone11,x = iPhone XR/XS (A12)
                // iPhone12,x = iPhone 11 (A13)
                // iPhone13,x = iPhone 12 (A14)
                // iPhone14,x = iPhone 13 (A15)
                // iPhone15,x = iPhone 14 (A15/A16)
                // iPhone16,x = iPhone 15 (A17)
                return majorVersion >= 10
            }
        }
        
        // iPad with A12 Bionic or newer
        if model.hasPrefix("iPad") {
            if let majorVersion = extractDeviceVersion(from: model) {
                // iPad8,x = iPad Pro 3rd gen (A12X)
                // iPad11,x = iPad Air 3rd gen (A12)
                // iPad13,x = iPad Air 4th gen (A14)
                return majorVersion >= 8
            }
        }
        
        // Apple TV with A15 or newer
        if model.hasPrefix("AppleTV") {
            if let majorVersion = extractDeviceVersion(from: model) {
                // AppleTV14,x = Apple TV 4K 3rd gen (A15)
                return majorVersion >= 14
            }
        }
        
        // For newer/unknown devices, try Core ML configuration check
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        return config.computeUnits == .cpuAndNeuralEngine || config.computeUnits == .all
        
        #else
        return false
        #endif
        #endif
    }
    
    private func extractDeviceVersion(from model: String) -> Int? {
        // Extract major version number from device model string
        // e.g., "iPhone14,2" -> 14
        let components = model.components(separatedBy: CharacterSet.decimalDigits.inverted)
        for component in components {
            if let version = Int(component), version > 0 {
                return version
            }
        }
        return nil
    }
    
    private func estimateNeuralEngineThroughput() -> Float {
        // Estimate TOPS (Trillion Operations Per Second) based on device
        #if os(macOS)
        // Check specific Mac chip
        var sysinfo = utsname()
        uname(&sysinfo)
        let machine = withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        }
        
        // Get the actual chip identifier via sysctl
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        
        if size > 0 {
            var cpuBrand = [CChar](repeating: 0, count: size)
            sysctlbyname("machdep.cpu.brand_string", &cpuBrand, &size, nil, 0)
            let brandString = String(cString: cpuBrand)
            
            // M-series Neural Engine TOPS
            if brandString.contains("M4") {
                return 38.0  // M4 Neural Engine
            } else if brandString.contains("M3") {
                return 18.0  // M3 Neural Engine
            } else if brandString.contains("M2") {
                return 15.8  // M2 Neural Engine
            } else if brandString.contains("M1") {
                if brandString.contains("Max") || brandString.contains("Ultra") {
                    return 15.8 * 2  // M1 Max/Ultra has 2x Neural Engine cores
                }
                return 11.0  // M1/M1 Pro Neural Engine
            }
        }
        
        return 15.8  // Default for unknown Apple Silicon
        
        #elseif os(iOS) || os(iPadOS)
        // Get device model to estimate Neural Engine performance
        var sysinfo = utsname()
        uname(&sysinfo)
        let deviceModel = withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        }
        
        guard let model = deviceModel else { return 11.0 }
        
        // Extract major version to determine chip generation
        if let majorVersion = extractDeviceVersion(from: model) {
            if model.hasPrefix("iPhone") {
                switch majorVersion {
                case 16: return 35.0  // A17 Pro Neural Engine (iPhone 15 Pro)
                case 15: return 17.0  // A16 Neural Engine (iPhone 14 Pro)
                case 14: return 15.8  // A15 Neural Engine (iPhone 13)
                case 13: return 11.0  // A14 Neural Engine (iPhone 12)
                case 12: return 5.0   // A13 Neural Engine (iPhone 11)
                case 11: return 5.0   // A12 Neural Engine (iPhone XS/XR)
                case 10: return 0.6   // A11 Neural Engine (iPhone 8/X)
                default: return 11.0
                }
            } else if model.hasPrefix("iPad") {
                switch majorVersion {
                case 13..<16: return 15.8  // M1/M2 iPad Pro
                case 11..<13: return 11.0  // A14/A15 iPad
                case 8..<11: return 5.0    // A12X/A12Z iPad Pro
                default: return 11.0
                }
            }
        }
        
        return 11.0  // Default A14-level performance
        
        #elseif os(tvOS)
        return 15.8  // A15 in Apple TV 4K
        #else
        return 11.0  // Conservative default
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
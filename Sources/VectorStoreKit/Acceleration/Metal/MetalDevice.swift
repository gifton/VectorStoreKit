// VectorStoreKit: Metal Device Wrapper
//
// Simple wrapper for MTLDevice to provide consistent API

import Foundation
@preconcurrency import Metal

/// Wrapper for MTLDevice to provide consistent API
public struct MetalDevice: Sendable {
    public let device: MTLDevice
    
    public init(device: MTLDevice) {
        self.device = device
    }
    
    public static var `default`: MetalDevice? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        return MetalDevice(device: device)
    }
}
import Testing
import Foundation
@testable import VectorStoreKit

@Suite("Neural Engine Detection Tests")
struct NeuralEngineDetectionTests {
    
    @Test("Neural Engine availability detection")
    func testNeuralEngineAvailability() async throws {
        let strategy = NeuralEngineAcceleratorStrategy()
        
        // Test capabilities can be retrieved
        let capabilities = strategy.capabilities()
        
        #expect(capabilities.maxModelSize > 0)
        #expect(!capabilities.supportedOperations.isEmpty)
        #expect(!capabilities.precision.isEmpty)
        #expect(capabilities.throughput > 0)
        
        // Log detection results for debugging
        print("Neural Engine Detection Results:")
        print("- Supported Operations: \(capabilities.supportedOperations)")
        print("- Max Model Size: \(capabilities.maxModelSize / 1024 / 1024) MB")
        print("- Precision Levels: \(capabilities.precision)")
        print("- Estimated Throughput: \(capabilities.throughput) TOPS")
        
        // Try to initialize if available
        do {
            let device = try await strategy.initialize()
            print("- Neural Engine Available: YES")
            print("- Configuration: \(device.configuration)")
        } catch {
            print("- Neural Engine Available: NO")
            print("- Reason: \(error)")
            
            // On simulator or Intel Mac, this is expected
            #if targetEnvironment(simulator) || (os(macOS) && arch(x86_64))
            // Expected to fail
            #else
            // On real device with Neural Engine, this should succeed
            // But we don't want to fail the test on devices without NE
            print("Note: This device may not have a Neural Engine")
            #endif
        }
    }
    
    @Test("Neural Engine throughput estimation")
    func testThroughputEstimation() {
        let strategy = NeuralEngineAcceleratorStrategy()
        let capabilities = strategy.capabilities()
        
        // Verify throughput is reasonable
        #expect(capabilities.throughput >= 0.6) // A11 minimum
        #expect(capabilities.throughput <= 50.0) // Reasonable maximum
        
        #if os(macOS)
        // On Mac, should be at least M1 level
        if ProcessInfo.processInfo.machineHardwareName?.contains("arm64") == true {
            #expect(capabilities.throughput >= 11.0)
        }
        #elseif os(iOS) || os(iPadOS)
        // On modern iOS devices, should be at least A11 level
        #expect(capabilities.throughput >= 0.6)
        #endif
    }
}

// Helper extension to get hardware name
extension ProcessInfo {
    var machineHardwareName: String? {
        var sysinfo = utsname()
        let result = uname(&sysinfo)
        guard result == 0 else { return nil }
        
        return withUnsafePointer(to: &sysinfo.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                String(validatingUTF8: $0)
            }
        }
    }
}
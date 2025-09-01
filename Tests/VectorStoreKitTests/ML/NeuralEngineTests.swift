import XCTest
import Foundation
import VectorStoreKit

struct TestNeuralEngine {
    func testMain() async throws {
        print("Testing Neural Engine Detection...")
        print("=" * 50)
        
        let strategy = NeuralEngineAcceleratorStrategy()
        let capabilities = strategy.capabilities()
        
        print("\nNeural Engine Capabilities:")
        print("- Supported Operations: \(capabilities.supportedOperations.map { "\($0)" }.joined(separator: ", "))")
        print("- Max Model Size: \(capabilities.maxModelSize / 1024 / 1024) MB")
        print("- Precision Levels: \(capabilities.precision.map { "\($0)" }.joined(separator: ", "))")
        print("- Estimated Throughput: \(capabilities.throughput) TOPS")
        
        print("\nChecking Neural Engine availability...")
        
        do {
            let device = try await strategy.initialize()
            print("✅ Neural Engine is AVAILABLE")
            print("- Configuration: \(device.configuration)")
            
            // Get system info
            var sysinfo = utsname()
            uname(&sysinfo)
            let machine = withUnsafePointer(to: &sysinfo.machine) {
                $0.withMemoryRebound(to: CChar.self, capacity: 1) {
                    String(validatingUTF8: $0)
                }
            }
            print("- Device: \(machine ?? "Unknown")")
            
            #if os(macOS)
            // Try to get CPU brand string
            var size = 0
            sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
            
            if size > 0 {
                var cpuBrand = [CChar](repeating: 0, count: size)
                sysctlbyname("machdep.cpu.brand_string", &cpuBrand, &size, nil, 0)
                let brandString = String(cString: cpuBrand)
                print("- CPU: \(brandString)")
            }
            #endif
            
        } catch {
            print("❌ Neural Engine is NOT AVAILABLE")
            print("- Error: \(error)")
            
            #if targetEnvironment(simulator)
            print("- Note: Running on simulator - Neural Engine not available")
            #elseif os(macOS) && arch(x86_64)
            print("- Note: Running on Intel Mac - Neural Engine not available")
            #else
            print("- Note: This device may not have Neural Engine support")
            #endif
        }
        
        print("\n" + "=" * 50)
    }
}

// Helper to repeat string
extension String {
    static func *(lhs: String, rhs: Int) -> String {
        String(repeating: lhs, count: rhs)
    }
}
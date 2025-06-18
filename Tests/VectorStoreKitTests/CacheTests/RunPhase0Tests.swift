// Temporary test runner for Phase 0 validation
// Run with: swift test --filter Phase0ValidationTests

import XCTest

@main
struct RunPhase0Tests {
    static func main() {
        // Run only Phase 0 validation tests
        XCTMain([
            testCase([
                ("testStructuredConcurrency", Phase0ValidationTests.testStructuredConcurrency),
                ("testFIFOBatchCapacityEnforcement", Phase0ValidationTests.testFIFOBatchCapacityEnforcement),
                ("testLRUCapacityEnforcement", Phase0ValidationTests.testLRUCapacityEnforcement),
                ("testLFUCapacityEnforcement", Phase0ValidationTests.testLFUCapacityEnforcement),
                ("testLRUNoRetainCycles", Phase0ValidationTests.testLRUNoRetainCycles),
                ("testLFUMinFrequencyBug", Phase0ValidationTests.testLFUMinFrequencyBug),
                ("testConcurrentSafetyUnderLoad", Phase0ValidationTests.testConcurrentSafetyUnderLoad),
            ])
        ])
    }
}
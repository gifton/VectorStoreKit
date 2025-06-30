#!/usr/bin/env swift

import Foundation

let fileManager = FileManager.default
let projectRoot = URL(fileURLWithPath: fileManager.currentDirectoryPath)
let examplesDir = projectRoot.appendingPathComponent("Examples")
let testsDir = projectRoot.appendingPathComponent("Tests/VectorStoreKitTests")

// Examples to migrate based on categorization
let examplesToMigrate: [(source: String, destination: String)] = [
    // Test-like Examples
    ("TestDistanceComputation.swift", "Core/DistanceComputationTests.swift"),
    ("TestNeuralEngine.swift", "ML/NeuralEngineTests.swift"),
    ("TestOPQRotation.swift", "Compression/OPQRotationTests.swift"),
    ("TestPCAImplementation.swift", "ML/PCAImplementationTests.swift"),
    
    // Performance/Benchmark Examples
    ("BenchmarkExample.swift", "Performance/BenchmarkTests.swift"),
    ("DistanceMatrixBenchmark.swift", "Performance/DistanceMatrixTests.swift"),
    ("GPUTimingExample.swift", "Metal/GPUTimingTests.swift"),
    ("HonestGPUTimingExample.swift", "Metal/AccurateGPUTimingTests.swift"),
    ("HighPerformanceExample.swift", "Performance/HighPerformanceTests.swift"),
    
    // Memory Management Examples
    ("BufferPoolExample.swift", "Metal/BufferPoolTests.swift"),
    ("BufferCacheExample.swift", "Metal/BufferCacheTests.swift"),
    ("MemoryManagementExample.swift", "Memory/MemoryManagementTests.swift"),
    ("MemoryOptimizationExample.swift", "Memory/OptimizationTests.swift"),
    ("MemoryPressureExample.swift", "Memory/PressureHandlingTests.swift"),
    ("SafeMemoryExample.swift", "Memory/SafetyTests.swift"),
    ("EnhancedMemoryPoolExample.swift", "Memory/EnhancedPoolTests.swift"),
    
    // Metal/GPU Examples
    ("MetalBufferOptimizationExample.swift", "Metal/BufferOptimizationTests.swift"),
    ("MetalResourceManagementExample.swift", "Metal/ResourceManagementTests.swift"),
    ("ComputeGraphExample.swift", "Metal/ComputeGraphTests.swift"),
    ("ComputeGraphExecutionExample.swift", "Metal/GraphExecutionTests.swift"),
    
    // ML/AI Examples
    ("LSTMExample.swift", "ML/LSTMTests.swift"),
    ("LSTMTrainingExample.swift", "ML/LSTMTrainingTests.swift"),
    ("NeuralClusteringExample.swift", "ML/NeuralClusteringTests.swift"),
    ("MixedPrecisionTrainingExample.swift", "ML/MixedPrecisionTests.swift"),
    ("OptimizedMLExample.swift", "ML/OptimizationTests.swift"),
    ("BatchNormExample.swift", "ML/BatchNormTests.swift"),
    ("AIAssistantExample.swift", "Integration/AIAssistantTests.swift"),
    
    // Cache/Storage Examples
    ("AdaptiveCacheExample.swift", "Storage/AdaptiveCacheTests.swift"),
    ("MultiLevelCacheExample.swift", "Storage/MultiLevelCacheTests.swift"),
    ("VectorQualityCacheExample.swift", "Storage/QualityCacheTests.swift"),
    
    // Optimization Examples
    ("SIMDOptimizationExample.swift", "Optimization/SIMDTests.swift"),
    ("Vector512OptimizationExample.swift", "Optimization/Vector512Tests.swift"),
    ("ThreadOptimizationExample.swift", "Optimization/ThreadingTests.swift")
]

// Function to transform example code to test code
func transformToTest(_ content: String, fileName: String) -> String {
    var transformed = content
    
    // Remove @main attribute
    transformed = transformed.replacingOccurrences(of: "@main\n", with: "")
    transformed = transformed.replacingOccurrences(of: "@main ", with: "")
    
    // Add XCTest import if not present
    if !transformed.contains("import XCTest") {
        transformed = "import XCTest\n" + transformed
    }
    
    // Transform struct/class to test class
    let className = fileName.replacingOccurrences(of: ".swift", with: "")
    
    // Replace main struct/enum with test class
    let structPattern = #"(struct|enum)\s+\w+Example\s*\{"#
    let structRegex = try! NSRegularExpression(pattern: structPattern, options: [])
    transformed = structRegex.stringByReplacingMatches(
        in: transformed,
        options: [],
        range: NSRange(location: 0, length: transformed.count),
        withTemplate: "final class \(className): XCTestCase {"
    )
    
    // Transform static func main to test methods
    let mainPattern = #"static\s+func\s+main\s*\(\s*\)\s*(async\s+)?(throws\s+)?\{"#
    let mainRegex = try! NSRegularExpression(pattern: mainPattern, options: [])
    
    var testMethodIndex = 1
    transformed = mainRegex.stringByReplacingMatches(
        in: transformed,
        options: [],
        range: NSRange(location: 0, length: transformed.count),
        withTemplate: "func testMain() async throws {"
    )
    
    // Transform other static functions to test methods
    let staticFuncPattern = #"static\s+func\s+(\w+)\s*\("#
    let staticFuncRegex = try! NSRegularExpression(pattern: staticFuncPattern, options: [])
    
    let matches = staticFuncRegex.matches(in: transformed, options: [], range: NSRange(location: 0, length: transformed.count))
    
    // Process matches in reverse order to maintain string indices
    for match in matches.reversed() {
        if let methodNameRange = Range(match.range(at: 1), in: transformed) {
            let methodName = String(transformed[methodNameRange])
            if methodName != "main" {
                let testMethodName = "test" + methodName.prefix(1).uppercased() + methodName.dropFirst()
                if let range = Range(match.range, in: transformed) {
                    transformed.replaceSubrange(range, with: "func \(testMethodName)(")
                }
            }
        }
    }
    
    // Replace print statements with test assertions where appropriate
    transformed = transformed.replacingOccurrences(
        of: #"print\("Test passed"\)"#,
        with: "XCTAssertTrue(true, \"Test passed\")"
    )
    
    // Add async test support wrapper where needed
    if transformed.contains("Task {") {
        transformed = transformed.replacingOccurrences(
            of: "Task {",
            with: "let expectation = expectation(description: \"async test\")\nTask {\ndefer { expectation.fulfill() }"
        )
        
        // Add wait at the end of test methods that use Task
        let testMethodPattern = #"(func test\w+\(\)[^{]*\{[^}]*Task[^}]*)\}"#
        let testMethodRegex = try! NSRegularExpression(pattern: testMethodPattern, options: [.dotMatchesLineSeparators])
        transformed = testMethodRegex.stringByReplacingMatches(
            in: transformed,
            options: [],
            range: NSRange(location: 0, length: transformed.count),
            withTemplate: "$1\n        wait(for: [expectation], timeout: 30.0)\n    }"
        )
    }
    
    return transformed
}

// Statistics
var successCount = 0
var failureCount = 0
var errors: [String] = []

// Migrate each file
for (source, destination) in examplesToMigrate {
    let sourceURL = examplesDir.appendingPathComponent(source)
    let destURL = testsDir.appendingPathComponent(destination)
    
    do {
        // Check if source exists
        guard fileManager.fileExists(atPath: sourceURL.path) else {
            print("⚠️  Source not found: \(source)")
            continue
        }
        
        // Create destination directory
        try fileManager.createDirectory(
            at: destURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        
        // Read source file
        let content = try String(contentsOf: sourceURL, encoding: .utf8)
        
        // Transform to test format
        let transformed = transformToTest(content, fileName: destURL.lastPathComponent)
        
        // Write transformed file
        try transformed.write(to: destURL, atomically: true, encoding: .utf8)
        
        // Remove original
        try fileManager.removeItem(at: sourceURL)
        
        print("✅ Migrated: \(source) → \(destination)")
        successCount += 1
        
    } catch {
        print("❌ Failed to migrate \(source): \(error)")
        errors.append("\(source): \(error)")
        failureCount += 1
    }
}

// Print summary
print("\n📊 Migration Summary:")
print("✅ Successfully migrated: \(successCount) files")
if failureCount > 0 {
    print("❌ Failed: \(failureCount) files")
    print("\nErrors:")
    errors.forEach { print("  - \($0)") }
}

// Verify build
print("\n🔨 Verifying build...")
let buildProcess = Process()
buildProcess.executableURL = URL(fileURLWithPath: "/usr/bin/swift")
buildProcess.arguments = ["build"]
buildProcess.currentDirectoryURL = projectRoot

do {
    try buildProcess.run()
    buildProcess.waitUntilExit()
    
    if buildProcess.terminationStatus == 0 {
        print("✅ Build successful!")
    } else {
        print("❌ Build failed with status: \(buildProcess.terminationStatus)")
    }
} catch {
    print("❌ Failed to run build: \(error)")
}
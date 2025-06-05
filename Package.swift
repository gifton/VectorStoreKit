// swift-tools-version: 5.10
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "VectorStoreKit",
    platforms: [
        .macOS(.v14),      // macOS 14.0+ 
        .iOS(.v17),        // iOS 17.0+
        .tvOS(.v17),       // tvOS 17.0+
        .watchOS(.v10),    // watchOS 10.0+
        .visionOS(.v1)     // visionOS 1.0+
    ],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "VectorStoreKit",
            targets: ["VectorStoreKit"]),
        .executable(
            name: "verify-memory",
            targets: ["VerifyMemory"]),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "VectorStoreKit",
            exclude: [
                "Acceleration/Metal/Shaders/DistanceShaders.metal",
                "Acceleration/Metal/Shaders/MatrixShaders.metal",
                "Acceleration/Metal/Shaders/QuantizationShaders.metal",
                "Acceleration/Metal/Shaders/ClusteringShaders.metal",
                "Benchmarks/BenchmarkRunner.swift",
                "Benchmarks/ComprehensiveBenchmarks.swift",
                "Benchmarks/PerformanceBenchmarks.swift",
                "Encoding/NeuralEncoder.swift"
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("BareSlashRegexLiterals"),
                .enableUpcomingFeature("ConciseMagicFile"),
                .enableUpcomingFeature("ForwardTrailingClosures"),
                .enableUpcomingFeature("ImplicitOpenExistentials"),
                .enableUpcomingFeature("DisableOutwardActorInference")
            ]
        ),
        .executableTarget(
            name: "VerifyMemory",
            dependencies: ["VectorStoreKit"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "TestNeuralEngine",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            sources: ["TestNeuralEngine.swift"]
        ),
        .executableTarget(
            name: "TestDistanceComputation",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            sources: ["TestDistanceComputation.swift"]
        ),
        .testTarget(
            name: "VectorStoreKitTests",
            dependencies: ["VectorStoreKit"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)
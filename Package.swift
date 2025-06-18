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
        .executable(
            name: "verify-components",
            targets: ["VerifyComponents"]),
        .executable(
            name: "performance-profile",
            targets: ["PerformanceProfile"]),
        .executable(
            name: "high-performance-example",
            targets: ["HighPerformanceExample"]),
        .executable(
            name: "adaptive-cache-example",
            targets: ["AdaptiveCacheExample"]),
        .executable(
            name: "vsk-benchmark",
            targets: ["VectorStoreKitBenchmark"]),
        .executable(
            name: "geospatial-example",
            targets: ["GeoSpatialExample"]),
        .executable(
            name: "financial-analysis-example",
            targets: ["FinancialAnalysisExample"]),
        .executable(
            name: "buffer-pool-example",
            targets: ["BufferPoolExample"]),
        .executable(
            name: "vectorstore",
            targets: ["VectorStoreCLI"]),
        .executable(
            name: "thread-optimization-example",
            targets: ["ThreadOptimizationExample"]),
        .executable(
            name: "vector512-optimization-example",
            targets: ["Vector512OptimizationExample"]),
        .executable(
            name: "VectorStoreValidation",
            targets: ["VectorStoreValidation"]),
        .executable(
            name: "distance-matrix-benchmark",
            targets: ["DistanceMatrixBenchmark"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "VectorStoreKit",
            path: "Sources/VectorStoreKit",
            exclude: [
                "Benchmarks/BenchmarkRunner.swift",
                "Benchmarks/ComprehensiveBenchmarks.swift",
                "Benchmarks/PerformanceBenchmarks.swift",
                "Encoding/NeuralEncoder.swift"
            ],
            resources: [
                .copy("Acceleration/Metal/Shaders/ClusteringShaders.metal"),
                .copy("Acceleration/Metal/Shaders/DistanceShaders.metal"),
                .copy("Acceleration/Metal/Shaders/DistanceMatrixShaders.metal"),
                .copy("Acceleration/Metal/Shaders/MatrixShaders.metal"),
                .copy("Acceleration/Metal/Shaders/QuantizationShaders.metal"),
                .copy("Acceleration/Metal/Shaders/WarpOptimizedShaders.metal"),
                .copy("Core/ML/Shaders/Activations.metal"),
                .copy("Core/ML/Shaders/ElementwiseOperations.metal"),
                .copy("Core/ML/Shaders/FusedOperations.metal"),
                .copy("Core/ML/Shaders/LossOperations.metal"),
                .copy("Core/ML/Shaders/MatrixOperations.metal"),
                .copy("Core/ML/Shaders/MixedPrecisionShaders.metal"),
                .copy("Core/ML/Shaders/NormalizationShaders.metal"),
                .copy("Core/ML/Shaders/OptimizationShaders.metal"),
                .copy("Core/ML/Shaders/OptimizedMLShaders.metal"),
                .copy("Core/ML/Shaders/OptimizedElementwiseOperations.metal"),
                .copy("Core/ML/Shaders/OptimizedLossOperations.metal"),
                .copy("Core/ML/Shaders/PCAShaders.metal")
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
            name: "VerifyComponents",
            dependencies: ["VectorStoreKit"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "PerformanceProfile",
            dependencies: ["VectorStoreKit"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "TestNeuralEngine",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            exclude: ["BenchmarkExample.swift", "NeuralClusteringExample.swift", "TestDistanceComputation.swift", "HighPerformanceExample.swift", "AdaptiveCacheExample.swift", "GeoSpatialExample.swift", "FinancialAnalysisExample.swift", "BufferPoolExample.swift"],
            sources: ["TestNeuralEngine.swift"]
        ),
        .executableTarget(
            name: "TestDistanceComputation",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            exclude: ["BenchmarkExample.swift", "NeuralClusteringExample.swift", "TestNeuralEngine.swift", "HighPerformanceExample.swift", "AdaptiveCacheExample.swift", "GeoSpatialExample.swift", "FinancialAnalysisExample.swift", "BufferPoolExample.swift"],
            sources: ["TestDistanceComputation.swift"]
        ),
        .executableTarget(
            name: "BenchmarkExample",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            exclude: ["TestNeuralEngine.swift", "NeuralClusteringExample.swift", "TestDistanceComputation.swift", "HighPerformanceExample.swift", "AdaptiveCacheExample.swift", "GeoSpatialExample.swift", "FinancialAnalysisExample.swift", "BufferPoolExample.swift"],
            sources: ["BenchmarkExample.swift"]
        ),
        .executableTarget(
            name: "NeuralClusteringExample",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            exclude: ["BenchmarkExample.swift", "TestNeuralEngine.swift", "TestDistanceComputation.swift", "HighPerformanceExample.swift", "AdaptiveCacheExample.swift", "GeoSpatialExample.swift", "FinancialAnalysisExample.swift", "BufferPoolExample.swift"],
            sources: ["NeuralClusteringExample.swift"]
        ),
        .executableTarget(
            name: "HighPerformanceExample",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            exclude: ["BenchmarkExample.swift", "TestNeuralEngine.swift", "TestDistanceComputation.swift", "NeuralClusteringExample.swift", "AdaptiveCacheExample.swift", "GeoSpatialExample.swift", "FinancialAnalysisExample.swift", "BufferPoolExample.swift"],
            sources: ["HighPerformanceExample.swift"]
        ),
        .executableTarget(
            name: "AdaptiveCacheExample",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            exclude: ["BenchmarkExample.swift", "TestNeuralEngine.swift", "TestDistanceComputation.swift", "NeuralClusteringExample.swift", "HighPerformanceExample.swift", "GeoSpatialExample.swift", "FinancialAnalysisExample.swift", "BufferPoolExample.swift"],
            sources: ["AdaptiveCacheExample.swift"]
        ),
        .executableTarget(
            name: "GeoSpatialExample",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            exclude: ["BenchmarkExample.swift", "TestNeuralEngine.swift", "TestDistanceComputation.swift", "NeuralClusteringExample.swift", "HighPerformanceExample.swift", "AdaptiveCacheExample.swift", "FinancialAnalysisExample.swift", "BufferPoolExample.swift"],
            sources: ["GeoSpatialExample.swift"]
        ),
        .executableTarget(
            name: "FinancialAnalysisExample",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            exclude: ["BenchmarkExample.swift", "TestNeuralEngine.swift", "TestDistanceComputation.swift", "NeuralClusteringExample.swift", "HighPerformanceExample.swift", "AdaptiveCacheExample.swift", "GeoSpatialExample.swift", "BufferPoolExample.swift"],
            sources: ["FinancialAnalysisExample.swift"]
        ),
        .executableTarget(
            name: "BufferPoolExample",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            exclude: ["BenchmarkExample.swift", "TestNeuralEngine.swift", "TestDistanceComputation.swift", "NeuralClusteringExample.swift", "HighPerformanceExample.swift", "AdaptiveCacheExample.swift", "GeoSpatialExample.swift", "FinancialAnalysisExample.swift"],
            sources: ["BufferPoolExample.swift"]
        ),
        .testTarget(
            name: "VectorStoreKitTests",
            dependencies: ["VectorStoreKit"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "VectorStoreKitBenchmark",
            dependencies: [
                "VectorStoreKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "VectorStoreCLI",
            dependencies: [
                "VectorStoreKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "ThreadOptimizationExample",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            exclude: ["BenchmarkExample.swift", "TestNeuralEngine.swift", "TestDistanceComputation.swift", "NeuralClusteringExample.swift", "HighPerformanceExample.swift", "AdaptiveCacheExample.swift", "GeoSpatialExample.swift", "FinancialAnalysisExample.swift", "BufferPoolExample.swift", "Vector512OptimizationExample.swift"],
            sources: ["ThreadOptimizationExample.swift"]
        ),
        .executableTarget(
            name: "Vector512OptimizationExample",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            exclude: ["BenchmarkExample.swift", "TestNeuralEngine.swift", "TestDistanceComputation.swift", "NeuralClusteringExample.swift", "HighPerformanceExample.swift", "AdaptiveCacheExample.swift", "GeoSpatialExample.swift", "FinancialAnalysisExample.swift", "BufferPoolExample.swift", "ThreadOptimizationExample.swift"],
            sources: ["Vector512OptimizationExample.swift"]
        ),
        .executableTarget(
            name: "VectorStoreValidation",
            dependencies: [
                "VectorStoreKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "DistanceMatrixBenchmark",
            dependencies: ["VectorStoreKit"],
            path: "Examples",
            sources: ["DistanceMatrixBenchmark.swift"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)
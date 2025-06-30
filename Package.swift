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
        // Main library
        .library(
            name: "VectorStoreKit",
            targets: ["VectorStoreKit"]),
        
        // Verification tools
        .executable(
            name: "verify-memory",
            targets: ["VerifyMemory"]),
        .executable(
            name: "verify-components",
            targets: ["VerifyComponents"]),
        
        // Performance tools
        .executable(
            name: "performance-profile",
            targets: ["PerformanceProfile"]),
        
        // CLI tool
        .executable(
            name: "vectorstore",
            targets: ["VectorStoreCLI"]),
        
        // Validation tool
        .executable(
            name: "VectorStoreValidation",
            targets: ["VectorStoreValidation"]),
        
        // Remaining examples (10 focused examples) - COMMENTED FOR BUILD SPEED
        // .executable(
        //     name: "semantic-search-example",
        //     targets: ["SemanticSearchExample"]),
        // .executable(
        //     name: "performance-optimization-example",
        //     targets: ["PerformanceOptimizationExample"]),
        // .executable(
        //     name: "ml-integration-example",
        //     targets: ["MLIntegrationExample"]),
        // .executable(
        //     name: "production-deployment-example",
        //     targets: ["ProductionDeploymentExample"]),
        // .executable(
        //     name: "batch-indexing-example",
        //     targets: ["BatchIndexingExample"]),
        // .executable(
        //     name: "advanced-distance-metrics-example",
        //     targets: ["AdvancedDistanceMetricsExample"]),
        // .executable(
        //     name: "migration-example",
        //     targets: ["MigrationExample"]),
        // .executable(
        //     name: "multi-modal-search-example",
        //     targets: ["MultiModalSearchExample"]),
        // .executable(
        //     name: "recommendation-system-example",
        //     targets: ["RecommendationSystemExample"]),
        // .executable(
        //     name: "monitoring-example",
        //     targets: ["MonitoringExample"]),
        // .executable(
        //     name: "rag-example",
        //     targets: ["RAGExample"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        // Main library target
        .target(
            name: "VectorStoreKit",
            path: "Sources/VectorStoreKit",
            exclude: [
                "Encoding/NeuralEncoder.swift",
                // Exclude README files
                "Acceleration/Metal/GPU_TIMING_README.md",
                "Core/ML/ML_FIX_IMPLEMENTATION_PLAN.md",
                "Core/ValidationREADME.md",
                "Core/BEST_PRACTICES.md",
                "Acceleration/Metal/Shaders/README.md",
                "Acceleration/Metal/BufferCache_README.md"
            ],
            // TEMPORARILY COMMENTED METAL SHADERS FOR BUILD SPEED DEBUGGING
            resources: [
                // .copy("Acceleration/Metal/Shaders/ClusteringShaders.metal"),
                // .copy("Acceleration/Metal/Shaders/DistanceShaders.metal"),
                // .copy("Acceleration/Metal/Shaders/DistanceMatrixShaders.metal"),
                // .copy("Acceleration/Metal/Shaders/MatrixShaders.metal"),
                // .copy("Acceleration/Metal/Shaders/QuantizationShaders.metal"),
                // .copy("Acceleration/Metal/Shaders/WarpOptimizedShaders.metal"),
                // .copy("Core/ML/Shaders/Activations.metal"),
                // .copy("Core/ML/Shaders/ElementwiseOperations.metal"),
                // .copy("Core/ML/Shaders/FusedOperations.metal"),
                // .copy("Core/ML/Shaders/LossOperations.metal"),
                // .copy("Core/ML/Shaders/MatrixOperations.metal"),
                // .copy("Core/ML/Shaders/MixedPrecisionShaders.metal"),
                // .copy("Core/ML/Shaders/NormalizationShaders.metal"),
                // .copy("Core/ML/Shaders/OptimizationShaders.metal"),
                // .copy("Core/ML/Shaders/OptimizedMLShaders.metal"),
                // .copy("Core/ML/Shaders/OptimizedElementwiseOperations.metal"),
                // .copy("Core/ML/Shaders/OptimizedLossOperations.metal"),
                // .copy("Core/ML/Shaders/PCAShaders.metal")
            ],
            swiftSettings: [
                // TEMPORARILY DISABLED FOR BUILD SPEED
                // .enableExperimentalFeature("StrictConcurrency"),
                .enableUpcomingFeature("BareSlashRegexLiterals"),
                .enableUpcomingFeature("ConciseMagicFile"),
                .enableUpcomingFeature("ForwardTrailingClosures"),
                .enableUpcomingFeature("ImplicitOpenExistentials"),
                .enableUpcomingFeature("DisableOutwardActorInference")
            ]
        ),
        
        // Verification targets
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
        
        // Performance targets
        .executableTarget(
            name: "PerformanceProfile",
            dependencies: ["VectorStoreKit"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        
        // CLI and validation
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
            name: "VectorStoreValidation",
            dependencies: [
                "VectorStoreKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        
        // Example targets (10 focused examples) - COMMENTED FOR BUILD SPEED
        // .executableTarget(
        //     name: "SemanticSearchExample",
        //     dependencies: ["VectorStoreKit"],
        //     path: "Examples",
        //     sources: ["SemanticSearchExample.swift"],
        //     swiftSettings: [
        //         .enableExperimentalFeature("StrictConcurrency")
        //     ]
        // ),
        // COMMENTED OUT ALL EXAMPLE TARGETS FOR BUILD SPEED
        // .executableTarget(
        //     name: "PerformanceOptimizationExample",
        //     dependencies: ["VectorStoreKit"],
        //     path: "Examples",
        //     sources: ["PerformanceOptimizationExample.swift"],
        //     swiftSettings: [
        //         .enableExperimentalFeature("StrictConcurrency")
        //     ]
        // ),
        // Additional example targets commented out...
        
        // Test target
        .testTarget(
            name: "VectorStoreKitTests",
            dependencies: ["VectorStoreKit"],
            swiftSettings: [
                // TEMPORARILY DISABLED FOR BUILD SPEED
                // .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)
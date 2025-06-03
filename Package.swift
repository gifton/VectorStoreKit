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
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "VectorStoreKit",
            exclude: [
                "Acceleration/Metal/Shaders/DistanceShaders.metal",
                "Acceleration/Metal/Shaders/MatrixShaders.metal",
                "Acceleration/Metal/Shaders/QuantizationShaders.metal"
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
        .testTarget(
            name: "VectorStoreKitTests",
            dependencies: ["VectorStoreKit"],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)
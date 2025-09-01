// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "MultiModalSearchExample",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "../..")
    ],
    targets: [
        .executableTarget(
            name: "MultiModalSearchExample",
            dependencies: [
                .product(name: "VectorStoreKit", package: "VectorStoreKit")
            ],
            path: ".",
            sources: ["MultiModalSearchExample.swift"]
        )
    ]
)
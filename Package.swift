// swift-tools-version:6.0
import PackageDescription

// Root manifest so this repository (published as `proteinmpnn-mlx`) is consumable
// as a git-URL SwiftPM dependency — SwiftPM requires Package.swift at the repo
// root. The package sources live under MPNNKit/ (which keeps its own standalone
// manifest, MPNNKit/Package.swift, for local development); the targets below
// point at that subdirectory via `path:`. Keep this manifest in sync with
// MPNNKit/Package.swift.
let package = Package(
    name: "MPNNKit",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "MPNNKit", targets: ["MPNNKit"]),
    ],
    dependencies: [
        // Transitive heavy dependency: MLX (Metal kernels + Cmlx). Source-built + SPM-cached.
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
    ],
    targets: [
        .target(
            name: "MPNNKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "MPNNKit/Sources/MPNNKit"
        ),
        .testTarget(
            name: "MPNNKitTests",
            dependencies: ["MPNNKit", .product(name: "MLX", package: "mlx-swift")],
            path: "MPNNKit/Tests/MPNNKitTests"
        ),
    ]
)

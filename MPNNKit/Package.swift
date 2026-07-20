// swift-tools-version:6.0
import PackageDescription

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
            ]
        ),
        .testTarget(
            name: "MPNNKitTests",
            dependencies: ["MPNNKit", .product(name: "MLX", package: "mlx-swift")]
        ),
    ]
)

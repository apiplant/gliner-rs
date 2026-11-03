// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "GLiNERMLX",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "GLiNERMLX", targets: ["GLiNERMLX"]),
        .executable(name: "gliner-mlx", targets: ["GLiNERCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.33.0"),
    ],
    targets: [
        .target(name: "GLiNERMLX", dependencies: [
            .product(name: "MLX", package: "mlx-swift"),
            .product(name: "MLXNN", package: "mlx-swift"),
        ]),
        .executableTarget(name: "GLiNERCLI", dependencies: ["GLiNERMLX", .product(name: "MLX", package: "mlx-swift")]),
        .testTarget(name: "GLiNERMLXTests", dependencies: ["GLiNERMLX"]),
    ],
    swiftLanguageModes: [.v5]
)

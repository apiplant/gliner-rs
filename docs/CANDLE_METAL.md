# Rust/Candle Metal on Apple devices

The Rust library supports an opt-in Metal backend on Apple Silicon. Build with the `metal` Cargo feature and select `--metal`; CPU remains the default. Metal initialization errors are reported rather than falling back to CPU. For example, from the repository root:

```sh
cargo +1.99.1 run --release --no-default-features --features cli,metal --bin gliner -- \
  --metal --text 'Michele vive a Milano.' --entities person,location --spans --confidence
```

The CLI downloads its default model if needed, or accepts a local checkpoint via `--model DIRECTORY`. The library selects a Metal device explicitly:

```rust
use candle_core::{DType, Device};
use gliner_rs::GLiNER2;
let device = Device::new_metal(0)?;
let model = GLiNER2::load("path/to/checkpoint", &device, DType::F32)?;
```

The default Metal path streams weights using bounded staging and Private GPU storage on macOS, and direct Shared storage on iOS. F32 relative-attention score operations use a fused epilogue; attention remains dense O(T²). `--metal-mmap` and `--metal-original-attention` independently select the original paths. Library callers can use `GLiNER2::load_with_options` and `RustLoadOptions`; the C interface in [`ffi/include/gliner_candle.h`](../ffi/include/gliner_candle.h) uses bit 0 for mmap and bit 1 for original attention.

## iOS and macOS demo

The Xcode project normally builds the Swift/MLX engine. To include Rust/Candle, first build its static library, then enable the compile condition and link it. On macOS:

```sh
SDKROOT="$(xcrun --sdk macosx --show-sdk-path)" MACOSX_DEPLOYMENT_TARGET=14.0 \
  cargo +1.99.1 build --locked --manifest-path ffi/Cargo.toml --release --target aarch64-apple-darwin
xcodebuild -project ios/GLiNERDemo.xcodeproj -scheme GLiNERDemo -configuration Release \
  -destination 'platform=macOS,arch=arm64' CODE_SIGNING_ALLOWED=NO CANDLE_SWIFT_FLAGS=CANDLE_ENABLED \
  "CANDLE_LINK_FLAGS=$PWD/ffi/target/aarch64-apple-darwin/release/libgliner_candle_ffi.a -lc++ -framework Accelerate -framework Metal" build
```

For iPhone, install the Rust target and build against the device SDK:

```sh
rustup +1.99.1 target add aarch64-apple-ios
SDKROOT="$(xcrun --sdk iphoneos --show-sdk-path)" IPHONEOS_DEPLOYMENT_TARGET=17.0 \
  cargo +1.99.1 build --locked --manifest-path ffi/Cargo.toml --release --target aarch64-apple-ios
xcodebuild -project ios/GLiNERDemo.xcodeproj -scheme GLiNERDemo -configuration Release \
  -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO CANDLE_SWIFT_FLAGS=CANDLE_ENABLED \
  "CANDLE_LINK_FLAGS=$PWD/ffi/target/aarch64-apple-ios/release/libgliner_candle_ffi.a -lc++ -framework Accelerate -framework Metal" build
```

These builds are unsigned. To install on a physical device, use Xcode with your own development team and signing profile, while retaining the Candle build settings. The demo serializes model access and downloads supported models on first use. Declared deployment targets are macOS 14 and iOS 17; those minimum versions were not independently tested.

The [Apple technical report](../website/public/reports/gliner-apple-report.html) preserves the recorded validation and benchmark data, including the distinction between RSS and physical footprint. Its results cover the tested devices and workloads, not all Apple hardware.

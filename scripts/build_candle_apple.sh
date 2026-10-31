#!/bin/bash
# Build isolated Rust/Candle Metal static archives. No Apple account or signing changes.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
toolchain="${RUST_TOOLCHAIN:-1.99.1}"
mode="${1:-all}"
case "$mode" in all|macos|ios|app-macos|app-ios) ;; *) echo "Usage: $0 [all|macos|ios|app-macos|app-ios]" >&2; exit 2;; esac
build_one() {
    local sdk="$1" target="$2" destination="$3"
    local sdkroot
    sdkroot="$(xcrun --sdk "$sdk" --show-sdk-path)"
    echo "Building Candle F32/Metal for $sdk ($target)"
    local target_dir="$root/.models/candle-ffi-target"
    SDKROOT="$sdkroot" IPHONEOS_DEPLOYMENT_TARGET=17.0 MACOSX_DEPLOYMENT_TARGET=14.0 \
      CARGO_TARGET_DIR="$target_dir" \
      cargo "+$toolchain" build --locked --manifest-path "$root/ffi/Cargo.toml" --release --target "$target"
    mkdir -p "$root/.models/candle-libs/$destination"
    cp "$target_dir/$target/release/libgliner_candle_ffi.a" "$root/.models/candle-libs/$destination/"
}
if [[ "$mode" == all || "$mode" == macos || "$mode" == app-macos ]]; then build_one macosx aarch64-apple-darwin macosx; fi
if [[ "$mode" == all || "$mode" == ios || "$mode" == app-ios ]]; then build_one iphoneos aarch64-apple-ios iphoneos; fi

if [[ "$mode" == app-ios || "$mode" == app-macos ]]; then
    if [[ "$mode" == app-ios ]]; then sdk=iphoneos; destination='generic/platform=iOS'; else sdk=macosx; destination='platform=macOS,arch=arm64'; fi
    # Optional build settings leave the ordinary MLX-only app build unaffected.
    xcodebuild -project "$root/ios/GLiNERDemo.xcodeproj" -scheme GLiNERDemo -configuration Release \
      -destination "$destination" -derivedDataPath "$root/.derivedData-candle" \
      -clonedSourcePackagesDirPath "$root/swift/.build" CODE_SIGNING_ALLOWED=NO CANDLE_SWIFT_FLAGS=CANDLE_ENABLED \
      "CANDLE_LINK_FLAGS=$root/.models/candle-libs/$sdk/libgliner_candle_ffi.a -lc++ -framework Accelerate -framework Metal" build
fi

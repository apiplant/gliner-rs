#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_mode="${1:-macos}"
task_config="${CONFIGURATION:-}"
if [ ! -f "$task_root/swift/.build/checkouts/mlx-swift/Package.swift" ]; then
    (cd "$task_root/swift" && swift package resolve)
fi
case "$task_mode" in
  macos)
    task_config="${task_config:-Release}"
    cd "$task_root/swift"
    xcodebuild -scheme gliner-mlx -configuration "$task_config" -destination 'platform=macOS,arch=arm64' \
      -derivedDataPath "$task_root/.derivedData" -clonedSourcePackagesDirPath "$task_root/swift/.build" -quiet build
    ;;
  test)
    task_config="${task_config:-Debug}"
    cd "$task_root/swift"
    xcodebuild -scheme GLiNERMLX-Package -configuration "$task_config" -destination 'platform=macOS,arch=arm64' \
      -derivedDataPath "$task_root/.derivedData" -clonedSourcePackagesDirPath "$task_root/swift/.build" -quiet test
    ;;
  ios|simulator|app-macos)
    task_config="${task_config:-Debug}"
    case "$task_mode" in
      ios) task_destination='generic/platform=iOS' ;;
      simulator) task_destination='generic/platform=iOS Simulator' ;;
      app-macos) task_destination='platform=macOS,arch=arm64'; task_config="${CONFIGURATION:-Release}" ;;
    esac
    xcodebuild -project "$task_root/ios/GLiNERDemo.xcodeproj" -scheme GLiNERDemo \
      -configuration "$task_config" -destination "$task_destination" \
      -derivedDataPath "$task_root/.derivedData" -clonedSourcePackagesDirPath "$task_root/swift/.build" CODE_SIGNING_ALLOWED=NO -quiet build
    ;;
  *) echo "Usage: $0 [macos|test|ios|simulator|app-macos]" >&2; exit 2 ;;
esac

#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$task_root/scripts/build_apple.sh" macos >&2
exec "$task_root/.derivedData/Build/Products/${CONFIGURATION:-Release}/gliner-mlx" "$@"

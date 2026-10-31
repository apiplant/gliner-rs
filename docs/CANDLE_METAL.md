# Rust/Candle Metal on Apple devices

The existing Rust model can execute on the Apple GPU through Candle Metal.
A Swift rewrite is not required for Metal or for embedding the model in an
iPhone app. This repository also contains a separate Swift/MLX implementation;
the [technical HTML report](../reports/gliner-apple-report.html) compares their
correctness, request latency and process memory.

## macOS command line

Use an Apple Silicon Mac with Xcode and the Rust 1.99.1 toolchain used for the
recorded builds. Download a pinned checkpoint, then enable both the Cargo
feature and the runtime device flag:

```sh
python3 scripts/download_apple_model.py --variant multi
cargo +1.99.1 run --release --no-default-features --features cli,metal --bin gliner -- \
  --metal --model .models/gliner2.5-multi-v1 \
  --text 'Michele vive a Milano e lavora per Acme.' \
  --entities person,company,location --spans --confidence
```

The default device remains CPU. `--metal` and `--cuda` are mutually exclusive.
Requesting Metal fails explicitly if initialization fails; it never silently
falls back to CPU. The comparison uses F32; the existing `--fp16` flag is not
evidence of verified F16 parity on Metal.

The public Rust model API already accepts a device:

```rust
use candle_core::{DType, Device};
use gliner_rs::GLiNER2;

let device = Device::new_metal(0)?;
let model = GLiNER2::load(".models/gliner2.5-multi-v1", &device, DType::F32)?;
```

## iPhone app and C boundary

The `ffi/` crate builds a Rust static library. Its C interface is documented in
[`ffi/include/gliner_candle.h`](../ffi/include/gliner_candle.h). The SwiftUI demo
serializes calls through an actor and can select either engine. Rust owns its
tokenizer, encoder, heads and decoding; Swift supplies the app interface and
converts request/result JSON at the boundary.

The app declares iOS 17 and macOS 14 deployment targets. Execution has been
tested on a physical iPhone 17 Pro and an M3 Pro Mac running the OS versions
recorded in the report. The minimum OS versions have not been tested. The
resolved `objc2-metal` wrapper uses `MTLCreateSystemDefaultDevice` on iOS; the
name `MTLCopyAllDevices` in Candle does not itself impose an iOS 18 requirement.

```sh
rustup +1.99.1 target add aarch64-apple-ios
bash scripts/build_candle_apple.sh app-ios
```

This produces an unsigned Release app at
`.derivedData-candle/Build/Products/Release-iphoneos/GLiNERDemo.app`.
For macOS, use `bash scripts/build_candle_apple.sh app-macos`.
The ordinary `build_apple.sh` builds retain their MLX-only behavior.

To install on an iPhone, sign with a development profile that covers your
bundle identifier and device, using its matching local keychain identity:

```sh
python3 scripts/sign_apple_demo.py \
  --app .derivedData-candle/Build/Products/Release-iphoneos/GLiNERDemo.app \
  --profile /absolute/path/to/your-development-profile.mobileprovision
xcrun devicectl device install app --device DEVICE_ID \
  .derivedData-candle/Build/Products/Release-iphoneos/GLiNERDemo.app
```

The signing helper reuses a local profile and identity. It does not create or
modify App Store Connect or provisioning-portal resources. No profile, private
key, model checkpoint or compiled archive belongs in the repository.

## Reproduce correctness checks and benchmarks

On Mac:

```sh
cargo +1.99.1 build --release --locked --no-default-features --features metal --example apple_reference
target/release/examples/apple_reference .models/gliner2.5-small-v1 scripts/apple_cases.json --metal > .models/candle.jsonl
target/release/examples/apple_reference .models/gliner2.5-small-v1 scripts/apple_cases.json > .models/cpu.jsonl
python3 scripts/compare_apple_results.py .models/cpu.jsonl .models/candle.jsonl

bash scripts/build_apple.sh macos
python3 scripts/benchmark_metal_comparison.py
```

The benchmark downloads nothing: both Small and Multi checkpoints must already
exist locally. It runs three trials with alternating backend order, five
warm-ups and 30 measured calls per request/backend/process. All outputs are
checked against fresh CPU reference outputs; direct GPU-to-GPU comparison is
also strict. Model loading is separate from request latency. Both CLI timers
are monotonic and include extraction, formatting, JSON and synchronous output.
Filesystem and shader caches are not flushed.

On an installed, signed iPhone demo:

```sh
python3 scripts/run_candle_iphone_suite.py DEVICE_ID --variant small
python3 scripts/run_candle_iphone_suite.py DEVICE_ID --variant multi
```

The device runner verifies unique run IDs before accepting result files, checks
the complete 28-case suite for each backend, and records repeated measurements
for the same three Italian requests used on Mac. Keep the phone available and
avoid other foreground workloads during measurements. Raw diagnostic files
remain under `.models/`; committed summaries are in `docs/benchmarks/`.

## Interpretation and limits

Full F32 output comparison between the GPU engines passes all 28 cases for
Small and Multi on Mac. Against CPU, Multi has the same bounded entity near-tie
ordering exception previously observed with MLX; it is reported separately.
The core compatibility fix creates the DeBERTa mask floor in F32 directly,
avoiding an unsupported intermediate F64 tensor while preserving its final
value.

The measured Mac workloads show a repeatable Swift/MLX latency advantage of
about 23–35% and roughly half the **load-inclusive peak process RSS**. These
measurements do not establish a permanent twofold difference in resident
memory: peak RSS can include mapped checkpoint pages, copies and temporary
allocations during loading. They do not compare language speed or establish
performance on every model/device. The current evidence therefore does not
satisfy the condition for removing Swift/MLX as an equivalent backend.

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

## Metal loading and attention options

The default Metal loader streams weights into persistent buffers: macOS uses a
bounded 16 MiB staging buffer and Private GPU storage; iOS writes directly to
Shared storage. F32 DeBERTa relative-attention score gathering, arithmetic and
masking use a fused epilogue and reuse relative-distance buckets. Attention
remains dense and O(T²); this is not FlashAttention. Other dtypes retain the
original attention path.

The CLI flags `--metal-mmap` and `--metal-original-attention` independently
restore the original mmap loader and original attention. Use both to reproduce
the original configuration. The Rust API exposes the same independent controls:

```rust
use gliner_rs::{GLiNER2, MetalWeightLoading, RustLoadOptions};
let options = RustLoadOptions {
    metal_weight_loading: MetalWeightLoading::Mmap,
    metal_attention_optimizations: false,
};
let model = GLiNER2::load_with_options(".models/gliner2.5-multi-v1", &device, DType::F32, options)?;
```

`RustLoadOptions::default()` selects streaming and optimized attention; CPU and
CUDA retain their existing loaders. The C API `gliner_candle_open_with_options`
uses flags 0–3: bit 0 selects mmap, bit 1 selects original attention. Thus 0
selects both optimizations and 3 restores both original paths. The demo's Rust
optimization toggle selects those combined settings and reloads the model when
changed.

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
exist locally. For the three-engine comparison, provide an archived original
executable with `--baseline-rust-binary PATH` and optionally declare its source
with `--baseline-source-ref REF`. The benchmark rotates engine order across
request/trial pairs, checks exact prepared inputs and every output against a
fresh CPU reference, and compares original against optimized Candle and
optimized Candle against MLX directly. Loading is separate from request latency;
filesystem and shader caches are not flushed.

For the balanced three-way iPhone experiment, first generate local CPU
references with the Release `apple_reference` binary built above. Both pinned
checkpoints must already be downloaded:

```sh
python3 - <<'PY'
import json, os, subprocess
from pathlib import Path

out = Path('.models/iphone-rust-references')
out.mkdir(parents=True, exist_ok=True)
cases = json.loads(Path('scripts/apple_cases.json').read_text())
cases += json.loads(Path('scripts/apple_advanced_cases.json').read_text())
requests = json.loads(Path('docs/benchmarks/rust-vs-mlx.json').read_text())['requests']
for variant in ('small', 'multi'):
    groups = [('parity', cases)] + [(r['name'], [r]) for r in requests]
    for name, values in groups:
        inputs = out / f'{variant}-{name}-cases.json'
        inputs.write_text(json.dumps(values, ensure_ascii=False))
        with (out / f'{variant}-{name}-cpu-reference.jsonl').open('w') as output:
            subprocess.run(['target/release/examples/apple_reference',
                            f'.models/gliner2.5-{variant}-v1', str(inputs)],
                           env=dict(os.environ, RAYON_NUM_THREADS='4'),
                           stdout=output, check=True)
PY

# App must already be signed/installed; choose a fresh artifact tag.
python3 scripts/run_candle_iphone_rotating.py DEVICE_ID --variant small --tag my-run
python3 scripts/run_candle_iphone_rotating.py DEVICE_ID --variant multi --tag my-run
```

The rotating runner transfers each local checkpoint, then executes original
Candle, optimized Candle and MLX in a rotating order within each trial. It
verifies unique run IDs before accepting result files, checks
the complete 28-case suite for each backend, and records repeated measurements
for the same three Italian requests used on Mac. Keep the phone available and
avoid other foreground workloads during measurements. Raw diagnostic files
remain under `.models/`; committed summaries are in `docs/benchmarks/`.
`run_candle_iphone_suite.py` remains available for the simpler two-engine run;
it does not reproduce the balanced three-configuration experiment.

## Interpretation and limits

Full F32 output comparison between the GPU engines passes all 28 cases for
Small and Multi on Mac. Against CPU, Multi has the same bounded entity near-tie
ordering exception previously observed with MLX; it is reported separately.
The core compatibility fix creates the DeBERTa mask floor in F32 directly,
avoiding an unsupported intermediate F64 tensor while preserving its final
value.

In the three-engine Release comparison, each device recorded 54 runs and
1,890 calls across Small/Multi and three short requests, including warm-ups.
Median request latency decreased from original to optimized Candle by 18–31%
on the M3 Pro and 13–31% on iPhone 17 Pro. MLX remained 7–18% lower than
optimized Candle on Mac and 18–23% lower on iPhone. These are workload-specific
comparisons, not language benchmarks.

Peak RSS is not total physical RAM: file-mapped pages and Private/Shared
allocations have different accounting. For the short Mac requests optimized
Candle's peak physical footprint was 0.3–2.7% higher than original, while iPhone
Multi was approximately unchanged. In a separate single mixed long-suite run,
optimized Candle's physical peak was lower by 26%/21% on Mac and 28%/21% on
iPhone for Small/Multi, respectively. The two workloads measure different
peaks; do not infer a general reduction from either. RSS and physical footprint
must not be added. See the [report](../reports/gliner-apple-report.html),
[Mac](benchmarks/candle-optimized-mac.json),
[iPhone](benchmarks/candle-optimized-iphone.json), and
[optimization summary](benchmarks/candle-metal-optimization.json) for methods,
individual values and validation. The earlier two-engine results remain
historical evidence.

Current checks include 23 Rust CPU tests, targeted GPU tests, 2 FFI tests and
45 Python tests; the prior Swift validation recorded 32 XCTest passes.

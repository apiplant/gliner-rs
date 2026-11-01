# Swift/MLX on Apple Silicon

The `GLiNERMLX` Swift package runs boundary/DeBERTa-v2 GLiNER2 checkpoints with MLX and Metal. It supports macOS 14 and iOS 17; the tested devices were an M3 Pro Mac and an iPhone 17 Pro. Building requires Xcode, Swift 6.3 or later, and the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain` if missing). The minimum OS versions were not tested on those versions.

## Model and CLI

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli), then from the repository root download a pinned checkpoint and build with Xcode:

```sh
hf download fastino/gliner2.5-multi-v1 config.json encoder_config/config.json tokenizer.json model.safetensors --revision cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3 --local-dir .models/gliner2.5-multi-v1
cd swift
xcodebuild -scheme gliner-mlx -configuration Release -destination 'platform=macOS,arch=arm64' -derivedDataPath ../.derivedData build
cd ..
.derivedData/Build/Products/Release/gliner-mlx --model .models/gliner2.5-multi-v1 --text 'Michele vive a Milano.' --entities person,location
```

Small and Base checkpoints are also supported. The model directory must contain `config.json`, `encoder_config/config.json`, `tokenizer.json`, and `model.safetensors`. The demo can download Small or Multi itself; models can be reused offline.

## App and library

Open `ios/GLiNERDemo.xcodeproj` in Xcode to build the macOS or iOS app. For a physical iPhone, select a device and configure your own development team in Signing & Capabilities. The standard project builds the Swift/MLX path; see [Candle Metal](../docs/CANDLE_METAL.md) to include the Rust engine.

Add the local `swift` package and its `GLiNERMLX` product to an app. Configure MLX before any MLX computation and serialize access to each model, for example with an actor:

```swift
import Foundation
import GLiNERMLX

actor Extractor {
    private let model: GLiNERModel
    init(directory: URL) throws {
        let profile = InferenceProfile.accurate
        try GLiNERRuntime.configure(profile.runtimeConfiguration)
        model = try GLiNERModel(modelDirectory: directory, precision: profile.precision,
                                runtimeConfiguration: profile.runtimeConfiguration)
    }
    func extract(_ text: String) throws -> String {
        let result = try model.extract(text, schema: Schema(entities: ["person", "location"]))
        return try result.prettyPrinted()
    }
}
```

F32 is the default. `fast` and `low-memory` use experimental F16, which can change results. TF32 is process-global and cached by MLX at its first relevant computation; restart to change it. Cache settings may be updated between serialized requests. The [technical report](../website/public/reports/gliner-apple-report.html) records validation, numerical tolerances and benchmark limitations.

Run package tests from `swift/` with `xcodebuild -scheme GLiNERMLX-Package -destination 'platform=macOS,arch=arm64' test`. Model-dependent tests skip without a Small checkpoint; set `GLINER_MODEL_DIR` to its directory.

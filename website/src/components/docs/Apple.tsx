import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Pre, FlagTable, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsApple() {
  return (
    <DocsLayout>
      <H1>Apple Silicon: MLX and Candle Metal</H1>
      <Lead>
        The native Swift/MLX backend and the Rust/Candle backend can both run on Metal. The
        optional Candle Metal route keeps inference in Rust; it does not replace the Swift package
        or the macOS/iOS demo.
      </Lead>

      <Section>
        <H2 id="requirements">Requirements</H2>
        <P>
          The Swift package declares macOS 14 and iOS 17 as minimum versions. Building requires
          Xcode with Swift 6.3 or later and the Metal toolchain; the checked-in implementation was
          tested with Xcode 27.2 and Swift 6.4.2 on an M3 Pro Mac and a physical iPhone 17 Pro.
          Install the Metal component if Xcode reports it missing:
        </P>
        <CopyBlock command="xcodebuild -downloadComponent MetalToolchain" />
        <P>
          Tokenization, input preparation and decoding run on the CPU; MLX executes model tensor
          operations using Metal. The implementation supports boundary checkpoints with a
          DeBERTa-v2 encoder, including GLiNER2.5 small, base and multi. Span architecture and
          ModernBERT checkpoints are not supported; quantization is not currently available.
        </P>
        <P>
          The optional Rust/Candle Metal library declares macOS 14 and iOS 17 targets as well. A
          physical iPhone 17 Pro was tested with both engines in Release. The declared minimum
          OS versions have not been tested on devices running those exact versions.
        </P>
      </Section>

      <Section>
        <H2 id="candle-metal">Optional Rust/Candle Metal CLI</H2>
        <P>
          Rust continues to use CPU by default. The <IC>metal</IC> feature plus <IC>--metal</IC>
          explicitly selects Candle’s Metal device and fails if it is unavailable; it never falls
          back to CPU. The command below uses F32 and the already downloaded local checkpoint:
        </P>
        <CopyBlock command={`cargo +1.99.1 run --release --no-default-features --features cli,metal --bin gliner -- \\
  --metal \\
  --model .models/gliner2.5-multi-v1 \\
  --text 'Michele vive a Milano.' \\
  --entities person,location \\
  --spans --confidence`} />
        <P>
          Existing Rust CPU and CUDA behavior is unchanged unless you opt in. The demo’s optional
          static-library/C bridge integration and its platform requirements are described in{" "}
          <a class="text-accent underline" href="https://github.com/apiplant/gliner-rs/blob/cd7b50f6ac7b5b395b92c373397ef19368f03a55/docs/CANDLE_METAL.md" target="_blank" rel="noreferrer noopener">the Candle Metal build notes</a>.
        </P>
      </Section>

      <Section>
        <H2 id="cli">Download, build and run</H2>
        <P>From a source checkout on an Apple Silicon Mac:</P>
        <CopyBlock command={`git clone https://github.com/apiplant/gliner-rs.git
cd gliner-rs
python3 scripts/download_apple_model.py --variant multi
bash scripts/build_apple.sh macos
bash scripts/run_apple.sh --model .models/gliner2.5-multi-v1 --text 'Michele vive a Milano e lavora per Acme.' --entities person,company,location`} />
        <P>
          The download script also accepts <IC>small</IC> and <IC>base</IC>. Models are stored
          under <IC>.models/</IC>; after the files are downloaded, inference does not require a
          network connection. Use <IC>--chunked</IC> for long text or <IC>--cases</IC> for a
          JSON case collection.
        </P>
      </Section>

      <Section>
        <H2 id="apps">macOS and iOS demo</H2>
        <P>Build and open the macOS demo, or build the iOS and simulator targets:</P>
        <CopyBlock command={`bash scripts/build_apple.sh app-macos
open .derivedData/Build/Products/Release/GLiNERDemo.app
bash scripts/build_apple.sh ios
bash scripts/build_apple.sh simulator`} />
        <P>
          These scripted app builds disable code signing. To install on a physical iPhone, open
          <IC>ios/GLiNERDemo.xcodeproj</IC>, select a connected device and configure your own
          account and team in Xcode’s Signing &amp; Capabilities, then run the app. In the demo,
          select Small or Multi and press “Scarica modello” (Download model). The downloaded
          files remain available offline.
        </P>
      </Section>

      <Section>
        <H2 id="swift-api">Swift API</H2>
        <P>
          Add the local <IC>swift</IC> package and its <IC>GLiNERMLX</IC> product to your app.
          A model holds MLX state and a mutable processor; serialize access, for example with an
          actor. Configure the runtime at startup before other MLX computation:
        </P>
        <Pre caption="Swift">{`import Foundation
import GLiNERMLX

actor Extractor {
    private let model: GLiNERModel

    init(directory: URL) throws {
        let profile = InferenceProfile.accurate
        try GLiNERRuntime.configure(profile.runtimeConfiguration)
        model = try GLiNERModel(
            modelDirectory: directory,
            precision: profile.precision,
            runtimeConfiguration: profile.runtimeConfiguration
        )
    }

    func extract(_ text: String) throws -> String {
        let result = try model.extract(
            text,
            schema: Schema(entities: ["person", "company", "location"]),
            options: .init(includeSpans: true, includeConfidence: true)
        )
        return try result.prettyPrinted()
    }
}

let extractor = try Extractor(directory: URL(fileURLWithPath: ".models/gliner2.5-multi-v1"))
print(try await extractor.extract("Michele vive a Milano e lavora per Acme."))`}</Pre>
        <P>
          Schemas also support classification, relations and structures. See the public types in
          <IC>swift/Sources/GLiNERMLX/Schema.swift</IC> for their options.
        </P>
      </Section>

      <Section>
        <H2 id="profiles">Precision and runtime controls</H2>
        <FlagTable rows={[
          { flag: "accurate", meaning: "Default: F32 and platform cache policy." },
          { flag: "fast", meaning: "Experimental F16 encoder; retains cache after requests." },
          { flag: "low-memory", meaning: "Experimental F16 encoder; 20 MiB cache limit and clear after each request." },
        ]} />
        <P>
          Select a profile with <IC>--profile accurate|fast|low-memory</IC>. On macOS the platform
          default uses the system MLX cache and retains it; on iOS it limits the cache to 20 MiB
          and clears it after each request. TF32 defaults to off.
        </P>
        <UL>
          <LI><IC>--fp16</IC> or <IC>--fp32</IC> overrides the profile precision.</LI>
          <LI><IC>--tf32</IC> or <IC>--no-tf32</IC> independently controls TF32.</LI>
          <LI><IC>--cache-mb auto|0|N</IC> chooses the system policy, disables unused-buffer caching, or sets an integer MiB limit.</LI>
          <LI><IC>--clear-cache</IC> or <IC>--retain-cache</IC> controls cleanup after each request.</LI>
        </UL>
        <P>
          TF32 is process-global and MLX caches the setting at its first relevant computation;
          restart the process to change it. Cache settings can be updated between requests with
          <IC>model.updateRuntimeConfiguration(...)</IC>; serialize those updates with model use.
          The CLI prints effective settings to stderr, and <IC>--repeat N</IC> reports timings.
        </P>
      </Section>

      <Section>
        <H2 id="validation">Validation and measurements</H2>
        <P>
          The Swift/MLX path on Mac passed the 28-case F32 suite for small, base and multi at a
          confidence tolerance of 0.0001. Multi has one near-tie entity reorder: 27 cases match
          strict ordering and all 28 pass when that narrow exception is allowed. On a physical
          iPhone, MLX passed small in strict order and multi with the same near-tie exception; full
          encoder values were compared at an absolute tolerance of 0.001. Separately, the Mac
          Candle Metal versus MLX Metal parity run returned strict-order agreement on all 28 cases
          for both small and multi. These checks cover the test cases, not every input.
        </P>
        <P>
          F16 is approximate. In an exploratory 21-case comparison at a looser 0.02 confidence
          tolerance, small matched 20 cases and multi 18; differences included a relation span,
          confidence values and selected candidates. Evaluate it on your own workload.
        </P>
        <P>
          The optimized Candle F32 path streams weights using 16 MiB staging and Private storage
          on macOS, and direct Shared storage on iOS. Its fused relative-attention score epilogue
          remains dense O(T²), not FlashAttention. <IC>--metal-mmap</IC> and{" "}
          <IC>--metal-original-attention</IC> independently restore the original paths; Rust
          <IC>RustLoadOptions</IC>, C API flags 0–3 and the demo toggle expose these choices.
        </P>
        <P>
          Release comparisons on an M3 Pro and iPhone 17 Pro used small and multi, three short
          requests and three trials, with five warm-ups and 30 measured calls per request and
          engine. Each device recorded 54 runs and 1,890 calls. Original-to-optimized Candle
          median latency decreased 18–31% on Mac and 13–31% on iPhone; MLX remained 7–18% and
          18–23% lower, respectively. These results apply to the measured workloads.
        </P>
        <P>
          Peak RSS is not total physical RAM: mapped files and Private/Shared allocations have
          different accounting. For short Mac requests, optimized Candle peak physical footprint
          was 0.3–2.7% higher; iPhone multi was approximately unchanged. In a separate single mixed
          long-suite run, physical peaks fell 26%/21% on Mac and 28%/21% on iPhone for small/multi.
          RSS and physical footprint overlap and must not be added. The report includes per-trial
          measurements, validation and methods.
        </P>
        <P>
          An earlier comparison of Rust/Candle CPU against Swift/MLX Metal reported a 9.59–10.35×
          median latency ratio on the same three inputs. It changed both the runtime and the device
          path, so it is historical CPU-versus-GPU context rather than evidence about Candle Metal
          versus MLX Metal.
        </P>
        <P>
          See the <a class="text-accent underline" href="/reports/gliner-apple-report.html" target="_blank" rel="noreferrer noopener">Apple implementation and benchmark report</a> for
          the full measurements and validation evidence.
        </P>
        <P>
          The model-dependent Swift tests skip without a local checkpoint. Download small to
          reproduce all 32 Swift tests; the verifier also builds the Rust reference using the
          installed Rust 1.99.1 toolchain:
        </P>
        <CopyBlock command={`python3 scripts/download_apple_model.py --variant small
bash scripts/verify_apple.sh multi
python3 -m unittest discover -s tests -p 'test_*.py'`} />
      </Section>
    </DocsLayout>
  );
}

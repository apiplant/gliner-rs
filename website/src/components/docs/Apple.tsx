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
          A new Metal-to-Metal comparison used Release builds on an M3 Pro, F32, pinned small and
          multi checkpoints, and three Italian extraction requests of 64, 97 and 130 words. Each
          request/backend pair ran in a fresh process across three independent trials, with five
          warm-ups and 30 measured calls per trial. Backend order alternated. Timings include
          preparation, inference and JSON output; model loading was measured separately. Before
          timing, prepared input records matched exactly. Across 630 per-run GPU output pairs, spans,
          labels, order and confidence values stayed within the strict comparison tolerance.
        </P>
        <FlagTable rows={[
          { flag: "small · 64 / 97 / 130 words", meaning: "Candle Metal 13.2 / 18.7 / 23.0 ms; MLX Metal 10.2 / 12.2 / 15.1 ms." },
          { flag: "multi · 64 / 97 / 130 words", meaning: "Candle Metal 25.0 / 35.6 / 49.6 ms; MLX Metal 18.8 / 24.9 / 33.4 ms." },
        ]} />
        <P>
          These warm medians show MLX 23–35% lower latency on these six Mac workloads. Peak process
          RSS, including loaded weights, was about 0.81 versus 0.40 GB for small and 2.69 versus
          1.34 GB for multi (Candle versus MLX). These are measurements of complete processes, not
          GPU-only allocation totals, persistent resident-memory differences or memory guarantees. The report includes p95, individual
          trials, model-load timings, checksums and the complete method. It does not establish a
          general advantage across devices, tasks or workloads.
        </P>
        <P>
          This run establishes strict output parity, but performance and process memory are not
          comparable: MLX was faster and used about half the peak RSS on these workloads. The
          evidence therefore does not meet a criterion that would make the Swift/MLX path
          redundant based on comparable outputs, performance and memory.
        </P>
        <P>
          The physical iPhone 17 Pro Release comparison also passed all 28 cases in direct
          GPU-to-GPU order for Small and Multi, plus all 630 paired benchmark outputs. Across
          the six workloads, MLX median request latency was 30–43% lower. Both engines ran in
          the same SwiftUI harness; each process covered all three requests. Candle had lower
          model-load times with warm system/shader caches. Phone timing excludes diagnostic
          JSONL writing, so absolute phone and CLI times should not be compared directly.
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

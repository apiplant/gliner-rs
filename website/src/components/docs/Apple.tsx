import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Pre, FlagTable, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsApple() {
  return (
    <DocsLayout>
      <H1>Apple: Swift and MLX</H1>
      <Lead>
        Run the native Swift implementation on Apple Silicon with MLX and Metal. This is a
        separate inference path from the Rust CPU/CUDA implementation and the browser WASM build.
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
          On Mac, F32 small, base and multi passed 28 test cases with a confidence tolerance of
          0.0001. Multi has one near-tie entity reorder: 27 cases match strict ordering and all 28
          pass with that exception. Physical iPhone tests covered small (28/28 in strict order)
          and multi (the same near-tie exception); encoder values were compared with an absolute
          tolerance of 0.001. These results cover the tested cases, not every input.
        </P>
        <P>
          F16 is approximate. In an exploratory 21-case comparison at a looser 0.02 confidence
          tolerance, small matched 20 cases and multi 18; differences included a relation span,
          confidence values and selected candidates. Evaluate it on your own workload.
        </P>
        <P>
          In a separate paired comparison on an M3 Pro, Release builds using the multi checkpoint
          and F32 measured warm median Rust/Candle CPU times of 201, 285 and 386 ms versus
          Swift/MLX Metal GPU times of 21, 28 and 37 ms for 64, 97 and 130-word inputs. Each
          configuration had five warm-up calls and 30 measured calls; CPU used eight Rayon
          threads. This compares runtimes and devices, not language speed. Cold start and other
          workloads can differ.
        </P>
        <P>
          See the <a class="text-accent underline" href="/reports/gliner-apple-report.html" target="_blank" rel="noreferrer noopener">Apple implementation and benchmark report</a> for
          methods, p95, numerical results and source data.
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

import { For } from "solid-js";
import { Badge, LinkButton, Mono } from "./ui";
import { CopyBlock } from "./Code";
import { Pre } from "./docs/Prose";
import { highlight } from "../lib/highlight";
import { GITHUB_URL } from "../lib/links";
import { PLATFORMS, LATEST_RELEASE_URL, assetName, downloadUrl } from "../lib/release";

/* ------------------------------------------------------------------ */
/* A fake terminal panel showing a real gliner invocation and output.  */
/* ------------------------------------------------------------------ */

function TerminalDemo() {
  const command =
    'gliner --text "Alice works for Acme in Paris." \\\n' +
    "  --entities person,company,location \\\n" +
    "  --relations works_for,located_in \\\n" +
    '  --classify "sentiment=positive,negative,neutral" \\\n' +
    "  --spans --confidence";
  const output = `{
  "entities": {
    "person": [{"text":"Alice","start":0,"end":5,"confidence":0.998}],
    "company": [{"text":"Acme","start":16,"end":20,"confidence":0.994}],
    "location": [{"text":"Paris","start":24,"end":29,"confidence":0.991}]
  },
  "relations": [
    {"head":"Alice","tail":"Acme","type":"works_for","confidence":0.973}
  ],
  "classification": {"sentiment":{"label":"neutral","confidence":0.812}}
}`;

  return (
    <div class="overflow-hidden rounded-xl border border-line shadow-2xl">
      <div class="flex items-center gap-2 border-b border-line bg-surface px-4 py-2.5">
        <span class="h-2.5 w-2.5 rounded-full bg-danger" />
        <span class="h-2.5 w-2.5 rounded-full bg-warn" />
        <span class="h-2.5 w-2.5 rounded-full bg-success" />
        <span class="ml-2 font-mono text-xs text-faint">gliner</span>
      </div>
      <pre class="overflow-x-auto bg-code-bg px-4 py-4 font-mono text-[0.78rem] leading-relaxed">
        <code class="language-bash">
          <span class="select-none text-faint">$ </span>
          <span innerHTML={highlight(command, "bash")} />
        </code>
        {"\n\n"}
        <code class="language-json">
          <span innerHTML={highlight(output, "json")} />
        </code>
      </pre>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The four binaries, one card each.                                  */
/* ------------------------------------------------------------------ */

const BINARIES = [
  {
    name: "gliner",
    href: "/docs/cli",
    tagline: "Entities, relations, classification, structure",
    body: "The generic extraction CLI. One schema of entities, relations, a classification task and structured extraction fields, run in a single encoder pass.",
  },
  {
    name: "gliner-classify",
    href: "/docs/classify",
    tagline: "Zero-shot classification from the terminal",
    body: "One-shot flags or an interactive shell with history. Single- or multi-label, several tasks at once, descriptions, few-shot examples, JSON/JSONL/TSV output.",
  },
  {
    name: "gliner-pii",
    href: "/docs/pii",
    tagline: "PII detection and redaction",
    body: "Preloaded with a 42-label PII taxonomy. Prints per-label spans as JSON, or redact matched spans in place with [LABEL] placeholders.",
  },
  {
    name: "gliner-guardrails",
    href: "/docs/guardrails",
    tagline: "LLM prompt/response moderation",
    body: "Safety verdicts plus toxicity categories and jailbreak-strategy detection on prompts, or refusal-vs-compliance on responses.",
  },
];

function Binaries() {
  return (
    <div class="grid gap-4 sm:grid-cols-2">
      <For each={BINARIES}>
        {(b) => (
          <a
            href={b.href}
            class="group block rounded-xl border border-line bg-surface p-5 transition-colors hover:border-line-strong"
          >
            <div class="flex items-center justify-between gap-2">
              <h3 class="font-mono text-[0.9375rem] font-semibold tracking-tight text-ink">
                {b.name}
              </h3>
              <span class="text-xs text-faint transition-colors group-hover:text-accent">docs →</span>
            </div>
            <p class="mt-1 text-xs font-medium uppercase tracking-[0.1em] text-accent">{b.tagline}</p>
            <p class="mt-2.5 text-sm leading-relaxed text-muted">{b.body}</p>
          </a>
        )}
      </For>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Features.                                                          */
/* ------------------------------------------------------------------ */

const FEATURES = [
  {
    title: "Zero-shot entity extraction",
    body: "Labels, descriptions, per-label thresholds, list/str dtypes, and the abstention head. Overlap policies: flat, nested, allow, longest.",
  },
  {
    title: "Text classification",
    body: "Single- or multi-label, label descriptions, few-shot examples — any number of tasks scored in one pass.",
  },
  {
    title: "Relation extraction",
    body: "Typed capped pair proposal, the biaffine relation scorer, and edge deduplication across a document.",
  },
  {
    title: "Structured extraction",
    body: "extract_json with the record head in natural, latent or anchorless mode. Cardinality, exclusive fields, choice fields, thresholds and descriptions all compose.",
  },
  {
    title: "Long documents",
    body: "extract_entities_long and friends chunk overrunning text into overlapping word windows, then remap and merge spans back onto the original document.",
  },
  {
    title: "Batching",
    body: "extract_batch and friends run several independent texts against the same schema in one padded encoder pass, with results identical to calling one at a time.",
  },
  {
    title: "Entity attributes",
    body: "Extra typed properties force-scored at an already-extracted entity's exact span — a color or sentiment attribute on a product entity, single- or multi-label.",
  },
  {
    title: "Regex validators",
    body: "Post-filter an extracted span's surface text with a full or partial regex match, optionally inverted, per entity or field.",
  },
  {
    title: "CPU or CUDA",
    body: "One dependency-free binary either way. The encoder runs in F32 or F16; the heads always run in F32 for numerical stability.",
  },
];

function Features() {
  return (
    <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <For each={FEATURES}>
        {(f) => (
          <div class="rounded-xl border border-line bg-surface p-5">
            <h3 class="text-[0.9375rem] font-semibold tracking-tight text-ink">{f.title}</h3>
            <p class="mt-2 text-sm leading-relaxed text-muted">{f.body}</p>
          </div>
        )}
      </For>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Install: numbered step cards, the apiplant layout.                  */
/* ------------------------------------------------------------------ */

const stepCard =
  "grid min-w-0 gap-5 rounded-2xl border bg-surface p-5 sm:p-6 lg:grid-cols-[minmax(14rem,0.7fr)_minmax(0,1.3fr)] lg:items-start";

function InstallSteps() {
  const homebrewCommands = `brew tap apiplant/tap
brew install apiplant/tap/gliner-rs`;
  const pacmanCommands = `curl -sSfL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg
keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ { print $5; exit }') && sudo pacman-key --add /tmp/apiplant.gpg && sudo pacman-key --finger "$keyid" && sudo pacman-key --lsign-key "$keyid"
printf '\\n[apiplant]\\nSigLevel = Required DatabaseOptional\\nServer = https://apiplant.github.io/pacman/$arch\\n' | sudo tee -a /etc/pacman.conf > /dev/null
sudo pacman -Sy gliner-rs`;
  const aptCommands = `curl -sSfL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main" | sudo tee /etc/apt/sources.list.d/apiplant.list > /dev/null
sudo apt update && sudo apt install gliner-rs`;
  const cargoCommands = `cargo add gliner-rs         # as a library dependency
cargo install gliner-rs     # gliner / gliner-classify / gliner-pii / gliner-guardrails
cargo install gliner-rs --features cuda  # with CUDA support`;

  return (
    <div class="mt-8 space-y-4 sm:mt-10">
      <div class={`${stepCard} border-accent-line`}>
        <div>
          <div class="flex items-center gap-2">
            <span class="font-mono text-xs text-accent">01</span>
            <Badge tone="accent">Recommended</Badge>
          </div>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Use a package manager</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            macOS (Apple Silicon), Arch Linux and Debian/Ubuntu are all published to the apiplant
            shared repositories.
          </p>
        </div>

        <div class="min-w-0 space-y-5">
          <div>
            <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">Homebrew</p>
            <CopyBlock command={homebrewCommands} />
          </div>

          <div>
            <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">
              Arch Linux / pacman
            </p>
            <CopyBlock command={pacmanCommands} />
          </div>

          <div>
            <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">
              Debian / Ubuntu
            </p>
            <CopyBlock command={aptCommands} />
          </div>
        </div>
      </div>

      <div class={`${stepCard} border-line`}>
        <div>
          <span class="font-mono text-xs text-accent">02</span>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Download the archive</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            One archive per platform, holding all four binaries — <Mono>gliner</Mono>,{" "}
            <Mono>gliner-classify</Mono>, <Mono>gliner-pii</Mono>, <Mono>gliner-guardrails</Mono> —
            and the README. No installation needed; unpack and run. On Linux x86_64 with an NVIDIA
            GPU, grab the CUDA archive instead and pass <Mono>--cuda</Mono> to use it.
          </p>
        </div>

        <ul class="min-w-0 space-y-1 border-t border-line pt-4 lg:border-t-0 lg:pt-0">
          <For each={PLATFORMS}>
            {(platform) => (
              <li class="min-w-0">
                <a
                  href={downloadUrl(platform)}
                  title={assetName(platform)}
                  class="flex min-w-0 items-baseline justify-between gap-3 rounded-md py-1 text-muted transition-colors hover:text-ink"
                >
                  <span class="shrink-0 text-sm">{platform.label}</span>
                  <span class="min-w-0 truncate font-mono text-xs text-accent">
                    {assetName(platform)}
                  </span>
                </a>
              </li>
            )}
          </For>
        </ul>

        <a
          href={LATEST_RELEASE_URL}
          target="_blank"
          rel="noreferrer noopener"
          class="lg:col-start-2 text-sm font-medium text-accent hover:text-accent-dim"
        >
          All releases and checksums
        </a>
      </div>

      <div class={`${stepCard} border-line`}>
        <div>
          <span class="font-mono text-xs text-faint">03</span>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Cargo</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            As a library dependency, or to build the CLIs from source via crates.io.
          </p>
        </div>

        <div class="min-w-0">
          <CopyBlock command={cargoCommands} />
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page.                                                              */
/* ------------------------------------------------------------------ */

export function Home() {
  return (
    <div class="mx-auto w-full max-w-6xl px-5">
      {/* Hero */}
      <section class="grid items-center gap-10 py-16 sm:py-20 lg:grid-cols-2 lg:gap-12">
        <div>
          <div class="flex flex-wrap items-center gap-2">
            <Badge tone="accent">v{__VERSION__}</Badge>
            <Badge>Rust</Badge>
            <Badge>candle</Badge>
            <Badge>Apache-2.0</Badge>
          </div>
          <h1 class="mt-5 text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
            GLiNER2 inference, <span class="text-accent">in Rust</span>.
          </h1>
          <p class="mt-4 max-w-lg text-lg leading-relaxed text-muted">
            Zero-shot entity extraction, classification, relation extraction, structured
            extraction, PII detection and LLM guardrails — from one dependency-free binary, or
            as a library. No Python, no server.
          </p>
          <div class="mt-7">
            <LinkButton
              href="/demo"
              variant="primary"
              class="!px-8 !py-4 !text-lg shadow-lg shadow-accent/20"
            >
              ▶ Try it now — runs in your browser
            </LinkButton>
          </div>
          <div class="mt-4 flex flex-wrap gap-3">
            <LinkButton href={GITHUB_URL} size="lg">
              View on GitHub
            </LinkButton>
            <LinkButton href="/docs" size="lg">
              Read the docs
            </LinkButton>
            <LinkButton href="/#install" size="lg">
              Install
            </LinkButton>
          </div>
          <p class="mt-5 text-sm text-faint">
            Works with <Mono>fastino/gliner2.5-multi-v1</Mono> and other checkpoints in the
            GLiNER2 boundary-architecture family.
          </p>
        </div>
        <TerminalDemo />
      </section>

      {/* Binaries */}
      <section class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink">Four binaries, one crate</h2>
        <p class="mt-2 max-w-2xl text-muted">
          A generic extraction CLI plus three task-specific tools built on the same encoder and
          checkpoint family.
        </p>
        <div class="mt-8">
          <Binaries />
        </div>
      </section>

      {/* Features */}
      <section id="features" class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink">
          Everything the GLiNER2 architecture supports
        </h2>
        <p class="mt-2 max-w-2xl text-muted">
          Any mix of entities, relations, structures and classification runs as a single encoder
          pass — as a library call or a CLI flag.
        </p>
        <div class="mt-8">
          <Features />
        </div>
      </section>

      {/* Install */}
      <section id="install" class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">Install</h2>
        <p class="mt-3 max-w-2xl leading-relaxed text-muted">
          Use Homebrew, pacman or apt when your platform has it. Otherwise take the prebuilt
          archive, or pull the library straight from crates.io.
        </p>

        <InstallSteps />
      </section>

      {/* Library */}
      <section class="border-t border-line pb-20 pt-16">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <h2 class="text-2xl font-semibold tracking-tight text-ink">As a library</h2>
          <LinkButton href="/docs/library" size="sm">
            Full library docs →
          </LinkButton>
        </div>
        <p class="mt-3 max-w-2xl leading-relaxed text-muted">
          One <Mono>Schema</Mono>, one call. Entities, relations and a classification task, all
          scored in a single encoder pass.
        </p>
        <div class="mt-6">
          <CopyBlock command="cargo add gliner-rs" />
        </div>
        <Pre caption="src/main.rs" lang="rust">{`use candle_core::{DType, Device};
use gliner_rs::{ClassificationSpec, ExtractOptions, GLiNER2, Schema};

let model = GLiNER2::load("path/to/checkpoint", &Device::Cpu, DType::F32)?;
let schema = Schema::new()
    .entities(["person", "company"])
    .relations(["works_for"])
    .classification(ClassificationSpec::new("sentiment", ["positive", "negative"]));
let opts = ExtractOptions { include_spans: true, include_confidence: true, ..Default::default() };
let result = model.extract("Alice works for Acme.", &schema, &opts)?;`}</Pre>
      </section>
    </div>
  );
}

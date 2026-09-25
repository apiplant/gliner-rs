import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Pre, Section } from "./Prose";
import { LinkButton, Mono } from "../ui";
import { CopyBlock } from "../Code";
import { GITHUB_URL } from "../../lib/links";

export function DocsOverview() {
  return (
    <DocsLayout>
      <H1>Documentation</H1>
      <Lead>
        gliner-rs is a Rust (<Mono>candle</Mono>) port of the inference path of{" "}
        <a
          href="https://github.com/fastino-ai/GLiNER2"
          target="_blank"
          rel="noreferrer noopener"
          class="text-accent hover:text-accent-dim"
        >
          fastino-ai/GLiNER2
        </a>
        , for GLiNER2 <strong class="text-ink font-medium">boundary-architecture</strong>{" "}
        checkpoints. It ships as a library crate and four command-line binaries.
      </Lead>

      <Section>
        <H2>Checkpoints</H2>
        <P>
          Any checkpoint in the boundary-architecture family works, including{" "}
          <IC>fastino/gliner2.5-multi-v1</IC> (the default — 205M parameters, all languages),{" "}
          <IC>gliner2.5-small-v1</IC>, <IC>gliner2.5-base-v1</IC> (smaller, faster, English-leaning),{" "}
          <IC>GLiNER2.5-Decide</IC>, <IC>GLiNER2.5-multi-Decide</IC>, <IC>GLiNER2.5-Decide-1B</IC>{" "}
          (decision/verification fine-tunes),{" "}
          <IC>gliner2-privacy-filter-PII-multi</IC>, <IC>gliguard-LLMGuardrails-300M</IC>,{" "}
          <IC>GLiNER2-Guardrails-PII-Multi</IC>, and other fine-tunes on the same architecture. On
          first use, each binary downloads its checkpoint automatically into a cache directory —
          see <a href="#checkpoint-resolution" class="text-accent hover:text-accent-dim">checkpoint resolution</a>{" "}
          below.
        </P>
      </Section>

      <Section>
        <H2>The four binaries</H2>
        <UL>
          <LI>
            <a href="/docs/cli" class="text-accent hover:text-accent-dim"><IC>gliner</IC></a> — the
            generic extraction CLI: entities, relations, classification and structured extraction,
            any mix, in one schema.
          </LI>
          <LI>
            <a href="/docs/classify" class="text-accent hover:text-accent-dim"><IC>gliner-classify</IC></a> —
            zero-shot text classification, one-shot or in an interactive shell.
          </LI>
          <LI>
            <a href="/docs/pii" class="text-accent hover:text-accent-dim"><IC>gliner-pii</IC></a> —
            PII detection and redaction, preloaded with a 42-label taxonomy.
          </LI>
          <LI>
            <a href="/docs/guardrails" class="text-accent hover:text-accent-dim"><IC>gliner-guardrails</IC></a> —
            LLM prompt/response safety moderation.
          </LI>
        </UL>
        <P>
          For using gliner-rs as a Rust dependency instead of (or alongside) the CLIs, see{" "}
          <a href="/docs/library" class="text-accent hover:text-accent-dim">As a library</a>.
        </P>
      </Section>

      <Section>
        <H2>Install</H2>
        <P>Build every binary from the crate in one shot:</P>
        <CopyBlock command={`cargo build --release                  # CPU\ncargo build --release --features cuda  # CUDA`} />
        <P>
          Or install from crates.io, or take a prebuilt archive / package for your platform — see
          the <a href="/#install" class="text-accent hover:text-accent-dim">install section</a> on
          the home page for Homebrew, pacman, apt and direct-download options.
        </P>
      </Section>

      <Section>
        <H2 id="checkpoint-resolution">CLI checkpoint resolution</H2>
        <P>
          <IC>gliner</IC>, <IC>gliner-classify</IC>, <IC>gliner-pii</IC> and{" "}
          <IC>gliner-guardrails</IC> all resolve their model directory the same way when you don't
          pass <IC>--model</IC>/<IC>GLINER_MODEL</IC> explicitly: each variant's checkpoint lives in
          the gliner-rs cache directory, <IC>$XDG_CACHE_HOME/gliner-rs</IC> (default{" "}
          <IC>~/.cache/gliner-rs</IC>), under a subdirectory named after its Hugging Face repo
          (e.g. <IC>gliner2.5-multi-v1</IC>). If it isn't there yet, it's downloaded automatically
          on first use.
        </P>
        <P>
          Set <IC>GLINER_OFFLINE=1</IC> to disable downloading — resolution then fails with an
          error naming the missing file unless the checkpoint is already fully cached.
        </P>
        <P>
          If you already have checkpoints downloaded elsewhere (e.g. via <IC>git lfs</IC> or{" "}
          <IC>huggingface-cli</IC>), symlink the whole collection into the cache directory instead
          of re-downloading:
        </P>
        <Pre caption="shell">{`ln -s /path/to/your/gliner/checkpoints ~/.cache/gliner-rs`}</Pre>
        <P>
          where the target contains subdirectories named after the Hugging Face repos (
          <IC>gliner2.5-multi-v1</IC>, <IC>gliner2-privacy-filter-PII-multi</IC>, ...).
        </P>
      </Section>

      <Section>
        <H2>Source and issues</H2>
        <P>The crate, source, and issue tracker all live on GitHub.</P>
        <div class="mt-4">
          <LinkButton href={GITHUB_URL} variant="primary">
            View on GitHub
          </LinkButton>
        </div>
      </Section>
    </DocsLayout>
  );
}

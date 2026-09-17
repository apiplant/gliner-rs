import { DocsLayout } from "./DocsLayout";
import { H1, H2, H3, Lead, P, UL, LI, IC, Pre, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsLibrary() {
  return (
    <DocsLayout>
      <H1>As a library</H1>
      <Lead>
        Add gliner-rs as a Rust dependency to run GLiNER2 extraction directly in your process —
        no CLI, no subprocess.
      </Lead>

      <Section>
        <H2>Install</H2>
        <CopyBlock command="cargo add gliner-rs" />
        <P>
          Enable CUDA with the <IC>cuda</IC> feature: <IC>cargo add gliner-rs --features cuda</IC>.
        </P>
      </Section>

      <Section>
        <H2>Load a model and run a schema</H2>
        <P>
          One <IC>Schema</IC>, one call. Entities, relations and a classification task, all scored
          in a single encoder pass:
        </P>
        <Pre caption="src/main.rs" lang="rust">{`use candle_core::{DType, Device};
use gliner_rs::{ClassificationSpec, ExtractOptions, GLiNER2, Schema};

let model = GLiNER2::load("path/to/checkpoint", &Device::Cpu, DType::F32)?;
let schema = Schema::new()
    .entities(["person", "company"])
    .relations(["works_for"])
    .classification(ClassificationSpec::new("sentiment", ["positive", "negative"]));
let opts = ExtractOptions { include_spans: true, include_confidence: true, ..Default::default() };
let result = model.extract("Alice works for Acme.", &schema, &opts)?;`}</Pre>
        <P>
          <IC>start</IC> and <IC>end</IC> in the result are character (code point) offsets — the
          same convention as Python string indexing.
        </P>
      </Section>

      <Section>
        <H2>Structured extraction</H2>
        <P>
          <IC>extract_json</IC> is a shorthand over <IC>Schema::structure</IC> for the common case:
          a list of records with typed fields.
        </P>
        <Pre caption="structured extraction" lang="rust">{`let orders = model.extract_json(
    "Order #1234 for 3 laptops was shipped via FedEx.",
    &[("order", &["order_id::str", "status::[shipped|pending]", "carrier::str"])],
    &ExtractOptions::default(),
)?;`}</Pre>
        <P>
          The record head runs in <IC>natural</IC> mode by default — a field is the anchor (the
          first by default, or pick one with <IC>StructureSpec::anchor</IC>) and each record keeps
          its own values. <IC>latent</IC> mode has no declared anchor: the head's own learned
          selector chooses which mention seeds each instance. <IC>anchorless</IC> mode uses
          document-conditioned learned instance queries, for records not seeded by any single span.
          <IC>StructureMode::Legacy</IC> gives the aggregate decoder instead of the record head.
        </P>
        <P>
          Per-field <IC>cardinality</IC> (<IC>optional_one</IC>/<IC>required_one</IC>/
          <IC>zero_or_more</IC>/<IC>one_or_more</IC>, via <IC>FieldSpec::cardinality</IC>) and{" "}
          <IC>exclusive</IC> (<IC>FieldSpec::exclusive</IC>) refine how mentions bind to fields;{" "}
          <IC>str</IC>/<IC>list</IC> fields, descriptions, thresholds, and choice fields (
          <IC>status::[shipped|pending]</IC>, including literal-mention binding and prefix scoring)
          all work in every mode.
        </P>
      </Section>

      <Section>
        <H2>Entity attributes</H2>
        <P>
          Attach extra typed properties to an already-extracted entity's exact span, instead of
          proposing spans of their own:
        </P>
        <Pre caption="entity attributes" lang="rust">{`use gliner_rs::AttributeGroup;

let schema = Schema::new().entities(["product"]).entity_attributes([(
    "color",
    AttributeGroup::new(["red", "blue", "green"]).applies_to(["product"]),
)]);
let result = model.extract("Apple released a red iPhone this year.", &schema, &opts)?;
// entities.product[0] == {"text": "iPhone", "color": {"label": "red", "confidence": 0.94}, ...}`}</Pre>
        <P>
          A group is single-label by default (softmax + argmax over its values); call{" "}
          <IC>.multi_label(threshold)</IC> for independent sigmoid decisions instead.{" "}
          <IC>.qualify_labels()</IC> prefixes a group's values with its name in the model-facing
          prompt (<IC>"color: red"</IC>) while keeping the returned label unqualified, for when the
          same value string is meaningful in more than one group.
        </P>
      </Section>

      <Section>
        <H2>Long documents</H2>
        <P>
          Every extraction method has a <IC>_long</IC> counterpart for text that overruns the
          encoder's context window (the boundary architecture's default checkpoints top out at 512
          tokens):
        </P>
        <Pre caption="chunked extraction" lang="rust">{`use gliner_rs::ChunkOptions;

let result = model.extract_entities_long(
    &long_document,
    &["person", "company", "location"],
    &ExtractOptions { include_spans: true, ..Default::default() },
    ChunkOptions::default(), // 384-word chunks, 64-word overlap
)?;`}</Pre>
        <P>
          The document is split into overlapping word windows (<IC>ChunkOptions::chunk_size</IC>/
          <IC>chunk_overlap</IC>, in words as counted by the active word splitter), each window
          runs through the ordinary single-pass API, and every span in the result is remapped to
          character offsets in the <em>original</em> document before merging. Exact re-detections
          in overlap regions keep their highest-confidence copy; genuine span overlaps resolve with{" "}
          <IC>opts.overlap_policy</IC> (default <IC>flat</IC>). Relation edges are always merged
          with <IC>allow</IC>. <IC>extract_relations_long</IC> and <IC>extract_json_long</IC> never
          synthesize relations or record fields across a chunk boundary.
        </P>
        <P>
          <IC>extract_long</IC> is the schema-based generic entry point the convenience methods
          build on, for schemas that mix entities, relations, structures and classification in one
          chunked pass.
        </P>
      </Section>

      <Section>
        <H2>Batching</H2>
        <P>
          Every extraction and classification method has a <IC>_batch</IC> counterpart for running
          several <em>independent</em> texts against the same schema in one encoder pass instead of
          one call per text:
        </P>
        <Pre caption="batched extraction" lang="rust">{`let texts = ["Apple released a new iPhone.", "Nvidia unveiled a new GPU."];
let results = model.extract_entities_batch(
    &texts,
    &["company", "product"],
    &ExtractOptions::default(),
)?; // one Value per text, in order`}</Pre>
        <P>
          Texts are tokenized independently, padded to the batch's longest sequence, and run
          through the encoder together; padding is masked out of attention so every text's result
          is identical to what the single-text method would give it alone — batching only saves
          encoder calls, it never changes an answer. One call takes one schema: every text in a
          batch is scored against the same entities/labels/structures, only the documents vary.
        </P>
      </Section>

      <Section>
        <H2>Word splitters</H2>
        <P>
          The default word splitter is whitespace-based. Pass a character-level (CJK) splitter for
          Chinese and Japanese text where whitespace doesn't mark word boundaries.
        </P>
      </Section>

      <Section>
        <H2>Overlap policies</H2>
        <UL>
          <LI><IC>flat</IC> (default) — the maximum-total-confidence non-overlapping set.</LI>
          <LI><IC>nested</IC> — spans may nest inside one another, but not partially overlap.</LI>
          <LI><IC>allow</IC> — every candidate span is kept, overlaps and all.</LI>
          <LI><IC>longest</IC> — the longest span wins any overlap.</LI>
        </UL>
      </Section>

      <Section>
        <H2>CPU vs. CUDA</H2>
        <P>
          Build with <IC>--features cuda</IC> and pass <IC>&Device::cuda_if_available(0)?</IC> (or
          your own <IC>Device::Cuda</IC>) to <IC>GLiNER2::load</IC>. The encoder can run in{" "}
          <IC>DType::F32</IC> or <IC>DType::F16</IC>; the heads always run in F32 regardless, for
          numerical stability.
        </P>
      </Section>

      <Section>
        <H3>Implementation notes</H3>
        <UL>
          <LI><IC>deberta.rs</IC> — DeBERTa-v2 with disentangled c2p/p2c attention, shared keys and log buckets.</LI>
          <LI>
            <IC>heads.rs</IC> — the boundary encoder (windowed attention plus SwiGLU), query
            marginals with a centered inside prefix, the shared document candidate pool (top-k
            union, per-query quota, dedup), the FiLM pool scorer, the classifier MLP and the
            relation scorer.
          </LI>
          <LI>
            <IC>processor.rs</IC> — prompt layout, first-subword routing, and the reference
            collator's quirk of appending <IC>.</IC> when the text doesn't end in <IC>.</IC>,{" "}
            <IC>!</IC> or <IC>?</IC>.
          </LI>
          <LI>
            <IC>records.rs</IC> — natural-mode instance formation, a global exclusive assignment
            (Hungarian solver with the reference tie-breaking), and literal choice-mention binding.
          </LI>
          <LI><IC>decode.rs</IC> — weighted-interval overlap resolution, relation pair proposal, and relation edge deduplication.</LI>
        </UL>
      </Section>
    </DocsLayout>
  );
}

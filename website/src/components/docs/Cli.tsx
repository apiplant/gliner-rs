import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Pre, FlagTable, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsCli() {
  return (
    <DocsLayout>
      <H1><IC>gliner</IC></H1>
      <Lead>
        The generic extraction CLI: entities, relations, a classification task and structured
        extraction, any mix, run as a single encoder pass over one schema.
      </Lead>

      <Section>
        <H2>Build</H2>
        <CopyBlock command={`cargo build --release                  # CPU\ncargo build --release --features cuda  # CUDA`} />
      </Section>

      <Section>
        <H2>Basic usage</H2>
        <CopyBlock
          command={`gliner \\
  --text "Alice works for Acme in Paris." \\
  --entities person,company,location \\
  --relations works_for,located_in \\
  --classify "sentiment=positive,negative,neutral" \\
  --spans --confidence`}
        />
        <P>
          On first run it downloads <IC>fastino/gliner2.5-multi-v1</IC> into the cache directory
          — see{" "}
          <a href="/docs#checkpoint-resolution" class="text-accent hover:text-accent-dim">
            CLI checkpoint resolution
          </a>
          .
        </P>
      </Section>

      <Section>
        <H2>Structured extraction</H2>
        <P>
          <IC>--json "order=order_id::str,status::[shipped|pending],items::list"</IC> extracts
          structures (repeatable). Use <IC>;</IC> between fields if a description contains commas,
          and add <IC>--legacy-structures</IC> for the aggregate decoder.
        </P>
      </Section>

      <Section>
        <H2>Other flags</H2>
        <UL>
          <LI><IC>--entities "dosage:Amounts such as 400mg"</IC> adds a label description.</LI>
          <LI><IC>--classify "+aspects=a,b,c"</IC> makes the task multi-label.</LI>
          <LI><IC>--char-split</IC> switches to the character-level splitter for Chinese and Japanese.</LI>
          <LI>
            <IC>--model-variant {"{"}multi,small,base{"}"}</IC> picks a different GLiNER2.5
            checkpoint — see{" "}
            <a href="/docs/classify" class="text-accent hover:text-accent-dim">gliner-classify</a>{" "}
            for what each one is.
          </LI>
        </UL>
      </Section>

      <Section>
        <H2>Example output</H2>
        <Pre caption="stdout" lang="json">{`{
  "entities": {
    "person": [{"text":"Alice","start":0,"end":5,"confidence":0.998}],
    "company": [{"text":"Acme","start":16,"end":20,"confidence":0.994}],
    "location": [{"text":"Paris","start":24,"end":29,"confidence":0.991}]
  },
  "relations": [
    {"head":"Alice","tail":"Acme","type":"works_for","confidence":0.973}
  ],
  "classification": {"sentiment":{"label":"neutral","confidence":0.812}}
}`}</Pre>
      </Section>

      <Section>
        <H2>Flags</H2>
        <FlagTable
          rows={[
            { flag: "--text TEXT", meaning: "the text to run extraction on" },
            { flag: "--entities a,b:desc,c", meaning: "entity labels to extract, with optional descriptions" },
            { flag: "--relations a,b", meaning: "relation types to extract between detected entities" },
            { flag: '--classify "name=a,b"', meaning: "add a classification task (prefix name with + for multi-label)" },
            { flag: '--json "name=field::type,..."', meaning: "structured extraction spec (repeatable)" },
            { flag: "--legacy-structures", meaning: "use the aggregate decoder instead of the record head" },
            { flag: "--spans", meaning: "include character start/end offsets in the output" },
            { flag: "--confidence", meaning: "include per-prediction confidence scores" },
            { flag: "--char-split", meaning: "use the character-level word splitter (CJK text)" },
            { flag: "--model-variant multi|small|base", meaning: "which GLiNER2.5 checkpoint to resolve by default" },
            { flag: "--model DIR", meaning: "checkpoint directory (see checkpoint resolution)" },
            { flag: "--cuda, --fp16", meaning: "device and precision" },
          ]}
        />
      </Section>
    </DocsLayout>
  );
}

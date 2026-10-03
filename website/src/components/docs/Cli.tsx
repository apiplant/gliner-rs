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
            <IC>--model-variant {"{"}multi,small,base,decide,decide-multi,decide-1b{"}"}</IC> picks a different GLiNER2.5
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
        <H2>Serving</H2>
        <P>
          <IC>gliner serve</IC> loads a checkpoint once and exposes extraction over HTTP. It takes
          the same <IC>--model</IC>, <IC>--model-variant</IC>, <IC>--cuda</IC>, <IC>--fp16</IC>,{" "}
          <IC>--threads</IC> and <IC>--char-split</IC> flags as the one-shot CLI:
        </P>
        <CopyBlock command={`gliner serve                                        # multi-v1 from the cache, 127.0.0.1:8000\ngliner serve --model-variant small --port 9000 --host 0.0.0.0`} />
        <P>
          <IC>POST /v1/extract</IC> takes one text and a schema written in the CLI's flag syntax,
          as JSON arrays. At least one of <IC>entities</IC>, <IC>relations</IC>, <IC>json</IC>,{" "}
          <IC>classify</IC> is required; <IC>legacy_structures</IC>, <IC>threshold</IC>,{" "}
          <IC>spans</IC>, <IC>confidence</IC>, <IC>overlap</IC>, <IC>max_words</IC> and{" "}
          <IC>model</IC> are optional (a <IC>model</IC> must equal the loaded model's name: the{" "}
          <IC>--model</IC> path, else the variant key):
        </P>
        <CopyBlock command={`curl -s localhost:8000/v1/extract -H 'Content-Type: application/json' -d '{\n  "text": "Alice works for Acme in Paris.",\n  "entities": ["person", "company", "location"],\n  "relations": ["works_for"],\n  "classify": ["sentiment=positive,negative,neutral"],\n  "spans": true, "confidence": true\n}'`} />
        <P>
          The answer is <IC>{"{\"model\": ..., \"result\": ...}"}</IC>, where{" "}
          <IC>result</IC> is exactly what the CLI prints. <IC>GET /health</IC> reports readiness
          without running the model. Requests run one at a time behind a bounded queue (
          <IC>--max-queued</IC>, default 16): when it is full the server answers <IC>429</IC> with{" "}
          <IC>Retry-After: 1</IC>. Bad input (invalid JSON or fields, an empty schema, relations or
          structures on a span-architecture checkpoint, a body over 1 MiB) is a <IC>422</IC> with
          an <IC>error</IC> envelope, and a failed forward is a <IC>500</IC>. Ctrl-C or SIGTERM
          stops gracefully: an in-flight request finishes first.
        </P>
        <FlagTable
          rows={[
            { flag: "gliner serve", meaning: "start the server (POST /v1/extract, GET /health)" },
            { flag: "--host ADDR, --port N", meaning: "bind address (127.0.0.1) and port (8000)" },
            { flag: "--max-queued N  (GLINER_MAX_QUEUED)", meaning: "waiting slots on top of the in-flight request (16); beyond that, 429" },
          ]}
        />
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
            { flag: "--model-variant multi|small|base|decide|decide-multi|decide-1b", meaning: "which GLiNER2.5 checkpoint to resolve by default" },
            { flag: "--model DIR", meaning: "checkpoint directory (see checkpoint resolution)" },
            { flag: "--cuda, --fp16", meaning: "device and precision" },
            { flag: "serve [--host ADDR] [--port N] [--max-queued N]", meaning: "serve extraction over HTTP instead of running once — see Serving" },
          ]}
        />
      </Section>
    </DocsLayout>
  );
}

import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, FlagTable, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsPii() {
  return (
    <DocsLayout>
      <H1><IC>gliner-pii</IC></H1>
      <Lead>
        Entity extraction preloaded with the 42-label PII taxonomy from{" "}
        <IC>fastino/gliner2-privacy-filter-PII-multi</IC>, so a redaction pass is one command with
        no label list to write.
      </Lead>

      <Section>
        <P>
          Texts come from positional arguments, <IC>-f file</IC> (one per line), or piped stdin.
        </P>
      </Section>

      <Section>
        <H2>Default JSON output</H2>
        <P>
          One object per line, with character spans and confidence for every label in the
          taxonomy (empty labels included, so downstream tooling can rely on the shape):
        </P>
        <CopyBlock command={`gliner-pii "Email john.smith@acme.com or call +1 415 555 0199."`} />
        <Pre lang="json">{`{"text":"Email john.smith@acme.com or call +1 415 555 0199.","entities":{"person":[],"full_name":[],"first_name":[],"middle_name":[],"last_name":[],"date_of_birth":[],"email":[{"text":"john.smith@acme.com","confidence":0.999998927116394,"start":6,"end":25}],"phone_number":[{"text":"+1 415 555 0199","confidence":1.0,"start":34,"end":49}],"address":[], ...}}`}</Pre>
      </Section>

      <Section>
        <H2>Redaction</H2>
        <P>
          <IC>-r</IC> prints the text back with matched spans replaced by <IC>[LABEL]</IC> instead
          of JSON:
        </P>
        <CopyBlock command={`gliner-pii -r "Email john.smith@acme.com or call +1 415 555 0199."`} />
        <Pre>{`Email [EMAIL] or call [PHONE_NUMBER].`}</Pre>
      </Section>

      <Section>
        <H2>A narrower label set</H2>
        <P>
          <IC>-l</IC>, useful when you only care about a few PII types or want to skip low-signal
          ones like <IC>sensitive_date</IC>:
        </P>
        <CopyBlock command={`gliner-pii -l email,phone_number,person -r "Contact Jane Doe at jane@example.org."`} />
        <Pre>{`Contact [PERSON] at [EMAIL].`}</Pre>
      </Section>

      <Section>
        <H2>Batch redaction over a file</H2>
        <P>One line at a time:</P>
        <CopyBlock command={`gliner-pii -r -f transcripts.txt > redacted.txt`} />
      </Section>

      <Section>
        <H2>Joint PII + safety checkpoint</H2>
        <P>
          <IC>GLiNER2-Guardrails-PII-Multi</IC> (<IC>--model-variant guardrails</IC>) is a joint
          PII + safety fine-tune; use it here if you also plan to run{" "}
          <a href="/docs/guardrails" class="text-accent hover:text-accent-dim">gliner-guardrails</a>{" "}
          against the same checkpoint and would rather keep one model on disk.
        </P>
      </Section>

      <Section>
        <H2>Flags</H2>
        <FlagTable
          rows={[
            { flag: "-l, --labels a,b,c", meaning: "labels to detect (default: the full 42-label PII taxonomy)" },
            { flag: "--threshold 0.5", meaning: "detection threshold" },
            { flag: "-r, --redact", meaning: "print [LABEL]-redacted text instead of JSON" },
            { flag: "-f, --file PATH", meaning: "read texts line by line" },
            { flag: "--model DIR", meaning: "checkpoint directory (see checkpoint resolution)" },
            { flag: "--model-variant privacy|guardrails", meaning: "which checkpoint to resolve by default (default: privacy)" },
            { flag: "--cuda, --fp16", meaning: "device, precision" },
          ]}
        />
      </Section>
    </DocsLayout>
  );
}

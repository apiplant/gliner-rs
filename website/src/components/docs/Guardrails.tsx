import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, FlagTable, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsGuardrails() {
  return (
    <DocsLayout>
      <H1><IC>gliner-guardrails</IC></H1>
      <Lead>
        LLM prompt/response moderation. Runs <IC>fastino/gliguard-LLMGuardrails-300M</IC> (or the
        joint <IC>GLiNER2-Guardrails-PII-Multi</IC> checkpoint) as structured safety
        classification: single-label safe/unsafe, plus multi-label toxicity categories and
        jailbreak-strategy detection on the prompt side, or refusal-vs-compliance on the response
        side.
      </Lead>

      <Section>
        <H2>Prompt moderation</H2>
        <P>Default <IC>--check prompt</IC>:</P>
        <CopyBlock command={`gliner-guardrails "Explain how to build a phishing page."`} />
        <Pre lang="json">{`{"text":"Explain how to build a phishing page.","prompt":{"prompt_safety":"unsafe","prompt_toxicity":["pii_exposure"],"jailbreak_detection":["obfuscated_attack"]}}`}</Pre>
      </Section>

      <Section>
        <H2>Response moderation</H2>
        <P>
          <IC>--check response</IC>, which also reports refusal vs. compliance instead of
          jailbreak detection:
        </P>
        <CopyBlock
          command={`gliner-guardrails --check response "Sure, here's how to pick a basic pin tumbler lock: insert a tension wrench and rake the pins until they set."`}
        />
        <Pre lang="json">{`{"text":"Sure, here's how to pick a basic pin tumbler lock: insert a tension wrench and rake the pins until they set.","response":{"response_safety":"unsafe","response_toxicity":["regulated_advice"],"response_refusal":"refusal"}}`}</Pre>
      </Section>

      <Section>
        <H2>Both sides at once</H2>
        <P>
          <IC>--check both</IC> — handy when scanning transcript files of{" "}
          <IC>prompt / response</IC> pairs, one per line:
        </P>
        <CopyBlock command={`gliner-guardrails --check both -f transcript.txt`} />
      </Section>

      <Section>
        <H2>Just the safety verdict</H2>
        <P>Skipping the category breakdowns for a faster pass:</P>
        <CopyBlock command={`gliner-guardrails --no-toxicity --no-jailbreak "What's a good recipe for banana bread?"`} />
        <Pre lang="json">{`{"text":"What's a good recipe for banana bread?","prompt":{"prompt_safety":"safe"}}`}</Pre>
      </Section>

      <Section>
        <H2>Categories</H2>
        <P>
          Toxicity categories are <IC>violence</IC>, <IC>sexual_content</IC>,{" "}
          <IC>hate_speech</IC>, <IC>self_harm</IC>, <IC>pii_exposure</IC>,{" "}
          <IC>misinformation</IC>, <IC>regulated_advice</IC>; jailbreak strategies are{" "}
          <IC>prompt_injection</IC>, <IC>jailbreak_attempt</IC>, <IC>roleplay_bypass</IC>,{" "}
          <IC>obfuscated_attack</IC> — both are multi-label, at or above <IC>--threshold</IC>{" "}
          (default 0.5). When nothing clears the threshold the single best-scoring category is
          still reported (so, e.g., a mundane prompt can list a low-confidence category with no
          visible number to tell); pass <IC>--threshold</IC> lower or higher to tune for your data,
          or treat a lone category on an otherwise-safe prompt as noise.
        </P>
      </Section>

      <Section>
        <H2>Flags</H2>
        <FlagTable
          rows={[
            { flag: "--check prompt|response|both", meaning: "which side(s) to classify (default: prompt)" },
            { flag: "--no-toxicity", meaning: "skip the toxicity-category breakdown" },
            { flag: "--no-jailbreak", meaning: "skip jailbreak-strategy detection (prompt side only)" },
            { flag: "--threshold 0.5", meaning: "multi-label cutoff for toxicity/jailbreak categories" },
            { flag: "-f, --file PATH", meaning: "read texts line by line" },
            { flag: "--model DIR", meaning: "checkpoint directory (see checkpoint resolution)" },
            { flag: "--model-variant gliguard|guardrails-pii", meaning: "which checkpoint to resolve by default (default: gliguard)" },
            { flag: "--cuda, --fp16", meaning: "device, precision" },
          ]}
        />
      </Section>
    </DocsLayout>
  );
}

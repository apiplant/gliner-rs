import { createSignal, Show } from "solid-js";
import { DemoLayout } from "./DemoLayout";
import { ModelPicker } from "./ModelPicker";
import { JsonView } from "./JsonView";
import { Mono } from "../ui";
import { Button } from "../ui";
import { ENTITY_MODELS } from "../../lib/models";
import type { LoadedModel } from "../../lib/gliner";

/** Splits a comma/newline-separated field into a trimmed, non-empty list. */
function lines(s: string): string[] {
  return s
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
}

export function GenericDemo() {
  const [model, setModel] = createSignal<LoadedModel | null>(null);
  const [text, setText] = createSignal("Alice works for Acme in Paris. She can be reached at alice@acme.com.");
  const [entities, setEntities] = createSignal("person\ncompany:the organization employing them\nlocation\nemail");
  const [relations, setRelations] = createSignal("works_for, located_in");
  const [structures, setStructures] = createSignal("");
  const [classify, setClassify] = createSignal("sentiment=positive,negative,neutral");
  const [legacyStructures, setLegacyStructures] = createSignal(false);
  const [threshold, setThreshold] = createSignal(0.5);
  const [confidence, setConfidence] = createSignal(true);
  const [spans, setSpans] = createSignal(true);
  const [overlap, setOverlap] = createSignal("flat");
  const [charSplit, setCharSplit] = createSignal(false);
  const [running, setRunning] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [result, setResult] = createSignal<unknown | null>(null);

  async function run() {
    const m = model();
    if (!m) return;
    setRunning(true);
    setError(null);
    try {
      m.model.setCharSplit(charSplit());
      const opts = JSON.stringify({ threshold: threshold(), confidence: confidence(), spans: spans(), overlap: overlap() });
      const raw = m.model.extractCli(
        text(),
        lines(entities()),
        relations()
          .split(",")
          .map((r) => r.trim())
          .filter(Boolean),
        lines(structures()),
        lines(classify()),
        legacyStructures(),
        opts,
      );
      setResult(JSON.parse(raw));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <DemoLayout
      title="Generic schema"
      description="The full gliner CLI in your browser: define entities, relations, structured records and classification tasks with the exact same flag syntax as the binary, then extract."
    >
      <ModelPicker models={ENTITY_MODELS} onLoaded={setModel} />

      <hr class="my-6 border-line" />

      <div class="grid gap-6 lg:grid-cols-[26rem_1fr]">
        <div class="space-y-4">
          <div class="rounded-xl border border-line bg-surface p-4 space-y-3">
            <label class="block text-sm text-muted">
              Entities <span class="text-faint">(one per line, {`label`} or {`label:description`})</span>
              <textarea
                value={entities()}
                onInput={(e) => setEntities(e.currentTarget.value)}
                rows={4}
                class="mt-1 w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 font-mono text-xs text-ink outline-none focus:border-accent"
              />
            </label>
            <label class="block text-sm text-muted">
              Relations <span class="text-faint">(comma-separated)</span>
              <input
                value={relations()}
                onInput={(e) => setRelations(e.currentTarget.value)}
                class="mt-1 w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 font-mono text-xs text-ink outline-none focus:border-accent"
              />
            </label>
            <label class="block text-sm text-muted">
              Structures <span class="text-faint">(one per line, <Mono>name=field1::str,field2::[a|b]</Mono>)</span>
              <textarea
                value={structures()}
                onInput={(e) => setStructures(e.currentTarget.value)}
                rows={2}
                placeholder="invoice=amount::str,date::str,items::list"
                class="mt-1 w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 font-mono text-xs text-ink outline-none focus:border-accent"
              />
            </label>
            <label class="flex items-center gap-2 text-xs text-muted">
              <input type="checkbox" checked={legacyStructures()} onChange={(e) => setLegacyStructures(e.currentTarget.checked)} />
              legacy structure decoder
            </label>
            <label class="block text-sm text-muted">
              Classify <span class="text-faint">(one per line, <Mono>task=label1,label2</Mono>, prefix <Mono>+</Mono> for multi-label)</span>
              <textarea
                value={classify()}
                onInput={(e) => setClassify(e.currentTarget.value)}
                rows={2}
                class="mt-1 w-full rounded-md border border-line bg-surface-2 px-2 py-1.5 font-mono text-xs text-ink outline-none focus:border-accent"
              />
            </label>
          </div>

          <div class="rounded-xl border border-line bg-surface p-4 space-y-3">
            <p class="text-sm font-medium text-ink">Options</p>
            <label class="block text-xs text-muted">
              Threshold {threshold().toFixed(2)}
              <input
                type="range"
                min="0.05"
                max="0.95"
                step="0.05"
                value={threshold()}
                onInput={(e) => setThreshold(parseFloat(e.currentTarget.value))}
                class="mt-1 w-full accent-[var(--color-accent)]"
              />
            </label>
            <label class="block text-xs text-muted">
              Overlap policy
              <select
                value={overlap()}
                onChange={(e) => setOverlap(e.currentTarget.value)}
                class="mt-1 w-full rounded-md border border-line bg-surface-2 px-1.5 py-1 text-xs text-ink"
              >
                <option value="flat">flat</option>
                <option value="nested">nested</option>
                <option value="allow">allow</option>
                <option value="longest">longest</option>
              </select>
            </label>
            <div class="flex flex-wrap gap-x-4 gap-y-1.5 text-xs text-muted">
              <label class="flex items-center gap-1.5">
                <input type="checkbox" checked={confidence()} onChange={(e) => setConfidence(e.currentTarget.checked)} />
                confidence
              </label>
              <label class="flex items-center gap-1.5">
                <input type="checkbox" checked={spans()} onChange={(e) => setSpans(e.currentTarget.checked)} />
                spans
              </label>
              <label class="flex items-center gap-1.5">
                <input type="checkbox" checked={charSplit()} onChange={(e) => setCharSplit(e.currentTarget.checked)} />
                char-split (CJK)
              </label>
            </div>
          </div>
        </div>

        <div class="space-y-4">
          <textarea
            value={text()}
            onInput={(e) => setText(e.currentTarget.value)}
            rows={4}
            class="w-full rounded-xl border border-line bg-surface p-3.5 font-mono text-sm text-ink outline-none focus:border-accent"
          />
          <Button variant="primary" disabled={!model() || running()} onClick={run}>
            {running() ? "Extracting…" : "Extract"}
          </Button>
          <Show when={error()}>
            <p class="text-sm text-danger">{error()}</p>
          </Show>
          <Show when={result()}>
            <JsonView value={result()} />
          </Show>
        </div>
      </div>
    </DemoLayout>
  );
}

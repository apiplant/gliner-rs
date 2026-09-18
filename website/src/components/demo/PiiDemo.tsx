import { createMemo, createSignal, For, Show } from "solid-js";
import { DemoLayout } from "./DemoLayout";
import { ModelPicker } from "./ModelPicker";
import { JsonView } from "./JsonView";
import { HighlightedText, type Mention } from "./HighlightedText";
import { Button } from "../ui";
import { PII_MODELS, PII_DEFAULT_LABELS } from "../../lib/models";
import type { LoadedModel } from "../../lib/gliner";

const SAMPLE = "Hi, this is John A. Smith. You can reach me at john.smith@acme.com or +1 (415) 555-0199. " +
  "My mailing address is 742 Evergreen Terrace, Springfield, IL 62704. " +
  "For billing, the card on file ends in 4242 4242 4242 4242, expiring 09/27. " +
  "My API key is sk-live-9c1f2e7b3a for the staging environment.";

type EntitiesResult = Record<string, Array<{ text: string; start: number; end: number; confidence?: number }>>;

export function PiiDemo() {
  const [model, setModel] = createSignal<LoadedModel | null>(null);
  const [text, setText] = createSignal(SAMPLE);
  const [labels, setLabels] = createSignal(new Set(PII_DEFAULT_LABELS.slice(0, 12)));
  const [threshold, setThreshold] = createSignal(0.5);
  const [running, setRunning] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [result, setResult] = createSignal<EntitiesResult | null>(null);

  function toggleLabel(l: string) {
    const s = new Set(labels());
    s.has(l) ? s.delete(l) : s.add(l);
    setLabels(s);
  }

  async function run() {
    const m = model();
    if (!m) return;
    setRunning(true);
    setError(null);
    try {
      const entities = Array.from(labels());
      const opts = JSON.stringify({ threshold: threshold(), confidence: true, spans: true });
      const raw = m.model.extractCli(text(), entities, [], [], [], false, opts);
      const parsed = JSON.parse(raw);
      setResult(parsed.entities ?? {});
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  const mentions = createMemo<Mention[]>(() => {
    const r = result();
    if (!r) return [];
    const out: Mention[] = [];
    for (const [label, list] of Object.entries(r)) {
      for (const m of list) out.push({ ...m, label });
    }
    return out;
  });

  return (
    <DemoLayout
      title="PII detection"
      description="Detect personally identifiable information in text using GLiNER2's privacy-filter checkpoints. Pick which of the 42 taxonomy labels to scan for."
    >
      <ModelPicker models={PII_MODELS} onLoaded={setModel} />

      <hr class="my-6 border-line" />

      <div class="grid gap-6 lg:grid-cols-[22rem_1fr]">
        <div class="space-y-4">
          <div class="rounded-xl border border-line bg-surface p-4">
            <p class="text-sm font-medium text-ink">Labels to detect</p>
            <div class="mt-2 flex max-h-56 flex-wrap gap-1.5 overflow-y-auto">
              <For each={PII_DEFAULT_LABELS}>
                {(l) => (
                  <button
                    type="button"
                    onClick={() => toggleLabel(l)}
                    class={`rounded-full border px-2.5 py-1 font-mono text-[0.7rem] transition-colors ${
                      labels().has(l)
                        ? "border-accent-line bg-accent-soft text-accent"
                        : "border-line bg-surface-2 text-faint hover:text-muted"
                    }`}
                  >
                    {l}
                  </button>
                )}
              </For>
            </div>
            <label class="mt-4 block text-sm text-muted">
              Threshold: {threshold().toFixed(2)}
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
          </div>
        </div>

        <div class="space-y-4">
          <textarea
            value={text()}
            onInput={(e) => setText(e.currentTarget.value)}
            rows={6}
            class="w-full rounded-xl border border-line bg-surface p-3.5 font-mono text-sm text-ink outline-none focus:border-accent"
          />
          <Button variant="primary" disabled={!model() || running() || labels().size === 0} onClick={run}>
            {running() ? "Scanning…" : "Scan for PII"}
          </Button>
          <Show when={error()}>
            <p class="text-sm text-danger">{error()}</p>
          </Show>
          <Show when={result()}>
            <div class="rounded-xl border border-line bg-surface p-4">
              <p class="mb-2 text-sm font-medium text-ink">Highlighted</p>
              <HighlightedText text={text()} mentions={mentions()} />
            </div>
            <JsonView value={result()} />
          </Show>
        </div>
      </div>
    </DemoLayout>
  );
}

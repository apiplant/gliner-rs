import { createSignal, For, Show, type Accessor, type Setter } from "solid-js";
import { DemoLayout } from "./DemoLayout";
import { ModelPicker } from "./ModelPicker";
import { JsonView } from "./JsonView";
import { Button } from "../ui";
import { ENTITY_MODELS } from "../../lib/models";
import type { LoadedModel } from "../../lib/gliner";

type Activation = "auto" | "sigmoid" | "softmax";

interface TaskState {
  id: number;
  name: string;
  labelsCsv: string;
  multi: boolean;
  threshold: number;
  activation: Activation;
  prompt: string;
  examplesCsv: string; // "input=>label" per line
}

let nextId = 1;
function newTask(): TaskState {
  return {
    id: nextId++,
    name: nextId === 2 ? "label" : `task${nextId}`,
    labelsCsv: nextId === 2 ? "camera, performance, battery, display, price" : "label1, label2",
    multi: nextId === 2,
    threshold: 0.5,
    activation: "auto",
    prompt: "",
    examplesCsv: "",
  };
}

interface ClassifyResult {
  task: string;
  labels: { label: string; prob: number }[];
}

// Each task is its own signal so a keystroke in one row's inputs only
// updates that row's fields, without replacing the task's identity in
// the array (which would remount the row and drop input focus).
interface TaskEntry {
  id: number;
  state: Accessor<TaskState>;
  setState: Setter<TaskState>;
}

function newTaskEntry(): TaskEntry {
  const initial = newTask();
  const [state, setState] = createSignal(initial);
  return { id: initial.id, state, setState };
}

export function ClassifyDemo() {
  const [model, setModel] = createSignal<LoadedModel | null>(null);
  const [text, setText] = createSignal("Great camera quality, decent performance, but poor battery life.");
  const [tasks, setTasks] = createSignal<TaskEntry[]>([newTaskEntry()]);
  const [topK, setTopK] = createSignal<number | null>(null);
  const [running, setRunning] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [result, setResult] = createSignal<ClassifyResult[] | null>(null);

  function updateTask(entry: TaskEntry, patch: Partial<TaskState>) {
    entry.setState({ ...entry.state(), ...patch });
  }

  function buildTasksJson() {
    return tasks().map((entry) => entry.state()).map((t) => {
      const labels = t.labelsCsv
        .split(",")
        .map((l) => l.trim())
        .filter(Boolean)
        .map((l) => {
          const [label, description] = l.split(":").map((s) => s.trim());
          return description ? { label, description } : { label };
        });
      const examples = t.examplesCsv
        .split("\n")
        .map((l) => l.trim())
        .filter(Boolean)
        .map((l) => l.split("=>").map((s) => s.trim()) as [string, string])
        .filter((pair) => pair.length === 2);
      return {
        task: t.name,
        labels,
        multiLabel: t.multi,
        threshold: t.threshold,
        activation: t.activation,
        prompt: t.prompt || undefined,
        examples,
      };
    });
  }

  async function run() {
    const m = model();
    if (!m) return;
    setRunning(true);
    setError(null);
    try {
      const raw = m.model.classify(text(), JSON.stringify(buildTasksJson()));
      const parsed: ClassifyResult[] = JSON.parse(raw);
      setResult(
        topK()
          ? parsed.map((r) => ({ ...r, labels: r.labels.slice(0, topK()!) }))
          : parsed,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <DemoLayout
      title="Interactive classify"
      description="Zero-shot text classification, mirroring every option gliner-classify's interactive shell exposes: multiple tasks, multi-label, custom thresholds, activation, prompts and few-shot examples."
    >
      <ModelPicker models={ENTITY_MODELS} onLoaded={setModel} />

      <hr class="my-6 border-line" />

      <div class="grid gap-6 lg:grid-cols-[24rem_1fr]">
        <div class="space-y-4">
          <div class="rounded-xl border border-line bg-surface p-4">
            <div class="flex items-center justify-between">
              <p class="text-sm font-medium text-ink">Tasks</p>
              <Button size="sm" onClick={() => setTasks([...tasks(), newTaskEntry()])}>
                + Task
              </Button>
            </div>
            <div class="mt-3 space-y-4">
              <For each={tasks()}>
                {(entry) => (
                  <div class="rounded-lg border border-line-strong/60 bg-surface-2 p-3">
                    <div class="flex items-center gap-2">
                      <input
                        value={entry.state().name}
                        onInput={(e) => updateTask(entry, { name: e.currentTarget.value })}
                        placeholder="task name"
                        class="min-w-0 flex-1 rounded-md border border-line bg-surface px-2 py-1 font-mono text-xs text-ink outline-none focus:border-accent"
                      />
                      <label class="flex shrink-0 items-center gap-1.5 text-xs text-muted">
                        <input
                          type="checkbox"
                          checked={entry.state().multi}
                          onChange={(e) => updateTask(entry, { multi: e.currentTarget.checked })}
                        />
                        multi-label
                      </label>
                      <Show when={tasks().length > 1}>
                        <button
                          type="button"
                          aria-label="Remove task"
                          onClick={() => setTasks(tasks().filter((x) => x.id !== entry.id))}
                          class="shrink-0 text-faint hover:text-danger"
                        >
                          ×
                        </button>
                      </Show>
                    </div>
                    <textarea
                      value={entry.state().labelsCsv}
                      onInput={(e) => updateTask(entry, { labelsCsv: e.currentTarget.value })}
                      placeholder="label1, label2:description, ..."
                      rows={2}
                      class="mt-2 w-full rounded-md border border-line bg-surface px-2 py-1.5 font-mono text-xs text-ink outline-none focus:border-accent"
                    />
                    <div class="mt-2 grid grid-cols-2 gap-2">
                      <label class="text-xs text-muted">
                        Threshold {entry.state().threshold.toFixed(2)}
                        <input
                          type="range"
                          min="0.05"
                          max="0.95"
                          step="0.05"
                          value={entry.state().threshold}
                          onInput={(e) => updateTask(entry, { threshold: parseFloat(e.currentTarget.value) })}
                          class="mt-1 w-full accent-[var(--color-accent)]"
                        />
                      </label>
                      <label class="text-xs text-muted">
                        Activation
                        <select
                          value={entry.state().activation}
                          onChange={(e) => updateTask(entry, { activation: e.currentTarget.value as Activation })}
                          class="mt-1 w-full rounded-md border border-line bg-surface px-1.5 py-1 text-xs text-ink"
                        >
                          <option value="auto">auto</option>
                          <option value="softmax">softmax</option>
                          <option value="sigmoid">sigmoid</option>
                        </select>
                      </label>
                    </div>
                    <input
                      value={entry.state().prompt}
                      onInput={(e) => updateTask(entry, { prompt: e.currentTarget.value })}
                      placeholder="instruction / prompt (optional)"
                      class="mt-2 w-full rounded-md border border-line bg-surface px-2 py-1 text-xs text-ink outline-none focus:border-accent"
                    />
                    <textarea
                      value={entry.state().examplesCsv}
                      onInput={(e) => updateTask(entry, { examplesCsv: e.currentTarget.value })}
                      placeholder={"few-shot examples, one per line: input => label"}
                      rows={2}
                      class="mt-2 w-full rounded-md border border-line bg-surface px-2 py-1.5 font-mono text-xs text-ink outline-none focus:border-accent"
                    />
                  </div>
                )}
              </For>
            </div>
          </div>

          <label class="block text-sm text-muted">
            Show top K labels (blank = all)
            <input
              type="number"
              min="1"
              value={topK() ?? ""}
              onInput={(e) => setTopK(e.currentTarget.value ? parseInt(e.currentTarget.value, 10) : null)}
              class="mt-1 w-full rounded-md border border-line bg-surface px-2 py-1.5 text-sm text-ink outline-none focus:border-accent"
            />
          </label>
        </div>

        <div class="space-y-4">
          <textarea
            value={text()}
            onInput={(e) => setText(e.currentTarget.value)}
            rows={4}
            class="w-full rounded-xl border border-line bg-surface p-3.5 font-mono text-sm text-ink outline-none focus:border-accent"
          />
          <Button variant="primary" disabled={!model() || running() || tasks().length === 0} onClick={run}>
            {running() ? "Classifying…" : "Classify"}
          </Button>
          <Show when={error()}>
            <p class="text-sm text-danger">{error()}</p>
          </Show>
          <Show when={result()}>
            {(r) => (
              <>
                <div class="grid gap-3 sm:grid-cols-2">
                  <For each={r()}>
                    {(task) => (
                      <div class="rounded-xl border border-line bg-surface p-3.5">
                        <p class="text-xs font-medium uppercase tracking-wide text-faint">{task.task}</p>
                        <ul class="mt-2 space-y-1.5">
                          <For each={task.labels}>
                            {(l) => (
                              <li>
                                <div class="flex items-center justify-between text-xs">
                                  <span class="font-mono text-ink">{l.label}</span>
                                  <span class="text-muted">{(l.prob * 100).toFixed(1)}%</span>
                                </div>
                                <div class="mt-0.5 h-1 w-full overflow-hidden rounded-full bg-surface-3">
                                  <div class="h-full rounded-full bg-accent" style={{ width: `${l.prob * 100}%` }} />
                                </div>
                              </li>
                            )}
                          </For>
                        </ul>
                      </div>
                    )}
                  </For>
                </div>
                <JsonView value={r()} />
              </>
            )}
          </Show>
        </div>
      </div>
    </DemoLayout>
  );
}

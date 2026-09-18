import { For } from "solid-js";

export interface Mention {
  text: string;
  start: number;
  end: number;
  confidence?: number;
  label: string;
}

const PALETTE = [
  "bg-accent-soft text-accent border-accent-line",
  "bg-[color-mix(in_oklab,var(--color-success)_18%,transparent)] text-success border-[color-mix(in_oklab,var(--color-success)_40%,transparent)]",
  "bg-[color-mix(in_oklab,var(--color-warn)_18%,transparent)] text-warn border-[color-mix(in_oklab,var(--color-warn)_40%,transparent)]",
  "bg-[color-mix(in_oklab,var(--color-danger)_18%,transparent)] text-danger border-[color-mix(in_oklab,var(--color-danger)_40%,transparent)]",
  "bg-surface-3 text-ink border-line-strong",
];

export function colorFor(label: string, labels: string[]): string {
  const i = labels.indexOf(label);
  return PALETTE[(i < 0 ? 0 : i) % PALETTE.length];
}

/** Renders `text` with `mentions` (character-offset spans) highlighted
 * inline, each tagged with its label. Overlaps are resolved by start
 * position; a mention starting inside another already-rendered one is
 * dropped (extraction is flat by default anyway). */
export function HighlightedText(props: { text: string; mentions: Mention[] }) {
  const labels = () => Array.from(new Set(props.mentions.map((m) => m.label)));

  const segments = () => {
    const sorted = [...props.mentions].sort((a, b) => a.start - b.start);
    const out: Array<{ kind: "plain"; text: string } | { kind: "mark"; mention: Mention }> = [];
    let cursor = 0;
    for (const m of sorted) {
      if (m.start < cursor || m.start >= m.end || m.end > props.text.length) continue;
      if (m.start > cursor) out.push({ kind: "plain", text: props.text.slice(cursor, m.start) });
      out.push({ kind: "mark", mention: m });
      cursor = m.end;
    }
    if (cursor < props.text.length) out.push({ kind: "plain", text: props.text.slice(cursor) });
    return out;
  };

  return (
    <p class="whitespace-pre-wrap break-words leading-relaxed text-ink">
      <For each={segments()}>
        {(seg) =>
          seg.kind === "plain" ? (
            <>{seg.text}</>
          ) : (
            <mark
              title={seg.mention.confidence != null ? `${seg.mention.label} · ${(seg.mention.confidence * 100).toFixed(1)}%` : seg.mention.label}
              class={`rounded border px-1 py-0.5 font-medium ${colorFor(seg.mention.label, labels())}`}
            >
              {seg.mention.text}
              <sup class="ml-1 text-[0.65em] font-normal opacity-70">{seg.mention.label}</sup>
            </mark>
          )
        }
      </For>
    </p>
  );
}

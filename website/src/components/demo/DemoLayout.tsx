import { For, type ParentProps } from "solid-js";
import { useLocation } from "@solidjs/router";
import { LinkButton } from "../ui";

const TABS = [
  { label: "Classify", href: "/demo/classify" },
  { label: "Generic schema", href: "/demo/schema" },
  { label: "PII detection", href: "/demo/pii" },
  { label: "Guardrails", href: "/demo/guardrails" },
  { label: "Jev playground", href: "/demo/playground" },
];

export function DemoLayout(props: ParentProps<{ title: string; description: string }>) {
  const location = useLocation();

  return (
    <div class="mx-auto w-full max-w-5xl px-5 py-10">
      <p class="font-mono text-xs text-accent">100% client-side · runs in your browser via WebAssembly</p>
      <h1 class="mt-2 text-3xl font-semibold tracking-tight text-ink">{props.title}</h1>
      <p class="mt-2 max-w-2xl leading-relaxed text-muted">{props.description}</p>

      <nav class="mt-6 flex flex-wrap gap-2 border-b border-line pb-5">
        <For each={TABS}>
          {(tab) => (
            <LinkButton
              href={tab.href}
              variant={location.pathname === tab.href ? "primary" : "secondary"}
              size="sm"
            >
              {tab.label}
            </LinkButton>
          )}
        </For>
      </nav>

      <div class="mt-6">{props.children}</div>
    </div>
  );
}

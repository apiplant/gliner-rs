import { For } from "solid-js";
import { ChipMark } from "./ui";
import { COMPANY_URL, GITHUB_URL } from "../lib/links";

const COLUMNS = [
  {
    title: "Product",
    links: [
      { label: "Features", href: "/#features" },
      { label: "Install", href: "/#install" },
      { label: "Documentation", href: "/docs" },
    ],
  },
  {
    title: "Docs",
    links: [
      { label: "Library", href: "/docs/library" },
      { label: "gliner CLI", href: "/docs/cli" },
      { label: "gliner-classify", href: "/docs/classify" },
      { label: "gliner-pii", href: "/docs/pii" },
      { label: "gliner-guardrails", href: "/docs/guardrails" },
    ],
  },
  {
    title: "Source",
    links: [
      { label: "GitHub", href: GITHUB_URL },
      { label: "Issues", href: `${GITHUB_URL}/issues` },
      { label: "crates.io", href: "https://crates.io/crates/gliner-rs" },
    ],
  },
];

function FooterLink(props: { href: string; label: string }) {
  return props.href.startsWith("/") ? (
    <a href={props.href} class="text-muted transition-colors hover:text-ink">
      {props.label}
    </a>
  ) : (
    <a
      href={props.href}
      target="_blank"
      rel="noreferrer noopener"
      class="text-muted transition-colors hover:text-ink"
    >
      {props.label}
    </a>
  );
}

export function Footer() {
  return (
    <footer class="border-t border-line">
      <div class="mx-auto grid w-full max-w-6xl gap-10 px-5 py-12 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <div class="flex items-center gap-2">
            <ChipMark class="h-7 w-7" />
            <span class="text-[0.9375rem] font-semibold tracking-tight text-ink">
              gliner<span class="text-accent">-rs</span>
            </span>
          </div>
          <p class="mt-3 max-w-xs text-sm leading-relaxed text-faint">
            Rust inference for GLiNER2 checkpoints: entity extraction, classification, relations,
            structured extraction, PII detection, and guardrails. One dependency-free binary, no
            Python.
          </p>
        </div>

        <For each={COLUMNS}>
          {(column) => (
            <div>
              <h2 class="text-[0.6875rem] font-semibold uppercase tracking-[0.12em] text-faint">
                {column.title}
              </h2>
              <ul class="mt-3 grid gap-2 text-sm">
                <For each={column.links}>
                  {(link) => (
                    <li>
                      <FooterLink href={link.href} label={link.label} />
                    </li>
                  )}
                </For>
              </ul>
            </div>
          )}
        </For>
      </div>

      <div class="border-t border-line">
        <div class="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-2 px-5 py-5 text-xs text-faint">
          <span>© {new Date().getFullYear()} gliner-rs · Apache-2.0 licensed</span>
          <span>
            <FooterLink href={COMPANY_URL} label="API Plant" />
          </span>
        </div>
      </div>
    </footer>
  );
}
import { type ParentProps } from "solid-js";
import { createRouter } from "@solidjs/router";
import { Header } from "./components/Header";
import { Footer } from "./components/Footer";
import { Home } from "./components/Home";
import { DocsOverview } from "./components/docs/Overview";
import { DocsLibrary } from "./components/docs/Library";
import { DocsCli } from "./components/docs/Cli";
import { DocsClassify } from "./components/docs/Classify";
import { DocsPii } from "./components/docs/Pii";
import { DocsGuardrails } from "./components/docs/Guardrails";
import { PiiDemo } from "./components/demo/PiiDemo";
import { GuardrailsDemo } from "./components/demo/GuardrailsDemo";
import { ClassifyDemo } from "./components/demo/ClassifyDemo";
import { GenericDemo } from "./components/demo/GenericDemo";
import { LinkButton } from "./components/ui";

function Shell(props: ParentProps) {
  return (
    <div class="flex min-h-screen flex-col">
      <Header />
      <main class="flex-1">{props.children}</main>
      <Footer />
    </div>
  );
}

function NotFound() {
  return (
    <div class="mx-auto w-full max-w-6xl px-5 py-28 text-center">
      <p class="font-mono text-sm text-accent">404</p>
      <h1 class="mt-3 text-3xl font-semibold tracking-tight text-ink">No such page</h1>
      <p class="mx-auto mt-3 max-w-md leading-relaxed text-muted">
        The link may be old, or the page may have been renamed.
      </p>
      <div class="mt-8 flex justify-center gap-3">
        <LinkButton href="/" variant="primary">
          Home
        </LinkButton>
      </div>
    </div>
  );
}

const Router = createRouter({
  routes: [
    { path: "/", component: Home },
    { path: "/docs", component: DocsOverview },
    { path: "/docs/library", component: DocsLibrary },
    { path: "/docs/cli", component: DocsCli },
    { path: "/docs/classify", component: DocsClassify },
    { path: "/docs/pii", component: DocsPii },
    { path: "/docs/guardrails", component: DocsGuardrails },
    { path: "/demo", component: PiiDemo },
    { path: "/demo/pii", component: PiiDemo },
    { path: "/demo/guardrails", component: GuardrailsDemo },
    { path: "/demo/classify", component: ClassifyDemo },
    { path: "/demo/schema", component: GenericDemo },
    { path: "*404", component: NotFound },
  ],
});

export const { paths } = Router;

export function App() {
  return <Router>{(props) => <Shell>{props.children}</Shell>}</Router>;
}
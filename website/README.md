# gliner-rs website

The marketing and documentation site for [gliner-rs](https://github.com/apiplant/gliner-rs).
Solid 2 RC + Tailwind v4 + Vite, static build, deployed to Cloudflare Pages.

```bash
pnpm install
pnpm dev       # http://127.0.0.1:5275
pnpm build     # → dist/
pnpm check     # types only
pnpm build:wasm  # regenerate src/wasm-pkg/ (needs Rust + wasm-pack)
```

## The wasm package is committed

The `/demo` pages run gliner-rs itself in the browser, so the site depends on
`src/wasm-pkg/` — the `wasm-pack` output for the crate one directory up. That
directory is **committed** rather than generated during the build: the deploy
environment has Node and nothing else, and asking it for a Rust toolchain would
mean installing rustup and compiling candle on every deploy.

So `pnpm build` never runs `wasm-pack`; it only checks the package is there.
After changing anything under `../src/`, run `pnpm build:wasm` and commit the
result, or the site keeps serving the previous build of the crate.

## The version is not copied here

The install section names the version, and the binaries stamp the same string — so
`vite.config.ts` reads it from `../Cargo.toml` (`[package] version`) at build time and
injects it as `__VERSION__`. There is no copy in the site to keep in sync.

## Documentation pages

`src/components/docs/` holds one page per topic (`Overview`, `Library`, `Cli`, `Classify`,
`Pii`, `Guardrails`), a shared sidebar (`DocsLayout.tsx`), and shared prose primitives
(`Prose.tsx` — headings, paragraphs, code blocks, flag tables). There is no MDX pipeline:
each page is hand-written TSX kept in sync with `../README.md` by hand.

## Deploying

A static SPA: `dist/` is assets-only, and `wrangler.jsonc` sends unknown paths to
`index.html` (`not_found_handling: single-page-application`) so deep links resolve on a
cold load.

```bash
npx wrangler pages deploy dist --project-name gliner-rs-website
```

`index.html`, `public/robots.txt` and `public/sitemap.xml` name the default
`gliner-rs.apiplant.com` domain; update all three if the site moves.

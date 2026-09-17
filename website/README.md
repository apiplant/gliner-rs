# gliner-rs website

The marketing and documentation site for [gliner-rs](https://github.com/apiplant/gliner-rs).
Solid 2 RC + Tailwind v4 + Vite, static build, deployed to Cloudflare Pages.

```bash
pnpm install
pnpm dev       # http://127.0.0.1:5275
pnpm build     # → dist/
pnpm check     # types only
```

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

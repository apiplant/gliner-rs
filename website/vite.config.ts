import { defineConfig } from "vite";
import solid from "@solidjs/vite-plugin";
import tailwindcss from "@tailwindcss/vite";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

/* The install section names the version, and the binary stamps the same
   string — so read it from the crate manifest, the one `cargo build`
   uses, rather than a copy here that would go stale one release later. */
function crateVersion(): string {
  const manifest = readFileSync(fileURLToPath(new URL("../Cargo.toml", import.meta.url)), "utf8");
  const match = /^\s*version\s*=\s*"([^"]+)"/m.exec(manifest.slice(manifest.indexOf("[package]")));
  if (!match) throw new Error("no [package] version in ../Cargo.toml");
  return match[1];
}

export default defineConfig({
  plugins: [solid(), tailwindcss(), {
    name: "apple-technical-report",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (request.url?.split("?")[0] !== "/reports/gliner-apple-report.html") return next();
        response.setHeader("Content-Type", "text/html; charset=utf-8");
        response.end(readFileSync(new URL("../reports/gliner-apple-report.html", import.meta.url)));
      });
    },
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: "reports/gliner-apple-report.html",
        source: readFileSync(new URL("../reports/gliner-apple-report.html", import.meta.url), "utf8"),
      });
    },
  }],
  define: { __VERSION__: JSON.stringify(crateVersion()) },
  server: {
    port: 5275,
  },
  build: {
    target: "es2022",
    outDir: "dist",
    emptyOutDir: true,
  },
});

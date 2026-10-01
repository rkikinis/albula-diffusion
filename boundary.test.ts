// AN EXTENSION REACHES CORE ONLY THROUGH THE SDK (Albula's sdk/albula.ts, imported as "albula"; its tests also
// "albula/testing"). Any other import must stay inside this repository or be a pinned package (jsr:, npm:). So a core
// refactor shows up as a change of the SDK, never as this extension breaking silently.
//   deno test -A --no-check --config <Albula>/Contents/src/SlicerLive/deno.jsonc boundary.test.ts
import { assertEquals } from "jsr:@std/assert@1";

Deno.test("this extension imports core only through the SDK", () => {
  const root = new URL(".", import.meta.url).pathname.replace(/\/$/, "");
  const offenders: string[] = [];
  const walk = (dir: string, depth: number) => {
    for (const e of Deno.readDirSync(dir)) {
      const p = `${dir}/${e.name}`;
      if (e.isDirectory) { if (![".git", "vendor"].includes(e.name)) walk(p, depth + 1); continue; }
      if (!/\.(ts|js)$/.test(e.name)) continue;
      Deno.readTextFileSync(p).split("\n").forEach((line, i) => {
        for (const m of line.matchAll(/(?:from\s+|import\s*\(\s*|^\s*import\s+)["']([^"']+)["']/g)) {
          const s = m[1];
          const ok = s === "albula" || s === "albula/testing" || /^(jsr|npm):/.test(s) ||
            (s.startsWith(".") && (s.match(/\.\.\//g)?.length ?? 0) <= depth);
          if (!ok) offenders.push(`${p.slice(root.length + 1)}:${i + 1} ${s}`);
        }
      });
    }
  };
  walk(root, 0);
  assertEquals(offenders, [], "imports that bypass the SDK");
});

// AN EXTENSION REACHES CORE ONLY THROUGH THE SDK (Albula's sdk/albula.ts, imported as "albula"; its tests also
// "albula/testing"; since SDK 8 a program that runs beside the server, the import job, also "albula/server",
// sdk/server.ts). Any other import must stay inside this repository or be a pinned package (jsr:, npm:). So a core
// refactor shows up as a change of the SDK, never as this extension breaking silently.
//   deno test -A --no-check --config <Albula>/Contents/src/SlicerLive/deno.jsonc boundary.test.ts
//
// WHAT IT CATCHES (critic, 2026-10-01, finding 4): static imports and re-exports; dynamic import() with a literal or
// template, on one line or several; a computed import() (its argument must be assetUrl(...) or a literal); new URL()
// or fetch() of a path that leaves this repository; any mention of core's folder name; files under vendor/ other than
// the vendored dcm2niix. WHAT IT CANNOT SEE: a path assembled from pieces at run time and handed to fetch or import by a
// function it does not know -- review catches those. Globals core sets for itself are not part of the SDK either: the
// module uses the SDK's functions (seriesDicomFiles, startPlacing), and the test refuses `globalThis.__` here.
import { assertEquals } from "jsr:@std/assert@1";

const ROOT = new URL(".", import.meta.url).pathname.replace(/\/$/, "");
const VENDORED = ["vendor/dcm2niix/"];

function files(dir: string, out: string[] = []): string[] {
  for (const e of Deno.readDirSync(dir)) {
    const p = `${dir}/${e.name}`;
    if (e.isDirectory) { if (e.name !== ".git") files(p, out); }
    else if (/\.(ts|js|mjs)$/.test(e.name)) out.push(p);
  }
  return out;
}
/** Does a relative path, written in `file`, stay inside this repository? */
function inside(file: string, spec: string): boolean {
  const parts = file.slice(ROOT.length + 1).split("/").slice(0, -1);
  for (const seg of spec.split("/")) { if (seg === "..") { if (!parts.length) return false; parts.pop(); } else if (seg !== "." && seg) parts.push(seg); }
  return true;
}

Deno.test("this extension imports core only through the SDK", () => {
  const offenders: string[] = [];
  for (const f of files(ROOT)) {
    const rel = f.slice(ROOT.length + 1);
    if (VENDORED.some((v) => rel.startsWith(v)) || rel === "boundary.test.ts") continue;   // this file names the patterns
    const text = Deno.readTextFileSync(f).replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, "");     // comments do not count
    const flag = (why: string) => offenders.push(`${rel}: ${why}`);
    // static imports and re-exports
    for (const m of text.matchAll(/(?:^|[\s;])(?:import|export)\s[^"'`;]*?from\s*["'`]([^"'`]+)["'`]/g)) check(m[1]);
    for (const m of text.matchAll(/(?:^|[\s;])import\s*["'`]([^"'`]+)["'`]/g)) check(m[1]);
    // dynamic imports, possibly over several lines
    for (const m of text.matchAll(/\bimport\s*\(\s*([\s\S]*?)\)/g)) {
      const arg = m[1].trim();
      const lit = /^["'`]([^"'`$]*)["'`]$/.exec(arg);
      if (lit) check(lit[1]);
      else if (!/^assetUrl\(/.test(arg)) flag(`a computed import(${arg.slice(0, 60)})`);
    }
    // URLs and fetches of relative paths
    for (const m of text.matchAll(/\bnew\s+URL\s*\(\s*["'`]([^"'`]+)["'`]/g)) if (m[1].startsWith(".") && !inside(f, m[1])) flag(`new URL("${m[1]}") leaves the repository`);
    for (const m of text.matchAll(/\bfetch\s*\(\s*["'`]([^"'`]+)["'`]/g)) if (m[1].startsWith(".") && !inside(f, m[1])) flag(`fetch("${m[1]}") leaves the repository`);
    if (/src\/SlicerLive|SlicerLive\//.test(text)) flag("names core's folder");
    if (/globalThis\s*(?:as[^.]*)?\)?\s*\.?\s*__|\bglobalThis\.__/.test(text)) flag("reads a global core sets for itself (use the SDK)");
    function check(spec: string) {
      if (spec === "albula" || spec === "albula/testing" || spec === "albula/server" || /^(jsr|npm):/.test(spec)) return;
      if (spec.startsWith(".") && inside(f, spec)) return;
      flag(`imports ${spec}`);
    }
  }
  assertEquals(offenders, [], "ways into core that bypass the SDK");
});

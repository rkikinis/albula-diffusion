// THE CODE THAT SHAPES THE TRACTS, FINGERPRINTED BY WHAT IT DOES (Ron, 2026-10-22: "Yes, as proposed"). Used where the
// job compares code: a run started by hand without --only-missing. The server's own runs (--only-missing, since
// 2026-10-14) never remake a set because the code changed; they compare only the scans a set was made from. (The reason
// first given for this file -- that commits had made the server remake whole databases -- was wrong: the critic,
// 2026-10-22, reversed-scan regression; the remake of 2026-10-21 came from a lost reversed scan.)
//
// The files the job reaches (import-job.ts importGraph), each with its comments and type annotations removed and its
// whitespace reduced (esbuild's transform, pinned in core's deno.jsonc), hashed WITHOUT their file names and in an order
// that does not depend on them. So a comment, a reworded note, a moved or renamed file changes nothing; a change to what
// the code does changes the fingerprint and makes stored tracts stale, as before.
//
// Version 2 of the fingerprint (version 1: the files' full text with their names, import-job-main.ts until 2026-10-22).
// A new esbuild may write the same code differently: then every set looks stale once (said in the record's code field
// by a new value; the pin is in one place).
import * as esbuild from "npm:esbuild@0.28.2";
import { importGraph } from "./import-job.ts";

export const CODE_FINGERPRINT_VERSION = 2;

const sha = async (b: Uint8Array) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", b as Uint8Array<ArrayBuffer>))].map((x) => x.toString(16).padStart(2, "0")).join("");

/** One file's code without comments: TypeScript through esbuild (types and comments go); WGSL with its comments
 *  removed (WGSL has no strings, so the comment marks are never inside one) and its whitespace collapsed. */
export async function codeOnly(path: string, text: string): Promise<string> {
  if (path.endsWith(".wgsl")) return text.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "").replace(/\s+/g, " ").trim();
  const r = await esbuild.transform(text, { loader: "ts", minifyWhitespace: true, legalComments: "none", target: "esnext" });
  return r.code;
}

/** The fingerprint of the code `entry` reaches, `entry` itself left out (its messages and arguments do not shape the
 *  tracts): twelve hexadecimal digits. Stops esbuild's helper process before it returns. */
export async function codeFingerprint(entry: URL): Promise<string> {
  try {
    const files = (await importGraph(entry)).filter((u) => u !== entry.href);
    const each: string[] = [];
    for (const u of files) each.push(await sha(new TextEncoder().encode(await codeOnly(u, await Deno.readTextFile(new URL(u))))));
    each.sort();
    return (await sha(new TextEncoder().encode(each.join("\n")))).slice(0, 12);
  } finally {
    await esbuild.stop();
  }
}

// Run directly: the fingerprint of the job in a folder (any checkout of this extension) --
//   deno run -A code-fingerprint.ts [folder]
if (import.meta.main) {
  const dir = new URL(`${(Deno.args[0] ?? ".").replace(/\/+$/, "")}/`, `file://${Deno.cwd()}/`);
  console.log(await codeFingerprint(new URL("import-job-main.ts", dir)));
}

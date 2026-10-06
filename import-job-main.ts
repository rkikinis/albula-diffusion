// THE IMPORT-TIME JOB AS A PROGRAM (import-job.ts is the library; Contents/docs/DMRI-AT-IMPORT.md in the workspace).
// Albula's server will start it after a diffusion scan is indexed and for the sweep of a whole database; until then it is
// run by hand against a running server:
//
//   deno run -A --unstable-webgpu --config Contents/src/SlicerLive/deno.jsonc Contents/extensions/diffusion/import-job-main.ts \
//     --server http://127.0.0.1:<port> --db <database id> [--series <SeriesInstanceUID>] [--dry-run] [--force] [--counts-only] [--assets <dir>]
//
// It prints ONE JSON OBJECT A LINE on standard output -- {"event": "plan" | "case" | "progress" | "done" | "error", ...};
// what a person reads is in "said". `--counts-only` prints states and counts only: no identifiers, no series names, no
// file paths, no error texts (critic, 2026-10-05, finding 2: a run from a Claude session over a private database must not
// print an identifying value into the session). Exit 0 when every case ended made, current or would-be-made; 1 when a
// case failed; 2 for a wrong call; 3 when another run holds the database.
// `--assets` is where the networks' weights are (the app's vendor/diffusion/); by default beside this file.
import { parseArgs } from "jsr:@std/cli@1/parse-args";
import { setDicomLibrary, dcmjs } from "albula/server";
import "./hooks.ts";
import { loadModel, type ModelJson } from "./tractcloud/tractcloud.ts";
import { loadRapidParc } from "./rapidparc/rapidparc.ts";
import { importGraph, makeTracts, planStudy, studiesOf, synthstripVersion } from "./import-job.ts";

const args = parseArgs(Deno.args, { string: ["server", "db", "series", "assets"], boolean: ["dry-run", "force", "counts-only"] });
const quiet = args["counts-only"];
// In counts-only mode only these fields leave the program.
const QUIET_KEYS = new Set(["at", "event", "case", "of", "state", "cases", "studies", "streamlines", "stored", "named", "seconds", "wall", "made", "current", "waiting", "cannot", "failed", "wouldBeMade"]);
const say = (o: Record<string, unknown>) => {
  const line = { at: new Date().toISOString(), ...o };
  console.log(JSON.stringify(quiet ? Object.fromEntries(Object.entries(line).filter(([k]) => QUIET_KEYS.has(k))) : line));
};
if (!args.server || !args.db) { say({ event: "error", said: "needs --server and --db" }); Deno.exit(2); }
const server = args.server.replace(/\/+$/, "");

/** THE CODE THAT SHAPES THE RESULT, hashed (critic, finding 1): every module of the extension except its tests, the
 *  person's interface (module.ts, face.ts, review.ts) and this program. A change there makes stored tracts stale. */
async function codeFingerprint(): Promise<string> {
  const here = new URL("./", import.meta.url), parts: Uint8Array[] = [];
  // From the program itself (it brings hooks.ts and the vendors' readers), the program's own file left out: its
  // arguments and messages do not shape the tracts.
  for (const u of (await importGraph(new URL("./import-job-main.ts", here))).filter((u) => !u.endsWith("/import-job-main.ts"))) parts.push(new TextEncoder().encode(`${u.slice(here.href.length)}\n`), await Deno.readFile(new URL(u)));
  const all = new Uint8Array(parts.reduce((n, p) => n + p.length, 0)); let o = 0; for (const p of parts) { all.set(p, o); o += p.length; }
  return [...new Uint8Array(await crypto.subtle.digest("SHA-256", all as Uint8Array<ArrayBuffer>))].slice(0, 6).map((b) => b.toString(16).padStart(2, "0")).join("");
}
const hash = async (b: Uint8Array) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", b as Uint8Array<ArrayBuffer>))].slice(0, 6).map((x) => x.toString(16).padStart(2, "0")).join("");

let lock: string | undefined;
let exitCode = 0;
try {
  setDicomLibrary(dcmjs);
  const list = await (await fetch(`${server}/_db`)).json() as { databases: { id: string; path: string }[] };
  const db = list.databases.find((d) => d.id === args.db);
  if (!db) { say({ event: "error", said: `no database "${args.db}" is registered with the server` }); Deno.exit(2); }
  // ONE RUN AT A TIME per database (finding 8): a lock file holding this program's process id.
  lock = `${db.path}/SlicerAlbula-SEG/.fiber-tracts-job.lock`;
  await Deno.mkdir(`${db.path}/SlicerAlbula-SEG`, { recursive: true });
  const held = await Deno.readTextFile(lock).catch(() => "");
  if (held && (await new Deno.Command("kill", { args: ["-0", held.trim()], stdout: "null", stderr: "null" }).output()).success) {
    say({ event: "error", said: "another fiber-tract run is working on this database" }); lock = undefined; Deno.exit(3);
  }
  await Deno.writeTextFile(lock, String(Deno.pid));
  // The networks: TractCloud's table (the tract names and the 1,600 → 43 map) and RapidParc's weights, read once.
  const assets = args.assets ? args.assets.replace(/\/+$/, "") : undefined;
  const at = (group: string, file: string) => assets ? `${assets}/${group}/${file}` : new URL(`./${group}/model/${file}`, import.meta.url);
  const modelJson = Deno.readFileSync(at("tractcloud", "model.json"));
  const model = loadModel(Deno.readFileSync(at("tractcloud", "weights.f32")).buffer, JSON.parse(new TextDecoder().decode(modelJson)) as ModelJson);
  const rpBytes = Deno.readFileSync(at("rapidparc", "rapidparc.safetensors"));
  const synthstrip = await synthstripVersion(server);
  if (!synthstrip) { say({ event: "error", said: "the segmentation server did not say which SynthStrip it runs (is it running?); nothing was made" }); exitCode = 1; throw new Error("no SynthStrip version"); }
  // The naming network's weights too, not only its table (critic 2026-10-06, R3-2).
  const versions = { code: await codeFingerprint(), labeler: `rapidparc ${await hash(rpBytes)}, table ${await hash(modelJson)}, tractcloud ${await hash(Deno.readFileSync(at("tractcloud", "weights.f32")))}`, synthstrip };
  const labeler = loadRapidParc(rpBytes.buffer);
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("no graphics card is available to this program");
  const L = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage, maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxBufferSize: L.maxBufferSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize } });

  const { rows, studies } = await studiesOf(db.path, args.series);
  if (args.series && !studies.length) { say({ event: "error", said: `series ${args.series} is not an MR series in this database's index` }); exitCode = 2; }
  say({ event: "plan", studies: studies.length, said: `${studies.length} stud${studies.length === 1 ? "y" : "ies"} with MR series to look at`, versions });
  const n = { made: 0, wouldBeMade: 0, current: 0, waiting: 0, cannot: 0, failed: 0 };
  let k = 0;
  // A STUDY AT A TIME: its series read, its cases made, then let go (finding 3).
  for (const study of studies) {
    let cases;
    try { cases = await planStudy(db.path, rows, study, (line, why) => say({ event: "progress", said: why ? `${line}: ${why}` : line })); }
    catch (e) { n.failed++; exitCode = 1; say({ event: "case", state: "failed", said: `a study could not be read: ${(e as Error).message}` }); continue; }
    for (const plan of cases) {
      if (args.series && plan.dwi.facts.uid !== args.series) continue;
      k++;
      const t0 = performance.now();
      try {
        const out = await makeTracts(db.path, db.id, server, plan, device, model, labeler, versions,
          (line) => say({ event: "progress", case: k, series: plan.dwi.facts.uid, said: line }), { force: args.force, dryRun: args["dry-run"] });
        if (out.state === "made") n.made++; else if (out.state === "would be made") n.wouldBeMade++; else if (out.state === "current") n.current++; else n[out.state]++;
        if (out.state === "failed") exitCode = 1;
        say({ event: "case", case: k, series: plan.dwi.facts.uid, study, t1: plan.t1?.facts.uid ?? null, partner: plan.partner?.facts.uid ?? null, ...out,
          wall: +((performance.now() - t0) / 1000).toFixed(1),
          said: "said" in out ? `fiber tracts ${out.state}: ${out.said}${plan.t1 ? `; MRI of the anatomy: ${plan.t1.facts.description}` : ""}` : out.state === "current" ? "fiber tracts already made by the same code, rules and scans" : out.why });
      } catch (e) {
        n.failed++; exitCode = 1;
        say({ event: "case", case: k, series: plan.dwi.facts.uid, state: "failed", said: `fiber tracts could not be made: ${(e as Error).message}` });
      }
    }
  }
  say({ event: "done", ...n, said: `fiber tracts: ${n.made} made, ${n.wouldBeMade} would be made (dry run), ${n.current} already current, ${n.waiting} waiting, ${n.cannot} cannot be made on this Mac, ${n.failed} failed` });
  device.destroy();
} catch (e) {
  say({ event: "error", said: (e as Error).message });
  exitCode = 1;
} finally {
  if (lock) await Deno.remove(lock).catch(() => {});
}
Deno.exit(exitCode);

// THE IMPORT-TIME JOB AS A PROGRAM (import-job.ts is the library; Contents/docs/DMRI-AT-IMPORT.md in the workspace).
// Albula's server will start it after a diffusion scan is indexed and for the sweep of a whole database; until then it is
// run by hand against a running server:
//
//   deno run -A --unstable-webgpu --config Contents/src/SlicerLive/deno.jsonc Contents/extensions/diffusion/import-job-main.ts \
//     --server http://127.0.0.1:<port> --db <database id> [--series <SeriesInstanceUID>] [--dry-run] [--force] [--counts-only] [--only-missing] [--parent <pid>] [--assets <dir>]
//
// `--only-missing` (how Albula's server runs it since 2026-10-14, desktop/extension-jobs.ts): tracts are made only for a
// diffusion scan that has none made FROM ITS PRESENT INPUTS with its maps. Tracts an earlier version made are KEPT, never
// remade or replaced (Ron, 2026-10-14: "for reproducibility scientists would like to run a specific version of a pipeline
// on data. You need to rethink that concept of stale"): a result is not stale, it is what a version made, and that version
// is recorded in it. Such a case is reported as state "kept". A set made before its reversed scan or its T1 arrived, or
// without its maps, is not "there": a new set is made beside it (critic 2026-10-14, import-time jobs, finding 5).
// `--parent <pid>`: stop when that process (Albula's server) is gone (finding 10).
//
// It prints ONE JSON OBJECT A LINE on standard output -- {"event": "plan" | "case" | "progress" | "done" | "error", ...};
// what a person reads is in "said". `--counts-only` prints states and counts only: no identifiers, no series names, no
// file paths, no error texts (critic, 2026-10-05, finding 2: a run from a Claude session over a private database must not
// print an identifying value into the session). Exit 0 when every case ended made, current or would-be-made; 1 when a
// case failed; 2 for a wrong call; 3 when another run holds the database.
// `--assets` is where the networks' weights are (the app's vendor/diffusion/); by default beside this file.
import { parseArgs } from "jsr:@std/cli@1/parse-args";
import { setDicomLibrary, dcmjs } from "albula/server";
import { parseNiftiVolumes } from "albula";
import "./hooks.ts";
import { CLUSTERS_NAME, loadTable, TABLE_NAME, type TableJson } from "./naming/table.ts";
import { codeFingerprint } from "./code-fingerprint.ts";
import { loadRapidParc } from "./rapidparc/rapidparc.ts";
import { importGraph, jobRules, makeMaps, makeTracts, MAP_PATHS, planStudy, studiesOf, synthstripVersion } from "./import-job.ts";
import { appendRunLog } from "./run-log.ts";
import { dicomToTracts } from "./tracts-dicom.ts";
import { indexSeries, provenanceEdges, provenancePathFor, seriesFilePaths } from "albula/server";

const args = parseArgs(Deno.args, { string: ["server", "db", "series", "assets", "parent", "core-version"], boolean: ["dry-run", "force", "counts-only", "only-missing"] });
const quiet = args["counts-only"];
// In counts-only mode only these fields leave the program.
const QUIET_KEYS = new Set(["at", "event", "case", "of", "state", "cases", "studies", "streamlines", "stored", "named", "seconds", "wall", "made", "current", "kept", "waiting", "cannot", "failed", "wouldBeMade", "stopped"]);
const say = (o: Record<string, unknown>) => {
  const line = { at: new Date().toISOString(), ...o };
  console.log(JSON.stringify(quiet ? Object.fromEntries(Object.entries(line).filter(([k]) => QUIET_KEYS.has(k))) : line));
};
if (!args.server || !args.db) { say({ event: "error", said: "needs --server and --db" }); Deno.exit(2); }
const server = args.server.replace(/\/+$/, "");

/** THE CODE THAT SHAPES THE RESULT, hashed (critic, finding 1): every module of the extension except its tests, the
 *  person's interface (module.ts, face.ts, review.ts) and this program. A change there makes stored tracts stale. */
const hash = async (b: Uint8Array) => [...new Uint8Array(await crypto.subtle.digest("SHA-256", b as Uint8Array<ArrayBuffer>))].slice(0, 6).map((x) => x.toString(16).padStart(2, "0")).join("");

/** The commit and the "changes since" of this extension's folder, for the record (finding 6); the core's version comes
 *  from the server (`--core-version`): an extension does not reach into core's folders. */
async function gitState(dir: string): Promise<string> {
  const g = async (...a: string[]) => { try { const o = await new Deno.Command("git", { args: ["-C", dir, ...a], stdout: "piped", stderr: "null" }).output(); return o.success ? new TextDecoder().decode(o.stdout).trim() : undefined; } catch { return undefined; } };
  const head = await g("rev-parse", "--short=12", "HEAD");
  if (!head) return "unknown";
  return (await g("status", "--porcelain", "--untracked-files=no")) ? `${head}+changes` : head;
}

/** THE NEWEST STORED SET of a diffusion scan, cheaply (finding 15: every 28 MB object was read to learn whether one
 *  exists, 38.7 s on the test cases): the provenance store's "tracts" edges, the children this database's index holds,
 *  the highest series number -- the same rule the Diffusion module uses -- and only that one object read, for its inputs.
 *  The edges are read once per run. */
let edgesOnce: Promise<{ child: string; parent: string; kind: string }[]> | undefined;
async function newestStored(dbPath: string, rows: { seriesUID: string; seriesNumber: string }[], dwiUID: string): Promise<{ seriesUID: string; inputs?: { partner?: string | null; t1?: string | null } } | undefined> {
  edgesOnce ??= provenanceEdges(provenancePathFor(dbPath)).catch(() => []);
  const kids = new Set((await edgesOnce).filter((e) => e.kind === "tracts" && e.parent === dwiUID).map((e) => e.child));
  const top = rows.filter((r) => kids.has(r.seriesUID)).sort((a, b) => (Number(b.seriesNumber) || 0) - (Number(a.seriesNumber) || 0))[0];
  if (!top) return undefined;
  const file = (await seriesFilePaths(dbPath, top.seriesUID))[0];
  let inputs: { partner?: string | null; t1?: string | null } | undefined;
  try { const t = await dicomToTracts(await Deno.readFile(file)); inputs = (t.provenance?.inputs as typeof inputs) ?? undefined; } catch { /* unreadable: counts as missing inputs */ }
  return { seriesUID: top.seriesUID, inputs };
}

// THE PARENT (finding 10): when Albula's server goes, so does this run -- its result could not be stored anyway.
if (args.parent) {
  const parent = args.parent;
  setInterval(async () => {
    const o = await new Deno.Command("kill", { args: ["-0", parent], stdout: "null", stderr: "null" }).output().catch(() => null);
    if (o && !o.success) { if (lock) await Deno.remove(lock).catch(() => {}); Deno.exit(4); }
  }, 5000);
}

let lock: string | undefined;
let exitCode = 0;
/** Ends the run with a reason a person can read, in one fixed word in --counts-only (finding 14). */
const stop = (word: string, said: string, code = 1) => { say({ event: "error", stopped: word, said }); exitCode = code; };
try {
  setDicomLibrary(dcmjs);
  const list = await (await fetch(`${server}/_db`)).json() as { databases: { id: string; path: string }[] };
  const db = list.databases.find((d) => d.id === args.db);
  if (!db) { say({ event: "error", stopped: "no such database", said: `no database "${args.db}" is registered with the server` }); Deno.exit(2); }
  // ONE RUN AT A TIME per database: a lock made atomically (createNew), holding this program's process id; a lock left by
  // a run that is gone -- or whose number now belongs to another program (finding 9) -- is taken over.
  lock = `${db.path}/SlicerAlbula-SEG/.fiber-tracts-job.lock`;
  await Deno.mkdir(`${db.path}/SlicerAlbula-SEG`, { recursive: true });
  const take = async () => { const f = await Deno.open(lock!, { write: true, createNew: true }); await f.write(new TextEncoder().encode(String(Deno.pid))); f.close(); };
  try { await take(); } catch {
    const held = (await Deno.readTextFile(lock).catch(() => "")).trim();
    const ps = held ? await new Deno.Command("ps", { args: ["-p", held, "-o", "command="], stdout: "piped", stderr: "null" }).output().catch(() => null) : null;
    const alive = !!ps?.success && /import-job-main/.test(new TextDecoder().decode(ps.stdout));
    if (alive) { lock = undefined; say({ event: "error", stopped: "another run holds this database", said: "another fiber-tract run is working on this database" }); Deno.exit(3); }
    await Deno.remove(lock).catch(() => {});
    try { await take(); } catch { lock = undefined; say({ event: "error", stopped: "another run holds this database", said: "another run took the database at the same moment" }); Deno.exit(3); }
  }

  // FIRST THE CASES, THEN THE MACHINERY (finding 13): nothing is started -- no segmentation server, no graphics card, no
  // networks -- until a case needs making.
  // THE RUN LOG (run-log.ts): what this run tries, with what, and how each scan ends -- for people outside the project.
  const run = `${new Date().toISOString().replace(/[-:]/g, "").slice(0, 15)}-${crypto.randomUUID().slice(0, 6)}`;
  const log = (event: "run" | "versions" | "case", fields: Record<string, unknown>) => appendRunLog(db.path, event, { run, ...fields }).then((ok) => { if (!ok) say({ event: "progress", said: "the run log could not be written" }); });
  await log("run", { database: db.id, how: { onlyMissing: !!args["only-missing"], dryRun: !!args["dry-run"], force: !!args.force, series: args.series ?? null },
    extension: await gitState(new URL("./", import.meta.url).pathname), core: args["core-version"] ?? "unknown" });
  const { rows, studies } = await studiesOf(db.path, args.series);
  if (args.series && !studies.length) { say({ event: "error", stopped: "no such series", said: `series ${args.series} is not an MR series in this database's index` }); exitCode = 2; }
  say({ event: "plan", studies: studies.length, said: `${studies.length} stud${studies.length === 1 ? "y" : "ies"} with MR series to look at` });

  type Machinery = { device: GPUDevice; model: ReturnType<typeof loadTable>; labeler: ReturnType<typeof loadRapidParc>; versions: { code: string; labeler: string; synthstrip: string; core: string; extension: string; headTemplate?: string }; template: { t1: { dims: [number, number, number]; ijkToRAS: number[]; data: ArrayLike<number> }; brain: Uint8Array } };
  let machinery: Machinery | undefined;
  const ready = async (): Promise<Machinery> => {
    if (machinery) return machinery;
    // THE NAMING: RapidParc's network and TractCloud's 42-tract table on the ORG-800FC-100HCP clusters (NAMING.md), read
    // once. TractCloud's network is not loaded: it does not name here.
    const assets = args.assets ? args.assets.replace(/\/+$/, "") : undefined;
    const at = (group: string, file: string) => assets ? `${assets}/${group}/${file}` : new URL(`./${group}/model/${file}`, import.meta.url);
    const tableBytes = Deno.readFileSync(assets ? `${assets}/naming/tractcloud-42.json` : new URL("./naming/tractcloud-42.json", import.meta.url));
    const model = loadTable(JSON.parse(new TextDecoder().decode(tableBytes)) as TableJson);
    const rpBytes = Deno.readFileSync(at("rapidparc", "rapidparc.safetensors"));
    const synthstrip = await synthstripVersion(server);
    if (!synthstrip) throw Object.assign(new Error("the segmentation server did not say which SynthStrip it runs (is it running?)"), { word: "no segmentation server" });
    // The naming network's weights too, not only its table (critic 2026-10-06, R3-2); and the commits of the core and of
    // the extension, with "+changes" when their folders differ from them (finding 6).
    const here = new URL("./", import.meta.url).pathname;
    // THE RECORD'S NAMING FIELD (2026-10-22, NAMING.md): the network that names, the clusters it gives, the table that
    // names them -- each by what it is. (Until 2026-10-22: "rapidparc …, table …, tractcloud …, template …", which
    // listed TractCloud's weights, loaded but not run, and the head template; import-job.ts upgradeRules reads it.)
    const versions = { code: await codeFingerprint(new URL("./import-job-main.ts", import.meta.url)), labeler: `rapidparc ${await hash(rpBytes)}; clusters ${CLUSTERS_NAME}; tract table ${TABLE_NAME} ${await hash(tableBytes)}`, synthstrip,
      core: args["core-version"] ?? "unknown", extension: await gitState(here), headTemplate: "" };
    const labeler = loadRapidParc(rpBytes.buffer);
    // The template brain for the head's own frame (head-frame.ts), from the extension's template/ folder.
    const tp = (f: string) => assets ? `${assets}/template/${f}` : new URL(`./template/${f}`, import.meta.url);
    const tv = (await parseNiftiVolumes(Deno.readFileSync(tp("mni152-2009c-asym-t1w.nii.gz"))))[0], tb = (await parseNiftiVolumes(Deno.readFileSync(tp("mni152-2009c-asym-brain-mask.nii.gz"))))[0];
    // The head template of the head-frame step: a field of its own (it has nothing to do with naming).
    versions.headTemplate = await hash(Deno.readFileSync(tp("mni152-2009c-asym-t1w.nii.gz")));
    const template = { t1: { dims: tv.dims as [number, number, number], ijkToRAS: tv.ijkToRAS, data: tv.data as ArrayLike<number> }, brain: Uint8Array.from(tb.data as ArrayLike<number>, (x) => (x > 0.5 ? 1 : 0)) };
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw Object.assign(new Error("no graphics card is available to this program"), { word: "no graphics card" });
    const L = adapter.limits;
    const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage, maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxBufferSize: L.maxBufferSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize } });
    machinery = { device, model, labeler, versions, template };
    say({ event: "progress", said: "ready", versions });
    await log("versions", { versions, rules: jobRules(versions) });
    return machinery;
  };

  const n = { made: 0, wouldBeMade: 0, current: 0, kept: 0, waiting: 0, cannot: 0, failed: 0 };
  type Plan = Awaited<ReturnType<typeof planStudy>>[number];
  const who = (s: Plan["dwi"] | undefined) => s ? { uid: s.facts.uid, description: s.facts.description } : null;
  const logCase = (plan: Plan, fields: Record<string, unknown>) => log("case", {
    scan: { ...who(plan.dwi), patient: String(plan.dwi.header["00100020"]?.Value?.[0] ?? ""), study: plan.dwi.facts.studyUID },
    reversedScan: who(plan.partner), ...(plan.partnerRefusals ? { reversedScanRefused: plan.partnerRefusals } : {}), t1: who(plan.t1), ...fields });
  let rowsOnce: Promise<{ seriesUID: string; seriesNumber: string }[]> | undefined;
  let k = 0;
  // A STUDY AT A TIME: its series read, its cases made, then let go (finding 3).
  studies: for (const study of studies) {
    let cases;
    try { cases = await planStudy(db.path, rows, study, (line, why) => say({ event: "progress", said: why ? `${line}: ${why}` : line })); }
    catch (e) { n.failed++; exitCode = 1; say({ event: "case", state: "failed", said: `a study could not be read: ${(e as Error).message}` }); continue; }
    for (const plan of cases) {
      if (args.series && plan.dwi.facts.uid !== args.series) continue;
      k++;
      const t0 = performance.now();
      // THERE = the newest set, made from the inputs the database holds now (the reversed scan, the T1), with its maps
      // (MAP_PATHS) beside it. Its maps missing: they alone are added, no tracking (finding 5). Older inputs, or no set:
      // a new set is made beside the old, which stays.
      let mapsFor: string | undefined;
      if (args["only-missing"]) {
        const s = await newestStored(db.path, await (rowsOnce ??= indexSeries(db.path)), plan.dwi.facts.uid).catch(() => undefined);
        const sameInputs = !!s?.inputs && (s.inputs.partner ?? null) === (plan.partner?.facts.uid ?? null) && (s.inputs.t1 ?? null) === (plan.t1?.facts.uid ?? null);
        if (s && sameInputs) {
          const maps = (await Promise.all(MAP_PATHS.map((p) => Deno.stat(`${db.path}/${p(s.seriesUID)}`).then(() => true, () => false)))).every(Boolean);
          if (maps) {
            n.kept++;
            say({ event: "case", case: k, series: plan.dwi.facts.uid, state: "kept", said: "fiber tracts and maps are there, made from the inputs the database holds; kept" });
            await logCase(plan, { state: "kept", result: s.seriesUID, said: "made earlier from the same scans; kept" });
            continue;
          }
          mapsFor = s.seriesUID;
        }
        // NEVER REPLACED BY ONE MADE FROM LESS (critic 2026-10-22, reversed-scan regression, finding 7): the newest set was
        // made with a reversed scan or a T1 this run does not find. Making a new set without it would hide the loss behind
        // a newer, worse result -- as it did on 2026-10-21 for 37 scans. Refused, said, and logged; a person decides.
        const lost = s?.inputs ? [s.inputs.partner && !plan.partner ? "the reversed scan" : "", s.inputs.t1 && !plan.t1 ? "the MRI of the anatomy (T1)" : ""].filter(Boolean) : [];
        if (lost.length) {
          n.failed++; exitCode = 1;
          const said = `not made: the newest fiber tracts were made with ${lost.join(" and ")}, which this run does not find; a set made without it would replace a better one`;
          say({ event: "case", case: k, series: plan.dwi.facts.uid, state: "failed", said });
          await logCase(plan, { state: "refused", newest: s!.seriesUID, newestInputs: s!.inputs, said });
          continue;
        }
      }
      try {
        const m = await ready();
        const progress = (line: string) => say({ event: "progress", case: k, series: plan.dwi.facts.uid, said: line });
        const out = mapsFor && !args["dry-run"]
          ? await makeMaps(db.path, server, plan, mapsFor, m.device, m.model, m.labeler, progress)
          : await makeTracts(db.path, db.id, server, plan, m.device, m.model, m.labeler, m.versions, progress, { force: args.force, dryRun: args["dry-run"], template: m.template });
        if (out.state === "made") n.made++; else if (out.state === "would be made") n.wouldBeMade++; else if (out.state === "current") n.current++; else n[out.state]++;
        if (out.state === "failed") exitCode = 1;
        await logCase(plan, { ...out, wall: +((performance.now() - t0) / 1000).toFixed(1) });
        say({ event: "case", case: k, series: plan.dwi.facts.uid, study, t1: plan.t1?.facts.uid ?? null, partner: plan.partner?.facts.uid ?? null, ...out,
          wall: +((performance.now() - t0) / 1000).toFixed(1),
          said: "said" in out ? `fiber tracts ${out.state}: ${out.said}${plan.t1 ? `; MRI of the anatomy: ${plan.t1.facts.description}` : ""}` : out.state === "current" ? "fiber tracts already made by the same code, rules and scans" : out.why });
      } catch (e) {
        const word = (e as { word?: string }).word;
        if (word) { stop(word, (e as Error).message); break studies; }   // nothing more can be made in this run
        n.failed++; exitCode = 1;
        say({ event: "case", case: k, series: plan.dwi.facts.uid, state: "failed", said: `fiber tracts could not be made: ${(e as Error).message}` });
        await logCase(plan, { state: "failed", said: `fiber tracts could not be made: ${(e as Error).message}` });
      }
    }
  }
  say({ event: "done", ...n, said: `fiber tracts: ${n.made} made, ${n.wouldBeMade} would be made (dry run), ${n.current} already current, ${n.kept} kept as made, ${n.waiting} waiting, ${n.cannot} cannot be made on this Mac, ${n.failed} failed` });
  machinery?.device.destroy();
} catch (e) {
  stop("stopped by an error", (e as Error).message);
} finally {
  if (lock) await Deno.remove(lock).catch(() => {});
}
Deno.exit(exitCode);

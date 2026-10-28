// The import-time job's choices (import-job.ts), on made-up series: which scan is tracked, which series is its reversed
// partner, which image is its MRI of the anatomy, when stored tracts are current, and how streamlines become track sets
// (critic, 2026-10-05, qa/2026-10-05-dmri-import-job.md, findings 6, 7, 11, 14).
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc import-job.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { anatomyOf, b0Path, colorFaPath, importGraph, isCurrent, isDiffusionScan, jobRules, partnerOf, partnerRefusals, planCases, synthstripVersion, trackSets, measuredB0, writeB0, writeColorFA, type SeriesFacts, CODE_SAME, madeWith } from "./import-job.ts";
import { packRGB24 } from "albula";
import type { TensorFit } from "./tensor.ts";
import type { DiffusionSeries } from "./dwi.ts";
import { SHORT } from "./naming/name-tracts.ts";
import { UNNAMED } from "./tracts-dicom.ts";

const AX = [-2.5, 0, 0, 120, 0, 2.5, 0, -100, 0, 0, 2.5, -60, 0, 0, 0, 1];
/** A series: `b` its b-values (NaN for an image that is not diffusion), `dirs` its distinct directions. */
const S = (uid: string, description: string, b: number[], o: Partial<SeriesFacts> = {}): SeriesFacts => ({
  uid, studyUID: "st1", patientUID: "p1", description, volumes: b.length, bValues: b, directions: o.directions ?? (b.filter((x) => x > 50).length), ijkToRAS: AX, derived: false, ...o });
const dwi = (uid: string, n: number, o: Partial<SeriesFacts> = {}) => S(uid, `DTI ${n}`, [0, ...Array(n - 1).fill(1000)], o);
const t1 = (uid: string, description = "T1_mprage", o: Partial<SeriesFacts> = {}) => S(uid, description, [NaN], o);

Deno.test("a diffusion scan needs seven volumes and six directions", () => {
  assert(isDiffusionScan(dwi("a", 7)));
  assert(!isDiffusionScan(dwi("a", 6)));
  assert(!isDiffusionScan(S("t", "trace", [0, 1000, 1000, 1000, 1000, 1000, 1000, 1000], { directions: 1 })));
});

Deno.test("the reversed partner: the record's reversed pair first, then b = 0 only; a pair the record refuses is skipped; elsewhere placed or bigger never", () => {
  const scan = dwi("scan", 102, { phaseEncoding: "j-" });
  const b0 = S("b0", "PA b0", [0, 0]);
  const recorded = S("rec", "PA", [0, 0, 0], { phaseEncoding: "j" });
  const wrongAxis = S("lr", "LR", [0, 0], { phaseEncoding: "i-" });
  assertEquals(partnerOf(scan, [scan, b0, recorded])?.uid, "rec");
  assertEquals(partnerOf(scan, [scan, b0])?.uid, "b0");
  // A computed map is never the partner (Siemens' TENSOR_B0 on Steve's Prisma scan), and a "PA" the record says is NOT
  // reversed is skipped for one that is (that scan's dMRI_dir99_PA was taken in the AP direction).
  const tensorB0 = S("tb0", "dMRI_dir99_AP_TENSOR_B0", [0], { phaseEncoding: "j-", derived: true });
  const notReversed = S("pa99", "dMRI_dir99_PA", [0, 0, 0], { phaseEncoding: "j-" });
  assertEquals(partnerOf(scan, [scan, tensorB0, notReversed, recorded])?.uid, "rec");
  assertEquals(partnerOf(scan, [scan, tensorB0, notReversed]), undefined);
  assertEquals(partnerOf(scan, [scan, wrongAxis]), undefined);
  // A short diffusion scan with its own b = 0 (PA, 7 volumes) is a partner, not a scan of its own (finding 7).
  const shortPA = dwi("pa7", 7, { phaseEncoding: "j" });
  const cases = planCases([scan, shortPA, t1("t1")]);
  assertEquals(cases.map((c) => [c.dwi.uid, c.partner?.uid]), [["scan", "pa7"]]);
  // Placed elsewhere (another slab 40 mm away) or another study: not a partner.
  const far = S("far", "PA", [0, 0], { ijkToRAS: AX.map((v, i) => (i === 11 ? v + 40 : v)) });
  assertEquals(partnerOf(scan, [scan, far, S("other", "PA", [0, 0], { studyUID: "st2" })]), undefined);
});

Deno.test("a full-length reversed twin (the HCP protocols: dir98 / dir99, each in AP and PA) is the partner only when the record says reversed", () => {
  // AABC / HCP Lifespan: four full diffusion scans, two runs, each run in both directions; no b = 0-only series.
  const ap99 = dwi("ap99", 106, { phaseEncoding: "j-" }), pa99 = { ...dwi("pa99", 106, { phaseEncoding: "j" }), description: "dMRI_dir99_PA" };
  const ap98 = { ...dwi("ap98", 105, { phaseEncoding: "j-" }), description: "dMRI_dir98_AP" }, pa98 = { ...dwi("pa98", 105, { phaseEncoding: "j" }), description: "dMRI_dir98_PA" };
  ap99.description = "dMRI_dir99_AP";
  // Each run with its own other half (the twin before the other run's shorter reversed scan), in any order of the list.
  for (const list of [[ap99, pa99, ap98, pa98], [pa98, ap98, pa99, ap99], [pa99, ap98, ap99, pa98]]) {
    const cases = planCases([...list, t1("t1")]);
    assertEquals(cases.map((c) => [c.dwi.uid, c.partner?.uid]), [["ap99", "pa99"], ["ap98", "pa98"]]);
  }
  // Without the record, a series as long as the scan is not its partner (a repeat or another run): a case of its own.
  const plain = dwi("a", 60), plain2 = dwi("b", 60);
  assertEquals(planCases([plain, plain2]).map((c) => [c.dwi.uid, c.partner?.uid ?? null]), [["a", null], ["b", null]]);
  assertEquals(partnerRefusals(plain, [plain, plain2]).map((r) => r.why), ["as many volumes as the scan, and not recorded as its reversed twin (? against ?)"]);
  // Recorded as the same direction (Steve's Prisma dMRI_dir99_PA, taken AP): never.
  const sameDir = { ...dwi("pa-not", 106, { phaseEncoding: "j-" }), description: "dMRI_dir99_PA" };
  assertEquals(partnerOf(ap99, [ap99, sameDir]), undefined);
  // Longer than the scan: never its partner (the longer one is the scan, and takes the shorter as its own).
  assertEquals(partnerOf(ap98, [ap98, pa99]), undefined);
  assertEquals(planCases([ap98, pa99]).map((c) => [c.dwi.uid, c.partner?.uid]), [["pa99", "ap98"]]);
});

Deno.test("the MRI of the anatomy: named as a T1, one volume, not computed; same study first, then the same patient; none named so, none chosen", () => {
  const scan = dwi("scan", 30);
  assertEquals(anatomyOf(scan, [scan, S("flair", "FLAIR", [NaN]), t1("t1", "T1_mprage")])?.uid, "t1");
  assertEquals(anatomyOf(scan, [scan, S("flair", "FLAIR", [NaN]), S("adc", "ADC", [NaN], { derived: true })]), undefined);   // quality first: wait
  assertEquals(anatomyOf(scan, [scan, t1("t10", "T10 something")]), undefined);                                            // "T10" is not a T1
  assertEquals(anatomyOf(scan, [scan, t1("mp", "MPRAGE sag")])?.uid, "mp");
  assertEquals(anatomyOf(scan, [scan, t1("ours", "T1 resampled", { derived: true })]), undefined);
  assertEquals(anatomyOf(scan, [scan, t1("other", "T1", { studyUID: "st2" }), t1("own", "T1 post", { studyUID: "st1" })])?.uid, "own");
  assertEquals(anatomyOf(scan, [scan, t1("other", "T1", { studyUID: "st2" })])?.uid, "other");                             // same patient, other study
  assertEquals(anatomyOf(scan, [scan, t1("stranger", "T1", { studyUID: "st3", patientUID: "p2" })]), undefined);
});

Deno.test("stored tracts are current only for the same code, rules and inputs", () => {
  const v = { code: "abc", labeler: "rapidparc x", synthstrip: "haversack 0.15.0" };
  const rules = jobRules(v), inputs = { diffusion: "d", partner: "p", t1: "t" };
  assert(isCurrent({ rules, inputs }, rules, inputs));
  assert(!isCurrent({ rules: jobRules({ ...v, code: "abd" }), inputs }, rules, inputs), "a code change must make them stale");
  assert(!isCurrent({ rules, inputs: { ...inputs, partner: null } }, rules, inputs), "a reversed scan that arrived later must make them stale");
  assert(!isCurrent({ rules, inputs: { ...inputs, t1: "other" } }, rules, inputs));
  assert(!isCurrent({ rules: { ...rules, extra: 1 }, inputs }, rules, inputs));
  assert(!isCurrent(undefined, rules, inputs));
});

// 2026-10-22 (Ron: "Yes, as proposed"): the commits are not compared; an older record is read through upgradeRules, and
// only the equivalences stated in CODE_SAME and the naming field's rewording carry it over -- nothing else.
Deno.test("an older record: commits not compared, a stated code equivalence and the old naming field carry over; nothing else does", () => {
  const now = jobRules({ code: CODE_SAME["1b13f282b7ec"].same, labeler: "rapidparc 6a158dad6a0b; clusters ORG-800FC-100HCP; tract table TractCloud-42 0123456789ab", synthstrip: "haversack 0.15.0", headTemplate: "1f27aabea9f7" });
  const inputs = { diffusion: "d", partner: "p", t1: "t" };
  assertEquals("core" in now || "extension" in now, false, "the commits are not in the rules");
  assertEquals(madeWith({ core: "8b53f4d", extension: "f307711" }), { core: "8b53f4d", extension: "f307711" });
  const old = { ...now, code: "1b13f282b7ec", labeler: "rapidparc 6a158dad6a0b, table 75b1af321932, tractcloud 5a843c95eb60, template 1f27aabea9f7", core: "4d2ab4ae7387", extension: "deab2c4a0598" } as Record<string, unknown>;
  delete old.headTemplate;
  assert(isCurrent({ rules: old, inputs }, now, inputs), "the same code, the same naming: current");
  assert(!isCurrent({ rules: { ...old, code: "9925e5b2db07" }, inputs }, now, inputs), "older code with real changes: stale");
  assert(!isCurrent({ rules: { ...old, labeler: "rapidparc 999999999999, table 75b1af321932, tractcloud 5a843c95eb60, template 1f27aabea9f7" }, inputs }, now, inputs), "another network: stale");
  assert(!isCurrent({ rules: { ...old, labeler: "rapidparc 6a158dad6a0b, table 000000000000, tractcloud 5a843c95eb60, template 1f27aabea9f7" }, inputs }, now, inputs), "another table: stale");
  assert(!isCurrent({ rules: { ...old, labeler: "rapidparc 6a158dad6a0b, table 75b1af321932, tractcloud 5a843c95eb60, template 000000000000" }, inputs }, now, inputs), "another head template: stale");
  assert(!isCurrent({ rules: { ...old, tracking: 7 }, inputs }, now, inputs), "another tracking rule: stale");
  assert(isCurrent({ rules: { ...now, core: "anything", extension: "anything" }, inputs }, now, inputs), "a newer core or extension commit alone: still current");
});

Deno.test("track sets: one per named tract and side, the rest in Unnamed, last", () => {
  const names = [{ name: "arcuate" }, { name: "corpus callosum" }, { name: "Other" }];
  const sl = [0, 1, 2, 3, 4].map((i) => new Float32Array([i, 0, 0, i + 1, 0, 0]));
  const sets = trackSets(sl, Int32Array.from([0, 0, 1, 2, SHORT]), Int8Array.from([-1, 1, 0, 0, 0]), new Uint8Array(5), names);
  assertEquals(sets.map((s) => [s.label, s.side, s.streamlines.length]), [["arcuate", -1, 1], ["arcuate", 1, 1], ["corpus callosum", 0, 1], [UNNAMED, 0, 2]]);
});

Deno.test("the direction-colored map is stored beside the tracts: the Color FA, packed as the slice views draw it", async () => {
  const dir = await Deno.makeTempDir();
  try {
    // Two voxels: FA 0.5 along left-right (red), FA 1 along up-down (blue).
    const fit = { dims: [2, 1, 1], ijkToRAS: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], fa: new Float32Array([0.5, 1]), v1: new Float32Array([1, 0, 0, 0, 0, 1]) } as unknown as TensorFit;
    await writeColorFA(dir, "1.2.3", fit);
    const bytes = await Deno.readFile(`${dir}/${colorFaPath("1.2.3")}`);
    const text = new TextDecoder().decode(bytes.subarray(0, 400)), at = text.indexOf("\n\n") + 2;
    assert(/type: float/.test(text) && /encoding: gzip/.test(text) && /sizes: 2 1 1/.test(text), text);
    const raw = new Uint8Array(await new Response(new Blob([bytes.subarray(at)]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
    const v = new Float32Array(raw.buffer);
    assertEquals([...v], [packRGB24(128, 0, 0), packRGB24(0, 0, 255)]);
  } finally { await Deno.remove(dir, { recursive: true }); }
});

Deno.test("the b = 0 image beside the tracts: the measured b = 0 images' mean on the fit's grid, written whole", async () => {
  const dir = await Deno.makeTempDir();
  try {
    const M = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const fit = { dims: [2, 1, 1], ijkToRAS: M, S0: new Float32Array([790, 130]) } as unknown as TensorFit;
    const vol = (a: number, b: number) => ({ dims: [2, 1, 1], ijkToRAS: M, data: new Float32Array([a, b]) });
    // Two b = 0 images and one b = 1000: the mean of the two b = 0, not the fit's S0, and not the b = 1000.
    const dwi = { volumes: [vol(800, 140), vol(825, 160), vol(300, 20)], bValues: [0, 5, 1000], gradients: [[0, 0, 0], [0, 0, 0], [1, 0, 0]], ijkToRAS: M } as unknown as DiffusionSeries;
    assertEquals([...measuredB0(dwi, fit)], [812.5, 150]);
    // Another grid: the fit's S0.
    assertEquals([...measuredB0({ ...dwi, ijkToRAS: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1] } as DiffusionSeries, fit)], [790, 130]);
    await writeB0(dir, "1.2.3", dwi, fit);
    const bytes = await Deno.readFile(`${dir}/${b0Path("1.2.3")}`);
    const text = new TextDecoder().decode(bytes.subarray(0, 400)), at = text.indexOf("\n\n") + 2;
    assert(/type: float/.test(text) && /sizes: 2 1 1/.test(text), text);
    const raw = new Uint8Array(await new Response(new Blob([bytes.subarray(at)]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer());
    assertEquals([...new Float32Array(raw.buffer)], [812.5, 150]);
    assertEquals([...Deno.readDirSync(`${dir}/SlicerAlbula-Cache`)].filter((e) => e.name.endsWith(".tmp")).length, 0, "no temporary file left");
  } finally { await Deno.remove(dir, { recursive: true }); }
});

Deno.test("the SynthStrip version is waited for, never recorded as unknown (a server still starting answered '?')", async () => {
  let asked = 0;
  const ac = new AbortController();
  const srv = Deno.serve({ port: 0, signal: ac.signal, onListen: () => {} }, (req) => {
    const p = new URL(req.url).pathname;
    if (p.endsWith("/_status")) { asked++; return Response.json(asked < 3 ? {} : { health: { version: "0.15.0" } }); }
    return Response.json(asked < 3 ? {} : { weights_installed: [{ id: "synthstrip", version: "1" }] });
  });
  try {
    const base = `http://127.0.0.1:${(srv.addr as Deno.NetAddr).port}`;
    assertEquals(await synthstripVersion(base, { pollMs: 10, waitMs: 5000 }), "haversack 0.15.0, synthstrip weights 1");
    asked = -1000;
    assertEquals(await synthstripVersion(base, { pollMs: 10, waitMs: 50 }), undefined);
  } finally { ac.abort(); await srv.finished; }
});

Deno.test("a segmentation server without SynthStrip is said as such at once, not waited on (critic 2026-10-06, finding 14)", async () => {
  const ac = new AbortController();
  const srv = Deno.serve({ port: 0, signal: ac.signal, onListen: () => {} }, (req) => new URL(req.url).pathname.endsWith("/_status") ? Response.json({ reachable: true, health: { version: "0.15.0" } }) : new Response("no such task", { status: 404 }));
  try {
    const t0 = performance.now();
    assertEquals(await synthstripVersion(`http://127.0.0.1:${(srv.addr as Deno.NetAddr).port}`, { pollMs: 10, waitMs: 5000 }), "haversack 0.15.0, no SynthStrip");
    assert(performance.now() - t0 < 1000);
  } finally { ac.abort(); await srv.finished; }
});

Deno.test("the code fingerprint covers the files that make the tracts, not the review's drawing (critic R2-4, R3-2)", async () => {
  const here = new URL("./", import.meta.url);
  const g = await importGraph(new URL("./import-job-main.ts", here));
  const rel = g.map((u) => u.slice(here.href.length));
  for (const f of ["tracking.ts", "prepare.ts", "csd-worker.ts", "naming/name-tracts.ts", "naming/table.ts", "tracts-dicom.ts", "hooks.ts", "diffusion-vendors.ts"]) assert(rel.includes(f), `${f} is in the graph`);
  for (const f of ["tract-slice.ts", "review.ts", "module.ts", "face.ts"]) assert(!rel.includes(f), `${f} is not`);
  // Deno's own module graph of the program, the extension's files: every one of them is in ours (workers, which Deno's
  // static graph does not follow, are in ours besides).
  // (Without core's configuration: "albula" stays unresolved, which leaves the extension's own files -- all this needs.)
  const out = await new Deno.Command(Deno.execPath(), { args: ["info", "--json", new URL("./import-job-main.ts", here).pathname], stdout: "piped", stderr: "null" }).output();
  const deno = (JSON.parse(new TextDecoder().decode(out.stdout)) as { modules: { specifier: string }[] }).modules.map((m) => m.specifier).filter((u) => u.startsWith(here.href) && !u.includes("/vendor/"));
  // Left out on purpose: the list of code equivalences (bookkeeping about fingerprints, not code that makes tracts; were
  // it in, adding an equivalence would change the value it describes) and the tract table's data file (in the labeler
  // field by its own hash).
  const ASIDE = new Set(["code-equivalences.json", "naming/tractcloud-42.json"]);
  for (const u of deno) if (!ASIDE.has(u.slice(here.href.length))) assert(g.includes(u), `${u.slice(here.href.length)} is in Deno's graph but not ours`);
});

// THE SECOND OPINION AT IMPORT (2026-10-28, Ron: "yes, close both gaps"): the job's own call, on a real scan's files
// when the Martinos Prisma database is on this machine (anonymized, public: CLAUDE.md), and never a throw.
Deno.test("the import job's second opinion: dcm2niix agrees on a real scan; a scan without files is an answer, not a throw", async () => {
  const { secondOpinionOf, readSeries, withScanChecks } = await import("./import-job.ts");
  const { fromDicomVolumes } = await import("./dwi.ts");
  const { setDicomLibrary, dcmjs, indexSeries } = await import("albula/server");
  await import("./hooks.ts");
  setDicomLibrary(dcmjs);
  const none = await secondOpinionOf("/nonexistent", "1.2.3", { bValues: [], gradients: [], volumes: [] } as unknown as DiffusionSeries);
  assert("error" in none && none.program === "dcm2niix", JSON.stringify(none));
  // The record's own answer wins over the scan's attribute; the attribute fills a record without one.
  assertEquals(withScanChecks({ secondOpinion: { agree: true } }, { agree: false }).secondOpinion, { agree: true });
  assertEquals(withScanChecks({}, { agree: false }).secondOpinion, { agree: false });
  const dir = `${Deno.env.get("HOME")}/SlicerAlbulaDatabases/Martinos Prisma`;
  const row = await indexSeries(dir).then((r) => r.find((x) => x.description === "dMRI_dir98_AP")).catch(() => undefined);
  if (!row) { console.log("Martinos Prisma not on this machine: the real-scan half is skipped"); return; }
  const s = await readSeries(dir, row);
  const r = await secondOpinionOf(dir, row.seriesUID, fromDicomVolumes(s.frames, row.description));
  assert("agree" in r && r.agree && r.volumes === 99 && r.worstDeg === 0, JSON.stringify(r));
});

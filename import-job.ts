// THE FIBER TRACTS MADE AT IMPORT TIME (Contents/docs/DMRI-AT-IMPORT.md in the workspace; Ron, 2026-10-05: "go ahead
// with the import-time job"; "quality takes precedence over speed. That is why we do the slow stuff at import time").
// For each diffusion scan in a DICOM database: the whole-brain pipeline (pipeline.ts, the one the case runs and the
// regression test use) on the scan, its reversed phase-encoding partner and the T1, with SynthStrip's brain from
// haversack, and the named tracts stored as ONE DICOM Tractography Results object (tracts-dicom.ts) filed under the scan,
// recording what made it. The resident's button then only measures stored tracts against the tumor.
//
// It runs as a program beside Albula's server (sdk/server.ts): it reads the index and the files itself, read-only, and
// writes and indexes only through the server's `_write` route (the index's lock lives in the server process). This file
// is the library; import-job-main.ts is the program. The critic's round of 2026-10-05 (qa/2026-10-05-dmri-import-job.md)
// shaped it: a study at a time (finding 3), the module's rules for the T1 and the partner, tested (6, 7, 14), stored
// tracts current only for the same code, rules and inputs (1, 11), states that say what will and will not change (13).
import { packRGB24, parseInstances, synthstripBrainMask, startSegmentationServer, volumesOfSeries, type BrainMask, type Volume } from "albula";
import { indexSeries, seriesFilePaths, writeNrrd, type IndexSeries } from "albula/server";
import { fromDicomVolumes, type DiffusionSeries } from "./dwi.ts";
import { DIRECTION_CHECK_RULE } from "./gradient-check.ts";
import { wholeBrainTracts, PIPELINE_MAX_B } from "./pipeline.ts";
import { DISTORTION_RULE } from "./distortion.ts";
import { MOTION_RULE } from "./motion.ts";
import { REGISTRATION_RULE } from "./registration.ts";
import { TRACKING_RULE } from "./tracking-rules.ts";
import { OUTSIDE_RULE } from "./outside-brain.ts";
import { stageText, type StageTimes } from "./planning.ts";
import { colorFA, type TensorFit } from "./tensor.ts";
import { SHORT } from "./tractcloud/name-tracts.ts";
import { tractColor, TRACT_COLORS_VERSION } from "./tractcloud/tract-colors.ts";
import type { TractCloudModel } from "./tractcloud/tractcloud.ts";
import type { RapidParcModel } from "./rapidparc/rapidparc.ts";
import { b0Path, colorFaPath, dicomToTracts, tractsToDicom, UNNAMED, type TractSetData } from "./tracts-dicom.ts";

/** The description every stored tracts object carries: how the job finds its own objects in the index. */
export const TRACTS_DESCRIPTION = "Fiber tracts (whole brain)";
/** A series larger than this is not read to be classified (a 4D fMRI, a long dynamic study): said, not tried. */
const MAX_SERIES_FILES = 4000;

// ── What decides whether stored tracts are current ───────────────────────────────────────────────────────────────

/**
 * EVERYTHING THAT MADE THE TRACTS: the rule of each step, a fingerprint of the code (`code`: the extension's modules that
 * shape the result, hashed by the program; critic finding 1 -- a fix that moves the results without a new rule number
 * must still make stored tracts stale), the naming network's weights and table, and SynthStrip's version. Stored tracts
 * are current only when all of it AND the inputs are the same (Ron, 2026-10-05: "remake automatically, versioned").
 */
export function jobRules(versions: { code: string; labeler: string; synthstrip: string }): Record<string, string | number> {
  return { directions: DIRECTION_CHECK_RULE, distortion: DISTORTION_RULE, motion: MOTION_RULE, registration: REGISTRATION_RULE, tracking: TRACKING_RULE,
    outside: OUTSIDE_RULE.on ? OUTSIDE_RULE.id : 0, maxB: PIPELINE_MAX_B, tractColors: TRACT_COLORS_VERSION,
    code: versions.code, labeler: versions.labeler, synthstrip: versions.synthstrip };
}
/** The inputs a case was made from: a reversed scan or a T1 that arrives later makes the case stale (finding 11). */
export interface CaseInputs { diffusion: string; partner: string | null; t1: string | null }
/** Same rules (every key, both ways) and same inputs. */
export function isCurrent(stored: { rules?: Record<string, unknown>; inputs?: Partial<CaseInputs> } | undefined, rules: Record<string, unknown>, inputs: CaseInputs): boolean {
  const a = stored?.rules;
  if (!a || Object.keys(a).length !== Object.keys(rules).length || !Object.keys(rules).every((k) => String(a[k]) === String(rules[k]))) return false;
  const i = stored?.inputs ?? {};
  return i.diffusion === inputs.diffusion && (i.partner ?? null) === inputs.partner && (i.t1 ?? null) === inputs.t1;
}

// ── What a series is (from its volumes; the rules tested in import-job.test.ts) ───────────────────────────────────

/** What the job needs to know about a series to choose: from its volumes and the first header, no pixels. */
export interface SeriesFacts {
  uid: string; studyUID: string; patientUID: string; description: string;
  volumes: number; bValues: number[];
  /** Distinct gradient directions among the volumes with b > 50. */
  directions: number;
  /** The scanner's record of the phase-encoding direction ("j-", "ROW", …) when there is one. */
  phaseEncoding?: string;
  /** The first volume's placement (4×4 ijkToRAS). */
  ijkToRAS: number[];
  /** A computed map (ImageType ADC, TRACE, FA, …), or an image Albula made: never the anatomy. */
  derived: boolean;
}
/** A diffusion scan worth a whole-brain run: 7 volumes or more, 6 gradient directions or more (the tensor's minimum). */
export const isDiffusionScan = (f: SeriesFacts) => f.volumes >= 7 && f.directions >= 6;
const isB0Only = (f: SeriesFacts) => f.bValues.length >= 1 && f.bValues.every((b) => Number.isFinite(b) && b < 50);
const hasB0 = (f: SeriesFacts) => f.bValues.some((b) => Number.isFinite(b) && b < 50);
/** Placed as the scan is: origin within 30 mm, each axis within 15° (module.ts partnerFor). */
export function samePlacement(A: number[], B: number[]): boolean {
  if (Math.hypot(A[3] - B[3], A[7] - B[7], A[11] - B[11]) > 30) return false;
  for (let c = 0; c < 3; c++) {
    const u = [A[c], A[4 + c], A[8 + c]], w = [B[c], B[4 + c], B[8 + c]], cos = Math.abs(u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (Math.hypot(...u) * Math.hypot(...w));
    if (!(cos > Math.cos(15 * Math.PI / 180))) return false;
  }
  return true;
}
/** The recorded phase-encoding directions say the two are a reversed pair: the same axis, opposite signs. */
function recordedReversed(a?: string, b?: string): boolean | undefined {
  const axis = (d: string) => d === "ROW" ? "i" : d === "COL" || d === "COLUMN" ? "j" : d[0], signed = (d: string) => /^[ijk]-?$/.test(d);
  if (!a || !b) return undefined;
  if (axis(a) !== axis(b)) return false;
  return signed(a) && signed(b) ? a.endsWith("-") !== b.endsWith("-") : undefined;
}
/**
 * THE REVERSED PARTNER of a diffusion scan (finding 7): a series of the same study placed as the scan is, holding
 * b = 0 images, and not itself a bigger diffusion scan. First choice: one the scanner's record calls reversed; then a
 * b = 0-only series; then a smaller diffusion series with b = 0 images. One the record calls NOT reversed is skipped.
 */
export function partnerOf(dwi: SeriesFacts, all: SeriesFacts[]): SeriesFacts | undefined {
  const c = all.filter((s) => s.uid !== dwi.uid && s.studyUID === dwi.studyUID && hasB0(s) && s.volumes < dwi.volumes && samePlacement(s.ijkToRAS, dwi.ijkToRAS)
    && recordedReversed(dwi.phaseEncoding, s.phaseEncoding) !== false);
  return c.find((s) => recordedReversed(dwi.phaseEncoding, s.phaseEncoding) === true) ?? c.find(isB0Only) ?? c[0];
}
/** A T1 by the series' name (face.ts pickAnatomy's rule), or a 3D T1 sequence's usual names. */
export const T1_NAME = /(^|[^a-z0-9])t1([^0-9]|$)|mprage|mp-rage|bravo|spgr|tfl3d|t1w/i;
/**
 * THE MRI OF THE ANATOMY for a scan (finding 6): one volume, not diffusion, not b = 0, not computed, of the same study --
 * else of the same patient -- and named as a T1. Quality first: with no image named so, none is chosen and the case
 * waits (the module, which shows its choice to the person, takes the first image instead).
 */
export function anatomyOf(dwi: SeriesFacts, all: SeriesFacts[]): SeriesFacts | undefined {
  const ok = (s: SeriesFacts) => s.uid !== dwi.uid && s.volumes === 1 && !s.derived && !s.bValues.some((b) => Number.isFinite(b)) && T1_NAME.test(s.description);
  return all.find((s) => ok(s) && s.studyUID === dwi.studyUID) ?? all.find((s) => ok(s) && !!dwi.patientUID && s.patientUID === dwi.patientUID);
}
/** The cases of a set of series: every diffusion scan that is not itself another scan's partner. */
export function planCases(all: SeriesFacts[]): { dwi: SeriesFacts; partner?: SeriesFacts; t1?: SeriesFacts }[] {
  const scans = all.filter(isDiffusionScan).sort((a, b) => b.volumes - a.volumes);
  const used = new Set<string>(), out: { dwi: SeriesFacts; partner?: SeriesFacts; t1?: SeriesFacts }[] = [];
  for (const dwi of scans) {
    if (used.has(dwi.uid)) continue;
    const partner = partnerOf(dwi, all), t1 = anatomyOf(dwi, all);
    if (partner) used.add(partner.uid);
    out.push({ dwi, ...(partner ? { partner } : {}), ...(t1 ? { t1 } : {}) });
  }
  return out;
}

// ── Reading ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** One series read through the core reader, with what the tracts object needs from its headers. */
export interface ReadSeries {
  facts: SeriesFacts; frames: Volume[];
  /** The first instance's header as DICOM JSON (patient and study attributes, frame of reference). */
  header: Record<string, { vr?: string; Value?: unknown[] }>;
  instances: { sopClassUID: string; sopInstanceUID: string }[];
}
const metaOf = (v: Volume) => (v.meta ?? {}) as Record<string, unknown>;
const diffusionOf = (v: Volume) => metaOf(v).diffusion as { bValue?: number; gradient?: number[]; phaseEncoding?: string } | undefined;

/** Read a series' files and make its volumes (the duckn copy writer's path; the extension's hooks are loaded by the program). */
export async function readSeries(dbDir: string, row: IndexSeries): Promise<ReadSeries> {
  const paths = await seriesFilePaths(dbDir, row.seriesUID);
  if (!paths.length) throw new Error("no files in the index");
  if (paths.length > MAX_SERIES_FILES) throw new Error(`${paths.length} files: too large to be a diffusion scan, not read`);
  const buffers = await Promise.all(paths.map(async (p) => { const b = await Deno.readFile(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; }));
  const instances = await parseInstances(buffers, { headers: true });
  if (!instances.length) throw new Error(`none of its ${paths.length} files could be read as images`);
  const { frames } = volumesOfSeries(instances);
  const header = (instances[0].header?.json ?? {}) as ReadSeries["header"];
  const seen = new Set<string>(), refs: ReadSeries["instances"] = [];
  for (const i of instances) {
    const sop = i.sopInstanceUID, cls = String((i.header?.json as ReadSeries["header"] | undefined)?.["00080016"]?.Value?.[0] ?? "");
    if (sop && !seen.has(sop)) { seen.add(sop); refs.push({ sopClassUID: cls, sopInstanceUID: sop }); }
  }
  const bValues = frames.map((v) => Number(diffusionOf(v)?.bValue ?? NaN));
  const dirs = new Set(frames.filter((v) => Number(diffusionOf(v)?.bValue ?? 0) > 50).map((v) => (diffusionOf(v)?.gradient ?? []).map((x) => Math.round(Number(x) * 100) / 100).join(",")));
  const maker = String(header["00080070"]?.Value?.[0] ?? "");
  // A computed map by its ImageType values (ADC, trace, FA, exponential ADC), or an image Albula made. Not "DERIVED" alone:
  // a scan converted to DICOM (the BIDS import, other converters) says DERIVED and is still the scan.
  const derived = (header["00080008"]?.Value ?? []).map((x) => String(x).toUpperCase()).some((v) => ["ADC", "TRACE", "TRACEW", "FA", "EXP", "EADC", "CALC_BV", "COLFA"].includes(v)) || maker === "SlicerAlbula";
  return { facts: { uid: row.seriesUID, studyUID: row.studyUID, patientUID: row.patientUID, description: row.description, volumes: frames.length, bValues, directions: dirs.size,
    ...(frames[0] && diffusionOf(frames[0])?.phaseEncoding ? { phaseEncoding: diffusionOf(frames[0])!.phaseEncoding } : {}), ijkToRAS: frames[0]?.ijkToRAS ?? [], derived }, frames, header, instances: refs };
}

/** The studies to look at: those with an MR series that is not one of ours, with their MR series (and the patient's). */
export async function studiesOf(dbDir: string, only?: string): Promise<{ rows: IndexSeries[]; studies: string[] }> {
  const rows = await indexSeries(dbDir);
  const mr = (r: IndexSeries) => r.modality === "MR" && r.description !== TRACTS_DESCRIPTION;
  let studies = [...new Set(rows.filter(mr).map((r) => r.studyUID))];
  if (only) { const st = rows.find((r) => r.seriesUID === only)?.studyUID; studies = studies.filter((s) => s === st); }
  return { rows, studies };
}

/**
 * ONE STUDY'S CASES, read when its turn comes and let go after it (finding 3): every MR series of the study, and of the
 * patient's other studies when the study has no T1 (the module takes an image of the same patient too).
 */
export async function planStudy(dbDir: string, rows: IndexSeries[], study: string, onLine?: (line: string, why?: string) => void): Promise<{ dwi: ReadSeries; partner?: ReadSeries; t1?: ReadSeries }[]> {
  const mr = (r: IndexSeries) => r.modality === "MR" && r.description !== TRACTS_DESCRIPTION;
  const own = rows.filter((r) => mr(r) && r.studyUID === study);
  const read = new Map<string, ReadSeries>();
  const readAll = async (list: IndexSeries[]) => {
    for (const r of list) {
      if (read.has(r.seriesUID)) continue;
      try { read.set(r.seriesUID, await readSeries(dbDir, r)); onLine?.("read a series"); }
      catch (e) { onLine?.("a series could not be read", (e as Error).message); }
    }
  };
  await readAll(own);
  let cases = planCases([...read.values()].map((s) => s.facts));
  if (!cases.length) return [];
  // No T1 in the study: the patient's other studies (same patient, module.ts anatomyFor).
  if (cases.some((c) => !c.t1)) {
    const patient = own[0]?.patientUID;
    if (patient) { await readAll(rows.filter((r) => mr(r) && r.studyUID !== study && r.patientUID === patient)); cases = planCases([...read.values()].map((s) => s.facts)).filter((c) => c.dwi.studyUID === study); }
  }
  return cases.map((c) => ({ dwi: read.get(c.dwi.uid)!, ...(c.partner ? { partner: read.get(c.partner.uid)! } : {}), ...(c.t1 ? { t1: read.get(c.t1.uid)! } : {}) }));
}

/** The stored tracts objects of a diffusion series, with what made them (from the index as it is NOW: finding 8). */
export async function storedTracts(dbDir: string, dwiUID: string): Promise<{ seriesUID: string; seriesNumber: number; rules?: Record<string, unknown>; inputs?: Partial<CaseInputs>; made?: string }[]> {
  const rows = await indexSeries(dbDir), study = rows.find((y) => y.seriesUID === dwiUID)?.studyUID;
  const out: { seriesUID: string; seriesNumber: number; rules?: Record<string, unknown>; inputs?: Partial<CaseInputs>; made?: string }[] = [];
  for (const r of rows.filter((x) => x.description === TRACTS_DESCRIPTION && x.studyUID === study)) {
    for (const p of await seriesFilePaths(dbDir, r.seriesUID)) {
      try {
        const t = await dicomToTracts(await Deno.readFile(p));
        if (t.referencedSeries === dwiUID) out.push({ seriesUID: r.seriesUID, seriesNumber: Number(r.seriesNumber) || 0, rules: t.provenance?.rules as Record<string, unknown> | undefined, inputs: t.provenance?.inputs as Partial<CaseInputs> | undefined, made: String(t.provenance?.made ?? "") });
      } catch { /* not a readable tracts object of ours */ }
    }
  }
  return out.sort((a, b) => (b.made ?? "").localeCompare(a.made ?? ""));
}

// ── The brain mask ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * SynthStrip's brain on the T1, through the server's haversack proxy (`server` the server's root URL). Starts the
 * segmentation server when none answers. Quality first: no fallback to the scan's own mask. `kind` says whether waiting
 * can help: "waiting" (the server is not answering), "cannot" (no SynthStrip on this Mac), "failed" (it ran and failed).
 */
export async function brainOnT1(server: string, t1: ReadSeries, onProgress?: (line: string) => void): Promise<{ mask?: BrainMask; why?: string; kind?: "waiting" | "cannot" | "failed" }> {
  const root = server.replace(/\/+$/, ""), transport = { fetch: (...a: Parameters<typeof fetch>) => fetch(...a), base: `${root}/_haversack/` };
  // gzip INSIDE the NRRD, as the page sends it (logic/export.ts); haversack's reader will not take ".nrrd.gz".
  const upload = async () => ({ bytes: await writeNrrd(t1.frames[0], { encoding: "gzip" }), filename: "t1.nrrd" });
  const stub = { nodes: new Map() } as unknown as Parameters<typeof synthstripBrainMask>[0];
  const key = `t1:${t1.facts.uid}`;
  let r = await synthstripBrainMask(stub, key, onProgress, { transport, upload });
  const toRoot = { fetch: (input: Parameters<typeof fetch>[0], init?: RequestInit) => fetch(typeof input === "string" && input.startsWith("/") ? `${root}${input}` : input, init), base: transport.base };
  if (!r.ok && r.reason === "no-server") {
    onProgress?.("starting the segmentation server");
    const s = await startSegmentationServer(onProgress, toRoot);
    if (s.ok) r = await synthstripBrainMask(stub, key, onProgress, { transport, upload });
  }
  // A STUCK SERVER IS NOT RESTARTED FROM HERE (critic 2026-10-06, finding 12): a restart re-queues whatever else the server
  // holds and loses its progress -- a person's decision, from the Diffusion panel's button. The case waits, said so.
  if (r.ok) return { mask: r.mask };
  return { why: r.message, kind: r.reason === "no-synthstrip" ? "cannot" : r.reason === "failed" ? "failed" : "waiting" };
}
/**
 * THE CODE THAT MAKES THE TRACTS, hashed (critic 2026-10-06, R2-4): the files the job's own module reaches through its
 * relative imports (static, dynamic, and workers by `new URL("./x.ts", import.meta.url)`), not the whole folder -- a
 * change to the review's drawing or the panel must not remake every case and send Ron's verdicts to "older tracts".
 */
export async function importGraph(entry: URL): Promise<string[]> {
  const seen = new Set<string>(), todo = [entry.href];
  // `from "./x"`, `import("./x")`, a bare `import "./x"` (critic 2026-10-06, R3-2: hooks.ts, which brings the vendors'
  // b-value and gradient readers, came in that way and was missed), and workers by `new URL("./x", import.meta.url)`.
  const spec = /(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+|new URL\(\s*)"(\.{1,2}\/[^"]+\.(?:ts|wgsl))"/g;
  while (todo.length) {
    const u = todo.pop()!;
    if (seen.has(u)) continue;
    seen.add(u);
    if (!u.endsWith(".ts")) continue;
    // Comments out first (this one's own example would count as an import).
    const text = (await Deno.readTextFile(new URL(u))).replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:"'])\/\/.*$/gm, "$1");
    for (const m of text.matchAll(spec)) todo.push(new URL(m[1], u).href);
  }
  return [...seen].sort();
}
/** haversack's and SynthStrip's versions, as the server reports them (finding 10). */
export async function synthstripVersion(server: string, opts: { waitMs?: number; pollMs?: number; start?: boolean } = {}): Promise<string | undefined> {
  // THE VERSION IS PART OF THE STALENESS KEY (jobRules), so an unknown one must not be recorded: a segmentation server
  // still starting answered "?" on 2026-10-06, and every case made then would have counted as stale ever after. So:
  // the server is started when it does not answer (as brainOnT1 did; critic 2026-10-06, finding 14); a server without
  // SynthStrip (a Mac without the developer tools) is said as such -- its cases come out "cannot be made on this Mac";
  // otherwise asked again until both parts are known, up to `waitMs` (default 5 minutes); undefined when never.
  // SynthStrip's weights are reported once on disk; before its first job they are "not yet installed" -- makeTracts asks
  // again after its first brain mask and records the real version (critic 2026-10-06, R2-5).
  const root = server.replace(/\/+$/, ""), until = Date.now() + (opts.waitMs ?? 5 * 60_000);
  let started = false;
  for (;;) {
    const st = await fetch(`${root}/_haversack/_status`).then((r) => r.json()).catch(() => ({})) as { reachable?: boolean; health?: { version?: string } };
    const v = st.health?.version;
    if (!v && !started && opts.start !== false) {
      started = true;
      await startSegmentationServer(undefined, { fetch: (input, init) => fetch(typeof input === "string" && input.startsWith("/") ? `${root}${input}` : input, init), base: `${root}/_haversack/` });
      continue;
    }
    if (v) {
      const res = await fetch(`${root}/_haversack/tasks/synthstrip:mask`).catch(() => null);
      if (res && res.status === 404) return `haversack ${v}, no SynthStrip`;
      const task = res?.ok ? await res.json().catch(() => ({})) as { weights_installed?: { id?: string; version?: string }[] } : undefined;
      if (task) return `haversack ${v}, synthstrip weights ${task.weights_installed?.find((x) => x.id === "synthstrip")?.version ?? "not yet installed"}`;
    }
    if (Date.now() >= until) return undefined;
    await new Promise((r) => setTimeout(r, opts.pollMs ?? 5000));
  }
}

// ── One case ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** What happened to one diffusion scan; `why` in plain words. */
export type CaseOutcome =
  | { state: "made" | "would be made"; seriesUID: string; streamlines: number; stored: number; named: number; seconds: number; said: string }
  | { state: "current"; seriesUID: string }
  | { state: "waiting" | "cannot" | "failed"; why: string };

/** The patient and study attributes the tracts object copies from the scan (PersonName as its alphabetic form). */
const PATIENT_STUDY: [string, string][] = [["00100010", "PatientName"], ["00100020", "PatientID"], ["00100030", "PatientBirthDate"], ["00100040", "PatientSex"],
  ["0020000D", "StudyInstanceUID"], ["00080020", "StudyDate"], ["00080030", "StudyTime"], ["00200010", "StudyID"], ["00080050", "AccessionNumber"], ["00080090", "ReferringPhysicianName"]];
function patientStudyOf(h: ReadSeries["header"]): Record<string, unknown> {
  const o: Record<string, unknown> = {};
  for (const [tag, key] of PATIENT_STUDY) {
    const v = h[tag]?.Value?.[0];
    o[key] = v === undefined ? "" : typeof v === "object" && v && "Alphabetic" in (v as Record<string, unknown>) ? (v as { Alphabetic: string }).Alphabetic : v;
  }
  return o;
}

/** The track sets: one per named tract and side, and one "Unnamed" (too short to name, or no tract). */
export function trackSets(sl: Float32Array[], tract: Int32Array, side: Int8Array, outside: Uint8Array, names: { name: string }[]): TractSetData[] {
  const count = names.length, OTHER = count - 1, groups = new Map<string, TractSetData>();
  for (let i = 0; i < sl.length; i++) {
    const t = tract[i], named = t !== SHORT && t !== OTHER && !outside[i];
    const s = (named ? Math.sign(side[i]) : 0) as -1 | 0 | 1, key = named ? `${t}:${s}` : "unnamed";
    let g = groups.get(key);
    if (!g) { const c = tractColor(named ? t : -1, count); g = { label: named ? names[t].name : UNNAMED, side: s, color: [c[0], c[1], c[2]], streamlines: [] }; groups.set(key, g); }
    g.streamlines.push(sl[i]);
  }
  return [...groups.values()].sort((a, b) => (a.label === UNNAMED ? 1 : b.label === UNNAMED ? -1 : a.label.localeCompare(b.label) || a.side - b.side));
}

/**
 * ONE CASE: the pipeline on the planned scan, its tracts written and indexed through the server -- unless tracts made
 * by the same code, rules and inputs are already stored.
 */
export async function makeTracts(dbDir: string, dbId: string, server: string, plan: { dwi: ReadSeries; partner?: ReadSeries; t1?: ReadSeries }, device: GPUDevice, model: TractCloudModel,
  labeler: RapidParcModel, versions: { code: string; labeler: string; synthstrip: string }, onProgress?: (line: string) => void, opts: { force?: boolean; dryRun?: boolean } = {}): Promise<CaseOutcome> {
  let rules = jobRules(versions);
  const inputs: CaseInputs = { diffusion: plan.dwi.facts.uid, partner: plan.partner?.facts.uid ?? null, t1: plan.t1?.facts.uid ?? null };
  const stored = await storedTracts(dbDir, plan.dwi.facts.uid);
  // CURRENT = the same code, rules and inputs AND its Color FA beside it (critic 2026-10-06, finding 16: a job stopped
  // between storing the tracts and the map left tracts that counted as current forever, without the map).
  // ... and the b = 0 image beside it (2026-10-06: the substantia nigra for the tract review).
  const hasMap = async (uid: string) => !!(await Deno.stat(`${dbDir}/${colorFaPath(uid)}`).catch(() => null)) && !!(await Deno.stat(`${dbDir}/${b0Path(uid)}`).catch(() => null));
  let current: (typeof stored)[number] | undefined;
  for (const s of stored) if (isCurrent(s, rules, inputs) && await hasMap(s.seriesUID)) { current = s; break; }
  if (current && !opts.force) return { state: "current", seriesUID: current.seriesUID };
  // QUALITY FIRST: the tracking's brain is SynthStrip's on the T1; without the T1 the case waits for it.
  if (!plan.t1) return { state: "waiting", why: "waiting for an MRI of the anatomy named as a T1 (same study or patient)" };
  const t0 = performance.now();
  onProgress?.("finding the brain on the MRI of the anatomy");
  const brain = await brainOnT1(server, plan.t1, onProgress);
  if (!brain.mask) return { state: brain.kind ?? "waiting", why: `the brain on the MRI of the anatomy: ${brain.why}` };
  // SYNTHSTRIP'S WEIGHTS ARRIVE WITH ITS FIRST JOB (critic 2026-10-06, R2-5): on a fresh installation the version asked
  // before the run says "not yet installed". Asked again now that a mask was made; the shared `versions` is updated, so
  // this case and every later one record the real version (and the next run does not remake them all).
  if (/not yet installed/.test(versions.synthstrip)) {
    const v = await synthstripVersion(server, { waitMs: 0, start: false });
    if (v) { versions.synthstrip = v; rules = jobRules(versions); }
  }
  const dwi = fromDicomVolumes(plan.dwi.frames, plan.dwi.facts.description);
  const t1v = plan.t1.frames[0], stages: StageTimes = {};
  onProgress?.("correcting, aligning and tracking the whole brain");
  const r = await wholeBrainTracts({ dwi,
    ...(plan.partner ? { partner: { b0s: plan.partner.frames.filter((v) => Number(diffusionOf(v)?.bValue ?? 0) < 50).map((v) => v.data as ArrayLike<number>), grid: { dims: plan.partner.frames[0].dims, ijkToRAS: plan.partner.frames[0].ijkToRAS }, name: "the reversed scan" } } : {}),
    phaseEncoding: { scan: plan.dwi.facts.phaseEncoding, partner: plan.partner?.facts.phaseEncoding },
    t1: { dims: t1v.dims as [number, number, number], ijkToRAS: t1v.ijkToRAS, data: t1v.data as ArrayLike<number> },
    brainT1: brain.mask }, device, model, { labeler }, stages);
  const sets = trackSets(r.sl, r.named.tract, r.named.side, r.outside, model.json.tracts);
  const named = sets.filter((s) => s.label !== UNNAMED).reduce((n, s) => n + s.streamlines.length, 0);
  // On the T1's axes when aligned: the streamlines are in the T1's space, so the T1's frame of reference.
  const aligned = !!r.alignment && !r.alignment.doubt;
  const forUID = String(((aligned ? plan.t1.header : plan.dwi.header)["00200052"]?.Value?.[0]) ?? "");
  const seconds = (performance.now() - t0) / 1000;
  const provenance = { rules, inputs, made: new Date().toISOString(), seconds: +seconds.toFixed(1), stages: stageText(stages),
    streamlines: r.sl.length, onePointLeftOut: r.sl.filter((p) => p.length < 6).length, corrected: r.corrected, alignedToT1: aligned,
    trackingRuleApplied: r.rule.id, motionRuleApplied: r.prep.motionRule, ...(r.prep.directions ? { directions: { rule: r.prep.directions.rule, verdict: r.prep.directions.verdict, used: r.prep.directions.best.label, Q: r.prep.directions.best.Q, recordOverBest: r.prep.directions.recordOverBest } } : {}), ...(r.alignment ? { alignment: r.alignment } : {}), brain: r.fit.seedMaskRule ?? r.fit.maskRule };
  // Several objects can follow each other under one scan (remade after a change): numbered from 900 by their time -- one
  // above the highest stored, so a deleted older one never makes the newest share a number (critic 2026-10-06, R2-7).
  const seriesNumber = Math.max(899, ...stored.map((x) => x.seriesNumber)) + 1;
  const written = await tractsToDicom(sets, { patientStudy: patientStudyOf(plan.dwi.header), frameOfReferenceUID: forUID, seriesInstanceUID: plan.dwi.facts.uid, instances: plan.dwi.instances,
    alsoReferenced: [...(aligned ? [{ seriesInstanceUID: plan.t1.facts.uid, instances: plan.t1.instances }] : []), ...(plan.partner ? [{ seriesInstanceUID: plan.partner.facts.uid, instances: plan.partner.instances }] : [])] },
    { algorithmName: "UKF two-tensor (Albula's port of UKFTractography)", algorithmVersion: `tracking rule ${r.rule.id}`, algorithmParameters: JSON.stringify(r.rule.ukf).slice(0, 10240),
      model: "multi", provenance, seriesDescription: TRACTS_DESCRIPTION, seriesNumber });
  // A streamline of one point (a seed that stopped at once) has no line to store; the writer leaves it out, and says so.
  const onePoint = r.sl.length - written.tracks;
  const said = `${r.sl.length} streamlines (${written.tracks} stored${onePoint ? `, ${onePoint} of a single point left out` : ""}), ${named} named into ${sets.length - (sets.some((s) => s.label === UNNAMED) ? 1 : 0)} tracts, in ${seconds.toFixed(0)} s`;
  const out = { seriesUID: written.seriesInstanceUID, streamlines: r.sl.length, stored: written.tracks, named, seconds, said };
  if (opts.dryRun) return { state: "would be made", ...out };
  const now = new Date(), d2 = (n: number) => String(n).padStart(2, "0");
  const meta = { sopInstanceUID: written.sopInstanceUID, seriesInstanceUID: written.seriesInstanceUID, studyInstanceUID: plan.dwi.facts.studyUID, modality: "MR",
    seriesNumber, seriesDate: `${now.getFullYear()}${d2(now.getMonth() + 1)}${d2(now.getDate())}`, seriesTime: `${d2(now.getHours())}${d2(now.getMinutes())}${d2(now.getSeconds())}`,
    seriesDescription: TRACTS_DESCRIPTION, frameOfReferenceUID: forUID, derivedFrom: { parentSeriesUID: plan.dwi.facts.uid, kind: "tracts", label: "Fiber tracts" } };
  const res = await fetch(`${server.replace(/\/+$/, "")}/_db/${encodeURIComponent(dbId)}/_write/${encodeURIComponent(`tracts-${written.seriesInstanceUID}.dcm`)}`,
    { method: "POST", body: written.bytes as unknown as BodyInit, headers: { "content-type": "application/dicom", "x-albula-index": encodeURIComponent(JSON.stringify(meta)) } });
  if (!res.ok) return { state: "failed", why: `the server did not store the tracts (${res.status})` };
  // THE DIRECTION-COLORED MAP BESIDE THEM (the tract review, Contents/docs/TRACT-REVIEW.md, Ron's "1 a"): on the grid the
  // tracts are on, after every correction, so a viewer needs no fit. A display aid, regenerable from the scan: the
  // database's cache folder, named by the tracts' series. Its failure does not undo the tracts.
  try { await writeColorFA(dbDir, written.seriesInstanceUID, r.fit); await writeB0(dbDir, written.seriesInstanceUID, r.dwi, r.fit); }
  catch (e) { return { state: "made", ...out, said: `${out.said}; the direction-colored map or the b = 0 image was not stored (${(e as Error).message})` }; }
  return { state: "made", ...out };
}

export { b0Path, colorFaPath };

/** The Color FA of `fit` (tensor.ts colorFA: FA times the principal direction's components), one byte a color packed
 *  into one float sample as the slice views draw it (packRGB24, the Diffusion module's own form), as a gzipped NRRD. */
/** Write a cache file whole or not at all (critic 2026-10-06, b = 0 finding 2: a half-written file counted as current and
 *  the review could not open the case): a temporary name beside it, then renamed over. */
async function writeCacheFile(dbDir: string, rel: string, bytes: Uint8Array): Promise<void> {
  await Deno.mkdir(`${dbDir}/SlicerAlbula-Cache`, { recursive: true });
  const tmp = `${dbDir}/SlicerAlbula-Cache/.${crypto.randomUUID()}.tmp`;
  try { await Deno.writeFile(tmp, bytes); await Deno.rename(tmp, `${dbDir}/${rel}`); }
  catch (e) { await Deno.remove(tmp).catch(() => {}); throw e; }
}

/** The b = 0 image on the tracts' grid: the MEAN OF THE MEASURED b = 0 images of the corrected scan (heavily
 *  T2-weighted, so the substantia nigra and the red nucleus show dark from their iron -- Ron, 2026-10-06), not the tensor
 *  fit's S0, which runs a few percent low at the fluid's edges (critic 2026-10-06, b = 0 finding 4). The fit's S0 only
 *  when the scan is not on the fit's grid or has no b = 0 image. */
export function measuredB0(dwi: DiffusionSeries, fit: TensorFit): Float32Array {
  const zeros = dwi.bValues.map((b, i) => (b < 50 ? i : -1)).filter((i) => i >= 0);
  const n = fit.dims[0] * fit.dims[1] * fit.dims[2];
  const sameGrid = dwi.volumes[0] && dwi.volumes[0].dims.every((d, i) => d === fit.dims[i]) && dwi.ijkToRAS.every((m, i) => Math.abs(m - fit.ijkToRAS[i]) < 1e-4);
  if (!zeros.length || !sameGrid) return Float32Array.from(fit.S0);
  const out = new Float32Array(n);
  for (const i of zeros) { const d = dwi.volumes[i].data as ArrayLike<number>; for (let v = 0; v < n; v++) out[v] += d[v] / zeros.length; }
  return out;
}
export async function writeB0(dbDir: string, tractsSeriesUID: string, dwi: DiffusionSeries, fit: TensorFit): Promise<void> {
  const bytes = await writeNrrd({ dims: fit.dims, ijkToRAS: fit.ijkToRAS, data: measuredB0(dwi, fit), dtype: "<f4" } as Volume, { encoding: "gzip" });
  await writeCacheFile(dbDir, b0Path(tractsSeriesUID), bytes);
}
/** The Color FA of `fit` (tensor.ts colorFA: FA times the principal direction's components), one byte a color packed
 *  into one float sample as the slice views draw it (packRGB24, the Diffusion module's own form), as a gzipped NRRD. */
export async function writeColorFA(dbDir: string, tractsSeriesUID: string, fit: TensorFit): Promise<void> {
  const rgb = colorFA(fit), n = fit.fa.length, packed = new Float32Array(n), q = (x: number) => Math.max(0, Math.min(255, Math.round(x * 255)));
  for (let v = 0; v < n; v++) packed[v] = packRGB24(q(rgb[3 * v]), q(rgb[3 * v + 1]), q(rgb[3 * v + 2]));
  const bytes = await writeNrrd({ dims: fit.dims, ijkToRAS: fit.ijkToRAS, data: packed, dtype: "<f4" } as Volume, { encoding: "gzip" });
  await writeCacheFile(dbDir, colorFaPath(tractsSeriesUID), bytes);
}

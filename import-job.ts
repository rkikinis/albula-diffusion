// THE FIBER TRACTS MADE AT IMPORT TIME (Contents/docs/DMRI-AT-IMPORT.md in the workspace; Ron, 2026-10-05: "go ahead
// with the import-time job"; "quality takes precedence over speed. That is why we do the slow stuff at import time").
// For each diffusion scan in a DICOM database: the whole-brain pipeline (pipeline.ts, the one the case runs and the
// regression test use) on the scan, its reversed phase-encoding partner and the T1 of the same study, with SynthStrip's
// brain from haversack, and the named tracts stored as ONE DICOM Tractography Results object (tracts-dicom.ts) beside the
// scan, derived from it, recording the rules that made it. The resident's button then only measures stored tracts
// against the tumor.
//
// It runs as a program beside Albula's server (sdk/server.ts): it reads the index and the files itself, read-only, and
// writes and indexes only through the server's `_write` route (the index's lock lives in the server process). This file
// is the library; import-job-main.ts is the program.
import { parseInstances, synthstripBrainMask, startSegmentationServer, volumesOfSeries, type BrainMask, type Volume } from "albula";
import { indexSeries, seriesFilePaths, writeNrrd, type IndexSeries } from "albula/server";
import { fromDicomVolumes } from "./dwi.ts";
import { wholeBrainTracts, PIPELINE_MAX_B } from "./pipeline.ts";
import { DISTORTION_RULE } from "./distortion.ts";
import { MOTION_RULE } from "./motion.ts";
import { REGISTRATION_RULE } from "./registration.ts";
import { TRACKING_RULE } from "./tracking-rules.ts";
import { OUTSIDE_RULE } from "./outside-brain.ts";
import { stageText, type StageTimes } from "./planning.ts";
import { SHORT } from "./tractcloud/name-tracts.ts";
import { tractColor, TRACT_COLORS_VERSION } from "./tractcloud/tract-colors.ts";
import type { TractCloudModel } from "./tractcloud/tractcloud.ts";
import type { RapidParcModel } from "./rapidparc/rapidparc.ts";
import { dicomToTracts, tractsToDicom, UNNAMED, type TractSetData } from "./tracts-dicom.ts";

/** The description every stored tracts object carries: how the job finds its own objects in the index. */
export const TRACTS_DESCRIPTION = "Fiber tracts (whole brain)";

/**
 * EVERYTHING THAT DECIDES WHETHER STORED TRACTS ARE CURRENT: the rules of each step and the naming network. Stored tracts
 * whose `rules` differ are remade (Ron, 2026-10-05: "remake automatically, versioned"); the old object stays until the
 * new one is written. `labeler` names the naming network's weights.
 */
export function jobRules(labeler: string): Record<string, string | number> {
  return { pipeline: 1, distortion: DISTORTION_RULE, motion: MOTION_RULE, registration: REGISTRATION_RULE, tracking: TRACKING_RULE,
    outside: OUTSIDE_RULE.on ? OUTSIDE_RULE.id : 0, maxB: PIPELINE_MAX_B, labeler, tractColors: TRACT_COLORS_VERSION, writer: 1 };
}
const sameRules = (a: Record<string, unknown> | undefined, b: Record<string, unknown>) => !!a && Object.keys(b).every((k) => String(a[k]) === String(b[k])) && Object.keys(a).length === Object.keys(b).length;

/** One series read through the core reader, with what the tracts object needs from its headers. */
export interface ReadSeries {
  uid: string; description: string; frames: Volume[];
  /** The first instance's header as DICOM JSON (patient and study attributes, frame of reference). */
  header: Record<string, { vr?: string; Value?: unknown[] }>;
  instances: { sopClassUID: string; sopInstanceUID: string }[];
}

/** Read a series' files, parse them with every registered interpreter (the extension's hooks are loaded by the program),
 *  and make its volumes -- the duckn copy writer's path. */
export async function readSeries(dbDir: string, row: IndexSeries): Promise<ReadSeries> {
  const paths = await seriesFilePaths(dbDir, row.seriesUID);
  if (!paths.length) throw new Error(`${row.seriesUID}: no files in the index`);
  const buffers = await Promise.all(paths.map(async (p) => { const b = await Deno.readFile(p); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer; }));
  const instances = await parseInstances(buffers, { headers: true });
  if (!instances.length) throw new Error(`${row.seriesUID}: none of its ${paths.length} files could be read as images`);
  const { frames } = volumesOfSeries(instances);
  const header = (instances[0].header?.json ?? {}) as ReadSeries["header"];
  const seen = new Set<string>(), refs: ReadSeries["instances"] = [];
  for (const i of instances) {
    const sop = i.sopInstanceUID, cls = String((i.header?.json as ReadSeries["header"] | undefined)?.["00080016"]?.Value?.[0] ?? "");
    if (sop && !seen.has(sop)) { seen.add(sop); refs.push({ sopClassUID: cls, sopInstanceUID: sop }); }
  }
  return { uid: row.seriesUID, description: row.description, frames, header, instances: refs };
}

const bOf = (v: Volume) => Number(((v.meta as Record<string, unknown> | undefined)?.diffusion as { bValue?: number } | undefined)?.bValue ?? NaN);
const peOf = (v: Volume) => ((v.meta as Record<string, unknown> | undefined)?.diffusion as { phaseEncoding?: string } | undefined)?.phaseEncoding;
/** A diffusion scan, as the module decides it: seven volumes or more, some with b > 0 (module.ts). */
export const isDiffusionScan = (s: ReadSeries) => s.frames.length >= 7 && s.frames.some((v) => bOf(v) > 50);
/** b = 0 images only, every volume carrying a diffusion b-value under 50. */
const isB0Only = (s: ReadSeries) => s.frames.length >= 1 && s.frames.every((v) => Number.isFinite(bOf(v)) && bOf(v) < 50);
/** Placed as the scan is: origin within 30 mm, each axis within 15° (module.ts partnerFor). */
function samePlacement(a: Volume, b: Volume): boolean {
  const A = a.ijkToRAS, B = b.ijkToRAS;
  if (Math.hypot(A[3] - B[3], A[7] - B[7], A[11] - B[11]) > 30) return false;
  for (let c = 0; c < 3; c++) {
    const u = [A[c], A[4 + c], A[8 + c]], w = [B[c], B[4 + c], B[8 + c]], cos = (u[0] * w[0] + u[1] * w[1] + u[2] * w[2]) / (Math.hypot(...u) * Math.hypot(...w));
    if (!(cos > Math.cos(15 * Math.PI / 180))) return false;
  }
  return true;
}

/** What a diffusion scan's job needs, found in its study; or why it waits. */
export interface CasePlan { study: string; dwi: ReadSeries; partner?: ReadSeries; t1?: ReadSeries }

/**
 * THE STUDIES' DIFFUSION SCANS, each with its partner and T1 (the module's rules: partnerFor, anatomyFor and pickAnatomy
 * prefer a "T1" in the name). Every MR series of a study with an MR series is read; nothing else.
 */
export async function planDatabase(dbDir: string, onSeries?: (line: string) => void, only?: string): Promise<{ plans: CasePlan[]; rows: IndexSeries[] }> {
  const rows = await indexSeries(dbDir);
  const studies = new Map<string, IndexSeries[]>();
  for (const r of rows) if (r.modality === "MR" && r.description !== TRACTS_DESCRIPTION) (studies.get(r.studyUID) ?? studies.set(r.studyUID, []).get(r.studyUID)!).push(r);
  if (only) { const st = rows.find((r) => r.seriesUID === only)?.studyUID; for (const k of [...studies.keys()]) if (k !== st) studies.delete(k); }
  const plans: CasePlan[] = [];
  for (const [study, series] of studies) {
    const read: ReadSeries[] = [];
    for (const r of series) {
      try { read.push(await readSeries(dbDir, r)); onSeries?.(`read ${r.seriesUID}`); }
      catch (e) { onSeries?.(`${r.seriesUID}: not read (${(e as Error).message.slice(0, 160)})`); }
    }
    for (const dwi of read.filter(isDiffusionScan)) {
      if (only && dwi.uid !== only) continue;
      const partner = read.find((s) => s !== dwi && isB0Only(s) && samePlacement(s.frames[0], dwi.frames[0]));
      const anatomy = read.filter((s) => s.frames.length === 1 && !isDiffusionScan(s) && !isB0Only(s) && !((s.frames[0].meta as Record<string, unknown> | undefined)?.diffusion));
      const t1 = anatomy.find((s) => /t1/i.test(s.description)) ?? anatomy[0];
      plans.push({ study, dwi, ...(partner ? { partner } : {}), ...(t1 ? { t1 } : {}) });
    }
  }
  return { plans, rows };
}

/** The stored tracts objects of a diffusion series, with the rules they were made by. */
export async function storedTracts(dbDir: string, rows: IndexSeries[], dwiUID: string): Promise<{ seriesUID: string; rules?: Record<string, unknown> }[]> {
  const out: { seriesUID: string; rules?: Record<string, unknown> }[] = [];
  for (const r of rows.filter((x) => x.description === TRACTS_DESCRIPTION && x.studyUID === rows.find((y) => y.seriesUID === dwiUID)?.studyUID)) {
    for (const p of await seriesFilePaths(dbDir, r.seriesUID)) {
      try {
        const t = await dicomToTracts(await Deno.readFile(p));
        if (t.referencedSeries === dwiUID) out.push({ seriesUID: r.seriesUID, rules: t.provenance?.rules as Record<string, unknown> | undefined });
      } catch { /* not a tracts object of ours */ }
    }
  }
  return out;
}

/** What happened to one diffusion scan. */
export type CaseOutcome =
  | { state: "made"; seriesUID: string; streamlines: number; named: number; seconds: number; said: string }
  | { state: "current"; seriesUID: string }
  | { state: "waiting"; why: string };

/**
 * SynthStrip's brain on the T1, through the server's haversack proxy (`server` the server's root URL). Starts the
 * segmentation server when none answers, and waits for it (quality first: no fallback to the scan's own mask).
 */
export async function brainOnT1(server: string, t1: Volume, onProgress?: (line: string) => void): Promise<{ mask?: BrainMask; why?: string }> {
  const transport = { fetch: (...a: Parameters<typeof fetch>) => fetch(...a), base: `${server.replace(/\/+$/, "")}/_haversack/` };
  const upload = async () => ({ bytes: await writeNrrd(t1, { encoding: "gzip" }), filename: "t1.nrrd" })   // gzip INSIDE the NRRD, as the page sends it (logic/export.ts); ITK will not read ".nrrd.gz";
  const stub = { nodes: new Map() } as unknown as Parameters<typeof synthstripBrainMask>[0];
  let r = await synthstripBrainMask(stub, `t1:${(t1.meta as Record<string, unknown> | undefined)?.seriesInstanceUID ?? "?"}`, onProgress, { transport, upload });
  if (!r.ok && r.reason === "no-server") {
    onProgress?.("starting the segmentation server");
    const s = await startSegmentationServer(onProgress, { fetch: (input, init) => fetch(typeof input === "string" && input.startsWith("/") ? `${server.replace(/\/+$/, "")}${input}` : input, init), base: transport.base });
    if (s.ok) r = await synthstripBrainMask(stub, `t1:${(t1.meta as Record<string, unknown> | undefined)?.seriesInstanceUID ?? "?"}`, onProgress, { transport, upload });
  }
  return r.ok ? { mask: r.mask } : { why: r.message };
}

/** The patient and study attributes the tracts object copies from the scan (as they are). */
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

/**
 * ONE CASE: the pipeline on the planned scan, its tracts written and indexed through the server. `rules` from jobRules;
 * a scan whose stored tracts carry the same rules is left alone.
 */
export async function makeTracts(dbDir: string, dbId: string, server: string, plan: CasePlan, rows: IndexSeries[], device: GPUDevice, model: TractCloudModel,
  labeler: { model: RapidParcModel; name: string }, onProgress?: (line: string) => void, opts: { force?: boolean; dryRun?: boolean } = {}): Promise<CaseOutcome> {
  const rules = jobRules(labeler.name);
  const stored = await storedTracts(dbDir, rows, plan.dwi.uid);
  const current = stored.find((s) => sameRules(s.rules, rules));
  if (current && !opts.force) return { state: "current", seriesUID: current.seriesUID };
  // QUALITY FIRST: the tracking's brain is SynthStrip's on the T1; without the T1 or the service the case waits.
  if (!plan.t1) return { state: "waiting", why: "waiting for the MRI of the anatomy (T1) of this study" };
  const t0 = performance.now();
  onProgress?.("finding the brain on the MRI of the anatomy");
  const brain = await brainOnT1(server, plan.t1.frames[0], onProgress);
  if (!brain.mask) return { state: "waiting", why: `waiting for the brain-mask service (${brain.why})` };
  const dwi = fromDicomVolumes(plan.dwi.frames, plan.dwi.description);
  const t1v = plan.t1.frames[0];
  const stages: StageTimes = {};
  onProgress?.("correcting, aligning and tracking the whole brain");
  const r = await wholeBrainTracts({ dwi,
    ...(plan.partner ? { partner: { b0s: plan.partner.frames.map((v) => v.data as ArrayLike<number>), grid: { dims: plan.partner.frames[0].dims, ijkToRAS: plan.partner.frames[0].ijkToRAS }, name: plan.partner.description || "reversed scan" } } : {}),
    phaseEncoding: { scan: peOf(plan.dwi.frames[0]), partner: plan.partner ? peOf(plan.partner.frames[0]) : undefined },
    t1: { dims: t1v.dims as [number, number, number], ijkToRAS: t1v.ijkToRAS, data: t1v.data as ArrayLike<number> },
    brainT1: brain.mask }, device, model, { labeler: labeler.model }, stages);
  // THE TRACK SETS: one per named tract and side, and one "Unnamed" for the rest (too short to name, or no tract).
  const count = model.json.tracts.length, OTHER = count - 1, groups = new Map<string, TractSetData>();
  for (let i = 0; i < r.sl.length; i++) {
    const t = r.named.tract[i], named = t !== SHORT && t !== OTHER && !r.outside[i];
    const side = (named ? Math.sign(r.named.side[i]) : 0) as -1 | 0 | 1, key = named ? `${t}:${side}` : "unnamed";
    let g = groups.get(key);
    if (!g) { const c = named ? tractColor(t, count) : tractColor(-1, count); g = { label: named ? model.json.tracts[t].name : UNNAMED, side, color: [c[0], c[1], c[2]], streamlines: [] }; groups.set(key, g); }
    g.streamlines.push(r.sl[i]);
  }
  const sets = [...groups.values()].sort((a, b) => (a.label === UNNAMED ? 1 : b.label === UNNAMED ? -1 : a.label.localeCompare(b.label) || a.side - b.side));
  const named = sets.filter((s) => s.label !== UNNAMED).reduce((n, s) => n + s.streamlines.length, 0);
  // On the T1's axes when aligned: the streamlines are in the T1's space, so the T1's frame of reference.
  const aligned = r.alignment && !r.alignment.doubt;
  const forUID = String(((aligned ? plan.t1.header : plan.dwi.header)["00200052"]?.Value?.[0]) ?? "");
  const seconds = (performance.now() - t0) / 1000;
  const provenance = { rules, made: new Date().toISOString(), seconds: +seconds.toFixed(1), stages: stageText(stages),
    streamlines: r.sl.length, onePointLeftOut: r.sl.filter((p) => p.length < 6).length,
    inputs: { diffusion: plan.dwi.uid, partner: plan.partner?.uid ?? null, t1: plan.t1.uid }, corrected: r.corrected,
    trackingRuleApplied: r.rule.id, motionRuleApplied: r.prep.motionRule, ...(r.alignment ? { alignment: r.alignment } : {}), brain: r.fit.seedMaskRule ?? r.fit.maskRule };
  const written = await tractsToDicom(sets, { patientStudy: patientStudyOf(plan.dwi.header), frameOfReferenceUID: forUID, seriesInstanceUID: plan.dwi.uid, instances: plan.dwi.instances },
    { algorithmName: "UKF two-tensor (Albula's port of UKFTractography)", algorithmVersion: `tracking rule ${r.rule.id}`, algorithmParameters: JSON.stringify(r.rule.ukf).slice(0, 10240),
      model: "multi", provenance, seriesDescription: TRACTS_DESCRIPTION });
  // A streamline of one point (a seed that stopped at once) has no line to store; the writer leaves it out, and says so.
  const onePoint = r.sl.length - written.tracks;
  const said = `${r.sl.length} streamlines (${written.tracks} stored${onePoint ? `, ${onePoint} of a single point left out` : ""}), ${named} named into ${sets.length - (groups.has("unnamed") ? 1 : 0)} tracts, in ${seconds.toFixed(0)} s; ${r.corrected}`;
  if (opts.dryRun) return { state: "made", seriesUID: written.seriesInstanceUID, streamlines: r.sl.length, named, seconds, said: `(dry run, not written) ${said}` };
  const meta = { sopInstanceUID: written.sopInstanceUID, seriesInstanceUID: written.seriesInstanceUID, studyInstanceUID: plan.study, modality: "MR",
    seriesDescription: TRACTS_DESCRIPTION, frameOfReferenceUID: forUID, derivedFrom: { parentSeriesUID: plan.dwi.uid, kind: "tracts", label: "Fiber tracts" } };
  const res = await fetch(`${server.replace(/\/+$/, "")}/_db/${encodeURIComponent(dbId)}/_write/${encodeURIComponent(`tracts-${written.seriesInstanceUID}.dcm`)}`,
    { method: "POST", body: written.bytes as unknown as BodyInit, headers: { "content-type": "application/dicom", "x-albula-index": encodeURIComponent(JSON.stringify(meta)) } });
  if (!res.ok) throw new Error(`the server refused the tracts object: ${res.status} ${(await res.text()).slice(0, 300)}`);
  return { state: "made", seriesUID: written.seriesInstanceUID, streamlines: r.sl.length, named, seconds, said };
}

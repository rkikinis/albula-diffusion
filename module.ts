// THE DIFFUSION MODULE (milestone 1, Contents/docs/dmri-review-2026-09-28.md; mockup
// Contents/docs/mockups/diffusion-2026-09-29.html in the workspace; Ron, 2026-09-29: "1 yes" to building it).
//
// THE FACE (2026-10-01, mockup diffusion-workflow-v4; Ron: the user is a neurosurgery resident who knows neither the
// lingo nor the concepts; "One button as initial, everything else under advanced"):
//  1 · THE PATIENT'S CASE: diffusion MRI, MRI of the anatomy, tumor outline, each ticked when it is there; Open a
//      patient… / Scans on this computer… go to the database and Load / Save (core's, through the SDK); a missing
//      outline is grown from Tumor / Not tumor strokes (core's growIntoSegmentation, Steve's GPU GrowCut).
//  2 · FIBER TRACTS NEAR THE TUMOR: one button (the named tracts, below) and the tracts it made, named as
//      tract-info.ts says ("Arcuate fasciculus, right (AF)").
// Everything below is under ADVANCED, in the three sections this module had before:
//  - MAPS (blue band: a display): which diffusion scan, and what the slice views show of it -- the scan's own signal,
//    FA, or Color FA. Color FA is what a diffusion scan shows when it loads (Ron's yes on the mockup), over the MRI of the
//    anatomy at half opacity when one is loaded (putMap; Ron, 2026-10-01). The tensor is
//    fitted once per scan, on the processor (tensor.ts; about 1 s on PAT16, reading included).
//  - TRACTS (yellow band: it makes new things): tracts near a segment -- started in every white-matter voxel inside it
//    or within a distance of it -- or from a point clicked in a view.
//  - IN THE SCENE (green band: what exists): the tract groups, each with its eye and its ✕, drawn as tubes (Ron:
//    "tubes" first) or lines, and as dots where they cross the slices (drawSliceCrossings), ends shortened (drawn()).
// Tracking: UKF two-tensor on the graphics card (ukf-gpu.ts) by the current tracking rule (tracking-rules.ts; rule 3 since
// 2026-10-04: Mike Halle's tractline -- plain two-tensor, the shell nearest b = 3000, every brain voxel a seed -- inside the
// brain SynthStrip finds on the MRI of the anatomy, asked of the segmentation server, brainFor), or one
// tensor (tracking.ts). Names: RapidParc
// (rapidparc/; TractCloud's tractcloud/ until 2026-10-03, its table still shared), from whole-brain tracking; the named tracts near the chosen structure are shown whole (makeNamedTracts).
// NOT YET: the tract groups are the module's, not scene nodes -- Scene does not list them and a saved scene does not keep
// them (the mockup's "in Scene under the diffusion scan" is still to build).
//
// This file belongs to the diffusion extension and registers through core's queue (render/demos/extension-modules.ts);
// core never imports it. Critic round 2026-09-29 (qa/2026-09-29-diffusion-module.md) answered in this version.
import type { ModuleContext } from "albula";
import { queueModule } from "albula";
import type { MrsonNode } from "albula";
import { fetchZarrVolumeNative, type ZarrDesc } from "albula";
import { packRGB24 } from "albula";
import { browserFrames, sequenceBrowsers } from "albula";
import { fromDicomVolumes, type DiffusionSeries } from "./dwi.ts";
import { loadVolumeIntoScene, removeVolumeFromScene } from "albula";
import { IDENTITY4, worldForNode } from "albula";
import type { Volume } from "albula";
import { FiberField, type RGBA, type Strand } from "albula";
import { colorFA, fitTensors, type TensorFit } from "./tensor.ts";
import { type UkfData } from "./ukf.ts";
import { edgeFluid, OUTSIDE_RULE, outsideBrain } from "./outside-brain.ts";
import { seedsFor, TRACKING_RULE, TRACKING_RULES, ukfDataFor, withBrainFromT1, type TrackingRuleId } from "./tracking-rules.ts";
import type { FieldFit } from "./distortion.ts";
import { prepareScan } from "./prepare.ts";
import { MOTION_RULE, type MotionRuleId } from "./motion.ts";
/** "Stop below FA"'s default under Advanced (the single-tensor tracker's; the UKF's comes from the tracking rule). */
const ADV_MIN_FA = 0.15;
import { DCM2NIIX_VERSION, secondOpinion, type SecondOpinion } from "./second-opinion.ts";
import { assetUrl, holdDrawing, seriesDicomFiles, startPlacing, startSegmentationServer, synthstripBrainMask, type BrainMaskResult } from "albula";
import { createSegmentation, growIntoSegmentation, openDicomDatabase, openLoadFromDisk, paintInto, registerProbeRows, registerRayHits, runAction, saveSegmentationToDicom, showHideAllState } from "albula";
import { buildTractIndex, tractsNear, type TractIndex } from "./tract-index.ts";
import { sliceCrossings, trimEnds } from "./tract-slice.ts";
import { readMore, tnaLine, tractInfo, tractLabel, tractNote } from "./tract-info.ts";
import { faceNear as nearOnFace, isTumorName, matchesSearch, patientOf, pickAnatomy, tractGroupKey, TRACT_GROUPS, withoutLastRun } from "./face.ts";
import { loadModel, type ModelJson, type TractCloudModel } from "./tractcloud/tractcloud.ts";
import { nameAgainst, nameTracts, type Named } from "./tractcloud/name-tracts.ts";
import { loadRapidParc, type RapidParcModel } from "./rapidparc/rapidparc.ts";
import { correctWithReversed, denseSeeds, MIN_NEAR_STREAMLINES, NEAR_MM, GRAY_BAND_MM, otherSide, sortByDistance, stageText, streamlineDistances, tractName, trackUkfSeeds, wholeBrainSeeds, type Sorted, type StageTimes, type Structure, type TrackTiming } from "./planning.ts";
import { tractColor, UNNAMED } from "./tractcloud/tract-colors.ts";
import { seedsInSphere, trackFromSeeds, type Streamline, type TrackingOptions } from "./tracking.ts";
import { DIFFUSION_REFERENCES } from "./references.ts";
import { estimateResponses, kernelFromResponses } from "./responses.ts";
import { csdVolume, type FodVolume } from "./csd-volume.ts";
import { trackPttParallel } from "./ptt.ts";

/** A diffusion scan in the scene: a sequence whose frames carry diffusion values. */
interface Scan { browserId: string; name: string; frameIds: string[]; bValues: number[]; study?: string; patient?: string;
  /** The scanner's record of the phase-encoding direction (diffusion-vendors.ts phaseEncodingOf: "j-", or "ROW" / "COL"). */
  phaseEncoding?: string }
/** What has been computed for a scan, kept while the scan is in the scene. */
interface Computed { dwi: DiffusionSeries; fit: TensorFit; maxB: number; corrected: string; partnerId: string; /** motion.ts: the head-movement rule the scan was put in place by. */ motionRule: MotionRuleId; /** The anatomy MRI the scan was aligned to ("" none). */ anatomyId?: string; /** The tracking rule the fit's brain mask serves (tracking-rules.ts): 2 until rule 3's brain is put in (brainFor). */ rule: TrackingRuleId; /** Rule 3's brain: what was used, in words, and why not when it was not. */ brain: BrainState; faId?: string; colorFaId?: string; ukf?: UkfData; fod?: FodVolume }
/** RULE 3's BRAIN for a fitted scan (critic, 2026-10-04, findings 2, 4, 8, 9): asked of the segmentation server when the
 *  maps are made but not waited for -- the maps do not use it -- and waited for by the tracking (brainFor). A failure
 *  that can go away (no server, a server without SynthStrip, a failed job) is asked again at the next tracking. */
interface BrainState { note: string; reason?: "no-anatomy" | "moved" | "doubt" | "no-server" | "no-synthstrip" | "failed"; ask?: Promise<BrainMaskResult> }
/** A reversed phase-encoding scan for a diffusion scan: b = 0 images of the same study, on its own grid (aligned by the scanner's coordinates). */
interface Partner { id: string; name: string; frameIds: string[]; dims: number[]; ijkToRAS: number[]; phaseEncoding?: string }
/** A node's recorded phase-encoding direction, as the diffusion interpreter read it. */
const phaseEncodingOfNode = (n: MrsonNode | undefined) => ((n?.origin as Record<string, unknown> | undefined)?.diffusion as { phaseEncoding?: string } | undefined)?.phaseEncoding;
/** A group of tracts. */
interface TractGroup {
  id: number; name: string; scan: string; strands: Float32Array[]; visible: boolean; method: Method;
  /** Named by TractCloud: the tract (index into the model's list), its side (+1 right, -1 left, 0 none), and its
   *  closest distance to the structure it was measured against, in mm. `unnamed`: TractCloud gave no name. */
  tract?: number; side?: number; distanceMm?: number; within?: number; unnamed?: boolean;
  /** Made by "Show the fiber tracts near the tumor": the next run for the same scan replaces them (critic, finding 5). */
  run?: boolean;
  /** Within reach of the structure, with fewer streamlines than the minimum: listed in gray, hidden (Ron, 2026-10-01). */
  faint?: boolean;
  /** Streamlines that cross the fluid at the brain's edge (outside-brain.ts): possibly a cranial nerve, never a tract. */
  outside?: boolean;
  /** The same tract's streamlines on the other side, for comparing sides (undefined for a tract across the midline). */
  otherSide?: number;
  /** Streamlines "Add lines" added to this group. */
  more?: number;
}
/** The last whole-brain run for a scan, kept so "Add lines" can name new streamlines in its context and measure them
 *  against the same structure. */
interface Run { sl: Float32Array[]; named: Named; sorted: Sorted; structure: Structure; label: string; withinMm: number; method: Method;
  /** The tracking rule the whole-brain run was made under, and its brain in words (BrainState.note). */ rule?: TrackingRuleId; brain?: string;
  /** The tracts ("tract:side") Add lines has already added to: a second press would start from the same points and
   *  draw the same lines again, so they are skipped (Ron, 2026-10-01: he pressed it, then saw the corticospinal tract
   *  was not on). */
  added: Set<string>;
  /** Each run streamline's distance to the structure (mm, measured out to the margin and 2 mm beyond). */
  dist: Float64Array;
  /** The fit it was tracked on (critic, 2026-10-01, finding 5): Add lines on a different fit is refused. */
  maxB: number; partnerId: string;
  /** The anatomy MRI the scan was aligned to when the run was made ("" none): Add lines tracks in the same space only. */
  anatomyId: string }
/** How tracts are followed: one tensor per voxel (fast, the classic), or the two-tensor UKF (ukf.ts; crossing fibers;
 *  the method SlicerDMRI uses for tumor planning), with the settings of the current tracking rule (tracking-rules.ts). */
/** "ptt": fiber distributions from multi-shell CSD (csd.ts, responses from the scan) and parallel transport tracking
 *  (ptt.ts), on the processor's workers -- the smoother method Lauren O'Donnell recommended; about three times as long
 *  as UKF and no better on the development cases' meningiomas, so an option (Ron, 2026-10-01: "if both are close to
 *  equal in quality, we go with the faster and offer the other as option"). */
type Method = "ukf" | "single" | "ptt";
type Show = "signal" | "fa" | "colorfa";

const FIELD_KEY = "diffusion-tracts";
const RADIUS = { tubes: 0.5, lines: 0.12 };                // mm: SlicerDMRI's tube radius is 0.5 mm; "lines" are hairlines
const SEEDS_PER_BATCH = 1500;                               // tracking yields to the page between batches
/** Palette ids: 1..216 are the direction colors; named groups take ids from TRACT_ID up (the palette ends at 255). */
const GRAY_ID = 217, TRACT_ID = 218;

/** Directional colors (red left-right, green front-back, blue up-down, as Color FA): 6 levels per channel, ids 1..216. */
const LEVELS = 6;
function directionPalette(): Record<number, RGBA> {
  const pal: Record<number, RGBA> = {};
  for (let r = 0; r < LEVELS; r++) for (let g = 0; g < LEVELS; g++) for (let b = 0; b < LEVELS; b++) {
    const id = 1 + (r * LEVELS + g) * LEVELS + b, s = (x: number) => 0.1 + 0.9 * x / (LEVELS - 1);
    pal[id] = [s(r), s(g), s(b), 1];
  }
  return pal;
}
/**
 * EACH PIECE'S COLOR, by its own direction (red left-right, green front-back, blue up-down), as Slicer's "color by
 * orientation": the direction at point i is the average over two points either side (so one kinked step does not
 * flicker), normalized, its absolute components quantized to the palette's 6 levels.
 */
function pointIds(p: Float32Array): Uint8Array {
  const m = p.length / 3, out = new Uint8Array(m), W = 2;
  for (let i = 1; i < m; i++) {
    const a = Math.max(0, i - 1 - W), b = Math.min(m - 1, i + W);
    const dx = p[3 * b] - p[3 * a], dy = p[3 * b + 1] - p[3 * a + 1], dz = p[3 * b + 2] - p[3 * a + 2];
    const mx = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) || 1, q = (v: number) => Math.min(LEVELS - 1, Math.round((Math.abs(v) / mx) * (LEVELS - 1)));
    out[i] = 1 + (q(dx) * LEVELS + q(dy)) * LEVELS + q(dz);
  }
  out[0] = out[1] || 1;
  return out;
}
/** A streamline's color id: the mean of its segments' absolute directions, normalized. */
function directionId(p: Float32Array): number {
  let x = 0, y = 0, z = 0;
  for (let i = 3; i < p.length; i += 3) {
    const dx = p[i] - p[i - 3], dy = p[i + 1] - p[i - 2], dz = p[i + 2] - p[i - 1], l = Math.hypot(dx, dy, dz) || 1;
    x += Math.abs(dx) / l; y += Math.abs(dy) / l; z += Math.abs(dz) / l;
  }
  const m = Math.max(x, y, z) || 1, q = (v: number) => Math.min(LEVELS - 1, Math.round((v / m) * (LEVELS - 1)));
  return 1 + (q(x) * LEVELS + q(y)) * LEVELS + q(z);
}

const matVec = (M: number[], i: number, j: number, k: number): [number, number, number] =>
  [M[0] * i + M[1] * j + M[2] * k + M[3], M[4] * i + M[5] * j + M[6] * k + M[7], M[8] * i + M[9] * j + M[10] * k + M[11]];
/** The inverse of a row-major 4x4 affine (rotation/scale + translation). */
function invAffine(m: number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  const R = [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((x) => x / det);
  const t = [m[3], m[7], m[11]];
  return [R[0], R[1], R[2], -(R[0] * t[0] + R[1] * t[1] + R[2] * t[2]), R[3], R[4], R[5], -(R[3] * t[0] + R[4] * t[1] + R[5] * t[2]), R[6], R[7], R[8], -(R[6] * t[0] + R[7] * t[1] + R[8] * t[2]), 0, 0, 0, 1];
}
const isIdentity = (m: number[]) => m.every((v, i) => Math.abs(v - IDENTITY4[i]) < 1e-9);

export function registerDiffusionPanel(ctx: ModuleContext): void {
  const { shell, live, store, device, status } = ctx;
  const computed = new Map<string, Computed>();
  const groups: TractGroup[] = [];
  const runs = new Map<string, Run>();
  const seenScans = new Set<string>();
  let groupSeq = 0;
  let drawAs: "tubes" | "lines" = "tubes";
  let colorBy: "tract" | "direction" = "tract";
  /** TractCloud's network, loaded once from the files beside the bundle (vendor/diffusion/tractcloud/, model/README.md). */
  let tcModel: Promise<TractCloudModel> | undefined;
  const tractCloud = () => tcModel ??= (async () => {
    const get = async (f: string) => { const r = await fetch(assetUrl("diffusion", `tractcloud/${f}`)); if (!r.ok) throw new Error(`the tract-naming model is missing (${f}: ${r.status})`); return r; };
    const [w, j] = await Promise.all([get("weights.f32").then((r) => r.arrayBuffer()), get("model.json").then((r) => r.json())]);
    return loadModel(w, j as ModelJson);
  })().catch((e) => { tcModel = undefined; throw e; });
  /** RapidParc's network, the namer since 2026-10-03 (Ron: "2 yes"; rapidparc/, model/README.md); TractCloud's model.json
   *  stays as the table of clusters, tracts and names both share. */
  let rpModel: Promise<RapidParcModel> | undefined;
  const rapidParc = () => rpModel ??= (async () => {
    const r = await fetch(assetUrl("diffusion", "rapidparc/rapidparc.safetensors"));
    if (!r.ok) throw new Error(`the tract-naming model is missing (rapidparc.safetensors: ${r.status})`);
    return loadRapidParc(await r.arrayBuffer());
  })().catch((e) => { rpModel = undefined; throw e; });
  let field: FiberField | undefined;
  let chosen = "";                                          // browser id of the scan the panel is about
  let near = "", withinMm = NEAR_MM, busy = "", seeding = false, note = "";
  let cancelSeeding: (() => void) | undefined;
  const adv: Required<Pick<TrackingOptions, "minFA" | "maxAngleDeg" | "stepVoxels">> & { maxB: number } & { ukfStopFA: number } = { minFA: ADV_MIN_FA, maxAngleDeg: 45, stepVoxels: 0.5, maxB: 1500, ukfStopFA: TRACKING_RULES[TRACKING_RULE].ukf.stoppingFA };
  let root: HTMLElement | undefined;
  let advOpen = false;
  let moreOpen = false;
  /** The tract list's search and its folded groups (Ron, 2026-10-01: Segmentations as the template). */
  let tractSearch = "";
  const foldedGroups = new Set<string>();                                      // the face's Advanced (mockup v4: everything but the one button)
  /**
   * THE TUMOR OUTLINE BEING MADE (mockup diffusion-workflow-v4; Ron, 2026-10-01: grow from seeds "would do the job"):
   * strokes in a segmentation of their own (1 Tumor, 2 Not tumor) on the anatomical MRI; the grown outline in another,
   * made again from all the strokes at every Grow; Done keeps the outline and takes the strokes away.
   */
  let outline: { imageId: string; seedsId: string; resultId?: string; tool: 0 | 1 | 2; voxels?: number; mm3?: number } | undefined;
  /** The outline made here, after Done: offered for saving as an AI result is (Ron, 2026-10-01: "same behavior and
   *  appearance as with the haversack functionality"). */
  let kept: { segId: string; mm3: number; saved: boolean; saving?: boolean; savable: boolean } | undefined;
  const saveKept = async () => {
    if (!kept || kept.saving || kept.saved) return;     // one save at a time (critic, finding 12c: two SEG series)
    const k = kept;
    k.saving = true;
    say("Saving the tumor outline to the DICOM database…");
    const note = await saveSegmentationToDicom(k.segId).catch((e) => { k.saving = false; say(`The tumor outline was not saved: ${(e as Error).message}`); throw e; });
    k.saved = true; k.saving = false;
    say(`Saved — ${note}`);
    render();
  };
  let correct = true;                                        // distortion correction when a reversed scan is there
  let preCorrected = false;                                  // the scan was corrected for head movement before (a preprocessed dataset)
  let method: Method = "ukf";
  /** dcm2niix's second opinion per scan (second-opinion.ts): running, its answer, or why it could not run. */
  const checks = new Map<string, { running: true } | { result: SecondOpinion } | { error: string }>();
  /** Run the second opinion once per scan, in the background, on the scan's own DICOM files. */
  function check(scan: Scan, ours: DiffusionSeries) {
    if (checks.has(scan.browserId)) return;
    const uid = (live.nodes.get(scan.frameIds[0])?.origin as Record<string, unknown> | undefined)?.seriesInstanceUID as string | undefined;
    if (!uid) { checks.set(scan.browserId, { error: "not from the DICOM database: nothing for dcm2niix to read" }); return; }
    checks.set(scan.browserId, { running: true }); render();
    void (async () => {
      try {
        const files = await seriesDicomFiles(uid);
        if (files === null) throw new Error("not from the DICOM database: nothing for dcm2niix to read");
        if (!files.length) throw new Error("the scan's DICOM files could not be read");
        const r = await secondOpinion(files, ours);
        checks.set(scan.browserId, { result: r });
        if (!r.agree) say(`${scan.name}: ${r.said}`);
      } catch (e) { checks.set(scan.browserId, { error: (e as Error).message }); }
      if (computed.has(scan.browserId) || scans().some((s) => s.browserId === scan.browserId)) render();
    })();
  }

  /** Said where the person looks: the visible status bar, and the line under the module's buttons. */
  const say = (t: string) => { note = t; status(t); };

  // ── what is in the scene ────────────────────────────────────────────────────────────────────────────────────────
  const scans = (): Scan[] => sequenceBrowsers(live).flatMap((b) => {
    const { frames, sequence } = browserFrames(live, b.id);
    const nodes = frames.map((f) => live.nodes.get(f.node)).filter(Boolean) as MrsonNode[];
    const org = (n: MrsonNode) => (n.origin as Record<string, unknown> | undefined) ?? {};
    const bs = nodes.map((n) => Number((org(n).diffusion as { bValue?: number } | undefined)?.bValue ?? NaN));
    if (nodes.length < 7 || !bs.some((b) => b > 0)) return [];
    const o = org(nodes[0]);
    return [{ browserId: b.id, name: String(sequence?.name ?? b.name ?? "Diffusion scan"), frameIds: nodes.map((n) => n.id), bValues: bs, study: o.studyInstanceUID as string | undefined, patient: o.patientID as string | undefined, phaseEncoding: phaseEncodingOfNode(nodes[0]) }];
  });
  /**
   * The segments a scan's tracts may start near: those of the SAME PATIENT -- the segmentation's study is the scan's, or
   * the image it was drawn on belongs to the scan's study or patient. Critic, 2026-09-29, finding 1: with two patients
   * loaded, the list offered the other patient's tumor and made 13,270 tracts around it.
   */
  const segmentChoices = (scan: Scan | undefined) => {
    if (!scan) return [];
    const segs = [...live.nodes.values()].filter((n) => n.type === "segmentation" && n.id !== outline?.seedsId).filter((s) => {
      const so = (s.origin as Record<string, unknown> | undefined) ?? {};
      const src = live.nodes.get(((s.refs as Record<string, string[]> | undefined)?.source ?? [])[0] ?? "");
      const io = (src?.origin as Record<string, unknown> | undefined) ?? {};
      return (!!scan.study && (so.studyInstanceUID === scan.study || io.studyInstanceUID === scan.study)) || (!!scan.patient && io.patientID === scan.patient);
    });
    // The segmentation's own description after each structure (the patient is the scan's, and is not repeated).
    const own = (n: string) => n.replace(/^.*?·\s*(SEG\s+)?/, "");
    return segs.flatMap((s) => ((s.segments as { labelValue: number; name: string }[] | undefined) ?? []).map((g) => ({ key: `${s.id}#${g.labelValue}`, label: `${g.name}${segs.length > 1 ? ` (${own(String(s.name))})` : ""}`, seg: s, labelValue: g.labelValue })));
  };
  /**
   * WHICH OUTLINES ARE A TUMOR (critic, 2026-10-01, finding 1: any segmentation of the patient -- an AI brain
   * parcellation -- ticked "Tumor outline" and the tracts were measured from its first structure). A segment counts when
   * its name says so, or when it is the outline made in this module; the face measures only from these.
   */
  const tumorChoices = (scan: Scan | undefined) => segmentChoices(scan).filter((c) => isTumorName(c.label) || c.seg.id === kept?.segId);
  /**
   * THE MRI OF THE ANATOMY for a scan: an image of the same study (or patient) that is not a diffusion volume, a
   * reversed-phase volume, a computed map or a label map; a T1 by name first (the scan the tumor is outlined on).
   */
  const anatomyFor = (scan: Scan | undefined): MrsonNode | undefined => {
    if (!scan) return undefined;
    const inSequences = new Set(sequenceBrowsers(live).flatMap((b) => browserFrames(live, b.id).frames.map((f) => f.node)));
    const c = computed.get(scan.browserId);
    const cands = [...live.nodes.values()].filter((n) => {
      if (n.type !== "image" || n.labelmap || inSequences.has(n.id) || n.id === c?.faId || n.id === c?.colorFaId) return false;
      const o = (n.origin as Record<string, unknown> | undefined) ?? {};
      if ((o.diffusion as unknown) !== undefined) return false;
      return (!!scan.study && o.studyInstanceUID === scan.study) || (!!scan.patient && o.patientID === scan.patient);
    });
    return pickAnatomy(cands);   // face.ts: "T1" in the series part of the name (critic, finding 15)
  };
  const removeNode = (id: string) => {
    const n = live.nodes.get(id);
    if (!n) return;
    live.write({ op: "del", id });
    for (const d of ((n.refs as Record<string, string[]> | undefined)?.display ?? [])) if (live.nodes.get(d)) live.write({ op: "del", id: d });
  };
  /** Tumor or Not tumor strokes, from now on, into the strokes segmentation (made on first use, on the anatomy). */
  async function strokes(anat: MrsonNode, which: 1 | 2) {
    if (busy) return;
    try {
      if (!outline || outline.imageId !== anat.id || !live.nodes.get(outline.seedsId)) {
        const { segId } = await createSegmentation(live, store, anat.id, { name: "Tumor strokes" });
        live.write({ op: "patch", id: segId, path: "#/segments", value: [
          { labelValue: 1, name: "Tumor", color: [0.95, 0.8, 0.3], visible: true },
          { labelValue: 2, name: "Not tumor", color: [0.55, 0.6, 0.7], visible: true },
        ] });
        outline = { imageId: anat.id, seedsId: segId, tool: 0 };
      }
      putBackground(anat.id, null);    // strokes are drawn on the anatomy, not on Color FA
      const to = outline.tool === which ? 0 : which;
      if (!paintInto(outline.seedsId, to || null, 5)) { say("Drawing is not available in this app."); return; }
      outline.tool = to;
      say(to === 1 ? "Draw a stroke inside the tumor on several slices, from its first slice to its last." : to === 2 ? "Draw strokes in the brain around the tumor, on several slices, also just above and below it." : "Drawing is off.");
    } catch (e) { say(`The strokes could not be started: ${(e as Error).message}`); }
    render();
  }
  async function growOutline() {
    if (!outline || busy) return;
    busy = "Growing…"; render();
    try {
      const r = await growIntoSegmentation(live, store, outline.seedsId, 1, { resultId: outline.resultId, name: "Tumor", color: [0.95, 0.75, 0.25] });
      outline.resultId = r.segId; outline.voxels = r.voxels;
      const S = live.nodes.get(outline.imageId)?.ijkToRAS as number[] | undefined;
      const vox = S ? Math.abs(S[0] * (S[5] * S[10] - S[6] * S[9]) - S[1] * (S[4] * S[10] - S[6] * S[8]) + S[2] * (S[4] * S[9] - S[5] * S[8])) : 1;
      outline.mm3 = r.voxels * vox;
      say(`Outline grown${r.ms < 100 ? "" : ` in ${(r.ms / 1000).toFixed(1)} s`}: ${(outline.mm3 / 1000).toFixed(1)} mL. Check it on every slice it touches, and in the sagittal and coronal views; add strokes where it is wrong and grow again.`);
    } catch (e) { say((e as Error).message); }
    finally { busy = ""; render(); }
  }
  /** Leave the outline steps: strokes and a grown outline not kept are removed (critic, finding 9: no way out). */
  function outlineCancel() {
    if (!outline) return;
    paintInto(outline.seedsId, null);
    removeNode(outline.seedsId);
    if (outline.resultId) removeNode(outline.resultId);
    outline = undefined;
    say("The outline was not kept.");
    render();
  }
  function outlineDone() {
    if (!outline?.resultId) return;
    paintInto(outline.seedsId, null);
    removeNode(outline.seedsId);
    near = `${outline.resultId}#1`;
    // Savable only when the MRI it was drawn on came from the DICOM database (critic, finding 12b), as an AI result is.
    const anatOrigin = live.nodes.get(outline.imageId)?.origin as { seriesInstanceUID?: string; savedSeriesInstanceUID?: string } | undefined;
    kept = { segId: outline.resultId, mm3: outline.mm3 ?? 0, saved: false, savable: !!(anatOrigin?.seriesInstanceUID || anatOrigin?.savedSeriesInstanceUID) };
    outline = undefined;
    say(`The tumor outline is ready: ${(kept.mm3 / 1000).toFixed(1)} mL. It is in the scene now; ${kept.savable ? "not saved yet" : "it cannot be saved to the database, because the MRI it was drawn on is not from the database"}.`);
    render();
    // TOLD ONCE, AS AN AI RESULT IS (render/demos/ai-seg-panel.ts): only when the resident is elsewhere -- here the module
    // shows it (critic, finding 12a) -- and Save only when it can work.
    if (kept.savable && (shell as unknown as { activePanel?: () => string }).activePanel?.() !== "diffusion") shell.notify({
      title: `Tumor outline: ${(kept.mm3 / 1000).toFixed(1)} mL`,
      body: "<p>It is in the scene now; not saved yet.</p>",
      actions: [
        { label: "Show in Diffusion", onClick: () => { void shell.showPanel("diffusion"); } },
        { label: "Save to DICOM", primary: true, busyLabel: "Saving…", doneLabel: "Saved ✓", failedLabel: "Not saved", onClick: () => saveKept() },
        { label: "Later", onClick: () => {} },
      ],
    });
  }

  /** What the slice views show as background now, in this module's terms. */
  const shownNow = (scan: Scan | undefined): Show | "" => {
    if (!scan) return "";
    const refs = ([...live.nodes.values()].find((n) => n.type === "sliceComposite")?.refs as Record<string, string[]> | undefined) ?? {};
    const bg = (refs.background ?? [])[0], fg = (refs.foreground ?? [])[0];
    const c = computed.get(scan.browserId);
    if (c?.colorFaId && (bg === c.colorFaId || fg === c.colorFaId)) return "colorfa";
    if (c?.faId && (bg === c.faId || fg === c.faId)) return "fa";
    if (bg && scan.frameIds.includes(bg)) return "signal";
    return "";
  };
  /** A transform on the scan or a segmentation: refused in plain words, not ignored (critic, finding 5). */
  const moved = (n: MrsonNode | undefined) => !!n && !isIdentity(worldForNode(n, live.nodes));
  /**
   * THE REVERSED PHASE-ENCODING SCAN for a diffusion scan: another sequence (or single volume) of the same study, all of
   * whose images are b = 0 -- the pair distortion correction needs (distortion.ts). Its slab may be placed differently
   * (11 of ds001226's 29: 2.3 mm, 0.8°): it is then aligned by the scanner's coordinates (planning.ts resampleInto), as Mike
   * Halle's pipeline does; until 2026-10-02 such a partner was not found at all. Near enough to be one: within 30 mm and 15°.
   */
  const partnerFor = (scan: Scan): Partner | undefined => {
    const ref = live.nodes.get(scan.frameIds[0]);
    if (!ref) return undefined;
    const R = ref.ijkToRAS as number[];
    const same = (n: MrsonNode) => {
      const P = n.ijkToRAS as number[] | undefined;
      if (!P || (n.origin as Record<string, unknown> | undefined)?.studyInstanceUID !== scan.study) return false;
      if (Math.hypot(P[3] - R[3], P[7] - R[7], P[11] - R[11]) > 30) return false;
      for (const c of [0, 1, 2]) {
        const a = [P[c], P[4 + c], P[8 + c]], b = [R[c], R[4 + c], R[8 + c]];
        if ((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(...a) * Math.hypot(...b)) < Math.cos(15 * Math.PI / 180)) return false;
      }
      return true;
    };
    const b0 = (n: MrsonNode) => { const d = (n.origin as Record<string, unknown> | undefined)?.diffusion as { bValue?: number } | undefined; return d?.bValue !== undefined && d.bValue < 50; };
    for (const b of sequenceBrowsers(live)) {
      if (b.id === scan.browserId) continue;
      const { frames, sequence } = browserFrames(live, b.id);
      const nodes = frames.map((f) => live.nodes.get(f.node)).filter(Boolean) as MrsonNode[];
      if (nodes.length && nodes.every((n) => same(n) && b0(n))) return { id: b.id, name: String(sequence?.name ?? b.name ?? "reversed scan"), frameIds: nodes.map((n) => n.id), dims: nodes[0].dims as number[], ijkToRAS: nodes[0].ijkToRAS as number[], phaseEncoding: phaseEncodingOfNode(nodes[0]) };
    }
    for (const n of live.nodes.values()) if (n.type === "image" && !(n as { hidden?: boolean }).hidden && !scan.frameIds.includes(n.id) && same(n) && b0(n)) return { id: n.id, name: String(n.name ?? "reversed scan"), frameIds: [n.id], dims: n.dims as number[], ijkToRAS: n.ijkToRAS as number[], phaseEncoding: phaseEncodingOfNode(n) };
    return undefined;
  };
  /** DISTORTION CORRECTION with the reversed scan (planning.ts correctWithReversed). */
  async function correctDistortion(dwi: DiffusionSeries, partner: Partner, times: StageTimes, scanPE?: string, field?: { fit?: FieldFit; sign?: 1 | -1 }): Promise<string> {
    const vols: ArrayLike<number>[] = [];
    for (const id of partner.frameIds) { const n = live.nodes.get(id); if (!n) throw new Error("the reversed scan was taken out of the scene"); vols.push((await fetchZarrVolumeNative(live.blobBase(), n.zarr as ZarrDesc)).data); }
    // The scanner's record decides whether the two are a reversed pair, when the files carry it (2026-10-03).
    return await correctWithReversed(dwi, vols, partner.name, say, { partnerGrid: { dims: partner.dims, ijkToRAS: partner.ijkToRAS }, times, phaseEncoding: { scan: scanPE, partner: partner.phaseEncoding }, ...(field ? { apply: false, field } : {}) });
  }
  /** The maps' own steps (read, distortion, tensor) from the last time they were made, for the next run's timing; taken once. */
  let fitTimes: StageTimes | undefined;
  const takeFitTimes = (): StageTimes => { const t = fitTimes ?? {}; fitTimes = undefined; return t; };

  // ── computing ─────────────────────────────────────────────────────────────────────────────────────────────────
  async function ensureFit(scan: Scan): Promise<Computed> {
    const have = computed.get(scan.browserId);
    const partner = correct ? partnerFor(scan) : undefined;
    // THE MRI OF THE ANATOMY (2026-10-03, Ron: "Number three, go"; registration.ts): with one loaded, the scan is aligned
    // to it and read once onto its axes, so the maps and the tracts are made in its space.
    const anat = anatomyFor(scan), anatOk = anat && !moved(anat) ? anat : undefined;
    // Still valid when made with the same b range, the same reversed scan (or none) and the same anatomy (or none).
    const motionRule: MotionRuleId = preCorrected ? 0 : MOTION_RULE;
    if (have && have.maxB === adv.maxB && have.partnerId === (partner?.id ?? "") && (have.anatomyId ?? "") === (anatOk?.id ?? "") && have.motionRule === motionRule) return have;
    if (have) dropMaps(have);
    if (moved(live.nodes.get(scan.frameIds[0]))) throw new Error("the scan has a transform (Transforms module); the maps and tracts are not computed on a moved scan yet — harden or remove the transform first");
    say(`Reading ${scan.frameIds.length} diffusion volumes…`);
    const t0 = performance.now();
    const vols: Volume[] = [];
    for (const id of scan.frameIds) {
      const n = live.nodes.get(id);
      if (!n) throw new Error("the scan was taken out of the scene");
      const z = await fetchZarrVolumeNative(live.blobBase(), n.zarr as ZarrDesc);
      vols.push({ dims: n.dims as [number, number, number], ijkToRAS: n.ijkToRAS as number[], data: z.data as Volume["data"], dtype: z.dtype, meta: n.origin as Record<string, unknown> });
    }
    let dwi = fromDicomVolumes(vols, scan.name);
    // The second opinion reads the directions before any correction touches the volumes (the correction moves voxels,
    // not directions); it compares b-values and directions only.
    check(scan, { ...dwi, volumes: [] });
    const times: StageTimes = { read: performance.now() - t0 };
    const field: { fit?: FieldFit; sign?: 1 | -1 } = {};
    let corrected = partner ? await correctDistortion(dwi, partner, times, scan.phaseEncoding, field) : correct ? "not corrected (no reversed phase-encoding scan of this study is loaded)" : "not corrected (switched off)";
    // Head movement, the field and the alignment to the MRI of the anatomy in one resampling (prepare.ts, the case runs'
    // path too). A doubtful alignment is not used (until aligning by hand is built): the scanner's placement stands.
    const z = anatOk ? await fetchZarrVolumeNative(live.blobBase(), anatOk.zarr as ZarrDesc) : undefined;
    const prep = await prepareScan(dwi, { ...(field.fit ? { field: { fit: field.fit, sign: field.sign ?? -1 } } : {}),
      ...(anatOk && z ? { t1: { dims: anatOk.dims as [number, number, number], ijkToRAS: anatOk.ijkToRAS as number[], data: z.data as ArrayLike<number> } } : {}),
      motionRule, ...(scan.phaseEncoding ? { phaseEncoding: scan.phaseEncoding } : {}), times, say });
    dwi = prep.dwi;
    if (prep.said) corrected += `; ${prep.said}`;
    const aligned = !!prep.alignment && !prep.alignment.doubt;
    const t1 = performance.now();
    say("Fitting the diffusion tensor…");
    await new Promise((r) => setTimeout(r, 0));
    const fit = fitTensors(dwi, { maxB: adv.maxB });
    times.fit = performance.now() - t1;
    times.total = performance.now() - t0;
    fitTimes = times;
    const wantT1 = TRACKING_RULES[TRACKING_RULE].brain === "t1-synthstrip";
    const fromScan = "the brain was taken from the diffusion scan, because ";
    const brain: BrainState = !wantT1 ? { note: "" }
      : !anat ? { note: fromScan + "no MRI of the anatomy is loaded", reason: "no-anatomy" }
      : !anatOk ? { note: fromScan + "the MRI of the anatomy has a transform (Transforms module): harden or remove it", reason: "moved" }
      : !aligned ? { note: fromScan + "the scan could not be aligned to the MRI of the anatomy", reason: "doubt" }
      : { note: "", ask: askBrain(anatOk.id) };
    const c: Computed = { dwi, fit, maxB: adv.maxB, corrected, motionRule, partnerId: partner?.id ?? "", anatomyId: anatOk?.id ?? "", rule: wantT1 ? 2 : TRACKING_RULE, brain };
    computed.set(scan.browserId, c);
    say(`Maps made from ${fit.used.length} volumes up to b = ${adv.maxB}; distortion ${corrected}. ${stageText(times)}.`);
    return c;
  }
  const dropMaps = (c: Computed) => {
    for (const id of [c.faId, c.colorFaId]) if (id && live.nodes.get(id)) removeVolumeFromScene(live, id);
    c.faId = c.colorFaId = undefined;
  };

  async function showMap(scan: Scan, what: Show) {
    if (busy) return;                                         // one computation at a time (critic, finding 10)
    busy = what === "signal" ? "" : "Computing…";
    render();
    try {
      if (what === "signal") {
        const { frames, selected } = browserFrames(live, scan.browserId);
        putBackground(frames[selected]?.node ?? scan.frameIds[0], null);
        return;
      }
      const c = await ensureFit(scan);
      if (what === "fa") {
        if (!c.faId || !live.nodes.get(c.faId)) {
          // A computed map: it does not take the 3D view over from the scan (autoVolumeRendering: false).
          const r = await loadVolumeIntoScene(live, store, { dims: c.fit.dims, ijkToRAS: c.fit.ijkToRAS, data: c.fit.fa, dtype: "<f4", name: `${scan.name} FA` }, { name: `${scan.name} FA`, extra: { autoVolumeRendering: false, recomputable: true } });
          c.faId = r.imageId;
          live.write({ op: "patch", id: r.displayId, path: "#/window", value: 1 });
          live.write({ op: "patch", id: r.displayId, path: "#/level", value: 0.5 });
        }
        putMap(scan, c.faId);
      } else {
        if (!c.colorFaId || !live.nodes.get(c.colorFaId)) {
          // FA times the principal direction's absolute components, one byte per color (tensor.ts colorFA), packed into
          // one sample that the slice views draw as color (render/fields.ts ImageFieldOpts.rgb24).
          const rgb = colorFA(c.fit), n = c.fit.fa.length, packed = new Float32Array(n), q = (x: number) => Math.max(0, Math.min(255, Math.round(x * 255)));
          for (let v = 0; v < n; v++) packed[v] = packRGB24(q(rgb[3 * v]), q(rgb[3 * v + 1]), q(rgb[3 * v + 2]));
          const r = await loadVolumeIntoScene(live, store, { dims: c.fit.dims, ijkToRAS: c.fit.ijkToRAS, data: packed, dtype: "<f4", name: `${scan.name} Color FA` }, { name: `${scan.name} Color FA`, extra: { rgb24: true, autoVolumeRendering: false, recomputable: true } });
          c.colorFaId = r.imageId;
        }
        putMap(scan, c.colorFaId);
      }
    } catch (e) {
      say(`${what === "fa" ? "FA" : what === "colorfa" ? "Color FA" : "The scan"} could not be shown: ${(e as Error).message}`);
    } finally { busy = ""; render(); }
  }
  /** The slice views' background; `foreground` undefined leaves the layer over it as it is, null takes it away. */
  const putBackground = (imageId: string, foreground?: string | null, opacity?: number) => {
    for (const cmp of [...live.nodes.values()].filter((n) => n.type === "sliceComposite")) {
      live.write({ op: "patch", id: cmp.id, path: "#/refs/background", value: [imageId] });
      if (foreground !== undefined) live.write({ op: "patch", id: cmp.id, path: "#/refs/foreground", value: foreground ? [foreground] : [] });
      if (opacity !== undefined) live.write({ op: "patch", id: cmp.id, path: "#/foregroundOpacity", value: opacity });
    }
  };
  /**
   * A MAP (FA, Color FA) OVER THE ANATOMY (Ron, 2026-10-01: "when I click color fa, the slices revert to b0. can you
   * overlay them on the t1 instead?"): with the case's anatomical MRI loaded, the map is the layer over it at half
   * opacity (a slider the user moved stays where it is); without one, the map is the background, as before.
   */
  const putMap = (scan: Scan, mapId: string) => {
    const anat = anatomyFor(scan);
    if (!anat) { putBackground(mapId, null); return; }
    const cur = [...live.nodes.values()].find((n) => n.type === "sliceComposite")?.foregroundOpacity as number | undefined;
    putBackground(anat.id, mapId, cur && cur > 0.05 ? undefined : 0.5);
  };

  /**
   * Seeds: every voxel of the diffusion grid inside the brain and above the stopping FA that is INSIDE the segment, or
   * within `mm` of its boundary. The boundary is kept whole (every segment voxel with a neighbor outside it), so the
   * edge is exact; "within 0 mm" means inside (critic, finding 18).
   */
  async function seedsNear(fit: TensorFit, seg: MrsonNode, labelValue: number, mm: number): Promise<number[][]> {
    const z = await fetchZarrVolumeNative(live.blobBase(), seg.zarr as ZarrDesc);
    const [sx, sy, sz] = seg.dims as number[], S = seg.ijkToRAS as number[], Sinv = invAffine(S), lab = z.data;
    const at = (i: number, j: number, k: number) => i >= 0 && j >= 0 && k >= 0 && i < sx && j < sy && k < sz && Number(lab[(k * sy + j) * sx + i]) === labelValue;
    const edge: [number, number, number][] = [];
    for (let k = 0; k < sz; k++) for (let j = 0; j < sy; j++) for (let i = 0; i < sx; i++) {
      if (!at(i, j, k)) continue;
      if (!at(i - 1, j, k) || !at(i + 1, j, k) || !at(i, j - 1, k) || !at(i, j + 1, k) || !at(i, j, k - 1) || !at(i, j, k + 1)) edge.push(matVec(S, i, j, k));
    }
    if (!edge.length) return [];
    const cell = Math.max(mm, 2), key = (a: number, b: number, c: number) => `${a},${b},${c}`;
    const bins = new Map<string, [number, number, number][]>();
    for (const p of edge) { const k = key(Math.floor(p[0] / cell), Math.floor(p[1] / cell), Math.floor(p[2] / cell)); (bins.get(k) ?? bins.set(k, []).get(k)!).push(p); }
    const [nx, ny, nz] = fit.dims, M = fit.ijkToRAS, out: number[][] = [];
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const v = (k * ny + j) * nx + i;
        if (!fit.mask[v] || fit.fa[v] < adv.minFA) continue;
        const p = matVec(M, i, j, k);
        const q = matVec(Sinv, p[0], p[1], p[2]);
        let hit = at(Math.round(q[0]), Math.round(q[1]), Math.round(q[2]));
        if (!hit && mm > 0) {
          const c = [Math.floor(p[0] / cell), Math.floor(p[1] / cell), Math.floor(p[2] / cell)];
          for (let a = -1; a <= 1 && !hit; a++) for (let b = -1; b <= 1 && !hit; b++) for (let d = -1; d <= 1 && !hit; d++) {
            for (const e of bins.get(key(c[0] + a, c[1] + b, c[2] + d)) ?? []) if (Math.hypot(e[0] - p[0], e[1] - p[1], e[2] - p[2]) <= mm) { hit = true; break; }
          }
        }
        if (hit) out.push(p);
      }
      if (k % 8 === 7) await new Promise((r) => setTimeout(r, 0));
    }
    return out;
  }

  /** Tracking in batches, the page answering between them (critic, finding 15: 2.5 s of frozen page in one piece). */
  /**
   * UKF ON THE GRAPHICS CARD (ukf-gpu.ts), in batches. Seeds are RAS here and voxels there. Checked against the
   * processor version (ukf.ts) on PAT16, 1,200 seeds: the same seeds give fibers, half of them identical, tract-density
   * maps correlate 0.875 (overlap 0.885); one fiber in ten ended > 17 mm away -- found and fixed 2026-09-30 (90% of ends
   * now within 0.06 mm, the README's numbers)
   * (dmri-review-2026-09-28.md). 12x faster (3.2 s against 40.7 s).
   */
  /** The last tracking's breakdown, for the status line (planning.ts TrackTiming, plus the signal's preparation). */
  let lastTiming: (TrackTiming & { data: number }) | undefined;
  /** Rule 3's brain into a fit, waited for here -- the tracking is what uses it. Asked again when the last answer was a
   *  failure that can go away (the server was not running, ran without SynthStrip, or the job failed). */
  let waitingForBrain = false, brainLine = "";
  const askBrain = (anatomyId: string) => synthstripBrainMask(live, anatomyId, (line) => { brainLine = line; if (waitingForBrain) say(`Finding the brain on the MRI of the anatomy: ${line}…`); });
  async function brainFor(c: Computed): Promise<void> {
    const b = c.brain;
    if (c.rule === TRACKING_RULE || !c.anatomyId) return;
    if (b.reason === "no-anatomy" || b.reason === "moved" || b.reason === "doubt") return;
    b.ask ??= askBrain(c.anatomyId);
    waitingForBrain = true;
    say(`Finding the brain on the MRI of the anatomy${brainLine ? `: ${brainLine}` : ""}…`);
    const r = await b.ask.finally(() => { waitingForBrain = false; });
    b.ask = undefined;
    if (r.ok) {
      const w = withBrainFromT1(c.fit, r.mask, undefined, TRACKING_RULES[TRACKING_RULE].fluidMdMax, TRACKING_RULES[TRACKING_RULE].fluidShellVoxels);
      c.fit = w.fit; c.ukf = undefined; c.rule = TRACKING_RULE;
      b.reason = undefined;
      b.note = "the brain was found on the MRI of the anatomy (SynthStrip)" + (w.outsideGridMl >= 1 ? `; ${w.outsideGridMl.toFixed(0)} mL of it lies outside the diffusion scan's grid and is not tracked` : "");
    } else { b.reason = r.reason; b.note = `the brain was taken from the diffusion scan, because ${r.message}`; }
  }
  async function trackUkf(c: Computed, seedsRAS: number[][]): Promise<Float32Array[]> {
    await brainFor(c);
    const t0 = performance.now();
    // THE TRACKING RULE (tracking-rules.ts): the shell, the mask and the thresholds. "Stop below FA" under Advanced shows
    // and sets the two-tensor tracker's own value when that tracker is chosen (adv.ukfStopFA, the rule's 0.08 to start
    // with), the single-tensor tracker's (adv.minFA, 0.15) otherwise (critic, 2026-10-03, finding 14).
    const rule = TRACKING_RULES[c.rule];
    c.ukf ??= ukfDataFor(c.dwi, c.fit, rule);
    const stopFA = adv.ukfStopFA;
    const timing: TrackTiming & { data: number } = { prepare: 0, gpu: 0, assemble: 0, between: 0, data: performance.now() - t0 };
    // The 3D view's drawing is held while the card tracks (holdDrawing: macOS's watchdog took the card when tracking ran
    // beside the solid anatomy, Ron's window, 2026-10-03 build 16:45).
    const release = holdDrawing("tracking");
    const out = await trackUkfSeeds(device, c.ukf, seedsRAS, stopFA, (f) => { busy = `${adding ? "Adding lines" : "Making tracts"}… ${Math.round(100 * f)}%`; render(); }, timing, { ...rule.ukf, stoppingFA: stopFA }).finally(release);
    lastTiming = timing;
    return out;
  }
  /** PARALLEL TRANSPORT TRACKING on fiber distributions, in workers bundled with the extension (extension.json "workers").
   *  The distributions are kept with the scan's computations, so a second run tracks at once. */
  async function trackPtt(c: Computed, seedsRAS: number[][]): Promise<Float32Array[]> {
    if (!c.fod) {
      busy = "Fiber distributions… 0%"; render();
      const k = kernelFromResponses(estimateResponses(c.dwi));
      c.fod = await csdVolume(c.dwi, c.fit.mask, k, { workerUrl: assetUrl("diffusion", "workers/csd-worker.js"), onProgress: (f) => { busy = `Fiber distributions… ${Math.round(100 * f)}%`; render(); } });
    }
    return await trackPttParallel(c.fod, seedsRAS, { workerUrl: assetUrl("diffusion", "workers/ptt-worker.js"), onProgress: (f) => { busy = `Making tracts… ${Math.round(100 * f)}%`; render(); } });
  }
  /** The tracts from these seeds, by the method chosen. */
  async function follow(c: Computed, seeds: number[][]): Promise<Float32Array[]> {
    if (method === "ukf") return await trackUkf(c, seeds);
    if (method === "ptt") return await trackPtt(c, seeds);
    return (await track(c.fit, seeds)).map((x) => x.points);
  }

  async function track(fit: TensorFit, seeds: number[][]): Promise<Streamline[]> {
    const out: Streamline[] = [];
    for (let s = 0; s < seeds.length; s += SEEDS_PER_BATCH) {
      out.push(...trackFromSeeds(fit, seeds.slice(s, s + SEEDS_PER_BATCH), { minFA: adv.minFA, maxAngleDeg: adv.maxAngleDeg, stepVoxels: adv.stepVoxels }));
      if (seeds.length > SEEDS_PER_BATCH) { busy = `Making tracts… ${Math.round(100 * Math.min(1, (s + SEEDS_PER_BATCH) / seeds.length))}%`; render(); }
      await new Promise((r) => setTimeout(r, 0));
    }
    return out;
  }

  function addGroup(name: string, scan: Scan, strands: Float32Array[]) {
    groups.push({ id: ++groupSeq, name: `${name}${method === "ukf" ? "" : method === "ptt" ? " (smooth curves)" : " (single tensor)"}`, scan: scan.browserId, strands, visible: true, method });
    redraw3d();
  }

  async function makeTracts() {
    const scan = scans().find((s) => s.browserId === chosen);
    const target = segmentChoices(scan).find((c) => c.key === near);
    if (!scan || !target || busy) return;
    if (moved(target.seg)) { say(`"${target.seg.name}" has a transform (Transforms module); tracts are not started near a moved segmentation yet — harden or remove the transform first.`); render(); return; }
    busy = "Making tracts…"; render();
    if (method === "ukf" || method === "ptt") { await makeNamedTracts(scan, target); return; }
    try {
      const c = await ensureFit(scan);
      say(`Finding white matter within ${withinMm} mm of ${target.label}…`);
      const t0 = performance.now();
      const seeds = await seedsNear(c.fit, target.seg, target.labelValue, withinMm);
      if (!seeds.length) { say(`No white matter ${withinMm ? `within ${withinMm} mm of` : "inside"} ${target.label} (inside the brain and above FA ${adv.minFA}).`); return; }
      const sl = await follow(c, seeds);
      addGroup(`Near ${target.label}`, scan, sl);
      say(`${sl.length.toLocaleString()} tracts from ${seeds.length.toLocaleString()} starting points ${withinMm ? `within ${withinMm} mm of` : "inside"} ${target.label}, in ${((performance.now() - t0) / 1000).toFixed(1)} s.`);
    } catch (e) { say(`Tracts could not be made: ${(e as Error).message}`); }
    finally { busy = ""; render(); }
  }

  /**
   * TRACTS NEAR A STRUCTURE, NAMED (Ron, 2026-09-30: whole brain first, then keep what is near the tumor, whole): UKF
   * through the whole brain, TractCloud names every streamline (tractcloud/), and each named tract that comes within
   * `withinMm` of the structure is shown -- all of it, since planning asks what a connection would lose. Unnamed
   * streamlines that pass that close are shown too, in gray (hiding a real tract is worse than showing an unnamed one);
   * everything else is kept, hidden, as "Rest of the brain".
   */
  async function makeNamedTracts(scan: Scan, target: { seg: MrsonNode; labelValue: number; label: string }) {
    try {
      const c = await ensureFit(scan);
      const times: StageTimes = takeFitTimes(), before = times.total ?? 0;
      const t0 = performance.now();
      if (method === "ukf") await brainFor(c);
      const seeds = method === "ukf" ? seedsFor(c.fit, TRACKING_RULES[c.rule]) : wholeBrainSeeds(c.fit);
      times.seeds = performance.now() - t0;
      say(`Following tracts through the whole brain from ${seeds.length.toLocaleString()} starting points…`);
      lastTiming = undefined;
      const sl = method === "ptt" ? await trackPtt(c, seeds) : await trackUkf(c, seeds);
      const t1 = performance.now();
      times.track = t1 - t0 - times.seeds;
      if (lastTiming) times.trackDetail = lastTiming;
      busy = "Naming tracts…"; render();
      const model = await tractCloud();
      tractCount = model.json.tracts.length;
      const rp = await rapidParc(), releaseN = holdDrawing("naming tracts");
      const named = await nameTracts(device, model, sl, { rapidParc: rp }).finally(releaseN);
      times.name = named.seconds * 1000;
      const tDist = performance.now();
      busy = "Measuring distances…"; render();
      const z = await fetchZarrVolumeNative(live.blobBase(), target.seg.zarr as ZarrDesc), lab = z.data;
      const structure: Structure = { dims: target.seg.dims as number[], ijkToRAS: target.seg.ijkToRAS as number[], inside: (v) => Number(lab[v]) === target.labelValue };
      const grayMm = withinMm + GRAY_BAND_MM;
      const dist = await streamlineDistances(structure, sl, grayMm + 2);
      // Streamlines that cross the fluid at the brain's edge would keep no tract's name (outside-brain.ts; Ron, 2026-10-04:
      // "The only bad thing is mislabeling them"). OFF: it also took corticospinal fibers along the medulla.
      const outside = OUTSIDE_RULE.on ? outsideBrain(sl, c.fit, edgeFluid(c.fit)) : undefined;
      const sorted = sortByDistance(model, named, dist, withinMm, MIN_NEAR_STREAMLINES, grayMm, outside), nearTracts = sorted.near;
      times.distances = performance.now() - tDist;
      const tDraw = performance.now();
      // STILL WANTED? A scan removed while this ran leaves nothing behind (critic, 2026-10-01, finding 8; CONSTRAINTS).
      if (!scans().some((s) => s.browserId === scan.browserId)) return;
      // A NEW RUN REPLACES THE LAST ONE for this scan (critic, finding 5: a second press doubled every tract).
      groups.splice(0, groups.length, ...withoutLastRun(groups, scan.browserId));   // face.ts
      const pick = (idx: number[]) => idx.map((i) => sl[i]);
      runs.set(scan.browserId, { sl, named, sorted, structure, label: target.label, withinMm, method, added: new Set(), dist, maxB: c.maxB, partnerId: c.partnerId, anatomyId: c.anatomyId ?? "", rule: method === "ukf" ? c.rule : undefined, brain: method === "ukf" ? c.brain.note : "" });
      // The anatomy behind the slices, where the tracts' crossings are drawn (Yogesh Rathi via Ron, 2026-10-01), and over
      // it this scan's own map -- never another patient's left from before (critic, 2026-10-01, finding 3).
      const anat = anatomyFor(scan);
      if (anat) {
        const fg = ((([...live.nodes.values()].find((n) => n.type === "sliceComposite")?.refs as Record<string, string[]> | undefined)?.foreground) ?? [])[0];
        const mine = fg !== undefined && (fg === c.colorFaId || fg === c.faId);
        putBackground(anat.id, mine ? undefined : (c.colorFaId && live.nodes.get(c.colorFaId) ? c.colorFaId : null));
      }
      // The near tracts shown; the faint ones (within reach, fewer than the minimum) listed after them, hidden.
      for (const [e, faint] of [...nearTracts.map((e) => [e, false] as const), ...sorted.faint.map((e) => [e, true] as const)]) {
        const t = model.json.tracts[e.tract];
        groups.push({ id: ++groupSeq, name: t ? tractLabel(t.abbr, t.name, e.side) : tractName(model, e.tract, e.side), scan: scan.browserId, strands: pick(e.idx), visible: !faint, method,
          tract: e.tract, side: e.side, distanceMm: e.d, within: e.within, run: true, faint, otherSide: e.side ? sorted.total(e.tract, otherSide(e.side)) : undefined });
      }
      // The rest: the unnamed far ones and the far tracts not listed as faint (a faint tract is its own row).
      const unnamedNear = sorted.unnamedNear, faintSet = new Set(sorted.faint), rest = [...sorted.unnamedFar, ...sorted.outsideFar, ...sorted.far.filter((e) => !faintSet.has(e)).flatMap((e) => e.idx)];
      if (sorted.outsideNear.length) groups.push({ id: ++groupSeq, name: `Outside the brain, possibly a cranial nerve, within ${withinMm} mm of ${target.label}`, scan: scan.browserId, strands: pick(sorted.outsideNear), visible: true, method, unnamed: true, run: true, outside: true });
      if (unnamedNear.length) groups.push({ id: ++groupSeq, name: `Not named, within ${withinMm} mm of ${target.label}`, scan: scan.browserId, strands: pick(unnamedNear), visible: true, method, unnamed: true, run: true });
      if (rest.length) groups.push({ id: ++groupSeq, name: "Rest of the brain", scan: scan.browserId, strands: pick(rest), visible: false, method, unnamed: true, run: true });
      redraw3d();
      const t2 = performance.now();
      times.draw = t2 - tDraw;
      times.total = before + (t2 - t0);
      say(`${nearTracts.length} named tracts come within ${withinMm} mm of ${target.label} (at least ${MIN_NEAR_STREAMLINES} streamlines each)${sorted.faint.length ? `; ${sorted.faint.length} more come that close with fewer, or within ${grayMm} mm (listed in gray, hidden)` : ""}${unnamedNear.length ? `, and ${unnamedNear.length.toLocaleString()} streamlines no name fits` : ""}${sorted.outsideNear.length ? `; ${sorted.outsideNear.length.toLocaleString()} streamlines that close run outside the brain (possibly a cranial nerve) and keep no tract's name` : ""}. ` +
        `${sl.length.toLocaleString()} streamlines through the whole brain. Step by step: ${stageText(times)}.`);
    } catch (e) { say(`Tracts could not be made: ${(e as Error).message}`); }
    finally { busy = ""; render(); }
  }

  /**
   * ADD LINES TO THE CHOSEN TRACTS, BOTH SIDES (Ron, 2026-10-01: "artificially prop up tracts like the right uncinate by
   * doing a second run with more seed points just in the tracts that were selected by the user"; then: "User select
   * tracts of interest. The updated workflow adds a button 'Add lines'" -- the tracts shown are the ones chosen, as he
   * chooses them by turning the others off); O'Donnell et al. 2017 seeded
   * tumor patients at 20 a voxel and suggest tracking again "in the region of the detected fiber tracts"): starting
   * points in every voxel the tract's streamlines pass through, on both sides so the sides stay comparable, followed by
   * the run's method, named against the whole-brain run (nameAgainst), and the ones named as this tract added to it.
   * Measured on PAT16 (Deno, 2026-10-01): the right uncinate 1 -> 18 streamlines; about 5-10% of the new streamlines
   * belong to the tract (the rest cross it), so one press is held to 16,000 starting points (MORE_MAX_SEEDS).
   */
  async function addLines(chosen: TractGroup[]) {
    const scanId = chosen[0]?.scan, run = scanId ? runs.get(scanId) : undefined, scan = scans().find((s) => s.browserId === scanId);
    const picked = chosen.filter((g) => g.scan === scanId && g.run && g.tract !== undefined && !run?.added.has(`${g.tract}:${g.side ?? 0}`));
    if (!run || !scan || !picked.length || busy) return;
    // Every chosen tract, on both of its sides -- those not added to already.
    const want = new Map<string, { tract: number; side: number }>();
    for (const g of picked) for (const sd of g.side ? [g.side, otherSide(g.side)] : [0]) if (!run.added.has(`${g.tract}:${sd}`)) want.set(`${g.tract}:${sd}`, { tract: g.tract!, side: sd });
    busy = "Making tracts…"; adding = true; render();
    try {
      const c = await ensureFit(scan), t0 = performance.now();
      // THE SAME SPACE (critic, 2026-10-03, finding 1): the scan aligned to the same anatomy MRI, or to none, as the run.
      if (c.maxB !== run.maxB || c.partnerId !== run.partnerId || (c.anatomyId ?? "") !== run.anatomyId) {
        say("The maps were made again since these tracts were found (Highest b, the distortion correction, or the MRI of the anatomy changed): press “Show the fiber tracts near the tumor” again first.");
        return;
      }
      const all = [...run.sorted.near, ...run.sorted.far];
      const foot = [...want.values()].flatMap(({ tract, side }) => all.find((e) => e.tract === tract && e.side === side)?.idx ?? []).map((i) => run.sl[i]);
      const times: StageTimes = takeFitTimes(), before = times.total ?? 0, ts = performance.now();
      const seeds = denseSeeds(foot, { dims: c.fit.dims, ijkToRAS: c.fit.ijkToRAS });
      times.seeds = performance.now() - ts;
      const what = picked.length === 1 ? picked[0].name.replace(/, (right|left)/, "") : `${new Set(picked.map((g) => g.tract)).size} tracts`;
      say(`Adding lines to ${what} from ${seeds.length.toLocaleString()} starting points…`);
      lastTiming = undefined;
      const tt = performance.now();
      const sl2 = run.method === "ptt" ? await trackPtt(c, seeds) : await trackUkf(c, seeds);
      times.track = performance.now() - tt;
      if (lastTiming) times.trackDetail = lastTiming;
      busy = "Naming tracts…"; render();
      const model = await tractCloud();
      const rp = await rapidParc(), releaseN = holdDrawing("naming tracts");
      const r = await nameAgainst(device, model, run.sl, sl2, { rapidParc: rp }).finally(releaseN);
      const outside2 = OUTSIDE_RULE.on ? outsideBrain(sl2, c.fit, edgeFluid(c.fit)) : new Uint8Array(sl2.length);   // added lines that cross the fluid join no tract
      times.name = r.added.seconds * 1000;
      const tDist = performance.now();
      if (!scans().some((s) => s.browserId === scanId) || runs.get(scanId!) !== run) return;   // left, or run again, meanwhile
      let total = 0;
      busy = "Measuring distances…"; render();
      for (const { tract, side: sd } of want.values()) {
        const mine = [...sl2.keys()].filter((i) => !outside2[i] && r.added.tract[i] === tract && r.added.side[i] === sd).map((i) => sl2[i]);
        if (!mine.length) continue;
        const d = await streamlineDistances(run.structure, mine, run.withinMm + 2);
        let h = groups.find((x) => x.scan === scanId && x.run && x.tract === tract && x.side === sd);
        if (!h) {
          // The other side was not listed (not within reach): it joins the list for comparison, hidden (Ron, 2026-10-01:
          // "add lines turns on the left tracts even if they were not turned on") -- WITH its streamlines from the run,
          // taken out of "Rest of the brain", so both sides count run plus added (critic, 2026-10-01, finding 2).
          const t = model.json.tracts[tract];
          const idx = all.find((e) => e.tract === tract && e.side === sd)?.idx ?? [];
          const own = idx.map((i) => run.sl[i]), ownSet = new Set(own);
          const rest = groups.find((x) => x.scan === scanId && x.run && x.name === "Rest of the brain");
          if (rest && own.length) rest.strands = rest.strands.filter((f) => !ownSet.has(f));
          h = { id: ++groupSeq, name: tractLabel(t.abbr, t.name, sd), scan: scanId!, strands: own, visible: false, method: run.method, tract, side: sd,
            distanceMm: idx.reduce((m, i) => Math.min(m, run.dist[i]), Infinity), within: idx.filter((i) => run.dist[i] <= run.withinMm).length, run: true };
          groups.push(h);
        }
        h.strands = [...h.strands, ...mine];
        h.more = (h.more ?? 0) + mine.length;
        h.within = (h.within ?? 0) + [...d].filter((x) => x <= run.withinMm).length;
        h.distanceMm = Math.min(h.distanceMm ?? Infinity, ...d);
        // Faint by the module's own rule, every time (critic, 2026-10-01, finding 11): within reach, fewer than the minimum.
        // Gray as the run made it: a band row (none within the distance) stays gray until 5 come within it; a row added for
        // comparison (the other side) is gray only when some, but fewer than 5, come within it (critic, 2026-10-04, finding 6).
        h.faint = (h.within ?? 0) < MIN_NEAR_STREAMLINES && (!!h.faint || (h.within ?? 0) > 0);
        // Shown or hidden as it was: only what the person turned on is shown.
        total += mine.length;
      }
      // Each side's count of the other, as it now stands.
      for (const { tract, side: sd } of want.values()) {
        if (!sd) continue;
        const h = groups.find((x) => x.scan === scanId && x.run && x.tract === tract && x.side === sd), o = groups.find((x) => x.scan === scanId && x.run && x.tract === tract && x.side === otherSide(sd));
        if (h) h.otherSide = o ? o.strands.length : run.sorted.total(tract, otherSide(sd));
      }
      times.distances = performance.now() - tDist;
      const tDraw = performance.now();
      redraw3d();
      for (const k of want.keys()) run.added.add(k);
      times.draw = performance.now() - tDraw;
      times.total = before + (performance.now() - t0);
      say(`${total.toLocaleString()} lines added to ${what}, both sides, from ${seeds.length.toLocaleString()} starting points. Step by step: ${stageText(times)}.`);
    } catch (e) { say(`Lines could not be added: ${(e as Error).message}`); }
    finally { busy = ""; adding = false; render(); }
  }

  /**
   * "Seed where I click…": the next point placed in a view, while placement is on for THIS, seeds a 5 mm ball there.
   * Leaving placement (Escape, another module) disarms it; a point placed later elsewhere is not taken (critic, finding 8).
   */
  function seedWhereIClick() {
    const scan = scans().find((s) => s.browserId === chosen);
    if (!scan || busy) return;
    const pointsNow = () => new Map([...live.nodes.values()].filter((n) => n.type === "markup").map((n) => [n.id, ((n.controlPoints as unknown[] | undefined) ?? []).length]));
    const before = pointsNow();
    let sawPlace = false;
    seeding = true; render();
    say("Click in a view where the tracts should start (Escape cancels).");
    if (!startPlacing("fiducial", false)) { seeding = false; render(); say("Placing points is not available in this app."); return; }
    // Placement is on from here (startPlace wrote it before anything below subscribed): leaving it without a point is
    // a cancel.
    sawPlace = live.nodes.get("local-interaction")?.mode === "place";
    const stop = () => { seeding = false; off(); cancelSeeding = undefined; render(); };
    cancelSeeding = stop;
    const off = live.subscribe(async (ch) => {
      if (!seeding) return;
      const inter = live.nodes.get("local-interaction");
      if (inter?.mode === "place") sawPlace = true;
      const fresh = [...live.nodes.values()].filter((n) => n.type === "markup" && ((n.controlPoints as unknown[] | undefined) ?? []).length > (before.get(n.id) ?? 0));
      const p = fresh.length ? ((fresh[0].controlPoints as { position: number[] }[]).at(-1)?.position) : undefined;
      if (!p) {
        // Placement ended without a point: cancelled.
        if (ch.id === "local-interaction" && sawPlace && inter?.mode !== "place") { stop(); say("Seeding cancelled."); }
        return;
      }
      stop();
      busy = "Making tracts…"; render();
      try {
        const c = await ensureFit(scan);
        const sl = await follow(c, seedsInSphere(p, 5));
        addGroup(`Seed ${groups.filter((x) => x.name.startsWith("Seed")).length + 1}`, scan, sl);
        say(sl.length ? `${sl.length.toLocaleString()} tracts from the point clicked.` : "No tracts from that point: it is not in white matter (FA below the stopping value).");
      } catch (e) { say(`Tracts could not be made: ${(e as Error).message}`); }
      finally { busy = ""; render(); }
    });
  }

  /**
   * SHORTER ENDS, FOR DRAWING (Ron, 2026-10-01, after Add lines: "the ends of the tracts are 'frazzled' any way make them
   * slightly shorter?"): every streamline drawn -- in 3D, as dots on the slices, and for the probe -- loses `trimVoxels`
   * voxels of the scan's grid at each end (tract-slice.ts trimEnds). The streamlines kept, measured and saved are whole.
   */
  let trimVoxels = 2;
  /** "Add lines" is running: its button shows the step (tracking with a percentage, naming, measuring). */
  let adding = false;
  const trimCache = new WeakMap<TractGroup, { key: string; strands: Float32Array[] }>();
  const voxelMm = (scanId: string) => { const M = computed.get(scanId)?.fit.ijkToRAS; return M ? Math.min(...[0, 1, 2].map((c) => Math.hypot(M[c], M[4 + c], M[8 + c]))) : 2; };
  function drawn(g: TractGroup): Float32Array[] {
    const mm = trimVoxels * voxelMm(g.scan), key = `${mm}:${g.strands.length}`;
    const hit = trimCache.get(g); if (hit && hit.key === key) return hit.strands;
    const strands = mm > 0 ? g.strands.map((f) => trimEnds(f, mm)).filter((f): f is Float32Array => !!f && f.length >= 6) : g.strands;
    trimCache.set(g, { key, strands });
    return strands;
  }
  /** How many tracts the model names (Other included), once it is loaded; colors need it. */
  let tractCount = 43;
  /** The tracts shown, indexed for the data probe (tract-index.ts); rebuilt with every redraw. */
  let probeIndex: { ix: TractIndex; shown: TractGroup[] } | undefined;
  function redraw3d() {
    const shownNow = groups.filter((g) => g.visible);
    probeIndex = shownNow.length ? { ix: buildTractIndex(shownNow.map(drawn)), shown: shownNow } : undefined;
    const view = live.view;
    if (!view) return;
    // Colored by tract: each named group its tract's color (tract-colors.ts), unnamed ones gray; otherwise, and for
    // groups started by hand, by direction.
    const pal = directionPalette(), count = tractCount;
    pal[GRAY_ID] = UNNAMED;
    let slot = TRACT_ID;
    const strands: Strand[] = groups.filter((g) => g.visible).flatMap((g) => {
      let id = 0;
      if (colorBy === "tract" && g.tract !== undefined && slot <= 255) { id = slot++; pal[id] = tractColor(g.tract, count); }
      else if (colorBy === "tract" && g.unnamed) id = GRAY_ID;
      return drawn(g).map((p) => id ? { points: p, bundle: id } : { points: p, bundle: directionId(p), pointBundles: pointIds(p) });
    });
    drawSliceCrossings();
    const old = field; field = undefined;
    if (!strands.length) view.removeField(FIELD_KEY);
    else { field = new FiberField(device, strands, { radius: RADIUS[drawAs], bundleColors: pal }); view.setField(FIELD_KEY, field); }
    old?.destroy?.();
  }

  /**
   * THE TRACTS ON THE SLICES (Yogesh Rathi via Ron, 2026-10-01: "on the cross sections show the T1 image and
   * intersections of the tubes on the slices"): a dot wherever a shown streamline crosses a slice view's plane, in the
   * color it has in 3D (tract-slice.ts). Each view's plane is read from its slice node; a view that moves draws again.
   */
  const CROSSINGS_LAYER = "diffusion-tracts", MAX_CROSSINGS = 30000;
  let crossingsQueued = false;
  function drawSliceCrossings() {
    const view = live.view as { setOverlay?: (cell: string, layer: string, items: unknown[]) => void } | undefined;
    if (!view?.setOverlay) return;
    const color = (g: TractGroup, dir: number[]): number[] => {
      if (colorBy === "tract" && g.tract !== undefined) { const c = tractColor(g.tract, tractCount); return [c[0], c[1], c[2]]; }
      if (colorBy === "tract" && g.unnamed) return [UNNAMED[0], UNNAMED[1], UNNAMED[2]];
      return [Math.abs(dir[0]), Math.abs(dir[1]), Math.abs(dir[2])];    // by direction, as in 3D: red left-right ...
    };
    // WHOSE TRACTS ON WHICH VIEW (critic, 2026-10-01, finding 3): a view shows the tracts of the scan whose images it
    // shows (its anatomy, its frames, its maps); with tracts for one scan only, that scan's, whatever is underneath.
    const withTracts = [...new Set(groups.filter((g) => g.visible && g.strands.length).map((g) => g.scan))];
    const imagesOf = (scanId: string) => {
      const sc = scans().find((x) => x.browserId === scanId), c = computed.get(scanId);
      return new Set([anatomyFor(sc)?.id, ...(sc?.frameIds ?? []), c?.faId, c?.colorFaId].filter((x): x is string => !!x));
    };
    for (const n of live.nodes.values()) {
      if (n.type !== "view" || n.kind !== "slice" || !Array.isArray(n.sliceToRAS)) continue;
      const cell = String(n.layoutName ?? n.name ?? "");
      const comp = [...live.nodes.values()].find((x) => x.type === "sliceComposite" && x.layoutName === cell);
      const bg = (((comp?.refs as Record<string, string[]> | undefined)?.background) ?? [])[0];
      const scansHere = withTracts.length === 1 ? withTracts : withTracts.filter((id) => bg !== undefined && imagesOf(id).has(bg));
      const shown = groups.filter((g) => g.visible && g.strands.length && scansHere.includes(g.scan));
      const items: unknown[] = [];
      if (shown.length) {
        const m = n.sliceToRAS as number[], L = Math.hypot(m[2], m[6], m[10]) || 1, nrm: [number, number, number] = [m[2] / L, m[6] / L, m[10] / L];
        const d = typeof n.offset === "number" ? n.offset : m[3] * nrm[0] + m[7] * nrm[1] + m[11] * nrm[2];
        const cs = sliceCrossings(shown.map(drawn), { origin: [nrm[0] * d, nrm[1] * d, nrm[2] * d], normal: nrm });
        const step = Math.max(1, Math.ceil(cs.length / MAX_CROSSINGS));      // a very dense slice is thinned evenly
        for (let i = 0; i < cs.length; i += step) items.push({ kind: "point", ras: cs[i].p, color: color(shown[cs[i].set], cs[i].dir), radiusPx: 1.6, inPlaneOnly: true });
      }
      // THIS VIEW'S DOTS ONLY (critic, 2026-10-01, finding 13): in a layer for every view, another plane's crossings
      // showed along the lines where the planes meet.
      view.setOverlay(cell, CROSSINGS_LAYER, items);
    }
  }
  live.subscribe((c) => {
    if (!((c.type === "view" && String(c.id ?? "").startsWith("nativeSlice-")) || c.type === "sliceComposite") || crossingsQueued || !groups.length) return;
    crossingsQueued = true;
    requestAnimationFrame(() => { crossingsQueued = false; drawSliceCrossings(); });
  });

  /** What left the scene takes its computations and tracts with it (critic, finding 9; CONSTRAINTS: a copy is held only
   *  while something reads it). */
  function prune() {
    const ids = new Set(scans().map((s) => s.browserId));
    let changed = false;
    for (const [id, c] of computed) if (!ids.has(id)) { computed.delete(id); dropMaps(c); changed = true; }
    for (const id of [...runs.keys()]) if (!ids.has(id)) runs.delete(id);
    for (let i = groups.length - 1; i >= 0; i--) if (!ids.has(groups[i].scan)) { groups.splice(i, 1); changed = true; }
    for (const id of [...seenScans]) if (!ids.has(id)) seenScans.delete(id);
    if (changed) redraw3d();
    return changed;
  }

  // ── the panel ─────────────────────────────────────────────────────────────────────────────────────────────────
  function render() {
    if (!root) return;
    const list = scans();
    if (!list.find((s) => s.browserId === chosen)) chosen = list[0]?.browserId ?? "";
    const scan = list.find((s) => s.browserId === chosen);
    const segs = segmentChoices(scan);
    if (!segs.find((s) => s.key === near)) near = segs[0]?.key ?? "";
    if (outline && (!live.nodes.get(outline.seedsId) || !live.nodes.get(outline.imageId))) outline = undefined;   // finding 9
    const tumors = tumorChoices(scan);
    // The face measures from a tumor only; Advanced may still measure from any structure (its own "Near").
    const faceNear = nearOnFace(tumors.map((t) => t.key), near);   // face.ts
    root.innerHTML = "";
    // THE FACE (mockup diffusion-workflow-v4; Ron, 2026-10-01: the user is a neurosurgery resident who knows neither the
    // lingo nor the concepts -- "One button as initial, everything else under advanced", tooltips "with lay person
    // level"). 1 · what the case needs, each ticked when it is there; 2 · the one button and its answer; then Advanced.
    const anat = anatomyFor(scan);
    const caseSec = shell.section(root, "1 · The patient's case", { band: "yellow", open: true, note: scan && anat && tumors.length ? "✓" : `${[scan, anat, tumors.length].filter(Boolean).length} of 3` });
    // WHICH PATIENT (critic, finding 10): named on the face; with several, chosen on the face.
    if (list.length > 1) {
      const pp = document.createElement("select");
      for (const s0 of list) pp.append(new Option(`${patientOf(s0.name) || "Patient"} — ${s0.name.replace(/^.*?·\s*(MR\s+)?/, "")}`, s0.browserId, false, s0.browserId === chosen));
      pp.title = "More than one patient's diffusion MRI is loaded: the one the steps below are about.";
      pp.onchange = () => { chosen = pp.value; render(); };
      shell.row(caseSec, "Patient").append(pp);
    } else if (scan && patientOf(scan.name)) {
      const pl = document.createElement("div"); pl.className = "sl-hint"; pl.textContent = `Patient: ${patientOf(scan.name)}`; caseSec.append(pl);
    }
    const need = (ok: boolean, what: string, detail: string, tip: string) => {
      const r = document.createElement("div");
      r.style.cssText = "display:grid;grid-template-columns:16px 1fr;gap:6px;margin:3px 0";
      r.title = tip;
      const m = document.createElement("span"); m.textContent = ok ? "✓" : "○"; m.style.color = ok ? "var(--sl-ok)" : "var(--sl-warn)";
      const t = document.createElement("span"); t.innerHTML = `<b>${what}</b>`;
      const d = document.createElement("span"); d.className = "sl-hint"; d.textContent = ` — ${detail}`; t.append(d);
      r.append(m, t); caseSec.append(r);
    };
    need(!!scan, "Diffusion MRI", scan ? scan.name.replace(/^.*?·\s*(MR\s+)?/, "") : "not loaded; it is often named DTI, DWI or diffusion", "The scan that shows the brain's wiring.");
    need(!!anat, "MRI of the anatomy", anat ? String(anat.name ?? "").replace(/^.*?·\s*(MR\s+)?/, "") : "not loaded; usually the T1 with contrast the tumor was seen on", "The scan the tumor is outlined on.");
    need(tumors.length > 0, "Tumor outline", tumors.length ? (tumors.length > 1 ? `${tumors.length} to choose from (below)` : tumors[0].label) : anat ? `none yet: make it below${segs.length ? " (the outlines loaded are not named as a tumor)" : ""}` : "none yet", "The tumor's outline, drawn on the anatomical MRI. The fiber tracts are measured from its edge. An outline counts when its name says tumor (or glioma, meningioma, lesion, …), or when it was made here.");
    if (!scan || !anat) {
      const bar0 = shell.actions(caseSec);
      const open = document.createElement("button"); open.className = "sl-primary"; open.textContent = "Open a patient…";
      open.title = "Opens the DICOM database Albula uses by default. Choose the patient, tick the diffusion MRI, the anatomical MRI and the tumor outline if there is one, and load them.";
      open.onclick = () => { if (!openDicomDatabase()) say("The DICOM database cannot be opened from here; use Load / Save."); };
      const disk = document.createElement("button"); disk.textContent = "Scans on this computer…";
      disk.title = "The scans are in a folder (copied from the scanner, a CD or a USB stick): opens Load / Save at “From disk”, where a folder is chosen and added to the database.";
      disk.onclick = () => { if (!openLoadFromDisk()) say("Load / Save cannot be opened from here."); };
      bar0.append(disk, open);
      const p = document.createElement("p"); p.className = "sl-hint"; p.style.margin = "4px 0 0"; p.textContent = "Everything stays on this computer: nothing is sent anywhere.";
      caseSec.append(p);
    } else if (!tumors.length || outline) {
      // OUTLINE THE TUMOR: strokes in, strokes around, grow, correct, done.
      const steps = document.createElement("div");
      steps.style.cssText = "display:flex;flex-direction:column;gap:4px;margin:6px 0 2px";
      const step = (n: string, label: string, tip: string, on: boolean, disabled: boolean, act: () => void, words: string) => {
        const r = document.createElement("div"); r.style.cssText = "display:flex;align-items:center;gap:6px";
        const k = document.createElement("span"); k.className = "sl-hint"; k.textContent = n; k.style.flex = "0 0 1em";
        const b = document.createElement("button"); b.textContent = label; b.title = tip; b.disabled = disabled || !!busy;
        if (on) b.className = "sl-primary";
        b.onclick = act;
        const w = document.createElement("span"); w.className = "sl-hint"; w.textContent = words;
        r.append(k, b, w); steps.append(r);
      };
      step("1", "Tumor", "Draw a stroke inside the tumor on several slices, from the slice where it starts to the slice where it ends (the brush paints where you drag in a slice view; a stroke along it in the sagittal or coronal view does the same). Click again to stop drawing.", outline?.tool === 1, false, () => void strokes(anat, 1), "strokes inside the tumor");
      step("2", "Not tumor", "Draw strokes in the brain around the tumor, on several slices and also just above and below it, so the outline knows where to stop.", outline?.tool === 2, false, () => void strokes(anat, 2), "strokes around it");
      step("3", busy === "Growing…" ? "Growing…" : "Grow the outline", "Albula fills the tumor out from your strokes up to where the image changes. Check every slice it touches; add strokes where it is wrong and grow again.", false, !outline, () => void growOutline(), outline?.mm3 ? `${(outline.mm3 / 1000).toFixed(1)} mL` : "");
      step("4", "Done", "Keeps the outline and removes the strokes.", false, !outline?.resultId, () => outlineDone(), "");
      if (outline) step("", "Cancel", "Removes the strokes and the outline grown from them; nothing is kept.", false, false, () => outlineCancel(), "");
      caseSec.append(steps);
    }
    // THE OUTLINE MADE HERE, NOT SAVED YET: the yellow Save to DICOM, as AI Segmentations' Result section has it.
    if (kept && kept.savable && live.nodes.get(kept.segId) && !outline) {
      const bar1 = shell.actions(caseSec);
      const sv = document.createElement("button");
      sv.className = kept.saved ? "" : "sl-primary";
      sv.textContent = kept.saved ? "Saved ✓" : "Save to DICOM";
      sv.disabled = kept.saved;
      sv.title = "Write the tumor outline into the DICOM database, under the MRI it was drawn on, so it is there next time.";
      sv.onclick = () => { void runAction(sv, () => saveKept(), { busyLabel: "Saving…", doneLabel: "Saved ✓", failedLabel: "Not saved" }).catch(() => {}); };
      bar1.append(sv);
    }
    if (tumors.length > 1) {
      const nearSel0 = document.createElement("select");
      for (const s0 of tumors) nearSel0.append(new Option(s0.label, s0.key, false, s0.key === faceNear));
      nearSel0.title = "Which outline the fiber tracts are measured from.";
      nearSel0.onchange = () => { near = nearSel0.value; render(); };
      shell.row(caseSec, "Tumor").append(nearSel0);
    }
    const face = shell.section(root, "2 · Fiber tracts near the tumor", { band: "yellow", open: true, note: groups.some((g) => g.tract !== undefined && g.run && !g.faint && (g.within ?? 0) >= MIN_NEAR_STREAMLINES) ? `${groups.filter((g) => g.tract !== undefined && g.run && !g.faint && (g.within ?? 0) >= MIN_NEAR_STREAMLINES).length} found` : "" });
    const go = document.createElement("button");
    go.className = "sl-primary";
    go.style.cssText = "width:100%;margin:4px 0";
    go.textContent = busy && !adding && !busy.startsWith("Grow") && !busy.startsWith("Comput") ? busy : "Show the fiber tracts near the tumor";
    // Says the cut (critic, finding 4: "every" was not true): a named tract is listed when at least 5 of its fibers come
    // that close.
    go.title = `Finds the brain's main nerve fiber tracts, names them, and shows each named tract with at least ${MIN_NEAR_STREAMLINES} of its fibers within ${withinMm} mm of the tumor — whole, in its own color, with how close it comes. ${method === "ptt" ? "About three minutes (Smooth curves, chosen under Advanced)." : method === "single" ? "Uses Two-tensor: Single tensor (chosen under Advanced) does not name tracts. About half a minute." : "About half a minute."} Tracts that come within 2 mm more, or close with fewer fibers, are listed after them in gray, hidden.`;
    go.disabled = !!busy || !scan || !faceNear || seeding || !!outline;
    go.onclick = () => { if (method === "single") method = "ukf"; near = faceNear; void makeTracts(); };   // the face names tracts, from a tumor
    face.append(go);
    // THE SERVER, WHERE THE PERSON IS (critic, 2026-10-04, finding 9): when the last tracking took the brain from the
    // diffusion scan because the segmentation server was not running, the button that starts it is here.
    const cNow = scan ? computed.get(scan.browserId) : undefined;
    if (cNow?.brain.reason === "no-server") {
      const st = document.createElement("button");
      st.textContent = "Start the segmentation server";
      st.title = "Starts the program that finds the brain on the MRI of the anatomy, so the next run follows tracts through all of it (the first start takes a minute or two).";
      st.style.cssText = "width:100%;margin:0 0 4px";
      st.disabled = !!busy;
      st.onclick = () => { void runAction(st, async () => { const r = await startSegmentationServer((l) => say(l)); say(r.ok ? `${r.message[0].toUpperCase()}${r.message.slice(1)}. Press the button above again.` : `The segmentation server did not start: ${r.message}.`); if (r.ok) cNow.brain.reason = undefined; render(); }, { busyLabel: "Starting…", doneLabel: "Started", failedLabel: "Did not start" }).catch(() => {}); };
      face.append(st);
    }
    if (!scan || !faceNear) { const p = document.createElement("p"); p.className = "sl-hint"; p.textContent = "Waits until the case has all three."; face.append(p); }
    if (note) { const p = document.createElement("p"); p.className = "sl-hint"; p.style.margin = "4px 0 0"; p.textContent = note; face.append(p); }
    const listBox = document.createElement("div");
    face.append(listBox);
    if (!scan) return;
    const more = shell.section(root, "Advanced", { band: "none", open: moreOpen });
    (more.closest("details") as HTMLDetailsElement | null)?.addEventListener("toggle", (e) => { moreOpen = (e.target as HTMLDetailsElement).open; });
    const showing = shownNow(scan);
    // MAPS
    const maps = shell.section(more, "Maps", { band: "3d", open: true, note: `${scan.frameIds.length} volumes · b ${[...new Set(scan.bValues.map((b) => Math.round(b)))].sort((a, b) => a - b).join("/")}` });
    const pick = document.createElement("select");
    for (const s of list) pick.append(new Option(s.name, s.browserId, false, s.browserId === chosen));
    pick.title = "The diffusion scan these maps and tracts come from.";
    pick.onchange = () => { chosen = pick.value; const s = scans().find((x) => x.browserId === chosen); if (s) void showMap(s, "colorfa"); };
    shell.row(maps, "Scan").append(pick);
    const seg = document.createElement("div");
    seg.style.cssText = "display:flex;gap:3px";
    for (const [k, label, tip] of [["signal", "Signal", "The scan itself, one volume at a time (the transport at the top steps through them)."], ["fa", "FA", "Fractional anisotropy: bright where water moves along one direction, as in white matter tracts."], ["colorfa", "Color FA", "FA colored by direction: red left-right, green front-back, blue up-down."]] as const) {
      const b = document.createElement("button");
      b.textContent = label; b.title = tip; b.disabled = !!busy;
      b.className = "sl-sh-look-b" + (showing === k ? " sl-on" : ""); b.setAttribute("aria-pressed", String(showing === k));
      b.onclick = () => { void showMap(scan, k); };
      seg.append(b);
    }
    shell.row(maps, "Show").append(seg);
    // Distortion: corrected by default when the reversed phase-encoding scan is loaded; what was done is said.
    const partner = partnerFor(scan), c0 = computed.get(scan.browserId);
    const dist = document.createElement("label");
    dist.style.cssText = "display:flex;align-items:center;gap:6px";
    const dcb = document.createElement("input"); dcb.type = "checkbox"; dcb.checked = correct && !!partner; dcb.disabled = !partner || !!busy;
    dcb.onchange = () => {
      correct = dcb.checked;
      const was = shownNow(scan);
      const c = computed.get(scan.browserId); if (c) { dropMaps(c); computed.delete(scan.browserId); }
      if (was === "fa" || was === "colorfa") void showMap(scan, was); else render();
    };
    // The series' own description: the patient is the scan's, and is already named above.
    const short = (n: string) => n.replace(/^.*?·\s*(MR\s+)?/, "");
    dist.append(dcb, partner ? `Correct with ${short(partner.name)}` : "Not corrected: no reversed phase-encoding scan loaded");
    dist.title = c0 ? `This scan's maps and tracts: distortion ${c0.corrected}.` : "Corrects the scan's stretching along its phase-encoding direction, using the scan taken with the opposite direction.";
    shell.row(maps, "Distortion").append(dist);
    // Head movement (motion.ts): corrected unless the scan was corrected before -- a preprocessed dataset (Lauren
    // O'Donnell, 2026-10-05: BIDS datasets are often shared preprocessed). Shown once the correction is on by default.
    if (MOTION_RULE !== 0) {
      const mv = document.createElement("label");
      mv.style.cssText = "display:flex;align-items:center;gap:6px";
      const mcb = document.createElement("input"); mcb.type = "checkbox"; mcb.checked = preCorrected; mcb.disabled = !!busy;
      mcb.onchange = () => {
        preCorrected = mcb.checked;
        const was = shownNow(scan);
        const c = computed.get(scan.browserId); if (c) { dropMaps(c); computed.delete(scan.browserId); }
        if (was === "fa" || was === "colorfa") void showMap(scan, was); else render();
      };
      mv.append(mcb, "Already corrected (a preprocessed scan)");
      mv.title = c0?.motionRule ? `This scan's images were put back where the head was: ${c0.corrected.split("; ").find((x) => x.startsWith("head movement")) ?? "corrected"}.` : "Tick when the scan was corrected for head movement before it was loaded, so it is not corrected twice.";
      shell.row(maps, "Head movement").append(mv);
    }
    // dcm2niix's reading of the same files, compared (second-opinion.ts).
    const ck = checks.get(scan.browserId), cRow = document.createElement("span");
    cRow.style.cssText = "font-size:12px";
    if (!ck) cRow.textContent = "not yet (runs when the maps are first made)";
    else if ("running" in ck) cRow.textContent = "dcm2niix is reading the same files…";
    else if ("error" in ck) cRow.textContent = `not checked: ${ck.error}`;
    else {
      cRow.textContent = ck.result.agree ? `✓ ${ck.result.said}` : `⚠ ${ck.result.said}`;
      if (!ck.result.agree) cRow.style.color = "var(--sl-warn)";
    }
    cRow.title = `Albula reads the scan's diffusion values with its own reader; dcm2niix ${DCM2NIIX_VERSION}, maintained by Chris Rorden and first to learn new scanners, reads the same files, and the two are compared.`;
    shell.row(maps, "Checked").append(cRow);
    // TRACTS
    const tr = shell.section(more, "Tracts", { band: "yellow", open: true });
    const nearSel = document.createElement("select");
    if (!segs.length) nearSel.append(new Option("no segmentation of this patient", ""));
    for (const s of segs) nearSel.append(new Option(s.label, s.key, false, s.key === near));
    nearSel.title = method === "ukf" ? "The structure tracts are measured against, such as a tumor: the named tracts that pass close to it are shown, whole." : "Tracts start in the white matter inside and around this structure.";
    nearSel.onchange = () => { near = nearSel.value; };
    shell.row(tr, "Near").append(nearSel);
    const mm = document.createElement("input");
    mm.type = "number"; mm.min = "0"; mm.max = "60"; mm.step = "1"; mm.value = String(withinMm);
    mm.style.cssText = "flex:0 0 4.5em;width:4.5em;min-width:0";
    mm.title = method === "ukf" ? `How close a tract must come to the structure to be shown, in millimeters (0: only tracts reaching into it). Tracts within ${GRAY_BAND_MM} mm more are listed in gray, hidden.` : "How far around the structure tracts may start, in millimeters (0: only inside it).";
    mm.onchange = () => { withinMm = Math.max(0, Math.min(60, Number(mm.value) || 0)); mm.value = String(withinMm); };
    const mmWrap = document.createElement("span"); mmWrap.style.cssText = "display:inline-flex;align-items:center;gap:6px"; mmWrap.append(mm, "mm");
    shell.row(tr, "Within").append(mmWrap);
    const meth = document.createElement("div");
    meth.style.cssText = "display:flex;gap:3px";
    for (const [k, label, tip] of [["ukf", "Two-tensor", "Follows two crossing fiber directions, as the atlas the tract names come from was made: the method used for tumor planning. The default: under a minute."], ["ptt", "Smooth curves", "Follows fiber directions as smooth curves, from a model of all the directions in each voxel (CSD and parallel transport tracking). About three minutes; on the development cases no better near meningiomas than two-tensor."], ["single", "Single tensor", "One direction per voxel: fast, but stops or turns where fibers cross."]] as const) {
      const b = document.createElement("button"); b.textContent = label; b.title = tip; b.disabled = !!busy;
      b.className = "sl-sh-look-b" + (method === k ? " sl-on" : ""); b.setAttribute("aria-pressed", String(method === k));
      b.onclick = () => { method = k; render(); };
      meth.append(b);
    }
    shell.row(tr, "Method").append(meth);
    // Advanced, folded; stays open across redraws
    const ad = shell.section(tr, "Advanced", { band: "none", open: advOpen });
    (ad.closest("details") as HTMLDetailsElement | null)?.addEventListener("toggle", (e) => { advOpen = (e.target as HTMLDetailsElement).open; });
    const num = (label: string, tip: string, get: () => number, set: (v: number) => void, step: number, lo: number, hi: number) => {
      const i = document.createElement("input"); i.type = "number"; i.step = String(step); i.min = String(lo); i.max = String(hi); i.value = String(get()); i.title = tip;
      i.style.cssText = "flex:0 0 5em;width:5em;min-width:0";
      // Out of range is refused and said, not taken (critic, finding 15: a step of 0 was accepted).
      i.onchange = () => { const v = Number(i.value); if (Number.isFinite(v) && v >= lo && v <= hi) set(v); else { say(`${label}: between ${lo} and ${hi}.`); i.value = String(get()); } };
      shell.row(ad, label).append(i);
    };
    num("Stop below FA", "A tract ends where the white matter becomes this faint (for the tracker chosen above).", () => (method === "ukf" ? adv.ukfStopFA : adv.minFA), (v) => { if (method === "ukf") adv.ukfStopFA = v; else adv.minFA = v; }, 0.01, 0.01, 0.9);
    num("Largest turn (°)", "A tract ends where it would bend more sharply than this.", () => adv.maxAngleDeg, (v) => { adv.maxAngleDeg = v; }, 5, 5, 90);
    num("Step (voxels)", "How finely a tract follows the white matter: smaller is smoother and slower.", () => adv.stepVoxels, (v) => { adv.stepVoxels = v; }, 0.1, 0.1, 2);
    num("Highest b used", "Which images the maps (FA, Color FA) and the single-tensor tracts are made from: those up to this b-value. Two-tensor tracts use the scan's images nearest b = 3000, as the atlas behind the tract names was made. Changing it makes the maps again.", () => adv.maxB, (v) => {
      adv.maxB = v;
      const s = scans().find((x) => x.browserId === chosen), was = shownNow(s);
      for (const c of computed.values()) dropMaps(c);
      computed.clear();
      if (s && (was === "fa" || was === "colorfa")) void showMap(s, was);
    }, 100, 500, 5000);
    // The buttons: the second one, then the one yellow action at the bottom right.
    const bar = shell.actions(tr);
    const seedBtn = document.createElement("button");
    seedBtn.textContent = seeding ? "Click in a view…" : "Seed where I click…";
    seedBtn.title = seeding ? "Waiting for a click in a view. Escape cancels." : "Start tracts at the next point you click in a slice or the 3D view.";
    seedBtn.disabled = !!busy;
    seedBtn.onclick = () => { if (seeding) { cancelSeeding?.(); say("Seeding cancelled."); } else seedWhereIClick(); };
    const make = document.createElement("button");
    make.className = "sl-primary";
    make.textContent = busy.startsWith("Making") ? busy : "Make tracts";
    make.title = method === "ukf" ? "Follow tracts through the whole brain, name them, and show every named tract that comes within the distance of the structure chosen above, whole." : "Make tracts starting inside and around the structure chosen above.";
    make.disabled = !!busy || !near || seeding;
    make.onclick = () => { void makeTracts(); };
    bar.append(seedBtn, make);
    // THE TRACTS, under the face's button (what was made, whatever made it): a search, Show / Hide all, and the tracts in
    // groups -- the corpus callosum together -- each group with its eye and its fold (Ron, 2026-10-01: "we need a select
    // all/none button like in segmentations. we need to group the tract. All corpus callosum together"; "and a search").
    if (groups.length) {
      const abbrOf = (g: TractGroup) => /\(([^)]+)\)$/.exec(g.name)?.[1];
      const shownGroups = groups.filter((g) => { const a = abbrOf(g); return matchesSearch(tractSearch, g.name, a ? tractInfo(a)?.tna?.latin : undefined); });
      const tools = document.createElement("div");
      tools.style.cssText = "display:flex;gap:6px;align-items:center;margin:4px 0";
      const find = document.createElement("input");
      find.type = "search"; find.placeholder = "Find a tract…"; find.value = tractSearch; find.className = "sl-tract-find";
      find.title = "Shows only the tracts whose name, abbreviation or anatomical term contains this.";
      find.style.cssText = "flex:1;min-width:0";
      find.oninput = () => { tractSearch = find.value; const at = find.selectionStart; render(); const f2 = root?.querySelector<HTMLInputElement>(".sl-tract-find"); f2?.focus(); if (at !== null) f2?.setSelectionRange(at, at); };
      const st = showHideAllState(shownGroups.map((g) => ({ labelValue: g.id, visible: g.visible })), !!tractSearch.trim());
      const all = document.createElement("button"); all.textContent = st.label; all.title = st.show ? "Show every tract in the list below" : "Hide every tract in the list below"; all.disabled = !shownGroups.length;
      all.onclick = () => { for (const g of shownGroups) g.visible = st.show; redraw3d(); render(); };
      // ADD LINES to the tracts shown (Ron, 2026-10-01).
      const shownNamed = groups.filter((g) => g.visible && g.run && g.tract !== undefined && runs.has(g.scan));
      const chosenNow = shownNamed.filter((g) => !runs.get(g.scan)!.added.has(`${g.tract}:${g.side ?? 0}`));
      // While it runs it says which step it is on (Ron, 2026-10-01: "add progress when the add lines button is clicked").
      const addL = document.createElement("button"); addL.textContent = adding && busy ? busy.replace("Making tracts", "Adding lines") : "Add lines"; addL.className = "sl-tract-add";
      addL.title = chosenNow.length ? `Follow many more lines in the ${chosenNow.length === 1 ? "tract" : `${chosenNow.length} tracts`} shown that ${chosenNow.length === 1 ? "has" : "have"} none added yet, on both sides, to see ${chosenNow.length === 1 ? "it" : "them"} better and compare the sides. About as long as the first run. The more tracts shown, the thinner the extra lines are spread: for the most lines in a thin tract, show it alone.`
        : shownNamed.length ? "Lines are already added to every tract shown. Turn on another tract to add lines to it." : "Show the tracts you want to see better, then press this.";
      addL.disabled = !!busy || seeding || !chosenNow.length;
      addL.onclick = () => { void addLines(chosenNow); };
      tools.append(find, all, addL);
      listBox.append(tools);
      const searching = !!tractSearch.trim();
      for (const grp of TRACT_GROUPS) {
        const members = shownGroups.filter((g) => tractGroupKey(g.tract !== undefined ? tractInfo(abbrOf(g) ?? "")?.category : undefined, g.tract === undefined) === grp.key);
        if (!members.length) continue;
        const folded = foldedGroups.has(grp.key) && !searching;     // a search opens every group, as in Segmentations
        const head = document.createElement("div");
        head.style.cssText = "display:flex;align-items:center;gap:6px;margin:4px 0 1px;font-weight:600;cursor:pointer";
        const caret = document.createElement("span"); caret.textContent = folded ? "▸" : "▾"; caret.style.cssText = "flex:0 0 10px;opacity:0.7";
        const hl = document.createElement("span"); hl.textContent = `${grp.label} (${members.length})`; hl.style.cssText = "flex:1;min-width:0";
        const anyOn = members.some((g) => g.visible);
        const geye = document.createElement("button");
        geye.style.cssText = "background:none;border:none;padding:0 2px;cursor:pointer;color:inherit;font-size:13px;flex:0 0 auto";
        geye.textContent = anyOn ? "👁" : "🚫"; geye.title = anyOn ? `Hide every tract in ${grp.label}` : `Show every tract in ${grp.label}`;
        geye.onclick = (e) => { e.stopPropagation(); for (const g of members) g.visible = !anyOn; redraw3d(); render(); };
        head.onclick = () => { if (foldedGroups.has(grp.key)) foldedGroups.delete(grp.key); else foldedGroups.add(grp.key); render(); };
        head.title = folded ? "Show the tracts in this group" : "Fold this group";
        head.append(caret, hl, geye);
        listBox.append(head);
        if (folded) continue;
        for (const g of members) {
        const row = document.createElement("div");
        row.style.cssText = "display:flex;align-items:center;gap:8px;min-width:0;padding:1px 0";
        const name = document.createElement("span"); name.textContent = g.name; name.title = g.outside ? `${g.name}\nFibers that leave the brain, run through the fluid around it and come back. They are not one of the brain's tracts, so they get no tract's name; at the skull base they are often a cranial nerve, such as the trigeminal nerve.` : g.name;
        // READ MORE (Ron, 2026-10-01): the paper that describes the tract, opened in the browser from the name.
        const abbr = /\(([^)]+)\)$/.exec(g.name)?.[1];
        if (g.tract !== undefined && abbr) {
          const ref = readMore(abbr);
          const tna = tnaLine(abbr);
          const note = tractNote(abbr);
          name.title = `${g.name}${tna ? `\n${tna}` : ""}${note ? `\n${note}` : ""}\nRead more (click): ${ref.cite}`;
          name.onclick = () => { void fetch(`/_open?url=${encodeURIComponent(ref.link)}`).then((r) => { if (!r.ok) throw new Error(); }).catch(() => { globalThis.open?.(ref.link, "_blank"); }); };
        }
        name.style.cssText = `flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap${name.onclick ? ";cursor:pointer;text-decoration:underline dotted" : ""}`;
        const n = document.createElement("span"); n.style.opacity = "0.7";
        n.textContent = g.within !== undefined ? `${g.within.toLocaleString()}/${g.strands.length.toLocaleString()}` : g.strands.length.toLocaleString();
        n.title = g.within !== undefined ? `${g.within.toLocaleString()} of this tract's ${g.strands.length.toLocaleString()} streamlines come within the distance of the structure` : "streamlines in this group";
        if (g.more) n.title += ` (${g.more.toLocaleString()} of them added by Add lines)`;
        // THE OTHER SIDE, for comparing (Ron, 2026-10-01; O'Donnell et al. 2017: surgeons compare the tumor's side with
        // the healthy side).
        const other = document.createElement("span");
        if (g.otherSide !== undefined && g.side) {
          const os = g.side > 0 ? "left" : "right";
          other.textContent = `${os} ${g.otherSide.toLocaleString()}`;
          other.title = `The same tract on the ${os} side has ${g.otherSide.toLocaleString()} streamlines, this side ${g.strands.length.toLocaleString()}.`;
          other.style.cssText = "flex:0 0 auto;opacity:0.55;font-size:11px";
        }
        if (g.faint) {
          name.style.opacity = "0.55"; n.style.opacity = "0.45";
          name.title += (g.within ?? 0) > 0
            ? `\nOnly ${g.within} of its streamlines come within the distance — fewer than ${MIN_NEAR_STREAMLINES}, so it is listed but hidden. It may be thin here, or hidden by swelling. Show it and press Add lines to see it better.`
            : `\nIt comes ${Number.isFinite(g.distanceMm ?? Infinity) ? `within ${(g.distanceMm as number).toFixed(1)} mm` : "no closer than the distance measured"} — farther than the distance chosen, so it is listed but hidden. Show it to see how it passes the structure.`;
        }
        if (g.tract !== undefined || g.unnamed) {
          // The tract's own color, as it is drawn (and as its card will show it).
          const c = g.tract !== undefined ? tractColor(g.tract, tractCount) : UNNAMED;
          const sw = document.createElement("span");
          sw.style.cssText = `flex:0 0 10px;width:10px;height:10px;border:1px solid var(--sl-border, #000);background:rgb(${c.slice(0, 3).map((v) => Math.round(v * 255)).join(",")})`;
          row.append(sw);
        }
        const dist = document.createElement("span");
        if (g.distanceMm !== undefined) {
          // Measured out to the margin and 2 mm beyond (streamlineDistances); farther is "far" (Ron's screenshot,
          // 2026-10-01: the other side, added by Add lines, read "Infinity mm").
          dist.textContent = g.distanceMm < 0.05 ? "touches" : Number.isFinite(g.distanceMm) ? `${g.distanceMm.toFixed(1)} mm` : "far";
          dist.title = g.distanceMm < 0.05 ? "This tract reaches into the structure." : "This tract's closest distance to the structure.";
          dist.style.cssText = "flex:0 0 auto;opacity:0.85";
        }
        const icon = "background:none;border:none;padding:0 2px;cursor:pointer;color:inherit;font-size:13px;min-width:0;flex:0 0 auto";
        const eye = document.createElement("button"); eye.style.cssText = icon; eye.textContent = g.visible ? "👁" : "🚫"; eye.title = g.visible ? "Hide these tracts" : "Show these tracts";
        eye.onclick = () => { g.visible = !g.visible; redraw3d(); render(); };
        const x = document.createElement("button"); x.style.cssText = icon; x.textContent = "✕"; x.title = "Remove these tracts";
        x.onclick = () => { groups.splice(groups.indexOf(g), 1); redraw3d(); render(); };
        row.style.paddingLeft = "14px";
        row.append(name, dist, n, other, eye, x);
        listBox.append(row);
        }
      }
      // THE CAVEAT (Ron, 2026-10-01; O'Donnell et al. 2017: a missing tract cannot be told destroyed from hidden by edema).
      if (groups.some((g) => g.run && g.tract !== undefined)) {
        const cav = document.createElement("p"); cav.className = "sl-hint";
        cav.textContent = "A tract that looks thin or missing on the tumor's side may be destroyed by the tumor, or still there but hidden by swelling (edema). Compare it with the other side, and press Add lines for a closer look.";
        listBox.append(cav);
        // Which brain these tracts were followed in, kept with the run (critic, 2026-10-04, finding 9).
        const runNow = runs.get(scan.browserId);
        if (runNow?.brain) { const bl = document.createElement("p"); bl.className = "sl-hint"; bl.textContent = `${runNow.brain[0].toUpperCase()}${runNow.brain.slice(1)}.`; listBox.append(bl); }
      }
      const as = document.createElement("div");
      as.style.cssText = "display:flex;gap:3px";
      for (const [k, label, tip] of [["lines", "Lines", "Thin lines: many tracts stay readable."], ["tubes", "Tubes", "Round, lit tubes: depth is easier to see."]] as const) {
        const b = document.createElement("button"); b.textContent = label; b.title = tip;
        b.className = "sl-sh-look-b" + (drawAs === k ? " sl-on" : ""); b.setAttribute("aria-pressed", String(drawAs === k));
        b.onclick = () => { drawAs = k; redraw3d(); render(); };
        as.append(b);
      }
      shell.row(more, "Draw as").append(as);
      // Shorter ends (drawing only).
      const tw = document.createElement("span"); tw.style.cssText = "display:flex;gap:6px;align-items:center";
      const ti = document.createElement("input"); ti.type = "number"; ti.min = "0"; ti.max = "6"; ti.step = "0.5"; ti.value = String(trimVoxels); ti.style.width = "4em";
      const tl = document.createElement("span"); tl.style.opacity = "0.7";
      const vmm = groups.length ? voxelMm(groups[0].scan) : 2;
      tl.textContent = `voxels at each end (≈ ${(trimVoxels * vmm).toFixed(1)} mm on this ${vmm.toFixed(1)} mm grid)`;
      ti.title = "Draws every tract a little shorter at both ends, where the fibers fan out. The tracts themselves stay whole. 0 draws them whole.";
      ti.onchange = () => { const v = Number(ti.value); trimVoxels = Number.isFinite(v) ? Math.max(0, Math.min(6, v)) : trimVoxels; redraw3d(); render(); };
      tw.append(ti, tl);
      shell.row(more, "Shorten ends").append(tw);
      if (groups.some((g) => g.tract !== undefined)) {
        const cb = document.createElement("div");
        cb.style.cssText = "display:flex;gap:3px";
        for (const [k, label, tip] of [["tract", "Tract", "Each named tract in its own color; streamlines without a name in gray."], ["direction", "Direction", "Every piece by its direction: red left-right, green front-back, blue up-down."]] as const) {
          const b = document.createElement("button"); b.textContent = label; b.title = tip;
          b.className = "sl-sh-look-b" + (colorBy === k ? " sl-on" : ""); b.setAttribute("aria-pressed", String(colorBy === k));
          b.onclick = () => { colorBy = k; redraw3d(); render(); };
          cb.append(b);
        }
        shell.row(more, "Color").append(cb);
      }
    }
  }

  // THE TUBES IN FRONT, for the 3D probe (critic, 2026-10-01, finding 4): where along the mouse ray the first shown tube
  // is -- the first step that comes within a tube's radius (plus half a millimeter) of a drawn streamline point.
  registerRayHits((o, d) => {
    if (!probeIndex) return null;
    const r = (drawAs === "tubes" ? RADIUS.tubes : RADIUS.lines) + 0.5;
    // Only the stretch of the ray inside the box around the shown streamlines (the slab method).
    const { xyz } = probeIndex.ix;
    let t0 = 0, t1 = Infinity;
    for (let a = 0; a < 3; a++) {
      let lo = Infinity, hi = -Infinity;
      for (let i = a; i < xyz.length; i += 3) { if (xyz[i] < lo) lo = xyz[i]; if (xyz[i] > hi) hi = xyz[i]; }
      lo -= r; hi += r;
      if (Math.abs(d[a]) < 1e-12) { if (o[a] < lo || o[a] > hi) return null; continue; }
      const ta = (lo - o[a]) / d[a], tb = (hi - o[a]) / d[a];
      t0 = Math.max(t0, Math.min(ta, tb)); t1 = Math.min(t1, Math.max(ta, tb));
    }
    if (!(t1 >= t0)) return null;
    for (let t = t0; t <= t1; t += 0.5) {
      const p: [number, number, number] = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
      if (tractsNear(probeIndex.ix, p, r).length) return t;
    }
    return null;
  });

  // THE TRACTS IN THE DATA PROBE (Ron, 2026-10-01: "the tracts are not in the data probe"): every tract shown that passes
  // within 2 mm of the point under the pointer, with its color, how many of its fibers, and how close.
  registerProbeRows((ras) => {
    if (!probeIndex) return [];
    return tractsNear(probeIndex.ix, ras, 2).slice(0, 6).map((h) => {
      const g = probeIndex!.shown[h.set];
      const c = g.tract !== undefined ? tractColor(g.tract, tractCount) : g.unnamed ? UNNAMED : undefined;
      return { color: c ? [c[0], c[1], c[2]] as [number, number, number] : undefined,
        text: `${g.name} · ${h.streamlines} fiber${h.streamlines === 1 ? "" : "s"} within 2 mm${h.closestMm < 0.5 ? "" : `, closest ${h.closestMm.toFixed(1)} mm`}`, source: "fiber tracts (Diffusion)" };
    });
  });

  shell.registerPanel({
    id: "diffusion",
    title: "Diffusion",
    groups: ["Display"],
    tip: "Which of the brain's fiber tracts run near a tumor, for planning an operation.",
    help: "<p><b>For preparing a case.</b> The module needs three things, and ticks each off when it is there: the <b>diffusion MRI</b> (the scan that shows the brain's wiring; often named DTI, DWI or diffusion), the <b>MRI of the anatomy</b> (usually the T1 with contrast), and the <b>tumor's outline</b>. <b>Open a patient…</b> opens the DICOM database; <b>Scans on this computer…</b> opens Load / Save, where a folder from a CD, a USB stick or an export is added to the database. Without an outline, draw a few <b>Tumor</b> strokes inside it and a few <b>Not tumor</b> strokes around it, on a few slices, then <b>Grow the outline</b>, check it, correct with more strokes, and press <b>Done</b>. Then <b>Show the fiber tracts near the tumor</b>: each named tract with at least 5 of its fibers within 6 mm of the tumor (the distance can be changed under Advanced) is shown whole, in its own color, with how close it comes; the tracts that come within 2 mm more, or that come close with fewer fibers, are listed after them in gray, hidden; pressing it again replaces the list. Each tract shows how many fibers the same tract has on the other side. Turn on the tracts you care about and press <b>Add lines</b>: it follows many more fibers in the tracts shown that have none added yet, on both sides (the other side stays hidden unless it is on), to see a thin tract better and compare the sides (about as long as the first run). A tract that looks thin or missing on the tumor's side may be destroyed by the tumor, or still there but hidden by swelling (edema). Only an outline named as a tumor (or made here) counts. Everything stays on this computer.</p><p><b>Advanced</b> holds the maps and the settings behind that button:</p><p>Shows what a diffusion MRI scan measures: <b>FA</b>, how strongly water moves along one direction (bright in white matter tracts), and <b>Color FA</b>, that direction as a color (red left-right, green front-back, blue up-down). A diffusion scan shows Color FA when it loads. When the case has an MRI of the anatomy, FA and Color FA are shown over it at half opacity; otherwise they fill the slice views. The gear in each slice view chooses the image, the one over it and how much shows through.</p><p><b>Tracts</b> follow the main direction of water movement from voxel to voxel. Under Advanced, <b>Make tracts</b> with <b>Single tensor</b> starts them in the white matter inside and around the chosen structure; with <b>Two-tensor</b> or <b>Smooth curves</b> it does what the face's button does, from the structure chosen there; <b>Seed where I click…</b> starts them at one point. <b>Smooth curves</b> follows fibers as gently bending lines from a model of every direction in each voxel (CSD and parallel transport tracking): about three minutes, an option. Tracts are drawn in 3D as tubes or lines, and as a dot in the tract's color wherever a shown tract crosses a slice, with the MRI of the anatomy behind it. Each tract is drawn 2 voxels shorter at each end, where fibers fan out (<b>Shorten ends</b>; 0 draws them whole); the tracts themselves stay whole. Each group can be hidden or removed.</p><p><b>Two-tensor</b> (UKF, the default) follows two fiber directions, from a starting point in every voxel of the brain, with the settings of the atlas the tract names come from (as Mike Halle's tractline does), so tracts reach into the swelling around a tumor; it runs on the graphics card and agrees with the original UKFTractography program (on a whole brain, 93% of fibers end within 0.1 mm of the original's from the same starting points). <b>Single tensor</b> follows one direction per voxel; where tracts cross, that direction is an average, and a tract may stop or turn. With <b>Two-tensor</b>, <b>Make tracts</b> follows tracts through the whole brain and names them with RapidParc (Bisten, Schultz et al., University of Bonn), a network trained on an atlas of 800 fiber clusters (Zhang, O'Donnell et al.); each named tract that comes within the distance of the chosen structure is shown whole, in its own color, with its closest distance to the structure. Streamlines no name fits and pass close are shown in gray; the rest of the brain is kept, hidden.</p><p><b>Licenses.</b> Research software: not reviewed or approved by the FDA or any other agency; clinical applications are neither recommended nor advised. The two-tensor tracking is a port of UKFTractography (authors: Yogesh Rathi, Stefan Lienhard, Yinpeng Li, Martin Styner, Ipek Oguz, Yundi Shi, Christian Baumgartner, Ryan Eckbo, Tashrif Billah and Dheshan Mohandass; github.com/pnlbwh/ukftractography). All or portions of this licensed product (such portions are the \"Software\") have been obtained under license from The Brigham and Women's Hospital, Inc. and are subject to the following terms and conditions: <a href=\"./vendor/diffusion/licenses/LICENSE-UKF.txt\" target=\"_blank\">the UKF Tractography Contribution and Software License Agreement</a> (this is a modified version: translated to TypeScript and WGSL). RapidParc's trained network is under <a href=\"./vendor/diffusion/rapidparc/LICENSE.txt\" target=\"_blank\">its BSD license</a> (University of Bonn); TractCloud's table of tract names under <a href=\"./vendor/diffusion/tractcloud/LICENSE.txt\" target=\"_blank\">3D Slicer's license</a>; dcm2niix under <a href=\"./vendor/diffusion/dcm2niix/LICENSE.txt\" target=\"_blank\">its own (BSD)</a>; the rest of this extension under the <a href=\"./vendor/diffusion/licenses/LICENSE\" target=\"_blank\">Apache License 2.0</a> (<a href=\"./vendor/diffusion/licenses/NOTICE\" target=\"_blank\">NOTICE</a>).</p>",
    acknowledgements: DIFFUSION_REFERENCES.map((r) => `${r.cite}${r.link ? ` ${r.link}` : ""} — ${r.usedFor}${r.verified ? "" : " (citation to be checked)"}`),
    mount(el) { root = el; render(); },
    onShow() { render(); },
  });

  // REDRAW ONLY WHEN WHAT THE PANEL SHOWS CHANGED (critic, finding 14: every sequence step closed Advanced and dropped a
  // value being typed). And COLOR FA WHEN A DIFFUSION SCAN LOADS (Ron's yes on the mockup).
  let lastSig = "";
  const sig = () => {
    const list = scans(), s = list.find((x) => x.browserId === chosen) ?? list[0];
    return JSON.stringify([list.map((x) => x.browserId), segmentChoices(s).map((c) => c.key + c.label), shownNow(s)]);
  };
  live.subscribe((c) => {
    if (!(c.type === "sequenceBrowser" || c.type === "sequence" || c.type === "segmentation" || c.type === "sliceComposite" || c.type === "image" || c.kind === "reset" || c.kind === "remove")) return;
    const pruned = prune();
    // THE REVERSED SCAN ARRIVED (or left) AFTER THE MAPS WERE MADE: a series loads in turn, and the diffusion scan often
    // comes before its partner. The maps on screen would say "corrected" and not be; they are made again.
    for (const s of scans()) {
      const c = computed.get(s.browserId);
      if (!c || busy) continue;
      const want = correct ? partnerFor(s)?.id ?? "" : "";
      if (want === c.partnerId) continue;
      const was = shownNow(s);
      dropMaps(c); computed.delete(s.browserId);
      if (was === "fa" || was === "colorfa") void showMap(s, was);
    }
    for (const s of scans()) {
      if (seenScans.has(s.browserId)) continue;
      seenScans.add(s.browserId);
      chosen = s.browserId;
      void showMap(s, "colorfa");
    }
    const now = sig();
    if (now !== lastSig || pruned) { lastSig = now; render(); }
  });
}

queueModule(registerDiffusionPanel);

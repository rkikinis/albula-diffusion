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
//  - MAPS (blue band: a display): which diffusion scan, and what the slice views show of it -- its b0, corrected and aligned (Ron, 2026-10-15),
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
import type { GradientCheck } from "./gradient-check.ts";
import { MOTION_RULE, type MotionRuleId } from "./motion.ts";
/** "Stop below FA"'s default under Advanced (the single-tensor tracker's; the UKF's comes from the tracking rule). */
const ADV_MIN_FA = 0.15;
import { DCM2NIIX_VERSION, secondOpinion, type SecondOpinion } from "./second-opinion.ts";
import { databaseSeries, openDatabaseId } from "albula";
import { adPath, b0Path, colorFaPath, dicomToTracts, faPath, mdPath, rdPath, UNNAMED as UNNAMED_SET } from "./tracts-dicom.ts";
import { howMade } from "./methods.ts";
import { byName, laterality, measure, MEASURE_RULE, openingWay, throughBox, toCsv, voxelsOf, type ExportRow, type Maps, type Measures, type Way } from "./selection.ts";
import { databaseFileUrl, nrrdDecode, nrrdGeometry, nrrdSplitHeader } from "albula";
import { addRowMenu, fitList, makeNameEditable, pickRectangle, type PickedRectangle, saveToDownloads, showInDownloads, sliceOffset, sliceOrientation, startEditing, viewBanner, type RowAction } from "albula";
import { assetUrl, holdDrawing, seriesDicomFiles, startPlacing, restartSegmentationServer, startSegmentationServer, synthstripBrainMask, type BrainMaskResult } from "albula";
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
interface Computed { dwi: DiffusionSeries; fit: TensorFit; maxB: number; corrected: string; partnerId: string; /** motion.ts: the head-movement rule asked for (the cache's key), the rule actually applied, and what it did in words. */ motionAsked: MotionRuleId; motionRule: MotionRuleId; movement?: string; /** gradient-check.ts: the directions' check, in words, when they were not used as recorded. */ directions?: string; /** The anatomy MRI the scan was aligned to ("" none). */ anatomyId?: string; /** The tracking rule the fit's brain mask serves (tracking-rules.ts): 2 until rule 3's brain is put in (brainFor). */ rule: TrackingRuleId; /** Rule 3's brain: what was used, in words, and why not when it was not. */ brain: BrainState; faId?: string; colorFaId?: string; /** The corrected b0 (the fit's S0), when shown. */ b0Id?: string; ukf?: UkfData; fod?: FodVolume }
/** RULE 3's BRAIN for a fitted scan (critic, 2026-10-04, findings 2, 4, 8, 9): asked of the segmentation server when the
 *  maps are made but not waited for -- the maps do not use it -- and waited for by the tracking (brainFor). A failure
 *  that can go away (no server, a server without SynthStrip, a failed job) is asked again at the next tracking. */
interface BrainState { note: string; reason?: "no-anatomy" | "moved" | "doubt" | "no-server" | "no-synthstrip" | "stuck" | "failed"; ask?: Promise<BrainMaskResult> }
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
  /** v3: how it was chosen -- by name (ticked) or through the box -- when not by a near-a-structure run. */
  by?: "name" | "box";
  /** A name the person gave the row (one click on it; the module template's rename). */
  label?: string;
  /** The other side of a listed tract, there to compare (hidden; it does not make the row near or faint). */
  compare?: boolean;
  /** The color the person chose for the row (Ron, 2026-10-16: "How do I give it a solid color?"): a solid color, or
   *  "direction"; absent, the default -- the tract's color, gray when unnamed, by direction when seeded by hand. */
  color?: RGBA | "direction";
}
/** The last whole-brain run for a scan, kept so "Add lines" can name new streamlines in its context and measure them
 *  against the same structure. */
interface Run { sl: Float32Array[]; named: Named; sorted: Sorted; structure: Structure; label: string; withinMm: number; method: Method;
  /** The tracking rule the whole-brain run was made under, and its brain in words (BrainState.note). */ rule?: TrackingRuleId; brain?: string;
  /** gradient-check.ts, in words, when the directions were not confirmed as recorded: kept with the tracts it shaped (critic, finding 6). */ directions?: string;
  /** The tracts ("tract:side") Add lines has already added to: a second press would start from the same points and
   *  draw the same lines again, so they are skipped (Ron, 2026-10-01: he pressed it, then saw the corticospinal tract
   *  was not on). */
  added: Set<string>;
  /** Each run streamline's distance to the structure (mm, measured out to the margin and 2 mm beyond). */
  dist: Float64Array;
  /** The fit it was tracked on (critic, 2026-10-01, finding 5): Add lines on a different fit is refused. */
  maxB: number; partnerId: string;
  /** The anatomy MRI the scan was aligned to when the run was made ("" none): Add lines tracks in the same space only. */
  anatomyId: string;
  /** The run's streamlines are the ones made at import (DMRI-AT-IMPORT.md): corrected and aligned to the T1 by the job,
   *  so Add lines checks only that this scan is aligned to an anatomy MRI as well, not that its fit is the one recorded. */
  stored?: boolean;
  /** The stored set's inputs (its reversed scan, its T1), for Add lines to track the way it was made. */
  storedInputs?: { partner?: string | null; t1?: string | null } }
/** How tracts are followed: one tensor per voxel (fast, the classic), or the two-tensor UKF (ukf.ts; crossing fibers;
 *  the method SlicerDMRI uses for tumor planning), with the settings of the current tracking rule (tracking-rules.ts). */
/** "ptt": fiber distributions from multi-shell CSD (csd.ts, responses from the scan) and parallel transport tracking
 *  (ptt.ts), on the processor's workers -- the smoother method Lauren O'Donnell recommended; about three times as long
 *  as UKF and no better on the development cases' meningiomas, so an option (Ron, 2026-10-01: "if both are close to
 *  equal in quality, we go with the faster and offer the other as option"). */
type Method = "ukf" | "single" | "ptt";
/** What the slice views show of a diffusion scan (Ron, 2026-10-15: "b0 (corrected) is the right option for our use case
 *  scenario" -- the scan as acquired, on its own grid and uncorrected, read as "the registration hasn't happened"). */
type Show = "b0" | "fa" | "colorfa";

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
  /** gradient-check.ts per scan: it depends on the scan alone, so a re-made map does not pay for it again (critic, finding 8). */
  const directionsChecked = new Map<string, GradientCheck>();
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
  const preCorrected = new Set<string>();                    // scans (browser ids) corrected for head movement before loading (preprocessed)
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
  let noteFor = "";
  const say = (t: string) => { note = t; noteFor = chosen; status(t); if (busy) { bannerStep = t; syncBanner(); } };
  // WHAT IS GOING ON, ACROSS THE TOP OF THE 3D VIEW (Ron, 2026-10-16, seeding: "How do I know that something is
  // happening? A progress bar immediately when I click … Perhaps the progress bar at the top of the 3d viewer?"; SDK 14
  // viewBanner): while the module waits for a click to seed, and while any work runs, with the step it is on and its
  // progress when it has one.
  let banner: ReturnType<typeof viewBanner> = null, bannerStep = "";
  function syncBanner() {
    const pct = /(\d+)%/.exec(busy)?.[1];
    const want = seeding
      ? { title: "Seed: click where the tracts should start, in a slice or the 3D view", hint: "Esc cancels", progress: undefined, actions: [{ label: "Cancel", title: "Stop waiting for a click (Esc)", onClick: () => { cancelSeeding?.(); say("Seeding cancelled."); } }] }
      : busy ? { title: busy.replace(/\s*\d+%$/, ""), hint: !pct && bannerStep && bannerStep !== busy ? bannerStep.replace(/…$/, "") : undefined, progress: pct ? Number(pct) / 100 : null, actions: [] }
      : null;
    if (!want) { banner?.close(); banner = null; bannerStep = ""; return; }
    if (banner) banner.update(want); else banner = viewBanner(want);
  }

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
      if (n.type !== "image" || n.labelmap || inSequences.has(n.id) || n.id === c?.faId || n.id === c?.colorFaId || n.id === c?.b0Id) return false;
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
    const c = computed.get(scan.browserId), m = storedMaps.get(scan.browserId);
    if ((m?.b0 && bg === m.b0) || (c?.b0Id && bg === c.b0Id)) return "b0";
    if (m?.colorfa && (bg === m.colorfa || fg === m.colorfa)) return "colorfa";
    if (m?.fa && (bg === m.fa || fg === m.fa)) return "fa";
    if (c?.colorFaId && (bg === c.colorFaId || fg === c.colorFaId)) return "colorfa";
    if (c?.faId && (bg === c.faId || fg === c.faId)) return "fa";
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
    const motionRule: MotionRuleId = preCorrected.has(scan.browserId) ? 0 : MOTION_RULE;
    if (have && have.maxB === adv.maxB && have.partnerId === (partner?.id ?? "") && (have.anatomyId ?? "") === (anatOk?.id ?? "") && have.motionAsked === motionRule) return have;
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
      motionRule, ...(scan.phaseEncoding ? { phaseEncoding: scan.phaseEncoding } : {}), times, say,
      ...(directionsChecked.has(scan.browserId) ? { directions: directionsChecked.get(scan.browserId) } : {}) });
    if (prep.directions) directionsChecked.set(scan.browserId, prep.directions);
    dwi = prep.dwi;
    // The distortion line keeps the alignment; the movement has its own row (critic, 2026-10-05, finding 7).
    if (prep.alignmentSaid) corrected += `; ${prep.alignmentSaid}`;
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
    const c: Computed = { dwi, fit, maxB: adv.maxB, corrected, motionAsked: motionRule, motionRule: prep.motionRule, ...(prep.movementSaid ? { movement: prep.movementSaid } : {}), ...(prep.directionsSaid ? { directions: prep.directionsSaid } : {}), partnerId: partner?.id ?? "", anatomyId: anatOk?.id ?? "", rule: wantT1 ? 2 : TRACKING_RULE, brain };
    computed.set(scan.browserId, c);
    say(`Maps made from ${fit.used.length} volumes up to b = ${adv.maxB}${c.directions ? `; ${c.directions}` : ""}; distortion ${corrected}${c.movement ? `; ${c.movement}` : ""}. ${stageText(times)}.`);
    return c;
  }
  const dropMaps = (c: Computed) => {
    for (const id of [c.faId, c.colorFaId, c.b0Id]) if (id && live.nodes.get(id)) removeVolumeFromScene(live, id);
    c.faId = c.colorFaId = c.b0Id = undefined;
  };

  async function showMap(scan: Scan, what: Show) {
    if (busy) return;                                         // one computation at a time (critic, finding 10)
    busy = "Computing…";
    render();
    try {
      // The stored map when the import made one (no fit); else the module's own.
      const kept = await storedMap(scan, what).catch(() => undefined);
      // The corrected b0 fills the slice views alone, as the diffusion scan itself did; FA and Color FA go over the T1.
      if (kept) { if (what === "b0") putBackground(kept, null); else putMap(scan, kept); return; }
      const c = await ensureFit(scan);
      if (what === "b0") {
        // This module's own fit: its S0, the b = 0 signal after the corrections, on the fit's grid.
        if (!c.b0Id || !live.nodes.get(c.b0Id)) {
          const name = `${scan.name} b0 (corrected)`;
          const r = await loadVolumeIntoScene(live, store, { dims: c.fit.dims, ijkToRAS: c.fit.ijkToRAS, data: c.fit.S0, dtype: "<f4", name }, { name, extra: { autoVolumeRendering: false, recomputable: true } });
          c.b0Id = r.imageId;
        }
        putBackground(c.b0Id, null);
      } else if (what === "fa") {
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
  /** The map this module put over each scan's slices (scan -> image id), for putting it back (mapsBack). */
  const mapOn = new Map<string, string>();
  /** When each scan arrived (performance.now()), for the log line saying how long its Color FA took. */
  const arrivedAt = new Map<string, number>();
  const putMap = (scan: Scan, mapId: string) => {
    mapOn.set(scan.browserId, mapId);
    const t = arrivedAt.get(scan.browserId);
    if (t !== undefined) { arrivedAt.delete(scan.browserId); status(`Color FA on the slices ${((performance.now() - t) / 1000).toFixed(1)} s after the diffusion MRI arrived.`); }
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

  // ── THE TRACTS MADE AT IMPORT (DMRI-AT-IMPORT.md in the workspace; Ron, 2026-10-05: "automatic import"; "q3: A" --
  // the button waits for the background run and says so). Albula's server runs the extension's job after scans are
  // added to a database (desktop/extension-jobs.ts). Its Tractography Results object, stored under the diffusion series,
  // holds the whole brain's streamlines already named; the button then only measures them against the structure.
  // WHICH ONE: the newest the database holds for the scan. Every object records the version that made it; none is
  // "stale" (Ron, 2026-10-14: "for reproducibility scientists would like to run a specific version of a pipeline on
  // data"): choosing among versions is the versioning design still to come.
  const storedState = new Map<string, { state: "stored" | "none" | "unknown"; made?: string; seriesUID?: string; tracking?: number; rules?: Record<string, unknown>; inputs?: { partner?: string | null; t1?: string | null } }>();
  let jobBusyNow = false;
  // While a loaded scan has no stored tracts: look again every 10 s (the server may be making them right now).
  setInterval(() => {
    const waiting = scans().filter((x) => storedState.get(x.browserId)?.state !== "stored");
    if (!waiting.length || !root) return;
    void (async () => {
      const was = jobBusyNow, before = waiting.map((x) => storedState.get(x.browserId)?.state).join();
      jobBusyNow = await jobRunning();
      // Asked of the database only while the server works on it, or once when it just finished (finding 19: every 10 s,
      // forever, on a Mac without jobs).
      if (jobBusyNow || was) for (const x of waiting) await findStored(x).catch(() => {});
      if (was !== jobBusyNow || before !== waiting.map((x) => storedState.get(x.browserId)?.state).join()) render();
    })();
  }, 10_000);
  const dwiUidOf = (scan: Scan) => (live.nodes.get(scan.frameIds[0])?.origin as Record<string, unknown> | undefined)?.seriesInstanceUID as string | undefined;
  async function findStored(scan: Scan): Promise<{ seriesUID: string } | null> {
    const uid = dwiUidOf(scan);
    if (!uid) { storedState.set(scan.browserId, { state: "none" }); return null; }
    const db = await databaseSeries({ fresh: true }).catch(() => null);
    if (!db) { storedState.set(scan.browserId, { state: "unknown" }); return null; }
    const num = new Map(db.series.map((x) => [x.seriesInstanceUID, x.seriesNumber ?? 0]));
    const kids = db.edges.filter((e) => e.kind === "tracts" && e.parent === uid && num.has(e.child)).sort((a, b) => (num.get(b.child) ?? 0) - (num.get(a.child) ?? 0));
    if (!kids.length) { storedState.set(scan.browserId, { state: "none" }); return null; }
    storedState.set(scan.browserId, { state: "stored", seriesUID: kids[0].child });
    return { seriesUID: kids[0].child };
  }
  /** The stored whole-brain tracts as the module's own run would have made them: the streamlines and their names. */
  async function loadStored(scan: Scan, model: TractCloudModel): Promise<{ sl: Float32Array[]; named: Named; made?: string } | null> {
    const at = await findStored(scan);
    if (!at) return null;
    const files = await seriesDicomFiles(at.seriesUID);
    if (!files?.length) return null;
    const t = await dicomToTracts(new Uint8Array(files[0]));
    const names = model.json.tracts, OTHER = names.length - 1, index = new Map(names.map((x, i) => [x.name, i]));
    const sl: Float32Array[] = [], tract: number[] = [], side: number[] = [];
    for (const set of t.sets) {
      const ti = set.label === UNNAMED_SET ? OTHER : index.get(set.label) ?? OTHER;
      for (const p of set.streamlines) { sl.push(p); tract.push(ti); side.push(ti === OTHER ? 0 : set.side); }
    }
    const made = typeof t.provenance?.made === "string" ? t.provenance.made : undefined;
    const rules = t.provenance?.rules as { tracking?: number } | undefined;
    storedState.set(scan.browserId, { state: "stored", seriesUID: at.seriesUID, made, tracking: rules?.tracking, rules: t.provenance?.rules as Record<string, unknown> | undefined, inputs: t.provenance?.inputs as { partner?: string | null; t1?: string | null } | undefined });
    return { sl, named: { tract: Int32Array.from(tract), side: Int8Array.from(side), draws: 0, seconds: 0 }, made };
  }
  // THE MAPS MADE AT IMPORT (Ron, 2026-10-14: "Why not do all proper registrations as part of the import? That would allow
  // to ignore how much time they need"): FA and Color FA, stored by the job beside its tracts, after every correction and
  // lined up with the T1 at the scan's own spacing -- shown as they are, so a case opens without fitting the scan (PAT16: 45 s of fit).
  const storedMaps = new Map<string, { fa?: string; colorfa?: string; b0?: string }>();   // scan -> image node ids
  async function storedMap(scan: Scan, what: Show): Promise<string | undefined> {
    const have = storedMaps.get(scan.browserId)?.[what];
    if (have && live.nodes.get(have)) return have;
    const st = storedState.get(scan.browserId)?.state === "stored" ? storedState.get(scan.browserId) : await findStored(scan).then(() => storedState.get(scan.browserId)).catch(() => undefined);
    if (st?.state !== "stored" || !st.seriesUID) return undefined;
    const url = await databaseFileUrl(what === "fa" ? faPath(st.seriesUID) : what === "b0" ? b0Path(st.seriesUID) : colorFaPath(st.seriesUID));
    const r = url ? await fetch(url).catch(() => null) : null;
    if (!r?.ok) return undefined;
    const { f, body } = nrrdSplitHeader(new Uint8Array(await r.arrayBuffer()));
    const sizes = (f["sizes"] ?? "").split(/\s+/).filter(Boolean).map(Number) as [number, number, number];
    const raw = await nrrdDecode(f, body, sizes[0] * sizes[1] * sizes[2] * 4);
    const data = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    const name = `${scan.name} ${what === "fa" ? "FA" : what === "b0" ? "b0 (corrected)" : "Color FA"}`;
    const res = await loadVolumeIntoScene(live, store, { dims: sizes, ijkToRAS: nrrdGeometry(f).ijkToRAS, data, dtype: "<f4", name },
      { name, extra: { ...(what === "colorfa" ? { rgb24: true } : {}), autoVolumeRendering: false, recomputable: true } });
    if (what === "fa") { live.write({ op: "patch", id: res.displayId, path: "#/window", value: 1 }); live.write({ op: "patch", id: res.displayId, path: "#/level", value: 0.5 }); }
    storedMaps.set(scan.browserId, { ...storedMaps.get(scan.browserId), [what]: res.imageId });
    return res.imageId;
  }
  /** A float NRRD the import job stored in the database's cache folder, or null. */
  async function storedNrrd(path: string): Promise<{ dims: [number, number, number]; ijkToRAS: number[]; data: Float32Array } | null> {
    const url = await databaseFileUrl(path);
    const r = url ? await fetch(url).catch(() => null) : null;
    if (!r?.ok) return null;
    const { f, body } = nrrdSplitHeader(new Uint8Array(await r.arrayBuffer()));
    const dims = (f["sizes"] ?? "").split(/\s+/).filter(Boolean).map(Number) as [number, number, number];
    const raw = await nrrdDecode(f, body, dims[0] * dims[1] * dims[2] * 4);
    return { dims, ijkToRAS: nrrdGeometry(f).ijkToRAS, data: new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)) };
  }
  /**
   * THE MAPS A TRACT IS MEASURED ON (v3: FA, MD, AD, RD per tract and side; selection.ts measure): the ones the import job
   * stored beside the tracts used (after every correction, lined up with the T1 at the scan's own spacing), else this module's own fit. Kept per scan
   * and per source, so tracts made here are measured on the fit they were made from.
   */
  const mapsKept = new Map<string, { key: string; maps: Promise<Maps | null> }>();
  function mapsFor(scan: Scan): Promise<Maps | null> {
    const w = whole.get(scan.browserId), st = storedState.get(scan.browserId);
    const key = w && !w.stored ? `fit:${w.key}` : `stored:${st?.seriesUID ?? ""}`;
    const have = mapsKept.get(scan.browserId);
    if (have && have.key === key) return have.maps;
    const maps = (async (): Promise<Maps | null> => {
      if (key.startsWith("stored:") && st?.seriesUID) {
        const uid = st.seriesUID;
        const [fa, md, ad, rd] = await Promise.all([faPath(uid), mdPath(uid), adPath(uid), rdPath(uid)].map((p) => storedNrrd(p).catch(() => null)));
        const g = fa ?? md;
        if (!g) return null;
        return { dims: g.dims, ijkToRAS: g.ijkToRAS, fa: fa?.data, md: md?.data, ad: ad?.data, rd: rd?.data };
      }
      const c = w?.c ?? computed.get(scan.browserId);
      if (!c) return null;
      const n = c.fit.fa.length, ad = new Float32Array(n), rd = new Float32Array(n), ev = c.fit.evals;
      for (let v = 0; v < n; v++) { ad[v] = ev[3 * v]; rd[v] = (ev[3 * v + 1] + ev[3 * v + 2]) / 2; }
      return { dims: c.fit.dims, ijkToRAS: c.fit.ijkToRAS, fa: c.fit.fa, md: c.fit.md, ad, rd };
    })().catch(() => null);
    mapsKept.set(scan.browserId, { key, maps });
    return maps;
  }
  /** Is the server preparing this extension's work right now? (GET /_jobs, desktop/extension-jobs.ts) */
  // ONLY FOR THE DATABASE THIS WINDOW HAS OPEN (critic 2026-10-14, import-time jobs, finding 4: the button waited for work
  // on any database). It cannot say which scan the run is on (counts only), so it waits for the run on this database.
  const jobRunning = async (): Promise<boolean> => {
    const s = await fetch("/_jobs", { cache: "no-store" }).then((r) => r.ok ? r.json() : null).catch(() => null) as { running?: { extension?: string; db?: string }; queued?: string[] } | null;
    if (!s) return false;
    const open = await openDatabaseId().catch(() => null);
    if (!open) return false;
    return (s.running?.extension === "diffusion" && s.running.db === open) || !!s.queued?.includes(open);
  };
  /** The server starts no job while this window tracks on the graphics card (finding 3); renewed while the run lasts. */
  const holdJobs = (): (() => void) => {
    const ask = () => { void fetch("/_jobs/hold", { method: "POST" }).catch(() => {}); };
    ask();
    const t = setInterval(ask, 60_000);
    return () => clearInterval(t);
  };
  // THE STORED SET ONLY WHEN IT IS WHAT THIS MODULE WOULD MAKE (finding 7): the default method and settings under
  // Advanced. Anything else the person chose is made here, as chosen.
  const ADV_DEFAULT = JSON.stringify(adv);
  const usesDefaults = (scan: Scan) => method === "ukf" && correct && JSON.stringify(adv) === ADV_DEFAULT && !preCorrected.has(scan.browserId);

  /**
   * TRACTS NEAR A STRUCTURE, NAMED (Ron, 2026-09-30: whole brain first, then keep what is near the tumor, whole): UKF
   * through the whole brain, TractCloud names every streamline (tractcloud/), and each named tract that comes within
   * `withinMm` of the structure is shown -- all of it, since planning asks what a connection would lose. Unnamed
   * streamlines that pass that close are shown too, in gray (hiding a real tract is worse than showing an unnamed one);
   * everything else is kept, hidden, as "Rest of the brain".
   */
  // ── THE WHOLE BRAIN'S RUN, ONCE PER SCAN (v3, Ron 2026-10-15): every way of choosing tracts -- by name, through a
  // box, near a segment or a tract -- picks from the same named whole-brain streamlines: the ones made at import when
  // the database has them, else the ones this module makes, once, with the settings in force (kept until a setting or
  // the scan changes).
  interface Whole { sl: Float32Array[]; named: Named; stored: boolean; storedInputs?: { partner?: string | null; t1?: string | null }; made?: string; c?: Computed; times: StageTimes; key: string }
  const whole = new Map<string, Whole>();
  // THE KEY NAMES WHAT A RUN MADE HERE WAS MADE FROM (critic 2026-10-15, finding 7): the settings, and the reversed scan
  // and the T1 present then -- one arriving later makes the run again.
  const settingsKey = (scan: Scan) => JSON.stringify([method, correct, adv, correct ? partnerFor(scan)?.id ?? "" : "", anatomyFor(scan)?.id ?? ""]);
  // ONE RUN IN PROGRESS PER SCAN (finding 6: a run for one scan was handed to another).
  const wholeBusy = new Map<string, Promise<Whole>>();
  async function ensureWholeBrain(scan: Scan): Promise<Whole> {
    const have = whole.get(scan.browserId);
    if (have && (have.stored ? usesDefaults(scan) : have.key === settingsKey(scan))) return have;
    const going = wholeBusy.get(scan.browserId);
    if (going) return await going;
    const run = (async () => {
      let releaseHold: (() => void) | undefined;
      try {
        const t0 = performance.now();
        const model = await tractCloud();
        tractCount = model.json.tracts.length;
        // THE TRACTS MADE AT IMPORT, when the database has them (UKF, the default method, is what the job runs); while the
        // server is still preparing them, wait for it and say so (Ron's "q3: A": one run, never two).
        let stored = usesDefaults(scan) ? await loadStored(scan, model).catch(() => null) : null;
        if (!usesDefaults(scan)) say("Your settings under Advanced differ from the ones the tracts made at import used: making them here, as set.");
        if (!stored && usesDefaults(scan) && await jobRunning()) {
          const tw = performance.now();
          while (await jobRunning() && performance.now() - tw < 15 * 60_000) {
            busy = `Waiting for the fiber tracts being prepared… ${Math.round((performance.now() - tw) / 1000)} s`; render();
            await new Promise((r) => setTimeout(r, 3000));
          }
          stored = await loadStored(scan, model).catch(() => null);
          busy = "Making tracts…"; render();
        }
        // THE SCAN'S OWN FIT only when the tracts are made here: stored tracts were corrected and aligned by the job, and
        // choosing among them needs neither (PAT16: 48 s, 45 of them the fit). A fit made already is used.
        const c = stored ? computed.get(scan.browserId) : await ensureFit(scan);
        const times: StageTimes = stored ? {} : takeFitTimes();
        let w: Whole;
        if (stored) {
          times.track = 0; times.name = 0;
          w = { sl: stored.sl, named: stored.named, stored: true, storedInputs: storedState.get(scan.browserId)?.inputs, made: stored.made, c, times, key: "stored" };
          say(`Using the whole brain's fiber tracts prepared for this scan: ${stored.sl.length.toLocaleString()} streamlines.`);
        } else {
          if (!c) throw new Error("the scan's maps are not made");
          releaseHold = holdJobs();
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
          const rp = await rapidParc(), releaseN = holdDrawing("naming tracts");
          const named = await nameTracts(device, model, sl, { rapidParc: rp }).finally(releaseN);
          times.name = named.seconds * 1000;
          w = { sl, named, stored: false, c, times, key: settingsKey(scan) };
        }
        whole.set(scan.browserId, w);
        return w;
      } finally {
        releaseHold?.(); wholeBusy.delete(scan.browserId);
        // ITS OWN WAITING WORDS GO WITH IT (finding 2: a box on a scan without stored tracts left "Naming tracts…" for good,
        // every button disabled).
        if (/^(Waiting for the fiber tracts|Making tracts|Naming tracts|Getting the whole brain)/.test(busy)) busy = "";
      }
    })();
    wholeBusy.set(scan.browserId, run);
    return await run;
  }

  /** The stored whole-brain tracts, read once into `whole` when the scan arrives (nothing made here: no fit, no tracking). */
  async function prefetchStored(scan: Scan): Promise<void> {
    if (whole.has(scan.browserId) || !usesDefaults(scan) || storedState.get(scan.browserId)?.state !== "stored") return;
    const model = await tractCloud();
    tractCount = model.json.tracts.length;
    const st = await loadStored(scan, model);
    if (!st || whole.has(scan.browserId) || !scans().some((x) => x.browserId === scan.browserId)) return;
    whole.set(scan.browserId, { sl: st.sl, named: st.named, stored: true, storedInputs: storedState.get(scan.browserId)?.inputs, made: st.made, c: computed.get(scan.browserId), times: { track: 0, name: 0 }, key: "stored" });
  }
  /** TRACTS NEAR A SEGMENT: its voxels as the structure (the near-the-tumor list, and any segment). */
  async function makeNamedTracts(scan: Scan, target: { seg: MrsonNode; labelValue: number; label: string }) {
    const z = await fetchZarrVolumeNative(live.blobBase(), target.seg.zarr as ZarrDesc), lab = z.data;
    await measureNear(scan, { dims: target.seg.dims as number[], ijkToRAS: target.seg.ijkToRAS as number[], inside: (v) => Number(lab[v]) === target.labelValue }, target.label);
  }
  /** TRACTS NEAR A TRACT (v3's "Tracts near this one…"; Ron, 2026-10-14: "could mean: near the uncinate fasciculus"): the
   *  voxels its streamlines pass through, on the scan's maps' grid, as the structure. */
  async function tractsNearTract(scan: Scan, strands: Float32Array[], label: string) {
    const maps = await mapsFor(scan);
    if (!maps) { say("The tract's place cannot be measured: the scan's maps are not ready."); return; }
    const vox = voxelsOf(strands, maps);
    await measureNear(scan, { dims: maps.dims, ijkToRAS: maps.ijkToRAS, inside: (v) => vox.has(v) }, label);
  }

  /**
   * TRACTS NEAR A STRUCTURE, NAMED (Ron, 2026-09-30: whole brain first, then keep what is near the tumor, whole): each
   * named tract that comes within `withinMm` of the structure is shown -- all of it, since planning asks what a connection
   * would lose. Unnamed streamlines that pass that close are shown too, in gray; everything else is kept, hidden, as
   * "Rest of the brain".
   */
  async function measureNear(scan: Scan, structure: Structure, label: string) {
    try {
      const t0 = performance.now();
      busy = busy || "Making tracts…"; render();
      const w = await ensureWholeBrain(scan), model = await tractCloud();
      const { sl, named, c } = w, times: StageTimes = { ...w.times }, before = 0;
      const tDist = performance.now();
      busy = "Measuring distances…"; render();
      const grayMm = withinMm + GRAY_BAND_MM;
      const dist = await streamlineDistances(structure, sl, grayMm + 2);
      // Streamlines that cross the fluid at the brain's edge would keep no tract's name (outside-brain.ts; Ron, 2026-10-04:
      // "The only bad thing is mislabeling them"). OFF: it also took corticospinal fibers along the medulla.
      const outside = OUTSIDE_RULE.on && c ? outsideBrain(sl, c.fit, edgeFluid(c.fit)) : undefined;
      const sorted = sortByDistance(model, named, dist, withinMm, MIN_NEAR_STREAMLINES, grayMm, outside), nearTracts = sorted.near;
      times.distances = performance.now() - tDist;
      const tDraw = performance.now();
      // STILL WANTED? A scan removed while this ran leaves nothing behind (critic, 2026-10-01, finding 8; CONSTRAINTS).
      if (!scans().some((s) => s.browserId === scan.browserId)) return;
      // A NEW RUN REPLACES THE LAST ONE for this scan (critic, finding 5: a second press doubled every tract).
      // WHAT THE PERSON SET ON THE LAST LIST'S ROWS -- a color, a side turned off -- carries over to the same tract and
      // side in the new list (critic 2026-10-16, shown-list finding 5: a Within change dropped them silently). Lines
      // taken out by hand do not: the new list is made afresh, and the banner of a removal in progress ends.
      const carried = new Map(groups.filter((g) => g.scan === scan.browserId && g.run && g.tract !== undefined).map((g) => [`${g.tract}:${g.side ?? 0}`, { color: g.color, visible: g.visible }]));
      groups.splice(0, groups.length, ...withoutLastRun(groups, scan.browserId));   // face.ts
      const pick = (idx: number[]) => idx.map((i) => sl[i]);
      runs.set(scan.browserId, { sl, named, sorted, structure, label, withinMm, method, added: new Set(), dist, stored: w.stored, storedInputs: w.storedInputs, maxB: c?.maxB ?? 0, partnerId: c?.partnerId ?? "", anatomyId: c?.anatomyId ?? "", rule: method === "ukf" && c ? c.rule : undefined, brain: method === "ukf" && c ? c.brain.note : "", ...(c?.directions ? { directions: c.directions } : {}) });
      // The anatomy behind the slices, where the tracts' crossings are drawn (Yogesh Rathi via Ron, 2026-10-01), and over
      // it this scan's own map -- never another patient's left from before (critic, 2026-10-01, finding 3).
      const anat = anatomyFor(scan);
      if (anat) {
        const fg = ((([...live.nodes.values()].find((n) => n.type === "sliceComposite")?.refs as Record<string, string[]> | undefined)?.foreground) ?? [])[0];
        const mapId = storedMaps.get(scan.browserId)?.colorfa ?? c?.colorFaId;
        const mine = fg !== undefined && (fg === mapId || fg === c?.faId || fg === storedMaps.get(scan.browserId)?.fa);
        putBackground(anat.id, mine ? undefined : (mapId && live.nodes.get(mapId) ? mapId : null));
      }
      // The near tracts shown; the faint ones (within reach, fewer than the minimum) listed after them, hidden.
      for (const [e, faint] of [...nearTracts.map((e) => [e, false] as const), ...sorted.faint.map((e) => [e, true] as const)]) {
        const t = model.json.tracts[e.tract];
        groups.push({ id: ++groupSeq, name: t ? tractLabel(t.abbr, t.name, e.side) : tractName(model, e.tract, e.side), scan: scan.browserId, strands: pick(e.idx), visible: !faint, method,
          tract: e.tract, side: e.side, distanceMm: e.d, within: e.within, run: true, faint, otherSide: e.side ? sorted.total(e.tract, otherSide(e.side)) : undefined });
        const was = carried.get(`${e.tract}:${e.side ?? 0}`);
        if (was) { const g = groups[groups.length - 1]; g.color = was.color; g.visible = was.visible; }
      }
      // THE OTHER SIDE OF EACH LISTED TRACT, hidden, so the row has Left and Right to compare (v3; Ron, 2026-10-01 and
      // O'Donnell et al. 2017: the tumor's side against the healthy side), measured like the rest.
      const listedKeys = new Set(groups.filter((g) => g.scan === scan.browserId && g.run && g.tract !== undefined).map((g) => `${g.tract}:${g.side}`));
      const compared = new Set<number>(), every = [...sorted.near, ...sorted.far];
      for (const g of groups.filter((x) => x.scan === scan.browserId && x.run && x.tract !== undefined && x.side)) {
        const os = otherSide(g.side!), k = `${g.tract}:${os}`;
        if (listedKeys.has(k)) continue;
        const e = every.find((x) => x.tract === g.tract && x.side === os);
        if (!e?.idx.length) continue;
        listedKeys.add(k); for (const i of e.idx) compared.add(i);
        const t = model.json.tracts[g.tract!];
        groups.push({ id: ++groupSeq, name: t ? tractLabel(t.abbr, t.name, os) : tractName(model, g.tract!, os), scan: scan.browserId, strands: pick(e.idx), visible: false, method, tract: g.tract, side: os, distanceMm: e.d, within: e.within, run: true, compare: true });
      }
      // ONE GROUP PER TRACT AND SIDE (critic 2026-10-15, finding 1: a tract ticked by name and then listed here was drawn
      // twice and exported twice): the run's group stays, shown if the ticked one was.
      for (let i = groups.length - 1; i >= 0; i--) {
        const g = groups[i];
        if (g.scan !== scan.browserId || g.by !== "name") continue;
        const k = `${g.tract}:${g.side}`;
        if (!listedKeys.has(k)) continue;
        const r = groups.find((x) => x.scan === scan.browserId && x.run && `${x.tract}:${x.side}` === k);
        if (r && g.visible) r.visible = true;
        groups.splice(i, 1);
      }
      // The rest: the unnamed far ones and the far tracts not listed as faint or for comparison (each is its own row).
      const unnamedNear = sorted.unnamedNear, faintSet = new Set(sorted.faint), rest = [...sorted.unnamedFar, ...sorted.outsideFar, ...sorted.far.filter((e) => !faintSet.has(e)).flatMap((e) => e.idx)].filter((i) => !compared.has(i));
      if (sorted.outsideNear.length) groups.push({ id: ++groupSeq, name: `Outside the brain, possibly a cranial nerve, within ${withinMm} mm of ${label}`, scan: scan.browserId, strands: pick(sorted.outsideNear), visible: true, method, unnamed: true, run: true, outside: true });
      if (unnamedNear.length) groups.push({ id: ++groupSeq, name: `Not named, within ${withinMm} mm of ${label}`, scan: scan.browserId, strands: pick(unnamedNear), visible: true, method, unnamed: true, run: true });
      if (rest.length) groups.push({ id: ++groupSeq, name: "Rest of the brain", scan: scan.browserId, strands: pick(rest), visible: false, method, unnamed: true, run: true });
      redraw3d();
      const t2 = performance.now();
      times.draw = t2 - tDraw;
      times.total = before + (t2 - t0);
      // ONE COUNT, AS THE LIST HAS IT (finding 19): tracts, both sides one row; the timing to the log only (finding 22).
      const nTracts = new Set(nearTracts.map((e) => e.tract)).size;
      say(`${nTracts} named tract${nTracts === 1 ? "" : "s"} come within ${withinMm} mm of ${label}${sorted.faint.length ? `; ${new Set(sorted.faint.map((e) => e.tract)).size} more are listed in gray, hidden` : ""}.`);
      status(`Near ${label}: ${nearTracts.length} tract sides of ${sl.length.toLocaleString()} streamlines through the whole brain. Step by step: ${stageText(times)}.`);
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
    const releaseHold = holdJobs();   // the server starts no job while these lines are followed (finding 3)
    try {
      const c = await ensureFit(scan), t0 = performance.now();
      // THE SAME SPACE (critic, 2026-10-03, finding 1): the scan aligned to the same anatomy MRI, or to none, as the run.
      // ON STORED TRACTS (critic 2026-10-14, import-time jobs, finding 7): new lines are followed the way the stored set was
      // made -- aligned to the T1, corrected with the reversed scan the set used, inside SynthStrip's brain (rule 3) -- or
      // not at all, said in plain words.
      if (run.stored) {
        const why = !c.anatomyId ? "the MRI of the anatomy (T1) is not loaded with the scan"
          : run.storedInputs?.partner && !c.partnerId ? "the scan taken with the reversed direction is not loaded (the stored tracts were corrected with it)"
          : !usesDefaults(scan) ? "the settings under Advanced differ from the ones the stored tracts were made with" : "";
        if (why) { say(`Lines cannot be added to the tracts made at import here: ${why}.`); return; }
        await brainFor(c);
      } else if (c.maxB !== run.maxB || c.partnerId !== run.partnerId || (c.anatomyId ?? "") !== run.anatomyId) {
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
    finally { releaseHold(); busy = ""; adding = false; render(); }
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
  /** LEFT AND RIGHT IN TWO SHADES OF THE TRACT'S COLOR (the v3 mockup, "drawn in two shades of its color"; critic
   *  2026-10-15, finding 20): the left side a darker shade, so a lopsided tract can be read in the view. */
  const sideShade = (c: RGBA, side?: number): RGBA => side === -1 ? [c[0] * 0.62, c[1] * 0.62, c[2] * 0.62, c[3]] : c;
  function redraw3d() {
    const shownNow = groups.filter((g) => g.visible);
    probeIndex = shownNow.length ? { ix: buildTractIndex(shownNow.map(drawn)), shown: shownNow } : undefined;
    const view = live.view;
    if (!view) return;
    drawSliceCrossings();
    // SHOWING OR HIDING IS A COLOR, NOT A REBUILD (render-audit 2026-10-15, finding 5: hiding one tract built a new field
    // from every shown one -- 36 MB sent to the card, the page held 0.4 s -- and hiding the last or showing the first
    // rebuilt the 3D shader). Each group in the field has its own palette slot; when every group to be shown is already
    // in the field, drawn the same way, a hidden one is made fully transparent (no tube, no halo) and a shown one gets its
    // color back, and the view redraws. The field is rebuilt only for something new, a different look, or direction
    // coloring (one slot per direction, shared by groups).
    const look = `${drawAs}|${trimVoxels}|${colorBy}|${tractCount}`;
    // NOT WHEN NOTHING IS SHOWN (critic 2026-10-15, render-audit-fixes finding 5: with every tract hidden or removed, or
    // the scan gone from the scene, the field stayed on the card -- 38 MB -- and every ray still walked its grid): then
    // the field goes, below, and showing a tract again builds it.
    const fast = !!field && fieldSlots && colorBy === "tract" && fieldLook === look && shownNow.length > 0
      && shownNow.every((g) => fieldSlots!.get(g)?.n === g.strands.length);
    if (fast) {
      for (const [g, s] of fieldSlots!) field!.setBundleColor(s.id, g.visible && groups.includes(g) ? s.color : [0, 0, 0, 0]);
      view.redraw();
      return;
    }
    // Colored by tract: each COLOR its own slot -- the tract's color (tract-colors.ts), the left side a shade darker, the
    // unnamed ones all gray in one (GRAY_ID); a group seeded by hand, and every group when colored by direction, by
    // direction. Groups of one color share a slot (critic 2026-10-15, render-audit-fixes finding 6: each unnamed group
    // took a slot of its own, and past the 38 slots tracts were drawn by direction in 3D while their rows and their
    // dots on the slices kept the tract's color); only past 38 different colors at once does a group fall back.
    const pal = directionPalette(), count = tractCount, slots = new Map<TractGroup, { id: number; n: number; color: RGBA }>();
    const byColor = new Map<string, number>();
    let slot = TRACT_ID, shared = false;
    const strands: Strand[] = shownNow.flatMap((g) => {
      let id = 0;
      // THE PERSON'S CHOICE FIRST (the row's color square): a solid color for any group, or by direction.
      const own = colorBy === "tract" ? g.color : undefined;
      if (colorBy === "tract" && own !== "direction" && (own || g.tract !== undefined || g.unnamed)) {
        const c = own ? sideShade(own, g.side) : g.tract !== undefined ? sideShade(tractColor(g.tract, count), g.side) : UNNAMED;
        const k = own || g.tract !== undefined ? c.join(",") : "unnamed";
        let sid = byColor.get(k);
        if (sid !== undefined) shared = true;
        else if (!own && g.tract === undefined) sid = GRAY_ID;
        else if (slot <= 255) sid = slot++;
        if (sid !== undefined) { byColor.set(k, sid); pal[sid] = c; id = sid; slots.set(g, { id, n: g.strands.length, color: c }); }
      }
      return drawn(g).map((p) => id ? { points: p, bundle: id } : { points: p, bundle: directionId(p), pointBundles: pointIds(p) });
    });
    // Every shown group has a slot of its own: the next show or hide can be a color. A shared slot cannot hide one of
    // its groups alone, so then the next change rebuilds.
    fieldSlots = !shared && slots.size === shownNow.length ? slots : null;
    fieldLook = look;
    // NOTHING SHOWN: when the field could stay (tract colors, same look), the fast path above already made every slot
    // transparent and kept it, so hiding the last tract or showing the first again rebuilds nothing (finding 5); here the
    // look changed or the coloring is by direction, and the field goes.
    if (!strands.length) { const old = field; field = undefined; fieldSlots = null; view.removeField(FIELD_KEY); old?.destroy?.(); return; }
    const old = field;
    field = new FiberField(device, strands, { radius: RADIUS[drawAs], bundleColors: pal });
    view.setField(FIELD_KEY, field);
    old?.destroy?.();
  }
  /** The groups in the tract field and their palette slots (null: the next change rebuilds it), and the look it was made in. */
  let fieldSlots: Map<TractGroup, { id: number; n: number; color: RGBA }> | null = null;
  let fieldLook = "";

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
      if (colorBy === "tract" && g.color && g.color !== "direction") { const c = sideShade(g.color, g.side); return [c[0], c[1], c[2]]; }
      if (colorBy === "tract" && g.color === "direction") return [Math.abs(dir[0]), Math.abs(dir[1]), Math.abs(dir[2])];
      if (colorBy === "tract" && g.tract !== undefined) { const c = sideShade(tractColor(g.tract, tractCount), g.side); return [c[0], c[1], c[2]]; }
      if (colorBy === "tract" && g.unnamed) return [UNNAMED[0], UNNAMED[1], UNNAMED[2]];
      return [Math.abs(dir[0]), Math.abs(dir[1]), Math.abs(dir[2])];    // by direction, as in 3D: red left-right ...
    };
    // WHOSE TRACTS ON WHICH VIEW (critic, 2026-10-01, finding 3): a view shows the tracts of the scan whose images it
    // shows (its anatomy, its frames, its maps); with tracts for one scan only, that scan's, whatever is underneath.
    const withTracts = [...new Set(groups.filter((g) => g.visible && g.strands.length).map((g) => g.scan))];
    const imagesOf = (scanId: string) => {
      const sc = scans().find((x) => x.browserId === scanId), c = computed.get(scanId);
      return new Set([anatomyFor(sc)?.id, ...(sc?.frameIds ?? []), c?.faId, c?.colorFaId, c?.b0Id, storedMaps.get(scanId)?.b0].filter((x): x is string => !!x));
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
    // The stored maps of a scan that left go with it, as the computed ones do.
    for (const [id, m] of storedMaps) if (!ids.has(id)) { for (const n of [m.fa, m.colorfa, m.b0]) if (n && live.nodes.get(n)) removeVolumeFromScene(live, n); storedMaps.delete(id); storedState.delete(id); changed = true; }
    for (const id of [...runs.keys()]) if (!ids.has(id)) runs.delete(id);
    for (const id of [...whole.keys()]) if (!ids.has(id)) whole.delete(id);         // the whole brain's streamlines go with the scan
    for (const id of [...mapsKept.keys()]) if (!ids.has(id)) mapsKept.delete(id);
    for (const id of [...mapOn.keys()]) if (!ids.has(id)) mapOn.delete(id);
    for (const id of [...storedState.keys()]) if (!ids.has(id)) storedState.delete(id);
    for (const [id, b] of [...boxes]) if (!ids.has(id)) { if (live.nodes.get(b)) live.write({ op: "del", id: b }); boxes.delete(id); boxLines.delete(id); }
    for (const id of [...wayBy.keys()]) if (!ids.has(id)) wayBy.delete(id);
    for (const id of [...exportedFor.keys()]) if (!ids.has(id)) exportedFor.delete(id);
    for (let i = groups.length - 1; i >= 0; i--) if (!ids.has(groups[i].scan)) { groups.splice(i, 1); changed = true; }
    for (const id of [...seenScans]) if (!ids.has(id)) seenScans.delete(id);
    if (changed) redraw3d();
    return changed;
  }

  // ── the panel ─────────────────────────────────────────────────────────────────────────────────────────────────
  // ── THE v3 FACE (Ron, 2026-10-15: "v3: 1: yes, 2: yes"; the workspace's Contents/docs/DMRI-WORKFLOW-RETHINK.md and
  // mockups/diffusion-generic-v3-2026-10-14.html, checked against MODULE-DESIGN-RULES.md there). One face for a patient
  // with a tumor and for a normal-appearing brain: "The scans" only when something is missing; CHOOSE TRACTS -- near a
  // structure (the default; a tumor case opens on it), by name (any other brain opens on it), or through a box; SHOWN --
  // one row per tract, Left and Right side by side with lines and FA, the rest in the row's tooltip and the export.
  // Changes from the mockup, said to Ron: Add lines stays a visible button under the list (he did not find it in the
  // tract list's top row); Draw as and Color stay under Advanced (his "yes, keep them in the module").
  // PER SCAN (critic 2026-10-15, findings 5, 8, 24): the way it is chosen by, its box, its last export.
  const wayBy = new Map<string, { way: Way; touched: boolean }>();
  let wantOutline = false;
  const boxes = new Map<string, string>(), boxLines = new Map<string, string>();
  let boxQueued = false;
  const exportedFor = new Map<string, string>();
  /** "Tracts near this one…": the tract that is the structure now, per scan (finding 18). */
  const nearTract = new Map<string, { strands: Float32Array[]; label: string }>();
  const measured = new WeakMap<TractGroup, { n: number; m: Measures }>();
  let measuringP: Promise<void> | undefined;
  const nameOf = (g: TractGroup) => g.label ?? g.name;
  /** A tract's name without its side, as a row shows it. */
  const tractTitle = (tract: number) => { const t = tractCountModel?.json.tracts[tract]; return t ? tractLabel(t.abbr, t.name, 0) : `Tract ${tract}`; };
  let tractCountModel: TractCloudModel | undefined;

  /** The numbers of every listed tract of a scan, measured once its maps are there (each voxel once; selection.ts). */
  // ONE MEASURING AT A TIME, AND A CALLER WAITS FOR IT (finding 3: an export pressed while measuring wrote a partial file).
  async function ensureMeasures(scan: Scan): Promise<boolean> {
    while (measuringP) await measuringP;
    const todo = groups.filter((g) => g.scan === scan.browserId && (g.tract !== undefined || g.by === "box") && measured.get(g)?.n !== g.strands.length);
    if (!todo.length) return true;
    let ok = false;
    measuringP = (async () => {
      const maps = await mapsFor(scan);
      if (!maps) return;
      for (const g of todo) { measured.set(g, { n: g.strands.length, m: measure(g.strands, maps) }); await new Promise((r) => setTimeout(r, 0)); }
      ok = true;
    })().finally(() => { measuringP = undefined; });
    await measuringP;
    render();
    return ok;
  }

  /** BY NAME: ticking a tract shows both its sides from the whole brain's run; unticking hides them (Ron's v3: "Tick a
   *  tract and it is drawn; untick it and it goes. Nothing to press."). */
  async function showByName(scan: Scan, tract: number, on: boolean) {
    const mine = groups.filter((g) => g.scan === scan.browserId && g.tract === tract);
    if (!on) {
      for (let i = groups.length - 1; i >= 0; i--) if (groups[i].scan === scan.browserId && groups[i].tract === tract) { if (groups[i].by === "name") groups.splice(i, 1); else groups[i].visible = false; }
      redraw3d(); render(); return;
    }
    if (mine.length) { for (const g of mine) g.visible = true; redraw3d(); render(); return; }
    try {
      if (!whole.get(scan.browserId)) { busy = "Getting the whole brain's tracts…"; render(); }
      const w = await ensureWholeBrain(scan);
      const idx = byName(w.named, [{ tract, side: -1 }, { tract, side: 0 }, { tract, side: 1 }]);
      const model = await tractCloud(); tractCountModel = model;
      const t = model.json.tracts[tract];
      for (const side of [-1, 0, 1]) {
        const ids = idx.get(`${tract}:${side}`) ?? [];
        if (!ids.length) continue;
        groups.push({ id: ++groupSeq, name: t ? tractLabel(t.abbr, t.name, side) : tractName(model, tract, side), scan: scan.browserId, strands: ids.map((i) => w.sl[i]), visible: true, method, tract, side, by: "name" });
      }
      if (!groups.some((g) => g.scan === scan.browserId && g.tract === tract)) say(`No streamlines of ${t?.name ?? "this tract"} were found in this brain.`);
      redraw3d();
    } catch (e) { say(`Tracts could not be shown: ${(e as Error).message}`); }
    finally { busy = ""; render(); void ensureMeasures(scan); }
  }

  /** THROUGH A PLACE: one box (a Markups box), put at the slice views' crossing; the streamlines through it follow it as
   *  it moves, with nothing to press (Ron's v3 "2: yes"). */
  function putBox(scan: Scan) {
    const c = [0, 0, 0];
    for (const cell of ["Red", "Yellow", "Green"]) {
      const o = sliceOrientation(cell), d = sliceOffset(cell);
      if (d === undefined) continue;
      if (o === "Sagittal") c[0] = d; else if (o === "Coronal") c[1] = d; else if (o === "Axial") c[2] = d;
    }
    const boxId = `diffusion-box-${scan.browserId}`;
    boxes.set(scan.browserId, boxId);
    live.write({ op: "put", id: boxId, node: { type: "markup", id: boxId, name: "Tracts through this box", markupType: "roi", frame: "RAS", center: c, size: [20, 20, 20], visible: true, locked: false, points: [], source: { mrmlClass: "vtkMRMLMarkupsROINode" }, origin: { local: true } } as unknown as MrsonNode });
    void boxMoved(scan);
  }
  async function boxMoved(scan: Scan) {
    const boxId = boxes.get(scan.browserId);
    const n = boxId ? live.nodes.get(boxId) : undefined;
    // THE BOX DELETED (in Markups): its lines and its words go too (finding 10).
    if (!n || !n.center || !n.size) {
      for (let i = groups.length - 1; i >= 0; i--) if (groups[i].scan === scan.browserId && groups[i].by === "box") groups.splice(i, 1);
      boxes.delete(scan.browserId); boxLines.delete(scan.browserId); redraw3d(); render(); return;
    }
    const t0 = performance.now();
    const w = await ensureWholeBrain(scan).catch((e) => { say(`Tracts could not be shown: ${(e as Error).message}`); return undefined; }).finally(() => render());
    if (!w) return;
    const box = { center: n.center as number[], size: n.size as number[], orientation: n.orientation as number[] | undefined };
    const idx = throughBox(w.sl, box);
    for (let i = groups.length - 1; i >= 0; i--) if (groups[i].scan === scan.browserId && groups[i].by === "box") groups.splice(i, 1);
    if (idx.length) groups.push({ id: ++groupSeq, name: "Through the box", scan: scan.browserId, strands: idx.map((i) => w.sl[i]), visible: true, method, unnamed: true, by: "box" });
    // Which tract most of them belong to, said in the line under the box.
    const count = new Map<string, number>();
    for (const i of idx) { const k = `${w.named.tract[i]}:${w.named.side[i]}`; count.set(k, (count.get(k) ?? 0) + 1); }
    const model = await tractCloud(); tractCountModel = model;
    const OTHER = model.json.tracts.length - 1;
    // The commonest NAMED tract: in the deep white matter most lines through a small box have no name.
    const top = [...count.entries()].filter(([k]) => Number(k.split(":")[0]) !== OTHER).sort((a, b) => b[1] - a[1])[0];
    const topName = top ? tractLabel(model.json.tracts[Number(top[0].split(":")[0])].abbr, model.json.tracts[Number(top[0].split(":")[0])].name, Number(top[0].split(":")[1])) : "";
    boxLines.set(scan.browserId, idx.length ? `Through the box: ${idx.length.toLocaleString()} lines${topName ? `; the named tract with the most of them: ${topName} (${Math.round(100 * top[1] / idx.length)}%)` : ", none of them named"}` : "No lines pass through the box.");
    console.info(`diffusion box: ${idx.length} lines in ${Math.round(performance.now() - t0)} ms`);
    redraw3d(); render();
  }
  // THE BOX'S OWN SCAN, whichever is chosen (finding 5); a quarter second between updates while it is dragged (finding 12).
  live.subscribe((c) => {
    const owner = [...boxes].find(([, b]) => b === c.id)?.[0];
    if (!owner || boxQueued) return;
    boxQueued = true;
    setTimeout(() => { boxQueued = false; const s = scans().find((x) => x.browserId === owner); if (s) void boxMoved(s); }, 250);
  });

  /** ONE TABLE FILE at the top of Downloads (Ron's v3 export, CSV): a row per tract and side, and Show in Finder. */
  async function exportNumbers(scan: Scan) {
    // EVERY LISTED TRACT MEASURED FIRST (finding 3), and a failure is a failure on the button (finding 11).
    if (!await ensureMeasures(scan)) throw new Error("the scan's maps are not ready");
    const st = storedState.get(scan.browserId), w = whole.get(scan.browserId), run = runs.get(scan.browserId), maps = await mapsFor(scan);
    const gridMm = maps ? [0, 1, 2].map((c) => Math.hypot(maps.ijkToRAS[c], maps.ijkToRAS[4 + c], maps.ijkToRAS[8 + c])) : undefined;
    // WHAT MADE THE NUMBERS (finding 23): the stored set's record with both commits, or the method and rule the run used.
    const madeBy = (g: TractGroup) => w?.stored ? `made at import; tracking rule ${st?.tracking ?? "?"}; Albula ${String(st?.rules?.core ?? "?")}; diffusion extension ${String(st?.rules?.extension ?? "?")}`
      : `made here (${g.method}); tracking rule ${run?.rule ?? "?"}`;
    const rows: ExportRow[] = [];
    for (const g of groups.filter((x) => x.scan === scan.browserId && x.tract !== undefined)) {
      const m = measured.get(g)?.m; if (!m) continue;
      const t = tractCountModel?.json.tracts[g.tract!];
      rows.push({ scan: scan.name, tract: g.label ?? (t ? tractLabel(t.abbr, t.name, 0).replace(/ \([^)]*\)$/, "") : g.name), abbreviation: t?.abbr ?? "", side: g.side === -1 ? "left" : g.side === 1 ? "right" : "both", m, madeBy: madeBy(g), gridMm, rule: MEASURE_RULE });
    }
    if (!rows.length) throw new Error("there are no tracts with numbers to export yet");
    // A FILE PER EXPORT (finding 11: a second export the same day overwrote the first): the time in the name.
    const now = new Date(), stamp = `${now.toISOString().slice(0, 10)} ${String(now.getHours()).padStart(2, "0")}-${String(now.getMinutes()).padStart(2, "0")}-${String(now.getSeconds()).padStart(2, "0")}`;
    const name = `${(patientOf(scan.name) || "scan").replace(/[^\w.-]+/g, "_")} fiber tract numbers ${stamp}.csv`;
    const r = await saveToDownloads(name, toCsv(rows));
    if (!r.ok) throw new Error(r.why ?? "the file could not be written");
    exportedFor.set(scan.browserId, name);
    say(`Saved ${rows.length} rows to Downloads: ${name}`);
    render();
  }

  /** The row's color choice (its color square): eight solid colors, by direction, or the tract's own. */
  const CHOICE_COLORS: RGBA[] = [[0.973, 0.843, 0.392, 1], [0.82, 0.27, 0.23, 1], [0.30, 0.79, 0.54, 1], [0.36, 0.61, 0.84, 1], [0.75, 0.31, 0.82, 1], [1.0, 0.60, 0.36, 1], [0.98, 0.98, 0.96, 1], [0.54, 0.49, 0.42, 1]];
  function colorChoice(anchor: HTMLElement, gs: TractGroup[], named: boolean) {
    const pop = document.createElement("div");
    const r = anchor.getBoundingClientRect();
    pop.style.cssText = `position:fixed;left:${Math.round(r.left)}px;top:${Math.round(r.bottom + 4)}px;z-index:1000;background:var(--sl-surface);border:1px solid var(--sl-control-edge);border-radius:6px;padding:8px;display:grid;gap:6px;box-shadow:var(--sl-shadow-float);width:210px`;
    const pal = document.createElement("div"); pal.style.cssText = "display:grid;grid-template-columns:repeat(8,1fr);gap:4px";
    const close = () => { pop.remove(); removeEventListener("pointerdown", outside, true); removeEventListener("keydown", esc, true); };
    // Choosing a color while every line is drawn by direction (Advanced › Color by) brings the colors back, so the
    // choice is seen (critic, finding 14).
    const pick = (c: RGBA | "direction" | undefined) => { for (const g of gs) g.color = c; if (c !== "direction") colorBy = "tract"; fieldSlots = null; redraw3d(); render(); close(); };
    for (const c of CHOICE_COLORS) {
      const b = document.createElement("button"); b.type = "button"; b.title = "This color";
      b.style.cssText = `height:20px;min-width:0;padding:0;border:1px solid var(--sl-line-strong);border-radius:3px;cursor:pointer;background:rgb(${c.slice(0, 3).map((v) => Math.round(v * 255)).join(",")})`;
      b.onclick = () => pick(c); pal.append(b);
    }
    const dir = document.createElement("button"); dir.type = "button"; dir.textContent = "By direction"; dir.title = "Red left–right, green front–back, blue up–down"; dir.onclick = () => pick("direction");
    pop.append(pal, dir);
    if (named) { const own = document.createElement("button"); own.type = "button"; own.textContent = "Its tract's color"; own.onclick = () => pick(undefined); pop.append(own); }
    const outside = (e: PointerEvent) => { if (!pop.contains(e.target as Node)) close(); };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") { e.preventDefault(); close(); } };
    document.body.append(pop);
    addEventListener("pointerdown", outside, true); addEventListener("keydown", esc, true);
  }

  // STRAY LINES, TAKEN OUT BY THE PERSON (Ron, 2026-10-16: "let the user do it. draw a rectangle to identify the culprits.
  // have an undo option"): a rectangle dragged in the 3D view or a slice (SDK 14 pickRectangle) takes out every SHOWN
  // line of the row that passes through it, as drawn; each rectangle is one step of undo, kept until the scan's list is
  // made again.
  const undoStack: { batch: number; g: TractGroup; gone: Float32Array[] }[] = [];
  /** Undo steps of rows no longer in the list go (critic, finding 4: they held the removed lines for the session). */
  const pruneUndo = () => { for (let i = undoStack.length - 1; i >= 0; i--) if (!groups.includes(undoStack[i].g)) undoStack.splice(i, 1); };
  /** What each row showed when the header's eye turned everything off, per scan: turned on, exactly that comes back. */
  const allOffMemory = new Map<string, Map<TractGroup, boolean>>();
  let undoBatch = 0, strayActive = false;
  function undoRemoval(gs: TractGroup[]): number {
    const last = [...undoStack].reverse().find((u) => gs.includes(u.g));
    if (!last) return 0;
    const batch = undoStack.filter((u) => u.batch === last.batch);
    let n = 0;
    // THE LINES TAKEN OUT GO BACK IN, and nothing else changes (critic, finding 3: restoring the earlier list threw away
    // the lines "Add lines" had added since).
    for (const u of batch) { n += u.gone.length; u.g.strands = [...u.g.strands, ...u.gone]; undoStack.splice(undoStack.indexOf(u), 1); }
    fieldSlots = null; redraw3d(); render();
    return n;
  }
  async function removeStray(gs: TractGroup[], base: string) {
    if (strayActive) return;
    strayActive = true;
    const stop = new AbortController();
    const done = { label: "Done", primary: true, title: "Stop removing lines (Esc)", onClick: () => stop.abort() };
    const hint = "drag a rectangle around them, in the 3D view or a slice · Esc or Done ends · the views turn again after Done";
    const b = viewBanner({ title: `Remove stray lines: ${base}`, hint, actions: [done] });
    const undoAct = () => ({ label: "Undo (⌘Z)", title: "Put back the lines the last rectangle took out", onClick: () => undoOnce() });
    const say2 = (title: string) => b?.update({ title, hint, actions: undoStack.some((u) => gs.includes(u.g)) ? [undoAct(), done] : [done] });
    const undoOnce = () => { const n = undoRemoval(gs); say2(n ? `${n.toLocaleString()} line${n === 1 ? "" : "s"} put back in ${base}` : `Nothing to put back in ${base}`); };
    const onZ = (e: KeyboardEvent) => { if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") { e.preventDefault(); e.stopPropagation(); undoOnce(); } };
    addEventListener("keydown", onZ, true);
    /** The lines of the shown groups a rectangle holds, as drawn: in a slice view where they cross its plane. */
    const takenBy = (r: PickedRectangle) => {
      const out = new Map<TractGroup, Set<Float32Array>>();
      for (const g of gs) {
        if (!g.visible) continue;                       // only what is shown
        const mm = trimVoxels * voxelMm(g.scan), hit = new Set<Float32Array>();
        for (const st of g.strands) {
          const d = mm > 0 ? trimEnds(st, mm) : st;     // the part drawn
          if (!d || d.length < 6) continue;
          for (let i = 0; i + 5 < d.length; i += 3) if (r.insideSegment([d[i], d[i + 1], d[i + 2]], [d[i + 3], d[i + 4], d[i + 5]])) { hit.add(st); break; }
        }
        if (hit.size) out.set(g, hit);
      }
      return out;
    };
    // THE MODE ENDS WITH ITS LIST (critic, findings 2 and 4): when the module is left, or its rows are no longer in the
    // list (a new run, the scan closed).
    const stale = () => !root?.isConnected || root.offsetParent === null || gs.some((g) => !groups.includes(g));
    const watch = setInterval(() => { if (stale()) stop.abort(); }, 300);
    try {
      for (;;) {
        const r = await pickRectangle({ signal: stop.signal, onDrag: (p) => {
          if (!p) { say2(`Remove stray lines: ${base}`); return; }
          let n = 0; for (const set of takenBy(p).values()) n += set.size;
          b?.update({ title: n ? `This rectangle would take ${n.toLocaleString()} line${n === 1 ? "" : "s"} of ${base}` : `No line of ${base} in this rectangle` });
        } });
        if (!r || stale()) break;
        const batch = ++undoBatch;
        let n = 0;
        for (const [g, hit] of takenBy(r)) {
          const gone = g.strands.filter((st) => hit.has(st));
          g.strands = g.strands.filter((st) => !hit.has(st));
          undoStack.push({ batch, g, gone }); n += gone.length;
        }
        if (n) { fieldSlots = null; redraw3d(); render(); }
        say2(n ? `${n.toLocaleString()} line${n === 1 ? "" : "s"} removed from ${base}` : `No line of ${base} in that rectangle`);
      }
    } finally { clearInterval(watch); removeEventListener("keydown", onZ, true); b?.close(); strayActive = false; pruneUndo(); }
  }

  function render() {
    syncBanner();
    pruneUndo();
    if (!root) return;
    if (!tractCountModel) void tractCloud().then((m) => { if (!tractCountModel) { tractCountModel = m; render(); } }).catch(() => {});
    const list = scans();
    if (!list.find((s) => s.browserId === chosen)) chosen = list[0]?.browserId ?? "";
    const scan = list.find((s) => s.browserId === chosen);
    const segs = segmentChoices(scan);
    if (outline && (!live.nodes.get(outline.seedsId) || !live.nodes.get(outline.imageId))) outline = undefined;   // finding 9
    const tumors = tumorChoices(scan);
    // THE WAY IT OPENS ON (Ron, 2026-10-14: near a structure the default; "Any other brain opens switched to By name").
    let wb = scan ? wayBy.get(scan.browserId) : undefined;
    if (scan && !wb) { wb = { way: openingWay(tumors.length > 0), touched: false }; wayBy.set(scan.browserId, wb); if (tumors.length) near = tumors[0].key; }
    // A TUMOR OUTLINE THAT LOADS AFTER THE SCAN (finding 8): an untouched face turns to Near, from the tumor.
    if (scan && wb && !wb.touched && wb.way === "name" && tumors.length) { wb.way = "near"; near = tumors[0].key; }
    const way: Way = wb?.way ?? "near";
    const setWay = (w: Way) => { if (wb) { wb.way = w; wb.touched = true; } render(); };
    if (!segs.find((s) => s.key === near)) near = (tumors[0] ?? segs[0])?.key ?? "";
    const anat = anatomyFor(scan);
    root.innerHTML = "";

    // ── THE PATIENT, AND HOW THE TRACTS WERE MADE: quiet lines, no section ───────────────────────────────────────────
    const top = document.createElement("div"); top.style.cssText = "margin:2px 0 6px";
    if (list.length > 1) {
      const pp = document.createElement("select");
      for (const s0 of list) pp.append(new Option(`${patientOf(s0.name) || "Patient"} — ${s0.name.replace(/^.*?·\s*(MR\s+)?/, "")}`, s0.browserId, false, s0.browserId === chosen));
      pp.title = "More than one diffusion MRI is loaded: the one this module is about.";
      pp.onchange = () => { chosen = pp.value; render(); };
      shell.row(top, "Scan").append(pp);
    } else if (scan) {
      const pl = document.createElement("div"); pl.className = "sl-hint"; pl.textContent = `${patientOf(scan.name) || "Scan"} · ${scan.name.replace(/^.*?·\s*(MR\s+)?/, "")}`; top.append(pl);
    }
    if (scan) {
      const st = storedState.get(scan.browserId);
      // WHAT, NOT WHEN (Ron, 2026-10-15): the steps, each with its method (methods.ts); the date and rule numbers in the tooltip.
      const steps = st?.state === "stored" ? howMade(st) : [];
      if (steps.length) {
        const box = document.createElement("details"); box.className = "sl-hint"; box.style.margin = "2px 0";
        const sum = document.createElement("summary"); sum.textContent = "How these fiber tracts were made"; sum.style.cursor = "pointer"; box.append(sum);
        for (const line of steps) { const d = document.createElement("div"); d.textContent = `· ${line}`; d.style.paddingLeft = "8px"; box.append(d); }
        const more = document.createElement("a"); more.href = "#"; more.textContent = "Details and references: Help";
        more.onclick = (e) => { e.preventDefault(); let p: HTMLElement | null = root ?? null; let help: HTMLDetailsElement | null = null; while (p && !help) { help = p.querySelector("details.sl-help"); p = p.parentElement; } if (help) { help.open = true; help.scrollIntoView({ block: "start", behavior: "smooth" }); } };
        box.append(more);
        box.open = howOpen; box.addEventListener("toggle", () => { howOpen = box.open; });
        box.title = `Made when the scan came into the database${st?.made ? `, ${new Date(st.made).toLocaleString()}` : ""}. Rules: ${Object.entries(st?.rules ?? {}).filter(([k]) => !["code", "labeler", "synthstrip", "core", "extension"].includes(k)).map(([k, v]) => `${k} ${v}`).join(", ")}. Code: Albula ${String(st?.rules?.core ?? "?")}, diffusion extension ${String(st?.rules?.extension ?? "?")}.`;
        top.append(box);
      } else {
        const tl = document.createElement("div"); tl.className = "sl-hint";
        const madeHere = whole.get(scan.browserId);
        tl.textContent = madeHere && !madeHere.stored ? `Fiber tracts: made here from this scan (${madeHere.c ? `tracking rule ${madeHere.c.rule}` : "this module's settings"})`
          : jobBusyNow ? "Fiber tracts: being prepared in the background now" : st?.state === "none" ? "Fiber tracts: not made yet; the first choice below makes them (a minute or more)" : "";
        if (tl.textContent) top.append(tl);
      }
    }
    root.append(top);

    // ── THE SCANS, only when something is missing (Ron's v3) ─────────────────────────────────────────────────────────
    if (!scan || !anat) {
      const sc = shell.section(root, "The scans", { band: "yellow", open: true, note: `${[scan, anat].filter((x) => !x).length} missing` });
      const need = (ok: boolean, what: string, detail: string, tip: string) => {
        const r = document.createElement("div"); r.style.cssText = "display:grid;grid-template-columns:16px 1fr;gap:6px;margin:3px 0"; r.title = tip;
        const m = document.createElement("span"); m.textContent = ok ? "✓" : "○"; m.style.color = ok ? "var(--sl-ok)" : "var(--sl-warn)";
        const t = document.createElement("span"); t.innerHTML = `<b>${what}</b>`;
        const d = document.createElement("span"); d.className = "sl-hint"; d.textContent = ` — ${detail}`; t.append(d);
        r.append(m, t); sc.append(r);
      };
      need(!!scan, "Diffusion MRI", scan ? scan.name.replace(/^.*?·\s*(MR\s+)?/, "") : "not loaded; it is often named DTI, DWI or diffusion", "The scan that shows the brain's wiring.");
      need(!!anat, "MRI of the anatomy (T1)", anat ? String(anat.name ?? "").replace(/^.*?·\s*(MR\s+)?/, "") : "not loaded", "The scan the tracts are lined up with, and a tumor is outlined on.");
      const bar0 = shell.actions(sc);
      const disk = document.createElement("button"); disk.textContent = "Scans on this computer…";
      disk.title = "Scans in a folder (from the scanner, a CD or a USB stick): opens Load / Save, where a folder is added to the database.";
      disk.onclick = () => { if (!openLoadFromDisk()) say("Load / Save cannot be opened from here."); };
      const open = document.createElement("button"); open.className = "sl-primary"; open.textContent = "Open from the database…";
      open.title = "Opens the DICOM database. Tick the diffusion MRI and the MRI of the anatomy (and a tumor outline, if there is one), and load them.";
      open.onclick = () => { if (!openDicomDatabase()) say("The DICOM database cannot be opened from here; use Load / Save."); };
      bar0.append(disk, open);
    }

    // ── CHOOSE TRACTS ───────────────────────────────────────────────────────────────────────────────────────────────
    const ch = shell.section(root, "Choose tracts", { band: "yellow", open: true });
    const storedHere = !!scan && storedState.get(scan.browserId)?.state === "stored";
    if (!scan || (!anat && !storedHere)) {
      const p = document.createElement("p"); p.className = "sl-hint"; p.textContent = !scan ? "Waits for the diffusion MRI." : "Waits for the MRI of the anatomy (T1)."; ch.append(p);
    } else {
      // THE WAY, a pull-down whose triangle is seen at arm's length (Ron, 2026-10-14; PALETTE.md: the full ▼, ~14 px, accent).
      const wrap = document.createElement("span"); wrap.style.cssText = "position:relative;display:inline-flex;align-items:center";
      const ws = document.createElement("select");
      for (const [k, label] of [["near", "Near a structure"], ["name", "By name"], ["box", "Through a place"]] as const) ws.append(new Option(label, k, false, k === way));
      ws.style.cssText = "appearance:none;-webkit-appearance:none;padding-right:24px;font-weight:600";
      ws.title = "How to choose the tracts: those that pass near a structure (a tumor, any segment, another tract), by their names, or those through a box you place.";
      ws.onchange = () => setWay(ws.value as Way);
      const tri = document.createElement("span"); tri.textContent = "▼"; tri.setAttribute("aria-hidden", "true");
      tri.style.cssText = "position:absolute;right:6px;pointer-events:none;font-size:14px;color:var(--sl-accent)";
      wrap.append(ws, tri);
      shell.row(ch, "Choose").append(wrap);

      if (way === "near") {
        const ns = document.createElement("select");
        if (!segs.length) ns.append(new Option("no segment of this patient", ""));
        for (const s of segs) ns.append(new Option(s.label, s.key, false, s.key === near && !wantOutline));
        // THE TRACT THAT IS THE STRUCTURE NOW, after "Tracts near this one…" (finding 18).
        const nt = nearTract.get(scan.browserId);
        if (nt) ns.append(new Option(nt.label, "__tract__", false, !wantOutline));
        if (anat) ns.append(new Option("Outline one here…", "__outline__", false, wantOutline));
        ns.title = "The structure the tracts are measured from: a tumor, any segment of this patient, or one you outline here. A tract as the structure: its ⋯ menu, Tracts near this one.";
        ns.onchange = () => { if (ns.value === "__tract__") { wantOutline = false; render(); return; } nearTract.delete(scan.browserId); if (ns.value === "__outline__") { wantOutline = true; } else { wantOutline = false; near = ns.value; } render(); };
        shell.row(ch, "Near").append(ns);
        const wm = document.createElement("select");
        for (const v of [2, 4, 6, 8, 10, 15, 20]) wm.append(new Option(`${v} mm`, String(v), false, v === withinMm));
        if (![2, 4, 6, 8, 10, 15, 20].includes(withinMm)) wm.append(new Option(`${withinMm} mm`, String(withinMm), false, true));
        wm.title = `How close a tract must come to the structure to be listed. Tracts within ${GRAY_BAND_MM} mm more are listed in gray, hidden.`;
        // A run already made from this structure is measured again at once: nothing to press (rule 3).
        wm.onchange = () => {
          withinMm = Number(wm.value);
          const r = runs.get(scan.browserId), nt2 = nearTract.get(scan.browserId);
          if (r && !busy && nt2 && r.label === nt2.label) void tractsNearTract(scan, nt2.strands, nt2.label);
          else if (r && !busy && segmentChoices(scan).find((c) => c.key === near)?.label === r.label) void makeTracts();
          else render();
        };
        shell.row(ch, "Within").append(wm);
        // OUTLINE ONE HERE: strokes in, strokes around, grow, correct, done (the resident's steps, unchanged).
        if (anat && (wantOutline || outline)) {
          const stepsBox = document.createElement("div");
          stepsBox.style.cssText = "display:flex;flex-direction:column;gap:4px;margin:6px 0 2px";
          const step = (n: string, label: string, tip: string, on: boolean, disabled: boolean, act: () => void, words: string) => {
            const r = document.createElement("div"); r.style.cssText = "display:flex;align-items:center;gap:6px";
            const k = document.createElement("span"); k.className = "sl-hint"; k.textContent = n; k.style.flex = "0 0 1em";
            const b = document.createElement("button"); b.textContent = label; b.title = tip; b.disabled = disabled || !!busy;
            if (on) b.className = "sl-primary";
            b.onclick = act;
            const w = document.createElement("span"); w.className = "sl-hint"; w.textContent = words;
            r.append(k, b, w); stepsBox.append(r);
          };
          step("1", "Inside", "Draw a stroke inside the structure on several slices, from where it starts to where it ends. Click again to stop drawing.", outline?.tool === 1, false, () => void strokes(anat, 1), "strokes inside it");
          step("2", "Around", "Draw strokes around it, on several slices and just above and below it, so the outline knows where to stop.", outline?.tool === 2, false, () => void strokes(anat, 2), "strokes around it");
          step("3", busy === "Growing…" ? "Growing…" : "Grow the outline", "Albula fills the structure out from your strokes up to where the image changes. Check every slice it touches; add strokes where it is wrong and grow again.", false, !outline, () => void growOutline(), outline?.mm3 ? `${(outline.mm3 / 1000).toFixed(1)} mL` : "");
          step("4", "Done", "Keeps the outline and removes the strokes.", false, !outline?.resultId, () => { outlineDone(); wantOutline = false; }, "");
          step("", "Cancel", "Removes the strokes and the outline grown from them; nothing is kept.", false, false, () => { if (outline) outlineCancel(); wantOutline = false; render(); }, "");
          ch.append(stepsBox);
        }
        if (kept && kept.savable && live.nodes.get(kept.segId) && !outline) {
          const bar1 = shell.actions(ch);
          const sv = document.createElement("button"); sv.className = kept.saved ? "" : "sl-primary";
          sv.textContent = kept.saved ? "Saved ✓" : "Save to DICOM"; sv.disabled = kept.saved;
          sv.title = "Write the outline into the DICOM database, under the MRI it was drawn on, so it is there next time.";
          sv.onclick = () => { void runAction(sv, () => saveKept(), { busyLabel: "Saving…", doneLabel: "Saved ✓", failedLabel: "Not saved" }).catch(() => {}); };
          bar1.append(sv);
        }
        // THE SERVER, WHERE THE PERSON IS: start it, or restart a stuck one (brain-mask.ts).
        const cNow = computed.get(scan.browserId);
        if (cNow?.brain.reason === "no-server" || cNow?.brain.reason === "stuck") {
          const stuck = cNow.brain.reason === "stuck";
          const st = document.createElement("button"); st.style.cssText = "width:100%;margin:4px 0"; st.disabled = !!busy;
          st.textContent = stuck ? "Restart the segmentation server" : "Start the segmentation server";
          st.title = stuck ? "The program that finds the brain on the MRI of the anatomy has stopped making progress. This stops it and starts it again (a minute or two). Anything it was doing stops, and the line at the top of the window names it." : "Starts the program that finds the brain on the MRI of the anatomy (the first start takes a minute or two).";
          st.onclick = () => { void runAction(st, async () => { const r = await (stuck ? restartSegmentationServer : startSegmentationServer)((l) => say(l)); say(r.ok ? `${r.message[0].toUpperCase()}${r.message.slice(1)}. Press the button below again.` : `The segmentation server did not ${stuck ? "restart" : "start"}: ${r.message}.`); if (r.ok) cNow.brain.reason = undefined; render(); }, { busyLabel: stuck ? "Restarting…" : "Starting…", doneLabel: stuck ? "Restarted" : "Started", failedLabel: "Did not work" }).catch(() => {}); };
          ch.append(st);
        }
        // THE ONE YELLOW ACTION, last row, right (PALETTE.md).
        const ntGo = nearTract.get(scan.browserId);
        const target = ntGo ? undefined : segs.find((s) => s.key === near);
        const isTumor = !!target && tumors.some((t) => t.key === target.key);
        const bar = shell.actions(ch);
        const go = document.createElement("button"); go.className = "sl-primary";
        go.textContent = busy && !adding && !busy.startsWith("Grow") ? busy : isTumor ? "Show the fiber tracts near the tumor" : "Show the tracts near it";
        go.title = `Shows each named tract with at least ${MIN_NEAR_STREAMLINES} of its lines within ${withinMm} mm of ${ntGo?.label ?? target?.label ?? "the structure"} — whole, in its own color, with how close it comes. Tracts that come within ${GRAY_BAND_MM} mm more, or close with fewer lines, are listed after them in gray, hidden.`;
        go.disabled = !!busy || (!target && !ntGo) || seeding || !!outline || wantOutline;
        // SINGLE TENSOR NAMES NO TRACTS (finding 9): said, not silently swapped; its own tracts start under For developers.
        go.onclick = () => {
          if (method === "single") { method = "ukf"; say("Single tensor names no tracts, so this list uses two-tensor tracking. Single-tensor tracts start under Advanced › For developers."); }
          if (ntGo) void tractsNearTract(scan, ntGo.strands, ntGo.label); else void makeTracts();
        };
        bar.append(go);
      } else if (way === "name") {
        // BY NAME: the atlas's tracts in their groups, a search, a tick each.
        const model = tractCountModel;
        if (!model) { const p = document.createElement("p"); p.className = "sl-hint"; p.textContent = "Reading the list of tracts…"; ch.append(p); void tractCloud().then((m) => { tractCountModel = m; render(); }).catch(() => {}); }
        else {
          const find = document.createElement("input"); find.type = "search"; find.placeholder = "Find a tract, e.g. \"uncinate\" or \"UF\""; find.value = tractSearch; find.className = "sl-tract-find"; find.style.cssText = "width:100%;margin:2px 0 4px";
          find.title = "Shows only the tracts whose name, abbreviation or anatomical term contains this.";
          find.oninput = () => { tractSearch = find.value; const at = find.selectionStart; render(); const f2 = root?.querySelector<HTMLInputElement>(".sl-tract-find"); f2?.focus(); if (at !== null) f2?.setSelectionRange(at, at); };
          ch.append(find);
          const OTHER = model.json.tracts.length - 1;
          const shownT = new Set(groups.filter((g) => g.scan === scan.browserId && g.visible && g.tract !== undefined).map((g) => g.tract!));
          const box = document.createElement("div"); box.style.cssText = "border:1px solid var(--sl-border);border-radius:var(--sl-radius);padding:2px 6px";
          // The mockup's order: association, projection, commissural, then the rest.
          const order = ["association", "projection", "cc", "cerebellar", "superficial"];
          for (const grp of [...TRACT_GROUPS].filter((g) => g.key !== "unnamed").sort((a, b) => order.indexOf(a.key) - order.indexOf(b.key))) {
            const members = model.json.tracts.map((t, i) => ({ t, i })).filter(({ t, i }) => i !== OTHER && tractGroupKey(tractInfo(t.abbr)?.category, false) === grp.key && matchesSearch(tractSearch, tractLabel(t.abbr, t.name, 0), tractInfo(t.abbr)?.tna?.latin));
            if (!members.length) continue;
            const h = document.createElement("div"); h.textContent = grp.label; h.className = "sl-hint"; h.style.cssText = "font-weight:600;margin:4px 0 1px"; box.append(h);
            for (const { t, i } of members) {
              const lab = document.createElement("label"); lab.style.cssText = "display:flex;align-items:center;gap:6px;padding:1px 0 1px 8px";
              const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = shownT.has(i); cb.disabled = !!busy;
              cb.onchange = () => { void showByName(scan, i, cb.checked); };
              const c = tractColor(i, tractCount);
              const sw = document.createElement("span"); sw.style.cssText = `flex:0 0 10px;width:10px;height:10px;border:1px solid var(--sl-border);background:rgb(${c.slice(0, 3).map((v) => Math.round(v * 255)).join(",")})`;
              const nm = document.createElement("span"); nm.textContent = tractLabel(t.abbr, t.name, 0);
              const tna = tnaLine(t.abbr); lab.title = `${tractLabel(t.abbr, t.name, 0)}${tna ? `\n${tna}` : ""}`;
              lab.append(cb, sw, nm); box.append(lab);
            }
          }
          fitList(box, "diffusion-tract-names");
          ch.append(box);
          if (busy) { const p = document.createElement("p"); p.className = "sl-hint"; p.textContent = busy; ch.append(p); }
        }
      } else {
        // THROUGH A PLACE: one box.
        const bId = boxes.get(scan.browserId), has = !!(bId && live.nodes.get(bId)), bLine = boxLines.get(scan.browserId);
        if (bLine && has) { const p = document.createElement("p"); p.className = "sl-hint"; p.style.margin = "4px 0"; p.textContent = bLine; ch.append(p); }
        const bar = shell.actions(ch);
        const pb = document.createElement("button"); pb.className = has ? "" : "sl-primary";
        pb.textContent = has ? "Put the box at the crossing again" : "Put a box";
        pb.title = "A box at the point where the slice views cross; move and resize it with its handles in any view. The lines through it are shown as it moves.";
        pb.disabled = !!busy;
        pb.onclick = () => putBox(scan);
        bar.append(pb);
      }
    }
    if (note && noteFor === chosen) { const p = document.createElement("p"); p.className = "sl-hint"; p.style.margin = "4px 0 0"; p.textContent = note; ch.append(p); }

    // ── SHOWN: one row per tract, Left and Right side by side ─────────────────────────────────────────────────────────
    const mineAll = scan ? groups.filter((g) => g.scan === scan.browserId) : [];
    if (scan && mineAll.length) {
      const run = runs.get(scan.browserId);
      type Row = { key: string; tract?: number; left?: TractGroup; right?: TractGroup; mid?: TractGroup; other?: TractGroup };
      const rows: Row[] = [], byKey = new Map<string, Row>();
      for (const g of mineAll) {
        const key = g.tract !== undefined ? `t${g.tract}` : `g${g.id}`;
        let r = byKey.get(key); if (!r) { r = { key, tract: g.tract }; byKey.set(key, r); rows.push(r); }
        if (g.tract === undefined) r.other = g; else if (g.side === -1) r.left = g; else if (g.side === 1) r.right = g; else r.mid = g;
      }
      const gs = (r: Row) => [r.left, r.right, r.mid, r.other].filter((x): x is TractGroup => !!x);
      // A side emptied by hand no longer counts as near (critic, finding 7).
      const near0 = (r: Row) => Math.min(...gs(r).filter((g) => !g.compare && g.strands.length > 0).map((g) => g.distanceMm ?? Infinity));
      const faintRow = (r: Row) => { const own = gs(r).filter((g) => !g.compare); return own.length > 0 && own.every((g) => g.faint); };
      rows.sort((a, b) => Number(faintRow(a)) - Number(faintRow(b)) || (a.other ? 1 : 0) - (b.other ? 1 : 0) || near0(a) - near0(b));
      const listed = rows.filter((r) => !faintRow(r) && !r.other).length;
      const sh = shell.section(root, "Shown", { band: "green", open: true, note: run ? `near ${run.label}: ${listed}` : `${listed} tract${listed === 1 ? "" : "s"}` });
      const hasDist = mineAll.some((g) => g.distanceMm !== undefined);
      // AS THE MOCKUP HAS IT: near a structure, closest and the lines per side (FA in the tooltip); otherwise lines and FA
      // per side. Five number columns did not leave the name room in the module column.
      const cols = hasDist ? "18px 14px minmax(0,1fr) 50px 52px 52px 22px" : "18px 14px minmax(0,1fr) 52px 34px 52px 34px 22px";
      const head = document.createElement("div"); head.className = "sl-hint";
      head.style.cssText = `display:grid;grid-template-columns:${cols};gap:4px;align-items:end;font-size:11px;margin:2px 0`;
      head.innerHTML = hasDist
        ? `<span></span><span></span><span></span><span style="text-align:right">closest</span><span style="text-align:right">Left</span><span style="text-align:right">Right</span><span></span>`
        : `<span></span><span></span><span></span><span style="grid-column:span 2;text-align:center">Left: lines · FA</span><span style="grid-column:span 2;text-align:center">Right: lines · FA</span><span></span>`;
      // NOTHING UNDER THE LIST (Ron, 2026-10-15: "anything under the list is effectively hidden"; MODULE-DESIGN-RULES.md):
      // the export, the caveat and what the tracts were followed in sit above it; Add lines is in each row's ⋯ (the
      // mockup's place), its progress said here while it runs.
      const bar = document.createElement("div"); bar.style.cssText = "display:flex;gap:6px;align-items:center;justify-content:flex-end;flex-wrap:wrap;margin:2px 0 4px";
      if (adding && busy) { const pr = document.createElement("span"); pr.className = "sl-hint"; pr.style.marginRight = "auto"; pr.textContent = busy.replace("Making tracts", "Adding lines"); bar.append(pr); }
      const exported = exportedFor.get(scan.browserId);
      if (exported) {
        const fb = document.createElement("button"); fb.textContent = "Show in Finder"; fb.title = exported;
        fb.onclick = () => showInDownloads(exported); bar.append(fb);
      }
      const ex = document.createElement("button"); ex.textContent = "Export the numbers";
      ex.title = "One table file at the top of Downloads, a row per tract and side: lines, FA, MD, AD, RD, volume, length, and what made them.";
      ex.onclick = () => { void runAction(ex, () => exportNumbers(scan), { busyLabel: "Exporting…", doneLabel: "Exported ✓", failedLabel: "Not exported" }).catch(() => {}); };
      bar.append(ex);
      sh.append(bar);
      // THE CAVEAT (Ron, 2026-10-01; O'Donnell et al. 2017), with a list measured near a tumor -- above it, to be read.
      if (run && tumors.some((t) => t.label === run.label)) {
        const cav = document.createElement("p"); cav.className = "sl-hint"; cav.style.margin = "0 0 4px";
        cav.textContent = "A tract that looks thin or missing on the tumor's side may be destroyed by the tumor, or still there but hidden by swelling (edema). Compare it with the other side; ⋯ › Add lines gives a closer look.";
        sh.append(cav);
      }
      if (run?.brain) { const bl = document.createElement("p"); bl.className = "sl-hint"; bl.style.margin = "0 0 4px"; bl.textContent = `${run.brain[0].toUpperCase()}${run.brain.slice(1)}.`; sh.append(bl); }
      // ALL ON OR OFF (Ron, 2026-10-16: "where is the toggle for turn all on/off?"): the eye in the header, above the list
      // so it never scrolls away. Off: every row hidden; on: the listed rows shown again (not the gray ones or the
      // comparison sides, which stay as the person left them hidden).
      {
        const allOn = mineAll.some((g) => g.visible);
        const he = document.createElement("button"); he.textContent = allOn ? "👁" : "🚫";
        he.style.cssText = "background:none;border:none;padding:0;cursor:pointer;color:inherit;font-size:14px;min-width:0;width:18px;height:18px";
        he.title = allOn ? "Hide every tract in the list" : "Show every listed tract"; he.setAttribute("aria-label", he.title);
        // Off remembers what was shown; on puts back exactly that (or, with nothing remembered, the listed tracts -- not
        // the gray ones, the comparison sides or "Rest of the brain", which start hidden).
        he.onclick = () => {
          if (allOn) { allOffMemory.set(scan.browserId, new Map(mineAll.map((g) => [g, g.visible]))); for (const g of mineAll) g.visible = false; }
          else { const m = allOffMemory.get(scan.browserId); for (const g of mineAll) g.visible = m?.has(g) ? m.get(g)! : !g.faint && !g.compare && !(g.run && g.name === "Rest of the brain"); allOffMemory.delete(scan.browserId); }
          redraw3d(); render();
        };
        head.firstElementChild?.replaceWith(he);
      }
      sh.append(head);
      const box = document.createElement("div");
      const f2 = (x?: number) => x === undefined ? "–" : x.toFixed(2);
      const mdTxt = (x?: number) => x === undefined ? "–" : (x * 1000).toFixed(2);
      for (const r of rows) {
        const row = document.createElement("div");
        row.style.cssText = `display:grid;grid-template-columns:${cols};gap:4px;align-items:center;padding:1px 0`;
        const all = gs(r), anyOn = all.some((g) => g.visible);
        const eye = document.createElement("button"); eye.textContent = anyOn ? "👁" : "🚫";
        eye.style.cssText = "background:none;border:none;padding:0;cursor:pointer;color:inherit;font-size:14px;min-width:0;width:18px;height:18px";   // a glyph button: not the 64 px of a button with words (PALETTE.md)
        eye.onclick = () => { for (const g of all) g.visible = !anyOn; redraw3d(); render(); };
        // ITS COLOR, AND THE CHOICE OF ANOTHER (Ron, 2026-10-16: "How do I give it a solid color?"): the square shows the
        // row's color -- by direction as a gradient -- and opens the choice.
        const chosenC = all[0].color;
        const byDir = colorBy === "direction" || chosenC === "direction" || (!chosenC && r.tract === undefined && !all[0].unnamed);
        const c = chosenC && chosenC !== "direction" ? chosenC : r.tract !== undefined ? tractColor(r.tract, tractCount) : UNNAMED;
        const sw = document.createElement("button"); sw.type = "button";
        sw.style.cssText = `width:14px;height:14px;padding:0;min-width:0;border:1px solid var(--sl-control-edge);border-radius:2px;cursor:pointer;background:${byDir ? "linear-gradient(135deg,#d33,#3b3,#33d)" : `rgb(${c.slice(0, 3).map((v) => Math.round(v * 255)).join(",")})`}`;
        sw.title = byDir ? "Colored by direction: click to choose a color" : "Its color: click to choose another";
        sw.onclick = (e) => { e.stopPropagation(); colorChoice(sw, all, r.tract !== undefined); };
        const nm = document.createElement("span"); nm.style.cssText = "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
        const base = r.other ? nameOf(r.other) : (all[0].label ?? tractTitle(r.tract!));
        eye.title = `${anyOn ? "Hide" : "Show"} ${base}`; eye.setAttribute("aria-label", eye.title);
        // THE ABBREVIATION FIRST (finding 16: five rows read "Corpus callosum …"); the full name in the tooltip.
        const tAbbr = r.tract !== undefined && !all[0].label ? tractCountModel?.json.tracts[r.tract]?.abbr : undefined;
        if (tAbbr) { const b = document.createElement("b"); b.textContent = tAbbr; nm.append(b, ` ${base.replace(/ \([^)]*\)$/, "")}`); } else nm.textContent = base;
        makeNameEditable(nm, { get: () => base, commit: (v) => { for (const g of all) g.label = v; render(); } });
        if (faintRow(r)) { nm.style.opacity = "0.55"; }
        row.append(eye, sw, nm);
        if (hasDist) {
          const d = near0(r), ds = document.createElement("span"); ds.style.cssText = "font-size:12px;opacity:0.85;text-align:right";
          ds.textContent = !Number.isFinite(d) ? "" : d < 0.05 ? "touches" : `${d.toFixed(1)} mm`; row.append(ds);
        }
        // LEFT AND RIGHT EACH WITH ITS OWN EYE (Ron, 2026-10-16: "I would like to have the left and right sides separate.
        // Or at least the ability to turn them on/off separately"); the row's eye still turns both.
        const cell = (g?: TractGroup, side?: string) => {
          const a = document.createElement("span"), b = document.createElement("span");
          a.style.cssText = b.style.cssText = "font-size:12px;text-align:right;font-variant-numeric:tabular-nums";
          a.style.cssText += ";display:flex;align-items:center;justify-content:flex-end;gap:2px";
          if (g) {
            const se = document.createElement("button"); se.textContent = g.visible ? "👁" : "🚫";
            se.style.cssText = "background:none;border:none;padding:0;cursor:pointer;color:inherit;font-size:12px;min-width:0;width:16px;height:16px";
            se.title = `${g.visible ? "Hide" : "Show"} the ${side} side`; se.setAttribute("aria-label", se.title);
            se.onclick = (e) => { e.stopPropagation(); g.visible = !g.visible; redraw3d(); render(); };
            const n = document.createElement("span"); n.textContent = g.strands.length.toLocaleString();
            a.append(se, n);
          } else a.textContent = "–";
          b.textContent = g ? f2(measured.get(g)?.m.fa) : "–";
          if (g && !g.visible) { a.style.opacity = b.style.opacity = "0.55"; }
          return hasDist ? [a] : [a, b];
        };
        if (r.left || r.right) row.append(...cell(r.left, "left"), ...cell(r.right, "right"));
        else { const one = r.mid ?? r.other!; const s = document.createElement("span"); s.style.cssText = `grid-column:span ${hasDist ? 2 : 4};font-size:12px;text-align:center;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis`; s.textContent = `${one.strands.length.toLocaleString()} lines${measured.get(one)?.m.fa !== undefined ? ` · FA ${f2(measured.get(one)!.m.fa)}` : ""}${r.mid ? " (across the midline)" : ""}`; row.append(s); }
        // THE REST IN THE TOOLTIP (Ron's v3): per side MD, AD, RD, volume, length; how lopsided (left vs right).
        const tip: string[] = [base];
        for (const [side, g] of [["Left", r.left], ["Right", r.right], ["", r.mid ?? r.other]] as const) {
          const m = g ? measured.get(g)?.m : undefined; if (!g || !m) continue;
          tip.push(`${side ? `${side}: ` : ""}${m.lines.toLocaleString()} lines, FA ${f2(m.fa)}, MD ${mdTxt(m.md)}, AD ${mdTxt(m.ad)}, RD ${mdTxt(m.rd)} (10⁻³ mm²/s), volume ${m.volumeMl.toFixed(1)} mL, median length ${m.lengthMm.toFixed(0)} mm`);
        }
        const ml = r.left ? measured.get(r.left)?.m : undefined, mr = r.right ? measured.get(r.right)?.m : undefined;
        if ([r.left, r.right, r.mid, r.other].some((g) => g && measured.get(g) && measured.get(g)!.m.md === undefined)) tip.push("MD, AD and RD: not made yet for this scan; Albula adds them in the background.");
        if (ml && mr) {
          const li = (a?: number, b?: number) => { const v = laterality(a, b); return v === undefined ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(2)}`; };
          tip.push(`How lopsided, left against right (above 0: more on the left): FA ${li(ml.fa, mr.fa)}, MD ${li(ml.md, mr.md)}, volume ${li(ml.volumeMl, mr.volumeMl)}. The number of lines depends on the tracking, not the brain, so it is not compared (Jones et al. 2013).`);
        }
        if (r.tract !== undefined) { const t = tractCountModel?.json.tracts[r.tract]; const tna = t ? tnaLine(t.abbr) : ""; if (tna) tip.push(tna); }
        if (faintRow(r)) tip.push("Listed in gray and hidden: it comes that close with fewer lines, or only within the gray band. Show it to see how it passes.");
        row.title = tip.join("\n");
        // ⋯: Rename, Remove, Tracts near this one…, Add lines, About this tract.
        addRowMenu(row, () => {
          const acts: RowAction[] = [
            { label: "Rename", run: () => startEditing(nm, base, (v) => { for (const g of all) g.label = v; render(); }) },
            { label: "Remove", run: () => { for (const g of all) groups.splice(groups.indexOf(g), 1); redraw3d(); render(); } },
            { label: "Remove stray lines…", title: "Drag a rectangle around lines to take them out of this tract, in the 3D view or a slice; each removal can be undone.", run: () => { void removeStray(all, base); } },
          ];
          if (undoStack.some((u) => all.includes(u.g))) acts.push({ label: "Undo the last removal", title: "Puts back the lines the last rectangle took out of this tract.", run: () => undoRemoval(all) });
          if (r.tract !== undefined) acts.push({ label: "Tracts near this one…", title: "Lists the tracts that pass near this one, as near a structure.", run: () => { const strands = all.flatMap((g) => g.strands); nearTract.set(scan.browserId, { strands, label: base }); if (wb) { wb.way = "near"; wb.touched = true; } void tractsNearTract(scan, strands, base); } });
          if (all.some((g) => g.run) && r.tract !== undefined) acts.push({ label: "Add lines", title: "Follows many more lines in this tract, on both sides.", run: () => { void addLines(all.filter((g) => g.run)); } });
          if (r.tract !== undefined) { const t = tractCountModel?.json.tracts[r.tract]; if (t) { const ref = readMore(t.abbr); acts.push({ label: "About this tract", title: ref.cite, run: () => { void fetch(`/_open?url=${encodeURIComponent(ref.link)}`).then((x) => { if (!x.ok) throw new Error(); }).catch(() => { globalThis.open?.(ref.link, "_blank"); }); } }); } }
          return acts;
        }, base);
        box.append(row);
      }
      fitList(box, "diffusion-shown", { min: 150 });   // about six rows at least
      sh.append(box);
      void ensureMeasures(scan);
    }
    if (!scan) return;

    // ── ADVANCED ─────────────────────────────────────────────────────────────────────────────────────────────────────
    const more = shell.section(root, "Advanced", { band: "none", open: moreOpen });
    (more.closest("details") as HTMLDetailsElement | null)?.addEventListener("toggle", (e) => { moreOpen = (e.target as HTMLDetailsElement).open; });
    const showing = shownNow(scan);
    const maps = shell.section(more, "Maps", { band: "3d", open: true, note: `${scan.frameIds.length} volumes · b ${[...new Set(scan.bValues.map((b) => Math.round(b)))].sort((a, b) => a - b).join("/")}` });
    const seg = document.createElement("div"); seg.style.cssText = "display:flex;gap:3px";
    for (const [k, label, tip] of [["b0", "b0 (corrected)", "The diffusion MRI's image without diffusion weighting, corrected and lined up with the MRI of the anatomy: the image the maps and the tracts are made from."], ["fa", "FA", "Fractional anisotropy: bright where water moves along one direction, as in white matter tracts."], ["colorfa", "Color FA", "FA colored by direction: red left-right, green front-back, blue up-down."]] as const) {
      const b = document.createElement("button"); b.textContent = label; b.title = tip; b.disabled = !!busy;
      b.className = "sl-sh-look-b" + (showing === k ? " sl-on" : ""); b.setAttribute("aria-pressed", String(showing === k));
      b.onclick = () => { void showMap(scan, k); };
      seg.append(b);
    }
    shell.row(maps, "Show").append(seg);
    const acq = document.createElement("label"); acq.style.cssText = "display:flex;align-items:center;gap:6px";
    const acb = document.createElement("input"); acb.type = "checkbox"; acb.checked = live.nodes.get(scan.browserId)?.transport === "shown"; acb.disabled = !!busy;
    acb.onchange = () => {
      live.write({ op: "patch", id: scan.browserId, path: "#/transport", value: acb.checked ? "shown" : "hidden" });
      if (acb.checked) { const { frames, selected } = browserFrames(live, scan.browserId); putBackground(frames[selected]?.node ?? scan.frameIds[0], null); render(); }
      else void showMap(scan, "b0");
    };
    acq.append(acb, "Step through the scan as acquired");
    acq.title = "Shows the diffusion MRI as the scanner wrote it, one volume at a time, with the player in the top bar to step through them. Not corrected and not lined up with the MRI of the anatomy.";
    shell.row(maps, "Volumes").append(acq);
    const partner = partnerFor(scan), c0 = computed.get(scan.browserId);
    const dist = document.createElement("label"); dist.style.cssText = "display:flex;align-items:center;gap:6px";
    const dcb = document.createElement("input"); dcb.type = "checkbox"; dcb.checked = correct && !!partner; dcb.disabled = !partner || !!busy;
    dcb.onchange = () => { correct = dcb.checked; const was = shownNow(scan); const c = computed.get(scan.browserId); if (c) { dropMaps(c); computed.delete(scan.browserId); } if (was) void showMap(scan, was); else render(); };
    const short = (n: string) => n.replace(/^.*?·\s*(MR\s+)?/, "");
    dist.append(dcb, partner ? `Correct with ${short(partner.name)}` : "Not corrected: no reversed phase-encoding scan loaded");
    dist.title = c0 ? `This scan's maps and tracts: distortion ${c0.corrected}.` : "Corrects the scan's stretching along its phase-encoding direction, using the scan taken with the opposite direction.";
    shell.row(maps, "Distortion").append(dist);
    if (MOTION_RULE !== 0) {
      const mv = document.createElement("label"); mv.style.cssText = "display:flex;align-items:center;gap:6px";
      const mcb = document.createElement("input"); mcb.type = "checkbox"; mcb.checked = preCorrected.has(scan.browserId); mcb.disabled = !!busy;
      mcb.onchange = () => { if (mcb.checked) preCorrected.add(scan.browserId); else preCorrected.delete(scan.browserId); const was = shownNow(scan); const c = computed.get(scan.browserId); if (c) { dropMaps(c); computed.delete(scan.browserId); } if (was) void showMap(scan, was); else render(); };
      mv.append(mcb, "Already corrected (a preprocessed scan)");
      mv.title = c0?.movement ? `This scan: ${c0.movement}.` : "Tick when this scan was corrected for head movement before it was loaded, so it is not corrected twice.";
      shell.row(maps, "Head movement").append(mv);
    }
    const ck = checks.get(scan.browserId), cRow = document.createElement("span"); cRow.style.cssText = "font-size:12px";
    if (!ck) cRow.textContent = "not yet (runs when the maps are first made)"; else if ("running" in ck) cRow.textContent = "dcm2niix is reading the same files…"; else if ("error" in ck) cRow.textContent = `not checked: ${ck.error}`; else { cRow.textContent = ck.result.agree ? `✓ ${ck.result.said}` : `⚠ ${ck.result.said}`; if (!ck.result.agree) cRow.style.color = "var(--sl-warn)"; }
    cRow.title = `Albula reads the scan's diffusion values with its own reader; dcm2niix ${DCM2NIIX_VERSION} reads the same files, and the two are compared.`;
    shell.row(maps, "Checked").append(cRow);
    // DRAWING (Ron, 2026-10-15: "yes, keep them in the module under Advanced too").
    const draw = shell.section(more, "Drawing", { band: "3d", open: false });
    const as = document.createElement("div"); as.style.cssText = "display:flex;gap:3px";
    for (const [k, label, tip] of [["lines", "Lines", "Thin lines: many tracts stay readable."], ["tubes", "Tubes", "Round, lit tubes: depth is easier to see."]] as const) {
      const b = document.createElement("button"); b.textContent = label; b.title = tip; b.className = "sl-sh-look-b" + (drawAs === k ? " sl-on" : ""); b.setAttribute("aria-pressed", String(drawAs === k));
      b.onclick = () => { drawAs = k; redraw3d(); render(); }; as.append(b);
    }
    shell.row(draw, "Draw as").append(as);
    const cb2 = document.createElement("div"); cb2.style.cssText = "display:flex;gap:3px";
    for (const [k, label, tip] of [["tract", "Tract", "Each named tract in its own color; lines without a name in gray."], ["direction", "Direction", "Every piece by its direction: red left-right, green front-back, blue up-down."]] as const) {
      const b = document.createElement("button"); b.textContent = label; b.title = tip; b.className = "sl-sh-look-b" + (colorBy === k ? " sl-on" : ""); b.setAttribute("aria-pressed", String(colorBy === k));
      b.onclick = () => { colorBy = k; redraw3d(); render(); }; cb2.append(b);
    }
    shell.row(draw, "Color").append(cb2);
    const tw = document.createElement("span"); tw.style.cssText = "display:flex;gap:6px;align-items:center";
    const ti = document.createElement("input"); ti.type = "number"; ti.min = "0"; ti.max = "6"; ti.step = "0.5"; ti.value = String(trimVoxels); ti.style.width = "4em";
    const vmm = voxelMm(scan.browserId);
    const tl2 = document.createElement("span"); tl2.style.opacity = "0.7"; tl2.textContent = `voxels at each end (≈ ${(trimVoxels * vmm).toFixed(1)} mm on this ${vmm.toFixed(1)} mm grid)`;
    ti.title = "Draws every tract a little shorter at both ends, where the fibers fan out. The tracts themselves stay whole.";
    ti.onchange = () => { const v = Number(ti.value); trimVoxels = Number.isFinite(v) ? Math.max(0, Math.min(6, v)) : trimVoxels; redraw3d(); render(); };
    tw.append(ti, tl2);
    shell.row(draw, "Shorten ends").append(tw);
    // FOR DEVELOPERS (the tracking's numbers, the method, a seed by click): folded, out of the way (rule 4).
    const dev = shell.section(more, "For developers", { band: "none", open: advOpen });
    (dev.closest("details") as HTMLDetailsElement | null)?.addEventListener("toggle", (e) => { advOpen = (e.target as HTMLDetailsElement).open; });
    const meth = document.createElement("div"); meth.style.cssText = "display:flex;gap:3px";
    for (const [k, label, tip] of [["ukf", "Two-tensor", "Follows two crossing fiber directions, as the atlas the tract names come from was made. The default."], ["ptt", "Smooth curves", "Follows fiber directions as smooth curves (CSD and parallel transport tracking). About three minutes."], ["single", "Single tensor", "One direction per voxel: fast, but stops or turns where fibers cross. Names no tracts."]] as const) {
      const b = document.createElement("button"); b.textContent = label; b.title = tip; b.disabled = !!busy;
      b.className = "sl-sh-look-b" + (method === k ? " sl-on" : ""); b.setAttribute("aria-pressed", String(method === k));
      b.onclick = () => { method = k; render(); }; meth.append(b);
    }
    shell.row(dev, "Method").append(meth);
    const num = (label: string, tip: string, get: () => number, set: (v: number) => void, step: number, lo: number, hi: number) => {
      const i = document.createElement("input"); i.type = "number"; i.step = String(step); i.min = String(lo); i.max = String(hi); i.value = String(get()); i.title = tip; i.style.cssText = "flex:0 0 5em;width:5em;min-width:0";
      i.onchange = () => { const v = Number(i.value); if (Number.isFinite(v) && v >= lo && v <= hi) set(v); else { say(`${label}: between ${lo} and ${hi}.`); i.value = String(get()); } };
      shell.row(dev, label).append(i);
    };
    num("Stop below FA", "A tract ends where the white matter becomes this faint.", () => (method === "ukf" ? adv.ukfStopFA : adv.minFA), (v) => { if (method === "ukf") adv.ukfStopFA = v; else adv.minFA = v; }, 0.01, 0.01, 0.9);
    num("Largest turn (°)", "A tract ends where it would bend more sharply than this.", () => adv.maxAngleDeg, (v) => { adv.maxAngleDeg = v; }, 5, 5, 90);
    num("Step (voxels)", "How finely a tract follows the white matter: smaller is smoother and slower.", () => adv.stepVoxels, (v) => { adv.stepVoxels = v; }, 0.1, 0.1, 2);
    num("Highest b used", "Which images the maps and single-tensor tracts are made from: those up to this b-value. Changing it makes the maps again.", () => adv.maxB, (v) => { adv.maxB = v; const was = shownNow(scan); for (const c of computed.values()) dropMaps(c); computed.clear(); if (was) void showMap(scan, was); }, 100, 500, 5000);
    const bar9 = shell.actions(dev);
    if (method === "single") {
      const mk = document.createElement("button"); mk.textContent = busy.startsWith("Making") ? busy : "Make single-tensor tracts";
      mk.title = "Starts single-tensor tracts in the white matter inside and around the structure chosen under Near (they get no names).";
      mk.disabled = !!busy || !near || seeding; mk.onclick = () => { void makeTracts(); };
      bar9.append(mk);
    }
    const seedBtn = document.createElement("button"); seedBtn.textContent = seeding ? "Click in a view…" : "Seed where I click…";
    seedBtn.title = seeding ? "Waiting for a click in a view. Escape cancels." : "Start tracts at the next point you click in a slice or the 3D view.";
    seedBtn.disabled = !!busy; seedBtn.onclick = () => { if (seeding) { cancelSeeding?.(); say("Seeding cancelled."); } else seedWhereIClick(); };
    bar9.append(seedBtn);
  }
  let howOpen = false;

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
        text: `${g.label ?? g.name} · ${h.streamlines} fiber${h.streamlines === 1 ? "" : "s"} within 2 mm${h.closestMm < 0.5 ? "" : `, closest ${h.closestMm.toFixed(1)} mm`}`, source: "fiber tracts (Diffusion)" };
    });
  });

  shell.registerPanel({
    id: "diffusion",
    title: "Diffusion",
    groups: ["Display"],
    tip: "The brain's fiber tracts from a diffusion MRI: those near a tumor or any structure, by name, or through a box; their numbers, left and right.",
    help: "<p><b>What it shows.</b> The brain's main fiber tracts, found in a diffusion MRI (often named DTI, DWI or diffusion) and lined up with the MRI of the anatomy (the T1). When a diffusion MRI comes into the database, Albula finds and names the whole brain's tracts once, in the background, so choosing them here takes about a second. The top of the module says how they were made, each step with its method; <b>Details and references</b> leads here.</p><p><b>Choose tracts</b> in one of three ways (the ▼ beside Choose). <b>Near a structure</b>: the named tracts that come within a distance (<b>Within</b>) of a tumor, any segment of the patient, or a structure you outline here (<b>Outline one here…</b>: strokes inside, strokes around, Grow, Done); each is shown whole, in its own color, with how close it comes; tracts that come within 2 mm more, or close with fewer lines, are listed in gray, hidden. A case with a tumor opens on this way. <b>By name</b>: tick a tract and both its sides are drawn; untick it and they go. Any other brain opens on this way. <b>Through a place</b>: a box put where the slice views cross; move and resize it with its handles, and the lines through it follow.</p><p><b>Shown</b> lists one row per tract with its left and right side next to each other: near a structure, how close it comes and the lines on each side; otherwise the lines and FA on each side. The row's tooltip gives MD, AD and RD, the volume, the length, and how lopsided the tract is (left against right). The left side is drawn a shade darker than the right. Click a name to rename it; the ⋯ at the end of the row (or a right-click) removes it, lists the tracts near it, adds lines to it, or opens the paper that describes it. <b>Add lines</b> (in the row's ⋯) follows many more lines in a tract listed near a structure, on both sides. <b>Export the numbers</b>, above the list, puts one table file (CSV) at the top of Downloads: a row per tract and side with lines, FA, MD, AD, RD, volume and length, and what made them. A tract that looks thin or missing on a tumor's side may be destroyed by the tumor, or still there but hidden by swelling (edema): compare it with the other side. Everything stays on this computer.</p><p><b>Advanced</b> holds the maps (<b>b0 (corrected)</b>, <b>FA</b>: how strongly water moves along one direction, and <b>Color FA</b>: that direction as a color, red left-right, green front-back, blue up-down; shown over the T1), the scan as acquired, the distortion and head-movement corrections, how tracts are drawn (lines or tubes, colored by tract or by direction, ends shortened), and, under For developers, the tracking method and its numbers.</p><p><b>Two-tensor</b> (UKF, the default) follows two fiber directions, from a starting point in every voxel of the brain, with the settings of the atlas the tract names come from (as Mike Halle's tractline does), so tracts reach into the swelling around a tumor; it runs on the graphics card and agrees with the original UKFTractography program (on a whole brain, 93% of fibers end within 0.1 mm of the original's from the same starting points). <b>Single tensor</b> follows one direction per voxel; where tracts cross, that direction is an average, and a tract may stop or turn. With <b>Two-tensor</b>, <b>Make tracts</b> follows tracts through the whole brain and names them with RapidParc (Bisten, Schultz et al., University of Bonn), a network trained on an atlas of 800 fiber clusters (Zhang, O'Donnell et al.); each named tract that comes within the distance of the chosen structure is shown whole, in its own color, with its closest distance to the structure. Streamlines no name fits and pass close are shown in gray; the rest of the brain is kept, hidden.</p><p><b>Licenses.</b> Research software: not reviewed or approved by the FDA or any other agency; clinical applications are neither recommended nor advised. The two-tensor tracking is a port of UKFTractography (authors: Yogesh Rathi, Stefan Lienhard, Yinpeng Li, Martin Styner, Ipek Oguz, Yundi Shi, Christian Baumgartner, Ryan Eckbo, Tashrif Billah and Dheshan Mohandass; github.com/pnlbwh/ukftractography). All or portions of this licensed product (such portions are the \"Software\") have been obtained under license from The Brigham and Women's Hospital, Inc. and are subject to the following terms and conditions: <a href=\"./vendor/diffusion/licenses/LICENSE-UKF.txt\" target=\"_blank\">the UKF Tractography Contribution and Software License Agreement</a> (this is a modified version: translated to TypeScript and WGSL). RapidParc's trained network is under <a href=\"./vendor/diffusion/rapidparc/LICENSE.txt\" target=\"_blank\">its BSD license</a> (University of Bonn); TractCloud's table of tract names under <a href=\"./vendor/diffusion/tractcloud/LICENSE.txt\" target=\"_blank\">3D Slicer's license</a>; dcm2niix under <a href=\"./vendor/diffusion/dcm2niix/LICENSE.txt\" target=\"_blank\">its own (BSD)</a>; the rest of this extension under the <a href=\"./vendor/diffusion/licenses/LICENSE\" target=\"_blank\">Apache License 2.0</a> (<a href=\"./vendor/diffusion/licenses/NOTICE\" target=\"_blank\">NOTICE</a>).</p>",
    acknowledgements: DIFFUSION_REFERENCES.map((r) => `${r.cite}${r.link ? ` ${r.link}` : ""} — ${r.usedFor}${r.verified ? "" : " (citation to be checked)"}`),
    mount(el) { root = el; render(); },
    onShow() { render(); },
  });

  // THE MAP STAYS OVER THE ANATOMY WHILE THE CASE LOADS (Ron, 2026-10-15: "the two D viewers show the T1 which i like
  // but they don't show the color by orientation overlay"): every new volume takes the slice views' background
  // (logic/ingest.ts), so a scan of the same case that finishes after the diffusion MRI -- the reversed scan, the T1 --
  // took the T1 or the map away. When another volume of the case arrives, the map this module put is put back over the
  // anatomy; a volume of another patient, or the person's own choice in the slice gear, is left alone.
  const knownImages = new Set<string>();
  function mapsBack() {
    const fresh = [...live.nodes.values()].filter((n) => n.type === "image" && !knownImages.has(n.id));
    for (const n of fresh) knownImages.add(n.id);
    if (!fresh.length) return;
    for (const s of scans()) {
      const mapId = mapOn.get(s.browserId);
      if (!mapId || !live.nodes.get(mapId) || busy) continue;
      const ours = new Set([mapId, ...s.frameIds, ...[...storedMaps.values()].flatMap((m) => [m.fa, m.colorfa, m.b0]), computed.get(s.browserId)?.faId, computed.get(s.browserId)?.colorFaId, computed.get(s.browserId)?.b0Id]);
      const ofCase = fresh.some((n) => {
        if (ours.has(n.id)) return false;
        const o = (n.origin as Record<string, unknown> | undefined) ?? {};
        return (!!s.study && o.studyInstanceUID === s.study) || (!!s.patient && o.patientID === s.patient);
      });
      if (ofCase && s.browserId === chosen) putMap(s, mapId);
    }
  }

  // THE PLAYER IN THE TOP BAR stays hidden for every diffusion series -- the scan and its reversed partner (Ron, 2026-10-15:
  // their volumes are directions, not moments; "probably hide with the ability to bring it up under advanced"); Advanced ›
  // Maps › Volumes brings the scan's back. Set once per series, so the person's own choice is kept.
  function hideDiffusionPlayers() {
    for (const b of sequenceBrowsers(live)) {
      if (b.transport !== undefined) continue;
      const first = browserFrames(live, b.id).frames[0], n = first ? live.nodes.get(first.node) : undefined;
      if ((n?.origin as Record<string, unknown> | undefined)?.diffusion !== undefined) live.write({ op: "patch", id: b.id, path: "#/transport", value: "hidden" });
    }
  }

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
    mapsBack();
    hideDiffusionPlayers();
    for (const s of scans()) {
      if (seenScans.has(s.browserId)) continue;
      seenScans.add(s.browserId);
      // THE PLAYER IN THE TOP BAR stays hidden for a diffusion scan (Ron, 2026-10-15: its volumes are directions, not
      // moments; "probably hide with the ability to bring it up under advanced"): Advanced › Maps › Volumes brings it back.
      arrivedAt.set(s.browserId, performance.now());
      chosen = s.browserId;
      void showMap(s, "colorfa");
      // The stored tracts read as the scan arrives (about a second on PAT16): the face then says how they were made, and
      // every way of choosing tracts starts from them at once.
      void findStored(s).then(() => { render(); return prefetchStored(s); }).then(() => render()).catch(() => {});
    }
    const now = sig();
    if (now !== lastSig || pruned) { lastSig = now; render(); }
  });
}

queueModule(registerDiffusionPanel);
// The second module of this extension: judging the corticospinal tract case by case (review.ts; it registers itself).
import "./review.ts";

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
//    FA, or Color FA. Color FA is what a diffusion scan shows when it loads (Ron's yes on the mockup). The tensor is
//    fitted once per scan, on the processor (tensor.ts; about 1 s on PAT16, reading included).
//  - TRACTS (yellow band: it makes new things): tracts near a segment -- started in every white-matter voxel inside it
//    or within a distance of it -- or from a point clicked in a view.
//  - IN THE SCENE (green band: what exists): the tract groups, each with its eye and its ✕, drawn as tubes (Ron:
//    "tubes" first) or lines.
// Tracking: UKF two-tensor free water on the graphics card (ukf-gpu.ts), or one tensor (tracking.ts). Names: TractCloud
// (tractcloud/), from whole-brain tracking; the named tracts near the chosen structure are shown whole (makeNamedTracts).
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
import { prepareUkfData, type UkfData } from "./ukf.ts";
import { DCM2NIIX_VERSION, secondOpinion, type SecondOpinion } from "./second-opinion.ts";
import { assetUrl, seriesDicomFiles, startPlacing } from "albula";
import { createSegmentation, growIntoSegmentation, openDicomDatabase, openLoadFromDisk, paintInto, runAction, saveSegmentationToDicom } from "albula";
import { readMore, tnaLine, tractLabel } from "./tract-info.ts";
import { faceNear as nearOnFace, isTumorName, patientOf, pickAnatomy, withoutLastRun } from "./face.ts";
import { loadModel, type ModelJson, type TractCloudModel } from "./tractcloud/tractcloud.ts";
import { nameTracts } from "./tractcloud/name-tracts.ts";
import { correctWithReversed, MIN_NEAR_STREAMLINES, sortByDistance, streamlineDistances, tractName, trackUkfSeeds, wholeBrainSeeds } from "./planning.ts";
import { tractColor, UNNAMED } from "./tractcloud/tract-colors.ts";
import { seedsInSphere, trackFromSeeds, type Streamline, type TrackingOptions } from "./tracking.ts";
import { DIFFUSION_REFERENCES } from "./references.ts";
import { estimateResponses, kernelFromResponses } from "./responses.ts";
import { csdVolume, type FodVolume } from "./csd-volume.ts";
import { trackPttParallel } from "./ptt.ts";

/** A diffusion scan in the scene: a sequence whose frames carry diffusion values. */
interface Scan { browserId: string; name: string; frameIds: string[]; bValues: number[]; study?: string; patient?: string }
/** What has been computed for a scan, kept while the scan is in the scene. */
interface Computed { dwi: DiffusionSeries; fit: TensorFit; maxB: number; corrected: string; partnerId: string; faId?: string; colorFaId?: string; ukf?: UkfData; fod?: FodVolume }
/** A reversed phase-encoding scan for a diffusion scan: b = 0 images on the same grid, same study. */
interface Partner { id: string; name: string; frameIds: string[] }
/** A group of tracts. */
interface TractGroup {
  id: number; name: string; scan: string; strands: Float32Array[]; visible: boolean; method: Method;
  /** Named by TractCloud: the tract (index into the model's list), its side (+1 right, -1 left, 0 none), and its
   *  closest distance to the structure it was measured against, in mm. `unnamed`: TractCloud gave no name. */
  tract?: number; side?: number; distanceMm?: number; within?: number; unnamed?: boolean;
  /** Made by "Show the fiber tracts near the tumor": the next run for the same scan replaces them (critic, finding 5). */
  run?: boolean;
}
/** How tracts are followed: one tensor per voxel (fast, the classic), or the two-tensor free-water UKF (ukf.ts; crossing
 *  fibers, and free water -- edema -- modeled; the method SlicerDMRI uses for tumor planning). */
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
  let field: FiberField | undefined;
  let chosen = "";                                          // browser id of the scan the panel is about
  let near = "", withinMm = 8, busy = "", seeding = false, note = "";
  let cancelSeeding: (() => void) | undefined;
  const adv: Required<Pick<TrackingOptions, "minFA" | "maxAngleDeg" | "stepVoxels">> & { maxB: number } = { minFA: 0.15, maxAngleDeg: 45, stepVoxels: 0.5, maxB: 1500 };
  let root: HTMLElement | undefined;
  let advOpen = false;
  let moreOpen = false;                                      // the face's Advanced (mockup v4: everything but the one button)
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
    return [{ browserId: b.id, name: String(sequence?.name ?? b.name ?? "Diffusion scan"), frameIds: nodes.map((n) => n.id), bValues: bs, study: o.studyInstanceUID as string | undefined, patient: o.patientID as string | undefined }];
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
      putBackground(anat.id);          // strokes are drawn on the anatomy, not on Color FA
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
    const bg = ((([...live.nodes.values()].find((n) => n.type === "sliceComposite")?.refs as Record<string, string[]> | undefined)?.background) ?? [])[0];
    const c = computed.get(scan.browserId);
    if (bg && c?.colorFaId === bg) return "colorfa";
    if (bg && c?.faId === bg) return "fa";
    if (bg && scan.frameIds.includes(bg)) return "signal";
    return "";
  };
  /** A transform on the scan or a segmentation: refused in plain words, not ignored (critic, finding 5). */
  const moved = (n: MrsonNode | undefined) => !!n && !isIdentity(worldForNode(n, live.nodes));
  /**
   * THE REVERSED PHASE-ENCODING SCAN for a diffusion scan: another sequence (or single volume) of the same study, on the
   * same grid, all of whose images are b = 0 -- the pair distortion correction needs (distortion.ts).
   */
  const partnerFor = (scan: Scan): Partner | undefined => {
    const ref = live.nodes.get(scan.frameIds[0]);
    if (!ref) return undefined;
    const same = (n: MrsonNode) => JSON.stringify(n.dims) === JSON.stringify(ref.dims) && (n.ijkToRAS as number[]).every((v, i) => Math.abs(v - (ref.ijkToRAS as number[])[i]) < 1e-3)
      && ((n.origin as Record<string, unknown> | undefined)?.studyInstanceUID === scan.study);
    const b0 = (n: MrsonNode) => { const d = (n.origin as Record<string, unknown> | undefined)?.diffusion as { bValue?: number } | undefined; return d?.bValue !== undefined && d.bValue < 50; };
    for (const b of sequenceBrowsers(live)) {
      if (b.id === scan.browserId) continue;
      const { frames, sequence } = browserFrames(live, b.id);
      const nodes = frames.map((f) => live.nodes.get(f.node)).filter(Boolean) as MrsonNode[];
      if (nodes.length && nodes.every((n) => same(n) && b0(n))) return { id: b.id, name: String(sequence?.name ?? b.name ?? "reversed scan"), frameIds: nodes.map((n) => n.id) };
    }
    for (const n of live.nodes.values()) if (n.type === "image" && !(n as { hidden?: boolean }).hidden && !scan.frameIds.includes(n.id) && same(n) && b0(n)) return { id: n.id, name: String(n.name ?? "reversed scan"), frameIds: [n.id] };
    return undefined;
  };
  /** DISTORTION CORRECTION with the reversed scan (planning.ts correctWithReversed). */
  async function correctDistortion(dwi: DiffusionSeries, partner: Partner): Promise<string> {
    const vols: ArrayLike<number>[] = [];
    for (const id of partner.frameIds) { const n = live.nodes.get(id); if (!n) throw new Error("the reversed scan was taken out of the scene"); vols.push((await fetchZarrVolumeNative(live.blobBase(), n.zarr as ZarrDesc)).data); }
    return await correctWithReversed(dwi, vols, partner.name, say);
  }

  // ── computing ─────────────────────────────────────────────────────────────────────────────────────────────────
  async function ensureFit(scan: Scan): Promise<Computed> {
    const have = computed.get(scan.browserId);
    const partner = correct ? partnerFor(scan) : undefined;
    // Still valid when made with the same b range and the same reversed scan (or none).
    if (have && have.maxB === adv.maxB && have.partnerId === (partner?.id ?? "")) return have;
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
    const dwi = fromDicomVolumes(vols, scan.name);
    // The second opinion reads the directions before any correction touches the volumes (the correction moves voxels,
    // not directions); it compares b-values and directions only.
    check(scan, { ...dwi, volumes: [] });
    const corrected = partner ? await correctDistortion(dwi, partner) : correct ? "not corrected (no reversed phase-encoding scan of this study is loaded)" : "not corrected (switched off)";
    const t1 = performance.now();
    say("Fitting the diffusion tensor…");
    await new Promise((r) => setTimeout(r, 0));
    const fit = fitTensors(dwi, { maxB: adv.maxB });
    const c: Computed = { dwi, fit, maxB: adv.maxB, corrected, partnerId: partner?.id ?? "" };
    computed.set(scan.browserId, c);
    say(`Tensor fitted in ${((performance.now() - t0) / 1000).toFixed(1)} s (reading and distortion ${((t1 - t0) / 1000).toFixed(1)} s), from ${fit.used.length} volumes up to b = ${adv.maxB}; distortion ${corrected}.`);
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
        putBackground(frames[selected]?.node ?? scan.frameIds[0]);
        return;
      }
      const c = await ensureFit(scan);
      if (what === "fa") {
        if (!c.faId || !live.nodes.get(c.faId)) {
          // A computed map: it does not take the 3D view over from the scan (autoVolumeRendering: false).
          const r = await loadVolumeIntoScene(live, store, { dims: c.fit.dims, ijkToRAS: c.fit.ijkToRAS, data: c.fit.fa, dtype: "<f4", name: `${scan.name} FA` }, { name: `${scan.name} FA`, extra: { autoVolumeRendering: false } });
          c.faId = r.imageId;
          live.write({ op: "patch", id: r.displayId, path: "#/window", value: 1 });
          live.write({ op: "patch", id: r.displayId, path: "#/level", value: 0.5 });
        } else putBackground(c.faId);
      } else {
        if (!c.colorFaId || !live.nodes.get(c.colorFaId)) {
          // FA times the principal direction's absolute components, one byte per color (tensor.ts colorFA), packed into
          // one sample that the slice views draw as color (render/fields.ts ImageFieldOpts.rgb24).
          const rgb = colorFA(c.fit), n = c.fit.fa.length, packed = new Float32Array(n), q = (x: number) => Math.max(0, Math.min(255, Math.round(x * 255)));
          for (let v = 0; v < n; v++) packed[v] = packRGB24(q(rgb[3 * v]), q(rgb[3 * v + 1]), q(rgb[3 * v + 2]));
          const r = await loadVolumeIntoScene(live, store, { dims: c.fit.dims, ijkToRAS: c.fit.ijkToRAS, data: packed, dtype: "<f4", name: `${scan.name} Color FA` }, { name: `${scan.name} Color FA`, extra: { rgb24: true, autoVolumeRendering: false } });
          c.colorFaId = r.imageId;
        } else putBackground(c.colorFaId);
      }
    } catch (e) {
      say(`${what === "fa" ? "FA" : what === "colorfa" ? "Color FA" : "The scan"} could not be shown: ${(e as Error).message}`);
    } finally { busy = ""; render(); }
  }
  const putBackground = (imageId: string) => {
    for (const cmp of [...live.nodes.values()].filter((n) => n.type === "sliceComposite")) live.write({ op: "patch", id: cmp.id, path: "#/refs/background", value: [imageId] });
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
  async function trackUkf(c: Computed, seedsRAS: number[][]): Promise<Float32Array[]> {
    c.ukf ??= prepareUkfData(c.dwi, c.fit.mask);
    return await trackUkfSeeds(device, c.ukf, seedsRAS, adv.minFA, (f) => { busy = `Making tracts… ${Math.round(100 * f)}%`; render(); });
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
      const t0 = performance.now();
      const seeds = wholeBrainSeeds(c.fit);
      say(`Following tracts through the whole brain from ${seeds.length.toLocaleString()} starting points…`);
      const sl = method === "ptt" ? await trackPtt(c, seeds) : await trackUkf(c, seeds);
      const t1 = performance.now();
      busy = "Naming tracts…"; render();
      const model = await tractCloud();
      tractCount = model.json.tracts.length;
      const named = await nameTracts(device, model, sl);
      busy = "Measuring distances…"; render();
      const z = await fetchZarrVolumeNative(live.blobBase(), target.seg.zarr as ZarrDesc), lab = z.data;
      const dist = await streamlineDistances({ dims: target.seg.dims as number[], ijkToRAS: target.seg.ijkToRAS as number[], inside: (v) => Number(lab[v]) === target.labelValue }, sl, withinMm + 2);
      const sorted = sortByDistance(model, named, dist, withinMm), nearTracts = sorted.near;
      // A NEW RUN REPLACES THE LAST ONE for this scan (critic, finding 5: a second press doubled every tract).
      groups.splice(0, groups.length, ...withoutLastRun(groups, scan.browserId));   // face.ts
      const pick = (idx: number[]) => idx.map((i) => sl[i]);
      for (const e of nearTracts) {
        const t = model.json.tracts[e.tract];
        groups.push({ id: ++groupSeq, name: t ? tractLabel(t.abbr, t.name, e.side) : tractName(model, e.tract, e.side), scan: scan.browserId, strands: pick(e.idx), visible: true, method,
          tract: e.tract, side: e.side, distanceMm: e.d, within: e.within, run: true });
      }
      const unnamedNear = sorted.unnamedNear, rest = [...sorted.unnamedFar, ...sorted.far.flatMap((e) => e.idx)];
      if (unnamedNear.length) groups.push({ id: ++groupSeq, name: `Not named, within ${withinMm} mm of ${target.label}`, scan: scan.browserId, strands: pick(unnamedNear), visible: true, method, unnamed: true, run: true });
      if (rest.length) groups.push({ id: ++groupSeq, name: "Rest of the brain", scan: scan.browserId, strands: pick(rest), visible: false, method, unnamed: true, run: true });
      redraw3d();
      const t2 = performance.now();
      say(`${nearTracts.length} named tracts come within ${withinMm} mm of ${target.label} (at least ${MIN_NEAR_STREAMLINES} streamlines each)${unnamedNear.length ? `, and ${unnamedNear.length.toLocaleString()} streamlines no name fits` : ""}. ` +
        `${sl.length.toLocaleString()} streamlines through the whole brain in ${((t1 - t0) / 1000).toFixed(1)} s, named in ${named.seconds.toFixed(1)} s (TractCloud), ${((t2 - t0) / 1000).toFixed(1)} s in all.`);
    } catch (e) { say(`Tracts could not be made: ${(e as Error).message}`); }
    finally { busy = ""; render(); }
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

  /** How many tracts the model names (Other included), once it is loaded; colors need it. */
  let tractCount = 43;
  function redraw3d() {
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
      return g.strands.map((p) => id ? { points: p, bundle: id } : { points: p, bundle: directionId(p), pointBundles: pointIds(p) });
    });
    const old = field; field = undefined;
    if (!strands.length) view.removeField(FIELD_KEY);
    else { field = new FiberField(device, strands, { radius: RADIUS[drawAs], bundleColors: pal }); view.setField(FIELD_KEY, field); }
    old?.destroy?.();
  }

  /** What left the scene takes its computations and tracts with it (critic, finding 9; CONSTRAINTS: a copy is held only
   *  while something reads it). */
  function prune() {
    const ids = new Set(scans().map((s) => s.browserId));
    let changed = false;
    for (const [id, c] of computed) if (!ids.has(id)) { computed.delete(id); dropMaps(c); changed = true; }
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
    const face = shell.section(root, "2 · Fiber tracts near the tumor", { band: "yellow", open: true, note: groups.some((g) => g.tract !== undefined) ? `${groups.filter((g) => g.tract !== undefined).length} found` : "" });
    const go = document.createElement("button");
    go.className = "sl-primary";
    go.style.cssText = "width:100%;margin:4px 0";
    go.textContent = busy && !busy.startsWith("Grow") && !busy.startsWith("Comput") ? busy : "Show the fiber tracts near the tumor";
    // Says the cut (critic, finding 4: "every" was not true): a named tract is listed when at least 5 of its fibers come
    // that close.
    go.title = `Finds the brain's main nerve fiber tracts, names them, and shows each named tract with at least ${MIN_NEAR_STREAMLINES} of its fibers within ${withinMm} mm of the tumor — whole, in its own color, with how close it comes. ${method === "ptt" ? "About three minutes (Smooth curves, chosen under Advanced)." : method === "single" ? "Uses Two-tensor: Single tensor (chosen under Advanced) does not name tracts. About half a minute." : "About half a minute."}`;
    go.disabled = !!busy || !scan || !faceNear || seeding || !!outline;
    go.onclick = () => { if (method === "single") method = "ukf"; near = faceNear; void makeTracts(); };   // the face names tracts, from a tumor
    face.append(go);
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
    mm.type = "number"; mm.min = "0"; mm.max = "60"; mm.step = "5"; mm.value = String(withinMm);
    mm.style.cssText = "flex:0 0 4.5em;width:4.5em;min-width:0";
    mm.title = method === "ukf" ? "How close a tract must come to the structure to be shown, in millimeters (0: only tracts reaching into it)." : "How far around the structure tracts may start, in millimeters (0: only inside it).";
    mm.onchange = () => { withinMm = Math.max(0, Math.min(60, Number(mm.value) || 0)); mm.value = String(withinMm); };
    const mmWrap = document.createElement("span"); mmWrap.style.cssText = "display:inline-flex;align-items:center;gap:6px"; mmWrap.append(mm, "mm");
    shell.row(tr, "Within").append(mmWrap);
    const meth = document.createElement("div");
    meth.style.cssText = "display:flex;gap:3px";
    for (const [k, label, tip] of [["ukf", "Two-tensor", "Follows two crossing fiber directions and the free water around them (edema): the method used for tumor planning. The default: about half a minute."], ["ptt", "Smooth curves", "Follows fiber directions as smooth curves, from a model of all the directions in each voxel (CSD and parallel transport tracking). About three minutes; on the development cases no better near meningiomas than two-tensor."], ["single", "Single tensor", "One direction per voxel: fast, but stops or turns where fibers cross."]] as const) {
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
    num("Stop below FA", "A tract ends where the white matter becomes this faint.", () => adv.minFA, (v) => { adv.minFA = v; }, 0.05, 0.05, 0.9);
    num("Largest turn (°)", "A tract ends where it would bend more sharply than this.", () => adv.maxAngleDeg, (v) => { adv.maxAngleDeg = v; }, 5, 5, 90);
    num("Step (voxels)", "How finely a tract follows the white matter: smaller is smoother and slower.", () => adv.stepVoxels, (v) => { adv.stepVoxels = v; }, 0.1, 0.1, 2);
    num("Highest b used", "Which part of the scan the maps and tracts are made from: its images up to this b-value. Changing it makes the maps again.", () => adv.maxB, (v) => {
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
    // THE TRACTS, under the face's button (what was made, whatever made it)
    if (groups.length) {
      const sc = listBox;
      for (const g of groups) {
        const row = document.createElement("div");
        row.style.cssText = "display:flex;align-items:center;gap:8px;min-width:0;padding:1px 0";
        const name = document.createElement("span"); name.textContent = g.name; name.title = g.name;
        // READ MORE (Ron, 2026-10-01): the paper that describes the tract, opened in the browser from the name.
        const abbr = /\(([^)]+)\)$/.exec(g.name)?.[1];
        if (g.tract !== undefined && abbr) {
          const ref = readMore(abbr);
          const tna = tnaLine(abbr);
          name.title = `${g.name}${tna ? `\n${tna}` : ""}\nRead more (click): ${ref.cite}`;
          name.onclick = () => { void fetch(`/_open?url=${encodeURIComponent(ref.link)}`).then((r) => { if (!r.ok) throw new Error(); }).catch(() => { globalThis.open?.(ref.link, "_blank"); }); };
        }
        name.style.cssText = `flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap${name.onclick ? ";cursor:pointer;text-decoration:underline dotted" : ""}`;
        const n = document.createElement("span"); n.style.opacity = "0.7";
        n.textContent = g.within !== undefined ? `${g.within.toLocaleString()}/${g.strands.length.toLocaleString()}` : g.strands.length.toLocaleString();
        n.title = g.within !== undefined ? `${g.within.toLocaleString()} of this tract's ${g.strands.length.toLocaleString()} streamlines come within the distance of the structure` : "streamlines in this group";
        if (g.tract !== undefined || g.unnamed) {
          // The tract's own color, as it is drawn (and as its card will show it).
          const c = g.tract !== undefined ? tractColor(g.tract, tractCount) : UNNAMED;
          const sw = document.createElement("span");
          sw.style.cssText = `flex:0 0 10px;width:10px;height:10px;border:1px solid var(--sl-border, #000);background:rgb(${c.slice(0, 3).map((v) => Math.round(v * 255)).join(",")})`;
          row.append(sw);
        }
        const dist = document.createElement("span");
        if (g.distanceMm !== undefined) {
          dist.textContent = g.distanceMm < 0.05 ? "touches" : `${g.distanceMm.toFixed(1)} mm`;
          dist.title = g.distanceMm < 0.05 ? "This tract reaches into the structure." : "This tract's closest distance to the structure.";
          dist.style.cssText = "flex:0 0 auto;opacity:0.85";
        }
        const icon = "background:none;border:none;padding:0 2px;cursor:pointer;color:inherit;font-size:13px;min-width:0;flex:0 0 auto";
        const eye = document.createElement("button"); eye.style.cssText = icon; eye.textContent = g.visible ? "👁" : "🚫"; eye.title = g.visible ? "Hide these tracts" : "Show these tracts";
        eye.onclick = () => { g.visible = !g.visible; redraw3d(); render(); };
        const x = document.createElement("button"); x.style.cssText = icon; x.textContent = "✕"; x.title = "Remove these tracts";
        x.onclick = () => { groups.splice(groups.indexOf(g), 1); redraw3d(); render(); };
        row.append(name, dist, n, eye, x);
        sc.append(row);
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

  shell.registerPanel({
    id: "diffusion",
    title: "Diffusion",
    groups: ["Display"],
    tip: "Which of the brain's fiber tracts run near a tumor, for planning an operation.",
    help: "<p><b>For preparing a case.</b> The module needs three things, and ticks each off when it is there: the <b>diffusion MRI</b> (the scan that shows the brain's wiring; often named DTI, DWI or diffusion), the <b>MRI of the anatomy</b> (usually the T1 with contrast), and the <b>tumor's outline</b>. <b>Open a patient…</b> opens the DICOM database; <b>Scans on this computer…</b> opens Load / Save, where a folder from a CD, a USB stick or an export is added to the database. Without an outline, draw a few <b>Tumor</b> strokes inside it and a few <b>Not tumor</b> strokes around it, on a few slices, then <b>Grow the outline</b>, check it, correct with more strokes, and press <b>Done</b>. Then <b>Show the fiber tracts near the tumor</b>: each named tract with at least 5 of its fibers within 8 mm of the tumor (the distance can be changed under Advanced) is shown whole, in its own color, with how close it comes; pressing it again replaces the list. Only an outline named as a tumor (or made here) counts. Everything stays on this computer.</p><p><b>Advanced</b> holds the maps and the settings behind that button:</p><p>Shows what a diffusion MRI scan measures: <b>FA</b>, how strongly water moves along one direction (bright in white matter tracts), and <b>Color FA</b>, that direction as a color (red left-right, green front-back, blue up-down). A diffusion scan shows Color FA when it loads.</p><p><b>Tracts</b> follow the main direction of water movement from voxel to voxel. Under Advanced, <b>Make tracts</b> with <b>Single tensor</b> starts them in the white matter inside and around the chosen structure; with <b>Two-tensor</b> or <b>Smooth curves</b> it does what the face's button does, from the structure chosen there; <b>Seed where I click…</b> starts them at one point. <b>Smooth curves</b> follows fibers as gently bending lines from a model of every direction in each voxel (CSD and parallel transport tracking): about three minutes, an option. Tracts are drawn in 3D as tubes or lines; each group can be hidden or removed.</p><p><b>Two-tensor</b> (UKF, the default) follows two fiber directions and the free water around them; it runs on the graphics card and agrees with the reference computation on the processor (fiber ends within a fraction of a millimeter for 90% of fibers). <b>Single tensor</b> follows one direction per voxel; where tracts cross, that direction is an average, and a tract may stop or turn. With <b>Two-tensor</b>, <b>Make tracts</b> follows tracts through the whole brain and names them with TractCloud, a network trained on an atlas of 800 fiber clusters (Zhang, O'Donnell et al.); each named tract that comes within the distance of the chosen structure is shown whole, in its own color, with its closest distance to the structure. Streamlines no name fits and pass close are shown in gray; the rest of the brain is kept, hidden.</p><p><b>Licenses.</b> Research software: not reviewed or approved by the FDA or any other agency; clinical applications are neither recommended nor advised. The two-tensor tracking is a port of UKFTractography (authors: Yogesh Rathi, Stefan Lienhard, Yinpeng Li, Martin Styner, Ipek Oguz, Yundi Shi, Christian Baumgartner, Ryan Eckbo, Tashrif Billah and Dheshan Mohandass; github.com/pnlbwh/ukftractography). All or portions of this licensed product (such portions are the \"Software\") have been obtained under license from The Brigham and Women's Hospital, Inc. and are subject to the following terms and conditions: <a href=\"./vendor/diffusion/licenses/LICENSE-UKF.txt\" target=\"_blank\">the UKF Tractography Contribution and Software License Agreement</a> (this is a modified version: translated to TypeScript and WGSL). TractCloud's trained network is under <a href=\"./vendor/diffusion/tractcloud/LICENSE.txt\" target=\"_blank\">3D Slicer's license</a>; dcm2niix under <a href=\"./vendor/diffusion/dcm2niix/LICENSE.txt\" target=\"_blank\">its own (BSD)</a>; the rest of this extension under the <a href=\"./vendor/diffusion/licenses/LICENSE\" target=\"_blank\">Apache License 2.0</a> (<a href=\"./vendor/diffusion/licenses/NOTICE\" target=\"_blank\">NOTICE</a>).</p>",
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

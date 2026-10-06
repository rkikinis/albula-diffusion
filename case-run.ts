// ONE CASE OF THE CASE LIBRARY, from its BIDS files to the named tracts near its tumor -- the pipeline the batch tool
// (Contents/tools/dmri-cases.ts) runs over the library and the regression test (cases.regression.test.ts) runs on one
// case, so the test checks exactly what produced the stored results. Moved here from the tool on 2026-10-01 (Ron: "Are
// there tests that you can add/improve now?").
//
// The steps, with the app's defaults: read the diffusion scan and its reversed-phase pair (NIfTI + FSL), correct the
// distortion, fit the tensor (b up to 1500), seed the whole brain, follow tracts (UKF on the graphics card, or CSD + PTT
// on the processor's workers), name them (TractCloud), and measure each tract's streamlines against the dataset's tumor
// mask (inside at 0.5 and above).
import { parseNiftiVolumes } from "albula";
import { fromFsl } from "./dwi.ts";
import { fitTensors } from "./tensor.ts";
import { sortByDistance, stageText, streamlineDistances, tractName, type StageTimes } from "./planning.ts";
import { wholeBrainTracts } from "./pipeline.ts";
import { TRACKING_RULE, TRACKING_RULES, type MaskGrid, type TrackingRuleId } from "./tracking-rules.ts";
import type { Rigid } from "./registration.ts";
import type { MotionRuleId } from "./motion.ts";
import { OUTSIDE_RULE } from "./outside-brain.ts";
import type { TractCloudModel } from "./tractcloud/tractcloud.ts";
import { SHORT, type Named } from "./tractcloud/name-tracts.ts";
import { loadRapidParc, type RapidParcModel } from "./rapidparc/rapidparc.ts";
/** Where tracking rule 3's brain mask for a library case lives (Contents/tools/synthstrip-masks.ts writes it). */
export const synthstripMaskPath = (ds: string, id: string) => `${ds}/derivatives/synthstrip/sub-${id}/anat/sub-${id}_ses-preop_desc-synthstrip_mask.nii.gz`;
/** RapidParc's standard weights from the file beside this module (case runs are Deno's), read once. */
let rp: RapidParcModel | undefined;
const defaultRapidParc = () => rp ??= loadRapidParc(Deno.readFileSync(new URL("./rapidparc/model/rapidparc.safetensors", import.meta.url)).buffer);
import { addScanNoise, noiseSigma } from "./agreement.ts";

export const CASE_DEFAULTS = { stopFA: 0.15, maxB: 1500, margins: [0, 5, 6, 8] } as const;

export interface CaseTract { tract: string; abbr: string; category: string; total: number; closest: number; within0: number; within5: number; within6: number; within8: number }
export interface CaseResult {
  id: string; corrected: string; streamlines: number; short: number; other: number; named: number; tumorVoxels: number;
  method: "ukf" | "ptt";
  /** tracking-rules.ts: the rule the case was tracked by, and the shell it used (undefined: every shell). */
  trackingRule?: TrackingRuleId; shell?: number;
  /** What the brain mask (seeds and tracking boundary) was made from (TensorFit.seedMaskRule), and for rule 3 how much of
   *  the T1's brain (mL) lay outside the diffusion grid. */
  brain?: string; brainOutsideGridMl?: number;
  /** registration.ts: the move from the diffusion scan to the T1 the case was tracked on, and a doubt about it. */
  alignment?: { T: Rigid; doubt?: string };
  /** motion.ts: the head-movement rule the case was put in place by (0: none), and with rule 1 what it found. */
  motionRule: MotionRuleId; motion?: { largest: { mm: number; degrees: number }; residual: { before: number; after: number } };
  seconds: { read_correct_fit: number; csd: number; track: number; name: number; total: number };
  /** Step by step (planning.ts StageTimes, milliseconds), and the same in words. */
  stages: StageTimes;
  stagesText: string;
  /** With opts.keep: the streamlines and their names, for agreement measures (agreement.ts). */
  kept?: { sl: Float32Array[]; named: Named; outside: Uint8Array };
  /** With opts.noiseSeed: the noise level that was added (σ, the scan's own, measured from its background). */
  noiseSigma?: number;
  /** Streamlines that cross the fluid at the brain's edge (outside-brain.ts), kept out of the tracts -- when the test is on
   *  (outsideRule its id; 0 = off, and then outsideBrain is 0). */
  outsideBrain: number; outsideRule: number;
  /** Every named tract with at least one streamline within the largest margin, with its counts. */
  tracts: CaseTract[];
}

/** Run case `id` of the BIDS dataset at `ds` (ds001226's layout: ses-preop, acq-AP / acq-PA, derivatives/tumor_masks). */
export async function runCase(ds: string, id: string, device: GPUDevice, model: TractCloudModel, method: "ukf" | "ptt" = "ukf", opts: { pttWorkerUrl?: URL; csdWorkerUrl?: URL; /** The card tracker's options (checking variants, e.g. onePass). */ ukf?: Record<string, unknown>; /** The naming draw's seed (label noise floor). */ nameSeed?: number; /** Who names: RapidParc (the default since 2026-10-03; its weights) or TractCloud ("tractcloud"). */ labeler?: RapidParcModel | "tractcloud"; /** Add the scan's own noise again, seeded (scan noise floor). */ noiseSeed?: number; /** Return the streamlines and names. */ keep?: boolean; /** How the brain mask is made (tensor.ts). */ maskMethod?: "head" | "median-otsu"; /** Whole-brain starting points (planning.ts WHOLE_BRAIN_SEEDS). */ seeds?: number; /** Their draw's seed (seed noise floor). */ seedDraw?: number; /** false: seeds anywhere in the head mask, as before 2026-10-02. */ seedMask?: boolean; /** One seed at the center of EVERY brain voxel, left to the tracker's own seed threshold (the ORG atlas's tracking, Mike Halle's mail of 2026-10-01; with opts.ukf's freeWater: false and his settings). Checking only: about ten times the seeds. */ seedEveryVoxel?: boolean; /** The distortion correction's rule (distortion.ts DISTORTION_RULE; default 2). */ distortionRule?: 1 | 2; /** tracking-rules.ts TRACKING_RULE; default the current one. */ trackingRule?: TrackingRuleId; /** false: stay in the diffusion scan's space even when the case has a T1 (as before 2026-10-03). */ onT1?: boolean; /** motion.ts MOTION_RULE; default the current one. */ motionRule?: MotionRuleId; /** Its rounds (default two). */ motionRounds?: number } = {}): Promise<CaseResult> {
  const t0 = performance.now(), p = `${ds}/sub-${id}/ses-preop`;
  const rd = (f: string) => Deno.readFileSync(`${p}/${f}`), tx = (f: string) => Deno.readTextFileSync(`${p}/${f}`);
  const sidecarPE = (f: string) => { try { return (JSON.parse(tx(f)) as { PhaseEncodingDirection?: string }).PhaseEncodingDirection; } catch { return undefined; } };
  const stages: StageTimes = {};
  let dwi = fromFsl(await parseNiftiVolumes(rd(`dwi/sub-${id}_ses-preop_acq-AP_dwi.nii.gz`)), tx(`dwi/sub-${id}_ses-preop_acq-AP_dwi.bval`), tx(`dwi/sub-${id}_ses-preop_acq-AP_dwi.bvec`));
  const rev = fromFsl(await parseNiftiVolumes(rd(`dwi/sub-${id}_ses-preop_acq-PA_dwi.nii.gz`)), tx(`dwi/sub-${id}_ses-preop_acq-PA_dwi.bval`), tx(`dwi/sub-${id}_ses-preop_acq-PA_dwi.bvec`));
  stages.read = performance.now() - t0;
  // THE PARTNER ON ITS OWN GRID: 11 of the 29 have a PA slab placed 2.3 mm and 0.8° away; until 2026-10-02 it was paired
  // voxel by voxel here (Mike Halle's pipeline aligns it by the scanner's coordinates; so does this now).
  let added: number | undefined;
  if (opts.noiseSeed !== undefined) {
    // THE SCAN'S OWN NOISE AGAIN (agreement.ts): measured from the background outside the brain on the scan's own grid,
    // added to the scan AS ACQUIRED, before the correction and the resampling onto the T1 (critic, 2026-10-03, finding
    // 10: added after the resampling it was 57-70% of the real noise -- interpolation smooths, a third was zero fill).
    const m = fitTensors(dwi, { maxB: CASE_DEFAULTS.maxB, maskMethod: opts.maskMethod, seedMask: false }).mask;
    added = noiseSigma(dwi, m);
    const noisy = addScanNoise(dwi, added, opts.noiseSeed);
    dwi.volumes = noisy.volumes;
    // The reversed scan too (a second acquisition is noisy on both): the same σ (same scanner and visit), its own draw.
    rev.volumes = addScanNoise(rev, added, opts.noiseSeed + 7919).volumes;
  }
  // THE T1 (2026-10-03, Ron: "Number three, go"; registration.ts): when the case has one, the field is not applied here
  // but with the move to the T1, in one resampling, and everything after is in the T1's space.
  const t1Vol = opts.onT1 === false ? undefined : await (async () => { try { return (await parseNiftiVolumes(rd(`anat/sub-${id}_ses-preop_T1w.nii.gz`)))[0]; } catch { return undefined; } })();
  // RULE 3's BRAIN (tracking-rules.ts): SynthStrip's mask of the T1, made beforehand by Contents/tools/synthstrip-masks.ts
  // (haversack). A case run asked for rule 3 without it stops and says so rather than tracking under another rule.
  const asked = TRACKING_RULES[opts.trackingRule ?? TRACKING_RULE];
  let brainT1: MaskGrid | undefined;
  if (method === "ukf" && asked.brain === "t1-synthstrip") {
    if (!t1Vol) throw new Error(`${id}: tracking rule 3 needs the T1 (onT1 is false or the case has none)`);
    const f = synthstripMaskPath(ds, id);
    try { const m = (await parseNiftiVolumes(Deno.readFileSync(f)))[0]; brainT1 = { dims: m.dims, ijkToRAS: m.ijkToRAS, data: m.data as ArrayLike<number> }; }
    catch { throw new Error(`${id}: tracking rule 3 needs SynthStrip's brain mask at ${f} (run Contents/tools/synthstrip-masks.ts ${id})`); }
  }
  const r = await wholeBrainTracts({ dwi,
    partner: { b0s: rev.volumes.filter((_, i) => rev.bValues[i] < 50).map((v) => v.data), grid: { dims: rev.volumes[0].dims, ijkToRAS: rev.volumes[0].ijkToRAS }, name: "PA" },
    // The scanner's record of the phase-encoding directions (the BIDS sidecars), when present: PAT03's and CON02's "PA"
    // scans are not reversed pairs (planning.ts correctWithReversed).
    phaseEncoding: { scan: sidecarPE(`dwi/sub-${id}_ses-preop_acq-AP_dwi.json`), partner: sidecarPE(`dwi/sub-${id}_ses-preop_acq-PA_dwi.json`) },
    ...(t1Vol ? { t1: { dims: t1Vol.dims as [number, number, number], ijkToRAS: t1Vol.ijkToRAS, data: t1Vol.data as ArrayLike<number> } } : {}),
    ...(brainT1 ? { brainT1 } : {}) }, device, model,
    { method, labeler: opts.labeler === "tractcloud" ? "tractcloud" : opts.labeler ?? defaultRapidParc(),
      ...(opts.distortionRule ? { distortionRule: opts.distortionRule } : {}), ...(opts.trackingRule ? { trackingRule: opts.trackingRule } : {}),
      ...(opts.motionRule !== undefined ? { motionRule: opts.motionRule } : {}), ...(opts.motionRounds ? { motionRounds: opts.motionRounds } : {}),
      ...(opts.maskMethod ? { maskMethod: opts.maskMethod } : {}), ...(opts.seedMask !== undefined ? { seedMask: opts.seedMask } : {}),
      ...(opts.seeds !== undefined ? { seeds: opts.seeds } : {}), ...(opts.seedDraw !== undefined ? { seedDraw: opts.seedDraw } : {}),
      ...(opts.seedEveryVoxel ? { seedEveryVoxel: true } : {}), ...(opts.ukf ? { ukf: opts.ukf } : {}), ...(opts.nameSeed !== undefined ? { nameSeed: opts.nameSeed } : {}),
      ...(opts.pttWorkerUrl ? { pttWorkerUrl: opts.pttWorkerUrl } : {}), ...(opts.csdWorkerUrl ? { csdWorkerUrl: opts.csdWorkerUrl } : {}) }, stages);
  const { sl, named, outside, corrected, alignment, prep, rule, shell, outsideGridMl, csdSeconds } = r, fit = r.fit;
  const t2 = performance.now();
  // As before the pipeline was shared: tracking includes its signal and seeds; reading, correction and fit are the rest.
  const trackMs = (stages.track ?? 0) + (stages.seeds ?? 0), nameMs = named.seconds * 1000, readFitMs = (t2 - t0) - trackMs - csdSeconds * 1000 - nameMs;
  const tDist = performance.now();
  const OTHER = model.json.tracts.length - 1;
  let short = 0, other = 0;
  for (const t of named.tract) { if (t === SHORT) short++; else if (t === OTHER) other++; }
  // The streamlines crossing the fluid at the brain's edge (pipeline.ts; none while OUTSIDE_RULE is off).
  let outsideCount = 0; for (const o of outside) outsideCount += o;
  let tracts: CaseTract[] = [], tumorVoxels = 0;
  const maskFile = `${ds}/derivatives/tumor_masks/sub-${id}/anat/sub-${id}_space_T1_label-tumor.nii`;
  try {
    const m = (await parseNiftiVolumes(Deno.readFileSync(maskFile)))[0];
    const data = m.data as ArrayLike<number>;
    for (let v = 0; v < data.length; v++) if (data[v] >= 0.5) tumorVoxels++;
    const margins = CASE_DEFAULTS.margins, widest = Math.max(...margins);
    const dist = await streamlineDistances({ dims: m.dims, ijkToRAS: m.ijkToRAS, inside: (v) => data[v] >= 0.5 }, sl, widest + 2);
    const sorted = sortByDistance(model, named, dist, widest, 1, widest, outside);   // every tract, with its counts
    tracts = sorted.near.map((e) => ({ tract: tractName(model, e.tract, e.side), abbr: model.json.tracts[e.tract].abbr, category: model.json.tracts[e.tract].category,
      total: e.idx.length, closest: +e.d.toFixed(2), within0: e.idx.filter((i) => dist[i] <= 0).length, within5: e.idx.filter((i) => dist[i] <= 5).length, within6: e.idx.filter((i) => dist[i] <= 6).length, within8: e.idx.filter((i) => dist[i] <= 8).length }));
  } catch (e) { if (!id.startsWith("CON")) throw e; }   // healthy volunteers have no tumor mask
  const t3 = performance.now();
  stages.distances = t3 - tDist;
  stages.total = t3 - t0;
  return { id, corrected, ...(alignment ? { alignment } : {}), motionRule: prep.motionRule, ...(prep.motion ? { motion: { largest: prep.motion.largest, residual: { before: +prep.motion.residual.before.toFixed(4), after: +prep.motion.residual.after.toFixed(4) } } } : {}), trackingRule: method === "ukf" ? rule.id : undefined, brain: fit.seedMaskRule ?? fit.maskRule, ...(outsideGridMl !== undefined ? { brainOutsideGridMl: +outsideGridMl.toFixed(1) } : {}), shell, stages, stagesText: stageText(stages), ...(opts.keep ? { kept: { sl, named, outside } } : {}), ...(added !== undefined ? { noiseSigma: +added.toFixed(2) } : {}), streamlines: sl.length, short, other, named: sl.length - short - other - outsideCount, outsideBrain: outsideCount, outsideRule: OUTSIDE_RULE.on ? OUTSIDE_RULE.id : 0, tumorVoxels, method,
    seconds: { read_correct_fit: +(readFitMs / 1000).toFixed(1), csd: +csdSeconds.toFixed(1), track: +(trackMs / 1000).toFixed(1), name: +named.seconds.toFixed(1), total: +((t3 - t0) / 1000).toFixed(1) }, tracts };
}

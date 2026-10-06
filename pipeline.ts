// THE WHOLE-BRAIN PIPELINE, ONE PATH FOR EVERY INPUT (Contents/docs/DMRI-AT-IMPORT.md, build plan step 1): the scans
// already read -- the diffusion scan, its reversed phase-encoding partner's b = 0 images, the T1, SynthStrip's brain on
// the T1 -- through the distortion correction, the head's movement and the alignment to the T1 (one resampling), the
// tensor, the whole-brain tracking and the naming. case-run.ts (a BIDS case from files) and the import-time job (a
// DICOM series from a database) both call it, so the regression test checks what the job stores. The tumor is not part
// of it: the near-tumor list is measured afterwards against whatever outline the resident has.
import type { DiffusionSeries } from "./dwi.ts";
import { fitTensors, type TensorFit } from "./tensor.ts";
import { correctWithReversed, trackUkfSeeds, wholeBrainSeeds, type StageTimes, type TrackTiming } from "./planning.ts";
import { everyBrainVoxel, seedsFor, TRACKING_RULE, TRACKING_RULES, ukfDataFor, withBrainFromT1, type MaskGrid, type TrackingRuleId } from "./tracking-rules.ts";
import type { Grid3, Rigid } from "./registration.ts";
import type { FieldFit } from "./distortion.ts";
import { prepareScan, type Prepared } from "./prepare.ts";
import { MOTION_RULE, type MotionRuleId } from "./motion.ts";
import { edgeFluid, outsideBrain, OUTSIDE_RULE } from "./outside-brain.ts";
import type { TractCloudModel } from "./tractcloud/tractcloud.ts";
import { nameTracts, type Named } from "./tractcloud/name-tracts.ts";
import type { RapidParcModel } from "./rapidparc/rapidparc.ts";
import { estimateResponses, kernelFromResponses } from "./responses.ts";
import { csdVolume } from "./csd-volume.ts";
import { trackPttParallel } from "./ptt.ts";

/** The tensor's highest b (the maps); the tracking's shell is the tracking rule's. */
export const PIPELINE_MAX_B = 1500;

export interface PipelineInput {
  /** The diffusion scan as acquired (nothing applied). */
  dwi: DiffusionSeries;
  /** The reversed phase-encoding scan's b = 0 images, on their own grid, with a name for the record. */
  partner?: { b0s: ArrayLike<number>[]; grid: { dims: number[]; ijkToRAS: number[] }; name: string };
  /** The scanner's record of the phase-encoding directions ("j-", or DICOM's "ROW" / "COL"), when known. */
  phaseEncoding?: { scan?: string; partner?: string };
  /** The MRI of the anatomy of the same study. */
  t1?: Grid3;
  /** SynthStrip's brain on the T1 (tracking rule 3). */
  brainT1?: MaskGrid;
}

export interface PipelineOptions {
  method?: "ukf" | "ptt";
  /** Who names: RapidParc's weights, or TractCloud. */
  labeler: RapidParcModel | "tractcloud";
  /** distortion.ts DISTORTION_RULE; tracking-rules.ts TRACKING_RULE; motion.ts MOTION_RULE and its rounds. */
  distortionRule?: 1 | 2; trackingRule?: TrackingRuleId; motionRule?: MotionRuleId; motionRounds?: number;
  /** How the brain mask is made when not from the T1 (tensor.ts), and seeds anywhere in the head (false). */
  maskMethod?: "head" | "median-otsu"; seedMask?: boolean;
  /** Whole-brain starting points and their draw; one seed in every brain voxel (checking only). */
  seeds?: number; seedDraw?: number; seedEveryVoxel?: boolean;
  /** The card tracker's options (checking variants); the naming draw's seed. */
  ukf?: Record<string, unknown>; nameSeed?: number;
  pttWorkerUrl?: URL; csdWorkerUrl?: URL;
}

export interface PipelineResult {
  /** The scan as tracked (corrected, put in place, on the T1's axes when there is one) and its tensor fit. */
  dwi: DiffusionSeries; fit: TensorFit;
  sl: Float32Array[]; named: Named;
  /** Streamlines crossing the fluid at the brain's edge (only when OUTSIDE_RULE is on). */
  outside: Uint8Array;
  /** What was done, in words: the distortion, the movement, the alignment. */
  corrected: string;
  alignment?: { T: Rigid; doubt?: string };
  /** prepare.ts's result: the movement rule applied and what it found. */
  prep: Prepared;
  /** The tracking rule actually applied (rule 3 falls back to 2 without SynthStrip's brain or after a doubtful alignment). */
  rule: (typeof TRACKING_RULES)[TrackingRuleId];
  shell?: number; outsideGridMl?: number; csdSeconds: number;
}

/** The whole pipeline; `stages` collects the step times (planning.ts stageText). */
export async function wholeBrainTracts(input: PipelineInput, device: GPUDevice, model: TractCloudModel, opts: PipelineOptions, stages: StageTimes = {}): Promise<PipelineResult> {
  let dwi = input.dwi;
  const method = opts.method ?? "ukf";
  // THE DISTORTION FIELD from the reversed pair, kept (not applied): the movement and the T1 alignment take it into the
  // one resampling (prepare.ts).
  const field: { fit?: FieldFit; sign?: 1 | -1 } = {};
  let corrected = input.partner
    ? await correctWithReversed(dwi, input.partner.b0s, input.partner.name, undefined,
      { partnerGrid: { dims: input.partner.grid.dims, ijkToRAS: input.partner.grid.ijkToRAS }, times: stages, rule: opts.distortionRule, apply: false, field,
        ...(input.phaseEncoding ? { phaseEncoding: input.phaseEncoding } : {}) })
    : "not corrected (no reversed phase-encoding scan)";
  // Head movement, the field and the T1 in one resampling.
  const prep = await prepareScan(dwi, { ...(field.fit ? { field: { fit: field.fit, sign: field.sign ?? -1 } } : {}),
    ...(input.t1 ? { t1: input.t1 } : {}),
    motionRule: opts.motionRule ?? MOTION_RULE, ...(opts.motionRounds ? { motionRounds: opts.motionRounds } : {}),
    ...(input.phaseEncoding?.scan ? { phaseEncoding: input.phaseEncoding.scan } : {}), times: stages });
  dwi = prep.dwi;
  if (prep.said) corrected += `; ${prep.said}`;
  const alignment = prep.alignment;
  const tFit = performance.now();
  let fit = fitTensors(dwi, { maxB: PIPELINE_MAX_B, maskMethod: opts.maskMethod, seedMask: opts.seedMask });
  // On the T1-aligned grid only: a doubtful alignment left the scan in the scanner's place, where a T1 mask does not fit,
  // and the case is then tracked under rule 2 and recorded so (critic, 2026-10-04, finding 10).
  let rule = TRACKING_RULES[opts.trackingRule ?? TRACKING_RULE], outsideGridMl: number | undefined;
  if (input.brainT1 && alignment && !alignment.doubt) ({ fit, outsideGridMl } = withBrainFromT1(fit, input.brainT1, undefined, rule.fluidMdMax, rule.fluidShellVoxels));
  else if (rule.brain === "t1-synthstrip") rule = TRACKING_RULES[2];
  const t1 = performance.now();
  stages.fit = t1 - tFit;
  const detail: TrackTiming = { prepare: 0, gpu: 0, assemble: 0, between: 0 };
  let shell: number | undefined, sl: Float32Array[], csdSeconds = 0;
  if (method === "ptt") {
    const k = kernelFromResponses(estimateResponses(dwi));
    const fod = await csdVolume(dwi, fit.mask, k, opts.csdWorkerUrl ? { workerUrl: opts.csdWorkerUrl } : {});
    csdSeconds = fod.seconds;
    sl = await trackPttParallel(fod, wholeBrainSeeds(fit, opts.seeds, opts.seedDraw), opts.pttWorkerUrl ? { workerUrl: opts.pttWorkerUrl } : {});
  } else {
    // THE TRACKING RULE (tracking-rules.ts): its shell, mask, seeds and thresholds; opts.ukf overrides single settings.
    const tData = performance.now(), data = ukfDataFor(dwi, fit, rule), ts = performance.now();
    shell = data.shell;
    const seeds = opts.seedEveryVoxel ? everyBrainVoxel(fit) : seedsFor(fit, rule, { count: opts.seeds, draw: opts.seedDraw });
    stages.seeds = performance.now() - ts;
    const ukf = { ...rule.ukf, ...(opts.ukf ?? {}) };
    sl = await trackUkfSeeds(device, data, seeds, ukf.stoppingFA as number, undefined, detail, ukf);
    stages.trackDetail = { ...detail, data: ts - tData };
  }
  const t2 = performance.now();
  stages.track = t2 - t1 - csdSeconds * 1000 - (stages.seeds ?? 0);
  const rapidParc = opts.labeler === "tractcloud" ? undefined : opts.labeler;
  const named = await nameTracts(device, model, sl, { ...(opts.nameSeed !== undefined ? { seed: opts.nameSeed } : {}), ...(rapidParc ? { rapidParc } : {}) });
  stages.name = named.seconds * 1000;
  // Streamlines that cross the fluid at the brain's edge would keep no tract's name (outside-brain.ts; off: OUTSIDE_RULE.on).
  const outside = OUTSIDE_RULE.on ? outsideBrain(sl, fit, edgeFluid(fit)) : new Uint8Array(sl.length);
  return { dwi, fit, sl, named, outside, corrected, ...(alignment ? { alignment } : {}), prep, rule, ...(shell !== undefined ? { shell } : {}), ...(outsideGridMl !== undefined ? { outsideGridMl } : {}), csdSeconds };
}

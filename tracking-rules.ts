// HOW THE WHOLE BRAIN IS TRACKED FOR NAMING, as numbered rules (Ron, 2026-09-25: "as modular as possible and also
// versioned"): which model, which thresholds, which part of the scan, where the seeds go. The case runs (case-run.ts) and
// the module's "Show the fiber tracts near the tumor" both take their settings from here; neither holds a copy.
//
// RULE 1 (2026-09-29 .. 2026-10-03): two-tensor UKF with free water, every shell up to the scan's highest, 16,000 seeds
// drawn among brain voxels above FA 0.2, seed FA 0.18, stop at FA 0.15 or mean signal 0.1, a point every 0.9 mm, tracked
// inside the head mask.
//
// RULE 2 (2026-10-03, the default; Ron: "Number one, please follow Mike's lead. I think what he's doing makes sense to
// me"): Mike Halle's tractline, which is the ORG atlas's tracking -- the conditions the naming networks (TractCloud,
// RapidParc) were trained on: plain two-tensor UKF (no free water), the b = 0 images and ONE shell, the one nearest the
// atlas's b = 3000 (ds001226: 2800, 50 directions), every voxel of the brain mask a seed, seed FA 0.1, stop at FA 0.08
// or mean signal 0.06, a point every 1.8 mm, tracked inside the brain mask (DIPY's median_otsu; tractline's prep.py).
// Why: on PAT14 and PAT25 tractline's fibers reached the tumor where rule 1's stopped 8-9 mm short (validation step 6,
// dmri-review 2026-10-03). Caveat carried over from tensor.ts: median_otsu cut the top of PAT16's brain; tractline
// tracks inside it all the same.
import type { DiffusionSeries } from "./dwi.ts";
import type { TensorFit } from "./tensor.ts";
import { prepareUkfData, type UkfData, type UkfOptions } from "./ukf.ts";
import { wholeBrainSeeds } from "./planning.ts";
import { rng } from "./tractcloud/tractcloud.ts";

export interface TrackingRule {
  id: 1 | 2;
  /** The tracker's settings (ukf.ts UkfOptions); stoppingFA is also the module's "Stop below FA" default. */
  ukf: Required<Pick<UkfOptions, "freeWater" | "seedingThreshold" | "stoppingFA" | "stoppingThreshold" | "recordLength">>;
  /** One shell: the b-value it is nearest to (the b = 0 images are always kept); undefined: every shell. */
  shellNear?: number;
  /** Where the seeds go: a draw among brain voxels above FA 0.2 (planning.ts wholeBrainSeeds), or every brain voxel. */
  seeding: "sample" | "every-voxel";
  /** The tracking boundary: the head mask (TensorFit.mask) or the brain mask (TensorFit.seedMask, median_otsu). */
  trackIn: "head" | "brain";
}

export const TRACKING_RULES: Record<1 | 2, TrackingRule> = {
  1: { id: 1, ukf: { freeWater: true, seedingThreshold: 0.18, stoppingFA: 0.15, stoppingThreshold: 0.1, recordLength: 0.9 }, seeding: "sample", trackIn: "head" },
  2: { id: 2, ukf: { freeWater: false, seedingThreshold: 0.1, stoppingFA: 0.08, stoppingThreshold: 0.06, recordLength: 1.8 }, shellNear: 3000, seeding: "every-voxel", trackIn: "brain" },
};
export const TRACKING_RULE: 1 | 2 = 2;

/** The b-value of the shell nearest `near` (shells: b-values over 50, grouped within 50 of each other). */
export function shellNearest(bValues: number[], near: number): number | undefined {
  const shells: number[] = [];
  for (const b of bValues) if (b > 50 && !shells.some((s) => Math.abs(s - b) < 50)) shells.push(b);
  return shells.sort((a, b) => Math.abs(a - near) - Math.abs(b - near) || b - a)[0];
}

/** The tracker's input for a rule: the series cut to the rule's shell (with the b = 0 images), inside the rule's mask. */
export function ukfDataFor(dwi: DiffusionSeries, fit: TensorFit, rule: TrackingRule): UkfData & { shell?: number } {
  const mask = rule.trackIn === "brain" ? fit.seedMask ?? fit.mask : fit.mask;
  if (rule.shellNear === undefined) return prepareUkfData(dwi, mask);
  const shell = shellNearest(dwi.bValues, rule.shellNear);
  if (shell === undefined) return prepareUkfData(dwi, mask);
  const keep = dwi.bValues.map((b, i) => (b <= 50 || Math.abs(b - shell) < 50 ? i : -1)).filter((i) => i >= 0);
  const sub: DiffusionSeries = { ...dwi, volumes: keep.map((i) => dwi.volumes[i]), bValues: keep.map((i) => dwi.bValues[i]), gradients: keep.map((i) => dwi.gradients[i]) };
  return { ...prepareUkfData(sub, mask), shell };
}

/** Every voxel of the brain mask that the tracking mask also holds, at its center, RAS mm. With `draw`, each point is
 *  moved to a seeded random place inside its voxel instead: the seed noise floor for every-voxel seeding (dmri-floors.ts),
 *  where there is no draw of voxels to vary. */
export function everyBrainVoxel(fit: TensorFit, draw?: number): number[][] {
  const [nx, ny, nz] = fit.dims, M = fit.ijkToRAS, sm = fit.seedMask ?? fit.mask, out: number[][] = [];
  const r = draw !== undefined ? rng(draw) : undefined;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = (k * ny + j) * nx + i;
    if (!(sm[v] && fit.mask[v])) continue;
    const a = i + (r ? r() - 0.5 : 0), b = j + (r ? r() - 0.5 : 0), c = k + (r ? r() - 0.5 : 0);
    out.push([M[0] * a + M[1] * b + M[2] * c + M[3], M[4] * a + M[5] * b + M[6] * c + M[7], M[8] * a + M[9] * b + M[10] * c + M[11]]);
  }
  return out;
}

/** The whole-brain seeds for a rule (RAS mm). */
export function seedsFor(fit: TensorFit, rule: TrackingRule, opts: { count?: number; draw?: number } = {}): number[][] {
  return rule.seeding === "every-voxel" ? everyBrainVoxel(fit, opts.draw) : wholeBrainSeeds(fit, opts.count, opts.draw);
}

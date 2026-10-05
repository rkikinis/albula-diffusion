// HOW THE WHOLE BRAIN IS TRACKED FOR NAMING, as numbered rules (Ron, 2026-09-25: "as modular as possible and also
// versioned"): which model, which thresholds, which part of the scan, where the seeds go. The case runs (case-run.ts) and
// the module's "Show the fiber tracts near the tumor" both take their settings from here; neither holds a copy.
//
// RULE 1 (2026-09-29 .. 2026-10-03): two-tensor UKF with free water, every shell up to the scan's highest, 16,000 seeds
// drawn among brain voxels above FA 0.2, seed FA 0.18, stop at FA 0.15 or mean signal 0.1, a point every 0.9 mm, tracked
// inside the head mask.
//
// RULE 2 (2026-10-03, the default until 2026-10-04; Ron: "Number one, please follow Mike's lead. I think what he's doing makes sense to
// me"): Mike Halle's tractline, which is the ORG atlas's tracking -- the conditions the naming networks (TractCloud,
// RapidParc) were trained on: plain two-tensor UKF (no free water), the b = 0 images and ONE shell, the one nearest the
// atlas's b = 3000 (ds001226: 2800, 50 directions), every voxel of the brain mask a seed, seed FA 0.1, stop at FA 0.08
// or mean signal 0.06, a point every 1.8 mm, tracked inside the brain mask (DIPY's median_otsu; tractline's prep.py).
// Why: on PAT14 and PAT25 tractline's fibers reached the tumor where rule 1's stopped 8-9 mm short (validation step 6,
// dmri-review 2026-10-03). Caveat carried over from tensor.ts: median_otsu cut the top of PAT16's brain; tractline
// tracks inside it all the same.
//
// RULE 3 (2026-10-04; Ron: "yes, SynthStrip through haversack"): rule 2 with a different brain. median_otsu is made on the
// diffusion scan's b = 0 after a 9-voxel median filter run four times; at the skull base that filter averages the brain
// with the bone and air around it, and the mask leaves out the lower temporal lobes and part of the cerebellum (PAT13,
// PAT14, PAT23; dmri-review 2026-10-04). Rule 3 takes the brain from the T1 instead -- SynthStrip (Hoopes et al. 2022)
// run by haversack on the T1 the scan is aligned to -- as the seeds and the tracking boundary. FastSurfer was tried and
// not taken: it labels most of a large tumor and much of the tissue around every tumor as not brain. Needs a T1 and its
// SynthStrip mask; without them the caller falls back to rule 2 and says so.
//
// RULE 4 (2026-10-04, tried and set aside the same evening): rule 3 without the fluid. SynthStrip's mask holds the CSF around the brain, and in
// PAT29 rule 3 followed the left trigeminal nerve through the cistern and named it uncinate (Ron: "Looks more like the
// trigeminus nerve"; "I prefer tight, if we don't lose too much"). Voxels whose mean diffusivity says fluid
// (fluidMdMax) are left out of the brain: the cisterns and the nerves lying in them, the ventricles, the sulci; edema,
// whose diffusivity is lower, stays.
import type { DiffusionSeries } from "./dwi.ts";
import type { TensorFit } from "./tensor.ts";
import { prepareUkfData, type UkfData, type UkfOptions } from "./ukf.ts";
import { wholeBrainSeeds } from "./planning.ts";
import { rng } from "./tractcloud/tractcloud.ts";

export interface TrackingRule {
  id: 1 | 2 | 3;
  /** The tracker's settings (ukf.ts UkfOptions); stoppingFA is also the module's "Stop below FA" default. */
  ukf: Required<Pick<UkfOptions, "freeWater" | "seedingThreshold" | "stoppingFA" | "stoppingThreshold" | "recordLength">>;
  /** One shell: the b-value it is nearest to (the b = 0 images are always kept); undefined: every shell. */
  shellNear?: number;
  /** Where the seeds go: a draw among brain voxels above FA 0.2 (planning.ts wholeBrainSeeds), or every brain voxel. */
  seeding: "sample" | "every-voxel";
  /** The tracking boundary: the head mask (TensorFit.mask) or the brain mask (TensorFit.seedMask). */
  trackIn: "head" | "brain";
  /** What the brain mask (TensorFit.seedMask) is: median_otsu on the scan's b = 0, or SynthStrip's mask of the T1 put on
   *  the scan's grid (withBrainFromT1). */
  brain: "median-otsu" | "t1-synthstrip";
  /** Rule 4: voxels of the brain whose mean diffusivity is above this (mm²/s) are fluid and left out -- the cisterns
   *  with the cranial nerves in them, the ventricles, the sulci. */
  fluidMdMax?: number;
  /** Rule 5: the fluid is left out only within this many voxels of the brain's outer edge (the cisterns, the sulci), not
   *  deep (the ventricles, whose roof shares voxels with the corpus callosum; tumors). */
  fluidShellVoxels?: number;
}

export type TrackingRuleId = 1 | 2 | 3 | 4 | 5;
export const TRACKING_RULES: Record<TrackingRuleId, TrackingRule> = {
  1: { id: 1, ukf: { freeWater: true, seedingThreshold: 0.18, stoppingFA: 0.15, stoppingThreshold: 0.1, recordLength: 0.9 }, seeding: "sample", trackIn: "head", brain: "median-otsu" },
  2: { id: 2, ukf: { freeWater: false, seedingThreshold: 0.1, stoppingFA: 0.08, stoppingThreshold: 0.06, recordLength: 1.8 }, shellNear: 3000, seeding: "every-voxel", trackIn: "brain", brain: "median-otsu" },
  3: { id: 3, ukf: { freeWater: false, seedingThreshold: 0.1, stoppingFA: 0.08, stoppingThreshold: 0.06, recordLength: 1.8 }, shellNear: 3000, seeding: "every-voxel", trackIn: "brain", brain: "t1-synthstrip" },
  4: { id: 4, ukf: { freeWater: false, seedingThreshold: 0.1, stoppingFA: 0.08, stoppingThreshold: 0.06, recordLength: 1.8 }, shellNear: 3000, seeding: "every-voxel", trackIn: "brain", brain: "t1-synthstrip", fluidMdMax: 2.5e-3 },
  5: { id: 5, ukf: { freeWater: false, seedingThreshold: 0.1, stoppingFA: 0.08, stoppingThreshold: 0.06, recordLength: 1.8 }, shellNear: 3000, seeding: "every-voxel", trackIn: "brain", brain: "t1-synthstrip", fluidMdMax: 2.5e-3, fluidShellVoxels: 3 },
};
/** The default: 3 (Ron, 2026-10-04: "yes, make rule 3 the default"). Rules 4 and 5 -- the fluid left out of the mask --
 *  were tried that evening and set aside: rule 4 took about a quarter of the corpus callosum's streamlines and cut into
 *  tumors, rule 5 a band along the cortex and a surface tumor's cyst (Ron: "treatment worse than the problem"). The
 *  trigeminal nerve that rule 3 can track keeps a wrong name for now: the test that would take it out of the tracts
 *  (outside-brain.ts) is built but off. Earlier wording, kept: rule 3 (Ron: "yes, make rule 3 the default"), after the comparison with rule 2 on the
 *  twelve (Contents/tools/dmri-rule3-compare.ts) and of both with rule 3's own run-to-run variation on three of them
 *  (Contents/tools/dmri-rule3-noise.ts): the list changes are about the size of that variation (dmri-review, 2026-10-04). */
export const TRACKING_RULE: TrackingRuleId = 3;

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

/** A mask on a grid (1 inside), e.g. SynthStrip's brain mask of the T1. */
export interface MaskGrid { dims: number[]; ijkToRAS: number[]; data: ArrayLike<number> }

/** A T1-space mask put on the fit's grid: each fit voxel is inside when the mask is inside at its center (nearest mask
 *  voxel). The fit's grid is the T1-aligned one (registration.ts), so no move is involved. */
export function brainFromT1(fit: Pick<TensorFit, "dims" | "ijkToRAS">, mask: MaskGrid): Uint8Array {
  const [nx, ny, nz] = fit.dims, M = fit.ijkToRAS, [mx, my, mz] = mask.dims, A = mask.ijkToRAS;
  // Inverse of the mask's affine (its 3×3 by cofactors, then the translation).
  const a = A[0], b = A[1], c = A[2], d = A[4], e = A[5], f = A[6], g = A[8], h = A[9], i9 = A[10];
  const det = a * (e * i9 - f * h) - b * (d * i9 - f * g) + c * (d * h - e * g);
  const Ri = [(e * i9 - f * h) / det, (c * h - b * i9) / det, (b * f - c * e) / det, (f * g - d * i9) / det, (a * i9 - c * g) / det, (c * d - a * f) / det, (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det];
  const out = new Uint8Array(nx * ny * nz);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = [0, 1, 2].map((r) => M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3] - A[4 * r + 3]);
    const p = Math.round(Ri[0] * x[0] + Ri[1] * x[1] + Ri[2] * x[2]), q = Math.round(Ri[3] * x[0] + Ri[4] * x[1] + Ri[5] * x[2]), r = Math.round(Ri[6] * x[0] + Ri[7] * x[1] + Ri[8] * x[2]);
    if (p >= 0 && q >= 0 && r >= 0 && p < mx && q < my && r < mz && mask.data[(r * my + q) * mx + p] > 0) out[(k * ny + j) * nx + i] = 1;
  }
  return out;
}

/** The fit with its brain mask (seedMask) taken from a T1-space mask: tracking rule 3. Kept only where the diffusion scan
 *  has data (the fit's own mask, TensorFit.mask): SynthStrip's brain runs down the brainstem, below a slab that stops
 *  short (critic, 2026-10-04, finding 6: PAT29, 3,447 voxels of zero fill). Also says how much of the T1's brain lies
 *  outside the diffusion grid altogether (mL), which the grid's margin is meant to keep at zero. */
export function withBrainFromT1(fit: TensorFit, mask: MaskGrid, what = "SynthStrip's brain mask of the T1", fluidMdMax?: number, fluidShellVoxels?: number): { fit: TensorFit; outsideGridMl: number } {
  const t1 = brainFromT1(fit, mask), seedMask = new Uint8Array(t1.length);
  for (let v = 0; v < t1.length; v++) seedMask[v] = t1[v] && fit.mask[v] ? 1 : 0;
  if (fluidMdMax !== undefined) {
    // Rule 5: only in the outer shell -- the voxels within fluidShellVoxels of the brain's outside (6-neighbor steps).
    const shell = fluidShellVoxels !== undefined ? outerShell(seedMask, fit.dims, fluidShellVoxels) : undefined;
    for (let v = 0; v < seedMask.length; v++) if (seedMask[v] && fit.md[v] > fluidMdMax && (!shell || shell[v])) seedMask[v] = 0;
  }
  return { fit: { ...fit, seedMask, seedMaskRule: what }, outsideGridMl: maskOutsideGrid(fit, mask) };
}

/** The volume (mL) of a mask's inside voxels whose centers fall outside a grid. */
export function maskOutsideGrid(grid: Pick<TensorFit, "dims" | "ijkToRAS">, mask: MaskGrid): number {
  const [nx, ny, nz] = grid.dims, M = grid.ijkToRAS, [mx, my, mz] = mask.dims, A = mask.ijkToRAS;
  const a = M[0], b = M[1], c = M[2], d = M[4], e = M[5], f = M[6], g = M[8], h = M[9], i9 = M[10];
  const det = a * (e * i9 - f * h) - b * (d * i9 - f * g) + c * (d * h - e * g);
  const Ri = [(e * i9 - f * h) / det, (c * h - b * i9) / det, (b * f - c * e) / det, (f * g - d * i9) / det, (a * i9 - c * g) / det, (c * d - a * f) / det, (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det];
  const voxMl = Math.abs(A[0] * (A[5] * A[10] - A[6] * A[9]) - A[1] * (A[4] * A[10] - A[6] * A[8]) + A[2] * (A[4] * A[9] - A[5] * A[8])) / 1000;
  let out = 0;
  for (let k = 0; k < mz; k++) for (let j = 0; j < my; j++) for (let i = 0; i < mx; i++) {
    if (!(mask.data[(k * my + j) * mx + i] > 0)) continue;
    const x0 = A[0] * i + A[1] * j + A[2] * k + A[3] - M[3], x1 = A[4] * i + A[5] * j + A[6] * k + A[7] - M[7], x2 = A[8] * i + A[9] * j + A[10] * k + A[11] - M[11];
    const p = Math.round(Ri[0] * x0 + Ri[1] * x1 + Ri[2] * x2), q = Math.round(Ri[3] * x0 + Ri[4] * x1 + Ri[5] * x2), r = Math.round(Ri[6] * x0 + Ri[7] * x1 + Ri[8] * x2);
    if (p < 0 || q < 0 || r < 0 || p >= nx || q >= ny || r >= nz) out++;
  }
  return out * voxMl;
}

/** The voxels of a mask within `k` 6-neighbor steps of its outside (a voxel at the grid's edge counts as next to it). */
export function outerShell(mask: Uint8Array, dims: number[], k: number): Uint8Array {
  const [nx, ny, nz] = dims, n = mask.length, depth = new Int16Array(n).fill(-1), queue = new Int32Array(n);
  let head = 0, tail = 0;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const v = (z * ny + y) * nx + x;
    if (!mask[v]) continue;
    const edge = x === 0 || y === 0 || z === 0 || x === nx - 1 || y === ny - 1 || z === nz - 1 ||
      !mask[v - 1] || !mask[v + 1] || !mask[v - nx] || !mask[v + nx] || !mask[v - nx * ny] || !mask[v + nx * ny];
    if (edge) { depth[v] = 1; queue[tail++] = v; }
  }
  while (head < tail) {
    const v = queue[head++], d = depth[v];
    if (d >= k) continue;
    const x = v % nx, y = Math.floor(v / nx) % ny, z = Math.floor(v / (nx * ny));
    for (const [w, ok] of [[v - 1, x > 0], [v + 1, x < nx - 1], [v - nx, y > 0], [v + nx, y < ny - 1], [v - nx * ny, z > 0], [v + nx * ny, z < nz - 1]] as [number, boolean][])
      if (ok && mask[w] && depth[w] < 0) { depth[w] = d + 1; queue[tail++] = w; }
  }
  const out = new Uint8Array(n);
  for (let v = 0; v < n; v++) out[v] = depth[v] > 0 ? 1 : 0;
  return out;
}

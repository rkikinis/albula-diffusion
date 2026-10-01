// THE STEPS FROM A DIFFUSION SCAN TO THE TRACTS NEAR A TUMOR, without the page: the Diffusion module (module.ts) and the
// case-library run (Contents/tools/dmri-cases.ts in the workspace; Ron's rules for tools: "a case library with 10 or
// more cases", half for development, half for testing) call the same code, so what is measured is what the app does.
import type { DiffusionSeries } from "./dwi.ts";
import { applyField, estimateField, fieldAtCenters, type FieldFit } from "./distortion.ts";
import type { TensorFit } from "./tensor.ts";
import { rng, type TractCloudModel } from "./tractcloud/tractcloud.ts";
import { SHORT, type Named } from "./tractcloud/name-tracts.ts";
import { distanceMap } from "./distance.ts";
import type { UkfData } from "./ukf.ts";
import { trackUkfGpu } from "./ukf-gpu.ts";

const matVec = (M: number[], i: number, j: number, k: number): [number, number, number] =>
  [M[0] * i + M[1] * j + M[2] * k + M[3], M[4] * i + M[5] * j + M[6] * k + M[7], M[8] * i + M[9] * j + M[10] * k + M[11]];
function invAffine(m: number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  const R = [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((x) => x / det);
  const t = [m[3], m[7], m[11]];
  return [R[0], R[1], R[2], -(R[0] * t[0] + R[1] * t[1] + R[2] * t[2]), R[3], R[4], R[5], -(R[3] * t[0] + R[4] * t[1] + R[5] * t[2]), R[6], R[7], R[8], -(R[6] * t[0] + R[7] * t[1] + R[8] * t[2]), 0, 0, 0, 1];
}
const yieldNow = () => new Promise((r) => setTimeout(r, 0));

/**
 * DISTORTION CORRECTION (distortion.ts, from the papers): the field is fitted between the mean b = 0 of the scan and of
 * its reversed partner, and every volume of the scan is corrected before the tensor. The phase-encoding axis is
 * MEASURED, not assumed: both in-plane axes are fitted and the one that brings the two scans closer is kept (PAT16: 15%
 * of the difference left along j, 59% along i). Refused, and said, when even the better axis leaves more than half: then
 * the two do not look like a reversed pair. Changes `dwi` in place; returns what was done, in words.
 */
export async function correctWithReversed(dwi: DiffusionSeries, reversed: ArrayLike<number>[], name: string, progress?: (s: string) => void): Promise<string> {
  const meanOf = (vols: ArrayLike<number>[]) => { const o = new Float32Array(vols[0].length); for (const d of vols) for (let v = 0; v < o.length; v++) o[v] += d[v] / vols.length; return o; };
  const plus = meanOf(reversed), minus = meanOf(dwi.volumes.filter((_, i) => dwi.bValues[i] < 50).map((v) => v.data));
  const dims = dwi.volumes[0].dims;
  let best: FieldFit | undefined, bestLeft = Infinity;
  for (const axis of [0, 1] as const) {
    progress?.(`Correcting distortion with ${name}: trying phase encoding along ${axis ? "the columns" : "the rows"}…`);
    await yieldNow();
    const f = estimateField({ dims, plus, minus, axis });
    const left = f.levels.at(-1)?.residual ?? Infinity;
    if (left < bestLeft) { best = f; bestLeft = left; }
  }
  if (!best || bestLeft > 0.5) return `not corrected: ${name} and this scan do not look like a reversed pair (${Math.round(bestLeft * 100)}% of their difference would remain)`;
  for (const v of dwi.volumes) { v.data = applyField(best, v.data, -1); v.dtype = "<f4"; }
  const M = dwi.volumes[0].ijkToRAS, col = best.axis, mmPerVox = Math.hypot(M[col], M[4 + col], M[8 + col]);
  let mx = 0; for (const x of fieldAtCenters(best)) mx = Math.max(mx, Math.abs(x));
  return `corrected with ${name} (shifts up to ${(mx * mmPerVox).toFixed(1)} mm; ${Math.round(bestLeft * 100)}% of the two scans' difference left)`;
}

/** WHOLE-BRAIN TRACKING for naming: TractCloud learned from about 10,000 streamlines a brain. 25,000 starting points in
 *  white matter above FA 0.2 give about 10,000 on PAT16 (measured 2026-09-30: 10,772 in Deno, 11,894 in the app). */
export const WHOLE_BRAIN_SEEDS = 25000, SEED_FA = 0.2;

/**
 * Starting points through the whole brain: up to WHOLE_BRAIN_SEEDS white-matter voxels above SEED_FA, chosen and placed
 * inside their voxel by a seeded generator, so the same scan always gives the same tracts. RAS mm.
 * IN ANATOMICAL ORDER, NOT STORAGE ORDER: the voxels are visited right-left, then front-back, then bottom-top, whatever
 * order the file keeps its slices in. PAT16's DICOM copy stores its slices in the reverse order of its NIfTI file (the
 * same grid in space); drawn in storage order, the two gave different seeds and so different tracts (2026-09-30).
 */
export function wholeBrainSeeds(fit: TensorFit): number[][] {
  const dims = fit.dims, M = fit.ijkToRAS;
  // For each RAS axis r, the grid axis that runs most along it, and whether it runs backward.
  const axisOf: number[] = [], flip: boolean[] = [], used = new Set<number>();
  for (let r = 0; r < 3; r++) {
    let best = -1, bv = -1;
    for (let a = 0; a < 3; a++) if (!used.has(a) && Math.abs(M[4 * r + a]) > bv) { bv = Math.abs(M[4 * r + a]); best = a; }
    used.add(best); axisOf.push(best); flip.push(M[4 * r + best] < 0);
  }
  const n = axisOf.map((a) => dims[a]);
  const toGrid = (c: number[]) => { const g = [0, 0, 0]; for (let r = 0; r < 3; r++) g[axisOf[r]] = flip[r] ? n[r] - 1 - c[r] : c[r]; return g; };
  const cand: number[][] = [];
  for (let c2 = 0; c2 < n[2]; c2++) for (let c1 = 0; c1 < n[1]; c1++) for (let c0 = 0; c0 < n[0]; c0++) {
    const g = toGrid([c0, c1, c2]), v = (g[2] * dims[1] + g[1]) * dims[0] + g[0];
    if (fit.mask[v] && fit.fa[v] > SEED_FA) cand.push([c0, c1, c2]);
  }
  const r = rng(20260930), count = Math.min(WHOLE_BRAIN_SEEDS, cand.length), out: number[][] = [];
  for (let i = 0; i < count; i++) {
    const j = i + Math.floor(r() * (cand.length - i)); const t = cand[j]; cand[j] = cand[i]; cand[i] = t;
    const c = cand[i].map((x) => x + r() - 0.5), g = toGrid(c);            // jitter in the anatomical frame too
    out.push(matVec(M, g[0], g[1], g[2]));
  }
  return out;
}

/** UKF ON THE GRAPHICS CARD (ukf-gpu.ts) from RAS seeds, in batches of 2,000 (short command buffers; the page answers
 *  between them). Streamlines in RAS mm. */
export async function trackUkfSeeds(device: GPUDevice, data: UkfData, seedsRAS: number[][], stoppingFA: number, onBatch?: (done: number) => void | Promise<void>): Promise<Float32Array[]> {
  const R = invAffine(data.ijkToRAS);
  const ijk = seedsRAS.map(([x, y, z]) => [R[0] * x + R[1] * y + R[2] * z + R[3], R[4] * x + R[5] * y + R[6] * z + R[7], R[8] * x + R[9] * y + R[10] * z + R[11]]);
  const out: Float32Array[] = [], BATCH = 2000;
  for (let s = 0; s < ijk.length; s += BATCH) {
    const r = await trackUkfGpu(device, data, ijk.slice(s, s + BATCH), { stoppingFA });
    for (const fb of r.fibers) out.push(fb.points);
    await onBatch?.(Math.min(1, (s + BATCH) / ijk.length));
    await yieldNow();
  }
  return out;
}

/** A structure on a grid: which voxels belong to it. */
export interface Structure { dims: number[]; ijkToRAS: number[]; inside: (index: number) => boolean }

/** Each streamline's closest distance to a structure, in mm (0 = it enters it); farther than `padMm` counts as Infinity
 *  (the distance map covers the structure's box grown by that much, not the whole grid). */
export async function streamlineDistances(s: Structure, sl: Float32Array[], padMm: number): Promise<Float64Array> {
  const out = new Float64Array(sl.length).fill(Infinity);
  const [sx, sy, sz] = s.dims, S = s.ijkToRAS, Sinv = invAffine(S);
  const lo = [sx, sy, sz], hi = [-1, -1, -1];
  for (let k = 0; k < sz; k++) for (let j = 0; j < sy; j++) for (let i = 0; i < sx; i++) {
    if (!s.inside((k * sy + j) * sx + i)) continue;
    lo[0] = Math.min(lo[0], i); lo[1] = Math.min(lo[1], j); lo[2] = Math.min(lo[2], k); hi[0] = Math.max(hi[0], i); hi[1] = Math.max(hi[1], j); hi[2] = Math.max(hi[2], k);
  }
  if (hi[0] < 0) return out;
  const full = [sx, sy, sz];
  for (let a = 0; a < 3; a++) {
    const pad = Math.ceil(padMm / Math.hypot(S[a], S[4 + a], S[8 + a])) + 1;
    lo[a] = Math.max(0, lo[a] - pad); hi[a] = Math.min(full[a] - 1, hi[a] + pad);
  }
  const bx = hi[0] - lo[0] + 1, by = hi[1] - lo[1] + 1, bz = hi[2] - lo[2] + 1;
  const dm = distanceMap((b) => {
    const i = b % bx + lo[0], j = Math.floor(b / bx) % by + lo[1], k = Math.floor(b / (bx * by)) + lo[2];
    return s.inside((k * sy + j) * sx + i);
  }, [bx, by, bz], S);
  for (let n = 0; n < sl.length; n++) {
    const p = sl[n]; let d = Infinity;
    for (let q = 0; q < p.length; q += 3) {
      const v = matVec(Sinv, p[q], p[q + 1], p[q + 2]);
      const i = Math.round(v[0]) - lo[0], j = Math.round(v[1]) - lo[1], k = Math.round(v[2]) - lo[2];
      if (i < 0 || j < 0 || k < 0 || i >= bx || j >= by || k >= bz) continue;
      d = Math.min(d, dm[(k * by + j) * bx + i]);
    }
    out[n] = d <= padMm ? d : Infinity;
    if (n % 2000 === 1999) await yieldNow();
  }
  return out;
}

/** One named tract (a tract on one side): its streamlines, its closest distance, and how many come within the margin. */
export interface NearTract { tract: number; side: number; idx: number[]; d: number; within: number }
/** `faint`: the far tracts that DO come within the margin, with fewer than the minimum of streamlines (shown in gray,
 *  hidden; Ron, 2026-10-01: the right uncinate came within reach with 4). `total`: a tract's streamlines on one side, for
 *  comparing sides. */
export interface Sorted { near: NearTract[]; far: NearTract[]; faint: NearTract[]; unnamedNear: number[]; unnamedFar: number[]; total: (tract: number, side: number) => number }

/** How many of a tract's streamlines must come within the margin for it to count as near (Ron, 2026-10-01: "yes for
 *  now", on the case library's development half: 43% of the tracts listed with "any streamline" had fewer than 5). */
export const MIN_NEAR_STREAMLINES = 5;

/** Sort a named whole-brain tractography by distance to a structure: the named tracts at least `minStreamlines` of
 *  whose streamlines come within `withinMm` (closest first), the others, and the unnamed streamlines (TractCloud's
 *  Other, and those too short to name). */
export function sortByDistance(model: TractCloudModel, named: Named, dist: Float64Array, withinMm: number, minStreamlines = MIN_NEAR_STREAMLINES): Sorted {
  const OTHER = model.json.tracts.length - 1, by = new Map<string, NearTract>(), unnamedNear: number[] = [], unnamedFar: number[] = [];
  for (let i = 0; i < named.tract.length; i++) {
    const t = named.tract[i];
    if (t === SHORT || t === OTHER) { (dist[i] <= withinMm ? unnamedNear : unnamedFar).push(i); continue; }
    const key = `${t}:${named.side[i]}`;
    const e = by.get(key) ?? by.set(key, { tract: t, side: named.side[i], idx: [], d: Infinity, within: 0 }).get(key)!;
    e.idx.push(i); e.d = Math.min(e.d, dist[i]); if (dist[i] <= withinMm) e.within++;
  }
  const all = [...by.values()];
  const isNear = (e: NearTract) => e.d <= withinMm && e.within >= minStreamlines;
  const far = all.filter((e) => !isNear(e)), faint = far.filter((e) => e.d <= withinMm && e.within > 0).sort((a, b) => a.d - b.d || b.within - a.within);
  return { near: all.filter(isNear).sort((a, b) => a.d - b.d || b.within - a.within), far, faint, unnamedNear, unnamedFar,
    total: (tract, side) => by.get(`${tract}:${side}`)?.idx.length ?? 0 };
}

/** The other side of a tract: right for left, left for right; 0 (a tract across the midline) has none. */
export const otherSide = (side: number) => -side;

/** How many starting points a voxel gets in "Add lines": O'Donnell et al. 2017 seeded tumor patients at 20 a voxel
 *  (NeuroImage: Clinical 13:138). At most as many in one press as the whole-brain run, so a press takes about as long
 *  as the first one (PAT16 in Deno, 2026-10-01: 21,000 seeds tracked in 49 s, 57,000 in 144 s). */
export const MORE_PER_VOXEL = 20, MORE_MAX_SEEDS = WHOLE_BRAIN_SEEDS;

/**
 * ADD LINES TO CHOSEN TRACTS (Ron, 2026-10-01: "artificially prop up tracts like the right uncinate by doing a second run
 * with more seed points just in the tracts that were selected by the user"): every voxel the given streamlines pass
 * through gets `perVoxel` starting points, placed at random inside it by a seeded generator. Over `max` in all, every
 * voxel gets fewer (at least one), and past that a seeded sample of the voxels is kept. RAS mm.
 */
export function denseSeeds(strands: Float32Array[], grid: { dims: number[]; ijkToRAS: number[] }, perVoxel = MORE_PER_VOXEL, max = MORE_MAX_SEEDS, seed = 20261001): number[][] {
  const M = grid.ijkToRAS, [nx, ny, nz] = grid.dims;
  // RAS -> voxel: the inverse of the 3x3 part, then the offset.
  const a = M[0], b = M[1], c = M[2], d = M[4], e = M[5], f = M[6], g = M[8], h = M[9], k = M[10];
  const det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
  const R = [(e * k - f * h) / det, (c * h - b * k) / det, (b * f - c * e) / det, (f * g - d * k) / det, (a * k - c * g) / det, (c * d - a * f) / det, (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det];
  const voxels = new Set<number>();
  for (const s of strands) for (let i = 0; i < s.length; i += 3) {
    const x = s[i] - M[3], y = s[i + 1] - M[7], z = s[i + 2] - M[11];
    const vi = Math.round(R[0] * x + R[1] * y + R[2] * z), vj = Math.round(R[3] * x + R[4] * y + R[5] * z), vk = Math.round(R[6] * x + R[7] * y + R[8] * z);
    if (vi >= 0 && vj >= 0 && vk >= 0 && vi < nx && vj < ny && vk < nz) voxels.add((vk * ny + vj) * nx + vi);
  }
  const r = rng(seed), list = [...voxels].sort((p, q) => p - q);
  const each = Math.max(1, Math.min(perVoxel, Math.floor(max / Math.max(1, list.length))));
  if (list.length * each > max) {
    for (let i = 0; i < max; i++) { const j = i + Math.floor(r() * (list.length - i)); const t = list[i]; list[i] = list[j]; list[j] = t; }
    list.length = max;
  }
  const out: number[][] = [];
  for (const v of list) {
    const vi = v % nx, vj = Math.floor(v / nx) % ny, vk = Math.floor(v / (nx * ny));
    for (let n = 0; n < each; n++) {
      const p = vi + r() - 0.5, q = vj + r() - 0.5, w = vk + r() - 0.5;
      out.push([M[0] * p + M[1] * q + M[2] * w + M[3], M[4] * p + M[5] * q + M[6] * w + M[7], M[8] * p + M[9] * q + M[10] * w + M[11]]);
    }
  }
  return out;
}

/** A tract's name in words, with its side. */
export const tractName = (model: TractCloudModel, tract: number, side: number) => `${model.json.tracts[tract].name}${side > 0 ? ", right" : side < 0 ? ", left" : ""}`;

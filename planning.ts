// THE STEPS FROM A DIFFUSION SCAN TO THE TRACTS NEAR A TUMOR, without the page: the Diffusion module (module.ts) and the
// case-library run (Contents/tools/dmri-cases.ts in the workspace; Ron's rules for tools: "a case library with 10 or
// more cases", half for development, half for testing) call the same code, so what is measured is what the app does.
import type { DiffusionSeries } from "./dwi.ts";
import { applyField, estimateField, estimateFieldWithMotion, fieldAtCenters, type FieldFit } from "./distortion.ts";
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

/** A grid: its size and where its voxels are (voxel index → RAS mm). */
export interface Grid { dims: number[]; ijkToRAS: number[] }

/** Do two grids put their voxels in the same places (to 0.001 mm and 0.001 in the axes)? */
export const sameGrid = (a: Grid, b: Grid) => a.dims.every((d, i) => d === b.dims[i]) && a.ijkToRAS.every((v, i) => Math.abs(v - b.ijkToRAS[i]) < 1e-3);

/**
 * `data` on grid `from`, sampled at the voxels of grid `to` through the scanner's coordinates (trilinear; 0 outside).
 * Used for a reversed phase-encoding scan whose slab the scanner placed differently -- 11 of ds001226's 29 people have a
 * partner about 2.3 mm and 0.8° away (measured 2026-10-02). Mike Halle's pipeline aligns it this way and found it in 4 of
 * his 12; Albula paired those voxel by voxel in the case runs and refused them in the app.
 */
export function resampleInto(data: ArrayLike<number>, from: Grid, to: Grid): Float32Array {
  const [nx, ny, nz] = from.dims, [mx, my, mz] = to.dims;
  const R = invAffine(from.ijkToRAS), M = to.ijkToRAS;
  const A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map((k) => { const r = Math.floor(k / 4), c = k % 4;
    return R[4 * r] * M[c] + R[4 * r + 1] * M[4 + c] + R[4 * r + 2] * M[8 + c] + (c === 3 ? R[4 * r + 3] : 0); });   // to-ijk → from-ijk
  const out = new Float32Array(mx * my * mz);
  const at = (i: number, j: number, k: number) => data[(k * ny + j) * nx + i] as number;
  for (let k = 0; k < mz; k++) for (let j = 0; j < my; j++) for (let i = 0; i < mx; i++) {
    const x = A[0] * i + A[1] * j + A[2] * k + A[3], y = A[4] * i + A[5] * j + A[6] * k + A[7], z = A[8] * i + A[9] * j + A[10] * k + A[11];
    const x0 = Math.floor(x), y0 = Math.floor(y), z0 = Math.floor(z);
    if (x0 < 0 || y0 < 0 || z0 < 0 || x0 > nx - 1 || y0 > ny - 1 || z0 > nz - 1) continue;
    const x1 = Math.min(x0 + 1, nx - 1), y1 = Math.min(y0 + 1, ny - 1), z1 = Math.min(z0 + 1, nz - 1);
    const fx = x - x0, fy = y - y0, fz = z - z0;
    const c00 = at(x0, y0, z0) * (1 - fx) + at(x1, y0, z0) * fx, c10 = at(x0, y1, z0) * (1 - fx) + at(x1, y1, z0) * fx;
    const c01 = at(x0, y0, z1) * (1 - fx) + at(x1, y0, z1) * fx, c11 = at(x0, y1, z1) * (1 - fx) + at(x1, y1, z1) * fx;
    out[(k * my + j) * mx + i] = (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
  }
  return out;
}

/**
 * WHERE THE TIME GOES, step by step, in Mike Halle's stages (his mail, 2026-10-02): read, distortion (align the partner,
 * estimate the field, apply it), mask and tensor, tracking (with its own breakdown), naming, distances, drawing. Milliseconds;
 * a step not run is absent. Said in the status line, so it reaches the session log too (Ron: "extend the timing reporting").
 */
export interface StageTimes {
  read?: number; align?: number; field?: number; apply?: number; register?: number; resample?: number; fit?: number;
  seeds?: number; track?: number; trackDetail?: TrackTiming & { data?: number };
  name?: number; distances?: number; draw?: number; total?: number;
}
const sec = (ms: number) => `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
export function stageText(t: StageTimes): string {
  const d = t.trackDetail;
  const parts: string[] = [];
  const add = (label: string, v?: number, extra = "") => { if (v !== undefined) parts.push(`${label} ${sec(v)}${extra}`); };
  add("read", t.read); add("align partner", t.align); add("distortion field", t.field); add("apply field", t.apply); add("align to the T1", t.register); add("onto the T1", t.resample); add("mask + tensor", t.fit);
  add("seeds", t.seeds);
  add("tracking", t.track, d ? ` (signal ${sec(d.data ?? 0)}, seeds ${sec(d.prepare)}, card ${sec(d.gpu)}, fibers ${sec(d.assemble)}, page ${sec(d.between)})` : "");
  add("naming", t.name); add("distances", t.distances); add("drawing", t.draw);
  return parts.join(" · ") + (t.total !== undefined ? ` — ${sec(t.total)} in all` : "");
}

/**
 * DISTORTION CORRECTION (distortion.ts, from the papers): the field is fitted between the mean b = 0 of the scan and of
 * its reversed partner, and every volume of the scan is corrected before the tensor. The phase-encoding axis is
 * MEASURED, not assumed: both in-plane axes are fitted and the one that brings the two scans closer is kept (PAT16: 15%
 * of the difference left along j, 59% along i). Refused, and said, when even the better axis leaves more than half: then
 * the two do not look like a reversed pair. A partner on a grid of its own (`partnerGrid`) is first sampled onto the
 * scan's grid through the scanner's coordinates. Changes `dwi` in place; returns what was done, in words; `times` gets
 * align, field and apply.
 * RULE 2 (distortion.ts, 2026-10-03, the default): the axis is chosen with the field alone (rule 1, about a second each),
 * then the field is fitted again on that axis together with the partner's movement between the two scans (about 4 s).
 * THE SCANNER'S RECORD FIRST (2026-10-03): where the phase-encoding directions are known (`phaseEncoding`, BIDS's
 * "i", "j-", … along the voxel axes of the two grids), a pair that is not reversed along one axis is refused, and a
 * reversed one is fitted on its recorded axis only. Found by Mike Halle's tractline (its ds001226 loader): PAT03's and
 * CON02's "PA" scans are phase-encoded left-right ("i-") against the AP's front-back ("j-"); measured alone, the axes
 * let PAT03 through (47% of the difference left, under the 50% refusal) while a true pair, CON01, leaves 40% -- the
 * residual cannot tell them apart.
 */
export async function correctWithReversed(dwi: DiffusionSeries, reversed: ArrayLike<number>[], name: string, progress?: (s: string) => void, opts: { partnerGrid?: Grid; times?: StageTimes; /** distortion.ts DISTORTION_RULE; default 2. */ rule?: 1 | 2; /** The recorded phase-encoding directions, BIDS style ("j-"), when known. */ phaseEncoding?: { scan?: string; partner?: string }; /** false: the field is found but not applied; it is handed back in `field` (registration.ts resampleOntoT1 applies it with the move to the T1, in one resampling). */ apply?: boolean; field?: { fit?: FieldFit; sign?: 1 | -1 } } = {}): Promise<string> {
  // Either form: BIDS's signed voxel axis ("j-"), or DICOM's In-plane Phase Encoding Direction (ROW = the voxel axis i,
  // COL = j; no sign -- then only the axis is checked). diffusion-vendors.ts phaseEncodingOf reads both from a file.
  // COMPARED IN THE PATIENT, NOT BY NAME (critic, 2026-10-03, finding 2): each record names an axis of ITS OWN grid; the
  // partner may be stored on another one (partnerGrid), so each is turned into a direction in patient space through its
  // own grid before they are compared, and only then is the scan's own voxel axis taken for the fit.
  const pe = opts.phaseEncoding, axisOf = (d: string) => d === "ROW" ? 0 : d === "COL" ? 1 : "ijk".indexOf(d[0]), signed = (d: string) => /^[ijk]-?$/.test(d);
  const scanGrid = dwi.volumes[0].ijkToRAS;
  /** The record's direction in patient space (unit; the sign only for a signed record). */
  const dirOf = (d: string, M: number[]) => { const c = axisOf(d); if (c < 0) return undefined; const v = [M[c], M[4 + c], M[8 + c]], l = Math.hypot(v[0], v[1], v[2]) || 1, s = d.endsWith("-") ? -1 : 1; return v.map((x) => s * x / l); };
  let recordedAxis: 0 | 1 | 2 | undefined;
  if (pe?.scan && pe.partner) {
    const ds = dirOf(pe.scan, scanGrid), dp = dirOf(pe.partner, opts.partnerGrid?.ijkToRAS ?? scanGrid);
    if (ds && dp) {
      const dot = ds[0] * dp[0] + ds[1] * dp[1] + ds[2] * dp[2];
      // Along one line in the patient: within 30°. A reversed pair: opposite, when both records carry a sign.
      const parallel = Math.abs(dot) >= Math.cos(Math.PI / 6), same = signed(pe.scan) && signed(pe.partner) && dot > 0;
      const words = (d: number[]) => { const k = [0, 1, 2].reduce((m, i) => (Math.abs(d[i]) > Math.abs(d[m]) ? i : m), 0); return ["left-right", "front-back", "top-bottom"][k]; };
      if (!parallel || same) return `not corrected: the scanner's record says ${name} was taken with its distortion ${parallel ? "in the same direction as" : `${words(dp)}, against ${words(ds)} for`} this scan, so the two are not a reversed pair`;
      recordedAxis = axisOf(pe.scan) as 0 | 1 | 2;
    }
  }
  const meanOf = (vols: ArrayLike<number>[]) => { const o = new Float32Array(vols[0].length); for (const d of vols) for (let v = 0; v < o.length; v++) o[v] += d[v] / vols.length; return o; };
  const grid: Grid = { dims: dwi.volumes[0].dims, ijkToRAS: dwi.volumes[0].ijkToRAS };
  const ta = performance.now();
  let plus: Float32Array = meanOf(reversed), aligned = "";
  if (opts.partnerGrid && !sameGrid(opts.partnerGrid, grid)) {
    plus = resampleInto(plus, opts.partnerGrid, grid);
    const P = opts.partnerGrid.ijkToRAS, G = grid.ijkToRAS;
    const shift = Math.hypot(P[3] - G[3], P[7] - G[7], P[11] - G[11]);
    let cos = 1; for (const c of [0, 1, 2]) { const a = [P[c], P[4 + c], P[8 + c]], b = [G[c], G[4 + c], G[8 + c]]; cos = Math.min(cos, (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (Math.hypot(...a) * Math.hypot(...b))); }
    aligned = `; ${name} was placed differently by the scanner (${shift.toFixed(1)} mm, ${(Math.acos(Math.min(1, cos)) * 180 / Math.PI).toFixed(1)}°) and was aligned by the scanner's coordinates first`;
    if (opts.times) opts.times.align = performance.now() - ta;
  }
  const minus = meanOf(dwi.volumes.filter((_, i) => dwi.bValues[i] < 50).map((v) => v.data));
  const dims = grid.dims as [number, number, number];
  const tf = performance.now();
  let best: FieldFit | undefined, bestLeft = Infinity;
  for (const axis of (recordedAxis !== undefined ? [recordedAxis] : [0, 1]) as (0 | 1 | 2)[]) {
    progress?.(`Correcting distortion with ${name}: trying phase encoding ${["along the rows", "along the columns", "across the slices"][axis]}…`);
    await yieldNow();
    const f = estimateField({ dims, plus, minus, axis });
    const left = f.levels.at(-1)?.residual ?? Infinity;
    if (left < bestLeft) { best = f; bestLeft = left; }
  }
  let moved = "";
  if (best && bestLeft <= 0.5 && (opts.rule ?? 2) === 2) {
    progress?.(`Correcting distortion with ${name}: allowing for movement between the two scans…`);
    await yieldNow();
    const M0 = grid.ijkToRAS, voxel = [0, 1, 2].map((c) => Math.hypot(M0[c], M0[4 + c], M0[8 + c])) as [number, number, number];
    const f = estimateFieldWithMotion({ dims, plus, minus, axis: best.axis, voxel });
    best = f; bestLeft = f.levels.at(-1)?.residual ?? bestLeft;
    // The movement itself is not reported: the fit is not a measurement of it (critic, 2026-10-03, finding 12: a known
    // movement came back at 3-45% on the phantom); what it improves is the field.
    moved = `; movement between the two scans allowed for`;
  }
  if (opts.times) opts.times.field = performance.now() - tf;
  if (!best || bestLeft > 0.5) return `not corrected: ${name} and this scan do not look like a reversed pair (${Math.round(bestLeft * 100)}% of their difference would remain)${aligned}`;
  const tp = performance.now();
  if (opts.field) { opts.field.fit = best; opts.field.sign = -1; }
  if (opts.apply !== false) for (const v of dwi.volumes) { v.data = applyField(best, v.data, -1); v.dtype = "<f4"; }
  if (opts.times) opts.times.apply = performance.now() - tp;
  const M = dwi.volumes[0].ijkToRAS, col = best.axis, mmPerVox = Math.hypot(M[col], M[4 + col], M[8 + col]);
  let mx = 0; for (const x of fieldAtCenters(best)) mx = Math.max(mx, Math.abs(x));
  return `corrected with ${name} (shifts up to ${(mx * mmPerVox).toFixed(1)} mm; ${Math.round(bestLeft * 100)}% of the two scans' difference left)${moved}${aligned}`;
}

/** WHOLE-BRAIN TRACKING for naming: TractCloud learned from about 10,000 streamlines a brain. Seeded in the brain only
 *  (TensorFit.seedMask, DIPY's median_otsu; since 2026-10-02) 16,000 starting points in white matter above FA 0.2 give
 *  about that many (measured in the record, 2026-10-02). Seeded in the whole head it took 25,000 -- a third of them in the
 *  scalp, making nothing (10,705 on PAT16); the same count from fewer seeds is several seconds less tracking a case. */
export const WHOLE_BRAIN_SEEDS = 16000, SEED_FA = 0.2;

/**
 * Starting points through the whole brain: up to WHOLE_BRAIN_SEEDS white-matter voxels above SEED_FA, chosen and placed
 * inside their voxel by a seeded generator, so the same scan always gives the same tracts. RAS mm.
 * IN ANATOMICAL ORDER, NOT STORAGE ORDER: the voxels are visited right-left, then front-back, then bottom-top, whatever
 * order the file keeps its slices in. PAT16's DICOM copy stores its slices in the reverse order of its NIfTI file (the
 * same grid in space); drawn in storage order, the two gave different seeds and so different tracts (2026-09-30).
 */
export function wholeBrainSeeds(fit: TensorFit, count = WHOLE_BRAIN_SEEDS, seed = 20260930): number[][] {
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
    if ((fit.seedMask ?? fit.mask)[v] && fit.mask[v] && fit.fa[v] > SEED_FA) cand.push([c0, c1, c2]);   // in the brain (seedMask), tracked in the head (mask)
  }
  const r = rng(seed), take = Math.min(count, cand.length), out: number[][] = [];
  for (let i = 0; i < take; i++) {
    const j = i + Math.floor(r() * (cand.length - i)); const t = cand[j]; cand[j] = cand[i]; cand[i] = t;
    const c = cand[i].map((x) => x + r() - 0.5), g = toGrid(c);            // jitter in the anatomical frame too
    out.push(matVec(M, g[0], g[1], g[2]));
  }
  return out;
}

/** UKF ON THE GRAPHICS CARD (ukf-gpu.ts) from RAS seeds, in batches of 2,000 (short command buffers; the page answers
 *  between them). Streamlines in RAS mm. */
/** Where tracking's time goes, ms, summed over the batches (2026-10-01: Ron's WebKit window took 39.8 s where Chrome
 *  took 18.6 s on the same case; this says which part): `prepare` the seeds on the processor, `gpu` the card's steps
 *  and reading them back, `assemble` the fibers on the processor, `between` the page's own work between batches
 *  (drawing, the progress shown). */
export interface TrackTiming { prepare: number; gpu: number; assemble: number; between: number }

export async function trackUkfSeeds(device: GPUDevice, data: UkfData, seedsRAS: number[][], stoppingFA: number, onBatch?: (done: number) => void | Promise<void>, timing?: TrackTiming, gpuOpts: Partial<Parameters<typeof trackUkfGpu>[3]> = {}): Promise<Float32Array[]> {
  const R = invAffine(data.ijkToRAS);
  const ijk = seedsRAS.map(([x, y, z]) => [R[0] * x + R[1] * y + R[2] * z + R[3], R[4] * x + R[5] * y + R[6] * z + R[7], R[8] * x + R[9] * y + R[10] * z + R[11]]);
  // 10,000 SEEDS A CALL, 64 STEPS A DISPATCH (2026-10-03; was 2,000 and 16): every brain voxel a seed (tracking rule 2)
  // made the old setting take 108 s on PAT16. Measured on 18,765 of its seeds: 2,000 / 16 took 22.6 s in 466 dispatches
  // of 48 ms; 10,000 / 64 7.2 s in 28 dispatches of 0.26 s; 20,000 / 64 6.8 s (0.48 s each); 20,000 / 128 5.9 s (0.84 s)
  // -- the fibers bit for bit the same in every one (the card does the same steps; only how many it is given at once
  // changes). A dispatch is kept near a quarter of a second, well inside what macOS's watchdog allows a window's card
  // (2026-09-23: a long command buffer killed the views); the page answers between dispatches.
  // NOT held to a short time per dispatch: tried 2026-10-03 after Ron's window lost its card (macOS's watchdog, tracking
  // beside the 3D view's solid anatomy) -- each dispatch reloads and stores every fiber's state, so 50 ms dispatches took
  // 111 s where 64 steps took 7.6 s (18,765 PAT16 seeds); and the crash came with the old, already short dispatches.
  // ukf-gpu.ts targetMsPerDispatch stays as an option.
  const g = gpuOpts as { seedsPerCall?: number; batch?: number; stepsPerDispatch?: number };
  const out: Float32Array[] = [], BATCH = g.seedsPerCall ?? 10000;
  gpuOpts = { ...gpuOpts, batch: g.batch ?? 2 * BATCH, stepsPerDispatch: g.stepsPerDispatch ?? 64 };
  for (let s = 0; s < ijk.length; s += BATCH) {
    const r = await trackUkfGpu(device, data, ijk.slice(s, s + BATCH), { ...gpuOpts, stoppingFA });
    for (const fb of r.fibers) out.push(fb.points);
    const t = performance.now();
    await onBatch?.(Math.min(1, (s + BATCH) / ijk.length));
    await yieldNow();
    if (timing) { timing.prepare += r.ms.prepare; timing.gpu += r.ms.gpu; timing.assemble += r.ms.total - r.ms.prepare - r.ms.gpu; timing.between += performance.now() - t; }
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
 *  hidden; Ron, 2026-10-01: the right uncinate came within reach with 4), and those that come within `grayMm` but not
 *  within the margin (since 2026-10-04: the 6-8 mm band). `total`: a tract's streamlines on one side, for comparing
 *  sides. */
export interface Sorted { near: NearTract[]; far: NearTract[]; faint: NearTract[]; unnamedNear: number[]; unnamedFar: number[];
  /** The streamlines that cross the fluid at the brain's edge (outside-brain.ts, when it is on), kept out of every tract. */
  outsideNear: number[]; outsideFar: number[]; total: (tract: number, side: number) => number }

/** How many of a tract's streamlines must come within the margin for it to count as near (Ron, 2026-10-01: "yes for
 *  now", on the case library's development half: 43% of the tracts listed with "any streamline" had fewer than 5). */
export const MIN_NEAR_STREAMLINES = 5;
/** How close a tract must come to the tumor to be listed: 6 mm since 2026-10-04 (8 before). Ron, after the lists at 5 /
 *  6 / 7 / 8 mm on the twelve and on seven library cases (dmri-review, 2026-10-04): "6mm with gray" -- the tracts that come
 *  within the next GRAY_BAND_MM are listed after the near ones, in gray and hidden, so a corticospinal tract listed only
 *  at 8 mm (PAT06's left) is not missed. */
export const NEAR_MM = 6, GRAY_BAND_MM = 2;

/** Sort a named whole-brain tractography by distance to a structure: the named tracts at least `minStreamlines` of
 *  whose streamlines come within `withinMm` (closest first), the others, and the unnamed streamlines (TractCloud's
 *  Other, and those too short to name). */
export function sortByDistance(model: TractCloudModel, named: Named, dist: Float64Array, withinMm: number, minStreamlines = MIN_NEAR_STREAMLINES, grayMm = withinMm, outside?: Uint8Array): Sorted {
  const OTHER = model.json.tracts.length - 1, by = new Map<string, NearTract>(), unnamedNear: number[] = [], unnamedFar: number[] = [], outsideNear: number[] = [], outsideFar: number[] = [];
  for (let i = 0; i < named.tract.length; i++) {
    if (outside?.[i]) { (dist[i] <= withinMm ? outsideNear : outsideFar).push(i); continue; }
    const t = named.tract[i];
    if (t === SHORT || t === OTHER) { (dist[i] <= withinMm ? unnamedNear : unnamedFar).push(i); continue; }
    const key = `${t}:${named.side[i]}`;
    const e = by.get(key) ?? by.set(key, { tract: t, side: named.side[i], idx: [], d: Infinity, within: 0 }).get(key)!;
    e.idx.push(i); e.d = Math.min(e.d, dist[i]); if (dist[i] <= withinMm) e.within++;
  }
  const all = [...by.values()];
  const isNear = (e: NearTract) => e.d <= withinMm && e.within >= minStreamlines;
  const far = all.filter((e) => !isNear(e)), faint = far.filter((e) => e.d <= Math.max(withinMm, grayMm)).sort((a, b) => a.d - b.d || b.within - a.within);
  return { near: all.filter(isNear).sort((a, b) => a.d - b.d || b.within - a.within), far, faint, unnamedNear, unnamedFar, outsideNear, outsideFar,
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

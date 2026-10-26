// THE DIFFUSION SCAN ONTO THE T1 (Ron, 2026-10-02/03: "Why not warp the dmri data and then create the streamlines", and
// "Number three, go"; the plan: Contents/docs/dmri-registration-review-2026-10-02.md, "Ron's answers"). Two parts:
//
//   1. RIGID ALIGNMENT of the corrected b = 0 to the T1, as Mike Halle's tractline does it (t1check.py rigid_start):
//      from the scanner's coordinates (same session), refined rigidly on NORMALIZED GRADIENT FIELDS -- 1 − (n_b0 · n_T1)²
//      averaged over the brain, where n is an image's gradient over its length with an edge floor η (a tenth of the
//      mean gradient length): it compares where edges run, not how bright they are, so a T1 against the T2-weighted
//      b = 0 works (Haber & Modersitzki, "Intensity gradient based registration and fusion of multi-modal images",
//      MICCAI 2006). Images blurred to FWHM 8 then 4 mm. Written from that description, not from his code; checked
//      against his (registration.test.ts) and against BRAINSFit (Slicer's, a check tool).
//      Ours: the T1's gradient is taken once per level on its own grid and sampled at the mapped points, turned back
//      by the rotation (the gradient of T1(T x) is Rᵀ ∇T1(T x)); the optimizer is BFGS on central differences over a
//      fixed sample of brain voxels.
//   2. ONE RESAMPLING onto a grid lined up with the T1 at the diffusion scan's own spacing: each target voxel maps
//      back through the rigid move to the scan as acquired, then through the distortion field along the phase-encoding
//      axis (with its stretch), and is read there once; the gradient directions are turned by the rotation (exact for a
//      rigid move). The streamlines, FA and Color FA are then born in the T1's space; tracking runs as before.
import type { DiffusionSeries } from "./dwi.ts";
import { applyField, fieldAtCenters, type FieldFit } from "./distortion.ts";
import { medianOtsuMask } from "./median-otsu.ts";

/** Let the page answer (a long step in the app; nothing in Deno). */
const yieldNow = () => new Promise<void>((r) => setTimeout(r, 0));

/** The default rule. 1: from the scanner's placement, against the whole T1 (same session). 2: brain centers, the shift, 27
 *  starting tilts, and rule 1's path as a candidate (rigidToT1Wide), against the whole T1. 3: rule 2 against the T1 with
 *  only SynthStrip's brain kept (brainOnlyT1) -- THE DEFAULT since 2026-10-25 (Ron: "rule 2 sounds better", against the
 *  brain-only T1; the size limit dropped, "your recommendation"). The known-move test (96 known moves of 10-40 degrees and
 *  mm on two ds001226 and two HCP cases) found rule 1 32/96, rule 2 85/96, rule 3 96/96 (as FLIRT with a brain-extracted
 *  reference); on the 34 real scans rule 3 agrees with brain-only FLIRT (0.5-2.1 mm) and keeps the 22 same-session answers
 *  (0.06-0.28 mm); the tumor outlines lie inside SynthStrip's brain (22 of 25 lose at most 1%, and keeping them changes the
 *  answer by at most 0.02 mm). WORKING-STATE.md, 2026-10-25. */
export const REGISTRATION_RULE: RegistrationRuleId = 3;
export type RegistrationRuleId = 1 | 2 | 3;

/** A 3D image on a grid: x fastest, ijkToRAS row-major 4×4 (or 3×4). */
export interface Grid3 { dims: [number, number, number]; ijkToRAS: number[]; data: ArrayLike<number> }

/** A rigid move y = R (x − c) + c + t in RAS mm, from the diffusion scan's space to the T1's. */
export interface Rigid { R: number[]; t: [number, number, number]; c: [number, number, number] }

export const spacingOf = (M: number[]): [number, number, number] => [0, 1, 2].map((c) => Math.hypot(M[c], M[4 + c], M[8 + c])) as [number, number, number];
export function inv4(M: number[]): number[] {
  const a = [[M[0], M[1], M[2]], [M[4], M[5], M[6]], [M[8], M[9], M[10]]];
  const det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
  const c = (r: number, k: number) => { const rr = [0, 1, 2].filter((x) => x !== r), cc = [0, 1, 2].filter((x) => x !== k); return ((r + k) % 2 ? -1 : 1) * (a[rr[0]][cc[0]] * a[rr[1]][cc[1]] - a[rr[0]][cc[1]] * a[rr[1]][cc[0]]); };
  const I = [0, 1, 2].map((i) => [0, 1, 2].map((j) => c(j, i) / det));
  const t = [M[3], M[7], M[11]];
  return [...I[0], -(I[0][0] * t[0] + I[0][1] * t[1] + I[0][2] * t[2]), ...I[1], -(I[1][0] * t[0] + I[1][1] * t[1] + I[1][2] * t[2]), ...I[2], -(I[2][0] * t[0] + I[2][1] * t[1] + I[2][2] * t[2]), 0, 0, 0, 1];
}
/** R = Rz·Ry·Rx from three angles (radians). */
export function rotation(a: number, b: number, c: number): number[] {
  const ca = Math.cos(a), sa = Math.sin(a), cb = Math.cos(b), sb = Math.sin(b), cc = Math.cos(c), sc = Math.sin(c);
  return [cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa, sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa, -sb, cb * sa, cb * ca];
}
export function applyRigid(T: Rigid, p: ArrayLike<number>): [number, number, number] {
  const d = [p[0] - T.c[0], p[1] - T.c[1], p[2] - T.c[2]], R = T.R;
  return [R[0] * d[0] + R[1] * d[1] + R[2] * d[2] + T.c[0] + T.t[0], R[3] * d[0] + R[4] * d[1] + R[5] * d[2] + T.c[1] + T.t[1], R[6] * d[0] + R[7] * d[1] + R[8] * d[2] + T.c[2] + T.t[2]];
}
/** The rotation angle of a rigid move (degrees) and its translation's length at its center (mm). */
export function rigidSize(T: Rigid): { degrees: number; mm: number } {
  const tr = T.R[0] + T.R[4] + T.R[8];
  return { degrees: Math.acos(Math.max(-1, Math.min(1, (tr - 1) / 2))) * 180 / Math.PI, mm: Math.hypot(...T.t) };
}

/** Gaussian blur with σ per axis in voxels (separable; the edge value beyond the edges). */
export function blur(img: ArrayLike<number>, dims: [number, number, number], sigma: [number, number, number]): Float32Array {
  const [nx, ny, nz] = dims, n = [nx, ny, nz], st = [1, nx, nx * ny];
  let a = Float32Array.from(img);
  for (let ax = 0; ax < 3; ax++) {
    const s = sigma[ax];
    if (!(s > 0.05)) continue;
    const R = Math.ceil(3 * s), w: number[] = []; let sum = 0;
    for (let d = -R; d <= R; d++) { const x = Math.exp(-d * d / (2 * s * s)); w.push(x); sum += x; }
    for (let i = 0; i < w.length; i++) w[i] /= sum;
    const b = new Float32Array(a.length);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const at = [i, j, k][ax], v = (k * ny + j) * nx + i; let acc = 0;
      for (let d = -R; d <= R; d++) acc += w[d + R] * a[v + (Math.min(n[ax] - 1, Math.max(0, at + d)) - at) * st[ax]];
      b[v] = acc;
    }
    a = b;
  }
  return a;
}
const blurMm = (g: Grid3, fwhmMm: number) => { const s = spacingOf(g.ijkToRAS), sig = fwhmMm / (2 * Math.sqrt(2 * Math.log(2))); return blur(g.data, g.dims, s.map((v) => sig / v) as [number, number, number]); };

/** The image's gradient in RAS mm (central differences on its grid, turned into world axes). */
export function worldGradient(img: Float32Array, g: Grid3): Float32Array {
  const [nx, ny, nz] = g.dims, M = g.ijkToRAS, out = new Float32Array(3 * img.length);
  // ∇_world = J⁻ᵀ ∇_ijk with J the ijk → RAS linear part.
  const I = inv4(M);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = (k * ny + j) * nx + i;
    const di = (img[v + (i + 1 < nx ? 1 : 0)] - img[v - (i > 0 ? 1 : 0)]) / ((i + 1 < nx ? 1 : 0) + (i > 0 ? 1 : 0) || 1);
    const dj = (img[v + (j + 1 < ny ? nx : 0)] - img[v - (j > 0 ? nx : 0)]) / ((j + 1 < ny ? 1 : 0) + (j > 0 ? 1 : 0) || 1);
    const dk = (img[v + (k + 1 < nz ? nx * ny : 0)] - img[v - (k > 0 ? nx * ny : 0)]) / ((k + 1 < nz ? 1 : 0) + (k > 0 ? 1 : 0) || 1);
    for (let r = 0; r < 3; r++) out[3 * v + r] = I[r] * di + I[4 + r] * dj + I[8 + r] * dk;
  }
  return out;
}
/** Trilinear sample of a field with `ch` channels per voxel at voxel position (x, y, z); zeros outside. */
export function sampleAt(f: ArrayLike<number>, dims: [number, number, number], ch: number, x: number, y: number, z: number, out: Float64Array): boolean {
  const [nx, ny, nz] = dims;
  // Zero beyond half a voxel past the outermost voxel centers; within it, the edge voxel (each face of the scan kept
  // whole: critic, 2026-10-03, finding 18).
  if (!(x >= -0.5 && y >= -0.5 && z >= -0.5 && x <= nx - 0.5 && y <= ny - 0.5 && z <= nz - 0.5)) { out.fill(0); return false; }
  x = Math.min(Math.max(x, 0), nx - 1); y = Math.min(Math.max(y, 0), ny - 1); z = Math.min(Math.max(z, 0), nz - 1);
  const i0 = Math.min(Math.floor(x), nx - 2 < 0 ? 0 : nx - 2), j0 = Math.min(Math.floor(y), ny - 2 < 0 ? 0 : ny - 2), k0 = Math.min(Math.floor(z), nz - 2 < 0 ? 0 : nz - 2);
  const fx = x - i0, fy = y - j0, fz = z - k0, sx = nx > 1 ? 1 : 0, sy = ny > 1 ? nx : 0, sz = nz > 1 ? nx * ny : 0, v = (k0 * ny + j0) * nx + i0;
  for (let c = 0; c < ch; c++) {
    const g = (o: number) => f[(v + o) * ch + c];
    const c00 = g(0) * (1 - fx) + g(sx) * fx, c10 = g(sy) * (1 - fx) + g(sy + sx) * fx, c01 = g(sz) * (1 - fx) + g(sz + sx) * fx, c11 = g(sz + sy) * (1 - fx) + g(sz + sy + sx) * fx;
    out[c] = (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
  }
  return true;
}

/** The image averaged in blocks to about `mm` per voxel (integer factors per axis; a block's centre keeps its place). */
export function coarsen(g: Grid3, mm: number): Grid3 {
  const sp = spacingOf(g.ijkToRAS), f = sp.map((v) => Math.max(1, Math.round(mm / v))), [nx, ny, nz] = g.dims;
  if (f.every((x) => x === 1)) return g;
  const d = [0, 1, 2].map((a) => Math.floor(g.dims[a] / f[a])) as [number, number, number], out = new Float32Array(d[0] * d[1] * d[2]);
  for (let k = 0; k < d[2]; k++) for (let j = 0; j < d[1]; j++) for (let i = 0; i < d[0]; i++) {
    let s = 0;
    for (let c = 0; c < f[2]; c++) for (let b = 0; b < f[1]; b++) for (let a = 0; a < f[0]; a++) s += Number(g.data[((k * f[2] + c) * ny + (j * f[1] + b)) * nx + (i * f[0] + a)]);
    out[(k * d[1] + j) * d[0] + i] = s / (f[0] * f[1] * f[2]);
  }
  const M = g.ijkToRAS, o = [0, 1, 2].map((r) => M[4 * r + 3] + M[4 * r] * (f[0] - 1) / 2 + M[4 * r + 1] * (f[1] - 1) / 2 + M[4 * r + 2] * (f[2] - 1) / 2);
  return { dims: d, data: out, ijkToRAS: [M[0] * f[0], M[1] * f[1], M[2] * f[2], o[0], M[4] * f[0], M[5] * f[1], M[6] * f[2], o[1], M[8] * f[0], M[9] * f[1], M[10] * f[2], o[2], 0, 0, 0, 1] };
}

export interface RigidResult { T: Rigid; /** The cost at the result and at the scanner's placement, both at the last level. */ cost: number; costAtStart: number; levels: { fwhmMm: number; iterations: number; cost: number }[]; ms: number }

/**
 * RIGID ALIGNMENT of `moving` (the corrected b = 0, diffusion grid) to `fixed` (the T1), on normalized gradient fields,
 * inside `mask` (on the moving grid: the brain, dilated -- its edge carries the signal). From the scanner's coordinates.
 */
export async function rigidToT1(moving: Grid3, fixedFull: Grid3, mask: Uint8Array, opts: { fwhmMm?: number[]; samples?: number; fixedMm?: number } = {}): Promise<RigidResult> {
  const t0 = performance.now(), P = ngfProblem(moving, fixedFull, mask, opts);
  const levels: RigidResult["levels"] = [];
  let p = [0, 0, 0, 0, 0, 0], costAtStart = NaN, cost = NaN;
  for (const fwhm of opts.fwhmMm ?? [8, 4]) {
    const f = P.level(fwhm);
    // The scanner's placement's cost at THIS level's blur: the last level's is what the result is compared with (costs at
    // different blurs are not comparable).
    costAtStart = f([0, 0, 0, 0, 0, 0]);
    const d = await descend(f, p);
    p = d.p;
    levels.push({ fwhmMm: fwhm, iterations: d.iterations, cost: +d.cost.toFixed(5) });
    cost = d.cost;
  }
  return { T: P.toRigid(p), cost, costAtStart, levels, ms: performance.now() - t0 };
}

/** Rule 2's starting tilts, per axis (degrees): every combination, 27 starts, about the brain's center. */
export const TILT_STARTS_DEG = [-20, 0, 20];

export interface Candidate {
  /** Where it started: "scanner" (rule 1's path, unchanged) or "brain centers" with its tilt. */
  start: string; tiltDeg?: [number, number, number];
  /** Its cost at the finest level, and how far (mean over the sample, mm) it puts the brain from the chosen one. */
  cost: number; apartMm: number;
}

/**
 * RULE 2 (Ron, 2026-10-25: DIPY as the roadmap, with "yes" to building it): a large capture range before the fine stage,
 * for a T1 and a diffusion scan made in different sessions (HCP-Aging: 5-42 mm and 1-26 degrees apart; rule 1 settled 26
 * degrees off on HCA9640889). DIPY's order of steps (dipy/align/_public.py affine_registration: centers of mass, then
 * translation alone, then rigid), with the brains' centers instead of the whole images' (a T1 with neck against a b = 0
 * of the brain moves the center of mass; mri_robust_register's page warns of it, and DIPY with one mask failed 10 of 12):
 *   1. the b = 0 brain's center onto the center of SynthStrip's brain on the T1 (skipped without it);
 *   2. the shift alone, at the coarse blur;
 *   3. FLIRT's lesson, small: 27 starting tilts about the brain's center, each a short rigid search at the coarse blur;
 *      the best three that are not the same answer (more than 2 mm apart) are finished at both blurs;
 *   4. rule 1's own path from the scanner's placement is always a candidate too, so a scan already in place can only gain.
 * The lowest cost at the finest blur wins. All candidates are returned with their cost and distance, for the doubt
 * (alignToT1) and the record. Literature and measurements: Contents/docs/registration-capture-range-literature-2026-10-25.md.
 */
export async function rigidToT1Wide(moving: Grid3, fixedFull: Grid3, mask: Uint8Array, brainT1?: Grid3, opts: { fwhmMm?: number[]; samples?: number; fixedMm?: number } = {}): Promise<RigidResult & { candidates: Candidate[] }> {
  const t0 = performance.now(), P = ngfProblem(moving, fixedFull, mask, opts), levels = opts.fwhmMm ?? [8, 4];
  const coarse = P.level(levels[0]), search = P.level(levels[0], SEARCH_THIN), fine = P.level(levels[levels.length - 1]);
  type Tried = { start: string; tiltDeg?: [number, number, number]; p: number[]; cost: number; levels: RigidResult["levels"] };
  // Each level in turn from `p` (rule 1's path when p is the scanner's placement).
  const run = async (p: number[], fws: number[], ls: RigidResult["levels"] = []) => { for (const fw of fws) { const d = await descend(P.level(fw), p); p = d.p; ls.push({ fwhmMm: fw, iterations: d.iterations, cost: +d.cost.toFixed(5) }); } return { p, levels: ls }; };
  // 4. Rule 1's path, from the scanner's placement.
  const scanner = await run([0, 0, 0, 0, 0, 0], levels);
  const tried: Tried[] = [{ start: "scanner", p: scanner.p, cost: fine(scanner.p), levels: scanner.levels }];
  // 1. Brain center onto brain center.
  let shift = [0, 0, 0];
  if (brainT1) {
    const [bx, by] = brainT1.dims, M = brainT1.ijkToRAS, s = [0, 0, 0]; let n = 0;
    for (let v = 0; v < brainT1.data.length; v++) if (Number(brainT1.data[v]) > 0.5) {
      const i = v % bx, j = Math.floor(v / bx) % by, k = Math.floor(v / (bx * by));
      for (let r = 0; r < 3; r++) s[r] += M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3];
      n++;
    }
    if (n) shift = s.map((x, r) => x / n - P.center[r]);
  }
  // 2. The shift alone.
  const shifted = (await descend(search, [...shift, 0, 0, 0], SHIFT)).p;
  // 3. The tilts: a short search from each on the thinned sample; the best three distinct ones then run the coarse level
  // in full, and those still distinct from each other and from the scanner's path go on to the fine levels.
  const searched: { tiltDeg: [number, number, number]; p: number[]; cost: number }[] = [];
  for (const a of TILT_STARTS_DEG) for (const b of TILT_STARTS_DEG) for (const c of TILT_STARTS_DEG) {
    const q = [shifted[0], shifted[1], shifted[2], ...[a, b, c].map((d) => (d * Math.PI / 180) / PS[3])];
    const d = await descend(search, q, ALL, 10);
    searched.push({ tiltDeg: [a, b, c], p: d.p, cost: d.cost });
  }
  searched.sort((x, y) => x.cost - y.cost);
  const keep: typeof searched = [];
  for (const s of searched) { if (keep.length >= 3) break; if (keep.every((k) => P.apart(k.p, s.p) > SAME_MM)) keep.push(s); }
  const coarseDone: { k: typeof keep[number]; p: number[]; ls: RigidResult["levels"] }[] = [];
  for (const k of keep) {
    const c = await run(k.p, [levels[0]]);
    if (P.apart(c.p, scanner.p) > SAME_MM && coarseDone.every((o) => P.apart(o.p, c.p) > SAME_MM)) coarseDone.push({ k, p: c.p, ls: c.levels });
  }
  for (const o of coarseDone) { const f = await run(o.p, levels.slice(1), o.ls); tried.push({ start: "brain centers", tiltDeg: o.k.tiltDeg, p: f.p, cost: fine(f.p), levels: f.levels }); }
  // The scanner's path stands unless another is clearly better: a scan already in place comes out exactly as under rule 1.
  const best = tried.slice(1).reduce((b, t) => (t.cost < b.cost - CLEARLY_BETTER ? t : b), tried[0]);
  const candidates: Candidate[] = tried.map((t) => ({ start: t.start, ...(t.tiltDeg ? { tiltDeg: t.tiltDeg } : {}), cost: +t.cost.toFixed(5), apartMm: +P.apart(t.p, best.p).toFixed(2) }));
  return { T: P.toRigid(best.p), cost: best.cost, costAtStart: fine([0, 0, 0, 0, 0, 0]), levels: best.levels, ms: performance.now() - t0, candidates };
}
/** Rule 2's search uses every 4th point of the sample; two answers within 2 mm (mean over the brain) are the same answer;
 *  a candidate replaces the scanner's path only when its cost is lower by more than 1e-4 (the doubt's tolerance). */
const SEARCH_THIN = 4, SAME_MM = 2, CLEARLY_BETTER = 1e-4;

/** Parameters: t (mm) and three angles (radians, in units of 0.01 so all six move on a similar scale). */
const PS = [1, 1, 1, 0.01, 0.01, 0.01];
const ALL = [0, 1, 2, 3, 4, 5], SHIFT = [0, 1, 2];

/**
 * THE COST, set up once: the sample (brain voxels of the moving grid, every k-th, fixed so the cost is deterministic for
 * the optimizer), its world points and their center (the rotations turn about it), and per blur level the cost function
 * of the six parameters (made once per level and kept).
 */
function ngfProblem(moving: Grid3, fixedFull: Grid3, mask: Uint8Array, opts: { samples?: number; fixedMm?: number } = {}) {
  // The T1 averaged to about 2 mm first (the finest blur is 4 mm FWHM): its blur and gradient on a 1 mm grid were most
  // of the time (2026-10-03).
  const fixed = coarsen(fixedFull, opts.fixedMm ?? 2);
  const [nx, ny] = moving.dims, Mm = moving.ijkToRAS, Fi = inv4(fixed.ijkToRAS);
  const all: number[] = []; for (let v = 0; v < mask.length; v++) if (mask[v]) all.push(v);
  const step = Math.max(1, Math.floor(all.length / (opts.samples ?? 40000))), idx = all.filter((_, i) => i % step === 0);
  const world = new Float64Array(3 * idx.length);
  idx.forEach((v, s) => { const i = v % nx, j = Math.floor(v / nx) % ny, k = Math.floor(v / (nx * ny)); for (let r = 0; r < 3; r++) world[3 * s + r] = Mm[4 * r] * i + Mm[4 * r + 1] * j + Mm[4 * r + 2] * k + Mm[4 * r + 3]; });
  const c = [0, 0, 0]; for (let s = 0; s < idx.length; s++) for (let r = 0; r < 3; r++) c[r] += world[3 * s + r] / idx.length;
  const center = c as [number, number, number];
  const toRigid = (q: number[]): Rigid => { const s = q.map((x, i) => x * PS[i]); return { R: rotation(s[3], s[4], s[5]), t: [s[0], s[1], s[2]], c: center }; };
  const made = new Map<string, (q: number[]) => number>();
  /** The cost at a blur; `thin` > 1 uses every thin-th point of the sample (rule 2's search: the same points, center and
   *  parameters, fewer of them). */
  const level = (fwhm: number, thin = 1): (q: number[]) => number => {
    const key = `${fwhm}/${thin}`, had = made.get(key);
    if (had) return had;
    const base = thin > 1 ? level(fwhm) : undefined;
    if (base) { const f = thinned.get(fwhm)!(thin); made.set(key, f); return f; }
    // The moving image's normalized gradient at the sample, once per level.
    const mb = blurMm(moving, fwhm), mg = worldGradient(mb, moving);
    let em = 0; for (const v of idx) em += Math.hypot(mg[3 * v], mg[3 * v + 1], mg[3 * v + 2]); em = (em / idx.length) * 0.1;
    const nm = new Float64Array(3 * idx.length);
    idx.forEach((v, s) => { const gx = mg[3 * v], gy = mg[3 * v + 1], gz = mg[3 * v + 2], l = Math.sqrt(gx * gx + gy * gy + gz * gz + em * em); nm[3 * s] = gx / l; nm[3 * s + 1] = gy / l; nm[3 * s + 2] = gz / l; });
    // The fixed image's world gradient on its grid, and its edge floor (over its own non-zero voxels).
    const fb = blurMm(fixed, fwhm), fg = worldGradient(fb, fixed);
    let ef = 0, nf = 0; for (let v = 0; v < fb.length; v += 7) if (fb[v] > 0) { ef += Math.hypot(fg[3 * v], fg[3 * v + 1], fg[3 * v + 2]); nf++; } ef = (ef / Math.max(1, nf)) * 0.1;
    const g3 = new Float64Array(3);
    const cost = (q: number[], thin: number): number => {
      const T = toRigid(q), R = T.R; let s = 0, n = 0;
      for (let k = 0; k < idx.length; k += thin) {
        const y = applyRigid(T, world.subarray(3 * k, 3 * k + 3));
        const X = Fi[0] * y[0] + Fi[1] * y[1] + Fi[2] * y[2] + Fi[3], Y = Fi[4] * y[0] + Fi[5] * y[1] + Fi[6] * y[2] + Fi[7], Z = Fi[8] * y[0] + Fi[9] * y[1] + Fi[10] * y[2] + Fi[11];
        sampleAt(fg, fixed.dims, 3, X, Y, Z, g3);
        // ∇ of T1(T x) with respect to x: Rᵀ ∇T1.
        const gx = R[0] * g3[0] + R[3] * g3[1] + R[6] * g3[2], gy = R[1] * g3[0] + R[4] * g3[1] + R[7] * g3[2], gz = R[2] * g3[0] + R[5] * g3[1] + R[8] * g3[2];
        const l = Math.sqrt(gx * gx + gy * gy + gz * gz + ef * ef), d = (nm[3 * k] * gx + nm[3 * k + 1] * gy + nm[3 * k + 2] * gz) / l;
        s += 1 - d * d; n++;
      }
      return s / n;
    };
    const f = (q: number[]) => cost(q, 1);
    thinned.set(fwhm, (thin: number) => (q: number[]) => cost(q, thin));
    made.set(key, f);
    return f;
  };
  const thinned = new Map<number, (thin: number) => (q: number[]) => number>();
  /** The mean distance (mm) between where two parameter sets put the sample's points. */
  const apart = (a: number[], b: number[]): number => {
    const A = toRigid(a), B = toRigid(b); let s = 0;
    for (let k = 0; k < idx.length; k++) { const x = world.subarray(3 * k, 3 * k + 3), u = applyRigid(A, x), v = applyRigid(B, x); s += Math.hypot(u[0] - v[0], u[1] - v[1], u[2] - v[2]); }
    return s / idx.length;
  };
  return { center, toRigid, level, apart };
}

/** BFGS on central differences, backtracking line search, from `p`; only the parameters in `free` move (the others' gradient is 0, so the approximate inverse Hessian keeps them still). */
async function descend(f: (q: number[]) => number, p: number[], free: number[] = ALL, maxIterations = 60): Promise<{ p: number[]; cost: number; iterations: number }> {
  const H = 1e-3, grad = (q: number[]) => q.map((_, i) => { if (!free.includes(i)) return 0; const a = q.slice(), b = q.slice(); a[i] += H; b[i] -= H; return (f(a) - f(b)) / (2 * H); });
  let B: number[][] = [0, 1, 2, 3, 4, 5].map((i) => [0, 1, 2, 3, 4, 5].map((j) => (i === j ? 1 : 0)));
  let fx = f(p), gx = grad(p), it = 0;
  for (; it < maxIterations; it++) {
    await yieldNow();
    const d = B.map((row) => -row.reduce((s, v, j) => s + v * gx[j], 0));
    let slope = d.reduce((s, v, i) => s + v * gx[i], 0);
    if (!(slope < 0)) { B = B.map((row, i) => row.map((_, j) => (i === j ? 1 : 0))); d.splice(0, 6, ...gx.map((v) => -v)); slope = -gx.reduce((s, v) => s + v * v, 0); }
    let a = 1, q = p, fq = fx;
    for (let ls = 0; ls < 20; ls++) { q = p.map((v, i) => v + a * d[i]); fq = f(q); if (fq <= fx + 1e-4 * a * slope) break; a /= 2; }
    if (!(fq < fx)) break;
    const gq = grad(q), s = q.map((v, i) => v - p[i]), y = gq.map((v, i) => v - gx[i]), sy = s.reduce((acc, v, i) => acc + v * y[i], 0);
    const small = Math.max(...s.map(Math.abs)) < 1e-3 && fx - fq < 1e-7;
    p = q; fx = fq; gx = gq;
    if (sy > 1e-12) {
      const By = B.map((row) => row.reduce((acc, v, j) => acc + v * y[j], 0)), yBy = y.reduce((acc, v, i) => acc + v * By[i], 0);
      B = B.map((row, i) => row.map((v, j) => v + ((sy + yBy) * s[i] * s[j]) / (sy * sy) - (By[i] * s[j] + s[i] * By[j]) / sy));
    }
    if (small) break;
  }
  return { p, cost: fx, iterations: it };
}

/** The field at cell centers (voxels along its axis) and its slope along the axis (the stretch), for sampling anywhere. */
export function fieldWithSlope(fit: FieldFit): { b: Float32Array; slope: Float32Array } {
  const fc = fieldAtCenters(fit), ax = fit.axis, [sx, sy] = fit.dims, slope = new Float32Array(fc.length), stride = [1, sx, sx * sy][ax], m = fit.dims[ax];
  for (let v = 0; v < fc.length; v++) { const q = Math.floor(v / stride) % m, up = q + 1 < m ? fc[v + stride] : fc[v], dn = q > 0 ? fc[v - stride] : fc[v]; slope[v] = (up - dn) / ((q + 1 < m ? 1 : 0) + (q > 0 ? 1 : 0) || 1); }
  return { b: fc, slope };
}

/**
 * THE DIFFUSION SCAN ON A GRID LINED UP WITH THE T1, read once per volume: each target voxel goes back through the rigid
 * move (T1 space → scan space) and, when a distortion field is given, along the phase-encoding axis through it (the
 * correction distortion.ts applyField makes, with its stretch), and the scan as acquired is read there. Target grid: the
 * T1's axes at the scan's own spacing, over the box the scan covers once moved. Gradients turned by the rotation.
 */
export async function resampleOntoT1(dwi: DiffusionSeries, T: Rigid, t1: Pick<Grid3, "dims" | "ijkToRAS">, field?: { fit: FieldFit; sign: 1 | -1 }, opts: { /** The scan's voxels to cover (the brain, on the scan's grid): the box is theirs plus a margin, not the whole field of view. */ cover?: Uint8Array; marginMm?: number; /** Head movement (motion.ts): per volume, the move from the scan's reference to where the head was, and its eddy-current shift d = g·(y − c)/100 mm along e. */ motion?: (Rigid & { ec?: { g: number[]; e: number[] } })[]; /** The registration rule that made the move, for the record. */ rule?: RegistrationRuleId } = {}): Promise<DiffusionSeries> {
  const src = dwi.volumes[0], [sx, sy, sz] = src.dims, Ms = src.ijkToRAS, Si = inv4(Ms), sp = spacingOf(Ms);
  const Mt = t1.ijkToRAS, tsp = spacingOf(Mt), axes = [0, 1, 2].map((c) => [Mt[c] / tsp[c], Mt[4 + c] / tsp[c], Mt[8 + c] / tsp[c]]);
  // THE SPACING keeps the scan's voxel volume (critic, 2026-10-03, finding 7: the smallest spacing in all three directions
  // made a 0.9 × 0.9 × 5 mm scan six times as many voxels -- and, every voxel a seed, six times the tracking): the cube
  // root of the three spacings, which is the spacing itself for an isotropic scan.
  const h = Math.cbrt(sp[0] * sp[1] * sp[2]);
  // THE BOX: the scan's voxels to cover (opts.cover, the brain; else every voxel) moved into T1 space, in the T1's axes,
  // grown by a margin -- a third of the field-of-view box was zero fill (PAT16).
  const pts: number[][] = [], cov = opts.cover;
  if (cov) { for (let k = 0; k < sz; k++) for (let j = 0; j < sy; j++) for (let i = 0; i < sx; i++) if (cov[(k * sy + j) * sx + i]) pts.push(applyRigid(T, [0, 1, 2].map((r) => Ms[4 * r] * i + Ms[4 * r + 1] * j + Ms[4 * r + 2] * k + Ms[4 * r + 3]))); }
  if (!pts.length) for (const a of [0, sx - 1]) for (const b of [0, sy - 1]) for (const c of [0, sz - 1]) pts.push(applyRigid(T, [0, 1, 2].map((r) => Ms[4 * r] * a + Ms[4 * r + 1] * b + Ms[4 * r + 2] * c + Ms[4 * r + 3])));
  // THE MARGIN: 4 voxels (2 until 2026-10-04). Tracking rule 3's brain comes from the T1 (SynthStrip), and a box made from
  // the scan's own median_otsu with 2 voxels left up to 0.26 mL of it outside on 4 of ds001226's 29 cases; 3 voxels left
  // 18 µL on one, 4 none (critic, 2026-10-04, finding 18 and the measurement after it). A fixed margin, not one made from
  // the T1's mask: the maps are made before the mask arrives, and the grid must not depend on when it did. Whole voxels, so
  // the voxel centers inside are the same as with 2 -- but median_otsu's 9-voxel filter reaches the grid's edge, so rule 2's
  // brain, and its tracts, can differ slightly (PAT07: 48,944 streamlines with 2, 49,331 with 4).
  const margin = cov ? (opts.marginMm ?? 4 * h) : 0;
  const lo = [0, 1, 2].map((ax) => { let m = Infinity; for (const p of pts) m = Math.min(m, p[0] * axes[ax][0] + p[1] * axes[ax][1] + p[2] * axes[ax][2]); return m - margin; });
  const hi = [0, 1, 2].map((ax) => { let m = -Infinity; for (const p of pts) m = Math.max(m, p[0] * axes[ax][0] + p[1] * axes[ax][1] + p[2] * axes[ax][2]); return m + margin; });
  const dims = [0, 1, 2].map((ax) => Math.floor((hi[ax] - lo[ax]) / h) + 1) as [number, number, number];
  const origin = [0, 1, 2].map((r) => lo[0] * axes[0][r] + lo[1] * axes[1][r] + lo[2] * axes[2][r]);
  const M = [axes[0][0] * h, axes[1][0] * h, axes[2][0] * h, origin[0], axes[0][1] * h, axes[1][1] * h, axes[2][1] * h, origin[1], axes[0][2] * h, axes[1][2] * h, axes[2][2] * h, origin[2], 0, 0, 0, 1];
  // Inverse rigid (T1 space → scan space): x = Rᵀ (y − c − t) + c.
  const R = T.R, inv = (y: number[]) => { const d = [y[0] - T.c[0] - T.t[0], y[1] - T.c[1] - T.t[1], y[2] - T.c[2] - T.t[2]]; return [R[0] * d[0] + R[3] * d[1] + R[6] * d[2] + T.c[0], R[1] * d[0] + R[4] * d[1] + R[7] * d[2] + T.c[1], R[2] * d[0] + R[5] * d[1] + R[8] * d[2] + T.c[2]]; };
  // The field at cell centers (voxels along its axis) and its derivative along the axis, for sampling anywhere.
  const fd = field ? fieldWithSlope(field.fit) : undefined, fc = fd?.b, dfc = fd?.slope, ax = field?.fit.axis ?? 0;
  // Where each target voxel reads (source voxel coordinates) and with what weight (the stretch), once for all volumes.
  // With head movement (opts.motion, motion.ts): where it lies in the scan's reference space (x) and the field's shift
  // there, kept, and each volume reads through its own move. The field is taken where the head is in the reference:
  // it moves with the head (the susceptibility field is the head's own), as FSL's eddy assumes.
  const mv = opts.motion, n = dims[0] * dims[1] * dims[2], pos = new Float32Array(3 * n), wgt = new Float32Array(n), tmp = new Float64Array(1);
  const xs = mv ? new Float32Array(3 * n) : undefined, shift = mv ? new Float32Array(n) : undefined;
  for (let k = 0; k < dims[2]; k++) for (let j = 0; j < dims[1]; j++) for (let i = 0; i < dims[0]; i++) {
    const v = (k * dims[1] + j) * dims[0] + i;
    const y = [0, 1, 2].map((r) => M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3]), x = inv(y);
    const q = [0, 1, 2].map((r) => Si[4 * r] * x[0] + Si[4 * r + 1] * x[1] + Si[4 * r + 2] * x[2] + Si[4 * r + 3]);
    let w = 1, s = 0;
    if (fc && dfc && field) {
      sampleAt(fc, src.dims, 1, q[0], q[1], q[2], tmp); const b = tmp[0];
      sampleAt(dfc, src.dims, 1, q[0], q[1], q[2], tmp); const db = tmp[0];
      s = field.sign * b; q[ax] += s; w = 1 + field.sign * db;
    }
    pos[3 * v] = q[0]; pos[3 * v + 1] = q[1]; pos[3 * v + 2] = q[2]; wgt[v] = w;
    if (xs && shift) { xs[3 * v] = x[0]; xs[3 * v + 1] = x[1]; xs[3 * v + 2] = x[2]; shift[v] = s; }
  }
  const one = new Float64Array(1);
  const volumes: DiffusionSeries["volumes"] = [];
  for (const [vi, vol] of dwi.volumes.entries()) {
    const out = new Float32Array(n), m = mv?.[vi];
    if (m && xs && shift) {
      // y = R (x − c) + c + t, then into the scan's voxels, then the field's shift along its axis.
      const r = m.R, a = [0, 1, 2].map((q) => m.c[q] + m.t[q] - (r[3 * q] * m.c[0] + r[3 * q + 1] * m.c[1] + r[3 * q + 2] * m.c[2]));
      const A = [0, 1, 2].map((q) => [0, 1, 2].map((p) => Si[4 * q] * r[p] + Si[4 * q + 1] * r[3 + p] + Si[4 * q + 2] * r[6 + p]));
      const b = [0, 1, 2].map((q) => Si[4 * q] * a[0] + Si[4 * q + 1] * a[1] + Si[4 * q + 2] * a[2] + Si[4 * q + 3]);
      const [A0, A1, A2] = A;
      // The eddy-current shift (motion rule 2): d = g·(y − c)/100 mm along e, y where the tissue was; in voxels Si·e d;
      // the brightness corrected for its stretch.
      const g = m.ec?.g ?? [0, 0, 0], e = m.ec?.e ?? [0, 0, 0], ev = [0, 1, 2].map((q) => Si[4 * q] * e[0] + Si[4 * q + 1] * e[1] + Si[4 * q + 2] * e[2]);
      const stretch = 1 + (g[0] * e[0] + g[1] * e[1] + g[2] * e[2]) / 100, ecOn = !!m.ec;
      for (let v = 0; v < n; v++) {
        const x0 = xs[3 * v], x1 = xs[3 * v + 1], x2 = xs[3 * v + 2];
        const q = [A0[0] * x0 + A0[1] * x1 + A0[2] * x2 + b[0], A1[0] * x0 + A1[1] * x1 + A1[2] * x2 + b[1], A2[0] * x0 + A2[1] * x1 + A2[2] * x2 + b[2]];
        if (ecOn) {
          const d0 = x0 - m.c[0], d1 = x1 - m.c[1], d2 = x2 - m.c[2];
          const r0 = r[0] * d0 + r[1] * d1 + r[2] * d2 + m.t[0], r1 = r[3] * d0 + r[4] * d1 + r[5] * d2 + m.t[1], r2 = r[6] * d0 + r[7] * d1 + r[8] * d2 + m.t[2];
          const dd = (g[0] * r0 + g[1] * r1 + g[2] * r2) / 100;
          q[0] += dd * ev[0]; q[1] += dd * ev[1]; q[2] += dd * ev[2];
        }
        q[ax] += shift[v];
        sampleAt(vol.data as ArrayLike<number>, src.dims, 1, q[0], q[1], q[2], one); out[v] = one[0] * wgt[v] * stretch;
      }
    } else for (let v = 0; v < n; v++) { sampleAt(vol.data as ArrayLike<number>, src.dims, 1, pos[3 * v], pos[3 * v + 1], pos[3 * v + 2], one); out[v] = one[0] * wgt[v]; }
    volumes.push({ ...vol, dims, ijkToRAS: M, data: out, dtype: "<f4" } as DiffusionSeries["volumes"][number]);
    if (volumes.length % 8 === 0) await yieldNow();
  }
  // Each gradient turned with the head: into the reference by its volume's move (Rᵥᵀ g), then onto the T1 (R).
  const gradients = dwi.gradients.map((g0, vi) => {
    const m = mv?.[vi]?.R, g = m ? [m[0] * g0[0] + m[3] * g0[1] + m[6] * g0[2], m[1] * g0[0] + m[4] * g0[1] + m[7] * g0[2], m[2] * g0[0] + m[5] * g0[1] + m[8] * g0[2]] : g0;
    return [R[0] * g[0] + R[1] * g[1] + R[2] * g[2], R[3] * g[0] + R[4] * g[1] + R[5] * g[2], R[6] * g[0] + R[7] * g[1] + R[8] * g[2]] as [number, number, number];
  });
  return { ...dwi, volumes, ijkToRAS: M, gradients, source: `${dwi.source}; onto the T1 (registration rule ${opts.rule ?? REGISTRATION_RULE})` };
}

/** Rule 3's target: the T1 with everything outside SynthStrip's brain set to zero (FSL asks for a brain-extracted reference
 *  for FLIRT; the T1's scalp, skull and neck were what pulled rules 1 and 2 to wrong answers). The brain mask is read on
 *  the T1's own grid, or by nearest voxel through world coordinates when its grid differs. */
export function brainOnlyT1(t1: Grid3, brainT1: Grid3): Grid3 {
  const [nx, ny, nz] = t1.dims, out = new Float32Array(nx * ny * nz), same = brainT1.dims.join() === t1.dims.join() && brainT1.ijkToRAS.slice(0, 12).every((v, i) => Math.abs(v - t1.ijkToRAS[i]) < 1e-3);
  const M = t1.ijkToRAS, Bi = inv4(brainT1.ijkToRAS), [bx, by, bz] = brainT1.dims;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = (k * ny + j) * nx + i;
    let inside: boolean;
    if (same) inside = Number(brainT1.data[v]) > 0.5;
    else {
      const w = [0, 1, 2].map((r) => M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3]);
      const I = Math.round(Bi[0] * w[0] + Bi[1] * w[1] + Bi[2] * w[2] + Bi[3]), J = Math.round(Bi[4] * w[0] + Bi[5] * w[1] + Bi[6] * w[2] + Bi[7]), K = Math.round(Bi[8] * w[0] + Bi[9] * w[1] + Bi[10] * w[2] + Bi[11]);
      inside = I >= 0 && J >= 0 && K >= 0 && I < bx && J < by && K < bz && Number(brainT1.data[(K * by + J) * bx + I]) > 0.5;
    }
    if (inside) out[v] = Number(t1.data[v]);
  }
  return { dims: t1.dims, ijkToRAS: t1.ijkToRAS, data: out };
}

/** A move too large to be the same session's -- under rule 1 the alignment is then shown as doubtful (the person aligns by
 *  hand). Rules 2 and 3 do not use it (HCP-Aging's T1 and diffusion scans are 5-42 mm and 1-26 degrees apart). */
export const DOUBTFUL = { mm: 10, degrees: 10 };

/** The b = 0 images' mean, the field applied to each, and the brain in it (median_otsu, grown by 2 voxels: the edge
 *  carries the alignment's signal), on the scan's grid. */
export function scanBrain(dwi: DiffusionSeries, field?: { fit: FieldFit; sign: 1 | -1 }): { b0: Float32Array; mask: Uint8Array } {
  const src = dwi.volumes[0], n = src.dims[0] * src.dims[1] * src.dims[2];
  const b0i = dwi.bValues.map((b, i) => (b < 50 ? i : -1)).filter((i) => i >= 0);
  const b0 = new Float32Array(n);
  for (const i of b0i.length ? b0i : [0]) { const d = field ? applyField(field.fit, dwi.volumes[i].data as ArrayLike<number>, field.sign) : dwi.volumes[i].data as ArrayLike<number>; for (let v = 0; v < n; v++) b0[v] += Number(d[v]) / Math.max(1, b0i.length); }
  const one = { ...dwi, volumes: [{ ...src, data: b0, dtype: "<f4" }], bValues: [0], gradients: [[0, 0, 0]] } as unknown as DiffusionSeries;
  const brain = medianOtsuMask(one, [0]).mask, [nx, ny, nz] = src.dims, mask = new Uint8Array(n);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    let on = 0;
    for (let c = -2; c <= 2 && !on; c++) for (let b = -2; b <= 2 && !on; b++) for (let a = -2; a <= 2 && !on; a++) { const I = i + a, J = j + b, K = k + c; if (I >= 0 && J >= 0 && K >= 0 && I < nx && J < ny && K < nz && brain[(K * ny + J) * nx + I]) on = 1; }
    mask[(k * ny + j) * nx + i] = on;
  }
  return { b0, mask };
}

/**
 * THE WHOLE STEP, for a case run and for the module: the b = 0 images corrected (the field applied to them alone), their
 * mean aligned to the T1 inside the brain (median_otsu of that mean, grown by 2 voxels), and the whole scan read once onto
 * the T1-aligned grid through the move and the field. Returns the new series, the move, and what was done, in words --
 * with a doubt when the move is larger than one session's or the alignment did not improve the match.
 */
export async function alignToT1(dwi: DiffusionSeries, t1: Grid3, field?: { fit: FieldFit; sign: 1 | -1 }, times?: { register?: number; resample?: number },
  /** Head movement (motion.ts): each volume's move, and the b = 0 mean and brain it made (corrected for the field and the movement). */
  motion?: { moves: Rigid[]; b0: Float32Array; mask: Uint8Array },
  /** The rule (default REGISTRATION_RULE), and SynthStrip's brain on the T1: rule 2 starts from its center (without it,
   *  from the scanner's placement); rule 3 aligns to the T1 inside it, and without it is rule 2 -- said in the record and
   *  in `rule` (the import job refuses a case without the brain, so this is the module's path). */
  opts: { rule?: RegistrationRuleId; brainT1?: Grid3 } = {}): Promise<{ dwi: DiffusionSeries; T: Rigid; result: RigidResult & { candidates?: Candidate[] }; said: string; doubt?: string; rule: RegistrationRuleId }> {
  const t0 = performance.now(), src = dwi.volumes[0];
  let b0: Float32Array, mask: Uint8Array;
  if (motion) ({ b0, mask } = motion);
  else ({ b0, mask } = scanBrain(dwi, field));
  const asked = opts.rule ?? REGISTRATION_RULE, rule: RegistrationRuleId = asked === 3 && !opts.brainT1 ? 2 : asked;
  const moving: Grid3 = { dims: src.dims as [number, number, number], ijkToRAS: src.ijkToRAS, data: b0 };
  const result = rule === 3 ? await rigidToT1Wide(moving, brainOnlyT1(t1, opts.brainT1!), mask, opts.brainT1)
    : rule === 2 ? await rigidToT1Wide(moving, t1, mask, opts.brainT1) : await rigidToT1(moving, t1, mask);
  const t1Done = performance.now();
  if (times) times.register = t1Done - t0;
  const out = await resampleOntoT1(dwi, result.T, t1, field, { cover: mask, rule, ...(motion ? { motion: motion.moves } : {}) });
  if (times) times.resample = performance.now() - t1Done;
  const sz = rigidSize(result.T);
  // Rule 1: a move larger than one session's is doubtful. Rules 2 and 3 are made for scans sessions apart, so the size says
  // nothing; their doubt is "the match got worse" (Ron, 2026-10-25: the size limit dropped, "your recommendation"; how much
  // the match improved does not separate -- same-session cases improve 2.7-30%, the failure 6%).
  const tooFar = rule === 1 && (sz.mm > DOUBTFUL.mm || sz.degrees > DOUBTFUL.degrees);
  const doubt = tooFar ? `it had to move ${sz.mm.toFixed(0)} mm and ${sz.degrees.toFixed(0)}°, more than within one visit`
    : result.cost > result.costAtStart + 1e-4 ? "the match got worse" : undefined;     // equal: the scan was already in place (critic, finding 18)
  return { dwi: out, T: result.T, result, rule, said: `aligned to the MRI of the anatomy (moved ${sz.mm.toFixed(1)} mm and ${sz.degrees.toFixed(1)}°${rule !== asked ? "; to the whole MRI: no brain was found on it" : ""})`, ...(doubt ? { doubt } : {}) };
}

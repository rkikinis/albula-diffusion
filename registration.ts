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

export const REGISTRATION_RULE = 1;

/** A 3D image on a grid: x fastest, ijkToRAS row-major 4×4 (or 3×4). */
export interface Grid3 { dims: [number, number, number]; ijkToRAS: number[]; data: ArrayLike<number> }

/** A rigid move y = R (x − c) + c + t in RAS mm, from the diffusion scan's space to the T1's. */
export interface Rigid { R: number[]; t: [number, number, number]; c: [number, number, number] }

const spacingOf = (M: number[]): [number, number, number] => [0, 1, 2].map((c) => Math.hypot(M[c], M[4 + c], M[8 + c])) as [number, number, number];
function inv4(M: number[]): number[] {
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
function worldGradient(img: Float32Array, g: Grid3): Float32Array {
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
function sampleAt(f: ArrayLike<number>, dims: [number, number, number], ch: number, x: number, y: number, z: number, out: Float64Array): boolean {
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
  // The T1 averaged to about 2 mm first (the finest blur is 4 mm FWHM): its blur and gradient on a 1 mm grid were most
  // of the time (2026-10-03).
  const fixed = coarsen(fixedFull, opts.fixedMm ?? 2);
  const t0 = performance.now(), [nx, ny, nz] = moving.dims, Mm = moving.ijkToRAS, Fi = inv4(fixed.ijkToRAS);
  // The sample: brain voxels of the moving grid, every k-th (fixed: the cost is deterministic for the optimizer).
  const all: number[] = []; for (let v = 0; v < mask.length; v++) if (mask[v]) all.push(v);
  const step = Math.max(1, Math.floor(all.length / (opts.samples ?? 40000))), idx = all.filter((_, i) => i % step === 0);
  const world = new Float64Array(3 * idx.length);
  idx.forEach((v, s) => { const i = v % nx, j = Math.floor(v / nx) % ny, k = Math.floor(v / (nx * ny)); for (let r = 0; r < 3; r++) world[3 * s + r] = Mm[4 * r] * i + Mm[4 * r + 1] * j + Mm[4 * r + 2] * k + Mm[4 * r + 3]; });
  let c = [0, 0, 0]; for (let s = 0; s < idx.length; s++) for (let r = 0; r < 3; r++) c[r] += world[3 * s + r] / idx.length;
  const center = c as [number, number, number];
  const levels: RigidResult["levels"] = [];
  let p = [0, 0, 0, 0, 0, 0];                               // t (mm), angles (rad, scaled by 0.01 below)
  const PS = [1, 1, 1, 0.01, 0.01, 0.01];
  const toRigid = (q: number[]): Rigid => { const s = q.map((x, i) => x * PS[i]); return { R: rotation(s[3], s[4], s[5]), t: [s[0], s[1], s[2]], c: center }; };
  let costAtStart = NaN, cost = NaN;
  for (const fwhm of opts.fwhmMm ?? [8, 4]) {
    // The moving image's normalized gradient at the sample, once per level.
    const mb = blurMm(moving, fwhm), mg = worldGradient(mb, moving);
    let em = 0; for (const v of idx) em += Math.hypot(mg[3 * v], mg[3 * v + 1], mg[3 * v + 2]); em = (em / idx.length) * 0.1;
    const nm = new Float64Array(3 * idx.length);
    idx.forEach((v, s) => { const gx = mg[3 * v], gy = mg[3 * v + 1], gz = mg[3 * v + 2], l = Math.sqrt(gx * gx + gy * gy + gz * gz + em * em); nm[3 * s] = gx / l; nm[3 * s + 1] = gy / l; nm[3 * s + 2] = gz / l; });
    // The fixed image's world gradient on its grid, and its edge floor (over its own non-zero voxels).
    const fb = blurMm(fixed, fwhm), fg = worldGradient(fb, fixed);
    let ef = 0, nf = 0; for (let v = 0; v < fb.length; v += 7) if (fb[v] > 0) { ef += Math.hypot(fg[3 * v], fg[3 * v + 1], fg[3 * v + 2]); nf++; } ef = (ef / Math.max(1, nf)) * 0.1;
    const g3 = new Float64Array(3);
    const f = (q: number[]): number => {
      const T = toRigid(q), R = T.R; let s = 0, n = 0;
      for (let k = 0; k < idx.length; k++) {
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
    // The scanner's placement's cost at THIS level's blur: the last level's is what the result is compared with (costs at
    // different blurs are not comparable).
    costAtStart = f([0, 0, 0, 0, 0, 0]);
    // BFGS on central differences, backtracking line search.
    const H = 1e-3, grad = (q: number[]) => q.map((_, i) => { const a = q.slice(), b = q.slice(); a[i] += H; b[i] -= H; return (f(a) - f(b)) / (2 * H); });
    let B = [0, 1, 2, 3, 4, 5].map((i) => [0, 1, 2, 3, 4, 5].map((j) => (i === j ? 1 : 0)));
    let fx = f(p), gx = grad(p), it = 0;
    for (; it < 60; it++) {
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
    levels.push({ fwhmMm: fwhm, iterations: it, cost: +fx.toFixed(5) });
    cost = fx;
  }
  return { T: toRigid(p), cost, costAtStart, levels, ms: performance.now() - t0 };
}

/**
 * THE DIFFUSION SCAN ON A GRID LINED UP WITH THE T1, read once per volume: each target voxel goes back through the rigid
 * move (T1 space → scan space) and, when a distortion field is given, along the phase-encoding axis through it (the
 * correction distortion.ts applyField makes, with its stretch), and the scan as acquired is read there. Target grid: the
 * T1's axes at the scan's own spacing, over the box the scan covers once moved. Gradients turned by the rotation.
 */
export async function resampleOntoT1(dwi: DiffusionSeries, T: Rigid, t1: Pick<Grid3, "dims" | "ijkToRAS">, field?: { fit: FieldFit; sign: 1 | -1 }, opts: { /** The scan's voxels to cover (the brain, on the scan's grid): the box is theirs plus a margin, not the whole field of view. */ cover?: Uint8Array; marginMm?: number; /** Points in T1 space (RAS mm) the box must also hold: the T1's own brain mask (tracking rule 3), so a box made from the scan's median_otsu cannot cut what SynthStrip keeps. */ alsoCover?: number[][] } = {}): Promise<DiffusionSeries> {
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
  if (cov && opts.alsoCover) for (const p of opts.alsoCover) pts.push(p);
  if (!pts.length) for (const a of [0, sx - 1]) for (const b of [0, sy - 1]) for (const c of [0, sz - 1]) pts.push(applyRigid(T, [0, 1, 2].map((r) => Ms[4 * r] * a + Ms[4 * r + 1] * b + Ms[4 * r + 2] * c + Ms[4 * r + 3])));
  const margin = cov ? (opts.marginMm ?? 2 * h) : 0;
  const lo = [0, 1, 2].map((ax) => { let m = Infinity; for (const p of pts) m = Math.min(m, p[0] * axes[ax][0] + p[1] * axes[ax][1] + p[2] * axes[ax][2]); return m - margin; });
  const hi = [0, 1, 2].map((ax) => { let m = -Infinity; for (const p of pts) m = Math.max(m, p[0] * axes[ax][0] + p[1] * axes[ax][1] + p[2] * axes[ax][2]); return m + margin; });
  const dims = [0, 1, 2].map((ax) => Math.floor((hi[ax] - lo[ax]) / h) + 1) as [number, number, number];
  const origin = [0, 1, 2].map((r) => lo[0] * axes[0][r] + lo[1] * axes[1][r] + lo[2] * axes[2][r]);
  const M = [axes[0][0] * h, axes[1][0] * h, axes[2][0] * h, origin[0], axes[0][1] * h, axes[1][1] * h, axes[2][1] * h, origin[1], axes[0][2] * h, axes[1][2] * h, axes[2][2] * h, origin[2], 0, 0, 0, 1];
  // Inverse rigid (T1 space → scan space): x = Rᵀ (y − c − t) + c.
  const R = T.R, inv = (y: number[]) => { const d = [y[0] - T.c[0] - T.t[0], y[1] - T.c[1] - T.t[1], y[2] - T.c[2] - T.t[2]]; return [R[0] * d[0] + R[3] * d[1] + R[6] * d[2] + T.c[0], R[1] * d[0] + R[4] * d[1] + R[7] * d[2] + T.c[1], R[2] * d[0] + R[5] * d[1] + R[8] * d[2] + T.c[2]]; };
  // The field at cell centers (voxels along its axis) and its derivative along the axis, for sampling anywhere.
  const fc = field ? fieldAtCenters(field.fit) : undefined, ax = field?.fit.axis ?? 0;
  let dfc: Float32Array | undefined;
  if (fc) {
    dfc = new Float32Array(fc.length); const stride = [1, sx, sx * sy][ax], m = src.dims[ax];
    for (let v = 0; v < fc.length; v++) { const q = Math.floor(v / stride) % m, up = q + 1 < m ? fc[v + stride] : fc[v], dn = q > 0 ? fc[v - stride] : fc[v]; dfc[v] = (up - dn) / ((q + 1 < m ? 1 : 0) + (q > 0 ? 1 : 0) || 1); }
  }
  // Where each target voxel reads (source voxel coordinates) and with what weight (the stretch), once for all volumes.
  const n = dims[0] * dims[1] * dims[2], pos = new Float32Array(3 * n), wgt = new Float32Array(n), tmp = new Float64Array(1);
  for (let k = 0; k < dims[2]; k++) for (let j = 0; j < dims[1]; j++) for (let i = 0; i < dims[0]; i++) {
    const v = (k * dims[1] + j) * dims[0] + i;
    const y = [0, 1, 2].map((r) => M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3]), x = inv(y);
    const q = [0, 1, 2].map((r) => Si[4 * r] * x[0] + Si[4 * r + 1] * x[1] + Si[4 * r + 2] * x[2] + Si[4 * r + 3]);
    let w = 1;
    if (fc && dfc && field) {
      sampleAt(fc, src.dims, 1, q[0], q[1], q[2], tmp); const b = tmp[0];
      sampleAt(dfc, src.dims, 1, q[0], q[1], q[2], tmp); const db = tmp[0];
      q[ax] += field.sign * b; w = 1 + field.sign * db;
    }
    pos[3 * v] = q[0]; pos[3 * v + 1] = q[1]; pos[3 * v + 2] = q[2]; wgt[v] = w;
  }
  const one = new Float64Array(1);
  const volumes: DiffusionSeries["volumes"] = [];
  for (const vol of dwi.volumes) {
    const out = new Float32Array(n);
    for (let v = 0; v < n; v++) { sampleAt(vol.data as ArrayLike<number>, src.dims, 1, pos[3 * v], pos[3 * v + 1], pos[3 * v + 2], one); out[v] = one[0] * wgt[v]; }
    volumes.push({ ...vol, dims, ijkToRAS: M, data: out, dtype: "<f4" } as DiffusionSeries["volumes"][number]);
    if (volumes.length % 8 === 0) await yieldNow();
  }
  const gradients = dwi.gradients.map((g) => [R[0] * g[0] + R[1] * g[1] + R[2] * g[2], R[3] * g[0] + R[4] * g[1] + R[5] * g[2], R[6] * g[0] + R[7] * g[1] + R[8] * g[2]] as [number, number, number]);
  return { ...dwi, volumes, ijkToRAS: M, gradients, source: `${dwi.source}; onto the T1 (registration rule ${REGISTRATION_RULE})` };
}

/** The RAS points (mm) of a mask's inside voxels, every `stride`-th along each axis -- enough for a bounding box. */
export function maskPoints(m: { dims: number[]; ijkToRAS: number[]; data: ArrayLike<number> }, stride = 1): number[][] {
  const [nx, ny, nz] = m.dims, A = m.ijkToRAS, out: number[][] = [];
  for (let k = 0; k < nz; k += stride) for (let j = 0; j < ny; j += stride) for (let i = 0; i < nx; i += stride)
    if (m.data[(k * ny + j) * nx + i] > 0) out.push([0, 1, 2].map((r) => A[4 * r] * i + A[4 * r + 1] * j + A[4 * r + 2] * k + A[4 * r + 3]));
  return out;
}

/** A move too large to be the same session's -- the alignment is then shown as doubtful (the person aligns by hand). */
export const DOUBTFUL = { mm: 10, degrees: 10 };

/**
 * THE WHOLE STEP, for a case run and for the module: the b = 0 images corrected (the field applied to them alone), their
 * mean aligned to the T1 inside the brain (median_otsu of that mean, grown by 2 voxels), and the whole scan read once onto
 * the T1-aligned grid through the move and the field. Returns the new series, the move, and what was done, in words --
 * with a doubt when the move is larger than one session's or the alignment did not improve the match.
 */
export async function alignToT1(dwi: DiffusionSeries, t1: Grid3, field?: { fit: FieldFit; sign: 1 | -1 }, times?: { register?: number; resample?: number }, opts: { /** The T1's brain mask (tracking rule 3): the resampled box holds it too. */ brainT1?: { dims: number[]; ijkToRAS: number[]; data: ArrayLike<number> } } = {}): Promise<{ dwi: DiffusionSeries; T: Rigid; result: RigidResult; said: string; doubt?: string }> {
  const t0 = performance.now(), src = dwi.volumes[0], n = src.dims[0] * src.dims[1] * src.dims[2];
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
  const result = await rigidToT1({ dims: src.dims as [number, number, number], ijkToRAS: src.ijkToRAS, data: b0 }, t1, mask);
  const t1Done = performance.now();
  if (times) times.register = t1Done - t0;
  const out = await resampleOntoT1(dwi, result.T, t1, field, { cover: mask, ...(opts.brainT1 ? { alsoCover: maskPoints(opts.brainT1, 2) } : {}) });
  if (times) times.resample = performance.now() - t1Done;
  const sz = rigidSize(result.T);
  const doubt = sz.mm > DOUBTFUL.mm || sz.degrees > DOUBTFUL.degrees ? `it had to move ${sz.mm.toFixed(0)} mm and ${sz.degrees.toFixed(0)}°, more than within one visit`
    : result.cost > result.costAtStart + 1e-4 ? "the match got worse" : undefined;     // equal: the scan was already in place (critic, finding 18)
  return { dwi: out, T: result.T, result, said: `aligned to the MRI of the anatomy (moved ${sz.mm.toFixed(1)} mm and ${sz.degrees.toFixed(1)}°)`, ...(doubt ? { doubt } : {}) };
}

// HEAD MOVEMENT DURING THE DIFFUSION SCAN (Lauren O'Donnell, 2026-10-05: motion and eddy-current correction were
// missing from the pipeline; Ron: "go ahead"). A diffusion scan is about a hundred images taken one after another over
// ten to fifteen minutes, and the head drifts between them: on ds001226, from the first b = 0 image to the last, 0.5-1 mm
// and 0.5-1.5° is typical, and up to 3 mm or 3° (measured 2026-10-05 on the 29 library cases). Each image is put back
// where the head was in the reference before anything is fitted.
//
// HOW (motion rule 1), after the idea of FSL's eddy (Andersson & Sotiropoulos, NeuroImage 125:1063, 2016), with a simpler
// model of the signal and written from that description, not from its code:
//   - the reference is the mean of the b = 0 images; each b = 0 is aligned to it;
//   - a diffusion-weighted image cannot be aligned to the b = 0 (at b 2800 it is a different picture), so it is aligned
//     to what it should look like: a prediction from ALL the other images, per voxel the kurtosis model on the log
//     signal (scanPredictor; for a scan with one diffusion-weighted shell, a polynomial of the gradient direction of
//     degree 4, or 2 under 22 images), fitted with the image itself left out;
//   - the diffusion-weighted images' common position is invisible to their predictions of each other. With three shells
//     or more it is found by extrapolating them to b = 0 (the fit's intercept, a b = 0-like picture) and aligning that to
//     the b = 0 mean; with one or two shells each shell is pinned instead to the b = 0 images' movement at the times its
//     images were taken (interpolated between the b = 0 images spread through the scan);
//   - two rounds: the second builds its predictions from images already put back.
// The alignment is rigid, least squares with a gain and an offset, Gauss-Newton with Levenberg-Marquardt damping, on a
// grid twice as coarse and then on the scan's own, inside the brain (median_otsu of the b = 0 mean, grown by 2 voxels).
// The distortion field moves with the head: each image is read at the moved point plus the field's shift taken where the
// point is in the reference (as eddy assumes). The moves go into the ONE resampling (registration.ts resampleOntoT1), so
// the data are still interpolated once, and each gradient direction turns with its image.
//
// EDDY CURRENTS (motion rule 2): switching the diffusion gradients on induces currents in the scanner's metal whose stray
// field shifts each diffusion-weighted image along the phase-encoding axis, differently for every image (it follows the
// gradient's direction and strength). Modeled as eddy's linear model (--flm=linear): the shift is a linear function of
// where the point is in the scanner, d = g·(y − c) / 100 mm, three numbers g a image, estimated with the movement against
// the same prediction; the image's brightness is corrected for the stretch (1 + g·e/100, e the phase-encoding axis). The
// constant part of the shift is the same thing as a movement along e and is left to the movement. Not for b = 0 images
// (no diffusion gradient, no eddy currents).
//
// TIED TO THE GRADIENT (motion rule 3; Ron, 2026-10-05: "build C"): on ds001226 the per-image slopes of rule 2 were noise
// beyond one axis, for FSL's eddy as for us (the physics check, dmri-review 2026-10-05 night). Eddy currents follow the
// gradient pulse, whose strength goes with √b at fixed timing, so the slopes are fitted, over all diffusion-weighted
// images, as g = K (√(b/1000) · direction) + g₀ (K 3×3, g₀ a constant; eddy's --slm=linear), each image's slopes set to
// that fit, and the movement estimated again with them held. Twelve numbers for the scan instead of three an image.
//
// Not here yet: eddy's quadratic model (its default), slice-wise movement and signal dropout (--mporder, --repol).
import type { DiffusionSeries } from "./dwi.ts";
import type { FieldFit } from "./distortion.ts";
import { coarsen, fieldWithSlope, inv4, rigidSize, sampleAt, scanBrain, spacingOf, worldGradient, type Grid3, type Rigid } from "./registration.ts";

/** The numbered rules (Ron, 2026-09-25: "as modular as possible and also versioned"). */
export const MOTION_RULES = {
  0: "none: the images are used as they are (a scan corrected before, such as a preprocessed dataset)",
  1: "rigid per image: the b = 0 images to their mean, each diffusion-weighted image to a prediction from all the others (the kurtosis model on the log signal, itself left out), the diffusion-weighted images' common offset from their extrapolation to b = 0, two rounds",
  2: "rule 1 and eddy currents: each diffusion-weighted image also stretched and sheared along the phase-encoding axis (a shift linear in position, three numbers an image), estimated with its movement",
  3: "rule 2 with the eddy currents tied to the gradient: each image's three slopes a fixed linear map of √b × its gradient direction, plus a constant, fitted over all the images (eddy's second-level model), then the movement estimated again with the slopes held",
} as const;
export type MotionRuleId = keyof typeof MOTION_RULES;
/** The rule in use. Rule 1 (Ron, 2026-10-05: "A now"), after the comparison with FSL's eddy (movement 0.20-0.30 mm root mean
 *  square on PAT16 and PAT25); eddy currents (rule 2) left out, their per-image estimates being noise beyond one axis on
 *  ds001226's scanner (dmri-review, 2026-10-05 night). */
export const MOTION_RULE: MotionRuleId = 1;
/** A common move of all the diffusion-weighted images larger than this is a failed fit, not a movement (step 4). */
const COMMON_MOVE_LIMIT = { mm: 3, degrees: 3 };

type Field = { fit: FieldFit; sign: 1 | -1 };
/** A move with, for a diffusion-weighted image under rule 2, its eddy-current shift along the phase-encoding axis e (a
 *  unit vector in RAS): d(y) = g·(y − c) / 100 mm at the point y where the tissue was. */
export interface Move extends Rigid { ec?: { g: [number, number, number]; e: [number, number, number] } }
const yieldNow = () => new Promise<void>((r) => setTimeout(r, 0));

export interface MotionResult {
  rule: MotionRuleId;
  /** Per image: the move from the reference to where the head was (scan RAS, about the brain's center), and under rule 2
   *  its eddy-current shift. */
  moves: Move[];
  /** The b = 0 mean, corrected for the field and the movement, and the brain grown by 2 voxels, on the scan's grid. */
  b0: Float32Array; mask: Uint8Array;
  /** Each image's move from the reference (mm at the brain's center, degrees). */
  sizes: { mm: number; degrees: number }[];
  largest: { mm: number; degrees: number };
  /** Rule 2: the largest eddy-current shift at the brain's edge (mm), over all images. */
  eddyMm?: number;
  /** How far the diffusion-weighted images are from their predictions over the brain (root mean square, relative to the
   *  signal), as acquired and after the correction: the check that the correction made the images agree better. */
  residual: { before: number; after: number };
  ms: number;
  said: string;
}

// ── Rotations ───────────────────────────────────────────────────────────────────────────────────

const I3 = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];
/** exp of a rotation vector (radians): Rodrigues. Row-major 3×3. */
export function rotVec(w: ArrayLike<number>): number[] {
  const th = Math.hypot(w[0], w[1], w[2]);
  if (th < 1e-12) return I3();
  const k0 = w[0] / th, k1 = w[1] / th, k2 = w[2] / th, c = Math.cos(th), s = Math.sin(th), v = 1 - c;
  return [c + k0 * k0 * v, k0 * k1 * v - k2 * s, k0 * k2 * v + k1 * s, k1 * k0 * v + k2 * s, c + k1 * k1 * v, k1 * k2 * v - k0 * s, k2 * k0 * v - k1 * s, k2 * k1 * v + k0 * s, c + k2 * k2 * v];
}
/** log of a rotation: its rotation vector (radians). */
export function logRot(R: ArrayLike<number>): [number, number, number] {
  const th = Math.acos(Math.max(-1, Math.min(1, (R[0] + R[4] + R[8] - 1) / 2))), f = th < 1e-6 ? 0.5 : th / (2 * Math.sin(th));
  return [(R[7] - R[5]) * f, (R[2] - R[6]) * f, (R[3] - R[1]) * f];
}
const mul3 = (A: number[], B: number[]) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c]));

/** Solve A x = b (n×n, row-major) by elimination with partial pivoting; undefined when singular. */
function solve(A: Float64Array, b: Float64Array, n: number): Float64Array | undefined {
  const M = Float64Array.from(A), x = Float64Array.from(b);
  // Singular, or nearly (critic, 2026-10-05, finding 2: an exact-zero test let roundoff through as huge weights).
  let scale = 0; for (let i = 0; i < n * n; i++) scale = Math.max(scale, Math.abs(M[i]));
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r * n + c]) > Math.abs(M[p * n + c])) p = r;
    if (!(Math.abs(M[p * n + c]) > 1e-12 * scale)) return undefined;
    if (p !== c) { for (let k = 0; k < n; k++) { const t = M[c * n + k]; M[c * n + k] = M[p * n + k]; M[p * n + k] = t; } const t = x[c]; x[c] = x[p]; x[p] = t; }
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r * n + c] / M[c * n + c]; if (f) { for (let k = c; k < n; k++) M[r * n + k] -= f * M[c * n + k]; x[r] -= f * x[c]; } }
  }
  for (let c = 0; c < n; c++) x[c] /= M[c * n + c];
  return x;
}

// ── Points of the reference, and reading an image at them through a move ───────────────────────

/** Points of the reference (RAS), each with the field's shift there (mm, along the phase-encoding axis) and its stretch. */
interface Points { n: number; x: Float64Array; shift: Float32Array; w: Float32Array }

function pointsAt(ras: Float64Array, grid: { dims: [number, number, number]; ijkToRAS: number[] }, fd: { b: Float32Array; slope: Float32Array; axis: number; sign: 1 | -1 } | undefined): Points {
  const n = ras.length / 3, shift = new Float32Array(3 * n), w = new Float32Array(n).fill(1);
  if (fd) {
    const M = grid.ijkToRAS, Si = inv4(M), a = [M[fd.axis], M[4 + fd.axis], M[8 + fd.axis]], tmp = new Float64Array(1);
    for (let p = 0; p < n; p++) {
      const x = ras[3 * p], y = ras[3 * p + 1], z = ras[3 * p + 2];
      const q0 = Si[0] * x + Si[1] * y + Si[2] * z + Si[3], q1 = Si[4] * x + Si[5] * y + Si[6] * z + Si[7], q2 = Si[8] * x + Si[9] * y + Si[10] * z + Si[11];
      sampleAt(fd.b, grid.dims, 1, q0, q1, q2, tmp); const s = fd.sign * tmp[0];
      sampleAt(fd.slope, grid.dims, 1, q0, q1, q2, tmp); w[p] = 1 + fd.sign * tmp[0];
      shift[3 * p] = s * a[0]; shift[3 * p + 1] = s * a[1]; shift[3 * p + 2] = s * a[2];
    }
  }
  return { n, x: ras, shift, w };
}

/** The voxel centers of `grid` where `pick` is set, in RAS. */
function voxelsRas(grid: { dims: [number, number, number]; ijkToRAS: number[] }, pick: (v: number, i: number, j: number, k: number) => boolean): { ras: Float64Array; idx: Int32Array } {
  const [nx, ny, nz] = grid.dims, M = grid.ijkToRAS, xs: number[] = [], id: number[] = [];
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = (k * ny + j) * nx + i;
    if (!pick(v, i, j, k)) continue;
    id.push(v); xs.push(M[0] * i + M[1] * j + M[2] * k + M[3], M[4] * i + M[5] * j + M[6] * k + M[7], M[8] * i + M[9] * j + M[10] * k + M[11]);
  }
  return { ras: Float64Array.from(xs), idx: Int32Array.from(id) };
}

/** An image (on the scan's grid) read at the reference points through a move: w · V(m(x) + shift + eddy shift). */
function readMoved(data: ArrayLike<number>, dims: [number, number, number], Si: number[], P: Points, m: Move): Float32Array {
  const out = new Float32Array(P.n), one = new Float64Array(1), R = m.R, c = m.c, t = m.t;
  const g = m.ec?.g ?? [0, 0, 0], e = m.ec?.e ?? [0, 0, 0], stretch = 1 + (g[0] * e[0] + g[1] * e[1] + g[2] * e[2]) / 100;
  for (let p = 0; p < P.n; p++) {
    const d0 = P.x[3 * p] - c[0], d1 = P.x[3 * p + 1] - c[1], d2 = P.x[3 * p + 2] - c[2];
    const r0 = R[0] * d0 + R[1] * d1 + R[2] * d2 + t[0], r1 = R[3] * d0 + R[4] * d1 + R[5] * d2 + t[1], r2 = R[6] * d0 + R[7] * d1 + R[8] * d2 + t[2];
    const dd = (g[0] * r0 + g[1] * r1 + g[2] * r2) / 100;
    const y0 = r0 + c[0] + P.shift[3 * p] + dd * e[0], y1 = r1 + c[1] + P.shift[3 * p + 1] + dd * e[1], y2 = r2 + c[2] + P.shift[3 * p + 2] + dd * e[2];
    sampleAt(data, dims, 1, Si[0] * y0 + Si[1] * y1 + Si[2] * y2 + Si[3], Si[4] * y0 + Si[5] * y1 + Si[6] * y2 + Si[7], Si[8] * y0 + Si[9] * y1 + Si[10] * y2 + Si[11], one);
    out[p] = one[0] * P.w[p] * stretch;
  }
  return out;
}

// ── One image to its target: rigid + gain + offset, Gauss-Newton with damping ───────────────────

interface Level { grid: { dims: [number, number, number]; ijkToRAS: number[] }; Si: number[]; mm: number; P: Points; idx: Int32Array; iterations: number }

/** The image and its gradient (RAS mm) interleaved, 4 values a voxel, on a level's grid. */
function packed(g: Grid3): Float32Array {
  const v = g.data instanceof Float32Array ? g.data : Float32Array.from(g.data), gr = worldGradient(v, g), out = new Float32Array(4 * v.length);
  for (let i = 0; i < v.length; i++) { out[4 * i] = v[i]; out[4 * i + 1] = gr[3 * i]; out[4 * i + 2] = gr[3 * i + 1]; out[4 * i + 3] = gr[3 * i + 2]; }
  return out;
}

/** Trilinear sample of a 4-value-a-voxel image (value and gradient) at voxel position (x, y, z), as registration.ts's
 *  sampleAt does (zeros beyond half a voxel past the edge, the edge voxel within it), unrolled: the alignment's inner loop. */
function sample4(f: Float32Array, nx: number, ny: number, nz: number, x: number, y: number, z: number, out: Float64Array): boolean {
  if (!(x >= -0.5 && y >= -0.5 && z >= -0.5 && x <= nx - 0.5 && y <= ny - 0.5 && z <= nz - 0.5)) return false;
  x = x < 0 ? 0 : x > nx - 1 ? nx - 1 : x; y = y < 0 ? 0 : y > ny - 1 ? ny - 1 : y; z = z < 0 ? 0 : z > nz - 1 ? nz - 1 : z;
  let i0 = Math.floor(x), j0 = Math.floor(y), k0 = Math.floor(z);
  if (i0 > nx - 2) i0 = nx > 1 ? nx - 2 : 0; if (j0 > ny - 2) j0 = ny > 1 ? ny - 2 : 0; if (k0 > nz - 2) k0 = nz > 1 ? nz - 2 : 0;
  const fx = x - i0, fy = y - j0, fz = z - k0, sx = nx > 1 ? 4 : 0, sy = ny > 1 ? 4 * nx : 0, sz = nz > 1 ? 4 * nx * ny : 0, b = 4 * ((k0 * ny + j0) * nx + i0);
  const w000 = (1 - fx) * (1 - fy) * (1 - fz), w100 = fx * (1 - fy) * (1 - fz), w010 = (1 - fx) * fy * (1 - fz), w110 = fx * fy * (1 - fz);
  const w001 = (1 - fx) * (1 - fy) * fz, w101 = fx * (1 - fy) * fz, w011 = (1 - fx) * fy * fz, w111 = fx * fy * fz;
  for (let c = 0; c < 4; c++) {
    const o = b + c;
    out[c] = w000 * f[o] + w100 * f[o + sx] + w010 * f[o + sy] + w110 * f[o + sy + sx] + w001 * f[o + sz] + w101 * f[o + sz + sx] + w011 * f[o + sz + sy] + w111 * f[o + sz + sy + sx];
  }
  return true;
}

/** The parameters of one alignment: the move (R, t about c), the gain a and offset b, and the eddy-current slopes g
 *  along e (when `e` is given). */
interface Params { R: number[]; t: number[]; a: number; b: number; g: number[] }

/** The sum of squares and (with H, gr) the normal equations: 8 unknowns (rotation, translation, gain, offset), 11 with
 *  the eddy-current slopes. */
function normalEq(L: Level, img: Float32Array, F: Float32Array, q: Params, c: number[], e: number[] | undefined, H?: Float64Array, gr?: Float64Array, ecHeld = false): { cost: number; n: number } {
  // ecHeld: the slopes q.g are applied along e but not estimated (rule 3's last round).
  const P = L.P, Si = L.Si, s4 = new Float64Array(4), np = e && !ecHeld ? 11 : 8, J = new Float64Array(np), [nx, ny, nz] = L.grid.dims;
  const { R, t, a, b } = q, g = e ? q.g : [0, 0, 0], E = e ?? [0, 0, 0], stretch = 1 + (g[0] * E[0] + g[1] * E[1] + g[2] * E[2]) / 100;
  let cost = 0, n = 0;
  if (H) H.fill(0); if (gr) gr.fill(0);
  for (let p = 0; p < P.n; p++) {
    const d0r = P.x[3 * p] - c[0], d1r = P.x[3 * p + 1] - c[1], d2r = P.x[3 * p + 2] - c[2];
    const d0 = R[0] * d0r + R[1] * d1r + R[2] * d2r, d1 = R[3] * d0r + R[4] * d1r + R[5] * d2r, d2 = R[6] * d0r + R[7] * d1r + R[8] * d2r;
    // r: where the tissue was, from the brain's center; then the field's shift and the eddy-current shift along e.
    const r0 = d0 + t[0], r1 = d1 + t[1], r2 = d2 + t[2], dd = (g[0] * r0 + g[1] * r1 + g[2] * r2) / 100;
    const y0 = r0 + c[0] + P.shift[3 * p] + dd * E[0], y1 = r1 + c[1] + P.shift[3 * p + 1] + dd * E[1], y2 = r2 + c[2] + P.shift[3 * p + 2] + dd * E[2];
    if (!sample4(img, nx, ny, nz, Si[0] * y0 + Si[1] * y1 + Si[2] * y2 + Si[3], Si[4] * y0 + Si[5] * y1 + Si[6] * y2 + Si[7], Si[8] * y0 + Si[9] * y1 + Si[10] * y2 + Si[11], s4)) continue;
    const w = P.w[p] * stretch, V = w * s4[0], g0 = w * s4[1], g1 = w * s4[2], g2 = w * s4[3];
    const res = a * V + b - F[p];
    cost += res * res; n++;
    if (!H || !gr) continue;
    // A move of the tissue point also moves where the eddy shift is taken: the effective gradient is ∇V + (∇V·e) g/100.
    const ge = g0 * E[0] + g1 * E[1] + g2 * E[2], h0 = g0 + ge * g[0] / 100, h1 = g1 + ge * g[1] / 100, h2 = g2 + ge * g[2] / 100;
    // d/dω = a (d × h) for the update R ← exp([ω]×) R; d/dt = a h; d/da = V; d/db = 1;
    // d/dg_k = a [(∇V·e)(r_k)/100 + w_field V_raw e_k/100] (the shift, and the stretch's brightness).
    J[0] = a * (d1 * h2 - d2 * h1); J[1] = a * (d2 * h0 - d0 * h2); J[2] = a * (d0 * h1 - d1 * h0);
    J[3] = a * h0; J[4] = a * h1; J[5] = a * h2; J[6] = V; J[7] = 1;
    if (e && !ecHeld) { const vr = P.w[p] * s4[0] / 100; J[8] = a * (ge * r0 / 100 + vr * E[0]); J[9] = a * (ge * r1 / 100 + vr * E[1]); J[10] = a * (ge * r2 / 100 + vr * E[2]); }
    for (let i = 0; i < np; i++) { gr[i] += J[i] * res; for (let k = i; k < np; k++) H[i * np + k] += J[i] * J[k]; }
  }
  if (H) for (let i = 0; i < np; i++) for (let k = 0; k < i; k++) H[i * np + k] = H[k * np + i];
  return { cost: n ? cost / n : Infinity, n };
}

/** Align one image (scan grid) to its target (values at each level's points), from `start`; with `e`, its eddy-current
 *  slopes along e too. */
function alignOne(data: ArrayLike<number>, full: { dims: [number, number, number]; ijkToRAS: number[] }, levels: Level[], targets: Float32Array[], start: Move, e?: [number, number, number], ecHeld = false): Move {
  // ecHeld: start.ec's slopes are applied, not estimated; only the movement (and gain, offset) is.
  if (ecHeld && start.ec) e = start.ec.e;
  const held = ecHeld && !!e, c = start.c, np = e && !held ? 11 : 8, H = new Float64Array(np * np), gr = new Float64Array(np);
  let q: Params = { R: start.R.slice(), t: [...start.t], a: 1, b: 0, g: e ? [...(start.ec?.g ?? [0, 0, 0])] : [0, 0, 0] };
  for (const [li, L] of levels.entries()) {
    const G: Grid3 = { dims: full.dims, ijkToRAS: full.ijkToRAS, data };
    const img = packed(L.mm > 0 ? coarsen(G, L.mm) : G), F = targets[li];
    // Gain and offset by regression at the start (the images match in brightness up to these; at b = 0 a = 1, b = 0):
    // one Gauss-Newton step on a and b alone is exactly that regression.
    {
      const H0 = new Float64Array(np * np), g0 = new Float64Array(np);
      normalEq(L, img, F, { ...q, a: 1, b: 0 }, c, e, H0, g0, held);
      const A2 = new Float64Array([H0[6 * np + 6], H0[6 * np + 7], H0[7 * np + 6], H0[7 * np + 7]]), d = solve(A2, new Float64Array([-g0[6], -g0[7]]), 2);
      // With a = 1 the derivative in a is V itself, so the step lands on the regression's a and b.
      if (d && H0[7 * np + 7] > 10) q = { ...q, a: 1 + d[0], b: d[1] };
    }
    let cur = normalEq(L, img, F, q, c, e, H, gr, held), lambda = 1e-3;
    for (let it = 0; it < L.iterations; it++) {
      const A = Float64Array.from(H); for (let i = 0; i < np; i++) A[i * np + i] += lambda * (H[i * np + i] || 1);
      const d = solve(A, gr.map((x) => -x), np);
      if (!d) break;
      const q2: Params = { R: mul3(rotVec([d[0], d[1], d[2]]), q.R), t: [q.t[0] + d[3], q.t[1] + d[4], q.t[2] + d[5]], a: q.a + d[6], b: q.b + d[7], g: e && !held ? [q.g[0] + d[8], q.g[1] + d[9], q.g[2] + d[10]] : q.g };
      const H2 = new Float64Array(np * np), g2 = new Float64Array(np), next = normalEq(L, img, F, q2, c, e, H2, g2, held);
      if (next.cost < cur.cost) {
        q = q2; cur = next; H.set(H2); gr.set(g2); lambda = Math.max(1e-6, lambda / 3);
        if (Math.hypot(d[3], d[4], d[5]) < 1e-3 && Math.hypot(d[0], d[1], d[2]) < 2e-5 && (!e || held || Math.hypot(d[8], d[9], d[10]) < 1e-3)) break;
      } else { lambda *= 10; if (lambda > 1e4) break; }
    }
  }
  return { R: q.R, t: q.t as [number, number, number], c, ...(e ? { ec: { g: q.g as [number, number, number], e } } : {}) };
}

// ── The prediction of a diffusion-weighted image from the rest of its shell ─────────────────────

/** The shells: images with b > 50 and a direction, grouped by b (within 5% or 50 s/mm²). */
export function shellsOf(dwi: Pick<DiffusionSeries, "bValues" | "gradients">): number[][] {
  const idx = dwi.bValues.map((b, i) => i).filter((i) => dwi.bValues[i] > 50 && Math.hypot(...dwi.gradients[i]) > 0.5).sort((a, b) => dwi.bValues[a] - dwi.bValues[b]);
  const out: number[][] = [];
  for (const i of idx) { const s = out[out.length - 1], b = dwi.bValues[i]; if (s && b - dwi.bValues[s[0]] <= Math.max(50, 0.05 * b)) s.push(i); else out.push([i]); }
  return out.map((s) => s.sort((a, b) => a - b));
}
/** The even monomials of one degree in the direction's components (degree 4: 15 of them, the span of spherical
 *  harmonics of orders 0, 2 and 4 on the sphere; degree 2: 6). */
function monomials(g: ArrayLike<number>, deg: number): number[] {
  const out: number[] = [];
  for (let i = deg; i >= 0; i--) for (let j = deg - i; j >= 0; j--) out.push(g[0] ** i * g[1] ** j * g[2] ** (deg - i - j));
  return out;
}
/**
 * The prediction of every image from all the others, voxel by voxel: the LOGARITHM of the signal as the kurtosis model
 * (Jensen et al., MRM 53:1432, 2005) -- log S = c − b·Q₂(g) + b²·Q₄(g), Q₂ and Q₄ the even polynomials of degree 2 and
 * 4 in the gradient direction, 22 numbers a voxel -- fitted by least squares over every image of the scan, the b = 0 ones
 * included, with the image itself left out: its prediction is exp(l_i − (l_i − fitted_i) / (1 − h_ii)), l the log
 * signal, h the fit's leverage. In the logarithm a single tensor is exactly the b term; crossing fibers and restriction
 * are mostly the b² term. A scan with one diffusion-weighted shell: log S = c − b·Q₄(g) (16 numbers; Q₂ if the shell has
 * fewer than 22 images).
 * Why this model (measured 2026-10-05 on synthetic single, crossing and free-water voxels with ds001226's own gradient
 * table, noise σ 15 on a b = 0 of 500): prediction errors 2.5%, 3.3% and 9.5% at b 700, 1200 and 2800, against 4.2%,
 * 6.8% and 9.7% for a separate polynomial per shell, whose higher leverage (0.37-0.50 against 0.10-0.30) also made the
 * rounds drift; a Gaussian process on the directions (eddy's kind of model) did worse than both, 5-15% without noise.
 */
export function scanPredictor(bValues: number[], dirs: ArrayLike<number>[]): ((Y: Float32Array[]) => Float32Array[]) | undefined {
  const N = bValues.length, shells = shellsOf({ bValues, gradients: dirs.map((d) => [d[0], d[1], d[2]] as [number, number, number]) });
  const use = bValues.map((b, i) => b < 50 || shells.some((s) => s.includes(i)));   // b = 0 and directional images
  const dwis = shells.flat().length;
  if (!dwis || dwis < 9) return undefined;
  const one = shells.length === 1, deg = shells[0].length >= 22 ? 4 : 2;
  const rows = bValues.map((b0, i) => {
    const b = b0 < 50 ? 0 : b0 / 1000, d = dirs[i], l = Math.hypot(d[0], d[1], d[2]) || 1, g = [d[0] / l, d[1] / l, d[2] / l];
    if (one) return [1, ...monomials(g, deg).map((x) => -b * x)];
    return [1, ...monomials(g, 2).map((x) => -b * x), ...monomials(g, 4).map((x) => b * b * x)];
  });
  const idx = bValues.map((_, i) => i).filter((i) => use[i]), n = idx.length, X = idx.map((i) => rows[i]), p = X[0].length;
  if (n <= p + 3) return undefined;
  const XtX = new Float64Array(p * p);
  for (let a = 0; a < p; a++) for (let c = 0; c < p; c++) { let s = 0; for (let r = 0; r < n; r++) s += X[r][a] * X[r][c]; XtX[a * p + c] = s; }
  const inv = new Float64Array(p * p);
  for (let c = 0; c < p; c++) { const e = new Float64Array(p); e[c] = 1; const col = solve(XtX, e, p); if (!col) return undefined; for (let r = 0; r < p; r++) inv[r * p + c] = col[r]; }
  // The hat matrix H = X (XᵀX)⁻¹ Xᵀ (n×n, the same for every voxel), and the leave-one-out factor 1 / (1 − h_ii).
  const T = X.map((x) => Array.from({ length: p }, (_, a) => { let t = 0; for (let c = 0; c < p; c++) t += inv[a * p + c] * x[c]; return t; }));
  const H = new Float64Array(n * n), loo = new Float64Array(n);
  for (let r = 0; r < n; r++) for (let q = 0; q < n; q++) { let s = 0; for (let a = 0; a < p; a++) s += X[r][a] * T[q][a]; H[r * n + q] = s; }
  for (let r = 0; r < n; r++) loo[r] = 1 / Math.max(0.05, 1 - H[r * n + r]);
  return (Y) => {
    const m = Y[idx[0]].length, out: Float32Array[] = new Array(N), l = new Float64Array(n);
    for (let r = 0; r < n; r++) if (bValues[idx[r]] >= 50) out[idx[r]] = new Float32Array(m);
    for (let v = 0; v < m; v++) {
      let top = 0; for (let r = 0; r < n; r++) top = Math.max(top, Y[idx[r]][v]);
      if (!(top > 0)) continue;
      // A floor under the logarithm: noise at the bottom of a strongly weighted image is not a signal of zero.
      const floor = Math.max(1e-3 * top, 1e-6);
      for (let r = 0; r < n; r++) l[r] = Math.log(Math.max(Y[idx[r]][v], floor));
      for (let r = 0; r < n; r++) {
        const o = out[idx[r]]; if (!o) continue;
        let f = 0; const h = r * n; for (let q = 0; q < n; q++) f += H[h + q] * l[q];
        o[v] = Math.min(2 * top, Math.exp(l[r] - (l[r] - f) * loo[r]));
      }
    }
    return out;
  };
}

/**
 * The diffusion-weighted images extrapolated to b = 0, voxel by voxel: the intercept of the kurtosis model fitted to them
 * alone (no b = 0 image in the fit), as weights on their log signals. It looks like a b = 0 image and sits where the
 * diffusion-weighted images sit, so aligning it to the b = 0 mean finds their common offset -- the one movement their
 * predictions from each other cannot see. Undefined for a single shell (one b cannot be extrapolated).
 */
function interceptWeights(bValues: number[], dirs: ArrayLike<number>[], idx: number[]): Float64Array | undefined {
  // The constant, the b terms and the b² terms are told apart only with THREE distinct shells or more (critic,
  // 2026-10-05, finding 2: two shells passed and threw the images 7-10 mm and 20° off); shells as shellsOf groups them.
  if (shellsOf({ bValues, gradients: dirs.map((d) => [d[0], d[1], d[2]] as [number, number, number]) }).length < 3) return undefined;
  const X = idx.map((i) => { const b = bValues[i] / 1000, d = dirs[i], l = Math.hypot(d[0], d[1], d[2]) || 1, g = [d[0] / l, d[1] / l, d[2] / l]; return [1, ...monomials(g, 2).map((x) => -b * x), ...monomials(g, 4).map((x) => b * b * x)]; });
  const n = X.length, p = X[0].length;
  if (n <= p + 3) return undefined;
  const XtX = new Float64Array(p * p);
  for (let a = 0; a < p; a++) for (let c = 0; c < p; c++) { let s = 0; for (let r = 0; r < n; r++) s += X[r][a] * X[r][c]; XtX[a * p + c] = s; }
  const e = new Float64Array(p); e[0] = 1;
  const row = solve(XtX, e, p);                       // the first row of (XᵀX)⁻¹ (symmetric)
  if (!row) return undefined;
  return Float64Array.from(X, (x) => x.reduce((s, v, a) => s + v * row[a], 0));
}

/** Rule 3: the slopes as a linear function of the gradient vector q = √(b/1000) · direction, plus a constant, by least
 *  squares over the images: g = K q + g₀ (12 numbers). Undefined when there are too few images to fit it. */
export function tieToGradient(rows: { q: number[]; g: number[] }[]): ((q: number[]) => [number, number, number]) | undefined {
  if (rows.length < 12) return undefined;
  const X = rows.map((r) => [r.q[0], r.q[1], r.q[2], 1]), XtX = new Float64Array(16);
  for (const x of X) for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) XtX[a * 4 + b] += x[a] * x[b];
  const W = [0, 1, 2].map((k) => { const Xty = new Float64Array(4); X.forEach((x, n) => { for (let a = 0; a < 4; a++) Xty[a] += x[a] * rows[n].g[k]; }); return solve(XtX, Xty, 4); });
  if (W.some((w) => !w)) return undefined;
  return (q) => [0, 1, 2].map((k) => W[k]![0] * q[0] + W[k]![1] * q[1] + W[k]![2] * q[2] + W[k]![3]) as [number, number, number];
}

// ── The whole estimate ──────────────────────────────────────────────────────────────────────────

/** A move as six numbers (rotation vector, translation) and back, all about the same center. */
const toVec = (m: Rigid) => [...logRot(m.R), ...m.t];
const fromVec = (v: number[], c: [number, number, number]): Rigid => ({ R: rotVec(v.slice(0, 3)), t: [v[3], v[4], v[5]], c });

/**
 * The head's movement for every image of `dwi` (on the scan's grid, as acquired), with the distortion field when there is
 * one. Returns each image's move from the reference, the corrected b = 0 mean and brain, and how much better the images
 * agree with their predictions afterwards.
 */
export async function estimateMotion(dwi: DiffusionSeries, field?: Field, opts: { rounds?: number; /** 1: movement; 2: and eddy currents. */ rule?: 1 | 2 | 3; /** The phase-encoding axis of the scan's voxels (0 i, 1 j, 2 k) when no field gives it. */ peAxis?: 0 | 1 | 2 } = {}): Promise<MotionResult> {
  const t0 = performance.now(), src = dwi.volumes[0], full = { dims: src.dims as [number, number, number], ijkToRAS: src.ijkToRAS };
  const Si = inv4(full.ijkToRAS), N = dwi.volumes.length, [nx, ny, nz] = full.dims, nv = nx * ny * nz;
  const b0i = dwi.bValues.map((b, i) => (b < 50 ? i : -1)).filter((i) => i >= 0), shells = shellsOf(dwi);
  const fdRaw = field ? fieldWithSlope(field.fit) : undefined, fd = fdRaw && field ? { ...fdRaw, axis: field.fit.axis, sign: field.sign } : undefined;
  const { mask } = scanBrain(dwi, field);
  // The center the rotations turn about: the brain's (so a move's mm are the brain center's).
  let cx = 0, cy = 0, cz = 0, cn = 0;
  const M = full.ijkToRAS;
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (mask[(k * ny + j) * nx + i]) { cx += i; cy += j; cz += k; cn++; }
  cx /= cn || 1; cy /= cn || 1; cz /= cn || 1;
  const c: [number, number, number] = [0, 1, 2].map((r) => M[4 * r] * cx + M[4 * r + 1] * cy + M[4 * r + 2] * cz + M[4 * r + 3]) as [number, number, number];
  const identity = (): Move => ({ R: I3(), t: [0, 0, 0], c });
  let moves: Move[] = dwi.volumes.map(identity);
  // Eddy currents (rule 2) shift along the phase-encoding axis: the distortion field's, else the scanner's record.
  const peAxis = field?.fit.axis ?? opts.peAxis, rule = opts.rule ?? 1;
  const e = (rule === 2 || rule === 3) && peAxis !== undefined ? (() => { const v = [M[peAxis], M[4 + peAxis], M[8 + peAxis]], l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l] as [number, number, number]; })() : undefined;
  const data = (i: number) => dwi.volumes[i].data as ArrayLike<number>;
  // Nothing to align to: no b = 0, no brain, or too few diffusion-weighted images to predict them from each other (a
  // six-direction scan; critic, 2026-10-05, finding 7: it said "corrected … NaN% better").
  const why = !b0i.length ? "no b = 0 image" : !cn ? "no brain found" : !scanPredictor(dwi.bValues, dwi.gradients) ? "too few diffusion-weighted images to predict them from each other" : "";
  if (why) {
    const b0 = scanBrain(dwi, field).b0;
    return { rule: 0, moves, b0, mask, sizes: moves.map(() => ({ mm: 0, degrees: 0 })), largest: { mm: 0, degrees: 0 }, residual: { before: NaN, after: NaN }, ms: performance.now() - t0, said: `head movement not corrected (${why})` };
  }

  // The region the images are read on: the brain grown by 2 more voxels (the coarse level's blocks reach one beyond it).
  const region = new Uint8Array(nv);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    let on = 0;
    for (let cc = -2; cc <= 2 && !on; cc++) for (let b = -2; b <= 2 && !on; b++) for (let a = -2; a <= 2 && !on; a++) { const I = i + a, J = j + b, K = k + cc; if (I >= 0 && J >= 0 && K >= 0 && I < nx && J < ny && K < nz && mask[(K * ny + J) * nx + I]) on = 1; }
    region[(k * ny + j) * nx + i] = on;
  }
  const reg = voxelsRas(full, (v) => region[v] === 1), regP = pointsAt(reg.ras, full, fd);
  const scatter = (vals: Float32Array) => { const o = new Float32Array(nv); for (let p = 0; p < reg.idx.length; p++) o[reg.idx[p]] = vals[p]; return o; };
  // The levels: twice the scan's smallest spacing (block means), then the scan's own grid at every second voxel.
  const sp = spacingOf(full.ijkToRAS), coarseMm = 2 * Math.min(...sp);
  const cMask = coarsen({ ...full, data: mask }, coarseMm), cGrid = { dims: cMask.dims, ijkToRAS: cMask.ijkToRAS };
  // Not within 2 voxels (and 4 mm) of the field of view's faces: there a moved image reads nothing, and the slab often cuts
  // through the head (ds001226 ends in the cerebellum).
  const edge = sp.map((h) => Math.max(2, Math.ceil(4 / h))), inside = (i: number, j: number, k: number) => i >= edge[0] && j >= edge[1] && k >= edge[2] && i < nx - edge[0] && j < ny - edge[1] && k < nz - edge[2];
  const cf = [0, 1, 2].map((a) => full.dims[a] / cGrid.dims[a]);
  const cPts = voxelsRas(cGrid, (v, i, j, k) => Number(cMask.data[v]) >= 0.5 && inside((i + 0.5) * cf[0], (j + 0.5) * cf[1], (k + 0.5) * cf[2]));
  const fPts = voxelsRas(full, (v, i, j, k) => mask[v] === 1 && i % 2 === 0 && j % 2 === 0 && k % 2 === 0 && inside(i, j, k));
  const levels: Level[] = [
    { grid: cGrid, Si: inv4(cGrid.ijkToRAS), mm: coarseMm, P: pointsAt(cPts.ras, full, fd), idx: cPts.idx, iterations: 15 },
    { grid: full, Si, mm: 0, P: pointsAt(fPts.ras, full, fd), idx: fPts.idx, iterations: 8 },
  ];
  // The same levels for images ALREADY in the reference (field removed): step 4 aligns two of those, and reading them
  // through the field again moved every diffusion-weighted image by it (critic, 2026-10-05, finding 1: 2.2 mm on a still
  // synthetic head, 0.37 mm on PAT16).
  const levelsInReference: Level[] = levels.map((L) => ({ ...L, P: pointsAt(L.P.x, full, undefined) }));
  const targetsOf = (img: Float32Array) => levels.map((L) => { const g = L.mm > 0 ? coarsen({ ...full, data: img }, L.mm) : { ...full, data: img }; const o = new Float32Array(L.idx.length); for (let p = 0; p < o.length; p++) o[p] = Number(g.data[L.idx[p]]); return o; });

  // Each image put back, at each level's points (block means on the coarse level).
  const atLevels = (i: number, m: Move) => targetsOf(scatter(readMoved(data(i), full.dims, Si, regP, m)));
  // The predictions of every diffusion-weighted image at each level, and how well the images agree with them (relative
  // root mean square at the fine level's points).
  const predictAll = (mv: Move[]) => {
    let se = 0, sy = 0;
    const pred = new Map<number, Float32Array[]>();
    const dirs = dwi.gradients.map((g, i) => { const m = mv[i].R; return [m[0] * g[0] + m[3] * g[1] + m[6] * g[2], m[1] * g[0] + m[4] * g[1] + m[7] * g[2], m[2] * g[0] + m[5] * g[1] + m[8] * g[2]]; });
    const predict = scanPredictor(dwi.bValues, dirs);
    if (!predict) return { pred, rel: NaN };
    const Y = dwi.volumes.map((_, i) => atLevels(i, mv[i]));
    const per = levels.map((_, li) => predict(Y.map((y) => y[li])));
    for (const i of shells.flat()) {
      pred.set(i, levels.map((_, li) => per[li][i]));
      const fine = Y[i][1], pf = per[1][i];
      for (let p = 0; p < fine.length; p++) { const e = fine[p] - pf[p]; se += e * e; sy += fine[p] * fine[p]; }
    }
    return { pred, rel: sy > 0 ? Math.sqrt(se / sy) : NaN };
  };
  const before = predictAll(moves).rel;

  // Moves between the b = 0 images, at any image's place in the scan (linear in the six numbers).
  const atTime = (i: number, vecs: Map<number, number[]>) => {
    const s = [...vecs.keys()].sort((a, b) => a - b);
    if (i <= s[0]) return vecs.get(s[0])!;
    if (i >= s[s.length - 1]) return vecs.get(s[s.length - 1])!;
    let k = 0; while (s[k + 1] < i) k++;
    const u = (i - s[k]) / (s[k + 1] - s[k]), A = vecs.get(s[k])!, B = vecs.get(s[k + 1])!;
    return A.map((a, q) => a + u * (B[q] - a));
  };

  // Rule 3 needs a free round before the held one (critic, 2026-10-05, finding 6).
  const rounds = Math.max(opts.rounds ?? 2, rule === 3 && e ? 2 : 1);
  for (let round = 0; round < rounds; round++) {
    // 1. The b = 0 images to their mean (put back as far as known).
    if (b0i.length > 1) {
      const mean = new Float32Array(reg.idx.length);
      for (const i of b0i) { const r = readMoved(data(i), full.dims, Si, regP, moves[i]); for (let p = 0; p < r.length; p++) mean[p] += r[p] / b0i.length; }
      const T = targetsOf(scatter(mean));
      for (const i of b0i) { moves[i] = alignOne(data(i), full, levels, T, moves[i]); await yieldNow(); }
    }
    const b0Vecs = new Map(b0i.map((i) => [i, toVec(moves[i])]));
    // 2. Every other image starts, in the first round, where the b = 0 images say the head was.
    if (round === 0) for (let i = 0; i < N; i++) if (!b0Vecs.has(i)) moves[i] = fromVec(atTime(i, b0Vecs), c);
    // 3. Each diffusion-weighted image to its prediction from the rest of its shell.
    const { pred } = predictAll(moves);
    for (const s of shells) {
      if (!pred.has(s[0])) continue;
      // Rule 3's last round: the slopes held at the fit over all images, only the movement estimated again.
      const hold = rule === 3 && !!e && round === rounds - 1;
      for (const i of s) { moves[i] = alignOne(data(i), full, levels, pred.get(i)!, moves[i], e, hold); if (i % 4 === 0) await yieldNow(); }
    }
    // Rule 3: the slopes tied to the gradient, g = K (√(b/1000) · direction) + g₀, fitted over all diffusion-weighted images
    // after each free round.
    if (rule === 3 && e && round < rounds - 1) {
      const fitted = tieToGradient(shells.flat().map((i) => ({ q: dwi.gradients[i].map((x) => x * Math.sqrt(dwi.bValues[i] / 1000)), g: moves[i].ec?.g ?? [0, 0, 0] })));
      if (fitted) for (const i of shells.flat()) { const q = dwi.gradients[i].map((x) => x * Math.sqrt(dwi.bValues[i] / 1000)); moves[i] = { ...moves[i], ec: { g: fitted(q), e } }; }
    }
    // 4. All diffusion-weighted images onto the b = 0 images' frame: their extrapolation to b = 0 aligned to the b = 0
    //    mean (same contrast, both already in the reference), and the move found added to every one of them. A scan with
    //    one or two shells cannot be extrapolated: its images are pinned instead to where the b = 0 images say the head
    //    was at the same times. So is a scan whose common move comes out larger than a visit allows (a failed fit).
    const dw = shells.flat().sort((a, b) => a - b);
    const dirsNow = dwi.gradients.map((g, i) => { const m = moves[i].R; return [m[0] * g[0] + m[3] * g[1] + m[6] * g[2], m[1] * g[0] + m[4] * g[1] + m[7] * g[2], m[2] * g[0] + m[5] * g[1] + m[8] * g[2]]; });
    const w = dw.length ? interceptWeights(dwi.bValues, dirsNow, dw) : undefined;
    let m: Move | undefined;
    if (w) {
      const Y = dw.map((i) => readMoved(data(i), full.dims, Si, regP, moves[i])), icpt = new Float32Array(reg.idx.length);
      for (let p = 0; p < icpt.length; p++) {
        let top = 0; for (const y of Y) top = Math.max(top, y[p]);
        if (!(top > 0)) continue;
        const floor = Math.max(1e-3 * top, 1e-6); let l = 0;
        for (let r = 0; r < Y.length; r++) l += w[r] * Math.log(Math.max(Y[r][p], floor));
        icpt[p] = Math.min(Math.exp(l), 20 * top);
      }
      const b0Mean = new Float32Array(reg.idx.length);
      for (const i of b0i) { const r = readMoved(data(i), full.dims, Si, regP, moves[i]); for (let p = 0; p < r.length; p++) b0Mean[p] += r[p] / b0i.length; }
      m = alignOne(scatter(icpt), full, levelsInReference, targetsOf(scatter(b0Mean)), identity());
      const sz = rigidSize(m);
      if (sz.mm > COMMON_MOVE_LIMIT.mm || sz.degrees > COMMON_MOVE_LIMIT.degrees) m = undefined;
    }
    if (m) {
      // The common move first, then each image's own: y = Rᵢ(R_m(x − c) + c + t_m − c) + c + tᵢ.
      const cm = m;
      for (const i of dw) { const a = moves[i], R = mul3(a.R, cm.R), Rt = [0, 1, 2].map((r) => a.R[3 * r] * cm.t[0] + a.R[3 * r + 1] * cm.t[1] + a.R[3 * r + 2] * cm.t[2]); moves[i] = { ...a, R, t: [a.t[0] + Rt[0], a.t[1] + Rt[1], a.t[2] + Rt[2]], c }; }
    } else for (const s of shells) {
      const want = [0, 0, 0, 0, 0, 0], have = [0, 0, 0, 0, 0, 0];
      for (const i of s) { const a = atTime(i, b0Vecs), h = toVec(moves[i]); for (let q = 0; q < 6; q++) { want[q] += a[q] / s.length; have[q] += h[q] / s.length; } }
      for (const i of s) { const h = toVec(moves[i]); moves[i] = { ...moves[i], ...fromVec(h.map((x, q) => x + want[q] - have[q]), c) }; }
    }
  }
  const after = predictAll(moves).rel;

  // The corrected b = 0 mean on the whole grid (the alignment to the T1 reads it), field and movement both.
  const all = voxelsRas(full, () => true), allP = pointsAt(all.ras, full, fd), b0 = new Float32Array(nv);
  for (const i of b0i) { const r = readMoved(data(i), full.dims, Si, allP, moves[i]); for (let v = 0; v < nv; v++) b0[v] += r[v] / b0i.length; }
  const sizes = moves.map((m) => { const s = rigidSize(m); return { mm: +s.mm.toFixed(3), degrees: +s.degrees.toFixed(3) }; });
  const largest = { mm: Math.max(...sizes.map((s) => s.mm)), degrees: Math.max(...sizes.map((s) => s.degrees)) };
  // The largest eddy-current shift inside the brain (at the fine level's points), over all images.
  let eddyMm: number | undefined;
  if (e) {
    eddyMm = 0;
    const X = levels[1].P.x;
    for (const m of moves) { const g = m.ec?.g; if (!g) continue; for (let p = 0; p < X.length; p += 3) eddyMm = Math.max(eddyMm, Math.abs(g[0] * (X[p] - c[0]) + g[1] * (X[p + 1] - c[1]) + g[2] * (X[p + 2] - c[2])) / 100); }
    eddyMm = +eddyMm.toFixed(2);
  }
  const ms = performance.now() - t0;
  const eddySaid = e ? `; eddy-current stretch up to ${eddyMm!.toFixed(1)} mm` : rule >= 2 ? "; eddy currents not corrected (the phase-encoding direction is not known)" : "";
  return { rule: e ? (rule === 3 ? 3 : 2) : 1, moves, b0, mask, sizes, largest, ...(eddyMm !== undefined ? { eddyMm } : {}), residual: { before, after }, ms,
    said: `head movement corrected (largest ${largest.mm.toFixed(1)} mm and ${largest.degrees.toFixed(1)}°${eddySaid}; the images agree with each other ${(100 * (1 - after / before)).toFixed(0)}% better)` };
}

/** The scan with each image put back (field and movement, one interpolation) on its own grid: for a case without a T1. */
export async function applyMotion(dwi: DiffusionSeries, motion: Pick<MotionResult, "moves">, field?: Field): Promise<DiffusionSeries> {
  // The eddy-current shifts ride in each move (readMoved applies them); the gradients turn with the movement only.
  const src = dwi.volumes[0], full = { dims: src.dims as [number, number, number], ijkToRAS: src.ijkToRAS }, Si = inv4(full.ijkToRAS);
  const fdRaw = field ? fieldWithSlope(field.fit) : undefined, fd = fdRaw && field ? { ...fdRaw, axis: field.fit.axis, sign: field.sign } : undefined;
  const all = voxelsRas(full, () => true), P = pointsAt(all.ras, full, fd), volumes: DiffusionSeries["volumes"] = [];
  for (const [i, vol] of dwi.volumes.entries()) { volumes.push({ ...vol, data: readMoved(vol.data as ArrayLike<number>, full.dims, Si, P, motion.moves[i]), dtype: "<f4" } as DiffusionSeries["volumes"][number]); if (i % 8 === 7) await yieldNow(); }
  const gradients = dwi.gradients.map((g, i) => { const m = motion.moves[i].R; return [m[0] * g[0] + m[3] * g[1] + m[6] * g[2], m[1] * g[0] + m[4] * g[1] + m[7] * g[2], m[2] * g[0] + m[5] * g[1] + m[8] * g[2]] as [number, number, number]; });
  return { ...dwi, volumes, gradients };
}

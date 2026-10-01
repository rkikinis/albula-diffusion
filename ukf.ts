// UKF TRACTOGRAPHY -- the two-tensor model with free water (and the plain two-tensor model by a switch), on the
// processor: the reference the graphics-card version (ukf-gpu.ts) is checked against. Ron, 2026-09-29: "go ahead with
// UKF. GPU and parallelize as much as possible." Review: Contents/docs/ukf-review-2026-09-29.md (workspace).
//
// UKFTractography's authors (its README): Yogesh Rathi, Stefan Lienhard, Yinpeng Li, Martin Styner, Ipek Oguz, Yundi Shi,
// Christian Baumgartner, Ryan Eckbo, Tashrif Billah, Dheshan Mohandass. Its license: LICENSE-UKF.txt beside this file.
// PORTED FROM UKFTractography (github.com/pnlbwh/ukftractography, files ukf/filter_Simple2T_FW.cc,
// unscented_kalman_filter.cc, tractography.cc Init / Follow2T / Step2T / UnpackTensor, NrrdData.cc Interp3Signal,
// dwi_normalize.cc, cli.cc defaults), which is under the UKF Tractography Contribution and Software License Agreement:
//   "All or portions of this licensed product (such portions are the "Software") have been obtained under license from
//    The Brigham and Women's Hospital, Inc. and are subject to the following terms and conditions."
//   This is a modified version: translated to TypeScript and restructured, with the changes listed below. The Software is
//   for research use; it has not been reviewed or approved by the Food and Drug Administration or any other agency, and
//   clinical applications are neither recommended nor advised.
// The method: Malcolm, Shenton, Rathi, "Filtered multitensor tractography", IEEE TMI 29:1664, 2010; the information-form
// filter: Reddy & Rathi, Front Neurosci 10:166, 2016; free water in the filter: Baumgartner et al., CDMRI 2012.
//
// THE MODEL (Simple2T_FW, state of 11): x = [m1 (3), λ∥1, λ⊥1, m2 (3), λ∥2, λ⊥2, w], eigenvalues in 1e-6 mm²/s;
//   S/S0 = w·(½e^(−b uᵀD1u) + ½e^(−b uᵀD2u)) + (1 − w)·e^(−b·0.003),  Dk = λ⊥k·I + (λ∥k − λ⊥k)·mk·mkᵀ.
// Without free water (Simple2T, state of 10) w is absent and the tensor term stands alone.
//
// WHAT DIFFERS FROM THE ORIGINAL, each on purpose:
//  1. Coordinates: gradients and tensor directions live in the scan's VOXEL-AXIS frame (the direction cosines applied
//     to our patient-space gradients, DWI_CONVENTION 1), as the original does; points come out in patient RAS.
//  2. The original lists every gradient twice (with its opposite); both copies predict the same signal and see the same
//     measurement, so here each is used once at twice the weight (R = 2/Rs) -- the same information update, half the work.
//  3. Constraints (w in [0, 1], eigenvalues >= 0): the original solves a general quadratic program for the projection in
//     the filter's metric; for these box constraints the same projection is computed exactly by an active-set solve
//     (`project`), and the number of times it acts is counted.
//  4. The curvature stop is omitted: as the original computes it (radius = 1 / (|v2 − v1| / 2) of unit steps), the radius
//     is never below 1, so its threshold of 0.87 can never stop a fiber.
//  5. Seeds come as points (voxel coordinates) from the caller; the original's random offsets (seeds per voxel > 1)
//     are made by `seedsInVoxels` with a fixed pseudo-random sequence, so runs repeat exactly.

import { type DiffusionSeries, isotropicVolumes } from "./dwi.ts";

export const UKF_RULE = 1;

export interface UkfOptions {
  freeWater?: boolean;          // default true (Simple2T_FW)
  Qm?: number; Ql?: number; Qw?: number; Rs?: number;   // defaults 0.001, 50, 0.0015, 0.02 (the original's for 2T)
  stepLength?: number;          // mm, default 0.3
  recordLength?: number;        // mm, default 0.9 (a point every 3 steps)
  maxHalfFiberLength?: number;  // mm, default 250
  seedingThreshold?: number;    // FA of the seed tensor, default 0.18
  stoppingFA?: number;          // default 0.15
  stoppingThreshold?: number;   // mean predicted signal, default 0.1
  p0?: number;                  // initial covariance, default 0.01
  sigmaSignal?: number;         // mm, default: the smallest voxel size
}

/** The data the filter reads: signals divided by the mean b=0, voxel-major, and the geometry. */
export interface UkfData {
  dims: [number, number, number];
  voxel: [number, number, number];
  ijkToRAS: number[];
  /** Gradients (unit) in the voxel-axis frame, and b-values (s/mm²), for the volumes used. */
  g: Float64Array;              // G × 3
  b: Float64Array;              // G
  G: number;
  /** n × G normalized signal, voxel-major. */
  signal: Float32Array;
  mask: Uint8Array;
}

/** Prepare the data: drop b=0 (b <= 50) and trace volumes, divide by the mean b=0, turn gradients into the voxel frame. */
export function prepareUkfData(dwi: DiffusionSeries, mask: Uint8Array): UkfData {
  const [nx, ny, nz] = dwi.volumes[0].dims, n = nx * ny * nz, M = dwi.ijkToRAS;
  const cols = [0, 1, 2].map((c) => [M[c], M[4 + c], M[8 + c]]);
  const voxel = cols.map((c) => Math.hypot(c[0], c[1], c[2])) as [number, number, number];
  const R = cols.map((c, k) => c.map((x) => x / voxel[k]));            // R[k] = unit direction of voxel axis k, in RAS
  const iso = new Set(isotropicVolumes(dwi));
  const b0 = dwi.bValues.map((b, i) => (b <= 50 ? i : -1)).filter((i) => i >= 0);
  const use = dwi.bValues.map((b, i) => (b > 50 && !iso.has(i) ? i : -1)).filter((i) => i >= 0);
  if (!b0.length) throw new Error("UKF needs a b=0 volume to normalize by");
  const G = use.length, g = new Float64Array(3 * G), bv = new Float64Array(G);
  use.forEach((vi, q) => {
    const gr = dwi.gradients[vi];
    for (let k = 0; k < 3; k++) g[3 * q + k] = R[k][0] * gr[0] + R[k][1] * gr[1] + R[k][2] * gr[2];
    bv[q] = dwi.bValues[vi];
  });
  const base = new Float32Array(n);
  for (const i of b0) { const d = dwi.volumes[i].data; for (let v = 0; v < n; v++) base[v] += d[v] / b0.length; }
  const signal = new Float32Array(n * G);
  use.forEach((vi, q) => { const d = dwi.volumes[vi].data; for (let v = 0; v < n; v++) signal[v * G + q] = base[v] ? d[v] / base[v] : 0; });
  return { dims: [nx, ny, nz], voxel, ijkToRAS: M, g, b: bv, G, signal, mask };
}

// ── Small dense linear algebra (row-major, n ≤ 11) ────────────────────────────────────────────────────────────────

/** Lower Cholesky factor of a symmetric positive matrix (in a new array); null if not positive definite. */
export function cholesky(A: Float64Array, n: number): Float64Array | null {
  const L = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    let s = A[j * n + j];
    for (let k = 0; k < j; k++) s -= L[j * n + k] * L[j * n + k];
    if (!(s > 0)) return null;
    const d = Math.sqrt(s); L[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let t = A[i * n + j];
      for (let k = 0; k < j; k++) t -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = t / d;
    }
  }
  return L;
}
/** Inverse of a symmetric positive matrix through its Cholesky factor; null if not positive definite. */
export function spdInverse(A: Float64Array, n: number): Float64Array | null {
  const L = cholesky(A, n);
  if (!L) return null;
  const Li = new Float64Array(n * n);                 // L⁻¹ (lower)
  for (let j = 0; j < n; j++) {
    Li[j * n + j] = 1 / L[j * n + j];
    for (let i = j + 1; i < n; i++) {
      let s = 0;
      for (let k = j; k < i; k++) s -= L[i * n + k] * Li[k * n + j];
      Li[i * n + j] = s / L[i * n + i];
    }
  }
  const out = new Float64Array(n * n);                // (L⁻¹)ᵀ L⁻¹
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = 0;
    for (let k = i; k < n; k++) s += Li[k * n + i] * Li[k * n + j];
    out[i * n + j] = out[j * n + i] = s;
  }
  return out;
}

// ── The model ─────────────────────────────────────────────────────────────────────────────────────────────────────

const D_ISO = 0.003;
const UNPACK = 1e-6;

/** Predicted normalized signal for one state, into y (G values). */
function predict(x: Float64Array, off: number, fw: boolean, data: UkfData, y: Float64Array, yoff: number) {
  const { g, b, G } = data;
  const n1 = Math.hypot(x[off], x[off + 1], x[off + 2]) || 1, n2 = Math.hypot(x[off + 5], x[off + 6], x[off + 7]) || 1;
  const m1x = x[off] / n1, m1y = x[off + 1] / n1, m1z = x[off + 2] / n1, m2x = x[off + 5] / n2, m2y = x[off + 6] / n2, m2z = x[off + 7] / n2;
  const l1a = Math.max(x[off + 3], 0), l1p = Math.max(x[off + 4], 0), l2a = Math.max(x[off + 8], 0), l2p = Math.max(x[off + 9], 0);
  const w = fw ? Math.min(Math.max(x[off + 10], 0), 1) : 1;
  for (let q = 0; q < G; q++) {
    const ux = g[3 * q], uy = g[3 * q + 1], uz = g[3 * q + 2], bq = b[q] * UNPACK;
    const c1 = ux * m1x + uy * m1y + uz * m1z, c2 = ux * m2x + uy * m2y + uz * m2z;
    const t = 0.5 * Math.exp(-bq * (l1p + (l1a - l1p) * c1 * c1)) + 0.5 * Math.exp(-bq * (l2p + (l2a - l2p) * c2 * c2));
    y[yoff + q] = fw ? w * t + (1 - w) * Math.exp(-b[q] * D_ISO) : t;
  }
}

/** Box constraints of the state: indices and bounds (w in [0,1]; the four eigenvalues >= 0). */
function boxes(fw: boolean): { i: number; lo: number; hi: number }[] {
  const out = [3, 4, 8, 9].map((i) => ({ i, lo: 0, hi: Infinity }));
  if (fw) out.push({ i: 10, lo: 0, hi: 1 });
  return out;
}

/**
 * Project x onto the box constraints in the metric W (minimize (x' − x)ᵀW(x' − x)): the coordinates at a bound are
 * fixed there and the others move by −W_FF⁻¹ W_FA (x'_A − x_A); repeated until nothing else is violated. Exact for
 * these constraints when the active set settles, which it does in one or two rounds. Returns whether it acted.
 */
function project(x: Float64Array, off: number, W: Float64Array, n: number, fw: boolean): boolean {
  const bx = boxes(fw);
  const out = (v: number, c: { lo: number; hi: number }) => v < c.lo - 1e-12 || v > c.hi + 1e-12;
  if (!bx.some((c) => out(x[off + c.i], c))) return false;
  const orig = x.slice(off, off + n);
  const active = new Map<number, number>();
  for (let round = 0; round < 4; round++) {
    for (const c of bx) { const v = active.has(c.i) ? active.get(c.i)! : x[off + c.i]; if (out(v, c) || active.has(c.i)) active.set(c.i, Math.min(Math.max(v, c.lo), c.hi)); }
    const A = [...active.keys()], F = [...Array(n).keys()].filter((k) => !active.has(k));
    // y_F = −W_FF⁻¹ W_FA y_A, y_A = bound − orig.
    const yA = A.map((k) => active.get(k)! - orig[k]);
    const WFF = new Float64Array(F.length * F.length);
    F.forEach((r, a) => F.forEach((c, bb) => (WFF[a * F.length + bb] = W[r * n + c])));
    const inv = spdInverse(WFF, F.length);
    const rhs = F.map((r) => A.reduce((s, k, t) => s + W[r * n + k] * yA[t], 0));
    for (let k = 0; k < n; k++) x[off + k] = orig[k];
    A.forEach((k, t) => (x[off + k] = orig[k] + yA[t]));
    if (inv) F.forEach((r, a) => { let s = 0; for (let bb = 0; bb < F.length; bb++) s += inv[a * F.length + bb] * rhs[bb]; x[off + r] = orig[r] - s; });
    if (!bx.some((c) => out(x[off + c.i], c))) break;
  }
  for (const c of bx) x[off + c.i] = Math.min(Math.max(x[off + c.i], c.lo), c.hi);     // exact bounds after rounding
  return true;
}

export interface UkfFilter {
  n: number; fw: boolean; Q: Float64Array; R: number; kappa: number; weights: Float64Array; data: UkfData;
  /** How often the constraint projection acted (sigma points and states). */
  projections: number;
}

export function makeFilter(data: UkfData, opts: UkfOptions = {}): UkfFilter {
  const fw = opts.freeWater ?? true, n = fw ? 11 : 10;
  const Q = new Float64Array(n * n);
  const Qm = opts.Qm ?? 0.001, Ql = opts.Ql ?? 50, Qw = opts.Qw ?? 0.0015;
  for (const i of [0, 1, 2, 5, 6, 7]) Q[i * n + i] = Qm;
  for (const i of [3, 4, 8, 9]) Q[i * n + i] = Ql;
  if (fw) Q[10 * n + 10] = Qw;
  const kappa = 0.01, weights = new Float64Array(2 * n + 1);
  weights[0] = kappa / (n + kappa);
  for (let i = 1; i <= 2 * n; i++) weights[i] = 0.5 / (n + kappa);
  // Each gradient once, at twice the weight of the original's two copies (header, point 2).
  return { n, fw, Q, R: 2 / (opts.Rs ?? 0.02), kappa, weights, data, projections: 0 };
}

/** One filter step (the original's UnscentedKalmanFilter::Filter, information form). Returns false if it fails. */
export function filterStep(f: UkfFilter, x: Float64Array, P: Float64Array, z: Float64Array): boolean {
  const { n, fw, Q, R, weights, data } = f, S = 2 * n + 1, G = data.G;
  // Sigma points: x, x ± sqrt(n + κ)·L columns.
  const L = cholesky(P, n);
  if (!L) return false;
  const sc = Math.sqrt(n + f.kappa), X = new Float64Array(S * n);          // X[s*n + k]
  for (let s = 0; s < S; s++) for (let k = 0; k < n; k++) X[s * n + k] = x[k];
  for (let c = 0; c < n; c++) for (let k = 0; k < n; k++) { const v = sc * L[k * n + c]; X[(1 + c) * n + k] += v; X[(1 + n + c) * n + k] -= v; }
  for (let s = 0; s < S; s++) if (project(X, s * n, P, n, fw)) f.projections++;
  // F: normalize the directions (and round tiny negative values to zero).
  for (let s = 0; s < S; s++) {
    for (const o of [0, 5]) { const a = s * n + o, l = Math.hypot(X[a], X[a + 1], X[a + 2]) || 1; X[a] /= l; X[a + 1] /= l; X[a + 2] /= l; }
    for (const i of fw ? [3, 4, 8, 9, 10] : [3, 4, 8, 9]) if (X[s * n + i] < 0 && X[s * n + i] >= -1e-4) X[s * n + i] = 0;
  }
  const xh = new Float64Array(n);
  for (let s = 0; s < S; s++) for (let k = 0; k < n; k++) xh[k] += weights[s] * X[s * n + k];
  const Xd = new Float64Array(S * n);
  for (let s = 0; s < S; s++) for (let k = 0; k < n; k++) Xd[s * n + k] = X[s * n + k] - xh[k];
  const Pn = Float64Array.from(Q);
  for (let s = 0; s < S; s++) for (let i = 0; i < n; i++) { const wi = weights[s] * Xd[s * n + i]; for (let j = 0; j < n; j++) Pn[i * n + j] += wi * Xd[s * n + j]; }
  const Yk = spdInverse(Pn, n);
  if (!Yk) return false;
  const yh = new Float64Array(n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) yh[i] += Yk[i * n + j] * xh[j];
  // Predicted signals of every sigma point, their mean, and Pxz = Σ w Xd Zdᵀ (n × G).
  const Z = new Float64Array(S * G);
  for (let s = 0; s < S; s++) predict(X, s * n, fw, data, Z, s * G);
  const zh = new Float64Array(G);
  for (let s = 0; s < S; s++) for (let q = 0; q < G; q++) zh[q] += weights[s] * Z[s * G + q];
  const Pxz = new Float64Array(n * G);
  for (let s = 0; s < S; s++) for (let q = 0; q < G; q++) { const zd = weights[s] * (Z[s * G + q] - zh[q]); for (let k = 0; k < n; k++) Pxz[k * G + q] += Xd[s * n + k] * zd; }
  // Ht = Yk Pxz; I = R Ht Htᵀ; i = R Ht ((z − ẑ) + Pxzᵀ ŷ).
  const Ht = new Float64Array(n * G);
  for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) { const y = Yk[i * n + k]; if (y) for (let q = 0; q < G; q++) Ht[i * G + q] += y * Pxz[k * G + q]; }
  const innov = new Float64Array(G);
  for (let q = 0; q < G; q++) { let s = z[q] - zh[q]; for (let k = 0; k < n; k++) s += Pxz[k * G + q] * yh[k]; innov[q] = s; }
  const W = Float64Array.from(Yk), iv = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) { let s = 0; for (let q = 0; q < G; q++) s += Ht[i * G + q] * Ht[j * G + q]; W[i * n + j] += R * s; if (j < i) W[j * n + i] += R * s; }
    let s = 0; for (let q = 0; q < G; q++) s += Ht[i * G + q] * innov[q]; iv[i] = R * s + yh[i];
  }
  const Pnew = spdInverse(W, n);
  if (!Pnew) return false;
  for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += Pnew[i * n + j] * iv[j]; x[i] = s; }
  P.set(Pnew);
  if (project(x, 0, W, n, fw)) f.projections++;
  return true;
}

// ── Signal at a point, the seed tensor, tracking ──────────────────────────────────────────────────────────────────

/** The normalized signal at voxel position p (ijk), a Gaussian-weighted 3×3×3 average (the original's Interp3Signal). */
export function signalAt(data: UkfData, p: number[], sigma: number, out: Float64Array) {
  const [nx, ny, nz] = data.dims, G = data.G, v = data.voxel;
  out.fill(0);
  let wsum = 1e-16;
  const ci = Math.round(p[0]), cj = Math.round(p[1]), ck = Math.round(p[2]);
  for (let a = -1; a <= 1; a++) {
    const i = ci + a; if (i < 0 || i >= nx) continue;
    const dx = (i - p[0]) * v[0];
    for (let bb = -1; bb <= 1; bb++) {
      const j = cj + bb; if (j < 0 || j >= ny) continue;
      const dy = (j - p[1]) * v[1];
      for (let c = -1; c <= 1; c++) {
        const k = ck + c; if (k < 0 || k >= nz) continue;
        const dz = (k - p[2]) * v[2];
        const w = Math.exp(-(dx * dx + dy * dy + dz * dz) / sigma), o = ((k * ny + j) * nx + i) * G;
        for (let q = 0; q < G; q++) out[q] += w * data.signal[o + q];
        wsum += w;
      }
    }
  }
  for (let q = 0; q < G; q++) out[q] /= wsum;
}

function l2fa(l1: number, l2: number, l3: number): number {
  if (l2 === l3) return Math.abs(l1 - l2) / Math.sqrt(l1 * l1 + 2 * l2 * l2) || 0;
  return Math.sqrt(0.5 * ((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2) / (l1 * l1 + l2 * l2 + l3 * l3)) || 0;
}

/** The seed tensor from the normalized signal (least squares on the logs, no S0 term -- the original's UnpackTensor). */
export function seedTensor(data: UkfData, z: Float64Array): { m: number[]; l: [number, number, number] } | null {
  const { g, b, G } = data, AtA = new Float64Array(36), Atb = new Float64Array(6);
  for (let q = 0; q < G; q++) {
    const u = [g[3 * q], g[3 * q + 1], g[3 * q + 2]], bq = b[q];
    const row = [-bq * u[0] * u[0], -bq * 2 * u[0] * u[1], -bq * 2 * u[0] * u[2], -bq * u[1] * u[1], -bq * 2 * u[1] * u[2], -bq * u[2] * u[2]];
    const s = Math.log(z[q] > 0 ? z[q] : 10e-8);
    for (let i = 0; i < 6; i++) { Atb[i] += row[i] * s; for (let j = 0; j < 6; j++) AtA[i * 6 + j] += row[i] * row[j]; }
  }
  const inv = spdInverse(AtA, 6);
  if (!inv) return null;
  const d = new Float64Array(6);
  for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) d[i] += inv[i * 6 + j] * Atb[j];
  // Eigen decomposition of the symmetric 3×3 (Jacobi): the original takes singular values (= eigenvalues for a positive tensor).
  const A = [[d[0], d[1], d[2]], [d[1], d[3], d[4]], [d[2], d[4], d[5]]], V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 30; sweep++) {
    let offd = 0;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      offd += A[p][q] ** 2;
      if (Math.abs(A[p][q]) < 1e-30) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq; }
      for (let k = 0; k < 3; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < 3; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
    if (offd < 1e-40) break;
  }
  const ev = [0, 1, 2].map((k) => ({ l: Math.abs(A[k][k]), v: [V[0][k], V[1][k], V[2][k]] })).sort((a, c) => c.l - a.l);
  return { m: ev[0].v, l: [ev[0].l / UNPACK, ev[1].l / UNPACK, ev[2].l / UNPACK] };
}

export interface UkfFiber {
  /** Points in patient RAS, mm (x, y, z …), the backward half reversed and then the forward half. */
  points: Float32Array;
  /** FA of the leading tensor and free-water fraction w at each point. */
  fa: Float32Array;
  freeWater: Float32Array;
  seed: number;
}

export interface UkfResult { fibers: UkfFiber[]; seedsUsed: number; seedsRejected: number; steps: number; projections: number; ms: number }

/** Points (voxel coordinates) around each seed voxel: the voxel center, and with k > 1 also k − 1 offsets of 0.5 voxel
 *  in fixed pseudo-random directions (the original draws them with rand(); fixed here so runs repeat). */
export function seedsInVoxels(voxels: [number, number, number][], perVoxel = 1): number[][] {
  const out: number[][] = [];
  let s = 12345;
  const rnd = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1;
  const dirs: number[][] = [[0, 0, 0]];
  for (let k = 1; k < perVoxel; k++) { const d = [rnd(), rnd(), rnd()], l = Math.hypot(d[0], d[1], d[2]) || 1; dirs.push(d.map((x) => (0.5 * x) / l)); }
  for (const v of voxels) for (const d of dirs) out.push([v[0] + d[0], v[1] + d[1], v[2] + d[2]]);
  return out;
}

/** Track from every seed (voxel coordinates) in both directions, the original's Init + Follow2T + Step2T. */
export function trackUkf(data: UkfData, seeds: number[][], opts: UkfOptions = {}): UkfResult {
  const t0 = performance.now();
  const f = makeFilter(data, opts), n = f.n, fw = f.fw, G = data.G;
  const step = opts.stepLength ?? 0.3, perRecord = Math.max(1, Math.round((opts.recordLength ?? 0.9) / step));
  const maxSteps = Math.ceil((opts.maxHalfFiberLength ?? 250) / step), p0 = opts.p0 ?? 0.01;
  const seedFA = opts.seedingThreshold ?? 0.18, stopFA = opts.stoppingFA ?? 0.15, stopSignal = opts.stoppingThreshold ?? 0.1;
  const sigma = opts.sigmaSignal ?? Math.min(...data.voxel), M = data.ijkToRAS, [nx, ny, nz] = data.dims;
  const z = new Float64Array(G), yp = new Float64Array(G);
  const inBrain = (p: number[]) => {
    const i = Math.round(p[0]), j = Math.round(p[1]), k = Math.round(p[2]);
    return i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz && data.mask[(k * ny + j) * nx + i] > 0;
  };
  const toRAS = (p: number[]) => [M[0] * p[0] + M[1] * p[1] + M[2] * p[2] + M[3], M[4] * p[0] + M[5] * p[1] + M[6] * p[2] + M[7], M[8] * p[0] + M[9] * p[1] + M[10] * p[2] + M[11]];
  const fibers: UkfFiber[] = [];
  let used = 0, rejected = 0, steps = 0;

  /** One half fiber from a seed state; returns the recorded points (voxel coords) with FA and w. */
  const follow = (start: number[], x: Float64Array, P: Float64Array, m1Start: number[], faStart: number) => {
    const pts: number[][] = [start.slice()], fas: number[] = [faStart], ws: number[] = [fw ? x[10] : 1];
    const p = start.slice();
    let m1 = m1Start.slice();
    for (let stepnr = 1; ; stepnr++) {
      // Step2T: filter at the current point, orient and order the two tensors, move along the first.
      signalAt(data, p, sigma, z);
      if (!filterStep(f, x, P, z)) break;
      steps++;
      const old = m1;
      const unit = (o: number) => { const l = Math.hypot(x[o], x[o + 1], x[o + 2]) || 1; return [x[o] / l, x[o + 1] / l, x[o + 2] / l]; };
      let a = unit(0), c = unit(5);
      if (a[0] * old[0] + a[1] * old[1] + a[2] * old[2] < 0) a = a.map((v) => -v);
      if (c[0] * old[0] + c[1] * old[1] + c[2] * old[2] < 0) c = c.map((v) => -v);
      let la = [Math.max(x[3], 0), Math.max(x[4], 0)], lc = [Math.max(x[8], 0), Math.max(x[9], 0)];
      const angle = (Math.acos(Math.max(-1, Math.min(1, a[0] * c[0] + a[1] * c[1] + a[2] * c[2]))) * 180) / Math.PI;
      const swap = () => {
        [a, c] = [c, a]; [la, lc] = [lc, la];
        for (let k = 0; k < 5; k++) { const t = x[k]; x[k] = x[5 + k]; x[5 + k] = t; }
        // The covariance's two tensor blocks (and their rows / columns with w) exchange places.
        const perm = [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, ...(fw ? [10] : [])], Pc = Float64Array.from(P);
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) P[i * n + j] = Pc[perm[i] * n + perm[j]];
      };
      if (a[0] * old[0] + a[1] * old[1] + a[2] * old[2] < c[0] * old[0] + c[1] * old[1] + c[2] * old[2]) swap();
      let fa1 = l2fa(la[0], la[1], la[1]), fa2 = l2fa(lc[0], lc[1], lc[1]);
      if (angle <= 20 && Math.min(fa1, fa2) <= 0.2 && !(fa1 > 0.2)) { swap(); [fa1, fa2] = [fa2, fa1]; }
      const fa = la[0] < la[1] ? 0 : fa1;
      m1 = a;
      for (let k = 0; k < 3; k++) p[k] += (a[k] / data.voxel[k]) * step;
      // Follow2T's stops: out of the brain, predicted mean signal or FA too low, too long.
      predict(x, 0, fw, data, yp, 0);
      let mean = 0; for (let q = 0; q < G; q++) mean += yp[q]; mean /= G;
      if (!inBrain(p) || mean < stopSignal || fa < stopFA || stepnr > maxSteps) break;
      if ((stepnr + 1) % perRecord === 0) { pts.push(p.slice()); fas.push(fa); ws.push(fw ? x[10] : 1); }
    }
    return { pts, fas, ws };
  };

  seeds.forEach((seed, idx) => {
    signalAt(data, seed, sigma, z);
    for (let q = 0; q < G; q++) if (!(z[q] >= 0) || !Number.isFinite(z[q])) { rejected++; return; }
    const t = seedTensor(data, z);
    if (!t) { rejected++; return; }
    // THE SEED'S FA, as the original computes it for the simple model: from the major eigenvalue and the MEAN of the
    // two minor ones (tractography.cc 523-529, 541, 552). From the three as they are it came out higher, and the port
    // tracked 77 seeds of PAT16's 2,000 the original rejects (50 of them fibers of 10+ points; investigation,
    // 2026-10-01). With this, the port makes the original's 838 fibers.
    const l2 = (t.l[1] + t.l[2]) / 2, fa = l2fa(t.l[0], l2, l2);
    if (!(fa > seedFA)) { rejected++; return; }
    used++;
    const halves = [1, -1].map((sgn) => {
      const x = new Float64Array(n), P = new Float64Array(n * n);
      const m = t.m.map((v) => sgn * v);
      x.set([m[0], m[1], m[2], t.l[0], l2, t.m[0], t.m[1], t.m[2], t.l[0], l2]);     // the second tensor as in the original
      if (fw) x[10] = 1;
      for (let k = 0; k < n; k++) P[k * n + k] = p0;
      return follow(seed, x, P, m, fa);
    });
    const [fwd, bwd] = halves;
    const all = [...bwd.pts.slice(1).reverse(), ...fwd.pts], faAll = [...bwd.fas.slice(1).reverse(), ...fwd.fas], wAll = [...bwd.ws.slice(1).reverse(), ...fwd.ws];
    const pts = new Float32Array(3 * all.length);
    all.forEach((pp, i) => { const r = toRAS(pp); pts[3 * i] = r[0]; pts[3 * i + 1] = r[1]; pts[3 * i + 2] = r[2]; });
    fibers.push({ points: pts, fa: Float32Array.from(faAll), freeWater: Float32Array.from(wAll), seed: idx });
  });
  return { fibers, seedsUsed: used, seedsRejected: rejected, steps, projections: f.projections, ms: performance.now() - t0 };
}

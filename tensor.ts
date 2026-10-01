// THE DIFFUSION TENSOR, per voxel, on the processor -- the reference the graphics-card version (milestone 1, step 2:
// FA and color FA on the card, Contents/docs/dmri-review-2026-09-28.md) is checked against, and a checked
// implementation in its own right.
//
// The model: the signal of volume i is S_i = S0 · exp(−b_i · gᵢᵀ D gᵢ), D a symmetric 3×3 tensor (six numbers). Taking
// logs makes it linear in (ln S0, D), fitted over the volumes with b up to `maxB` (the tensor model holds at low b; a
// 2800 s/mm² shell bends it, so DTI is fitted on b ≤ 1500 by default, as DIPY and MRtrix3 advise for multi-shell data).
//
// Fitting: ordinary least squares on the logs first, then one weighted pass with weights S_fit² (the standard
// "WLS" fit: the log turns the noise's size upside down at low signal, and the weights put it back; Salvador 2005,
// Veraart 2013). Gradients are in patient RAS (DWI_CONVENTION 1), so D and its eigenvectors are in patient RAS too:
// the principal direction's x is left-right, whatever the scan's tilt.
//
// Checked against DIPY 1.12.1 (BSD; a check tool, not a dependency) on OpenNeuro ds001226 PAT16: FA within 0.0006,
// principal direction within 0.02°, mean diffusivity within 5e-7 (relative, 99th percentile), in all 151,244 brain voxels,
// with the same floor for zero signals (see minSignal). The physics check is in the test. 0.7 s for the brain on this Mac.
//
// Outputs per voxel: D (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz) in mm²/s, S0, the eigenvalues λ1 ≥ λ2 ≥ λ3, the principal
// eigenvector v1, FA and MD. Voxels outside `mask` (default: brainMask, below) are 0.
import { type DiffusionSeries, isotropicVolumes } from "./dwi.ts";

export interface TensorFit {
  dims: [number, number, number];
  ijkToRAS: number[];
  /** Six per voxel: Dxx, Dxy, Dxz, Dyy, Dyz, Dzz (mm²/s), patient RAS. */
  D: Float32Array;
  S0: Float32Array;
  /** λ1 ≥ λ2 ≥ λ3, three per voxel (mm²/s). */
  evals: Float32Array;
  /** The principal eigenvector, three per voxel, unit, patient RAS (its sign is arbitrary, as for any axis). */
  v1: Float32Array;
  fa: Float32Array;
  md: Float32Array;
  mask: Uint8Array;
  /** Which volumes were used, and how the mask was made -- for the record. */
  used: number[];
  maskRule: string;
}

/** Eigenvalues of a symmetric 3x3 (a, b, c on the diagonal; d = xy, e = xz, f = yz), descending. Closed form (Smith 1961). */
export function symEigenvalues(a: number, d: number, e: number, b: number, f: number, c: number): [number, number, number] {
  const p1 = d * d + e * e + f * f;
  if (p1 < 1e-30 * (a * a + b * b + c * c + 1e-300)) {
    const v = [a, b, c].sort((x, y) => y - x);
    return [v[0], v[1], v[2]];
  }
  const q = (a + b + c) / 3;
  const p2 = (a - q) ** 2 + (b - q) ** 2 + (c - q) ** 2 + 2 * p1;
  const p = Math.sqrt(p2 / 6);
  const B = [(a - q) / p, d / p, e / p, d / p, (b - q) / p, f / p, e / p, f / p, (c - q) / p];
  const detB = B[0] * (B[4] * B[8] - B[5] * B[7]) - B[1] * (B[3] * B[8] - B[5] * B[6]) + B[2] * (B[3] * B[7] - B[4] * B[6]);
  const r = Math.max(-1, Math.min(1, detB / 2));
  const phi = Math.acos(r) / 3;
  const l1 = q + 2 * p * Math.cos(phi);
  const l3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  return [l1, 3 * q - l1 - l3, l3];
}

/** The unit eigenvector of a symmetric 3x3 for eigenvalue `l`: the largest cross product of two rows of (M − l·I). */
export function symEigenvector(a: number, d: number, e: number, b: number, f: number, c: number, l: number): [number, number, number] {
  const r0 = [a - l, d, e], r1 = [d, b - l, f], r2 = [e, f, c - l];
  const cr = (u: number[], v: number[]) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  let best = [1, 0, 0], bn = 0;
  for (const v of [cr(r0, r1), cr(r0, r2), cr(r1, r2)]) {
    const n = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if (n > bn) { bn = n; best = v; }
  }
  if (bn === 0) return [1, 0, 0];
  const s = 1 / Math.sqrt(bn);
  return [best[0] * s, best[1] * s, best[2] * s];
}

export function fractionalAnisotropy(l1: number, l2: number, l3: number): number {
  const den = l1 * l1 + l2 * l2 + l3 * l3;
  if (!(den > 0)) return 0;
  return Math.min(1, Math.sqrt(0.5 * ((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2) / den));
}

/** Solve the 7x7 symmetric positive system A x = y in place (Cholesky); false when it is not positive definite. */
function chol7(A: Float64Array, y: Float64Array, x: Float64Array): boolean {
  const n = 7;
  for (let j = 0; j < n; j++) {
    let s = A[j * n + j];
    for (let k = 0; k < j; k++) s -= A[j * n + k] * A[j * n + k];
    if (!(s > 1e-300)) return false;
    const ljj = Math.sqrt(s);
    A[j * n + j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let t = A[i * n + j];
      for (let k = 0; k < j; k++) t -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = t / ljj;
    }
  }
  for (let i = 0; i < n; i++) { let t = y[i]; for (let k = 0; k < i; k++) t -= A[i * n + k] * x[k]; x[i] = t / A[i * n + i]; }
  for (let i = n - 1; i >= 0; i--) { let t = x[i]; for (let k = i + 1; k < n; k++) t -= A[k * n + i] * x[k]; x[i] = t / A[i * n + i]; }
  return true;
}

export interface TensorOptions {
  /** Highest b-value used (s/mm²); default 1500. */
  maxB?: number;
  /** Voxels to fit; default: brainMask (Otsu on the log b=0, holes filled). */
  mask?: Uint8Array;
  /** false: ordinary least squares only. Default true (one weighted pass). */
  weighted?: boolean;
  /** The floor for a signal at or below zero, whose logarithm does not exist; default 1, the smallest value an integer
   *  scan stores. It decides the tensor only in voxels where the signal has fallen to nothing (fluid at high b: about 1% of
   *  PAT16's brain). DIPY's default is 0.0001, which pulls those voxels hard; with min_signal=1 DIPY 1.12.1 and this fit
   *  agree on PAT16 to 0.0006 in FA and 0.02° in direction everywhere (2026-09-28). */
  minSignal?: number;
}

/**
 * THE BRAIN MASK from the b=0 volumes: Otsu's threshold on log(1 + mean b=0), then every hole the threshold left
 * inside the head filled (anything not connected to the grid's border through non-mask voxels is inside).
 *
 * Why this rule: on a b=0 image white matter is DARK and fluid very bright (PAT16: corpus callosum about 85, ventricles
 * up to 800), so a cut at a fraction of the bright end -- the first rule here, 2026-09-28 -- ran through the white
 * matter and left the corpus callosum full of holes, which stopped every streamline there after one step. The log puts
 * tissue and fluid near each other and far from the background, where Otsu's two-class split then falls; the hole
 * filling takes back anything dark that is surrounded by head.
 */
export function brainMask(dwi: DiffusionSeries, b0s: number[]): { mask: Uint8Array; rule: string } {
  const [nx, ny, nz] = dwi.volumes[0].dims;
  const n = nx * ny * nz;
  const lg = new Float32Array(n);
  for (const i of b0s) { const d = dwi.volumes[i].data; for (let v = 0; v < n; v++) lg[v] += Math.max(d[v], 0) / b0s.length; }
  let hi = 0;
  for (let v = 0; v < n; v++) { lg[v] = Math.log1p(lg[v]); if (lg[v] > hi) hi = lg[v]; }
  // Otsu on a 256-bin histogram of the logs.
  const bins = 256, hist = new Float64Array(bins);
  for (let v = 0; v < n; v++) hist[Math.min(bins - 1, Math.floor((lg[v] / (hi || 1)) * bins))]++;
  let sumAll = 0; for (let b = 0; b < bins; b++) sumAll += b * hist[b];
  let wB = 0, sumB = 0, best = -1, cut = 0;
  for (let b = 0; b < bins; b++) {
    wB += hist[b]; if (!wB) continue;
    const wF = n - wB; if (!wF) break;
    sumB += b * hist[b];
    const between = wB * wF * (sumB / wB - (sumAll - sumB) / wF) ** 2;
    if (between > best) { best = between; cut = b; }
  }
  const t = ((cut + 1) / bins) * hi;
  const mask = new Uint8Array(n);
  for (let v = 0; v < n; v++) mask[v] = lg[v] > t ? 1 : 0;
  // Fill enclosed holes: flood the outside from the border through non-mask voxels; what the flood never reaches is in.
  const outside = new Uint8Array(n), stack: number[] = [];
  const push = (v: number) => { if (!mask[v] && !outside[v]) { outside[v] = 1; stack.push(v); } };
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    if (i === 0 || j === 0 || k === 0 || i === nx - 1 || j === ny - 1 || k === nz - 1) push((k * ny + j) * nx + i);
  }
  while (stack.length) {
    const v = stack.pop()!, i = v % nx, j = Math.floor(v / nx) % ny, k = Math.floor(v / (nx * ny));
    if (i > 0) push(v - 1); if (i < nx - 1) push(v + 1);
    if (j > 0) push(v - nx); if (j < ny - 1) push(v + nx);
    if (k > 0) push(v - nx * ny); if (k < nz - 1) push(v + nx * ny);
  }
  let filled = 0;
  for (let v = 0; v < n; v++) if (!mask[v] && !outside[v]) { mask[v] = 1; filled++; }
  return { mask, rule: `Otsu on log(1 + mean b=0) (cut at b=0 ≈ ${Math.expm1(t).toFixed(1)}), ${filled} enclosed voxels filled` };
}

export function fitTensors(dwi: DiffusionSeries, opts: TensorOptions = {}): TensorFit {
  const maxB = opts.maxB ?? 1500;
  // Volumes up to maxB; a b > 0 volume with no direction (a trace image) says nothing about direction and is left out.
  const iso = new Set(isotropicVolumes(dwi));
  const used = dwi.bValues.map((b, i) => (b <= maxB && !iso.has(i) ? i : -1)).filter((i) => i >= 0);
  const b0s = used.filter((i) => dwi.bValues[i] < 50);
  if (!b0s.length) throw new Error("no b=0 volume: the tensor needs a reference signal");
  if (used.length - b0s.length < 6) throw new Error(`${used.length - b0s.length} diffusion-weighted volumes up to b=${maxB}: the tensor needs at least 6 directions`);
  const [nx, ny, nz] = dwi.volumes[0].dims;
  const n = nx * ny * nz;

  let mask = opts.mask, maskRule = "given";
  if (!mask) ({ mask, rule: maskRule } = brainMask(dwi, b0s));

  // Design rows: [ -b gx², -2b gx gy, -2b gx gz, -b gy², -2b gy gz, -b gz², 1 ] · (Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, ln S0) = ln S.
  const m = used.length;
  const X = new Float64Array(m * 7);
  used.forEach((vi, r) => {
    const b = dwi.bValues[vi], g = dwi.gradients[vi];
    const row = [-b * g[0] * g[0], -2 * b * g[0] * g[1], -2 * b * g[0] * g[2], -b * g[1] * g[1], -2 * b * g[1] * g[2], -b * g[2] * g[2], 1];
    for (let c = 0; c < 7; c++) X[r * 7 + c] = row[c];
  });
  // OLS once for all voxels: (XᵀX)⁻¹Xᵀ, 7 × m.
  const XtX = new Float64Array(49);
  for (let r = 0; r < m; r++) for (let i = 0; i < 7; i++) for (let j = 0; j < 7; j++) XtX[i * 7 + j] += X[r * 7 + i] * X[r * 7 + j];
  const pinv = new Float64Array(7 * m);
  {
    const A = new Float64Array(49), y = new Float64Array(7), x = new Float64Array(7);
    for (let r = 0; r < m; r++) {
      A.set(XtX);
      for (let c = 0; c < 7; c++) y[c] = X[r * 7 + c];
      if (!chol7(A, y, x)) throw new Error("the gradient directions do not determine a tensor (too few, or all in one plane)");
      for (let c = 0; c < 7; c++) pinv[c * m + r] = x[c];
    }
  }

  const D = new Float32Array(6 * n), S0 = new Float32Array(n), evals = new Float32Array(3 * n), v1 = new Float32Array(3 * n);
  const fa = new Float32Array(n), md = new Float32Array(n);
  const logS = new Float64Array(m), beta = new Float64Array(7), A = new Float64Array(49), y = new Float64Array(7);
  const vols = used.map((i) => dwi.volumes[i].data);
  const weighted = opts.weighted ?? true;
  const floor = opts.minSignal ?? 1;
  for (let v = 0; v < n; v++) {
    if (!mask[v]) continue;
    // Signals at or below zero have no logarithm; the floor (opts.minSignal) keeps them in the fit.
    for (let r = 0; r < m; r++) logS[r] = Math.log(Math.max(vols[r][v], floor));
    for (let c = 0; c < 7; c++) { let s = 0; for (let r = 0; r < m; r++) s += pinv[c * m + r] * logS[r]; beta[c] = s; }
    if (weighted) {
      A.fill(0); y.fill(0);
      for (let r = 0; r < m; r++) {
        let pred = 0;
        for (let c = 0; c < 7; c++) pred += X[r * 7 + c] * beta[c];
        const w = Math.exp(2 * Math.min(pred, 50));     // S_fit²
        for (let i = 0; i < 7; i++) {
          const wxi = w * X[r * 7 + i];
          y[i] += wxi * logS[r];
          for (let j = 0; j <= i; j++) A[i * 7 + j] += wxi * X[r * 7 + j];
        }
      }
      for (let i = 0; i < 7; i++) for (let j = i + 1; j < 7; j++) A[i * 7 + j] = A[j * 7 + i];
      const x = new Float64Array(7);
      if (chol7(A, y, x)) beta.set(x);
    }
    for (let c = 0; c < 6; c++) D[6 * v + c] = beta[c];
    S0[v] = Math.exp(beta[6]);
    const [l1, l2, l3] = symEigenvalues(beta[0], beta[1], beta[2], beta[3], beta[4], beta[5]);
    evals[3 * v] = l1; evals[3 * v + 1] = l2; evals[3 * v + 2] = l3;
    const e = symEigenvector(beta[0], beta[1], beta[2], beta[3], beta[4], beta[5], l1);
    v1[3 * v] = e[0]; v1[3 * v + 1] = e[1]; v1[3 * v + 2] = e[2];
    // Noise can make λ3 negative; for FA the eigenvalues are clipped at 0, as DIPY's decompose_tensor does (min_diffusivity 0).
    fa[v] = fractionalAnisotropy(Math.max(l1, 0), Math.max(l2, 0), Math.max(l3, 0));
    md[v] = (l1 + l2 + l3) / 3;
  }
  return { dims: [nx, ny, nz], ijkToRAS: dwi.ijkToRAS, D, S0, evals, v1, fa, md, mask, used, maskRule };
}

/** Color FA, RGB 0..1 per voxel: |v1| (x red = left-right, y green = front-back, z blue = up-down) times FA -- the
 *  standard coloring (Pajevic and Pierpaoli 1999), in patient space, so an oblique scan colors as the anatomy runs. */
export function colorFA(fit: TensorFit): Float32Array {
  const n = fit.fa.length, out = new Float32Array(3 * n);
  for (let v = 0; v < n; v++) for (let c = 0; c < 3; c++) out[3 * v + c] = Math.abs(fit.v1[3 * v + c]) * fit.fa[v];
  return out;
}

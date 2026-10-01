// RESPONSE FUNCTIONS FOR MULTI-SHELL MULTI-TISSUE CSD (csd.ts): what one tissue's signal looks like at each b-value --
// white matter as a single straight bundle (an axially symmetric tensor), gray matter and fluid as isotropic. Version 1
// follows DIPY's `mask_for_response_msmt` / `response_from_mask_msmt` / `multi_shell_fiber_response` (BSD) step for
// step, so it can be checked against DIPY's numbers (responses.test.ts); the method is Jeurissen et al., NeuroImage
// 103:411, 2014 (responses from tissue masks), the masks here from the tensor instead of a T1 segmentation.
//
// 1. MASKS, in a cube of 21 voxels around the image's center (DIPY's region; roiRadius 10), from a tensor fit on all
//    shells: white matter FA > 0.7; gray matter FA < 0.3 and MD < 1.0e-3 mm²/s; fluid FA < 0.15 and 0 < MD < 3.2e-3
//    (DIPY's multi-shell tutorial's thresholds). NOTE: DIPY's fluid mask takes MD BELOW its threshold (as documented
//    there), so it includes gray matter; csfRule "high-md" takes MD ABOVE it instead -- which one suits patients (edema
//    has high MD too) is to be measured (Contents/docs/csd-ptt-plan-2026-10-01.md in the workspace).
// 2. Per tissue and shell, a tensor fit on the mean b=0 image plus that shell, over the tissue's mask: the mean λ1 and the
//    mean λ2 give the response tensor [λ1, λ2, λ2]; the mean b=0 its S0.
// 3. THE KERNEL per shell: the white matter's tensor signal S0·exp(−b(λ1cos²θ + λ2sin²θ)) projected on the order-l zonal
//    harmonics (Gauss–Legendre quadrature; DIPY fits the same on a dense sphere); gray matter and fluid
//    S0·exp(−b·λ1) / Y00 (DIPY's form: the isotropic tissues' first eigenvalue). b = 0: S0 only.
import type { DiffusionSeries } from "./dwi.ts";
import type { Volume } from "albula";
import { fitTensors } from "./tensor.ts";
import type { Kernel } from "./csd.ts";

export const RESPONSE_RULE = 1;

export interface ResponseOptions {
  roiRadius?: number; wmFA?: number; gmFA?: number; gmMD?: number; csfFA?: number; csfMD?: number;
  csfRule?: "dipy" | "high-md"; b0Tolerance?: number; lmax?: number;
}
const DEFAULTS = { roiRadius: 10, wmFA: 0.7, gmFA: 0.3, gmMD: 0.001, csfFA: 0.15, csfMD: 0.0032, csfRule: "dipy" as const, b0Tolerance: 20, lmax: 8 };

/** The shells: b-values clustered within `tol` (DIPY's unique_bvals_tolerance), each cluster named by its smallest. */
export function shellsOf(bValues: ArrayLike<number>, tol = 20): number[] {
  const sorted = Float64Array.from(bValues).sort(), out: number[] = [];
  for (const b of sorted) if (!out.length || b - out[out.length - 1] > tol) out.push(b);
  return out;
}

/** One tissue's response per shell (b > 0): [λ1, λ2, λ2, S0]. */
export type TissueResponse = [number, number, number, number][];
export interface Responses { shells: number[]; wm: TissueResponse; gm: TissueResponse; csf: TissueResponse; counts: { wm: number; gm: number; csf: number }; rule: number }

export function estimateResponses(dwi: DiffusionSeries, opts: ResponseOptions = {}): Responses {
  const o = { ...DEFAULTS, ...opts };
  const [nx, ny, nz] = dwi.volumes[0].dims, n = nx * ny * nz;
  // 1. the region and the masks
  const cx = Math.floor(nx / 2), cy = Math.floor(ny / 2), cz = Math.floor(nz / 2), r = o.roiRadius;
  const roi = new Uint8Array(n);
  for (let k = Math.max(0, cz - r); k <= Math.min(nz - 1, cz + r); k++) for (let j = Math.max(0, cy - r); j <= Math.min(ny - 1, cy + r); j++) for (let i = Math.max(0, cx - r); i <= Math.min(nx - 1, cx + r); i++) roi[(k * ny + j) * nx + i] = 1;
  const t = fitTensors(dwi, { maxB: Infinity, mask: roi });
  const wm = new Uint8Array(n), gm = new Uint8Array(n), csf = new Uint8Array(n);
  for (let v = 0; v < n; v++) {
    if (!roi[v]) continue;
    const fa = Number.isFinite(t.fa[v]) ? t.fa[v] : 0, md = Number.isFinite(t.md[v]) ? t.md[v] : 0;
    if (fa > o.wmFA) wm[v] = 1;
    if (md < o.gmMD && fa < o.gmFA && fa > 0) gm[v] = 1;
    if (fa < o.csfFA && fa > 0 && (o.csfRule === "dipy" ? md < o.csfMD && md > 0 : md > o.csfMD)) csf[v] = 1;
  }
  // 2. per shell, per tissue
  const shells = shellsOf(dwi.bValues, o.b0Tolerance);
  const b0i = dwi.bValues.map((b, i) => (Math.abs(b - shells[0]) <= o.b0Tolerance ? i : -1)).filter((i) => i >= 0);
  const b0 = new Float32Array(n);
  for (const i of b0i) { const d = dwi.volumes[i].data; for (let v = 0; v < n; v++) b0[v] += Number(d[v]) / b0i.length; }
  const geo = dwi.volumes[0];
  const b0vol: Volume = { ...geo, data: b0, dtype: "<f4" } as Volume;
  const per = (mask: Uint8Array): TissueResponse => shells.slice(1).map((sh) => {
    const idx = dwi.bValues.map((b, i) => (Math.abs(b - sh) <= o.b0Tolerance ? i : -1)).filter((i) => i >= 0);
    const sub: DiffusionSeries = { ...dwi, volumes: [b0vol, ...idx.map((i) => dwi.volumes[i])], bValues: [0, ...idx.map((i) => dwi.bValues[i])], gradients: [[0, 0, 0], ...idx.map((i) => dwi.gradients[i])] };
    const f = fitTensors(sub, { maxB: Infinity, mask });
    let l1 = 0, l2 = 0, s0 = 0, c = 0;
    for (let v = 0; v < n; v++) { if (!mask[v]) continue; l1 += f.evals[3 * v]; l2 += f.evals[3 * v + 1]; s0 += b0[v]; c++; }
    return c ? [l1 / c, l2 / c, l2 / c, s0 / c] : [0, 0, 0, 0];
  });
  const count = (m: Uint8Array) => m.reduce((a, b) => a + b, 0);
  return { shells, wm: per(wm), gm: per(gm), csf: per(csf), counts: { wm: count(wm), gm: count(gm), csf: count(csf) }, rule: RESPONSE_RULE };
}

/** Legendre polynomial P_l(x). */
function legendre(l: number, x: number): number {
  let p0 = 1, p1 = x; if (l === 0) return 1;
  for (let k = 2; k <= l; k++) { const p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
  return p1;
}
/** Gauss–Legendre nodes and weights on [−1, 1] (Newton on P_n). */
function gaussLegendre(n: number): { x: number[]; w: number[] } {
  const x: number[] = [], w: number[] = [];
  for (let i = 1; i <= n; i++) {
    let z = Math.cos(Math.PI * (i - 0.25) / (n + 0.5)), dp = 0;
    for (let it = 0; it < 100; it++) {
      const p = legendre(n, z), q = legendre(n - 1, z);
      dp = n * (z * p - q) / (z * z - 1);
      const dz = p / dp; z -= dz; if (Math.abs(dz) < 1e-15) break;
    }
    x.push(z); w.push(2 / ((1 - z * z) * dp * dp));
  }
  return { x, w };
}

/** The kernel CSD uses: per shell (b = 0 first), [fluid, gray matter, white matter at l = 0, 2, …, lmax]. */
export function kernelFromResponses(r: Responses, lmax = 8): Kernel {
  const Y00 = 0.5 / Math.sqrt(Math.PI), { x, w } = gaussLegendre(64);
  const zonal = (f: (c: number) => number) => {
    const out: number[] = [];
    for (let l = 0; l <= lmax; l += 2) { let s = 0; for (let q = 0; q < x.length; q++) s += w[q] * f(x[q]) * legendre(l, x[q]); out.push(2 * Math.PI * Math.sqrt((2 * l + 1) / (4 * Math.PI)) * s); }
    return out;
  };
  const response: number[][] = [];
  const hasB0 = r.shells[0] < 20;
  if (hasB0) {
    const [, , , s0] = r.wm[0];
    response.push([r.csf[0][3] / Y00, r.gm[0][3] / Y00, ...zonal(() => s0)]);
  }
  r.shells.slice(hasB0 ? 1 : 0).forEach((b, i) => {
    const [l1, l2, , s0] = r.wm[i];
    response.push([r.csf[i][3] * Math.exp(-b * r.csf[i][0]) / Y00, r.gm[i][3] * Math.exp(-b * r.gm[i][0]) / Y00,
      ...zonal((c) => s0 * Math.exp(-b * (l1 * c * c + l2 * (1 - c * c))))]);
  });
  return { shells: r.shells, response, iso: 2, lmax };
}

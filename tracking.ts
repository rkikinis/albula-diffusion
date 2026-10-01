// TRACKING FROM SEEDS -- milestone 1, step 3 (Contents/docs/dmri-review-2026-09-28.md): "tracking from seeds placed in
// the view, drawn next to the tumor". This is the processor reference: deterministic streamlines that follow the
// tensor's principal direction, the method of Basser et al. 2000 and of Slicer's own seeding (SlicerDMRI's
// "Tractography Interactive Seeding" follows the same principal-direction rule for its single-tensor tracts).
//
// The rule, per step:
//  - the tensor at the current point is interpolated trilinearly from the eight surrounding voxels' tensors (the
//    tensor, not its direction: directions have no sign and do not average);
//  - the step follows its principal eigenvector, oriented to continue the previous step (sign chosen by the dot
//    product), second-order (a midpoint step, Runge-Kutta 2);
//  - stop when FA falls below `minFA`, the turn between two steps exceeds `maxAngleDeg`, the point leaves the mask or
//    the grid, or the length reaches `maxSteps`.
// Each seed is tracked both ways and the two halves joined, so a seed in the middle of a bundle gives the whole bundle.
//
// UNITS: stepping happens in VOXELS (Ron, 2026-09-24: "work in voxels, not mm"): `stepVoxels` is the step length in
// voxel units of the diffusion grid (0.5 by default: 1.25 mm on PAT16's 2.5 mm grid). Directions are turned from patient
// RAS (where the tensor lives, DWI_CONVENTION 1) into voxel steps by the grid's own matrix, so a tilted scan tracks as
// the anatomy runs. Output points are in patient RAS (mm), ready to draw with anything else in the scene.
import type { TensorFit } from "./tensor.ts";
import { fractionalAnisotropy, symEigenvalues, symEigenvector } from "./tensor.ts";

export interface TrackingOptions {
  /** Step length in voxels of the diffusion grid. Default 0.5. */
  stepVoxels?: number;
  /** Stop below this FA. Default 0.15. */
  minFA?: number;
  /** Stop when two consecutive steps turn by more than this. Default 45°. */
  maxAngleDeg?: number;
  /** Most steps in each direction from the seed. Default 2000. */
  maxSteps?: number;
  /** Shortest streamline kept, in points. Default 5. */
  minPoints?: number;
}

export interface Streamline {
  /** x, y, z per point, patient RAS, mm. */
  points: Float32Array;
  /** The seed's index in the list given. */
  seed: number;
  /** Why each end stopped. */
  stopped: [string, string];
}

function inv3x4(m: number[]): number[] {
  const a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], k = m[10];
  const A = e * k - f * h, B = -(d * k - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  const R = [A, -(b * k - c * h), b * f - c * e, B, a * k - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((x) => x / det);
  const t = [m[3], m[7], m[11]];
  return [R[0], R[1], R[2], -(R[0] * t[0] + R[1] * t[1] + R[2] * t[2]), R[3], R[4], R[5], -(R[3] * t[0] + R[4] * t[1] + R[5] * t[2]), R[6], R[7], R[8], -(R[6] * t[0] + R[7] * t[1] + R[8] * t[2])];
}

/** Track from each seed (patient RAS, mm) through a fitted tensor field. */
export function trackFromSeeds(fit: TensorFit, seedsRAS: number[][], opts: TrackingOptions = {}): Streamline[] {
  const step = opts.stepVoxels ?? 0.5, minFA = opts.minFA ?? 0.15, maxSteps = opts.maxSteps ?? 2000, minPoints = opts.minPoints ?? 5;
  const cosMax = Math.cos(((opts.maxAngleDeg ?? 45) * Math.PI) / 180);
  const [nx, ny, nz] = fit.dims, M = fit.ijkToRAS, Mi = inv3x4(M);
  // Patient-space direction -> voxel-space direction (the inverse's 3x3), then scaled to `step` voxels.
  const toVox = (d: number[]) => [Mi[0] * d[0] + Mi[1] * d[1] + Mi[2] * d[2], Mi[4] * d[0] + Mi[5] * d[1] + Mi[6] * d[2], Mi[8] * d[0] + Mi[9] * d[1] + Mi[10] * d[2]];
  const toRAS = (p: number[]) => [M[0] * p[0] + M[1] * p[1] + M[2] * p[2] + M[3], M[4] * p[0] + M[5] * p[1] + M[6] * p[2] + M[7], M[8] * p[0] + M[9] * p[1] + M[10] * p[2] + M[11]];
  const D6 = new Float64Array(6);

  /** The interpolated tensor's FA and principal direction (patient RAS) at voxel point p, or why not. */
  const probe = (p: number[]): { fa: number; dir: number[] } | string => {
    const x = p[0], y = p[1], z = p[2];
    if (!(x >= 0 && y >= 0 && z >= 0 && x <= nx - 1 && y <= ny - 1 && z <= nz - 1)) return "left the grid";
    const i0 = Math.min(Math.floor(x), nx - 2 < 0 ? 0 : nx - 2), j0 = Math.min(Math.floor(y), ny - 2 < 0 ? 0 : ny - 2), k0 = Math.min(Math.floor(z), nz - 2 < 0 ? 0 : nz - 2);
    const fx = x - i0, fy = y - j0, fz = z - k0;
    D6.fill(0);
    let wIn = 0;
    for (let c = 0; c < 8; c++) {
      const di = c & 1, dj = (c >> 1) & 1, dk = (c >> 2) & 1;
      const w = (di ? fx : 1 - fx) * (dj ? fy : 1 - fy) * (dk ? fz : 1 - fz);
      if (w === 0) continue;
      const v = ((k0 + dk) * ny + (j0 + dj)) * nx + (i0 + di);
      if (!fit.mask[v]) continue;
      wIn += w;
      for (let q = 0; q < 6; q++) D6[q] += w * fit.D[6 * v + q];
    }
    if (wIn < 0.5) return "left the brain";
    for (let q = 0; q < 6; q++) D6[q] /= wIn;
    const ev = symEigenvalues(D6[0], D6[1], D6[2], D6[3], D6[4], D6[5]);
    const fa = fractionalAnisotropy(Math.max(ev[0], 0), Math.max(ev[1], 0), Math.max(ev[2], 0));
    if (fa < minFA) return `FA below ${minFA}`;
    return { fa, dir: symEigenvector(D6[0], D6[1], D6[2], D6[3], D6[4], D6[5], ev[0]) };
  };
  /** A step of `step` voxels along patient direction d. */
  const stepAlong = (d: number[]) => { const v = toVox(d), l = Math.hypot(v[0], v[1], v[2]); return v.map((x) => (x / l) * step); };

  const half = (seed: number[], sign: 1 | -1): { pts: number[][]; why: string } => {
    const pts: number[][] = [];
    let p = seed.slice();
    const first = probe(p);
    if (typeof first === "string") return { pts, why: first };
    let prev = first.dir.map((x) => x * sign);
    for (let s = 0; s < maxSteps; s++) {
      // Midpoint step: direction here, half a step, direction there (oriented to continue), the full step from p.
      const a = probe(p);
      if (typeof a === "string") return { pts, why: a };
      let d1 = a.dir; if (d1[0] * prev[0] + d1[1] * prev[1] + d1[2] * prev[2] < 0) d1 = d1.map((x) => -x);
      const h = stepAlong(d1).map((x) => x / 2);
      const mid = [p[0] + h[0], p[1] + h[1], p[2] + h[2]];
      const b = probe(mid);
      if (typeof b === "string") return { pts, why: b };
      let d2 = b.dir; if (d2[0] * d1[0] + d2[1] * d1[1] + d2[2] * d1[2] < 0) d2 = d2.map((x) => -x);
      if (d2[0] * prev[0] + d2[1] * prev[1] + d2[2] * prev[2] < cosMax) return { pts, why: `turned more than ${opts.maxAngleDeg ?? 45}°` };
      const st = stepAlong(d2);
      p = [p[0] + st[0], p[1] + st[1], p[2] + st[2]];
      pts.push(p);
      prev = d2;
    }
    return { pts, why: `reached ${maxSteps} steps` };
  };

  const out: Streamline[] = [];
  seedsRAS.forEach((s, idx) => {
    const seed = [Mi[0] * s[0] + Mi[1] * s[1] + Mi[2] * s[2] + Mi[3], Mi[4] * s[0] + Mi[5] * s[1] + Mi[6] * s[2] + Mi[7], Mi[8] * s[0] + Mi[9] * s[1] + Mi[10] * s[2] + Mi[11]];
    const fwd = half(seed, 1), back = half(seed, -1);
    const all = [...back.pts.reverse(), seed, ...fwd.pts];
    if (all.length < minPoints) return;
    const pts = new Float32Array(3 * all.length);
    all.forEach((p, i) => { const r = toRAS(p); pts[3 * i] = r[0]; pts[3 * i + 1] = r[1]; pts[3 * i + 2] = r[2]; });
    out.push({ points: pts, seed: idx, stopped: [back.why, fwd.why] });
  });
  return out;
}

/** Seeds on a regular grid inside a sphere (patient RAS, mm): `perAxis` points per axis, spacing radius·2/perAxis. */
export function seedsInSphere(center: number[], radiusMm: number, perAxis = 5): number[][] {
  const s: number[][] = [];
  for (let a = 0; a < perAxis; a++) for (let b = 0; b < perAxis; b++) for (let c = 0; c < perAxis; c++) {
    const o = [a, b, c].map((t) => ((t + 0.5) / perAxis * 2 - 1) * radiusMm);
    if (Math.hypot(o[0], o[1], o[2]) <= radiusMm) s.push([center[0] + o[0], center[1] + o[1], center[2] + o[2]]);
  }
  return s;
}

// PARALLEL TRANSPORT TRACTOGRAPHY on the processor: the reference the graphics-card version will be checked against.
// Aydogan & Shi, "Parallel transport tractography", IEEE TMI 40(2):635-647, 2021 (Trekker); DIPY's PTTDirectionGetter
// (BSD) read as a second description. Plan: Contents/docs/csd-ptt-plan-2026-10-01.md (workspace).
//
// A streamline carries a FRAME (tangent T, normal N, binormal B) and two curvatures (k1 along N, k2 along B). A step of
// arc length s moves the point and turns the frame by the parallel-transport propagator (second order in s). At each
// step, candidate curvature pairs are drawn uniformly in the disc of radius kMax; each candidate's DATA SUPPORT is the
// FOD's value along a short probe ahead (probeQuality points over probeLength, each in the probe's own tangent
// direction); one is accepted by rejection sampling (support against twice the largest of a first sample). No accepted
// candidate within the tries ends the streamline. Every draw comes from a seeded generator: the same seeds give the same
// streamlines.
//
// UNITS: positions and steps in mm (patient RAS), stated per voxel in the options and converted with the grid's mean
// voxel size (Ron, 2026-09-24: "work in voxels, not mm"). The FOD's directions are patient RAS (csd-volume.ts).
//
// DIFFERS FROM DIPY, on purpose: DIPY moves its probe by the voxel size where its own step divides by it (it multiplies
// in calculate_data_support, divides in propagate), so on a 2.5 mm grid its probe reaches about 6x further than its step
// would -- to be confirmed with DIPY before saying more; here both use one unit (mm).
import { shBasis, shCount } from "./csd.ts";
import type { FodVolume } from "./csd-volume.ts";

export interface PttOptions {
  /** Step, in voxels (default 0.25). */
  stepVoxels?: number;
  /** Probe length, in voxels (default 1), and its points (default 4). */
  probeVoxels?: number; probeQuality?: number;
  /** Smallest radius of curvature, in voxels (default 1): kMax = 1 / radius. */
  minRadiusVoxels?: number;
  /** An FOD value below this fraction of the typical white-matter peak counts as no support (default 0.1). */
  fodThreshold?: number;
  /** Rejection sampling: candidates to estimate the largest support (10), tries (100). */
  samples?: number; tries?: number;
  /** Longest streamline, mm (default 250). */
  maxLengthMm?: number;
  seed?: number;
}

const DEF = { stepVoxels: 0.25, probeVoxels: 1, probeQuality: 4, minRadiusVoxels: 1, fodThreshold: 0.1, samples: 10, tries: 100, maxLengthMm: 250, seed: 20261001 };

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: number[]) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/** The FOD of a scan, read anywhere: coefficients interpolated between voxel centers, evaluated in any direction. */
export class FodField {
  readonly nc: number; private readonly inv: number[]; readonly voxelMm: number; readonly peak: number;
  constructor(readonly fod: FodVolume) {
    this.nc = shCount(fod.lmax);
    const m = fod.ijkToRAS, [a, b, c, d, e, f, g, h, i] = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
    const R = [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((x) => x / det);
    this.inv = [R[0], R[1], R[2], -(R[0] * m[3] + R[1] * m[7] + R[2] * m[11]), R[3], R[4], R[5], -(R[3] * m[3] + R[4] * m[7] + R[5] * m[11]), R[6], R[7], R[8], -(R[6] * m[3] + R[7] * m[7] + R[8] * m[11])];
    this.voxelMm = [0, 1, 2].map((k) => Math.hypot(m[k], m[4 + k], m[8 + k])).reduce((s, v) => s + v, 0) / 3;
    // the typical white-matter peak: the median, over voxels with a clear fiber, of the FOD's largest value on 200 directions
    const dirs = new Float64Array(600); for (let q = 0; q < 200; q++) { const z = 1 - (q + 0.5) / 200, r = Math.sqrt(1 - z * z), ph = q * Math.PI * (3 - Math.sqrt(5)); dirs.set([r * Math.cos(ph), r * Math.sin(ph), z], 3 * q); }
    const Bd = shBasis(fod.lmax, dirs), peaks: number[] = [], n = fod.coeffs.length / fod.nx;
    for (let v = 0; v < n; v += 7) {
      const o = v * fod.nx + 2; let mx = 0;
      for (let q = 0; q < 200; q++) { let s = 0; for (let k = 0; k < this.nc; k++) s += Bd[q * this.nc + k] * fod.coeffs[o + k]; mx = Math.max(mx, s); }
      if (mx > 0) peaks.push(mx);
    }
    peaks.sort((x, y) => x - y);
    this.peak = peaks.length ? peaks[Math.floor(peaks.length * 0.75)] : 1;
  }
  /** Voxel coordinates of a RAS point. */
  ijk(p: number[]): number[] { const M = this.inv; return [M[0] * p[0] + M[1] * p[1] + M[2] * p[2] + M[3], M[4] * p[0] + M[5] * p[1] + M[6] * p[2] + M[7], M[8] * p[0] + M[9] * p[1] + M[10] * p[2] + M[11]]; }
  inside(p: number[]): boolean {
    const [x, y, z] = this.ijk(p), [nx, ny, nz] = this.fod.dims;
    return x >= 0 && y >= 0 && z >= 0 && x <= nx - 1 && y <= ny - 1 && z <= nz - 1;
  }
  /** The FOD's value at a RAS point in a RAS direction (0 outside the grid), relative to the typical peak. */
  value(p: number[], dir: number[]): number {
    const [x, y, z] = this.ijk(p), [nx, ny, nz] = this.fod.dims;
    if (!(x >= 0 && y >= 0 && z >= 0 && x <= nx - 1 && y <= ny - 1 && z <= nz - 1)) return 0;
    const x0 = Math.min(nx - 2, Math.floor(x)), y0 = Math.min(ny - 2, Math.floor(y)), z0 = Math.min(nz - 2, Math.floor(z));
    const fx = x - x0, fy = y - y0, fz = z - z0, Y = shBasis(this.fod.lmax, dir);
    let s = 0;
    for (let c = 0; c < 8; c++) {
      const dx = c & 1, dy = (c >> 1) & 1, dz = (c >> 2) & 1, w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy) * (dz ? fz : 1 - fz);
      if (!w) continue;
      const o = (((z0 + dz) * ny + (y0 + dy)) * nx + (x0 + dx)) * this.fod.nx + 2;
      let a = 0; for (let k = 0; k < this.nc; k++) a += Y[k] * this.fod.coeffs[o + k];
      s += w * a;
    }
    return s / this.peak;
  }
  /** The direction of the FOD's largest value at a point, among `n` directions (a hemisphere), refined once. */
  peakDirection(p: number[], n = 400): number[] | undefined {
    let best: number[] | undefined, bv = 0;
    for (let q = 0; q < n; q++) {
      const z = 1 - (q + 0.5) / n, r = Math.sqrt(1 - z * z), ph = q * Math.PI * (3 - Math.sqrt(5)), d = [r * Math.cos(ph), r * Math.sin(ph), z];
      const v = this.value(p, d); if (v > bv) { bv = v; best = d; }
    }
    return best;
  }
}

/** One streamline (both directions from the seed, joined), or undefined when no direction is supported at the seed. */
export function pttStreamline(field: FodField, seed: number[], o: Required<PttOptions>, rand: () => number): Float32Array | undefined {
  const vox = field.voxelMm, step = o.stepVoxels * vox, probe = o.probeVoxels * vox, kMax = 1 / (o.minRadiusVoxels * vox);
  const dq = probe / (o.probeQuality - 1), thr = o.fodThreshold;
  const d0 = field.peakDirection(seed);
  if (!d0 || field.value(seed, d0) < thr) return undefined;
  const halves: number[][][] = [];
  for (const sign of [1, -1]) {
    let T = d0.map((x) => sign * x), N = norm(cross(T, Math.abs(T[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0])), B = cross(T, N);
    let pos = seed.slice(), k1 = 0, k2 = 0;
    const pts: number[][] = [];
    // a candidate's support: the FOD along the probe, each point in the probe's own tangent
    const support = (c1: number, c2: number): number => {
      let p = pos.slice(), t = T.slice(), n = N.slice(), b = B.slice(), sum = field.value(p, t) >= thr ? field.value(p, t) : 0;
      for (let q = 1; q < o.probeQuality; q++) {
        [p, t, n, b] = transport(p, t, n, b, c1, c2, dq);
        const v = field.value(p, t); sum += v >= thr ? v : 0;
      }
      return sum / o.probeQuality;
    };
    const draw = () => { for (;;) { const a = 2 * rand() - 1, c = 2 * rand() - 1; if (a * a + c * c <= 1) return [a * kMax, c * kMax]; } };
    let length = 0;
    for (;;) {
      // rejection sampling of the next curvature pair
      let maxS = 0; for (let i = 0; i < o.samples; i++) { const [a, c] = draw(); maxS = Math.max(maxS, support(a, c)); }
      if (maxS <= 0) break;
      let accepted = false;
      for (let i = 0; i < o.tries; i++) { const [a, c] = draw(); if (rand() * 2 * maxS <= support(a, c)) { k1 = a; k2 = c; accepted = true; break; } }
      if (!accepted) break;
      [pos, T, N, B] = transport(pos, T, N, B, k1, k2, step);
      if (!field.inside(pos)) break;
      pts.push(pos.slice()); length += step;
      if (length >= o.maxLengthMm / 2) break;
    }
    halves.push(pts);
  }
  const all = [...halves[1].reverse(), seed, ...halves[0]];
  if (all.length < 3) return undefined;
  const out = new Float32Array(all.length * 3); all.forEach((p, i) => out.set(p, 3 * i));
  return out;
}

/** One step of arc length s along the curve with curvatures (k1, k2): the parallel-transport propagator, second order. */
function transport(p: number[], T: number[], N: number[], B: number[], k1: number, k2: number, s: number): [number[], number[], number[], number[]] {
  const h = s * s / 2;
  const np = [0, 1, 2].map((i) => p[i] + s * T[i] + k1 * h * N[i] + k2 * h * B[i]);
  const t = norm([0, 1, 2].map((i) => (1 - (k1 * k1 + k2 * k2) * h) * T[i] + k1 * s * N[i] + k2 * s * B[i]));
  let n = [0, 1, 2].map((i) => -k1 * s * T[i] + (1 - k1 * k1 * h) * N[i] - k1 * k2 * h * B[i]);
  const b = norm(cross(t, n)); n = cross(b, t);
  return [np, t, n, b];
}

/** Streamlines from seeds (RAS mm), on the processor, in this thread. */
export function trackPtt(fod: FodVolume, seedsRAS: number[][], opts: PttOptions = {}): { streamlines: Float32Array[]; field: FodField } {
  const o = { ...DEF, ...opts } as Required<PttOptions>, field = new FodField(fod), rand = rng(o.seed), out: Float32Array[] = [];
  for (const s of seedsRAS) { const sl = pttStreamline(field, s, o, rand); if (sl) out.push(sl); }
  return { streamlines: out, field };
}

/** Streamlines from seeds in parallel workers (ptt-worker.ts). Each seed has its own seeded generator, so the result is
 *  the same however the seeds are split. */
export async function trackPttParallel(fod: FodVolume, seedsRAS: number[][], opts: PttOptions & { workers?: number; workerUrl?: URL; onProgress?: (f: number) => void } = {}): Promise<Float32Array[]> {
  const o = { ...DEF, ...opts } as Required<PttOptions>;
  const W = Math.max(1, opts.workers ?? Math.min(8, (navigator.hardwareConcurrency ?? 4) - 1)), CH = 500;
  const chunks: number[] = []; for (let s = 0; s < seedsRAS.length; s += CH) chunks.push(s);
  const result: (Float32Array | null)[] = new Array(seedsRAS.length).fill(null);
  let done = 0;
  const url = opts.workerUrl ?? new URL("./ptt-worker.ts", import.meta.url);
  const plain = { seed: o.seed, stepVoxels: o.stepVoxels, probeVoxels: o.probeVoxels, probeQuality: o.probeQuality, minRadiusVoxels: o.minRadiusVoxels, fodThreshold: o.fodThreshold, samples: o.samples, tries: o.tries, maxLengthMm: o.maxLengthMm };
  await Promise.all(Array.from({ length: Math.min(W, chunks.length) }, async () => {
    const w = new Worker(url, { type: "module" });
    try {
      w.postMessage({ init: { fod, opts: plain } });
      for (let first = chunks.shift(); first !== undefined; first = chunks.shift()) {
        const n = Math.min(CH, seedsRAS.length - first), seeds = new Float64Array(3 * n);
        for (let i = 0; i < n; i++) seeds.set(seedsRAS[first + i], 3 * i);
        const out: (Float32Array | null)[] = await new Promise((res, rej) => { w.onmessage = (e) => res(e.data); w.onerror = (e) => rej(new Error(e.message)); w.postMessage({ seeds, first }); });
        out.forEach((sl, i) => { result[first + i] = sl; });
        done += n; opts.onProgress?.(done / seedsRAS.length);
      }
    } finally { w.terminate(); }
  }));
  return result.filter((x): x is Float32Array => !!x);
}

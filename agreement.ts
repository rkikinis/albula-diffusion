// HOW MUCH TWO RUNS AGREE, AND HOW MUCH THEY WOULD DISAGREE ANYWAY -- Mike Halle's standard (his mail to Ron, 2026-10-02):
// a difference between two ways of computing something is judged against the scan's own noise, not against zero. Two
// measures he reports, written here from his description:
//
//   tract mix r      Pearson correlation of the share of streamlines each tract (and side) gets, between two runs;
//   center shift     for each tract both runs name (at least MIN_FOR_CENTER streamlines in each), the distance between
//                    the tract's centers (the mean of all its points); reported as the median and 95th percentile.
//
// and, for the same streamlines named twice, label agreement: the share that gets the same tract and side.
//
// THE NOISE FLOORS. Label noise: TractCloud's random context draws with another seed (`nameTracts` opts.seed). Scan noise:
// the scan with fresh noise of its own size added (`addScanNoise`): the noise level is measured from the background
// (outside the head, where a magnitude image is pure noise: Rayleigh, median = σ·√(2 ln 2)), and each voxel becomes
// |s + n₁ + i·n₂| with n₁, n₂ ~ N(0, σ²) -- a second acquisition's worth of noise on top of the first.
import type { DiffusionSeries } from "./dwi.ts";
import type { Named } from "./tractcloud/name-tracts.ts";
import { rng } from "./tractcloud/tractcloud.ts";

export const MIN_FOR_CENTER = 10;

export interface Agreement {
  /** Correlation of the tract shares (1 = the same mix). */
  mixR: number;
  /** Median and 95th percentile of the distance between the same tract's centers, mm; how many tracts were compared. */
  centerMedianMm: number;
  center95Mm: number;
  tractsCompared: number;
  /** For the same streamlines named twice: the share with the same tract and side; undefined when the streamlines differ. */
  labelAgreement?: number;
}

const key = (named: Named, i: number) => `${named.tract[i]}:${named.side[i]}`;

/** Agreement of two runs: streamlines and their names. Unnamed and short streamlines take part like any tract. */
export function agreement(slA: Float32Array[], namedA: Named, slB: Float32Array[], namedB: Named): Agreement {
  const share = (sl: Float32Array[], named: Named) => { const m = new Map<string, number>(); for (let i = 0; i < sl.length; i++) m.set(key(named, i), (m.get(key(named, i)) ?? 0) + 1 / sl.length); return m; };
  const a = share(slA, namedA), b = share(slB, namedB);
  const keys = [...new Set([...a.keys(), ...b.keys()])];
  const xa = keys.map((k) => a.get(k) ?? 0), xb = keys.map((k) => b.get(k) ?? 0);
  const mixR = pearson(xa, xb);
  const centers = (sl: Float32Array[], named: Named) => {
    const m = new Map<string, { s: number[]; n: number; count: number }>();
    for (let i = 0; i < sl.length; i++) {
      const k = key(named, i), c = m.get(k) ?? { s: [0, 0, 0], n: 0, count: 0 };
      const p = sl[i];
      for (let q = 0; q < p.length; q += 3) { c.s[0] += p[q]; c.s[1] += p[q + 1]; c.s[2] += p[q + 2]; c.n++; }
      c.count++; m.set(k, c);
    }
    return m;
  };
  const ca = centers(slA, namedA), cb = centers(slB, namedB);
  const d: number[] = [];
  for (const [k, u] of ca) {
    const v = cb.get(k);
    if (!v || u.count < MIN_FOR_CENTER || v.count < MIN_FOR_CENTER || !u.n || !v.n) continue;
    d.push(Math.hypot(u.s[0] / u.n - v.s[0] / v.n, u.s[1] / u.n - v.s[1] / v.n, u.s[2] / u.n - v.s[2] / v.n));
  }
  d.sort((x, y) => x - y);
  const q = (f: number) => d.length ? d[Math.min(d.length - 1, Math.floor(f * (d.length - 1) + 0.5))] : NaN;
  let labelAgreement: number | undefined;
  if (slA.length === slB.length && slA.every((s, i) => s === slB[i] || (s.length === slB[i].length && s.every((x, j) => x === slB[i][j])))) {
    let same = 0; for (let i = 0; i < slA.length; i++) if (key(namedA, i) === key(namedB, i)) same++;
    labelAgreement = slA.length ? same / slA.length : 1;
  }
  return { mixR, centerMedianMm: q(0.5), center95Mm: q(0.95), tractsCompared: d.length, ...(labelAgreement !== undefined ? { labelAgreement } : {}) };
}

function pearson(x: number[], y: number[]): number {
  const n = x.length; if (n < 2) return NaN;
  const mx = x.reduce((s, v) => s + v, 0) / n, my = y.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return sxy / Math.sqrt(sxx * syy);
}

/**
 * The scan's noise level σ from its background: one b = 0 volume's voxels outside `mask`. A magnitude image's background
 * is Rayleigh, whose MEDIAN is σ·√(2 ln 2) -- the median, not the mean, so ghosts and scalp in the upper tail do not count.
 */
export function noiseSigma(dwi: DiffusionSeries, mask: Uint8Array): number {
  const b0 = dwi.volumes.find((_, i) => dwi.bValues[i] < 50) ?? dwi.volumes[0];
  const vals: number[] = [];
  for (let v = 0; v < mask.length; v++) if (!mask[v]) vals.push(Math.abs(Number(b0.data[v])));
  vals.sort((a, b) => a - b);
  return (vals[Math.floor(vals.length / 2)] ?? 0) / Math.sqrt(2 * Math.LN2);
}

/** A copy of the scan with fresh Rician noise of size `sigma` (seeded, so a run can be repeated). */
export function addScanNoise(dwi: DiffusionSeries, sigma: number, seed: number): DiffusionSeries {
  const r = rng(seed);
  const gauss = () => { let u = 0; while (!u) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); };
  return {
    ...dwi,
    volumes: dwi.volumes.map((vol) => {
      const out = new Float32Array(vol.data.length);
      for (let v = 0; v < out.length; v++) { const s = Number(vol.data[v]); out[v] = Math.hypot(s + sigma * gauss(), sigma * gauss()); }
      return { ...vol, data: out, dtype: "<f4" };
    }),
  };
}

// TWO SETS OF FIBERS FROM THE SAME STARTING POINTS, COMPARED -- our UKF port against the original UKFTractography
// (ukf-reference.test.ts), and later against haversack's pipeline (Ron, 2026-10-01: the comparison is a test "forward
// looking"). Each starting point's fiber in one set is matched to the fiber in the other set that passes closest to
// that point; then, per matched pair: the length, and how far apart the two ends are (the ends paired whichever way is
// closer); and for the sets as a whole, how much of the brain each fiber set passes through (a density map on a grid)
// and how alike those maps are.

export interface FiberComparison {
  starts: number;
  /** Starting points with a fiber in A, in B, in both. */
  inA: number; inB: number; inBoth: number;
  /** Over the pairs: end distance (mm) median and 90th percentile; length ratio B/A median. */
  endMedianMm: number; end90Mm: number; lengthRatioMedian: number;
  /** Density maps (points per cell of `cellMm`) of the two whole sets: Pearson correlation, and the share of cells
   *  either set reaches that both reach. */
  densityCorrelation: number; densityOverlap: number;
}

const len = (f: Float32Array) => { let s = 0; for (let i = 3; i < f.length; i += 3) s += Math.hypot(f[i] - f[i - 3], f[i + 1] - f[i - 2], f[i + 2] - f[i - 1]); return s; };
const pctl = (a: number[], p: number) => { if (!a.length) return NaN; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * (s.length - 1) + 0.5))]; };

/** For each start, the index of the fiber passing closest to it (within `tolMm`), or -1. */
export function matchToStarts(fibers: Float32Array[], starts: number[][], tolMm = 1): number[] {
  const cell = Math.max(tolMm, 1), key = (x: number, y: number, z: number) => `${Math.floor(x / cell)},${Math.floor(y / cell)},${Math.floor(z / cell)}`;
  const grid = new Map<string, number[]>();
  fibers.forEach((f, fi) => { for (let i = 0; i < f.length; i += 3) { const k = key(f[i], f[i + 1], f[i + 2]); const l = grid.get(k) ?? grid.set(k, []).get(k)!; if (l[l.length - 1] !== fi) l.push(fi); } });
  return starts.map(([x, y, z]) => {
    let best = -1, bd = tolMm;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
      for (const fi of grid.get(key(x + a * cell, y + b * cell, z + c * cell)) ?? []) {
        const f = fibers[fi];
        for (let i = 0; i < f.length; i += 3) { const d = Math.hypot(f[i] - x, f[i + 1] - y, f[i + 2] - z); if (d < bd) { bd = d; best = fi; } }
      }
    }
    return best;
  });
}

export function compareFibers(a: Float32Array[], b: Float32Array[], starts: number[][], opts: { tolMm?: number; cellMm?: number } = {}): FiberComparison {
  const ma = matchToStarts(a, starts, opts.tolMm), mb = matchToStarts(b, starts, opts.tolMm);
  const ends: number[] = [], ratios: number[] = [];
  let inA = 0, inB = 0, inBoth = 0;
  starts.forEach((_, s) => {
    const fa = ma[s] >= 0 ? a[ma[s]] : undefined, fb = mb[s] >= 0 ? b[mb[s]] : undefined;
    if (fa) inA++; if (fb) inB++;
    if (!fa || !fb || fa.length < 6 || fb.length < 6) return;
    inBoth++;
    const e = (f: Float32Array, last: boolean) => last ? [f[f.length - 3], f[f.length - 2], f[f.length - 1]] : [f[0], f[1], f[2]];
    const d = (p: number[], q: number[]) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    const same = (d(e(fa, false), e(fb, false)) + d(e(fa, true), e(fb, true))) / 2, swap = (d(e(fa, false), e(fb, true)) + d(e(fa, true), e(fb, false))) / 2;
    ends.push(Math.min(same, swap));
    ratios.push(len(fb) / Math.max(1e-6, len(fa)));
  });
  // Density maps on a common grid.
  const cell = opts.cellMm ?? 2, da = new Map<string, number>(), db = new Map<string, number>();
  const fill = (set: Float32Array[], m: Map<string, number>) => { for (const f of set) for (let i = 0; i < f.length; i += 3) { const k = `${Math.floor(f[i] / cell)},${Math.floor(f[i + 1] / cell)},${Math.floor(f[i + 2] / cell)}`; m.set(k, (m.get(k) ?? 0) + 1); } };
  fill(a, da); fill(b, db);
  const keys = new Set([...da.keys(), ...db.keys()]);
  const xs: number[] = [], ys: number[] = [];
  let both = 0;
  for (const k of keys) { const x = da.get(k) ?? 0, y = db.get(k) ?? 0; xs.push(x); ys.push(y); if (x && y) both++; }
  const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / v.length, mx = mean(xs), my = mean(ys);
  let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < xs.length; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; syy += (ys[i] - my) ** 2; }
  return { starts: starts.length, inA, inB, inBoth, endMedianMm: pctl(ends, 0.5), end90Mm: pctl(ends, 0.9), lengthRatioMedian: pctl(ratios, 0.5),
    // Flat maps have no variance: identical flat maps correlate 1, different ones 0 (no NaN).
    densityCorrelation: sxx && syy ? sxy / Math.sqrt(sxx * syy) : xs.every((x, i) => x === ys[i]) ? 1 : 0, densityOverlap: keys.size ? both / keys.size : 0 };
}

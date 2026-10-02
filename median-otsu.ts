// THE BRAIN MASK AS DIPY MAKES IT: `dipy.segment.mask.median_otsu` with its defaults (DIPY 1.12.1; median_radius 4, numpass 4,
// no dilation, no hole or island removal, no cropping), written from its source (BSD-3) so the two can be compared voxel for
// voxel -- Mike Halle's pipeline uses a mask "identical voxel for voxel to DIPY's median_otsu" (his mail, 2026-10-02), the
// field's common mask. Steps, as DIPY does them:
//   1. b0 = the mean of the b = 0 volumes (float64);
//   2. a 9×9×9 median filter, four times (scipy.ndimage.median_filter, mode "reflect": the edge is mirrored, edge voxel
//      repeated -- d c b a | a b c d | d c b a);
//   3. Otsu's threshold on a 256-bin histogram of the filtered volume -- scikit-image's, which `median_otsu` uses when
//      scikit-image is installed (see otsuThreshold);
//   4. mask = filtered > threshold.
import type { DiffusionSeries } from "./dwi.ts";

export const MEDIAN_RADIUS = 4, NUMPASS = 4;

/** scipy's "reflect" index: -1 → 0, -2 → 1, n → n-1, n+1 → n-2. */
const reflect = (i: number, n: number) => { while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1; return i; };

/** One pass of the cubic median filter of radius r (window (2r+1)³, an odd count: the middle value). */
export function medianFilter(data: Float64Array, dims: number[], r: number): Float64Array {
  const [nx, ny, nz] = dims, w = 2 * r + 1, count = w * w * w, mid = count >> 1;
  const out = new Float64Array(data.length), buf = new Float64Array(count);
  const xi = new Int32Array(nx + 2 * r), yi = new Int32Array(ny + 2 * r), zi = new Int32Array(nz + 2 * r);
  for (let i = -r; i < nx + r; i++) xi[i + r] = reflect(i, nx);
  for (let j = -r; j < ny + r; j++) yi[j + r] = reflect(j, ny);
  for (let k = -r; k < nz + r; k++) zi[k + r] = reflect(k, nz);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    let c = 0;
    for (let dk = 0; dk < w; dk++) { const zo = zi[k + dk] * ny;
      for (let dj = 0; dj < w; dj++) { const yo = (zo + yi[j + dj]) * nx;
        for (let di = 0; di < w; di++) buf[c++] = data[yo + xi[i + di]]; } }
    out[(k * ny + j) * nx + i] = select(buf, mid);
  }
  return out;
}

/** The k-th smallest of `a` (Hoare's quickselect, median of three; `a` is reordered). */
function select(a: Float64Array, k: number): number {
  let lo = 0, hi = a.length - 1;
  while (hi > lo) {
    const m = (lo + hi) >> 1;
    if (a[m] < a[lo]) { const t = a[m]; a[m] = a[lo]; a[lo] = t; }
    if (a[hi] < a[lo]) { const t = a[hi]; a[hi] = a[lo]; a[lo] = t; }
    if (a[hi] < a[m]) { const t = a[hi]; a[hi] = a[m]; a[m] = t; }
    const p = a[m];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < p) i++;
      while (a[j] > p) j--;
      if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
    }
    if (k <= j) hi = j; else if (k >= i) lo = i; else return a[k];
  }
  return a[k];
}

/**
 * OTSU'S THRESHOLD AS `median_otsu` GETS IT: DIPY imports scikit-image's `threshold_otsu` when scikit-image is installed
 * (it is with DIPY 1.12.1 from pip, scikit-image 0.26) and its own `otsu` only otherwise -- the two differ (PAT16: 92,951
 * against 93,157 voxels; found 2026-10-02). The scikit-image one: np.histogram of 256 bins over [min, max], the bin
 * CENTERS as class values, the threshold the center of the best bin.
 */
export function otsuThreshold(v: Float64Array, nbins = 256): number {
  let mn = Infinity, mx = -Infinity;
  for (const x of v) { if (x < mn) mn = x; if (x > mx) mx = x; }
  if (mn === mx) return mn;                                            // one value: that value
  const edges = new Float64Array(nbins + 1);
  for (let b = 0; b <= nbins; b++) edges[b] = mn + (mx - mn) * b / nbins;     // np.linspace(mn, mx, nbins + 1)
  const hist = new Float64Array(nbins), norm = nbins / (mx - mn);
  for (const x of v) {
    let b = Math.floor((x - mn) * norm);
    if (b >= nbins) b = nbins - 1;
    if (b > 0 && x < edges[b]) b--;                                   // numpy's correction at the edges
    else if (b < nbins - 1 && x >= edges[b + 1]) b++;
    hist[b]++;
  }
  const c = new Float64Array(nbins);
  for (let b = 0; b < nbins; b++) c[b] = (edges[b] + edges[b + 1]) / 2;
  const w1 = new Float64Array(nbins), m1 = new Float64Array(nbins), w2 = new Float64Array(nbins), m2 = new Float64Array(nbins);
  let cw = 0, cm = 0;
  for (let b = 0; b < nbins; b++) { cw += hist[b]; cm += hist[b] * c[b]; w1[b] = cw; m1[b] = cm / cw; }
  cw = 0; cm = 0;
  for (let b = nbins - 1; b >= 0; b--) { cw += hist[b]; cm += hist[b] * c[b]; w2[b] = cw; m2[b] = cm / cw; }
  let best = -Infinity, idx = 0;
  for (let b = 0; b < nbins - 1; b++) {
    const v12 = w1[b] * w2[b + 1] * (m1[b] - m2[b + 1]) ** 2;
    if (v12 > best) { best = v12; idx = b; }                          // np.argmax: the first of equal maxima
  }
  return c[idx];
}

/**
 * THE SAME FILTER, FAST: a median depends only on the ORDER of the values, so the volume is replaced by the ranks of its
 * distinct values, and the median of each window is kept with a sliding histogram of ranks (Huang, Yang & Tang 1979, in
 * 3D): moving one voxel along a row takes out one 9×9 plane of the window and puts in another, and the median's rank moves
 * from where it was. Every pass picks values already in the volume, so all passes stay in the same ranks. The answer is
 * the same as `medianFilter` (tested); 12.6 s → about 1.4 s on PAT16.
 */
export function medianFilterPasses(data: Float64Array, dims: number[], r: number, passes: number): Float64Array {
  const n = data.length, [nx, ny, nz] = dims, w = 2 * r + 1, mid = (w * w * w) >> 1;
  const order = Float64Array.from(data).sort();
  const values: number[] = [];
  for (let q = 0; q < n; q++) if (q === 0 || order[q] !== order[q - 1]) values.push(order[q]);
  const rankOf = (x: number) => { let lo = 0, hi = values.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (values[m] < x) lo = m + 1; else hi = m; } return lo; };
  let cur = new Int32Array(n); for (let v = 0; v < n; v++) cur[v] = rankOf(data[v]);
  const xi = new Int32Array(nx + 2 * r), yi = new Int32Array(ny + 2 * r), zi = new Int32Array(nz + 2 * r);
  for (let i = -r; i < nx + r; i++) xi[i + r] = reflect(i, nx);
  for (let j = -r; j < ny + r; j++) yi[j + r] = reflect(j, ny);
  for (let k = -r; k < nz + r; k++) zi[k + r] = reflect(k, nz);
  // Two levels: each rank's count, and each block of BLOCK ranks' total, so the median can jump whole blocks -- after a
  // distortion correction the b0 has as many distinct values as voxels, and stepping rank by rank took 43 s on PAT16.
  const BLOCK = 256, hist = new Int32Array(values.length + 1), coarse = new Int32Array(Math.ceil((values.length + 1) / BLOCK) + 1);
  for (let p = 0; p < passes; p++) {
    const out = new Int32Array(n);
    let m = 0, below = 0;
    // The plane of the window at x index xx (already reflected), for row (j, k): w×w values, added (+1) or taken out (-1).
    const plane = (xx: number, j: number, k: number, sign: number) => {
      for (let dk = 0; dk < w; dk++) { const zo = zi[k + dk] * ny;
        for (let dj = 0; dj < w; dj++) { const v = cur[(zo + yi[j + dj]) * nx + xx]; hist[v] += sign; coarse[(v / BLOCK) | 0] += sign; if (v < m) below += sign; } }
    };
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) {
      for (let di = 0; di < w; di++) plane(xi[di], j, k, 1);                    // the row's first window
      for (let i = 0; i < nx; i++) {
        if (i > 0) { plane(xi[i - 1], j, k, -1); plane(xi[i + w - 1], j, k, 1); }
        while (below > mid) {
          if (m % BLOCK === 0 && m >= BLOCK && below - coarse[m / BLOCK - 1] > mid) { m -= BLOCK; below -= coarse[m / BLOCK]; }
          else { m--; below -= hist[m]; }
        }
        while (below + hist[m] <= mid) {
          if (m % BLOCK === 0 && below + coarse[m / BLOCK] <= mid) { below += coarse[m / BLOCK]; m += BLOCK; }
          else { below += hist[m]; m++; }
        }
        out[(k * ny + j) * nx + i] = m;
      }
      for (let di = 0; di < w; di++) plane(xi[nx - 1 + di], j, k, -1);          // empty the histogram for the next row
    }
    cur = out;
  }
  const res = new Float64Array(n); for (let v = 0; v < n; v++) res[v] = values[cur[v]];
  return res;
}

/** The mask (1 inside), DIPY's way, and the rule in words. */
export function medianOtsuMask(dwi: DiffusionSeries, b0s: number[]): { mask: Uint8Array; rule: string; threshold: number } {
  const dims = dwi.volumes[0].dims, n = dims[0] * dims[1] * dims[2];
  let f: Float64Array = new Float64Array(n);
  for (const i of b0s) { const d = dwi.volumes[i].data; for (let v = 0; v < n; v++) f[v] += Number(d[v]); }
  for (let v = 0; v < n; v++) f[v] /= b0s.length;
  f = medianFilterPasses(f, dims, MEDIAN_RADIUS, NUMPASS);
  const threshold = otsuThreshold(f);
  const mask = new Uint8Array(n);
  for (let v = 0; v < n; v++) mask[v] = f[v] > threshold ? 1 : 0;
  return { mask, threshold, rule: `DIPY's median_otsu (median radius ${MEDIAN_RADIUS}, ${NUMPASS} passes, Otsu threshold ${threshold.toFixed(1)}) on the mean of ${b0s.length} b=0 volumes` };
}

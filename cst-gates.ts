// THE CORTICOSPINAL TRACT'S ANATOMICAL GATES, automatic (Ron, 2026-10-07: "Now we need to automate. I am not a scalable
// resource"; "define the internal capsule and then CST is in the blue area. Depending on the level, that blue area also
// contains other fibers such as the sensory"). Two slices in the head's own frames (head-frame.ts), each read on the
// direction-colored map colored in that frame:
//   THE CRUS (brainstem frame): the crus is the largest region per side that runs up-down WITH a real left-right share
//     (pink: blue > 0.25 and red > 0.5 of FA, FA > 0.25), the green band (front-back, green > 0.8 of the larger of red and
//     blue, widened by one cell) cut out of it, so the band separates the crus from the tegmentum behind. A crossing is
//     errant when it lies outside the crus (1 mm of slack) and the way to the crus crosses the green band, or when it is
//     more than 3 mm from it. Against Ron's drawn borders on ten cases: 83% agreement per crossing (TRACT-REVIEW.md).
//   THE POSTERIOR LIMB (Talairach frame): the internal capsule's up-down (blue) band beside the thalamus -- the largest
//     region per side where blue is the larger part (> 0.6 of FA, FA > 0.25) within 10-35 mm of the midline. A crossing
//     outside it (1 mm of slack) is errant.
// A streamline passes a gate when it crosses the gate's slice and no crossing there is errant; one that does not reach
// the crus slice fails it -- a corticospinal fiber goes to the body (a thalamocortical sensory fiber, which shares the
// posterior limb's blue, ends in the thalamus). Display only: the stored tracts are not changed.

import { sliceCrossings } from "./tract-slice.ts";
import type { M4 } from "./head-frame.ts";

export const CST_GATES_RULE = 2;   // 2 (2026-10-07): the crus gate is the FA ridge's border

/** A slice of the color map, sampled on a plane: `rgb` per cell (row-major, v then u), in-plane coordinates in mm. */
export interface PlaneImage { nu: number; nv: number; step: number; u0: number; v0: number; rgb: Float32Array }
export interface PackedColorMap { dims: [number, number, number]; ijkToRAS: number[]; data: ArrayLike<number> }

/** The color map (packed RGB24, packRGB24) sampled on a slice matrix (sliceToRAS: x, y, normal columns, origin), nearest
 *  cell, a square of `half` mm around the origin's in-plane position. */
export function samplePlane(map: PackedColorMap, P: M4, half = 30, step = 0.5, center: [number, number] = [0, 0]): PlaneImage {
  const M = map.ijkToRAS, inv = invert3x4(M), [nx, ny, nz] = map.dims;
  const nu = Math.round((2 * half) / step), nv = nu, u0 = center[0] - half, v0 = center[1] - half, rgb = new Float32Array(3 * nu * nv);
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    const u = u0 + i * step, v = v0 + j * step;
    const p = [0, 1, 2].map((r) => P[4 * r + 3] + u * P[4 * r] + v * P[4 * r + 1]);
    const q = [0, 1, 2].map((r) => Math.round(inv[4 * r] * p[0] + inv[4 * r + 1] * p[1] + inv[4 * r + 2] * p[2] + inv[4 * r + 3]));
    if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[0] >= nx || q[1] >= ny || q[2] >= nz) continue;
    const x = Number(map.data[q[0] + nx * (q[1] + ny * q[2])]) | 0, c = 3 * (j * nu + i);
    rgb[c] = (x & 255) / 255; rgb[c + 1] = ((x >> 8) & 255) / 255; rgb[c + 2] = ((x >> 16) & 255) / 255;
  }
  return { nu, nv, step, u0, v0, rgb };
}
function invert3x4(M: number[]): number[] {
  const a = M[0], b = M[1], c = M[2], d = M[4], e = M[5], f = M[6], g = M[8], h = M[9], k = M[10];
  const det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
  const I = [(e * k - f * h) / det, (c * h - b * k) / det, (b * f - c * e) / det, (f * g - d * k) / det, (a * k - c * g) / det, (c * d - a * f) / det, (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det];
  const t = [M[3], M[7], M[11]];
  return [I[0], I[1], I[2], -(I[0] * t[0] + I[1] * t[1] + I[2] * t[2]), I[3], I[4], I[5], -(I[3] * t[0] + I[4] * t[1] + I[5] * t[2]), I[6], I[7], I[8], -(I[6] * t[0] + I[7] * t[1] + I[8] * t[2])];
}

const dilate = (m: Uint8Array, nu: number, nv: number, times: number): Uint8Array => {
  let a = m;
  for (let t = 0; t < times; t++) {
    const b = a.slice();
    for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) if (a[j * nu + i]) for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const x = i + di, y = j + dj; if (x >= 0 && y >= 0 && x < nu && y < nv) b[y * nu + x] = 1;
    }
    a = b;
  }
  return a;
};
/** The largest 4-connected region (edges, not corners: two regions touching at a corner stay two) of `m` whose cells' mean in-plane x has the sign `side` (and, optionally, lies within
 *  [minX, maxX] of |x|), at least `minCells`. */
function largestOnSide(m: Uint8Array, img: PlaneImage, side: 1 | -1, minCells: number, absX?: [number, number]): Uint8Array {
  const { nu, nv } = img, lab = new Int32Array(nu * nv); let best: { n: number; cells: number[] } | undefined, next = 0;
  for (let s = 0; s < m.length; s++) {
    if (!m[s] || lab[s]) continue;
    const cells: number[] = [s]; lab[s] = ++next;
    for (let k = 0; k < cells.length; k++) {
      const c = cells[k], i = c % nu, j = (c / nu) | 0;
      for (const [di, dj] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const x = i + di, y = j + dj; if (x < 0 || y < 0 || x >= nu || y >= nv) continue;
        const q = y * nu + x; if (m[q] && !lab[q]) { lab[q] = next; cells.push(q); }
      }
    }
    if (cells.length < minCells) continue;
    const mx = cells.reduce((a, c) => a + (img.u0 + (c % nu) * img.step), 0) / cells.length;
    if (Math.sign(mx) !== side) continue;
    if (absX && (Math.abs(mx) < absX[0] || Math.abs(mx) > absX[1])) continue;
    if (!best || cells.length > best.n) best = { n: cells.length, cells };
  }
  const out = new Uint8Array(nu * nv); for (const c of best?.cells ?? []) out[c] = 1;
  return out;
}

const fa = (img: PlaneImage, c: number) => Math.hypot(img.rgb[3 * c], img.rgb[3 * c + 1], img.rgb[3 * c + 2]);

/** The crus and the green band on a brainstem-frame slice; `side` is the in-plane x sign of the side (the slice's x is
 *  the head's left: the left side is +1). */
export function crusOf(img: PlaneImage, side: 1 | -1): { crus: Uint8Array; green: Uint8Array } {
  const n = img.nu * img.nv, green = new Uint8Array(n), cand = new Uint8Array(n);
  for (let c = 0; c < n; c++) {
    const r = img.rgb[3 * c], g = img.rgb[3 * c + 1], b = img.rgb[3 * c + 2], f = fa(img, c);
    if (f > 0.15 && g > 0.8 * Math.max(r, b)) green[c] = 1;
  }
  const g1 = dilate(green, img.nu, img.nv, 1);
  for (let c = 0; c < n; c++) {
    const r = img.rgb[3 * c], b = img.rgb[3 * c + 2], f = fa(img, c);
    if (f > 0.25 && b > 0.25 * f && r > 0.5 * f && !g1[c]) cand[c] = 1;
  }
  return { crus: largestOnSide(cand, img, side, 16), green };
}

/** The posterior limb's blue on a Talairach-frame slice at the internal capsule. */
export function posteriorLimbOf(img: PlaneImage, side: 1 | -1): Uint8Array {
  const n = img.nu * img.nv, cand = new Uint8Array(n);
  for (let c = 0; c < n; c++) { const b = img.rgb[3 * c + 2], f = fa(img, c); if (f > 0.25 && b > 0.6 * f) cand[c] = 1; }
  return largestOnSide(cand, img, side, 16, [10, 35]);
}

const cellOf = (img: PlaneImage, p: [number, number]) => {
  const i = Math.round((p[0] - img.u0) / img.step), j = Math.round((p[1] - img.v0) / img.step);
  return i < 0 || j < 0 || i >= img.nu || j >= img.nv ? -1 : j * img.nu + i;
};

/** Whether a crossing (in-plane mm) is errant at the crus. */
export function crusErrant(img: PlaneImage, crus: Uint8Array, green: Uint8Array, p: [number, number], crusSlack = dilate(crus, img.nu, img.nv, 2)): boolean {
  const c = cellOf(img, p); if (c < 0) return true;
  if (crusSlack[c]) return false;
  // The nearest crus cell (search outward up to 3 mm) and whether the way to it crosses the green band.
  const ci = c % img.nu, cj = (c / img.nu) | 0, R = Math.ceil(3 / img.step);
  let best: [number, number] | undefined, bd = Infinity;
  for (let dj = -R; dj <= R; dj++) for (let di = -R; di <= R; di++) {
    const x = ci + di, y = cj + dj; if (x < 0 || y < 0 || x >= img.nu || y >= img.nv || !crus[y * img.nu + x]) continue;
    const d = di * di + dj * dj; if (d < bd) { bd = d; best = [x, y]; }
  }
  if (!best || Math.sqrt(bd) * img.step > 3) return true;
  const steps = Math.max(Math.abs(best[0] - ci), Math.abs(best[1] - cj)) + 1;
  for (let k = 0; k <= steps; k++) { const t = k / steps, x = Math.round(ci + (best[0] - ci) * t), y = Math.round(cj + (best[1] - cj) * t); if (green[y * img.nu + x]) return true; }
  return false;
}

/**
 * THE CRUS BORDER FROM THE FA RIDGE (Ron, 2026-10-07: "Have you looked at FA in the crus region?" -- "yes, build the FA
 * ridge rule"). Across Ron's borders on ten cases the FA is a ridge 1-3 mm in front of his line (~0.58), falls through it
 * (~0.48) to a level behind (~0.43, with a faint dip -- the blurred substantia nigra). So, per column (in-plane x) on one
 * side, coming from the front: the brain's surface (smoothed FA past 0.35), the crus ridge (the FA maximum within 8 mm of
 * the surface where the red share -- the left-right part, the crus's oblique course -- is at least 0.35), and the border:
 * the first point behind the ridge where FA has fallen half-way to the level 3-6 mm behind the ridge. Against Ron's
 * borders (with cleanBorder): median 94% of crossings agree (mean 90%), median distance 2.8 mm; moving the slice 1-2 mm changes
 * the kept share by at most 22% on 19 of the 20 sides (PAT23 left 54%; the color rule: 30% or more on 17 of 72). The miss:
 * PAT23 (45% / 59%), tilted front to back. `side`: the in-plane x sign of the side.
 */
export const RIDGE = { enter: 0.35, ridgeMm: 8, red: 0.35, sigmaMm: 0.75, xMin: 3, xMax: 22, behind: [3, 6] as const, minRise: 0.05, maxDrop: 8 };
export function crusBorderFromRidge(img: PlaneImage, side: 1 | -1, o = RIDGE): [number, number][] {
  const { nu, nv, step } = img, n = nu * nv, fa = new Float32Array(n), red = new Float32Array(n);
  for (let c = 0; c < n; c++) { const r = img.rgb[3 * c], g = img.rgb[3 * c + 1], b = img.rgb[3 * c + 2]; fa[c] = Math.hypot(r, g, b); red[c] = r; }
  const fs = blur2(fa, nu, nv, o.sigmaMm / step), rs = blur2(red, nu, nv, o.sigmaMm / step);
  const v = (j: number) => img.v0 + j * step, out: [number, number][] = [];
  for (let i = 0; i < nu; i++) {
    const x = img.u0 + i * step; if (side * x < o.xMin || side * x > o.xMax) continue;
    const at = (j: number) => fs[j * nu + i], share = (j: number) => rs[j * nu + i] / Math.max(at(j), 1e-6);
    let j0 = -1; for (let j = nv - 1; j >= 0; j--) if (v(j) < 15 && at(j) > o.enter) { j0 = j; break; }
    if (j0 < 0) continue;
    let jr = -1; for (let j = j0; j >= 0 && v(j) >= v(j0) - o.ridgeMm; j--) if (share(j) >= o.red && (jr < 0 || at(j) > at(jr))) jr = j;
    if (jr < 0) continue;
    let sum = 0, cnt = 0; for (let j = jr; j >= 0; j--) { const d = v(jr) - v(j); if (d > o.behind[1]) break; if (d >= o.behind[0]) { sum += at(j); cnt++; } }
    if (!cnt) continue;
    const peak = at(jr), plateau = sum / cnt, half = plateau + 0.5 * (peak - plateau);
    if (peak - plateau < o.minRise) continue;
    let jb = -1; for (let j = jr - 1; j >= 0; j--) if (at(j) < half) { jb = j; break; }
    if (jb < 0 || v(jr) - v(jb) > o.maxDrop) continue;
    out.push([x, v(jb)]);
  }
  return cleanBorder(out);
}
/** A border made robust: a running median of y over +-2.5 mm of x; points more than 2 mm off it dropped (a column that
 *  caught the wrong surface or ridge -- spikes on CON03, CON04, PAT05, PAT07, PAT24, PAT28); the rest smoothed the same way. */
export function cleanBorder(pts: [number, number][], win = 2.5, tol = 2): [number, number][] {
  if (pts.length < 3) return pts;
  const s = [...pts].sort((a, b) => a[0] - b[0]);
  const med = (arr: [number, number][], x: number) => { const ys = arr.filter((p) => Math.abs(p[0] - x) <= win).map((p) => p[1]).sort((a, b) => a - b); const n = ys.length; return n % 2 ? ys[n >> 1] : (ys[n / 2 - 1] + ys[n / 2]) / 2; };
  const kept = s.filter((p) => Math.abs(p[1] - med(s, p[0])) <= tol);
  if (kept.length < 3) return [];
  return kept.map((p) => [p[0], med(kept, p[0])] as [number, number]);
}
/** Separable Gaussian blur of a 2D image (sigma in cells), edges clamped. */
function blur2(a: Float32Array, nu: number, nv: number, sigma: number): Float32Array {
  const r = Math.ceil(3 * sigma), k = Array.from({ length: 2 * r + 1 }, (_, i) => Math.exp(-((i - r) ** 2) / (2 * sigma * sigma))), ks = k.reduce((s, x) => s + x, 0);
  const tmp = new Float32Array(a.length), out = new Float32Array(a.length);
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) { let s = 0; for (let t = -r; t <= r; t++) s += k[t + r] * a[j * nu + Math.min(nu - 1, Math.max(0, i + t))]; tmp[j * nu + i] = s / ks; }
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) { let s = 0; for (let t = -r; t <= r; t++) s += k[t + r] * tmp[Math.min(nv - 1, Math.max(0, j + t)) * nu + i]; out[j * nu + i] = s / ks; }
  return out;
}
/** Is a point (in-plane mm) dorsal to a border given as points (x, y)? Compared with the border's y at the point's x
 *  (interpolated; beyond its ends, the nearer end's). */
export function dorsalOfBorder(border: [number, number][], p: [number, number]): boolean {
  const b = [...border].sort((a, c) => a[0] - c[0]); if (b.length < 2) return false;
  const [x, y] = p;
  if (x <= b[0][0]) return y < b[0][1];
  if (x >= b[b.length - 1][0]) return y < b[b.length - 1][1];
  let i = 1; while (b[i][0] < x) i++;
  const [x0, y0] = b[i - 1], [x1, y1] = b[i]; return y < y0 + ((x - x0) / Math.max(x1 - x0, 1e-9)) * (y1 - y0);
}

export interface GateResult { kept: Float32Array[]; failedCrus: number; failedLimb: number; noCrus: boolean; noLimb: boolean;
  /** The automatic crus border (in-plane mm) and the slice it lies on, for drawing it. */
  crusBorder?: [number, number][]; crusPlane?: M4 }

/** In-plane coordinates on a slice matrix. */
const inPlaneOf = (P: M4, p: ArrayLike<number>): [number, number] => {
  const d = [p[0] - P[3], p[1] - P[7], p[2] - P[11]];
  return [d[0] * P[0] + d[1] * P[4] + d[2] * P[8], d[0] * P[1] + d[1] * P[5] + d[2] * P[9]];
};

/**
 * THE GATES ON ONE SIDE'S TRACT. `crusPlane` / `limbPlane`: the two slices (sliceToRAS, radiological: x = the head's
 * left); `bsMap` / `talMap`: the color map colored in the brainstem and the Talairach frame; `side` -1 left, 1 right
 * (the head's). A gate whose anatomy cannot be found on its slice is not applied (said in noCrus / noLimb).
 */
export function gateTract(sl: Float32Array[], side: -1 | 1, crusPlane: M4, bsMap: PackedColorMap, limbPlane: M4, talMap: PackedColorMap): GateResult & { crusImg: PlaneImage; crus: Uint8Array; green: Uint8Array; crusBorder: [number, number][]; limbImg: PlaneImage; limb: Uint8Array } {
  const inSide = (side < 0 ? 1 : -1) as 1 | -1;   // the slice's x is the head's left
  const crusImg = samplePlane(bsMap, crusPlane, 30, 0.5, [0, 0]), { crus, green } = crusOf(crusImg, inSide);
  // The crus gate is the FA ridge's border (RULE 2, 2026-10-07): a crossing dorsal to it is errant. The color region
  // (crusOf) is kept for the picture only.
  const crusBorder = crusBorderFromRidge(crusImg, inSide);
  const limbImg = samplePlane(talMap, limbPlane, 40, 0.5, [0, 0]), limb = posteriorLimbOf(limbImg, inSide), limbSlack = dilate(limb, limbImg.nu, limbImg.nv, 2);
  const noCrus = crusBorder.length < 3, noLimb = !limb.some(Boolean);
  const plane = (P: M4) => ({ origin: [P[3], P[7], P[11]] as [number, number, number], normal: [P[2], P[6], P[10]] as [number, number, number] });
  const kept: Float32Array[] = []; let failedCrus = 0, failedLimb = 0;
  for (const f of sl) {
    if (!noCrus) {
      const cs = sliceCrossings([[f]], plane(crusPlane)).map((c) => inPlaneOf(crusPlane, c.p));
      if (!cs.length || cs.some((p) => dorsalOfBorder(crusBorder, p))) { failedCrus++; continue; }
    }
    if (!noLimb) {
      const cs = sliceCrossings([[f]], plane(limbPlane)).map((c) => inPlaneOf(limbPlane, c.p));
      if (!cs.length || cs.some((p) => { const c = cellOf(limbImg, p); return c < 0 || !limbSlack[c]; })) { failedLimb++; continue; }
    }
    kept.push(f);
  }
  return { kept, failedCrus, failedLimb, noCrus, noLimb, crusImg, crus, green, crusBorder, crusPlane, limbImg, limb };
}

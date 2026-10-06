// WHERE THE TRACTS CROSS A SLICE (Yogesh Rathi via Ron, 2026-10-01: "on the cross sections show the T1 image and
// intersections of the tubes on the slices"): every place a shown streamline passes through a slice plane, as a point
// on the plane, with the streamline's direction there (so a crossing can be drawn as a dot, or as a short stroke
// that shows which way the tract runs). Pure geometry; module.ts draws what this returns.

export interface Plane { origin: [number, number, number]; normal: [number, number, number] }
export interface Crossing { p: [number, number, number]; dir: [number, number, number]; set: number }

/**
 * Crossings of `sets[k]`'s streamlines (points in RAS mm) with the plane: a segment whose ends lie on opposite sides
 * (or one end on it) gives one crossing, interpolated along it. `set` says which entry of `sets` it came from.
 */
export function sliceCrossings(sets: Float32Array[][], plane: Plane): Crossing[] {
  const [nx0, ny0, nz0] = plane.normal, l = Math.hypot(nx0, ny0, nz0) || 1, nx = nx0 / l, ny = ny0 / l, nz = nz0 / l;
  const off = nx * plane.origin[0] + ny * plane.origin[1] + nz * plane.origin[2];
  const out: Crossing[] = [];
  sets.forEach((set, k) => {
    for (const f of set) {
      let prev = nx * f[0] + ny * f[1] + nz * f[2] - off;
      for (let i = 3; i < f.length; i += 3) {
        const cur = nx * f[i] + ny * f[i + 1] + nz * f[i + 2] - off;
        // One crossing per segment; a point exactly on the plane counts for the segment that ends there only.
        if ((prev < 0 && cur >= 0) || (prev > 0 && cur <= 0)) {
          const t = prev / (prev - cur), ax = f[i - 3], ay = f[i - 2], az = f[i - 1];
          const dx = f[i] - ax, dy = f[i + 1] - ay, dz = f[i + 2] - az, dl = Math.hypot(dx, dy, dz) || 1;
          out.push({ p: [ax + t * dx, ay + t * dy, az + t * dz], dir: [dx / dl, dy / dl, dz / dl], set: k });
        }
        prev = cur;
      }
    }
  });
  return out;
}

/**
 * A STREAMLINE WITH ITS ENDS SHORTENED, for drawing (Ron, 2026-10-01, after Add lines: "the ends of the tracts are
 * 'frazzled' any way make them slightly shorter?"). `mm` of arc length off each end, the cut points interpolated.
 * The streamline itself is not changed (leave the data, modulate the appearance). One too short to keep anything
 * gives null.
 */
export function trimEnds(f: Float32Array, mm: number): Float32Array | null {
  const n = f.length / 3;
  if (mm <= 0 || n < 2) return f;
  const cum = new Float64Array(n);
  for (let i = 1; i < n; i++) cum[i] = cum[i - 1] + Math.hypot(f[3 * i] - f[3 * i - 3], f[3 * i + 1] - f[3 * i - 2], f[3 * i + 2] - f[3 * i - 1]);
  const a = mm, b = cum[n - 1] - mm;
  if (b - a <= 1e-6) return null;
  const at = (t: number): number[] => {
    let i = 1; while (i < n - 1 && cum[i] < t) i++;
    const s = (t - cum[i - 1]) / Math.max(1e-12, cum[i] - cum[i - 1]);
    return [0, 1, 2].map((c) => f[3 * (i - 1) + c] + s * (f[3 * i + c] - f[3 * (i - 1) + c]));
  };
  const out: number[] = [...at(a)];
  for (let i = 0; i < n; i++) if (cum[i] > a && cum[i] < b) out.push(f[3 * i], f[3 * i + 1], f[3 * i + 2]);
  out.push(...at(b));
  return Float32Array.from(out);
}

/**
 * THE OUTLINE OF WHERE A TRACT CROSSES A SLICE (Ron, 2026-10-06, on the Tract review: dots hide the direction-colored
 * map under them, and that map is what shows the tract in the posterior limb of the internal capsule and in front of the
 * substantia nigra). Every crossing is widened to a disk of `radiusMm`, and the edge of their union is returned as
 * closed loops in RAS mm: the bundle becomes one outline with the map visible inside it, a stray fiber a small circle
 * of its own. `e1`, `e2` are the slice's in-plane axes (unit vectors), `origin` a point on it. Grid: `cellMm`.
 */
export function crossingOutlines(points: [number, number, number][], origin: [number, number, number], e1: [number, number, number], e2: [number, number, number], radiusMm = 1, cellMm = 0.5): [number, number, number][][] {
  if (!points.length) return [];
  const uv = points.map((p) => {
    const d = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    return [d[0] * e1[0] + d[1] * e1[1] + d[2] * e1[2], d[0] * e2[0] + d[1] * e2[1] + d[2] * e2[2]];
  });
  const pad = radiusMm + 2 * cellMm;
  let u0 = Infinity, v0 = Infinity, u1 = -Infinity, v1 = -Infinity;
  for (const [u, v] of uv) { u0 = Math.min(u0, u); v0 = Math.min(v0, v); u1 = Math.max(u1, u); v1 = Math.max(v1, v); }
  u0 -= pad; v0 -= pad;
  const W = Math.ceil((u1 + pad - u0) / cellMm), H = Math.ceil((v1 + pad - v0) / cellMm);
  if (W * H > 4e6) return [];   // a slice-wide spray: no outline is meaningful (and none is drawn)
  const fill = new Uint8Array(W * H), rc = Math.ceil(radiusMm / cellMm) + 1, r2 = radiusMm * radiusMm;
  for (const [u, v] of uv) {
    const ci = Math.floor((u - u0) / cellMm), cj = Math.floor((v - v0) / cellMm);
    for (let j = Math.max(0, cj - rc); j <= Math.min(H - 1, cj + rc); j++) for (let i = Math.max(0, ci - rc); i <= Math.min(W - 1, ci + rc); i++) {
      const du = u0 + (i + 0.5) * cellMm - u, dv = v0 + (j + 0.5) * cellMm - v;
      if (du * du + dv * dv <= r2) fill[j * W + i] = 1;
    }
  }
  const at = (i: number, j: number) => (i >= 0 && j >= 0 && i < W && j < H ? fill[j * W + i] : 0);
  // The filled cells' boundary edges, each running counterclockwise around its cell, keyed by the corner it starts at.
  const next = new Map<number, number[]>(), key = (i: number, j: number) => j * (W + 1) + i;
  const add = (a: number, b: number) => { const l = next.get(a); if (l) l.push(b); else next.set(a, [b]); };
  for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
    if (!fill[j * W + i]) continue;
    if (!at(i, j - 1)) add(key(i, j), key(i + 1, j));
    if (!at(i + 1, j)) add(key(i + 1, j), key(i + 1, j + 1));
    if (!at(i, j + 1)) add(key(i + 1, j + 1), key(i, j + 1));
    if (!at(i - 1, j)) add(key(i, j + 1), key(i, j));
  }
  const loops: [number, number, number][][] = [];
  for (const [start, outs] of next) {
    while (outs.length) {
      const corners: number[] = [start];
      let cur = outs.pop()!;
      while (cur !== start) { corners.push(cur); const o = next.get(cur); if (!o?.length) break; cur = o.pop()!; }
      // Corner cutting (two rounds) takes the grid's stairs off; the loop stays within half a cell of the edge.
      let pts = corners.map((c) => [u0 + (c % (W + 1)) * cellMm, v0 + Math.floor(c / (W + 1)) * cellMm]);
      for (let k = 0; k < 2; k++) pts = pts.flatMap((p, n) => { const q = pts[(n + 1) % pts.length]; return [[0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1]], [0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1]]]; });
      loops.push(pts.map(([u, v]) => [origin[0] + u * e1[0] + v * e2[0], origin[1] + u * e1[1] + v * e2[1], origin[2] + u * e1[2] + v * e2[2]]));
    }
  }
  return loops;
}

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

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

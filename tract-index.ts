// WHICH TRACTS PASS NEAR A POINT -- for the data probe (Ron, 2026-10-01: "the tracts are not in the data probe"). Every
// point of every streamline shown goes into a grid of cells; a question about one point looks in the cells around it
// only, so it answers at mouse speed with tens of thousands of streamlines. The answer per tract: how many of its
// streamlines pass within the radius, and the closest one's distance.
//
// THE CELL IS A SIZE OF THE DATA, not 2 mm (Ron, 2026-10-20: "did you forget about scales? from astrophysics to electron
// microscope"): the caller gives a voxel of the scan; without one, the streamlines' own point spacing is the cell. The
// answer does not depend on the cell (every point found is measured exactly); only the work does, so a radius many cells
// wide is answered by looking at every point instead of at more cells than there are points.

export interface TractIndex { cell: number; cells: Map<string, number[]>; xyz: Float32Array; owner: Int32Array; strand: Int32Array }

/** `sets[k]` is one tract's streamlines (points in patient RAS). `cell`: the grid's cell, in the scene's units (a voxel
 *  of the scan); by default the mean distance between a streamline's neighboring points. */
export function buildTractIndex(sets: Float32Array[][], cell?: number): TractIndex {
  let n = 0; for (const s of sets) for (const f of s) n += f.length / 3;
  const c = cell !== undefined && cell > 0 && Number.isFinite(cell) ? cell : pointSpacing(sets);
  const xyz = new Float32Array(n * 3), owner = new Int32Array(n), strand = new Int32Array(n), cells = new Map<string, number[]>();
  let p = 0, sid = 0;
  sets.forEach((s, k) => s.forEach((f) => {
    for (let i = 0; i < f.length; i += 3, p++) {
      xyz[3 * p] = f[i]; xyz[3 * p + 1] = f[i + 1]; xyz[3 * p + 2] = f[i + 2]; owner[p] = k; strand[p] = sid;
      const key = `${Math.floor(f[i] / c)},${Math.floor(f[i + 1] / c)},${Math.floor(f[i + 2] / c)}`;
      const l = cells.get(key); if (l) l.push(p); else cells.set(key, [p]);
    }
    sid++;
  }));
  return { cell: c, cells, xyz, owner, strand };
}

/** The mean distance between neighboring points of the streamlines (1 when there are none): the data's own scale. */
function pointSpacing(sets: Float32Array[][]): number {
  let sum = 0, m = 0;
  for (const s of sets) for (const f of s) for (let i = 3; i < f.length && m < 100000; i += 3, m++) sum += Math.hypot(f[i] - f[i - 3], f[i + 1] - f[i - 2], f[i + 2] - f[i - 1]);
  const d = m ? sum / m : 0;
  return d > 0 && Number.isFinite(d) ? d : 1;
}

/** The tracts with a streamline within `radius` of `ras` (the scene's units): index into `sets`, streamlines that
 *  close, the closest distance. */
export function tractsNear(ix: TractIndex, ras: [number, number, number], radius: number): { set: number; streamlines: number; closestMm: number }[] {
  const per = new Map<number, { strands: Set<number>; d: number }>();
  const visit = (p: number) => {
    const d = Math.hypot(ix.xyz[3 * p] - ras[0], ix.xyz[3 * p + 1] - ras[1], ix.xyz[3 * p + 2] - ras[2]);
    if (d > radius) return;
    const k = ix.owner[p], hit = per.get(k) ?? per.set(k, { strands: new Set(), d }).get(k)!;
    hit.strands.add(ix.strand[p]); if (d < hit.d) hit.d = d;
  };
  const r = Math.ceil(radius / ix.cell), n = ix.owner.length;
  // More cells around the point than points in all: every point, once (a radius far wider than the cell).
  if (!(Number.isFinite(r) && (2 * r + 1) ** 3 <= Math.max(27, n))) for (let p = 0; p < n; p++) visit(p);
  else {
    const c = ras.map((v) => Math.floor(v / ix.cell));
    for (let a = -r; a <= r; a++) for (let b = -r; b <= r; b++) for (let e = -r; e <= r; e++) {
      for (const p of ix.cells.get(`${c[0] + a},${c[1] + b},${c[2] + e}`) ?? []) visit(p);
    }
  }
  return [...per].map(([set, h]) => ({ set, streamlines: h.strands.size, closestMm: h.d })).sort((x, y) => x.closestMm - y.closestMm);
}

/**
 * WHERE ALONG A RAY THE FIRST STREAMLINE IS (the 3D probe, through the tubes in front): the first step that comes within
 * `radius` of a streamline point, on the stretch of the ray inside the box around them (the slab method), or null.
 * `d` is a unit direction; the answer is the distance along it. The step is the caller's (a size of the data), widened
 * so the march takes at most `maxSteps` (Ron, 2026-10-20: "did you forget about scales?" -- a fixed 0.5 mm step was
 * 6e21 steps across a 3e19 mm grid's tracts).
 */
export function firstAlongRay(ix: TractIndex, o: ArrayLike<number>, d: ArrayLike<number>, radius: number, step: number, maxSteps = 4096): number | null {
  const { xyz } = ix;
  if (!xyz.length) return null;
  let t0 = 0, t1 = Infinity;
  for (let a = 0; a < 3; a++) {
    let lo = Infinity, hi = -Infinity;
    for (let i = a; i < xyz.length; i += 3) { if (xyz[i] < lo) lo = xyz[i]; if (xyz[i] > hi) hi = xyz[i]; }
    lo -= radius; hi += radius;
    if (Math.abs(d[a]) < 1e-12) { if (o[a] < lo || o[a] > hi) return null; continue; }
    const ta = (lo - o[a]) / d[a], tb = (hi - o[a]) / d[a];
    t0 = Math.max(t0, Math.min(ta, tb)); t1 = Math.min(t1, Math.max(ta, tb));
  }
  if (!(t1 >= t0) || !Number.isFinite(t1)) return null;
  const s = Math.max(step > 0 ? step : 0, (t1 - t0) / maxSteps), n = s > 0 ? Math.floor((t1 - t0) / s) : 0;
  for (let k = 0; k <= n; k++) {
    const t = t0 + k * s;
    if (tractsNear(ix, [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t], radius).length) return t;
  }
  return null;
}

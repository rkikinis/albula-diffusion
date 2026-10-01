// WHICH TRACTS PASS NEAR A POINT -- for the data probe (Ron, 2026-10-01: "the tracts are not in the data probe"). Every
// point of every streamline shown goes into a grid of cells; a question about one point looks in the cells around it
// only, so it answers at mouse speed with tens of thousands of streamlines. The answer per tract: how many of its
// streamlines pass within the radius, and the closest one's distance.

export interface TractIndex { cell: number; cells: Map<string, number[]>; xyz: Float32Array; owner: Int32Array; strand: Int32Array }

/** `sets[k]` is one tract's streamlines (points in patient RAS, mm). */
export function buildTractIndex(sets: Float32Array[][], cellMm = 2): TractIndex {
  let n = 0; for (const s of sets) for (const f of s) n += f.length / 3;
  const xyz = new Float32Array(n * 3), owner = new Int32Array(n), strand = new Int32Array(n), cells = new Map<string, number[]>();
  let p = 0, sid = 0;
  sets.forEach((s, k) => s.forEach((f) => {
    for (let i = 0; i < f.length; i += 3, p++) {
      xyz[3 * p] = f[i]; xyz[3 * p + 1] = f[i + 1]; xyz[3 * p + 2] = f[i + 2]; owner[p] = k; strand[p] = sid;
      const key = `${Math.floor(f[i] / cellMm)},${Math.floor(f[i + 1] / cellMm)},${Math.floor(f[i + 2] / cellMm)}`;
      const l = cells.get(key); if (l) l.push(p); else cells.set(key, [p]);
    }
    sid++;
  }));
  return { cell: cellMm, cells, xyz, owner, strand };
}

/** The tracts with a streamline within `radiusMm` of `ras`: index into `sets`, streamlines that close, the closest distance. */
export function tractsNear(ix: TractIndex, ras: [number, number, number], radiusMm = 2): { set: number; streamlines: number; closestMm: number }[] {
  const r = Math.ceil(radiusMm / ix.cell), c = ras.map((v) => Math.floor(v / ix.cell));
  const per = new Map<number, { strands: Set<number>; d: number }>();
  for (let a = -r; a <= r; a++) for (let b = -r; b <= r; b++) for (let e = -r; e <= r; e++) {
    for (const p of ix.cells.get(`${c[0] + a},${c[1] + b},${c[2] + e}`) ?? []) {
      const d = Math.hypot(ix.xyz[3 * p] - ras[0], ix.xyz[3 * p + 1] - ras[1], ix.xyz[3 * p + 2] - ras[2]);
      if (d > radiusMm) continue;
      const k = ix.owner[p], hit = per.get(k) ?? per.set(k, { strands: new Set(), d }).get(k)!;
      hit.strands.add(ix.strand[p]); if (d < hit.d) hit.d = d;
    }
  }
  return [...per].map(([set, h]) => ({ set, streamlines: h.strands.size, closestMm: h.d })).sort((x, y) => x.closestMm - y.closestMm);
}

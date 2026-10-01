// DISTANCE TO A SEGMENT, in mm, for every voxel of the segment's grid: 0 inside, the Euclidean distance to the nearest
// inside voxel center outside (Felzenszwalb & Huttenlocher's exact squared distance transform, one axis at a time; the
// grid's spacing per axis from ijkToRAS's columns). Used for each tract's closest distance to the tumor.

function pass(f: Float64Array, n: number, w2: number, d: Float64Array, v: Int32Array, z: Float64Array) {
  let k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    if (f[q] === Infinity) continue;
    if (f[v[k]] === Infinity) { v[k] = q; continue; }
    let s: number;
    for (;;) {
      s = ((f[q] + w2 * q * q) - (f[v[k]] + w2 * v[k] * v[k])) / (2 * w2 * (q - v[k]));
      if (s <= z[k] && k > 0) k--; else break;
    }
    k++; v[k] = q; z[k] = s; z[k + 1] = Infinity;
  }
  if (f[v[0]] === Infinity && k === 0) { d.fill(Infinity, 0, n); return; }
  k = 0;
  for (let q = 0; q < n; q++) { while (z[k + 1] < q) k++; const r = q - v[k]; d[q] = w2 * r * r + f[v[k]]; }
}

export function distanceMap(inside: (index: number) => boolean, dims: number[], ijkToRAS: number[]): Float32Array {
  const [nx, ny, nz] = dims, n = nx * ny * nz, g = new Float64Array(n);
  for (let i = 0; i < n; i++) g[i] = inside(i) ? 0 : Infinity;
  const sp = [0, 1, 2].map((c) => Math.hypot(ijkToRAS[c], ijkToRAS[4 + c], ijkToRAS[8 + c]));
  const m = Math.max(nx, ny, nz), f = new Float64Array(m), d = new Float64Array(m), v = new Int32Array(m), z = new Float64Array(m + 1);
  const axis = (len: number, count: number, at: (line: number, q: number) => number, w: number) => {
    for (let line = 0; line < count; line++) {
      for (let q = 0; q < len; q++) f[q] = g[at(line, q)];
      pass(f, len, w * w, d, v, z);
      for (let q = 0; q < len; q++) g[at(line, q)] = d[q];
    }
  };
  axis(nx, ny * nz, (l, q) => l * nx + q, sp[0]);
  axis(ny, nx * nz, (l, q) => (Math.floor(l / nx) * ny + q) * nx + (l % nx), sp[1]);
  axis(nz, nx * ny, (l, q) => q * nx * ny + l, sp[2]);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sqrt(g[i]);
  return out;
}

// PARALLEL TRANSPORT TRACTOGRAPHY on synthetic fiber distributions (ptt.ts): one fiber everywhere gives straight
// streamlines along it; the same seeds give the same streamlines; a streamline stays where the data supports it.
import { assert, assertEquals } from "jsr:@std/assert@1";
import { shBasis, shCount } from "./csd.ts";
import type { FodVolume } from "./csd-volume.ts";
import { trackPtt } from "./ptt.ts";

/** A grid of 2 mm voxels whose FOD is a sharp single fiber along `dir` where `where(i,j,k)`, nothing elsewhere. */
function synthetic(dims: [number, number, number], dir: number[], where: (i: number, j: number, k: number) => boolean): FodVolume {
  const lmax = 8, nc = shCount(lmax), nx = 2 + nc, [X, Y, Z] = dims, coeffs = new Float32Array(X * Y * Z * nx);
  // the SH coefficients of a narrow lobe: the basis at the fiber direction, weighted toward low orders (a smooth peak)
  const Yd = shBasis(lmax, dir), w = [1, 1, 1, 1, 1];          // a truncated delta: as sharp as order 8 allows
  const lobe = new Float32Array(nc); let c = 0;
  for (let l = 0; l <= lmax; l += 2) for (let m = -l; m <= l; m++, c++) lobe[c] = Yd[c] * w[l / 2];
  for (let k = 0; k < Z; k++) for (let j = 0; j < Y; j++) for (let i = 0; i < X; i++) if (where(i, j, k)) coeffs.set(lobe, ((k * Y + j) * X + i) * nx + 2);
  return { dims, ijkToRAS: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1], coeffs, nx, lmax, frame: "RAS", seconds: 0, voxels: 0 };
}

Deno.test("one fiber along x everywhere: streamlines run straight along x, the whole grid", () => {
  const fod = synthetic([30, 12, 12], [1, 0, 0], () => true);
  const { streamlines } = trackPtt(fod, [[30, 12, 12], [20, 10, 14]], { seed: 1 });
  assertEquals(streamlines.length, 2);
  for (const s of streamlines) {
    const n = s.length / 3, dx = Math.abs(s[3 * (n - 1)] - s[0]);
    let off = 0; for (let i = 0; i < n; i++) off = Math.max(off, Math.hypot(s[3 * i + 1] - s[1], s[3 * i + 2] - s[2]));
    assert(dx > 50, `runs ${dx.toFixed(1)} mm along x (the grid is 58 mm)`);
    assert(off < 6, `strays ${off.toFixed(1)} mm sideways`);
  }
});

Deno.test("the same seeds give the same streamlines", () => {
  const fod = synthetic([20, 20, 20], [0.6, 0.8, 0], () => true);
  const seeds = [[20, 20, 20], [10, 30, 16]];
  const a = trackPtt(fod, seeds, { seed: 7 }).streamlines, b = trackPtt(fod, seeds, { seed: 7 }).streamlines;
  assertEquals(a.map((s) => Array.from(s)), b.map((s) => Array.from(s)));
});

Deno.test("a streamline ends where the data stops supporting it", () => {
  // fiber along x only in the band 8 <= j <= 12 (16-24 mm in y); the streamline stays inside it
  const fod = synthetic([30, 20, 10], [1, 0, 0], (_i, j) => j >= 8 && j <= 12);
  const { streamlines } = trackPtt(fod, [[30, 20, 10]], { seed: 3 });
  assertEquals(streamlines.length, 1);
  const s = streamlines[0];
  for (let i = 1; i < s.length; i += 3) assert(s[i] > 13 && s[i] < 27, `left the band: y = ${s[i].toFixed(1)}`);
});

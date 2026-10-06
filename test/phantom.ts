// THE SYNTHETIC HEAD for the head-movement and eddy-current tests (motion.test.ts) and for the check against FSL's eddy
// (Contents/tools/eddy-phantom.ts in the workspace): a diffusion series made exactly from a continuous model, with the
// tissue moved and the gradient turned with it, no interpolation, and eddy currents of a known size if asked.
import type { DiffusionSeries } from "../dwi.ts";
import { rotVec } from "../motion.ts";
import type { Rigid } from "../registration.ts";

export const DIMS: [number, number, number] = [48, 52, 38], H = 2.5;
export const M = [-H, 0, 0, 58.75, 0, H, 0, -63.75, 0, 0, H, -46.25, 0, 0, 0, 1];   // a scan in LAS-like orientation; the head inside it

/** The head at a point (RAS mm): its b = 0 brightness and its diffusion tensor (mm²/s), smooth and not symmetric. */
export function tissue(x: number, y: number, z: number): { s0: number; D: number[] } {
  const e = (x / 42) ** 2 + (y / 50) ** 2 + (z / 34) ** 2;
  if (e > 1.3) return { s0: 0, D: [0, 0, 0, 0, 0, 0] };
  const edge = 1 / (1 + Math.exp((e - 1) * 25));
  let s0 = 600 * edge + 500 * edge * Math.exp(-((x - 12) ** 2 + (y + 8) ** 2 + z * z) / 200) + 350 * edge * Math.exp(-((x + 15) ** 2 + (y - 18) ** 2 + (z - 8) ** 2) / 150);
  // Folds: a texture of ridges, as a cortex gives a real scan its edges (without it a smooth head barely shows a turn).
  s0 *= 1 + 0.25 * Math.sin(x / 5 + 0.5 * Math.sin(z / 8)) * Math.sin(y / 6 - z / 9);
  // Fibers: around an axis off the middle and tilted, along it further out; a "ventricle" of free water.
  const X = x - 8, Y = y + 6 - 0.3 * z, r = Math.hypot(X, Y) || 1, w = Math.exp(-((r - 18) ** 2) / 120);
  const d = [(-Y / r) * w + 0.3 * (1 - w), (X / r) * w, 1 - w], dl = Math.hypot(d[0], d[1], d[2]);
  const u = d.map((v) => v / dl), l1 = 1.7e-3, l2 = 0.3e-3, fw = Math.exp(-((x - 4) ** 2 + (y - 4) ** 2 + (z + 6) ** 2) / 80);
  const D = [0, 1, 2].flatMap((i) => [0, 1, 2].map((j) => (1 - fw) * ((l1 - l2) * u[i] * u[j] + (i === j ? l2 : 0)) + fw * (i === j ? 3e-3 : 0)));
  s0 += 400 * fw;
  return { s0, D: [D[0], D[4], D[8], D[1], D[2], D[5]] };
}

/** A move about a center: reference point → where it was in that image. */
export const move = (w: number[], t: number[], c: [number, number, number]): Rigid => ({ R: rotVec(w.map((a) => a * Math.PI / 180)), t: t as [number, number, number], c });

/** Eddy currents in the phantom: per image a shift along e (RAS unit vector) of d0 + g·p/100 mm at scanner point p. */
export interface Eddy { e: [number, number, number]; g: number[][]; d0: number[] }

export function scan(moves: Rigid[], bValues: number[], gradients: [number, number, number][], noise = 0, eddy?: Eddy): DiffusionSeries {
  const [nx, ny, nz] = DIMS;
  let seed = 7;
  const rnd = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const volumes = moves.map((m, v) => {
    const data = new Float32Array(nx * ny * nz), R = m.R, g0 = gradients[v];
    // The gradient as the tissue felt it: Rᵀ g. The tissue at scanner point p came from the reference point R⁻¹(p − c − t) + c.
    const g = [R[0] * g0[0] + R[3] * g0[1] + R[6] * g0[2], R[1] * g0[0] + R[4] * g0[1] + R[7] * g0[2], R[2] * g0[0] + R[5] * g0[1] + R[8] * g0[2]];
    const eg = eddy?.g[v] ?? [0, 0, 0], ee = eddy?.e ?? [0, 0, 0], ed0 = eddy?.d0[v] ?? 0, ge = (eg[0] * ee[0] + eg[1] * ee[1] + eg[2] * ee[2]) / 100;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const pa = [0, 1, 2].map((r) => M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3]);
      // The signal from scanner point p appears at p + (d0 + g·p/100) e: solved for p, and its brightness spread by the stretch.
      const dd = (ed0 + (eg[0] * pa[0] + eg[1] * pa[1] + eg[2] * pa[2]) / 100) / (1 + ge), p = [pa[0] - dd * ee[0], pa[1] - dd * ee[1], pa[2] - dd * ee[2]];
      const d = [p[0] - m.c[0] - m.t[0], p[1] - m.c[1] - m.t[1], p[2] - m.c[2] - m.t[2]];
      const x = [R[0] * d[0] + R[3] * d[1] + R[6] * d[2] + m.c[0], R[1] * d[0] + R[4] * d[1] + R[7] * d[2] + m.c[1], R[2] * d[0] + R[5] * d[1] + R[8] * d[2] + m.c[2]];
      const { s0, D } = tissue(x[0], x[1], x[2]);
      const q = D[0] * g[0] * g[0] + D[1] * g[1] * g[1] + D[2] * g[2] * g[2] + 2 * (D[3] * g[0] * g[1] + D[4] * g[0] * g[2] + D[5] * g[1] * g[2]);
      data[(k * ny + j) * nx + i] = Math.max(0, s0 * Math.exp(-bValues[v] * q) / (1 + ge) + noise * gauss());
    }
    return { dims: DIMS, ijkToRAS: M, data, dtype: "<f4" };
  });
  return { volumes, bValues, gradients, ijkToRAS: M, source: "synthetic", convention: 1 } as unknown as DiffusionSeries;
}

/** ds001226's layout: 102 images, b = 0 at 0, 1, 26, 51, 76 and 101, three shells interleaved (48 at b 2800, 32 at
 *  1200, 16 at 700), each shell's directions spread evenly (golden spiral) over the whole sphere. */
export function protocol() {
  const bValues: number[] = [], gradients: [number, number, number][] = [], B0 = new Set([0, 1, 26, 51, 76, 101]);
  const cycle = [2800, 1200, 2800, 700, 2800, 1200], count = new Map([[2800, 48], [1200, 32], [700, 16]]), used = new Map<number, number>();
  let k = 0;
  for (let v = 0; v < 102; v++) {
    if (B0.has(v)) { bValues.push(0); gradients.push([0, 0, 0]); continue; }
    const b = cycle[k++ % cycle.length], n = count.get(b)!, i = (used.get(b) ?? 0) + 0.5; used.set(b, i + 0.5);
    // Every other direction turned to its opposite: the same diffusion measurement, but the directions then cover the whole
    // sphere, as ds001226's do -- which is what lets eddy currents (opposite for opposite directions) be told apart from
    // the signal (the same for both). On a half sphere they cannot be.
    const z = 1 - i / n, r = Math.sqrt(1 - z * z), ph = Math.PI * (3 - Math.sqrt(5)) * i + b / 1000, sg = Math.floor(i) % 2 ? -1 : 1;
    bValues.push(b); gradients.push([sg * r * Math.cos(ph), sg * r * Math.sin(ph), sg * z]);
  }
  return { bValues, gradients };
}

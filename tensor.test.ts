// The tensor fit (extensions/diffusion/tensor.ts): a synthetic tensor recovered exactly, and on real data (OpenNeuro
// ds001226 PAT16, when on disk) the physics check -- in the corpus callosum's middle the main direction runs left-right.
//   deno test -A --no-check extensions/diffusion/tensor.test.ts
import { assert, assertAlmostEquals } from "jsr:@std/assert@1";
import { colorFA, fitTensors, fractionalAnisotropy, symEigenvalues, symEigenvector } from "./tensor.ts";
import { DWI_CONVENTION, type DiffusionSeries, fromFsl } from "./dwi.ts";
import { parseNiftiVolumes } from "albula";
import { ABSENT, testData } from "albula/testing";

Deno.test("eigen: closed form against a rotated diagonal tensor", () => {
  // R = rotation by 30° about z then 20° about x; D = R diag(1.7, 0.4, 0.2) Rᵀ (×1e-3).
  const c1 = Math.cos(Math.PI / 6), s1 = Math.sin(Math.PI / 6), c2 = Math.cos(Math.PI / 9), s2 = Math.sin(Math.PI / 9);
  const Rz = [[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]], Rx = [[1, 0, 0], [0, c2, -s2], [0, s2, c2]];
  const mm = (A: number[][], B: number[][]) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
  const R = mm(Rx, Rz), L = [1.7e-3, 0.4e-3, 0.2e-3];
  const Dm = [0, 1, 2].map((i) => [0, 1, 2].map((j) => R[i][0] * L[0] * R[j][0] + R[i][1] * L[1] * R[j][1] + R[i][2] * L[2] * R[j][2]));
  const ev = symEigenvalues(Dm[0][0], Dm[0][1], Dm[0][2], Dm[1][1], Dm[1][2], Dm[2][2]);
  ev.forEach((l, i) => assertAlmostEquals(l, L[i], 1e-12));
  const v = symEigenvector(Dm[0][0], Dm[0][1], Dm[0][2], Dm[1][1], Dm[1][2], Dm[2][2], ev[0]);
  const dot = Math.abs(v[0] * R[0][0] + v[1] * R[1][0] + v[2] * R[2][0]);
  assertAlmostEquals(dot, 1, 1e-9);
  assertAlmostEquals(fractionalAnisotropy(1, 1, 1), 0, 1e-12);
  assertAlmostEquals(fractionalAnisotropy(1, 0, 0), 1, 1e-12);
});

Deno.test("fit: noise-free signals give back the tensor, S0, FA and direction; b above maxB is left out", () => {
  // One voxel, 1 b0 + 12 directions at b=1000 + 6 at b=3000 (which the default fit must ignore: they carry a kurtosis
  // term here that the tensor model does not have, so using them would bend the result).
  const dirs: number[][] = [];
  for (let i = 0; i < 18; i++) { const z = 1 - (2 * (i + 0.5)) / 18, r = Math.sqrt(1 - z * z), t = i * 2.39996; dirs.push([r * Math.cos(t), r * Math.sin(t), z]); }
  const D = [[1.2e-3, 0.3e-3, 0.1e-3], [0.3e-3, 0.6e-3, -0.05e-3], [0.1e-3, -0.05e-3, 0.4e-3]];
  const q = (g: number[]) => [0, 1, 2].reduce((s, i) => s + [0, 1, 2].reduce((t, j) => t + g[i] * D[i][j] * g[j], 0), 0);
  const bs = [0, ...Array(12).fill(1000), ...Array(6).fill(3000)];
  const gs = [[0, 0, 0], ...dirs.slice(0, 12), ...dirs.slice(12)];
  const S0 = 900;
  const vols = bs.map((b, i) => {
    const adc = b ? q(gs[i]) : 0;
    const s = S0 * Math.exp(-b * adc + (b > 2000 ? (b * adc) ** 2 / 6 : 0));
    return { dims: [1, 1, 1] as [number, number, number], ijkToRAS: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], data: new Float32Array([s]), dtype: "<f4" };
  });
  const dwi: DiffusionSeries = { volumes: vols, bValues: bs, gradients: gs as [number, number, number][], ijkToRAS: vols[0].ijkToRAS, source: "synthetic", convention: DWI_CONVENTION };
  const fit = fitTensors(dwi, { mask: new Uint8Array([1]) });
  const want = [D[0][0], D[0][1], D[0][2], D[1][1], D[1][2], D[2][2]];
  want.forEach((w, i) => assertAlmostEquals(fit.D[i], w, 1e-8));
  assertAlmostEquals(fit.S0[0], S0, 1e-2);
  assert(fit.used.length === 13, `used ${fit.used.length} volumes`);
  const ev = symEigenvalues(D[0][0], D[0][1], D[0][2], D[1][1], D[1][2], D[2][2]);
  assertAlmostEquals(fit.fa[0], fractionalAnisotropy(...ev), 1e-5);
  const rgb = colorFA(fit);
  assert(rgb[0] > rgb[1] && rgb[0] > rgb[2], "x is the main direction of this tensor");
});

// THE PHYSICS CHECK on a real scan, in two places where anatomy fixes the direction. At the midline, at and above the
// brain's center (the corpus callosum), fibers cross between the hemispheres: left-right (x in patient RAS). At the
// midline well below the center (the brainstem), the long tracts run up-down: z. A sign or an axis lost anywhere between
// the file and the tensor -- the classic silent error of the field -- breaks one or both. Positions are taken relative
// to the brain mask's own center, not to the scanner's coordinates. (Seen on PAT16, 2026-09-28: 168 of 209 left-right
// at the callosum's level, 36 of 39 up-down in the brainstem; the rest at the top are the cingulum and fornix, which run
// front-back.)
const D = testData("openneuro-ds001226", "sub-PAT16/ses-preop/dwi") ?? ABSENT;
const HAVE = (() => { try { Deno.statSync(`${D}sub-PAT16_ses-preop_acq-AP_dwi.nii.gz`); return true; } catch { return false; } })();
Deno.test({
  name: "PAT16: high-FA voxels at the midline run left-right in the corpus callosum and up-down in the brainstem",
  ignore: !HAVE,
  fn: async () => {
    const I = `${D}sub-PAT16_ses-preop_acq-AP_dwi`;
    const dwi = fromFsl(await parseNiftiVolumes(Deno.readFileSync(`${I}.nii.gz`)), Deno.readTextFileSync(`${I}.bval`), Deno.readTextFileSync(`${I}.bvec`));
    const t0 = performance.now();
    const fit = fitTensors(dwi);
    const secs = (performance.now() - t0) / 1000;
    const [nx, ny, nz] = fit.dims, M = fit.ijkToRAS;
    const at = (i: number, j: number, k: number) => [0, 1, 2].map((r) => M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3]);
    const c = [0, 0, 0];
    let cnt = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      if (!fit.mask[(k * ny + j) * nx + i]) continue;
      const p = at(i, j, k); c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; cnt++;
    }
    c.forEach((_, q) => (c[q] /= cnt));
    const count = { upper: [0, 0, 0], lower: [0, 0, 0] };
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = (k * ny + j) * nx + i;
      if (fit.fa[v] < 0.6) continue;
      const p = at(i, j, k);
      if (Math.abs(p[0] - c[0]) > 3) continue;
      const main = [0, 1, 2].reduce((b, q) => (Math.abs(fit.v1[3 * v + q]) > Math.abs(fit.v1[3 * v + b]) ? q : b), 0);
      const dz = p[2] - c[2];
      if (dz >= -5 && dz <= 35) count.upper[main]++;
      else if (dz < -35) count.lower[main]++;
    }
    const sum = (a: number[]) => a[0] + a[1] + a[2];
    console.log(`PAT16 tensor fit: ${secs.toFixed(1)} s, ${cnt} voxels in the mask; midline FA ≥ 0.6 -- callosum level LR/AP/SI ${count.upper.join("/")}, brainstem ${count.lower.join("/")}`);
    assert(sum(count.upper) > 30 && count.upper[0] / sum(count.upper) > 0.75, `callosum level: ${count.upper.join("/")} (LR/AP/SI)`);
    assert(sum(count.lower) > 5 && count.lower[2] / sum(count.lower) > 0.75, `brainstem: ${count.lower.join("/")} (LR/AP/SI)`);
  },
});

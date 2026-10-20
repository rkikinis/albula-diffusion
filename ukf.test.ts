// UKF tractography (ukf.ts): the linear algebra, a synthetic crossing the two-tensor filter must go straight through
// (where one tensor per voxel is confused), free water recovered, and on OpenNeuro ds001226 PAT16 (when on disk)
// callosal fibers that join the hemispheres.
//   deno test -A --no-check extensions/diffusion/ukf.test.ts
import { assert, assertAlmostEquals, assertEquals as assertEqualsL } from "jsr:@std/assert@1";
import { cholesky, prepareUkfData, spdInverse, trackUkf, ukfLengths, ukfSigma, type UkfData } from "./ukf.ts";
import { fromFsl, DWI_CONVENTION, type DiffusionSeries } from "./dwi.ts";
import { brainMask, fitTensors } from "./tensor.ts";
import { parseNiftiVolumes } from "albula";
import { ABSENT, testData } from "albula/testing";

Deno.test("SPD inverse and Cholesky", () => {
  const A = new Float64Array([4, 2, 0.6, 2, 5, 1, 0.6, 1, 3]);
  const inv = spdInverse(A, 3)!;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let s = 0; for (let k = 0; k < 3; k++) s += A[i * 3 + k] * inv[k * 3 + j];
    assertAlmostEquals(s, i === j ? 1 : 0, 1e-12);
  }
  assert(cholesky(new Float64Array([1, 2, 2, 1]), 2) === null, "not positive definite");
});

/** A synthetic scan: two bundles crossing at `angleDeg` in the x-y plane, free-water fraction 1 − w everywhere, on a
 *  grid of `voxel` mm. */
function crossing(angleDeg: number, w: number, voxel = 1): { dwi: DiffusionSeries; mask: Uint8Array } {
  const dims: [number, number, number] = [40, 40, 6], [nx, ny, nz] = dims, n = nx * ny * nz;
  const dirs: [number, number, number][] = [[0, 0, 0]];
  for (let i = 0; i < 60; i++) { const z = 1 - (2 * (i + 0.5)) / 60, r = Math.sqrt(1 - z * z), t = i * 2.39996; dirs.push([r * Math.cos(t), r * Math.sin(t), z]); }
  const bvals = [0, ...Array(30).fill(1000), ...Array(30).fill(2000)];
  const a = (angleDeg * Math.PI) / 180, e1 = [1, 0, 0], e2 = [Math.cos(a), Math.sin(a), 0];
  const l1 = 1.7e-3, l2 = 0.3e-3;
  const vols = bvals.map((b, q) => {
    const u = dirs[q], data = new Float32Array(n);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      // Bundle 1: horizontal band |y − 20| < 4; bundle 2: a band along e2 through (20, 20) of half-width 4.
      const inA = Math.abs(j - 20) < 4, inB = Math.abs(-(i - 20) * e2[1] + (j - 20) * e2[0]) < 4;
      const t = (e: number[]) => Math.exp(-b * (l2 + (l1 - l2) * (u[0] * e[0] + u[1] * e[1] + u[2] * e[2]) ** 2));
      let s: number;
      if (inA && inB) s = 0.5 * t(e1) + 0.5 * t(e2);
      else if (inA) s = t(e1);
      else if (inB) s = t(e2);
      else s = Math.exp(-b * 0.8e-3);                          // isotropic tissue
      data[(k * ny + j) * nx + i] = 1000 * (w * s + (1 - w) * Math.exp(-b * 0.003));
    }
    return { dims, ijkToRAS: [voxel, 0, 0, 0, 0, voxel, 0, 0, 0, 0, voxel, 0, 0, 0, 0, 1], data, dtype: "<f4" };
  });
  const dwi: DiffusionSeries = { volumes: vols, bValues: bvals, gradients: dirs, ijkToRAS: vols[0].ijkToRAS, source: "synthetic", convention: DWI_CONVENTION };
  return { dwi, mask: new Uint8Array(n).fill(1) };
}

Deno.test("synthetic 60° crossing: the fiber seeded on bundle 1 goes straight through; free water is recovered", () => {
  const { dwi, mask } = crossing(60, 0.8);
  const data: UkfData = prepareUkfData(dwi, mask);
  const r = trackUkf(data, [[6, 20, 3]], { stoppingFA: 0.1, sigmaSignal: 1 });
  assert(r.fibers.length === 1, `${r.fibers.length} fibers`);
  const p = r.fibers[0].points, n = p.length / 3;
  let worst = 0, xmin = Infinity, xmax = -Infinity;
  for (let q = 0; q < n; q++) { worst = Math.max(worst, Math.abs(p[3 * q + 1] - 20)); xmin = Math.min(xmin, p[3 * q]); xmax = Math.max(xmax, p[3 * q]); }
  // The filter starts every fiber at w = 1 (as the original does) and moves it gradually (Qw is small): along this fiber
  // it goes 0.95 -> 0.81 (seen 2026-09-29). The estimate is read where it has settled: the last third of the points.
  const fwv = r.fibers[0].freeWater;
  const w = fwv.slice(Math.floor((2 * n) / 3)).reduce((s2, v) => s2 + v, 0) / (n - Math.floor((2 * n) / 3));
  console.log(`crossing: ${n} points, x ${xmin.toFixed(1)}..${xmax.toFixed(1)}, strays at most ${worst.toFixed(2)} voxel from its bundle; tissue fraction over the last third ${w.toFixed(3)} (true 0.8); ${r.steps} steps in ${r.ms.toFixed(0)} ms; ${r.projections} projections`);
  assert(xmin < 16 && xmax > 26, "passes through the crossing at x 16..24");
  assert(worst < 3, `strays ${worst} voxels (the bundle's half-width is 4)`);
  assert(Math.abs(w - 0.8) < 0.05, `settled tissue fraction ${w}`);
});

// THE STEP IN VOXELS OF THE SCAN (ukfLengths; Ron, 2026-10-20: "did you forget about scales? from astrophysics to
// electron microscope"): the original's 0.3 mm, 0.9 mm and 250 mm, exactly, on every clinical grid (so the port's
// agreement with UKFTractography is untouched), and the same number of steps and points at 1 nm and 1 pc voxels.
Deno.test("the step and the smoothing: the original's on clinical grids exactly; at 1 nm and 1 pc in voxels, 834 steps at most", () => {
  for (const v of [1, 2, 2.5]) {
    assertEqualsL(ukfLengths([v, v, v], {}), { step: 0.3, perRecord: 3, maxSteps: 834 }, `${v} mm`);
    assertEqualsL(ukfLengths([v, v, v], { recordLength: 1.8 }), { step: 0.3, perRecord: 6, maxSteps: 834 }, `${v} mm, rule 2's record`);
  }
  const nm = ukfLengths([1e-6, 1e-6, 1e-6], {}), pc = ukfLengths([3.086e19, 3.086e19, 3.086e19], { recordLength: 1.8 });
  assertAlmostEquals(nm.step / 1e-6, 0.5, 1e-12); assertEqualsL([nm.perRecord, nm.maxSteps], [3, 834], "1 nm");
  assertAlmostEquals(pc.step / 3.086e19, 0.05, 1e-12); assertEqualsL([pc.perRecord, pc.maxSteps], [6, 834], "1 pc");
  assertEqualsL(ukfLengths([2, 2, 2], { stepLength: 0.5 }).step, 0.5, "a step given is used as given");
  // The signal's smoothing: σ the voxel in mm on clinical grids, as the original. A neighbor n voxels away weighs
  // exp(-n² v² / σ); outside 0.6 to 6 mm, v² / σ is held at 0.6 or 6, as on those grids.
  for (const v of [1, 2, 2.5]) assertEqualsL(ukfSigma([v, v, 2 * v], {}), v);
  assertAlmostEquals((1e-6) ** 2 / ukfSigma([1e-6, 1e-6, 1e-6], {}), 0.6, 1e-12);
  assertAlmostEquals(3.086e19 ** 2 / ukfSigma([3.086e19, 3.086e19, 3.086e19], {}), 6, 1e-9);
  assertEqualsL(ukfSigma([2, 2, 2], { sigmaSignal: 1 }), 1);
});

Deno.test("the synthetic crossing at 1 nm and 1 pc voxels: the fiber still goes straight through, in a bounded run", () => {
  for (const v of [1e-6, 3.086e19]) {
    const { dwi, mask } = crossing(60, 0.8, v);
    const r = trackUkf(prepareUkfData(dwi, mask), [[6, 20, 3]], { stoppingFA: 0.1 });
    assert(r.fibers.length === 1, `${v} mm: ${r.fibers.length} fibers`);
    const p = r.fibers[0].points, n = p.length / 3;
    let worst = 0, xmin = Infinity, xmax = -Infinity;
    for (let q = 0; q < n; q++) { worst = Math.max(worst, Math.abs(p[3 * q + 1] / v - 20)); xmin = Math.min(xmin, p[3 * q] / v); xmax = Math.max(xmax, p[3 * q] / v); }
    assert(xmin < 16 && xmax > 26, `${v} mm: x ${xmin.toFixed(1)}..${xmax.toFixed(1)} voxels`);
    assert(worst < 3, `${v} mm: strays ${worst.toFixed(2)} voxels`);
    assert(r.steps <= 2 * 834, `${v} mm: ${r.steps} steps`);
  }
});

const DIR = testData("openneuro-ds001226", "sub-PAT16/ses-preop/dwi") ?? ABSENT;
const HAVE = (() => { try { Deno.statSync(`${DIR}sub-PAT16_ses-preop_acq-AP_dwi.nii.gz`); return true; } catch { return false; } })();
Deno.test({
  name: "PAT16: two-tensor free-water fibers from the corpus callosum join the hemispheres",
  ignore: !HAVE,
  fn: async () => {
    const I = `${DIR}sub-PAT16_ses-preop_acq-AP_dwi`;
    const dwi = fromFsl(await parseNiftiVolumes(Deno.readFileSync(`${I}.nii.gz`)), Deno.readTextFileSync(`${I}.bval`), Deno.readTextFileSync(`${I}.bvec`));
    const b0s = dwi.bValues.map((b, i) => (b < 50 ? i : -1)).filter((i) => i >= 0);
    const { mask } = brainMask(dwi, b0s);
    const fit = fitTensors(dwi, { mask }), [nx, ny, nz] = fit.dims, M = fit.ijkToRAS;
    // Seeds: voxels within a voxel of the midline, above the brain's center, FA 0.5-0.95, main direction left-right.
    let cx = 0, cz = 0, c = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (mask[(k * ny + j) * nx + i]) { cx += M[0] * i + M[1] * j + M[2] * k + M[3]; cz += M[8] * i + M[9] * j + M[10] * k + M[11]; c++; }
    cx /= c; cz /= c;
    const seeds: number[][] = [];
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = (k * ny + j) * nx + i, x = M[0] * i + M[1] * j + M[2] * k + M[3], z = M[8] * i + M[9] * j + M[10] * k + M[11];
      if (Math.abs(x - cx) < 2.5 && z > cz && fit.fa[v] > 0.5 && fit.fa[v] < 0.95 && Math.abs(fit.v1[3 * v]) > 0.9) seeds.push([i, j, k]);
    }
    const r = trackUkf(prepareUkfData(dwi, mask), seeds.slice(0, 40));
    const join = r.fibers.filter((f) => { const p = f.points, l = p.length / 3 - 1, a = p[0] - cx, b = p[3 * l] - cx; return a * b < 0 && Math.min(Math.abs(a), Math.abs(b)) > 10; }).length;
    console.log(`PAT16 UKF: ${r.fibers.length} fibers from ${Math.min(40, seeds.length)} seeds, ${join} end in both hemispheres (> 10 mm out); ${r.steps} steps, ${(r.ms / 1000).toFixed(1)} s (${(r.ms / r.steps).toFixed(3)} ms a step)`);
    assert(r.fibers.length >= 20 && join / r.fibers.length > 0.6, `${join} of ${r.fibers.length}`);
  },
});

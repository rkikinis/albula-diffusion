// Tracking from seeds (logic/diffusion/tracking.ts): a synthetic curved bundle followed to its ends, and on real data
// (OpenNeuro ds001226 PAT16, when on disk) the anatomy: seeds in the midline corpus callosum give streamlines that
// join the two hemispheres; seeds in the brainstem give streamlines that climb toward the cortex.
//   deno test -A --no-check extensions/diffusion/tracking.test.ts
import { assert } from "jsr:@std/assert@1";
import { seedsInSphere, trackFromSeeds } from "./tracking.ts";
import { fitTensors, type TensorFit } from "./tensor.ts";
import { fromFsl } from "./dwi.ts";
import { parseNiftiVolumes } from "albula";
import { ABSENT, testData } from "albula/testing";

Deno.test("a quarter-circle bundle is followed from a seed in its middle to both of its ends", () => {
  // A 40x40x3 grid, 1 mm voxels. Fibers run along circles around (0,0) with radius 15..25 mm, FA ~0.8; elsewhere isotropic.
  const n = [40, 40, 3] as [number, number, number], N = n[0] * n[1] * n[2];
  const D = new Float32Array(6 * N), fa = new Float32Array(N), mask = new Uint8Array(N).fill(1);
  for (let k = 0; k < n[2]; k++) for (let j = 0; j < n[1]; j++) for (let i = 0; i < n[0]; i++) {
    const v = (k * n[1] + j) * n[0] + i, r = Math.hypot(i, j);
    let t = [1, 0, 0], l1 = 0.7e-3, l2 = 0.7e-3;
    if (r >= 15 && r <= 25) { t = [-j / r, i / r, 0]; l1 = 1.7e-3; l2 = 0.3e-3; }
    const Dm = (a: number, b: number) => l2 * (a === b ? 1 : 0) + (l1 - l2) * t[a] * t[b];
    [Dm(0, 0), Dm(0, 1), Dm(0, 2), Dm(1, 1), Dm(1, 2), Dm(2, 2)].forEach((x, q) => (D[6 * v + q] = x));
  }
  const fit = { dims: n, ijkToRAS: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], D, fa, mask } as unknown as TensorFit;
  const s = trackFromSeeds(fit, [[20 * Math.SQRT1_2, 20 * Math.SQRT1_2, 1]], { stepVoxels: 0.5 });
  assert(s.length === 1, "one streamline");
  const p = s[0].points, last = p.length / 3 - 1;
  // Every point stays on the circle of radius 20 (the seed's) to within half a voxel, and the ends reach the axes.
  let worst = 0;
  for (let q = 0; q <= last; q++) worst = Math.max(worst, Math.abs(Math.hypot(p[3 * q], p[3 * q + 1]) - 20));
  assert(worst < 0.5, `strays ${worst.toFixed(2)} mm from its circle`);
  const ends = [[p[0], p[1]], [p[3 * last], p[3 * last + 1]]];
  assert(ends.some((e) => e[0] < 1.5) && ends.some((e) => e[1] < 1.5), `ends at ${JSON.stringify(ends.map((e) => e.map((x) => x.toFixed(1))))}`);
});

const DIR = testData("openneuro-ds001226", "sub-PAT16/ses-preop/dwi") ?? ABSENT;
const HAVE = (() => { try { Deno.statSync(`${DIR}sub-PAT16_ses-preop_acq-AP_dwi.nii.gz`); return true; } catch { return false; } })();
Deno.test({
  name: "PAT16: callosal seeds join the hemispheres; brainstem seeds climb",
  ignore: !HAVE,
  fn: async () => {
    const I = `${DIR}sub-PAT16_ses-preop_acq-AP_dwi`;
    const fit = fitTensors(fromFsl(await parseNiftiVolumes(Deno.readFileSync(`${I}.nii.gz`)), Deno.readTextFileSync(`${I}.bval`), Deno.readTextFileSync(`${I}.bvec`)));
    const [nx, ny, nz] = fit.dims, M = fit.ijkToRAS;
    const at = (i: number, j: number, k: number) => [0, 1, 2].map((r) => M[4 * r] * i + M[4 * r + 1] * j + M[4 * r + 2] * k + M[4 * r + 3]);
    // The brain's center; then each seed region at a real voxel of its bundle: the midline voxels at the callosum's
    // level whose main direction is left-right (FA 0.5-0.95: above 0.95 the fit is noise), and those 35+ mm below the
    // center running up-down (the brainstem) -- in each set, the voxel nearest the set's average. (The average itself
    // is no seed: the callosum is an arch, and the middle of an arch is the ventricle.)
    const c = [0, 0, 0]; let cnt = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (fit.mask[(k * ny + j) * nx + i]) { const p = at(i, j, k); c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; cnt++; }
    c.forEach((_, q) => (c[q] /= cnt));
    const ccSet: number[][] = [], bsSet: number[][] = [];
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const v = (k * ny + j) * nx + i, p = at(i, j, k);
      if (Math.abs(p[0] - c[0]) > 3 || fit.fa[v] < 0.5 || fit.fa[v] > 0.95) continue;
      const dz = p[2] - c[2];
      if (dz >= -5 && dz <= 35 && Math.abs(fit.v1[3 * v]) > 0.9) ccSet.push(p);
      if (dz < -35 && Math.abs(fit.v1[3 * v + 2]) > 0.9) bsSet.push(p);
    }
    const nearestToMean = (set: number[][]) => {
      const m = [0, 1, 2].map((q) => set.reduce((s, p) => s + p[q], 0) / set.length);
      return set.reduce((b, p) => (Math.hypot(p[0] - m[0], p[1] - m[1], p[2] - m[2]) < Math.hypot(b[0] - m[0], b[1] - m[1], b[2] - m[2]) ? p : b));
    };
    assert(ccSet.length > 10 && bsSet.length > 5, `seed regions: ${ccSet.length} callosal, ${bsSet.length} brainstem voxels`);
    const cc = nearestToMean(ccSet), bs = nearestToMean(bsSet);
    const t0 = performance.now();
    const callosal = trackFromSeeds(fit, seedsInSphere(cc, 2));
    const climbing = trackFromSeeds(fit, seedsInSphere(bs, 2));
    const ms = performance.now() - t0;
    const length = (p: Float32Array) => { let L = 0; for (let q = 3; q < p.length; q += 3) L += Math.hypot(p[q] - p[q - 3], p[q + 1] - p[q - 2], p[q + 2] - p[q - 1]); return L; };
    // Callosal: the two ends in opposite hemispheres (each more than 3 mm from the midline) and longer than 60 mm --
    // the body's fibers cross, run out 15-20 mm, then turn up toward the cortex (seen on PAT16, 2026-09-28).
    const crossing = callosal.filter((s) => {
      const p = s.points, l = p.length / 3 - 1, a = p[0] - c[0], b = p[3 * l] - c[0];
      return a * b < 0 && Math.min(Math.abs(a), Math.abs(b)) > 3 && length(p) > 60;
    }).length;
    // Brainstem: most climb out of the pons (25 mm or more above the seed), and a good share reach the top of the brain
    // (80 mm or more). Seen on PAT16, 2026-09-28: the climb's quartiles 27 / 30 / 104 mm -- a seed AT the midline runs
    // into the midbrain's crossing fibers (the decussations) about 30 mm up, where most stop; the rest go on to the
    // cortex. The lower ends leave the scan's field of view.
    const climb = climbing.map((s) => { let top = -Infinity; for (let q = 2; q < s.points.length; q += 3) top = Math.max(top, s.points[q]); return top - bs[2]; });
    const out = climb.filter((d) => d >= 25).length, top = climb.filter((d) => d >= 80).length;
    console.log(`PAT16 tracking: ${callosal.length} callosal streamlines, ${crossing} join the hemispheres over 60 mm or more; ${climbing.length} brainstem streamlines, ${out} climb 25 mm or more, ${top} 80 mm or more; ${ms.toFixed(0)} ms`);
    assert(callosal.length >= 10 && crossing / callosal.length > 0.5, `callosal: ${crossing} of ${callosal.length}`);
    assert(climbing.length >= 10 && out / climbing.length > 0.5 && top / climbing.length > 0.2, `brainstem: ${out} and ${top} of ${climbing.length}`);
  },
});

// CSD AGAINST DIPY, ON THE PROBLEM'S OWN TERMS (csd.ts; the "csd-reference" test data, made by DIPY's
// MultiShellDeconvModel with cvxpy on three slices of PAT16). The fiber distributions may differ where several shapes
// fit the signal equally well, so the check is what the method promises: our misfit to the signal is no worse than
// DIPY's, and our distribution is not negative on the constraint directions. Measured on all 6,770 voxels
// (2026-10-01): misfit equal within 0.1% in 6,742, ours lower in 28, DIPY's lower in none; distributions agree to 1%
// (median, voxels with fibers). A sample of every 13th voxel here, to stay quick.
import { assert } from "jsr:@std/assert@1";
import { parseNiftiVolumes } from "albula";
import { ABSENT, testData } from "albula/testing";
import { csdFit, csdModel, shBasis, shCount, type Kernel } from "./csd.ts";

const REF = (testData("csd-reference") ?? ABSENT) + "PAT16/";
const DWI = testData("openneuro-ds001226", "sub-PAT16/ses-preop/dwi") ?? ABSENT;
const have = (p: string) => { try { Deno.statSync(p); return true; } catch { return false; } };

Deno.test("the spherical-harmonic basis is orthonormal (a quadrature check on many directions)", () => {
  const n = 4000, dirs = new Float64Array(3 * n);
  for (let i = 0; i < n; i++) {                                 // Fibonacci sphere
    const z = 1 - 2 * (i + 0.5) / n, r = Math.sqrt(1 - z * z), ph = i * Math.PI * (3 - Math.sqrt(5));
    dirs.set([r * Math.cos(ph), r * Math.sin(ph), z], 3 * i);
  }
  const B = shBasis(8, dirs), nc = shCount(8);
  let worst = 0;
  for (let a = 0; a < nc; a++) for (let b = 0; b <= a; b++) {
    let s = 0; for (let i = 0; i < n; i++) s += B[i * nc + a] * B[i * nc + b];
    s *= 4 * Math.PI / n;
    worst = Math.max(worst, Math.abs(s - (a === b ? 1 : 0)));
  }
  assert(worst < 2e-3, `basis not orthonormal: ${worst}`);
});

Deno.test({ name: "CSD against DIPY on PAT16: no worse a fit, the constraint kept", ignore: !have(REF + "objective.f32") || !have(DWI), fn: async () => {
  const f32 = (n: string) => new Float32Array(Deno.readFileSync(REF + n).buffer);
  const slab = JSON.parse(Deno.readTextFileSync(REF + "slab.json")), kj = JSON.parse(Deno.readTextFileSync(REF + "kernel.json"));
  const P = DWI + "sub-PAT16_ses-preop_acq-AP_dwi";
  const vols = await parseNiftiVolumes(Deno.readFileSync(P + ".nii.gz"));
  const bvals = Deno.readTextFileSync(P + ".bval").trim().split(/\s+/).map(Number);
  const vec = Deno.readTextFileSync(P + ".bvec").trim().split("\n").map((l) => l.trim().split(/\s+/).map(Number));
  const g = new Float64Array(bvals.length * 3); for (let i = 0; i < bvals.length; i++) for (let c = 0; c < 3; c++) g[3 * i + c] = vec[c][i];
  const sphere = f32("sphere.f32"), objRef = f32("objective.f32"), mask = new Uint8Array(Deno.readFileSync(REF + "mask.u8"));
  const k: Kernel = { shells: kj.shells, response: kj.response, iso: 2, lmax: 8 };
  const model = csdModel(g, bvals, k, sphere), Bs = shBasis(8, sphere), ns = sphere.length / 3;
  const [nx, ny] = vols[0].dims;
  let n = 0, checked = 0, worse = 0, worstNeg = 0; const ratios: number[] = [], peaks: number[] = [];
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) for (let zz = 0; zz < slab.slices; zz++) {
    if (!mask[(zz * ny + y) * nx + x]) continue;
    const i = n++;
    if (i % 13) continue;
    const v = ((slab.first_slice + zz) * ny + y) * nx + x, s = vols.map((vol) => Number(vol.data[v]));
    const xf = csdFit(model, s);
    let o = 0; for (let r = 0; r < model.rows; r++) { let p = 0; for (let a = 0; a < model.nx; a++) p += model.X[r * model.nx + a] * xf[a]; o += (p - s[r]) ** 2; }
    const ratio = o / Math.max(objRef[i], 1e-12); ratios.push(ratio); if (ratio > 1.001) worse++;
    let scale = 0, mn = Infinity;
    for (let d = 0; d < ns; d++) { let a = 0; for (let c = 0; c < 45; c++) a += Bs[d * 45 + c] * xf[2 + c]; mn = Math.min(mn, a); scale = Math.max(scale, Math.abs(a)); }
    worstNeg = Math.min(worstNeg, mn); peaks.push(scale);
    checked++;
  }
  ratios.sort((a, b) => a - b);
  // Negative values measured against the TYPICAL peak (a fluid voxel's distribution is near zero, so its own peak is no scale).
  const typical = [...peaks].sort((a, b) => a - b)[peaks.length >> 1]; worstNeg /= typical;
  console.log(`  ${checked} voxels: misfit ours/DIPY median ${ratios[checked >> 1].toFixed(4)}, worst ${ratios[checked - 1].toFixed(4)}; ${worse} worse by more than 0.1%; lowest distribution value ${worstNeg.toExponential(1)} of the typical peak`);
  assert(worse <= checked * 0.01, `${worse} of ${checked} voxels fit worse than DIPY's`);
  assert(worstNeg > -1e-3, `the distribution goes negative: ${worstNeg} of the typical peak`);
} });

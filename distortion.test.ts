// Distortion correction from a reversed phase-encoding pair (distortion.ts): a phantom distorted by a known field is
// put back, and on OpenNeuro ds001226 PAT16 (when on disk) the corrected b=0 image agrees better with the undistorted
// T1 of the same session, and the two corrected directions agree with each other.
//   deno test -A --no-check extensions/diffusion/distortion.test.ts
import { assert } from "jsr:@std/assert@1";
import { applyField, estimateField, fieldAtCenters } from "./distortion.ts";
import { parseNiftiVolumes } from "albula";
import { ABSENT, testData } from "albula/testing";

/** Distort `img` along axis 1 by field b (voxels, at cell centers): I±(z) with z = x ± b(x), intensity I/(1 ± ∂b). */
function distort(img: Float32Array, b: Float32Array, dims: [number, number, number], sign: 1 | -1): Float32Array {
  const [nx, ny, nz] = dims, out = new Float32Array(img.length);
  for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) {
    // Forward map of this line, sampled finely; then read it back at the whole-voxel positions z.
    const S = 20, zs: number[] = [], vs: number[] = [];
    for (let s = 0; s <= (ny - 1) * S; s++) {
      const x = s / S, j0 = Math.min(Math.floor(x), ny - 2), f = x - j0;
      const at = (a: Float32Array, j: number) => a[(k * ny + j) * nx + i];
      const bx = at(b, j0) * (1 - f) + at(b, j0 + 1) * f, db = at(b, j0 + 1) - at(b, j0);
      zs.push(x + sign * bx); vs.push((at(img, j0) * (1 - f) + at(img, j0 + 1) * f) / (1 + sign * db));
    }
    for (let j = 0; j < ny; j++) {
      let q = 0; while (q < zs.length - 1 && zs[q + 1] < j) q++;
      const t = zs[q + 1] > zs[q] ? Math.min(1, Math.max(0, (j - zs[q]) / (zs[q + 1] - zs[q]))) : 0;
      out[(k * ny + j) * nx + i] = j < zs[0] || j > zs[zs.length - 1] ? 0 : vs[q] * (1 - t) + vs[q + 1] * t;
    }
  }
  return out;
}

Deno.test("a phantom distorted both ways by a known field is put back: field within 0.3 voxel, image much closer", () => {
  const dims: [number, number, number] = [24, 48, 12], [nx, ny, nz] = dims, n = nx * ny * nz;
  const img = new Float32Array(n), b = new Float32Array(n);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const v = (k * ny + j) * nx + i;
    // A "head": an ellipsoid with inner structure (stripes along the phase-encoding axis, so there is something to align).
    const e = ((i - 12) / 10) ** 2 + ((j - 24) / 20) ** 2 + ((k - 6) / 5) ** 2;
    img[v] = e < 1 ? 100 + 60 * Math.sin(j / 2.2) * Math.cos(i / 3) : 0;
    // The field: a smooth bump toward the front (as near the frontal sinus), up to 2.5 voxels.
    b[v] = 2.5 * Math.exp(-(((j - 34) / 8) ** 2 + ((i - 12) / 9) ** 2 + ((k - 6) / 6) ** 2));
  }
  const plus = distort(img, b, dims, 1), minus = distort(img, b, dims, -1);
  const fit = estimateField({ dims, plus, minus, axis: 1 });
  const est = fieldAtCenters(fit), cp = applyField(fit, plus, 1), cm = applyField(fit, minus, -1);
  let worst = 0, sumErr = 0, cnt = 0, before = 0, after = 0;
  for (let v = 0; v < n; v++) {
    if (img[v] <= 0) continue;
    const e = Math.abs(est[v] - b[v]); worst = Math.max(worst, e); sumErr += e; cnt++;
    before += (plus[v] - img[v]) ** 2; after += (0.5 * (cp[v] + cm[v]) - img[v]) ** 2;
  }
  console.log(`phantom: field error mean ${(sumErr / cnt).toFixed(3)}, worst ${worst.toFixed(2)} voxel; image error ${Math.sqrt(before / cnt).toFixed(1)} -> ${Math.sqrt(after / cnt).toFixed(1)}; ${fit.levels.map((l) => `${l.dims.join("x")}: ${l.iterations} it, r ${l.residual.toFixed(3)}`).join("; ")}; ${fit.ms.toFixed(0)} ms`);
  assert(sumErr / cnt < 0.3, `mean field error ${sumErr / cnt}`);
  assert(after < 0.25 * before, `image error ${Math.sqrt(before / cnt)} -> ${Math.sqrt(after / cnt)}`);
});

/** Mutual information of two images over voxels where both are non-zero (32 bins each). */
function mutualInformation(a: Float32Array, c: Float32Array): number {
  const B = 32, h = new Float64Array(B * B);
  const top = (x: Float32Array) => { const s = Array.from(x.filter((v) => v > 0)).sort((p, q) => p - q); return s[Math.floor(s.length * 0.995)] || 1; };
  const ta = top(a), tc = top(c);
  let n = 0;
  for (let v = 0; v < a.length; v++) {
    if (!(a[v] > 0 && c[v] > 0)) continue;
    const x = Math.min(B - 1, Math.floor((a[v] / ta) * B)), y = Math.min(B - 1, Math.floor((c[v] / tc) * B));
    h[x * B + y]++; n++;
  }
  const pa = new Float64Array(B), pc = new Float64Array(B);
  for (let x = 0; x < B; x++) for (let y = 0; y < B; y++) { pa[x] += h[x * B + y] / n; pc[y] += h[x * B + y] / n; }
  let mi = 0;
  for (let x = 0; x < B; x++) for (let y = 0; y < B; y++) { const p = h[x * B + y] / n; if (p > 0) mi += p * Math.log(p / (pa[x] * pc[y])); }
  return mi;
}

const D = testData("openneuro-ds001226", "sub-PAT16/ses-preop") ?? ABSENT;
const HAVE = (() => { try { Deno.statSync(`${D}dwi/sub-PAT16_ses-preop_acq-PA_dwi.nii.gz`); return true; } catch { return false; } })();
Deno.test({
  name: "PAT16: the corrected b=0 agrees better with the T1, and the two directions agree with each other",
  ignore: !HAVE,
  fn: async () => {
    const ap = await parseNiftiVolumes(Deno.readFileSync(`${D}dwi/sub-PAT16_ses-preop_acq-AP_dwi.nii.gz`));
    const pa = await parseNiftiVolumes(Deno.readFileSync(`${D}dwi/sub-PAT16_ses-preop_acq-PA_dwi.nii.gz`));
    const bval = Deno.readTextFileSync(`${D}dwi/sub-PAT16_ses-preop_acq-AP_dwi.bval`).trim().split(/\s+/).map(Number);
    const mean = (vols: typeof ap, idx: number[]) => { const o = new Float32Array(vols[0].data.length); for (const t of idx) for (let v = 0; v < o.length; v++) o[v] += vols[t].data[v] / idx.length; return o; };
    const apB0 = mean(ap, bval.map((b, i) => (b < 50 ? i : -1)).filter((i) => i >= 0)), paB0 = mean(pa, [0, 1]);
    const dims = ap[0].dims;
    // AP is phase-encoded along -j ("j-" in its sidecar), PA along +j: plus = PA, minus = AP, axis j.
    const fit = estimateField({ dims, plus: paB0, minus: apB0, axis: 1 });
    const cPA = applyField(fit, paB0, 1), cAP = applyField(fit, apB0, -1);
    const f = fieldAtCenters(fit);
    let fmax = 0; for (const v of f) fmax = Math.max(fmax, Math.abs(v));
    // Agreement of the two directions, before and after.
    let d0 = 0, d1 = 0, s = 0;
    for (let v = 0; v < apB0.length; v++) { d0 += (apB0[v] - paB0[v]) ** 2; d1 += (cAP[v] - cPA[v]) ** 2; s += apB0[v] ** 2; }
    // The T1 on the diffusion grid (same session, scanner coordinates; trilinear).
    const t1 = (await parseNiftiVolumes(Deno.readFileSync(`${D}anat/sub-PAT16_ses-preop_T1w.nii.gz`)))[0];
    const inv = (m: number[]) => { const a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], ff = m[6], g = m[8], h = m[9], k = m[10]; const A = e * k - ff * h, B = -(d * k - ff * g), C = d * h - e * g, det = a * A + b * B + c * C; const R = [A, -(b * k - c * h), b * ff - c * e, B, a * k - c * g, -(a * ff - c * d), C, -(a * h - b * g), a * e - b * d].map((x) => x / det); const t = [m[3], m[7], m[11]]; return [R[0], R[1], R[2], -(R[0] * t[0] + R[1] * t[1] + R[2] * t[2]), R[3], R[4], R[5], -(R[3] * t[0] + R[4] * t[1] + R[5] * t[2]), R[6], R[7], R[8], -(R[6] * t[0] + R[7] * t[1] + R[8] * t[2])]; };
    const M = ap[0].ijkToRAS, Ti = inv(t1.ijkToRAS), [tx, ty, tz] = t1.dims, [nx, ny, nz] = dims;
    const t1on = new Float32Array(apB0.length);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const x = M[0] * i + M[1] * j + M[2] * k + M[3], y = M[4] * i + M[5] * j + M[6] * k + M[7], z = M[8] * i + M[9] * j + M[10] * k + M[11];
      const u = Ti[0] * x + Ti[1] * y + Ti[2] * z + Ti[3], vv = Ti[4] * x + Ti[5] * y + Ti[6] * z + Ti[7], w = Ti[8] * x + Ti[9] * y + Ti[10] * z + Ti[11];
      const i0 = Math.floor(u), j0 = Math.floor(vv), k0 = Math.floor(w);
      if (i0 < 0 || j0 < 0 || k0 < 0 || i0 + 1 >= tx || j0 + 1 >= ty || k0 + 1 >= tz) continue;
      const fx = u - i0, fy = vv - j0, fz = w - k0;
      let s2 = 0;
      for (let c = 0; c < 8; c++) { const a = c & 1, b = (c >> 1) & 1, e = c >> 2; s2 += (a ? fx : 1 - fx) * (b ? fy : 1 - fy) * (e ? fz : 1 - fz) * t1.data[((k0 + e) * ty + j0 + b) * tx + i0 + a]; }
      t1on[(k * ny + j) * nx + i] = s2;
    }
    const miAP = mutualInformation(apB0, t1on), miPA = mutualInformation(paB0, t1on);
    const miCAP = mutualInformation(cAP, t1on), miC = mutualInformation(new Float32Array(cAP.map((v, i) => 0.5 * (v + cPA[i]))), t1on);
    console.log(`PAT16: field up to ${fmax.toFixed(2)} voxels (${(fmax * 2.5).toFixed(1)} mm; ${(fmax / 0.0266003).toFixed(0)} Hz at 26.6 ms readout); AP-PA difference ${Math.sqrt(d0 / s).toFixed(3)} -> ${Math.sqrt(d1 / s).toFixed(3)}; MI with T1: AP ${miAP.toFixed(3)}, PA ${miPA.toFixed(3)} -> corrected AP ${miCAP.toFixed(3)}, corrected mean ${miC.toFixed(3)}; ${fit.levels.map((l) => `${l.dims.join("x")}: ${l.iterations} it`).join("; ")}; ${(fit.ms / 1000).toFixed(1)} s`);
    assert(d1 < 0.6 * d0, "the two directions agree better after correction");
    assert(miCAP > miAP, "the corrected AP b=0 agrees better with the T1");
  },
});

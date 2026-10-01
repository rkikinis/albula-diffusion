// The diffusion reader's conventions (logic/diffusion/dwi.ts): synthetic cases for each rule, and a real one checked
// against Slicer's DWIConvert on OpenNeuro ds001226 sub-PAT16 when that data is on disk (skipped otherwise).
//   deno test -A --no-check logic/diffusion/dwi.test.ts
import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@1";
import { fromDicomVolumes, fromFsl, fromNrrdDwi } from "./dwi.ts";
import { parseNiftiVolumes, type Volume } from "albula";
import { ABSENT, testData } from "albula/testing";

const vol = (ijkToRAS: number[], meta?: Record<string, unknown>): Volume =>
  ({ dims: [2, 2, 2], ijkToRAS, data: new Float32Array(8), dtype: "<f4", meta });
const close = (a: number[], b: number[], eps = 1e-6) => a.forEach((x, i) => assertAlmostEquals(x, b[i], eps));

Deno.test("FSL bvec: along the image axes, x mirrored when the determinant is positive", () => {
  // RAS storage (det > 0): FSL's first axis is mirrored, so bvec +x means patient left.
  const ras = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  let s = fromFsl([vol(ras), vol(ras)], "0 1000", "0 1\n0 0\n0 0");
  close(s.gradients[1], [-1, 0, 0]);
  assertEquals(s.gradients[0], [0, 0, 0]);
  // LAS storage (det < 0): no mirror; the first image axis already points left, so +x is patient left too.
  const las = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  s = fromFsl([vol(las), vol(las)], "0 1000", "0 1\n0 0\n0 0");
  close(s.gradients[1], [-1, 0, 0]);
  // The second and third axes are never mirrored; spacing does not matter.
  const las2 = [-2.5, 0, 0, 0, 0, 2.5, 0, 0, 0, 0, 2.5, 0, 0, 0, 0, 1];
  s = fromFsl([vol(las2), vol(las2)], "0 1000", "0 0\n0 1\n0 0");
  close(s.gradients[1], [0, 1, 0]);
  // One row per volume is accepted too.
  s = fromFsl([vol(las), vol(las)], "0 1000", "0 0 0\n0 0 1");
  close(s.gradients[1], [0, 0, 1]);
});

Deno.test("NRRD DWI: list axis first or last, b from gradient length, measurement frame, LPS to RAS", async () => {
  const enc = new TextEncoder();
  const head = (kinds: string, sizes: string, mf: string) => `NRRD0005
type: float
dimension: 4
space: left-posterior-superior
sizes: ${sizes}
space directions: ${kinds.startsWith("list") ? "none (1,0,0) (0,1,0) (0,0,1)" : "(1,0,0) (0,1,0) (0,0,1) none"}
kinds: ${kinds}
endian: little
encoding: raw
space origin: (0,0,0)
measurement frame: ${mf}
modality:=DWMRI
DWMRI_b-value:=2000
DWMRI_gradient_0000:=0 0 0
DWMRI_gradient_0001:=0.70710678 0 0
DWMRI_gradient_0002:=0 1 0
`;
  // Two voxels in space (2x1x1), three gradients. Volume t holds values 10t, 10t+1.
  const dataFirst = new Float32Array([0, 10, 20, 1, 11, 21]);          // list first: voxel-major
  const dataLast = new Float32Array([0, 1, 10, 11, 20, 21]);           // list last: volume-major
  const a = await fromNrrdDwi(enc.encode(head("list space space space", "3 2 1 1", "(1,0,0) (0,1,0) (0,0,1)")), new Uint8Array(dataFirst.buffer));
  const b = await fromNrrdDwi(enc.encode(head("space space space list", "2 1 1 3", "(1,0,0) (0,1,0) (0,0,1)")), new Uint8Array(dataLast.buffer));
  for (const s of [a, b]) {
    assertEquals([...s.volumes[2].data], [20, 21]);
    close(s.bValues, [0, 1000, 2000], 1e-3);                            // |g|² = 0.5 of the longest
    close(s.gradients[1], [-1, 0, 0]);                                  // LPS +x is patient left: RAS -x
    close(s.gradients[2], [0, -1, 0]);                                  // LPS +y is posterior: RAS -y
  }
  // A measurement frame that swaps x and y (columns (0,1,0) (1,0,0)): the frame's x lands on the space's y.
  const c = await fromNrrdDwi(enc.encode(head("space space space list", "2 1 1 3", "(0,1,0) (1,0,0) (0,0,1)")), new Uint8Array(dataLast.buffer));
  close(c.gradients[1], [0, -1, 0]);
});

Deno.test("DICOM: the reader's LPS gradient becomes RAS", () => {
  const s = fromDicomVolumes([vol([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], { diffusion: { bValue: 0 } }),
    vol([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], { diffusion: { bValue: 1000, gradient: [0.6, 0.8, 0] } })]);
  assertEquals(s.gradients[0], [0, 0, 0]);
  close(s.gradients[1], [-0.6, -0.8, 0]);
});

// THE REAL CASE, against an independent implementation: Slicer's DWIConvert (FSLToNrrd) on the same NIfTI + bval/bvec.
// Same b-values, same gradient directions in patient space, and the same value at the same patient position in every
// volume -- DWIConvert stores the front-back axis the other way round, so positions are compared, not array indices.
const D = testData("openneuro-ds001226") ?? ABSENT;
const REF = `${D}derivatives/albula-reference/PAT16-dwiconvert.nhdr`;
const HAVE = (() => { try { Deno.statSync(REF); return true; } catch { return false; } })();
Deno.test({
  name: "PAT16: our NIfTI + FSL reading agrees with Slicer's DWIConvert voxel for voxel, and gradient for gradient once its missing tilt is put back",
  ignore: !HAVE,
  fn: async () => {
    const I = `${D}sub-PAT16/ses-preop/dwi/sub-PAT16_ses-preop_acq-AP_dwi`;
    const ours = fromFsl(await parseNiftiVolumes(Deno.readFileSync(`${I}.nii.gz`)), Deno.readTextFileSync(`${I}.bval`), Deno.readTextFileSync(`${I}.bvec`));
    const ref = await fromNrrdDwi(Deno.readFileSync(REF), Deno.readFileSync(REF.replace(/\.nhdr$/, ".raw")));
    assertEquals(ours.volumes.length, ref.volumes.length);
    close(ours.bValues, ref.bValues, 1e-3);
    // DWIConvert's FSLToNrrd DROPS THE SCAN'S TILT: it turns the image-axis vectors into patient space with the axis
    // flips alone, not the rotation (PAT16 is tilted ~5.5°, so its directions are off by up to 5.5° -- found
    // 2026-09-28; MRtrix3's documentation: FSL vectors are "with respect to the image axes" and are converted by
    // "rotating the gradient vectors according to the rotation part of the transform"). So: ours must equal
    // DWIConvert's with the tilt put back -- R·Aᵀ·g, R our full rotation, A its axis-aligned version.
    const unit3 = (v: number[]) => { const l = Math.hypot(v[0], v[1], v[2]); return v.map((x) => x / l); };
    const Mo = ours.ijkToRAS;
    const cols = [0, 1, 2].map((c) => unit3([Mo[c], Mo[4 + c], Mo[8 + c]]));
    const Rm = [0, 1, 2].map((r) => cols.map((c) => c[r]));
    const Am = Rm.map((row) => row.map((x) => (Math.abs(x) > 0.5 ? Math.sign(x) : 0)));
    const mv = (A: number[][], v: number[]) => A.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]);
    const At = [0, 1, 2].map((r) => [0, 1, 2].map((c) => Am[c][r]));
    let worst = 1, worstRaw = 1;
    ours.gradients.forEach((g, i) => {
      if (ours.bValues[i] === 0) { assertEquals(ref.gradients[i], [0, 0, 0]); return; }
      const r = ref.gradients[i], fixed = unit3(mv(Rm, mv(At, r)));
      worst = Math.min(worst, g[0] * fixed[0] + g[1] * fixed[1] + g[2] * fixed[2]);
      worstRaw = Math.min(worstRaw, g[0] * r[0] + g[1] * r[1] + g[2] * r[2]);
    });
    assert(worst > 0.9999999, `with DWIConvert's missing tilt put back, the directions must agree: worst cos ${worst}`);
    assert(worstRaw < 0.999, "DWIConvert's tilt omission is expected on this oblique scan -- if this fails, it was fixed upstream");
    // Positions: voxel (i,j,k) of ours maps through our ijkToRAS and the reference's inverse to a voxel of the reference.
    const M = ours.ijkToRAS, N = ref.ijkToRAS, [nx, ny, nz] = ours.volumes[0].dims;
    const inv3 = (m: number[]) => {       // inverse of the affine 4x4 (row-major)
      const a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], k = m[10];
      const A = e * k - f * h, B = -(d * k - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
      const R = [A, -(b * k - c * h), b * f - c * e, B, a * k - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((x) => x / det);
      const t = [m[3], m[7], m[11]];
      return [R[0], R[1], R[2], -(R[0] * t[0] + R[1] * t[1] + R[2] * t[2]), R[3], R[4], R[5], -(R[3] * t[0] + R[4] * t[1] + R[5] * t[2]), R[6], R[7], R[8], -(R[6] * t[0] + R[7] * t[1] + R[8] * t[2])];
    };
    const Ni = inv3(N);
    let mismatches = 0, off = 0;
    for (const t of [0, 1, 50, 101]) {
      const a = ours.volumes[t].data, b = ref.volumes[t].data;
      for (let k = 0; k < nz; k += 3) for (let j = 0; j < ny; j += 3) for (let i = 0; i < nx; i += 3) {
        const x = M[0] * i + M[1] * j + M[2] * k + M[3], y = M[4] * i + M[5] * j + M[6] * k + M[7], z = M[8] * i + M[9] * j + M[10] * k + M[11];
        const u = Ni[0] * x + Ni[1] * y + Ni[2] * z + Ni[3], v = Ni[4] * x + Ni[5] * y + Ni[6] * z + Ni[7], w = Ni[8] * x + Ni[9] * y + Ni[10] * z + Ni[11];
        const ri = Math.round(u), rj = Math.round(v), rk = Math.round(w);
        off = Math.max(off, Math.abs(u - ri), Math.abs(v - rj), Math.abs(w - rk));
        if (ri < 0 || rj < 0 || rk < 0 || ri >= nx || rj >= ny || rk >= nz) { mismatches++; continue; }
        if (a[(k * ny + j) * nx + i] !== b[(rk * ny + rj) * nx + ri]) mismatches++;
      }
    }
    assert(off < 1e-3, `the two grids do not coincide (off by ${off} voxel)`);
    assertEquals(mismatches, 0, "every sampled voxel has the same value at the same patient position");
  },
});

Deno.test("NRRD DWI in a space with no patient meaning is refused; a b > 0 volume without direction is named", async () => {
  const enc = new TextEncoder();
  const head = (space: string, g2: string) => `NRRD0005
type: float
dimension: 4
space: ${space}
sizes: 2 1 1 3
space directions: (1,0,0) (0,1,0) (0,0,1) none
kinds: space space space list
endian: little
encoding: raw
space origin: (0,0,0)
modality:=DWMRI
DWMRI_b-value:=1000
DWMRI_gradient_0000:=0 0 0
DWMRI_gradient_0001:=1 0 0
DWMRI_gradient_0002:=${g2}
`;
  const data = new Uint8Array(new Float32Array(6).buffer);
  let err = "";
  try { await fromNrrdDwi(enc.encode(head("scanner-xyz", "0 1 0")), data); } catch (e) { err = String(e); }
  assert(/no patient orientation/.test(err), err || "accepted");
  // FSL: a b = 1000 row with an all-zero vector (a trace image).
  const ras = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const s = fromFsl([vol(ras), vol(ras), vol(ras), vol(ras)], "0 1000 1000 1000", "0 1 0 0\n0 0 1 0\n0 0 0 0");
  assert(/1 volume\(s\) with b > 0 and no direction/.test(s.source), s.source);
});

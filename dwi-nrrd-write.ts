// A DIFFUSION SERIES AS AN NRRD DWI FILE -- the input UKFTractography (the original C++ program) reads, so the comparison
// against it (Contents/tools/ukf-reference.ts; Ron, 2026-10-01: "We will need it as test forward looking") gives both
// programs exactly the same signal. The layout is Slicer's: the gradient axis first (`kinds: list space space space`),
// space RAS, the measurement frame the identity in RAS, one DWMRI_b-value (the largest), each gradient scaled by
// sqrt(b / largest) so its b-value is DWMRI_b-value · |g|² (NRRD's DWI convention, as dwi.ts reads it back).
// Checked by a round trip through fromNrrdDwi (dwi-nrrd-write.test.ts).
import type { DiffusionSeries } from "./dwi.ts";

export function writeNrrdDwi(s: DiffusionSeries): Uint8Array {
  const [nx, ny, nz] = s.volumes[0].dims, N = s.volumes.length, n3 = nx * ny * nz;
  const M = s.ijkToRAS;
  const col = (c: number) => `(${M[c]},${M[4 + c]},${M[8 + c]})`;
  const bmax = Math.max(...s.bValues);
  const lines = [
    "NRRD0005", "# written by albula-diffusion dwi-nrrd-write.ts", "type: float", "dimension: 4", "space: right-anterior-superior",
    `sizes: ${N} ${nx} ${ny} ${nz}`, `space directions: none ${col(0)} ${col(1)} ${col(2)}`, "kinds: list space space space",
    "endian: little", "encoding: raw", `space origin: (${M[3]},${M[7]},${M[11]})`, "measurement frame: (1,0,0) (0,1,0) (0,0,1)",
    "modality:=DWMRI", `DWMRI_b-value:=${bmax}`,
    ...s.gradients.map((g, i) => {
      const k = s.bValues[i] > 0 && bmax > 0 ? Math.sqrt(s.bValues[i] / bmax) : 0;
      return `DWMRI_gradient_${String(i).padStart(4, "0")}:=${(g[0] * k).toFixed(8)} ${(g[1] * k).toFixed(8)} ${(g[2] * k).toFixed(8)}`;
    }),
  ];
  const head = new TextEncoder().encode(lines.join("\n") + "\n\n");
  const data = new Float32Array(n3 * N);
  s.volumes.forEach((v, q) => { const d = v.data; for (let i = 0; i < n3; i++) data[i * N + q] = Number(d[i]); });
  const out = new Uint8Array(head.length + data.byteLength);
  out.set(head); out.set(new Uint8Array(data.buffer), head.length);
  return out;
}

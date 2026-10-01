// A DIFFUSION SERIES AS AN NRRD DWI FILE -- the input UKFTractography (the original C++ program) reads, so the comparison
// against it (Contents/tools/ukf-reference.ts; Ron, 2026-10-01: "We will need it as test forward looking") gives both
// programs exactly the same signal. The layout is Slicer's: the gradient axis first (`kinds: list space space space`),
// space RAS, the measurement frame the identity in RAS, one DWMRI_b-value (the largest), each gradient scaled by
// sqrt(b / largest) so its b-value is DWMRI_b-value · |g|² (NRRD's DWI convention, as dwi.ts reads it back).
// Checked by a round trip through fromNrrdDwi (dwi-nrrd-write.test.ts).
import type { DiffusionSeries } from "./dwi.ts";

/**
 * `space: "LPS"` writes Slicer's own convention (left-posterior-superior: x and y flipped in the directions, the origin and
 * the gradients), which is what UKFTractography is tested on; "RAS" (the default) is the same data in RAS.
 */
export function writeNrrdDwi(s: DiffusionSeries, opts: { space?: "RAS" | "LPS" } = {}): Uint8Array {
  const [nx, ny, nz] = s.volumes[0].dims, N = s.volumes.length, n3 = nx * ny * nz;
  const lps = opts.space === "LPS", f = lps ? [-1, -1, 1] : [1, 1, 1];
  const M = s.ijkToRAS.map((v, i) => (i < 12 ? v * f[Math.floor(i / 4)] : v));
  const col = (c: number) => `(${M[c]},${M[4 + c]},${M[8 + c]})`;
  const bmax = Math.max(...s.bValues);
  const lines = [
    "NRRD0005", "# written by albula-diffusion dwi-nrrd-write.ts", "type: float", "dimension: 4", `space: ${lps ? "left-posterior-superior" : "right-anterior-superior"}`,
    `sizes: ${N} ${nx} ${ny} ${nz}`, `space directions: none ${col(0)} ${col(1)} ${col(2)}`, "kinds: list space space space",
    "endian: little", "encoding: raw", `space origin: (${M[3]},${M[7]},${M[11]})`, "measurement frame: (1,0,0) (0,1,0) (0,0,1)",
    "modality:=DWMRI", `DWMRI_b-value:=${bmax}`,
    ...s.gradients.map((g, i) => {
      const k = s.bValues[i] > 0 && bmax > 0 ? Math.sqrt(s.bValues[i] / bmax) : 0;
      return `DWMRI_gradient_${String(i).padStart(4, "0")}:=${(g[0] * k * f[0]).toFixed(8)} ${(g[1] * k * f[1]).toFixed(8)} ${(g[2] * k * f[2]).toFixed(8)}`;
    }),
  ];
  const head = new TextEncoder().encode(lines.join("\n") + "\n\n");
  const data = new Float32Array(n3 * N);
  s.volumes.forEach((v, q) => { const d = v.data; for (let i = 0; i < n3; i++) data[i * N + q] = Number(d[i]); });
  const out = new Uint8Array(head.length + data.byteLength);
  out.set(head); out.set(new Uint8Array(data.buffer), head.length);
  return out;
}

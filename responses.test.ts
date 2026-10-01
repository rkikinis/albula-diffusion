// RESPONSES AND KERNEL AGAINST DIPY (responses.ts; the "csd-reference" data: DIPY's responses.json and kernel.json for
// PAT16). The kernel from DIPY's own responses must match DIPY's kernel closely (the same formula; DIPY fits on a dense
// sphere where this integrates exactly). The responses estimated here from the scan must come close to DIPY's (the
// masks depend on two tensor fits, ours and DIPY's, at their thresholds).
import { assert } from "jsr:@std/assert@1";
import { parseNiftiVolumes } from "albula";
import { ABSENT, testData } from "albula/testing";
import { fromFsl } from "./dwi.ts";
import { estimateResponses, kernelFromResponses, shellsOf, type Responses } from "./responses.ts";

const REF = (testData("csd-reference") ?? ABSENT) + "PAT16/";
const DWI = testData("openneuro-ds001226", "sub-PAT16/ses-preop/dwi") ?? ABSENT;
const have = (p: string) => { try { Deno.statSync(p); return true; } catch { return false; } };

Deno.test("shells: b-values clustered within 20", () => {
  const s = shellsOf([0, 5, 700, 705, 1200, 1195, 2800, 0]);
  assert(JSON.stringify(s) === JSON.stringify([0, 700, 1195, 2800]), JSON.stringify(s));
});

Deno.test({ name: "the kernel from DIPY's responses matches DIPY's kernel (PAT16)", ignore: !have(REF + "kernel.json"), fn: () => {
  const rj = JSON.parse(Deno.readTextFileSync(REF + "responses.json")), kj = JSON.parse(Deno.readTextFileSync(REF + "kernel.json"));
  const r: Responses = { shells: rj.shells, wm: rj.responses.wm, gm: rj.responses.gm, csf: rj.responses.csf, counts: { wm: 0, gm: 0, csf: 0 }, rule: 1 };
  const k = kernelFromResponses(r);
  let worst = 0;
  k.response.forEach((row, s) => row.forEach((v, c) => { const ref = kj.response[s][c], scale = Math.abs(kj.response[s][2]); worst = Math.max(worst, Math.abs(v - ref) / scale); }));
  console.log(`  largest difference ${worst.toExponential(1)} of the shell's white-matter order-0 term`);
  assert(worst < 2e-3, String(worst));
} });

Deno.test({ name: "responses estimated from PAT16 against DIPY's", ignore: !have(REF + "responses.json") || !have(DWI), fn: async () => {
  const P = DWI + "sub-PAT16_ses-preop_acq-AP_dwi";
  const dwi = fromFsl(await parseNiftiVolumes(Deno.readFileSync(P + ".nii.gz")), Deno.readTextFileSync(P + ".bval"), Deno.readTextFileSync(P + ".bvec"));
  const r = estimateResponses(dwi), rj = JSON.parse(Deno.readTextFileSync(REF + "responses.json"));
  const rel = (a: number, b: number) => Math.abs(a - b) / Math.abs(b);
  const lines: string[] = []; let worst = 0;
  for (const t of ["wm", "gm", "csf"] as const) r[t].forEach((v, s) => {
    const ref = rj.responses[t][s], e = Math.max(rel(v[0], ref[0]), rel(v[1], ref[1]), rel(v[3], ref[3]));
    worst = Math.max(worst, e); lines.push(`${t} b${r.shells[s + 1]}: λ1 ${v[0].toExponential(3)}/${ref[0].toExponential(3)} λ2 ${v[1].toExponential(3)}/${ref[1].toExponential(3)} S0 ${v[3].toFixed(1)}/${ref[3].toFixed(1)}`);
  });
  console.log(`  voxels: wm ${r.counts.wm}, gm ${r.counts.gm}, csf ${r.counts.csf}; largest relative difference ${(100 * worst).toFixed(1)}%\n  ` + lines.join("\n  "));
  assert(worst < 0.05, `responses differ from DIPY's by ${(100 * worst).toFixed(1)}%`);
} });

// @full-tier -- the port against the ORIGINAL UKFTractography (Ron, 2026-10-01: "We will need it as test forward
// looking"), on the reference data Contents/tools/ukf-reference.ts made (test-data "ukf-reference": PAT16, 2,000 seeds,
// free water). Paired by seed point (fiber-compare.ts pairBySeed). As measured 2026-10-01 night, after the seed-FA fix:
// every fiber of the original pairs, the port makes none the original does not, 815 of 838 ends within 0.1 mm. The test
// holds that line -- no worse than today -- and is tightened when the constraint step is settled (critic,
// qa/2026-10-01-ukf-port-vs-original.md, finding 1).
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc ukf-reference.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { ABSENT, testData } from "albula/testing";
import { fromNrrdDwi } from "./dwi.ts";
import { prepareUkfData, trackUkf } from "./ukf.ts";
import { readVtkFibers } from "./vtk-fibers.ts";
import { pairBySeed } from "./fiber-compare.ts";
import { nrrdDecode, nrrdSplitHeader } from "albula";

const R = testData("ukf-reference", "PAT16/") ?? ABSENT;
const have = (() => { try { return Deno.statSync(R + "ukf.vtk").isFile; } catch { return false; } })();

Deno.test({ name: "the port against the original UKFTractography, paired by seed point (PAT16, 2,000 seeds, free water)", ignore: !have, fn: async () => {
  const dwi = await fromNrrdDwi(Deno.readFileSync(R + "dwi.nrrd"));
  // The mask as Contents/tools/ukf-reference.ts wrote it: one unsigned byte a voxel.
  const { f, body } = nrrdSplitHeader(Deno.readFileSync(R + "mask.nrrd"));
  if (!/uchar|unsigned char|uint8/.test(f.type ?? "")) throw new Error(`mask type ${f.type}`);
  const n = (f.sizes ?? "").trim().split(/\s+/).reduce((a, x) => a * Number(x), 1);
  const mask = Uint8Array.from(await nrrdDecode(f, body, n), (v) => (v ? 1 : 0));
  const { starts } = JSON.parse(Deno.readTextFileSync(R + "starts.json")) as { starts: number[][] };
  const M = dwi.ijkToRAS, A = [[M[0], M[1], M[2]], [M[4], M[5], M[6]], [M[8], M[9], M[10]]];
  const det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
  const cof = (r: number, c: number) => { const rr = [0, 1, 2].filter((x) => x !== r), cc = [0, 1, 2].filter((x) => x !== c); return ((r + c) % 2 ? -1 : 1) * (A[rr[0]][cc[0]] * A[rr[1]][cc[1]] - A[rr[0]][cc[1]] * A[rr[1]][cc[0]]); };
  const inv = [0, 1, 2].map((i) => [0, 1, 2].map((j) => cof(j, i) / det));
  const ijk = starts.map((p) => { const q = [p[0] - M[3], p[1] - M[7], p[2] - M[11]]; return [0, 1, 2].map((i) => inv[i][0] * q[0] + inv[i][1] * q[1] + inv[i][2] * q[2]); });
  const original = readVtkFibers(Deno.readTextFileSync(R + "ukf.vtk")).map((f) => new Float32Array(f));
  const port = trackUkf(prepareUkfData(dwi, mask), ijk, { freeWater: true }).fibers.map((f) => f.points);
  const r = pairBySeed(original, port, starts);
  const within = (t: number) => r.ends.filter((x) => x < t).length;
  console.log(`original vs port: ${r.pairs} pairs, only original ${r.onlyA}, only port ${r.onlyB}; ends within 0.001 mm ${within(0.001)}, 0.1 mm ${within(0.1)}, worst ${Math.max(...r.ends).toFixed(1)} mm`);
  assertEquals(r.onlyA, 0, "every fiber of the original has the port's from the same seed");
  assertEquals(r.onlyB, 0, "the port makes no fiber the original does not");
  assert(r.pairs >= 838, `pairs ${r.pairs}`);
  assert(within(0.1) >= 815, `ends within 0.1 mm: ${within(0.1)} (815 on 2026-10-01)`);
}});

// The head's own frame (head-frame.ts): the matrix arithmetic, the two frames' tilts, and the color map in a frame.
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc head-frame.test.ts
import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@1";
import { AFIDS, apply4, frames, I4, mul4, pitch4, rigidM4, riseDeg, TALAIRACH_RISE_DEG } from "./head-frame.ts";
import { applyRigid, rotation } from "./registration.ts";
import { colorFAInFrame } from "./import-job.ts";
import type { TensorFit } from "./tensor.ts";

Deno.test("a rigid move as a matrix moves points as applyRigid does", () => {
  const T = { R: rotation(0.1, -0.2, 0.3), t: [3, -4, 5] as [number, number, number], c: [10, 20, -5] as [number, number, number] };
  for (const p of [[0, 0, 0], [12, -7, 30], [-40, 5, 2]]) {
    const a = applyRigid(T, p), b = apply4(rigidM4(T), p);
    for (let r = 0; r < 3; r++) assertAlmostEquals(a[r], b[r], 1e-9);
  }
  assertEquals(mul4(I4, pitch4(10)), pitch4(10));
});

Deno.test("the frames' tilts: the brainstem plane tips about 36 degrees front-down, the Talairach line rises 1.7", () => {
  assertAlmostEquals(riseDeg(AFIDS.infracollicularSulcus, AFIDS.PMJ), -35.9, 0.2);
  assertAlmostEquals(riseDeg(AFIDS.PC, AFIDS.AC), -5.5, 0.1, "the centers' line (Schaltenbrand)");
  // With the template where the patient is (identity), the brainstem frame's front-back axis runs along the ICS->PMJ line.
  const fr = frames({ templateToPatient: I4, stages: [] }, TALAIRACH_RISE_DEG);
  const B = fr.brainstem.toPatient, y = [B[1], B[5], B[9]], line = [0, AFIDS.PMJ[1] - AFIDS.infracollicularSulcus[1], AFIDS.PMJ[2] - AFIDS.infracollicularSulcus[2]];
  const cos = (y[1] * line[1] + y[2] * line[2]) / (Math.hypot(...y) * Math.hypot(...line));
  assert(cos > 0.999, `brainstem frame's axis along the PMJ line (cos ${cos})`);
  assertEquals(apply4(fr.talairach.toPatient, [0, 0, 0]).map((v) => +v.toFixed(3)), AFIDS.AC.map((v) => +v.toFixed(3)), "origin at the AC");
});

Deno.test("the color map in a frame: a fiber along the frame's up-down axis is blue whatever the scanner's tilt", () => {
  // A frame pitched 30 degrees: its up-down axis is (0, -sin 30, cos 30) in the scanner's RAS.
  const F = pitch4(30), up = [F[2], F[6], F[10]];
  const fit = { fa: new Float32Array([1, 1]), v1: new Float32Array([...up, 1, 0, 0]) } as unknown as TensorFit;
  const c = colorFAInFrame(fit, F);
  assertAlmostEquals(c[2], 1, 1e-6); assertAlmostEquals(c[0], 0, 1e-6); assertAlmostEquals(c[1], 0, 1e-6);
  assertAlmostEquals(c[3], 1, 1e-6, "left-right stays red: the pitch keeps the left-right axis");
});

// The gradient-direction check (gradient-check.ts): its candidates, and the fact it and its test on the case library rest
// on -- a table changed by an orthogonal Q gives exactly the tensors Q D Qᵀ, so each candidate's tensors can be turned
// instead of refitted, and a corrupted table scores the same candidates relabeled.
//   deno test -A --no-check gradient-check.test.ts
import { assert } from "jsr:@std/assert@1";
import { candidates } from "./gradient-check.ts";
import { fitTensors } from "./tensor.ts";
import { M, move, protocol, scan } from "./test/phantom.ts";

const det = (Q: number[]) => Q[0] * (Q[4] * Q[8] - Q[5] * Q[7]) - Q[1] * (Q[3] * Q[8] - Q[5] * Q[6]) + Q[2] * (Q[3] * Q[7] - Q[4] * Q[6]);
const orthonormal = (Q: number[]) => [0, 1, 2].every((a) => [0, 1, 2].every((b) => Math.abs(Q[3 * a] * Q[3 * b] + Q[3 * a + 1] * Q[3 * b + 1] + Q[3 * a + 2] * Q[3 * b + 2] - (a === b ? 1 : 0)) < 1e-9));
// Two tables are the same check if Q and -Q (a direction has no sign).
const same = (A: number[], B: number[]) => A.every((x, i) => Math.abs(x - B[i]) < 1e-9) || A.every((x, i) => Math.abs(x + B[i]) < 1e-9);

Deno.test("candidates: 24 distinct swaps and flips, the record first; two lost tilts only for a tilted scan", () => {
  const straight = candidates(M);
  assert(straight.length === 24, `${straight.length} candidates for an untilted scan`);
  assert(straight[0].label === "as recorded" && same(straight[0].Q, [1, 0, 0, 0, 1, 0, 0, 0, 1]));
  for (const c of straight) assert(orthonormal(c.Q), c.label);
  for (let a = 0; a < 24; a++) for (let b = a + 1; b < 24; b++) assert(!same(straight[a].Q, straight[b].Q), `${straight[a].label} = ${straight[b].label}`);
  // A scan tilted 10° about the left-right axis.
  const th = 10 * Math.PI / 180, c = Math.cos(th), s = Math.sin(th);
  const tilted = [-2.5, 0, 0, 0, 0, 2.5 * c, -2.5 * s, 0, 0, 2.5 * s, 2.5 * c, 0, 0, 0, 0, 1];
  const t = candidates(tilted);
  assert(t.length === 26, `${t.length} candidates for a tilted scan`);
  for (const k of [24, 25]) {
    assert(orthonormal(t[k].Q) && det(t[k].Q) > 0, `${t[k].label}: not a rotation`);
    const angle = Math.acos(Math.min(1, (t[k].Q[0] + t[k].Q[4] + t[k].Q[8] - 1) / 2)) * 180 / Math.PI;
    assert(Math.abs(angle - 10) < 1e-6, `${t[k].label}: turns ${angle.toFixed(3)}°, not the scan's 10°`);
  }
});

Deno.test("a table turned by Q gives the tensors turned by Q (Q D Qᵀ), so candidates are turned, not refitted", () => {
  const { bValues, gradients } = protocol(), c: [number, number, number] = [0, 0, 0];
  const dwi = scan(bValues.map(() => move([0, 0, 0], [0, 0, 0], c)), bValues, gradients, 4);
  const fit = fitTensors(dwi, { maxB: 1500 });
  for (const k of [1, 7, 16]) {
    const Q = candidates(M)[k].Q;
    const turnedTable = { ...dwi, gradients: gradients.map((g) => [0, 1, 2].map((r) => Q[3 * r] * g[0] + Q[3 * r + 1] * g[1] + Q[3 * r + 2] * g[2]) as [number, number, number]) };
    const refit = fitTensors(turnedTable, { maxB: 1500, mask: fit.mask });
    let worst = 0, scale = 0;
    for (let v = 0; v < fit.mask.length; v++) {
      if (!fit.mask[v]) continue;
      const o = 6 * v, d = fit.D, S = [d[o], d[o + 1], d[o + 2], d[o + 1], d[o + 3], d[o + 4], d[o + 2], d[o + 4], d[o + 5]];
      const QS = [0, 1, 2].flatMap((r) => [0, 1, 2].map((cc) => Q[3 * r] * S[cc] + Q[3 * r + 1] * S[3 + cc] + Q[3 * r + 2] * S[6 + cc]));
      const T = [0, 1, 2].flatMap((r) => [0, 1, 2].map((cc) => QS[3 * r] * Q[3 * cc] + QS[3 * r + 1] * Q[3 * cc + 1] + QS[3 * r + 2] * Q[3 * cc + 2]));
      const want = [T[0], T[1], T[2], T[4], T[5], T[8]];
      for (let q = 0; q < 6; q++) { worst = Math.max(worst, Math.abs(refit.D[o + q] - want[q])); scale = Math.max(scale, Math.abs(want[q])); }
    }
    assert(worst < 1e-5 * scale, `candidate ${k}: refitted tensors differ from turned ones by ${worst} (largest ${scale})`);
  }
});

// The gradient-direction check (gradient-check.ts): its candidates; the check itself on the synthetic head (the record
// kept, a scan without a shell at b ≤ 1500 still checked, too few directions said, not thrown); and the fact it rests on
// -- a table changed by an orthogonal Q gives exactly the tensors Q D Qᵀ, so candidates can be turned, not refitted.
//   deno test -A --no-check gradient-check.test.ts
import { assert } from "jsr:@std/assert@1";
import { candidates, checkGradientTable, withCheckedDirections } from "./gradient-check.ts";
import { fitTensors } from "./tensor.ts";
import { M, move, protocol, scan } from "./test/phantom.ts";

const det = (Q: number[]) => Q[0] * (Q[4] * Q[8] - Q[5] * Q[7]) - Q[1] * (Q[3] * Q[8] - Q[5] * Q[6]) + Q[2] * (Q[3] * Q[7] - Q[4] * Q[6]);
const orthonormal = (Q: number[]) => [0, 1, 2].every((a) => [0, 1, 2].every((b) => Math.abs(Q[3 * a] * Q[3 * b] + Q[3 * a + 1] * Q[3 * b + 1] + Q[3 * a + 2] * Q[3 * b + 2] - (a === b ? 1 : 0)) < 1e-9));
// Two tables are the same check if Q and -Q (a direction has no sign).
const same = (A: number[], B: number[]) => A.every((x, i) => Math.abs(x - B[i]) < 1e-9) || A.every((x, i) => Math.abs(x + B[i]) < 1e-9);

Deno.test("candidates: the record first; the 24 swaps and flips of the patient's axes; those of the image's axes that differ, each paired with its twin; two lost tilts from 10°", () => {
  const straight = candidates(M);
  assert(straight.length === 24, `${straight.length} candidates for an untilted scan`);
  assert(straight[0].label === "as recorded" && same(straight[0].Q, [1, 0, 0, 0, 1, 0, 0, 0, 1]));
  const distinct = (cs: { Q: number[]; label: string }[]) => { for (let a = 0; a < cs.length; a++) for (let b = a + 1; b < cs.length; b++) assert(!same(cs[a].Q, cs[b].Q), `${cs[a].label} = ${cs[b].label}`); };
  for (const c of straight) assert(orthonormal(c.Q), c.label);
  distinct(straight);
  // Tilted about the left-right axis: 12° (a lost tilt is offered) and 6° (it is not: below GRADIENT_CHECK.tiltMinDeg).
  const tilted = (deg: number) => { const th = deg * Math.PI / 180, c = Math.cos(th), s = Math.sin(th); return [-2.5, 0, 0, 0, 0, 2.5 * c, -2.5 * s, 0, 0, 2.5 * s, 2.5 * c, 0, 0, 0, 0, 1]; };
  const t12 = candidates(tilted(12)), t6 = candidates(tilted(6));
  for (const cs of [t12, t6]) {
    distinct(cs);
    for (const c of cs) assert(orthonormal(c.Q), c.label);
    assert(cs.filter((c) => c.kind === "patient").length === 23, "the 23 swaps and flips of the patient's axes besides the record");
    for (const [i, c] of cs.entries()) if (c.twin !== undefined) assert(cs[c.twin].twin === i && cs[c.twin].kind !== c.kind, `${c.label}: twin not mutual`);
  }
  assert(t6.every((c) => c.kind !== "tilt"), "a 6° tilt is offered");
  const tilts = t12.filter((c) => c.kind === "tilt");
  assert(tilts.length === 2, `${tilts.length} lost tilts for a 12° scan`);
  for (const c of tilts) {
    assert(det(c.Q) > 0, `${c.label}: not a rotation`);
    const angle = Math.acos(Math.min(1, (c.Q[0] + c.Q[4] + c.Q[8] - 1) / 2)) * 180 / Math.PI;
    assert(Math.abs(angle - 12) < 1e-6, `${c.label}: turns ${angle.toFixed(3)}°, not the scan's 12°`);
  }
  // At 45° the nearest axis-aligned frame is still a rotation away (critic finding 12: per-column rounding made it singular).
  for (const c of candidates(tilted(45))) assert(orthonormal(c.Q), `45°: ${c.label}`);
});

// Detection itself (a flip or swap found) is tested on real brains, by Contents/tools/gradient-check-sweep.ts: the synthetic
// head's fibers circle a nearly vertical axis, so an up-down flip scores about as well as the truth there (270 against
// 263 mm a seed) -- a symmetry no brain has.
Deno.test("the check on the synthetic head: never 'corrected' on a right record; a scan without a shell at b ≤ 1500 is still checked", async () => {
  const { bValues, gradients } = protocol(), c: [number, number, number] = [0, 0, 0];
  const dwi = scan(bValues.map(() => move([0, 0, 0], [0, 0, 0], c)), bValues, gradients, 4);
  const ok = await checkGradientTable(dwi);
  assert(ok.best.kind === "record" && (ok.verdict === "as recorded" || ok.verdict === "unconfirmed"), `the record: ${ok.verdict}, ${ok.best.label}`);
  assert(withCheckedDirections(dwi, ok) === dwi, "the record's directions changed");
  // No shell at or below b = 1500 (critic finding 5): the lowest shell is fitted instead of refusing.
  const high = { ...dwi, bValues: bValues.map((b) => (b > 0 ? 2800 : 0)) };
  const h = await checkGradientTable(high);
  assert(h.verdict !== "not checked", `b = 2800 only: ${h.verdict} ${h.why ?? ""}`);
  // Too few directions for any tensor: "not checked", said, no throw.
  const few = { ...dwi, volumes: dwi.volumes.slice(0, 5), bValues: bValues.slice(0, 5), gradients: gradients.slice(0, 5) };
  const f = await checkGradientTable(few);
  assert(f.verdict === "not checked" && f.best.kind === "record" && !!f.why, `five volumes: ${f.verdict}`);
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

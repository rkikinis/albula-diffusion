// The corticospinal tract checked against the thalamus (cst-gate.ts, rule 1).
import { assertEquals } from "jsr:@std/assert@1";
import { coreOf, gateCorticospinal, isCorticospinal } from "./cst-gate.ts";

// A 20 x 20 x 20 grid of 1 mm voxels, identity placement; a thalamus block x 5..10, y 5..10, z 5..10 (label 7).
const N = 20, lab = new Uint8Array(N * N * N);
for (let k = 5; k <= 10; k++) for (let j = 5; j <= 10; j++) for (let i = 5; i <= 10; i++) lab[(k * N + j) * N + i] = 7;
const brain = { dims: [N, N, N] as const, ijkToRAS: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], lab };
const line = (x: number, y: number) => { const f = new Float32Array(3 * 20); for (let z = 0; z < 20; z++) f.set([x, y, z], 3 * z); return f; };
type Set = { label: string; side: number; streamlines: Float32Array[] };
const unnamed = (sl: Float32Array[]): Set => ({ label: "Unnamed", side: 0, streamlines: sl });

Deno.test("a corticospinal streamline through the thalamus goes to the unnamed ones; one beside it stays", () => {
  const sets: Set[] = [
    { label: "corticospinal tract", side: -1, streamlines: [line(8, 8), line(14, 8)] },   // through the block; 3 mm beside it
    { label: "arcuate fasciculus", side: -1, streamlines: [line(8, 8)] },                 // not the corticospinal tract: kept as it is
  ];
  const r = gateCorticospinal(sets, brain, new Set([7]), unnamed);
  assertEquals(r.sets.find((s) => s.label === "corticospinal tract")!.streamlines.length, 1);
  assertEquals(r.sets.find((s) => s.label === "arcuate fasciculus")!.streamlines.length, 1);
  assertEquals(r.sets.find((s) => s.label === "Unnamed")!.streamlines.length, 1);
  assertEquals(r.moved, [{ label: "corticospinal tract", side: -1, of: 2, moved: 1 }]);
});

Deno.test("grazing the thalamus's edge is not passing through it: the core is one voxel in", () => {
  const core = coreOf(brain, new Set([7]));
  assertEquals(core[(8 * N + 8) * N + 5], 0, "the block's face is not core");
  assertEquals(core[(8 * N + 8) * N + 6], 1);
  // Along x = 5 (the block's face) and x = 10: in the label, never in the core.
  const r = gateCorticospinal([{ label: "corticospinal tract", side: 1, streamlines: [line(5, 8), line(10, 8)] }], brain, new Set([7]), unnamed);
  assertEquals(r.moved[0].moved, 0);
  assertEquals(r.sets.length, 1, "no unnamed set made when nothing moved");
});

Deno.test("an existing unnamed set takes the moved streamlines; names are matched as TractCloud writes them", () => {
  const r = gateCorticospinal([{ label: "corticospinal tract", side: 1, streamlines: [line(8, 8)] }, unnamed([line(1, 1)])], brain, new Set([7]), unnamed);
  assertEquals(r.sets.map((s) => s.label), ["Unnamed"], "an emptied tract is left out");
  assertEquals(r.sets[0].streamlines.length, 2);
  assertEquals([isCorticospinal("corticospinal tract"), isCorticospinal("Cortico-spinal"), isCorticospinal("cortico-ponto-cerebellar")], [true, true, false]);
});

import { cstFails, coreOf as core2, voxelIndexer, CST_RULE_2 } from "./cst-gate.ts";
Deno.test("rule 2: each test, in order -- thalamus, lentiform, crus, the end's parcel", () => {
  // Labels on the 20^3 grid: thalamus 7 (block above), putamen 8 at x 14..16, brainstem 9 at z 0..2, cortex: allowed 20 at
  // z 18..19 for x < 10, another gyrus 21 at z 18..19 for x >= 10.
  const L = new Uint8Array(lab);
  for (let k = 0; k < N; k++) for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const v = (k * N + j) * N + i;
    if (i >= 14 && i <= 16 && k >= 5 && k <= 10) L[v] = 8;
    if (k <= 2 && !L[v]) L[v] = 9;
    if (k >= 18) L[v] = i < 10 ? 20 : 21;
  }
  const brain2 = { dims: [N, N, N] as const, ijkToRAS: brain.ijkToRAS, lab: L };
  const A = { brain: brain2, thalamus: new Set([7]), lentiform: new Set([8]), cortex: new Set([20, 21]), allowedEnd: new Set([20]),
    crus: { origin: [0, 0, 1.5] as [number, number, number], normal: [0, 0, 1] as [number, number, number], errant: new Set([9]) } };
  const at = voxelIndexer(brain2), tc = core2(brain2, new Set([7]));
  assertEquals(cstFails(line(8, 8), A, at, tc), "thalamus");
  assertEquals(cstFails(line(15, 2), A, at, tc), "lentiform");
  assertEquals(cstFails(line(3, 3), A, at, tc), "crus", "crosses z = 1.5 in the brainstem");
  const noCrus = { ...A, crus: undefined };
  assertEquals(cstFails(line(3, 3), noCrus, at, tc), undefined, "ends in the allowed gyrus");
  assertEquals(cstFails(line(12, 3), noCrus, at, tc), "end", "ends in another gyrus");
  assertEquals(CST_RULE_2.id, 2);
});

import { CST_RULE_3, gateCorticospinal2 } from "./cst-gate.ts";
Deno.test("rule 3: the end's parcel by majority, not by axis order; the crus test only where the plane is in the ventral diencephalon", () => {
  // An end in white matter at (10, 10, 15): one voxel of gyrus 21 at +i (rule 2's first neighbor), nine of gyrus 20 at -i.
  const L = new Uint8Array(N * N * N), at3 = (i: number, j: number, k: number) => (k * N + j) * N + i;
  L[at3(11, 10, 15)] = 21;
  for (let k = 14; k <= 16; k++) for (let j = 9; j <= 11; j++) L[at3(9, j, k)] = 20;
  for (let k = 0; k <= 2; k++) for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) L[at3(i, j, k)] = 9;   // brainstem low down
  const g = { dims: [N, N, N] as const, ijkToRAS: brain.ijkToRAS, lab: L };
  const short = () => { const f = new Float32Array(3 * 16); for (let z = 0; z <= 15; z++) f.set([10, 10, z], 3 * z); return f; };
  const A = { brain: g, thalamus: new Set<number>(), lentiform: new Set<number>(), cortex: new Set([20, 21]), allowedEnd: new Set([20]) };
  const at = voxelIndexer(g), tc = core2(g, new Set());
  assertEquals(cstFails(short(), A, at, tc, CST_RULE_2), "end", "rule 2: the +i neighbor decides");
  assertEquals(cstFails(short(), A, at, tc, CST_RULE_3), undefined, "rule 3: the majority decides");
  // The crus plane at z = 1.5 crosses the brainstem label: with no ventral diencephalon there, the test is skipped and said.
  const crus = { origin: [0, 0, 1.5] as [number, number, number], normal: [0, 0, 1] as [number, number, number], errant: new Set([9]) };
  const sets = [{ label: "corticospinal tract", side: 1, streamlines: [short()] }];
  const skipped = gateCorticospinal2(sets, { ...A, crus: { ...crus, ventral: new Set([30]) } }, unnamed, CST_RULE_3);
  assertEquals(skipped.crus, { ventralVoxels: 0, brainstemVoxels: 800, ventralShareOfCrossings: 0, tested: false }, "the plane at z = 1.5 cuts two 20 x 20 layers of brainstem");
  assertEquals(skipped.moved[0].kept, 1);
  const tested = gateCorticospinal2(sets, { ...A, crus: { ...crus, ventral: new Set([9]) } }, unnamed, CST_RULE_3);
  assertEquals(tested.crus?.tested, true);
  assertEquals(tested.moved[0].crus, 1);
  assertEquals(gateCorticospinal2(sets, { ...A, crus }, unnamed, CST_RULE_2).crus, undefined, "rule 2 does not check the plane");
});

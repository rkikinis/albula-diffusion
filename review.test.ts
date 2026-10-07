// The Tract review's own arithmetic (review.ts): which streamlines are judged, on which side, at which levels.
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc review.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { carriedVerdict, CST, cstOf, dorsalTo, fibersFingerprint, frameAxial, frameCoronal, inPlane, intoFrame, levelsFromPair, levelsOf, mergeCase, midlineX, sideToJudge, type Review, type ReviewFile } from "./review.ts";
import type { TractSetData } from "./tracts-dicom.ts";

/** A synthetic corticospinal tract as the real ones are (the test cases, 2026-10-06): 60 fibers from z = -40 (brainstem)
 *  to z = +60 (cortex) around x = cx, a narrow trunk 3-4 mm across up to z = +20, then a fan widening to 30 mm. */
function tract(cx: number): Float32Array[] {
  const width = (z: number) => z < 20 ? 3.5 - 0.01 * z : 3.3 + (z - 20) * 0.7;
  return Array.from({ length: 60 }, (_, k) => {
    const a = (2 * Math.PI * k) / 60, r = 0.4 + 0.6 * ((k * 7) % 10) / 10, pts: number[] = [];
    for (let z = -40; z <= 60; z += 1) pts.push(cx + Math.cos(a) * r * width(z), 10 + Math.sin(a) * r * width(z) * 0.6, z);
    return new Float32Array(pts);
  });
}

Deno.test("the corticospinal tract of each side, the midline between them, and the side away from the tumor", () => {
  const left = tract(-20), right = tract(24);
  const sets: TractSetData[] = [
    { label: CST, side: -1, color: [1, 0, 0], streamlines: left }, { label: CST, side: 1, color: [1, 0, 0], streamlines: right },
    { label: "arcuate fasciculus", side: -1, color: [0, 1, 0], streamlines: tract(-40) },
  ];
  const c = cstOf(sets);
  assertEquals([c.left.length, c.right.length], [60, 60]);
  const mid = midlineX(c.left, c.right);
  assert(Math.abs(mid - 2) < 0.5, `midline at ${mid}, not 2`);
  assertEquals(sideToJudge(30, mid), -1, "a tumor on the right: the left is judged");
  assertEquals(sideToJudge(-30, mid), 1, "a tumor on the left: the right is judged");
  assertEquals(sideToJudge(undefined, mid), -1, "no tumor (a control): the left");
  assertEquals(sideToJudge(1.5, mid), 1, "the midline is the tracts', not the scanner's x = 0");
  // One side missing: the midline from the other, 25 mm across.
  assertEquals(midlineX(c.left, []), midlineX(c.left, c.left) + 25);
});

Deno.test("the levels: the internal capsule 6 mm below where the fan begins, the peduncle 22 mm below it, the coronal through the tract", () => {
  const l = levelsOf(tract(-20));
  assert(l, "no levels");
  // The spread passes 1.4 times the trunk's a few millimeters above z = 20 (smoothed): the internal capsule near 20.
  assert(Math.abs(l!.ic - 20) <= 4, `internal capsule at ${l!.ic}, not near 20`);
  assertEquals(l!.crus, l!.ic - 22);
  assert(Math.abs(l!.coronal - 10) <= 1, `coronal at ${l!.coronal}, not through the tract (y = 10)`);
  assertEquals(levelsOf([]), undefined);
});

/** A pair of corticospinal tracts whose half-separation is 4 mm up to z = -10, then grows 0.6 mm per mm of height. */
function pair(): { left: Float32Array[]; right: Float32Array[] } {
  const half = (z: number) => (z < -10 ? 4 : 4 + (z + 10) * 0.6);
  const one = (side: number) => Array.from({ length: 30 }, (_, k) => {
    const pts: number[] = [];
    for (let z = -40; z <= 60; z += 1) pts.push(side * half(z) + Math.cos(k) * 0.8, 10 + Math.sin(k) * 0.8, z);
    return new Float32Array(pts);
  });
  return { left: one(-1), right: one(1) };
}

Deno.test("the levels from the two tracts' separation: the peduncle where each is 11 mm from the midline, the internal capsule at 22 mm", () => {
  const { left, right } = pair();
  const l = levelsFromPair(left, right, -1)!;
  assert(l, "no levels");
  assert(Math.abs(l.crus - 2) <= 2, `peduncle at ${l.crus}, not near 2 (half-separation 11 mm)`);
  assert(Math.abs(l.ic - 20) <= 2, `internal capsule at ${l.ic}, not near 20`);
  assert(Math.abs(l.coronal - 10) <= 1);
  assertEquals(levelsFromPair(left, [], -1), undefined, "one side missing: not from the pair");
});

Deno.test("a pair that never gets far apart: the internal capsule at 85% of the widest, not at its peak (critic R2-3)", () => {
  // Half-separation 4 mm up to z = -10, rising 0.6 mm per mm to a peak of 22.3 mm at z = 20.5, then falling.
  const half = (z: number) => (z < -10 ? 4 : z < 20.5 ? 4 + (z + 10) * 0.6 : 22.3 - (z - 20.5) * 0.6);
  const one = (side: number) => Array.from({ length: 30 }, (_, k) => {
    const pts: number[] = [];
    for (let z = -40; z <= 60; z += 1) pts.push(side * half(z) + Math.cos(k) * 0.8, 10 + Math.sin(k) * 0.8, z);
    return new Float32Array(pts);
  });
  const l = levelsFromPair(one(-1), one(1), 1)!;
  assert(l, "levels found");
  // 85% of ~22 is ~19: reached near z = 15, five or so millimeters below the peak.
  assert(l.ic <= 16 && l.ic >= 12, `internal capsule at ${l.ic}, below the peak`);
});

Deno.test("an action changes only what it names: a window with an out-of-date copy cannot erase a verdict (critic R2-1)", () => {
  const onDisk: ReviewFile = { version: 2, cases: {
    a: { patient: "A", side: "left", levels: { crus: -10, ic: 5, coronal: 3 }, judgments: { "tracts-1": { verdict: "acceptable", judgedAt: "t1" }, "tracts-2": { verdict: "acceptable", judgedAt: "t2", note: "fine" } } },
    b: { patient: "B", side: "right", judgments: { "tracts-9": { verdict: "not acceptable" } } },
  } };
  const base = { patient: "A", side: "left" as const, tracts: "tracts-2" };
  // A note from another window: the verdict on the same tracts stays, the levels stay, case b stays.
  const n = mergeCase(onDisk, "a", { ...base, note: "second look" });
  assertEquals(n.cases.a.judgments["tracts-2"].verdict, "acceptable");
  assertEquals(n.cases.a.judgments["tracts-2"].note, "second look");
  assertEquals(n.cases.a.levels, { crus: -10, ic: 5, coronal: 3 });
  assertEquals(n.cases.b, onDisk.cases.b);
  assertEquals(n.cases.a.judgments["tracts-1"].verdict, "acceptable", "the verdict on older tracts kept (Ron's '6')");
  // Levels moved: the verdict and the note stay.
  const l = mergeCase(n, "a", { ...base, levels: { crus: 16, ic: 25, coronal: 12.7 } });
  assertEquals(l.cases.a.judgments["tracts-2"], n.cases.a.judgments["tracts-2"]);
  assertEquals(l.cases.a.levels!.crus, 16);
  // A verdict: the note stays; an empty note removes it.
  const v = mergeCase(l, "a", { ...base, verdict: { verdict: "not acceptable", judgedAt: "t3" } });
  assertEquals([v.cases.a.judgments["tracts-2"].verdict, v.cases.a.judgments["tracts-2"].note], ["not acceptable", "second look"]);
  assertEquals(mergeCase(v, "a", { ...base, note: "" }).cases.a.judgments["tracts-2"].note, undefined);
  // A case not on disk yet.
  assertEquals(mergeCase({ version: 2, cases: {} }, "c", { patient: "C", side: "right", tracts: "t", verdict: { verdict: "acceptable", judgedAt: "x" } }).cases.c.judgments.t.verdict, "acceptable");
});

Deno.test("a remake that drew the very same fibers carries the verdict over; different fibers do not (critic R2-4)", () => {
  const sl = tract(-20), same = fibersFingerprint(sl.map((f) => f.slice())), other = fibersFingerprint(tract(-21));
  assertEquals(fibersFingerprint(sl), same);
  assert(same !== other);
  const rec: Review = { patient: "A", side: "left", judgments: { old: { verdict: "acceptable", judgedAt: "t1", fibers: same }, older: { verdict: "not acceptable", judgedAt: "t0", fibers: same } } };
  assertEquals(carriedVerdict(rec, "new", same)?.from, "old", "the newest verdict on the same fibers");
  assertEquals(carriedVerdict(rec, "new", other), undefined);
  assertEquals(carriedVerdict({ ...rec, judgments: { ...rec.judgments, new: { verdict: "not acceptable" } } }, "new", same), undefined, "a verdict of its own wins");
});

Deno.test("crossings dorsal to a drawn crus border: compared with the border's height at their own x", () => {
  // A border sloping from (0, 10) to (20, 0): y = 10 - x / 2.
  const border: [number, number][] = [[20, 0], [0, 10], [10, 5]];
  assertEquals(dorsalTo(border, [[10, 4], [10, 6], [0, 9], [20, 1], [-5, 9], [25, -1]]), 4, "below the line at their x: (10,4), (0,9), and beyond the ends (-5,9), (25,-1)");
  assertEquals(dorsalTo([[0, 0]], [[0, -5]]), 0, "a border of one point counts nothing");
});

Deno.test("a crus border is kept per side; drawing one side keeps the other and the verdicts", () => {
  const onDisk: ReviewFile = { version: 2, cases: { a: { patient: "A", side: "left", judgments: { t: { verdict: "acceptable" } }, crusBorder: { right: { points: [[1, 2, 3], [4, 5, 3]], z: 3, drawnAt: "x" } } } } };
  const m = mergeCase(onDisk, "a", { patient: "A", side: "left", tracts: "t", crusBorder: { side: "left", points: [[-1, 2, 3], [-4, 5, 3]], z: 3, drawnAt: "y" } });
  assertEquals(Object.keys(m.cases.a.crusBorder!).sort(), ["left", "right"]);
  assertEquals(m.cases.a.judgments.t.verdict, "acceptable");
  assertEquals(mergeCase(m, "a", { patient: "A", side: "left", tracts: "t", note: "n" }).cases.a.crusBorder, m.cases.a.crusBorder, "another action keeps the borders");
});

Deno.test("the head-frame planes: an axial at height h and a coronal at y, and coordinates in them", async () => {
  const { pitch4, mul4, apply4 } = await import("./head-frame.ts");
  // A frame tipped 30 degrees with its origin at (5, -10, 2).
  const F = mul4([1, 0, 0, 5, 0, 1, 0, -10, 0, 0, 1, 2, 0, 0, 0, 1], pitch4(30));
  const A = frameAxial(F, 7), o = apply4(F, [0, 0, 7]);
  assertEquals([A[3], A[7], A[11]].map((v) => +v.toFixed(6)), o.map((v) => +v.toFixed(6)), "the axial's origin is the frame's (0, 0, h)");
  const p = apply4(F, [3, 4, 7]), q = inPlane(A, p);
  assert(Math.abs(q[0] + 3) < 1e-9 && Math.abs(q[1] - 4) < 1e-9, "in-plane x is the head's left (radiological), y its front");
  const C = frameCoronal(F, -6), n = [C[2], C[6], C[10]], fy = [F[1], F[5], F[9]];
  assert(Math.abs(n[0] * fy[0] + n[1] * fy[1] + n[2] * fy[2] - 1) < 1e-9, "the coronal's normal is the frame's front-back axis");
  // Streamlines carried into the frame land at the frame's coordinates.
  const back = intoFrame([new Float32Array(p)], (await import("./registration.ts")).inv4(F))[0];
  assert(Math.abs(back[0] - 3) < 1e-4 && Math.abs(back[1] - 4) < 1e-4 && Math.abs(back[2] - 7) < 1e-4);
});

Deno.test("frame borders and levels are kept apart from the scanner's; a redraw keeps the old line in the history (critic 2026-10-07)", () => {
  const onDisk: ReviewFile = { version: 2, cases: { a: { patient: "A", side: "left", levels: { crus: 1, ic: 2, coronal: 3 }, judgments: {},
    crusBorder: { left: { points: [[1, 2, 3], [4, 5, 3]], z: 3, drawnAt: "scanner" } } } } };
  const plane = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -20, 0, 0, 0, 1];
  const base = { patient: "A", side: "left" as const, tracts: "t" };
  const m = mergeCase(onDisk, "a", { ...base, crusBorder: { side: "left", points: [[0, 0, -20], [3, 1, -20]], z: -20, drawnAt: "frame", plane } });
  assertEquals(m.cases.a.crusBorder!.left!.drawnAt, "scanner", "the scanner line is untouched");
  assertEquals(m.cases.a.frameBorder!.left!.plane, plane, "the frame line keeps its plane");
  const m2 = mergeCase(m, "a", { ...base, crusBorder: { side: "left", points: [[1, 1, -20], [4, 2, -20]], z: -20, drawnAt: "frame2", plane } });
  assertEquals(m2.cases.a.frameBorder!.left!.drawnAt, "frame2");
  assertEquals(m2.cases.a.borderHistory!.map((b) => b.drawnAt), ["frame"], "the replaced line is kept");
  const m3 = mergeCase(m2, "a", { ...base, levels: { crus: -18, ic: 9, coronal: 2, frame: "head-1" } });
  assertEquals(m3.cases.a.levels, { crus: 1, ic: 2, coronal: 3 }, "the scanner levels stay");
  assertEquals(m3.cases.a.frameLevels!.crus, -18);
});

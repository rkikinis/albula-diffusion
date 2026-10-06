// The Tract review's own arithmetic (review.ts): which streamlines are judged, on which side, at which levels.
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc review.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { CST, cstOf, levelsFromPair, levelsOf, mergeCase, midlineX, sideToJudge, type Review, type ReviewFile } from "./review.ts";
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

Deno.test("the levels from the two tracts' separation: the peduncle where each is 13 mm from the midline, the internal capsule at 22 mm", () => {
  const { left, right } = pair();
  const l = levelsFromPair(left, right, -1)!;
  assert(l, "no levels");
  assert(Math.abs(l.crus - 5) <= 2, `peduncle at ${l.crus}, not near 5`);
  assert(Math.abs(l.ic - 20) <= 2, `internal capsule at ${l.ic}, not near 20`);
  assert(Math.abs(l.coronal - 10) <= 1);
  assertEquals(levelsFromPair(left, [], -1), undefined, "one side missing: not from the pair");
});

Deno.test("a verdict is saved into the file as it is on disk: other cases kept, and the verdict on older tracts kept (Ron's '6')", () => {
  const onDisk: ReviewFile = { version: 2, cases: {
    a: { patient: "A", side: "left", judgments: { "tracts-1": { verdict: "acceptable", judgedAt: "t1" } } },
    b: { patient: "B", side: "right", judgments: { "tracts-9": { verdict: "not acceptable" } } },
  } };
  // Case a's tracts were remade (tracts-2); this window has only the new, unjudged-then-judged record.
  const mine: Review = { patient: "A", side: "left", levels: { crus: -10, ic: 5, coronal: 3 }, judgments: { "tracts-2": { verdict: "not acceptable", judgedAt: "t2" } } };
  const merged = mergeCase(onDisk, "a", mine);
  assertEquals(merged.cases.b, onDisk.cases.b, "another case kept");
  assertEquals(merged.cases.a.judgments["tracts-1"].verdict, "acceptable", "the verdict on the old tracts kept");
  assertEquals(merged.cases.a.judgments["tracts-2"].verdict, "not acceptable");
  assertEquals(merged.cases.a.levels, mine.levels);
});

import { assert, assertEquals } from "jsr:@std/assert@1";
import { flatten, recordDiff, recordDiffText } from "./record-diff.ts";
import { anatomyReason, partnerReason, shellsOf, type SeriesFacts } from "./import-job.ts";

// Item 4 (2026-10-27): the morning's redo would have shown "not corrected" as the first difference.
Deno.test("recordDiff lists every differing path, grouped, the times last", () => {
  const a = { rules: { tracking: 3, distortion: 2 }, inputs: { partner: "P98" }, corrected: "corrected with the reversed scan", made: "t1", streamlines: 281587 };
  const b = { rules: { tracking: 3, distortion: 2 }, inputs: { partner: "P99" }, corrected: "not corrected: same direction", made: "t2", streamlines: 279035, namer: { network: "TractCloud" } };
  const d = recordDiff(a, b);
  assertEquals(d.map((x) => `${x.group}:${x.path}`), ["inputs:inputs.partner", "namer:namer.network", "corrections:corrected", "counts:streamlines", "times:made"]);
  assertEquals(d[1].a, undefined);
  assert(recordDiffText(d).includes("corrected\n    A: corrected with the reversed scan\n    B: not corrected: same direction"));
  assertEquals(recordDiff(a, structuredClone(a)), []);
});

Deno.test("flatten keeps arrays of plain values whole and indexes arrays of objects", () => {
  const f = flatten({ m: [1, 2, 3], refused: [{ uid: "x", why: "y" }], e: {} });
  assertEquals([...f.keys()], ["m", "refused[0].uid", "refused[0].why", "e"]);
});

// Item 3 (2026-10-27): the reasons and the shells written into each tract set's record.
const facts = (o: Partial<SeriesFacts>): SeriesFacts => ({ uid: "u", studyUID: "s", patientUID: "p", description: "d", volumes: 100, bValues: [], directions: 0, ijkToRAS: [], derived: false, ...o });
Deno.test("partnerReason follows partnerOf's order; anatomyReason says same study or another", () => {
  const dwi = facts({ phaseEncoding: "j", volumes: 100 });
  assertEquals(partnerReason(dwi, facts({ phaseEncoding: "j-", volumes: 100 })), "a full-length scan the scanner recorded with the opposite phase encoding");
  assertEquals(partnerReason(dwi, facts({ phaseEncoding: "j-", volumes: 3, bValues: [0, 0, 0] })), "the scanner recorded it with the opposite phase encoding");
  assertEquals(partnerReason(facts({}), facts({ volumes: 3, bValues: [0, 0, 0] })), "b = 0 images placed like the scan; the scanner did not record the phase-encoding direction");
  assertEquals(anatomyReason(dwi, facts({ studyUID: "s" })), "named as a T1, in the same study");
  assertEquals(anatomyReason(dwi, facts({ studyUID: "other" })), "named as a T1, in another study of the same patient");
});

Deno.test("shellsOf groups b-values within 50 and counts the volumes", () => {
  assertEquals(shellsOf([0, 5, 1495, 1500, 1505, 2995, 3000, NaN]), [{ b: 0, volumes: 2 }, { b: 1495, volumes: 3 }, { b: 2995, volumes: 2 }]);
});

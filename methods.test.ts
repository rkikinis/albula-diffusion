// methods.ts: what was done to a scan, in words, from the record its tracts carry.
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc methods.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { howMade, notAligned } from "./methods.ts";
import { jobRules } from "./import-job.ts";

Deno.test("the record the import job writes today is said step by step, every step named by its method", () => {
  const rules = jobRules({ code: "x", labeler: "rapidparc", synthstrip: "y" });
  const lines = howMade({ rules, inputs: { partner: "p", t1: "t" }, corrected: "corrected with the reversed scan (shifts up to 17.8 mm; 16% left)" });
  assertEquals(lines.length, 5);
  assert(lines[0].startsWith("Head movement corrected"));
  assert(lines[1].startsWith("Scanner distortion corrected with the scan taken in the opposite direction"));
  assert(lines[2].startsWith("Aligned to the T1"));
  assert(lines[3].startsWith("Whole-brain tractography"));
  assert(lines[4].includes("RapidParc"));
  // Every rule number the job writes today has words: nothing reads "does not describe".
  assert(!lines.some((l) => l.includes("does not describe")));
});

Deno.test("a missing reversed scan or T1 is said; a rule this version does not know is said as such, not guessed", () => {
  const lines = howMade({ rules: { motion: 1, distortion: 2, registration: 1, tracking: 9 }, inputs: { partner: null, t1: null } });
  assertEquals(lines[1], "Scanner distortion: not corrected (no reversed scan)");
  assertEquals(lines[2], "Not aligned to a T1 (none in the study)");
  assert(lines[3].includes("does not describe (rule 9)"));
  assertEquals(howMade(undefined), []);
});

Deno.test("a refused alignment is said instead of the alignment's method; rule 3 has its words", () => {
  const ok = howMade({ rules: { motion: 1, distortion: 2, registration: 3, tracking: 3 }, inputs: { partner: "p", t1: "t" }, alignedToT1: true });
  assert(ok.some((l) => /against the brain found on the T1 by SynthStrip/.test(l)), ok.join(" | "));
  const no = howMade({ rules: { motion: 1, distortion: 2, registration: 3, tracking: 3 }, inputs: { partner: "p", t1: "t" }, alignedToT1: false, alignment: { doubt: "the scan matches its own mirror image about as well (0.99)" } });
  assert(no.some((l) => /^Not lined up with the MRI of the anatomy.*mirror image.*where the scanner placed them/.test(l)), no.join(" | "));
  assert(!no.some((l) => /^Aligned to the T1/.test(l)), no.join(" | "));
  assertEquals(notAligned({ inputs: { t1: "t" }, alignedToT1: true }), undefined);
});

// Version 2 (2026-10-28): the distortion line follows what the record says was done.
Deno.test("the distortion line says what the record says, not that a partner existed", async () => {
  const { distortionLine } = await import("./methods.ts");
  const p = { inputs: { partner: "p", t1: "t" } };
  assert(distortionLine({ ...p, corrected: "corrected with the reversed scan (shifts up to 17.8 mm; 16% left)" }).startsWith("Scanner distortion corrected"));
  assertEquals(distortionLine({ ...p, corrected: "not corrected: the reversed scan and this scan do not look like a reversed pair" }), "Scanner distortion: not corrected (the scan taken in the opposite direction could not be used)");
  assertEquals(distortionLine({ inputs: { partner: null }, corrected: "not corrected (no reversed phase-encoding scan)" }), "Scanner distortion: not corrected (no reversed scan)");
  assert(!/^Scanner distortion corrected/.test(distortionLine(p)));   // a record that does not say claims nothing
});

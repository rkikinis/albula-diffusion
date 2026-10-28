// methods.ts: what was done to a scan, in words, from the record its tracts carry.
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc methods.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { howMade, notAligned } from "./methods.ts";
import { jobRules } from "./import-job.ts";

Deno.test("the record the import job writes today is said step by step, every step named by its method", () => {
  const rules = jobRules({ code: "x", labeler: "rapidparc", synthstrip: "y" });
  const lines = howMade({ rules, inputs: { partner: "p", t1: "t" }, corrected: "corrected with the reversed scan (shifts up to 17.8 mm; 16% left)",
    directions: { verdict: "as recorded", used: "as recorded" }, secondOpinion: { agree: true, said: "dcm2niix agrees" } });
  assertEquals(lines.length, 7);
  assertEquals(lines[0], "Diffusion values read a second time by dcm2niix (Li et al. 2016): the two readings agree");
  assertEquals(lines[1], "Diffusion directions checked against the images (Jeurissen, Leemans and Sijbers 2014): the scanner's record matches");
  assert(lines[2].startsWith("Head movement corrected"));
  assert(lines[3].startsWith("Scanner distortion corrected with the scan taken in the opposite direction"));
  assert(lines[4].startsWith("Aligned to the T1"));
  assert(lines[5].startsWith("Whole-brain tractography"));
  assert(lines[6].includes("RapidParc"));
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

// Version 3 (2026-10-28, Lauren O'Donnell: "I hope the user is informed about the outcome of the test"): both checks of the
// scanner's diffusion record are said with their outcome, whatever it was -- and a set made before dcm2niix ran at
// import says that it did not.
Deno.test("the directions' check and dcm2niix's second reading are said with their outcome", async () => {
  const { directionsLine, secondOpinionLine } = await import("./methods.ts");
  assertEquals(directionsLine(undefined), undefined);
  assert(/did not match, so the directions that do were used \(patient axes: x y -z\)/.test(directionsLine({ verdict: "corrected", used: "patient axes: x y -z" })!));
  assert(/could not be fully confirmed and was used as recorded/.test(directionsLine({ verdict: "unconfirmed" })!));
  assert(/treat these tracts with caution/.test(directionsLine({ verdict: "uncertain", used: "image axes: y x z" })!));
  assert(/does not describe \(sideways\)/.test(directionsLine({ verdict: "sideways" })!));
  assert(/differ -- dcm2niix reads this scan differently: directions differ by up to 3.0°/.test(secondOpinionLine({ agree: false, said: "dcm2niix reads this scan differently: directions differ by up to 3.0°. Check before making tracts." })));
  assert(/could not be done \(no files\)/.test(secondOpinionLine({ error: "no files" })));
  assertEquals(secondOpinionLine(undefined), "Diffusion values read a second time by dcm2niix (Li et al. 2016): not done when these tracts were made");
  // A job's set from before the second opinion ran at import says so; a record without the job's rules says nothing of it.
  const before = howMade({ rules: { directions: 2, motion: 1, distortion: 2, registration: 1, tracking: 2 }, inputs: { partner: null, t1: null } });
  assert(before[0].endsWith("not done when these tracts were made"), before.join(" | "));
  assert(!howMade({ rules: { motion: 1 }, inputs: {} }).some((l) => /dcm2niix/.test(l)));
});

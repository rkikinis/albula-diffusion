// The face's decisions (face.ts): what counts as a tumor outline, which patient, which image is the T1, what is measured
// from, and that a new run replaces the last. Each case here is one a resident meets.
//   deno test -A --no-check face.test.ts   (with core's config, as the rebuild runs it)
import { assertEquals } from "jsr:@std/assert";
import { faceNear, isTumorName, matchesSearch, patientOf, pickAnatomy, seriesPart, tractGroupKey, TRACT_GROUPS, withoutLastRun } from "./face.ts";
import { TRACT_INFO } from "./tract-info.ts";

Deno.test("a tumor outline is one named as a tumor; an AI brain parcellation's structures are not", () => {
  for (const yes of ["Tumor (tumor_masks)", "Tumor", "Tumour", "Glioblastoma", "Meningioma, right", "Lesion 1", "Neoplasm", "Brain metastasis", "Vestibular schwannoma"]) {
    assertEquals(isTumorName(yes), true, yes);
  }
  for (const no of ["Left lateral ventricle", "Brain-Stem", "Right cerebral white matter", "Cerebellum cortex", "Segment_1", "Not tumor", "non-tumor tissue"]) {
    assertEquals(isTumorName(no), false, no);   // "Not tumor": the word is there, the meaning is the opposite
  }
});

Deno.test("the patient is the part of the name before the dot", () => {
  assertEquals(patientOf("ds001226-sub-PAT16 · MR T1_mprage"), "ds001226-sub-PAT16");
  assertEquals(patientOf("T1_mprage"), "");
  assertEquals(seriesPart("ds001226-sub-PAT16 · MR T1_mprage"), "T1_mprage");
  assertEquals(seriesPart("Patient1 · SEG Tumor (tumor_masks)"), "Tumor (tumor_masks)");
});

Deno.test("the T1 is found in the series name, never in the patient's (critic, finding 15)", () => {
  const flair = { name: "ds001226-sub-PAT10 · MR FLAIR" }, t1 = { name: "ds001226-sub-PAT10 · MR T1_mprage" };
  assertEquals(pickAnatomy([flair, t1]), t1, "PAT10 has a 1 after T: the patient part must not count");
  assertEquals(pickAnatomy([{ name: "Patient1 · MR FLAIR" }, { name: "Patient1 · MR T1 post contrast" }])?.name, "Patient1 · MR T1 post contrast");
  assertEquals(pickAnatomy([{ name: "X · MR T10 map" }, { name: "X · MR t1w" }])?.name, "X · MR t1w", "T10 is not T1");
  assertEquals(pickAnatomy([flair])?.name, flair.name, "no T1: the first image");
  assertEquals(pickAnatomy([]), undefined);
});

Deno.test("the face measures from a tumor only", () => {
  assertEquals(faceNear(["seg2#1"], "seg1#4"), "seg2#1", "a ventricle chosen under Advanced is not used by the face");
  assertEquals(faceNear(["seg2#1", "seg3#1"], "seg3#1"), "seg3#1");
  assertEquals(faceNear([], "seg1#4"), "", "no tumor: nothing to measure from; the button waits");
});

Deno.test("a new run replaces the last run for that scan, and only that (critic, finding 5)", () => {
  const g = [{ scan: "a", run: true, n: 1 }, { scan: "a", n: 2 }, { scan: "b", run: true, n: 3 }];
  assertEquals(withoutLastRun(g, "a").map((x) => x.n), [2, 3]);
});

Deno.test("the tract list's groups: all of the corpus callosum together; every named tract has a group (Ron, 2026-10-01)", () => {
  for (const t of TRACT_INFO.filter((x) => x.abbr.startsWith("CC"))) assertEquals(tractGroupKey(t.category, false), "cc", t.abbr);
  for (const t of TRACT_INFO) assertEquals(TRACT_GROUPS.some((g) => g.key === tractGroupKey(t.category, false) && g.key !== "unnamed"), true, t.abbr);
  assertEquals(tractGroupKey("Association", true), "unnamed", "streamlines with no name go last");
});

Deno.test("the search finds a tract by its name, its abbreviation or its anatomical term", () => {
  assertEquals(matchesSearch("arcuate", "Arcuate fasciculus, right (AF)"), true);
  assertEquals(matchesSearch("(af)", "Arcuate fasciculus, right (AF)"), true);
  assertEquals(matchesSearch("radiatio", "Thalamo-frontal, left (TF)", "radiatio thalamica anterior"), true);
  assertEquals(matchesSearch("cingulum", "Arcuate fasciculus, right (AF)"), false);
  assertEquals(matchesSearch("  ", "anything"), true, "an empty search shows everything");
});

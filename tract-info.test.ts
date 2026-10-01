// The tract table the module's face reads (tract-info.ts): one row per tract TractCloud names, and the face's label.
//   deno test -A --no-check tract-info.test.ts   (with core's config, as the rebuild runs it)
import { assertEquals } from "jsr:@std/assert";
import { ATLAS_REFERENCE, readMore, tnaLine, TRACT_INFO, tractLabel } from "./tract-info.ts";
import { DIFFUSION_REFERENCES } from "./references.ts";

const model = JSON.parse(await Deno.readTextFile(new URL("./tractcloud/model/model.json", import.meta.url))) as { tracts: { abbr: string }[] };

Deno.test("every tract TractCloud names has one row, and only those", () => {
  const named = model.tracts.map((t) => t.abbr).filter((a) => a !== "Other");
  assertEquals(TRACT_INFO.map((t) => t.abbr), named);
});

Deno.test("the face's label: name, side, abbreviation in parentheses (Ron, 2026-10-01)", () => {
  assertEquals(tractLabel("AF", "arcuate fasciculus", 1), "Arcuate fasciculus, right (AF)");
  assertEquals(tractLabel("CC3", "corpus callosum 3", 0), "Corpus callosum 3 (CC3)");
  assertEquals(tractLabel("XYZ", "new tract", -1), "New tract, left (XYZ)");
});

Deno.test("a reference, when one is entered, has its citation and its link", () => {
  for (const t of TRACT_INFO) if (t.reference) { assertEquals(!!t.reference.cite && /^https:\/\//.test(t.reference.link), true, t.abbr); }
});

Deno.test("Read more falls back to the atlas paper, the same entry the checked reference list has", () => {
  assertEquals(readMore("AF"), ATLAS_REFERENCE);
  assertEquals(DIFFUSION_REFERENCES.some((r) => r.link === ATLAS_REFERENCE.link && r.cite === ATLAS_REFERENCE.cite && r.verified), true);
});

Deno.test("the tooltip's Terminologia Neuroanatomica line says how close the term is", () => {
  assertEquals(tnaLine("AF"), "Terminologia Neuroanatomica (2017): fasciculus arcuatus — arcuate fasciculus, TAH:U6269");
  assertEquals(tnaLine("TF").endsWith("(the nearest term; not the same definition)"), true);
  assertEquals(tnaLine("CR-F").endsWith("(the larger structure this tract is part of)"), true);
  assertEquals(tnaLine("SF"), "", "no entry, no line");
  for (const t of TRACT_INFO) if (t.tna) assertEquals(/^TAH:U\d+$/.test(t.tna.id), true, t.abbr);
});

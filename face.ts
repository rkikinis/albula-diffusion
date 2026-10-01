// THE FACE'S DECISIONS, as plain functions so they are tested (face.test.ts) -- Ron, 2026-10-01, after the critic found
// three of them wrong in one round: "Are there tests that you can add/improve now?"; Mike, the same day: "an algorithm
// is not implemented if it isn't tested" (Bill Lorensen). module.ts calls these; it holds no copy of the rules.

/**
 * A segment counts as a tumor outline when its name says so (critic, 2026-10-01, finding 1: an AI brain parcellation
 * ticked "Tumor outline" and the tracts were measured from its first structure).
 */
export const TUMOR = /tumou?r|neoplas|lesion|glioma|glioblastoma|meningioma|metasta|cancer|carcinoma|lymphoma|schwannoma/i;
export const isTumorName = (label: string): boolean => TUMOR.test(label) && !/\b(not|non)[\s-]+tumou?r/i.test(label);

/** The patient a node belongs to, as its name says it: the part before "·" ("" when there is none). */
export const patientOf = (name: string): string => name.includes("·") ? name.slice(0, name.indexOf("·")).trim() : "";

/** The series part of a node's name: after "·", without the modality ("ds001226-sub-PAT16 · MR T1_mprage" -> "T1_mprage"). */
export const seriesPart = (name: string): string => name.replace(/^.*?·\s*/, "").replace(/^(MR|CT|SEG|PT)\s+/, "");

/**
 * The MRI of the anatomy among candidate images: a T1 by the SERIES part of its name (critic, finding 15: "T1" in a
 * patient ID such as PAT10 or Patient1 made every image a T1), else the first.
 */
export function pickAnatomy<T extends { name?: unknown }>(candidates: T[]): T | undefined {
  return candidates.find((n) => /(^|[^a-z0-9])t1([^0-9]|$)/i.test(seriesPart(String(n.name ?? "")))) ?? candidates[0];
}

/** What the face measures from: the chosen key when it is a tumor, else the first tumor, else nothing. */
export const faceNear = (tumorKeys: string[], chosen: string): string => tumorKeys.includes(chosen) ? chosen : tumorKeys[0] ?? "";

/** Groups after a new run for `scan`: the last run's groups for that scan go (critic, finding 5); others stay. */
export function withoutLastRun<G extends { scan: string; run?: boolean }>(groups: G[], scan: string): G[] {
  return groups.filter((g) => !(g.scan === scan && g.run));
}

/**
 * THE GROUPS OF THE TRACT LIST, Segmentations as the template (Ron, 2026-10-01: "we need to group the tract. All corpus
 * callosum together. Use segmentations as a concept template"): the atlas's own categories, the commissural one named
 * for what it is. Streamlines with no name are their own group, last.
 */
export const TRACT_GROUPS: { key: string; label: string; category?: string }[] = [
  { key: "cc", label: "Corpus callosum", category: "Commissural" },
  { key: "association", label: "Association tracts", category: "Association" },
  { key: "projection", label: "Projection tracts", category: "Projection" },
  { key: "cerebellar", label: "Cerebellar tracts", category: "Cerebellar" },
  { key: "superficial", label: "Superficial (short) fibers", category: "Superficial" },
  { key: "unnamed", label: "Not named" },
];
export function tractGroupKey(category: string | undefined, unnamed: boolean): string {
  if (unnamed || !category) return "unnamed";
  return TRACT_GROUPS.find((g) => g.category === category)?.key ?? "unnamed";
}

/** The search over the tract list: the shown name (abbreviation included) and the anatomical term, case-insensitive. */
export function matchesSearch(query: string, ...texts: (string | undefined)[]): boolean {
  const q = query.trim().toLowerCase();
  return !q || texts.some((t) => !!t && t.toLowerCase().includes(q));
}

// WHAT THE RESIDENT READS ABOUT EACH TRACT: its name on the module's face, what it is for, and a recent paper to read more.
//
// Ron, 2026-10-01: tract names on the face with "the abbreviation in parenthesis"; what each tract is for, in lay words
// -- "I will do it" (he writes or checks that column); and "when we name the fasciculi, a recent reference for more
// information". On TA2: "TA2 has no tracts. Nothing to check." (Ron, 2026-10-01, with Paul Neumann's note that the
// standard deliberately does not name every structure.) So the names are the atlas's own. So this is ONE table, versioned like the other tables (CLAUDE.md, "modular and
// versioned"), that the module reads and nothing else holds a copy of.
//
// VERSION 1 (2026-10-01): the names are TractCloud's own (its atlas, tractcloud/model/model.json), capitalized and
// nothing more; the abbreviations are TractCloud's. The `function` and `reference` columns are EMPTY on purpose: they
// are clinical claims, Ron writes or checks them, and the face shows nothing rather than an unchecked claim.

export const TRACT_INFO_VERSION = 1;

export interface TractInfo {
  /** TractCloud's abbreviation, the key (shown in parentheses after the name). */
  abbr: string;
  /** The name on the face. */
  name: string;
  category: string;
  /** What the tract is for, in words a lay person understands (Ron writes or checks these). */
  function?: string;
  /** A recent paper for more information, checked against the publisher's record before it is entered. */
  reference?: { cite: string; link: string };
}

export const TRACT_INFO: TractInfo[] = [
  { abbr: "AF", name: "Arcuate fasciculus", category: "Association" },
  { abbr: "CB", name: "Cingulum bundle", category: "Association" },
  { abbr: "EC", name: "External capsule", category: "Association" },
  { abbr: "EmC", name: "Extreme capsule", category: "Association" },
  { abbr: "ILF", name: "Inferior longitudinal fasciculus", category: "Association" },
  { abbr: "IOFF", name: "Inferior occipito-frontal fasciculus", category: "Association" },
  { abbr: "MdLF", name: "Middle longitudinal fasciculus", category: "Association" },
  { abbr: "SLF-I", name: "Superior longitudinal fasciculus I", category: "Association" },
  { abbr: "SLF-II", name: "Superior longitudinal fasciculus II", category: "Association" },
  { abbr: "SLF-III", name: "Superior longitudinal fasciculus III", category: "Association" },
  { abbr: "UF", name: "Uncinate fasciculus", category: "Association" },
  { abbr: "CST", name: "Corticospinal tract", category: "Projection" },
  { abbr: "CR-F", name: "Corona radiata frontal", category: "Projection" },
  { abbr: "CR-P", name: "Corona radiata parietal", category: "Projection" },
  { abbr: "SF", name: "Striato-frontal", category: "Projection" },
  { abbr: "SO", name: "Striato-occipital", category: "Projection" },
  { abbr: "SP", name: "Striato-parietal", category: "Projection" },
  { abbr: "TF", name: "Thalamo-frontal", category: "Projection" },
  { abbr: "TO", name: "Thalamo-occipital", category: "Projection" },
  { abbr: "TT", name: "Thalamo-temporal", category: "Projection" },
  { abbr: "TP", name: "Thalamo-parietal", category: "Projection" },
  { abbr: "PLIC", name: "Posterior limb of internal capsule", category: "Projection" },
  { abbr: "CC1", name: "Corpus callosum 1", category: "Commissural" },
  { abbr: "CC2", name: "Corpus callosum 2", category: "Commissural" },
  { abbr: "CC3", name: "Corpus callosum 3", category: "Commissural" },
  { abbr: "CC4", name: "Corpus callosum 4", category: "Commissural" },
  { abbr: "CC5", name: "Corpus callosum 5", category: "Commissural" },
  { abbr: "CC6", name: "Corpus callosum 6", category: "Commissural" },
  { abbr: "CC7", name: "Corpus callosum 7", category: "Commissural" },
  { abbr: "CPC", name: "Cortico-ponto-cerebellar", category: "Cerebellar" },
  { abbr: "ICP", name: "Inferior cerebellar peduncle", category: "Cerebellar" },
  { abbr: "Intra-CBLM-I-P", name: "Intracerebellar input and Purkinje tract", category: "Cerebellar" },
  { abbr: "Intra-CBLM-PaT", name: "Intracerebellar parallel tract", category: "Cerebellar" },
  { abbr: "MCP", name: "Middle cerebellar peduncle", category: "Cerebellar" },
  { abbr: "Sup-F", name: "Superficial frontal", category: "Superficial" },
  { abbr: "Sup-FP", name: "Superficial frontal-parietal", category: "Superficial" },
  { abbr: "Sup-O", name: "Superficial occipital", category: "Superficial" },
  { abbr: "Sup-OT", name: "Superficial occipital-temporal", category: "Superficial" },
  { abbr: "Sup-P", name: "Superficial parietal", category: "Superficial" },
  { abbr: "Sup-PO", name: "Superficial parietal-occipital", category: "Superficial" },
  { abbr: "Sup-PT", name: "Superficial parietal-temporal", category: "Superficial" },
  { abbr: "Sup-T", name: "Superficial temporal", category: "Superficial" },
];

const byAbbr = new Map(TRACT_INFO.map((t) => [t.abbr, t]));
/** The row for a TractCloud abbreviation, if there is one. */
export const tractInfo = (abbr: string): TractInfo | undefined => byAbbr.get(abbr);

/** The name as the face shows it: "Arcuate fasciculus, right (AF)". */
export function tractLabel(abbr: string, fallbackName: string, side: number): string {
  const t = byAbbr.get(abbr);
  const name = t?.name ?? (fallbackName[0]?.toUpperCase() ?? "") + fallbackName.slice(1);
  return `${name}${side > 0 ? ", right" : side < 0 ? ", left" : ""} (${abbr})`;
}

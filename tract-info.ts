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
  /**
   * THE ANATOMICAL TERM, shown in the tooltip (Ron, 2026-10-01: "add TA 2017 to the tooltips", meaning Terminologia
   * Neuroanatomica). Taken from FIPAT's entry pages at ifaa.unifr.ch/Public/TNAEntryPage, which are Terminologia
   * Anatomica Humana (TAH) -- FIPAT's provisional revision of TA98 that carries the TNA terms -- with TAH unit ids; the
   * 2017 TNA publication itself was not consulted (critic, 2026-10-01, finding 6: say what the source is). The official
   * term is the Latin one, exactly as on the page (without "(par)"); `english` as the page gives it, in US spelling
   * (CLAUDE.md: TA terms in American spelling). `match`: "same", "part" (the larger structure this tract is part of),
   * "close" (the nearest term; not the same definition). Read 2026-10-01; corrected the same day (six Latin terms).
   */
  tna?: { latin: string; id: string; english?: string; match: "same" | "part" | "close" };
}

export const TRACT_INFO: TractInfo[] = [
  { abbr: "AF", name: "Arcuate fasciculus", category: "Association", tna: { latin: "fasciculus arcuatus", id: "TAH:U6269", english: "arcuate fasciculus", match: "same" } },
  { abbr: "CB", name: "Cingulum bundle", category: "Association", tna: { latin: "cingulum", id: "TAH:U6270", english: "girdle", match: "same" } },
  { abbr: "EC", name: "External capsule", category: "Association", tna: { latin: "capsula externa", id: "TAH:U6266", english: "external capsule", match: "same" } },
  { abbr: "EmC", name: "Extreme capsule", category: "Association", tna: { latin: "capsula extrema", id: "TAH:U6267", english: "extreme capsule", match: "same" } },
  { abbr: "ILF", name: "Inferior longitudinal fasciculus", category: "Association", tna: { latin: "fasciculus longitudinalis inferior telencephali", id: "TAH:U6271", english: "inferior longitudinal fasciculus of telencephalon", match: "same" } },
  { abbr: "IOFF", name: "Inferior occipito-frontal fasciculus", category: "Association", tna: { latin: "fasciculus occipitofrontalis inferior", id: "TAH:U6276", english: "inferior occipitofrontal fasciculus", match: "same" } },
  { abbr: "MdLF", name: "Middle longitudinal fasciculus", category: "Association", tna: { latin: "fasciculus longitudinalis medius telencephali", id: "TAH:U14222", english: "middle longitudinal fasciculus of telencephalon", match: "same" } },
  { abbr: "SLF-I", name: "Superior longitudinal fasciculus I", category: "Association", tna: { latin: "fasciculus longitudinalis superior I telencephali", id: "TAH:U9484", english: "superior longitudinal fasciculus I of telencephalon", match: "same" } },
  { abbr: "SLF-II", name: "Superior longitudinal fasciculus II", category: "Association", tna: { latin: "fasciculus longitudinalis superior II telencephali", id: "TAH:U9485", english: "superior longitudinal fasciculus II of telencephalon", match: "same" } },
  { abbr: "SLF-III", name: "Superior longitudinal fasciculus III", category: "Association", tna: { latin: "fasciculus longitudinalis superior III telencephali", id: "TAH:U9486", english: "superior longitudinal fasciculus III of telencephalon", match: "same" } },
  { abbr: "UF", name: "Uncinate fasciculus", category: "Association", tna: { latin: "fasciculus uncinatus cerebri", id: "TAH:U6275", english: "uncinate fasciculus of brain", match: "same" } },
  { abbr: "CST", name: "Corticospinal tract", category: "Projection", tna: { latin: "tractus corticospinalis", id: "TAH:U8527", english: "corticospinal tract", match: "same" } },
  { abbr: "CR-F", name: "Corona radiata frontal", category: "Projection", tna: { latin: "corona radiata", id: "TAH:U6265", english: "corona radiata", match: "part" } },
  { abbr: "CR-P", name: "Corona radiata parietal", category: "Projection", tna: { latin: "corona radiata", id: "TAH:U6265", english: "corona radiata", match: "part" } },
  { abbr: "SF", name: "Striato-frontal", category: "Projection" },
  { abbr: "SO", name: "Striato-occipital", category: "Projection" },
  { abbr: "SP", name: "Striato-parietal", category: "Projection" },
  { abbr: "TF", name: "Thalamo-frontal", category: "Projection", tna: { latin: "radiatio thalamica anterior", id: "TAH:U5877", english: "anterior thalamic radiation", match: "close" } },
  { abbr: "TO", name: "Thalamo-occipital", category: "Projection", tna: { latin: "radiatio thalamica posterior", id: "TAH:U5886", english: "posterior thalamic radiation", match: "close" } },
  { abbr: "TT", name: "Thalamo-temporal", category: "Projection" },
  { abbr: "TP", name: "Thalamo-parietal", category: "Projection", tna: { latin: "radiatio thalamica centralis", id: "TAH:U5878", english: "central thalamic radiation", match: "close" } },
  { abbr: "PLIC", name: "Posterior limb of internal capsule", category: "Projection", tna: { latin: "crus posterius capsulae internae", id: "TAH:U6246", english: "posterior limb of internal capsule", match: "same" } },
  { abbr: "CC1", name: "Corpus callosum 1", category: "Commissural", tna: { latin: "corpus callosum", id: "TAH:U6077", english: "corpus callosum", match: "part" } },
  { abbr: "CC2", name: "Corpus callosum 2", category: "Commissural", tna: { latin: "corpus callosum", id: "TAH:U6077", english: "corpus callosum", match: "part" } },
  { abbr: "CC3", name: "Corpus callosum 3", category: "Commissural", tna: { latin: "corpus callosum", id: "TAH:U6077", english: "corpus callosum", match: "part" } },
  { abbr: "CC4", name: "Corpus callosum 4", category: "Commissural", tna: { latin: "corpus callosum", id: "TAH:U6077", english: "corpus callosum", match: "part" } },
  { abbr: "CC5", name: "Corpus callosum 5", category: "Commissural", tna: { latin: "corpus callosum", id: "TAH:U6077", english: "corpus callosum", match: "part" } },
  { abbr: "CC6", name: "Corpus callosum 6", category: "Commissural", tna: { latin: "corpus callosum", id: "TAH:U6077", english: "corpus callosum", match: "part" } },
  { abbr: "CC7", name: "Corpus callosum 7", category: "Commissural", tna: { latin: "corpus callosum", id: "TAH:U6077", english: "corpus callosum", match: "part" } },
  { abbr: "CPC", name: "Cortico-ponto-cerebellar", category: "Cerebellar", tna: { latin: "tractus corticopontini", id: "TAH:U12543", english: "corticopontine tracts", match: "close" } },
  { abbr: "ICP", name: "Inferior cerebellar peduncle", category: "Cerebellar", tna: { latin: "pedunculus cerebellaris inferior", id: "TAH:U5756", english: "inferior cerebellar peduncle", match: "same" } },
  { abbr: "Intra-CBLM-I-P", name: "Intracerebellar input and Purkinje tract", category: "Cerebellar" },
  { abbr: "Intra-CBLM-PaT", name: "Intracerebellar parallel tract", category: "Cerebellar" },
  { abbr: "MCP", name: "Middle cerebellar peduncle", category: "Cerebellar", tna: { latin: "pedunculus cerebellaris medius", id: "TAH:U5759", english: "middle cerebellar peduncle", match: "same" } },
  { abbr: "Sup-F", name: "Superficial frontal", category: "Superficial", tna: { latin: "fibrae associationis breves", id: "TAH:U6274", english: "short association fibers", match: "close" } },
  { abbr: "Sup-FP", name: "Superficial frontal-parietal", category: "Superficial", tna: { latin: "fibrae associationis breves", id: "TAH:U6274", english: "short association fibers", match: "close" } },
  { abbr: "Sup-O", name: "Superficial occipital", category: "Superficial", tna: { latin: "fibrae associationis breves", id: "TAH:U6274", english: "short association fibers", match: "close" } },
  { abbr: "Sup-OT", name: "Superficial occipital-temporal", category: "Superficial", tna: { latin: "fibrae associationis breves", id: "TAH:U6274", english: "short association fibers", match: "close" } },
  { abbr: "Sup-P", name: "Superficial parietal", category: "Superficial", tna: { latin: "fibrae associationis breves", id: "TAH:U6274", english: "short association fibers", match: "close" } },
  { abbr: "Sup-PO", name: "Superficial parietal-occipital", category: "Superficial", tna: { latin: "fibrae associationis breves", id: "TAH:U6274", english: "short association fibers", match: "close" } },
  { abbr: "Sup-PT", name: "Superficial parietal-temporal", category: "Superficial", tna: { latin: "fibrae associationis breves", id: "TAH:U6274", english: "short association fibers", match: "close" } },
  { abbr: "Sup-T", name: "Superficial temporal", category: "Superficial", tna: { latin: "fibrae associationis breves", id: "TAH:U6274", english: "short association fibers", match: "close" } },
];

/**
 * "READ MORE" UNTIL A TRACT HAS ITS OWN (Ron, 2026-10-01: TractCloud's papers are "a good startpoint"): the atlas that
 * defines every tract TractCloud names, open access -- the same entry as in references.ts (checked against Crossref).
 */
export const ATLAS_REFERENCE = {
  cite: "Zhang F, Wu Y, Norton I, Rigolo L, Rathi Y, Makris N, O'Donnell LJ. An anatomically curated fiber clustering white matter atlas for consistent white matter tract parcellation across the lifespan. NeuroImage 179:429-447, 2018.",
  link: "https://doi.org/10.1016/j.neuroimage.2018.06.027",
};
/** The paper to read more about a tract: its own when Ron has entered one, else the atlas's. */
export const readMore = (abbr: string) => byAbbr.get(abbr)?.reference ?? ATLAS_REFERENCE;

const byAbbr = new Map(TRACT_INFO.map((t) => [t.abbr, t]));
/** The row for a TractCloud abbreviation, if there is one. */
export const tractInfo = (abbr: string): TractInfo | undefined => byAbbr.get(abbr);

/** What the tooltip says about a tract's anatomical term (FIPAT's TAH entry pages), or "" when it has no entry. */
export function tnaLine(abbr: string): string {
  const t = byAbbr.get(abbr)?.tna;
  if (!t) return "";
  const how = t.match === "same" ? "" : t.match === "part" ? " (the larger structure this tract is part of)" : " (the nearest term; not the same definition)";
  return `Anatomical term (FIPAT, Terminologia Anatomica Humana, provisional): ${t.latin}${t.english ? ` — ${t.english}` : ""}, ${t.id}${how}`;
}

/** The name as the face shows it: "Arcuate fasciculus, right (AF)". */
export function tractLabel(abbr: string, fallbackName: string, side: number): string {
  const t = byAbbr.get(abbr);
  const name = t?.name ?? (fallbackName[0]?.toUpperCase() ?? "") + fallbackName.slice(1);
  return `${name}${side > 0 ? ", right" : side < 0 ? ", left" : ""} (${abbr})`;
}

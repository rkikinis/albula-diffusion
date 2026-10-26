// WHAT WAS DONE TO A SCAN, IN WORDS (Ron, 2026-10-15, on the module's "made when the scan was imported (10/15/2026) ·
// tracking rule 3": "This information is important to us, but not to the user … They want to know not when things were
// done but what they are: geometric alignment to the t1 using xxxx, whole brain tractography using xxxx, and so on. For
// more detail and references see the help section."). One line per step, from the record the tracts carry (the numbered
// rules and the inputs, import-job.ts jobRules): what the step did and which method did it, by its published name. The
// Methods are named by their papers, never by a program Albula does not run (FSL is a check tool only). The
// date, the rule numbers and the commits stay in the record and the tooltip. A new rule number needs its words here;
// an unknown one is said as such, never guessed (methods.test.ts).

/** The record a stored set of tracts carries (DICOM Tractography Results, the private block's provenance). */
export interface HowMadeRecord { rules?: Record<string, unknown>; inputs?: { partner?: string | null; t1?: string | null };
  /** Whether the alignment was used, and its doubt when it was not (critic 2026-10-26, mirror test, finding 1). */
  alignedToT1?: boolean; alignment?: { doubt?: string } }

const MOTION: Record<number, string> = {
  0: "Head movement: not corrected (the scan came corrected)",
  1: "Head movement corrected (Andersson and Sotiropoulos 2016)",
  2: "Head movement and eddy currents corrected (Andersson and Sotiropoulos 2016)",
  3: "Head movement and eddy currents corrected, tied to the gradient (Andersson and Sotiropoulos 2016)",
};
const TRACKING: Record<number, string> = {
  1: "Whole-brain tractography: two-tensor UKF (Malcolm, Shenton and Rathi 2010)",
  2: "Whole-brain tractography: two-tensor UKF (Malcolm, Shenton and Rathi 2010), from every voxel of the brain",
  3: "Whole-brain tractography: two-tensor UKF (Malcolm, Shenton and Rathi 2010), inside the brain found on the T1 by SynthStrip (Hoopes et al. 2022)",
};
const REGISTRATION: Record<number, string> = {
  1: "Aligned to the T1: rigid, normalized gradient fields (Haber and Modersitzki 2006)",
  2: "Aligned to the T1: rigid, normalized gradient fields (Haber and Modersitzki 2006), from a wide search of starting positions",
  3: "Aligned to the T1: rigid, normalized gradient fields (Haber and Modersitzki 2006), from a wide search of starting positions, against the brain found on the T1 by SynthStrip (Hoopes et al. 2022)",
};

/** THE ALIGNMENT REFUSED: said first and plainly -- the tracts are then in the scanner's placement, not on the T1. */
export function notAligned(rec: HowMadeRecord | undefined): string | undefined {
  if (!rec?.inputs?.t1 || rec.alignedToT1 !== false) return undefined;
  return `Not lined up with the MRI of the anatomy: the automatic alignment looked wrong${rec.alignment?.doubt ? ` (${rec.alignment.doubt})` : ""}. The tracts are shown where the scanner placed them.`;
}

/** The steps, in the order they were done, each one line. */
export function howMade(rec: HowMadeRecord | undefined): string[] {
  const r = rec?.rules ?? {}, i = rec?.inputs ?? {};
  const num = (k: string) => (r[k] === undefined ? undefined : Number(r[k]));
  const unknown = (what: string, n: number) => `${what}: a method this version of Albula does not describe (rule ${n})`;
  const out: string[] = [];
  const m = num("motion"), d = num("distortion"), g = num("registration"), t = num("tracking");
  if (m !== undefined) out.push(MOTION[m] ?? unknown("Head movement", m));
  if (d !== undefined) out.push(i.partner ? "Scanner distortion corrected with the reversed scan (Andersson, Skare and Ashburner 2003)" : "Scanner distortion: not corrected (no reversed scan)");
  if (g !== undefined) out.push(!i.t1 ? "Not aligned to a T1 (none in the study)" : notAligned(rec) ?? REGISTRATION[g] ?? unknown("Aligned to the T1", g));
  if (t !== undefined) out.push(TRACKING[t] ?? unknown("Whole-brain tractography", t));
  if (r.labeler !== undefined) out.push("Tracts named by RapidParc (Bisten et al. 2026) on the fiber-cluster atlas of Zhang, O'Donnell et al. (2018)");
  return out;
}

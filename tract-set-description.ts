// WHAT A TRACT SET IS, FOR SOMEONE WHO DOES NOT KNOW US (Ron, 2026-10-27, on eight rows all reading "Fiber tracts (whole
// brain)": "Useless and puzzling. Remember, our user in the use case is a resident in surgery. Bright but no technical
// background"; "How would a surgery resident in Perth know what Mike's tractline is?"; the mockup
// docs/mockups/tract-sets-in-browser-2026-10-27.html in the workspace, approved the same evening). From a set's own
// record (its provenance) and its motor-pathway counts: a TITLE that says what the tracts are and what sets them apart, in
// a doctor's words (no program, person or project names, no numbers of ours); a LINE with the method that made them and
// its reference (author and year; the full references are in references.ts and the module's Help) and the motor
// pathway's count each side; a TOOLTIP that says what one sees first and stands on its own (MODULE-DESIGN-RULES, Words:
// "Say what was done"). The DICOM browser shows them; this module is the one place that reads a record's meaning.

/** THE DESCRIPTIONS' VERSION, stored beside them as `display_version` (Ron, 2026-10-28: "Are you keeping track of the
 *  proper versioning?"). 1, 2026-10-27: the tooltip said "Corrected for the scanner's distortion and for head movement"
 *  for every set Albula made. 2, 2026-10-28: what was done is read from the record (`preparation`). */
export const DESCRIPTION_VERSION = 2;

export interface TractSetDescription { title: string; line: string; tip: string }

/** The motor pathway's streamline counts, left and right, when the set has one. */
export interface MotorCounts { left: number; right: number }

type Rec = Record<string, unknown>;
const str = (v: unknown) => typeof v === "string" ? v : "";

/** The parts of a record that change what a set is, read once (the record's fields as the writers set them). */
export function traits(rec: Rec) {
  const rules = (rec.rules ?? {}) as Rec;
  const made = JSON.stringify(rec.madeWith ?? {}) + " " + str(rec.note);
  const otherProgram = /tractline/i.test(made);
  const brainFromT1 = otherProgram ? /mask replaced|brain mask replaced|FastSurfer's brain/i.test(made) : /synthstrip|T1/i.test(str(rec.brain)) || Number(rules.tracking) >= 3;
  return {
    otherProgram,
    brainFromT1,
    brainBy: otherProgram ? (brainFromT1 ? "FastSurfer" : "diffusion") : brainFromT1 ? "SynthStrip" : "diffusion",
    stopsAtCortex: Number(rules.tracking) >= 6 || Number(rec.trackingRuleApplied) >= 6,
    secondNamer: /^tractcloud/i.test(str(rules.labeler)),
    motorChecked: !!rec.cstGate,
    motorByAnatomy: !!rec.cstByAnatomy,
  };
}

/** Title, line and tooltip of a tract set from its record. */
export function describeTractSet(rec: Rec, motor?: MotorCounts): TractSetDescription {
  const t = traits(rec);
  const title = t.motorByAnatomy ? "Fiber tracts, motor pathway by anatomy"
    : t.motorChecked ? (t.stopsAtCortex ? "Fiber tracts stopping at the cortex, motor pathway checked" : "Fiber tracts, motor pathway checked")
    : t.otherProgram ? (t.brainFromT1 ? "Fiber tracts, brain outline from the T1" : "Fiber tracts, brain outline from the diffusion scan")
    : t.secondNamer ? "Fiber tracts, named by a second network"
    : t.stopsAtCortex ? "Fiber tracts stopping at the cortex"
    : "Fiber tracts, standard";
  const method = t.otherProgram ? "Two-tensor tractography (tractline, Halle 2026)" : "Two-tensor tractography (Malcolm 2010)";
  const brain = t.brainBy === "diffusion" ? "brain found on the diffusion scan" : t.brainBy === "FastSurfer" ? "brain from the T1 (FastSurfer, Henschel 2020)" : "brain from the T1 (SynthStrip, Hoopes 2022)";
  const naming = t.secondNamer ? "named by TractCloud (Xue 2023)" : "named against an atlas (Zhang 2018)";
  const parts = [method, brain, ...(t.stopsAtCortex ? ["fibers end at the cortex (FastSurfer, Henschel 2020)"] : []), naming];
  const motorWord = t.motorByAnatomy ? "motor pathway by anatomy" : t.motorChecked ? "motor pathway checked" : "motor pathway";
  const line = `${parts.join(", ")}${motor ? ` — ${motorWord} ${motor.left} left, ${motor.right} right` : ""}`;

  const tip: string[] = ["The brain's main nerve fiber tracts, traced in this diffusion scan and named against an atlas of 100 healthy adults."];
  if (t.otherProgram) tip.push(t.brainFromT1
    ? "Traced with a second research program, tractline (M. Halle, 2026), inside the brain found on the T1 scan."
    : "Traced with a second research program, tractline (M. Halle, 2026), which finds the brain on the diffusion scan itself; that can leave out part of the brainstem, so fibers may stop at the internal capsule.");
  else tip.push(preparation(rec, t.brainFromT1));
  if (t.stopsAtCortex) tip.push("Each fiber is stopped where it reaches the cortex, so fewer end outside the brain.");
  if (t.secondNamer) tip.push("The fibers are named by a second naming program, to show how much the names depend on the program.");
  if (t.motorChecked) tip.push("Fibers named motor pathway are kept only if they avoid the thalamus and putamen, pass the crus of the cerebral peduncle and end in the motor cortex.");
  if (t.motorByAnatomy) tip.push("The motor pathway is every fiber that runs through the front of the cerebral peduncle to the motor cortex, whatever the atlas called it; fibers called motor pathway that run elsewhere are not counted.");
  return { title, line, tip: tip.join(" ") };
}

/** What was done to the scan before tracing, from the record's own words (the job's `corrected` sentence and
 *  `alignedToT1`), never assumed: on 2026-10-28 the tooltip said "Corrected for the scanner's distortion" for all of
 *  them, and 26 of the 96 stored sets were not (no reversed scan, or one that did not match). */
export function preparation(rec: Rec, brainFromT1: boolean): string {
  const c = str(rec.corrected);
  const distortion = /^corrected/i.test(c) ? "Corrected for the scanner's distortion"
    : /^not corrected/i.test(c) ? "Not corrected for the scanner's distortion (no usable second scan taken in the opposite direction), so fibers may be shifted, most near the skull base and the sinuses"
    : "";
  const movement = /head movement corrected/i.test(c) ? "corrected for head movement"
    : /head movement not corrected/i.test(c) ? "not corrected for head movement (the correction did not improve the images)" : "";
  const aligned = rec.alignedToT1 === true ? "aligned to the T1 scan" : rec.alignedToT1 === false ? "not aligned to the T1 scan" : "";
  const where = brainFromT1 ? "traced inside the brain found on the T1" : "traced inside the brain found on the diffusion scan";
  if (/^Corrected/.test(distortion) && /^corrected/.test(movement) && rec.alignedToT1 === true && brainFromT1)
    return "Corrected for the scanner's distortion and for head movement, aligned to the T1 scan, and traced inside the brain found on the T1.";
  const parts = [distortion, movement, aligned, where].filter(Boolean);
  const s = parts.length > 1 ? `${parts.slice(0, -1).join("; ")}; ${parts.at(-1)}` : parts[0];
  return s[0].toUpperCase() + s.slice(1) + ".";
}

/** The attribute keys the DICOM browser reads (core's load-panel.ts). */
export const DISPLAY_KEYS = { title: "display_title", line: "display_line", tip: "display_tip" } as const;
/** The key the descriptions' version is stored under. */
export const DISPLAY_VERSION_KEY = "display_version";

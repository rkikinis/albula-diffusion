// THE SCAN PUT IN PLACE before anything is fitted, one path for the module and for a case run (case-run.ts), so the
// regression test checks what the app does: the head's movement between images (motion.ts, when its rule is on), the
// distortion field (distortion.ts, made beforehand from the reversed scan), and the alignment to the MRI of the anatomy
// (registration.ts) -- all three applied in ONE resampling, so the data are interpolated once.
import type { DiffusionSeries } from "./dwi.ts";
import { applyField, type FieldFit } from "./distortion.ts";
import { applyMotion, estimateMotion, MOTION_RULE, type MotionResult, type MotionRuleId } from "./motion.ts";
import { alignToT1, type Grid3, type Rigid } from "./registration.ts";
import type { StageTimes } from "./planning.ts";

export interface Prepared {
  dwi: DiffusionSeries;
  /** What was done, in words, to add to the scan's line ("head movement corrected (…); aligned to …"); and its two parts. */
  said: string;
  movementSaid?: string; alignmentSaid?: string;
  /** The move onto the T1, and a doubt (then not used: the scanner's placement stands). */
  alignment?: { T: Rigid; doubt?: string };
  motion?: MotionResult;
  /** The head-movement rule actually applied (0 when it was off, or could not run: no b = 0, too few images). */
  motionRule: MotionRuleId;
}

/**
 * `dwi` as acquired (the field NOT applied: correctWithReversed with apply false), the field when there is one, the T1
 * when there is one. `motionRule` 0 when the scan was corrected before (a preprocessed dataset) or the rule is off.
 */
export async function prepareScan(dwi: DiffusionSeries, opts: { field?: { fit: FieldFit; sign: 1 | -1 }; t1?: Grid3; motionRule?: MotionRuleId; /** The scanner's record of the phase-encoding direction ("j-", or DICOM's "ROW" / "COL"), for the eddy currents when no field gives the axis. */ phaseEncoding?: string; times?: StageTimes; say?: (s: string) => void } = {}): Promise<Prepared> {
  const rule = opts.motionRule ?? MOTION_RULE, field = opts.field, said: string[] = [];
  let motion: MotionResult | undefined;
  if (rule === 1 || rule === 2 || rule === 3) {
    opts.say?.(rule >= 2 ? "Correcting the head's movement and the eddy-current distortion between the images…" : "Correcting the head's movement between the images…");
    const t = performance.now(), pe = opts.phaseEncoding;
    const peAxis = pe === "ROW" ? 0 : pe === "COL" ? 1 : pe && /^[ijk]/.test(pe) ? "ijk".indexOf(pe[0]) as 0 | 1 | 2 : undefined;
    motion = await estimateMotion(dwi, field, { rule, ...(peAxis !== undefined ? { peAxis } : {}) });
    if (opts.times) opts.times.motion = performance.now() - t;
    said.push(motion.said);
  }
  const inPlace = async () => {
    if (motion) return await applyMotion(dwi, motion, field);
    if (field) for (const v of dwi.volumes) { v.data = applyField(field.fit, v.data as ArrayLike<number>, field.sign); v.dtype = "<f4"; }
    return dwi;
  };
  // Recorded: the rule actually applied, not the one asked for (critic, 2026-10-05, finding 7).
  const done = { ...(motion ? { motion, movementSaid: motion.said } : {}), motionRule: (motion?.rule ?? 0) as MotionRuleId };
  if (!opts.t1) return { dwi: await inPlace(), said: said.join("; "), ...done };
  opts.say?.("Aligning the diffusion scan to the MRI of the anatomy…");
  const a = await alignToT1(dwi, opts.t1, field, opts.times, motion);
  const alignment = { T: a.T, ...(a.doubt ? { doubt: a.doubt } : {}) };
  // A DOUBTFUL ALIGNMENT IS NOT USED (until aligning by hand is built): the scanner's placement stands.
  if (a.doubt) {
    const alignmentSaid = `not aligned to the MRI of the anatomy: the automatic alignment looked wrong (${a.doubt}), so the scanner's placement is used`;
    return { dwi: await inPlace(), said: [...said, alignmentSaid].join("; "), alignmentSaid, alignment, ...done };
  }
  return { dwi: a.dwi, said: [...said, a.said].join("; "), alignmentSaid: a.said, alignment, ...done };
}

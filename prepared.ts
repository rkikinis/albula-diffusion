// WHAT THE IMPORT DID TO A SCAN BEFORE TRACKING, KEPT WITH ITS TRACT SET (SPEED-2026-10-28.md in the workspace, A5; Ron,
// 2026-10-28: "do all including the non gpu optimization"). The GPU review found that the module's Add lines and tracking
// from a point redo the distortion field, the head movement and the direction check in the window -- about 138 s of
// work in V8 on the Prisma scan, slower in the window's engine -- though the import made them already. The import job
// now writes them beside the set's maps (tracts-dicom.ts preparedPath); the module takes them when they were made under
// today's rules for the same reversed scan, as it takes the import's alignment (module.ts importAlignment). The one
// place that writes and reads this file.
import type { FieldFit } from "./distortion.ts";
import type { GradientCheck } from "./gradient-check.ts";
import type { Move } from "./motion.ts";

/** The file's version: a reader refuses any other. */
export const PREPARED_VERSION = 1;

export interface Prepared {
  version: number;
  /** The rules the steps were made under (import-job.ts jobRules) and the reversed scan used (null: none). */
  rules: { distortion: number; motion: number; directions: number };
  partner: string | null;
  /** The field as applied: its sign and the fit (its b as the float32 bytes, base64). */
  field: { sign: 1 | -1; fit: Omit<FieldFit, "b"> & { b: string } } | null;
  /** The movement applied (null: not corrected, or the correction was undone; `movementSaid` says which). */
  moves: Move[] | null; motionRule: number; movementSaid?: string;
  directions?: GradientCheck;
}

const b64 = (a: Float32Array) => { const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); let s = ""; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode(...u.subarray(i, i + 0x8000)); return btoa(s); };
const unb64 = (s: string) => { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Float32Array(u.buffer); };

/** The file's text, from what the pipeline applied. */
export function preparedJson(p: { rules: Prepared["rules"]; partner: string | null; field?: { fit: FieldFit; sign: 1 | -1 }; moves?: Move[]; motionRule: number; movementSaid?: string; directions?: GradientCheck }): string {
  const out: Prepared = { version: PREPARED_VERSION, rules: p.rules, partner: p.partner,
    field: p.field ? { sign: p.field.sign, fit: { ...p.field.fit, b: b64(p.field.fit.b) } } : null,
    moves: p.moves ?? null, motionRule: p.motionRule, ...(p.movementSaid ? { movementSaid: p.movementSaid } : {}), ...(p.directions ? { directions: p.directions } : {}) };
  return JSON.stringify(out);
}

/** The file read back, when it was made under these rules for this reversed scan; otherwise undefined (the module then
 *  does the steps itself, as before). */
export function readPrepared(text: string, want: { rules: Prepared["rules"]; partner: string | null }): { field?: { fit: FieldFit; sign: 1 | -1 }; moves: Move[] | null; motionRule: number; movementSaid?: string; directions?: GradientCheck } | undefined {
  let p: Prepared;
  try { p = JSON.parse(text) as Prepared; } catch { return undefined; }
  if (p.version !== PREPARED_VERSION || p.partner !== want.partner) return undefined;
  if (p.rules.distortion !== want.rules.distortion || p.rules.motion !== want.rules.motion || p.rules.directions !== want.rules.directions) return undefined;
  return { ...(p.field ? { field: { sign: p.field.sign, fit: { ...p.field.fit, b: unb64(p.field.fit.b) } as FieldFit } } : {}),
    moves: p.moves, motionRule: p.motionRule, ...(p.movementSaid ? { movementSaid: p.movementSaid } : {}), ...(p.directions ? { directions: p.directions } : {}) };
}

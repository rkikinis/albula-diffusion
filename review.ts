// TRACT REVIEW -- Ron judges the corticospinal tract on the side without a tumor, case after case (Contents/docs/
// TRACT-REVIEW.md in the workspace; his design and answers, 2026-10-06; mockup docs/mockups/tract-review-2026-10-06.html).
// The ground truth the pipeline lacked: not agreement with another program, but a neuroradiologist's verdict.
//
// A case is a diffusion scan whose fiber tracts were made at import (the import job's Tractography Results object, with
// the direction-colored map it stored beside them). Opening one loads its T1 and its tumor outline from the database,
// lays the map over the T1 (Ron's "1 a"), draws ONLY the corticospinal tract of the side judged, in one color ("2"), in
// Conventional Widescreen: 3D from the front, an axial slice at the cerebral peduncle (crus cerebri, red view), an axial
// slice at the posterior limb of the internal capsule (yellow view), a coronal slice through the tract (green view, "4").
// The two levels are placed from the tract; Ron moves them when he does not like them, and the levels he left are kept
// for next time ("3: … you record for next time"). The side: away from the tumor's center; a control, the left ("5: pick
// one"). Acceptable / Not acceptable, a note, Previous / Next; the verdicts in tract-review.json in the database's folder,
// with the rules of the tracts judged ("6"), so a remake of the tracts knows which verdicts were about the old ones.
import {
  closeScene, databaseFileUrl, databaseSeries, FiberField, fetchZarrVolumeNative, LAYOUT, loadDatabaseSeries, loadVolumeIntoScene,
  lookFrom3D, nrrdDecode, nrrdGeometry, nrrdSplitHeader, orientView, queueModule, seriesDicomFiles, setLayout, setSliceOffset,
  sliceOffset, sliceOrientation, writeDatabaseFile, startPlacing, placingMarkupId, endPlacing, type DatabaseSeries, type ModuleContext, type ZarrDesc,
} from "albula";
import { isTumorName } from "./face.ts";
import { b0Path, colorFaPath, dicomToTracts, type TractSetData } from "./tracts-dicom.ts";
import { crossingOutlines, sliceCrossings } from "./tract-slice.ts";

export const CST = "corticospinal tract";
/** The fibers' one color (the mockup's yellow) and their tube radius in 3D (mm). */
const FIBER_RGB: [number, number, number] = [1, 0.83, 0.3], RADIUS = 0.35;
/** How far around each crossing the outline on a slice runs (mm; about 1 voxel of the 1 mm T1). */
const OUTLINE_MM = 1;
const REVIEWS = "tract-review.json";

// ── The pure parts (tested: review.test.ts) ─────────────────────────────────────────────────────────────────────────

/** The corticospinal tract's streamlines on each side (-1 left, 1 right). */
export function cstOf(sets: TractSetData[]): { left: Float32Array[]; right: Float32Array[] } {
  const pick = (side: number) => sets.filter((s) => s.label === CST && s.side === side).flatMap((s) => s.streamlines);
  return { left: pick(-1), right: pick(1) };
}

const meanX = (sl: Float32Array[]) => { let s = 0, n = 0; for (const f of sl) for (let i = 0; i < f.length; i += 3) { s += f[i]; n++; } return n ? s / n : NaN; };

/** The midline (RAS x, mm): halfway between the two corticospinal tracts -- the scanner's x = 0 need not be the head's. */
export function midlineX(left: Float32Array[], right: Float32Array[]): number {
  const l = meanX(left), r = meanX(right);
  return Number.isFinite(l) && Number.isFinite(r) ? (l + r) / 2 : Number.isFinite(l) ? l + 25 : Number.isFinite(r) ? r - 25 : 0;
}

/** The side judged: away from the tumor's center (RAS x > midline is the right); no tumor, the left (-1). */
export function sideToJudge(tumorX: number | undefined, midline: number): -1 | 1 {
  return tumorX === undefined || !Number.isFinite(tumorX) ? -1 : tumorX > midline ? -1 : 1;
}

export interface Levels { crus: number; ic: number; coronal: number }

/**
 * THE TWO AXIAL LEVELS FROM THE TRACT (RAS z, mm) and the coronal one (RAS y). On real tracts (the test cases, 2026-10-06)
 * the bundle is narrow -- 5 to 7 mm across -- from the brainstem up through the internal capsule, and then fans out to
 * the cortex. So: the crossing points' spread (root mean square distance from their center) per millimeter of height,
 * smoothed over 5 mm; the trunk's width, the median spread over the lower half of the tract's height; where the fan begins,
 * the first height above 30% of the tract where the spread reaches 1.4 times the trunk's and stays so for 6 mm. The
 * internal capsule (its posterior limb) is taken 6 mm below that, the cerebral peduncle 22 mm below the internal capsule
 * (a typical distance), the coronal slice through the bundle at the internal capsule. A first guess; Ron moves them, and
 * the levels he leaves are kept ("you record for next time").
 */
export function levelsOf(sl: Float32Array[]): Levels | undefined {
  const zs: number[] = [];
  for (const f of sl) for (let i = 2; i < f.length; i += 3) zs.push(f[i]);
  if (zs.length < 50) return undefined;
  zs.sort((a, b) => a - b);
  const lo = zs[Math.floor(zs.length * 0.05)], hi = zs[Math.floor(zs.length * 0.95)], span = hi - lo;
  if (!(span > 30)) return undefined;
  const at: { z: number; spread: number; y: number }[] = [];
  for (let z = Math.ceil(lo); z <= hi; z++) {
    const cs = sliceCrossings([sl], { origin: [0, 0, z], normal: [0, 0, 1] });
    if (cs.length < 5) { at.push({ z, spread: NaN, y: NaN }); continue; }
    let cx = 0, cy = 0; for (const c of cs) { cx += c.p[0] / cs.length; cy += c.p[1] / cs.length; }
    let s = 0; for (const c of cs) s += ((c.p[0] - cx) ** 2 + (c.p[1] - cy) ** 2) / cs.length;
    at.push({ z, spread: Math.sqrt(s), y: cy });
  }
  const smooth = at.map((_, i) => { const w = at.slice(Math.max(0, i - 2), i + 3).filter((b) => Number.isFinite(b.spread)); return w.length ? w.reduce((s, b) => s + b.spread, 0) / w.length : NaN; });
  const lower = smooth.filter((s, i) => Number.isFinite(s) && at[i].z <= lo + span / 2).sort((a, b) => a - b);
  if (!lower.length) return undefined;
  const trunk = lower[lower.length >> 1], wide = 1.4 * trunk;
  let fan = -1;
  for (let i = 0; i < at.length && fan < 0; i++) {
    if (at[i].z < lo + 0.3 * span || !(smooth[i] >= wide)) continue;
    if (at.slice(i, i + 7).every((_, k) => !(smooth[i + k] < wide))) fan = i;
  }
  if (fan < 0) return undefined;
  const ic = at[fan].z - 6, crus = ic - 22;
  const near = at.reduce((b, a, i) => (Math.abs(a.z - ic) < Math.abs(at[b].z - ic) ? i : b), 0);
  const y = Number.isFinite(at[near].y) ? at[near].y : at[fan].y;
  return { crus, ic, coronal: Number.isFinite(y) ? +y.toFixed(1) : 0 };
}

/**
 * THE LEVELS FROM ANATOMY, when both corticospinal tracts are there (critic 2026-10-06, finding 4: the fan rule above put
 * PAT08's "peduncle" in the lateral ventricles). The two tracts run close together in the pons and the medulla (each
 * 5-8 mm from the midline), about 9-11 mm from it through the cerebral peduncles (measured; CRUS_HALF_MM) and 22-25 mm through the posterior limb
 * of the internal capsule. So: half the distance between the two tracts' centers per millimeter of height, smoothed over
 * 5 mm; the peduncle where it first reaches CRUS_HALF_MM going up, the internal capsule where it first reaches 22 mm (or 85% of
 * its peak, when lower) above that
 * (on the 59 stored tracts objects of the test cases: 10-25 mm apart, typically 15); the coronal slice through the judged
 * tract at the internal capsule. Undefined when either is not reached.
 */
/** Half the two tracts' separation at the cerebral peduncle (mm). 13 put the slice at the junction of the midbrain and
 *  the diencephalon, the third ventricle behind it (critic 2026-10-06, anatomy finding 1, CON07 and PAT08); 11 puts it in
 *  the midbrain proper -- crura, interpeduncular fossa, tegmentum, aqueduct -- on 7 of 7 cases looked at on the T1
 *  (PAT19, PAT08, PAT28, PAT26, CON07, PAT11, PAT14), 2-4 mm lower. */
export const CRUS_HALF_MM = 11;
export function levelsFromPair(left: Float32Array[], right: Float32Array[], judged: -1 | 1): Levels | undefined {
  if (left.length < 5 || right.length < 5) return undefined;
  const zs: number[] = [];
  for (const f of [...left, ...right]) for (let i = 2; i < f.length; i += 3) zs.push(f[i]);
  zs.sort((a, b) => a - b);
  const lo = Math.ceil(zs[Math.floor(zs.length * 0.05)]), hi = Math.floor(zs[Math.floor(zs.length * 0.95)]);
  const at: { z: number; half: number; y: number }[] = [];
  const center = (sl: Float32Array[], z: number) => { const cs = sliceCrossings([sl], { origin: [0, 0, z], normal: [0, 0, 1] }); if (cs.length < 5) return undefined; let x = 0, y = 0; for (const c of cs) { x += c.p[0] / cs.length; y += c.p[1] / cs.length; } return { x, y }; };
  for (let z = lo; z <= hi; z++) {
    const l = center(left, z), r = center(right, z);
    at.push({ z, half: l && r ? (r.x - l.x) / 2 : NaN, y: (judged < 0 ? l : r)?.y ?? NaN });
  }
  const sm = at.map((_, i) => { const w = at.slice(Math.max(0, i - 2), i + 3).filter((b) => Number.isFinite(b.half)); return w.length ? w.reduce((s, b) => s + b.half, 0) / w.length : NaN; });
  const first = (from: number, thr: number) => at.findIndex((a, i) => a.z >= from && sm[i] >= thr);
  const c = first(lo, CRUS_HALF_MM);
  if (c < 0) return undefined;
  // The internal capsule at 22 mm -- or at 85% of the widest the two get, when that is less than 26 mm: on PAT13 and
  // PAT31 the separation peaks at 22.2-22.3 mm, in the corona radiata, and 22 was met only there (critic 2026-10-06,
  // R2-3); of the other 34 stored cases, 19 are unchanged and 15 move 1-7 mm lower.
  let peak = -Infinity;
  sm.forEach((v, i) => { if (at[i].z >= at[c].z && v > peak) peak = v; });
  const k = first(at[c].z + 5, Math.min(22, 0.85 * peak));
  if (k < 0) return undefined;
  return { crus: at[c].z, ic: at[k].z, coronal: Number.isFinite(at[k].y) ? +at[k].y.toFixed(1) : 0 };
}

/** One case's verdict on one version of its tracts (Ron's "6": a remake of the tracts keeps the verdicts on the old ones).
 *  `fibers` fingerprints the streamlines judged, so a remake that drew the very same ones carries the verdict over
 *  (`carriedFrom`, the older tracts series; critic 2026-10-06, R2-4). */
export interface Judgment { verdict?: "acceptable" | "not acceptable"; note?: string; judgedAt?: string; rules?: Record<string, unknown>; fibers?: string; carriedFrom?: string }
/** One case's record in tract-review.json (version 2): the side, the levels Ron left (kept across remakes of the tracts:
 *  they are anatomy), and a judgment per tracts series. */
export interface Review {
  patient: string; side: "left" | "right";
  /** The levels as Ron left them (only when he moved them; "you record for next time"). */
  levels?: Levels;
  /** Per tracts series: the verdict, the note, when, and the rules the tracts were made under. */
  judgments: Record<string, Judgment>;
  /** The border between each crus and the substantia nigra as Ron drew it on the peduncle slice (RAS points, the slice's
   *  height), for counting the fibers dorsal to it (2026-10-06: "I could draw a line to separate the crus from sn"). */
  crusBorder?: Partial<Record<"left" | "right", CrusBorder>>;
}
export interface CrusBorder { points: [number, number, number][]; z: number; drawnAt: string }
export interface ReviewFile { version: 2; cases: Record<string, Review> }

/** WHAT ONE ACTION CHANGES in a case: a verdict, a note, or the levels -- never the whole record. */
export interface CasePatch {
  patient: string; side: "left" | "right"; tracts: string;
  levels?: Levels; rules?: Record<string, unknown>; fibers?: string;
  verdict?: { verdict: NonNullable<Judgment["verdict"]>; judgedAt: string; carriedFrom?: string };
  /** A note; "" removes it; undefined leaves it. */
  note?: string;
  /** A crus border drawn (one side). */
  crusBorder?: { side: "left" | "right" } & CrusBorder;
}

/** Apply one action to the file AS IT IS ON DISK NOW (read just before writing). Only the fields the action names change:
 *  a window whose copy is out of date can no longer erase a verdict, a note or levels another window -- or an earlier
 *  session -- left (critic 2026-10-06, finding 18 and R2-1: the window's whole record used to win). */
export function mergeCase(onDisk: ReviewFile, key: string, p: CasePatch): ReviewFile {
  const had = onDisk.cases[key];
  const j: Judgment = { ...(had?.judgments?.[p.tracts] ?? {}) };
  if (p.rules) j.rules = p.rules;
  if (p.fibers) j.fibers = p.fibers;
  if (p.verdict) { j.verdict = p.verdict.verdict; j.judgedAt = p.verdict.judgedAt; if (p.verdict.carriedFrom) j.carriedFrom = p.verdict.carriedFrom; else delete j.carriedFrom; }
  if (p.note !== undefined) { if (p.note) j.note = p.note; else delete j.note; }
  const levels = p.levels ?? had?.levels;
  const crusBorder = p.crusBorder ? { ...(had?.crusBorder ?? {}), [p.crusBorder.side]: { points: p.crusBorder.points, z: p.crusBorder.z, drawnAt: p.crusBorder.drawnAt } } : had?.crusBorder;
  const rec: Review = { patient: p.patient, side: p.side, ...(levels ? { levels } : {}), judgments: { ...(had?.judgments ?? {}), [p.tracts]: j }, ...(crusBorder ? { crusBorder } : {}) };
  return { version: 2, cases: { ...onDisk.cases, [key]: rec } };
}

/**
 * HOW MANY CROSSINGS LIE DORSAL TO A DRAWN BORDER (on the axial slice; RAS: dorsal = smaller y). The border is a line of
 * points drawn from one end of the crus to the other; a crossing is compared with the border's y at its own x
 * (interpolated between the two border points around it; beyond the border's ends, the nearer end's y).
 */
export function dorsalTo(border: [number, number][], crossings: [number, number][]): number {
  if (border.length < 2) return 0;
  const b = [...border].sort((p, q) => p[0] - q[0]);
  const yAt = (x: number) => {
    if (x <= b[0][0]) return b[0][1];
    if (x >= b[b.length - 1][0]) return b[b.length - 1][1];
    let i = 1; while (b[i][0] < x) i++;
    const [x0, y0] = b[i - 1], [x1, y1] = b[i], t = x1 > x0 ? (x - x0) / (x1 - x0) : 0;
    return y0 + t * (y1 - y0);
  };
  return crossings.filter(([x, y]) => y < yAt(x)).length;
}

/** A fingerprint of the streamlines judged (their count and every coordinate), for carrying a verdict over a remake that
 *  drew the very same fibers. */
export function fibersFingerprint(sl: Float32Array[]): string {
  let h1 = 0x811c9dc5, h2 = 0x01000193 ^ sl.length;
  for (const f of sl) {
    const b = new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
    for (let i = 0; i < b.length; i++) { h1 = Math.imul(h1 ^ b[i], 0x01000193); h2 = Math.imul(h2 ^ b[i], 0x5bd1e995) ^ (h2 >>> 15); }
    h1 = Math.imul(h1 ^ 0xff, 0x01000193);
  }
  return `${sl.length}:${(h1 >>> 0).toString(16).padStart(8, "0")}${(h2 >>> 0).toString(16).padStart(8, "0")}`;
}

/** The verdict to carry to `tracts` (unjudged): the newest one given on an older tracts series of the same fibers. */
export function carriedVerdict(rec: Review | undefined, tracts: string, fibers: string): { from: string; j: Judgment } | undefined {
  if (!rec || rec.judgments?.[tracts]?.verdict) return undefined;
  const same = Object.entries(rec.judgments ?? {}).filter(([s, j]) => s !== tracts && j.verdict && j.fibers === fibers)
    .sort((a, b) => String(b[1].judgedAt ?? "").localeCompare(String(a[1].judgedAt ?? "")));
  return same.length ? { from: same[0][0], j: same[0][1] } : undefined;
}

// ── The module ──────────────────────────────────────────────────────────────────────────────────────────────────────

interface Case { dwi: DatabaseSeries; tracts: string; patient: string }

function registerTractReview(ctx: ModuleContext): void {
  const { shell, live, store, device } = ctx;
  let root: HTMLElement | undefined, cases: Case[] = [], file: ReviewFile = { version: 2, cases: {} };
  /** The verdicts file as last read: "ok" or "absent" can be written; "failed" (no answer, a server error) is read again
   *  before anything is saved; "unreadable" (it read but is not a verdicts file) is never written over (critic
   *  2026-10-06, finding 2 and R2-2). `fileProblem` stays on screen beside the buttons until a read succeeds. */
  let fileState: "unknown" | "ok" | "absent" | "failed" | "unreadable" = "unknown", fileProblem = "";
  const canWrite = () => fileState === "ok" || fileState === "absent";
  /** The open case's own facts, for the actions' patches. */
  let openPatient = "", openSide: "left" | "right" = "left", openRules: Record<string, unknown> | undefined, openFibers = "";
  /** Both corticospinal tracts of the open case (the crus borders are drawn on both sides), and the side being drawn. */
  let openCst: { left: Float32Array[]; right: Float32Array[] } = { left: [], right: [] }, drawing: "left" | "right" | undefined;
  /** The case open, by its diffusion scan's UID (critic 2026-10-06, finding 8: a row number moves when the list does). */
  let openKey = "", openTracts = "", busy = "", note = "", field: FiberField | undefined, drawn: Float32Array[] = [];
  let placed: Levels | undefined, levelsSaid = "";
  const say = (s: string) => { note = s; ctx.status(s); render(); };
  const currentIndex = () => cases.findIndex((c) => c.dwi.seriesInstanceUID === openKey);

  async function readFile(): Promise<{ state: "ok" | "absent" | "failed" | "unreadable"; file?: ReviewFile }> {
    const url = await databaseFileUrl(`SlicerAlbula-SEG/${REVIEWS}`);
    if (!url) return { state: "failed" };
    const r = await fetch(url, { cache: "no-store" }).catch(() => null);
    if (!r) return { state: "failed" };
    if (r.status === 404) return { state: "absent" };
    if (!r.ok) return { state: "failed" };
    try { const j = await r.json() as { version?: number; cases?: Record<string, unknown> }; if (j?.version === 2 && j.cases) return { state: "ok", file: j as ReviewFile }; } catch { /* below */ }
    return { state: "unreadable" };
  }
  const problemOf = (state: string) => state === "unreadable"
    ? `The verdicts file (${REVIEWS}, in the database's SlicerAlbula-SEG folder) is not a verdicts file this version can read. Nothing will be saved over it, so nothing in it is lost; move it aside to start a new one.`
    : state === "failed" ? "The verdicts could not be read just now (the database did not answer). Nothing is saved until they are read." : "";
  /** Read the verdicts again (on showing the module, on opening a case, on Try again): the disk is the record, and
   *  another window -- or this one before a failed read -- may have changed it. The window's database is the one read. */
  async function refreshFile(): Promise<void> {
    const r = await readFile();
    fileState = r.state; fileProblem = problemOf(r.state);
    if (r.state === "ok") file = r.file!;
    else if (r.state === "absent") file = { version: 2, cases: {} };
  }
  /** Apply ONE action to the file as it is on disk now; never over a file that does not read. THE WINDOW'S SAVES RUN ONE
   *  AFTER ANOTHER (critic 2026-10-06, R3-1): one click can start several (a moved level, a note left by the click, the
   *  verdict); run at once, each read the same file and the last write erased the others' fields. A save that could not
   *  be made is kept, and Try again makes it. */
  let saving: Promise<unknown> = Promise.resolve(), unsaved: { key: string; patch: CasePatch }[] = [];
  function saveCase(key: string, patch: CasePatch): Promise<boolean> {
    const run = saving.then(() => saveNow(key, patch));
    saving = run.catch(() => false);
    return run;
  }
  async function saveNow(key: string, patch: CasePatch): Promise<boolean> {
    const r = await readFile();
    if (r.state === "failed" || r.state === "unreadable") { fileState = r.state; fileProblem = problemOf(r.state); unsaved.push({ key, patch }); render(); return false; }
    const merged = mergeCase(r.file ?? { version: 2, cases: {} }, key, patch);
    const w = await writeDatabaseFile(REVIEWS, JSON.stringify(merged, null, 2) + "\n");
    if (!w.ok) { fileProblem = `The verdict could not be saved: ${w.why}.`; unsaved.push({ key, patch }); render(); return false; }
    file = merged; fileState = "ok"; fileProblem = "";
    return true;
  }
  /** Try again: read the verdicts, then make the saves that could not be made, in their order. */
  async function tryAgain(): Promise<void> {
    await refreshFile();
    if (canWrite()) { const todo = unsaved; unsaved = []; for (const u of todo) await saveCase(u.key, u.patch); }
    render();
  }
  const patchOf = (extra: Partial<CasePatch>): CasePatch => ({ patient: openPatient, side: openSide, tracts: openTracts, ...(openRules ? { rules: openRules } : {}), ...(openFibers ? { fibers: openFibers } : {}), ...extra });

  async function listCases(): Promise<void> {
    const db = await databaseSeries({ fresh: true });
    if (!db) { cases = []; return; }
    const byUid = new Map(db.series.map((s) => [s.seriesInstanceUID, s]));
    // The newest tracts of a scan (a remake is a new series; a series not in the list yet counts as newest).
    const num = (uid: string) => byUid.get(uid)?.seriesNumber ?? Infinity;
    cases = db.edges.filter((e) => e.kind === "tracts" && byUid.has(e.parent)).map((e) => ({ dwi: byUid.get(e.parent)!, tracts: e.child }))
      .filter((c, i, all) => !all.some((o, j) => j !== i && o.dwi.seriesInstanceUID === c.dwi.seriesInstanceUID && (num(o.tracts) > num(c.tracts) || (num(o.tracts) === num(c.tracts) && j > i))))
      .map((c) => ({ ...c, patient: String(c.dwi.patientName ?? c.dwi.patientID ?? "?") }))
      .sort((a, b) => a.patient.localeCompare(b.patient));
  }

  /** Ron's levels, when he moved them: the three views still in the orientations set, and at other positions. */
  function keepLevels(): void {
    if (!openKey || !openTracts || !placed || !hasCase() || !canWrite()) return;
    if (sliceOrientation("Red") !== "Axial" || sliceOrientation("Yellow") !== "Axial" || sliceOrientation("Green") !== "Coronal") return;
    const c = sliceOffset("Red"), k = sliceOffset("Yellow"), y = sliceOffset("Green");
    if (c === undefined || k === undefined || y === undefined) return;
    const now = { crus: +c.toFixed(1), ic: +k.toFixed(1), coronal: +y.toFixed(1) };
    if (now.crus === placed.crus && now.ic === placed.ic && now.coronal === placed.coronal) return;
    placed = now;
    void saveCase(openKey, patchOf({ levels: now }));
  }
  /** Is the case still what the scene shows (its T1 or map loaded)? After File › Close Scene it is not. */
  const hasCase = () => [...live.nodes.values()].some((n) => n.type === "image");
  const sceneHasData = () => [...live.nodes.values()].some((n) => n.type === "image" || n.type === "segmentation");

  function draw(sl: Float32Array[]): void {
    const view = live.view; if (!view) return;
    const old = field; field = undefined; drawn = sl;
    if (sl.length) { field = new FiberField(device, sl.map((p) => ({ points: p, bundle: 1 })), { radius: RADIUS, bundleColors: { 1: [...FIBER_RGB, 1] } }); view.setField("tract-review", field); }
    else view.removeField("tract-review");
    old?.destroy?.();
    drawDots();
  }
  /** Where the tract crosses each slice view, as OUTLINES (Ron, 2026-10-06: dots hid the direction-colored map, which is
   *  what shows the tract in the internal capsule's posterior limb and ventral to the substantia nigra). */
  function drawDots(): void {
    const view = live.view; if (!view?.setOverlay) return;
    for (const n of live.nodes.values()) {
      if (n.type !== "view" || n.kind !== "slice" || !Array.isArray(n.sliceToRAS)) continue;
      const m = n.sliceToRAS as number[], L = Math.hypot(m[2], m[6], m[10]) || 1, nrm: [number, number, number] = [m[2] / L, m[6] / L, m[10] / L];
      const d = typeof n.offset === "number" ? n.offset : m[3] * nrm[0] + m[7] * nrm[1] + m[11] * nrm[2];
      const o: [number, number, number] = [nrm[0] * d, nrm[1] * d, nrm[2] * d];
      const cs = drawn.length ? sliceCrossings([drawn], { origin: o, normal: nrm }) : [];
      const unit = (a: number, b: number, c: number): [number, number, number] => { const l = Math.hypot(a, b, c) || 1; return [a / l, b / l, c / l]; };
      const loops = crossingOutlines(cs.map((c) => c.p), o, unit(m[0], m[4], m[8]), unit(m[1], m[5], m[9]), OUTLINE_MM);
      // Ron's crus borders, on the view whose level they were drawn at.
      const borders = Object.values(file.cases[openKey]?.crusBorder ?? {}).filter((b) => b && Math.abs(nrm[2]) > 0.9 && Math.abs(b.z - d * Math.sign(nrm[2])) < 1.5);
      view.setOverlay(String(n.layoutName ?? n.name), "tract-review", [...loops.map((points) => ({ kind: "polyline" as const, points, color: [...FIBER_RGB, 1], widthPx: 1.5, closed: true })),
        ...borders.map((b) => ({ kind: "polyline" as const, points: b!.points, color: [0.35, 0.85, 1, 1], widthPx: 2 }))]);
    }
  }
  live.subscribe((c) => {
    if (c.type === "view" && String(c.id).startsWith("nativeSlice-") && drawn.length) requestAnimationFrame(drawDots);
    // The scene emptied (File › Close Scene): the review's own fibers go with it.
    if (drawn.length && !hasCase()) { draw([]); openKey = ""; render(); }
  });

  async function loadColorFA(tracts: string, name: string): Promise<string | undefined> {
    return await loadStored(colorFaPath(tracts), name, true);
  }
  /** A volume the import job stored beside the tracts (the Color FA, packed as RGB; the b = 0 image, plain). */
  async function loadStored(path: string, name: string, rgb: boolean): Promise<string | undefined> {
    const url = await databaseFileUrl(path);
    const r = url ? await fetch(url).catch(() => null) : null;
    if (!r?.ok) return undefined;
    const { f, body } = nrrdSplitHeader(new Uint8Array(await r.arrayBuffer()));
    const sizes = (f["sizes"] ?? "").split(/\s+/).filter(Boolean).map(Number) as [number, number, number];
    const raw = await nrrdDecode(f, body, sizes[0] * sizes[1] * sizes[2] * 4);
    const data = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    const res = await loadVolumeIntoScene(live, store, { dims: sizes, ijkToRAS: nrrdGeometry(f).ijkToRAS, data, dtype: "<f4", name }, { name, extra: { ...(rgb ? { rgb24: true } : {}), autoVolumeRendering: false, recomputable: true } });
    if (!rgb) {
      // THE b = 0's WINDOW ON THE BRAIN (critic 2026-10-06, b = 0 finding 5): the automatic one spans the eyes and the
      // ventricles, the brightest things on it, and leaves the brain dim. From the 30th to the 92nd percentile of the
      // non-zero voxels (on PAT08: about 60 to 380, the tissue's range).
      const v = Float32Array.from(data.filter((x) => x > 1)).sort(), q = (t: number) => v[Math.floor(t * (v.length - 1))];
      if (v.length > 100) { const lo = q(0.3), hi = q(0.92);
        for (const [k, val] of [["window", hi - lo], ["level", (hi + lo) / 2], ["autoWindowLevel", false]] as const) live.write({ op: "patch", id: res.displayId, path: `#/${k}`, value: val }); }
    }
    return res.imageId;
  }

  /** The tumor's center (RAS x), from the segments named as a tumor (face.ts isTumorName, the Diffusion module's own rule:
   *  "Not tumor" is not; critic 2026-10-06, R2-6). */
  async function tumorCenterX(segIds: string[]): Promise<number | undefined> {
    let sx = 0, n = 0;
    for (const id of segIds) {
      const s = live.nodes.get(id); if (!s?.zarr) continue;
      const labels = ((s.segments as { labelValue: number; name: string }[] | undefined) ?? []).filter((g) => isTumorName(g.name)).map((g) => g.labelValue);
      if (!labels.length) continue;
      const z = await fetchZarrVolumeNative(live.blobBase(), s.zarr as ZarrDesc);
      const [nx, ny] = s.dims as number[], M = s.ijkToRAS as number[], lab = z.data as ArrayLike<number>;
      for (let v = 0; v < lab.length; v++) if (labels.includes(Number(lab[v]))) { const i = v % nx, j = Math.floor(v / nx) % ny, k = Math.floor(v / (nx * ny)); sx += M[0] * i + M[1] * j + M[2] * k + M[3]; n++; }
    }
    return n ? sx / n : undefined;
  }

  async function openCase(key: string): Promise<void> {
    const c = cases.find((x) => x.dwi.seriesInstanceUID === key);
    if (busy || !c) return;
    if (hasCase()) keepLevels();
    // ONE CASE ON SCREEN (critic 2026-10-06, findings 3 and 9): whatever is in the scene is closed first -- every time, the
    // first case too; an empty scene is a closed one; "keep" (something unsaved, the person said so) stops here.
    if (sceneHasData() && !(await closeScene()) && sceneHasData()) return;
    draw([]); openKey = key; openTracts = c.tracts; placed = undefined; levelsSaid = ""; openPatient = c.patient; openRules = undefined; openFibers = "";
    busy = "Opening…"; render();
    try {
      await refreshFile();
      say(`Reading the fiber tracts of ${c.patient}…`);
      const files = await seriesDicomFiles(c.tracts);
      if (!files?.length) throw new Error("the stored fiber tracts are not in the open database");
      const t = await dicomToTracts(new Uint8Array(files[0]));
      const inputs = (t.provenance?.inputs ?? {}) as { t1?: string | null };
      const rules = t.provenance?.rules as Record<string, unknown> | undefined;
      openRules = rules;
      const db = await databaseSeries();
      const t1 = inputs.t1 ?? undefined, t1Entry = db?.series.find((s) => s.seriesInstanceUID === t1);
      // THE TUMOR OUTLINE (critic 2026-10-06, finding 6): a SEG of this patient -- the diffusion scan's study, the T1's,
      // or another of the patient's -- named as a tumor by the Diffusion module's rule; its segments are checked by name
      // once loaded. When the description says nothing, the SEGs of the scan's and the T1's studies are tried.
      const patientSegs = (db?.series ?? []).filter((s) => s.modality === "SEG" && (s.patientID ?? s.patientName) === (c.dwi.patientID ?? c.dwi.patientName));
      // The scan's own study (and the T1's) first: a patient with a later, post-operative study must not have that outline
      // decide the side (critic 2026-10-06, R2-7); the patient's other studies only when these have none.
      const sameStudy = (s: DatabaseSeries) => s.studyInstanceUID === c.dwi.studyInstanceUID || s.studyInstanceUID === t1Entry?.studyInstanceUID;
      const named = patientSegs.filter((s) => isTumorName(s.description ?? ""));
      const namedHere = named.filter(sameStudy);
      const segs = (namedHere.length ? namedHere : named.length ? named : patientSegs.filter(sameStudy)).map((s) => s.seriesInstanceUID);
      say(`Loading ${c.patient}'s MRI of the anatomy${segs.length ? " and the tumor outline" : ""}…`);
      const want = [...(t1 ? [t1] : []), ...segs];
      const r = want.length ? await loadDatabaseSeries(want, (l) => { note = l; render(); }) : { loaded: 0, failures: ["no MRI of the anatomy is recorded with the tracts"] };
      const nodeOf = (uid: string) => [...live.nodes.values()].find((n) => (n.origin as Record<string, unknown> | undefined)?.seriesInstanceUID === uid);
      const t1Node = t1 ? nodeOf(t1) : undefined;
      const segNodes = [...live.nodes.values()].filter((n) => n.type === "segmentation" && segs.includes(String((n.dicom as { seriesInstanceUID?: string } | undefined)?.seriesInstanceUID ?? (n.origin as Record<string, unknown> | undefined)?.seriesInstanceUID)));
      const mapId = await loadColorFA(c.tracts, `${c.patient} Color FA (made at import)`);
      // The b = 0 image (T2-weighted: the substantia nigra and the red nucleus dark, Ron 2026-10-06), for a view's gear ›
      // Image; tracts made before it was stored have none.
      const b0Id = await loadStored(b0Path(c.tracts), `${c.patient} b=0 (made at import)`, false);
      for (const cmp of [...live.nodes.values()].filter((n) => n.type === "sliceComposite")) {
        if (t1Node) live.write({ op: "patch", id: cmp.id, path: "#/refs/background", value: [t1Node.id] });
        live.write({ op: "patch", id: cmp.id, path: "#/refs/foreground", value: mapId ? [mapId] : [] });
        live.write({ op: "patch", id: cmp.id, path: "#/foregroundOpacity", value: 0.5 });
      }
      const { left, right } = cstOf(t.sets), mid = midlineX(left, right);
      openCst = { left, right };
      const tumorX = await tumorCenterX(segNodes.map((n) => n.id));
      const side = sideToJudge(tumorX, mid);
      const sl = side < 0 ? left : right;
      openSide = side < 0 ? "left" : "right"; openFibers = fibersFingerprint(sl);
      const had = file.cases[key];
      const pair = levelsFromPair(left, right, side), fan = pair ? undefined : levelsOf(sl);
      const levels = had?.levels ?? pair ?? fan;
      const bothThere = left.length >= 5 && right.length >= 5;
      levelsSaid = had?.levels ? "the levels you left last time" : pair ? "" : fan ? `levels placed from this tract alone (${bothThere ? "the two tracts never got far enough apart to place them" : "the other side's is missing"}): check them` : "the levels could not be found from the tracts: please place them";
      const lv = levels ?? (() => { const zs = sl.flatMap((f) => [...f].filter((_, i) => i % 3 === 2)).sort((a, b) => a - b); const m = zs.length ? zs[zs.length >> 1] : 0; return { crus: Math.round(m - 10), ic: Math.round(m + 5), coronal: 0 }; })();
      // A remake that drew the very same fibers keeps the verdict given on the older tracts (critic 2026-10-06, R2-4).
      const carry = carriedVerdict(had, c.tracts, openFibers);
      if (carry && canWrite()) await saveCase(key, patchOf({ verdict: { verdict: carry.j.verdict!, judgedAt: carry.j.judgedAt ?? new Date().toISOString(), carriedFrom: carry.from }, ...(carry.j.note ? { note: carry.j.note } : {}) }));
      setLayout(LAYOUT.conventionalWidescreen);
      orientView("Red", "axial"); orientView("Yellow", "axial"); orientView("Green", "coronal");
      setSliceOffset("Red", lv.crus); setSliceOffset("Yellow", lv.ic); setSliceOffset("Green", lv.coronal);
      placed = { crus: +lv.crus.toFixed(1), ic: +lv.ic.toFixed(1), coronal: +lv.coronal.toFixed(1) };
      draw(sl);
      lookFrom3D("A");
      const why = tumorX === undefined ? (patientSegs.length ? "no outline named as a tumor was found for this patient, so the left is shown" : "no tumor outline for this patient, so the left is shown") : `the tumor is on the ${side < 0 ? "right" : "left"}`;
      say([`${c.patient}: the ${side < 0 ? "left" : "right"} corticospinal tract (${why}), ${sl.length.toLocaleString()} fibers`, ...(mapId ? [] : ["the direction-colored map was not stored with these tracts"]), ...(b0Id ? [] : ["no b = 0 image was stored with these tracts"]), ...(levelsSaid ? [levelsSaid] : []), ...(r.failures.length ? [`not everything loaded: ${r.failures[0]}`] : [])].join("; ") + ".");
    } catch (e) {
      say(`${c.patient} could not be opened: ${(e as Error).message}`);
    } finally { busy = ""; render(); }
  }

  /** THE CRUS BORDER (Ron, 2026-10-06): a curve he places along the border between one crus and the substantia nigra on
   *  the peduncle slice; Done takes its points into the verdicts file and removes the markup (the line is drawn by the
   *  review from then on). */
  function startBorder(side: "left" | "right"): void {
    if (!openKey || !canWrite()) return;
    drawing = side;
    if (!startPlacing("curve", false)) { drawing = undefined; say("Drawing is not available in this app."); return; }
    say(`Click points along the border between the ${side} crus and the substantia nigra in the red view, from one end to the other; then Done.`);
  }
  async function finishBorder(keep: boolean): Promise<void> {
    const side = drawing, id = placingMarkupId();
    drawing = undefined; endPlacing();
    const node = id ? live.nodes.get(id) : undefined;
    const points = ((node?.controlPoints as { position: [number, number, number] }[] | undefined) ?? []).map((c) => c.position);
    if (id) live.write({ op: "del", id });
    if (keep && side && points.length >= 2) {
      const z = sliceOffset("Red") ?? points[0][2];
      await saveCase(openKey, patchOf({ crusBorder: { side, points, z: +z.toFixed(1), drawnAt: new Date().toISOString() } }));
      drawDots();
      say(`The ${side} crus border is saved with the case.`);
    } else if (keep) say("The border needs at least two points; nothing was saved.");
    render();
  }
  /** How many of a side's crossings at the border's level lie dorsal to it. */
  function borderCount(side: "left" | "right", b: CrusBorder): { dorsal: number; of: number } {
    const cs = sliceCrossings([openCst[side]], { origin: [0, 0, b.z], normal: [0, 0, 1] }).map((c) => [c.p[0], c.p[1]] as [number, number]);
    return { dorsal: dorsalTo(b.points.map((p) => [p[0], p[1]] as [number, number]), cs), of: cs.length };
  }

  async function judge(verdict: NonNullable<Judgment["verdict"]>): Promise<void> {
    if (!openKey || !openTracts || !hasCase() || !canWrite()) return;
    keepLevels();
    await saveCase(openKey, patchOf({ verdict: { verdict, judgedAt: new Date().toISOString() } }));
    render();
  }

  function render(): void {
    if (!root) return;
    root.replaceChildren();
    const verdictOf = (c: Case) => file.cases[c.dwi.seriesInstanceUID]?.judgments?.[c.tracts]?.verdict;
    const judged = cases.filter((c) => verdictOf(c)).length;
    const list = shell.section(root, "1 · Cases", { band: "yellow", open: true, note: cases.length ? `${judged} of ${cases.length} judged` : "" });
    if (!cases.length) { const p = document.createElement("p"); p.className = "sl-hint"; p.textContent = "No fiber tracts made at import in the open database. They are made for each diffusion scan with an MRI of the anatomy when the import job runs."; list.append(p); }
    const table = document.createElement("table"); table.style.cssText = "width:100%;border-collapse:collapse;font-size:11px";
    // The list scrolls in its own box, so the verdict buttons stay in sight with many cases (critic 2026-10-06, R2-7).
    const box = document.createElement("div"); box.style.cssText = "max-height:30vh;overflow-y:auto"; box.append(table);
    for (const c of cases) {
      const key = c.dwi.seriesInstanceUID, r = file.cases[key], v = verdictOf(c), older = r && Object.entries(r.judgments ?? {}).some(([s, j]) => s !== c.tracts && j.verdict);
      const tr = document.createElement("tr");
      tr.style.cssText = `cursor:pointer;${key === openKey ? "background:rgba(248,215,100,.14)" : ""}`;
      if (key === openKey) tr.dataset.open = "1";
      tr.innerHTML = `<td style="padding:2px 4px">${c.patient.replace(/[<&]/g, "")}</td><td style="padding:2px 4px;color:var(--sl-fg-muted)">${r?.side ?? (key === openKey ? openSide : "")}</td><td style="padding:2px 4px;color:${v === "acceptable" ? "var(--sl-ok)" : v ? "var(--sl-error)" : "var(--sl-fg-muted)"}">${v === "acceptable" ? "✓ acceptable" : v ? "✗ not acceptable" : older ? "— (judged on older tracts)" : "—"}</td>`;
      tr.title = `Open ${c.patient}: the T1, the direction-colored map and the corticospinal tract on the side without the tumor.`;
      tr.onclick = () => { void openCase(key); };
      table.append(tr);
    }
    list.append(box);
    box.querySelector<HTMLElement>("tr[data-open]")?.scrollIntoView({ block: "nearest" });
    const i = currentIndex(), c = cases[i], r = c && file.cases[c.dwi.seriesInstanceUID], j = r?.judgments?.[c.tracts];
    if (c && openPatient && hasCase()) {
      const here = shell.section(root, `2 · This case`, { band: "green", open: true, note: c.patient });
      const p = document.createElement("p"); p.className = "sl-hint";
      p.textContent = `The ${openSide} corticospinal tract. Red view: axial at the cerebral peduncle; yellow: axial at the internal capsule; green: coronal through the tract; 3D from the front. Move a slider when a level is off: the level you leave is kept for next time.${levelsSaid ? ` (${levelsSaid[0].toUpperCase()}${levelsSaid.slice(1)}.)` : ""}`;
      here.append(p);
      const v = shell.section(root, "3 · Verdict", { band: "yellow", open: true });
      if (fileProblem) {
        const warn = document.createElement("p"); warn.className = "sl-hint"; warn.style.color = "var(--sl-error)"; warn.textContent = fileProblem; v.append(warn);
        if (fileState !== "unreadable") { const again = document.createElement("button"); again.textContent = "Try again"; again.title = "Read the verdicts again."; again.onclick = () => { void tryAgain(); }; v.append(again); }
      }
      if (j?.carriedFrom) { const cf = document.createElement("p"); cf.className = "sl-hint"; cf.textContent = "This verdict was given on an earlier making of the tracts that drew exactly these fibers."; v.append(cf); }
      const crit = document.createElement("p"); crit.className = "sl-hint"; crit.textContent = "Judge by: at the internal capsule, the tract in the posterior limb (blue), nothing in the thalamus or the lentiform nucleus; at the peduncle, in the crus, ventral to the substantia nigra; how complete the fan is, seen from the front in 3D. The outline on a slice is where the tract crosses it; on the two axial slices, a small circle on its own is a stray fiber."; v.append(crit);
      const row = document.createElement("div"); row.style.cssText = "display:flex;gap:6px;margin:6px 0";
      const ok = document.createElement("button"), no = document.createElement("button");
      ok.textContent = "✓ Acceptable"; no.textContent = "✗ Not acceptable";
      ok.style.cssText = `flex:1;${j?.verdict === "acceptable" ? "background:var(--sl-ok);color:#000" : "color:var(--sl-ok);border-color:var(--sl-ok)"}`;
      no.style.cssText = `flex:1;${j?.verdict === "not acceptable" ? "background:var(--sl-error);color:#000" : "color:var(--sl-error);border-color:var(--sl-error)"}`;
      ok.title = "Few errant fibers on the two axial slices, and a complete fan from the front."; no.title = "Too many errant fibers, or the fan incomplete.";
      ok.disabled = no.disabled = !!busy || !canWrite();
      ok.onclick = () => { void judge("acceptable"); }; no.onclick = () => { void judge("not acceptable"); };
      row.append(ok, no); v.append(row);
      const ta = document.createElement("textarea"); ta.placeholder = "Note (optional)"; ta.rows = 2; ta.value = j?.note ?? ""; ta.style.cssText = "width:100%;box-sizing:border-box";
      ta.disabled = !canWrite();
      ta.onchange = () => { void saveCase(c.dwi.seriesInstanceUID, patchOf({ note: ta.value.trim() })).then(render); };
      v.append(ta);
      const nav = document.createElement("div"); nav.style.cssText = "display:flex;gap:6px;align-items:center;margin-top:8px";
      const prev = document.createElement("button"), next = document.createElement("button"), pos = document.createElement("span");
      prev.textContent = "◀ Previous"; next.textContent = "Next ▶"; next.className = "sl-primary"; pos.className = "sl-hint";
      prev.disabled = !!busy || i <= 0; next.disabled = !!busy || i >= cases.length - 1;
      prev.onclick = () => { void openCase(cases[i - 1].dwi.seriesInstanceUID); }; next.onclick = () => { void openCase(cases[i + 1].dwi.seriesInstanceUID); };
      pos.textContent = `${i + 1} of ${cases.length}`;
      nav.append(prev, next, pos); v.append(nav);
      const cb = shell.section(root, "4 · Crus border", { band: "green", open: true });
      const hint = document.createElement("p"); hint.className = "sl-hint";
      hint.textContent = "Optional: draw the border between each crus and the substantia nigra on the red view; the fibers of that side dorsal to your line are counted.";
      cb.append(hint);
      for (const sd of ["left", "right"] as const) {
        const b = r?.crusBorder?.[sd], line = document.createElement("div"); line.style.cssText = "display:flex;gap:6px;align-items:center;margin:4px 0";
        const btn = document.createElement("button");
        btn.textContent = drawing === sd ? "Done" : b ? `Redraw the ${sd} border` : `Draw the ${sd} border`;
        btn.title = drawing === sd ? "Take the line you placed." : `Place points along the border between the ${sd} crus and the substantia nigra in the red view.`;
        btn.disabled = !!busy || !canWrite() || (!!drawing && drawing !== sd);
        btn.onclick = () => { if (drawing === sd) void finishBorder(true); else startBorder(sd); };
        line.append(btn);
        if (drawing === sd) { const cancel = document.createElement("button"); cancel.textContent = "Cancel"; cancel.onclick = () => { void finishBorder(false); }; line.append(cancel); }
        const info = document.createElement("span"); info.className = "sl-hint";
        if (b) { const n = borderCount(sd, b); info.textContent = `${n.dorsal} of ${n.of} crossings dorsal (S ${b.z} mm)`; }
        line.append(info); cb.append(line);
      }
    }
    if (busy || note) { const s = document.createElement("p"); s.className = "sl-hint"; s.textContent = busy && !note ? busy : note; root.append(s); }
  }

  shell.registerPanel({
    id: "tract-review",
    title: "Tract review",
    groups: ["Display"],
    tip: "Judge the corticospinal tract on the side without a tumor, case after case, for checking the fiber tracts.",
    help: "<p><b>For checking the fiber tracts</b> against an expert's eye. Each case is a diffusion scan whose fiber tracts were made when it was imported. Click a case: its MRI of the anatomy is shown with the direction-colored map over it (red left-right, green front-back, blue up-down), and only the corticospinal tract on the side without the tumor, in one color. The red view is an axial slice at the cerebral peduncle, the yellow one an axial slice at the internal capsule, the green one coronal through the tract; the 3D view is seen from the front. The slice levels are found from the two tracts; move a slider when one is off, and the level you leave is used next time. The outline on a slice is where the tract crosses it, with the map visible inside; on the two axial slices a small circle on its own is a stray fiber (the coronal slice runs along the tract, so there it shows as many small circles). Judge by where the tract lies -- at the internal capsule in the posterior limb (blue on the map), nothing in the thalamus or the lentiform nucleus; at the peduncle in the crus, ventral to the substantia nigra -- and how complete the fan is in 3D: <b>Acceptable</b> or <b>Not acceptable</b>, with a note if you like; <b>Next</b> opens the next case. The verdicts are kept in the database's folder (tract-review.json), each with the version of the tracts it was about; when the tracts are made again, the earlier verdicts stay, and the case waits for a new one.</p>",
    async mount(el: HTMLElement) { root = el; await refreshFile(); await listCases(); say(`${cases.length} case${cases.length === 1 ? "" : "s"} with fiber tracts made at import.${fileProblem ? ` ${fileProblem}` : ""}`); },
    // Shown again (perhaps after the window's database changed): the verdicts and the list read again.
    onShow() { void refreshFile().then(listCases).then(render); },
  });
}

queueModule(registerTractReview);

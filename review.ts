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
  sliceOffset, sliceOrientation, writeDatabaseFile, type DatabaseSeries, type ModuleContext, type ZarrDesc,
} from "albula";
import { TUMOR } from "./face.ts";
import { colorFaPath, dicomToTracts, type TractSetData } from "./tracts-dicom.ts";
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
 * 5-8 mm from the midline), about 12-17 mm from it through the cerebral peduncles and 22-25 mm through the posterior limb
 * of the internal capsule. So: half the distance between the two tracts' centers per millimeter of height, smoothed over
 * 5 mm; the peduncle where it first reaches 13 mm going up, the internal capsule where it first reaches 22 mm above that
 * (on the 59 stored tracts objects of the test cases: 10-25 mm apart, typically 15); the coronal slice through the judged
 * tract at the internal capsule. Undefined when either is not reached.
 */
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
  const c = first(lo, 13);
  if (c < 0) return undefined;
  const k = first(at[c].z + 5, 22);
  if (k < 0) return undefined;
  return { crus: at[c].z, ic: at[k].z, coronal: Number.isFinite(at[k].y) ? +at[k].y.toFixed(1) : 0 };
}

/** One case's verdict on one version of its tracts (Ron's "6": a remake of the tracts keeps the verdicts on the old ones). */
export interface Judgment { verdict?: "acceptable" | "not acceptable"; note?: string; judgedAt?: string; rules?: Record<string, unknown> }
/** One case's record in tract-review.json (version 2): the side, the levels Ron left (kept across remakes of the tracts:
 *  they are anatomy), and a judgment per tracts series. */
export interface Review {
  patient: string; side: "left" | "right";
  /** The levels as Ron left them (only when he moved them; "you record for next time"). */
  levels?: Levels;
  /** Per tracts series: the verdict, the note, when, and the rules the tracts were made under. */
  judgments: Record<string, Judgment>;
}
export interface ReviewFile { version: 2; cases: Record<string, Review> }

/** Merge one case's record into the file as it is on disk now (read just before writing: another window's verdicts on
 *  other cases are kept; critic 2026-10-06, finding 18). */
export function mergeCase(onDisk: ReviewFile, key: string, mine: Review): ReviewFile {
  const had = onDisk.cases[key];
  const judgments = { ...(had?.judgments ?? {}), ...mine.judgments };
  return { version: 2, cases: { ...onDisk.cases, [key]: { ...mine, judgments } } };
}

// ── The module ──────────────────────────────────────────────────────────────────────────────────────────────────────

interface Case { dwi: DatabaseSeries; tracts: string; patient: string }

function registerTractReview(ctx: ModuleContext): void {
  const { shell, live, store, device } = ctx;
  let root: HTMLElement | undefined, cases: Case[] = [], file: ReviewFile = { version: 2, cases: {} };
  /** Whether the verdicts file can be written: it was read, or there is none yet. Never over a file that did not read
   *  (critic 2026-10-06, finding 2). */
  let fileState: "unknown" | "ok" | "absent" | "unreadable" = "unknown";
  /** The case open, by its diffusion scan's UID (critic 2026-10-06, finding 8: a row number moves when the list does). */
  let openKey = "", openTracts = "", busy = "", note = "", field: FiberField | undefined, drawn: Float32Array[] = [];
  let placed: Levels | undefined, levelsSaid = "";
  const say = (s: string) => { note = s; ctx.status(s); render(); };
  const currentIndex = () => cases.findIndex((c) => c.dwi.seriesInstanceUID === openKey);

  async function readFile(): Promise<{ state: "ok" | "absent" | "unreadable"; file?: ReviewFile }> {
    const url = await databaseFileUrl(`SlicerAlbula-SEG/${REVIEWS}`);
    if (!url) return { state: "unreadable" };
    const r = await fetch(url, { cache: "no-store" }).catch(() => null);
    if (!r) return { state: "unreadable" };
    if (r.status === 404) return { state: "absent" };
    if (!r.ok) return { state: "unreadable" };
    try { const j = await r.json() as { version?: number; cases?: Record<string, unknown> }; if (j?.version === 2 && j.cases) return { state: "ok", file: j as ReviewFile }; } catch { /* below */ }
    return { state: "unreadable" };
  }
  async function readReviews(): Promise<void> {
    const r = await readFile();
    fileState = r.state;
    if (r.file) file = r.file;
    if (r.state === "unreadable") say(`The verdicts file (${REVIEWS}, in the database's SlicerAlbula-SEG folder) could not be read: nothing will be saved until it is moved aside or repaired, so no verdict in it is lost.`);
  }
  /** Save ONE case, merged into the file as it is on disk now; never over a file that does not read. */
  async function saveCase(key: string): Promise<boolean> {
    const r = await readFile();
    if (r.state === "unreadable") { fileState = "unreadable"; say(`The verdicts file could not be read, so this was not saved (nothing in it was overwritten).`); return false; }
    const merged = mergeCase(r.file ?? { version: 2, cases: {} }, key, file.cases[key]);
    const w = await writeDatabaseFile(REVIEWS, JSON.stringify(merged, null, 2) + "\n");
    if (!w.ok) { say(`The verdict could not be saved: ${w.why}.`); return false; }
    file = merged; fileState = "ok";
    return true;
  }

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
    const r = file.cases[openKey];
    if (!r || !placed || !hasCase()) return;
    if (sliceOrientation("Red") !== "Axial" || sliceOrientation("Yellow") !== "Axial" || sliceOrientation("Green") !== "Coronal") return;
    const c = sliceOffset("Red"), k = sliceOffset("Yellow"), y = sliceOffset("Green");
    if (c === undefined || k === undefined || y === undefined) return;
    const now = { crus: +c.toFixed(1), ic: +k.toFixed(1), coronal: +y.toFixed(1) };
    if (now.crus === placed.crus && now.ic === placed.ic && now.coronal === placed.coronal) return;
    r.levels = now; placed = now;
    void saveCase(openKey);
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
   *  what shows the tract in the internal capsule's posterior limb and in front of the substantia nigra). */
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
      view.setOverlay(String(n.layoutName ?? n.name), "tract-review", loops.map((points) => ({ kind: "polyline" as const, points, color: [...FIBER_RGB, 1], widthPx: 1.5, closed: true })));
    }
  }
  live.subscribe((c) => {
    if (c.type === "view" && String(c.id).startsWith("nativeSlice-") && drawn.length) requestAnimationFrame(drawDots);
    // The scene emptied (File › Close Scene): the review's own fibers go with it.
    if (drawn.length && !hasCase()) { draw([]); openKey = ""; render(); }
  });

  async function loadColorFA(tracts: string, name: string): Promise<string | undefined> {
    const url = await databaseFileUrl(colorFaPath(tracts));
    const r = url ? await fetch(url).catch(() => null) : null;
    if (!r?.ok) return undefined;
    const { f, body } = nrrdSplitHeader(new Uint8Array(await r.arrayBuffer()));
    const sizes = (f["sizes"] ?? "").split(/\s+/).filter(Boolean).map(Number) as [number, number, number];
    const raw = await nrrdDecode(f, body, sizes[0] * sizes[1] * sizes[2] * 4);
    const data = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
    const res = await loadVolumeIntoScene(live, store, { dims: sizes, ijkToRAS: nrrdGeometry(f).ijkToRAS, data, dtype: "<f4", name }, { name, extra: { rgb24: true, autoVolumeRendering: false, recomputable: true } });
    return res.imageId;
  }

  /** The tumor's center (RAS x), from the segments named as a tumor (face.ts TUMOR, the Diffusion module's own rule). */
  async function tumorCenterX(segIds: string[]): Promise<number | undefined> {
    let sx = 0, n = 0;
    for (const id of segIds) {
      const s = live.nodes.get(id); if (!s?.zarr) continue;
      const labels = ((s.segments as { labelValue: number; name: string }[] | undefined) ?? []).filter((g) => TUMOR.test(g.name)).map((g) => g.labelValue);
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
    draw([]); openKey = key; openTracts = c.tracts; placed = undefined; levelsSaid = "";
    busy = "Opening…"; render();
    try {
      say(`Reading the fiber tracts of ${c.patient}…`);
      const files = await seriesDicomFiles(c.tracts);
      if (!files?.length) throw new Error("the stored fiber tracts are not in the open database");
      const t = await dicomToTracts(new Uint8Array(files[0]));
      const inputs = (t.provenance?.inputs ?? {}) as { t1?: string | null };
      const rules = t.provenance?.rules as Record<string, unknown> | undefined;
      const db = await databaseSeries();
      const t1 = inputs.t1 ?? undefined, t1Entry = db?.series.find((s) => s.seriesInstanceUID === t1);
      // THE TUMOR OUTLINE (critic 2026-10-06, finding 6): a SEG of this patient -- the diffusion scan's study, the T1's,
      // or another of the patient's -- named as a tumor by the Diffusion module's rule; its segments are checked by name
      // once loaded. When the description says nothing, the SEGs of the scan's and the T1's studies are tried.
      const patientSegs = (db?.series ?? []).filter((s) => s.modality === "SEG" && (s.patientID ?? s.patientName) === (c.dwi.patientID ?? c.dwi.patientName));
      const named = patientSegs.filter((s) => TUMOR.test(s.description ?? ""));
      const segs = (named.length ? named : patientSegs.filter((s) => s.studyInstanceUID === c.dwi.studyInstanceUID || s.studyInstanceUID === t1Entry?.studyInstanceUID)).map((s) => s.seriesInstanceUID);
      say(`Loading ${c.patient}'s MRI of the anatomy${segs.length ? " and the tumor outline" : ""}…`);
      const want = [...(t1 ? [t1] : []), ...segs];
      const r = want.length ? await loadDatabaseSeries(want, (l) => { note = l; render(); }) : { loaded: 0, failures: ["no MRI of the anatomy is recorded with the tracts"] };
      const nodeOf = (uid: string) => [...live.nodes.values()].find((n) => (n.origin as Record<string, unknown> | undefined)?.seriesInstanceUID === uid);
      const t1Node = t1 ? nodeOf(t1) : undefined;
      const segNodes = [...live.nodes.values()].filter((n) => n.type === "segmentation" && segs.includes(String((n.dicom as { seriesInstanceUID?: string } | undefined)?.seriesInstanceUID ?? (n.origin as Record<string, unknown> | undefined)?.seriesInstanceUID)));
      const mapId = await loadColorFA(c.tracts, `${c.patient} Color FA (made at import)`);
      for (const cmp of [...live.nodes.values()].filter((n) => n.type === "sliceComposite")) {
        if (t1Node) live.write({ op: "patch", id: cmp.id, path: "#/refs/background", value: [t1Node.id] });
        live.write({ op: "patch", id: cmp.id, path: "#/refs/foreground", value: mapId ? [mapId] : [] });
        live.write({ op: "patch", id: cmp.id, path: "#/foregroundOpacity", value: 0.5 });
      }
      const { left, right } = cstOf(t.sets), mid = midlineX(left, right);
      const tumorX = await tumorCenterX(segNodes.map((n) => n.id));
      const side = sideToJudge(tumorX, mid);
      const sl = side < 0 ? left : right;
      const had = file.cases[key];
      const pair = levelsFromPair(left, right, side), fan = pair ? undefined : levelsOf(sl);
      const levels = had?.levels ?? pair ?? fan;
      levelsSaid = had?.levels ? "the levels you left last time" : pair ? "" : fan ? "levels placed from this tract alone (the other side's is missing): check them" : "the levels could not be found from the tracts: please place them";
      const lv = levels ?? (() => { const zs = sl.flatMap((f) => [...f].filter((_, i) => i % 3 === 2)).sort((a, b) => a - b); const m = zs.length ? zs[zs.length >> 1] : 0; return { crus: Math.round(m - 10), ic: Math.round(m + 5), coronal: 0 }; })();
      file.cases[key] = { patient: c.patient, side: side < 0 ? "left" : "right", ...(had?.levels ? { levels: had.levels } : {}), judgments: { ...(had?.judgments ?? {}), [c.tracts]: { ...(had?.judgments?.[c.tracts] ?? {}), ...(rules ? { rules } : {}) } } };
      setLayout(LAYOUT.conventionalWidescreen);
      orientView("Red", "axial"); orientView("Yellow", "axial"); orientView("Green", "coronal");
      setSliceOffset("Red", lv.crus); setSliceOffset("Yellow", lv.ic); setSliceOffset("Green", lv.coronal);
      placed = { crus: +lv.crus.toFixed(1), ic: +lv.ic.toFixed(1), coronal: +lv.coronal.toFixed(1) };
      draw(sl);
      lookFrom3D("A");
      const why = tumorX === undefined ? (patientSegs.length ? "no outline named as a tumor was found for this patient, so the left is shown" : "no tumor outline for this patient, so the left is shown") : `the tumor is on the ${side < 0 ? "right" : "left"}`;
      say([`${c.patient}: the ${side < 0 ? "left" : "right"} corticospinal tract (${why}), ${sl.length.toLocaleString()} fibers`, ...(mapId ? [] : ["the direction-colored map was not stored with these tracts"]), ...(levelsSaid ? [levelsSaid] : []), ...(r.failures.length ? [`not everything loaded: ${r.failures[0]}`] : [])].join("; ") + ".");
    } catch (e) {
      say(`${c.patient} could not be opened: ${(e as Error).message}`);
    } finally { busy = ""; render(); }
  }

  async function judge(verdict: Judgment["verdict"]): Promise<void> {
    const r = file.cases[openKey]; if (!r || !openTracts || !hasCase()) return;
    const j = r.judgments[openTracts] ?? (r.judgments[openTracts] = {});
    j.verdict = verdict; j.judgedAt = new Date().toISOString();
    keepLevels();
    await saveCase(openKey);
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
    for (const c of cases) {
      const key = c.dwi.seriesInstanceUID, r = file.cases[key], v = verdictOf(c), older = r && Object.entries(r.judgments ?? {}).some(([s, j]) => s !== c.tracts && j.verdict);
      const tr = document.createElement("tr");
      tr.style.cssText = `cursor:pointer;${key === openKey ? "background:rgba(248,215,100,.14)" : ""}`;
      tr.innerHTML = `<td style="padding:2px 4px">${c.patient.replace(/[<&]/g, "")}</td><td style="padding:2px 4px;color:var(--sl-fg-muted)">${r?.side ?? ""}</td><td style="padding:2px 4px;color:${v === "acceptable" ? "var(--sl-ok)" : v ? "var(--sl-error)" : "var(--sl-fg-muted)"}">${v === "acceptable" ? "✓ acceptable" : v ? "✗ not acceptable" : older ? "— (judged on older tracts)" : "—"}</td>`;
      tr.title = `Open ${c.patient}: the T1, the direction-colored map and the corticospinal tract on the side without the tumor.`;
      tr.onclick = () => { void openCase(key); };
      table.append(tr);
    }
    list.append(table);
    const i = currentIndex(), c = cases[i], r = c && file.cases[c.dwi.seriesInstanceUID], j = r?.judgments?.[c.tracts];
    if (c && r && hasCase()) {
      const here = shell.section(root, `2 · This case`, { band: "green", open: true, note: c.patient });
      const p = document.createElement("p"); p.className = "sl-hint";
      p.textContent = `The ${r.side} corticospinal tract. Red view: axial at the cerebral peduncle; yellow: axial at the internal capsule; green: coronal through the tract; 3D from the front. Move a slider when a level is off: the level you leave is kept for next time.${levelsSaid ? ` (${levelsSaid[0].toUpperCase()}${levelsSaid.slice(1)}.)` : ""}`;
      here.append(p);
      const v = shell.section(root, "3 · Verdict", { band: "yellow", open: true });
      const crit = document.createElement("p"); crit.className = "sl-hint"; crit.textContent = "Judge by: at the internal capsule, the tract in the posterior limb (blue), nothing in the thalamus or the lentiform nucleus; at the peduncle, in the crus, in front of the substantia nigra; how complete the fan is, seen from the front in 3D. The outline on a slice is where the tract crosses it; a small circle is a stray fiber."; v.append(crit);
      const row = document.createElement("div"); row.style.cssText = "display:flex;gap:6px;margin:6px 0";
      const ok = document.createElement("button"), no = document.createElement("button");
      ok.textContent = "✓ Acceptable"; no.textContent = "✗ Not acceptable";
      ok.style.cssText = `flex:1;${j?.verdict === "acceptable" ? "background:var(--sl-ok);color:#000" : "color:var(--sl-ok);border-color:var(--sl-ok)"}`;
      no.style.cssText = `flex:1;${j?.verdict === "not acceptable" ? "background:var(--sl-error);color:#000" : "color:var(--sl-error);border-color:var(--sl-error)"}`;
      ok.title = "Few errant fibers on the two axial slices, and a complete fan from the front."; no.title = "Too many errant fibers, or the fan incomplete.";
      ok.disabled = no.disabled = !!busy || fileState === "unreadable";
      ok.onclick = () => { void judge("acceptable"); }; no.onclick = () => { void judge("not acceptable"); };
      row.append(ok, no); v.append(row);
      const ta = document.createElement("textarea"); ta.placeholder = "Note (optional)"; ta.rows = 2; ta.value = j?.note ?? ""; ta.style.cssText = "width:100%;box-sizing:border-box";
      ta.disabled = fileState === "unreadable";
      ta.onchange = () => { const jj = r.judgments[c.tracts] ?? (r.judgments[c.tracts] = {}); jj.note = ta.value.trim() || undefined; void saveCase(c.dwi.seriesInstanceUID); };
      v.append(ta);
      const nav = document.createElement("div"); nav.style.cssText = "display:flex;gap:6px;align-items:center;margin-top:8px";
      const prev = document.createElement("button"), next = document.createElement("button"), pos = document.createElement("span");
      prev.textContent = "◀ Previous"; next.textContent = "Next ▶"; next.className = "sl-primary"; pos.className = "sl-hint";
      prev.disabled = !!busy || i <= 0; next.disabled = !!busy || i >= cases.length - 1;
      prev.onclick = () => { void openCase(cases[i - 1].dwi.seriesInstanceUID); }; next.onclick = () => { void openCase(cases[i + 1].dwi.seriesInstanceUID); };
      pos.textContent = `${i + 1} of ${cases.length}`;
      nav.append(prev, next, pos); v.append(nav);
    }
    if (busy || note) { const s = document.createElement("p"); s.className = "sl-hint"; s.textContent = busy && !note ? busy : note; root.append(s); }
  }

  shell.registerPanel({
    id: "tract-review",
    title: "Tract review",
    groups: ["Display"],
    tip: "Judge the corticospinal tract on the side without a tumor, case after case, for checking the fiber tracts.",
    help: "<p><b>For checking the fiber tracts</b> against an expert's eye. Each case is a diffusion scan whose fiber tracts were made when it was imported. Click a case: its MRI of the anatomy is shown with the direction-colored map over it (red left-right, green front-back, blue up-down), and only the corticospinal tract on the side without the tumor, in one color. The red view is an axial slice at the cerebral peduncle, the yellow one an axial slice at the internal capsule, the green one coronal through the tract; the 3D view is seen from the front. The slice levels are found from the two tracts; move a slider when one is off, and the level you leave is used next time. The outline on a slice is where the tract crosses it, with the map visible inside; a small circle on its own is a stray fiber. Judge by where the tract lies -- at the internal capsule in the posterior limb (blue on the map), nothing in the thalamus or the lentiform nucleus; at the peduncle in the crus, in front of the substantia nigra -- and how complete the fan is in 3D: <b>Acceptable</b> or <b>Not acceptable</b>, with a note if you like; <b>Next</b> opens the next case. The verdicts are kept in the database's folder (tract-review.json), each with the version of the tracts it was about; when the tracts are made again, the earlier verdicts stay, and the case waits for a new one.</p>",
    async mount(el: HTMLElement) { root = el; await readReviews(); await listCases(); say(`${cases.length} case${cases.length === 1 ? "" : "s"} with fiber tracts made at import.`); },
    onShow() { void listCases().then(render); },
  });
}

queueModule(registerTractReview);

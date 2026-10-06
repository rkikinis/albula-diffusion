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
  sliceOffset, writeDatabaseFile, type DatabaseSeries, type ModuleContext, type ZarrDesc,
} from "albula";
import { colorFaPath, dicomToTracts, type TractSetData } from "./tracts-dicom.ts";
import { sliceCrossings } from "./tract-slice.ts";

export const CST = "corticospinal tract";
/** The fibers' one color (the mockup's yellow) and their tube radius in 3D (mm). */
const FIBER_RGB: [number, number, number] = [1, 0.83, 0.3], RADIUS = 0.35;
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

/** One case's record in tract-review.json. */
export interface Review {
  patient: string; side: "left" | "right";
  verdict?: "acceptable" | "not acceptable"; note?: string; judgedAt?: string;
  /** The levels as last left (Ron's, once he moved them; "you record for next time"). */
  levels?: Levels;
  /** The tracts judged: their series, and the rules they were made under (a remake of the tracts makes a new series). */
  tracts: { series: string; rules?: Record<string, unknown> };
}
export interface ReviewFile { version: 1; cases: Record<string, Review> }

// ── The module ──────────────────────────────────────────────────────────────────────────────────────────────────────

interface Case { dwi: DatabaseSeries; tracts: string; patient: string }

function registerTractReview(ctx: ModuleContext): void {
  const { shell, live, store, device } = ctx;
  let root: HTMLElement | undefined, cases: Case[] = [], file: ReviewFile = { version: 1, cases: {} };
  let current = -1, busy = "", note = "", field: FiberField | undefined, drawn: Float32Array[] = [], open: { side: -1 | 1; levels: Levels } | undefined;
  const say = (s: string) => { note = s; ctx.status(s); render(); };

  async function readReviews(): Promise<void> {
    const url = await databaseFileUrl(`SlicerAlbula-SEG/${REVIEWS}`);
    const r = url ? await fetch(url, { cache: "no-store" }).catch(() => null) : null;
    if (r?.ok) { try { const j = await r.json() as ReviewFile; if (j?.cases) file = j; } catch { /* a broken file is kept on disk; a new one is not written over it until a verdict */ } }
  }
  async function saveReviews(): Promise<void> {
    const w = await writeDatabaseFile(REVIEWS, JSON.stringify(file, null, 2) + "\n");
    if (!w.ok) say(`The verdicts could not be saved: ${w.why}.`);
  }

  async function listCases(): Promise<void> {
    const db = await databaseSeries();
    if (!db) { cases = []; return; }
    const byUid = new Map(db.series.map((s) => [s.seriesInstanceUID, s]));
    cases = db.edges.filter((e) => e.kind === "tracts").map((e) => ({ dwi: byUid.get(e.parent)!, tracts: e.child })).filter((c) => !!c.dwi)
      // The newest tracts of a scan (a remake is a new series, listed with the old until the old is removed).
      .filter((c, i, all) => !all.some((o, j) => j !== i && o.dwi.seriesInstanceUID === c.dwi.seriesInstanceUID && (byUid.get(o.tracts)?.seriesNumber ?? 0) > (byUid.get(c.tracts)?.seriesNumber ?? 0)))
      .map((c) => ({ ...c, patient: String(c.dwi.patientName ?? c.dwi.patientID ?? "?") }))
      .sort((a, b) => a.patient.localeCompare(b.patient));
  }

  /** The levels in the views now (red z, yellow z, green y): what Ron left them at. */
  const levelsNow = (): Levels | undefined => {
    const c = sliceOffset("Red"), k = sliceOffset("Yellow"), y = sliceOffset("Green");
    return c === undefined || k === undefined || y === undefined ? undefined : { crus: +c.toFixed(1), ic: +k.toFixed(1), coronal: +y.toFixed(1) };
  };
  /** Before leaving a case: the levels as left, for next time. */
  function keepLevels(): void {
    const c = cases[current], r = c && file.cases[c.dwi.seriesInstanceUID], l = levelsNow();
    if (r && l && open && (l.crus !== open.levels.crus || l.ic !== open.levels.ic || l.coronal !== open.levels.coronal)) { r.levels = l; void saveReviews(); }
  }

  function draw(sl: Float32Array[]): void {
    const view = live.view; if (!view) return;
    const old = field; field = undefined; drawn = sl;
    if (sl.length) { field = new FiberField(device, sl.map((p) => ({ points: p, bundle: 1 })), { radius: RADIUS, bundleColors: { 1: [...FIBER_RGB, 1] } }); view.setField("tract-review", field); }
    else view.removeField("tract-review");
    old?.destroy?.();
    drawDots();
  }
  /** Where the tract crosses each slice view, as dots (the Diffusion module's way of showing tracts on slices). */
  function drawDots(): void {
    const view = live.view; if (!view?.setOverlay) return;
    for (const n of live.nodes.values()) {
      if (n.type !== "view" || n.kind !== "slice" || !Array.isArray(n.sliceToRAS)) continue;
      const m = n.sliceToRAS as number[], L = Math.hypot(m[2], m[6], m[10]) || 1, nrm: [number, number, number] = [m[2] / L, m[6] / L, m[10] / L];
      const d = typeof n.offset === "number" ? n.offset : m[3] * nrm[0] + m[7] * nrm[1] + m[11] * nrm[2];
      const cs = drawn.length ? sliceCrossings([drawn], { origin: [nrm[0] * d, nrm[1] * d, nrm[2] * d], normal: nrm }) : [];
      view.setOverlay(String(n.layoutName ?? n.name), "tract-review", cs.map((c) => ({ kind: "point" as const, ras: c.p, color: [...FIBER_RGB, 1], radiusPx: 1.6, inPlaneOnly: true })));
    }
  }
  live.subscribe((c) => { if (drawn.length && c.type === "view" && String(c.id).startsWith("nativeSlice-")) requestAnimationFrame(drawDots); });

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

  async function tumorCenterX(segIds: string[]): Promise<number | undefined> {
    let sx = 0, n = 0;
    for (const id of segIds) {
      const s = live.nodes.get(id); if (!s?.zarr) continue;
      const z = await fetchZarrVolumeNative(live.blobBase(), s.zarr as ZarrDesc);
      const [nx, ny] = s.dims as number[], M = s.ijkToRAS as number[], lab = z.data as ArrayLike<number>;
      for (let v = 0; v < lab.length; v++) if (Number(lab[v])) { const i = v % nx, j = Math.floor(v / nx) % ny, k = Math.floor(v / (nx * ny)); sx += M[0] * i + M[1] * j + M[2] * k + M[3]; n++; }
    }
    return n ? sx / n : undefined;
  }

  async function openCase(i: number): Promise<void> {
    if (busy || i < 0 || i >= cases.length) return;
    keepLevels();
    if (current >= 0 && !(await closeScene())) return;            // kept: something unsaved, and Ron said keep
    draw([]); current = i; open = undefined;
    const c = cases[i];
    busy = "Opening…"; render();
    try {
      say(`Reading the fiber tracts of ${c.patient}…`);
      const files = await seriesDicomFiles(c.tracts);
      if (!files?.length) throw new Error("the stored fiber tracts are not in the open database");
      const t = await dicomToTracts(new Uint8Array(files[0]));
      const inputs = (t.provenance?.inputs ?? {}) as { t1?: string | null };
      const rules = t.provenance?.rules as Record<string, unknown> | undefined;
      const db = await databaseSeries();
      const t1 = inputs.t1 ?? undefined;
      // The tumor outline: a SEG of the same study (the T1's, as the outlines are drawn on it).
      const segs = (db?.series ?? []).filter((s) => s.studyInstanceUID === c.dwi.studyInstanceUID && s.modality === "SEG" && /tumou?r/i.test(s.description ?? "")).map((s) => s.seriesInstanceUID);
      say(`Loading ${c.patient}'s MRI of the anatomy${segs.length ? " and the tumor outline" : ""}…`);
      const want = [...(t1 ? [t1] : []), ...segs];
      const r = want.length ? await loadDatabaseSeries(want, (l) => { note = l; render(); }) : { loaded: 0, failures: ["no MRI of the anatomy is recorded with the tracts"] };
      if (r.failures.length) say(`Not everything loaded: ${r.failures[0]}`);
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
      const side = sideToJudge(await tumorCenterX(segNodes.map((n) => n.id)), mid);
      const sl = side < 0 ? left : right;
      const key = c.dwi.seriesInstanceUID, had = file.cases[key];
      const levels = (had?.tracts.series === c.tracts ? had.levels : undefined) ?? levelsOf(sl) ?? { crus: 0, ic: 15, coronal: 0 };
      file.cases[key] = { ...(had ?? {}), patient: c.patient, side: side < 0 ? "left" : "right", tracts: { series: c.tracts, ...(rules ? { rules } : {}) }, ...(had && had.tracts.series !== c.tracts ? { verdict: undefined, judgedAt: undefined } : {}) } as Review;
      setLayout(LAYOUT.conventionalWidescreen);
      orientView("Red", "axial"); orientView("Yellow", "axial"); orientView("Green", "coronal");
      setSliceOffset("Red", levels.crus); setSliceOffset("Yellow", levels.ic); setSliceOffset("Green", levels.coronal);
      draw(sl);
      lookFrom3D("A");
      open = { side, levels };
      say(sl.length ? `${c.patient}: the ${side < 0 ? "left" : "right"} corticospinal tract, ${sl.length.toLocaleString()} fibers${mapId ? "" : " (the direction-colored map was not stored with these tracts)"}.` : `${c.patient}: no fibers named corticospinal tract on the ${side < 0 ? "left" : "right"}.`);
    } catch (e) {
      say(`${c.patient} could not be opened: ${(e as Error).message}`);
    } finally { busy = ""; render(); }
  }

  async function judge(verdict: Review["verdict"]): Promise<void> {
    const c = cases[current]; if (!c) return;
    const r = file.cases[c.dwi.seriesInstanceUID]; if (!r) return;
    r.verdict = verdict; r.judgedAt = new Date().toISOString();
    const l = levelsNow(); if (l) r.levels = l;
    await saveReviews();
    render();
  }

  function render(): void {
    if (!root) return;
    root.replaceChildren();
    const judged = cases.filter((c) => file.cases[c.dwi.seriesInstanceUID]?.verdict).length;
    const list = shell.section(root, "1 · Cases", { band: "yellow", open: true, note: cases.length ? `${judged} of ${cases.length} judged` : "" });
    if (!cases.length) { const p = document.createElement("p"); p.className = "sl-hint"; p.textContent = "No fiber tracts made at import in the open database. They are made for each diffusion scan with an MRI of the anatomy when the import job runs."; list.append(p); }
    const table = document.createElement("table"); table.style.cssText = "width:100%;border-collapse:collapse;font-size:11px";
    cases.forEach((c, i) => {
      const r = file.cases[c.dwi.seriesInstanceUID], tr = document.createElement("tr");
      tr.style.cssText = `cursor:pointer;${i === current ? "background:rgba(248,215,100,.14)" : ""}`;
      const v = r?.verdict && r.tracts.series === c.tracts ? r.verdict : undefined;
      tr.innerHTML = `<td style="padding:2px 4px">${c.patient.replace(/[<&]/g, "")}</td><td style="padding:2px 4px;color:var(--sl-fg-muted)">${r?.side ?? ""}</td><td style="padding:2px 4px;color:${v === "acceptable" ? "var(--sl-ok)" : v ? "var(--sl-error)" : "var(--sl-fg-muted)"}">${v === "acceptable" ? "✓ acceptable" : v ? "✗ not acceptable" : "—"}</td>`;
      tr.title = `Open ${c.patient}: the T1, the direction-colored map and the corticospinal tract on the side without the tumor.`;
      tr.onclick = () => { void openCase(i); };
      table.append(tr);
    });
    list.append(table);
    const c = cases[current], r = c && file.cases[c.dwi.seriesInstanceUID];
    if (c && r) {
      const here = shell.section(root, `2 · This case`, { band: "green", open: true, note: c.patient });
      const p = document.createElement("p"); p.className = "sl-hint";
      p.textContent = `The ${r.side} corticospinal tract (${r.side === "left" ? "the tumor is on the right, or there is none" : "the tumor is on the left"}). Red view: axial at the cerebral peduncle; yellow: axial at the internal capsule; green: coronal through the tract; 3D from the front. Move a slider when a level is off: the level you leave is kept for next time.`;
      here.append(p);
      const v = shell.section(root, "3 · Verdict", { band: "yellow", open: true });
      const crit = document.createElement("p"); crit.className = "sl-hint"; crit.textContent = "Judge by: how many errant fibers on the two axial slices; how complete the fan is, seen from the front in 3D."; v.append(crit);
      const row = document.createElement("div"); row.style.cssText = "display:flex;gap:6px;margin:6px 0";
      const ok = document.createElement("button"), no = document.createElement("button");
      ok.textContent = "✓ Acceptable"; no.textContent = "✗ Not acceptable";
      ok.style.cssText = `flex:1;${r.verdict === "acceptable" ? "background:var(--sl-ok);color:#000" : "color:var(--sl-ok);border-color:var(--sl-ok)"}`;
      no.style.cssText = `flex:1;${r.verdict === "not acceptable" ? "background:var(--sl-error);color:#000" : "color:var(--sl-error);border-color:var(--sl-error)"}`;
      ok.title = "Few errant fibers on the two axial slices, and a complete fan from the front."; no.title = "Too many errant fibers, or the fan incomplete.";
      ok.disabled = no.disabled = !!busy;
      ok.onclick = () => { void judge("acceptable"); }; no.onclick = () => { void judge("not acceptable"); };
      row.append(ok, no); v.append(row);
      const ta = document.createElement("textarea"); ta.placeholder = "Note (optional)"; ta.rows = 2; ta.value = r.note ?? ""; ta.style.cssText = "width:100%;box-sizing:border-box";
      ta.onchange = () => { r.note = ta.value.trim() || undefined; void saveReviews(); };
      v.append(ta);
      const nav = document.createElement("div"); nav.style.cssText = "display:flex;gap:6px;align-items:center;margin-top:8px";
      const prev = document.createElement("button"), next = document.createElement("button"), pos = document.createElement("span");
      prev.textContent = "◀ Previous"; next.textContent = "Next ▶"; next.className = "sl-primary"; pos.className = "sl-hint";
      prev.disabled = !!busy || current <= 0; next.disabled = !!busy || current >= cases.length - 1;
      prev.onclick = () => { void openCase(current - 1); }; next.onclick = () => { void openCase(current + 1); };
      pos.textContent = `${current + 1} of ${cases.length}`;
      nav.append(prev, next, pos); v.append(nav);
    }
    if (busy || note) { const s = document.createElement("p"); s.className = "sl-hint"; s.textContent = busy && !note ? busy : note; root.append(s); }
  }

  shell.registerPanel({
    id: "tract-review",
    title: "Tract review",
    groups: ["Display"],
    tip: "Judge the corticospinal tract on the side without a tumor, case after case, for checking the fiber tracts.",
    help: "<p><b>For checking the fiber tracts</b> against an expert's eye. Each case is a diffusion scan whose fiber tracts were made when it was imported. Click a case: its MRI of the anatomy is shown with the direction-colored map over it (red left-right, green front-back, blue up-down), and only the corticospinal tract on the side without the tumor, in one color. The red view is an axial slice at the cerebral peduncle, the yellow one an axial slice at the internal capsule, the green one coronal through the tract; the 3D view is seen from the front. The slice levels are found from the tract; move a slider when one is off, and the level you leave is used next time. Judge by how many errant fibers the two axial slices show and how complete the fan is in 3D: <b>Acceptable</b> or <b>Not acceptable</b>, with a note if you like; <b>Next</b> opens the next case. The verdicts are kept in the database's folder (tract-review.json), with which version of the tracts they were about.</p>",
    async mount(el: HTMLElement) { root = el; await readReviews(); await listCases(); render(); },
    onShow() { void listCases().then(render); },
  });
}

queueModule(registerTractReview);

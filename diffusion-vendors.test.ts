// @full-tier -- reads 27 scanner data sets (about 10 s): the rebuild runs it only in the full tier (Contents/tools/Rebuild SlicerAlbula App.command).
// Diffusion DICOM from many scanners, read by Albula and checked against dcm2niix's reference conversions shipped with
// Chris Rorden's validation sets (github.com/neurolabusc/dcm_qa_*, BSD-2): the same b-values, the same directions in
// patient space, and the same value at the same patient position in every sampled voxel. The sets are fetched by
// Contents/tools/fetch-dwi-vendors.sh (workspace) into Contents/data/dwi-vendors/ (not in git); absent, they skip.
//   deno test -A --no-check logic/readers/diffusion-vendors.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { dcmjs } from "albula/testing";
import { setDicomLibrary } from "albula/testing";
import { groupSeries, parseInstances, volumesOfSeries } from "albula";
import { parseCsa } from "albula";
import { fromDicomVolumes, fromFsl } from "./dwi.ts";
import { parseNiftiVolumes, type Volume } from "albula";
import { ABSENT, testData } from "albula/testing";
import "./hooks.ts";                                         // the diffusion interpreter, as the app registers it

setDicomLibrary(dcmjs);
const ROOT = testData("dcm_qa") ?? ABSENT;
const have = (r: string) => { try { return Deno.statSync(`${ROOT}${r}/In`).isDirectory; } catch { return false; } };

Deno.test("CSA header: SV10 form, tags and multi-item values", () => {
  // Build a two-tag SV10 header: B_value = "1000", DiffusionGradientDirection = ["0.6", "0.8", "0"].
  const enc = new TextEncoder(), parts: number[] = [];
  const i32 = (v: number) => { const b = new Uint8Array(4); new DataView(b.buffer).setInt32(0, v, true); parts.push(...b); };
  parts.push(...enc.encode("SV10"), 4, 3, 2, 1); i32(2); i32(77);
  const tag = (name: string, items: string[]) => {
    const nb = new Uint8Array(64); nb.set(enc.encode(name)); parts.push(...nb);
    i32(items.length); parts.push(...enc.encode("DS\0\0")); i32(0); i32(items.length); i32(77);
    for (const it of items) { const b = enc.encode(it + "\0"); i32(b.length); i32(b.length); i32(77); i32(b.length); parts.push(...b); while (parts.length % 4) parts.push(0); }
  };
  tag("B_value", ["1000"]); tag("DiffusionGradientDirection", ["0.6", "0.8", "0"]);
  const t = parseCsa(new Uint8Array(parts));
  assertEquals(t.get("B_value"), ["1000"]);
  assertEquals(t.get("DiffusionGradientDirection"), ["0.6", "0.8", "0"]);
});

const inv = (m: number[]) => { const a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6], g = m[8], h = m[9], k = m[10]; const A = e * k - f * h, B = -(d * k - f * g), C = d * h - e * g, det = a * A + b * B + c * C; const R = [A, -(b * k - c * h), b * f - c * e, B, a * k - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map((x) => x / det); const t = [m[3], m[7], m[11]]; return [R[0], R[1], R[2], -(R[0] * t[0] + R[1] * t[1] + R[2] * t[2]), R[3], R[4], R[5], -(R[3] * t[0] + R[4] * t[1] + R[5] * t[2]), R[6], R[7], R[8], -(R[6] * t[0] + R[7] * t[1] + R[8] * t[2])]; };

/** Every acquisition in a set: ours against dcm2niix's reference (matched by SeriesDescription and volume count). */
async function compareSet(inDir: string, refDir: string, refSuffix = ""): Promise<{ name: string; volumes: number; bDiff: number; worstCos: number; off: number; differ: number; sampled: number; ratio?: number; traceSetAside?: number }[] & { unmatched: string[] }> {
  const files: string[] = [];
  const walk = (d: string) => { for (const e of Deno.readDirSync(d)) { const p = `${d}/${e.name}`; if (e.isDirectory) walk(p); else if (!e.name.startsWith(".")) files.push(p); } };
  walk(inDir);
  const inst = await parseInstances(files.map((f) => Deno.readFileSync(f).slice().buffer as ArrayBuffer), { names: files });
  const bases = new Map<string, ReturnType<typeof groupSeries>>();
  for (const s of groupSeries(inst)) { const k = s.seriesInstanceUID.split("#")[0]; if (!bases.has(k)) bases.set(k, []); bases.get(k)!.push(s); }
  const refs = [...Deno.readDirSync(refDir)].map((e) => e.name).filter((n) => n.endsWith(".json")).map((n) => n.replace(/\.json$/, "")).filter((n) => n.endsWith(refSuffix));
  const out = [] as unknown as Awaited<ReturnType<typeof compareSet>>;
  out.unmatched = [];
  for (const group of bases.values()) {
    group.sort((a, b) => (a.temporal?.index ?? 0) - (b.temporal?.index ?? 0));
    const frames: Volume[] = group.flatMap((g) => volumesOfSeries(g.instances).frames);
    if (frames.length < 2) continue;
    const all = fromDicomVolumes(frames), desc = String((group[0].instances[0] as { seriesDescription?: string }).seriesDescription ?? "").trim();
    // THE TRACE IMAGE (b > 0, no direction: the scanner's own average over directions, Philips' in particular) is a volume
    // dcm2niix sets aside (it writes it as _ADC); Albula keeps it. When the reference is exactly that much shorter, the
    // comparison sets ours aside too, and says how many.
    const traceAt = all.bValues.map((b, i) => b > 0 && Math.hypot(...all.gradients[i]) < 0.5 ? i : -1).filter((i) => i >= 0);
    const keep = (drop: boolean) => drop ? all.bValues.map((_, i) => i).filter((i) => !traceAt.includes(i)) : all.bValues.map((_, i) => i);
    const pick = (idx: number[]) => ({ ...all, volumes: idx.map((i) => all.volumes[i]), bValues: idx.map((i) => all.bValues[i]), gradients: idx.map((i) => all.gradients[i]) });
    let ours = all, traceSetAside = 0;
    const best = refs.find((r) => {
      try {
        const j = JSON.parse(Deno.readTextFileSync(`${refDir}/${r}.json`));
        let nb = -1; try { nb = Deno.readTextFileSync(`${refDir}/${r}.bval`).trim().split(/\s+/).length; } catch { /* no bval */ }
        if (String(j.SeriesDescription ?? "").trim() !== desc) return false;
        if (nb === all.bValues.length || (nb === -1 && !all.bValues.some((b) => b > 0))) { ours = all; traceSetAside = 0; return true; }
        if (traceAt.length && nb === all.bValues.length - traceAt.length) { ours = pick(keep(true)); traceSetAside = traceAt.length; return true; }
        return false;
      } catch { return false; }
    });
    // A DIFFUSION series that matches no reference is a failure, not a skip (critic, 2026-09-29, finding 10a).
    // A series of trace images only (the scanner's TRACEW, derived after the scan) has no reference in these sets.
    const directional = all.gradients.some((g, i) => all.bValues[i] > 0 && Math.hypot(...g) > 0.5);
    if (!best) { if (directional) out.unmatched.push(desc || group[0].seriesInstanceUID.slice(-12)); continue; }
    const niiPath = [`${refDir}/${best}.nii`, `${refDir}/${best}.nii.gz`].find((p) => { try { return Deno.statSync(p).isFile; } catch { return false; } })!;
    const nii = await parseNiftiVolumes(Deno.readFileSync(niiPath));
    if (nii.length !== ours.volumes.length) { if (directional) out.unmatched.push(`${best} (${ours.volumes.length} volumes, reference ${nii.length})`); continue; }
    let ref;
    try { ref = fromFsl(nii, Deno.readTextFileSync(`${refDir}/${best}.bval`), Deno.readTextFileSync(`${refDir}/${best}.bvec`)); }
    catch { ref = { volumes: nii, bValues: nii.map(() => 0), gradients: nii.map(() => [0, 0, 0]), ijkToRAS: nii[0].ijkToRAS }; }
    const bDiff = Math.max(...ours.bValues.map((b, i) => Math.abs(b - ref.bValues[i])));
    let worstCos = 1;
    // Every volume the reference gives a direction must have one here too: a missing direction is a failure (cos 0),
    // not a volume to skip (the first version of this test skipped them, and hid that GE MR29 directions were not read).
    ours.gradients.forEach((g, i) => { const r = ref.gradients[i]; if (ref.bValues[i] > 0 && Math.hypot(...r) > 0.5) worstCos = Math.min(worstCos, Math.hypot(...g) > 0.5 ? Math.abs(g[0] * r[0] + g[1] * r[1] + g[2] * r[2]) : 0); });
    // And the other way (critic, 2026-09-29, finding 10d): no direction where the reference has none (a b = 0 volume, a
    // trace image). An invented direction counts as cos 0.
    ours.gradients.forEach((g, i) => { const r = ref.gradients[i]; if (!(ref.bValues[i] > 0 && Math.hypot(...r) > 0.5) && Math.hypot(...g) > 0.5) worstCos = 0; });
    const M = ours.ijkToRAS, Ni = inv(ref.ijkToRAS), [nx, ny, nz] = ours.volumes[0].dims, [rx, ry, rz] = ref.volumes[0].dims;
    let differ = 0, sampled = 0, off = 0, ratio: number | undefined, ratioOk = true;
    // EVERY volume (critic, 2026-09-29, finding 10b: the first and last alone let a middle volume be wrong).
    for (let t = 0; t < ours.volumes.length; t++) for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j += 2) for (let i = 0; i < nx; i += 2) {
      const x = M[0] * i + M[1] * j + M[2] * k + M[3], y = M[4] * i + M[5] * j + M[6] * k + M[7], z = M[8] * i + M[9] * j + M[10] * k + M[11];
      const u = Ni[0] * x + Ni[1] * y + Ni[2] * z + Ni[3], w = Ni[4] * x + Ni[5] * y + Ni[6] * z + Ni[7], q = Ni[8] * x + Ni[9] * y + Ni[10] * z + Ni[11];
      const ri = Math.round(u), rj = Math.round(w), rk = Math.round(q);
      off = Math.max(off, Math.abs(u - ri), Math.abs(w - rj), Math.abs(q - rk));
      sampled++;
      if (ri < 0 || rj < 0 || rk < 0 || ri >= rx || rj >= ry || rk >= rz) { differ++; continue; }
      const a = ours.volumes[t].data[(k * ny + j) * nx + i], b = ref.volumes[t].data[(rk * ry + rj) * rx + ri];
      if (a !== b) differ++;
      // ONE CONSTANT FACTOR (Philips: the reference holds its private floating-point scale, we the standard rescale).
      if (b !== 0) { const q = a / b; if (ratio === undefined) ratio = q; else if (Math.abs(q - ratio) > 1e-5 * Math.abs(ratio)) ratioOk = false; } else if (a !== 0) ratioOk = false;
    }
    out.push({ name: best, volumes: ours.volumes.length, bDiff, worstCos, off, differ, sampled, ...(differ && ratioOk && ratio !== undefined ? { ratio } : {}), ...(traceSetAside ? { traceSetAside } : {}) });
  }
  return out;
}

// Each set: Siemens mosaics (tilted, and stored head-to-foot), GE, and whatever else has been fetched.
// Philips: identical up to ONE constant factor -- the reference holds Philips' private floating-point values (stored /
// (2005,100E)), Albula the standard's rescale (stored × RescaleSlope + intercept). Constant over a scan, the factor
// cancels in every diffusion ratio; were it to vary between volumes, the diffusion computation would have to use the
// floating-point values (dmri-review-2026-09-29-vendor-note.md).
const PHILIPS_SCALE = new Set(["dcm_qa_philips_dwi"]);
for (const [repo, what, minSeries, sub, suffix] of [
  ["dcm_qa_dti", "Siemens Prisma mosaic DTI, straight and tilted (yaw, pitch, roll, two axes)", 5, "", ""],
  ["dcm_qa_mosaic", "Siemens mosaic fMRI stored foot-to-head and head-to-foot (geometry and slice order)", 8, "", ""],
  ["dcm_qa_ge", "GE DTI, including multiband", 4, "", ""],
  ["dcm_qa_toshiba", "Toshiba DTI, axial, sagittal, coronal and tilted", 5, "", ""],
  ["dcm_qa_canon", "Canon DTI, axial, sagittal, coronal and tilted", 11, "", ""],
  ["dcm_qa_xa30", "Siemens XA30 diffusion (enhanced), AP and PA", 2, "", ""],
  ["dcm_qa_xa60", "Siemens XA60/XA61 diffusion (enhanced), multiband", 6, "", ""],
  ["dcm_qa_philips_dwi", "Philips DTI, classic single-frame files", 1, "/cl", "_cl"],
  ["dcm_qa_philips_dwi", "Philips DTI, enhanced multi-frame file", 1, "/enh", "_enh"],
] as const) {
  Deno.test({
    name: `${repo}${sub}: ${what} -- identical to dcm2niix${PHILIPS_SCALE.has(repo) ? " (values up to Philips' constant scale)" : ""}`,
    ignore: !have(repo),
    fn: async () => {
      const r = await compareSet(`${ROOT}${repo}/In${sub}`, `${ROOT}${repo}/Ref`, suffix);
      for (const s of r) console.log(`  ${s.name}: ${s.volumes} volumes; b diff ${s.bDiff}; worst |cos| ${s.worstCos.toFixed(6)}; grid off ${s.off.toFixed(4)}; ${s.differ}/${s.sampled} voxels differ${s.traceSetAside ? `; ${s.traceSetAside} trace volume set aside (dcm2niix leaves it out)` : ""}`);
      assert(r.length >= minSeries, `${r.length} series compared`);
      assertEquals(r.unmatched, [], "diffusion series that matched no reference");
      for (const s of r) {
        const values = s.differ === 0 || (PHILIPS_SCALE.has(repo) && s.ratio !== undefined);
        assert(s.bDiff < 1e-3 && s.worstCos > 0.99999 && s.off < 1e-3 && values, `${s.name}: ${JSON.stringify(s)}`);
      }
    },
  });
}

// THE DWIConvert TEST LIBRARY (BRAINSTools, fetched by content hash): older scanners the dcm_qa sets do not have. They
// ship no dcm2niix conversion, so Contents/tools/make-dwi-vendor-refs.sh makes one beside them (brainstools-ref/, the
// dcm2niix version recorded there); both absent, the test skips.
const BT = testData("brainstools") ?? ABSENT;
const BT_REF = testData("brainstools-ref") ?? ABSENT;
// NOT YET MATCHING: a set named here is kept in the list so it is seen, and skipped with its reason so the suite stays
// a pass/fail signal. Empty since 2026-09-29 (vendor rule 2): all 17 match.
const KNOWN: Record<string, string> = {};
const haveBt = (n: string) => { try { return Deno.statSync(`${BT}${n}`).isDirectory && Deno.statSync(`${BT_REF}${n}`).isDirectory; } catch { return false; } };
for (const [set, what] of [
  ["GeSignaHDx", "GE Signa HDx (software 14), directions only in GE's private attributes, in the image's frame"],
  ["GeSignaHDxBigEndian", "the same GE study stored explicit VR big endian (the byte order swapped on read)"],
  ["GeSignaHDxt", "GE Signa HDxt, private attributes"],
  ["PhilipsAchieva1", "Philips Achieva, older single-frame files"],
  ["PhilipsAchieva2", "Philips Achieva, older single-frame files"],
  ["PhilipsAchieva3", "Philips Achieva, older single-frame files"],
  ["PhilipsAchieva4", "Philips Achieva, older single-frame files"],
  ["PhilipsAchieva6", "Philips Achieva, older single-frame files"],
  ["PhilipsAchieva7", "Philips Achieva, older single-frame files"],
  ["PhilipsAchievaBigEndian1", "Philips Achieva stored explicit VR big endian"],
  ["SiemensTrio-Syngo2004A-1", "Siemens Trio, syngo 2004A"],
  ["SiemensTrio-Syngo2004A-2", "Siemens Trio, syngo 2004A"],
  ["SiemensTrioTim1", "Siemens Trio Tim"],
  ["SiemensTrioTim2", "Siemens Trio Tim"],
  ["SiemensTrioTim3", "Siemens Trio Tim"],
  ["SiemensTrioTimBigEndian1", "Siemens Trio Tim stored explicit VR big endian"],
  ["SiemensVerio", "Siemens Verio"],
] as const) {
  Deno.test({
    name: `${set}: ${what} -- identical to dcm2niix`,
    ignore: !haveBt(set) || set in KNOWN,
    fn: async () => {
      const r = await compareSet(`${BT}${set}`, `${BT_REF}${set}`);
      for (const s of r) console.log(`  ${s.name}: ${s.volumes} volumes; b diff ${s.bDiff}; worst |cos| ${s.worstCos.toFixed(6)}; grid off ${s.off.toFixed(4)}; ${s.differ}/${s.sampled} voxels differ${s.traceSetAside ? `; ${s.traceSetAside} trace volume set aside (dcm2niix leaves it out)` : ""}`);
      assert(r.length >= 1, `${r.length} series compared`);
      assertEquals(r.unmatched, [], "diffusion series that matched no reference");
      // Philips: values may differ from dcm2niix by one constant factor (the standard rescale versus Philips' private
      // scale; see the dcm_qa_philips_dwi test above).
      const scaled = set.startsWith("Philips");
      for (const s of r) assert(s.bDiff < 1e-3 && s.worstCos > 0.99999 && s.off < 1e-3 && (s.differ === 0 || (scaled && s.ratio !== undefined)), `${s.name}: ${JSON.stringify(s)}`);
    },
  });
}

// NOT DIFFUSION STAYS NOT DIFFUSION (critic, 2026-09-29, finding 3): GE writes (0043,1039) on every MR image, so a T1,
// an fMRI or a field map read as "b = 0" and every GE MR series was labeled diffusion. A series dcm2niix gives no .bval
// must carry no diffusion label here.
for (const repo of ["dcm_qa_ge", "dcm_qa_polar"]) {
  Deno.test({
    name: `${repo}: series that are not diffusion carry no diffusion label`,
    ignore: !have(repo),
    fn: async () => {
      const inDir = `${ROOT}${repo}/In`, refDir = `${ROOT}${repo}/Ref`;
      const refs = [...Deno.readDirSync(refDir)].map((e) => e.name).filter((n) => n.endsWith(".json")).map((n) => n.replace(/\.json$/, ""));
      const noBval = new Set(refs.filter((r) => { try { Deno.statSync(`${refDir}/${r}.bval`); return false; } catch { return true; } })
        .map((r) => String(JSON.parse(Deno.readTextFileSync(`${refDir}/${r}.json`)).SeriesDescription ?? "").trim()));
      let checked = 0; const labeled: string[] = [];
      for (const e of Deno.readDirSync(inDir)) {
        if (!e.isDirectory) continue;
        const files: string[] = [];
        const walk = (d: string) => { for (const f of Deno.readDirSync(d)) { const p = `${d}/${f.name}`; if (f.isDirectory) walk(p); else if (!f.name.startsWith(".")) files.push(p); } };
        walk(`${inDir}/${e.name}`);
        const inst = await parseInstances(files.map((f) => Deno.readFileSync(f).slice().buffer as ArrayBuffer), { names: files });
        const desc = String((inst[0] as { seriesDescription?: string } | undefined)?.seriesDescription ?? "").trim();
        if (!inst.length || !noBval.has(desc)) continue;
        checked++;
        if (inst.some((i) => i.volumeKeys?.diffusion !== undefined)) labeled.push(e.name);
      }
      console.log(`  ${checked} series that are not diffusion checked`);
      assert(checked > 0, "no series without a .bval was found to check");
      assertEquals(labeled, [], `labeled diffusion but are not: ${labeled.join(", ")}`);
    },
  });
}

// GE MULTI-SHELL (critic, 2026-09-29, finding 6; no public GE multi-shell set is in the test data): a lower shell is a
// shortened vector under the one maximum b. The b-value scales by the length squared, the direction stays a unit vector.
Deno.test("GE: a shortened gradient vector scales the b-value (a lower shell), as dcm2niix does", async () => {
  const { diffusionOf } = await import("./diffusion-vendors.ts");
  const ds = { Manufacturer: "GE MEDICAL SYSTEMS", InPlanePhaseEncodingDirection: "COL", ImageOrientationPatient: [1, 0, 0, 0, 1, 0] };
  const raw = (b: number, g: number[]) => ({ "00431039": { Value: [b, 8, 0, 0] }, "001910BB": { Value: [g[0]] }, "001910BC": { Value: [g[1]] }, "001910BD": { Value: [g[2]] } });
  const r = 1 / Math.sqrt(3);                                     // length 0.577 under b = 3000 -> b = 1000
  const low = diffusionOf(ds, raw(3000, [r, 0, 0]))!;
  assertEquals(low.bValue, 1000);
  assert(Math.abs(Math.hypot(...low.direction!) - 1) < 1e-9);
  assertEquals(diffusionOf(ds, raw(3000, [1, 0, 0]))!.bValue, 3000);      // full length: the maximum shell
  const row = diffusionOf({ ...ds, InPlanePhaseEncodingDirection: "ROW" }, raw(1000, [1, 0, 0]))!;
  assert(!row.direction && row.source.includes("phase encoding"), row.source);   // says the real reason
});

// A MOSAIC WHOSE CSA LISTS SliceNormalVector WITH EMPTY ITEMS (the syngo 2004 CSA shape; critic, 2026-09-29, finding 7)
// falls back to the image's own normal instead of giving NaN positions.
Deno.test("Siemens mosaic: an empty SliceNormalVector falls back to the image's normal (no NaN positions)", async () => {
  const { siemensMosaic } = await import("albula");
  const enc = new TextEncoder(), parts: number[] = [];
  const i32 = (v: number) => { const b = new Uint8Array(4); new DataView(b.buffer).setInt32(0, v, true); parts.push(...b); };
  parts.push(...enc.encode("SV10"), 4, 3, 2, 1); i32(2); i32(77);
  const tag = (name: string, items: string[]) => {
    const nb = new Uint8Array(64); nb.set(enc.encode(name)); parts.push(...nb);
    i32(items.length); parts.push(...enc.encode("DS\0\0")); i32(0); i32(items.length); i32(77);
    for (const it of items) { const b = enc.encode(it + "\0"); i32(b.length); i32(b.length); i32(77); i32(b.length); parts.push(...b); while (parts.length % 4) parts.push(0); }
  };
  tag("NumberOfImagesInMosaic", ["4"]); tag("SliceNormalVector", ["", "", ""]);
  const csa = new Uint8Array(parts);
  const ds = { ImageType: ["ORIGINAL", "PRIMARY", "M", "MOSAIC"], Rows: 128, Columns: 128, ImageOrientationPatient: [1, 0, 0, 0, 1, 0], ImagePositionPatient: [0, 0, 0], PixelSpacing: [2, 2], SliceThickness: 3, SpacingBetweenSlices: 3 };
  const m = siemensMosaic(ds, { "00291010": { Value: [csa.buffer] } })!;
  assertEquals(m.n, 4);
  assert(m.positions.every((p) => p.every(Number.isFinite)), JSON.stringify(m.positions));
  assert(Math.abs(m.positions[1][2] - m.positions[0][2] - 3) < 1e-9, "tiles step one slice spacing along the image normal");
});

// PRIVATE ELEMENTS BY THEIR CREATOR (critic, 2026-09-29, finding 12), not by a fixed block.
Deno.test("private elements are found in the block their creator reserved", async () => {
  const { privateTag } = await import("albula");
  assertEquals(privateTag({ "00190010": { Value: ["SIEMENS SMS-AX VIEW 1.0"] }, "00190011": { Value: ["GEMS_ACQU_01"] } }, "0019", "GEMS_ACQU_01", "BB"), "001911BB");
  assertEquals(privateTag({ "00190010": { Value: ["SIEMENS SMS-AX VIEW 1.0"] } }, "0019", "GEMS_ACQU_01", "BB"), undefined);   // someone else's
  assertEquals(privateTag({}, "0019", "GEMS_ACQU_01", "BB"), "001910BB");                                                       // no creators: the usual block
  const bytes = new TextEncoder().encode("GEMS_PARM_01").buffer;                                                                 // implicit VR: creator as bytes
  assertEquals(privateTag({ "00430010": { Value: [bytes] } }, "0043", "GEMS_PARM_01", "39"), "00431039");
});

// @full-tier -- THE APPLYING STAGE OF THE DISTORTION CORRECTION against FSL's applytopup (validation step 3, Ron,
// 2026-10-03; Mike Halle's standard: each stage against its original). The same field -- topup's own (df_01, its AP
// displacement) -- is applied to the whole AP diffusion scan by distortion.ts's applyField and by applytopup --method=jac
// (Contents/tools/applytopup-reference.ts made it), so only the applying differs: our linear interpolation along the
// phase-encoding line and our Jacobian from face differences, against FSL's splines. Compared where it matters: every
// corrected volume inside the brain, and the tensor fit on each (FA, and the main direction's angle).
// WHAT IT SHOWED (2026-10-03, PAT16 / PAT05 / PAT03 / PAT08): the geometry is the same (topup's field fits best at scale
// 1.0 and shift 0; the line totals are kept by both), and the 2-4% left per volume is the INTERPOLATION: applytopup
// interpolates with cubic splines -- except the FIRST volume of the file, which it interpolates linearly, as we do:
// volume 0 agrees within 1.3-1.4% (this test's brain mask; 1.1% in the b = 0 mask), the others 4.0-4.7% median (with a
// cubic B-spline along the line, the others agree within
// 0.75-0.98% and volume 0 then differs by 4.2%; PAT16, measured in a scratch script). topup's own corrected b = 0
// (--iout) is NOT a reference for the applying: it differs from applytopup's by 7.4% (its intensity scaling, b02b0.cnf).
// PAT03 agrees less well even with the same interpolation (first volume 7.5% here, 3.4% in the b = 0 brain mask; its
// "PA" is not a reversed pair at all -- phase-encoded left-right, found later the same day): the difference sits at the brain's edge along the phase-encoding axis, where its field is
// steepest and FSL takes the stretch from its spline's derivative, we from differences (from the field at the centers
// with central differences: 2.9%; PAT16 0.97%). So, as in distortion-topup.test.ts, the close match is required where
// the head held still and measured elsewhere.
// Skipped where the reference data are absent (no FSL on that machine).
//
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc distortion-apply.test.ts
import { assert } from "jsr:@std/assert@1";
import { ABSENT, testData } from "albula/testing";
import { parseNiftiVolumes } from "albula";
import { applyField, fieldFromCenters } from "./distortion.ts";
import { fromFsl, type DiffusionSeries } from "./dwi.ts";
import { fitTensors } from "./tensor.ts";

const ROOT = testData("topup-reference", "") ?? ABSENT;
const DS = testData("openneuro-ds001226", "") ?? ABSENT;
const ready = (name: string) => { try { return Deno.statSync(`${ROOT}/${name}/apply.json`).isFile; } catch { return false; } };
const cases = (() => { try { return [...Deno.readDirSync(ROOT)].filter((e) => e.isDirectory && ready(e.name)).map((e) => e.name).sort(); } catch { return []; } })();

const vols = async (f: string) => await parseNiftiVolumes(Deno.readFileSync(f));
const pct = (xs: number[], f: number) => { const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(f * s.length))]; };

Deno.test({ name: "applying the distortion field against FSL's applytopup (topup's field, the whole AP scan)", ignore: !cases.length, fn: async () => {
  const rows: { id: string; still: boolean; first: number; rel: number; fa50: number; fa99: number; ang50: number; ang90: number }[] = [];
  for (const id of cases) {
    const D = `${ROOT}/${id}`, P = `${DS}/sub-${id}/ses-preop/dwi/sub-${id}_ses-preop_acq-AP_dwi`;
    const bval = Deno.readTextFileSync(`${P}.bval`), bvec = Deno.readTextFileSync(`${P}.bvec`);
    const raw = await vols(`${P}.nii.gz`), fsl = await vols(`${D}/ap_applytopup.nii.gz`);
    const dims = raw[0].dims as [number, number, number], M = raw[0].ijkToRAS;
    assert(fsl.length === raw.length && fsl[0].ijkToRAS.every((x, i) => Math.abs(x - M[i]) < 1e-3), `${id}: applytopup's output is not on the scan's grid`);
    // topup's AP displacement along j (mm) as our field in voxels. The sign is checked, not assumed: applied with each
    // sign to the first volume (interpolated linearly by applytopup too; header), the one that reproduces it is kept.
    const df = await vols(`${D}/df_01.nii.gz`), dj = (df.length >= 3 ? df[1] : df[0]).data as ArrayLike<number>;
    const mmJ = Math.hypot(M[1], M[5], M[9]);
    const rms = (a: ArrayLike<number>, c: ArrayLike<number>, idx: number[]) => { let d = 0, m = 0; for (const v of idx) { d += (a[v] - c[v]) ** 2; m += c[v] ** 2; } return Math.sqrt(d / m); };
    const series = (data: (v: number) => ArrayLike<number>): DiffusionSeries => {
      const s = fromFsl(raw.map((r, i) => ({ ...r, data: data(i), dtype: "<f4" })), bval, bvec);
      return s;
    };
    const fitFsl = fitTensors(series((i) => Float32Array.from(fsl[i].data as ArrayLike<number>)), { maxB: 1500 });
    const inside: number[] = []; for (let v = 0; v < fitFsl.mask.length; v++) if (fitFsl.mask[v]) inside.push(v);
    const bySign = [1, -1].map((s) => { const fit = fieldFromCenters(dims, 1, Float32Array.from(dj, (x) => s * x / mmJ)); return { s, fit, err: rms(applyField(fit, raw[0].data as ArrayLike<number>, -1), fsl[0].data as ArrayLike<number>, inside) }; });
    const { s, fit, err: firstErr } = bySign[0].err <= bySign[1].err ? bySign[0] : bySign[1];
    // Every volume: ours against applytopup's, inside the brain.
    const ours = raw.map((r) => applyField(fit, r.data as ArrayLike<number>, -1));
    const rels = ours.map((o, i) => rms(o, fsl[i].data as ArrayLike<number>, inside));
    const fitOurs = fitTensors(series((i) => ours[i]), { maxB: 1500 });
    // FA and the main direction, in white matter (FA above 0.2 by applytopup's fit), where tracking happens.
    const dFA: number[] = [], ang: number[] = [];
    for (const v of inside) {
      dFA.push(Math.abs(fitOurs.fa[v] - fitFsl.fa[v]));
      if (fitFsl.fa[v] > 0.2) {
        const c = Math.abs(fitOurs.v1[3 * v] * fitFsl.v1[3 * v] + fitOurs.v1[3 * v + 1] * fitFsl.v1[3 * v + 1] + fitOurs.v1[3 * v + 2] * fitFsl.v1[3 * v + 2]);
        ang.push(Math.acos(Math.min(1, c)) * 180 / Math.PI);
      }
    }
    const apply = JSON.parse(Deno.readTextFileSync(`${D}/apply.json`)) as { seconds: number; fsl: string };
    const mv = Deno.readTextFileSync(`${D}/tu_movpar.txt`).trim().split("\n")[1].trim().split(/\s+/).map(Number);
    const still = Math.max(...mv.slice(3, 6).map((x) => Math.abs(x))) * 180 / Math.PI < 0.5;
    const row = { id, still, first: firstErr, rel: pct(rels, 0.5), fa50: pct(dFA, 0.5), fa99: pct(dFA, 0.99), ang50: pct(ang, 0.5), ang90: pct(ang, 0.9) };
    console.log(`${id}: topup's field ${s > 0 ? "as is" : "with its sign reversed"}; first volume ${(100 * firstErr).toFixed(2)}% (both linear); ` +
      `${raw.length} volumes vs applytopup: median ${(100 * row.rel).toFixed(2)}%, worst ${(100 * Math.max(...rels)).toFixed(2)}% (RMS difference / RMS, in the brain); ` +
      `FA |difference| median ${row.fa50.toFixed(4)}, 99th ${row.fa99.toFixed(4)}; main direction (FA > 0.2) median ${row.ang50.toFixed(2)}°, 90th ${row.ang90.toFixed(2)}° ` +
      `(applytopup ${apply.seconds} s, FSL ${apply.fsl})${still ? "" : "; the head moved between the scans: measured, not required"}`);
    rows.push(row);
  }
  // Guards at the level measured when written (2026-10-03), with room: the applying is interpolation, and FSL's splines
  // differ from our linear lines by a few percent of the signal; what must hold is that the FA and directions tracking
  // reads agree to well under the scan's own noise.
  for (const r of rows) {
    if (!r.still) continue;
    assert(r.first < 0.02, `${r.id}: with the same interpolation (the first volume) ours differs from applytopup's by ${(100 * r.first).toFixed(1)}%`);
    assert(r.rel < 0.05, `${r.id}: the corrected volumes differ from applytopup's by ${(100 * r.rel).toFixed(1)}% (median)`);
    assert(r.fa50 < 0.02 && r.ang50 < 3, `${r.id}: FA differs by ${r.fa50.toFixed(3)} (median), the main direction by ${r.ang50.toFixed(1)}°`);
  }
}});

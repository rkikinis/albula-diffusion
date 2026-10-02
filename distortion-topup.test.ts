// @full-tier -- Albula's distortion correction (distortion.ts, written from the papers) against FSL's topup on the same
// mean b = 0 images (Contents/tools/topup-reference.ts made them; Ron, 2026-10-02: FSL in testing, never in the pipeline).
// Measured as Mike Halle measured his (his mail, 2026-10-02): the two displacement fields' correlation inside the brain,
// their difference at the median and the 99th percentile (mm), and how far apart the corrected AP and PA images stay,
// by each method. Skipped where the reference data are absent (no FSL on that machine).
//
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc distortion-topup.test.ts
import { assert } from "jsr:@std/assert@1";
import { ABSENT, testData } from "albula/testing";
import { parseNiftiVolumes } from "albula";
import { applyField, estimateField, fieldAtCenters } from "./distortion.ts";
import { medianOtsuMask } from "./median-otsu.ts";
import type { DiffusionSeries } from "./dwi.ts";

const ROOT = testData("topup-reference", "") ?? ABSENT;
const done = (name: string) => { try { return Deno.statSync(`${ROOT}/${name}/manifest.json`).isFile; } catch { return false; } };   // a case topup is still working on has no manifest yet
const cases = (() => { try { return [...Deno.readDirSync(ROOT)].filter((e) => e.isDirectory && done(e.name)).map((e) => e.name).sort(); } catch { return []; } })();

const vol = async (f: string) => (await parseNiftiVolumes(Deno.readFileSync(f)));

Deno.test({ name: "the distortion field against FSL's topup (cases with one grid for AP and PA)", ignore: !cases.length, fn: async () => {
  const results: { id: string; r: number; ours: number; theirs: number; before: number; still: boolean; med: number; p99: number }[] = [];
  for (const id of cases) {
    const D = `${ROOT}/${id}`;
    const ap = (await vol(`${D}/ap_mean.nii.gz`))[0], pa = (await vol(`${D}/pa_mean.nii.gz`))[0];
    const dims = ap.dims as [number, number, number], n = dims[0] * dims[1] * dims[2];
    const A = Float32Array.from(ap.data as ArrayLike<number>), P = Float32Array.from(pa.data as ArrayLike<number>);
    // Phase encoding along j (the sidecars: AP j-, PA j); the reversed-pair model's "plus" is the +j image (PA).
    const t0 = performance.now();
    const fit = estimateField({ dims, plus: P, minus: A, axis: 1 });
    const seconds = (performance.now() - t0) / 1000;
    const mmJ = Math.hypot(ap.ijkToRAS[1], ap.ijkToRAS[5], ap.ijkToRAS[9]);
    // Ours: the AP image is displaced by −b along j (I(x) = I₋(x − b·v)(1 − ∂b)); in mm.
    const b = fieldAtCenters(fit), oursMm = Float64Array.from(b, (x) => -x * mmJ);
    // topup's displacement field for AP: three frames (x, y, z), mm; the j axis is the second.
    const df = await vol(`${D}/df_01.nii.gz`);
    const theirsMm = Float64Array.from((df.length >= 3 ? df[1] : df[0]).data as ArrayLike<number>);
    const series = { volumes: [{ dims, ijkToRAS: ap.ijkToRAS, data: A, dtype: "<f4" }], bValues: [0], gradients: [[0, 0, 0]], ijkToRAS: ap.ijkToRAS, source: "test", convention: 1 } as unknown as DiffusionSeries;
    const mask = medianOtsuMask(series, [0]).mask;
    const inside: number[] = []; for (let v = 0; v < n; v++) if (mask[v]) inside.push(v);
    // The sign convention of topup's field is checked, not assumed: the better-correlated sign is used, and said.
    const corr = (s: number) => {
      let mx = 0, my = 0; for (const v of inside) { mx += s * oursMm[v]; my += theirsMm[v]; } mx /= inside.length; my /= inside.length;
      let sxy = 0, sxx = 0, syy = 0; for (const v of inside) { const a = s * oursMm[v] - mx, c = theirsMm[v] - my; sxy += a * c; sxx += a * a; syy += c * c; }
      return sxy / Math.sqrt(sxx * syy);
    };
    const sign = corr(1) >= corr(-1) ? 1 : -1, r = corr(sign);
    const diff = inside.map((v) => Math.abs(sign * oursMm[v] - theirsMm[v])).sort((x, y) => x - y);
    const q = (f: number) => diff[Math.min(diff.length - 1, Math.floor(f * diff.length))];
    // The corrected AP and PA: how far apart they stay (RMS of the difference over RMS of the mean, inside the brain).
    const left = (a: ArrayLike<number>, c: ArrayLike<number>) => { let d = 0, m = 0; for (const v of inside) { d += (a[v] - c[v]) ** 2; m += ((a[v] + c[v]) / 2) ** 2; } return Math.sqrt(d / m); };
    const ours = left(applyField(fit, A, -1), applyField(fit, P, 1));
    const corrected = await vol(`${D}/corr.nii.gz`);
    const theirs = left(corrected[0].data as ArrayLike<number>, corrected[1].data as ArrayLike<number>);
    const before = left(A, P);
    const manifest = JSON.parse(Deno.readTextFileSync(`${D}/manifest.json`)) as { seconds: { topup: number }; fsl: string };
    console.log(`${id}: field r ${r.toFixed(3)}${sign < 0 ? " (topup's sign opposite)" : ""}; |difference| median ${q(0.5).toFixed(2)} mm, 99th ${q(0.99).toFixed(2)} mm; ` +
      `AP vs PA left after correction ${ours.toFixed(3)} (ours) / ${theirs.toFixed(3)} (topup), ${before.toFixed(3)} before; ` +
      `${seconds.toFixed(1)} s against topup's ${manifest.seconds.topup} s (FSL ${manifest.fsl})`);
    // MOTION BETWEEN THE TWO SCANS: topup models it (tu_movpar.txt: the second scan's translation in mm, rotation in
    // radians); distortion.ts does not, and soaks it up as distortion -- PAT03 (1.8° of rotation) gave a field unlike
    // topup's (r 0.12), PAT16 and PAT05 (under 0.5°) agree with it as Mike Halle's does (r 0.97). So the close match is
    // required where the head held still; the others are measured and said.
    const mv = Deno.readTextFileSync(`${D}/tu_movpar.txt`).trim().split("\n")[1].trim().split(/\s+/).map(Number);
    const rotDeg = Math.max(...mv.slice(3, 6).map((x) => Math.abs(x))) * 180 / Math.PI, still = rotDeg < 0.5;
    console.log(`  ${id}: motion between the scans per topup: ${mv.slice(0, 3).map((x) => x.toFixed(2)).join(", ")} mm, up to ${rotDeg.toFixed(2)}° ${still ? "(still)" : "(moved: not required to match)"}`);
    results.push({ id, r, ours, theirs, before, still, med: q(0.5), p99: q(0.99) });
  }
  for (const x of results) {
    assert(x.ours < x.before, `${x.id}: the correction brings AP and PA closer`);
    // Measured 2026-10-02 on the still cases: r 0.927-0.968, median 0.30-0.39 mm, 99th 2.00-2.88 mm (Mike's own against
    // topup: r 0.969-0.972, 0.21-0.27, 1.9-2.2). A guard at that level; even small motion lowers it (PAT08: 0.48 mm across
    // the phase-encoding direction, r 0.927), which a motion model would fix, not a lower bar.
    if (x.still) {
      assert(x.r > 0.9, `${x.id}: the head held still, and the fields correlate only ${x.r.toFixed(3)}`);
      assert(x.med < 0.5 && x.p99 < 3, `${x.id}: the fields differ by ${x.med.toFixed(2)} mm (median), ${x.p99.toFixed(2)} mm (99th)`);
    }
  }
}});

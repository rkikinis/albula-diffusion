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
import { applyField, estimateField, estimateFieldWithMotion, fieldAtCenters, fieldFromCenters, moveRigid, type FieldFit, type Motion } from "./distortion.ts";
import { medianOtsuMask } from "./median-otsu.ts";
import type { DiffusionSeries } from "./dwi.ts";

const ROOT = testData("topup-reference", "") ?? ABSENT;
const done = (name: string) => { try { return Deno.statSync(`${ROOT}/${name}/manifest.json`).isFile; } catch { return false; } };   // a case topup is still working on has no manifest yet
const cases = (() => { try { return [...Deno.readDirSync(ROOT)].filter((e) => e.isDirectory && done(e.name)).map((e) => e.name).sort(); } catch { return []; } })();

const vol = async (f: string) => (await parseNiftiVolumes(Deno.readFileSync(f)));

Deno.test({ name: "the distortion field against FSL's topup (cases with one grid for AP and PA)", ignore: !cases.length, fn: async () => {
  const results: { id: string; r: number; r1: number; ours: number; theirs: number; before: number; still: boolean; med: number; p99: number }[] = [];
  for (const id of cases) {
    const D = `${ROOT}/${id}`;
    const ap = (await vol(`${D}/ap_mean.nii.gz`))[0], pa = (await vol(`${D}/pa_mean.nii.gz`))[0];
    const dims = ap.dims as [number, number, number], n = dims[0] * dims[1] * dims[2];
    const A = Float32Array.from(ap.data as ArrayLike<number>), P = Float32Array.from(pa.data as ArrayLike<number>);
    // Phase encoding along j (the sidecars: AP j-, PA j); the reversed-pair model's "plus" is the +j image (PA).
    // Rule 2 (the default since 2026-10-03: the field with the PA scan's movement) and, for the record, rule 1.
    const M = ap.ijkToRAS, voxel = [0, 1, 2].map((c) => Math.hypot(M[c], M[4 + c], M[8 + c])) as [number, number, number];
    const t0 = performance.now();
    const fit = estimateFieldWithMotion({ dims, plus: P, minus: A, axis: 1, voxel });
    const seconds = (performance.now() - t0) / 1000;
    const fit1 = estimateField({ dims, plus: P, minus: A, axis: 1 });
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
    const corr = (s: number, ours = oursMm) => {
      let mx = 0, my = 0; for (const v of inside) { mx += s * ours[v]; my += theirsMm[v]; } mx /= inside.length; my /= inside.length;
      let sxy = 0, sxx = 0, syy = 0; for (const v of inside) { const a = s * ours[v] - mx, c = theirsMm[v] - my; sxy += a * c; sxx += a * a; syy += c * c; }
      return sxy / Math.sqrt(sxx * syy);
    };
    const sign = corr(1) >= corr(-1) ? 1 : -1, r = corr(sign);
    const r1 = corr(sign, Float64Array.from(fieldAtCenters(fit1), (x) => -x * mmJ));
    const diff = inside.map((v) => Math.abs(sign * oursMm[v] - theirsMm[v])).sort((x, y) => x - y);
    const q = (f: number) => diff[Math.min(diff.length - 1, Math.floor(f * diff.length))];
    // The corrected AP and PA: how far apart they stay (RMS of the difference over RMS of the mean, inside the brain).
    const left = (a: ArrayLike<number>, c: ArrayLike<number>) => { let d = 0, m = 0; for (const v of inside) { d += (a[v] - c[v]) ** 2; m += ((a[v] + c[v]) / 2) ** 2; } return Math.sqrt(d / m); };
    const ours = left(applyField(fit, A, -1), applyField(fit, P, 1));
    const corrected = await vol(`${D}/corr.nii.gz`);
    const theirs = left(corrected[0].data as ArrayLike<number>, corrected[1].data as ArrayLike<number>);
    const before = left(A, P);
    const manifest = JSON.parse(Deno.readTextFileSync(`${D}/manifest.json`)) as { seconds: { topup: number }; fsl: string };
    console.log(`${id}: field r ${r.toFixed(3)} (rule 1: ${r1.toFixed(3)}; PA's movement ${fit.motion.t.map((x) => x.toFixed(2)).join(", ")} mm, ${fit.motion.r.map((x) => (x * 180 / Math.PI).toFixed(2)).join(", ")}°)${sign < 0 ? " (topup's sign opposite)" : ""}; |difference| median ${q(0.5).toFixed(2)} mm, 99th ${q(0.99).toFixed(2)} mm; ` +
      `AP vs PA left after correction ${ours.toFixed(3)} (ours) / ${theirs.toFixed(3)} (topup), ${before.toFixed(3)} before; ` +
      `${seconds.toFixed(1)} s against topup's ${manifest.seconds.topup} s (FSL ${manifest.fsl})`);
    // MOTION BETWEEN THE TWO SCANS: topup models it (tu_movpar.txt: the second scan's translation in mm, rotation in
    // radians); distortion.ts does not, and soaks it up as distortion -- PAT03 (1.8° of rotation) gave a field unlike
    // topup's (r 0.12), PAT16 and PAT05 (under 0.5°) agree with it as Mike Halle's does (r 0.97). So the close match is
    // required where the head held still; the others are measured and said.
    const mv = Deno.readTextFileSync(`${D}/tu_movpar.txt`).trim().split("\n")[1].trim().split(/\s+/).map(Number);
    const rotDeg = Math.max(...mv.slice(3, 6).map((x) => Math.abs(x))) * 180 / Math.PI, still = rotDeg < 0.5;
    console.log(`  ${id}: motion between the scans per topup: ${mv.slice(0, 3).map((x) => x.toFixed(2)).join(", ")} mm, up to ${rotDeg.toFixed(2)}° ${still ? "(still)" : "(moved: not required to match)"}`);
    results.push({ id, r, r1, ours, theirs, before, still, med: q(0.5), p99: q(0.99) });
  }
  for (const x of results) {
    assert(x.ours < x.before, `${x.id}: the correction brings AP and PA closer`);
    // Measured 2026-10-02 on the still cases: r 0.927-0.968, median 0.30-0.39 mm, 99th 2.00-2.88 mm (Mike's own against
    // topup: r 0.969-0.972, 0.21-0.27, 1.9-2.2). A guard at that level; even small motion lowers it (PAT08: 0.48 mm across
    // the phase-encoding direction, r 0.927), which a motion model would fix, not a lower bar.
    if (x.still) {
      assert(x.r > 0.9, `${x.id}: the head held still, and the fields correlate only ${x.r.toFixed(3)}`);
      assert(x.med < 0.5 && x.p99 < 3, `${x.id}: the fields differ by ${x.med.toFixed(2)} mm (median), ${x.p99.toFixed(2)} mm (99th)`);
      // Rule 2 measured 2026-10-03: r 0.975-0.982 against rule 1's 0.927-0.968 on PAT16, PAT05, PAT08.
      assert(x.r > x.r1, `${x.id}: with the movement allowed for, the field agrees less with topup's (${x.r.toFixed(3)} against ${x.r1.toFixed(3)})`);
    }
  }
}});

// RULE 2 ON A PHANTOM WITH A KNOWN MOVEMENT: PAT16's b = 0 as topup corrected it is the undistorted head, topup's field the
// known field; the PA image is made from it with that field and then moved (0.4 mm, -0.3 mm across the phase-encoding
// axis; 0.5°, 0.3°, -0.4°), both with Rician noise. Measured 2026-10-03: rule 2's field error 0.24 mm (median) against
// rule 1's 0.33; the movement comes back in the right direction but at about 40% of its size (the phantom's PA was made by
// resampling, which blurs it -- real scans are not -- and the fit on half-size images; on real scans the movements are
// close to topup's own estimates). What the correction uses is the field, so the field is what is required to improve.
Deno.test({ name: "rule 2 on a phantom with a known movement between the scans: the field closer to the truth than rule 1's", ignore: !cases.includes("PAT16"), fn: async () => {
  const D = `${ROOT}/PAT16`;
  const corr = (await vol(`${D}/corr.nii.gz`))[0], df = await vol(`${D}/df_01.nii.gz`);
  const dims = corr.dims as [number, number, number], [nx, ny, nz] = dims, M = corr.ijkToRAS;
  const voxel = [0, 1, 2].map((c) => Math.hypot(M[c], M[4 + c], M[8 + c])) as [number, number, number];
  const head = Float32Array.from(corr.data as ArrayLike<number>), b = Float32Array.from((df.length >= 3 ? df[1] : df[0]).data as ArrayLike<number>, (x) => -x / voxel[1]);
  // The distorted images: the field applied forward (the inverse of applyField's correction), line by line.
  const distort = (sign: 1 | -1) => {
    const out = new Float32Array(head.length);
    for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) {
      const S = 20, zs: number[] = [], vs: number[] = [], at = (a: Float32Array, j: number) => a[(k * ny + j) * nx + i];
      for (let s = 0; s <= (ny - 1) * S; s++) { const x = s / S, j0 = Math.min(Math.floor(x), ny - 2), f = x - j0, bx = at(b, j0) * (1 - f) + at(b, j0 + 1) * f, db = at(b, j0 + 1) - at(b, j0); zs.push(x + sign * bx); vs.push((at(head, j0) * (1 - f) + at(head, j0 + 1) * f) / (1 + sign * db)); }
      let q = 0;
      for (let j = 0; j < ny; j++) { while (q < zs.length - 1 && zs[q + 1] < j) q++; const t = zs[q + 1] > zs[q] ? Math.min(1, Math.max(0, (j - zs[q]) / (zs[q + 1] - zs[q]))) : 0; out[(k * ny + j) * nx + i] = j < zs[0] || j > zs[zs.length - 1] ? 0 : vs[q] * (1 - t) + vs[q + 1] * t; }
    }
    return out;
  };
  let seed = 1; const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  const gauss = () => { let u = 0; while (!u) u = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()); };
  const noisy = (a: Float32Array) => a.map((x) => Math.hypot(x + 5 * gauss(), 5 * gauss()));
  const truth: Motion = { t: [0.4, 0, -0.3], r: [0.5, 0.3, -0.4].map((d) => d * Math.PI / 180) as [number, number, number] };
  const back: Motion = { t: truth.t.map((x) => -x) as [number, number, number], r: truth.r.map((x) => -x) as [number, number, number] };
  const minus = noisy(distort(-1)), plus = noisy(moveRigid(distort(1), dims, voxel, back));
  const inside: number[] = []; for (let v = 0; v < head.length; v++) if (head[v] > 100) inside.push(v);
  const err = (f: FieldFit) => { const e = fieldAtCenters(f), d = inside.map((v) => Math.abs(e[v] - b[v]) * voxel[1]).sort((x, y) => x - y); return d[Math.floor(d.length / 2)]; };
  const f2 = estimateFieldWithMotion({ dims, plus, minus, axis: 1, voxel }), f1 = estimateField({ dims, plus, minus, axis: 1 });
  const e1 = err(f1), e2 = err(f2), e0 = err(fieldFromCenters(dims, 1, new Float32Array(b.length)));
  console.log(`phantom: field error (median) rule 1 ${e1.toFixed(2)} mm, rule 2 ${e2.toFixed(2)} mm (no correction ${e0.toFixed(2)}); movement found ${f2.motion.t.map((x) => x.toFixed(2)).join(", ")} mm, ${f2.motion.r.map((x) => (x * 180 / Math.PI).toFixed(2)).join(", ")}° (true 0.40, 0, -0.30 mm; 0.50, 0.30, -0.40°)`);
  assert(e2 < e1, `rule 2's field error ${e2.toFixed(2)} mm is not below rule 1's ${e1.toFixed(2)} mm`);
  assert(f2.motion.t[0] > 0.1 && f2.motion.t[2] < -0.05, "the movement across the phase-encoding axis was not found in its direction");
}});

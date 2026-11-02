// WHOLE-BRAIN TRACKING BY MICHAEL HALLE'S TRACTLINE, for the import (tracking rule 6; tracking-rules.ts). Ron,
// 2026-11-01: "Use his for import use ours for points." His two-tensor UKF follows the same method and settings as ours
// (rule 3 is "Mike's lead"), about three times faster on a whole brain (12.9 s against 39.2 s on the ds001226 tumor
// scans; albula-vs-tractline-2026-10-29.md), because of how his kernel shares the card's work (SPEED-2026-10-28.md).
// For a few thousand points -- a click in the module -- ours is faster (2.3 s against 13-15 s for 500), so the module
// keeps ours.
//
// What is handed over: the scan Albula has already corrected (head movement, distortion, aligned to the T1), cut to the
// rule's shell with its b = 0 images, and the brain the rule tracks in (SynthStrip's brain of the T1, inside the scan's
// data). He seeds every voxel of that brain as his own pipeline does (ukf.seeds: the original program's seeding), tracks
// on the graphics card (Metal, 32-bit) and returns RAS millimeters. Nothing of his pipeline before or after the tracking
// is used: Albula's distortion correction (field maps included) and Albula's naming stay.
//
// How: his Python package, pinned to one commit (TRACTLINE below), started with uvx as haversack is
// (desktop/haversack-proxy.ts); the scan goes to a folder of plain arrays and a header, the lines come back as one file.
import type { DiffusionSeries } from "./dwi.ts";

/** The tractline the import tracks with: one release, one commit (CHANGELOG 0.8.1, 2026-10-28). */
export const TRACTLINE = { version: "0.8.1", commit: "b8dcdff5451cb8c1d241c5898b2a6ba59a9cbe69" } as const;
const FROM = `tractline[cli] @ git+https://github.com/mhalle/tractline@${TRACTLINE.commit}`;

/** The tracker's settings (tracking-rules.ts TrackingRule.ukf): his `ukf.track` keywords. */
export interface TractlineSettings { seedingThreshold: number; stoppingFA: number; stoppingThreshold: number; recordLength: number }

export interface TractlineResult { sl: Float32Array[]; seconds: { total: number; load: number; track: number }; device: string; version: string }

// The Python side, written next to the arrays at each run: reads them, tracks, writes the lines. Kept here so the
// extension carries it (no separate file for a bundler to lose).
const SCRIPT = `
import sys, json, time, numpy as np, torch
from tractline import ukf
d = sys.argv[1]; h = json.load(open(d + "/header.json"))
nx, ny, nz, G = h["dims"]
t0 = time.time()
dwi = np.fromfile(d + "/dwi.f32", np.float32).reshape(G, nz, ny, nx).transpose(3, 2, 1, 0)   # (i, j, k, G)
mask = np.fromfile(d + "/mask.u8", np.uint8).reshape(nz, ny, nx).transpose(2, 1, 0)        # (i, j, k)
D = ukf.from_arrays(dwi, h["nrrd"], mask, device="cpu")
t1 = time.time()
mps = torch.backends.mps.is_available()
kw = dict(backend="metal", dtype=torch.float32, device="mps") if mps else dict(backend="torch", dtype=torch.float64)
lines, _ = ukf.track(D, seeding_threshold=h["seedingThreshold"], stopping_fa=h["stoppingFA"],
                     stopping_threshold=h["stoppingThreshold"], record_length=h["recordLength"], **kw)
t2 = time.time()
lines = [np.asarray(l, np.float32).reshape(-1, 3) for l in lines if len(l)]
with open(d + "/lines.bin", "wb") as f:
    np.asarray([len(lines)], np.uint32).tofile(f)
    np.asarray([len(l) for l in lines], np.uint32).tofile(f)
    for l in lines: l.tofile(f)
json.dump({"load": t1 - t0, "track": t2 - t1, "device": "mps" if mps else "cpu"}, open(d + "/done.json", "w"))
`;

/** `uvx` where the installers put it (the Finder starts the app with a bare PATH), as desktop/haversack-proxy.ts looks. */
function findUvx(): string | undefined {
  const home = Deno.env.get("HOME") ?? "";
  for (const c of [`${home}/.local/bin/uvx`, `${home}/miniconda3/bin/uvx`, `${home}/anaconda3/bin/uvx`, "/opt/homebrew/bin/uvx", "/usr/local/bin/uvx"]) {
    try { if (Deno.statSync(c).isFile) return c; } catch { /* not there */ }
  }
  return undefined;
}

/** The header his reader takes (ukf.from_arrays: pynrrd's dict of an NRRD DWI), in RAS: the largest b as DWMRI_b-value,
 *  each gradient scaled by sqrt(b / largest), the measurement frame the identity -- dwi-nrrd-write.ts's convention. */
function nrrdHeader(s: DiffusionSeries): Record<string, unknown> {
  const M = s.ijkToRAS, bmax = Math.max(...s.bValues);
  const h: Record<string, unknown> = {
    "space": "right-anterior-superior", "space origin": [M[3], M[7], M[11]],
    "space directions": [[M[0], M[4], M[8]], [M[1], M[5], M[9]], [M[2], M[6], M[10]]],
    "measurement frame": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "DWMRI_b-value": String(bmax),
  };
  s.gradients.forEach((g, i) => {
    const k = s.bValues[i] > 0 && bmax > 0 ? Math.sqrt(s.bValues[i] / bmax) : 0;
    h[`DWMRI_gradient_${String(i).padStart(4, "0")}`] = `${(g[0] * k).toFixed(8)} ${(g[1] * k).toFixed(8)} ${(g[2] * k).toFixed(8)}`;
  });
  return h;
}

/**
 * Track the whole brain with tractline: every voxel of `brain` a seed (his seeding), tracked inside it. `s` is the series
 * already cut to the rule's shell (with its b = 0 images); `brain` is on its grid, (k, j, i) order, 1 inside.
 */
export async function trackWithTractline(s: DiffusionSeries, brain: Uint8Array, settings: TractlineSettings, opts: { uvx?: string; offline?: boolean } = {}): Promise<TractlineResult> {
  const t0 = performance.now();
  const uvx = opts.uvx ?? findUvx();
  if (!uvx) throw new Error("tractline needs uv (uvx was not found in ~/.local/bin, ~/miniconda3/bin, /opt/homebrew/bin, /usr/local/bin)");
  const [nx, ny, nz] = s.volumes[0].dims, n3 = nx * ny * nz, G = s.volumes.length;
  const dir = await Deno.makeTempDir({ prefix: "albula-tractline-" });
  try {
    const dwi = new Float32Array(n3 * G);
    s.volumes.forEach((v, q) => { const d = v.data; for (let i = 0; i < n3; i++) dwi[q * n3 + i] = Number(d[i]); });
    await Deno.writeFile(`${dir}/dwi.f32`, new Uint8Array(dwi.buffer));
    await Deno.writeFile(`${dir}/mask.u8`, brain);
    await Deno.writeTextFile(`${dir}/header.json`, JSON.stringify({ dims: [nx, ny, nz, G], nrrd: nrrdHeader(s), ...settings }));
    await Deno.writeTextFile(`${dir}/track.py`, SCRIPT);
    const args = ["--python", "3.12", ...(opts.offline ? ["--offline"] : []), "--from", FROM, "python", "-I", `${dir}/track.py`, dir];
    const r = await new Deno.Command(uvx, { args, stdout: "piped", stderr: "piped" }).output();
    if (!r.success) throw new Error(`tractline ${TRACTLINE.version} failed: ${new TextDecoder().decode(r.stderr).trim().split("\n").slice(-3).join(" / ")}`);
    const done = JSON.parse(await Deno.readTextFile(`${dir}/done.json`)) as { load: number; track: number; device: string };
    const bin = await Deno.readFile(`${dir}/lines.bin`), u = new Uint32Array(bin.buffer, bin.byteOffset, Math.floor(bin.byteLength / 4));
    const n = u[0], sl: Float32Array[] = [];
    let at = (1 + n) * 4;
    for (let i = 0; i < n; i++) {
      const len = u[1 + i] * 3;
      sl.push(new Float32Array(bin.buffer.slice(bin.byteOffset + at, bin.byteOffset + at + len * 4)));
      at += len * 4;
    }
    return { sl, seconds: { total: (performance.now() - t0) / 1000, load: done.load, track: done.track }, device: done.device, version: `${TRACTLINE.version} (${TRACTLINE.commit.slice(0, 7)})` };
  } finally {
    await Deno.remove(dir, { recursive: true }).catch(() => {});
  }
}

// THE PER-IMAGE PIECES OF THE HEAD-MOVEMENT CORRECTION IN A THREAD (SPEED-2026-10-28.md, A3): motion.ts's own functions
// -- read an image back through its move, align it to its target -- on the scan's images in shared memory. The same
// functions on the same numbers as on the main thread, so the moves are bit-identical.
import { alignOne, readMoved, scatterTo, targetsAt, type Level, type Move, type Points } from "./motion.ts";

let st: { full: { dims: [number, number, number]; ijkToRAS: number[] }; Si: number[]; levels: Level[]; regP: Points; regIdx: Int32Array; nv: number; volumes: ArrayLike<number>[] } | undefined;
self.onmessage = (e: MessageEvent) => {
  const d = e.data as { init?: typeof st; kind?: "align" | "atLevels"; i?: number; m?: Move; targets?: Float32Array[]; start?: Move; e?: [number, number, number]; hold?: boolean };
  if (d.init) { st = d.init; self.postMessage({ ready: true }); return; }
  try {
    const s = st!, img = s.volumes[d.i!];
    if (d.kind === "atLevels") self.postMessage({ ok: targetsAt(s.levels, s.full, scatterTo(s.regIdx, s.nv, readMoved(img, s.full.dims, s.Si, s.regP, d.m!))) });
    else self.postMessage({ ok: alignOne(img, s.full, s.levels, d.targets!, d.start!, d.e, d.hold ?? false) });
  } catch (err) { self.postMessage({ error: (err as Error).message }); }
};

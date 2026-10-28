// ONE RIGID ALIGNMENT IN ITS OWN THREAD (SPEED-2026-10-28.md, A1/A2; Ron, 2026-10-28: "do all including the non gpu
// optimization"). The same function, on the same numbers, as on the main thread -- registration.ts rigidToT1 -- so the
// result is bit-identical; independent alignments (the head frame's starting tilts, the mirror test) then run side by
// side instead of one after the other. A typed array arrives as an exact copy.
import { rigidToT1, rigidToT1Wide, type Grid3 } from "./registration.ts";

self.onmessage = async (e: MessageEvent) => {
  const { moving, fixed, mask, opts, wide, brainT1 } = e.data as { moving: Grid3; fixed: Grid3; mask: Uint8Array; opts: Parameters<typeof rigidToT1>[3]; wide?: boolean; brainT1?: Grid3 };
  try { self.postMessage({ ok: wide ? await rigidToT1Wide(moving, fixed, mask, brainT1, opts) : await rigidToT1(moving, fixed, mask, opts) }); }
  catch (err) { self.postMessage({ error: (err as Error).message }); }
};

// THE THREADS ARE THE IMPORT JOB'S, NEVER THE WINDOW'S (SPEED-2026-10-28.md in the workspace; the critic, 2026-10-28,
// speed part A, finding 10: protected by convention only). The window's module must reach none of the thread modules and
// hand no `threads` to the pipeline, so it runs in turn as before; the import job is the one that starts them.
//   deno test -A --no-check --config ../../src/SlicerLive/deno.jsonc threads.test.ts
import { assert } from "jsr:@std/assert@1";
import { importGraph } from "./import-job.ts";

const THREADED = ["rigid-worker.ts", "motion-worker.ts", "second-opinion-job.ts"];

Deno.test("the window's module reaches no thread module and asks for no threads", async () => {
  const graph = await importGraph(new URL("./module.ts", import.meta.url));
  const reached = graph.filter((u) => THREADED.some((t) => u.endsWith(`/${t}`)));
  assert(!reached.length, `module.ts reaches ${reached.join(", ")}`);
  const text = Deno.readTextFileSync(new URL("./module.ts", import.meta.url)).replace(/\/\/.*$/gm, "");
  assert(!/\bthreads\s*:/.test(text), "module.ts passes threads to the pipeline");
});

Deno.test("the import job does start them (so the check above checks something)", async () => {
  const graph = await importGraph(new URL("./import-job-main.ts", import.meta.url));
  for (const t of ["rigid-worker.ts", "motion-worker.ts"]) assert(graph.some((u) => u.endsWith(`/${t}`)), `import-job-main.ts does not reach ${t}`);
});

// A THREAD THAT FAILS IS DONE IN TURN, AND ENDS NOTHING (the critic, 2026-10-28, speed part A, findings 1-3): an alignment
// to a T1 with the mirror test, its threads pointed at a module that does not exist, gives the same answer as in turn.
Deno.test("a failing alignment thread falls back to the same answer, without ending the program", async () => {
  const { alignToT1 } = await import("./registration.ts");
  const { inThread } = await import("./head-frame.ts");
  // A phantom head: a bright ellipsoid with an inner structure, the T1 and the b = 0 the same shape, shifted 2 voxels.
  const dims: [number, number, number] = [40, 44, 36], n = dims[0] * dims[1] * dims[2], M = [2, 0, 0, -40, 0, 2, 0, -44, 0, 0, 2, -36, 0, 0, 0, 1];
  const img = (dx: number) => { const d = new Float32Array(n); for (let k = 0; k < dims[2]; k++) for (let j = 0; j < dims[1]; j++) for (let i = 0; i < dims[0]; i++) { const x = (i - 20 - dx) / 14, y = (j - 22) / 17, z = (k - 18) / 12, r = x * x + y * y + z * z; d[(k * dims[1] + j) * dims[0] + i] = r < 1 ? (r < 0.2 ? 300 : 100 + 50 * Math.sin(3 * x) * Math.cos(2 * y)) + (x > 0 ? 20 : 0) : 0; } return d; };
  const t1 = { dims, ijkToRAS: M, data: img(0) }, b0 = img(2), mask = Uint8Array.from(b0, (v) => (v > 0 ? 1 : 0)), brainT1 = { dims, ijkToRAS: M, data: Uint8Array.from(t1.data, (v) => (v > 0 ? 1 : 0)) };
  const dwi = { bValues: [0], gradients: [[0, 0, 0]], volumes: [{ dims, ijkToRAS: M, data: b0, dtype: "<f4" }] } as never;
  const bad = new URL("./no-such-worker.ts", import.meta.url);
  let rejected = false; await inThread(bad, t1, t1, mask, {}).catch(() => { rejected = true; });
  assert(rejected, "a thread that cannot start rejects");
  const motion = { moves: [], b0, mask };
  const inTurn = await alignToT1(dwi, t1, undefined, undefined, motion, { brainT1, check: 2 });
  const fallen = await alignToT1(dwi, t1, undefined, undefined, motion, { brainT1, check: 2, threads: { workerUrl: bad } });
  assert(JSON.stringify(inTurn.T) === JSON.stringify(fallen.T) && inTurn.mirrorRatio === fallen.mirrorRatio, "the fallback gives the same alignment and mirror ratio");
});

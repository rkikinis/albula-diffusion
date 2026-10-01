// A CSD worker (csd-volume.ts): builds the model once, then fits the chunks of voxels it is sent.
import { csdFitFast, csdGram, csdModel, type CsdModel, type Kernel } from "./csd.ts";

let model: CsdModel | undefined, gram: Float64Array | undefined;
self.onmessage = (e: MessageEvent) => {
  const d = e.data as { init?: { gradients: Float64Array; bValues: Float64Array; kernel: Kernel; constraint: Float64Array }; signals?: Float32Array; count?: number };
  if (d.init) { model = csdModel(d.init.gradients, d.init.bValues, d.init.kernel, d.init.constraint); gram = csdGram(model); return; }
  const m = model!, rows = m.rows, out = new Float32Array(d.count! * m.nx);
  let warm: number[] | undefined;
  for (let i = 0; i < d.count!; i++) {
    const r = csdFitFast(m, gram!, d.signals!.subarray(i * rows, (i + 1) * rows), warm);
    warm = r.passive; out.set(r.x, i * m.nx);
  }
  (self as unknown as Worker).postMessage(out, [out.buffer]);
};

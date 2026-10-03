// NAMING ADDED STREAMLINES AGAINST A WHOLE-BRAIN RUN (nameAgainst; Ron, 2026-10-01: "Add lines" to chosen tracts).
// The added streamlines must be named as the whole-brain run would name them: a copy of a run's own streamline gets that
// streamline's name, and the run's own names do not change when streamlines are added. On PAT16's reference streamlines.
//   deno test -A --no-check --unstable-webgpu extensions/diffusion/tractcloud/name-tracts.test.ts
import { assert, assertEquals } from "jsr:@std/assert@1";
import { ABSENT, testData } from "albula/testing";
import { loadModel, type ModelJson } from "./tractcloud.ts";
import { nameAgainst, nameTracts } from "./name-tracts.ts";

const REF = testData("tractcloud-reference") ?? ABSENT;
const have = (() => { try { return Deno.statSync(REF + "streamlines.f32").isFile; } catch { return false; } })();
const HERE = new URL("./model/", import.meta.url);

Deno.test({ name: "added copies get their originals' names; the run's own names are unchanged (PAT16)", ignore: !have || !("gpu" in navigator), fn: async () => {
  const model = loadModel(Deno.readFileSync(new URL("weights.f32", HERE)).buffer, JSON.parse(Deno.readTextFileSync(new URL("model.json", HERE))) as ModelJson);
  const raw = new Float32Array(Deno.readFileSync(REF + "streamlines.f32").buffer), sl: Float32Array[] = []; let o = 1;
  for (let i = 0; i < raw[0]; i++) { const n = raw[o]; sl.push(raw.slice(o + 1, o + 1 + 3 * n)); o += 1 + 3 * n; }
  const device = await (await navigator.gpu.requestAdapter())!.requestDevice();
  const alone = await nameTracts(device, model, sl);
  // Every 7th streamline added again, as copies -- and, so the added set is dense in one place, 200 more copies of one.
  const picks = sl.map((_, i) => i).filter((i) => i % 7 === 0), dense = Array.from({ length: 200 }, () => 3);
  const added = [...picks, ...dense].map((i) => sl[i].slice());
  const r = await nameAgainst(device, model, sl, added);
  assertEquals(Array.from(r.context.tract), Array.from(alone.tract), "the run's own names");
  assertEquals(Array.from(r.context.side), Array.from(alone.side), "the run's own sides");
  let same = 0;
  [...picks, ...dense].forEach((i, k) => { if (r.added.tract[k] === alone.tract[i] && r.added.side[k] === alone.side[i]) same++; });
  console.log(`added copies named as their originals: ${same} of ${added.length}`);
  assertEquals(same, added.length);
  assert(picks.length > 100);
  device.destroy();
} });

Deno.test({ name: "RapidParc: the run's own names are unchanged by added streamlines; added copies mostly keep their originals' names (PAT16)", ignore: !have || !("gpu" in navigator), fn: async () => {
  const { loadRapidParc } = await import("../rapidparc/rapidparc.ts");
  const model = loadModel(Deno.readFileSync(new URL("weights.f32", HERE)).buffer, JSON.parse(Deno.readTextFileSync(new URL("model.json", HERE))) as ModelJson);
  const rapidParc = loadRapidParc(Deno.readFileSync(new URL("../rapidparc/model/rapidparc.safetensors", import.meta.url)).buffer);
  const raw = new Float32Array(Deno.readFileSync(REF + "streamlines.f32").buffer), sl: Float32Array[] = []; let o = 1;
  for (let i = 0; i < raw[0]; i++) { const n = raw[o]; sl.push(raw.slice(o + 1, o + 1 + 3 * n)); o += 1 + 3 * n; }
  const device = await (await navigator.gpu.requestAdapter())!.requestDevice();
  const alone = await nameTracts(device, model, sl, { rapidParc });
  const picks = sl.map((_, i) => i).filter((i) => i % 7 === 0), dense = Array.from({ length: 200 }, () => 3);
  const added = [...picks, ...dense].map((i) => sl[i].slice());
  const r = await nameAgainst(device, model, sl, added, { rapidParc });
  assertEquals(Array.from(r.context.tract), Array.from(alone.tract), "the run's own names");
  // An added streamline is named in a group that is not its original's (500 added, 1,500 drawn from the run), so its
  // context differs and its name can: RapidParc's own draws agree on 98 % of tracts (tractline's docs/labelers.md).
  let same = 0, named = 0;
  [...picks, ...dense].forEach((i, k) => { if (alone.tract[i] >= 0) { named++; if (r.added.tract[k] === alone.tract[i] && r.added.side[k] === alone.side[i]) same++; } });
  console.log(`RapidParc: added copies named as their originals: ${same} of ${named}`);
  assert(same >= 0.95 * named, `${same} of ${named}`);
  device.destroy();
} });

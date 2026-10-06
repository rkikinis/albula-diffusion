// THE IMPORT-TIME JOB AS A PROGRAM (import-job.ts is the library; Contents/docs/DMRI-AT-IMPORT.md in the workspace).
// Albula's server starts it after a diffusion scan is indexed, and for the sweep of a whole database; it can also be run
// by hand against a running server:
//
//   deno run -A --unstable-webgpu --config Contents/src/SlicerLive/deno.jsonc Contents/extensions/diffusion/import-job-main.ts \
//     --server http://127.0.0.1:<port> --db <database id> [--series <SeriesInstanceUID>] [--dry-run] [--force] [--assets <dir>]
//
// It prints ONE JSON OBJECT A LINE on standard output -- {"event": "plan" | "case" | "progress" | "done" | "error", ...} --
// which the server reads for its status route (the top-bar line); everything a person reads is in plain words in "said".
// `--assets` is where the networks' weights are (the app's vendor/diffusion/); by default beside this file.
import { parseArgs } from "jsr:@std/cli@1/parse-args";
import { setDicomLibrary, dcmjs } from "albula/server";
import "./hooks.ts";
import { loadModel, type ModelJson } from "./tractcloud/tractcloud.ts";
import { loadRapidParc } from "./rapidparc/rapidparc.ts";
import { makeTracts, planDatabase } from "./import-job.ts";

const args = parseArgs(Deno.args, { string: ["server", "db", "series", "assets"], boolean: ["dry-run", "force"] });
const say = (o: Record<string, unknown>) => console.log(JSON.stringify({ at: new Date().toISOString(), ...o }));
if (!args.server || !args.db) { say({ event: "error", said: "needs --server and --db" }); Deno.exit(2); }
const server = args.server.replace(/\/+$/, "");

try {
  setDicomLibrary(dcmjs);
  // The database's folder from the server's own list (GET /_db), so the job and the server agree on what "--db" means.
  const list = await (await fetch(`${server}/_db`)).json() as { databases: { id: string; path: string }[] };
  const db = list.databases.find((d) => d.id === args.db);
  if (!db) throw new Error(`no database "${args.db}" is registered with the server`);
  // The networks: TractCloud's table (the tract names and the 1,600 → 43 map) and RapidParc's weights, read once.
  const assets = args.assets ? args.assets.replace(/\/+$/, "") : undefined;
  const at = (group: string, file: string) => assets ? `${assets}/${group}/${file}` : new URL(`./${group}/model/${file}`, import.meta.url);
  const model = loadModel(Deno.readFileSync(at("tractcloud", "weights.f32")).buffer, JSON.parse(Deno.readTextFileSync(at("tractcloud", "model.json"))) as ModelJson);
  const rpBytes = Deno.readFileSync(at("rapidparc", "rapidparc.safetensors"));
  const rpHash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", rpBytes))].slice(0, 6).map((b) => b.toString(16).padStart(2, "0")).join("");
  const labeler = { model: loadRapidParc(rpBytes.buffer), name: `rapidparc ${rpHash}` };
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error("no graphics card is available to this program");
  const L = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage, maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize, maxBufferSize: L.maxBufferSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize } });

  const { plans, rows } = await planDatabase(db.path, (line) => say({ event: "progress", said: line }), args.series);
  say({ event: "plan", cases: plans.length, said: `${plans.length} diffusion scan${plans.length === 1 ? "" : "s"} to check` });
  let made = 0, current = 0, waiting = 0;
  for (const [i, plan] of plans.entries()) {
    const t0 = performance.now();
    try {
      const out = await makeTracts(db.path, db.id, server, plan, rows, device, model, labeler, (line) => say({ event: "progress", case: i + 1, of: plans.length, series: plan.dwi.uid, said: line }),
        { force: args.force, dryRun: args["dry-run"] });
      if (out.state === "made") made++; else if (out.state === "current") current++; else waiting++;
      say({ event: "case", case: i + 1, of: plans.length, series: plan.dwi.uid, study: plan.study, ...out, wall: +((performance.now() - t0) / 1000).toFixed(1),
        said: out.state === "made" ? `fiber tracts made: ${out.said}` : out.state === "current" ? "fiber tracts already made with the current rules" : out.why });
    } catch (e) {
      waiting++;
      say({ event: "case", case: i + 1, of: plans.length, series: plan.dwi.uid, state: "failed", said: `fiber tracts could not be made: ${(e as Error).message}` });
    }
  }
  say({ event: "done", made, current, waiting, said: `fiber tracts: ${made} made, ${current} already current, ${waiting} waiting or failed` });
  device.destroy();
} catch (e) {
  say({ event: "error", said: (e as Error).message });
  Deno.exit(1);
}

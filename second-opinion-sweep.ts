// THE SECOND OPINION FOR THE TRACT SETS ALREADY STORED (Ron, 2026-10-28: "yes, close both gaps"). From 2026-10-28 the
// import job runs dcm2niix on every diffusion scan it makes tracts from and keeps the answer in the set's record
// (import-job.ts secondOpinionOf). The sets made before have none, and a stored set's record is not rewritten; so this
// runs the job's own second opinion once on each such set's diffusion scan, keeps the answer as the scan's series
// attribute (SECOND_OPINION_KEY: what the module and the descriptions read when the record has none), and writes the
// descriptions of the scan's sets again. A scan already answered for this dcm2niix version is skipped.
//
//   deno run -A --no-check --config ../../src/SlicerLive/deno.jsonc second-opinion-sweep.ts --server http://127.0.0.1:<port> --db <id> [--dry-run]
//
// It prints one JSON object a line per scan: the tract sets' series numbers and the answer.
import { parseArgs } from "jsr:@std/cli@1/parse-args";
import { setDicomLibrary, dcmjs, indexSeries } from "albula/server";
import "./hooks.ts";
import { dicomToTracts } from "./tracts-dicom.ts";
import { fromDicomVolumes } from "./dwi.ts";
import { readSeries, secondOpinionOf, TRACTS_DESCRIPTION, withScanChecks, writeDescription } from "./import-job.ts";
import { DCM2NIIX_VERSION, SECOND_OPINION_KEY } from "./second-opinion.ts";

setDicomLibrary(dcmjs);
const a = parseArgs(Deno.args, { string: ["server", "db"], boolean: ["dry-run"] });
const server = a.server!.replace(/\/+$/, "");
const db = ((await (await fetch(`${server}/_db`)).json()) as { databases: { id: string; path: string }[] }).databases.find((d) => d.id === a.db);
if (!db) { console.error(`no database "${a.db}"`); Deno.exit(2); }
const attributes = async () => ((await (await fetch(`${server}/_db/${encodeURIComponent(db.id)}/_provenance`)).json()) as { attributes?: { uid: string; key: string; value: string }[] }).attributes ?? [];
const rows = await indexSeries(db.path);
const byUid = new Map(rows.map((r) => [r.seriesUID, r]));
// The stored sets, by the diffusion scan they were made from.
const sets = new Map<string, { uid: string; number: string; rec: Record<string, unknown>; tracts: Awaited<ReturnType<typeof dicomToTracts>>["sets"] }[]>();
for (const r of rows.filter((x) => x.description === TRACTS_DESCRIPTION)) {
  try {
    const t = await dicomToTracts(await Deno.readFile(`${db.path}/SlicerAlbula-SEG/tracts-${r.seriesUID}.dcm`));
    const rec = (t.provenance ?? {}) as Record<string, unknown>, scan = String((rec.inputs as { diffusion?: string } | undefined)?.diffusion ?? t.referencedSeries ?? "");
    if (!scan) continue;
    if (!sets.has(scan)) sets.set(scan, []);
    sets.get(scan)!.push({ uid: r.seriesUID, number: String(r.seriesNumber), rec, tracts: t.sets });
  } catch (e) { console.log(JSON.stringify({ series: r.seriesNumber, read: false, why: (e as Error).message })); }
}
const known = new Map((await attributes()).filter((x) => x.key === SECOND_OPINION_KEY).map((x) => [x.uid, x.value]));
for (const [scan, list] of sets) {
  const numbers = list.map((s) => s.number);
  // Every set of this scan has its own answer in its record: nothing to add.
  if (list.every((s) => s.rec.secondOpinion)) { console.log(JSON.stringify({ sets: numbers, skipped: "in the record" })); continue; }
  let answer: Record<string, unknown> | undefined;
  const had = known.get(scan);
  if (had && (JSON.parse(had) as { version?: string }).version === DCM2NIIX_VERSION) answer = JSON.parse(had);
  else {
    const row = byUid.get(scan);
    if (!row) { console.log(JSON.stringify({ sets: numbers, checked: false, why: "the diffusion scan is not in this database's index" })); continue; }
    const s = await readSeries(db.path, row).catch((e) => e as Error);
    if (s instanceof Error) { console.log(JSON.stringify({ sets: numbers, checked: false, why: s.message })); continue; }
    answer = await secondOpinionOf(db.path, scan, fromDicomVolumes(s.frames, row.description));
    if (!a["dry-run"]) {
      const r = await fetch(`${server}/_db/${encodeURIComponent(db.id)}/_attribute`, { method: "PUT", headers: { "content-type": "application/json" },
        body: JSON.stringify({ uid: scan, key: SECOND_OPINION_KEY, value: JSON.stringify(answer), source: "albula-diffusion" }) });
      await r.body?.cancel();
      if (!r.ok) { console.log(JSON.stringify({ sets: numbers, checked: true, stored: false, why: `the server said ${r.status}` })); continue; }
    }
  }
  if (!a["dry-run"]) for (const s of list) await writeDescription(server, db.id, s.uid, withScanChecks(s.rec, answer), s.tracts).catch(() => {});
  console.log(JSON.stringify({ sets: numbers, scan: row(scan), ...answer }));
}
function row(uid: string) { const r = byUid.get(uid); return r ? `${r.seriesNumber} ${r.description}` : uid.slice(-8); }

// THE DESCRIPTIONS OF THE TRACT SETS ALREADY STORED (tract-set-description.ts; Ron, 2026-10-27): every tract set of a
// database read once -- its record and its motor-pathway counts -- and its title, line and tooltip written as its series
// attributes, as the import job now writes them for every new set. Run again after any tool stores a set.
//
//   deno run -A --no-check --config ../../src/SlicerLive/deno.jsonc describe-tract-sets.ts --server http://127.0.0.1:<port> --db <id> [--dry-run]
import { parseArgs } from "jsr:@std/cli@1/parse-args";
import { setDicomLibrary, dcmjs, indexSeries } from "albula/server";
import { dicomToTracts } from "./tracts-dicom.ts";
import { TRACTS_DESCRIPTION, writeDescription, motorCounts, withScanChecks } from "./import-job.ts";
import { SECOND_OPINION_KEY } from "./second-opinion.ts";
import { describeTractSet } from "./tract-set-description.ts";

setDicomLibrary(dcmjs);
const a = parseArgs(Deno.args, { string: ["server", "db"], boolean: ["dry-run"] });
const server = a.server!.replace(/\/+$/, "");
const db = ((await (await fetch(`${server}/_db`)).json()) as { databases: { id: string; path: string }[] }).databases.find((d) => d.id === a.db);
if (!db) { console.error(`no database "${a.db}"`); Deno.exit(2); }
const rows = (await indexSeries(db.path)).filter((r: { description: string }) => r.description === TRACTS_DESCRIPTION);
// dcm2niix's answer for a scan whose sets were made before the job ran it (second-opinion-sweep.ts), so a set's
// description keeps it.
const scanAnswers = new Map((((await (await fetch(`${server}/_db/${encodeURIComponent(db.id)}/_provenance`)).json()) as { attributes?: { uid: string; key: string; value: string }[] }).attributes ?? [])
  .filter((x) => x.key === SECOND_OPINION_KEY).map((x) => [x.uid, JSON.parse(x.value) as Record<string, unknown>]));
for (const r of rows) {
  try {
    const t = await dicomToTracts(await Deno.readFile(`${db.path}/SlicerAlbula-SEG/tracts-${r.seriesUID}.dcm`));
    const own = (t.provenance ?? {}) as Record<string, unknown>;
    const rec = withScanChecks(own, scanAnswers.get(String((own.inputs as { diffusion?: string } | undefined)?.diffusion ?? t.referencedSeries ?? "")));
    if (a["dry-run"]) { const d = describeTractSet(rec, motorCounts(t.sets)); console.log(JSON.stringify({ series: r.seriesNumber, title: d.title, line: d.line })); continue; }
    await writeDescription(server, db.id, r.seriesUID, rec, t.sets);
    console.log(JSON.stringify({ series: r.seriesNumber, described: true }));
  } catch (e) { console.log(JSON.stringify({ series: r.seriesNumber, described: false, why: (e as Error).message })); }
}

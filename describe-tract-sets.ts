// THE DESCRIPTIONS OF THE TRACT SETS ALREADY STORED (tract-set-description.ts; Ron, 2026-10-27): every tract set of a
// database read once -- its record and its motor-pathway counts -- and its title, line and tooltip written as its series
// attributes, as the import job now writes them for every new set. Run again after any tool stores a set.
//
//   deno run -A --no-check --config ../../src/SlicerLive/deno.jsonc describe-tract-sets.ts --server http://127.0.0.1:<port> --db <id> [--dry-run]
import { parseArgs } from "jsr:@std/cli@1/parse-args";
import { setDicomLibrary, dcmjs, indexSeries } from "albula/server";
import { dicomToTracts } from "./tracts-dicom.ts";
import { TRACTS_DESCRIPTION, writeDescription, motorCounts } from "./import-job.ts";
import { describeTractSet } from "./tract-set-description.ts";

setDicomLibrary(dcmjs);
const a = parseArgs(Deno.args, { string: ["server", "db"], boolean: ["dry-run"] });
const server = a.server!.replace(/\/+$/, "");
const db = ((await (await fetch(`${server}/_db`)).json()) as { databases: { id: string; path: string }[] }).databases.find((d) => d.id === a.db);
if (!db) { console.error(`no database "${a.db}"`); Deno.exit(2); }
const rows = (await indexSeries(db.path)).filter((r: { description: string }) => r.description === TRACTS_DESCRIPTION);
for (const r of rows) {
  try {
    const t = await dicomToTracts(await Deno.readFile(`${db.path}/SlicerAlbula-SEG/tracts-${r.seriesUID}.dcm`));
    const rec = (t.provenance ?? {}) as Record<string, unknown>;
    if (a["dry-run"]) { const d = describeTractSet(rec, motorCounts(t.sets)); console.log(JSON.stringify({ series: r.seriesNumber, title: d.title, line: d.line })); continue; }
    await writeDescription(server, db.id, r.seriesUID, rec, t.sets);
    console.log(JSON.stringify({ series: r.seriesNumber, described: true }));
  } catch (e) { console.log(JSON.stringify({ series: r.seriesNumber, described: false, why: (e as Error).message })); }
}

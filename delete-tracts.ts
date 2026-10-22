// DELETE TRACT SETS, RECORDED FIRST (Ron, 2026-10-22: "Once everything is properly documented, you should delete the
// garbage"). For each tract set named: its whole stored record (the provenance inside the file: the scans it was made
// from, every rule and setting, the code and naming versions, the times, what was corrected) and the maps beside it are
// copied into the database's run log (run-log.ts, event "deleted") with the reason and the evidence; then the server
// deletes it (DELETE /_db/<id>/_series/<uid>: its index rows, its file, its provenance link, its cached maps -- one backup
// and one audit). A set whose record cannot be written to the log is not deleted.
//
//   deno run -A --no-check --config ../../src/SlicerLive/deno.jsonc delete-tracts.ts --server http://127.0.0.1:<port> \
//     --db <database id> --sets <uid,uid,...> --reason "<why>" --evidence "<where it is shown>" [--dry-run]
import { parseArgs } from "jsr:@std/cli@1/parse-args";
import { setDicomLibrary, dcmjs } from "albula/server";
import { adPath, b0Path, colorFaBrainstemPath, colorFaPath, colorFaTalairachPath, dicomToTracts, faPath, mdPath, rdPath } from "./tracts-dicom.ts";
/** Every map the job stores beside a tract set (tracts-dicom.ts). */
const MAPS_OF = (uid: string) => [colorFaPath, colorFaTalairachPath, colorFaBrainstemPath, faPath, mdPath, adPath, rdPath, b0Path].map((f) => f(uid));
import { appendRunLog } from "./run-log.ts";

setDicomLibrary(dcmjs);
const a = parseArgs(Deno.args, { string: ["server", "db", "sets", "reason", "evidence"], boolean: ["dry-run"] });
if (!a.server || !a.db || !a.sets || !a.reason || !a.evidence) { console.error("needs --server --db --sets --reason --evidence"); Deno.exit(2); }
const server = a.server.replace(/\/+$/, "");
const dbs = await fetch(`${server}/_db`).then((r) => r.json()) as { databases: { id: string; path: string }[] };
const db = dbs.databases.find((d) => d.id === a.db);
if (!db) { console.error(`no database ${a.db} at ${server}`); Deno.exit(2); }
const sets = a.sets.split(",").map((s) => s.trim()).filter(Boolean);
let deleted = 0, refused = 0;
for (const uid of sets) {
  const file = `${db.path}/SlicerAlbula-SEG/tracts-${uid}.dcm`;
  let bytes: Uint8Array;
  try { bytes = await Deno.readFile(file); } catch { console.log(JSON.stringify({ set: uid, deleted: false, why: "no such tract set file" })); refused++; continue; }
  const t = await dicomToTracts(bytes);
  const maps = MAPS_OF(uid).filter((p) => { try { Deno.statSync(`${db.path}/${p}`); return true; } catch { return false; } });
  const record = {
    set: uid, scan: t.referencedSeries ?? null, reason: a.reason, evidence: a.evidence,
    streamlines: t.sets.reduce((n, s) => n + s.streamlines.length, 0), tracts: t.sets.length, maps,
    storedRecord: t.provenance ?? null,
  };
  if (a["dry-run"]) { console.log(JSON.stringify({ set: uid, wouldDelete: true, scan: record.scan, made: (t.provenance as { made?: string } | undefined)?.made, maps: maps.length })); continue; }
  if (!await appendRunLog(db.path, "deleted", record)) { console.log(JSON.stringify({ set: uid, deleted: false, why: "its record could not be written to the run log" })); refused++; continue; }
  const r = await fetch(`${server}/_db/${encodeURIComponent(db.id)}/_series/${uid}`, { method: "DELETE" });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) {
    await appendRunLog(db.path, "note", { about: uid, said: `the deletion recorded above did NOT happen: the server said ${r.status} ${(body as { error?: string }).error ?? ""}` });
    console.log(JSON.stringify({ set: uid, deleted: false, why: (body as { error?: string }).error ?? r.status }));
    refused++; continue;
  }
  deleted++;
  console.log(JSON.stringify({ set: uid, deleted: true, scan: record.scan, maps: maps.length }));
}
console.log(JSON.stringify({ done: true, deleted, refused, of: sets.length }));

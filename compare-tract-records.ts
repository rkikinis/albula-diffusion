// TWO STORED TRACT SETS' RECORDS, SIDE BY SIDE (record-diff.ts; Ron, 2026-10-27, item 4). Each argument is a tract set's
// DICOM file, or its SeriesInstanceUID with --db (the database folder): the file is then <db>/SlicerAlbula-SEG/tracts-<uid>.dcm.
// Prints every difference in inputs, rules, versions, scanner facts, the namer's settings, corrections and counts, times
// last; --json prints them as data.
//
//   deno run -A --no-check --config ../../src/SlicerLive/deno.jsonc compare-tract-records.ts [--db <folder>] <set A> <set B> [--json]
import { parseArgs } from "jsr:@std/cli@1/parse-args";
import { setDicomLibrary, dcmjs } from "albula/server";
import { dicomToTracts } from "./tracts-dicom.ts";
import { recordDiff, recordDiffText } from "./record-diff.ts";

setDicomLibrary(dcmjs);
const a = parseArgs(Deno.args, { string: ["db"], boolean: ["json"] });
const [A, B] = a._.map(String);
if (!A || !B) { console.error("needs two tract sets (files, or SeriesInstanceUIDs with --db)"); Deno.exit(2); }
const fileOf = (x: string) => /\.dcm$/i.test(x) || x.includes("/") ? x : a.db ? `${a.db}/SlicerAlbula-SEG/tracts-${x}.dcm` : x;
const recordOf = async (x: string) => {
  const t = await dicomToTracts(await Deno.readFile(fileOf(x)));
  return (t.provenance ?? {}) as Record<string, unknown>;
};
const diffs = recordDiff(await recordOf(A), await recordOf(B));
const short = (x: string) => x.replace(/^.*tracts-/, "").replace(/\.dcm$/i, "").slice(-12);
console.log(a.json ? JSON.stringify(diffs, null, 1) : recordDiffText(diffs, [short(A), short(B)]));

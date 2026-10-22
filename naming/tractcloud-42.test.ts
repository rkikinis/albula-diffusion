// The table every naming network uses (tractcloud-42.json) is TractCloud's, unchanged: the same 1,600 classes -> tracts,
// the same tracts with their names, the same average streamline as TractCloud's converted release (model.json).
//
//   deno test -A --no-check naming/tractcloud-42.test.ts
import { assertEquals } from "jsr:@std/assert@1";
import { loadTable } from "./table.ts";

Deno.test("tractcloud-42.json is TractCloud's table, unchanged, and loads alone", () => {
  const table = JSON.parse(Deno.readTextFileSync(new URL("./tractcloud-42.json", import.meta.url)));
  const model = JSON.parse(Deno.readTextFileSync(new URL("../tractcloud/model/model.json", import.meta.url)));
  assertEquals(table.clusterToTract, model.clusterToTract);
  assertEquals(table.tracts, model.tracts);
  assertEquals(table.massCenter, model.massCenter);
  assertEquals(table.settings.points, model.settings.points);
  const t = loadTable(table);
  assertEquals([t.P, t.json.tracts.length, t.json.clusterToTract.length], [15, 43, 1600]);
  assertEquals(t.json.tracts[42].abbr, "Other");
});

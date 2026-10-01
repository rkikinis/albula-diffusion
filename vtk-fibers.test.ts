// readVtkFibers on both legacy layouts VTK writes: version 4 ("n i0 i1 ...") and version 5 (OFFSETS / CONNECTIVITY).
//   deno test -A --no-check vtk-fibers.test.ts   (with core's config)
import { assertEquals } from "jsr:@std/assert";
import { readVtkFibers } from "./vtk-fibers.ts";

const want = [[0, 0, 0, 1, 0, 0, 2, 0, 0], [5, 5, 5, 6, 6, 6]];

Deno.test("legacy VTK 4 polydata: two lines", () => {
  const v4 = `# vtk DataFile Version 4.2\nfibers\nASCII\nDATASET POLYDATA\nPOINTS 5 float\n0 0 0 1 0 0 2 0 0\n5 5 5 6 6 6\nLINES 2 7\n3 0 1 2\n2 3 4\nPOINT_DATA 5\nFIELD x 0\n`;
  assertEquals(readVtkFibers(v4).map((f) => [...f]), want);
});

Deno.test("legacy VTK 5 polydata: OFFSETS and CONNECTIVITY", () => {
  const v5 = `# vtk DataFile Version 5.1\nfibers\nASCII\nDATASET POLYDATA\nPOINTS 5 float\n0 0 0 1 0 0 2 0 0\n5 5 5 6 6 6\nLINES 3 5\nOFFSETS vtktypeint64\n0 3 5\nCONNECTIVITY vtktypeint64\n0 1 2 3 4\n`;
  assertEquals(readVtkFibers(v5).map((f) => [...f]), want);
});

// FIBERS FROM A LEGACY VTK POLYDATA FILE -- what UKFTractography (the original C++ program) writes with
// --writeAsciiTracts, read for the comparison with our port (Contents/tools/ukf-reference.ts, cases ukf.reference test).
// Points are in patient RAS when the input NRRD's space is RAS (UKFTractography's vtk_writer.cc: p = i2r * ijk).
// ASCII POINTS (float/double) and LINES only; point data and cell data after them are skipped.

export function readVtkFibers(text: string): Float32Array[] {
  const tok = text.split(/\s+/).filter(Boolean);
  let i = tok.indexOf("POINTS");
  if (i < 0) throw new Error("VTK: no POINTS section");
  const np = Number(tok[i + 1]);
  i += 3;
  const pts = new Float64Array(np * 3);
  for (let k = 0; k < np * 3; k++) pts[k] = Number(tok[i + k]);
  if (Number.isNaN(pts[np * 3 - 1])) throw new Error("VTK: POINTS ended early (a binary file? write it with --writeAsciiTracts)");
  let j = tok.indexOf("LINES", i + np * 3);
  if (j < 0) throw new Error("VTK: no LINES section");
  const nl = Number(tok[j + 1]);
  j += 3;
  // VTK 5 files write OFFSETS / CONNECTIVITY arrays; VTK 4 writes "n i0 i1 ..." per line.
  if (tok[j] === "OFFSETS") {
    const offType = 2; j += offType;
    const offsets: number[] = []; for (let k = 0; k < nl; k++) offsets.push(Number(tok[j + k]));
    j += nl;
    if (tok[j] !== "CONNECTIVITY") throw new Error("VTK: CONNECTIVITY expected after OFFSETS");
    j += 2;
    const out: Float32Array[] = [];
    for (let l = 0; l + 1 < nl; l++) {
      const a = offsets[l], b = offsets[l + 1], f = new Float32Array((b - a) * 3);
      for (let q = a; q < b; q++) { const p = Number(tok[j + q]); f.set(pts.subarray(3 * p, 3 * p + 3), (q - a) * 3); }
      out.push(f);
    }
    return out;
  }
  const out: Float32Array[] = [];
  for (let l = 0; l < nl; l++) {
    const n = Number(tok[j]); j++;
    const f = new Float32Array(n * 3);
    for (let q = 0; q < n; q++) { const p = Number(tok[j + q]); f.set(pts.subarray(3 * p, 3 * p + 3), q * 3); }
    j += n;
    out.push(f);
  }
  return out;
}

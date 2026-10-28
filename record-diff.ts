// TWO TRACT SETS' RECORDS, COMPARED (Ron, 2026-10-27, item 4 of the facts proposal: "we are not doing well in the facts
// only and in the consistency domain"). That morning a redo of the Prisma scan was reported as a result although its own
// record said "not corrected"; set side by side with the set it was meant to replace, the difference would have been the
// first line. Every value of each record is flattened to a path ("rules.tracking", "choices.partner.uid"), and every path
// whose value differs is listed, grouped by what it is: the inputs and why they were chosen, the rules, the versions of
// the code and the networks, the scanner's facts relied on, the namer's settings, what was corrected and how it was
// aligned, the counts -- and the times last (they always differ and say nothing about the result).

/** The groups, by the record's top-level field; anything not named here is "other". */
export const RECORD_GROUPS: Record<string, string[]> = {
  inputs: ["inputs", "choices", "fromTracts", "namedFrom"],
  rules: ["rules", "trackingRuleApplied", "motionRuleApplied", "cstGate"],
  versions: ["madeWith"],
  "scanner facts": ["scannerFacts", "directions"],
  namer: ["namer"],
  corrections: ["corrected", "alignedToT1", "alignment", "brain", "headFrame"],
  counts: ["streamlines", "onePointLeftOut"],
  times: ["made", "seconds", "stages", "gatedAt", "namedAt", "nameSeconds"],
};
const GROUP_OF = new Map(Object.entries(RECORD_GROUPS).flatMap(([g, keys]) => keys.map((k) => [k, g] as const)));

/** A record flattened to paths: objects by key, arrays of plain values kept whole, arrays of objects by index. */
export function flatten(o: unknown, prefix = "", out: Map<string, unknown> = new Map()): Map<string, unknown> {
  if (o && typeof o === "object" && !Array.isArray(o)) {
    const entries = Object.entries(o as Record<string, unknown>);
    if (!entries.length && prefix) out.set(prefix, {});
    for (const [k, v] of entries) flatten(v, prefix ? `${prefix}.${k}` : k, out);
  } else if (Array.isArray(o) && o.some((x) => x && typeof x === "object")) {
    if (!o.length) out.set(prefix, []);
    o.forEach((x, i) => flatten(x, `${prefix}[${i}]`, out));
  } else out.set(prefix, o);
  return out;
}

export interface RecordDifference { group: string; path: string; a: unknown; b: unknown }

/** Every path whose value differs between two records (absent on one side is a difference), in group order. */
export function recordDiff(a: Record<string, unknown>, b: Record<string, unknown>): RecordDifference[] {
  const fa = flatten(a), fb = flatten(b), out: RecordDifference[] = [];
  for (const path of new Set([...fa.keys(), ...fb.keys()])) {
    const va = fa.get(path), vb = fb.get(path);
    if (JSON.stringify(va) === JSON.stringify(vb)) continue;
    const top = path.split(/[.[]/)[0];
    // A head frame's own time, and that it was taken from an earlier scan of the same T1, are times, not corrections
    // (the critic, 2026-10-28, speed part A, finding 7).
    const timeInside = path === "headFrame.seconds" || path === "headFrame.takenFrom" || path === "secondOpinion.ms";
    out.push({ group: timeInside ? "times" : GROUP_OF.get(top) ?? "other", path, a: va, b: vb });
  }
  const order = [...Object.keys(RECORD_GROUPS).filter((g) => g !== "times"), "other", "times"];
  return out.sort((x, y) => order.indexOf(x.group) - order.indexOf(y.group) || x.path.localeCompare(y.path));
}

/** The differences as plain text, one line each, a group heading before its lines. */
export function recordDiffText(diffs: RecordDifference[], names: [string, string] = ["A", "B"]): string {
  if (!diffs.length) return "The two records are the same.";
  const show = (v: unknown) => v === undefined ? "(absent)" : typeof v === "string" ? v : JSON.stringify(v);
  const lines: string[] = [];
  let group = "";
  for (const d of diffs) {
    if (d.group !== group) { group = d.group; lines.push(`${group}:`); }
    lines.push(`  ${d.path}\n    ${names[0]}: ${show(d.a)}\n    ${names[1]}: ${show(d.b)}`);
  }
  return lines.join("\n");
}

// THE RUN LOG: what was tried on each diffusion scan, with what, and how it ended -- for people outside this project to
// study and report later (Ron, 2026-10-22: "keep a log file for third parties as to what was tried with what parameters
// and how it failed, so that this can be studied and reported down the road").
//
// One file per database, `SlicerAlbula-Logs/diffusion-runs.jsonl` beside its index: one JSON object a line, ADDED TO AND
// NEVER EDITED, so the history survives when results are deleted. This module is its only writer (the import job, and
// the deletion of tract sets, delete-tracts.ts). The databases hold anonymized public data only, so the file can be
// given out as it is.
//
// Lines, by `event` (every line also has `log`, `v` -- this format's version -- and `at`, the time written):
//   "run"        a job run began: its id, the database, how it was started, the extension's and the core's commits
//   "versions"   what the run makes tracts with: every rule number, the code fingerprint, the naming network and table,
//                the head template, SynthStrip's version (once a run, when the machinery is ready)
//   "case"       one diffusion scan: the scan, the reversed scan and the T1 used -- or, for the reversed scan, every
//                candidate of the study and why each was refused -- the outcome (made, kept, current, waiting, cannot,
//                failed) with its reason, the result's series UID, counts and times
//   "deleted"    a tract set deleted from the database: why, the evidence, and its whole stored record, copied here first
//   "note"       a statement about earlier lines, written by a person (a correction, a finding), with its evidence

export const RUN_LOG_VERSION = 1;
export const RUN_LOG_FOLDER = "SlicerAlbula-Logs";
export const runLogPath = (dbDir: string) => `${dbDir}/${RUN_LOG_FOLDER}/diffusion-runs.jsonl`;

export type RunLogEvent = "run" | "versions" | "case" | "deleted" | "note";

/** Add one line. Never throws: a log that cannot be written must not stop the work (the job says so in its output). */
export async function appendRunLog(dbDir: string, event: RunLogEvent, fields: Record<string, unknown>): Promise<boolean> {
  try {
    await Deno.mkdir(`${dbDir}/${RUN_LOG_FOLDER}`, { recursive: true });
    const line = JSON.stringify({ log: "albula-diffusion run log", v: RUN_LOG_VERSION, at: new Date().toISOString(), event, ...fields });
    await Deno.writeTextFile(runLogPath(dbDir), line + "\n", { append: true });
    return true;
  } catch {
    return false;
  }
}

/** Every line of a database's run log, oldest first (an unreadable line is skipped and counted). */
export async function readRunLog(dbDir: string): Promise<{ lines: Record<string, unknown>[]; unreadable: number }> {
  let text = "";
  try { text = await Deno.readTextFile(runLogPath(dbDir)); } catch { return { lines: [], unreadable: 0 }; }
  const lines: Record<string, unknown>[] = [];
  let unreadable = 0;
  for (const l of text.split("\n")) {
    if (!l.trim()) continue;
    try { lines.push(JSON.parse(l)); } catch { unreadable++; }
  }
  return { lines, unreadable };
}

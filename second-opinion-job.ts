// dcm2niix's loader FOR THE IMPORT-TIME JOB (Deno): imported here by its path in this folder, by the job only -- the
// page loads it from the app's assets instead (second-opinion.ts), so the page's bundle never carries it. Its worker
// starts silenced (second-opinion-worker.js).
import { Dcm2niix } from "./vendor/dcm2niix/index.js";
import type { SecondOpinionPlace } from "./second-opinion.ts";

export const besideThisFile = (): SecondOpinionPlace => ({ Cls: Dcm2niix, quietWorker: new URL("./second-opinion-worker.js", import.meta.url) });

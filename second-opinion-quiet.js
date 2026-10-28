// dcm2niix's worker, SILENT: under Deno (the import-time job) a worker's console is the program's own standard output,
// which carries one JSON object a line and, with --counts-only, no names or paths -- and dcm2niix prints the series
// name, date and file count of every conversion. This module is imported first (second-opinion-worker.js), so the
// Emscripten glue captures these empty printers when it starts; the results come back by message, not by the console.
const quiet = () => {};
console.log = quiet; console.info = quiet; console.warn = quiet; console.error = quiet; console.debug = quiet;
